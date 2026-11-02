"""Multi-view 1D CNN: nested cross-validation by TIC, ensemble and calibration; CUDA-only training."""
import copy
import hashlib
import json
import os
import pickle

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from data import (ROOT, CAT, DATA, LABELS, rows, eligible, downloaded_stars, load_star, select_sectors,
                  masks_for, save_json)
from lightcurve import VIEWS
from vetting import SCALARS, analyze

FEATURE_VERSION = 2
INPUT_SIZE = sum(c * b for _, c, b in VIEWS) + len(SCALARS)
MODEL = ROOT / 'models' / 'transit-cnn.pt'
MIN_STARS = 25


def star_examples(tic, tois):
    """Per labeled TOI: one example with all sectors and one per sector (training augmentation)."""
    star = load_star(tic)
    examples, skipped = [], []
    for r in tois:
        if r['tfopwg_disp'] not in LABELS:
            continue
        params = (float(r['pl_orbper']), float(r['pl_tranmid']), float(r['pl_trandurh']))
        errors = (abs(float(r.get('pl_orbpererr1') or 0)), abs(float(r.get('pl_tranmiderr1') or 0)))
        for sector in [None] + sorted(set(int(s) for s in star['sector'])):
            data = star if sector is None else select_sectors(star, [sector])
            try:
                x, metrics = analyze(data, *params, masks=masks_for(tois, r), errors=errors)
            except ValueError as exc:
                skipped.append({'toi': r['toi'], 'tic': tic, 'sector': sector, 'reason': str(exc)})
                continue
            examples.append({'toi': r['toi'], 'tic': tic, 'label': LABELS[r['tfopwg_disp']],
                             'kind': 'object' if sector is None else 'sector', 'sector': sector,
                             'x': x, 'metrics': metrics})
    return examples, skipped


def cached_examples(tic, tois):
    key = hashlib.sha256(json.dumps({
        'version': FEATURE_VERSION,
        'code': [hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in ('lightcurve.py', 'vetting.py')],
        'tois': [[r['toi'], r['tfopwg_disp'], r['pl_orbper'], r['pl_tranmid'], r['pl_trandurh']] for r in tois],
        'curves': sorted(json.loads(p.read_text())['sha256'] for p in (DATA / 'curves' / str(tic)).glob('*.json')),
    }, sort_keys=True).encode()).hexdigest()
    path = DATA / 'features' / f'{tic}.pkl'
    if path.exists():
        with path.open('rb') as f:
            stored = pickle.load(f)
        if stored['key'] == key:
            return stored['examples'], stored['skipped']
    examples, skipped = star_examples(tic, tois)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as f:
        pickle.dump({'key': key, 'examples': examples, 'skipped': skipped}, f)
    path.with_suffix('.tmp').replace(path)
    return examples, skipped


def dataset(workers=12):
    """Labeled examples from all downloaded stars, cached per star in data/features/."""
    from collections import defaultdict
    from concurrent.futures import ProcessPoolExecutor
    by_star = defaultdict(list)
    for r in rows():
        if eligible(r):
            by_star[int(r['tid'])].append(r)
    stars = [t for t in downloaded_stars() if any(r['tfopwg_disp'] in LABELS for r in by_star.get(t, []))]
    examples, skipped = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, (e, s) in enumerate(pool.map(cached_examples, stars, [by_star[t] for t in stars],
                                               chunksize=4), 1):
            examples += e
            skipped += s
            if done % 200 == 0 or done == len(stars):
                print(f'Features: {done}/{len(stars)} stars', flush=True)
    save_json(ROOT / 'reports' / 'skipped.json', skipped)
    return examples


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1), nn.GELU(),
                                  nn.Conv1d(channels, channels, 3, padding=1))

    def forward(self, x):
        return nn.functional.gelu(x + self.body(x))


class AveragePool(nn.Module):
    """Same windows as AdaptiveAvgPool1d, as a matrix product (deterministic backward on CUDA)."""
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        length = x.shape[-1]
        weights = torch.zeros(length, self.size, device=x.device, dtype=x.dtype)
        for i in range(self.size):
            lo, hi = (i * length) // self.size, -(-(i + 1) * length // self.size)
            weights[lo:hi, i] = 1 / (hi - lo)
        return x @ weights


class TransitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def branch(channels):
            return nn.Sequential(nn.Conv1d(channels, 32, 5, padding=2), nn.GELU(), Residual(32),
                                 nn.MaxPool1d(2), nn.Conv1d(32, 64, 3, padding=1), nn.GELU(), Residual(64),
                                 AveragePool(8), nn.Flatten())
        self.branches = nn.ModuleList(branch(c) for _, c, _ in VIEWS)
        self.head = nn.Sequential(nn.Linear(512 * len(VIEWS) + len(SCALARS), 128), nn.GELU(),
                                  nn.Dropout(.3), nn.Linear(128, 1))

    def forward(self, x):
        parts, start = [], 0
        for branch, (_, channels, bins) in zip(self.branches, VIEWS):
            parts.append(branch(x[:, start:start + channels * bins].reshape(-1, channels, bins)))
            start += channels * bins
        parts.append(x[:, start:])
        return self.head(torch.cat(parts, dim=1)).squeeze(1)


def inputs(X):
    X = np.asarray(X, dtype=np.float32).copy()
    # Fixed amplitude compression of the flux channels, without learning statistics from the test set.
    start = 0
    for _, channels, bins in VIEWS:
        for c in range(0, channels, 2):
            region = slice(start + c * bins, start + (c + 1) * bins)
            X[:, region] = np.arcsinh(X[:, region] / 5)
        start += channels * bins
    if not np.isfinite(X).all():
        raise ValueError('CNN input contains NaN or infinity')
    return X


def outer_folds(y, groups, k=5):
    """Each star is in the test set of exactly one fold; validation comes from the remaining stars."""
    folds = []
    zeros = np.zeros(len(y))
    for rest, test in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42).split(zeros, y, groups):
        inner_train, inner_val = next(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
                                      .split(zeros[rest], y[rest], groups[rest]))
        train, validation = rest[inner_train], rest[inner_val]
        for a, b in ((train, validation), (train, test), (validation, test)):
            if set(groups[a]) & set(groups[b]):
                raise ValueError('Star leakage between partitions')
        if any(len(set(y[i])) < 2 for i in (train, validation, test)):
            raise ValueError('Partition without both classes; enlarge the sample.')
        folds.append((train, validation, test))
    return folds


def choose_threshold(y, probabilities, target=.95, minimum=5):
    """Maximize recall subject to empirical precision and minimum support."""
    candidates = []
    for threshold in np.unique(probabilities):
        detected = probabilities >= threshold
        if detected.sum() >= minimum and y[detected].mean() >= target:
            candidates.append((float(y[detected].sum() / max(1, y.sum())), float(threshold)))
    return max(candidates, key=lambda x: (x[0], -x[1]))[1] if candidates else None


def wilson(success, total):
    if total == 0:
        return None
    z, p = 1.95996398454, success / total
    denom = 1 + z*z / total
    center = (p + z*z / (2 * total)) / denom
    radius = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denom
    return [float(center-radius), float(center+radius)]


def predict(models, x, batch=4096):
    """Mean of the model probabilities, in batches to stay within VRAM."""
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunk = x[start:start + batch]
            out.append(torch.stack([m(chunk).float().sigmoid() for m in models]).mean(0).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def fit(x, labels, train_rows, val_rows, args):
    model = TransitCNN().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    y = labels[train_rows].cpu().numpy()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y == 0).sum() / (y == 1).sum()), device='cuda'))
    scaler = torch.amp.GradScaler('cuda')
    best_loss, best, best_epoch, stale, history = float('inf'), None, 0, 0, []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in np.array_split(np.random.permutation(train_rows), max(1, int(np.ceil(len(train_rows) / args.batch_size)))):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x[batch])
            loss = criterion(logits.float(), labels[batch])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; training stopped')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(x[val_rows[i:i + 4096]]).float() for i in range(0, len(val_rows), 4096)])
            val_loss = float(criterion(logits, labels[val_rows]))
        history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), 'validation_loss': val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, best, best_epoch, stale = val_loss, copy.deepcopy(model.state_dict()), epoch + 1, 0
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f'  epoch {epoch+1}: train={np.mean(losses):.4f}, validation={val_loss:.4f}', flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(best)
    model.eval()
    return model, history, best_epoch


def at_threshold(y, scores, detected):
    tp = int(((y == 1) & detected).sum())
    return {
        'n': int(len(y)), 'positives': int(y.sum()),
        'average_precision': float(average_precision_score(y, scores)) if len(set(y)) == 2 else None,
        'roc_auc': float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None,
        'selected': int(detected.sum()),
        'precision': float(y[detected].mean()) if detected.any() else None,
        'precision_wilson95': wilson(tp, int(detected.sum())),
        'recall': float(tp / max(1, y.sum())),
        'confusion_matrix_rows_true_cols_pred_0_1': confusion_matrix(y, detected, labels=[0, 1]).tolist(),
    }


def subgroups(y, detected, values, edges):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = np.isfinite(values) & (values >= lo) & (values < hi)
        if not s.any():
            continue
        positive = s & (y == 1)
        rows.append({'range': [lo, hi], 'n': int(s.sum()), 'positives': int(positive.sum()),
                     'recall': float(detected[positive].mean()) if positive.any() else None,
                     'precision': float(y[s & detected].mean()) if (s & detected).any() else None})
    return rows


def vetting_rates(y, metrics):
    """Rejection rate of each test per class, to calibrate the thresholds in vetting.py."""
    out = {}
    for label, name in ((1, 'CP/KP'), (0, 'FP/FA')):
        chosen = [m for m, c in zip(metrics, y) if c == label]
        flags = sorted({k for m in chosen for k in m['flags']})
        out[name] = {'n': len(chosen), 'rejected': sum(m['verdict'] == 'fail' for m in chosen) / max(1, len(chosen)),
                     **{f: sum(m['flags'].get(f) == 'fail' for m in chosen) / max(1, len(chosen)) for f in flags}}
    return out


def train_network(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run make setup. There is no CPU fallback.')
    if not (0 < args.precision <= 1) or min(args.epochs, args.batch_size, args.patience) < 1:
        raise ValueError('Invalid configuration')
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    # Features before opening the GPU: the process pool uses fork.
    examples = dataset()
    torch.cuda.reset_peak_memory_stats()
    objects = np.array([i for i, e in enumerate(examples) if e['kind'] == 'object'])
    if len(objects) == 0:
        raise ValueError('No examples; run make curves')
    y = np.array([examples[i]['label'] for i in objects])
    groups = np.array([examples[i]['tic'] for i in objects])
    if any(len(set(groups[y == c])) < MIN_STARS for c in (0, 1)):
        raise ValueError(f'Cross-validation needs >={MIN_STARS} stars per class. Download more light curves.')
    x = torch.tensor(inputs([e['x'] for e in examples]), device='cuda')
    labels = torch.tensor([e['label'] for e in examples], dtype=torch.float32, device='cuda')
    tic_of = np.array([e['tic'] for e in examples])
    folds = outer_folds(y, groups)
    print(f'GPU: {torch.cuda.get_device_name(0)}; {len(objects)} objects from {len(set(groups))} stars, '
          f'{len(examples)} examples with sectors', flush=True)

    models, oof, detected, fold_reports = [], np.full(len(objects), np.nan), np.zeros(len(objects), bool), []
    for k, (train, validation, test) in enumerate(folds):
        print(f'Fold {k+1}/5: train={len(train)} validation={len(validation)} test={len(test)} objects', flush=True)
        # Training uses the object and each single sector of training stars; validation and test use objects only.
        train_rows = np.where(np.isin(tic_of, groups[train]))[0]
        model, history, best_epoch = fit(x, labels, train_rows, objects[validation], args)
        threshold = choose_threshold(y[validation], predict([model], x[objects[validation]]), args.precision)
        # The fold test set is scored only after weights and threshold are fixed.
        oof[test] = predict([model], x[objects[test]])
        detected[test] = oof[test] >= threshold if threshold is not None else False
        fold_reports.append({'fold': k + 1, 'best_epoch': best_epoch, 'threshold': threshold, 'history': history,
                             **at_threshold(y[test], oof[test], detected[test])})
        models.append(model)

    # Final ensemble threshold and calibration come from the out-of-fold predictions (cross-validation).
    threshold = choose_threshold(y, oof, args.precision)
    calibration = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(oof, y)
    metrics = [examples[i]['metrics'] for i in objects]
    pick = lambda key: np.array([m[key] if m[key] is not None else np.nan for m in metrics], dtype=float)
    star_scores = {}
    for tic, label, score in zip(groups, y, oof):
        best = star_scores.get(tic, (0, 0.))
        star_scores[tic] = (max(best[0], label), max(best[1], score))
    star_y, star_s = map(np.array, zip(*star_scores.values()))
    injection = None
    if args.injections:
        from search import injection_recovery
        injection = injection_recovery(models, threshold, args.injections)

    MODEL.parent.mkdir(exist_ok=True)
    torch.save({'state_dicts': [{k: v.cpu() for k, v in m.state_dict().items()} for m in models],
                'threshold': threshold, 'feature_version': FEATURE_VERSION,
                'calibration': [calibration.X_thresholds_.tolist(), calibration.y_thresholds_.tolist()]}, MODEL)
    report = {
        'experimental': True, 'architecture': 'multi-view residual 1D CNN, 5-fold ensemble',
        'device': torch.cuda.get_device_name(0), 'torch_version': str(torch.__version__),
        'cuda_version': torch.version.cuda, 'peak_vram_mb': torch.cuda.max_memory_allocated() / 2**20,
        'seed': 42, 'feature_version': FEATURE_VERSION, 'scalars': list(SCALARS),
        'catalog_sha256': hashlib.sha256(CAT.read_bytes()).hexdigest(),
        'code_sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                        for f in ('data.py', 'lightcurve.py', 'vetting.py', 'neural.py', 'search.py')},
        'configuration': {'epochs': args.epochs, 'batch_size': args.batch_size, 'patience': args.patience,
                          'target_precision': args.precision},
        'objects': int(len(objects)), 'stars': int(len(set(groups))), 'examples_with_sectors': len(examples),
        'cross_validated_test': at_threshold(y, oof, detected),
        'per_star': {'stars': int(len(star_y)), 'average_precision': float(average_precision_score(star_y, star_s))},
        'folds': fold_reports,
        'folds_average_precision_mean_std': [float(np.mean([f['average_precision'] for f in fold_reports])),
                                             float(np.std([f['average_precision'] for f in fold_reports]))],
        'ensemble_threshold': threshold,
        'subgroups': {
            'depth_ppm': subgroups(y, detected, pick('depth_ppm'), [0, 500, 1000, 3000, 10000, 1e9]),
            'period_days': subgroups(y, detected, pick('period_days'), [0, 1, 3, 10, 30, 1e9]),
            'snr': subgroups(y, detected, pick('snr'), [0, 7.3, 15, 30, 100, 1e9]),
            'sectors': subgroups(y, detected, np.array([len(m['sectors']) for m in metrics], float), [1, 2, 4, 8, 16, 1e9]),
        },
        'vetting_by_class': vetting_rates(y, metrics),
        'injection_recovery': injection,
        'note': ('Out-of-fold test metrics: each star scored once by a model that never saw it. '
                 'Ensemble threshold and calibration were fit on these same predictions, so they are '
                 'cross-validated estimates. Labels are TOI CP/KP vs FP/FA; they do not equal the planet '
                 'fraction among arbitrary signals.'),
        'test_predictions': [{'toi': examples[i]['toi'], 'tic': int(examples[i]['tic']), 'label': int(l),
                              'score': float(s), 'flags': examples[i]['metrics']['flags']}
                             for i, l, s in zip(objects, y, oof)],
    }
    save_json(ROOT / 'reports' / 'training.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('folds', 'test_predictions', 'code_sha256', 'subgroups')}, indent=2))


def load_ensemble():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    checkpoint = torch.load(MODEL, map_location='cuda', weights_only=True)
    if checkpoint.get('feature_version') != FEATURE_VERSION:
        raise RuntimeError('Model trained with another feature version; run make train')
    models = []
    for state in checkpoint['state_dicts']:
        model = TransitCNN().cuda()
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models, checkpoint


def score(models, checkpoint, X):
    """Ensemble score and score calibrated on out-of-fold predictions (base: TOI CP/KP vs FP/FA)."""
    raw = predict(models, torch.tensor(inputs(X), device='cuda'))
    cal_x, cal_y = checkpoint['calibration']
    return raw, np.interp(raw, cal_x, cal_y)


def gpu_check():
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch did not detect CUDA')
    model = TransitCNN().cuda()
    sample = torch.randn(64, INPUT_SIZE, device='cuda')
    loss = model(sample).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    print(json.dumps({'gpu': torch.cuda.get_device_name(0), 'torch': str(torch.__version__),
                      'cuda': torch.version.cuda, 'forward_backward': 'ok',
                      'parameters': sum(p.numel() for p in model.parameters())}, indent=2))


if __name__ == '__main__':
    gpu_check()
