"""Blind transit search (BLS), candidate pipeline and injection-recovery test."""
import csv
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from data import (ROOT, DATA, load_star, select_sectors, downloaded_stars, star_tois, known_signals,
                  save_json)
from lightcurve import prepare, trend, segments, wrap, SPAWN, GPU_WORKERS
from vetting import analyze, stellar_density, G, DAY

FPP_PYTHON = ROOT / '.venv-fpp' / 'bin' / 'python'
REPORTS = ROOT / 'reports' / 'candidates'
# Detection is gated on red-noise-aware SNR, like the SPOC TPS (MES >= 7.1); the SDE is only reported,
# because it is unstable for signals with few transits. Vetting does the false positive triage.
MIN_BLS_SNR = 7.1
# Consistency veto: without its strongest event, a real signal keeps about SNR·sqrt((N-1)/N).
MIN_EVENT_CONSISTENCY = .5
# Edge veto: scattered-light ramps and filter edge effects near data gaps were the main source of false
# alarms. Events sitting at a segment edge are dropped and the rest must still carry the signal (same rule
# as the consistency veto): a real planet can have one event near a gap, a pair of edge ramps cannot.
EDGE_DAYS = .5
MAX_EDGE_FRACTION = .5
MAX_SIGNALS = 3
BIN_DAYS = 10 / 1440
DURATION_HOURS = np.array([.5, .75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 13, 16, 20])
# Narrow period ranges, so each range can test durations up to a quarter of its shortest period.
PERIOD_EDGES = (.5, .8, 1.2, 1.8, 2.7, 4, 6, 9, 13.5, 20, 30, 45, 70, 100)
# Typical central transit duration for a Sun-like star, used to space the period grid.
SOLAR_DURATION = lambda period: .54 * (period / 365.25) ** (1 / 3)
# Blind-search detrending window in days (biweight), chosen by injection-recovery.
FLATTEN_WINDOW = .75


def flatten(time, flux, sector):
    """Detrend without knowing the transit (biweight, FLATTEN_WINDOW days) and clip flares."""
    cadence = np.median(np.diff(time)[np.diff(time) > 0])
    base = trend(time, flux, np.isfinite(flux), segments(time, sector, cadence), FLATTEN_WINDOW)
    rel = flux / base - 1
    noise = 1.4826 * np.nanmedian(np.abs(rel - np.nanmedian(rel)))
    rel[rel > 4 * noise] = np.nan
    return rel


def binned(time, rel):
    ok = np.isfinite(rel)
    cells = np.floor((time[ok] - time[ok][0]) / BIN_DAYS).astype(int)
    _, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
    return (np.bincount(inverse, time[ok]) / counts, np.bincount(inverse, rel[ok]) / counts)


def period_grid(p_min, p_max, baseline, oversample=2):
    """Period steps keeping the phase drift over the baseline below 1/oversample of the duration.

    With dP/P = D(P)/(oversample·baseline), the number of steps up to P is u(P) = ∫ oversample·baseline/D d(ln P);
    the grid is u⁻¹ at integers (half the solar duration covers smaller stars and grazing transits).
    """
    if p_max <= p_min:
        return np.zeros(0)
    log_p = np.linspace(np.log(p_min), np.log(p_max), 20001)
    width = np.maximum(.02, .5 * SOLAR_DURATION(np.exp(log_p)))
    rate = oversample * baseline / width
    steps = np.r_[0, np.cumsum((rate[1:] + rate[:-1]) / 2 * np.diff(log_p))]
    return np.exp(np.interp(np.arange(0, steps[-1]), steps, log_p))


def spectrum_trend(power, block=501):
    """Spectrum trend: interpolated block medians (O(N), instead of a running median)."""
    n = len(power) // block * block
    if n < 2 * block:
        return np.full(len(power), np.median(power))
    medians = np.median(power[:n].reshape(-1, block), axis=1)
    centers = np.arange(len(medians)) * block + block / 2
    return np.interp(np.arange(len(power)), centers, medians)


def fold(time, flux, p, nbins):
    """Phase fold and cumulative sums of counts and flux per phase bin."""
    import torch
    # Multiply by the reciprocal: float64 division is slow on the GPU.
    cells = ((time[None, :] * (1 / p)[:, None]) % 1 * nbins).long().clamp(max=nbins - 1)
    counts = torch.zeros(p.shape[0], nbins, dtype=time.dtype, device=time.device)
    counts.scatter_add_(1, cells, torch.ones_like(cells, dtype=time.dtype))
    sums = torch.zeros(p.shape[0], nbins, dtype=time.dtype, device=time.device)
    sums.scatter_add_(1, cells, flux.expand(p.shape[0], -1))
    zero = torch.zeros(p.shape[0], 1, dtype=time.dtype, device=time.device)
    return torch.cat([zero, counts, counts], 1).cumsum(1), torch.cat([zero, sums, sums], 1).cumsum(1)


def duration_noise(t, f, durations):
    """Noise of the in-transit mean for each duration, in per-bin units (correlated noise included).

    The standard deviation of the mean over windows of length d, times sqrt(bins per window), equals the
    per-bin white noise when the noise is white and grows when it is correlated (as in the TPS CDPP).
    """
    white = 1.4826 * np.median(np.abs(f - np.median(f)))
    out = []
    for d in durations:
        per_window = d / BIN_DAYS
        _, inverse, counts = np.unique(np.floor((t - t[0]) / d), return_inverse=True, return_counts=True)
        means = np.bincount(inverse, f) / counts
        full = counts >= per_window / 2
        if full.sum() < 10:
            out.append(white)
            continue
        red = 1.4826 * np.median(np.abs(means[full] - np.median(means[full]))) * np.sqrt(per_window)
        out.append(max(white, red))
    return np.array(out)


def consistent(t, f, noise, period, epoch, duration):
    """True when the detection does not rest on a single event (TPS-style consistency veto)."""
    inside = np.abs(wrap(t - epoch, period)) < duration / 2
    if inside.sum() < 3:
        return False
    _, inverse, counts = np.unique(np.round((t[inside] - epoch) / period), return_inverse=True, return_counts=True)
    if len(counts) < 2:
        return False
    sums = -np.bincount(inverse, f[inside])
    # Combined SNR of the in-transit mean is sum(depth)/(noise*sqrt(n)); drop the strongest event.
    total = sums.sum() / (noise * np.sqrt(counts.sum()))
    strongest = int(np.argmax(sums / np.sqrt(counts)))
    rest = (sums.sum() - sums[strongest]) / (noise * np.sqrt(counts.sum() - counts[strongest]))
    expected = total * np.sqrt((counts.sum() - counts[strongest]) / counts.sum())
    return bool(total > 0 and rest >= MIN_EVENT_CONSISTENCY * expected)


def edge_dominated(t, f, noise, period, epoch, duration):
    """True when the signal does not survive dropping the events that sit at segment edges."""
    gaps = np.where(np.diff(t) > .5)[0]
    edges = np.r_[t[0], t[gaps], t[gaps + 1], t[-1]]
    inside = np.abs(wrap(t - epoch, period)) < duration / 2
    if not inside.any():
        return True
    near = np.min(np.abs(edges[None, :] - t[inside][:, None]), axis=1) < EDGE_DAYS
    _, event = np.unique(np.round((t[inside] - epoch) / period), return_inverse=True)
    counts = np.bincount(event)
    at_edge = (np.bincount(event, near) / counts > MAX_EDGE_FRACTION)[event]
    if not at_edge.any():
        return False
    if at_edge.all():
        return True
    total = -f[inside].sum() / (noise * np.sqrt(inside.sum()))
    rest = -f[inside][~at_edge].sum() / (noise * np.sqrt((~at_edge).sum()))
    expected = total * np.sqrt((~at_edge).sum() / inside.sum())
    return bool(not (total > 0 and rest >= MIN_EVENT_CONSISTENCY * expected))


def fold_snr(t, f, noise, period, epoch, duration):
    """Box SNR of a given ephemeris (in-transit mean against zero), with the per-duration noise."""
    inside = np.abs(wrap(t - epoch, period)) < duration / 2
    if inside.sum() < 3:
        return -np.inf
    return float(-f[inside].sum() / (noise * np.sqrt(inside.sum())))


def resolve_alias(t, f, noise, signal):
    """Prefer P/2 or P/3 when folding there is stronger and consistent (DV-style period alias check)."""
    period, epoch, duration = signal['period_days'], signal['epoch_bjd'], signal['duration_hours'] / 24
    best_snr = fold_snr(t, f, noise, period, epoch, duration)
    for k in (2, 3):
        candidate = period / k
        if candidate < .5 or duration >= candidate / 2:
            continue
        snr = fold_snr(t, f, noise, candidate, epoch, duration)
        if snr > best_snr and consistent(t, f, noise, candidate, epoch, duration):
            best_snr, signal = snr, {**signal, 'period_days': candidate, 'alias_of': period}
    return signal


def box_scan(c_n, c_y, p, durations, total_n, total_y, noises, nbins):
    """Best transit window (SNR) per period, scanning every phase and duration of the batch."""
    import torch
    columns = torch.arange(nbins, device=c_n.device)
    best = torch.full_like(p, -float('inf'))
    best_t0, best_d, best_depth = torch.zeros_like(p), torch.zeros_like(p), torch.zeros_like(p)
    for i in range(durations.shape[0]):
        d, noise = durations[i], noises[i]
        k = (d / p * nbins).round().clamp(min=1).long()
        end = columns[None, :] + k[:, None]
        n_in = c_n.gather(1, end) - c_n[:, :nbins]
        y_in = c_y.gather(1, end) - c_y[:, :nbins]
        n_out = total_n - n_in
        depth = (total_y - y_in) / n_out.clamp(min=1) - y_in / n_in.clamp(min=1)
        snr = depth / (noise * torch.sqrt(1 / n_in.clamp(min=1) + 1 / n_out.clamp(min=1)))
        snr = torch.where((n_in >= 3) & (n_out >= 3), snr, torch.full_like(snr, -float('inf')))
        value, j = snr.max(1)
        better = value > best
        best = torch.where(better, value, best)
        best_t0 = torch.where(better, (j + k / 2) / nbins * p, best_t0)
        best_d = torch.where(better, d.expand_as(best), best_d)
        best_depth = torch.where(better, depth.gather(1, j[:, None])[:, 0], best_depth)
    return best, best_t0, best_d, best_depth


def bls_power(t, f, noises, periods, durations, oversample=3):
    """Vectorized BLS in PyTorch (GPU when available): depth SNR per period.

    For each batch of periods, folds the light curve into phase bins (scatter_add) and evaluates every
    transit window through cumulative sums. Returns SNR, mid-transit epoch, duration and depth.
    """
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    origin = float(t[0])
    # Always float64: phase precision does not degrade over multi-year baselines.
    real = torch.float64
    time = torch.as_tensor(t - origin, dtype=real, device=device)
    flux = torch.as_tensor(f, dtype=real, device=device)
    total_n, total_y = float(len(t)), float(flux.sum())
    batch = max(1, int(2e7 // max(len(t), oversample * periods.max() / durations.min())))
    out = [np.empty(len(periods)) for _ in range(4)]
    for start in range(0, len(periods), batch):
        p = torch.as_tensor(periods[start:start + batch], dtype=real, device=device)
        # Phase bins per batch (increasing periods): ~1/oversample of the shortest duration at each period.
        nbins = int(np.ceil(oversample * periods[start:start + batch].max() / durations.min()))
        c_n, c_y = fold(time, flux, p, nbins)
        best, best_t0, best_d, best_depth = box_scan(
            c_n, c_y, p, torch.as_tensor(durations, dtype=real, device=device),
            total_n, total_y, torch.as_tensor(noises, dtype=real, device=device), nbins)
        for array, value in zip(out, (best, best_t0, best_d, best_depth)):
            array[start:start + len(p)] = value.cpu().numpy()
    out[1] += origin  # in numpy float64: the absolute epoch (~2.46e6 d) does not fit in float32
    return out


def bls(time, rel, p_min=.5, p_max=100., max_signals=MAX_SIGNALS):
    """Periodic signals by BLS, masking each one before searching for the next.

    Power is the depth SNR with the noise measured at each duration's timescale (correlated noise
    included), comparable across period ranges. The SDE is reported for each signal, not used as a gate.
    """
    t, f = binned(time, rel)
    baseline = t[-1] - t[0]
    p_max = min(p_max, baseline / 2)
    found = []
    keep = np.ones(len(t), bool)
    # Vetoed detections are masked and the search continues, so allow extra passes (completeness over speed).
    for _ in range(2 * max_signals):
        if len(found) >= max_signals or p_max <= p_min or keep.sum() < 100:
            break
        all_durations = DURATION_HOURS / 24
        noises = duration_noise(t[keep], f[keep], all_durations)
        best = None
        edges = [p_min] + [e for e in PERIOD_EDGES if p_min < e < p_max] + [p_max]
        for lo, hi in zip(edges[:-1], edges[1:]):
            # Durations up to a quarter of the shortest period of the range and at most ~3x the central
            # duration of a Sun-like star at its longest period (low-density stars). No lower bound:
            # grazing transits can be short at any period (completeness over speed).
            chosen = (all_durations < .25 * lo) & (all_durations < 3 * SOLAR_DURATION(hi))
            grid = period_grid(lo, hi, baseline)
            if len(grid) < 10 or not chosen.any():
                continue
            snr, epochs, widths, depths = bls_power(t[keep], f[keep], noises[chosen], grid, all_durations[chosen])
            power = snr - spectrum_trend(snr)
            i = int(np.argmax(snr))
            # Robust SDE: harmonics of a strong signal do not inflate the spectrum scale.
            scale = 1.4826 * np.median(np.abs(power - np.median(power)))
            sde = (power[i] - np.median(power)) / scale if scale > 0 else np.inf
            if best is None or snr[i] > best['snr']:
                best = {'snr': float(snr[i]), 'sde': float(sde), 'period_days': float(grid[i]),
                        'noise': float(noises[chosen][int(np.argmin(np.abs(all_durations[chosen] - widths[i])))]),
                        'epoch_bjd': float(epochs[i]), 'duration_hours': float(widths[i] * 24),
                        'bls_depth_ppm': float(depths[i] * 1e6)}
        if best is None or best['snr'] < MIN_BLS_SNR:
            break
        noise = best.pop('noise')
        best = resolve_alias(t[keep], f[keep], noise, best)
        # Leftover of a signal already found (same period or harmonic): mask without recording again.
        repeated = any(abs(best['period_days'] / (prior['period_days'] * r) - 1) < .01
                       for prior in found for r in (1, 2, .5, 3, 1 / 3))
        shape = (best['period_days'], best['epoch_bjd'], best['duration_hours'] / 24)
        if (not repeated and consistent(t[keep], f[keep], noise, *shape)
                and not edge_dominated(t[keep], f[keep], noise, *shape)):
            found.append(best)
        # ±1.5 durations: BLS tends to underestimate the duration and leave ingress/egress for the next pass.
        keep &= np.abs(wrap(t - best['epoch_bjd'], best['period_days'])) > 1.5 * best['duration_hours'] / 24
    return found


def match_known(event, known):
    """Match the signal to a catalogued one (TOI, CTOI, planet, TCE or eclipsing binary): same period or a
    harmonic, and the same phase when the catalog ephemeris is still good at our epoch. The phase tolerance
    grows with the ephemeris drift (3σ of period and epoch, propagated); once that reaches a quarter of the
    period the phase is unknown and the period alone decides. Single-transit CTOIs (epoch, no period) match
    when one of our transits falls on their event; non-transiting planets and binaries match by period."""
    duration = event['duration_hours'] / 24
    for k in known:
        p, e = k['period'], k['epoch']
        d = max((k['duration_hours'] or 0) / 24, duration)
        found = None
        if p is None and e is not None:
            if abs(wrap(e - event['epoch_bjd'], event['period_days'])) < d + 3 * (k.get('epoch_err') or 0):
                found = {'period_ratio': None, 'phase_checked': True}
        elif p is not None:
            for ratio in (1, 2, .5, 3, 1 / 3):
                if abs(event['period_days'] / (p * ratio) - 1) < .01:
                    cycles = abs(event['epoch_bjd'] - e) / p if e is not None else 0
                    drift = 3 * np.hypot(cycles * (k.get('period_err') or 0), k.get('epoch_err') or 0)
                    if e is None or not k['transiting'] or drift > min(p, p * ratio) / 4:
                        found = {'period_ratio': ratio, 'phase_checked': False}
                    elif abs(wrap(event['epoch_bjd'] - e, min(p, p * ratio))) < d + drift:
                        found = {'period_ratio': ratio, 'phase_checked': True}
                    if found:
                        break
        if found:
            return {'source': k['source'], 'name': k['name'], 'toi': k.get('toi'),
                    'disposition': k['disposition'], **found}
    return None


def known_masks(known, event=None):
    """(period, epoch, duration) of every catalogued transiting signal with a full ephemeris.

    With an event, every entry matching it is left out: the same planet is often listed several times
    (TOI, published planet, TCEs of several DV runs), and masking any of them would erase the event."""
    return [(k['period'], k['epoch'], k['duration_hours']) for k in known
            if k['transiting'] and None not in (k['period'], k['epoch'], k['duration_hours'])
            and (event is None or match_known(event, [k]) is None)]


def star_events(tic, given=None):
    """Signals of one star (BLS search or a given event) with features and vetting tests."""
    star = load_star(tic)
    known_here = known_signals(tic)
    if given:
        events = [given]
    else:
        events = bls(star['time_bjd'], flatten(star['time_bjd'], star['flux'], star['sector']))
    out = []
    for i, event in enumerate(events):
        # Other signals of the same star (catalog and BLS) are masked while measuring this one.
        others = [(e['period_days'], e['epoch_bjd'], e['duration_hours']) for j, e in enumerate(events) if j != i]
        known = match_known(event, known_here)
        masks = others + known_masks(known_here, event)
        try:
            x, metrics = analyze(star, event['period_days'], event['epoch_bjd'], event['duration_hours'], masks)
        except ValueError as exc:
            out.append({'tic': tic, **event, 'known': known, 'error': str(exc)})
            continue
        out.append({'tic': tic, **event, 'known': known, 'masks': masks, 'x': x, 'metrics': metrics})
    return out


def safe_star_events(tic, given=None):
    try:
        return star_events(tic, given)
    except Exception as exc:
        return [{'tic': tic, 'error': f'{type(exc).__name__}: {exc}'}]


def fpp_job(event):
    """TRICERATOPS in the isolated environment; data folded within ±1.5 durations, in 5-minute bins."""
    if not FPP_PYTHON.exists():
        return {'error': '.venv-fpp environment missing; run make setup'}
    m = event['metrics']
    star = load_star(event['tic'])
    period, epoch, duration = m['period_days'], m['epoch_bjd'], m['duration_hours'] / 24
    prep = prepare(star['time_bjd'], star['flux'], period, epoch, m['duration_hours'], star['sector'],
                   event['masks'])
    # Up to 3 sectors (the most recent) that actually contain a transit: TRICERATOPS downloads pixels for each.
    sectors = sorted(set(prep['sector'][prep['valid'] & (np.abs(prep['phase']) < duration / 2)].astype(int)))[-3:]
    near = prep['valid'] & (np.abs(prep['phase']) < 1.5 * duration) & np.isin(prep['sector'], sectors)
    cells = np.floor(prep['phase'][near] / (5 / 1440)).astype(int)
    _, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
    phase = np.bincount(inverse, prep['phase'][near]) / counts
    flux = 1 + np.bincount(inverse, prep['corrected'][near]) / counts
    outside = np.abs(phase) > duration / 2
    job = {'tic': int(event['tic']), 'sectors': [int(s) for s in sectors], 'period': period,
           'depth': max(m['depth_ppm'] or 0, 1) / 1e6, 'time': phase.tolist(), 'flux': flux.tolist(),
           'flux_err': float(np.std(flux[outside])) if outside.sum() > 5 else float(prep['scatter']),
           # Local SPOC files: TRICERATOPS reads pixel geometry and apertures from them instead of the network.
           'fits': {str(s): str(ROOT / star['meta'][s]['fits']) for s in sectors}}
    work = DATA / 'fpp'
    work.mkdir(parents=True, exist_ok=True)
    source = work / f"{event['tic']}-{period:.6f}.json"
    target = source.with_suffix('.out.json')
    source.write_text(json.dumps(job))
    try:
        # cwd in data/fpp: TRICERATOPS writes helper files (Gaia background) to the current directory.
        result = subprocess.run([str(FPP_PYTHON), str(ROOT / 'fpp.py'), str(source), str(target)],
                                capture_output=True, text=True, timeout=3600, cwd=work)
    except subprocess.TimeoutExpired:
        return {'error': 'TRICERATOPS exceeded 1 hour'}
    if result.returncode:
        return {'error': (result.stderr or result.stdout).strip()[-1000:]}
    return json.loads(target.read_text())


def tier(cnn_selected, metrics, fpp):
    """Final tier; 'strong' uses FPP < 1% and NFPP < 0.1% (stricter version of the statistical
    validation criterion of Giacalone et al. 2021, which uses FPP < 1.5%)."""
    if metrics['verdict'] == 'fail':
        return 'rejected: vetting'
    if fpp and fpp.get('nfpp') is not None and fpp['nfpp'] > .1:
        return 'rejected: NFPP (likely nearby star)'
    if not cnn_selected:
        return 'inconclusive'
    if fpp and fpp.get('fpp') is not None:
        if fpp['fpp'] < .01 and fpp['nfpp'] < .001:
            return 'strong: FPP < 1% and NFPP < 0.1%'
        if fpp['fpp'] < .5 and fpp['nfpp'] < .001:
            return 'candidate: likely planet'
        return 'inconclusive'
    return 'candidate: no FPP'


def run_candidates(args):
    from neural import load_ensemble, score
    models, checkpoint = load_ensemble()
    threshold = checkpoint['threshold']
    given = None
    if args.period is not None:
        given = {'period_days': args.period, 'epoch_bjd': args.epoch, 'duration_hours': args.duration}
    stars = [args.tic] if args.tic else downloaded_stars()
    if args.field:
        field = {int(t) for t in json.loads((DATA / 'field.json').read_text())}
        stars = [t for t in stars if t in field]
    REPORTS.mkdir(parents=True, exist_ok=True)
    model_id = Path(ROOT / 'models' / 'transit-cnn.pt').stat().st_mtime
    if not args.tic:
        done = {int(p.stem) for p in REPORTS.glob('*.json')
                if json.loads(p.read_text()).get('model_mtime') == model_id}
        stars = [t for t in stars if t not in done]
    print(f'{len(stars)} stars to search', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=SPAWN) as pool, \
            ThreadPoolExecutor(max_workers=2) as fpp_pool:
        for n, events in enumerate(pool.map(safe_star_events, stars, [given] * len(stars)), 1):
            ok = [e for e in events if 'x' in e]
            if ok:
                raw, calibrated = score(models, checkpoint, np.stack([e['x'] for e in ok]))
                for e, r, c in zip(ok, raw, calibrated):
                    e['cnn_score'], e['cnn_calibrated'] = float(r), float(c)
                    e['cnn_selected'] = bool(threshold is not None and r >= threshold)
            pending = {}
            for e in ok:
                if not args.no_fpp and e['metrics']['verdict'] == 'pass' and (e['cnn_selected'] or given):
                    pending[id(e)] = fpp_pool.submit(fpp_job, e)
            for e in events:
                e.pop('x', None)
                e['fpp'] = pending[id(e)].result() if id(e) in pending else None
                if 'metrics' in e:
                    e['tier'] = tier(e['cnn_selected'], e['metrics'], e['fpp'] if e['fpp'] and 'error' not in e['fpp'] else None)
            tic = stars[n - 1]
            save_json(REPORTS / f'{tic}.json', {'tic': tic, 'model_mtime': model_id, 'threshold': threshold,
                                                'events': events})
            for e in events:
                if 'error' in e:
                    print(f"[{n}/{len(stars)}] TIC {tic} error: {e['error'][-200:]}", flush=True)
                if 'tier' in e:
                    print(f"[{n}/{len(stars)}] TIC {tic} P={e['period_days']:.4f} d "
                          f"{e['known']['name'] if e['known'] else 'not catalogued'} "
                          f"cnn={e['cnn_calibrated']:.3f} -> {e['tier']}", flush=True)
    summary()


def summary():
    """Single table with every vetted signal, most promising first."""
    rows = []
    for path in REPORTS.glob('*.json'):
        for e in json.loads(path.read_text())['events']:
            if 'tier' not in e:
                continue
            m, fpp = e['metrics'], e.get('fpp') or {}
            known = e['known'] or {}
            # Reports from before the CTOI cross-match only carry a TOI number.
            name = known.get('name') or (f"TOI {known['toi']}" if known.get('toi') else None)
            rows.append({'tic': e['tic'], 'tier': e['tier'], 'known': name, 'known_source': known.get('source'),
                         'known_disposition': known.get('disposition'),
                         'period_days': m['period_days'], 'epoch_bjd': m['epoch_bjd'],
                         'duration_hours': m['duration_hours'], 'depth_ppm': m['depth_ppm'],
                         'radius_earth': m['radius_earth'], 'snr': m['snr'], 'transits': m['transits'],
                         'cnn_calibrated': e['cnn_calibrated'], 'fpp': fpp.get('fpp'), 'nfpp': fpp.get('nfpp'),
                         'flags': ';'.join(f'{k}={v}' for k, v in m['flags'].items())})
    order = ['strong', 'candidate', 'inconclusive', 'rejected']
    # Unknown tiers (e.g. reports from older versions) sort last instead of failing.
    rank = lambda tier: next((i for i, o in enumerate(order) if tier.startswith(o)), len(order))
    rows.sort(key=lambda r: (rank(r['tier']), r['known'] is not None, -r['cnn_calibrated']))
    if rows:
        with (ROOT / 'reports' / 'candidates.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    counts = {}
    for r in rows:
        # "not catalogued": no match among TOIs, CTOIs, published planets, SPOC TCEs or known binaries.
        key = (r['tier'].split(':')[0], 'catalogued' if r['known'] else 'not catalogued')
        counts[key] = counts.get(key, 0) + 1
    print('Summary (reports/candidates.csv):', {f'{a} / {b}': c for (a, b), c in sorted(counts.items())})


def inject(time, flux, period, epoch, depth, star, rng):
    """Physical trapezoidal transit: duration from stellar density, random impact parameter."""
    density = stellar_density(star)
    if not np.isfinite(density):
        density = 1410.
    a = (G * density * (period * DAY) ** 2 / (3 * np.pi)) ** (1 / 3)
    k, b = np.sqrt(depth), rng.uniform(0, .7)
    t14 = period / (np.pi * a) * np.sqrt(max((1 + k) ** 2 - b ** 2, 1e-9))
    t23 = period / (np.pi * a) * np.sqrt(max((1 - k) ** 2 - b ** 2, 0))
    x = np.abs(wrap(time - epoch, period))
    shape = np.clip((t14 / 2 - x) / max((t14 - t23) / 2, 1e-6), 0, 1)
    return flux * (1 - depth * shape), t14 * 24


def planet_hosts():
    """Stars whose TOIs are all CP/KP: no eclipsing binaries that BLS would find before the injection."""
    hosts = []
    for tic in downloaded_stars():
        tois = star_tois(tic)
        # all() of an empty list is true: non-TOI stars (FIELD sample) do not count as hosts.
        if tois and all(r['tfopwg_disp'] in ('CP', 'KP') for r in tois):
            hosts.append(tic)
    return hosts


def injection_trial(seed, hosts):
    rng = np.random.default_rng(seed)
    tic = int(rng.choice(hosts))
    star = load_star(tic)
    sector = int(rng.choice(np.unique(star['sector'])))
    star = select_sectors(star, [sector])
    # Remove the star's known transits so the injection lands on a light curve with no cataloged signal.
    keep = np.ones(len(star['time_bjd']), bool)
    for p, e, d in known_masks(known_signals(tic)):
        keep &= np.abs(wrap(star['time_bjd'] - e, p)) > d / 24
    star = {k: (v[keep] if hasattr(v, 'shape') else v) for k, v in star.items()}
    if len(star['time_bjd']) < 1000:
        return None
    period = float(np.exp(rng.uniform(np.log(.5), np.log(13))))
    depth = float(np.exp(rng.uniform(np.log(2e-4), np.log(2e-2))))
    epoch = float(star['time_bjd'][0] + rng.uniform(0, period))
    info = star['meta'][sector]['star']
    star['flux'], duration = inject(star['time_bjd'], star['flux'], period, epoch, depth, info, rng)
    trial = {'tic': tic, 'sector': sector, 'period_days': period, 'depth_ppm': depth * 1e6,
             'duration_hours': duration, 'recovered': False, 'x': None}
    for event in bls(star['time_bjd'], flatten(star['time_bjd'], star['flux'], star['sector']), p_max=13.,
                     max_signals=2):
        if abs(event['period_days'] / period - 1) < .01 and abs(wrap(event['epoch_bjd'] - epoch, period)) < duration / 24:
            trial['recovered'] = True
            try:
                trial['x'], metrics = analyze(star, event['period_days'], event['epoch_bjd'], event['duration_hours'])
                trial['vetting'] = metrics['verdict']
            except ValueError:
                trial['vetting'] = 'no coverage'
            break
    return trial


def injection_recovery(models, threshold, n, workers=GPU_WORKERS):
    """Fraction of synthetic transits found by BLS, passing vetting and selected by the CNN."""
    from neural import predict, inputs
    import torch
    print(f'Injection-recovery: {n} synthetic transits', flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=SPAWN) as pool:
        hosts = planet_hosts()
        trials = [t for t in pool.map(injection_trial, range(n), [hosts] * n) if t]
    with_x = [t for t in trials if t['x'] is not None]
    if with_x:
        scores = predict(models, torch.tensor(inputs(np.stack([t['x'] for t in with_x])), device='cuda'))
        for t, s in zip(with_x, scores):
            t['cnn_score'] = float(s)
    for t in trials:
        t.pop('x')
        t['selected'] = bool(t['recovered'] and t.get('vetting') == 'pass' and threshold is not None
                             and t.get('cnn_score', 0) >= threshold)

    def table(key, edges):
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = [t for t in trials if lo <= t[key] < hi]
            if s:
                rows.append({'range': [lo, hi], 'n': len(s),
                             'bls_recovered': sum(t['recovered'] for t in s) / len(s),
                             'vetting_passed': sum(t.get('vetting') == 'pass' for t in s) / len(s),
                             'selected': sum(t['selected'] for t in s) / len(s)})
        return rows
    return {'trials': len(trials),
            'bls_recovered': sum(t['recovered'] for t in trials) / max(1, len(trials)),
            'vetting_passed': sum(t.get('vetting') == 'pass' for t in trials) / max(1, len(trials)),
            'selected': sum(t['selected'] for t in trials) / max(1, len(trials)),
            'by_depth_ppm': table('depth_ppm', [200, 500, 1000, 3000, 10000, 20001]),
            'by_period_days': table('period_days', [.5, 1, 3, 6, 13.01]),
            'note': ('One sector per injection on stars whose TOIs are all CP/KP, known transits removed, '
                     'no limb darkening.')}
