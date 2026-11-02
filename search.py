"""Busca cega de trânsitos (BLS), pipeline de candidatos e teste de injeção-recuperação."""
import csv
import json
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tess import (ROOT, DATA, LABELS, load_star, select_sectors, downloaded_stars, star_tois, masks_for,
                  trend, segments, wrap, save_json)
from vetting import analyze, stellar_density, G, DAY

FPP_PYTHON = ROOT / '.venv-fpp' / 'bin' / 'python'
REPORTS = ROOT / 'reports' / 'candidates'
# Processos que usam CUDA (BLS) não podem nascer por fork de um processo que já abriu a GPU.
SPAWN = multiprocessing.get_context('spawn')
GPU_WORKERS = 4
MIN_SDE = 7.
MIN_BLS_SNR = 7.1
MAX_SIGNALS = 3
BIN_DAYS = 10 / 1440
# Duração típica de trânsito central numa estrela solar, usada para espaçar a grade de períodos.
SOLAR_DURATION = lambda period: .54 * (period / 365.25) ** (1 / 3)


def flatten(time, flux, sector):
    """Remove tendências sem conhecer o trânsito (mediana de ~1 dia) e corta flares."""
    cadence = np.median(np.diff(time)[np.diff(time) > 0])
    base = trend(time, flux, np.isfinite(flux), segments(time, sector, cadence), int(1. / cadence) | 1)
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
    """Passo em período para que a deriva de fase no baseline fique abaixo de 1/oversample da duração.

    Com dP/P = D(P)/(oversample·baseline), o número de passos até P é u(P) = ∫ oversample·baseline/D d(ln P);
    a grade é u⁻¹ em inteiros (metade da duração solar cobre estrelas menores e trânsitos rasantes).
    """
    if p_max <= p_min:
        return np.zeros(0)
    log_p = np.linspace(np.log(p_min), np.log(p_max), 20001)
    width = np.maximum(.02, .5 * SOLAR_DURATION(np.exp(log_p)))
    rate = oversample * baseline / width
    steps = np.r_[0, np.cumsum((rate[1:] + rate[:-1]) / 2 * np.diff(log_p))]
    return np.exp(np.interp(np.arange(0, steps[-1]), steps, log_p))


def spectrum_trend(power, block=501):
    """Tendência do espectro: mediana por blocos interpolada (O(N), em vez de mediana móvel)."""
    n = len(power) // block * block
    if n < 2 * block:
        return np.full(len(power), np.median(power))
    medians = np.median(power[:n].reshape(-1, block), axis=1)
    centers = np.arange(len(medians)) * block + block / 2
    return np.interp(np.arange(len(power)), centers, medians)


def fold(time, flux, p, nbins):
    """Dobra em fase e somas acumuladas de contagem e fluxo por caixa de fase."""
    import torch
    # Multiplicar pelo inverso: divisão em float64 é lenta na GPU.
    cells = ((time[None, :] * (1 / p)[:, None]) % 1 * nbins).long().clamp(max=nbins - 1)
    counts = torch.zeros(p.shape[0], nbins, dtype=time.dtype, device=time.device)
    counts.scatter_add_(1, cells, torch.ones_like(cells, dtype=time.dtype))
    sums = torch.zeros(p.shape[0], nbins, dtype=time.dtype, device=time.device)
    sums.scatter_add_(1, cells, flux.expand(p.shape[0], -1))
    zero = torch.zeros(p.shape[0], 1, dtype=time.dtype, device=time.device)
    return torch.cat([zero, counts, counts], 1).cumsum(1), torch.cat([zero, sums, sums], 1).cumsum(1)


def box_scan(c_n, c_y, p, durations, total_n, total_y, noise, nbins):
    """Melhor janela de trânsito (SNR) por período, varrendo todas as fases e durações do lote."""
    import torch
    columns = torch.arange(nbins, device=c_n.device)
    best = torch.full_like(p, -float('inf'))
    best_t0, best_d, best_depth = torch.zeros_like(p), torch.zeros_like(p), torch.zeros_like(p)
    for i in range(durations.shape[0]):
        d = durations[i]
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


def bls_power(t, f, noise, periods, durations, oversample=3):
    """BLS vetorizada em PyTorch (GPU quando disponível): SNR da profundidade por período.

    Para cada lote de períodos, dobra a curva em caixas de fase (scatter_add) e avalia todas as
    janelas de trânsito por somas acumuladas. Retorna SNR, época central, duração e profundidade.
    """
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    origin = float(t[0])
    # float64 sempre: precisão de fase não se degrada com baselines de vários anos.
    real = torch.float64
    time = torch.as_tensor(t - origin, dtype=real, device=device)
    flux = torch.as_tensor(f, dtype=real, device=device)
    total_n, total_y = float(len(t)), float(flux.sum())
    batch = max(1, int(2e7 // max(len(t), oversample * periods.max() / durations.min())))
    out = [np.empty(len(periods)) for _ in range(4)]
    for start in range(0, len(periods), batch):
        p = torch.as_tensor(periods[start:start + batch], dtype=real, device=device)
        # Caixas de fase por lote (períodos crescentes): ~1/oversample da menor duração em cada período.
        nbins = int(np.ceil(oversample * periods[start:start + batch].max() / durations.min()))
        c_n, c_y = fold(time, flux, p, nbins)
        best, best_t0, best_d, best_depth = box_scan(
            c_n, c_y, p, torch.as_tensor(durations, dtype=real, device=device),
            total_n, total_y, float(noise), nbins)
        for array, value in zip(out, (best, best_t0, best_d, best_depth)):
            array[start:start + len(p)] = value.cpu().numpy()
    out[1] += origin  # em float64 no numpy: a época absoluta (~2,46e6 d) não cabe em float32
    return out


def bls(time, rel, p_min=.5, p_max=100., max_signals=MAX_SIGNALS):
    """Sinais periódicos por BLS, mascarando cada um antes de procurar o próximo.

    A potência é o SNR da profundidade (ruído robusto por caixa de 10 min), comparável entre
    faixas de período; o SDE é medido dentro de cada faixa contra o próprio espectro.
    """
    t, f = binned(time, rel)
    baseline = t[-1] - t[0]
    p_max = min(p_max, baseline / 2)
    hours = np.array([.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 13, 16])
    found = []
    keep = np.ones(len(t), bool)
    for _ in range(max_signals):
        if p_max <= p_min or keep.sum() < 100:
            break
        noise = 1.4826 * np.median(np.abs(f[keep] - np.median(f[keep])))
        best = None
        edges = [e for e in (p_min, 1.5, 5, 15, p_max) if p_min <= e <= p_max]
        # Faixas de período com durações compatíveis (BLS exige duração < período mínimo da faixa).
        for lo, hi in zip(edges[:-1], edges[1:]):
            durations = hours[hours / 24 < .15 * lo] / 24
            grid = period_grid(lo, hi, baseline)
            if len(grid) < 10 or len(durations) == 0:
                continue
            snr, epochs, widths, depths = bls_power(t[keep], f[keep], noise, grid, durations)
            power = snr - spectrum_trend(snr)
            i = int(np.argmax(snr))
            # SDE robusto: harmônicos de um sinal forte não inflam a escala do espectro.
            sde = (power[i] - np.median(power)) / (1.4826 * np.median(np.abs(power - np.median(power))))
            if best is None or snr[i] > best['snr']:
                best = {'snr': float(snr[i]), 'sde': float(sde), 'period_days': float(grid[i]),
                        'epoch_bjd': float(epochs[i]), 'duration_hours': float(widths[i] * 24),
                        'bls_depth_ppm': float(depths[i] * 1e6)}
        if best is None or best['snr'] < MIN_BLS_SNR or best['sde'] < MIN_SDE:
            break
        # Resto de um sinal já achado (mesmo período ou harmônico): mascara sem registrar de novo.
        repeated = any(abs(best['period_days'] / (f['period_days'] * r) - 1) < .01
                       for f in found for r in (1, 2, .5, 3, 1 / 3))
        if not repeated:
            found.append(best)
        # ±1,5 duração: a BLS tende a subestimar a duração e deixar ingresso/egresso para a próxima passada.
        keep &= np.abs(wrap(t - best['epoch_bjd'], best['period_days'])) > 1.5 * best['duration_hours'] / 24
    return found


def match_known(event, tois):
    """Associa o sinal a um TOI do catálogo (mesmo período ou harmônico, mesma fase)."""
    for r in tois:
        p, e, d = float(r['pl_orbper']), float(r['pl_tranmid']), float(r['pl_trandurh']) / 24
        for ratio in (1, 2, .5, 3, 1 / 3):
            if abs(event['period_days'] / (p * ratio) - 1) < .01:
                lag = abs(wrap(event['epoch_bjd'] - e, min(p, p * ratio)))
                if lag < max(d, event['duration_hours'] / 24):
                    return {'toi': r['toi'], 'disposition': r['tfopwg_disp'], 'period_ratio': ratio}
    return None


def star_events(tic, given=None):
    """Sinais de uma estrela (busca BLS ou evento informado) com features e testes de vetting."""
    star = load_star(tic)
    tois = star_tois(tic)
    if given:
        events = [given]
    else:
        events = bls(star['time_bjd'], flatten(star['time_bjd'], star['flux'], star['sector']))
    out = []
    for i, event in enumerate(events):
        # Outros sinais da mesma estrela (catálogo e BLS) são mascarados ao medir este.
        others = [(e['period_days'], e['epoch_bjd'], e['duration_hours']) for j, e in enumerate(events) if j != i]
        known = match_known(event, tois)
        masks = others + [m for r, m in zip(tois, masks_for(tois)) if not known or r['toi'] != known['toi']]
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
    """TRICERATOPS no ambiente isolado; dados dobrados em ±1,5 duração, em caixas de 5 minutos."""
    if not FPP_PYTHON.exists():
        return {'error': 'Ambiente .venv-fpp ausente; execute make setup'}
    from tess import prepare
    m = event['metrics']
    star = load_star(event['tic'])
    period, epoch, duration = m['period_days'], m['epoch_bjd'], m['duration_hours'] / 24
    prep = prepare(star['time_bjd'], star['flux'], period, epoch, m['duration_hours'], star['sector'],
                   event['masks'])
    # Até 3 setores (os mais recentes) que de fato contêm trânsito: o TRICERATOPS baixa pixels de cada um.
    sectors = sorted(set(prep['sector'][prep['valid'] & (np.abs(prep['phase']) < duration / 2)].astype(int)))[-3:]
    near = prep['valid'] & (np.abs(prep['phase']) < 1.5 * duration) & np.isin(prep['sector'], sectors)
    cells = np.floor(prep['phase'][near] / (5 / 1440)).astype(int)
    _, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
    phase = np.bincount(inverse, prep['phase'][near]) / counts
    flux = 1 + np.bincount(inverse, prep['corrected'][near]) / counts
    outside = np.abs(phase) > duration / 2
    job = {'tic': int(event['tic']), 'sectors': [int(s) for s in sectors], 'period': period,
           'depth': max(m['depth_ppm'] or 0, 1) / 1e6, 'time': phase.tolist(), 'flux': flux.tolist(),
           'flux_err': float(np.std(flux[outside])) if outside.sum() > 5 else float(prep['scatter'])}
    work = DATA / 'fpp'
    work.mkdir(parents=True, exist_ok=True)
    source = work / f"{event['tic']}-{period:.6f}.json"
    target = source.with_suffix('.out.json')
    source.write_text(json.dumps(job))
    try:
        # cwd em data/fpp: o TRICERATOPS grava arquivos auxiliares (fundo Gaia) no diretório atual.
        result = subprocess.run([str(FPP_PYTHON), str(ROOT / 'fpp.py'), str(source), str(target)],
                                capture_output=True, text=True, timeout=3600, cwd=work)
    except subprocess.TimeoutExpired:
        return {'error': 'TRICERATOPS excedeu 1 hora'}
    if result.returncode:
        return {'error': (result.stderr or result.stdout).strip()[-1000:]}
    return json.loads(target.read_text())


def tier(cnn_selected, metrics, fpp):
    """Classe final; 'forte' usa FPP < 1% e NFPP < 0,1% (versão mais rígida do critério de
    validação estatística de Giacalone et al. 2021, que usa FPP < 1,5%)."""
    if metrics['verdict'] == 'fail':
        return 'rejeitado: vetting'
    if fpp and fpp.get('nfpp') is not None and fpp['nfpp'] > .1:
        return 'rejeitado: NFPP (provável estrela vizinha)'
    if not cnn_selected:
        return 'inconclusivo'
    if fpp and fpp.get('fpp') is not None:
        if fpp['fpp'] < .01 and fpp['nfpp'] < .001:
            return 'forte: FPP < 1% e NFPP < 0,1%'
        if fpp['fpp'] < .5 and fpp['nfpp'] < .001:
            return 'candidato: provável planeta'
        return 'inconclusivo'
    return 'candidato: sem FPP'


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
    print(f'{len(stars)} estrelas para buscar', flush=True)
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
                    print(f"[{n}/{len(stars)}] TIC {tic} erro: {e['error'][-200:]}", flush=True)
                if 'tier' in e:
                    print(f"[{n}/{len(stars)}] TIC {tic} P={e['period_days']:.4f} d "
                          f"{'TOI ' + str(e['known']['toi']) if e['known'] else 'NOVO'} "
                          f"cnn={e['cnn_calibrated']:.3f} -> {e['tier']}", flush=True)
    summary()


def summary():
    """Tabela única com todos os sinais avaliados, os mais promissores primeiro."""
    rows = []
    for path in REPORTS.glob('*.json'):
        for e in json.loads(path.read_text())['events']:
            if 'tier' not in e:
                continue
            m, fpp = e['metrics'], e.get('fpp') or {}
            rows.append({'tic': e['tic'], 'tier': e['tier'], 'known_toi': (e['known'] or {}).get('toi'),
                         'known_disposition': (e['known'] or {}).get('disposition'),
                         'period_days': m['period_days'], 'epoch_bjd': m['epoch_bjd'],
                         'duration_hours': m['duration_hours'], 'depth_ppm': m['depth_ppm'],
                         'radius_earth': m['radius_earth'], 'snr': m['snr'], 'transits': m['transits'],
                         'cnn_calibrated': e['cnn_calibrated'], 'fpp': fpp.get('fpp'), 'nfpp': fpp.get('nfpp'),
                         'flags': ';'.join(f'{k}={v}' for k, v in m['flags'].items())})
    order = ['forte', 'candidato', 'inconclusivo', 'rejeitado']
    rows.sort(key=lambda r: (next(i for i, o in enumerate(order) if r['tier'].startswith(o)),
                             r['known_toi'] is not None, -r['cnn_calibrated']))
    if rows:
        with (ROOT / 'reports' / 'candidates.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    counts = {}
    for r in rows:
        key = (r['tier'].split(':')[0], 'conhecido' if r['known_toi'] else 'novo')
        counts[key] = counts.get(key, 0) + 1
    print('Resumo (reports/candidates.csv):', {f'{a} / {b}': c for (a, b), c in sorted(counts.items())})


def inject(time, flux, period, epoch, depth, star, rng):
    """Trânsito trapezoidal físico: duração pela densidade estelar e parâmetro de impacto aleatório."""
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
    """Estrelas cujos TOIs são todos CP/KP: sem binárias eclipsantes que a BLS acharia antes da injeção."""
    return [t for t in downloaded_stars() if all(r['tfopwg_disp'] in ('CP', 'KP') for r in star_tois(t))]


def injection_trial(seed, hosts):
    rng = np.random.default_rng(seed)
    tic = int(rng.choice(hosts))
    star = load_star(tic)
    sector = int(rng.choice(np.unique(star['sector'])))
    star = select_sectors(star, [sector])
    # Remove os trânsitos conhecidos da estrela para injetar sobre uma curva sem sinal catalogado.
    keep = np.ones(len(star['time_bjd']), bool)
    for p, e, d in masks_for(star_tois(tic)):
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
                trial['vetting'] = 'sem cobertura'
            break
    return trial


def injection_recovery(models, threshold, n, workers=GPU_WORKERS):
    """Fração de trânsitos sintéticos achados pela BLS, aprovados no vetting e selecionados pela CNN."""
    from neural import predict, inputs
    import torch
    print(f'Injeção-recuperação: {n} trânsitos sintéticos', flush=True)
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
            'note': ('Um setor por injeção em estrelas com TOIs só CP/KP, trânsitos conhecidos removidos, '
                     'sem escurecimento de borda.')}
