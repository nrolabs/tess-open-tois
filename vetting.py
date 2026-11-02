"""Testes de falso positivo calculados localmente para um evento de período, época e duração dados."""
import numpy as np
from scipy import stats
from scipy.optimize import least_squares
from tess import prepare, views, trend, segments, wrap, SCALARS

G = 6.674e-11
DAY = 86400.
REARTH_PER_RSUN = 109.1
PIXEL_ARCSEC = 21.
# Limiares: SNR de detecção do SPOC (TPS, 7,1σ).
MIN_SNR = 7.1
MIN_TRANSITS = 2
# Reprovação: vários testes simultâneos e sistemáticas entre setores pedem margem acima de 3σ.
ODD_EVEN_SIGMA = 4.
CENTROID_SIGMA = 4.
SECONDARY_SIGMA = 5.
MAX_RADIUS_EARTH = 25.
FAIL, WARN = 'fail', 'warn'


def window_noise(time, values, mask, width, cadence):
    """Ruído de uma média sobre `width` dias, medido fora do trânsito (inclui ruído correlacionado)."""
    t, v = time[mask], values[mask]
    expected = width / cadence
    if len(t) < 10:
        return np.nan, expected
    _, inverse, counts = np.unique(np.floor((t - t[0]) / width), return_inverse=True, return_counts=True)
    means = np.bincount(inverse, v) / counts
    full = counts >= expected / 2
    white = 1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(expected)
    if full.sum() < 5:
        return white, expected
    red = 1.4826 * np.median(np.abs(means[full] - np.median(means[full])))
    return max(red, white), expected


def trapezoid(x, depth, total, ingress):
    """Trânsito trapezoidal em unidades da duração do catálogo; ingress é fração de total."""
    half, ramp = total / 2, max(ingress * total, 1e-6)
    return -depth * np.clip((half - np.abs(x)) / ramp, 0, 1)


def fit_shape(prep):
    x = prep['phase'] / prep['duration']
    ok = prep['valid'] & (np.abs(x) < 2)
    edges = np.linspace(-2, 2, 81)
    centers, profile = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        chosen = ok & (x >= lo) & (x < hi)
        if chosen.sum() >= 3:
            centers.append((lo + hi) / 2)
            profile.append(np.median(prep['corrected'][chosen]))
    if len(centers) < 20:
        return np.nan, np.nan
    centers, profile = np.array(centers), np.array(profile)
    # Fluxo negativo (dado ruim) daria chute fora dos limites do ajuste.
    start = [min(max(-profile.min(), 1e-6), .99), 1., .2]
    fit = least_squares(lambda p: trapezoid(centers, *p) - profile, start,
                        bounds=([0, .3, .01], [1, 2.5, .5]))
    return float(fit.x[2]), float(fit.x[1] * prep['duration'])


def centroid_test(prep, centr1, centr2, depth, meta):
    """Posição da fonte do trânsito pelo deslocamento do centroide, comparada à posição do alvo.

    Centroide ponderado por fluxo: se a fonte em x_s escurece uma fração δ do fluxo da abertura,
    Δc = δ (c_fora − x_s) / (1 − δ). Combina os setores por média ponderada em cada eixo.
    """
    core = np.abs(prep['phase']) < prep['duration'] / 4
    near = prep['outside'] & (np.abs(prep['phase']) < 3 * prep['duration'])
    offsets = []
    for sector in np.unique(prep['sector']):
        info = meta.get(int(sector), {})
        if not info.get('target_pixel') or not depth > 0:
            continue
        aperture_depth = depth * (info.get('crowdsap') or 1.)
        select = prep['sector'] == sector
        shifts = []
        for values, target in zip((centr1, centr2), info['target_pixel']):
            good = select & np.isfinite(values)
            if (good & core).sum() < 3 or (good & near).sum() < 10:
                break
            residual = values - trend(prep['time'], values, good & prep['outside'],
                                      segments(prep['time'], prep['sector'], prep['cadence']), prep['window'])
            noise, expected = window_noise(prep['time'], residual, good & prep['outside'] & np.isfinite(residual),
                                           prep['duration'] / 2, prep['cadence'])
            inside = good & core & np.isfinite(residual)
            if not np.isfinite(noise) or inside.sum() < 3:
                break
            reference = good & near & np.isfinite(residual)
            shift = np.mean(residual[inside]) - np.median(residual[reference])
            factor = (1 - aperture_depth) / aperture_depth
            source = np.median(values[reference]) - shift * factor
            # Piso sistemático de 0,1 px: dispersão medida entre centroide e alvo em estrelas isoladas.
            error = np.hypot(noise * np.sqrt(expected / inside.sum()) * factor, .1)
            shifts.append((source - target, error))
        if len(shifts) == 2:
            offsets.append(shifts)
    if not offsets:
        return np.nan, np.nan
    total = []
    for axis in range(2):
        weights = np.array([1 / o[axis][1] ** 2 for o in offsets])
        values = np.array([o[axis][0] for o in offsets])
        total.append((np.sum(weights * values) / weights.sum(), 1 / np.sqrt(weights.sum())))
    chi2 = sum((o / e) ** 2 for o, e in total)
    sigma = stats.norm.isf(max(stats.chi2.sf(chi2, 2), 1e-300) / 2)
    return float(min(sigma, 40.)), float(np.hypot(*[o for o, _ in total]) * PIXEL_ARCSEC)


def stellar_density(star):
    """Densidade média (kg/m³) a partir de log g e raio do TIC."""
    if star.get('logg') is None or star.get('radius') is None or not star['radius'] > 0:
        return np.nan
    gravity = 10 ** star['logg'] / 100
    return 3 * gravity / (4 * np.pi * G * star['radius'] * 6.957e8)


def measure(prep, star=None, centr1=None, centr2=None, crowding=np.nan, meta=None):
    star = star or {}
    period, duration = prep['period'], prep['duration']
    corrected, valid, time = prep['corrected'], prep['valid'], prep['time']
    noise, expected = window_noise(time, corrected, prep['outside'], duration / 2, prep['cadence'])
    box_noise, box_expected = window_noise(time, corrected, prep['outside'], .8 * duration, prep['cadence'])
    number = np.round((time - prep['epoch']) / period)
    odd = number % 2 == 1

    def depth(offset=0., select=True, half=duration / 4):
        """Profundidade média; o erro é o maior entre o ruído fora do trânsito e a dispersão entre eventos,
        que captura diferenças sistemáticas entre setores (diluição, variabilidade)."""
        chosen = valid & select & (np.abs(wrap(prep['phase'] - offset * period, period)) < half)
        if chosen.sum() == 0:
            return np.nan, np.nan, 0
        sigma, points = (noise, expected) if half == duration / 4 else (box_noise, box_expected)
        error = sigma * np.sqrt(points / chosen.sum())
        cycle = np.round((time[chosen] - prep['epoch'] - offset * period) / period)
        _, inverse, counts = np.unique(cycle, return_inverse=True, return_counts=True)
        full = counts >= max(3, points / 4)
        events = -np.bincount(inverse, corrected[chosen])[full] / counts[full]
        # Dispersão robusta: um evento atípico (flare, falha) não deve dominar o erro.
        if len(events) >= 5:
            error = max(error, 1.4826 * np.median(np.abs(events - np.median(events))) / np.sqrt(len(events)))
        return float(-np.mean(corrected[chosen])), float(error), int(full.sum())

    d, d_err, transits = depth()
    box, box_err, _ = depth(half=.4 * duration)
    d_odd, e_odd, n_odd = depth(select=odd)
    d_even, e_even, n_even = depth(select=~odd)
    # Com um único evento num grupo, diferença entre setores e diferença ímpar/par se confundem.
    odd_even = abs(d_odd - d_even) / np.hypot(e_odd, e_even) if min(n_odd, n_even) >= 2 else np.nan

    # Busca do secundário em todas as fases fora do primário (órbitas excêntricas incluídas).
    nbins = max(64, int(np.ceil(8 * period / duration)))
    cells = np.floor((prep['phase'] / period % 1) * nbins).astype(int) % nbins
    sums = np.bincount(cells[valid], corrected[valid], nbins)
    hits = np.bincount(cells[valid], minlength=nbins).astype(float)
    width = max(1, int(round(nbins * duration / (2 * period))))
    kernel = np.ones(width)
    window_sum = np.convolve(np.r_[sums, sums[:width]], kernel, 'valid')[:nbins]
    window_hits = np.convolve(np.r_[hits, hits[:width]], kernel, 'valid')[:nbins]
    center = ((np.arange(nbins) + width / 2) / nbins) % 1
    allowed = (np.abs(center - .5) < .5 - 1.5 * duration / period) & (window_hits >= max(3, expected / 4))
    sec_depth, sec_sigma, sec_phase, brightening, sec_threshold = np.nan, 0., np.nan, 0., np.inf
    if allowed.any():
        depths = -window_sum / np.maximum(window_hits, 1)
        sigmas = depths / (noise * np.sqrt(expected / np.maximum(window_hits, 1)))
        best = int(np.argmax(np.where(allowed, sigmas, -np.inf)))
        sec_phase = float(center[best])
        sec_depth, sec_err, _ = depth(offset=sec_phase)
        sec_sigma = float(max(sec_depth / sec_err, 0.)) if sec_err > 0 else 0.
        brightening = float(max(-sigmas[allowed].min(), 0.))
        # Muitas fases testadas: limiar com 1% de falso alarme após correção de Bonferroni.
        sec_threshold = max(3., float(stats.norm.isf(.01 / max(1, allowed.sum() / width))))

    ingress, total = fit_shape(prep)
    density = stellar_density(star)
    k = np.sqrt(max(d, 0.))
    if np.isfinite(density):
        a_over_r = (G * density * (period * DAY) ** 2 / (3 * np.pi)) ** (1 / 3)
        albedo = sec_depth * a_over_r ** 2 / d if d > 0 and np.isfinite(sec_depth) else np.nan
    else:
        a_over_r = albedo = np.nan
    t14 = total if np.isfinite(total) else duration
    # Densidade máxima compatível com a duração observada (parâmetro de impacto zero, órbita circular).
    max_density = 3 * np.pi * (period / (np.pi * t14) * (1 + k)) ** 3 / (G * (period * DAY) ** 2)
    density_ratio = max_density / density if np.isfinite(density) else np.nan
    radius = k * star['radius'] * REARTH_PER_RSUN if star.get('radius') else np.nan
    if centr1 is not None:
        centroid_sigma, centroid_arcsec = centroid_test(prep, centr1[prep['index']], centr2[prep['index']], d,
                                                        meta or {})
    else:
        centroid_sigma = centroid_arcsec = np.nan
    snr = box / box_err if np.isfinite(box_err) and box_err > 0 else np.nan

    flags = {}
    if not snr >= MIN_SNR:
        flags['low_snr'] = FAIL
    if transits < MIN_TRANSITS:
        flags['single_transit'] = FAIL
    if odd_even > ODD_EVEN_SIGMA:
        flags['odd_even'] = FAIL
    elif odd_even > 3:
        flags['odd_even'] = WARN
    if (sec_sigma > sec_threshold and sec_sigma - brightening > 1
            and (albedo > 1 if np.isfinite(albedo) else sec_depth > .1 * d)):
        flags['secondary_eclipse'] = FAIL if sec_sigma > SECONDARY_SIGMA else WARN
    if radius > MAX_RADIUS_EARTH:
        flags['too_large'] = FAIL
    if centroid_sigma > CENTROID_SIGMA and centroid_arcsec > PIXEL_ARCSEC:
        flags['centroid_offset'] = FAIL
    elif centroid_sigma > 3 and centroid_arcsec > PIXEL_ARCSEC / 2:
        flags['centroid_offset'] = WARN
    if crowding < .8:
        flags['crowded'] = WARN
    if density_ratio < .2:
        flags['duration_too_long'] = WARN
    if ingress > .4:
        flags['v_shape'] = WARN

    def num(v):
        return float(v) if v is not None and np.isfinite(v) else None
    return {
        'period_days': period, 'epoch_bjd': prep['epoch'], 'duration_hours': duration * 24,
        'depth_ppm': num(d * 1e6), 'snr': num(snr), 'transits': transits,
        'odd_depth_ppm': num(d_odd * 1e6), 'even_depth_ppm': num(d_even * 1e6), 'odd_even_sigma': num(odd_even),
        'secondary_depth_ppm': num(sec_depth * 1e6), 'secondary_sigma': num(sec_sigma),
        'secondary_phase': num(sec_phase), 'secondary_albedo': num(albedo),
        'secondary_threshold_sigma': num(sec_threshold), 'brightening_sigma': num(brightening),
        'ingress_fraction': num(ingress), 'fitted_duration_hours': num(total * 24 if np.isfinite(total) else np.nan),
        'radius_earth': num(radius), 'a_over_rstar': num(a_over_r), 'density_ratio': num(density_ratio),
        'centroid_sigma': num(centroid_sigma), 'source_offset_arcsec': num(centroid_arcsec),
        'crowdsap': num(crowding), 'star': star, 'sectors': sorted(int(s) for s in np.unique(prep['sector'])),
        'flags': flags, 'verdict': FAIL if FAIL in flags.values() else 'pass',
    }


def scalars(m):
    """Métricas em escala comparável para a CNN; valores ausentes viram referências solares/neutras."""
    star = m['star'] or {}

    def get(v, default):
        return default if v is None or not np.isfinite(v) else v
    values = {
        'log_depth': np.log10(max(get(m['depth_ppm'], 1.), 1.)) - 3,
        'snr': np.arcsinh(get(m['snr'], 0.) / 10),
        'odd_even_sigma': np.arcsinh(get(m['odd_even_sigma'], 0.) / 3),
        'secondary_sigma': np.arcsinh(get(m['secondary_sigma'], 0.) / 3),
        'ingress_fraction': get(m['ingress_fraction'], .2),
        'log_duration_period': np.log10(m['duration_hours'] / 24 / m['period_days']) + 1.5,
        'log_transits': np.log10(max(m['transits'], 1)),
        'centroid_sigma': np.arcsinh(get(m['centroid_sigma'], 0.) / 3),
        'crowding': get(m['crowdsap'], 1.),
        'log_star_radius': np.log10(get(star.get('radius'), 1.)),
        'star_logg': get(star.get('logg'), 4.44) - 4.44,
        'star_teff': get(star.get('teff'), 5778.) / 5778 - 1,
        'star_tmag': get(star.get('tmag'), 10.) / 10 - 1,
        'log_density_ratio': np.clip(np.log10(get(m['density_ratio'], 1.)), -3, 3),
    }
    return np.array([values[k] for k in SCALARS], dtype=float)


def refine(star, period, epoch, duration_hours, period_err, epoch_err, masks=()):
    """Corrige a deriva da efeméride do catálogo entre setores distantes no tempo.

    Procura período e época dentro de 3σ das incertezas do catálogo (no máximo ±1 duração),
    maximizando o SNR de uma caixa no trânsito.
    """
    time, d = star['time_bjd'], duration_hours / 24
    cycles = np.round((time - epoch) / period)
    reference = float(np.round(np.median(cycles)))
    span = max(float(np.abs(cycles - reference).max()), 1.)
    epoch_range = min(3 * np.hypot(epoch_err, reference * period_err), d)
    period_range = min(3 * period_err, d / span)
    if epoch_range < d / 8 and period_range * span < d / 8:
        return period, epoch
    center = epoch + reference * period
    prep = prepare(time, star['flux'], period, center, min(2 * duration_hours, period * 24 / 2.2),
                   star['sector'], masks)
    near = prep['valid'] & (np.abs(prep['phase']) < 2 * d)
    t, f = prep['time'][near], prep['corrected'][near]
    best = (-np.inf, period, center)
    steps = lambda r: np.linspace(-r, r, min(33, 2 * int(np.ceil(r / (d / 8))) + 1))
    for dp in steps(period_range * span) / span:
        for de in steps(epoch_range):
            box = np.abs(wrap(t - center - de, period + dp)) < .4 * d
            if box.sum() >= 5:
                score = -f[box].sum() / np.sqrt(box.sum())
                if score > best[0]:
                    best = (score, period + dp, center + de)
    return best[1], best[2]


def analyze(star, period, epoch, duration_hours, masks=(), errors=None):
    """Vetor de entrada da CNN (vistas + escalares) e métricas de vetting de um evento.

    errors=(erro do período, erro da época) do catálogo ativa o refinamento da efeméride.
    """
    catalog = (period, epoch)
    if errors:
        # Só setores em que a efeméride do catálogo localiza o trânsito (3σ ≤ 1 duração).
        cycles = np.round((star['time_bjd'] - epoch) / period)
        precise = 3 * np.hypot(errors[1], cycles * errors[0]) <= duration_hours / 24
        sectors = [s for s in np.unique(star['sector']) if precise[star['sector'] == s].all()]
        if not sectors:
            raise ValueError('Efeméride do catálogo imprecisa demais para os setores disponíveis')
        if len(sectors) < len(np.unique(star['sector'])):
            star = {k: (v[np.isin(star['sector'], sectors)] if hasattr(v, 'shape') else v) for k, v in star.items()}
        period, epoch = refine(star, period, epoch, duration_hours, *errors, masks)
    prep = prepare(star['time_bjd'], star['flux'], period, epoch, duration_hours, star['sector'], masks)
    sectors = [int(s) for s in np.unique(prep['sector'])]
    meta = [star['meta'][s] for s in sectors if s in star.get('meta', {})]
    info = meta[0]['star'] if meta else {}
    crowding = np.nanmedian([m['crowdsap'] for m in meta if m.get('crowdsap') is not None] or [np.nan])
    metrics = measure(prep, info, star.get('centr1'), star.get('centr2'), crowding, star.get('meta'))
    metrics['tic'] = star.get('tic')
    metrics['catalog_ephemeris'] = list(catalog)
    return np.r_[views(prep), scalars(metrics)], metrics
