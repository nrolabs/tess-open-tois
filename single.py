"""Single- and duo-transit search for long periods, where BLS has too few events to work.

Long-period planets leave one to three transits in a multi-year light curve. Each event is searched
on its own (matched box filter with correlated noise per sector), vetted on its own (edges, symmetry,
systematics shared with other stars, centroid, size), and events of the same star with consistent
depth and duration are paired. A pair constrains the period to Δt/n; aliases are pruned when a
predicted transit falls on covered data without a dip (as in duotransit work on TESS) and ranked by
the period a central transit of that duration implies for the star's density.
"""
import json

import numpy as np
from data import DATA, load_star
from lightcurve import trend, segments, wrap
from vetting import analyze, stellar_density, G, DAY, FAIL, WARN, PIXEL_ARCSEC, CENTROID_SIGMA

# Durations searched: long periods mean long transits (a Sun-like star gives ~6.5 h at 100 d, ~13 h
# at one year); grazing and small-star transits are shorter.
SINGLE_HOURS = np.array([2, 2.5, 3, 4, 5, 6.5, 8, 10, 13, 16, 20, 25])
BIN_DAYS = 10 / 1440
# Detection threshold (single-event SNR with correlated noise), calibrated on inverted light curves
# of the field stars: see README.
MIN_SINGLE_SNR = 8.
MAX_EVENTS = 6
# Minimum fraction of the expected in-transit bins that must be present.
MIN_COVERAGE = .75
EDGE_DAYS = .5
ASYMMETRY_SIGMA = 4.
# ...and the halves must differ by this fraction of the depth (a ramp or step, not a slightly
# off-center window on a deep transit).
ASYMMETRY_FRACTION = .3
# Pairing: depths within 3σ and durations within a factor 1.5.
PAIR_SIGMA = 3.
PAIR_DURATION_RATIO = 1.5
MIN_ALIAS_DAYS = 2.
# Aliases whose implied central-transit period is this many times larger are disfavoured
# (allowance for density error, impact parameter and eccentricity).
DENSITY_ALLOWANCE = 4.
# The event must beat the strongest dip found in the star's own inverted light curve: variable stars,
# systematics and red noise produce dips and bumps alike, so this sets a per-star detection ceiling.
# Both are maxima of the same noise, so a margin is needed: noise events that passed reached up to
# 1.23 times the ceiling on the inverted field curves (calibration, see README).
MIN_CEILING_RATIO = 1.3
COMMON_MODE = DATA / 'common_mode.npz'
COMMON_BIN_DAYS = 30 / 1440
COMMON_WINDOW = 2.
COMMON_MIN_STARS = 5
COMMON_SIGMA = 4.


def detrended(t, f, sector, duration, protect=None, clip=True):
    """10-minute bins detrended by a biweight of three durations (at least half a day); flares clipped.

    Searched transits last at least 2 h (12 bins), so binning first loses nothing and makes the long
    windows affordable. Protected points (other signals) are left out. Returns (time, flux, sector).
    """
    f = np.where(protect, np.nan, f) if protect is not None else f
    t, f, sector, _ = binned(t, f, sector)
    sector = np.round(sector)
    level = trend(t, f, np.isfinite(f), segments(t, sector, BIN_DAYS), max(.5, 3 * duration))
    rel = f / level - 1
    noise = 1.4826 * np.nanmedian(np.abs(rel - np.nanmedian(rel)))
    keep = np.isfinite(rel) & ((rel <= 4 * noise) if clip else True)
    return t[keep], rel[keep], sector[keep]


def binned(t, rel, sector, width=BIN_DAYS):
    """Bin means on an absolute grid, keeping the sector of each bin."""
    ok = np.isfinite(rel)
    cells = np.floor(t[ok] / width).astype(np.int64)
    keys, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
    return ((keys + .5) * width, np.bincount(inverse, rel[ok]) / counts,
            np.bincount(inverse, sector[ok]) / counts, counts)


def sector_noise(t, f, sector, duration):
    """Per-bin noise of each sector at this duration's timescale (correlated noise included)."""
    from search import duration_noise
    noise = {}
    for s in np.unique(sector):
        select = sector == s
        if select.sum() > 50:
            noise[s] = float(duration_noise(t[select], f[select], [duration])[0])
    return noise


def device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def reference(t, f, sector, noise):
    """Binned series on the GPU (float64) with cumulative sums and per-bin noise, for window means."""
    import torch
    sigma = np.array([noise.get(s, np.inf) for s in np.round(sector)])
    as_tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device())
    return {'t': t, 'f': f, 'sector': sector, 'noise': noise,
            'gt': as_tensor(t), 'gsigma': as_tensor(sigma), 'gsums': as_tensor(np.r_[0, np.cumsum(f)])}


def window_stats(ref, centers, duration):
    """Bins, depth and depth error of the window of one duration around each center (tensors)."""
    import torch
    t = ref['gt']
    lo = torch.searchsorted(t, centers - duration / 2)
    hi = torch.searchsorted(t, centers + duration / 2, right=True)
    n = (hi - lo).to(torch.float64)
    depth = -(ref['gsums'][hi] - ref['gsums'][lo]) / n.clamp(min=1)
    error = ref['gsigma'][lo.clamp(max=len(t) - 1)] / n.clamp(min=1).sqrt()
    return n, depth, error


def event_scan(ref, duration):
    """Single-event SNR and depth with the window centered on every bin."""
    n, depth, error = window_stats(ref, ref['gt'], duration)
    snr = depth / error
    snr[n < MIN_COVERAGE * duration / BIN_DAYS] = -float('inf')
    return snr.cpu().numpy(), depth.cpu().numpy()


def search_events(star, masks=(), invert=False, max_events=MAX_EVENTS):
    """Isolated transit-like events: best matched-filter SNR over durations, masking each one found."""
    t, f, sector = star['time_bjd'], star['flux'], star['sector']
    f = 2 * np.nanmedian(f) - f if invert else f
    protected = np.zeros(len(t), bool)
    for p, e, d in masks:
        protected |= np.abs(wrap(t - e, p)) < d / 24
    series = []
    for hours in SINGLE_HOURS:
        duration = hours / 24
        tb, fb, sb = detrended(t, f, sector, duration, protected)
        snr, depth = event_scan(reference(tb, fb, sb, sector_noise(tb, fb, sb, duration)), duration)
        series.append((duration, tb, sb, snr, depth))
    events = []
    blocked = [np.zeros(len(s[1]), bool) for s in series]
    for _ in range(max_events):
        best = None
        for i, (duration, tb, sb, snr_all, depth) in enumerate(series):
            snr = np.where(blocked[i] | ~np.isfinite(snr_all), -np.inf, snr_all)
            if not len(snr):
                continue
            j = int(np.argmax(snr))
            if np.isfinite(snr[j]) and (best is None or snr[j] > best['snr']):
                best = {'snr': float(snr[j]), 'epoch_bjd': float(tb[j]), 'duration_hours': float(duration * 24),
                        'depth_ppm': float(depth[j] * 1e6), 'sector': int(round(sb[j]))}
        if best is None or (best['snr'] < MIN_SINGLE_SNR and not invert):
            break
        events.append(best)
        # Mask the event and its surroundings (1.5 durations each side) in every duration's series.
        for i, (_, tb, *_rest) in enumerate(series):
            blocked[i] |= np.abs(tb - best['epoch_bjd']) < 2 * best['duration_hours'] / 24
    return events


def refine_event(ref, event):
    """Center and duration of an event on a fine grid (10-minute steps, durations 0.5–1.5× the detected
    one): detection only tests a few durations, and a box longer than the transit sits off-center,
    which would bias the symmetry, coverage and shape tests."""
    import torch
    duration = event['duration_hours'] / 24
    dev = ref['gt'].device
    centers = torch.as_tensor(event['epoch_bjd'] + np.arange(-duration / 2, duration / 2 + 1e-9, BIN_DAYS),
                              dtype=torch.float64, device=dev)
    best = None
    # Within the searched durations: below 2 h, flares (inverted), cosmic rays and jitter dominate.
    trials = np.clip(duration * np.linspace(.5, 1.5, 21), SINGLE_HOURS[0] / 24, SINGLE_HOURS[-1] / 24)
    for d in np.unique(trials):
        n, depth, error = window_stats(ref, centers, d)
        snr = torch.where(n >= MIN_COVERAGE * d / BIN_DAYS, depth / error, torch.full_like(depth, -float('inf')))
        j = int(torch.argmax(snr))
        if best is None or float(snr[j]) > best[0]:
            best = (float(snr[j]), float(centers[j]), d, float(depth[j]))
    if best is None or not np.isfinite(best[0]):
        return event
    return {**event, 'snr': best[0], 'epoch_bjd': best[1], 'duration_hours': best[2] * 24, 'depth_ppm': best[3] * 1e6}


def density_period(duration_days, depth, star):
    """Period of a central circular transit with this duration around this star (days), or nan."""
    density = stellar_density(star)
    if not np.isfinite(density):
        return np.nan
    k = np.sqrt(max(depth, 0.))
    # T = P/π · (R*/a)(1+k) with a/R* = (Gρ P²/3π)^(1/3)  ⇒  P = (πT/(1+k))³ Gρ / (3π).
    seconds = (np.pi * duration_days * DAY / (1 + k)) ** 3 * G * density / (3 * np.pi)
    return float(seconds / DAY)


def coverage(ref, epoch, periods, duration, depth, skip):
    """For each trial period, the predicted transits on covered data, all periods at once on the GPU.

    Returns (excluded, supported) per period: a covered predicted transit without a dip (measured
    depth more than 3σ below the event depth) excludes the period; one with a consistent dip supports it.
    """
    import torch
    t = ref['t']
    dev = ref['gt'].device
    p = torch.as_tensor(np.asarray(periods, float), dtype=torch.float64, device=dev)
    first = torch.ceil((t[0] - epoch) / p)
    count = (torch.floor((t[-1] - epoch) / p) - first + 1).clamp(min=0).long()
    owner = torch.repeat_interleave(torch.arange(len(p), device=dev), count)
    start = torch.cumsum(count, 0) - count
    k = first[owner] + (torch.arange(len(owner), device=dev) - start[owner])
    centers = epoch + k * p[owner]
    keep = torch.ones_like(centers, dtype=torch.bool)
    for s in skip:
        keep &= (centers - s).abs() >= duration
    n, measured, error = window_stats(ref, centers, duration)
    covered = keep & (n >= MIN_COVERAGE * duration / BIN_DAYS)
    against = covered & (measured < depth - 3 * error)
    zeros = torch.zeros(len(p), dtype=torch.float64, device=dev)
    excluded = zeros.scatter_add(0, owner, against.double()) > 0
    supported = zeros.scatter_add(0, owner, (covered & ~against).double())
    return excluded.cpu().numpy(), supported.cpu().numpy().astype(int)


def single_periods(ref, event, star):
    """Periods allowed by the coverage for an isolated event, and the density-implied period."""
    t = ref['t']
    duration, depth = event['duration_hours'] / 24, event['depth_ppm'] / 1e6
    baseline = t[-1] - t[0]
    grid = np.exp(np.linspace(np.log(MIN_ALIAS_DAYS), np.log(4 * baseline), 3000))
    excluded, _ = coverage(ref, event['epoch_bjd'], grid, duration, depth, [event['epoch_bjd']])
    allowed = ~excluded
    return {'min_period_days': float(grid[allowed][0]) if allowed.any() else None,
            'allowed_fraction_below_baseline': float(allowed[grid < baseline].mean()),
            'density_period_days': density_period(duration, depth, star)}


def pair_aliases(ref, a, b, star):
    """Allowed periods Δt/n for two events with consistent shape, best (most plausible) first."""
    duration = max(a['duration_hours'], b['duration_hours']) / 24
    depth = min(a['depth_ppm'], b['depth_ppm']) / 1e6
    delta = abs(b['epoch_bjd'] - a['epoch_bjd'])
    p_density = density_period(duration, depth, star)
    numbers = np.arange(1, int(delta / MIN_ALIAS_DAYS) + 1)
    if not len(numbers):
        return [], p_density
    periods = delta / numbers
    excluded, supported = coverage(ref, a['epoch_bjd'], periods, duration, depth, [a['epoch_bjd'], b['epoch_bjd']])
    out = [{'period_days': float(period), 'n': int(n), 'extra_transits': int(extra),
            'density_disfavoured': bool(np.isfinite(p_density) and period * DENSITY_ALLOWANCE < p_density)}
           for period, n, extra, bad in zip(periods, numbers, supported, excluded) if not bad]
    # Most plausible first: not disfavoured, then the shortest (highest transit probability).
    out.sort(key=lambda x: (x['density_disfavoured'], x['period_days']))
    return out, p_density


def consistent_pair(a, b):
    if max(a['duration_hours'], b['duration_hours']) / min(a['duration_hours'], b['duration_hours']) > PAIR_DURATION_RATIO:
        return False
    error = np.hypot(a['depth_ppm'] / a['snr'], b['depth_ppm'] / b['snr'])
    return abs(a['depth_ppm'] - b['depth_ppm']) <= PAIR_SIGMA * error


def edge_distance(t, epoch):
    """Days from the event to the nearest data gap (> 0.5 d) or segment end."""
    gaps = np.where(np.diff(t) > .5)[0]
    edges = np.r_[t[0], t[gaps], t[gaps + 1], t[-1]]
    return float(np.min(np.abs(edges - epoch)))


def asymmetry(ref, epoch, duration):
    """Difference between the first and second half of the event: (σ, fraction of the depth).
    Ramps and steps are lopsided; a transit is symmetric."""
    t, f, sector, noise = ref['t'], ref['f'], ref['sector'], ref['noise']
    first = (t > epoch - duration / 2) & (t <= epoch)
    second = (t > epoch) & (t < epoch + duration / 2)
    if first.sum() < 3 or second.sum() < 3:
        return np.nan, np.nan
    s = int(round(np.median(sector[first | second])))
    sigma = noise.get(s, np.inf) * np.sqrt(1 / first.sum() + 1 / second.sum())
    a, b = np.mean(f[first]), np.mean(f[second])
    depth = -np.mean(f[first | second])
    return float(abs(a - b) / sigma), float(abs(a - b) / depth) if depth > 0 else np.inf


# ---- Systematics shared by stars on the same CCD (common mode) ----

def star_common_series(tic):
    """Detrended 30-minute series of every sector of a star, catalogued transits masked, keyed by CCD."""
    from astropy.io import fits
    from data import known_signals, ROOT
    from search import known_masks
    star = load_star(tic)
    t, f, sector = star['time_bjd'], star['flux'], star['sector']
    protected = np.zeros(len(t), bool)
    for p, e, d in known_masks(known_signals(tic)):
        protected |= np.abs(wrap(t - e, p)) < d / 24
    # 10-minute bins first: the common mode is a 30-minute series, so this loses nothing.
    f = np.where(protected, np.nan, f)
    t, f, sector, _ = binned(t, f, sector)
    sector = np.round(sector)
    level = trend(t, f, np.isfinite(f), segments(t, sector, BIN_DAYS), COMMON_WINDOW)
    rel = f / level - 1
    out = []
    for s in np.unique(sector):
        header = fits.getheader(ROOT / star['meta'][int(s)]['fits'])
        select = sector == s
        tb, fb, _, _ = binned(t[select], rel[select], sector[select], COMMON_BIN_DAYS)
        scale = 1.4826 * np.nanmedian(np.abs(fb - np.nanmedian(fb)))
        if np.isfinite(scale) and scale > 0:
            # Normalized, so bright and faint stars weigh alike in the median.
            out.append(((int(s), int(header['CAMERA']), int(header['CCD'])),
                        np.floor(tb / COMMON_BIN_DAYS).astype(np.int64), fb / scale))
    return out


def build_common_mode(stars, workers=None):
    """Median normalized flux of all downloaded stars per (sector, camera, CCD) and 30-minute bin."""
    from collections import defaultdict
    from concurrent.futures import ProcessPoolExecutor
    from lightcurve import SPAWN, GPU_WORKERS
    groups = defaultdict(list)
    with ProcessPoolExecutor(max_workers=workers or GPU_WORKERS, mp_context=SPAWN) as pool:
        for n, parts in enumerate(pool.map(star_common_series, stars, chunksize=4), 1):
            for key, bins, values in parts:
                groups[key].append((bins, values))
            if n % 200 == 0 or n == len(stars):
                print(f'Common mode: {n}/{len(stars)} stars', flush=True)
    import torch
    arrays = {}
    for (s, cam, ccd), parts in groups.items():
        # Stars × bins matrix (NaN where a star has no data), median over stars on the GPU.
        keys = np.unique(np.concatenate([b for b, _ in parts]))
        matrix = np.full((len(parts), len(keys)), np.nan)
        for row, (bins, values) in enumerate(parts):
            matrix[row, np.searchsorted(keys, bins)] = values
        values = torch.as_tensor(matrix, dtype=torch.float64, device=device())
        name = f's{s}c{cam}d{ccd}'
        arrays[name + '_bins'] = keys
        arrays[name + '_median'] = values.nanmedian(0).values.cpu().numpy()
        arrays[name + '_stars'] = (~values.isnan()).sum(0).cpu().numpy()
    np.savez(COMMON_MODE, stars=np.array(sorted(stars)), **arrays)


_common = None


def common_mode():
    global _common
    if _common is None:
        with np.load(COMMON_MODE) as data:
            _common = {k: data[k] for k in data.files}
    return _common


def ccd_of(star, sector):
    from astropy.io import fits
    from data import ROOT
    header = fits.getheader(ROOT / star['meta'][int(sector)]['fits'])
    return int(header['CAMERA']), int(header['CCD'])


def common_mode_sigma(star, period, epoch, duration):
    """Significance of a dip in the CCD-wide median flux at the event times (instrumental if high).

    Every covered transit is stacked; the error uses the scatter of the median series at the
    duration's timescale. Returns nan when too few stars share the CCD.
    """
    cm = common_mode()
    inside, around = [], []
    for s in np.unique(star['sector']):
        times = star['time_bjd'][star['sector'] == s]
        centers = epoch + period * np.arange(np.ceil((times[0] - epoch) / period - .5),
                                             np.floor((times[-1] - epoch) / period + .5) + 1)
        if not len(centers):
            continue
        cam, ccd = ccd_of(star, s)
        name = f's{int(s)}c{cam}d{ccd}'
        if name + '_bins' not in cm:
            continue
        tb = (cm[name + '_bins'] + .5) * COMMON_BIN_DAYS
        good = cm[name + '_stars'] >= COMMON_MIN_STARS
        values = cm[name + '_median']
        for c in centers:
            near = good & (np.abs(tb - c) < duration / 2 + 1.5 * max(duration, .25))
            core = near & (np.abs(tb - c) < duration / 2)
            if core.sum() >= 2 and (near & ~core).sum() >= 4:
                inside.append(values[core])
                around.append(values[near & ~core])
    if not inside:
        return np.nan
    values_in, values_out = np.concatenate(inside), np.concatenate(around)
    scatter = 1.4826 * np.median(np.abs(values_out - np.median(values_out)))
    # Correlated noise: scale by the ratio of binned to white scatter at the duration's timescale.
    per = max(1, int(round(duration / COMMON_BIN_DAYS)))
    if len(values_out) >= 4 * per:
        blocks = values_out[:len(values_out) // per * per].reshape(-1, per).mean(1)
        red = 1.4826 * np.median(np.abs(blocks - np.median(blocks))) * np.sqrt(per)
        scatter = max(scatter, red)
    if not scatter > 0:
        return np.nan
    dip = np.median(values_out) - np.mean(values_in)
    return float(dip / (scatter * np.sqrt(1 / len(values_in) + 1 / len(values_out))))


def add_common_mode(star, metrics, period, epoch, duration_hours):
    """Common-mode test as a vetting flag on any event (BLS, single or duo)."""
    sigma = common_mode_sigma(star, period, epoch, duration_hours / 24) if COMMON_MODE.exists() else np.nan
    metrics['common_mode_sigma'] = sigma if np.isfinite(sigma) else None
    if np.isfinite(sigma) and sigma > COMMON_SIGMA:
        metrics['flags']['systematic'] = FAIL
    elif np.isfinite(sigma) and sigma > 3:
        metrics['flags']['systematic'] = WARN
    metrics['verdict'] = FAIL if FAIL in metrics['flags'].values() else 'pass'


# ---- Per-star pipeline ----

def event_sectors(star, times, margin=1.):
    """Only the sectors holding the given event times: local noise, and no detrending of years of data
    to examine a few hours."""
    from data import select_sectors
    near = np.zeros(len(star['time_bjd']), bool)
    for t0 in times:
        near |= np.abs(star['time_bjd'] - t0) < margin
    return select_sectors(star, sorted(int(s) for s in np.unique(star['sector'][near])))


def centroid_shifts(star, duration):
    """Standardized centroid shift of a window of this duration at every time, per axis.

    Each sector is detrended on its own (biweight of three durations, 10-minute bins) and its
    window means are divided by that sector's robust spread. Returns {axis: (time, z, sector)}.
    """
    out = {}
    for axis, key in enumerate(('centr1', 'centr2')):
        times, scores, sectors = [], [], []
        for s in np.unique(star['sector']):
            mask = star['sector'] == s
            values = star[key][mask]
            if np.isfinite(values).sum() < 200:
                continue
            # No clipping: a real centroid shift must not be cut as an outlier.
            tb, cb, sb = detrended(star['time_bjd'][mask], values, star['sector'][mask], duration, clip=False)
            if len(tb) < 50:
                continue
            ref = reference(tb, cb * np.nanmedian(values), sb, {float(s): 1.})
            n, means, _ = window_stats(ref, ref['gt'], duration)
            n, means = n.cpu().numpy(), -means.cpu().numpy()
            full = n >= MIN_COVERAGE * duration / BIN_DAYS
            if full.sum() < 20:
                continue
            center = np.median(means[full])
            spread = 1.4826 * np.median(np.abs(means[full] - center))
            if spread > 0:
                times.append(tb[full])
                scores.append((means[full] - center) / spread)
                sectors.append(np.full(full.sum(), int(s)))
        if times:
            out[axis] = (np.concatenate(times), np.concatenate(scores), np.concatenate(sectors))
    return out


def tail_probability(null, x):
    """P(|z| ≥ x) from the null sample; beyond its range, from a generalized Pareto fit to the top 2%
    (extreme value theory), so the significance is not capped by the sample size."""
    from scipy import stats
    exceed = int(np.sum(null >= x))
    if exceed >= 10:
        return (exceed + 1) / (len(null) + 1)
    u = np.quantile(null, .98)
    tail = null[null > u] - u
    if len(tail) < 20 or x <= u:
        return (exceed + 1) / (len(null) + 1)
    shape, _, scale = stats.genpareto.fit(tail, floc=0)
    return float(max(.02 * stats.genpareto.sf(x - u, shape, 0, scale), 1e-300))


def centroid_single(star, epochs, duration, depth):
    """Transit source test for one or two events, with an empirical significance.

    The centroid wanders on hour timescales (pointing, momentum dumps): this averages out over many
    transits but not over one, and its distribution has heavy tails. The shift at each event is
    ranked among the shifts of the same window at every other time of all the star's sectors
    (each standardized by its sector's spread), with an extreme-value tail beyond the sample; the
    p-values of both axes and all events are combined (Fisher). The offset of the implied source from the target, Δc = δ (c − x_s) / (1 − δ),
    is reported as in the periodic test. Returns (sigma, offset arcsec).
    """
    from scipy import stats
    if 'centr1' not in star or not depth > 0:
        return np.nan, np.nan
    shifts = centroid_shifts(star, duration)
    if len(shifts) < 2:
        return np.nan, np.nan
    logs, offset = [], [[], []]
    for t0 in epochs:
        for axis, (t, z, sector) in shifts.items():
            here = np.abs(t - t0) < BIN_DAYS
            null = np.abs(z[np.all([np.abs(t - e) > 2 * duration for e in epochs], axis=0)])
            if not here.any() or len(null) < 100:
                continue
            score = float(np.mean(z[here]))
            p = tail_probability(null, abs(score))
            logs.append(np.log(p))
            s = int(np.median(sector[here]))
            info = star['meta'].get(s, {})
            if info.get('target_pixel'):
                aperture_depth = depth * (info.get('crowdsap') or 1.)
                near = (star['sector'] == s) & (np.abs(star['time_bjd'] - t0) < 3 * duration)
                values = star['centr1' if axis == 0 else 'centr2'][near]
                inside = np.abs(star['time_bjd'][near] - t0) < duration / 4
                around = np.abs(star['time_bjd'][near] - t0) > duration
                if inside.sum() >= 3 and around.sum() >= 10:
                    shift = np.nanmean(values[inside]) - np.nanmedian(values[around])
                    source = np.nanmedian(values[around]) - shift * (1 - aperture_depth) / aperture_depth
                    offset[axis].append(source - info['target_pixel'][axis])
    if not logs:
        return np.nan, np.nan
    chi2 = -2 * float(np.sum(logs))
    sigma = stats.norm.isf(max(stats.chi2.sf(chi2, 2 * len(logs)), 1e-300) / 2)
    arcsec = (float(np.hypot(np.mean(offset[0]), np.mean(offset[1])) * PIXEL_ARCSEC)
              if offset[0] and offset[1] else np.nan)
    return float(min(sigma, 40.)), arcsec


def apply_centroid(star, metrics, epochs, duration_hours):
    """Replace the periodic centroid test by the empirical one for single and duo events."""
    sigma, arcsec = centroid_single(star, epochs, duration_hours / 24, (metrics['depth_ppm'] or 0) / 1e6)
    metrics['centroid_sigma'] = sigma if np.isfinite(sigma) else None
    metrics['source_offset_arcsec'] = arcsec if np.isfinite(arcsec) else None
    metrics['flags'].pop('centroid_offset', None)
    if sigma > CENTROID_SIGMA and arcsec > PIXEL_ARCSEC:
        metrics['flags']['centroid_offset'] = FAIL
    elif sigma > 3 and arcsec > PIXEL_ARCSEC / 2:
        metrics['flags']['centroid_offset'] = WARN


def long_period_flags(metrics):
    """Tests that do not apply to one or two events are dropped; a V shape fails, because without
    odd/even and secondary tests it is the only handle on grazing eclipsing binaries."""
    flags = metrics['flags']
    for key in ('single_transit', 'odd_even', 'duration_too_long'):
        flags.pop(key, None)
    if 'v_shape' in flags:
        flags['v_shape'] = FAIL


def vet_single(star, event, masks, ref, ceiling):
    """Vetting of an isolated event: shared tests from analyze() plus edge, symmetry and systematics."""
    baseline = star['time_bjd'][-1] - star['time_bjd'][0]
    # A period longer than twice the baseline leaves exactly this event in the data.
    period = 2.5 * baseline
    x, metrics = analyze(event_sectors(star, [event['epoch_bjd']]), period, event['epoch_bjd'],
                         event['duration_hours'], masks)
    flags = metrics['flags']
    long_period_flags(metrics)
    apply_centroid(star, metrics, [event['epoch_bjd']], event['duration_hours'])
    # An isolated event has no secondary phase to test.
    flags.pop('secondary_eclipse', None)
    metrics['noise_ceiling_snr'] = ceiling
    if ceiling and event['snr'] < MIN_CEILING_RATIO * ceiling:
        flags['below_noise_ceiling'] = FAIL
    duration = event['duration_hours'] / 24
    metrics['edge_distance_days'] = edge_distance(star['time_bjd'], event['epoch_bjd'])
    if metrics['edge_distance_days'] < EDGE_DAYS:
        flags['edge'] = FAIL
    metrics['asymmetry_sigma'], metrics['asymmetry_fraction'] = asymmetry(ref, event['epoch_bjd'], duration)
    if metrics['asymmetry_sigma'] > ASYMMETRY_SIGMA and metrics['asymmetry_fraction'] > ASYMMETRY_FRACTION:
        flags['asymmetric'] = FAIL
    add_common_mode(star, metrics, period, event['epoch_bjd'], event['duration_hours'])
    return metrics


def single_star_events(star, masks, known):
    """Single and duo transit events of one star, vetted, with catalogue matches."""
    from search import match_single
    events = search_events(star, masks)
    if not events:
        return []
    protected = np.zeros(len(star['time_bjd']), bool)
    for p, e, d in masks:
        protected |= np.abs(wrap(star['time_bjd'] - e, p)) < d / 24
    mirror = search_events(star, masks, invert=True, max_events=1)
    ceiling = None
    if mirror:
        # Refined like the real events, so both sides of the comparison had the same number of trials.
        flipped = 2 * np.nanmedian(star['flux']) - star['flux']
        hours = mirror[0]['duration_hours']
        t, f, sector = detrended(star['time_bjd'], flipped, star['sector'], hours / 24, protected)
        ceiling = refine_event(reference(t, f, sector, sector_noise(t, f, sector, hours / 24)), mirror[0])['snr']
    refs = {}

    def ref_for(hours):
        """Reference series detrended for this duration (a shorter window would erode the transit)."""
        if hours not in refs:
            t, f, sector = detrended(star['time_bjd'], star['flux'], star['sector'], hours / 24, protected)
            refs[hours] = reference(t, f, sector, sector_noise(t, f, sector, hours / 24))
        return refs[hours]
    span = 2.5 * (star['time_bjd'][-1] - star['time_bjd'][0])
    out = []
    events = [refine_event(ref_for(e['duration_hours']), e) for e in events]
    for i, event in enumerate(events):
        others = [(span, e['epoch_bjd'], e['duration_hours']) for j, e in enumerate(events) if j != i]
        ref = ref_for(event['duration_hours'])
        try:
            metrics = vet_single(star, event, masks + others, ref, ceiling)
        except ValueError as exc:
            out.append({'search': 'single', **event, 'period_days': None, 'error': str(exc)})
            continue
        periods = single_periods(ref, event, metrics['star'])
        out.append({'search': 'single', **event, 'period_days': None, 'periods': periods,
                    'known': match_single(event, known), 'masks': masks + others, 'metrics': metrics})
    out += duo_events(star, [e for e in out if 'metrics' in e], masks, known, ref_for, span)
    return out


def duo_events(star, singles, masks, known, ref_for, span):
    """Pairs of vetted single events with consistent shape and at least one allowed period."""
    from search import match_known, match_single
    passing = [e for e in singles if e['metrics']['verdict'] == 'pass']
    duos, used = [], set()
    pairs = [(i, j) for i in range(len(passing)) for j in range(i + 1, len(passing))
             if consistent_pair(passing[i], passing[j])]
    # Strongest pairs first; each event joins at most one duo.
    pairs.sort(key=lambda ij: -(passing[ij[0]]['snr'] ** 2 + passing[ij[1]]['snr'] ** 2))
    for i, j in pairs:
        if i in used or j in used:
            continue
        a, b = sorted((passing[i], passing[j]), key=lambda e: e['epoch_bjd'])
        aliases, p_density = pair_aliases(ref_for(max(a['duration_hours'], b['duration_hours'])), a, b,
                                          a['metrics']['star'])
        if not aliases:
            continue
        period = aliases[0]['period_days']
        duration_hours = max(a['duration_hours'], b['duration_hours'])
        event = {'period_days': period, 'epoch_bjd': a['epoch_bjd'], 'duration_hours': duration_hours}
        others = [(span, e['epoch_bjd'], e['duration_hours']) for e in singles if e is not a and e is not b]
        try:
            x, metrics = analyze(event_sectors(star, [a['epoch_bjd'], b['epoch_bjd']]), period, a['epoch_bjd'],
                                 duration_hours, masks + others)
        except ValueError as exc:
            duos.append({'search': 'duo', **event, 'error': str(exc)})
            continue
        # The secondary search assumes the period is right; with several aliases it is only a warning.
        if metrics['flags'].get('secondary_eclipse') == FAIL and len(aliases) > 1:
            metrics['flags']['secondary_eclipse'] = WARN
        long_period_flags(metrics)
        apply_centroid(star, metrics, [a['epoch_bjd'], b['epoch_bjd']], duration_hours)
        # Both events were tested against the noise ceiling as singles.
        add_common_mode(star, metrics, period, a['epoch_bjd'], duration_hours)
        used |= {i, j}
        known_match = match_known(event, known) or match_single(a, known) or match_single(b, known)
        duos.append({'search': 'duo', **event, 'snr': float(np.hypot(a['snr'], b['snr'])),
                     'events_bjd': [a['epoch_bjd'], b['epoch_bjd']], 'aliases': aliases[:20],
                     'alias_count': len(aliases), 'density_period_days': p_density,
                     'known': known_match, 'masks': masks + others, 'metrics': metrics})
    return duos
