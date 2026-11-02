"""Photometric preparation: transit-protected detrending, phase folding and CNN views."""
import multiprocessing

# Worker processes that use CUDA must be spawned (a forked child cannot initialize CUDA).
SPAWN = multiprocessing.get_context('spawn')
# BIWEIGHT_BLOCK is sized so that this many GPU processes fit in 8 GB.
GPU_WORKERS = 4

# CNN input: (view, channels, bins); even channels = flux, odd channels = coverage.
VIEWS = (('global', 2, 101), ('local', 2, 61), ('odd_even', 4, 61), ('secondary', 2, 61))


def wrap(phase, period):
    return (phase + period / 2) % period - period / 2


BIWEIGHT_C = 5.
BIWEIGHT_ITERATIONS = 10
# Elements per GPU block: ~6 float64 arrays of this size per process, so 4 parallel processes fit in 8 GB.
BIWEIGHT_BLOCK = 4e6


def trend(time, values, base, boundaries, window):
    """Tukey biweight location in a sliding time window of `window` days, per segment (the wotan method).

    Robust to outliers and to the transit itself; points outside `base` (protected transits, other
    planets) get zero weight, so no interpolation is needed. Near gaps the window simply holds fewer
    points, without padding. Runs on the GPU in float64 when available.
    """
    import numpy as np
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result = np.full(len(time), np.nan)
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        t, v = time[a:b], values[a:b]
        ok = base[a:b] & np.isfinite(v)
        if ok.sum() < 10:
            continue
        lo = np.searchsorted(t, t - window / 2)
        hi = np.searchsorted(t, t + window / 2, side='right')
        width = int((hi - lo).max())
        x_all = torch.as_tensor(np.where(ok, v, 0.), dtype=torch.float64, device=device)
        ok_all = torch.as_tensor(ok, device=device)
        offsets = torch.arange(width, device=device)
        rows = max(1, int(BIWEIGHT_BLOCK // width))
        for start in range(0, b - a, rows):
            r_lo = torch.as_tensor(lo[start:start + rows], device=device)
            r_hi = torch.as_tensor(hi[start:start + rows], device=device)
            index = r_lo[:, None] + offsets[None, :]
            inside = index < r_hi[:, None]
            index = index.clamp(max=b - a - 1)
            x = x_all[index]
            mask = inside & ok_all[index]
            location = torch.where(mask, x, torch.full_like(x, float('nan'))).nanmedian(1).values
            for _ in range(BIWEIGHT_ITERATIONS):
                deviation = torch.where(mask, (x - location[:, None]).abs(), torch.full_like(x, float('nan')))
                scale = BIWEIGHT_C * deviation.nanmedian(1).values
                u = (x - location[:, None]) / scale.clamp(min=1e-12)[:, None]
                weight = torch.where(mask & (u.abs() < 1), (1 - u ** 2) ** 2, torch.zeros_like(u))
                total = weight.sum(1)
                updated = (weight * x).sum(1) / total.clamp(min=1e-300)
                location = torch.where(total > 0, updated, location)
            result[a + start:a + start + len(location)] = location.cpu().numpy()
    return result


def segments(time, sector, cadence):
    import numpy as np
    jumps = (np.diff(time) > max(0.5, 10 * cadence)) | (np.diff(sector) != 0)
    return np.r_[0, np.where(jumps)[0] + 1, len(time)]


def prepare(time, flux, period, epoch, duration_hours, sector=None, masks=()):
    """Detrend while protecting the transit and masking the other known planets of the star."""
    import numpy as np
    time, flux = np.asarray(time, dtype=float), np.asarray(flux, dtype=float)
    sector = np.zeros(len(time)) if sector is None else np.asarray(sector)
    index = np.argsort(time, kind='stable')
    index = index[np.isfinite(time[index]) & np.isfinite(flux[index])]
    time, flux, sector = time[index], flux[index], sector[index]
    duration = duration_hours / 24
    if len(time) < 200 or not (0 < duration < period / 2):
        raise ValueError('Invalid coverage or duration')
    phase = wrap(time - epoch, period)
    inside = np.abs(phase) < duration / 2
    others = np.zeros(len(time), dtype=bool)
    for p, e, d in masks:
        others |= np.abs(wrap(time - e, p)) < d / 24
    outside = (np.abs(phase) > duration) & ~others
    if (inside & ~others).sum() < 5 or outside.sum() < 100:
        raise ValueError('Not enough coverage of the predicted transit')
    cadence = np.median(np.diff(time)[np.diff(time) > 0])
    # Window of 3 transit durations (wotan recommendation), at least half a day; the transit has zero weight.
    window = max(.5, 3 * duration)
    base = trend(time, flux, outside, segments(time, sector, cadence), window)
    corrected = np.divide(flux, base, out=np.full(len(flux), np.nan), where=np.isfinite(base) & (base != 0)) - 1
    corrected[others] = np.nan
    valid = np.isfinite(corrected)
    if (valid & inside).sum() < 5 or (valid & outside).sum() < 100:
        raise ValueError('Not enough coverage after detrending')
    out = corrected[valid & outside]
    scatter = max(1.4826 * np.median(np.abs(out - np.median(out))), 1e-9)
    return {'time': time, 'phase': phase, 'corrected': corrected, 'normalized': corrected / scatter,
            'valid': valid, 'outside': outside & valid, 'scatter': scatter, 'cadence': cadence,
            'period': period, 'epoch': epoch, 'duration': duration, 'sector': sector, 'index': index,
            'window': window}


def views(prep):
    """Folded profiles in VIEWS layout: global, local, odd/even and secondary."""
    import numpy as np
    phase, period, duration = prep['phase'], prep['period'], prep['duration']
    normalized, valid = prep['normalized'], prep['valid']
    odd = np.round((prep['time'] - prep['epoch']) / period) % 2 == 1

    def bins(x, edges, select=True):
        values, mask = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            chosen = valid & select & (x >= lo) & (x < hi)
            mask.append(float(chosen.any()))
            values.append(float(np.median(normalized[chosen])) if chosen.any() else 0.)
        return values + mask
    local = np.linspace(-3, 3, 62)
    secondary = (phase % period) - period / 2
    return np.array(bins(phase / period, np.linspace(-.5, .5, 102)) + bins(phase / duration, local)
                    + bins(phase / duration, local, odd) + bins(phase / duration, local, ~odd)
                    + bins(secondary / duration, local), dtype=float)

