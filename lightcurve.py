"""Photometric preparation: transit-protected detrending, phase folding and CNN views."""

# CNN input: (view, channels, bins); even channels = flux, odd channels = coverage.
VIEWS = (('global', 2, 101), ('local', 2, 61), ('odd_even', 4, 61), ('secondary', 2, 61))


def wrap(phase, period):
    return (phase + period / 2) % period - period / 2


def trend(time, values, base, boundaries, window):
    """Running median per segment, interpolating over points outside base."""
    import numpy as np
    from scipy.ndimage import median_filter
    result = np.full(len(time), np.nan)
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        ok = base[a:b] & np.isfinite(values[a:b])
        if ok.sum() < 10:
            continue
        t = time[a:b]
        interpolated = np.interp(t, t[ok], values[a:b][ok])
        size = max(3, min(window, (b - a) // 2 * 2 - 1))
        result[a:b] = median_filter(interpolated, size=size, mode='nearest')
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
    window = max(11, int(max(1, 5 * duration) / cadence) | 1)
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

