"""Difference images from SPOC target pixel files: which pixel the transit comes from.

The light curve files hold only the aperture sum and its centroid. The target pixel file (TPF) holds
every pixel at every cadence, so the image out of transit minus the image in transit shows where
the light was lost. A Gaussian fit to that difference image locates the source; its offset from the
target's catalogue position (WCS of the same SPOC stamp) is the pixel-level source test, the
analogue of the SPOC DV difference image centroid offsets. TPFs (~50 MB per sector) are downloaded
only for signals that pass vetting, for up to three sectors that contain transits.
"""
import numpy as np
from data import DATA, ROOT, aws_url, parallel_download
from vetting import PIXEL_ARCSEC, FAIL, WARN

TPF_DIR = DATA / 'mast' / 'tp'
MAX_SECTORS = 3
# Difference image significance below which the location is not trusted (noise-dominated image).
MIN_DIFFERENCE_SNR = 5.
# A source this far from the target, this significantly, fails; difference images resolve
# offsets well below a pixel, so the bar is half a pixel (~10″) instead of the centroid test's one.
OFFSET_SIGMA = 4.
OFFSET_PIXELS = .5
# Systematic floor of the WCS and of the Gaussian model of the TESS PSF, in pixels.
SYSTEMATIC_PIXELS = .1


def tpf_file(star, sector):
    """Local TPF for a sector, downloaded on first use (public AWS copy first, MAST as fallback)."""
    from pathlib import Path
    name = Path(star['meta'][int(sector)]['fits']).name.replace('_lc.fits', '_tp.fits')
    path = TPF_DIR / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            parallel_download(aws_url(star['tic'], sector, name), path)
        except RuntimeError:
            parallel_download(f'https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/{name}', path)
    return path


def read_tpf(path):
    """Good cadences of a TPF: BJD time, flux cube, error cube, and the target position in the stamp."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from lightkurve.utils import TessQualityFlags
    with fits.open(path) as hdul:
        data, header = hdul[1].data, hdul[1].header
        time = data['TIME'] + header['BJDREFI'] + header['BJDREFF']
        flux, error = np.asarray(data['FLUX'], float), np.asarray(data['FLUX_ERR'], float)
        good = (TessQualityFlags.create_quality_mask(data['QUALITY'], 'default') & np.isfinite(time)
                & np.isfinite(flux).any(axis=(1, 2)))
        x, y = WCS(hdul[2].header).all_world2pix(hdul[0].header['RA_OBJ'], hdul[0].header['DEC_OBJ'], 0)
    return time[good], flux[good], error[good], (float(x), float(y))


def difference_image(time, flux, centers, duration):
    """Mean (out of transit − in transit) image over the covered transits, with its per-pixel error.

    In transit: the central half of the transit. Out of transit: the same length of time just before
    and just after, beyond 0.75 durations from the center (ingress and egress excluded). The error
    is the scatter between transits when there are at least three, else the out-of-transit scatter.
    """
    diffs, errors, oot = [], [], []
    for c in centers:
        inside = np.abs(time - c) < duration / 4
        outside = (np.abs(time - c) > .75 * duration) & (np.abs(time - c) < 1.25 * duration)
        if inside.sum() < 3 or outside.sum() < 6:
            continue
        before, after = outside & (time < c), outside & (time > c)
        if not before.any() or not after.any():
            continue
        out = np.nanmean(flux[outside], axis=0)
        diffs.append(out - np.nanmean(flux[inside], axis=0))
        errors.append(np.nanstd(flux[outside], axis=0) * np.sqrt(1 / inside.sum() + 1 / outside.sum()))
        oot.append(out)
    if not diffs:
        return None
    diffs = np.array(diffs)
    if len(diffs) >= 3:
        error = np.nanstd(diffs, axis=0, ddof=1) / np.sqrt(len(diffs))
        error = np.maximum(error, np.sqrt(np.nanmean(np.square(errors), axis=0) / len(diffs)))
    else:
        error = np.sqrt(np.nansum(np.square(errors), axis=0)) / len(diffs)
    return {'difference': np.nanmean(diffs, axis=0), 'error': error, 'oot': np.nanmean(oot, axis=0),
            'transits': len(diffs)}


def fit_source(image, error):
    """Gaussian + constant fit (weighted): (x, y, σx, σy, amplitude SNR), pixel coordinates."""
    from scipy.optimize import least_squares
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    good = np.isfinite(image) & np.isfinite(error) & (error > 0)
    if good.sum() < 9:
        return None
    z, w, xg, yg = image[good], 1 / error[good], xx[good], yy[good]
    j = int(np.argmax(np.where(good, image, -np.inf)))
    start = [image.flat[j], j % nx, j // nx, .9, np.nanmedian(z)]

    def residual(p):
        amplitude, x0, y0, sigma, offset = p
        model = amplitude * np.exp(-((xg - x0) ** 2 + (yg - y0) ** 2) / (2 * sigma ** 2)) + offset
        return (model - z) * w
    lower = [0, -.5, -.5, .3, -np.inf]
    upper = [np.inf, nx - .5, ny - .5, 3., np.inf]
    start[0] = max(start[0], 1e-12)
    try:
        result = least_squares(residual, start, bounds=(lower, upper))
    except ValueError:
        return None
    dof = max(1, good.sum() - 5)
    chi2 = float(np.sum(result.fun ** 2)) / dof
    try:
        covariance = np.linalg.inv(result.jac.T @ result.jac) * max(chi2, 1.)
    except np.linalg.LinAlgError:
        return None
    errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    amplitude, x0, y0 = result.x[:3]
    return {'x': float(x0), 'y': float(y0), 'x_err': float(errors[1]), 'y_err': float(errors[2]),
            'snr': float(amplitude / errors[0]) if errors[0] > 0 else 0.}


def transit_centers(time, period, epoch):
    if period is None or period > 2 * (time[-1] - time[0]) + 1e3:
        return np.array([epoch])
    k = np.arange(np.ceil((time[0] - epoch) / period - .5), np.floor((time[-1] - epoch) / period + .5) + 1)
    return epoch + k * period


def pixel_test(star, period, epoch, duration_hours, epochs=None):
    """Difference-image source offset from the target, combined over up to three sectors.

    `epochs` lists the event times for single and duo events (their period is not reliable).
    Returns a dict with the offset (pixels and arcsec), its significance and per-sector details.
    """
    from scipy import stats
    duration = duration_hours / 24
    times = np.asarray(epochs if epochs is not None else
                       transit_centers(star['time_bjd'], period, epoch))
    sectors = []
    for s in np.unique(star['sector']):
        t = star['time_bjd'][star['sector'] == s]
        if np.any((times > t[0] - duration) & (times < t[-1] + duration)):
            sectors.append(int(s))
    details = []
    for s in sectors[-MAX_SECTORS:]:
        time, flux, _, target = read_tpf(tpf_file(star, s))
        centers = times[(times > time[0]) & (times < time[-1])]
        image = difference_image(time, flux, centers, duration)
        if image is None:
            continue
        source = fit_source(image['difference'], image['error'])
        if source is None or source['snr'] < MIN_DIFFERENCE_SNR:
            details.append({'sector': s, 'transits': image['transits'],
                            'difference_snr': source['snr'] if source else None, 'used': False})
            continue
        dx, dy = source['x'] - target[0], source['y'] - target[1]
        ex = np.hypot(source['x_err'], SYSTEMATIC_PIXELS)
        ey = np.hypot(source['y_err'], SYSTEMATIC_PIXELS)
        details.append({'sector': s, 'transits': image['transits'], 'difference_snr': source['snr'],
                        'used': True, 'dx_pixels': float(dx), 'dy_pixels': float(dy),
                        'dx_err': float(ex), 'dy_err': float(ey)})
    used = [d for d in details if d['used']]
    if not used:
        return {'sigma': None, 'offset_pixels': None, 'offset_arcsec': None, 'sectors': details}
    combined = []
    for key in ('dx', 'dy'):
        w = np.array([1 / d[f'{key}_err'] ** 2 for d in used])
        v = np.array([d[f'{key}_pixels'] for d in used])
        combined.append((float(np.sum(w * v) / w.sum()), float(1 / np.sqrt(w.sum()))))
    chi2 = sum((v / e) ** 2 for v, e in combined)
    sigma = float(min(stats.norm.isf(max(stats.chi2.sf(chi2, 2), 1e-300) / 2), 40.))
    offset = float(np.hypot(combined[0][0], combined[1][0]))
    return {'sigma': sigma, 'offset_pixels': offset, 'offset_arcsec': offset * PIXEL_ARCSEC, 'sectors': details}


def apply_pixel_test(metrics, result):
    """Pixel-level flag on the event's vetting: a significant offset of half a pixel or more fails."""
    metrics['pixel_test'] = result
    sigma, offset = result.get('sigma'), result.get('offset_pixels')
    if sigma is None:
        return
    if sigma > OFFSET_SIGMA and offset > OFFSET_PIXELS:
        metrics['flags']['pixel_offset'] = FAIL
    elif sigma > 3 and offset > .6 * OFFSET_PIXELS:
        metrics['flags']['pixel_offset'] = WARN
    metrics['verdict'] = FAIL if FAIL in metrics['flags'].values() else 'pass'
