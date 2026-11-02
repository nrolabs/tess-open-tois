"""False positive probability with TRICERATOPS; runs in the isolated .venv-fpp environment."""
import json
import sys
import time

import numpy as np

START = time.time()


def log(step):
    print(f'[{time.time() - START:7.1f} s] {step}', file=sys.stderr, flush=True)


class LocalCutout:
    """Stands in for a TESScut download, built from our SPOC light curve file.

    TRICERATOPS downloads the full FFI cutout cube of every sector (thousands of frames) but only uses
    its WCS and CCD corner to place the neighbors on the pixel grid, plus the mean image for plots.
    The APERTURE extension of the SPOC file has the same WCS and corner (CRVAL1P/2P), the ones
    TRICERATOPS itself uses for the SPOC aperture, so the geometry is identical and nothing is
    downloaded. The 'image' is the aperture bitmask; it is only used by TRICERATOPS plots.
    """

    def __init__(self, path):
        from types import SimpleNamespace
        from astropy.io import fits
        with fits.open(path) as hdul:
            header = hdul['APERTURE'].header.copy()
            image = hdul['APERTURE'].data.astype(float)
        corner = fits.Header({'1CRV4P': header['CRVAL1P'], '2CRV4P': header['CRVAL2P']})
        self.hdu = [None, SimpleNamespace(header=corner, data={'FLUX': image[None]}),
                    SimpleNamespace(header=header)]


def use_local_files(paths):
    """Route TRICERATOPS' TESScut and SPOC aperture downloads to the local SPOC light curve files."""
    import lightkurve
    import triceratops.funcs as funcs
    from types import SimpleNamespace

    def search_tesscut(target, sector):
        return SimpleNamespace(download_all=lambda **_: [LocalCutout(paths[str(int(sector))])])
    lightkurve.search_tesscut = search_tesscut
    funcs.find_url = lambda ID, sector: paths[str(int(sector))]


def main(source, target):
    import triceratops.triceratops as tr
    job = json.loads(open(source).read())
    sectors = np.array(job['sectors'])
    use_local_files(job['fits'])
    log(f"querying TIC/Gaia for TIC {job['tic']}; pixel geometry from local SPOC files, sectors {list(sectors)}")
    star = tr.target(ID=job['tic'], sectors=sectors)
    # TRICERATOPS silently skips sectors that raise, leaving fewer pixel coordinate sets than sectors;
    # calc_depths would then fail with an opaque IndexError.
    if len(star.pix_coords) < len(sectors):
        raise SystemExit(f'Pixel geometry failed for {len(sectors) - len(star.pix_coords)} of '
                         f'{len(sectors)} sectors')
    log(f'{len(star.stars)} stars in the field; fetching SPOC apertures')
    apertures = star.get_spoc_apertures()
    keep = [i for i, a in enumerate(apertures) if a is not None]
    if not keep:
        raise SystemExit('No SPOC apertures for the sectors')
    if len(keep) < len(sectors):
        star = tr.target(ID=job['tic'], sectors=sectors[keep])
        apertures = [apertures[i] for i in keep]
    log('computing the depth each star would need')
    star.calc_depths(tdepth=job['depth'], all_ap_pixels=apertures)
    # The result is Monte Carlo: several runs give mean and spread.
    runs = []
    for run in range(job.get('runs', 3)):
        log(f'Monte Carlo {run + 1}')
        star.calc_probs(time=np.array(job['time']), flux_0=np.array(job['flux']),
                        flux_err_0=job['flux_err'], P_orb=job['period'], parallel=True, verbose=0)
        runs.append([float(star.FPP), float(star.NFPP)])
    runs = np.array(runs)
    probs = star.probs.sort_values('prob', ascending=False).head(5)
    result = {'fpp': float(runs[:, 0].mean()), 'nfpp': float(runs[:, 1].mean()),
              'fpp_std': float(runs[:, 0].std()), 'nfpp_std': float(runs[:, 1].std()),
              'sectors_used': [int(s) for s in sectors[keep]],
              'top_scenarios': [{'id': str(r['ID']), 'scenario': r['scenario'], 'prob': float(r['prob'])}
                                for _, r in probs.iterrows()]}
    log('done')
    with open(target, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main(*sys.argv[1:3])
