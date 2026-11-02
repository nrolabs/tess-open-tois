"""Public TESS data: TOI and known-signal catalogs, MAST sector manifests, light curve download and loading."""
import csv
import functools
import hashlib
import io
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
CAT = DATA / 'catalogs' / 'toi.json'
# Signals outside the TOI catalog: community TOIs (ExoFOP CTOIs), published planets (Exoplanet Archive),
# SPOC TCEs (every DV run, single- and multi-sector) and the TESS eclipsing binary catalog (Prša et al. 2022).
KNOWN = DATA / 'catalogs' / 'known.json'
TCE_INDEX = 'https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_tce.html'
TCE_FILE = re.compile(r'/missions/tess/catalogs/tce/[^"]+_dvr-tcestats\.csv')
# ExoFOP TOI table: dispositions plus the number of follow-up observations of each TOI.
EXOFOP_TOI = DATA / 'catalogs' / 'exofop_toi.csv'
EXOFOP_TOI_URL = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
FOLLOW_UP = ('Time Series Observations', 'Spectroscopy Observations', 'Imaging Observations')
EB_URL = 'https://archive.stsci.edu/hlsps/tess-ebs/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv'
TAP = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync'
CTOI_URL = 'https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv'
PLANETS_QUERY = ('select pl_name, tic_id, pl_orbper, pl_orbpererr1, pl_tranmid, pl_tranmiderr1, pl_trandur, tran_flag '
                 'from pscomppars where tic_id is not null')
LABELS = {'CP': 1, 'KP': 1, 'FP': 0, 'FA': 0}
MANIFEST = 'https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_{}_lc.sh'
PRODUCT = re.compile(r'https://\S+/(tess\d+-s(\d{4})-(\d{16})-\d{4}-s_lc\.fits)$')
CURVE_VERSION = 2
FIELD_SECTORS = 30


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


@functools.lru_cache(maxsize=1)
def rows():
    return json.loads(CAT.read_text())


def fetch_text(url, query=None):
    command = ['curl', '--fail', '--silent', '--show-error', '--retry', '2', '--retry-all-errors',
               '--connect-timeout', '15', '--max-time', '300', url]
    if query:
        command[-1:] = ['-G', url, '--data-urlencode', f'query={query}', '--data-urlencode', 'format=json']
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'Download failed ({url}): {result.stderr.strip()}')
    return result.stdout


def positive(value):
    """Float if the value is a finite positive number, else None (catalog fields are often empty)."""
    import math
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def known_catalog():
    """CTOIs, published planets, eclipsing binaries and SPOC TCEs in one format.

    Period, epoch or duration may be None."""
    ctois = list(csv.DictReader(io.StringIO(fetch_text(CTOI_URL))))
    planets = json.loads(fetch_text(TAP, PLANETS_QUERY))
    if not ctois or 'CTOI' not in ctois[0] or not planets or 'tic_id' not in planets[0]:
        raise ValueError('Unexpected CTOI or planet catalog response')
    known = [{'source': 'CTOI', 'name': f"CTOI {r['CTOI']}", 'tic': int(r['TIC ID']),
              'period': positive(r['Period (days)']), 'epoch': positive(r['Transit Epoch (BJD)']),
              'period_err': positive(r['Period (days) Error']), 'epoch_err': positive(r['Transit Epoch (BJD) err']),
              'duration_hours': positive(r['Duration (hrs)']),
              'disposition': r['TFOPWG Disposition'] or r['User Disposition'] or None, 'transiting': True}
             for r in ctois]
    known += [{'source': 'planet', 'name': r['pl_name'], 'tic': int(r['tic_id'].split()[-1]),
               'period': positive(r['pl_orbper']), 'epoch': positive(r['pl_tranmid']),
               'period_err': positive(r['pl_orbpererr1']), 'epoch_err': positive(r['pl_tranmiderr1']),
               'duration_hours': positive(r['pl_trandur']), 'disposition': 'published',
               'transiting': bool(r['tran_flag'])}
              for r in planets]
    known += [{'source': 'EB', 'name': f"EB TIC {r['tess_id']}", 'tic': int(r['tess_id']),
               'period': positive(r['period']),
               'epoch': positive(r['bjd0']) and positive(r['bjd0']) + 2457000, 'duration_hours': None,
               'disposition': 'eclipsing binary',
               # Matched by period only: the detected signal may be the primary or the secondary eclipse.
               'transiting': False}
              for r in csv.DictReader(io.StringIO(fetch_text(EB_URL)))]
    known += spoc_tces()
    return known


def resume_download(url, output, attempts=6):
    """Large file over a slow link: resume partial transfers and abort only when the transfer stalls."""
    part = output.with_suffix(output.suffix + '.part')
    for _ in range(attempts):
        result = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location', '--continue-at', '-',
                                 '--connect-timeout', '15', '--speed-limit', '1000', '--speed-time', '120',
                                 '--output', str(part), url], capture_output=True, text=True)
        if result.returncode == 0:
            part.replace(output)
            return
    raise RuntimeError(f'Download failed {url}: {result.stderr.strip()}')


def parallel_download(url, output, parts=16):
    """Large file in byte ranges over parallel connections (the archive limits each connection to
    ~50 KB/s; the link carries far more). Each range resumes on its own; falls back to one stream."""
    from concurrent.futures import ThreadPoolExecutor
    head = subprocess.run(['curl', '--fail', '--silent', '--head', '--location', '--connect-timeout', '15', url],
                          capture_output=True, text=True)
    sizes = re.findall(r'(?im)^content-length:\s*(\d+)', head.stdout)
    size = int(sizes[-1]) if head.returncode == 0 and sizes else 0
    if size < 4 * 2 ** 20:
        return resume_download(url, output)
    edges = np.linspace(0, size, parts + 1).astype(int)
    pieces = [output.with_suffix(output.suffix + f'.part{i}') for i in range(parts)]

    def fetch(i):
        lo, hi = edges[i], edges[i + 1] - 1
        for _ in range(6):
            have = pieces[i].stat().st_size if pieces[i].exists() else 0
            if have >= hi - lo + 1:
                return
            result = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
                                     '--connect-timeout', '15', '--speed-limit', '1000', '--speed-time', '120',
                                     '--range', f'{lo + have}-{hi}', url], capture_output=True)
            with pieces[i].open('ab') as f:
                f.write(result.stdout)
        if pieces[i].stat().st_size < hi - lo + 1:
            raise RuntimeError(f'Download failed {url} (range {lo}-{hi})')
    with ThreadPoolExecutor(parts) as pool:
        list(pool.map(fetch, range(parts)))
    tmp = output.with_suffix(output.suffix + '.part')
    with tmp.open('wb') as f:
        for piece in pieces:
            f.write(piece.read_bytes())
    if tmp.stat().st_size != size:
        raise RuntimeError(f'Download size mismatch {url}')
    tmp.replace(output)
    for piece in pieces:
        piece.unlink()


def spoc_tces():
    """Every SPOC TCE (all single- and multi-sector DV runs listed on the MAST bulk page).

    Each DV run file has a timestamped name and never changes, so it is kept in data/catalogs/tce/."""
    from concurrent.futures import ThreadPoolExecutor
    files = sorted(set(TCE_FILE.findall(fetch_text(TCE_INDEX))))
    if not files:
        raise ValueError('No TCE files found on the MAST bulk download page')
    folder = DATA / 'catalogs' / 'tce'
    folder.mkdir(parents=True, exist_ok=True)

    def read(path):
        run = re.search(r'-(s\d{4}-s\d{4})_', path).group(1)
        local = folder / Path(path).name
        if not local.exists():
            resume_download('https://archive.stsci.edu' + path, local)
        with local.open() as f:
            return [{'source': 'TCE', 'name': f"TCE {run} {r['tceid'].lstrip('0')}", 'tic': int(r['ticid']),
                     'period': positive(r['tce_period']), 'epoch': positive(r['tce_time0']),
                     'period_err': positive(r['tce_period_err']), 'epoch_err': positive(r['tce_time0_err']),
                     'duration_hours': positive(r['tce_duration']), 'disposition': None, 'transiting': True}
                    for r in csv.DictReader(line for line in f if not line.startswith('#'))]
    with ThreadPoolExecutor(8) as pool:
        tces = [t for part in pool.map(read, files) for t in part]
    print(f'{len(tces)} TCEs from {len(files)} DV runs', flush=True)
    return tces


def catalog(args):
    from collections import Counter
    if not args.local:
        records = json.loads(fetch_text(TAP, 'select * from toi'))
        if not isinstance(records, list) or not records or 'tid' not in records[0]:
            raise ValueError('Unexpected catalog response')
        save_json(CAT, records)
        save_json(KNOWN, known_catalog())
        EXOFOP_TOI.write_text(fetch_text(EXOFOP_TOI_URL))
        rows.cache_clear()
        known_signals_all.cache_clear()
        known_by_star.cache_clear()
    records = rows()
    digest = hashlib.sha256(CAT.read_bytes()).hexdigest()
    snapshot_dir = DATA / 'catalogs' / 'snapshots'
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / f'{digest}.json').write_bytes(CAT.read_bytes())
    with CAT.with_suffix('.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    summary = {
        'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync',
        'query': 'select * from toi',
        'sha256': hashlib.sha256(CAT.read_bytes()).hexdigest(),
        'objects': len(records), 'stars': len({r['tid'] for r in records}),
        'dispositions': dict(Counter(r['tfopwg_disp'] for r in records)),
        'known_signals': {'sources': [CTOI_URL, f'{TAP} ({PLANETS_QUERY})', EB_URL, TCE_INDEX],
                          'sha256': hashlib.sha256(KNOWN.read_bytes()).hexdigest() if KNOWN.exists() else None,
                          'counts': dict(Counter(k['source'] for k in known_signals_all())) if KNOWN.exists() else None},
    }
    save_json(DATA / 'catalogs' / 'snapshot.json', summary)
    print(json.dumps(summary, indent=2))


def eligible(r):
    import math
    return all(r.get(k) is not None and math.isfinite(float(r[k])) and float(r[k]) > 0
               for k in ('pl_orbper', 'pl_tranmid', 'pl_trandurh'))


def curl(url, output):
    """Download url to output; return False only for HTTP 404."""
    result = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location', '--retry', '3',
                             '--retry-all-errors', '--connect-timeout', '15', '--max-time', '600',
                             '--output', str(output), url], capture_output=True, text=True)
    if result.returncode == 22 and '404' in result.stderr:
        return False
    if result.returncode:
        raise RuntimeError(f'Download failed {url}: {result.stderr.strip()}')
    return True


def manifests():
    """Official MAST manifests, one per sector; the last cached sector is always refreshed."""
    from concurrent.futures import ThreadPoolExecutor
    folder = DATA / 'manifests'
    folder.mkdir(parents=True, exist_ok=True)
    last = max((int(p.stem.split('-')[1]) for p in folder.glob('sector-*.sh')), default=0)
    def fetch(sector):
        path = folder / f'sector-{sector}.sh'
        if path.exists() and sector != last:
            return path
        if not curl(MANIFEST.format(sector), path.with_suffix('.tmp')):
            return None
        path.with_suffix('.tmp').replace(path)
        return path
    # Sectors are contiguous: parallel batches until the first 404.
    paths = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        while True:
            batch = list(pool.map(fetch, range(len(paths) + 1, len(paths) + 17)))
            if None in batch:
                return paths + batch[:batch.index(None)]
            paths += batch


def aws_url(tic, sector, filename):
    """Same product in the public stpubdata bucket (AWS Open Data)."""
    t = f'{int(tic):016d}'
    return (f'https://stpubdata.s3.amazonaws.com/tess/public/tid/s{int(sector):04d}/'
            f'{t[0:4]}/{t[4:8]}/{t[8:12]}/{t[12:16]}/{filename}')


def fetch_product(tic, sector, filename, url):
    import numpy as np
    from astropy.io import fits
    from lightkurve.utils import TessQualityFlags
    # Original FITS files are kept; the .npz is derived from them.
    path = DATA / 'mast' / 'lc' / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        if not path.exists() or attempt:
            # The public AWS copy (Open Data) is faster; MAST is the fallback.
            try:
                copied = curl(aws_url(tic, sector, filename), path.with_suffix('.part'))
            except RuntimeError:
                copied = False
            if not copied:
                curl(url, path.with_suffix('.part'))
            path.with_suffix('.part').replace(path)
        try:
            # Read the last row to detect a file truncated by an interrupted download.
            with fits.open(path) as hdul:
                hdul[1].data['TIME'][-1]
            break
        except (OSError, TypeError, ValueError):
            if attempt:
                raise
    with fits.open(path) as hdul:
        head, data = hdul[0].header, hdul[1].data
        time = data['TIME'] + hdul[1].header['BJDREFI'] + hdul[1].header['BJDREFF']
        good = (TessQualityFlags.create_quality_mask(data['QUALITY'], 'default')
                & np.isfinite(time) & np.isfinite(data['PDCSAP_FLUX']))
        crowding = hdul[1].header.get('CROWDSAP')
        star = {k: head.get(key) for k, key in (('radius', 'RADIUS'), ('teff', 'TEFF'),
                                                  ('logg', 'LOGG'), ('tmag', 'TESSMAG'))}
        arrays = {'time_bjd': time[good], 'flux': data['PDCSAP_FLUX'][good],
                  'flux_err': data['PDCSAP_FLUX_ERR'][good],
                  'centr1': data['MOM_CENTR1'][good], 'centr2': data['MOM_CENTR2'][good]}
    out = DATA / 'curves' / str(tic) / f'sector-{sector}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix('.tmp').open('wb') as output:
        np.savez_compressed(output, **{k: np.asarray(v, dtype=float) for k, v in arrays.items()})
    out.with_suffix('.tmp').replace(out)
    save_json(out.with_suffix('.json'), {
        'version': CURVE_VERSION, 'tic': tic, 'sector': sector, 'author': 'SPOC', 'exptime': 120,
        'quality_bitmask': 'default', 'flux': 'PDCSAP_FLUX', 'centroid': 'MOM_CENTR', 'time': 'BJD TDB',
        'crowdsap': crowding, 'star': {k: (float(v) if v is not None else None) for k, v in star.items()},
        'product': filename, 'fits': str(path.relative_to(ROOT)),
        'target_pixel': target_pixel(path.relative_to(ROOT)),
        'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
        'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
    })


def current(npz):
    json_path = npz.with_suffix('.json')
    return (npz.exists() and json_path.exists()
            and json.loads(json_path.read_text()).get('version') == CURVE_VERSION)


def field_stars(products, toi, n):
    """Fixed sample (seed 42) of non-TOI stars with ≥30 sectors, with all sectors of each.

    Many sectors means many transits: that is where weak signals, missed sector by sector by SPOC,
    show up. The shuffle is stable: increasing n only appends stars to the previous sample.
    """
    import random
    pool = sorted(t for t, p in products.items() if t not in toi and len(p) >= FIELD_SECTORS)
    random.Random(42).shuffle(pool)
    return {tic: sorted(products[tic], key=int) for tic in pool[:n]}


def open_tois():
    """TOIs still open: TFOPWG disposition PC (or none yet) and no follow-up observation of any kind
    (time series, spectroscopy, imaging) on ExoFOP. Returns {TIC: [TOI, ...]}."""
    from collections import defaultdict
    if not EXOFOP_TOI.exists():
        raise FileNotFoundError(f'{EXOFOP_TOI} missing; run make catalog')
    out = defaultdict(list)
    for r in csv.DictReader(io.StringIO(EXOFOP_TOI.read_text())):
        if r['TFOPWG Disposition'] in ('PC', '') and all(int(r[k] or 0) == 0 for k in FOLLOW_UP):
            out[int(r['TIC ID'])].append(r['TOI'])
    return dict(out)


def download(args):
    """SPOC 120 s light curves of labeled stars, of a non-TOI sample (--field) or of the stars of open
    TOIs (--open), via MAST manifests."""
    from collections import defaultdict
    from concurrent.futures import ProcessPoolExecutor, as_completed
    products = defaultdict(dict)
    for manifest in manifests():
        for line in manifest.read_text().splitlines():
            m = PRODUCT.search(line)
            # The TIC comes from the file name: exact match, no name resolution to neighboring stars.
            if m:
                products[int(m[3])][str(int(m[2]))] = (m[1], m[0])
    if args.field:
        chosen = field_stars(products, {int(r['tid']) for r in rows()}, args.field)
        save_json(DATA / 'field.json', {str(t): s for t, s in chosen.items()})
        selected = {(t, s): products[t][s] for t, sectors in chosen.items() for s in sectors}
        print(f'{len(chosen)} non-TOI stars with >= {FIELD_SECTORS} sectors, all sectors', flush=True)
    elif args.open:
        chosen = open_tois()
        save_json(DATA / 'open.json', {str(t): tois for t, tois in sorted(chosen.items())})
        selected = {(t, s): p for t in chosen if t in products for s, p in products[t].items()}
        print(f'{len(chosen)} stars with open TOIs (PC, no follow-up); '
              f'{len({t for t, _ in selected})} have 120 s light curves, all sectors', flush=True)
    else:
        tics = {int(r['tid']) for r in rows() if eligible(r)
                and (r['tfopwg_disp'] in LABELS or args.include_candidates)}
        selected = {(t, s): p for t in tics if t in products for s, p in products[t].items()}
        print(f'{len({t for t, _ in selected})}/{len(tics)} stars with light curves', flush=True)
    pending = [(tic, sector, *product) for (tic, sector), product in sorted(selected.items())
               if not current(DATA / 'curves' / str(tic) / f'sector-{sector}.npz')]
    print(f'{len(selected)} files, {len(pending)} pending', flush=True)
    log = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_product, *item): item for item in pending}
        for done, future in enumerate(as_completed(futures), 1):
            tic, sector, filename, _ = futures[future]
            try:
                future.result()
            except Exception as exc:
                log.append({'tic': tic, 'sector': sector, 'product': filename, 'error': str(exc)[-500:]})
                print(f'[{done}/{len(pending)}] TIC {tic} sector {sector} error: {str(exc)[-200:]}', flush=True)
            if done % 100 == 0 or done == len(pending):
                name = 'download-field-log.json' if args.field else 'download-open-log.json' if args.open else 'download-log.json'
                save_json(DATA / name, log)
                print(f'[{done}/{len(pending)}] {len(log)} errors', flush=True)
    print(f'Done: {len(pending) - len(log)}/{len(pending)}; rerun the command to retry the errors.')


def downloaded_stars():
    return sorted(int(p.name) for p in (DATA / 'curves').glob('*') if any(p.glob('*.npz')))


def target_pixel(fits_path):
    """Target CCD column and row (same system as MOM_CENTR), from the APERTURE extension WCS."""
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(ROOT / fits_path) as hdul:
        aperture = hdul[2].header
        x, y = WCS(aperture).all_world2pix(hdul[0].header['RA_OBJ'], hdul[0].header['DEC_OBJ'], 0)
    return [float(x + aperture['CRVAL1P']), float(y + aperture['CRVAL2P'])]


def load_star(tic):
    """All downloaded sectors of a star, concatenated in time order."""
    import numpy as np
    parts, meta = [], {}
    for path in (DATA / 'curves' / str(tic)).glob('sector-*.npz'):
        if not current(path):
            continue
        sector = int(path.stem.split('-')[1])
        meta[sector] = json.loads(path.with_suffix('.json').read_text())
        if 'target_pixel' not in meta[sector]:
            meta[sector]['target_pixel'] = target_pixel(meta[sector]['fits'])
        with np.load(path) as lc:
            parts.append({k: lc[k] for k in lc.files} | {'sector': np.full(len(lc['time_bjd']), sector)})
    if not parts:
        raise ValueError(f'No up-to-date light curves for TIC {tic}; run make curves')
    star = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(star['time_bjd'], kind='stable')
    star = {k: v[order] for k, v in star.items()}
    star['tic'], star['meta'] = tic, meta
    return star


def select_sectors(star, sectors):
    import numpy as np
    keep = np.isin(star['sector'], list(sectors))
    return {k: (v[keep] if hasattr(v, 'shape') else v) for k, v in star.items()}


def star_tois(tic):
    return [r for r in rows() if int(r['tid']) == int(tic) and eligible(r)]


@functools.lru_cache(maxsize=1)
def known_signals_all():
    if not KNOWN.exists():
        raise FileNotFoundError(f'{KNOWN} missing; run make catalog')
    return json.loads(KNOWN.read_text())


@functools.lru_cache(maxsize=1)
def known_by_star():
    from collections import defaultdict
    index = defaultdict(list)
    for k in known_signals_all():
        index[k['tic']].append(k)
    return index


def known_signals(tic):
    """Every catalogued signal of the star (TOIs, CTOIs, planets, eclipsing binaries, TCEs), in one format."""
    tois = [{'source': 'TOI', 'name': f"TOI {r['toi']}", 'toi': r['toi'], 'tic': int(tic),
             'period': float(r['pl_orbper']), 'epoch': float(r['pl_tranmid']),
             'period_err': positive(r.get('pl_orbpererr1')), 'epoch_err': positive(r.get('pl_tranmiderr1')),
             'duration_hours': float(r['pl_trandurh']), 'disposition': r['tfopwg_disp'], 'transiting': True}
            for r in star_tois(tic)]
    return tois + known_by_star().get(int(tic), [])


def masks_for(tois, exclude=None):
    return [(float(r['pl_orbper']), float(r['pl_tranmid']), float(r['pl_trandurh']))
            for r in tois if exclude is None or r['toi'] != exclude['toi']]

