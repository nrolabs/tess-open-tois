"""Coleta pública do TESS, preparação das curvas e CLI do pipeline de candidatos."""
import argparse
import csv
import functools
import hashlib
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
CAT = DATA / 'catalogs' / 'toi.json'
LABELS = {'CP': 1, 'KP': 1, 'FP': 0, 'FA': 0}
MANIFEST = 'https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_{}_lc.sh'
PRODUCT = re.compile(r'https://\S+/(tess\d+-s(\d{4})-(\d{16})-\d{4}-s_lc\.fits)$')
CURVE_VERSION = 2
FIELD_SECTORS = 30
FEATURE_VERSION = 2
# Entrada da CNN: (vista, canais, bins); canais pares = fluxo, ímpares = cobertura.
VIEWS = (('global', 2, 101), ('local', 2, 61), ('odd_even', 4, 61), ('secondary', 2, 61))
SCALARS = ('log_depth', 'snr', 'odd_even_sigma', 'secondary_sigma', 'ingress_fraction',
           'log_duration_period', 'log_transits', 'centroid_sigma', 'crowding', 'log_star_radius',
           'star_logg', 'star_teff', 'star_tmag', 'log_density_ratio')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


@functools.lru_cache(maxsize=1)
def rows():
    return json.loads(CAT.read_text())


def catalog(args):
    from collections import Counter
    if not args.local:
        result = subprocess.run([
            'curl', '--fail', '--silent', '--show-error', '--retry', '2', '--retry-all-errors',
            '--connect-timeout', '15', '--max-time', '180', '-G',
            'https://exoplanetarchive.ipac.caltech.edu/TAP/sync',
            '--data-urlencode', 'query=select * from toi',
            '--data-urlencode', 'format=json',
        ], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f'Falha ao consultar TAP: {result.stderr.strip()}')
        records = json.loads(result.stdout)
        if not isinstance(records, list) or not records or 'tid' not in records[0]:
            raise ValueError('Resposta inesperada do catálogo')
        save_json(CAT, records)
        rows.cache_clear()
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
    }
    save_json(DATA / 'catalogs' / 'snapshot.json', summary)
    print(json.dumps(summary, indent=2))


def eligible(r):
    import math
    return all(r.get(k) is not None and math.isfinite(float(r[k])) and float(r[k]) > 0
               for k in ('pl_orbper', 'pl_tranmid', 'pl_trandurh'))


def curl(url, output):
    """Baixa url em output; devolve False somente para HTTP 404."""
    result = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location', '--retry', '3',
                             '--retry-all-errors', '--connect-timeout', '15', '--max-time', '600',
                             '--output', str(output), url], capture_output=True, text=True)
    if result.returncode == 22 and '404' in result.stderr:
        return False
    if result.returncode:
        raise RuntimeError(f'Falha ao baixar {url}: {result.stderr.strip()}')
    return True


def manifests():
    """Manifestos oficiais da MAST, um por setor; o último setor em cache é sempre renovado."""
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
    # Setores são contínuos: lotes paralelos até o primeiro 404.
    paths = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        while True:
            batch = list(pool.map(fetch, range(len(paths) + 1, len(paths) + 17)))
            if None in batch:
                return paths + batch[:batch.index(None)]
            paths += batch


def aws_url(tic, sector, filename):
    """Mesmo produto no bucket público stpubdata (AWS Open Data)."""
    t = f'{int(tic):016d}'
    return (f'https://stpubdata.s3.amazonaws.com/tess/public/tid/s{int(sector):04d}/'
            f'{t[0:4]}/{t[4:8]}/{t[8:12]}/{t[12:16]}/{filename}')


def fetch_product(tic, sector, filename, url):
    import numpy as np
    from astropy.io import fits
    from lightkurve.utils import TessQualityFlags
    # FITS originais ficam preservados; o .npz é derivado deles.
    path = DATA / 'mast' / 'lc' / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        if not path.exists() or attempt:
            # Cópia pública na AWS (Open Data) é mais rápida; a MAST fica como reserva.
            try:
                copied = curl(aws_url(tic, sector, filename), path.with_suffix('.part'))
            except RuntimeError:
                copied = False
            if not copied:
                curl(url, path.with_suffix('.part'))
            path.with_suffix('.part').replace(path)
        try:
            # Lê a última linha para detectar arquivo truncado por download interrompido.
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
    """Amostra fixa (semente 42) de estrelas sem TOI com ≥30 setores, com todos os setores de cada uma.

    Muitos setores = muitos trânsitos: é onde sinais fracos, que o SPOC deixa passar setor a setor,
    aparecem. O embaralhamento é estável: aumentar n só acrescenta estrelas à amostra anterior.
    """
    import random
    pool = sorted(t for t, p in products.items() if t not in toi and len(p) >= FIELD_SECTORS)
    random.Random(42).shuffle(pool)
    chosen = {tic: sorted(products[tic], key=int) for tic in pool[:n]}
    save_json(DATA / 'field.json', {str(t): s for t, s in chosen.items()})
    return chosen


def download(args):
    """Curvas SPOC 120 s das estrelas rotuladas, ou de uma amostra sem TOI (--field), via manifestos da MAST."""
    from collections import defaultdict
    from concurrent.futures import ProcessPoolExecutor, as_completed
    products = defaultdict(dict)
    for manifest in manifests():
        for line in manifest.read_text().splitlines():
            m = PRODUCT.search(line)
            # O TIC vem do nome do arquivo: correspondência exata, sem resolver nomes vizinhos.
            if m:
                products[int(m[3])][str(int(m[2]))] = (m[1], m[0])
    if args.field:
        chosen = field_stars(products, {int(r['tid']) for r in rows()}, args.field)
        selected = {(t, s): products[t][s] for t, sectors in chosen.items() for s in sectors}
        print(f'{len(chosen)} estrelas sem TOI com >= {FIELD_SECTORS} setores, todos os setores', flush=True)
    else:
        tics = {int(r['tid']) for r in rows() if eligible(r)
                and (r['tfopwg_disp'] in LABELS or args.include_candidates)}
        selected = {(t, s): p for t in tics if t in products for s, p in products[t].items()}
        print(f'{len({t for t, _ in selected})}/{len(tics)} estrelas com curva', flush=True)
    pending = [(tic, sector, *product) for (tic, sector), product in sorted(selected.items())
               if not current(DATA / 'curves' / str(tic) / f'sector-{sector}.npz')]
    print(f'{len(selected)} arquivos, {len(pending)} pendentes', flush=True)
    log = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_product, *item): item for item in pending}
        for done, future in enumerate(as_completed(futures), 1):
            tic, sector, filename, _ = futures[future]
            try:
                future.result()
            except Exception as exc:
                log.append({'tic': tic, 'sector': sector, 'product': filename, 'error': str(exc)[-500:]})
                print(f'[{done}/{len(pending)}] TIC {tic} setor {sector} erro: {str(exc)[-200:]}', flush=True)
            if done % 100 == 0 or done == len(pending):
                save_json(DATA / ('download-field-log.json' if args.field else 'download-log.json'), log)
                print(f'[{done}/{len(pending)}] {len(log)} erros', flush=True)
    print(f'Concluídos: {len(pending) - len(log)}/{len(pending)}; repita o comando para tentar os erros.')


def downloaded_stars():
    return sorted(int(p.name) for p in (DATA / 'curves').glob('*') if any(p.glob('*.npz')))


def target_pixel(fits_path):
    """Coluna e linha do alvo no CCD (mesmo sistema de MOM_CENTR), pela WCS da extensão APERTURE."""
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(ROOT / fits_path) as hdul:
        aperture = hdul[2].header
        x, y = WCS(aperture).all_world2pix(hdul[0].header['RA_OBJ'], hdul[0].header['DEC_OBJ'], 0)
    return [float(x + aperture['CRVAL1P']), float(y + aperture['CRVAL2P'])]


def load_star(tic):
    """Todos os setores baixados da estrela, concatenados em ordem temporal."""
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
        raise ValueError(f'Sem curvas atualizadas para TIC {tic}; execute make curves')
    star = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(star['time_bjd'], kind='stable')
    star = {k: v[order] for k, v in star.items()}
    star['tic'], star['meta'] = tic, meta
    return star


def select_sectors(star, sectors):
    import numpy as np
    keep = np.isin(star['sector'], list(sectors))
    return {k: (v[keep] if hasattr(v, 'shape') else v) for k, v in star.items()}


def wrap(phase, period):
    return (phase + period / 2) % period - period / 2


def trend(time, values, base, boundaries, window):
    """Mediana móvel por segmento, interpolando sobre pontos fora de base."""
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
    """Remove tendências protegendo o trânsito e mascarando os outros planetas conhecidos da estrela."""
    import numpy as np
    time, flux = np.asarray(time, dtype=float), np.asarray(flux, dtype=float)
    sector = np.zeros(len(time)) if sector is None else np.asarray(sector)
    index = np.argsort(time, kind='stable')
    index = index[np.isfinite(time[index]) & np.isfinite(flux[index])]
    time, flux, sector = time[index], flux[index], sector[index]
    duration = duration_hours / 24
    if len(time) < 200 or not (0 < duration < period / 2):
        raise ValueError('Cobertura ou duração inválida')
    phase = wrap(time - epoch, period)
    inside = np.abs(phase) < duration / 2
    others = np.zeros(len(time), dtype=bool)
    for p, e, d in masks:
        others |= np.abs(wrap(time - e, p)) < d / 24
    outside = (np.abs(phase) > duration) & ~others
    if (inside & ~others).sum() < 5 or outside.sum() < 100:
        raise ValueError('Sem cobertura suficiente do trânsito previsto')
    cadence = np.median(np.diff(time)[np.diff(time) > 0])
    window = max(11, int(max(1, 5 * duration) / cadence) | 1)
    base = trend(time, flux, outside, segments(time, sector, cadence), window)
    corrected = np.divide(flux, base, out=np.full(len(flux), np.nan), where=np.isfinite(base) & (base != 0)) - 1
    corrected[others] = np.nan
    valid = np.isfinite(corrected)
    if (valid & inside).sum() < 5 or (valid & outside).sum() < 100:
        raise ValueError('Cobertura insuficiente após remover tendência')
    out = corrected[valid & outside]
    scatter = max(1.4826 * np.median(np.abs(out - np.median(out))), 1e-9)
    return {'time': time, 'phase': phase, 'corrected': corrected, 'normalized': corrected / scatter,
            'valid': valid, 'outside': outside & valid, 'scatter': scatter, 'cadence': cadence,
            'period': period, 'epoch': epoch, 'duration': duration, 'sector': sector, 'index': index,
            'window': window}


def views(prep):
    """Perfis dobrados no formato de VIEWS: global, local, ímpar/par e secundário."""
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


def star_tois(tic):
    return [r for r in rows() if int(r['tid']) == int(tic) and eligible(r)]


def masks_for(tois, exclude=None):
    return [(float(r['pl_orbper']), float(r['pl_tranmid']), float(r['pl_trandurh']))
            for r in tois if exclude is None or r['toi'] != exclude['toi']]


def star_examples(tic, tois):
    """Por TOI rotulado: um exemplo com todos os setores e um por setor (aumento de dados no treino)."""
    from vetting import analyze
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
    import pickle
    key = hashlib.sha256(json.dumps({
        'version': FEATURE_VERSION,
        'code': hashlib.sha256((ROOT / 'vetting.py').read_bytes()).hexdigest(),
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
    """Exemplos rotulados de todas as estrelas baixadas, com cache por estrela em data/features/."""
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
                print(f'Features: {done}/{len(stars)} estrelas', flush=True)
    save_json(ROOT / 'reports' / 'skipped.json', skipped)
    return examples


def train(args):
    from neural import train_network
    train_network(args)


def candidates(args):
    from search import run_candidates
    run_candidates(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('catalog'); c.add_argument('--local', action='store_true'); c.set_defaults(func=catalog)
    d = sub.add_parser('download', help='Todas as curvas SPOC 120 s das estrelas rotuladas')
    d.add_argument('--workers', type=int, default=16, help='Downloads em paralelo')
    d.add_argument('--include-candidates', action='store_true', help='Também PC/APC (não entram no treino)')
    d.add_argument('--field', type=int, default=0,
                   help='Em vez dos rotulados: N estrelas sem TOI com >=30 setores (todos), para planetas novos')
    d.set_defaults(func=download)
    tr = sub.add_parser('train')
    tr.add_argument('--epochs', type=int, default=100)
    tr.add_argument('--batch-size', type=int, default=64)
    tr.add_argument('--patience', type=int, default=15)
    tr.add_argument('--precision', type=float, default=.95)
    tr.add_argument('--injections', type=int, default=300, help='Trânsitos sintéticos para injeção-recuperação')
    tr.set_defaults(func=train)
    s = sub.add_parser('candidates', help='Busca BLS, CNN, testes de falso positivo e TRICERATOPS')
    s.add_argument('--tic', type=int, help='Uma estrela; sem isso, todas as estrelas baixadas')
    s.add_argument('--period', type=float, help='Dias; com --epoch e --duration avalia um evento dado')
    s.add_argument('--epoch', type=float, help='BJD TDB')
    s.add_argument('--duration', type=float, help='Horas')
    s.add_argument('--no-fpp', action='store_true', help='Não executar TRICERATOPS')
    s.add_argument('--field', action='store_true', help='Só as estrelas sem TOI baixadas com make curves FIELD=N')
    s.add_argument('--workers', type=int, default=4, help='Processos em paralelo (cada um usa a GPU na BLS)')
    s.set_defaults(func=candidates)
    args = p.parse_args()
    if args.command == 'candidates':
        given = [args.period, args.epoch, args.duration]
        if any(v is not None for v in given) and (None in given or args.tic is None):
            p.error('--period, --epoch e --duration exigem os três valores e --tic')
    args.func(args)


if __name__ == '__main__':
    main()
