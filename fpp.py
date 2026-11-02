"""Probabilidade de falso positivo com TRICERATOPS; roda no ambiente isolado .venv-fpp."""
import json
import sys
import time

import numpy as np

START = time.time()


def log(step):
    print(f'[{time.time() - START:7.1f} s] {step}', file=sys.stderr, flush=True)


def main(source, target):
    import triceratops.triceratops as tr
    job = json.loads(open(source).read())
    sectors = np.array(job['sectors'])
    log(f"consultando TIC/Gaia ao redor do TIC {job['tic']}, setores {list(sectors)}")
    star = tr.target(ID=job['tic'], sectors=sectors)
    log(f'{len(star.stars)} estrelas no campo; buscando aberturas SPOC')
    apertures = star.get_spoc_apertures()
    keep = [i for i, a in enumerate(apertures) if a is not None]
    if not keep:
        raise SystemExit('Sem aperturas SPOC para os setores')
    if len(keep) < len(sectors):
        star = tr.target(ID=job['tic'], sectors=sectors[keep])
        apertures = [apertures[i] for i in keep]
    log('calculando profundidades exigidas de cada estrela')
    star.calc_depths(tdepth=job['depth'], all_ap_pixels=apertures)
    # O resultado é Monte Carlo: várias execuções dão média e dispersão.
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
    log('fim')
    with open(target, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main(*sys.argv[1:3])
