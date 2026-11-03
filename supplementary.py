"""Supplementary analysis for the paper: pixel test and TRICERATOPS for vetted signals of the open-TOI
sample that the CNN did not select (tier 'inconclusive'). The search decisions are not changed:
results go to reports/supplementary/, one file per star."""
import json
from concurrent.futures import ThreadPoolExecutor

from data import ROOT, DATA, save_json
from search import pixel_job, fpp_job, json_safe

OUT = ROOT / 'reports' / 'supplementary'


def main():
    chosen = {int(t) for t in json.loads((DATA / 'open.json').read_text())}
    jobs = []
    for tic in sorted(chosen):
        path = ROOT / 'reports' / 'candidates' / f'{tic}.json'
        if not path.exists() or (OUT / f'{tic}.json').exists():
            continue
        events = [e for e in json.loads(path.read_text())['events'] if e.get('tier', '').startswith('inconclusive')]
        if events:
            jobs.append((tic, events))
    print(f'{len(jobs)} stars, {sum(len(e) for _, e in jobs)} inconclusive signals', flush=True)
    with ThreadPoolExecutor(2) as fpp_pool, ThreadPoolExecutor(2) as pixel_pool:
        for n, (tic, events) in enumerate(jobs, 1):
            pending = [(e, pixel_pool.submit(pixel_job, e), fpp_pool.submit(fpp_job, e)) for e in events]
            out = []
            for e, pixel, fpp in pending:
                out.append({'period_days': e['period_days'], 'epoch_bjd': e['epoch_bjd'],
                            'duration_hours': e['duration_hours'], 'known': e['known'],
                            'cnn_calibrated': e.get('cnn_calibrated'), 'tier': e['tier'],
                            'pixel_test': pixel.result(), 'fpp': fpp.result()})
            save_json(OUT / f'{tic}.json', json_safe({'tic': tic, 'events': out}))
            for o in out:
                f = o['fpp'] or {}
                print(f"[{n}/{len(jobs)}] TIC {tic} P={o['period_days']:.4f} {(o['known'] or {}).get('name', 'not catalogued')} "
                      f"FPP={f.get('fpp')} NFPP={f.get('nfpp')} pixel={(o['pixel_test'] or {}).get('sigma')}", flush=True)


if __name__ == '__main__':
    main()
