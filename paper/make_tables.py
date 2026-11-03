"""Tables, numbers and figures of the paper, generated from the pipeline reports (run from tess/)."""
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import downloaded_stars, load_star  # noqa: E402

PAPER = ROOT / 'paper'
TABLES, FIGURES = PAPER / 'tables', PAPER / 'figures'
REPORTS, SUPPLEMENTARY, CTOI = ROOT / 'reports' / 'candidates', ROOT / 'reports' / 'supplementary', ROOT / 'reports' / 'ctoi'
CANDIDATES = [452464529, 89062142, 410056850, 219396153]


def write(name, text):
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text(text.strip() + '\n')


def num(x, fmt='{:.3g}'):
    return '--' if x is None or (isinstance(x, float) and not np.isfinite(x)) else fmt.format(x)


def pm(entry, digits):
    v, lo, hi = entry['value'], entry['minus'], entry['plus']
    return f'${v:.{digits}f}^{{+{hi:.{digits}f}}}_{{-{lo:.{digits}f}}}$'


def exp_tex(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '--'
    if x == 0 or x < 1e-15:
        return '$0$'
    if x >= 0.005:
        return f'{x:.2g}'
    exp = int(np.floor(np.log10(x)))
    coeff = round(x / (10.0 ** exp))
    if coeff >= 10:
        coeff = 1
        exp += 1
    if coeff == 1:
        return f'$1\\times 10^{{{exp}}}$'
    return f'${coeff}\\times 10^{{{exp}}}$'


def sample():
    open_tois = {int(t): v for t, v in json.loads((ROOT / 'data' / 'open.json').read_text()).items()}
    exo = {r['TOI']: r for r in csv.DictReader(open(ROOT / 'data' / 'catalogs' / 'exofop_toi.csv'))}
    stars = sorted(set(downloaded_stars()) & set(open_tois))
    return open_tois, exo, stars


def main():
    open_tois, exo, stars = sample()
    reports = {t: json.loads((REPORTS / f'{t}.json').read_text()) for t in stars}
    supplementary = {int(p.stem): json.loads(p.read_text()) for p in SUPPLEMENTARY.glob('*.json')}

    # Recovery of each open TOI.
    recovered, missed = {}, []
    for t in stars:
        by_name = {(e.get('known') or {}).get('name'): e for e in reports[t]['events'] if 'tier' in e and e.get('known')}
        for toi in open_tois[t]:
            if f'TOI {toi}' in by_name:
                recovered[toi] = (t, by_name[f'TOI {toi}'])
            else:
                missed.append(toi)
    n_open = sum(len(open_tois[t]) for t in stars)
    sectors = [len(set(load_star(t)['sector'].astype(int))) for t in stars]
    tmags = [next(iter(load_star(t)['meta'].values()))['star']['tmag'] for t in stars]
    tmags = [x for x in tmags if x]
    training = json.loads((ROOT / 'reports' / 'training.json').read_text())
    new = [(t, e) for t in stars for e in reports[t]['events'] if 'tier' in e and not e.get('known')
           and e['tier'].startswith(('inconclusive', 'candidate', 'strong'))]
    write('numbers.tex', f'''
\\newcommand{{\\nStars}}{{{len(stars)}}}
\\newcommand{{\\nOpen}}{{{n_open}}}
\\newcommand{{\\nRecovered}}{{{len(recovered)} ({100 * len(recovered) / n_open:.0f}\\%)}}
\\newcommand{{\\nNew}}{{{len(new)}}}
\\newcommand{{\\nOpenAll}}{{{sum(len(v) for v in open_tois.values())}}}
\\newcommand{{\\nOpenStarsAll}}{{{len(open_tois)}}}
\\newcommand{{\\tmagRange}}{{{min(tmags):.1f}--{max(tmags):.1f} (median {np.median(tmags):.1f})}}
\\newcommand{{\\sectorRange}}{{{min(sectors)}--{max(sectors)} (median {np.median(sectors):.0f})}}
\\newcommand{{\\trainStars}}{{{training['stars']}}}
''')

    # Why TOIs were missed.
    reasons = {'short': 0, 'long': 0, 'weak': 0, 'other': 0}
    for toi in missed:
        r = exo[toi]
        period, snr = float(r['Period (days)'] or 0), float(r['Planet SNR'] or 0)
        if 0 < period < .5:
            reasons['short'] += 1
        elif period == 0 or period > 100:
            reasons['long'] += 1
        elif snr < 10:
            reasons['weak'] += 1
        else:
            reasons['other'] += 1
    write('missed.tex', f'''
Of the {len(missed)} TOIs not recovered, {reasons['short']} {'has a period' if reasons['short'] == 1 else 'have periods'} below the 0.5\\,d limit of
the periodic search, {reasons['long']} are single transits or have periods beyond 100\\,d (their
events were found by the long-period search but rejected by its conservative vetting or not
detected), {reasons['weak']} have a catalogue SNR below 10, and {reasons['other']} have higher SNR:
for these the 120\\,s sectors available do not always include the sectors in which the TOI was
detected.''')

    # Tier summary.
    rows = {}
    for t in stars:
        for e in reports[t]['events']:
            if 'tier' not in e:
                continue
            kind = e.get('search', 'bls')
            tier = e['tier'].split(':')[0].split(' [')[0]
            key = (kind, tier)
            rows.setdefault(key, [0, 0])[0 if e.get('known') else 1] += 1
    body = '\n'.join(f'{"periodic" if k == "bls" else k} & {tier} & {c} & {n} \\\\'
                     for (k, tier), (c, n) in sorted(rows.items(), key=lambda x: (x[0][0], ['candidate', 'inconclusive', 'rejected'].index(x[0][1]) if x[0][1] in ('candidate', 'inconclusive', 'rejected') else 9)))
    write('summary.tex', f'''
\\begin{{deluxetable}}{{llcc}}
\\tablecaption{{Signals found in the {len(stars)} host stars of open TOIs.\\label{{tab:summary}}}}
\\tablehead{{\\colhead{{Search}} & \\colhead{{Tier}} & \\colhead{{Catalogued}} & \\colhead{{Not catalogued}}}}
\\startdata
{body}
\\enddata
\\tablecomments{{Tiers refer to the initial search, before supplementary testing. Catalogued signals match a TOI, CTOI, published planet, SPOC TCE or known eclipsing binary; these counts are not restricted to the open TOIs.}}
\\end{{deluxetable}}''')

    # CNN performance.
    cv = training['cross_validated_test']
    lo, hi = cv['precision_wilson95']
    folds = training['folds_average_precision_mean_std']
    tn, fp, fn, tp = np.array(cv['confusion_matrix_rows_true_cols_pred_0_1']).ravel()
    write('cnn.tex', f'''
\\begin{{deluxetable}}{{lc}}
\\tablecaption{{Cross-validated performance of the classifier.\\label{{tab:cnn}}}}
\\tablehead{{\\colhead{{Quantity}} & \\colhead{{Value}}}}
\\startdata
Labelled TOIs (stars) & {cv['n']} ({training['stars']}) \\\\
Positives (CP/KP) & {cv['positives']} \\\\
Average precision & {cv['average_precision']:.3f} \\\\
Average precision, per fold & {folds[0]:.3f} $\\pm$ {folds[1]:.3f} \\\\
ROC AUC & {cv['roc_auc']:.3f} \\\\
Precision at threshold & {cv['precision']:.3f} ({lo:.3f}--{hi:.3f}) \\\\
Recall at threshold & {cv['recall']:.3f} \\\\
Confusion (TP, FP, FN, TN) & {tp}, {fp}, {fn}, {tn} \\\\
\\enddata
\\tablecomments{{Out-of-fold scores of five models, folds grouped by star; fold-specific validation thresholds target 95\\% precision.
The precision interval is the Wilson 95\\% interval.}}
\\end{{deluxetable}}''')

    # Candidates.
    lines, figures = [], {}
    names = {}
    for t in CANDIDATES:
        folder = sorted(CTOI.glob(f'TIC{t}-bls-*'))[-1]
        s = json.loads((folder / 'summary.json').read_text())
        p, d, f, m = s['parameters'], s['derived'], s['fpp'] or {}, s['metrics']
        name = (s.get('known') or {}).get('name', f'TIC {t}')
        names[t] = name
        pixel = m.get('pixel_test') or {}
        lines.append(' & '.join([
            name.replace('TOI ', 'TOI-'), str(t), pm({**p['t0'], 'value': p['t0']['value'] - 2460000}, 4),
            pm(p['period'], 5), pm(p['k'], 4), pm(p['b'], 2), pm(d['duration_hours'], 2),
            pm(d['radius_earth'], 1), pm(d['teq_k'], 0) if 'teq_k' in d else '--',
            f"{s['event'].get('cnn_calibrated', 0):.2f}", f"{f.get('fpp', 0) * 100:.1f} $\\pm$ {f.get('fpp_error_of_mean', 0) * 100:.1f}",
            exp_tex(f.get('nfpp')), num(pixel.get('offset_arcsec'), '{:.1f}')]) + ' \\\\')
        FIGURES.mkdir(parents=True, exist_ok=True)
        shutil.copy(folder / 'folded.pdf', FIGURES / f'folded_{t}.pdf')
        if (folder / 'difference_image.pdf').exists():
            shutil.copy(folder / 'difference_image.pdf', FIGURES / f'difference_{t}.pdf')
    write('candidates.tex', '''
\\begin{deluxetable*}{lccccccccccc}
\\tablecaption{Candidates: transit fits, classifier score and false-positive probabilities.\\label{tab:candidates}}
\\tablehead{\\colhead{TOI} & \\colhead{TIC} & \\colhead{$T_0-2460000$} & \\colhead{$P$ (d)} & \\colhead{$R_p/R_\\star$} &
\\colhead{$b$} & \\colhead{$T_{14}$ (h)} & \\colhead{$R_p$ ($R_\\oplus$)} & \\colhead{$T_{\\rm eq}$ (K)} &
\\colhead{CNN} & \\colhead{FPP (\\%)} & \\colhead{NFPP}}
\\startdata
''' + '\n'.join(line.rsplit(' & ', 1)[0] + ' \\\\' for line in lines) + '''
\\enddata
\\tablecomments{$T_0$ in BJD$_{\\rm TDB}$; $T_{\\rm eq}$ for zero albedo and full redistribution. CNN is the calibrated
classifier score. FPP and NFPP from ten TRICERATOPS runs (mean and error of the mean), without
high-resolution imaging. Three significant difference images are consistent with the target position; the image for TOI-7888.01 is not significant. Zero NFPP values reflect numerical estimates, not an exclusion of all contaminating sources.}
\\end{deluxetable*}''')

    # Inconclusive recovered TOIs, with the supplementary pixel test and FPP.
    inconclusive = []
    for toi, (t, e) in sorted(recovered.items(), key=lambda x: float(x[0])):
        if not e['tier'].startswith('inconclusive'):
            continue
        extra = next((x for x in supplementary.get(t, {}).get('events', [])
                      if abs(x['period_days'] - e['period_days']) < 1e-6), None)
        inconclusive.append((toi, t, e, extra))
    done = [x for x in inconclusive if x[3]]
    fpps = [((x[3].get('fpp') or {}).get('fpp'), (x[3].get('fpp') or {}).get('nfpp'), (x[3].get('pixel_test') or {})) for x in done]
    valid_fpp = [f for f in fpps if f[0] is not None]
    low = sum(1 for f, n, _ in valid_fpp if f < .01 and n < .001)
    likely = sum(1 for f, n, _ in valid_fpp if f < .5 and n < .001)
    nearby = sum(1 for f, n, _ in valid_fpp if n is not None and n > .1)
    offset = sum(1 for *_, px in fpps if px.get('sigma') and px['sigma'] > 4 and (px.get('offset_pixels') or 0) > .5)
    threshold = json.loads(next((ROOT / 'reports' / 'candidates').glob('*.json')).read_text())['threshold']
    below = [x for x in inconclusive if not x[2].get('cnn_selected')]
    selected = [x for x in inconclusive if x[2].get('cnn_selected')]
    write('inconclusive_text.tex', f'''
A total of {len(inconclusive)} recovered TOIs passed the photometric vetting and were initially classified as inconclusive.
Of these, {len(below)} score below the classifier threshold (raw scores
{min(x[2].get('cnn_score') or 0 for x in below):.2f}--{max(x[2].get('cnn_score') or 0 for x in below):.3f}, threshold {threshold:.3f}).
The other {len(selected)} (marked with $\\dagger$ in Table~\\ref{{tab:inconclusive}}) are selected by the classifier,
but the search \\texttt{{TRICERATOPS}} result falls outside the candidate criterion
(FPP\\,$\\geq 50\\%$, or $0.1\\% \\leq$\\,NFPP\\,$\\leq 10\\%$).
Difference images and \\texttt{{TRICERATOPS}} (three runs) were computed for {'all of them' if len(done) == len(inconclusive) else f'{len(done)} of them'}. {'None has' if low == 0 else str(low) + ' have'} FPP\\,$<1\\%$ and NFPP\\,$<0.1\\%$,
while {likely} have FPP\\,$<50\\%$ and NFPP\\,$<0.1\\%$, {nearby} have NFPP\\,$>10\\%$ (signal likely from a
neighbour), and {offset} show a significant difference-image offset ($>4\\sigma$ and $>0.5$ pixel).
Signals above the NFPP or offset limits would be rejected by these tests; they are kept in the
table to show the full group.''')
    rows = []
    for toi, t, e, extra in inconclusive:
        f = (extra or {}).get('fpp') or {}
        px = (extra or {}).get('pixel_test') or {}
        m = e.get('metrics') or {}
        rows.append(' & '.join([f'TOI-{toi}' + ('$^\\dagger$' if e.get('cnn_selected') else ''), str(t), f"{e['period_days']:.4f}", num(m.get('depth_ppm'), '{:.0f}'),
                                num(m.get('snr'), '{:.1f}'), num(m.get('radius_earth'), '{:.1f}'),
                                num(e.get('cnn_calibrated'), '{:.2f}'), num(f.get('fpp'), '{:.3f}'),
                                exp_tex(f.get('nfpp')), num(px.get('offset_arcsec'), '{:.1f}'),
                                num(px.get('sigma'), '{:.1f}')]) + ' \\\\')
    write('inconclusive.tex', '''
\\startlongtable
\\begin{deluxetable*}{lcccccccccc}
\\tablecaption{Initially inconclusive recovered TOIs: supplementary source tests and false-positive probabilities.\\label{tab:inconclusive}}
\\tablehead{\\colhead{TOI} & \\colhead{TIC} & \\colhead{$P$ (d)} & \\colhead{Depth (ppm)} & \\colhead{SNR} &
\\colhead{$R_p$ ($R_\\oplus$)} & \\colhead{CNN} & \\colhead{FPP} & \\colhead{NFPP} & \\colhead{Offset ($''$)} & \\colhead{Offset ($\\sigma$)}}
\\startdata
''' + '\n'.join(rows) + '''
\\enddata
\\tablecomments{Depth, SNR and radius from the box model of the vetting; FPP and NFPP from three
TRICERATOPS runs; offset of the difference-image source from the target. Dashes: not computed or
image not significant. Membership in this table follows the initial search tier; supplementary rejection criteria are discussed in the text.}
\\end{deluxetable*}''')

    # Uncatalogued signals that pass the vetting.
    rows = []
    for t, e in new:
        m = e.get('metrics') or {}
        extra = next((x for x in supplementary.get(t, {}).get('events', [])
                      if abs(x['period_days'] - e['period_days']) < 1e-6), None)
        f = (extra or e).get('fpp') or {}
        px = (extra or {}).get('pixel_test') or m.get('pixel_test') or {}
        hosts = ', '.join(f'TOI-{x}' for x in open_tois[t])
        rows.append(' & '.join([str(t), hosts, f"{e['period_days']:.4f}", f"{e['epoch_bjd'] - 2460000:.4f}",
                                f"{e['duration_hours']:.2f}", num(m.get('depth_ppm'), '{:.0f}'), num(m.get('snr'), '{:.1f}'),
                                num(m.get('transits'), '{}'), num(m.get('radius_earth'), '{:.1f}'),
                                num(e.get('cnn_calibrated'), '{:.2f}'), num(f.get('fpp'), '{:.3f}'),
                                exp_tex(f.get('nfpp')), num(px.get('offset_arcsec'), '{:.1f}')]) + ' \\\\')
    write('new_text.tex', f'''
A total of {len(new)} signals in these stars pass the photometric vetting but match no catalogued signal
(Table~\\ref{{tab:new}}). None is selected by the classifier. Each lies in the host star of an open
TOI and does not coincide with it in period or phase; they are listed with the same supplementary
tests as a starting point for inspection, not as planet candidates.''')
    write('new.tex', '''
\\startlongtable
\\setlength{\\tabcolsep}{3.5pt}
\\begin{deluxetable*}{lcccccccccccc}
\\tabletypesize{\\scriptsize}
\\tablecaption{Signals that pass the photometric vetting and match no catalogued signal.\\label{tab:new}}
\\tablehead{\\colhead{TIC} & \\colhead{Host TOI} & \\colhead{$P$ (d)} & \\colhead{$T_0-2460000$} & \\colhead{$T_{14}$ (h)} &
\\colhead{Depth (ppm)} & \\colhead{SNR} & \\colhead{$N_{\\rm tr}$} & \\colhead{$R_p$ ($R_\\oplus$)} & \\colhead{CNN} &
\\colhead{FPP} & \\colhead{NFPP} & \\colhead{Offset ($''$)}}
\\startdata
''' + '\n'.join(rows) + '''
\\enddata
\\tablecomments{Box-model parameters from the vetting; $T_0$ in BJD$_{\\rm TDB}$. FPP, NFPP and difference-image offset from
the supplementary analysis (three TRICERATOPS runs).}
\\end{deluxetable*}''')

    # Long-period injection-recovery.
    injection = Path(sys.argv[1]) if len(sys.argv) > 1 else PAPER / 'data' / 'long_period_injections.json'
    if injection and injection.exists():
        res = json.loads(injection.read_text())
        fair = [r for r in res if not (r['flags'] or {}).get('too_large')]
        parts = []
        for lo_, hi_ in ((8, 12), (12, 20), (20, 40), (40, 1e9)):
            b = [r for r in fair if lo_ <= r['expected_snr'] < hi_]
            if b:
                label = f'{lo_}--{hi_}' if hi_ < 1e8 else f'$>{lo_}$'
                parts.append(f"{label}: {100 * np.mean([r['recovered'] for r in b]):.0f}\\% detected and "
                             f"{100 * np.mean([r['verdict'] == 'pass' for r in b]):.0f}\\% passing ({len(b)} injections)")
        write('single_injection.tex', 'For single transits with periods of 150--1000\\,d injected into the '
              'same light curves, by expected SNR: ' + '; '.join(parts) + '.')
    elif not (TABLES / 'single_injection.tex').exists():
        raise FileNotFoundError('Long-period injection results are required for the paper')
    print('tables written:', sorted(p.name for p in TABLES.glob('*.tex')))
    print(f'recovered {len(recovered)}/{n_open}, missed {len(missed)}, inconclusive {len(inconclusive)} '
          f'({len(done)} with supplementary), new {len(new)}')


if __name__ == '__main__':
    main()
