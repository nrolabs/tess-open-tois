"""Candidate report for an ExoFOP CTOI submission: MCMC transit fit, figures, LaTeX/PDF and the CTOI row.

The fit uses a limb-darkened transit model (batman, quadratic law with the Kipping 2013 q1/q2
parameterization) sampled with emcee. Free parameters: mid-transit time, period, Rp/R*, impact
parameter, stellar density (sets a/R*), q1, q2, a flux offset and the white noise. Correlated noise
is included by scaling the errors with the time-averaging β factor (Winn et al. 2008) measured on
the best-fit residuals. For single events the period is fixed at the shortest value the coverage
and the stellar density allow, and for duos at the best alias: only the shape is fitted then.

Nothing is submitted anywhere: the output is a folder with the PDF, the figures, the posterior
samples and a CSV row using the column names of the ExoFOP CTOI table.
"""
import json
import subprocess

import numpy as np
from data import ROOT, load_star, save_json
from lightcurve import prepare, wrap

OUT = ROOT / 'reports' / 'ctoi'
AUTHOR = 'Isak Paulo de Andrade Ruas'
INSTITUTION = 'NROLABS'
G_CGS = 6.674e-8
RSUN_CM = 6.957e10
AU_CM = 1.496e13
REARTH_PER_RSUN = 109.1
EXPOSURE_DAYS = 120 / 86400
WALKERS = 48
# The chain runs in blocks until it is longer than 50 autocorrelation times and the estimate is stable
# (emcee's recommended criterion); grazing transits correlate b, Rp/R* and density and need long chains.
BLOCK = 2000
MAX_STEPS = 60000


# ---- Event selection ----

def rank(tier):
    order = ['strong', 'candidate', 'inconclusive', 'rejected']
    return next((i for i, o in enumerate(order) if tier.startswith(o)), len(order))


def stored_event(tic, index=None):
    """Event from the star's candidate report: the given index, else the best-ranked one."""
    path = ROOT / 'reports' / 'candidates' / f'{tic}.json'
    if not path.exists():
        return None
    events = [e for e in json.loads(path.read_text())['events'] if 'tier' in e]
    if not events:
        return None
    if index is not None:
        return events[index]
    return min(events, key=lambda e: (rank(e['tier']), -(e.get('snr') or 0)))


def evaluate(tic, period, epoch, duration_hours):
    """Full candidate pipeline for one given event (vetting, CNN, pixels, FPP), without touching the
    search reports."""
    from neural import load_ensemble, score
    from search import star_events, pixel_job, fpp_job, tier
    from pixels import apply_pixel_test
    event = star_events(tic, {'period_days': period, 'epoch_bjd': epoch, 'duration_hours': duration_hours})[0]
    if 'metrics' not in event:
        raise SystemExit(f"Event could not be analysed: {event.get('error')}")
    models, checkpoint = load_ensemble()
    raw, calibrated = score(models, checkpoint, np.stack([event.pop('x')]))
    event['cnn_score'], event['cnn_calibrated'] = float(raw[0]), float(calibrated[0])
    event['cnn_selected'] = bool(raw[0] >= checkpoint['threshold'])
    if event['metrics']['verdict'] == 'pass':
        apply_pixel_test(event['metrics'], pixel_job(event))
        event['fpp'] = fpp_job(event)
    else:
        event['fpp'] = None
    fpp = event['fpp'] if event['fpp'] and 'error' not in event['fpp'] else None
    event['tier'] = tier(event['cnn_selected'], event['metrics'], fpp)
    return event


FPP_RUNS = 10


def precise_fpp(event):
    """TRICERATOPS again with FPP_RUNS Monte Carlo runs (the search uses 3): its FPP scatters by tens
    of percent between runs, so a report quotes the mean of many runs and the error of that mean."""
    from search import FPP_PYTHON
    from data import DATA
    period = event['metrics']['period_days']
    source = DATA / 'fpp' / f"{event['tic']}-{period:.6f}.json"
    if not source.exists():
        return event.get('fpp')
    job = json.loads(source.read_text())
    job['runs'] = FPP_RUNS
    work = source.parent
    target_in, target_out = work / f'{source.stem}-report.json', work / f'{source.stem}-report.out.json'
    target_in.write_text(json.dumps(job))
    result = subprocess.run([str(FPP_PYTHON), str(ROOT / 'fpp.py'), str(target_in), str(target_out)],
                            capture_output=True, text=True, cwd=work, timeout=7200)
    if result.returncode or not target_out.exists():
        print(f'Precise FPP failed, keeping the search value: {(result.stderr or result.stdout)[-300:]}', flush=True)
        return event.get('fpp')
    out = json.loads(target_out.read_text())
    out['runs'] = FPP_RUNS
    out['fpp_error_of_mean'] = out['fpp_std'] / np.sqrt(FPP_RUNS)
    out['nfpp_error_of_mean'] = out['nfpp_std'] / np.sqrt(FPP_RUNS)
    return out


# ---- Stellar parameters ----

def tic_star(tic):
    """TIC v8 radius, density, Teff and their errors (MAST); falls back to the light curve header."""
    try:
        from astroquery.mast import Catalogs
        row = Catalogs.query_criteria(catalog='Tic', ID=int(tic))[0]
        value = lambda k: float(row[k]) if row[k] is not None and np.isfinite(float(row[k])) else None
        return {k: value(k) for k in ('rad', 'e_rad', 'rho', 'e_rho', 'Teff', 'e_Teff', 'logg', 'Tmag', 'd',
                                      'ra', 'dec')}
    except Exception as exc:
        return {'error': str(exc)[-200:]}


# ---- Transit model and sampler ----

class Model:
    """batman model on fixed times; theta = [t0, P, k, b, log10 ρ*(g/cm³), q1, q2, f0, log σ]."""

    def __init__(self, t, f, fixed_period=None):
        import batman
        self.t, self.f, self.fixed_period = t, f, fixed_period
        self.params = batman.TransitParams()
        p = self.params
        p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0., 1., .1, 10., 90., 0., 90.
        p.u, p.limb_dark = [.4, .2], 'quadratic'
        # 120 s cadence: supersampling avoids smearing ingress and egress.
        self.model = batman.TransitModel(p, t, supersample_factor=7, exp_time=EXPOSURE_DAYS)
        self.beta = 1.

    def unpack(self, theta):
        if self.fixed_period is not None:
            t0, k, b, log_rho, q1, q2, f0, log_s = theta
            period = self.fixed_period
        else:
            t0, period, k, b, log_rho, q1, q2, f0, log_s = theta
        return t0, period, k, b, log_rho, q1, q2, f0, log_s

    def light_curve(self, theta):
        t0, period, k, b, log_rho, q1, q2, f0, _ = self.unpack(theta)
        a = a_over_r(10 ** log_rho, period)
        p = self.params
        p.t0, p.per, p.rp, p.a = t0, period, k, a
        p.inc = np.degrees(np.arccos(min(b / a, 1.)))
        p.u = [2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)]
        return self.model.light_curve(p) * (1 + f0)


def a_over_r(rho_cgs, period_days):
    return (G_CGS * rho_cgs * (period_days * 86400) ** 2 / (3 * np.pi)) ** (1 / 3)


_MODEL, _BOUNDS = None, None


def log_probability(theta):
    t0, period, k, b, log_rho, q1, q2, f0, log_s = _MODEL.unpack(theta)
    names = _MODEL_NAMES()
    for name, value in zip(names, theta):
        lo, hi = _BOUNDS[name]
        if not lo < value < hi:
            return -np.inf
    if b > 1 + k:
        return -np.inf
    a = a_over_r(10 ** log_rho, period)
    if a <= 1 + k:
        return -np.inf
    sigma = np.exp(log_s) * _MODEL.beta
    residual = _MODEL.f - _MODEL.light_curve(theta)
    return -.5 * np.sum(residual ** 2) / sigma ** 2 - len(residual) * np.log(sigma)


def _MODEL_NAMES():
    names = ['t0', 'period', 'k', 'b', 'log_rho', 'q1', 'q2', 'f0', 'log_sigma']
    return [n for n in names if not (n == 'period' and _MODEL.fixed_period is not None)]


def _init_worker(model, bounds):
    global _MODEL, _BOUNDS
    _MODEL, _BOUNDS = model, bounds


def beta_factor(t, residual, duration):
    """Time-averaging β (Winn et al. 2008): binned scatter over the white-noise expectation, averaged
    over bin sizes from 10 minutes to the transit duration; at least 1."""
    white = np.std(residual)
    betas = []
    for width in np.linspace(10 / 1440, max(duration, 20 / 1440), 8):
        cells = np.floor((t - t[0]) / width).astype(int)
        _, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
        means = np.bincount(inverse, residual) / counts
        full = counts >= max(3, .5 * np.median(counts))
        if full.sum() < 10:
            continue
        n = np.median(counts[full])
        expected = white / np.sqrt(n) * np.sqrt(full.sum() / max(full.sum() - 1, 1))
        betas.append(np.std(means[full]) / expected)
    return float(max(1., np.median(betas))) if betas else 1.


def fit(t, f, guess, fixed_period=None, max_steps=MAX_STEPS, workers=8):
    """MAP fit, β factor, then emcee. Returns samples (dict of arrays), the MAP vector and diagnostics."""
    import emcee
    import multiprocessing
    from scipy.optimize import minimize
    global _MODEL, _BOUNDS
    model = Model(t, f, fixed_period)
    duration = guess['duration']
    bounds = {'t0': (guess['t0'] - duration / 2, guess['t0'] + duration / 2),
              'period': (guess['period'] * (1 - 1e-3), guess['period'] * (1 + 1e-3)),
              'k': (1e-3, .5), 'b': (0., 1.5), 'log_rho': (-3., 2.), 'q1': (0., 1.), 'q2': (0., 1.),
              'f0': (-.01, .01), 'log_sigma': (-14., -2.)}
    _MODEL, _BOUNDS = model, bounds
    names = _MODEL_NAMES()
    start = {'t0': guess['t0'], 'period': guess['period'], 'k': np.sqrt(max(guess['depth'], 1e-6)),
             'b': .3, 'log_rho': np.log10(guess['rho']), 'q1': .36, 'q2': .3, 'f0': 0.,
             'log_sigma': np.log(np.std(f))}
    x0 = np.array([start[n] for n in names])
    best = minimize(lambda x: -log_probability(x) if np.isfinite(log_probability(x)) else 1e25, x0,
                    method='Nelder-Mead', options={'maxiter': 20000, 'maxfev': 20000, 'xatol': 1e-9, 'fatol': 1e-9})
    x_map = best.x if np.isfinite(log_probability(best.x)) else x0
    model.beta = beta_factor(t, f - model.light_curve(x_map), duration)
    ndim = len(names)
    scale = np.array([{'t0': 1e-4, 'period': 1e-6, 'k': 1e-3, 'b': .02, 'log_rho': .02, 'q1': .02,
                       'q2': .02, 'f0': 1e-5, 'log_sigma': .01}[n] for n in names])
    walkers = []
    rng = np.random.default_rng(1)
    while len(walkers) < WALKERS:
        trial = x_map + scale * rng.normal(size=ndim)
        if np.isfinite(log_probability(trial)):
            walkers.append(trial)
    # The model has no CUDA state, so forked workers are safe and use the CPU cores the search leaves free.
    converged, tau, previous = False, np.full(ndim, np.nan), np.inf
    with multiprocessing.get_context('fork').Pool(workers, initializer=_init_worker,
                                                  initargs=(model, bounds)) as pool:
        # Differential-evolution moves follow the strong b–Rp/R*–density correlation of grazing transits,
        # which the default stretch move explores slowly.
        moves = [(emcee.moves.DEMove(), .8), (emcee.moves.DESnookerMove(), .2)]
        sampler = emcee.EnsembleSampler(WALKERS, ndim, log_probability, pool=pool, moves=moves)
        state = np.array(walkers)
        while sampler.iteration < max_steps:
            state = sampler.run_mcmc(state, BLOCK, progress=False)
            tau = emcee.autocorr.integrated_time(sampler.get_chain(), quiet=True)
            worst = float(np.max(tau))
            print(f'  MCMC {sampler.iteration} steps: max autocorrelation {worst:.0f} steps', flush=True)
            if sampler.iteration > 50 * worst and abs(previous - worst) / worst < .05:
                converged = True
                break
            previous = worst
    # Burn-in: five autocorrelation times; thinning by half of one.
    burn = int(5 * np.max(tau)) if np.all(np.isfinite(tau)) else sampler.iteration // 2
    chain = sampler.get_chain(discard=burn, thin=max(1, int(np.min(tau) / 2)), flat=True)
    samples = {n: chain[:, i] for i, n in enumerate(names)}
    if fixed_period is not None:
        samples['period'] = np.full(len(chain), fixed_period)
    diagnostics = {'beta': model.beta, 'acceptance': float(np.mean(sampler.acceptance_fraction)),
                   'autocorrelation_steps': {n: float(v) for n, v in zip(names, tau)},
                   'steps': int(sampler.iteration), 'burn': burn, 'walkers': WALKERS, 'converged': converged,
                   'independent_samples': int(len(chain) * max(1, int(np.min(tau) / 2)) / np.max(tau))}
    return samples, dict(zip(names, x_map)), diagnostics, model


def derived(samples, star, rng):
    """Physical parameters per posterior sample (stellar radius and Teff drawn from the TIC errors)."""
    n = len(samples['k'])
    k, b, period = samples['k'], samples['b'], samples['period']
    rho = 10 ** samples['log_rho']
    a = a_over_r(rho, period)
    inc = np.degrees(np.arccos(np.clip(b / a, 0, 1)))
    q1, q2 = samples['q1'], samples['q2']
    u1, u2 = 2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)
    arg = np.sqrt(np.clip((1 + k) ** 2 - b ** 2, 0, None)) / (a * np.sin(np.radians(inc)))
    t14 = period / np.pi * np.arcsin(np.clip(arg, 0, 1)) * 24
    mu = np.sqrt(np.clip(1 - b ** 2, 0, 1))
    depth = k ** 2 * (1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2) / (1 - u1 / 3 - u2 / 6) * 1e6
    out = {'a_over_r': a, 'inclination_deg': inc, 'duration_hours': t14, 'depth_ppm': depth,
           'density_cgs': rho, 'u1': u1, 'u2': u2}
    if star.get('rad'):
        radius = star['rad'] + (star.get('e_rad') or .07 * star['rad']) * rng.normal(size=n)
        out['radius_earth'] = k * radius * REARTH_PER_RSUN
        out['a_au'] = a * radius * RSUN_CM / AU_CM
        if star.get('Teff'):
            teff = star['Teff'] + (star.get('e_Teff') or 150.) * rng.normal(size=n)
            out['teq_k'] = teff * np.sqrt(1 / (2 * a))
            out['insolation_earth'] = radius ** 2 * (teff / 5772.) ** 4 / out['a_au'] ** 2
    return out


def summarize(values):
    lo, mid, hi = np.percentile(values, [15.87, 50, 84.13])
    return {'value': float(mid), 'minus': float(mid - lo), 'plus': float(hi - mid)}


# ---- Figures ----

def figures(folder, prep, t, f, model, samples, event, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'figure.dpi': 150})
    period = float(np.median(samples['period']))
    t0 = float(np.median(samples['t0']))
    duration = float(np.median(derived(samples, {}, np.random.default_rng(0))['duration_hours'])) / 24
    names = list(_MODEL_NAMES())
    median = np.array([np.median(samples[n]) for n in names])
    files = {}

    # 1. Full light curve, 30-minute bins, transits marked.
    valid = prep['valid']
    time, flux = prep['time'][valid], prep['corrected'][valid]
    cells = np.floor((time - time[0]) / (30 / 1440)).astype(int)
    _, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
    tb, fb = np.bincount(inverse, time) / counts, np.bincount(inverse, flux) / counts
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    ax.plot(tb - 2457000, fb * 1e3, '.', ms=1.5, color='0.3')
    k = np.arange(np.ceil((time[0] - t0) / period), np.floor((time[-1] - t0) / period) + 1)
    for c in t0 + k * period:
        if np.min(np.abs(time - c)) < duration:
            ax.axvline(c - 2457000, color='tab:red', lw=.6, alpha=.6)
    ax.set_xlabel('BJD$_{\\rm TDB}$ − 2457000'), ax.set_ylabel('Relative flux (ppt)')
    ax.set_title('Detrended light curve (30-min bins); red: transits')
    files['lightcurve'] = save(fig, folder / 'lightcurve.pdf')

    # 2. Folded transit with the posterior median model and residuals.
    phase = wrap(t - t0, period) * 24
    order = np.argsort(phase)
    fine = np.linspace(phase.min(), phase.max(), 1500)
    import batman
    p = model.params
    curve = model.light_curve(median)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6, 4.4), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    top.plot(phase, (f - 1) * 1e3, '.', ms=1, color='0.75')
    bins = np.arange(phase.min(), phase.max() + 1e-9, 10 / 60)
    which = np.digitize(phase, bins)
    centers = [(phase[which == i].mean(), np.mean(f[which == i]), np.std(f[which == i]) / np.sqrt((which == i).sum()))
               for i in np.unique(which) if (which == i).sum() > 3]
    cx, cy, ce = map(np.array, zip(*centers))
    top.errorbar(cx, (cy - 1) * 1e3, ce * 1e3, fmt='o', ms=3, color='k', lw=.8)
    smooth = batman.TransitModel(p, t0 + fine / 24, supersample_factor=7, exp_time=EXPOSURE_DAYS).light_curve(p)
    top.plot(fine, (smooth * (1 + median[names.index('f0')]) - 1) * 1e3, color='tab:red', lw=1.4)
    top.set_ylabel('Relative flux (ppt)'), top.set_title('Phase-folded transit, posterior median model')
    residual = f - curve
    bottom.plot(phase, residual * 1e3, '.', ms=1, color='0.75')
    which_r = [np.mean(residual[which == i]) for i in np.unique(which) if (which == i).sum() > 3]
    bottom.plot(cx, np.array(which_r) * 1e3, 'o', ms=3, color='k')
    bottom.axhline(0, color='tab:red', lw=1)
    bottom.set_xlabel('Hours from mid-transit'), bottom.set_ylabel('Residual (ppt)')
    files['folded'] = save(fig, folder / 'folded.pdf')

    # 3. Odd and even transits.
    number = np.round((t - t0) / period)
    fig, ax = plt.subplots(figsize=(6, 3))
    for select, color, label in ((number % 2 == 0, 'tab:blue', 'even'), (number % 2 == 1, 'tab:orange', 'odd')):
        if select.sum() < 10:
            continue
        w = np.digitize(phase[select], bins)
        pts = [(phase[select][w == i].mean(), np.mean(f[select][w == i])) for i in np.unique(w) if (w == i).sum() > 3]
        if pts:
            x, y = map(np.array, zip(*pts))
            ax.plot(x, (y - 1) * 1e3, 'o-', ms=3, lw=.8, color=color, label=label)
    ax.legend(), ax.set_xlabel('Hours from mid-transit'), ax.set_ylabel('Relative flux (ppt)')
    ax.set_title(f"Odd/even depths: {metrics.get('odd_even_sigma') or 0:.1f}σ difference")
    files['odd_even'] = save(fig, folder / 'odd_even.pdf')

    # 4. Secondary eclipse window (phase 0.5) and full phase curve, 1-hour bins.
    if period < 2 * (prep['time'][-1] - prep['time'][0]):
        full_phase = (prep['phase'][valid] / period) % 1
        # Transit excluded, so the scale shows the out-of-transit phase curve.
        away = np.abs(wrap(prep['phase'][valid], period)) > duration
        edges = np.linspace(0, 1, max(50, int(period * 24)) + 1)
        w = np.digitize(full_phase, edges)
        pts = [(full_phase[away & (w == i)].mean(), np.mean(flux[away & (w == i)])) for i in np.unique(w)
               if (away & (w == i)).sum() > 3]
        x, y = map(np.array, zip(*pts))
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, y * 1e6, '.', ms=3, color='k')
        ax.axvspan(.5 - duration / period / 2, .5 + duration / period / 2, color='tab:red', alpha=.15)
        ax.set_xlabel('Orbital phase'), ax.set_ylabel('Relative flux (ppm)')
        ax.set_title(f"Out-of-transit phase curve; secondary: {metrics.get('secondary_sigma') or 0:.1f}σ "
                     f"(limit {metrics.get('secondary_threshold_sigma') or 0:.1f}σ)", fontsize=8)
        files['secondary'] = save(fig, folder / 'secondary.pdf')

    # 5. Posterior distributions.
    rows = [('k', '$R_p/R_\\star$'), ('b', '$b$'), ('log_rho', '$\\log_{10}\\rho_\\star$ (g cm$^{-3}$)'),
            ('q1', '$q_1$'), ('q2', '$q_2$'), ('log_sigma', '$\\ln\\sigma$')]
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4))
    for ax, (key, label) in zip(axes.flat, rows):
        ax.hist(samples[key], 50, color='0.4', histtype='stepfilled')
        ax.set_xlabel(label), ax.set_yticks([])
    fig.suptitle('Posterior distributions')
    files['posterior'] = save(fig, folder / 'posterior.pdf')
    return files


def pixel_figure(folder, star, event, metrics):
    """Out-of-transit image and difference image of the first usable sector, with target and source."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pixels import tpf_file, read_tpf, difference_image, fit_source, transit_centers
    test = metrics.get('pixel_test') or {}
    used = [d for d in test.get('sectors', []) if d.get('used')]
    if not used:
        return None
    sector = used[0]['sector']
    time, flux, _, target = read_tpf(tpf_file(star, sector))
    if event.get('search') in ('single', 'duo'):
        centers = np.array(event.get('events_bjd') or [event['epoch_bjd']])
    else:
        centers = transit_centers(time, metrics['period_days'], metrics['epoch_bjd'])
    centers = centers[(centers > time[0]) & (centers < time[-1])]
    image = difference_image(time, flux, centers, metrics['duration_hours'] / 24)
    source = fit_source(image['difference'], image['error'])
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    for ax, data, title in ((axes[0], image['oot'], 'Out of transit'),
                            (axes[1], image['difference'], 'Difference (out − in)')):
        shown = ax.imshow(data, origin='lower', cmap='viridis')
        fig.colorbar(shown, ax=ax, fraction=.046)
        ax.plot(*target, 'x', color='white', ms=9, mew=2, label='target (WCS)')
        if source:
            ax.plot(source['x'], source['y'], '+', color='tab:red', ms=11, mew=2, label='fitted source')
        ax.set_title(f'{title}, sector {sector}'), ax.set_xlabel('column'), ax.set_ylabel('row')
    axes[1].legend(fontsize=7, loc='upper right')
    return save(fig, folder / 'difference_image.pdf')


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path.name


# ---- LaTeX ----

def tex(text):
    return (str(text).replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('%', '\\%')
            .replace('&', '\\&').replace('#', '\\#').replace('$', '\\$').replace('~', '\\textasciitilde{}'))


def fmt(entry, digits=3):
    if entry is None:
        return '--'
    if isinstance(entry, dict):
        v, m, p = entry['value'], entry['minus'], entry['plus']
        scale = max(abs(m), abs(p), 1e-12)
        decimals = max(0, int(-np.floor(np.log10(scale))) + 1)
        return f'${v:.{decimals}f}^{{+{p:.{decimals}f}}}_{{-{m:.{decimals}f}}}$'
    if isinstance(entry, float):
        if abs(entry) >= 1000 and digits <= 4:
            return f'{entry:.0f}'
        return f'{entry:.{digits}g}'
    return tex(entry)


def latex(folder, info):
    p, d, m = info['parameters'], info['derived'], info['metrics']
    fpp = info.get('fpp') or {}
    pix = m.get('pixel_test') or {}
    known = info.get('known') or {}
    flags = m.get('flags') or {}
    tests = [
        ('Transit SNR (correlated noise)', fmt(m.get('snr')), '$\\ge 7.1$', flags.get('low_snr', 'pass')),
        ('Transits observed', str(m.get('transits')), '$\\ge 2$ (periodic)', flags.get('single_transit', 'pass')),
        ('Odd/even depth difference ($\\sigma$)', fmt(m.get('odd_even_sigma')), '$\\le 4$', flags.get('odd_even', 'pass')),
        ('Secondary eclipse ($\\sigma$)', fmt(m.get('secondary_sigma')),
         f"$\\le {m.get('secondary_threshold_sigma') or 0:.1f}$", flags.get('secondary_eclipse', 'pass')),
        ('Ingress fraction (V shape)', fmt(m.get('ingress_fraction')), '$\\le 0.4$', flags.get('v_shape', 'pass')),
        ('Planet radius ($R_\\oplus$)', fmt(m.get('radius_earth')), '$\\le 25$', flags.get('too_large', 'pass')),
        ('Centroid source offset (arcsec)', fmt(m.get('source_offset_arcsec')), '4$\\sigma$ and $>$21\\arcsec',
         flags.get('centroid_offset', 'pass')),
        ('Difference image offset (arcsec)', fmt(pix.get('offset_arcsec')), '4$\\sigma$ and $>$10.5\\arcsec',
         flags.get('pixel_offset', 'pass')),
        ('Difference image significance ($\\sigma$)', fmt(pix.get('sigma')), '--', '--'),
        ('CCD common mode ($\\sigma$)', fmt(m.get('common_mode_sigma')), '$\\le 4$', flags.get('systematic', 'pass')),
        ('Crowding (CROWDSAP)', fmt(m.get('crowdsap')), '$\\ge 0.8$', flags.get('crowded', 'pass')),
        ('Stellar density ratio (max/TIC)', fmt(m.get('density_ratio')), '$\\ge 0.2$',
         flags.get('duration_too_long', 'pass')),
    ]
    test_rows = '\n'.join(f'{a} & {b} & {c} & {tex(e)} \\\\' for a, b, c, e in tests)
    param_rows = '\n'.join(f'{label} & {fmt(value)} \\\\' for label, value in (
        ('Mid-transit time $T_0$ (BJD$_{\\rm TDB}$)', p.get('t0')),
        ('Period $P$ (days)', p.get('period')),
        ('Radius ratio $R_p/R_\\star$', p.get('k')),
        ('Impact parameter $b$', p.get('b')),
        ('Stellar density $\\rho_\\star$ (g cm$^{-3}$)', d.get('density_cgs')),
        ('Limb darkening $q_1$', p.get('q1')),
        ('Limb darkening $q_2$', p.get('q2')),
        ('Depth (ppm)', d.get('depth_ppm')),
        ('Duration $T_{14}$ (hours)', d.get('duration_hours')),
        ('$a/R_\\star$', d.get('a_over_r')),
        ('Inclination (deg)', d.get('inclination_deg')),
        ('Planet radius ($R_\\oplus$)', d.get('radius_earth')),
        ('Semi-major axis (au)', d.get('a_au')),
        ('Equilibrium temperature (K, $A=0$)', d.get('teq_k')),
        ('Insolation ($S_\\oplus$)', d.get('insolation_earth'))))
    star = info['star']
    figs = info['figures']
    figure_blocks = '\n'.join(
        f'\\begin{{figure}}[h!]\\centering\\includegraphics[width={w}\\textwidth]{{{figs[key]}}}'
        f'\\caption{{{caption}}}\\end{{figure}}'
        for key, w, caption in (
            ('lightcurve', '1', 'Detrended SPOC PDCSAP light curve (transit-protected biweight).'),
            ('folded', '.8', 'Phase-folded transit with the posterior median limb-darkened model.'),
            ('odd_even', '.7', 'Odd and even transits, binned to 10 minutes.'),
            ('secondary', '.7', 'Phase curve; the shaded band is the secondary eclipse window for a circular orbit.'),
            ('difference_image', '.85', 'SPOC target pixel file: out-of-transit image and difference image. '
                                        'White cross: target catalogue position; red plus: fitted source.'),
            ('posterior', '.9', 'Posterior distributions of the fitted parameters.'))
        if figs.get(key))
    fit = info['fit']
    caveats = [
        'This is a candidate with statistical evidence, not a confirmed planet: confirmation needs '
        'high-resolution imaging, reconnaissance spectroscopy and radial velocities.',
        'TRICERATOPS was run without a high-resolution imaging contrast curve, so close bound '
        'companions are not excluded.' if fpp else 'TRICERATOPS was not computed for this event.',
        f"The MCMC {'converged (chain longer than 50 autocorrelation times, stable estimate)' if fit['converged'] else 'did NOT converge within the step limit: treat the errors with caution'} "
        f"(acceptance {fit['acceptance']:.2f}, {fit['steps']} steps, about {fit['independent_samples']} independent samples); errors are scaled by the time-averaging factor "
        f"$\\beta = {fit['beta']:.2f}$ for correlated noise.",
        'Stellar radius and effective temperature come from TIC v8; if the TIC gives no error, 7\\% '
        '(radius) and 150 K were assumed.']
    if info['event'].get('search') in ('single', 'duo'):
        caveats.append(f"The period was fixed at {info['parameters']['period']['value']:.3f} d "
                       f"({'best alias of the pair' if info['event']['search'] == 'duo' else 'shortest value allowed by the coverage and the stellar density'}); "
                       'the true period may be longer.')
    caveat_rows = '\n'.join(f'\\item {c}' for c in caveats)
    body = f'''\\documentclass[11pt,a4paper]{{article}}
\\usepackage[margin=2.2cm]{{geometry}}
\\usepackage{{graphicx,booktabs,amsmath}}
\\newcommand{{\\arcsec}}{{$^{{\\prime\\prime}}$}}
\\title{{TESS planet candidate around TIC {info['tic']}}}
\\author{{{tex(AUTHOR)} \\\\ {tex(INSTITUTION)}}}
\\date{{\\today}}
\\begin{{document}}
\\maketitle

\\section*{{Summary}}
Signal found by the {tex(info['event'].get('search', 'bls'))} search of a public-data TESS pipeline
(SPOC 120\\,s PDCSAP light curves, sectors {tex(', '.join(str(s) for s in m.get('sectors', [])))}).
Final tier: \\textbf{{{tex(info['tier'])}}}.
Catalogue cross-match (TOIs, CTOIs, published planets, SPOC TCEs, TESS eclipsing binaries):
\\textbf{{{tex(known.get('name') or 'no match')}}}{(' (' + tex(known.get('source')) + ')') if known.get('source') else ''}.
CNN score (calibrated): {fmt(info['event'].get('cnn_calibrated'))}.
TRICERATOPS ({fpp.get('runs', 3)} Monte Carlo runs, mean $\\pm$ error of the mean):
FPP $= {fpp.get('fpp', float('nan')):.3g} \\pm {fpp.get('fpp_error_of_mean', fpp.get('fpp_std', float('nan'))):.2g}$,
NFPP $= {fpp.get('nfpp', float('nan')):.2g}$.

\\section*{{Host star (TIC v8)}}
\\begin{{tabular}}{{ll}}\\toprule
RA, Dec (deg) & {fmt(star.get('ra'), 8)}, {fmt(star.get('dec'), 8)} \\\\
TESS magnitude & {fmt(star.get('Tmag'))} \\\\
Radius ($R_\\odot$) & {fmt(star.get('rad'))} $\\pm$ {fmt(star.get('e_rad'))} \\\\
$T_{{\\rm eff}}$ (K) & {fmt(star.get('Teff'))} $\\pm$ {fmt(star.get('e_Teff'))} \\\\
Density ($\\rho_\\odot$) & {fmt(star.get('rho'))} $\\pm$ {fmt(star.get('e_rho'))} \\\\
Distance (pc) & {fmt(star.get('d'))} \\\\\\bottomrule
\\end{{tabular}}

\\section*{{Transit fit}}
Limb-darkened model (batman, quadratic law, Kipping $q_1,q_2$), emcee with {fit['walkers']} walkers,
{fit['steps']} steps ({fit['burn']} burn-in), run until the chain exceeded 50 autocorrelation times.
Medians and 68\\% intervals.

\\begin{{tabular}}{{lc}}\\toprule
{param_rows}
\\bottomrule\\end{{tabular}}

\\section*{{Vetting}}
\\begin{{tabular}}{{lccc}}\\toprule
Test & Value & Pass condition & Result \\\\\\midrule
{test_rows}
\\bottomrule\\end{{tabular}}

\\section*{{Caveats}}
\\begin{{itemize}}
{caveat_rows}
\\end{{itemize}}

{figure_blocks}
\\end{{document}}
'''
    (folder / 'report.tex').write_text(body)
    result = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', 'report.tex'],
                            cwd=folder, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('LaTeX failed:\n' + (result.stdout + result.stderr)[-2000:])
    subprocess.run(['latexmk', '-c'], cwd=folder, capture_output=True)
    return folder / 'report.pdf'


def ctoi_row(info):
    """CTOI values with the column names of the ExoFOP CTOI table (check the upload template on ExoFOP)."""
    p, d = info['parameters'], info['derived']
    err = lambda e: (e['minus'] + e['plus']) / 2 if e else None
    val = lambda e: e['value'] if e else None
    return {
        'TIC ID': info['tic'], 'Transit Epoch (BJD)': val(p['t0']), 'Transit Epoch (BJD) err': err(p['t0']),
        'Period (days)': val(p['period']), 'Period (days) Error': err(p['period']) if info['event'].get('search') not in ('single', 'duo') else None,
        'Depth ppm': val(d.get('depth_ppm')), 'Depth ppm Error': err(d.get('depth_ppm')),
        'Duration (hrs)': val(d.get('duration_hours')), 'Duration (hrs) Error': err(d.get('duration_hours')),
        'Impact Param': val(p['b']), 'Impact Param Error': err(p['b']),
        'Rad_p/Rad_s': val(p['k']), 'Rad_p/Rad_s Error': err(p['k']),
        'a/Rad_s': val(d.get('a_over_r')), 'a/Rad_s Error': err(d.get('a_over_r')),
        'Inclination (deg)': val(d.get('inclination_deg')), 'Inclination (deg) Error': err(d.get('inclination_deg')),
        'Planet Radius (R_Earth)': val(d.get('radius_earth')), 'Planet Radius (R_Earth) Error': err(d.get('radius_earth')),
        'Insolation (Earth Flux)': val(d.get('insolation_earth')),
        'Equilibrium Temp (K)': val(d.get('teq_k')),
        'Fitted Stellar Density (g/cm^3)': val(d.get('density_cgs')),
        'Fitted Stellar Density (g/cm^3) Error': err(d.get('density_cgs')),
        'User Disposition': 'PC',
        'Notes': (f"{info['event'].get('search', 'bls')} search; tier {info['tier']}; "
                  f"FPP {((info.get('fpp') or {}).get('fpp'))}; NFPP {((info.get('fpp') or {}).get('nfpp'))}; "
                  f"{info['author']} ({INSTITUTION})"),
    }


# ---- Main ----

def make_report(args):
    import csv
    from search import json_safe
    tic = args.tic
    if args.period is not None:
        # The vetting, pixel test and TRICERATOPS of a given event are cached: refits do not redo them.
        cache = OUT / 'evaluated' / f'TIC{tic}-{args.period:.7f}-{args.epoch:.5f}-{args.duration:.3f}.json'
        if cache.exists():
            event = json.loads(cache.read_text())
        else:
            event = json_safe(evaluate(tic, args.period, args.epoch, args.duration))
            save_json(cache, event)
    else:
        event = stored_event(tic, args.event)
        if event is None:
            raise SystemExit(f'No vetted event for TIC {tic}; run make candidates TIC={tic} or give the ephemeris')
    metrics = event['metrics']
    if event.get('fpp') and 'error' not in event['fpp'] and 'fpp_error_of_mean' not in event['fpp']:
        print(f'TRICERATOPS with {FPP_RUNS} Monte Carlo runs', flush=True)
        event['fpp'] = json_safe(precise_fpp(event))
        if args.period is not None:
            save_json(cache, event)
    star = load_star(tic)
    stellar = tic_star(tic)
    search_kind = event.get('search', 'bls')
    fixed = None
    if search_kind == 'single':
        fixed = event.get('fpp_period_days') or 4 * (star['time_bjd'][-1] - star['time_bjd'][0])
    elif search_kind == 'duo':
        fixed = event['period_days']
    period = fixed or metrics['period_days']
    duration = metrics['duration_hours'] / 24
    prep = prepare(star['time_bjd'], star['flux'], period if search_kind != 'single' else 2.5 * (
        star['time_bjd'][-1] - star['time_bjd'][0]), metrics['epoch_bjd'], metrics['duration_hours'],
        star['sector'], event.get('masks') or [])
    near = prep['valid'] & (np.abs(prep['phase']) < max(2.5 * duration, duration + .1))
    t, f = prep['time'][near], 1 + prep['corrected'][near]
    # Reference epoch near the middle of the observed transits decorrelates T0 and P.
    cycles = np.round((t - metrics['epoch_bjd']) / period) if search_kind != 'single' else np.zeros(len(t))
    t0 = metrics['epoch_bjd'] + np.round(np.median(cycles)) * period
    rho = stellar.get('rho')
    rho_cgs = (rho * 1.41) if rho else 1.41
    guess = {'t0': t0, 'period': period, 'depth': (metrics.get('depth_ppm') or 1000) / 1e6,
             'duration': duration, 'rho': rho_cgs}
    folder = OUT / f"TIC{tic}-{search_kind}-{t0:.3f}"
    folder.mkdir(parents=True, exist_ok=True)
    print(f'Fitting TIC {tic}: {len(t)} points around {len(np.unique(cycles))} transits', flush=True)
    samples, _, diagnostics, model = fit(t, f, guess, fixed, max_steps=args.steps)
    rng = np.random.default_rng(2)
    extra = derived(samples, stellar, rng)
    parameters = {n: summarize(v) for n, v in samples.items() if n != 'period' or fixed is None}
    parameters['period'] = summarize(samples['period']) if fixed is None else {'value': fixed, 'minus': 0., 'plus': 0.}
    info = {'tic': tic, 'author': AUTHOR, 'institution': INSTITUTION, 'event': {k: v for k, v in event.items()
            if k not in ('metrics', 'masks', 'fpp', 'known')}, 'metrics': metrics, 'fpp': event.get('fpp'),
            'known': event.get('known'), 'tier': event.get('tier', ''), 'star': stellar,
            'parameters': parameters, 'derived': {k: summarize(v) for k, v in extra.items()},
            'fit': diagnostics}
    info['figures'] = figures(folder, prep, t, f, model, samples, event, metrics)
    try:
        info['figures']['difference_image'] = pixel_figure(folder, star, event, metrics)
    except Exception as exc:
        info['figures']['difference_image'] = None
        print(f'Difference image figure skipped: {exc}', flush=True)
    np.savez_compressed(folder / 'posterior.npz', **samples, **extra)
    save_json(folder / 'summary.json', json_safe({k: v for k, v in info.items() if k != 'metrics'} |
                                                 {'metrics': metrics}))
    row = json_safe(ctoi_row(info))
    with (folder / 'ctoi.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    pdf = latex(folder, info)
    print(f'Report: {pdf}\nCTOI row: {folder / "ctoi.csv"}\nConverged: {diagnostics["converged"]}', flush=True)
    return folder
