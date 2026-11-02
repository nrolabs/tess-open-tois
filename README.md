# TESS — exoplanet candidate search and vetting

End-to-end pipeline on public TESS data:

1. **Collection** of the TOI catalog and of every SPOC 120 s light curve of the labeled stars.
2. **Blind search** for periodic transits (BLS on the GPU) in each star.
3. **Multi-view CNN** (5-model ensemble) scoring each signal.
4. **Local false positive tests**: SNR, number of transits, odd/even, secondary eclipse, shape,
   physical size, stellar density, transit source location from the centroid, and contamination.
5. **False positive probability** (FPP/NFPP) with TRICERATOPS, using TIC/Gaia neighbors and the
   SPOC apertures.
6. **Ranking** in `reports/candidates.csv`, separating catalogued signals from new ones. A signal is
   "catalogued" when it matches a TOI, a community TOI (ExoFOP CTOI) or a published planet (NASA
   Exoplanet Archive), by period or harmonic and phase; single-transit CTOIs match by event time and
   non-transiting planets by period. SPOC TCEs and known eclipsing binaries are not checked yet.

The final result is a **candidate with statistical evidence**, not a confirmed planet. Confirmation
needs observations a computer cannot replace (high-resolution imaging, spectroscopy, radial
velocities). Strong candidates can be submitted to the community through
[ExoFOP-TESS](https://exofop.ipac.caltech.edu/tess/).

## Commands

Requirements: Python 3.12 with `venv`/`pip` (main environment), Python 3.13 (TRICERATOPS
environment), `curl`, GNU Make, an NVIDIA GPU with CUDA, and internet access.

```bash
make setup PYTHON=/path/to/python3
make catalog
make curves
make test
make train
make candidates
```

| Command | What it does |
|---|---|
| `make setup` | Creates `.venv` (CNN, PyTorch CUDA) and `.venv-fpp` (TRICERATOPS) and tests the GPU |
| `make catalog` | Downloads the TOI catalog, CTOIs and published planets, and records source/hash |
| `make curves` | Downloads/resumes every SPOC 120 s light curve of CP/KP/FP/FA stars |
| `make curves FIELD=100` | Downloads 100 **non-TOI** stars with ≥30 sectors (fixed sample, all sectors) to search for new planets |
| `make train` | Cross-validation by star, final ensemble, calibration and injection-recovery |
| `make candidates` | Searches every downloaded star (BLS on the GPU); `TIC=...` for one star; `FIELD=1` for the non-TOI sample only |
| `make test` | Local tests, no network |

Vet a specific event (period in days, epoch in BJD TDB, duration in hours):

```bash
make candidates TIC=1528696 PERIOD=0.8820624 EPOCH=2458438.402842 DURATION=0.5759672
```

On this machine the system Python has no pip, hence the `PYTHON=` above. TRICERATOPS 1.1.0 needs
`numpy<2` on Python 3.12, so it runs isolated in `.venv-fpp` (Python 3.13; `FPP_PYTHON=` changes
the interpreter) and the pipeline calls it as a subprocess.

## Code

| File | Responsibility |
|---|---|
| `tess.py` | Command-line interface used by the Makefile |
| `data.py` | TOI catalog, MAST manifests, light curve download and loading |
| `lightcurve.py` | Transit-protected detrending, phase folding, CNN views |
| `vetting.py` | False positive tests and the CNN scalar features |
| `neural.py` | Dataset with per-star cache, CNN, cross-validated training, inference |
| `search.py` | GPU BLS search, candidate pipeline, injection-recovery |
| `fpp.py` | TRICERATOPS runner (isolated environment) |

## Data collection

- Catalog: [NASA Exoplanet Archive / TAP](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html), query `select * from toi`.
- [TOI definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html): `CP`/`KP` → class 1;
  `FP`/`FA` → class 0. `PC`, `APC` and unknown dispositions are left out of training.
- Light curves: the [MAST per-sector manifests](https://archive.stsci.edu/tess/bulk_downloads.html)
  (`tesscurl_sector_N_lc.sh`) list every SPOC 120 s file. Filtering them by catalog TICs gives the
  exact file list, with no star-by-star search. Files come from the public AWS copy (`stpubdata`,
  Open Data), with MAST as fallback. The TIC comes from the file name: exact match, no name
  resolution to neighboring stars.
- `--include-candidates` (in `python tess.py download`) also downloads PC/APC stars, which can be
  searched but are not used for training.
- Interrupt with Ctrl+C and rerun `make curves` to resume. FITS files truncated by interrupted
  downloads are detected and downloaded again. Original FITS files are kept.

In 2026-11 there were 2,657 eligible labeled TICs, of which 2,357 have SPOC 120 s light curves:
15,834 files, ~29 GB of FITS and ~2.5 GB of `.npz`. At ~0.9 MB/s this takes ~10 hours.

## Model

- Input: global view (full orbit), local view (±3 durations), odd and even transits side by side,
  and secondary view (phase 0.5), each with a coverage channel, plus 14 measured scalars (depth,
  SNR, odd/even, secondary, shape, duration/period, transits, centroid, contamination, radius,
  log g, Teff, magnitude and density). IDs, dispositions and catalog physical parameters are not
  inputs; the catalog period/epoch/duration only drive the alignment.
- Each TOI becomes one example with **all sectors combined**. In training, each single sector is
  also an example (augmentation); validation and test use only the combined example.
- Other planets of the same star are masked while measuring a signal.
- The catalog ephemeris is refined within 3σ of its uncertainties, and sectors where it cannot
  locate the transit (3σ > 1 duration) are dropped.
- 1D CNN with residual blocks, one branch per view. CUDA-only, deterministic, mixed-precision
  training.
- **Nested cross-validation by TIC**: 5 folds; each star is tested exactly once by a model that
  never saw it; each fold's validation set comes from the remaining stars. The final model is the
  ensemble of the 5. The threshold (target precision `PRECISION`) and the ensemble's isotonic
  calibration come from the out-of-fold predictions. If no threshold reaches the precision with 5
  selections, `threshold: null`.
- The report records per-fold, pooled and per-star metrics, metrics by depth, period, SNR and
  number of sectors, the rejection rate of each vetting test per class, and injection-recovery.

## False positive tests (`vetting.py`)

| Test | Rejects (`fail`) | Warns (`warn`) |
|---|---|---|
| SNR (box of 80% of the duration, correlated noise and event-to-event scatter) | < 7.1 | |
| Transits with data | < 2 | |
| Odd/even (≥2 events in each group) | > 4σ | > 3σ |
| Secondary eclipse at any phase, Bonferroni threshold, must exceed the largest brightening and imply albedo > 1 | > 5σ | above threshold |
| Companion radius (depth × stellar radius) | > 25 R⊕ | |
| Transit source from the centroid (source vs. target position, via WCS) | > 4σ and > 21″ | > 3σ and > 10″ |
| V shape (trapezoid fit) | | ingress > 40% |
| Duration too long for the stellar density | | ratio < 0.2 |
| Contamination (CROWDSAP) | | < 0.8 |

These thresholds will be recalibrated on the full dataset using `vetting_by_class` in
`reports/training.json`. On an early sample (116 TOIs), 16% of CP/KP and 36% of FP/FA were
rejected. Part of the planet rejections is low SNR on stars that still had few sectors downloaded.
Many catalog FPs were identified only with ground-based observations that vetting cannot see; those
are left to the CNN and the NFPP.

## Final tiers (`make candidates`)

| Tier | Criterion |
|---|---|
| `strong` | CNN above threshold, vetting passed, FPP < 1% and NFPP < 0.1% (stricter version of the [Giacalone et al. 2021](https://ui.adsabs.harvard.edu/abs/2021AJ....161...24G) criterion, which uses FPP < 1.5%) |
| `candidate: likely planet` | CNN above threshold, vetting passed, FPP < 50% and NFPP < 0.1% |
| `candidate: no FPP` | CNN and vetting passed, TRICERATOPS not computed |
| `inconclusive` | CNN below threshold or high FPP |
| `rejected` | Vetting failed, or NFPP > 10% (signal likely from a nearby star) |

Even `strong` is statistical validation **without** high-resolution imaging: without a contrast
curve, TRICERATOPS cannot rule out very close companions. Treat it as a follow-up priority.

TRICERATOPS downloads a TESScut image cutout per sector; with a slow or busy connection this can
take tens of minutes per candidate, and a failed cutout is reported as an FPP error.

## Transit search (`search.py`)

The BLS search runs on the GPU (PyTorch, float64): for each batch of periods the light curve is
folded into phase bins and every window and duration is evaluated with cumulative sums. Design,
following the SPOC TPS:

- **Red-noise-aware SNR.** The noise of the in-transit mean is measured at each tested duration's
  timescale, so correlated noise is included. Detection requires SNR ≥ 7.1 (the TPS MES threshold).
  The SDE is reported but not used as a gate: it rejected real signals with few transits.
- **Narrow period ranges** (0.5–100 d in 13 ranges), each testing durations from 0.5 h up to a
  quarter of its shortest period, so short-period, long-duration transits are not found at 2P.
- **Consistency veto.** A detection that does not survive removing its strongest event (it should
  keep about SNR·√((N−1)/N)) rests on a single event and is rejected.
- **Edge veto.** Events whose in-transit points mostly lie within 0.5 d of a data gap are dropped
  and the rest must still carry the signal (same rule as the consistency veto); scattered-light ramps
  and filter edges were the main source of false alarms.
- **Period alias check** (as in the SPOC DV): P/2 and P/3 are preferred when folding there is
  stronger and consistent.
- Up to 3 signals per star, masking ±1.5 durations after each one; harmonics of a signal already
  found are not reported again.

Calibration on 300 single-sector injections into planet-host stars (known transits removed),
binned by expected SNR (white noise at 10-minute bins), with false alarms measured on inverted
light curves, where no astrophysical transit exists:

| Expected SNR | 7.1–10 | 10–15 | 15–25 | 25–50 | > 50 | False alarms (inverted) |
|---|---|---|---|---|---|---|
| Before (white-noise SNR + SDE gate) | 13% | 27% | 65% | 85% | 83% | ~4% |
| Now | 12–18% | 67–73% | 88–90% | 93–98% | 98–99% | 19% |

Higher false alarms are the intended trade-off: a missed planet is lost for good, while a false
alarm still goes through vetting, the CNN and TRICERATOPS.

**Detrending** is a Tukey biweight in a 0.75-day sliding time window (the wotan method), on the GPU
in float64; it matches wotan within a median 0.8 ppm and runs 7.7× faster. The window was chosen on a
frozen tuning set (paired comparison of 0.5, 0.75, 1.0 and 1.5 d at matched false-alarm rate).

**Benchmark against the reference tools** (wotan biweight + TLS 2.0.4), on a held-out set of 300
single-sector injections into planet-host stars never used for tuning, each pipeline taking its single
best peak, evaluated once, at the same false-alarm rate on inverted light curves:

| Held-out set | Method | Recovery at 1% FA | 5% FA | 10% FA | SNR 7–10 | SNR 10–15 | SNR 15–25 | SNR > 25 |
|---|---|---|---|---|---|---|---|---|
| 1st (earlier version, global edge veto) | This pipeline | 81% | 85% | 87% | 0% | 69% | 91% | 92% |
| 1st | wotan + TLS (SDE) | 81% | 83% | 85% | 20% | 50% | 91% | 90% |
| 2nd (current version, new injections) | This pipeline | 74% | 79% | 84% | 17% | 64% | 76% | 93% |
| 2nd | wotan + TLS (SDE) | 69% | 77% | 77% | 50% | 60% | 70% | 87% |

On the second held-out set (197 detectable cases) the difference is +3.5, +2.7 and +5.9 points at
1%, 5% and 10% FA, with bootstrap 95% intervals of −6.3 to +16.0, −3.4 to +8.5 and −0.5 to +11.9:
consistently ahead on both independent sets, but not yet statistically established. TLS remains
better on the weakest signals (SNR 7–10). Rejected after testing on the tuning set: finer period,
phase and duration grids (more trials raise the noise peaks as much as they raise the signal).
A limb-darkened matched filter (the TLS idea) was evaluated and not adopted: for small planets its
theoretical SNR gain over the optimal box is only ~1%.

To avoid overfitting, settings are tuned only on one half of the host stars (split by TIC hash) and
reported on the other half with different injections. A star with an 8-year baseline (473 thousand
periods) takes ~3 min. The training report measures recovery of the full pipeline (BLS, vetting and
CNN).

## Generated files

| Path | Content |
|---|---|
| `data/catalogs/toi.json`, `.csv`, `snapshot.json`, `snapshots/` | Catalog, source, hash and versions |
| `data/catalogs/known.json` | CTOIs and published planets used to tell catalogued signals from new ones |
| `data/manifests/sector-N.sh` | Official per-sector manifests (cache) |
| `data/mast/lc/*.fits` | Original FITS files, kept |
| `data/curves/<TIC>/sector-N.npz` | BJD TDB time, PDCSAP flux, error and MOM_CENTR centroids |
| `data/curves/<TIC>/sector-N.json` | Product, CROWDSAP, TIC stellar parameters, target CCD position |
| `data/field.json` | Current non-TOI sample (`FIELD=N`) |
| `data/features/<TIC>.pkl` | Per-star feature and metric cache |
| `data/fpp/` | TRICERATOPS inputs and outputs |
| `data/download-log.json` | Errors from the last download |
| `models/transit-cnn.pt` | Ensemble, threshold and calibration |
| `reports/training.json` | Metrics, folds, subgroups, vetting per class, injection-recovery |
| `reports/skipped.json` | Excluded examples and reason |
| `reports/candidates/<TIC>.json` | Signals of each star with metrics, CNN, FPP and tier |
| `reports/candidates.csv` | Ranking of every signal |

## Limitations

- TOI labels (CP/KP vs FP/FA) do not represent the mix of signals in a blind search: the CNN
  calibration applies to that population, not to "any dip in brightness".
- The search only covers downloaded stars. There are 585,858 non-TOI stars with 120 s light curves
  (~3 TB); `FIELD=N` downloads a fixed sample of the 4,662 with ≥30 sectors (none reach 50), with all
  sectors. SPOC already searched these stars: the discovery space is in weak signals that only
  appear when sectors are combined.
- SPOC 120 s light curves only; FFIs (QLP, TESS-SPOC) and full pixel data are not used.
- Injection-recovery uses trapezoidal transits without limb darkening, one sector at a time.
- Systems with strong transit timing variations (TTV) may fail the ephemeris tests.
- The project never submits anything automatically to ExoFOP, Zooniverse or any other service.

## Roadmap (noted, not implemented)

Evolution proposal discussed on 2026-11-01. These are priorities, not demonstrated improvements.

| Priority | Feature | Expected benefit | Notes |
|---|---|---|---|
| P0 | Difference images and pixel analysis | Identify the true transit source | Needs TPFs (~10× the light curves): download only for candidates |
| P0 | Training on blind-search signals | Fewer false positives outside the TOI population | Population: SPOC TCEs ([TCE bulk](https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_tce.html)) + [TESS eclipsing binary catalog](https://cdsarc.cds.unistra.fr/ftp/J/ApJS/258/16) (Prša et al. 2022) |
| P0 | Benchmark against ExoMiner++ | Measurable comparison | Published scores for 147,568 TCEs S1–67 ([Zenodo](https://zenodo.org/record/15466292), [paper](https://arxiv.org/abs/2502.09790)); same TCEs and labels; superiority is not guaranteed |
| P0 | Cross-match signals with TCEs and known binaries | Reliable "new" label | TOIs, CTOIs and published planets are already checked; the binary catalog (VizieR) was unreachable when the cross-match was added |
| P1 | TLS + BLS + single-transit search | Signals BLS misses | Single transits: long-period planets |
| P1 | FFI data and 20 s light curves | More stars and signals | Much larger volume; limited by bandwidth and disk |
| P1 | Temporal model with attention | Dependencies across events and sectors | Evaluate against the current CNN in the benchmark |
| P1 | TTV handling | Planets that are not strictly periodic | |
| P1 | Correlated noise model | Sensitivity to weak signals | |
| P2 | Heterogeneous ensemble | Combine different errors | |
| P2 | Uncertainty and out-of-distribution detection | Avoid overconfidence | |
