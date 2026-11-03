# tess-open-tois

Search and vetting of TESS Objects of Interest (TOIs) that have no follow-up observations recorded
on ExoFOP, using SPOC 120 s light curves. Code, trained model and results of the paper in `paper/`.

The pipeline runs a blind periodic search (BLS) and a single/duo-transit search on the GPU, vets
each signal with photometric, centroid, CCD common-mode and difference-image tests, scores it with
a multi-view CNN, and computes false-positive probabilities with TRICERATOPS. Results are
candidates for follow-up, not validated planets.

## Requirements

Python 3.12 (pipeline), Python 3.13 (TRICERATOPS, isolated environment), an NVIDIA GPU with CUDA,
GNU Make, `curl`, and a LaTeX distribution for reports and the paper.

## Usage

```bash
make setup                # .venv (pipeline) and .venv-fpp (TRICERATOPS); PYTHON=/path/python3
make catalog              # TOIs, CTOIs, published planets, SPOC TCEs, eclipsing binaries
make curves               # light curves of labelled TOI hosts (training set)
make train                # cross-validated CNN ensemble -> models/transit-cnn.pt
make curves OPEN=1        # light curves of open-TOI hosts
make candidates OPEN=1    # search and vet them -> reports/candidates/
make report TIC=...       # MCMC fit, figures and PDF report for one star -> reports/ctoi/
make test                 # unit tests, no network
```

Paper:

```bash
.venv/bin/python supplementary.py      # pixel test and FPP for non-candidate signals
.venv/bin/python paper/make_tables.py  # tables and numbers from the reports
cd paper && latexmk -pdf paper.tex
```

## Layout

| Path | Content |
|---|---|
| `tess.py` | Command-line interface |
| `data.py` | Catalogues and light-curve downloads |
| `lightcurve.py` | Detrending, folding and CNN views |
| `search.py`, `single.py` | Periodic and single/duo-transit searches, tiers |
| `vetting.py`, `pixels.py` | Photometric, centroid and difference-image tests |
| `neural.py` | CNN training and inference |
| `fpp.py` | TRICERATOPS runner |
| `report.py` | Candidate transit fit and report |
| `models/transit-cnn.pt` | Trained ensemble, threshold and calibration |
| `data/open.json`, `data/catalogs/exofop_toi.csv` | Open-TOI sample (ExoFOP table of 2026-11-02) |
| `reports/` | Per-star results, supplementary tests, candidate reports, CNN metrics |
| `paper/` | Manuscript, tables, figures |

Light curves and target pixel files are downloaded from MAST and are not versioned.

## Author

Isak Paulo de Andrade Ruas, NROLABS ([ORCID 0000-0002-3986-8048](https://orcid.org/0000-0002-3986-8048)).
