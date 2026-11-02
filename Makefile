PYTHON ?= python3
FPP_PYTHON ?= python3.13
VENV := .venv
PY := $(VENV)/bin/python
EPOCHS ?= 100
BATCH_SIZE ?= 64
PRECISION ?= 0.95
TORCH_INDEX := https://download.pytorch.org/whl/cu128

.PHONY: help setup catalog curves train candidates report test
help:
	@echo 'make setup       Install environments (CNN + TRICERATOPS) and test the GPU (PYTHON=/path/python3)'
	@echo 'make catalog     Download TOIs and known signals (CTOIs, planets, TCEs, binaries)'
	@echo 'make curves      Download/resume all SPOC 120 s light curves of labeled stars'
	@echo '                 FIELD=N downloads N non-TOI stars with >=30 sectors (all) to search for new planets'
	@echo '                 OPEN=1 downloads the stars of open TOIs (PC, no follow-up on ExoFOP), all sectors'
	@echo 'make train       Cross-validation by star, final ensemble and injection-recovery on the GPU'
	@echo 'make candidates  Search signals (BLS), classify, run false positive tests and compute FPP'
	@echo '                 TIC=... one star; FIELD=1 non-TOI stars only; OPEN=1 open TOIs only;'
	@echo '                 PERIOD= EPOCH= DURATION= vets one event'
	@echo 'make report      CTOI package for one event: MCMC fit, figures, LaTeX/PDF, ExoFOP row (TIC=...)'
	@echo '                 Best event of reports/candidates/<TIC>.json, or PERIOD= EPOCH= DURATION='
	@echo 'make test        Local tests, no network'

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install 'torch==2.7.1' --index-url $(TORCH_INDEX)
	$(FPP_PYTHON) -m venv .venv-fpp
	.venv-fpp/bin/python -m pip install 'triceratops==1.1.0'
	$(PY) neural.py

catalog:
	$(PY) tess.py catalog

curves:
	$(PY) tess.py download $(if $(FIELD),--field $(FIELD)) $(if $(OPEN),--open)

train:
	$(PY) tess.py train --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --precision $(PRECISION)

candidates:
	$(PY) tess.py candidates $(if $(TIC),--tic $(TIC)) $(if $(FIELD),--field) $(if $(OPEN),--open) $(if $(PERIOD),--period $(PERIOD) --epoch $(EPOCH) --duration $(DURATION))

report:
	$(PY) tess.py report --tic $(TIC) $(if $(PERIOD),--period $(PERIOD) --epoch $(EPOCH) --duration $(DURATION))

test:
	$(PY) -m unittest discover -s tests -v
