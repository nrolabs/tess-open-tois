PYTHON ?= python3
FPP_PYTHON ?= python3.13
VENV := .venv
PY := $(VENV)/bin/python
EPOCHS ?= 100
BATCH_SIZE ?= 64
PRECISION ?= 0.95
TORCH_INDEX := https://download.pytorch.org/whl/cu128

.PHONY: help setup catalog curves train candidates test
help:
	@echo 'make setup       Instalar ambientes (CNN + TRICERATOPS) e testar a GPU (PYTHON=/caminho/python3)'
	@echo 'make catalog     Baixar catálogo TOI completo e registrar origem/hash'
	@echo 'make curves      Baixar/retomar todas as curvas SPOC 120 s das estrelas rotuladas'
	@echo '                 FIELD=N baixa N estrelas sem TOI com >=30 setores (todos) para buscar planetas novos'
	@echo 'make train       Validação cruzada por estrela, ensemble final e injeção-recuperação na GPU'
	@echo 'make candidates  Buscar sinais (BLS), classificar, testar falsos positivos e calcular FPP'
	@echo '                 TIC=... uma estrela; FIELD=1 só as estrelas sem TOI; PERIOD= EPOCH= DURATION= avalia um evento'
	@echo 'make test        Testes locais, sem rede'

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
	$(PY) tess.py download $(if $(FIELD),--field $(FIELD))

train:
	$(PY) tess.py train --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --precision $(PRECISION)

candidates:
	$(PY) tess.py candidates $(if $(TIC),--tic $(TIC)) $(if $(FIELD),--field) $(if $(PERIOD),--period $(PERIOD) --epoch $(EPOCH) --duration $(DURATION))

test:
	$(PY) -m unittest discover -s tests -v
