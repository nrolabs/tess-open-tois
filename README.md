# TESS — busca e triagem de candidatos a exoplanetas

Pipeline completo sobre dados públicos do TESS:

1. **Coleta** do catálogo TOI e de todas as curvas SPOC de 120 s das estrelas rotuladas.
2. **Busca cega** de trânsitos periódicos (BLS) em cada estrela.
3. **CNN multivista** (ensemble de 5 modelos) que pontua cada sinal.
4. **Testes de falso positivo** locais: SNR, número de trânsitos, ímpar/par, eclipse
   secundário, forma, tamanho físico, densidade estelar, localização da fonte pelo
   centroide e contaminação.
5. **Probabilidade de falso positivo** (FPP/NFPP) com TRICERATOPS, usando estrelas vizinhas
   do TIC/Gaia e as aberturas SPOC.
6. **Ranking** em `reports/candidates.csv`, separando sinais que correspondem a TOIs dos que estão
   fora do TOI. "Fora do TOI" ainda não exclui TCEs do SPOC nem binárias eclipsantes conhecidas.

O resultado final é um **candidato com evidência estatística**, não um planeta confirmado.
Confirmação exige observações que um computador não substitui (imagem de alta resolução,
espectroscopia, velocidade radial). Candidatos fortes podem ser enviados à comunidade pelo
[ExoFOP-TESS](https://exofop.ipac.caltech.edu/tess/).

## Comandos

Requisitos: Python 3.12 com `venv`/`pip` (ambiente principal), Python 3.13 (ambiente do
TRICERATOPS), `curl`, GNU Make, GPU NVIDIA com CUDA e internet.

```bash
make setup PYTHON=/path/to/python3
make catalog
make curves
make test
make train
make candidates
```

| Comando | O que faz |
|---|---|
| `make setup` | Cria `.venv` (CNN, PyTorch CUDA) e `.venv-fpp` (TRICERATOPS) e testa a GPU |
| `make catalog` | Baixa o catálogo TOI completo e registra origem/hash |
| `make curves` | Baixa/retoma todas as curvas SPOC 120 s das estrelas CP/KP/FP/FA |
| `make curves FIELD=100` | Baixa 100 estrelas **sem TOI** com ≥30 setores (amostra fixa, todos os setores) para buscar planetas novos |
| `make train` | Validação cruzada por estrela, ensemble final, calibração e injeção-recuperação |
| `make candidates` | Busca em todas as estrelas baixadas (BLS na GPU); `TIC=...` para uma estrela |
| `make test` | Testes locais, sem rede |

Avaliar um evento específico (período em dias, época em BJD TDB, duração em horas):

```bash
make candidates TIC=1528696 PERIOD=0.8820624 EPOCH=2458438.402842 DURATION=0.5759672
```

Neste computador o Python do sistema não tem pip, por isso o `PYTHON=` acima. O
TRICERATOPS 1.1.0 exige `numpy<2` no Python 3.12 e por isso roda isolado em `.venv-fpp`
(Python 3.13, `FPP_PYTHON=` muda o interpretador); o pipeline o chama por subprocesso.

## Coleta

- Catálogo: [NASA Exoplanet Archive / TAP](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html), consulta `select * from toi`.
- [Definições TOI](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html): `CP`/`KP` → classe 1;
  `FP`/`FA` → classe 0. `PC`, `APC` e desconhecidos ficam fora do treino.
- Curvas: os [manifestos por setor da MAST](https://archive.stsci.edu/tess/bulk_downloads.html)
  (`tesscurl_sector_N_lc.sh`) listam todos os arquivos SPOC 120 s. Filtrar esses manifestos pelos
  TICs do catálogo dá a lista exata, sem busca estrela por estrela. Os arquivos vêm da cópia pública
  na AWS (`stpubdata`, Open Data), com a MAST como reserva. O TIC sai do nome do arquivo:
  correspondência exata, sem resolver nomes de estrelas vizinhas.
- `--include-candidates` (em `python tess.py download`) baixa também PC/APC, que servem para
  busca mas não entram no treino.
- Interrompa com Ctrl+C e repita `make curves` para retomar. FITS truncados por downloads
  interrompidos são detectados e baixados de novo.

Em 2026-11 eram 2.657 TICs rotulados elegíveis, dos quais 2.357 têm curva SPOC 120 s:
15.834 arquivos, ~29 GB de FITS e ~2,5 GB de `.npz`.

## Modelo

- Entrada: vista global (ciclo inteiro), local (±3 durações), ímpar e par lado a lado, e
  secundária (fase 0,5), cada uma com canal de cobertura, mais 14 escalares medidos (profundidade,
  SNR, ímpar/par, secundário, forma, duração/período, trânsitos, centroide, contaminação, raio,
  log g, Teff, magnitude e densidade). IDs, disposição e parâmetros físicos do catálogo não entram;
  período/época/duração do catálogo só orientam o alinhamento.
- Cada TOI vira um exemplo com **todos os setores juntos**. No treino, cada setor isolado também
  vira exemplo (aumento de dados); validação e teste usam só o exemplo completo.
- Outros planetas da mesma estrela são mascarados ao medir um sinal.
- A efeméride do catálogo é refinada dentro de 3σ das suas incertezas, e setores em que ela não
  localiza o trânsito (3σ > 1 duração) são descartados.
- CNN 1D com blocos residuais, um ramo por vista. Treino exclusivamente CUDA, determinístico, AMP.
- **Validação cruzada aninhada por TIC**: 5 folds; cada estrela é testada exatamente uma vez por um
  modelo que nunca a viu; a validação de cada fold sai das estrelas restantes. O modelo final é o
  ensemble dos 5. O limiar (precisão-alvo `PRECISION`) e a calibração isotônica do ensemble vêm das
  predições fora do fold. Se nenhum limiar atinge a precisão com 5 seleções, `threshold: null`.
- O relatório registra métricas por fold, agregadas, por estrela, por faixa de profundidade,
  período, SNR e número de setores, a taxa de reprovação de cada teste de vetting por classe e a
  injeção-recuperação.

## Testes de falso positivo (`vetting.py`)

| Teste | Reprova (`fail`) | Avisa (`warn`) |
|---|---|---|
| SNR (caixa de 80% da duração, ruído correlacionado e dispersão entre eventos) | < 7,1 | |
| Trânsitos com dados | < 2 | |
| Ímpar/par (≥2 eventos em cada grupo) | > 4σ | > 3σ |
| Eclipse secundário em qualquer fase, limiar de Bonferroni, exigindo superar o maior abrilhantamento e albedo > 1 | > 5σ | acima do limiar |
| Raio do companheiro (profundidade × raio estelar) | > 25 R⊕ | |
| Fonte do sinal pelo centroide (posição da fonte vs. alvo, via WCS) | > 4σ e > 21″ | > 3σ e > 10″ |
| Forma em V (trapézio) | | ingresso > 40% |
| Duração longa demais para a densidade estelar | | razão < 0,2 |
| Contaminação (CROWDSAP) | | < 0,8 |

Esses limiares serão recalibrados com o conjunto completo, usando `vetting_by_class` em
`reports/training.json`. Na amostra inicial (116 TOIs), 16% dos CP/KP e 36% dos FP/FA foram
reprovados. Parte das reprovações de planetas é SNR baixo em estrelas ainda com poucos setores
baixados. Muitos FPs do catálogo só foram identificados com observações de solo, e o vetting
não os vê. Eles ficam para a CNN e o NFPP.

## Classes finais (`make candidates`)

| Classe | Critério |
|---|---|
| `forte` | CNN acima do limiar, vetting aprovado, FPP < 1% e NFPP < 0,1% (versão mais rígida do critério de [Giacalone et al. 2021](https://ui.adsabs.harvard.edu/abs/2021AJ....161...24G), que usa FPP < 1,5%) |
| `candidato: provável planeta` | CNN acima do limiar, vetting aprovado, FPP < 50% e NFPP < 0,1% |
| `candidato: sem FPP` | CNN e vetting aprovados, TRICERATOPS não calculado |
| `inconclusivo` | CNN abaixo do limiar ou FPP alto |
| `rejeitado` | vetting reprovado, ou NFPP > 10% (sinal provavelmente de estrela vizinha) |

Mesmo `forte` é validação estatística **sem** imagem de alta resolução: sem curva de contraste, o
TRICERATOPS não descarta companheiras muito próximas. Trate como prioridade para acompanhamento.

A busca BLS roda na GPU (PyTorch, float64): para cada lote de períodos, a curva é dobrada em caixas de fase e todas as janelas e durações são avaliadas por somas acumuladas; numa estrela com 8 anos de baseline (473 mil períodos) leva ~40 s contra ~5 min na CPU. Usa SNR com ruído robusto e SDE robusto por faixa de período. Em injeções num
setor de estrelas hospedeiras de planetas, ela recuperou 87% dos trânsitos acima de 3.000 ppm e
~47% entre 1.000 e 3.000 ppm, com detecção espúria em 4% das curvas sem sinal injetado. O
relatório de treino mede a recuperação do pipeline completo (BLS, vetting e CNN).

## Arquivos gerados

| Caminho | Conteúdo |
|---|---|
| `data/catalogs/toi.json`, `.csv`, `snapshot.json`, `snapshots/` | Catálogo, origem, hash e versões |
| `data/manifests/sector-N.sh` | Manifestos oficiais por setor (cache) |
| `data/mast/lc/*.fits` | FITS originais, preservados |
| `data/curves/<TIC>/sector-N.npz` | Tempo BJD TDB, fluxo PDCSAP, erro e centroides MOM_CENTR |
| `data/curves/<TIC>/sector-N.json` | Produto, CROWDSAP, parâmetros estelares do TIC, posição do alvo no CCD |
| `data/features/<TIC>.pkl` | Cache das features e métricas por estrela |
| `data/fpp/` | Entradas e saídas do TRICERATOPS |
| `data/download-log.json` | Erros do último download |
| `models/transit-cnn.pt` | Ensemble, limiar e calibração |
| `reports/training.json` | Métricas, folds, subgrupos, vetting por classe, injeção-recuperação |
| `reports/skipped.json` | Exemplos excluídos e motivo |
| `reports/candidates/<TIC>.json` | Sinais de cada estrela com métricas, CNN, FPP e classe |
| `reports/candidates.csv` | Ranking de todos os sinais |

## Limites

- Rótulos TOI (CP/KP vs FP/FA) não representam a mistura de sinais de uma busca cega: a calibração
  da CNN vale para essa população, não para "qualquer queda de brilho".
- A busca só cobre estrelas baixadas. Há 585.858 estrelas com curva 120 s sem TOI (~3 TB); `FIELD=N`
  baixa uma amostra fixa das 4.662 com ≥30 setores (nenhuma chega a 50), com todos os setores. O SPOC já buscou sinais nessas
  estrelas: o espaço de descoberta está em sinais fracos que só aparecem juntando setores.
- Só curvas SPOC 120 s; FFIs (QLP, TESS-SPOC) e dados de pixels completos não entram.
- A injeção-recuperação usa trânsitos trapezoidais sem escurecimento de borda, um setor por vez.
- Sistemas com variação de tempo de trânsito (TTV) forte podem falhar nos testes de efeméride.
- O projeto não envia nada automaticamente a ExoFOP, Zooniverse ou qualquer serviço.

## Próximas etapas (anotado, não implementado)

Proposta de evolução discutida em 2026-11-01. São prioridades, não melhorias já demonstradas.

| Prioridade | Recurso | Benefício esperado | Observação |
|---|---|---|---|
| P0 | Imagens de diferença e análise de pixels | Identificar a verdadeira fonte do trânsito | Exige TPFs (~10× as curvas): baixar só para candidatos |
| P0 | Treinamento com sinais de busca cega | Reduzir falsos positivos fora da população TOI | População: TCEs do SPOC ([bulk TCE](https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_tce.html)) + [catálogo de binárias eclipsantes TESS](https://cdsarc.cds.unistra.fr/ftp/J/ApJS/258/16) (Prša et al. 2022) |
| P0 | Benchmark contra ExoMiner++ | Comparação mensurável | Scores publicados de 147.568 TCEs S1–67 ([Zenodo](https://zenodo.org/record/15466292), [artigo](https://arxiv.org/abs/2502.09790)); mesmos TCEs e rótulos; superioridade não é garantida |
| P0 | Cruzar sinais com TCEs e binárias conhecidas | Rótulo "NOVO" confiável | Hoje "NOVO" só significa "não está no TOI" |
| P1 | TLS + BLS + busca de trânsitos únicos | Sinais que a BLS perde | Trânsito único: planetas de período longo |
| P1 | Dados FFI e curvas de 20 s | Mais estrelas e sinais | Volume muito maior; limitado por banda e disco |
| P1 | Modelo temporal com atenção | Dependências entre eventos e setores | Avaliar contra o CNN atual no benchmark |
| P1 | Tratamento de TTV | Planetas não estritamente periódicos | |
| P1 | Modelo de ruído correlacionado | Sensibilidade a sinais fracos | |
| P2 | Ensemble heterogêneo | Combinar erros distintos | |
| P2 | Incerteza e detecção fora de distribuição | Evitar confiança excessiva | |
