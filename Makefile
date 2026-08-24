# FinSentNet-C — reproduction entry points.
#
# Two distinct things live here and they were previously confused. `make verify` runs
# the pipeline on synthetic data with a known planted signal, which proves the machinery
# executes and recovers what was planted. `make reproduce` runs the real study and
# regenerates every table and figure in paper/ from it.
#
# Nothing in paper/tables or paper/figures is hand-edited. If it were, the provenance
# footnote each one carries would be a lie.

PY ?= python
OUT ?= paper
RUNS ?= runs
SEEDS ?= 0 1 2
PANEL ?= data/cache/study_panel.parquet

.PHONY: help install test lint skeleton smoke verify dataset study tables paper \
        reproduce clean check-clean-tree

help:
	@echo "install           install the package with dev extras"
	@echo "test              run the full test suite"
	@echo "lint              ruff check"
	@echo ""
	@echo "  -- synthetic acceptance, no market data required --"
	@echo "skeleton          emit every table and figure on a ZERO-SIGNAL panel"
	@echo "smoke             full protocol end to end on synthetic data"
	@echo "verify            tests + skeleton + smoke"
	@echo ""
	@echo "  -- the real study --"
	@echo "dataset           build the point-in-time panel from the price cache"
	@echo "study             walk-forward every model over that panel (hours)"
	@echo "tables            emit paper/tables and paper/figures from the run"
	@echo "paper             dataset + study + tables"
	@echo "bundle            self-contained folder + zip for Overleaf"
	@echo "reproduce         clean-tree check + tests + paper"
	@echo ""
	@echo "check-clean-tree  refuse to publish numbers from a dirty working tree"

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check finsent tests experiments

# --------------------------------------------------------------- synthetic acceptance
# Phase-1 acceptance test. The manuscript skeleton must exist before any result does.
skeleton:
	$(PY) -m finsent.eval.report --signal random --out $(OUT) --label skeleton \
		--signal-strength 0.0

# Whole protocol on synthetic data with a known planted signal: proves the pipeline
# executes and that the metrics recover what was planted.
smoke:
	$(PY) experiments/00_smoke_end_to_end.py --out $(RUNS)/smoke

verify: test skeleton smoke
	@echo ""
	@echo "Synthetic artifacts written to $(OUT)/ and $(RUNS)/."
	@echo "These prove the machinery works. They are not results about markets."

# ---------------------------------------------------------------------- the real study
dataset:
	$(PY) experiments/01_build_dataset.py

# Hours, not minutes. --skip-existing leaves finished models alone, so an interrupted
# run resumes instead of discarding the baselines to redo the network.
study:
	$(PY) -u experiments/02_run_study.py --models all --seeds $(SEEDS) --skip-existing

tables:
	$(PY) experiments/03_make_paper.py --study-panel $(PANEL)
	$(PY) $(OUT)/check_tex.py $(OUT)/main.tex

# Everything Overleaf needs in one folder: the manuscript, the generated tables it
# inputs, and the figures it includes. Only *_main artifacts are copied -- the skeleton
# tables are synthetic and must never travel next to real results.
bundle: check-clean-tree tables
	rm -rf $(OUT)/overleaf_bundle $(OUT)/finsentnet_overleaf.zip
	mkdir -p $(OUT)/overleaf_bundle/tables $(OUT)/overleaf_bundle/figures
	cp $(OUT)/main.tex $(OUT)/overleaf_bundle/
	cp $(OUT)/tables/*_main.tex $(OUT)/tables/keynumbers.tex $(OUT)/overleaf_bundle/tables/
	cp $(OUT)/figures/F5_main.pdf $(OUT)/figures/F6_main.pdf $(OUT)/figures/F7_main.pdf $(OUT)/figures/F9_main.pdf $(OUT)/overleaf_bundle/figures/
	$(PY) -c "import shutil; shutil.make_archive('$(OUT)/finsentnet_overleaf','zip','$(OUT)/overleaf_bundle')"
	@echo "Upload $(OUT)/finsentnet_overleaf.zip to Overleaf; main.tex is the root."

paper: dataset study tables
	@echo ""
	@echo "Tables and figures written to $(OUT)/tables and $(OUT)/figures."
	@echo "main.tex already has \resultsreadytrue; run make bundle to package for Overleaf."

# A dirty tree means the recorded git sha does not describe the code that produced the
# numbers, so the artifact is not reproducible. Fail loudly rather than stamp a lie.
check-clean-tree:
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "ERROR: working tree is dirty; commit before generating paper artifacts."; \
		git status --short; \
		exit 1; \
	fi

reproduce: check-clean-tree test paper

clean:
	rm -rf $(OUT)/tables $(OUT)/figures $(RUNS) .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
