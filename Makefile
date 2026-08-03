# FinSentNet-C — reproduction entry points.
#
# `make reproduce` regenerates every paper artifact from scratch. Nothing in paper/ is
# hand-edited; if it were, the provenance footnote on each table would be a lie.

PY ?= python
OUT ?= paper
RUNS ?= runs

.PHONY: help install test lint skeleton smoke reproduce clean check-clean-tree

help:
	@echo "install         install the package with dev extras"
	@echo "test            run the full test suite"
	@echo "lint            ruff check"
	@echo "skeleton        emit every table and figure on a ZERO-SIGNAL panel"
	@echo "smoke           full protocol end to end on synthetic data"
	@echo "reproduce       skeleton + smoke + tests, the whole artifact set"
	@echo "check-clean-tree  refuse to publish numbers from a dirty working tree"

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check finsent tests experiments

# Phase-1 acceptance test. The manuscript skeleton must exist before any result does.
skeleton:
	$(PY) -m finsent.eval.report --signal random --out $(OUT) --label skeleton \
		--signal-strength 0.0

# Whole protocol on synthetic data with a known planted signal: proves the pipeline
# executes and that the metrics recover what was planted.
smoke:
	$(PY) experiments/00_smoke_end_to_end.py --out $(RUNS)/smoke

# A dirty tree means the recorded git sha does not describe the code that produced the
# numbers, so the artifact is not reproducible. Fail loudly rather than stamp a lie.
check-clean-tree:
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "ERROR: working tree is dirty; commit before generating paper artifacts."; \
		git status --short; \
		exit 1; \
	fi

reproduce: test skeleton smoke
	@echo ""
	@echo "Artifacts written to $(OUT)/ and $(RUNS)/."
	@echo "These are SYNTHETIC. The real study is not run by this target."

clean:
	rm -rf $(OUT)/tables $(OUT)/figures $(RUNS) .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
