.PHONY: install run benchmark build-wine run-wine setup-wine freeze-deps

PYTHON  = .venv/bin/python
PIP     = .venv/bin/pip

# ── Entorno ────────────────────────────────────────────────
install:
	python -m venv .venv
	$(PIP) install -r requirements-dev.txt

# ── Desarrollo ─────────────────────────────────────────────
run:
	$(PYTHON) main.py

benchmark:
	$(PYTHON) scripts/benchmark.py

# ── Build Windows (Wine) ───────────────────────────────────
setup-wine:
	bash scripts/setup_wine.sh

build-wine:
	bash scripts/build_wine.sh

run-wine:
	bash scripts/run_wine.sh

# ── Mantenimiento ──────────────────────────────────────────
freeze-deps:
	$(PIP) freeze > requirements.lock
