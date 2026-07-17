PYTHON ?= python
OUT ?= results/full_v9
NX ?= 201
NU ?= 32
NC ?= 4 8 16 32
BITS ?= 3 5 8
NSTEPS ?= 20
TOL ?= 1e-8
MAX_SOURCE_ITERS ?= 200
MEMORY_INTERVAL_MS ?= 10

.PHONY: test smoke mixed-order-line-source memory-smoke full-v9 full-jcp clean

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

smoke:
	PYTHONPATH=src $(PYTHON) scripts/reproduce_mixed_order_line_source.py \
		--preset smoke --out results/smoke --skip-existing

memory-smoke:
	PYTHONPATH=src $(PYTHON) scripts/profile_command.py \
		--output results/memory_smoke.json --interval-ms 5 -- \
		$(PYTHON) -c "import numpy as np,time; a=np.ones((4000000,),dtype=np.float64); time.sleep(0.15); print(a.sum())"

mixed-order-line-source:
	PYTHONPATH=src $(PYTHON) scripts/reproduce_mixed_order_line_source.py \
		--preset custom --out $(OUT) --nx $(NX) --uncollided-order $(NU) \
		--collided-orders $(NC) --bits $(BITS) --n-steps $(NSTEPS) \
		--tol $(TOL) --max-source-iters $(MAX_SOURCE_ITERS) \
		--memory-interval-ms $(MEMORY_INTERVAL_MS) --skip-existing

full-v9:
	PYTHONPATH=src $(PYTHON) scripts/reproduce_mixed_order_line_source.py \
		--preset full-v9 --out results/full_v9 --skip-existing

full-jcp: full-v9

clean:
	rm -rf results .pytest_cache .numba_cache **/__pycache__
