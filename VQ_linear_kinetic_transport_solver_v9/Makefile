PYTHON ?= python
NX ?= 201
NU ?= 32
NC ?= 8,16
BITS ?= 5,8
MAX_SOURCE_ITERS ?= 1000
TOL ?= 1e-8
OUT ?= results/v9_mixed_order

.PHONY: test mixed-order-smoke mixed-order clean

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

mixed-order-smoke:
	PYTHONPATH=src $(PYTHON) scripts/reproduce_v9.py \
		--preset smoke --out results/v9_smoke \
		--max-source-iters 20 --tol 1e-6

mixed-order:
	PYTHONPATH=src $(PYTHON) scripts/reproduce_v9.py \
		--preset paper --out $(OUT) --nx $(NX) \
		--uncollided-order $(NU) --collided-orders $(NC) \
		--bits $(BITS) --max-source-iters $(MAX_SOURCE_ITERS) \
		--tol $(TOL) --skip-existing

clean:
	rm -rf results/v9_smoke results/v9_mixed_order .pytest_cache
