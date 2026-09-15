PYTHON ?= python3
CONFIG ?= configs/fused_index_topk_h20.json
VARIANT ?= deepgemm_flashinfer_topk_auto
CANDIDATE ?= fused_index_topk
RUN_ID ?= itk-manual
EVAL_RUN_ID ?= h20-eval-$(shell date -u +%Y%m%dT%H%M%SZ)
MODE ?= screening
RUNTIME_ROOT ?= /data/$(USER)
ARTIFACT_ROOT ?= $(RUNTIME_ROOT)/artifacts
CORRECTNESS ?= $(ARTIFACT_ROOT)/raw/$(RUN_ID)/$(VARIANT)/correctness.json
BENCHMARK ?= $(ARTIFACT_ROOT)/raw/$(RUN_ID)/$(VARIANT)/benchmark.json

.PHONY: test lint variants setup-h20 check bench evaluate nsys ncu reproduce-smoke

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

variants:
	$(PYTHON) -c 'from index_topk_perflab.registry import available_variants; print(*available_variants(), sep="\n")'

setup-h20:
	scripts/setup_h20.sh

check:
	scripts/run_h20.sh $(PYTHON) -m index_topk_perflab.cli check --config $(CONFIG) --variant $(VARIANT) --run-id $(RUN_ID)

bench:
	python3 scripts/live_progress.py --kind benchmark --label "benchmark $(VARIANT)" --config $(CONFIG) --artifact $(BENCHMARK) -- scripts/run_h20.sh $(PYTHON) -m index_topk_perflab.cli bench --config $(CONFIG) --variant $(VARIANT) --run-id $(RUN_ID) --correctness $(CORRECTNESS) --output $(BENCHMARK)

evaluate:
	scripts/evaluate_h20.sh --candidate $(CANDIDATE) --mode $(MODE) --run-id $(EVAL_RUN_ID) --config $(CONFIG) --artifact-root $(ARTIFACT_ROOT)

nsys:
	scripts/profile_nsys.sh --config $(CONFIG) --variant $(VARIANT) --target-length 16384 --run-id $(RUN_ID) --correctness $(CORRECTNESS) --output-root $(ARTIFACT_ROOT)/profiles

ncu:
	scripts/profile_ncu.sh --config $(CONFIG) --variant $(VARIANT) --target-length 16384 --stage indexer --run-id $(RUN_ID) --correctness $(CORRECTNESS) --output-root $(ARTIFACT_ROOT)/profiles

reproduce-smoke:
	ITK_RUNTIME_ROOT=$(RUNTIME_ROOT) scripts/reproduce.sh --level smoke
