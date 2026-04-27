# =============================================================================
# CfC Continual Learning - Project Makefile
# =============================================================================
# Usage:
#   make status          - Show experiment completion status
#   make validate        - Quick 1-epoch smoke test of all backbones
#   make run-tep         - Rerun TEP experiments (bugfix applied)
#   make run-ablations   - Run missing ablation experiments (H1/H2)
#   make run-all-missing - Run all missing experiments
#   make analyze         - Download WandB data and generate tables/plots
#   make paper           - Compile the paper
#   make clean-repo      - Archive stale files from repo root
#   make help            - Show this help
# =============================================================================

SHELL := /bin/bash
WORKSPACE := $(shell pwd)
MAMMOTH := $(WORKSPACE)/mammoth
VENV := $(WORKSPACE)/.venv/bin/activate
PAPER_DIR := $(WORKSPACE)/LTC_CFC_ContinualLearning
SCRIPTS := $(WORKSPACE)/scripts
LOG_DIR := $(WORKSPACE)/paper_results/logs
MAX_PARALLEL ?= 4
SEEDS := 0 1 2

# WandB settings (loaded from .secrets.json)
WANDB_ENTITY := $(shell python3 -c "import json; print(json.load(open('.secrets.json')).get('WANDB_ENTITY','fneubuerger'))" 2>/dev/null)
WANDB_PROJECT := mammoth

# ==== HELP ====
.PHONY: help
help:
	@echo "CfC Continual Learning - Available targets:"
	@echo ""
	@echo "  Status & Validation:"
	@echo "    make status          Check WandB experiment completion"
	@echo "    make validate        Quick 1-epoch smoke test"
	@echo ""
	@echo "  Experiments:"
	@echo "    make run-tep         Rerun TEP (fixed dataset loader)"
	@echo "    make run-ablations   Run H1/H2 ablation experiments"
	@echo "    make run-all-missing Run everything that's missing"
	@echo ""
	@echo "  Analysis & Paper:"
	@echo "    make analyze         Pull WandB data, generate tables"
	@echo "    make paper           Compile LaTeX paper"
	@echo "    make paper-clean     Remove LaTeX aux files"
	@echo "    make docs            Build Sphinx documentation"
	@echo "    make test            Run the Python test suite"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make clean-repo      Archive stale root files"
	@echo "    make clean-wandb     List failed/crashed WandB runs"
	@echo ""
	@echo "  Variables:"
	@echo "    MAX_PARALLEL=N       Max concurrent experiments (default: 4)"

# ==== STATUS ====
.PHONY: status
status:
	@cd $(WORKSPACE) && source $(VENV) && python3 scripts/analysis/check_status.py

# ==== TESTS ====
.PHONY: test
test:
	@echo "=== Running Python unit tests ==="
	@cd $(WORKSPACE) && source $(VENV) && python3 -m unittest discover -s tests

# ==== DOCS ====
.PHONY: docs
docs:
	@echo "=== Building Sphinx documentation ==="
	@cd $(WORKSPACE)/docs && make html

# ==== VALIDATE ====
.PHONY: validate
validate:
	@echo "=== Quick Smoke Test (1 epoch, debug mode) ==="
	@cd $(MAMMOTH) && source $(VENV) && \
	for bb in mnistcfc mnistltc mnist-random-sparse; do \
		echo -n "Testing $$bb... "; \
		python utils/main.py --dataset seq-mnist --model sgd --backbone $$bb \
			--n_epochs 1 --lr 0.01 --batch_size 64 --seed 0 --debug_mode 1 2>&1 | \
			grep -o "Logging results.*" || echo "FAILED"; \
	done
	@cd $(MAMMOTH) && source $(VENV) && \
	for bb in cnn-cfc cnn-ltc cnn-random-sparse; do \
		echo -n "Testing $$bb... "; \
		python utils/main.py --dataset seq-cifar10 --model sgd --backbone $$bb \
			--n_epochs 1 --lr 0.03 --batch_size 32 --seed 0 --debug_mode 1 2>&1 | \
			grep -o "Logging results.*" || echo "FAILED"; \
	done
	@cd $(MAMMOTH) && source $(VENV) && \
	for bb in tepcfc tepltc tep-random-sparse; do \
		echo -n "Testing $$bb... "; \
		python utils/main.py --dataset tennessee-eastman --model sgd --backbone $$bb \
			--n_epochs 1 --lr 0.001 --batch_size 32 --seed 0 --debug_mode 1 2>&1 | \
			grep -o "Logging results.*" || echo "FAILED"; \
	done
	@echo "=== Validation Complete ==="

# ==== RUN TEP (with fixed dataset) ====
.PHONY: run-tep
run-tep:
	@echo "=== TEP Experiments (fixed dataset loader) ==="
	@echo "Clearing old TEP logs..."
	@mkdir -p $(LOG_DIR)
	@# Remove old broken TEP logs so skip detection doesn't skip them
	@rm -f $(LOG_DIR)/tep_*.log
	$(SCRIPTS)/benchmarks/run_paper_benchmarks.sh --dataset tep --max-parallel $(MAX_PARALLEL) --force

# ==== RUN ABLATIONS (H1: wiring, H2: dynamics) ====
.PHONY: run-ablations
run-ablations: run-ablations-mnist run-ablations-cifar

.PHONY: run-ablations-mnist
run-ablations-mnist:
	@echo "=== MNIST Ablation Experiments ==="
	@echo "Running RandomSparse (H1) and LTC (H2) on MNIST"
	@cd $(MAMMOTH) && source $(VENV) && \
	for seed in $(SEEDS); do \
		for bb in mnistltc mnist-random-sparse; do \
			for method_args in \
				"sgd||" \
				"joint||" \
				"er|200|--buffer_size 200" \
				"er|500|--buffer_size 500" \
				"derpp|500|--buffer_size 500 --alpha 0.1 --beta 0.5" \
				"er_ace|200|--buffer_size 200"; \
			do \
				IFS='|' read -r model buf extra <<< "$$method_args"; \
				name="mnist_$${bb}_$${model}$${buf}"; \
				logfile="$(LOG_DIR)/$${name}_seed$${seed}.log"; \
				if grep -q "wandb: Synced\|Run history:" "$$logfile" 2>/dev/null; then \
					echo "  Skip: $$name seed $$seed (done)"; \
					continue; \
				fi; \
				echo "  Run: $$name seed $$seed"; \
				python utils/main.py --dataset seq-mnist --model $$model --backbone $$bb \
					--n_epochs 1 --lr 0.01 --batch_size 32 --seed $$seed \
					--num_workers 4 \
					--wandb_entity $(WANDB_ENTITY) --wandb_project $(WANDB_PROJECT) \
					--wandb_name $${name}_seed$${seed} \
					$$extra \
					2>&1 | tee "$$logfile"; \
			done; \
		done; \
	done

.PHONY: run-ablations-cifar
run-ablations-cifar:
	@echo "=== CIFAR-10 Ablation Experiments ==="
	@echo "Running RandomSparse (H1) and LTC (H2) on CIFAR-10"
	@cd $(MAMMOTH) && source $(VENV) && \
	for seed in $(SEEDS); do \
		for bb in cnn-ltc cnn-random-sparse; do \
			for method_args in \
				"sgd||" \
				"joint||" \
				"er|200|--buffer_size 200" \
				"er|500|--buffer_size 500" \
				"er|1000|--buffer_size 1000" \
				"derpp|500|--buffer_size 500 --alpha 0.1 --beta 0.5" \
				"er_ace|500|--buffer_size 500"; \
			do \
				IFS='|' read -r model buf extra <<< "$$method_args"; \
				name="cifar_$${bb}_$${model}$${buf}"; \
				logfile="$(LOG_DIR)/$${name}_seed$${seed}.log"; \
				if grep -q "wandb: Synced\|Run history:" "$$logfile" 2>/dev/null; then \
					echo "  Skip: $$name seed $$seed (done)"; \
					continue; \
				fi; \
				echo "  Run: $$name seed $$seed"; \
				python utils/main.py --dataset seq-cifar10 --model $$model --backbone $$bb \
					--n_epochs 50 --lr 0.03 --batch_size 32 --seed $$seed \
					--num_workers 4 \
					--wandb_entity $(WANDB_ENTITY) --wandb_project $(WANDB_PROJECT) \
					--wandb_name $${name}_seed$${seed} \
					$$extra \
					2>&1 | tee "$$logfile"; \
			done; \
		done; \
	done

# ==== RUN ALL MISSING ====
.PHONY: run-all-missing
run-all-missing: run-tep run-ablations
	@echo "=== All missing experiments launched ==="

# ==== RESUME ====
.PHONY: run-resume
run-resume:
	@if [ -z "$(RESUME_EXPERIMENT)" ]; then echo "ERROR: provide RESUME_EXPERIMENT=<experiment_name>"; exit 1; fi
	@echo "=== Resuming benchmark from $(RESUME_EXPERIMENT) ==="
	@$(SCRIPTS)/benchmarks/run_paper_benchmarks.sh --dataset tep --max-parallel $(MAX_PARALLEL) --resume-experiment $(RESUME_EXPERIMENT)

$(WORKSPACE)/paper_results/wandb_analysis/all_results.csv: scripts/analysis/analyze_wandb_results.py
	@echo "=== Pulling WandB Results ==="
	@mkdir -p $(WORKSPACE)/paper_results/wandb_analysis
	@cd $(WORKSPACE) && source $(VENV) && \
	python3 scripts/analysis/analyze_wandb_results.py \
		--output-dir paper_results/wandb_analysis \
		--min-year 2025
	@echo "Results saved to paper_results/wandb_analysis/"

# ==== PAPER ====
.PHONY: paper
paper:
	@echo "=== Compiling Paper ==="
	@cd $(PAPER_DIR) && \
		pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 && \
		bibtex main > /dev/null 2>&1 && \
		pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 && \
		pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
	@echo "Paper compiled: $(PAPER_DIR)/main.pdf ($$(wc -c < $(PAPER_DIR)/main.pdf | xargs) bytes)"

.PHONY: paper-clean
paper-clean:
	@cd $(PAPER_DIR) && rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
	@echo "LaTeX aux files cleaned"

# ==== REPO CLEANUP ====
.PHONY: clean-repo
clean-repo:
	@echo "=== Archiving stale root files ==="
	@mkdir -p $(WORKSPACE)/archive/stale_root
	@# Debug/temp scripts
	@for f in debug_ewc.sh debug_hope_import.py debug_hope_run.sh \
		debug_tep_params.py debug_tep_params_v2.py test_tep_fix.py \
		check_validation_summary.sh rerun_crashed.py rerun_crashed.sh \
		rerun_crashed.log rerun_log.txt cleanup_repo.sh; do \
		[ -f "$(WORKSPACE)/$$f" ] && mv "$(WORKSPACE)/$$f" $(WORKSPACE)/archive/stale_root/ && echo "  Archived: $$f" || true; \
	done
	@# Superseded launch/monitor scripts
	@for f in LAUNCH_FULL_BENCHMARKS.sh LAUNCH_GUIDE.sh QUICK_START_BENCHMARKS.sh \
		launch_hope_benchmark.sh launch_tep_benchmark.sh \
		monitor_hope.sh monitor_tep.sh; do \
		[ -f "$(WORKSPACE)/$$f" ] && mv "$(WORKSPACE)/$$f" $(WORKSPACE)/archive/stale_root/ && echo "  Archived: $$f" || true; \
	done
	@# Stale docs (keep RESEARCH_PLAN.md, PAPER_BENCHMARKS.md, README.md)
	@for f in BUG_RESOLUTION_PLAN.md DETAILED_BENCHMARK_REPORT.md \
		IMPLEMENTATION_SUMMARY.md LAUNCH_STATUS.md NEXT_STEPS.md \
		PAPER_REPO.md PAPER_SETUP_SUMMARY.md BENCHMARK_STATUS.md; do \
		[ -f "$(WORKSPACE)/$$f" ] && mv "$(WORKSPACE)/$$f" $(WORKSPACE)/archive/stale_root/ && echo "  Archived: $$f" || true; \
	done
	@# Empty validation JSONs
	@for f in validation_ml_lightgbm.json validation_ml_xgboost.json; do \
		[ -f "$(WORKSPACE)/$$f" ] && mv "$(WORKSPACE)/$$f" $(WORKSPACE)/archive/stale_root/ && echo "  Archived: $$f" || true; \
	done
	@# Move setup scripts to scripts/
	@[ -f "$(WORKSPACE)/download_tep.sh" ] && mv "$(WORKSPACE)/download_tep.sh" $(SCRIPTS)/ && echo "  Moved: download_tep.sh -> scripts/" || true
	@[ -f "$(WORKSPACE)/setup_env.sh" ] && mv "$(WORKSPACE)/setup_env.sh" $(SCRIPTS)/ && echo "  Moved: setup_env.sh -> scripts/" || true
	@echo "=== Cleanup complete ==="

.PHONY: clean-wandb
clean-wandb:
	@echo "=== Failed/Crashed WandB Runs ==="
	@cd $(WORKSPACE) && source $(VENV) && python3 -c "import wandb,json; api=wandb.Api(); e=json.load(open('.secrets.json')).get('WANDB_ENTITY','fneubuerger'); runs=api.runs(f'{e}/mammoth',filters={'state':{'\$$in':['crashed','failed']},'created_at':{'\$$gt':'2025-01-01'}}); [print(f'{r.state:8s} {r.name:40s} {r.config.get(\"backbone\",\"?\"):20s}') for r in runs]; print(f'Total: {len(runs)}')"

.DEFAULT_GOAL := help
