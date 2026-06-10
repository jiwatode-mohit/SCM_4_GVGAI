# SCM_4_GVGAI

Code and cached outputs for the ICPR 2026 paper "From Gameplay Traces to Game Mechanics: Causal Induction with Large Language Models".

This repository studies two related problems:

- Task 1: infer game mechanics from gameplay traces and compare model predictions against reference labels.
- Task 2: generate VGDL descriptions from those inferred mechanics and evaluate the generated games.

The repository already includes the result files used in the paper. In most cases, the fastest way to inspect or reproduce the reported figures and tables is to run the analysis scripts on those included outputs rather than rerunning every model.

## Overview

The codebase is organized around a simple pipeline. Gameplay observations in `observations/` are used for mechanics prediction in Task 1. Those predictions are aggregated into plots in `comparative_analysis_plots/`. Task 2 generates VGDL candidates into `VGDL_gen/`, evaluates them with `test_vgdls.py`, and summarizes the evaluation outputs in `overall_results.txt`.

If you only want the paper-level artifacts, use the included outputs and run the two aggregation scripts described below. If you want to regenerate predictions or VGDL files from scratch, the repository also supports full reruns.

## Repository Layout

- `observations/`: cached ASCII gameplay rollouts for the GVGAI games.
- `GVGAI_GYM/`: vendored GVGAI environment and canonical VGDL definitions.
- `main.py`: Task 1 runner for mechanics prediction.
- `overall_game_pred_analysis.py`: aggregates Task 1 outputs and writes plots to `comparative_analysis_plots/`.
- `vgdl_gen.py`: Task 2 runner for VGDL generation.
- `test_vgdls.py`: evaluates one generated VGDL JSON file and writes outputs to `vgdl_results/`.
- `overall_vgdl_results_analysis.py`: aggregates Task 2 evaluation outputs and refreshes `overall_results.txt`.
- `results_task_1/`, `VGDL_gen/`, `vgdl_results/`: included result files used by the paper.
- `environment.yml`: larger environment for local model reruns.
- `environment-minimal.yml`: lighter environment for reproducing plots and summary tables from included outputs.

## Setup

Run commands from the repository root:

```bash
cd SCM_4_GVGAI-main
```

For reproducing plots and summary tables from the included outputs, use the lighter environment:

```bash
conda env create -f environment-minimal.yml
conda activate llm_new_minimal
```

For local model generation and classification runs, use the larger environment instead:

```bash
conda env create -f environment.yml
conda activate llm_new
```

`environment.yml` reflects the Linux/CUDA setup used for local reruns. GPU access is strongly recommended if you plan to regenerate predictions or VGDL candidates. The lighter path is enough for the analysis scripts below.

`overall_game_pred_analysis.py` uses the sentence-transformer model `all-MiniLM-L6-v2` for semantic similarity plots. If the model is not already present in the local Hugging Face cache, it may be downloaded on first use.

## Quick Start

To regenerate the main analysis artifacts from the included outputs, run:

```bash
python overall_game_pred_analysis.py
python overall_vgdl_results_analysis.py
```

These two commands produce:

- `comparative_analysis_plots/`
- `overall_results.txt`

This is the shortest path to the figures and summary tables discussed in the paper.

## Reproducing Task 1

Task 1 evaluates model predictions of game mechanics using the cached JSON outputs in `results_task_1/`.

Run:

```bash
python overall_game_pred_analysis.py
```

Main output:

- `comparative_analysis_plots/`

Additional note:

- The script reads the JSON files already present in `results_task_1/`.
- Semantic similarity figures depend on `all-MiniLM-L6-v2`.

## Reproducing Task 2

Task 2 evaluates generated VGDL files using the cached statistics already stored in `vgdl_results/`.

Run:

```bash
python overall_vgdl_results_analysis.py
```

Main output:

- refreshed `overall_results.txt`

Additional note:

- The script discovers and aggregates committed `vgdl_results/*_new_paper_stats.txt` files.

## Full Reruns

The repository also supports end-to-end reruns. Those runs are more expensive and may not match the included outputs exactly because local inference stacks, hosted APIs, and model checkpoints can change over time.

### Task 1: Mechanics Prediction

Example:

```bash
python main.py --models nvidia/OpenReasoning-Nemotron-1.5B --emit-csv --analyze --num-iterations 5
```

Quantized example:

```bash
python main.py --models Qwen/QwQ-32B --quantization 4bit-nf4 --emit-csv --analyze --num-iterations 5
```

Generated outputs:

- `results_task_1/<model>@w_description*.json`
- `results_task_1/<model>@w_description.*.metrics.json`
- `results_task_1/<model>@w_description.*.predictions.csv`

### Task 2: VGDL Generation

Example:

```bash
python vgdl_gen.py --hf-generation-model Qwen/QwQ-32B --quantize
```

Generated outputs:

- `VGDL_gen/*.json`

### Task 2: VGDL Evaluation

Example:

```bash
python test_vgdls.py --json-file VGDL_gen/Qwen_QwQ-32B_4bit_results_.json
```

Generated outputs:

- `vgdl_results/*_full_analysis.json`
- `vgdl_results/*_preference_table.csv`
- `vgdl_results/*_new_paper_stats.txt`

Practical note:

- `test_vgdls.py` attempts to load `all-MiniLM-L6-v2` and a local `Qwen/Qwen3-8B` model at import time for similarity scoring and VGDL-to-text conversion. That step can trigger substantial downloads and memory use even if hosted judge API keys are unset.

## API Keys

`test_vgdls.py` can optionally call hosted judge models through:

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `ANTHROPIC_API_KEY`

If these variables are unset, the hosted judge calls are skipped.

## Paper Cross-Reference

The main paper artifacts map to the repository outputs as follows:

- Figure 1 comes from the clustering workflow in `clustering_results.ipynb`.
- Figure 3 maps to `comparative_analysis_plots/runtime_vs_accuracy.png`.
- Figure 5 maps to `comparative_analysis_plots/1a_model_performance_bar.png`.
- Figure 6 maps to `comparative_analysis_plots/global_confusion_matrix.png`.
- Table 1 maps to the `OVERALL WIN RATES` section in `overall_results.txt`.
- Table 2 maps to the `LEVEL ANALYSIS (SIMILARITY + WINS)` section in `overall_results.txt`.

Appendix material:

- The preliminary model-selection figure comes from the separate preliminary classification workflow described in the appendix.
- The clustering appendix figures come from `clustering_results.ipynb`.
- The appendix game-specific Task II table maps to the `GAME ANALYSIS (SIMILARITY + WINS)` section in `overall_results.txt`.

Additional Task 1 plots produced by `overall_game_pred_analysis.py` but not used in the main paper include:

- `comparative_analysis_plots/model_performance_line.png`
- `comparative_analysis_plots/stability_performance.png`
- `comparative_analysis_plots/18_description_type_similarity.png`
- `comparative_analysis_plots/19_cross_type_description_similarity.png`

## Notes

- The included outputs in `results_task_1/`, `VGDL_gen/`, and `vgdl_results/` are the reference files for the paper results in this repository.
- Full reruns are optional and may differ from the included outputs for reasons outside this codebase, including model and API drift.
