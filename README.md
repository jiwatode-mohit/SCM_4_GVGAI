# SCM_4_GVGAI
This repo is part of an ICPR 2026 submission.

## Structure
- `observations/`: cached ASCII rollouts (first/last ten frames).  
- `GVGAI_GYM/`: canonical VGDL definitions.  
- `main.py`: Task 1 classifier driver (w_description only).  
- `vgdl_gen_new.py`: Task 2 VGDL generator.  
- `test_vgdls.py`: compares generated VGDLs to ground truth.  
- `results_task_1/`, `VGDL_gen/`, `vgdl_results/`: cached outputs.  
- `environment.yml`: Conda environment.  
- `pthon_call_main.txt`, `python_call_task_2.txt`: sample commands.

## Usage
1. `conda/mamba env create -f environment.yml && conda activate llm_new`.
2. Task 1 classification: `python main.py --models <hf_id> --num-iterations 5`. Outputs land in `results/<model>@w_description*.json`.
3. Task 1 plots: `python overall_game_pred_analysis.py` (writes to `comparative_analysis_plots/`).
4. Task 2 generation: `python vgdl_gen_new.py --hf-generation-model <hf_id> --quantize`.
5. Task 2 evaluation: `python test_vgdls.py` → results in `vgdl_results/`.
6. Task 2 summary: `python overall_vgdl_results_analysis.py` → refreshes `overall_results.txt`.

Set API keys (see `keys.txt`) if you want LLM judges; leave unset to skip. Generated folders are gitignored, so copy outputs elsewhere if needed.
