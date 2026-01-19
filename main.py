"""Command-line entry point for running GVGAI evaluation jobs.

The script loads observations for a chosen job, runs a language model over
them using helper utilities, and persistently stores the model's responses as
well as aggregate metrics.  It supports the original classification setup and
an alternative evaluation flow that rewrites the official descriptions before
querying the model.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from utils import (
    load_model_tokenizer,
    generate_response,
    unload,
    extract_game_name,
    OnlineMetrics,
    AVAILABLE_GAMES,
    is_classification_job,
    save_metrics_json,
    save_predictions_csv,
    descriptions,
    build_variant_descriptions,
    strip_description_prefix,
    log_progress,
)


def _iteration_sort_key(name: str) -> tuple[int, int | str]:
    """Sort numeric iteration names before alphanumeric ones."""
    log_progress(f"_iteration_sort_key: computing sort key for {name}")
    return (0, int(name)) if str(name).isdigit() else (1, name)


def _result_basename(model_name: str, job: str, quantization: Optional[str]) -> str:
    base = model_name.replace('/', '__') + f"@{job}"
    if quantization:
        base += f"__{quantization}"
    return base

def iter_observations(
    # job: str,
    model_name: str,
    num_iterations: int | None = 5,
    subset: str | None = None,
) -> Iterable[tuple[str, str, str]]:
    """Yield ``(iteration, game_name, observation_text)`` triples for ``job``.

    Each job stores its observations differently, so we inspect the expected
    filesystem layout and normalise what we yield.  Some jobs cache additional
    metadata (e.g. descriptions or SCMs) in JSON files – those are stitched into
    the returned observation text so the downstream prompting code can treat
    everything uniformly.  ``num_iterations`` limits how many iterations are
    processed (in sorted order) when provided.
    """
    log_progress(
        f"iter_observations: subset={subset}, model={model_name}, limit={num_iterations}"
    )
    target_subset = subset or "first_10"
    base = Path("observations") / target_subset
    log_progress(f"iter_observations: scanning observation directory {base} for subset {target_subset}")
    if not base.exists():
        raise FileNotFoundError(f"Observation subset '{target_subset}' not found at {base}")
    iteration_dirs = [
        p
        for p in sorted(base.iterdir(), key=lambda p: _iteration_sort_key(p.name))
        if p.is_dir()
    ]
    log_progress(f"iter_observations: discovered {len(iteration_dirs)} iteration folders before truncation")
    if num_iterations is not None:
        iteration_dirs = iteration_dirs[:num_iterations]
        log_progress(f"iter_observations: truncated iteration list to {len(iteration_dirs)} items")
    # Walk every iteration directory in order so the output is deterministic
    for folder in iteration_dirs:
        if not folder.is_dir():
            continue
        log_progress(f"iter_observations: processing folder {folder}")
        # Each TXT file corresponds to a single game instance within that iteration
        for txt in sorted(folder.glob("*.txt")):
            log_progress(f"iter_observations: reading observation file {txt}")
            content = txt.read_text()
            log_progress("iter_observations: stripping description prefix for w_description job")
            content = strip_description_prefix(content)
            log_progress(
                f"iter_observations: yielding iteration={folder.name}, game={txt.stem}, subset={target_subset}"
            )
            yield folder.name, txt.stem, content


def evaluate_with_variant_descriptions(
    model_name: str,
    analyze: bool = False,
    emit_csv: bool = False,
    num_iterations: int | None = 5,
    quantization: str | None = None,
) -> None:
    """Evaluate ``model_name`` while using multiple description variants.

    The standard descriptions are rewritten via the constructive/destructive
    prompting flows before the model processes the observations.  We run the
    evaluation for each variant, write the per-sample explanations, and collect
    metrics/CSVs if requested.
    """
    log_progress(
        f"evaluate_with_variant_descriptions: model={model_name}, analyze={analyze}, emit_csv={emit_csv}, limit={num_iterations}"
    )
    job = "w_description"
    subset_candidates = [
        ("first_10", Path("observations") / "first_10"),
        ("last_10", Path("observations") / "last_10"),
    ]
    subset_order: list[str] = []
    entries_by_subset: Dict[str, list[tuple[str, str, str]]] = {}

    for subset_name, subset_path in subset_candidates:
        if not subset_path.exists():
            log_progress(
                f"evaluate_with_variant_descriptions: subset '{subset_name}' missing at {subset_path}, skipping"
            )
            continue
        subset_entries = list(
            iter_observations(
                model_name,
                num_iterations=num_iterations,
                subset=subset_name,
            )
        )
        log_progress(
            f"evaluate_with_variant_descriptions: collected {len(subset_entries)} entries for subset '{subset_name}'"
        )
        if not subset_entries:
            log_progress(
                f"evaluate_with_variant_descriptions: no entries found for subset '{subset_name}', omitting from run"
            )
            continue
        entries_by_subset[subset_name] = subset_entries
        subset_order.append(subset_name)

    if not subset_order:
        raise FileNotFoundError(
            "No observation subsets found for job 'w_description'. Expected directories: observations/first_10 and/or observations/last_10."
        )

    entry_iterations = {
        iteration
        for subset_entries in entries_by_subset.values()
        for iteration, _, _ in subset_entries
    }
    observed_iterations = {
        p.name
        for subset_name in subset_order
        for p in (Path("observations") / subset_name).iterdir()
        if p.is_dir()
    }
    iterations = sorted(
        observed_iterations or entry_iterations,
        key=_iteration_sort_key,
    )
    if num_iterations is not None:
        iterations = iterations[:num_iterations]
    allowed_iterations = set(iterations)
    log_progress(f"evaluate_with_variant_descriptions: iteration order resolved: {iterations}")

    for subset_name in subset_order:
        filtered_entries = [
            entry for entry in entries_by_subset[subset_name] if entry[0] in allowed_iterations
        ]
        if len(filtered_entries) != len(entries_by_subset[subset_name]):
            log_progress(
                f"evaluate_with_variant_descriptions: filtered {len(entries_by_subset[subset_name]) - len(filtered_entries)} entries outside allowed iteration set for subset '{subset_name}'"
            )
        entries_by_subset[subset_name] = filtered_entries

    modes = ["standard", "constructive", "destructive", "vgdl"]
    result_by_subset: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {
        subset: {mode: {} for mode in modes} for subset in subset_order
    }
    metrics_by_subset: Dict[str, Dict[str, OnlineMetrics]] = {
        subset: {mode: OnlineMetrics(AVAILABLE_GAMES) for mode in modes}
        for subset in subset_order
    }
    metrics_total = {mode: OnlineMetrics(AVAILABLE_GAMES) for mode in modes}
    rows = {mode: [] for mode in modes} if emit_csv else None

    log_progress(f"evaluate_with_variant_descriptions: loading model {model_name}")
    model, tokenizer = load_model_tokenizer(model_name, quantization=quantization)
    # Every mode keeps a mapping: iteration -> {game_name -> description}
    variant_maps: Dict[str, Dict[str, Dict[str, str]]] = {
        "standard": {it: descriptions.copy() for it in iterations}
    }

    try:
        # Ask the model to rewrite the canonical descriptions for each iteration/mode
        log_progress("evaluate_with_variant_descriptions: building constructive variants")
        variant_maps["constructive"] = build_variant_descriptions(
            model,
            tokenizer,
            model_name,
            "constructive",
            iterations,
            base_descriptions=descriptions,
            num_iterations=num_iterations,
        )
        log_progress("evaluate_with_variant_descriptions: building destructive variants")
        variant_maps["destructive"] = build_variant_descriptions(
            model,
            tokenizer,
            model_name,
            "destructive",
            iterations,
            base_descriptions=descriptions,
            num_iterations=num_iterations,
        )
        log_progress("evaluate_with_variant_descriptions: building vgdl variants")
        variant_maps["vgdl"] = build_variant_descriptions(
            model,
            tokenizer,
            model_name,
            "vgdl",
            iterations,
            base_descriptions=descriptions,
            num_iterations=num_iterations,
        )

        sanitized_model = model_name.replace("/", "__")

        for subset_name in subset_order:
            entries = entries_by_subset.get(subset_name, [])
            log_progress(
                f"evaluate_with_variant_descriptions: processing {len(entries)} entries for subset '{subset_name}'"
            )
            for iteration, game, observation in entries:
                log_progress(
                    f"evaluate_with_variant_descriptions: processing subset={subset_name}, iteration={iteration}, game={game}"
                )
                for mode in modes:
                    desc_map = variant_maps[mode].get(iteration)
                    if desc_map is None:
                        log_progress(
                            f"evaluate_with_variant_descriptions: no description map for mode={mode}, iteration={iteration}"
                        )
                        continue
                    # Feed the observation together with the mode-specific descriptions
                    log_progress(
                        f"evaluate_with_variant_descriptions: invoking generate_response for mode={mode}, subset={subset_name}"
                    )
                    reply, thought, _ = generate_response(
                        model,
                        tokenizer,
                        model_name,
                        observation,
                        job,
                        description_map=desc_map,
                    )

                    print(f"[{model_name}@{job}::{mode}::{subset_name}] Iteration: {iteration}, Game: {game}")
                    print(f"Reply: {reply}")
                    print(20 * "-")
                    print(f"Thought: {thought}")
                    print(20 * "=")
                    print("\n\n\n")

                    exp_dir = (
                        Path("llm_explanations") / job / subset_name / mode / str(iteration) / sanitized_model
                    )
                    exp_dir.mkdir(parents=True, exist_ok=True)
                    # Persist the model output for manual inspection / auditing later
                    (exp_dir / f"{game}.txt").write_text(
                        f"reply: {reply}" + (f"\n\nThought: {thought}" if thought else "")
                    )

                    prediction = extract_game_name(reply)
                    # Track per-mode accuracy online to avoid re-reading outputs later
                    metrics_by_subset[subset_name][mode].record(true_label=game, prediction=prediction)
                    metrics_total[mode].record(true_label=game, prediction=prediction)

                    if iteration not in result_by_subset[subset_name][mode]:
                        result_by_subset[subset_name][mode][iteration] = {}
                    result_by_subset[subset_name][mode][iteration][game] = {
                        "reply": reply,
                        "thought": thought,
                        "prediction": prediction,
                    }

                    if rows is not None:
                        # Buffer CSV rows so we only hit the filesystem once per mode
                        rows[mode].append({
                            "subset": subset_name,
                            "iteration": iteration,
                            "game": game,
                            "model": model_name,
                            "reply": reply,
                            "thought": thought,
                            "prediction": prediction,
                            "correct": prediction == game,
                        })
    finally:
        log_progress(f"evaluate_with_variant_descriptions: unloading model {model_name}")
        unload(model)

    Path("results").mkdir(exist_ok=True)
    out_base = _result_basename(model_name, job, quantization)
    out_path = Path("results") / f"{out_base}.json"

    payload: Dict[str, object] = {
        "subsets": {
            subset: {
                mode: result_by_subset[subset][mode]
                for mode in modes
            }
            for subset in subset_order
        },
        "modes": modes,
        "iterations": iterations,
        "description_variants": {
            "constructive": variant_maps.get("constructive", {}),
            "destructive": variant_maps.get("destructive", {}),
            "vgdl": variant_maps.get("vgdl", {}),
        },
    }
    log_progress(f"evaluate_with_variant_descriptions: writing combined results to {out_path}")
    out_path.write_text(json.dumps(payload, indent=4))

    for mode in modes:
        log_progress(f"evaluate_with_variant_descriptions: summarising metrics for mode={mode}")
        for subset in subset_order:
            subset_summary = metrics_by_subset[subset][mode].summary()
            subset_overall = subset_summary.get("overall", {})
            subset_corr = int(subset_overall.get("correct", 0))
            subset_att = int(subset_overall.get("attempts", 0))
            subset_acc = float(subset_overall.get("accuracy", 0.0))
            print(
                f"[{model_name}@{job}::{mode}::{subset}] Correct/Total: {subset_corr}/{subset_att} ({subset_acc:.2%})"
            )

        total_summary = metrics_total[mode].summary()
        total_overall = total_summary.get("overall", {})
        total_corr = int(total_overall.get("correct", 0))
        total_att = int(total_overall.get("attempts", 0))
        total_acc = float(total_overall.get("accuracy", 0.0))
        print(f"[{model_name}@{job}::{mode}::total] Correct/Total: {total_corr}/{total_att} ({total_acc:.2%})")

        if analyze:
            metrics_path = Path("results") / f"{out_base}.{mode}.metrics.json"
            log_progress(f"evaluate_with_variant_descriptions: saving metrics JSON to {metrics_path}")
            metrics_payload: Dict[str, object] = {
                subset: metrics_by_subset[subset][mode].summary() for subset in subset_order
            }
            metrics_payload["total"] = total_summary
            save_metrics_json(metrics_payload, metrics_path)

        if emit_csv and rows is not None:
            csv_path = Path("results") / f"{out_base}.{mode}.predictions.csv"
            log_progress(f"evaluate_with_variant_descriptions: saving predictions CSV to {csv_path}")
            save_predictions_csv(rows[mode], csv_path)


def evaluate(
    model_name: str,
    analyze: bool = False,
    emit_csv: bool = False,
    num_iterations: int | None = None,
    quantization: str | None = None,
) -> None:
    """Run ``model_name`` over every observation for ``job``.

    The helper drives both classification jobs (where we measure accuracy) and
    generation jobs (where we only archive the text).  It ensures we only load
    the model once, records metrics for classification problems, and writes the
    structured results to ``results/<model>@<job>.json``.
    """
    log_progress(f"evaluate: model={model_name}, analyze={analyze}, emit_csv={emit_csv}, limit={num_iterations}")
    log_progress("evaluate: delegating to evaluate_with_variant_descriptions")
    evaluate_with_variant_descriptions(
        model_name,
        analyze=analyze,
        emit_csv=emit_csv,
        num_iterations=num_iterations,
        quantization=quantization,
    )
    return

###
def main() -> None:
    """Parse CLI arguments and kick off evaluation for every requested model."""
    log_progress("main: parsing CLI arguments")
    parser = argparse.ArgumentParser(description="Unified GVGAI evaluator")
    parser.add_argument("--models", nargs="+", required=True, help="HuggingFace model names")
    parser.add_argument("--analyze", action="store_true", default=True, help="Compute and save metrics for classification jobs (default: True)")
    parser.add_argument("--emit-csv", action="store_true", default=True, help="Also write a predictions CSV for classification jobs (default: True)")
    parser.add_argument(
        "--quantization",
        choices=["none", "8bit", "4bit", "4bit-nf4", "4bit-fp4"],
        default="none",
        help="Optional bitsandbytes quantization mode to load the models with",
    )
    parser.add_argument("--num-iterations", type=int, default=5, help="Limit number of iterations to process (for testing)")
    args = parser.parse_args()
    log_progress(f"main: received arguments {args}")
    quantization = None if args.quantization == "none" else args.quantization

    for model in args.models:
        log_progress(f"main: starting evaluation for model {model}")
        evaluate(
            model,
            analyze=args.analyze,
            emit_csv=args.emit_csv,
            num_iterations=args.num_iterations,
            quantization=quantization,
        )

if __name__ == "__main__":
    main()
