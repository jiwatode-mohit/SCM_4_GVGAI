import json
import os
import re
import shutil
import warnings
import time
import torch
import csv
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings("ignore")

# --- GLOBAL MODEL LOADING ---
# 1. Sentence Transformer (for Vector Embeddings)
embedding_model = None
try:
    from sentence_transformers import SentenceTransformer, util
    print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    print("CRITICAL ERROR: 'sentence_transformers' library not found.")
    print("Please install: pip install sentence-transformers")

# 2. Qwen Model (for VGDL -> Natural Language AND LLM Comparator)
TEXT_GEN_MODEL_ID = "Qwen/Qwen3-8B" 
text_gen_model = None
text_gen_tokenizer = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading Text Generation model ({TEXT_GEN_MODEL_ID})... This may take a while.")
    try:
        text_gen_tokenizer = AutoTokenizer.from_pretrained(TEXT_GEN_MODEL_ID)
        text_gen_model = AutoModelForCausalLM.from_pretrained(
            TEXT_GEN_MODEL_ID, 
            torch_dtype="auto", 
            device_map="auto",
        )
        print("Text Generation model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load Qwen model. Semantic Text Similarity will use raw VGDL as fallback.")
        print(f"Error details: {e}")

except ImportError:
    print("Warning: 'transformers' or 'torch' not found.")

# --- API CLIENT SETUP ---
clients = {}

# OpenAI
try:
    from openai import OpenAI
    if "OPENAI_API_KEY" in os.environ:
        clients['openai'] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        print("OPENAI_API_KEY not found in environment. GPT evaluation will be skipped.")
except ImportError:
    print("OpenAI library not found. GPT evaluation will be skipped.")

# Google Gemini
try:
    from google import genai
    if "GOOGLE_API_KEY" in os.environ:
        # genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        clients['gemini'] = True
    else:
        print("GOOGLE_API_KEY not found in environment. Gemini evaluation will be skipped.")
except ImportError:
    print("Google Generative AI library not found. Gemini evaluation will be skipped.")

# Anthropic Claude
try:
    import anthropic
    if "ANTHROPIC_API_KEY" in os.environ:
        clients['anthropic'] = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    else:
        print("Claude_API_KEY not found in environment. Claude evaluation will be skipped.")
except ImportError:
    print("Anthropic library not found. Claude evaluation will be skipped.")


# --- UTILITY FUNCTIONS ---

def load_json_data(filepath):
    """Loads the generated results JSON."""
    if not os.path.exists(filepath):
        print(f"Error: JSON file '{filepath}' not found.")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def get_generated_vgdl_direct(data, game, level, iteration, variant, max_length=5000):
    """Retrieves the VGDL string from the JSON structure safely and truncates it."""
    try:
        path = data.get('results', {}).get(variant, {})
        path = path.get(level, {}).get('first_10', {})
        path = path.get(iteration, {}).get(game, {})
        vgdl = path.get('vgdl', None)

        if isinstance(vgdl, str) and len(vgdl) > max_length:
            return vgdl[:max_length]

        return vgdl
    except Exception:
        return None


def get_ground_truth_vgdl(game, base_gym_path):
    """Retrieves the ground truth VGDL from the GVGAI file structure."""
    file_path = os.path.join(base_gym_path, "gym_gvgai", "envs", "games", f"{game}_v0", f"{game}.txt")
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"[Error reading GT file for {game}: {str(e)}]")
        return None

# --- GENERATION FUNCTIONS ---

def vgdl_to_natural_language(vgdl_code):
    """Uses Qwen to convert VGDL code into a natural language description."""
    if not text_gen_model or not text_gen_tokenizer:
        return vgdl_code 

    prompt_text = f"""
    Analyze the following VGDL (Video Game Description Language) code. 
    Provide a concise natural language description of the game mechanics, rules, and interactions defined in this code.
    Focus on what the sprites do and how the game is won or lost.

    VGDL Code:
    ```
    {vgdl_code}
    ```

    Natural Language Description:
    """
    messages = [
        {"role": "system", "content": "You are an expert game engine developer parsing VGDL code."},
        {"role": "user", "content": prompt_text}
    ]
    
    try:
        text = text_gen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = text_gen_tokenizer([text], return_tensors="pt").to(text_gen_model.device)
        generated_ids = text_gen_model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = text_gen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    except Exception as e:
        print(f"Error generating description: {e}")
        return vgdl_code

# --- LLM API CALLS ---

def call_gpt(prompt):
    if 'openai' not in clients: return "Skipped: Client/Key missing"
    try:
        print("  > Calling GPT-5.2...")
        response = clients['openai'].chat.completions.create(
            model="gpt-5.2-2025-12-11",
            messages=[{"role": "user", "content": prompt}]
        )
        print("  > GPT-5.2 response received.")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling GPT: {e}")
        return f"Error: {str(e)}"

def call_gemini(prompt):
    if 'gemini' not in clients: return "Skipped: Client/Key missing"
    try:
        from google import genai
        print("  > Calling Gemini 2.5 Pro...")
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-pro", contents=prompt
        )
        print("  > Gemini response received.")
        return response.text
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return f"Error: {str(e)}"

def call_claude(prompt):
    if 'anthropic' not in clients: return "Skipped: Client/Key missing"
    try:
        print("  > Calling Claude Opus 4.5...")
        response = clients['anthropic'].messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        print("  > Claude response received.")
        return response.content[0].text
    except Exception as e:
        print(f"Error calling Claude: {e}")
        return f"Error: {str(e)}"

def call_qwen(prompt):
    """Uses the locally loaded Qwen model for comparison."""
    if not text_gen_model or not text_gen_tokenizer:
        return "Skipped: Qwen model not loaded locally"
    
    try:
        from utils import _separate_thinking
        print("  > Calling Qwen 3 8B (Local)...")
        messages = [{"role": "user", "content": prompt}]
        text = text_gen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = text_gen_tokenizer([text], return_tensors="pt").to(text_gen_model.device)
        generated_ids = text_gen_model.generate(**model_inputs, max_new_tokens=7000, do_sample=False)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = text_gen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response, thought = _separate_thinking(response)
        print("  > Qwen response received.")
        return response
    except Exception as e:
        print(f"Error calling Qwen: {e}")
        raise e

def run_llm_comparison(prompt):
    """Orchestrates calls to all 4 judges."""
    return {
        "gpt-5.2-2025-12-11": call_gpt(prompt),
        "gemini-2.5-pro": call_gemini(prompt),
        "claude-opus-4-5-20251101": call_claude(prompt),
        "qwen-3-8b": call_qwen(prompt)
    }

def extract_winner(response_text):
    """
    Parses the LLM response to find the {ABC}, {XYZ}, or {TIE} token.
    """
    if not response_text or response_text.startswith("Skipped") or response_text.startswith("Error"):
        return "N/A"
    
    # Check for strict format at start of string (allowing for potential whitespace)
    # Looking for {ABC}, {XYZ}, or {TIE} case-insensitive
    match = re.search(r"^\s*\{(ABC|XYZ|TIE)\}", response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    # Fallback: Check the first 100 characters in case the model chattered a bit first
    match = re.search(r"\{(ABC|XYZ|TIE)\}", response_text[:100], re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return "PARSE_ERROR"

# --- METRIC FUNCTIONS ---

def get_cosine_similarity(text1, text2):
    """Calculates cosine similarity between two texts using Sentence Transformers."""
    if not text1 or not text2 or embedding_model is None:
        return 0.0
    try:
        embeddings = embedding_model.encode([text1, text2])
        sim = util.cos_sim(embeddings[0], embeddings[1])
        return float(sim.item())
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def check_parsability(vgdl_content, game_name):
    """Checks if VGDL is parsable."""
    if not vgdl_content:
        return False, "Empty Content"
    if "SpriteSet" not in vgdl_content or "LevelMapping" not in vgdl_content:
        return False, "Missing Core Blocks (SpriteSet/LevelMapping)"
    try:
        import gym_gvgai
        return True, "Passed Basic Syntax Check"
    except ImportError:
        return True, "Parsability Skipped (gym_gvgai not installed) - Basic Syntax OK"
    except Exception as e:
        return False, str(e)

def create_comparison_prompt_text(non_scm_vgdl, scm_vgdl, gt_vgdl):
    """Generates the prompt string for the LLMs."""
    def format_content(title, content):
        if not content: return f"=== {title} ===\n[No Content]\n\n"
        clean_content = content.replace("\\n", "\n").replace("\\t", "\t").strip()
        return f"{'='*80}\n  {title}\n{'='*80}\n{clean_content}\n\n"

    prompt_text = (
        "Compare the generated VGDL (ABC) and generated VGDL (XYZ) against the ground truth VGDL.\n"
        "Determine which generated stream better preserves the semantic and logical integrity of the original game mechanics.\n"
        "STRICT OUTPUT FORMAT: You must start your response with either {ABC} or {XYZ} to indicate the winner. "
        "If you cannot decide or they are equal, use {TIE}. Follow this tag with your reasoning.\n\n"
    )
    
    prompt_text += format_content("GENERATED VGDL (ABC / Non-SCM)", non_scm_vgdl)
    prompt_text += format_content("GENERATED VGDL (XYZ / SCM)", scm_vgdl)
    prompt_text += format_content("GROUND TRUTH VGDL", gt_vgdl)

    return prompt_text

def analyze_variant(variant_name, generated_vgdl, gt_vgdl, gt_desc, game_name):
    """Runs individual metrics for a specific variant."""
    if not generated_vgdl:
        return {"error": "No generated VGDL found"}

    # 1. VGDL Cosine Similarity
    code_sim = get_cosine_similarity(generated_vgdl, gt_vgdl)
    
    # 2. Semantic Text Similarity
    gen_desc = vgdl_to_natural_language(generated_vgdl)
    semantic_sim = get_cosine_similarity(gen_desc, gt_desc)

    return {
        "variant": variant_name,
        "metrics": {
            "vgdl_cosine_similarity": round(code_sim, 4),
            "semantic_text_similarity": round(semantic_sim, 4),
        }
    }

def save_paper_statistics(full_analysis, output_path):
    """
    Aggregates metrics and win rates for paper reporting.
    Includes breakdowns by Context Level and by Game.
    """
    
    # --- Helper to create a fresh statistics container ---
    def init_stat_block():
        return {
            "non_scm": {"code_sim": [], "sem_sim": []},
            "scm": {"code_sim": [], "sem_sim": []},
            "wins": {
                "gpt-5.2-2025-12-11": {"ABC": 0, "XYZ": 0, "TIE": 0, "total": 0},
                "gemini-2.5-pro": {"ABC": 0, "XYZ": 0, "TIE": 0, "total": 0},
                "claude-opus-4-5-20251101": {"ABC": 0, "XYZ": 0, "TIE": 0, "total": 0},
                "qwen-3-8b": {"ABC": 0, "XYZ": 0, "TIE": 0, "total": 0}
            }
        }

    # --- Data Structures ---
    overall_stats = init_stat_block()
    level_stats = defaultdict(init_stat_block)
    game_stats = defaultdict(init_stat_block)
    
    missing_counts = {"non_scm": 0, "scm": 0, "total_entries": 0}

    # --- Aggregation Loop ---
    for entry in full_analysis:
        missing_counts["total_entries"] += 1
        level_key = entry['meta']['level']
        game_key = entry['meta']['game']
        
        # References to the 3 distinct blocks we need to update
        blocks_to_update = [
            overall_stats,
            level_stats[level_key],
            game_stats[game_key]
        ]

        # 1. Aggregate Non-SCM Metrics
        if "metrics" in entry.get("non_scm_analysis", {}):
            val_code = entry["non_scm_analysis"]["metrics"].get("vgdl_cosine_similarity", 0)
            val_sem = entry["non_scm_analysis"]["metrics"].get("semantic_text_similarity", 0)
            
            for block in blocks_to_update:
                block["non_scm"]["code_sim"].append(val_code)
                block["non_scm"]["sem_sim"].append(val_sem)
        else:
            missing_counts["non_scm"] += 1

        # 2. Aggregate SCM Metrics
        if "metrics" in entry.get("scm_analysis", {}):
            val_code = entry["scm_analysis"]["metrics"].get("vgdl_cosine_similarity", 0)
            val_sem = entry["scm_analysis"]["metrics"].get("semantic_text_similarity", 0)
            
            for block in blocks_to_update:
                block["scm"]["code_sim"].append(val_code)
                block["scm"]["sem_sim"].append(val_sem)
        else:
            missing_counts["scm"] += 1
        
        # 3. Aggregate LLM Wins
        winners = entry.get("llm_comparative_eval", {}).get("parsed_winners", {})
        for model, winner in winners.items():
            if model in overall_stats["wins"]:
                if winner in ["ABC", "XYZ", "TIE"]:
                    for block in blocks_to_update:
                        block["wins"][model][winner] += 1
                        block["wins"][model]["total"] += 1
    
    # --- Report Formatting Helpers ---
    def get_avg(lst): 
        return sum(lst) / len(lst) if lst else 0.0

    lines = []
    lines.append("="*80)
    lines.append("ANALYSIS")
    lines.append("="*80)
    lines.append(f"Total Entries Processed: {missing_counts['total_entries']}")
    lines.append(f"Missing VGDLs - Non-SCM: {missing_counts['non_scm']} | SCM: {missing_counts['scm']}")
    lines.append("\n")

    # --- SECTION 1: OVERALL STATISTICS ---
    lines.append("-" * 40)
    lines.append("1. OVERALL STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Non-SCM (ABC) - Avg Code Sim:     {get_avg(overall_stats['non_scm']['code_sim']):.4f}")
    lines.append(f"Non-SCM (ABC) - Avg Sem Sim:      {get_avg(overall_stats['non_scm']['sem_sim']):.4f}")
    lines.append(f"SCM (XYZ)     - Avg Code Sim:     {get_avg(overall_stats['scm']['code_sim']):.4f}")
    lines.append(f"SCM (XYZ)     - Avg Sem Sim:      {get_avg(overall_stats['scm']['sem_sim']):.4f}")
    lines.append("")
    lines.append("Overall LLM Win Rates:")
    lines.append(f"{'Judge Model':<30} | {'ABC %':<8} | {'XYZ %':<8} | {'TIE %':<8} | {'Total'}")
    for model, counts in overall_stats["wins"].items():
        total = counts["total"] if counts["total"] > 0 else 1
        lines.append(f"{model:<30} | {counts['ABC']/total*100:6.1f}%  | {counts['XYZ']/total*100:6.1f}%  | {counts['TIE']/total*100:6.1f}%  | {total}")
    lines.append("\n")

    # --- SECTION 2: CONTEXT LEVEL ANALYSIS ---
    lines.append("-" * 40)
    lines.append("2. ANALYSIS BY CONTEXT LEVEL")
    lines.append("-" * 40)
    
    sorted_levels = sorted(level_stats.keys())
    
    # Table for Quantitative Metrics
    lines.append("Quantitative Metrics per Level:")
    lines.append(f"{'Level':<15} | {'NS-Code':<8} | {'NS-Sem':<8} | {'SCM-Code':<8} | {'SCM-Sem':<8}")
    for lvl in sorted_levels:
        stats = level_stats[lvl]
        lines.append(f"{lvl:<15} | {get_avg(stats['non_scm']['code_sim']):<8.3f} | {get_avg(stats['non_scm']['sem_sim']):<8.3f} | {get_avg(stats['scm']['code_sim']):<8.3f} | {get_avg(stats['scm']['sem_sim']):<8.3f}")
    lines.append("")

    # Table for Win Rates (Aggregating all 4 judges for brevity in level breakdown)
    lines.append("Aggregated LLM Preference (Avg of all 4 judges) per Level:")
    lines.append(f"{'Level':<15} | {'ABC Wins':<8} | {'XYZ Wins':<8} | {'Ties':<8} | {'Total Votes'}")
    for lvl in sorted_levels:
        tot_abc = sum(level_stats[lvl]["wins"][m]["ABC"] for m in level_stats[lvl]["wins"])
        tot_xyz = sum(level_stats[lvl]["wins"][m]["XYZ"] for m in level_stats[lvl]["wins"])
        tot_tie = sum(level_stats[lvl]["wins"][m]["TIE"] for m in level_stats[lvl]["wins"])
        grand_total = tot_abc + tot_xyz + tot_tie
        grand_total_safe = grand_total if grand_total > 0 else 1
        lines.append(f"{lvl:<15} | {tot_abc/grand_total_safe*100:6.1f}%  | {tot_xyz/grand_total_safe*100:6.1f}%  | {tot_tie/grand_total_safe*100:6.1f}%  | {grand_total}")
    lines.append("\n")

    # --- SECTION 3: GAME-WISE ANALYSIS ---
    lines.append("-" * 40)
    lines.append("3. ANALYSIS BY GAME")
    lines.append("-" * 40)
    
    sorted_games = sorted(game_stats.keys())
    
    lines.append("Quantitative Metrics per Game:")
    lines.append(f"{'Game':<20} | {'NS-Code':<8} | {'NS-Sem':<8} | {'SCM-Code':<8} | {'SCM-Sem':<8}")
    for g in sorted_games:
        stats = game_stats[g]
        lines.append(f"{g:<20} | {get_avg(stats['non_scm']['code_sim']):<8.3f} | {get_avg(stats['non_scm']['sem_sim']):<8.3f} | {get_avg(stats['scm']['code_sim']):<8.3f} | {get_avg(stats['scm']['sem_sim']):<8.3f}")
    lines.append("")

    lines.append("Aggregated LLM Preference per Game:")
    lines.append(f"{'Game':<20} | {'ABC Wins':<8} | {'XYZ Wins':<8} | {'Ties':<8} | {'Total Votes'}")
    for g in sorted_games:
        tot_abc = sum(game_stats[g]["wins"][m]["ABC"] for m in game_stats[g]["wins"])
        tot_xyz = sum(game_stats[g]["wins"][m]["XYZ"] for m in game_stats[g]["wins"])
        tot_tie = sum(game_stats[g]["wins"][m]["TIE"] for m in game_stats[g]["wins"])
        grand_total = tot_abc + tot_xyz + tot_tie
        grand_total_safe = grand_total if grand_total > 0 else 1
        lines.append(f"{g:<20} | {tot_abc/grand_total_safe*100:6.1f}%  | {tot_xyz/grand_total_safe*100:6.1f}%  | {tot_tie/grand_total_safe*100:6.1f}%  | {grand_total}")

    # Write to file
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
        
    print(f"Statistics summary for paper saved to: {output_path}")


# --- MAIN EXECUTION ---

def main():
    # --- CONFIGURATION ---
    json_file_path = 'VGDL_gen/Qwen_Qwen3-14B_4bit_results_.json'
    gvgai_gym_path = 'GVGAI_GYM'
    output_folder = 'vgdl_results'
    # ---------------------

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    base_name = os.path.basename(json_file_path)
    model_name = os.path.splitext(base_name)[0].replace('_results', '')
    
    output_path = os.path.join(output_folder, f"{model_name}_full_analysis.json")
    csv_table_path = os.path.join(output_folder, f"{model_name}_preference_table.csv")
    stats_output_path = os.path.join(output_folder, f"{model_name}_new_paper_stats.txt")

    print(f"Loading data from: {json_file_path}")
    data = load_json_data(json_file_path)
    if not data: return

    # Traverse JSON structure
    # Structure: results -> non_scm -> level -> first_10 -> iteration -> game
    results_data = data.get('results', {})
    non_scm_root = results_data.get('non_scm', {})

    full_analysis = []
    csv_rows = []
    
    # Cache for Ground Truth VGDL and Descriptions to avoid re-reading/re-generating
    gt_cache = {} 

    print("Starting analysis loop...")

    # Iterate Levels
    for level, level_data in non_scm_root.items():
        if level == "vgdl": continue # Skip if incorrect key parsing
        
        iterations = level_data.get('first_10', {})
        
        # Iterate Iterations
        for iteration, iter_data in iterations.items():
            
            # Iterate Games
            for game, game_data in iter_data.items():
                
                print(f"Processing: Game={game}, Level={level}, Iteration={iteration}\n")
                print("Running GT VGDL retrieval and converting to Descriptions\n\n")
                # 1. Retrieve Data
                non_scm_vgdl = get_generated_vgdl_direct(data, game, level, iteration, 'non_scm')
                scm_vgdl = get_generated_vgdl_direct(data, game, level, iteration, 'scm')
                
                # 2. Retrieve Ground Truth (with caching)
                if game not in gt_cache:
                    raw_gt = get_ground_truth_vgdl(game, gvgai_gym_path)
                    if raw_gt:
                        desc_gt = vgdl_to_natural_language(raw_gt)
                    else:
                        raw_gt = ""
                        desc_gt = ""
                    gt_cache[game] = {"vgdl": raw_gt, "desc": desc_gt}
                
                gt_vgdl = gt_cache[game]["vgdl"]
                gt_desc = gt_cache[game]["desc"]

                # 3. Run Variant Metrics    
                metrics_abc = analyze_variant("non_scm", non_scm_vgdl, gt_vgdl, gt_desc, game)
                metrics_xyz = analyze_variant("scm", scm_vgdl, gt_vgdl, gt_desc, game)

                # 4. LLM Preference Evaluation (Comparative)
                llm_results = {}
                parsed_winners = {}
                print("Running LLM Comparative Evaluation...\n")
                if non_scm_vgdl and scm_vgdl and gt_vgdl:
                    print(f"  > querying LLMs for {game}...")
                    prompt = create_comparison_prompt_text(non_scm_vgdl, scm_vgdl, gt_vgdl)
                    llm_results = run_llm_comparison(prompt)
                    
                    # Analyze Winners using exact model keys
                    parsed_winners = {
                        "gpt-5.2-2025-12-11": extract_winner(llm_results.get("gpt-5.2-2025-12-11")),
                        "gemini-2.5-pro": extract_winner(llm_results.get("gemini-2.5-pro")),
                        "claude-opus-4-5-20251101": extract_winner(llm_results.get("claude-opus-4-5-20251101")),
                        "qwen-3-8b": extract_winner(llm_results.get("qwen-3-8b"))
                    }
                else:
                    print(f"  > Skipping LLM comparison for {game} due to missing VGDL.")
                    llm_results = {"status": "Skipped - Missing VGDL content"}
                    parsed_winners = {
                        "gpt-5.2-2025-12-11": "N/A", 
                        "gemini-2.5-pro": "N/A", 
                        "claude-opus-4-5-20251101": "N/A",
                        "qwen-3-8b": "N/A"
                    }

                # 5. Store Data for JSON
                entry = {
                    "meta": {
                        "game": game,
                        "level": level,
                        "iteration": iteration
                    },
                    "non_scm_analysis": metrics_abc,
                    "scm_analysis": metrics_xyz,
                    "llm_comparative_eval": {
                        "raw_responses": llm_results,
                        "parsed_winners": parsed_winners
                    }
                }
                full_analysis.append(entry)
                
                # 6. Store Data for CSV Table
                csv_rows.append({
                    "Game": game,
                    "Level": level,
                    "Iteration": iteration,
                    "GPT-5.2 Winner": parsed_winners["gpt-5.2-2025-12-11"],
                    "Gemini-3 Winner": parsed_winners["gemini-2.5-pro"],
                    "Claude-Opus Winner": parsed_winners["claude-opus-4-5-20251101"],
                    "Qwen-3 Winner": parsed_winners["qwen-3-8b"]
                })

    # Save consolidated JSON results
    final_output = {
        "model_name": model_name,
        "results": full_analysis
    }

    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=4)
        
    # Save CSV Table
    if csv_rows:
        headers = ["Game", "Level", "Iteration", "GPT-5.2 Winner", "Gemini-3 Winner", "Claude-Opus Winner", "Qwen-3 Winner"]
        with open(csv_table_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Preference table saved to: {csv_table_path}")
    
    # Save Paper Statistics
    save_paper_statistics(full_analysis, stats_output_path)

    print(f"\nFull analysis complete. processed {len(full_analysis)} entries.")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()