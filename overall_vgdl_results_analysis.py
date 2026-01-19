    # 'vgdl_results/_Qwen_Qwen3-8B_new_paper_stats.txt',
    # 'vgdl_results/Qwen_QwQ-32B_4bit_new_paper_stats.txt'

import os
import re

# List of specific file names to parse
FILES_TO_PARSE = [
    'vgdl_results/Qwen_Qwen3-8B_2_new_paper_stats.txt',
    'vgdl_results/Qwen_QwQ-32B_4bit__new_paper_stats.txt'
    # Add other filenames here
]

def clean_and_extract(file_path):
    """Cleans source tags and extracts both Win Rates and Similarity Metrics."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    model_name = os.path.basename(file_path).split('_new_paper_stats')[0]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # Remove tags to fix row fragmentation [cite: 1, 4, 7]
    content = re.sub(r'\\', '', content)
    
    # --- Extraction Helpers ---
    
    def get_win_rows(section_pattern, text):
        match = re.search(section_pattern, text, re.DOTALL)
        if not match: return {}
        # Pattern for: Name | % | % | % | Total [cite: 19, 45]
        pattern = r'([^|\n]+?)\s*\|\s*([\d\.]+%)\s*\|\s*([\d\.]+%)\s*\|\s*([\d\.]+%)\s*\|\s*(\d+)'
        rows = re.findall(pattern, match.group(1))
        return {r[0].strip().split('\n')[-1].strip(): r[1:] for r in rows}

    def get_sim_rows(section_pattern, text):
        match = re.search(section_pattern, text, re.DOTALL)
        if not match: return {}
        # Pattern for: Name | Float | Float | Float | Float 
        pattern = r'([^|\n]+?)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)'
        rows = re.findall(pattern, match.group(1))
        return {r[0].strip().split('\n')[-1].strip(): r[1:] for r in rows}

    # --- Parse Sections ---
    
    # Overall Win Rates (Doesn't have similarity metrics in the same way) [cite: 4]
    overall_wins = get_win_rows(r"Overall LLM Win Rates:(.*?)2\. ANALYSIS", content)
    
    # Context Level 
    level_sims = get_sim_rows(r"Quantitative Metrics per Level:(.*?)Aggregated", content)
    level_wins = get_win_rows(r"Aggregated LLM Preference.*?per Level:(.*?)3\. ANALYSIS", content)
    
    # Game Analysis 
    game_sims = get_sim_rows(r"Quantitative Metrics per Game:(.*?)Aggregated", content)
    game_wins = get_win_rows(r"Aggregated LLM Preference per Game:(.*)", content)

    # --- Combine Metrics into single rows ---
    
    combined_levels = []
    for lbl in level_wins:
        sim = level_sims.get(lbl, ("-", "-", "-", "-"))
        win = level_wins[lbl]
        combined_levels.append((model_name, lbl, *sim, *win))

    combined_games = []
    for lbl in game_wins:
        sim = game_sims.get(lbl, ("-", "-", "-", "-"))
        win = game_wins[lbl]
        combined_games.append((model_name, lbl, *sim, *win))

    overall_results = [(model_name, judge, *vals) for judge, vals in overall_wins.items()]

    return overall_results, combined_levels, combined_games

def write_txt_table(f, title, headers, rows):
    f.write(f"\n{title}\n" + "="*len(title) + "\n")
    if not rows:
        f.write("No data found.\n")
        return
    widths = [max(len(str(row[i])) for row in ([headers] + rows)) for i in range(len(headers))]
    fmt = " | ".join([f"{{:<{w}}}" for w in widths])
    f.write(fmt.format(*headers) + "\n")
    f.write("-+-".join(["-"*w for w in widths]) + "\n")
    for row in rows:
        f.write(fmt.format(*row) + "\n")

def generate_latex(title, headers, rows):
    if not rows: return ""
    col_config = "|" + "|".join(['l'] * len(headers)) + "|"
    latex = f"\\subsection*{{{title}}}\n"
    latex += "\\begin{table}[h]\n\\centering\n"
    latex += f"\\begin{{tabular}}{{{col_config}}}\n\\hline\n"
    latex += " & ".join(headers).replace("%", "\\%").replace("-", "\\textendash") + " \\\\\n\\hline\n"
    for row in rows:
        formatted_row = [str(item).replace("%", "\\%").replace("_", "\\_") for item in row]
        latex += " & ".join(formatted_row) + " \\\\\n"
    latex += "\\hline\n\\end{tabular}\n\\end{table}\n\n"
    return latex

def main():
    final_overall, final_levels, final_games = [], [], []

    for file_path in FILES_TO_PARSE:
        results = clean_and_extract(file_path)
        if results:
            o, l, g = results
            final_overall.extend(o)
            final_levels.extend(l)
            final_games.extend(g)

    if not final_overall:
        print("Extraction failed. Check file content.")
        return

    with open('overall_results.txt', 'w', encoding='utf-8') as f:
        f.write("CONSOLIDATED LLM BENCHMARK RESULTS\n\n")
        
        # Headers for combined metrics
        sim_win_headers = ["Model", "Target", "NS-Code", "NS-Sem", "SCM-Code", "SCM-Sem", "NS Win%", "SCM Win%", "Tie%", "Votes"]
        
        write_txt_table(f, "OVERALL WIN RATES", ["Model", "Judge", "NS %", "SCM %", "Tie %", "Total"], final_overall)
        write_txt_table(f, "LEVEL ANALYSIS (SIMILARITY + WINS)", sim_win_headers, final_levels)
        write_txt_table(f, "GAME ANALYSIS (SIMILARITY + WINS)", sim_win_headers, final_games)
        
        f.write("\n" + "="*40 + "\nLATEX CODE\n" + "="*40 + "\n\n")
        f.write(generate_latex("Overall Judge Win Rates", ["Model", "Judge", "NS %", "SCM %", "Tie %", "Total"], final_overall))
        f.write(generate_latex("Context Level Metrics", sim_win_headers, final_levels))
        f.write(generate_latex("Game-Specific Metrics", sim_win_headers, final_games))

    print("Success: results written to overall_results.txt")

if __name__ == "__main__":
    main()