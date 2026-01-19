import os
import json
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import re
from sklearn.metrics import confusion_matrix
from scipy.cluster import hierarchy
from scipy.spatial import distance

# New imports for the requested analysis
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: 'sentence_transformers' not found. Semantic analysis will be skipped.")
    SentenceTransformer = None

from adjustText import adjust_text   

# ==========================================
# 1. Configuration
# ==========================================
RESULTS_FOLDER = "./results"
OUTPUT_FOLDER = "./comparative_analysis_plots"

sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['savefig.bbox'] = 'tight'



#Time required for model runs, obtained from slurm (in seconds)
MODEL_RUN_TIMES = {
    "Olmo-3-7B-Think": "11:10:07",
    "Olmo-3-7B-Think (4bit-nf4)": "08:07:30",
    "OREAL-32B": "13:26:43",
    "OREAL-32B (4bit-nf4)": "00:23:20",
    "OREAL-7B": "02:31:36",
    "OREAL-7B (4bit-nf4)": "00:57:07",
    "Phi-4-reasoning (4bit-nf4)": "11:18:45",
    "Phi-4-reasoning": "22:09:05",
    "QwQ-32B": "8-19:55:28",
    "QwQ-32B (4bit-nf4)": "03:43:16",
    "Qwen3-14B": "03:51:50",
    "Qwen3-14B (4bit-nf4)": "02:23:26",
    "Qwen3-30B-A3B-Instruct-2507": "04:34:43",
    "Qwen3-30B-A3B-Instruct-2507 (4bit-nf4)": "01:52:55",
    "Qwen3-30B-A6B-16-Extreme": "2-06:37:13",
    "Qwen3-30B-A6B-16-Extreme (4bit-nf4)": "22:05:58",
    "Qwen3-32B (4bit-nf4)": "03:09:07",
    "Qwen3-8B": "03:25:02",
    "Qwen3-8B (4bit-nf4)": "02:32:13",
    "Qwen3-Coder-30B-A3B-Instruct": "04:01:49",
    "Qwen3-Coder-30B-A3B-Instruct (4bit-nf4)": "01:26:41",
    "Marco-o1 (4bit-nf4)": "00:13:04",
    "Marco-o1": "00:14:33",
    
}

# Convert the model runtime from dd-hh:mm:ss format to total seconds
def convert_runtime_to_seconds(runtime_str):
    if runtime_str is None:
        return None
    if '-' in runtime_str:
        days, time_part = runtime_str.split('-')
        days = int(days)
    else:
        days = 0
        time_part = runtime_str

    h, m, s = map(int, time_part.split(':'))
    total_seconds = days * 86400 + h * 3600 + m * 60 + s
    return total_seconds

MODEL_RUN_TIMES_SECONDS = {model: convert_runtime_to_seconds(runtime) for model, runtime in MODEL_RUN_TIMES.items()}


GAME_DESCRIPTION_OVERRIDES: Dict[str, str] = {
    "racebet": "The objective of this game is guessing which camel is going to win the race. Betting is performed by placing the player in one of the corresponding betting spots. No points are given in this game. You either bet for the one that wins (and the player wins) or you don't (and the player loses).",
    "digdug": "The player must collect all gems and gold coins in the cave, digging its way through it, to win the game. There are also enemies in the level that kill the player on collision with them. Also, the player can shoot boulders by pressing USE two consecutive time steps, which kill enemies.",
    "brainman": "The player must push keys into doors to open them, to reach the exit to win the game. Many gems around are available for collection, which give score to the player. Keys are pushed by colliding with them, and they are propelled in a straight line until finding a sprite to collide against, when they stop. If they hit a door, both the door and the key are destroyed.",
    "defender": "The objective of this game is to avoid that the enemies destroy your city. The enemies are spawned at the right of the screen and cross it while bombing the city. The player must eliminate these enemies as soon as possible so the city is not destroyed, otherwise the player loses. The player’s ammunition is supplied from the top of the screen and must be collected periodically to avoid running out of ammo.",
    "lasers2": "The objective of this game is to reach the exit, which is typically blocked by a mud sprite. The mud can be destroyed by directing a laser to it. There are several laser cannons whose orientation can be changed by the player shooting at them. Apart from this, there are several crystal blocks, which can also be rotated, that deflect the lasers in a certain direction. Lasers kill the player when in contact.",
    "boulderchase": "In this game, the player must dig inside a cave to collect at least 9 gems and reach the exit. As the player digs, boulders that are left without a ground fall and can kill the player upon collision. Enemies also move around the cave, being able to dig as well. When digging, they drop gems that can also be collected by the player. Enemies are also killed by boulders and kill the player upon contact.",
    "portals": "The player controls an avatar that needs to find the exit of a maze. The maze is full of entry and exit portals, and every time the player goes through an entry portal they will be teleported to an exit portal at random. The maze is also full of enemies and missiles that kill the player on contact.",
    "iceandfire": "The objective of this game is to find the exit of the maze. The maze has multiple traps, two different types of surfaces (ice and fire), and two types of boots (for ice and fire) that can be collected. The special surfaces kill the player on contact, unless the appropriate boots are picked up beforehand. There are several coins scattered around the level that give the player some score.",
    "cakybaky": "The objective of the game is to bake a cake. There are several ingredients that must be collected in order, to follow the recipe. This order to follow is flour, milk, eggs, sugar, butter and cherries, although not all levels have all these ingredients. There are angry chefs around the level that chase the player, although they only care about their favourite ingredient, so only the ones that prefer the next ingredient to be picked up are active at each time. If a chef collides with the player, the player loses."
}

# ==========================================
# 2. Data Loading & Parsing
# ==========================================

def parse_model_name(filename):
    """
    Extracts a distinct model identifier from the filename.
    Treats quantized versions (e.g., 4bit-nf4) as separate models.
    """
    base = os.path.basename(filename)
    name_part = os.path.splitext(base)[0]
    
    if '@' in name_part:
        parts = name_part.split('@')
        base_model = parts[0]
        try:
            # Try to clean up the prefix if it exists (e.g., internlm__OREAL -> OREAL)
            if '__' in base_model:
                base_model = base_model.split("__")[1]
        except IndexError:
            pass # Keep original if split fails
            
        suffix = parts[1]
        
        quant_markers = ['4bit', '8bit', 'nf4', 'gguf', 'awq', 'gptq', 'int4', 'int8', 'q4', 'q8']
        found_quants = [marker for marker in quant_markers if marker in suffix.lower()]
        
        if found_quants:
            quant_tag = suffix.replace('w_description', '').strip('_').strip('-')
            if quant_tag:
                return f"{base_model} ({quant_tag})"
            
        return base_model
    
    return name_part

def load_all_results(folder_path):
    all_records = []
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {folder_path}!")
        return pd.DataFrame()

    print(f"Found {len(json_files)} files. Processing...")

    for filepath in json_files:
        model_name = parse_model_name(filepath)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

        subsets = data.get('subsets', {})
        for subset_key, modes in subsets.items():
            for mode_key, iterations in modes.items():
                for iter_key, games in iterations.items():
                    for game_key, details in games.items():
                        ground_truth = game_key.lower().strip().replace('.', '')
                        prediction = details.get('prediction', 'none').lower().strip().replace('.', '')
                        is_correct = (ground_truth == prediction)
                        
                        all_records.append({
                            'Model': model_name,
                            'Mode': mode_key,
                            'Subset': subset_key,
                            'Iteration': int(iter_key),
                            'Game': ground_truth,
                            'Prediction': prediction,
                            'IsCorrect': 1 if is_correct else 0
                        })

    return pd.DataFrame(all_records)

def apply_mode_ordering(df):
    desired_order = ['destructive', 'standard', 'constructive', 'vgdl']
    existing = df['Mode'].unique()
    final_order = [m for m in desired_order if m in existing]
    final_order += [m for m in existing if m not in final_order]
    df['Mode'] = pd.Categorical(df['Mode'], categories=final_order, ordered=True)
    return df

def get_model_order(df, sort_by_size=True, metric_col='IsCorrect', ascending=False):
    if sort_by_size:
        def get_size(name):
            match = re.search(r'(\d+(?:\.\d+)?)[bB]', name)
            return float(match.group(1)) if match else 9999.0
        unique_models = df['Model'].unique()
        return sorted(unique_models, key=lambda x: (get_size(x), x))
    else:
        return df.groupby('Model')[metric_col].mean().sort_values(ascending=ascending).index.tolist()

# ==========================================
# 3. Visualization Functions
# ==========================================

def plot_figure_1a_bar(raw_df, output_dir, sort_by_size=True):
    plt.figure(figsize=(16, 8))
    iter_stats = raw_df.groupby(['Model', 'Mode', 'Iteration'])['IsCorrect'].mean().reset_index()
    iter_stats['Accuracy (%)'] = iter_stats['IsCorrect'] * 100
    order = get_model_order(iter_stats, sort_by_size, 'Accuracy (%)', ascending=False)
    xlabel_text = 'Model Architecture' if sort_by_size else 'Model Architecture'

    ax = sns.barplot(
        data=iter_stats, x='Model', y='Accuracy (%)', hue='Mode',
        order=order, errorbar='sd', capsize=0.1, palette='viridis', edgecolor='.2'
    )
    plt.title('Model Performance by Prompt Variants\n(Mean Accuracy \u00B1 Standard Deviation)', pad=20, fontweight='bold', fontsize=16)
    plt.xlabel(xlabel_text, fontweight='bold', fontsize=16)
    plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=16)
    plt.ylim(0, 105)
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.legend(title='Description Mode', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=16)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1a_model_performance_bar.png"))
    plt.show()

def plot_figure_1b_line(raw_df, output_dir, sort_by_size=True):
    plt.figure(figsize=(14, 8))
    iter_stats = raw_df.groupby(['Model', 'Mode', 'Iteration'])['IsCorrect'].mean().reset_index()
    iter_stats['Accuracy (%)'] = iter_stats['IsCorrect'] * 100
    order = get_model_order(iter_stats, sort_by_size, 'Accuracy (%)', ascending=False)
    xlabel_text = 'Model Architecture (Sorted by Size)' if sort_by_size else 'Model Architecture (Sorted by Accuracy)'

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    unique_modes = raw_df['Mode'].cat.categories
    mode_markers = {mode: markers[i % len(markers)] for i, mode in enumerate(unique_modes)}

    sns.pointplot(
        data=iter_stats, x='Model', y='Accuracy (%)', hue='Mode',
        order=order, errorbar='sd', capsize=0.1, dodge=0.1, palette='tab10',
        markers=[mode_markers[m] for m in unique_modes], linestyles=''
    )
    plt.title('Sensitivity to Description Variants (Line)\n(Mean Accuracy \u00B1 Standard Deviation)', pad=20, fontweight='bold')
    plt.xlabel(xlabel_text, fontweight='bold', fontsize=15)
    plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=15)
    plt.ylim(0, 105)
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(rotation=45, fontsize=15)
    plt.legend(title='Description Mode', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=15)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1b_model_performance_line.png"))
    plt.savefig(os.path.join(output_dir, "model_performance_line.png"))
    plt.show()

def plot_overall_leaderboard_with_std(raw_df, output_dir, sort_by_size=True):
    plt.figure(figsize=(12, 7))
    iter_stats = raw_df.groupby(['Model', 'Iteration'])['IsCorrect'].mean().reset_index()
    iter_stats['Accuracy (%)'] = iter_stats['IsCorrect'] * 100
    order = get_model_order(iter_stats, sort_by_size, 'Accuracy (%)', ascending=False)
    title_text = 'Fig 2: Overall Model Accuracy' + (' (Sorted by Size)' if sort_by_size else ' (Sorted by Rank)')

    ax = sns.barplot(
        data=iter_stats, x='Accuracy (%)', y='Model', order=order, hue='Model',
        errorbar='sd', capsize=0.2, palette='rocket', edgecolor='.2', legend=False
    )
    plt.title(f'{title_text}\n(Mean \u00B1 Standard Deviation)', pad=15, fontweight='bold')
    plt.xlabel('Global Average Accuracy (%)')
    plt.ylabel('')
    plt.xlim(0, 100)
    means = iter_stats.groupby('Model')['Accuracy (%)'].mean()
    for i, model in enumerate(order):
        val = means[model]
        ax.text(val + 2, i, f"{val:.1f}%", va='center', fontweight='bold', color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_overall_leaderboard_with_std.png"))
    plt.show()

def plot_stability_analysis(raw_df, output_dir, sort_by_size=True):
    plt.figure(figsize=(12, 6))
    iter_acc = raw_df.groupby(['Model', 'Mode', 'Iteration'])['IsCorrect'].mean().reset_index()
    std_per_mode = iter_acc.groupby(['Model', 'Mode']).agg(Std_Dev=('IsCorrect', 'std')).reset_index()
    # std_per_mode = iter_acc.groupby(['Model', 'Iteration']).agg(Std_Dev=('IsCorrect', 'std')).reset_index()
    std_per_mode['Std_Dev_Pct'] = std_per_mode['Std_Dev'] * 100
    order = get_model_order(std_per_mode, sort_by_size, 'Std_Dev_Pct', ascending=True)
    title_suffix = "(Sorted by Size)" if sort_by_size else "(Sorted by Stability)"

    sns.barplot(
        data=std_per_mode, x='Model', y='Std_Dev_Pct', hue='Model', order=order,
        palette='Reds_r', edgecolor='.2', errorbar=None, legend=False
    )
    plt.title(f'Fig 3: Model Stability Analysis {title_suffix}\n(Average Standard Deviation - Lower is Better)', pad=15, fontweight='bold')
    plt.ylabel('Standard Deviation (%)')
    plt.xlabel('Model Architecture')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_model_stability.png"))
    plt.show()

def plot_game_mastery_heatmap(raw_df, output_dir, sort_by_size=True):
    plt.figure(figsize=(16, 10))
    game_stats = raw_df.groupby(['Model', 'Game'])['IsCorrect'].mean().reset_index()
    heatmap_data = game_stats.pivot(index='Model', columns='Game', values='IsCorrect')
    order = get_model_order(game_stats, sort_by_size, 'IsCorrect', ascending=False)
    heatmap_data = heatmap_data.reindex(order)

    sns.heatmap(heatmap_data, annot=True, fmt=".0%", cmap="YlGnBu", linewidths=.5, cbar_kws={'label': 'Accuracy'})
    suffix = "Size" if sort_by_size else "Performance"
    plt.title(f'Fig 4: Game Mastery Profile (Models Sorted by {suffix})', pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_game_mastery_heatmap.png"))
    plt.show()

def plot_model_performance_per_game(raw_df, output_dir, sort_by_size=True):
    plt.figure(figsize=(20, 8))
    game_model_stats = raw_df.groupby(['Model', 'Game'])['IsCorrect'].mean().reset_index()
    game_model_stats['Accuracy (%)'] = game_model_stats['IsCorrect'] * 100
    hue_order = get_model_order(game_model_stats, sort_by_size, 'Accuracy (%)', ascending=False)
    
    sns.barplot(
        data=game_model_stats, x='Game', y='Accuracy (%)', hue='Model',
        hue_order=hue_order, palette='muted', edgecolor='.2'
    )
    plt.title('Fig 5: Deep Dive - Performance by Game Title', pad=20, fontweight='bold')
    plt.ylim(0, 105)
    plt.legend(title='Model', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "5_model_performance_per_game.png"))
    plt.show()

def plot_mode_effectiveness_per_game(raw_df, output_dir, sort_by_size=True):
    impact_stats = raw_df.groupby(['Game', 'Mode'])['IsCorrect'].mean().reset_index()
    heatmap_data = impact_stats.pivot(index='Game', columns='Mode', values='IsCorrect')
    mode_order = raw_df['Mode'].cat.categories
    heatmap_data = heatmap_data.reindex(columns=mode_order)
    
    row_means = pd.DataFrame(heatmap_data.mean(axis=1), columns=['Average'])
    col_means = pd.DataFrame(heatmap_data.mean(axis=0)).T
    col_means.index = ['Average']
    grand_mean = pd.DataFrame([heatmap_data.values.mean()], index=['Average'], columns=['Average'])

    plt.figure(figsize=(14, 10))
    n_rows, n_cols = heatmap_data.shape
    gs = gridspec.GridSpec(2, 2, width_ratios=[n_cols, 1], height_ratios=[n_rows, 1], wspace=0.02, hspace=0.02)
    vmin = min(heatmap_data.min().min(), row_means.min().min(), col_means.min().min())
    vmax = max(heatmap_data.max().max(), row_means.max().max(), col_means.max().max())
    cmap = "RdYlGn"

    ax_main = plt.subplot(gs[0, 0])
    sns.heatmap(heatmap_data, annot=True, fmt=".1%", cmap=cmap, cbar=False, vmin=vmin, vmax=vmax, ax=ax_main)
    ax_main.set_xlabel(''); ax_main.set_xticklabels([])
    
    ax_row = plt.subplot(gs[0, 1])
    sns.heatmap(row_means, annot=True, fmt=".1%", cmap=cmap, cbar=False, vmin=vmin, vmax=vmax, ax=ax_row, yticklabels=False)
    ax_row.set_title("Avg", fontweight='bold', fontsize=11, pad=10)
    ax_row.set_xlabel(''); ax_row.set_ylabel(''); ax_row.set_xticklabels([])

    ax_col = plt.subplot(gs[1, 0])
    sns.heatmap(col_means, annot=True, fmt=".1%", cmap=cmap, cbar=False, vmin=vmin, vmax=vmax, ax=ax_col, xticklabels=True)
    ax_col.set_ylabel('Average', fontweight='bold', rotation=0, ha='right', va='center')
    ax_col.set_xlabel('')
    
    ax_grand = plt.subplot(gs[1, 1])
    sns.heatmap(grand_mean, annot=True, fmt=".1%", cmap=cmap, cbar=False, vmin=vmin, vmax=vmax, ax=ax_grand, xticklabels=False, yticklabels=False)
    ax_grand.set_ylabel(''); ax_grand.set_xlabel('')

    cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=cbar_ax, label='Accuracy')

    plt.suptitle('Fig 6: Description Mode Effectiveness per Game\n(With Marginal Averages)', fontsize=16, fontweight='bold', y=0.95)
    plt.savefig(os.path.join(output_dir, "6_mode_effectiveness_heatmap.png"), bbox_inches='tight')
    plt.show()

def plot_error_correlation_heatmap(raw_df, output_dir, sort_by_size=True):
    plt.figure(figsize=(12, 10))
    confusion_counts = raw_df.groupby(['Model', 'Game', 'Prediction']).size().reset_index(name='Count')
    model_signatures = confusion_counts.pivot(index='Model', columns=['Game', 'Prediction'], values='Count').fillna(0)
    order = get_model_order(raw_df, sort_by_size, 'IsCorrect', ascending=False)
    order = [m for m in order if m in model_signatures.index]
    model_signatures = model_signatures.reindex(order)
    
    corr_matrix = model_signatures.T.corr(method='pearson')
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1, linewidths=.5, square=True)
    suffix = "Size" if sort_by_size else "Similarity"
    plt.title(f'Fig 7: Model Error Correlation Matrix (Ordered by {suffix})', pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "7_error_correlation_heatmap.png"))
    plt.show()

def plot_global_confusion_matrix(raw_df, output_dir, sort_by_size=True):
    plt.figure(figsize=(7, 5))
    labels = sorted(raw_df['Game'].unique())
    cm = confusion_matrix(raw_df['Game'], raw_df['Prediction'], labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 1. Assign the heatmap to a variable 'ax'
    # 2. Remove 'label' from cbar_kws (we will set it manually below)
    ax = sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt=".2f", 
        cmap="Blues", 
        xticklabels=labels, 
        yticklabels=labels, 
        linewidths=.5, 
        square=True, 
        annot_kws={"size": 8}, # Size of numbers inside the matrix
        cbar_kws={}             # Empty dict here (or remove it entirely)
    )

    # 3. Access the colorbar object
    cbar = ax.collections[0].colorbar

    # 4. Set the label size and weight here
    cbar.set_label('Recall (Row Normalized)', fontsize=12,  labelpad=15)

    # Optional: Increase the size of the tick numbers (0.0, 0.2...) on the colorbar
    cbar.ax.tick_params(labelsize=12)
    
    plt.title('Global Confusion Matrix (All Models)', pad=12, fontsize=10, fontweight='bold')
    plt.ylabel('Ground Truth',  fontsize=10, fontweight='bold')
    plt.xlabel('Prediction', fontsize=10, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "8_global_confusion_matrix.png"))
    plt.savefig(os.path.join(output_dir, "global_confusion_matrix.png"))
    
    plt.show()

def plot_game_hardness_distribution(raw_df, output_dir, sort_by_size=True):
    plt.figure(figsize=(14, 8))
    model_game_acc = raw_df.groupby(['Model', 'Game'])['IsCorrect'].mean().reset_index()
    model_game_acc['Accuracy (%)'] = model_game_acc['IsCorrect'] * 100
    order = model_game_acc.groupby('Game')['Accuracy (%)'].median().sort_values().index
    
    sns.boxplot(data=model_game_acc, x='Game', y='Accuracy (%)', hue='Game', order=order, palette="vlag", showfliers=False, legend=False)
    sns.stripplot(data=model_game_acc, x='Game', y='Accuracy (%)', order=order, color=".25", alpha=0.6, dodge=True)
    plt.title('Fig 9: Task Hardness & Model Variance', pad=15, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.xlabel('Game Title (Sorted by Median Difficulty)', fontweight='bold')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "9_game_hardness_distribution.png"))
    plt.show()

def plot_subset_consistency_by_model(raw_df, output_dir, sort_by_size=True):
    plt.figure(figsize=(14, 8))
    stats = raw_df.groupby(['Model', 'Subset', 'Iteration'])['IsCorrect'].mean().reset_index()
    stats['Accuracy (%)'] = stats['IsCorrect'] * 100
    order = get_model_order(stats, sort_by_size, 'Accuracy (%)', ascending=False)
    xlabel_text = 'Model Architecture (Sorted by Size)' if sort_by_size else 'Model Architecture (Sorted by Accuracy)'

    sns.pointplot(
        data=stats, x='Model', y='Accuracy (%)', hue='Subset', order=order,
        errorbar='sd', capsize=0.1, dodge=0.1, palette='coolwarm',
        markers=['o', 's'], linestyles=['-', '--']
    )
    plt.title('Fig 10: Performance Consistency Across Subsets (First 10 vs Last 10)', pad=20, fontweight='bold')
    plt.xlabel(xlabel_text, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.ylim(0, 105)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Data Subset', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "10_subset_consistency_by_model.png"))
    plt.show()

def plot_subset_consistency_by_game(raw_df, output_dir, sort_by_size=True):
    plt.figure(figsize=(14, 8))
    stats = raw_df.groupby(['Game', 'Subset', 'Iteration'])['IsCorrect'].mean().reset_index()
    stats['Accuracy (%)'] = stats['IsCorrect'] * 100
    game_order = stats.groupby('Game')['Accuracy (%)'].mean().sort_values().index
    
    sns.pointplot(
        data=stats, x='Game', y='Accuracy (%)', hue='Subset', order=game_order,
        errorbar='sd', capsize=0.1, dodge=0.1, palette='coolwarm',
        markers=['o', 's'], linestyles=['-', '--']
    )
    plt.title('Fig 11: Game Difficulty Consistency Across Subsets', pad=20, fontweight='bold')
    plt.xlabel('Game Title (Sorted by Difficulty)', fontweight='bold')
    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.ylim(0, 105)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Data Subset', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "11_subset_consistency_by_game.png"))
    plt.show()


def plot_robustness_performance_scatter(raw_df, output_dir, sort_by_size=True):
    # 1. Data Preparation
    
    # Step A: Calculate Accuracy for every specific run (Iteration) of every Prompt (Mode)
    # This gives us the raw performance for every single execution.
    per_run_acc = raw_df.groupby(['Model', 'Mode', 'Iteration'])['IsCorrect'].mean().reset_index()

    # Step B: Calculate Standard Deviation across ITERATIONS for each Mode
    # This measures "Wobble" or "Noise" for a specific prompt.
    per_mode_stability = per_run_acc.groupby(['Model', 'Mode'])['IsCorrect'].std().reset_index()
    
    # Step C: Aggregate to Model Level
    # X-Axis: Overall Mean Accuracy across all runs
    # Y-Axis: Average of the stability scores (Average 'Wobble')
    stats = raw_df.groupby('Model')['IsCorrect'].mean().reset_index(name='Mean_Accuracy')
    stability_score = per_mode_stability.groupby('Model')['IsCorrect'].mean().reset_index(name='Stability_Std')
    
    # Merge X and Y data
    stats = pd.merge(stats, stability_score, on='Model')

    # Convert to percentages
    stats['Mean_Accuracy_Pct'] = stats['Mean_Accuracy'] * 100
    stats['Stability_Std_Pct'] = stats['Stability_Std'] * 100
    
    # Create list of tuples: (Model, Accuracy, StdDev)
    data = []
    for i in range(len(stats)):
        data.append((
            stats.iloc[i]['Model'], 
            stats.iloc[i]['Mean_Accuracy_Pct'], 
            stats.iloc[i]['Stability_Std_Pct']
        ))

    # 2. Identify Pareto Frontier
    # Goal: Maximize Accuracy (X), Minimize Std Dev (Y).
    
    # Sort by Accuracy (Desc) AND THEN Std Dev (Asc).
    data.sort(key=lambda x: (-x[1], x[2]))
    
    pareto_points = []
    min_std_seen = float('inf') 
    
    for m, acc, std in data:
        # Only keep if this model has a strictly lower Std Dev than 
        # any model with higher (or equal) accuracy seen so far.
        if std < min_std_seen:
            pareto_points.append((m, acc, std))
            min_std_seen = std

    # Sort Pareto points by Accuracy (Ascending) for clean plotting from left to right
    pareto_points.sort(key=lambda x: x[1])

    # 3. Text Output (Console & LaTeX)
    print("\n====================== PARETO-OPTIMAL MODELS (Stability) ======================\n")
    print("Criteria: Maximize Accuracy, Minimize Std Dev (Run-to-Run Consistency)")
    header = f"{'Model':30s}      {'Accuracy (%)':>12s}  {'Std Dev (%)':>12s}"
    print(header)
    print("-" * len(header))
    for m, acc, std in pareto_points:
        print(f"{m:30s}      {acc:12.2f}  {std:12.2f}")
    print("\n==============================================================================\n")

    latex_rows = "\n".join(f"{m} & {acc:.2f} & {std:.2f} \\\\" for m, acc, std in pareto_points)
    
    latex_table = """
    \\begin{table}[ht]
        \\centering
        \\begin{tabular}{lrr}
            \\hline
            Model & Accuracy (\\%%) & Stability Std Dev (\\%%) \\\\
            \\hline
    %s
            \\hline
        \\end{tabular}
        \\caption{Pareto-optimal models (Stability vs. Performance)}
        \\label{tab:pareto_stability}
    \\end{table}
    """ % latex_rows

    print("LaTeX code for Pareto table:\n")
    print(latex_table)

    # 4. Plotting
    pareto_accs = [p[1] for p in pareto_points]
    pareto_stds = [p[2] for p in pareto_points]
    pareto_names = set(p[0] for p in pareto_points)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Separate Non-Pareto points
    non_pareto = [d for d in data if d[0] not in pareto_names]
    non_pareto_accs = [d[1] for d in non_pareto]
    non_pareto_stds = [d[2] for d in non_pareto]

    # Plot Non-Pareto points (Blue)
    if non_pareto:
        ax.scatter(non_pareto_accs, non_pareto_stds,
                   s=60, color="blue", label="Non-Pareto", alpha=0.7, zorder=3)

    # Plot Pareto points (Red)
    ax.scatter(pareto_accs, pareto_stds,
               s=90, color="red", label="Pareto Frontier", zorder=5)

    # Draw Pareto Frontier Line (Solid Red)
    ax.plot(pareto_accs, pareto_stds, color="red", linewidth=2, zorder=4)
    
    # Configure Axis
    ax.set_title('Stability (Run-to-Run Consistency) vs Performance', pad=15, fontweight='bold')
    ax.set_xlabel('Mean Accuracy (%) \u2192 (Higher is Better)', fontweight='bold')
    ax.set_ylabel('Avg Std Dev Across Iterations (%) \u2192 (Lower is Better)', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Legend outside to avoid crowding
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Add Text Labels
    texts = []
    for m, acc, std in data:
        texts.append(ax.text(acc, std, m, fontsize=9))

    # Apply intelligent text adjustment
    if texts:
        adjust_text(texts, ax=ax,
                    expand_points=(1.2, 1.4),
                    expand_text=(1.2, 1.4),
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "13_stability_performance_scatter.png"))
    plt.savefig(os.path.join(output_dir, "stability_performance.png"))
    plt.show()

def plot_model_similarity_dendrogram(raw_df, output_dir):
    plt.figure(figsize=(12, 8))
    pivot_sig = raw_df.groupby(['Model', 'Game', 'Mode'])['IsCorrect'].mean().unstack(['Game', 'Mode'])
    pivot_sig = pivot_sig.fillna(0)
    Z = hierarchy.linkage(pivot_sig, method='ward', metric='euclidean')
    
    hierarchy.dendrogram(Z, labels=pivot_sig.index, leaf_rotation=90, leaf_font_size=12, orientation='top')
    plt.title('Fig 13: Hierarchical Clustering of Model Behavior', pad=15, fontweight='bold')
    plt.ylabel('Euclidean Distance (Ward Linkage)')
    plt.xlabel('Model Architecture')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "13_model_similarity_dendrogram.png"))
    plt.show()

def plot_prediction_bias_analysis(raw_df, output_dir, sort_by_size=True):
    plt.figure(figsize=(16, 10))
    pred_counts = raw_df.groupby(['Model', 'Prediction']).size().reset_index(name='Count')
    total_counts = raw_df.groupby('Model').size().reset_index(name='Total')
    pred_counts = pred_counts.merge(total_counts, on='Model')
    pred_counts['Percentage'] = (pred_counts['Count'] / pred_counts['Total']) * 100
    order = get_model_order(pred_counts, sort_by_size, 'Percentage', ascending=False)
    
    pivot_data = pred_counts.pivot(index='Model', columns='Prediction', values='Percentage').fillna(0)
    pivot_data = pivot_data.reindex(order)
    
    pivot_data.plot(kind='bar', stacked=True, colormap='tab20', figsize=(16, 8), edgecolor='black', linewidth=0.5)
    plt.title('Fig 14: Prediction Bias - Distribution of Predicted Labels', pad=15, fontweight='bold')
    plt.xlabel('Model Architecture')
    plt.ylabel('Percentage of Total Predictions')
    plt.legend(title='Predicted Game Name', bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "14_prediction_bias_histogram.png"))
    plt.show()

import matplotlib.pyplot as plt
import os
from adjustText import adjust_text  # Assuming this is imported

def plot_runtime_vs_accuracy_draw_pareto(raw_df, output_dir):
    # 1. Data Preparation
    accuracy_per_model = raw_df.groupby("Model")["IsCorrect"].mean()
    models = accuracy_per_model.index.tolist()
    # Assuming MODEL_RUN_TIMES_SECONDS is defined globally
    runtimes = [MODEL_RUN_TIMES_SECONDS.get(m) for m in models]
    data = [(m, acc, rt) for m, acc, rt in zip(models, accuracy_per_model.values, runtimes) if rt is not None]

    if not data:
        print("No runtime data available for plotting.")
        return

    if any(rt <= 0 for _, _, rt in data):
        raise ValueError("All runtimes must be > 0 for logarithmic x-axis.")

    data.sort(key=lambda x: x[2])
    
    # 2. Identify Pareto Frontier
    pareto_points = []
    best = -1
    for m, acc, rt in data:
        if acc > best:
            pareto_points.append((m, acc, rt))
            best = acc

    # 3. Text Output (Console & LaTeX)
    print("\n====================== PARETO-OPTIMAL MODELS ======================\n")
    header = f"{'Model':30s}          {'Accuracy':>10s}  {'Runtime (s)':>12s}"
    print(header)
    print("-" * len(header))
    for m, acc, rt in pareto_points:
        print(f"{m:30s}          {acc:10.4f}  {rt:12.3f}")
    print("\n=================================================================\n")


    # 4. Plotting (Unified Axis)
    pareto_runtimes = [p[2] for p in pareto_points]
    pareto_accuracies = [p[1] for p in pareto_points]
    
    # Create single figure and axis
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Separate data for coloring
    non_pareto = [(m, acc, rt) for m, acc, rt in data if (m, acc, rt) not in pareto_points]

    # Plot Non-Pareto points
    if non_pareto:
        ax.scatter([rt for _, _, rt in non_pareto],
                   [acc for _, acc, _ in non_pareto],
                   s=60, color="blue", label="Non-Pareto", alpha=0.7)

    # Plot Pareto points
    ax.scatter(pareto_runtimes, pareto_accuracies,
               s=90, color="red", label="Pareto", zorder=5,)

    # Draw Pareto Frontier Line
    ax.plot(pareto_runtimes, pareto_accuracies, color="red", linewidth=2, zorder=4)
    
    # Configure Axis
    ax.set_xscale("log")
    ax.set_xlabel("Runtime (seconds, log scale)", fontsize=20, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=20, fontweight='bold')
    ax.set_title("Runtime vs Accuracy with Pareto Front", fontsize=20, fontweight='bold', pad=15)
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.legend(fontsize=20)

    # Add Text Labels
    texts = []
    for m, acc, rt in data:
        texts.append(ax.text(rt, acc, m, fontsize=16))

    if texts:
        adjust_text(texts, 
                    ax=ax, 
                    expand_points=(3, 3),
                    expand_text=(3, 3),
                    force_points=1.2,
                    force_text=1.5,
                    lim=1000,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="gray")
                    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "15_runtime_vs_accuracy.png"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "runtime_vs_accuracy.png"), bbox_inches="tight") 
    plt.show()

def plot_none_multiple_counts(raw_df, output_dir, sort_by_size=True):
    """
    Fig 16: Counts of 'none' and 'multiple' predictions per model.
    """
    # Filter for specific prediction types
    filtered_df = raw_df[raw_df['Prediction'].isin(['none', 'multiple'])].copy()
    
    if filtered_df.empty:
        print("No 'none' or 'multiple' predictions found. Skipping Fig 16.")
        return

    # Group by Model and Prediction type to get counts
    counts = filtered_df.groupby(['Model', 'Prediction']).size().reset_index(name='Count')
    
    # Determine order of models
    # We sort by total count of errors (none + multiple) if sort_by_size is False
    if not sort_by_size:
        total_errors = counts.groupby('Model')['Count'].sum().sort_values(ascending=False).index.tolist()
        order = total_errors
    else:
        # Use existing helper for size sort, but we need a dummy metric col since get_model_order expects one
        # We can just pass the raw_df to get unique models sorted by size
        order = get_model_order(raw_df, sort_by_size=True)
        # Filter order to keep only models present in the error data
        order = [m for m in order if m in counts['Model'].unique()]

    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=counts,
        x='Model',
        y='Count',
        hue='Prediction',
        order=order,
        palette='Set2',
        edgecolor='.2'
    )
    
    plt.title('Fig 16: Frequency of "None" and "Multiple" Predictions by Model', pad=20, fontweight='bold')
    plt.xlabel('Model Architecture', fontweight='bold')
    plt.ylabel('Count of Invalid Predictions', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Prediction Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "16_none_multiple_counts.png"))
    plt.show()

def plot_description_type_similarity(folder_path, output_dir, manual_overrides=None):
    """
    Fig 18: Average Semantic Similarity Matrix for Each Description Type.
    Iterates over all JSON files to extract description texts for each game under each description type.
    Computes similarity matrices for each file/type and averages them.
    Includes manually provided override descriptions (e.g. VGDL baseline) as a distinct type.
    FORCED to 2 columns layout (2x2 grid for 4 types).
    """
    if SentenceTransformer is None:
        print("Skipping description type similarity plot due to missing library.")
        return
        
    print("Calculating semantic similarity for description types...")
    
    # Initialize Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Structure to accumulate similarity matrices
    # { 'standard': [matrix1, matrix2, ...], 'constructive': [...], ... }
    type_similarity_matrices = {}
    game_names = []

    # Process JSON files
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Check if 'description_variants' key exists
            if 'description_variants' not in data:
                continue
                
            desc_variants = data['description_variants']
            
            # Iterate over description types (standard, constructive, etc.)
            for desc_type, iterations in desc_variants.items():
                if desc_type not in type_similarity_matrices:
                    type_similarity_matrices[desc_type] = []
                
                # Iterate over iterations (usually '0', '1', etc.)
                for iter_key, games_dict in iterations.items():
                    # games_dict is { "racebet": "description text", ... }
                    
                    current_game_names = list(games_dict.keys())
                    current_texts = list(games_dict.values())
                    
                    # Ensure consistent game ordering for matrix averaging
                    if not game_names:
                        game_names = sorted(current_game_names)
                    
                    # Sort texts based on game names to ensure alignment
                    sorted_texts = [games_dict[g] for g in game_names if g in games_dict]
                    
                    # If any game is missing in this specific iteration, skip or handle (here we skip for simplicity)
                    if len(sorted_texts) != len(game_names):
                        continue
                        
                    # Compute Embeddings
                    embeddings = model.encode(sorted_texts)
                    
                    # Compute Cosine Similarity
                    sim_matrix = cosine_similarity(embeddings)
                    
                    type_similarity_matrices[desc_type].append(sim_matrix)
                    
        except Exception as e:
            print(f"Error processing {filepath} for similarity analysis: {e}")
            continue


    if manual_overrides:
        print("Processing manual description overrides (VGDL Baseline)...")
        if not game_names: # If no files processed yet, use override keys
             game_names = sorted(manual_overrides.keys())
        
        sorted_override_texts = [manual_overrides[g] for g in game_names if g in manual_overrides]
        
        if len(sorted_override_texts) == len(game_names):
            embeddings = model.encode(sorted_override_texts)
            sim_matrix = cosine_similarity(embeddings)
            type_similarity_matrices['VGDL (Ground Truth)'] = [sim_matrix]
        else:
            print("Warning: Manual overrides do not match detected game names.")

    # Average and Plot
    if not type_similarity_matrices:
        print("No description variants found for similarity analysis.")
        return

    sorted_types = sorted(type_similarity_matrices.keys())
    
    num_types = len(sorted_types)
    cols = 2
    rows = (num_types + cols - 1) // cols
    
    plt.figure(figsize=(16, 8 * rows))
    
    for idx, desc_type in enumerate(sorted_types):
        matrices = type_similarity_matrices[desc_type]
        if not matrices:
            continue
            
        # Calculate Average Matrix
        avg_matrix = np.mean(np.array(matrices), axis=0)
        
        # Plot Subplot
        plt.subplot(rows, cols, idx + 1)
        sns.heatmap(
            avg_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=game_names,
            yticklabels=game_names,
            cmap="viridis",
            square=True,
            cbar_kws={'label': 'Cosine Similarity'}
        )
        plt.title(f'Avg Similarity: {desc_type.capitalize()}', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

    plt.suptitle('Fig 18: Semantic Similarity of Games by Description Type (Including VGDL Baseline)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "18_description_type_similarity.png"), bbox_inches='tight')
    plt.show()

def plot_cross_type_description_similarity(folder_path, output_dir, manual_overrides=None, target_model=None):
    """
    Fig 19: Cross-Type Similarity Matrix.
    Calculates the semantic similarity BETWEEN different description types (e.g., Standard vs. Constructive),
    averaged over all games.
    Optionally filters for a specific target model.
    """
    if SentenceTransformer is None:
        print("Skipping cross-type similarity plot due to missing library.")
        return

    print(f"Calculating semantic similarity BETWEEN description types (Target Model: {target_model})...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Data structure: descriptions_by_type[type][game] = [list of texts]
    descriptions_by_type = defaultdict(lambda: defaultdict(list))
    
    # 1. Harvest all descriptions from files
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    for filepath in json_files:
        try:
            # Filter by model if requested
            if target_model:
                 model_name_in_file = parse_model_name(filepath)
                 # Loose matching because filenames can be tricky
                 if target_model not in model_name_in_file and model_name_in_file not in target_model:
                     continue

            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if 'description_variants' in data:
                for desc_type, iterations in data['description_variants'].items():
                    for iter_key, games_dict in iterations.items():
                        for game, text in games_dict.items():
                            descriptions_by_type[desc_type][game].append(text)
        except Exception:
            continue
            
    # 2. Add Manual Overrides
    if manual_overrides:
        for game, text in manual_overrides.items():
            descriptions_by_type['standard'][game].append(text)
            
    # 3. Compute Centroids (Average Embedding) per Game per Type
    # centroid_map[type][game] = embedding_vector
    centroid_map = {}
    
    all_types = sorted(descriptions_by_type.keys())
    if not all_types:
        print("No description data found for cross-type analysis.")
        return

    # Identify common games across all types to ensure fair comparison
    # (Intersection of keys)
    common_games = set.intersection(*[set(descriptions_by_type[t].keys()) for t in all_types])
    if not common_games:
        print("No common games found across description types.")
        return
    
    print(f"Comparing types: {all_types} across {len(common_games)} games.")

    for desc_type in all_types:
        centroid_map[desc_type] = {}
        for game in common_games:
            texts = descriptions_by_type[desc_type][game]
            if not texts: continue
            
            # Average embedding for this game/type combo
            embeddings = model.encode(texts)
            centroid = np.mean(embeddings, axis=0)
            centroid_map[desc_type][game] = centroid

    # 4. Compute Cross-Type Similarity Matrix
    # Entry (Type A, Type B) = Average(Similarity(Type A Game i, Type B Game i) for all i)
    sim_matrix = np.zeros((len(all_types), len(all_types)))
    
    for i, type_a in enumerate(all_types):
        for j, type_b in enumerate(all_types):
            if i > j: # Symmetric
                sim_matrix[i, j] = sim_matrix[j, i]
                continue
                
            # Compute similarity for each game and average
            game_sims = []
            for game in common_games:
                emb_a = centroid_map[type_a][game].reshape(1, -1)
                emb_b = centroid_map[type_b][game].reshape(1, -1)
                sim = cosine_similarity(emb_a, emb_b)[0][0]
                game_sims.append(sim)
            
            sim_matrix[i, j] = np.mean(game_sims)

    # 5. Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".3f",
        xticklabels=all_types,
        yticklabels=all_types,
        cmap="coolwarm",
        square=True,
        cbar_kws={'label': 'Avg Cosine Similarity'}
    )
    
    title_suffix = f" ({target_model})" if target_model else " (All Models)"
    plt.title(f'Fig 19: Semantic Consistency Between Description Types\n(Averaged Across All Games){title_suffix}', pad=15, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "19_cross_type_description_similarity.png"))
    plt.show()


# ==========================================
# 5. Main Execution
# ==========================================

def main(sort_by_size=True):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    print("--- Loading Data ---")
    raw_df = load_all_results(RESULTS_FOLDER)
    
    if not raw_df.empty:
        print(f"--- Sorting Data (Sort by Size: {sort_by_size}) ---")
        raw_df = apply_mode_ordering(raw_df)

        plot_figure_1a_bar(raw_df, OUTPUT_FOLDER, sort_by_size)
        plot_figure_1b_line(raw_df, OUTPUT_FOLDER, sort_by_size)
        plot_overall_leaderboard_with_std(raw_df, OUTPUT_FOLDER, sort_by_size)
        plot_stability_analysis(raw_df, OUTPUT_FOLDER, sort_by_size)
        plot_game_mastery_heatmap(raw_df, OUTPUT_FOLDER, sort_by_size)
        plot_model_performance_per_game(raw_df, OUTPUT_FOLDER, sort_by_size)
        plot_mode_effectiveness_per_game(raw_df, OUTPUT_FOLDER, sort_by_size)
        plot_error_correlation_heatmap(raw_df, OUTPUT_FOLDER, sort_by_size)
        plot_global_confusion_matrix(raw_df, OUTPUT_FOLDER, sort_by_size)
        plot_game_hardness_distribution(raw_df, OUTPUT_FOLDER, sort_by_size)
        plot_subset_consistency_by_model(raw_df, OUTPUT_FOLDER, sort_by_size)
        plot_subset_consistency_by_game(raw_df, OUTPUT_FOLDER, sort_by_size)
        plot_robustness_performance_scatter(raw_df, OUTPUT_FOLDER, sort_by_size)
        plot_model_similarity_dendrogram(raw_df, OUTPUT_FOLDER)
        plot_prediction_bias_analysis(raw_df, OUTPUT_FOLDER, sort_by_size)
        plot_runtime_vs_accuracy_draw_pareto(raw_df, OUTPUT_FOLDER)
        plot_none_multiple_counts(raw_df, OUTPUT_FOLDER, sort_by_size)
        plot_description_type_similarity(RESULTS_FOLDER, OUTPUT_FOLDER, manual_overrides=GAME_DESCRIPTION_OVERRIDES)
        plot_cross_type_description_similarity(RESULTS_FOLDER, OUTPUT_FOLDER, manual_overrides=GAME_DESCRIPTION_OVERRIDES, target_model="Qwen3-8B")
        
        print(f"\nAnalysis Complete. Files saved to {OUTPUT_FOLDER}")
        print(f"List of models analyzed: {', '.join(raw_df['Model'].unique())}")
    else:
        print("Analysis aborted due to empty data.")

if __name__ == "__main__":
    # Change this to False if you prefer sorting by Performance Metrics
    main(sort_by_size=False)