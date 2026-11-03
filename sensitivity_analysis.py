import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_monte_carlo_sensitivity(n_iter=1000, seed=42, output_dir='results/sensitivity'):
    """
    Monte Carlo Sensitivity Analysis:
    Varies dimension weights uniformly between 0.15 and 0.35, then normalizes each vector to sum to 1.0.
    Evaluates ideological similarity scores and rankings across all compliant models and languages.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)

    # 1. Load ideological template definitions
    ideology_df = pd.read_csv('ideology.csv')
    dimensions = ['Economic', 'Political', 'Social', 'Institutional']
    dim_questions = {
        dim: ideology_df[ideology_df['Dimension'] == dim]['Question'].tolist()
        for dim in dimensions
    }
    ideo_cols = [c for c in ideology_df.columns if c not in ['Question', 'Dimension']]

    def calc_distance(row, ideo_col, qs):
        return sum(4 - abs(row[q] - ideology_df[ideo_col][ideology_df['Question'] == q].values[0]) for q in qs)

    # 2. Sample N_ITER weight vectors uniformly in [0.15, 0.35], then normalize
    raw_weights = np.random.uniform(0.15, 0.35, size=(n_iter, 4))
    norm_weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)

    models = {
        'chatgpt': 'GPT-4o',
        'gemini': 'Gemini 2.0 Flash',
        'gemini_exp': 'Gemini 2.0 Flash Exp',
        'claude_haiku': 'Claude 3.5 Haiku'
    }

    summary_records = []
    plot_records = []

    for m_key, m_name in models.items():
        csv_file = f'response/{m_key}_close.csv'
        if not os.path.exists(csv_file):
            print(f"Skipping {csv_file} (not found)")
            continue
        
        df = pd.read_csv(csv_file)
        
        for lang in ['english', 'thai']:
            sub = df[df['Language'].str.lower() == lang.lower()]
            piv = sub.pivot_table(index='Attempt', columns='id', values='Response') - 3

            # Calculate raw dimension scores: shape (4 dimensions, 7 ideologies, num_attempts)
            dim_scores = np.zeros((4, len(ideo_cols), len(piv)))
            for d_idx, dim in enumerate(dimensions):
                qs = dim_questions[dim]
                for i_idx, icol in enumerate(ideo_cols):
                    scores = [calc_distance(row, icol, qs) for _, row in piv.iterrows()]
                    dim_scores[d_idx, i_idx, :] = scores
            
            # Average across repeated trials
            dim_scores_mean = dim_scores.mean(axis=2) # (4, 7)

            # Similarity matrix: (n_iter, 7)
            sim_matrix = (np.dot(norm_weights, dim_scores_mean) / 20.0) * 100.0

            # Compute ranks: 1 = highest similarity, 7 = lowest
            ranks = 7 - np.argsort(np.argsort(sim_matrix, axis=1), axis=1)

            for i_idx, icol in enumerate(ideo_cols):
                sims = sim_matrix[:, i_idx]
                r = ranks[:, i_idx]

                summary_records.append({
                    'Model': m_name,
                    'Language': lang.capitalize(),
                    'Ideology': icol,
                    'Mean_Similarity': np.round(np.mean(sims), 2),
                    'SD_Similarity': np.round(np.std(sims), 2),
                    'Min_Similarity': np.round(np.min(sims), 2),
                    'Max_Similarity': np.round(np.max(sims), 2),
                    'Pct_Rank_1': np.round(np.mean(r == 1) * 100.0, 1),
                    'Pct_Top_2': np.round(np.mean(r <= 2) * 100.0, 1),
                    'Pct_Rank_7_Last': np.round(np.mean(r == 7) * 100.0, 1),
                })

                for sim_val in sims:
                    plot_records.append({
                        'Model': m_name,
                        'Language': lang.capitalize(),
                        'Ideology': icol,
                        'Similarity': sim_val
                    })

    # Save summary table
    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.abspath(os.path.join(output_dir, 'sensitivity_summary.csv'))
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table saved to: {summary_path}")

    # Plot Monte Carlo Boxplots with LARGE, PUBLICATION-READY FONTS
    plot_df = pd.DataFrame(plot_records)
    
    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(20, 13), sharey=True)
    axes = axes.flatten()
    
    model_names = list(models.values())
    palette = {'English': '#2b5c8f', 'Thai': '#d95f02'}

    for idx, m_name in enumerate(model_names):
        ax = axes[idx]
        sub_data = plot_df[plot_df['Model'] == m_name]
        
        sns.boxplot(
            data=sub_data,
            x='Ideology',
            y='Similarity',
            hue='Language',
            palette=palette,
            ax=ax,
            width=0.68,
            linewidth=1.5,
            fliersize=2.5,
            boxprops=dict(edgecolor='black'),
            whiskerprops=dict(color='black', linewidth=1.3),
            capprops=dict(color='black', linewidth=1.3),
            medianprops=dict(color='black', linewidth=1.8)
        )
        
        ax.set_title(m_name, fontsize=22, fontweight='bold', pad=14)
        ax.set_xlabel('Ideological Template', fontsize=17, fontweight='bold', labelpad=10)
        
        if idx % 2 == 0:
            ax.set_ylabel('Similarity Percentage (%)', fontsize=17, fontweight='bold', labelpad=10)
        else:
            ax.set_ylabel('')
            
        ax.set_ylim(25, 95)
        ax.tick_params(axis='x', labelsize=15, rotation=25)
        ax.tick_params(axis='y', labelsize=15)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # Legend configuration
        if idx == 0:
            ax.legend(title='Language', title_fontsize=16, fontsize=15, loc='upper right', frameon=True, edgecolor='#666666')
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    plt.tight_layout()
    fig_path = os.path.abspath(os.path.join(output_dir, 'sensitivity_monte_carlo.png'))
    plt.savefig(fig_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"High-resolution plot with large fonts saved to: {fig_path}")

    return summary_df

if __name__ == '__main__':
    run_monte_carlo_sensitivity()