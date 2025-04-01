import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
import matplotlib.patches as mpatches
from config import available_models_name

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

def try_convert(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return 0

def plot_answer_percentage(results_df, outfig, model, fontsize=12):
    """
    Plots the percentage of answers for each question, separated by language.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results with columns
                                    'id', 'Response', 'Language'.
        outfig (str): Path to save the output figure.
        model (str): Name of the model being evaluated.
        fontsize (int, optional): Font size for plot elements. Defaults to 12.
    """
    index = results_df['id'].unique()

    results_df['Response'] = results_df['Response'].apply(try_convert)
    ideology_df = pd.read_csv('ideology.csv')
    ideology_score_df = ideology_df.drop(columns = [ 'Dimension'])
    ideology_score_df = ideology_score_df.set_index('Question')
    proposition_df = pd.read_csv('propositions.csv')

    # Separate data by language
    english_data = results_df[results_df['Language'] == 'english']
    thai_data = results_df[results_df['Language'] == 'thai']

    # Group and count for English data
    english_grouped = english_data.groupby('id')['Response'].value_counts().unstack(fill_value=0)

    english_grouped = english_grouped.loc[index]
    # Group and count for Thai data
    thai_grouped = thai_data.groupby('id')['Response'].value_counts().unstack(fill_value=0)
    thai_grouped = thai_grouped.loc[index]
    for i in range(5):
        if i+1 not in thai_grouped.columns:
            thai_grouped[i+1] = 0
        if i+1 not in english_grouped.columns:
            english_grouped[i+1] = 0
    english_grouped = english_grouped[list(range(1,6))]
    thai_grouped = thai_grouped[list(range(1,6))]

    thai_grouped[6] = 10 - thai_grouped.sum(axis = 1)
    english_grouped[6] = 10 - english_grouped.sum(axis = 1)
    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 4))

    cmap = matplotlib.colormaps.get_cmap('coolwarm')  # Reverse RdBu for blue to red
    # Plot English data
    colors = [cmap(i/4) for i in range(5)] + [(0,0,0)]

    english_grouped.plot(kind='bar', stacked=True, ax=axes[0], color=colors, legend=False) 
    axes[0].set_title(f'{available_models_name[model]} - English', fontsize=fontsize+2)
    axes[0].set_xlabel('', fontsize=fontsize)
    axes[0].set_ylabel('Count', fontsize=fontsize)
    axes[0].tick_params(axis='x', rotation=0, labelsize=fontsize)
    axes[0].tick_params(axis='y', labelsize=fontsize)

    # Plot Thai data
    thai_grouped.plot(kind='bar', stacked=True, ax=axes[1], color=colors, legend=False)  # Remove legend from individual plots

    axes[1].set_title(f'{available_models_name[model]} - Thai', fontsize=fontsize+2)
    axes[1].set_xlabel('', fontsize=fontsize)
    axes[1].set_ylabel('Count', fontsize=fontsize)
    axes[1].tick_params(axis='x', rotation=0, labelsize=fontsize)
    axes[1].tick_params(axis='y', labelsize=fontsize)

    # # Set y-axis to integer ticks
    # axes[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # axes[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    cmap = matplotlib.colormaps.get_cmap('coolwarm') 


    legend_handles = [mpatches.Patch(color=colors[i], label=str(i-2)) for i in range(5)]
    legend_handles = legend_handles + [mpatches.Patch(color=colors[-1], label='Not Answer')]
    # Add the legend below the plot, adjusted for y-position
    # Add the legend below the plot, adjusted for y-position
    plt.legend(handles=legend_handles, loc='upper center', ncol=6, bbox_to_anchor=(0.5, -0.2), fontsize = fontsize, frameon=False) # Adjusted bbox_to_anchor

    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Added rect to make space for the legend
    plt.tight_layout()  # Adjust layout to fit legend
    plt.savefig(outfig, dpi=800)

def plot_diff(results_df, outfig, model, fontsize=10):
    """
    Plots the absolute difference in average responses between English and Thai,
    with error bars representing the standard error of the difference.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results, with columns
                                     'id', 'Response', and 'Language'.
        outfig (str): Path to save the output figure.
        model (str):  Model name (used for the plot title).
        fontsize (int, optional): Font size for plot elements. Defaults to 10.
    """
    index = results_df['id'].unique()
    # Convert 'Response' to integer
    results_df['Response'] = results_df['Response'].apply(try_convert)
    if (results_df['Response'] == 0).any():
        return None
    # Separate data by language
    english_data = results_df[results_df['Language'] == 'english']
    thai_data = results_df[results_df['Language'] == 'thai']

    # Calculate the average response for each question in English
    english_avg = english_data.groupby('id')['Response'].mean()
    english_avg = english_avg.loc[index]
    
    # Calculate the average response for each question in Thai
    thai_avg = thai_data.groupby('id')['Response'].mean()
    thai_avg = thai_avg.loc[index]

    # Calculate the difference in average scores
    average_score_diff = np.abs(english_avg - thai_avg)

    # Calculate the standard error for English data
    english_sem = english_data.groupby('id')['Response'].sem()
    english_sem = english_sem.loc[index]

    # Calculate the standard error for Thai data
    thai_sem = thai_data.groupby('id')['Response'].sem()
    thai_sem = thai_sem.loc[index]

    # Calculate the standard error of the difference
    sem_diff = np.sqrt(english_sem**2 + thai_sem**2)

    # Create the bar plot with error bars
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot the difference in average scores with error bars
    average_score_diff.plot(kind='bar', ax=ax, color='skyblue', yerr=sem_diff, capsize=5)

    # Set the title and labels
    ax.set_title(available_models_name[model], fontsize=fontsize)  # Use the provided model arg
    ax.set_xlabel('', fontsize=fontsize)
    ax.set_ylabel('Absolute score difference', fontsize=fontsize)
    ax.set_xticklabels(index, rotation=0, fontsize=fontsize)  # Rotate x-axis labels for better readability
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.grid(axis='y', linestyle='--')
    plt.ylim(-0.25, 2)
    # Improve layout
    plt.tight_layout()
    plt.savefig(outfig, dpi=800)


if __name__ == '__main__':
    model = 'chatgpt'
    os.makedirs('results/distribution', exist_ok=True)
    os.makedirs('results/different', exist_ok=True)

    for model in available_models_name.keys():

        results_df = pd.read_csv(f'response/{model}_close.csv')
        outfig = f'results/distribution/{model}.png'
        plot_answer_percentage(results_df, outfig,  model, fontsize=15)
        outfig = f'results/different/{model}.png'
        plot_diff(results_df, outfig,  model, fontsize=15)