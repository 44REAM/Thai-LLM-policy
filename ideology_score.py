import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from config import available_models_name

ideology_df = pd.read_csv('ideology.csv')
ideology_score_df = ideology_df.drop(columns = [ 'Dimension'])
ideology_score_df = ideology_score_df.set_index('Question')
proposition_df = pd.read_csv('propositions.csv')


def get_ideo_distance(score):
    # Separate economic and political questions
    economic_questions = ideology_df[ideology_df['Dimension'] == 'Economic']['Question'].tolist()
    political_questions = ideology_df[ideology_df['Dimension'] == 'Political']['Question'].tolist()
    social_questions = ideology_df[ideology_df['Dimension'] == 'Social']['Question'].tolist()
    instu_questions = ideology_df[ideology_df['Dimension'] == 'Institutional']['Question'].tolist()

    # Calculate Manhattan distance for each dimension
    def calculate_manhattan_distance(row, ideology_col, questions):
        distance = 0
        for question in questions:
            distance += 4-abs(row[question] - ideology_col[ideology_df['Question'] == question].values[0])
        return distance

    results = {}
    for col in ideology_df.columns[2:]:  # Iterate over ideology columns (Fascist, Cons, etc.)
        results[col] = {}
        for dimension in ['Economic', 'Political', 'Social', 'Institutional']:
            if dimension == 'Economic':
                questions = economic_questions
            elif dimension == 'Political':
                questions = political_questions
            elif dimension == 'Social':
                questions = social_questions
            elif dimension == 'Institutional':
                questions = instu_questions
            
            distances = []
            for index, row in score.iterrows():
                distances.append(calculate_manhattan_distance(row, ideology_df[col], questions))
            results[col][dimension] = np.array(distances)

    # Convert results to a DataFrame for better visualization
    result_df = pd.DataFrame({
        col: {
            'Economic': results[col]['Economic'],
            'Political': results[col]['Political'],
            'Social': results[col]['Social'],
            'Institutional': results[col]['Institutional']
        }
        for col in ideology_df.columns[2:]
    })

    return result_df
def get_percentage(distance_df):
    weight = np.array([0.35, 0.25,0.15,0.25]).reshape((-1,1))

    ideo_results = {}
    for ideo_type in distance_df.columns:
        cal = []
        for array in distance_df[ideo_type].to_numpy():
            cal.append(array)
        cal = np.array(cal)
        cal = cal*weight
        cal = cal.sum(axis = 0)/20*100
        ideo_results[ideo_type] = cal
    ideo_results = pd.DataFrame(ideo_results)
    return ideo_results

def plot(df, outfig, model_name):

    # Melt the DataFrame to long format for Seaborn
    df_melted = pd.melt(df, var_name='Ideology', value_name='Percentage')


    # Set a visually appealing color palette
    palette = sns.color_palette("viridis", n_colors=len(df.columns))

    # Create the bar chart with customizations
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Ideology', y='Percentage', data=df_melted, estimator='mean',
                    hue='Ideology', palette=palette, edgecolor='black', linewidth=1, legend=False)  # Corrected line

    # Add gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Improve title and labels
    plt.title(available_models_name[model_name], fontsize=25, fontweight='bold')
    plt.xlabel('Ideological Categories', fontsize=20)
    plt.ylabel('Average Percentage', fontsize=20)
    plt.ylim(0, 100)
    

    # Customize x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=20)

    # Customize y-axis labels
    plt.yticks(fontsize=20)

    # Improve layout
    plt.tight_layout()

    # Show the plot
    plt.savefig(outfig, dpi=800)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot(df_thai, df_eng, outfig, model_name, 
         title_fontsize=18, label_fontsize=18, tick_fontsize=18):
    # Melt the DataFrames to long format for Seaborn
    df_thai_melted = pd.melt(df_thai, var_name='Ideology', value_name='Percentage')
    df_eng_melted = pd.melt(df_eng, var_name='Ideology', value_name='Percentage')
    
    # Set a visually appealing color palette
    palette = sns.color_palette("viridis", n_colors=len(df_thai.columns))
    
    # Create a figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))
    
    # Thai Language Plot
    bars1 = sns.barplot(x='Ideology', y='Percentage', data=df_thai_melted, 
                         estimator='mean', hue='Ideology', palette=palette, 
                         edgecolor='black', linewidth=1, legend=False, ax=ax1)
    
    # Adding labels above each Thai bar
    for bar in ax1.patches:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,   # center of the bar
            height + 1,                          # a little above the bar
            f'{height:.1f}%',                    # format to one decimal (adjust if needed)
            ha='center', va='bottom', fontsize=tick_fontsize
        )
    
    # Make sure available_models_name exists in the context. For example:
    # available_models_name = {'model1': 'Model One', 'model2': 'Model Two'}
    ax1.set_title(f'{available_models_name[model_name]} - Thai', fontsize=title_fontsize, fontweight='bold')
    ax1.set_xlabel('', fontsize=label_fontsize)
    ax1.set_ylabel('Similarity (%)', fontsize=label_fontsize)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
    ax1.tick_params(axis='y', labelsize=tick_fontsize)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # English Language Plot
    bars2 = sns.barplot(x='Ideology', y='Percentage', data=df_eng_melted, 
                         estimator='mean', hue='Ideology', palette=palette, 
                         edgecolor='black', linewidth=1, legend=False, ax=ax2)
    
    # Adding labels above each English bar
    for bar in ax2.patches:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=tick_fontsize
        )
    
    ax2.set_title(f'{available_models_name[model_name]} - English', fontsize=title_fontsize, fontweight='bold')
    ax2.set_xlabel('', fontsize=label_fontsize)
    ax2.set_ylabel('Similarity (%)', fontsize=label_fontsize)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
    ax2.tick_params(axis='y', labelsize=tick_fontsize)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Improve overall layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(outfig, dpi=800)


if __name__ == '__main__':
    os.makedirs('results/ideology', exist_ok=True)
    samples = 10
    for model in available_models_name.keys():
        results_df = pd.read_csv(f'{model}_close.csv')
        print(model)
        thai_score = {}
        english_score = {}
        for id in results_df['id'].unique():
            tmpdf = results_df[results_df['id'] == id]

            tmpdf_thai = tmpdf[tmpdf['Language'] == 'thai']
            tmpdf_eng = tmpdf[tmpdf['Language'] == 'english']
            sample_thai  = tmpdf_thai.sample(n=samples, replace=True)['Response']
            sample_eng  = tmpdf_eng.sample(n=samples, replace=True)['Response']

            int_sample_thai = []
            for response in sample_thai:

                int_sample_thai.append(int(response)-3)

            int_sample_eng = []
            for response in sample_eng:

                int_sample_eng.append(int(response)-3)

            thai_score[id] = np.array(int_sample_thai)
            english_score[id] = np.array(int_sample_eng)

        thai_score = pd.DataFrame(thai_score)
        english_score = pd.DataFrame(english_score)

        
        distance_thai = get_ideo_distance(thai_score)
        distance_eng = get_ideo_distance(english_score)
        
        ideo_percent_thai = get_percentage(distance_thai)
        ideo_percent_eng = get_percentage(distance_eng)

        plot(ideo_percent_thai, ideo_percent_eng, f'results/ideology/{model}.png', model)
