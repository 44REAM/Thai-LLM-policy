from question_score import *
from stacking import *
from ideology_score import *

model = 'chatgpt'
os.makedirs('results/distribution', exist_ok=True)
os.makedirs('results/different', exist_ok=True)

for model in available_models_name.keys():

    results_df = pd.read_csv(f'response/{model}_close.csv')
    outfig = f'results/distribution/{model}.png'
    plot_answer_percentage(results_df, outfig,  model, fontsize=15)
    outfig = f'results/different/{model}.png'
    plot_diff(results_df, outfig,  model, fontsize=15)

folder_path = 'results/distribution'

outpath = f'results/distribution.png'
stack_images_grid(folder_path, dpi = (800,800), layout = (2,2), output_path = outpath)

folder_path = 'results/different'
outpath = f'results/different.png'
stack_images_grid(folder_path, dpi = (800,800), layout = (2,2), output_path = outpath)

os.makedirs('results/ideology', exist_ok=True)

###################### ideology

samples = 10
for model in available_models_name.keys():
    results_df = pd.read_csv(f'response/{model}_close.csv')
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

folder_path = 'results/ideology'
outpath = f'results/ideology.png'
stack_images_grid(folder_path, dpi = (800,800), layout = (2,2), output_path = outpath)
