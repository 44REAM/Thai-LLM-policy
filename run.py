import pandas as pd
import anthropic
from google import genai
from google.genai import types
import os
from openai import OpenAI
import time
from config import available_models


def send_prompt_to_models(prompt, language, model, max_attempts = 10):
    """
    Send prompt to different AI models.
    
    :param prompt: Formatted prompt
    :param language: Language of the prompt
    :param model: Specific model to use (optional)
    :return: Responses from selected models
    """
    # Define models for different languages

    if 'gemini' in model:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    elif 'claude' in model:
        client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    elif 'chatgpt' in model:
        client = OpenAI()
    elif 'deepseek' in model:
        client = OpenAI(api_key="", base_url="https://api.deepseek.com")
    for attempt in range(max_attempts):
        try:

            if model not in available_models:
                raise ValueError(f"Model {model} not found for language {language}")
            
            if 'gemini' in model:
                client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
                response = client.models.generate_content(
                    model=available_models[model],
                    config=types.GenerateContentConfig(
                        max_output_tokens=8000,
                        temperature=1
                    ),
                    contents=prompt,
                )
                return response.text

            elif 'claude' in model:
                client = anthropic.Anthropic()
                response = client.messages.create(
                    model=available_models[model],
                    max_tokens=8000,
                    temperature=1,
                    messages=[{"content": prompt, "role": "user"}]
                )

                return response.content[0].text

            elif 'chatgpt' in model:
                response = client.chat.completions.create(
                    model=available_models[model],
                    temperature=1,
                    max_tokens=8000,
                    messages=[{"content": prompt, "role": "user"}]
                )

                return response.choices[0].message.content
            elif 'deepseek' in model:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    temperature=1,
                    max_tokens=8000,
                    messages=[{"content": prompt, "role": "user"}]
                )
                return response.choices[0].message.content

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            sleep_time = 20
            print(f"Waiting {sleep_time:.2f} seconds before retrying...")
            time.sleep(sleep_time)
    return None

# CSV file path
csv_file_path = 'prompt_template.csv'
csv_question_path = 'propositions.csv'

os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
# Read prompts from CSV using pandas
prompts_df = pd.read_csv(csv_file_path)
propositions_df = pd.read_csv(csv_question_path)


model = 'deepseek_chat'
open_or_close = 'close'

output_filename = f'{model}_{open_or_close}.csv'

already_done = []
results = []
if os.path.exists(output_filename):
    previous_results_df = pd.read_csv(output_filename)
    already_done = previous_results_df['Original_Question'].unique()
    results = previous_results_df.to_dict(orient='records')
repeat = 10

# Process each prompt
for _, proposition_data in propositions_df.iterrows():
        # Construct prompts for both languages
    if proposition_data['question'] in already_done:
        continue

    print(proposition_data['question'])
    for _, prompt_data in prompts_df.iterrows():
        if prompt_data['type'] != open_or_close:
            continue
        language = prompt_data['language'].lower()
        type_key = proposition_data['type']
        if prompt_data['language'].lower() == proposition_data['language'].lower():
            # Replace proposition in prompt
            prompt =  prompt_data['prompt'].format(proposition=proposition_data['question'])
            
            for round in range(repeat):
                print(f"Round {round + 1} for {proposition_data['question']}")
                model_responses = send_prompt_to_models(prompt, language, model)
                # print(model_responses)
                # Prepare result row    
                result = {
                    'Original_Question': proposition_data['question'],
                    'id': proposition_data['id'],
                    'Type': type_key,
                    'Open': prompt_data['type'],
                    'Language': language,
                    'Prompt': prompt,
                    'Attempt': round,
                    'Response': model_responses
                }
                
                # Add result to list
                results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')