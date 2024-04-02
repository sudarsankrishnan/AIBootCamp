
import os
from openai import AzureOpenAI
 
azure_api_key = "168dfde591cd421ebd00ab8870defb3f"
azure_base_url = "https://usbank-bootcamp-2.openai.azure.com/"
deployment_name = 'gpt-35-turbo-instruct'
 
client = AzureOpenAI(
  api_key = azure_api_key,
  api_version = "2023-05-15",
  azure_endpoint = azure_base_url
)
 
# Send a completion call to generate an answer
print('USBank GenAI BootCamp Day2')
system_prompt = """You are a helpful and honest assistant. Always answer as helpfully as possible, while being safe. \
                    Please ensure your responses are socially unbiased and positive in nature. If you don't know  \
                    the answer to a question, please don't share false information. """
                    
user_prompt = input('User query: ')
 
if deployment_name == 'gpt-35-turbo-instruct':
    response = client.completions.create(model=deployment_name,
                                          prompt=system_prompt + user_prompt)
    text = response.choices[0].text.replace('\n', '').replace(' .', '.').strip()
    
else:
    response = client.chat.completions.create(model=deployment_name,
                                              messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                  {
                    "role": "user",
                    "content": user_prompt
                }
            ],)
    text = response.choices[0].message.content.replace('\n', '').replace(' .', '.').strip()
 
print("AI assistant response: ", text)