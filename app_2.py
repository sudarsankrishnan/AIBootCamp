
import os, json
from openai import AzureOpenAI
 
azure_api_key = "168dfde591cd421ebd00ab8870defb3f"
azure_base_url = "https://usbank-bootcamp-2.openai.azure.com/"
deployment_name = 'gpt-35-turbo-instruct'
 
client = AzureOpenAI(
  api_key = azure_api_key,
  api_version = "2023-05-15",
  azure_endpoint = azure_base_url
)
 
print('USBank GenAI Bootcamp Day2')
system_prompt = """
Your name is Mike (a Customer Service Agent) and you work for USBank.
Please ensure your respones are to the point and doesn't contain any extrenous details
only provide response to the user's latest request and do not provide responses to your own responses
"""
 
qna = [
  {"system": system_prompt}
]
 
def get_token_count(text:str):
  return len(text.split())
 
def do_qna(user_query:str):
  """
  system_prompt: str: The system prompt to be used for the completion
  user_query: str: The user query to be used for the completion
  qna_so_far: str: The qna so far to be used for the completion
  """
 
  qna_as_text = ""
  for key, value in enumerate(qna):
    qna_as_text += f"{key}: {value}\n\n"
 
  prompt = qna_as_text+"\n\n---\n\nUser query:"+user_query
  length = get_token_count(prompt)
 
  response = client.completions.create(model=deployment_name,
                                       max_tokens=8192,
                                       temperature=0.99,
                                       seed=42,
                                       prompt=prompt)
  
  text = response.choices[0].text
  qna.append({"user": user_query})
  qna.append({"system": text})
  return text, length
 
user_query = "start"
 
while user_query != 'exit':
  user_query = input("User query: ")
  text, length = do_qna(user_query)
  print(f"\nContext Length: {length}\nResponse: {text}")
 
 