
import os, json
import shutil
from openai import AzureOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.html import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.embeddings import AzureOpenAIEmbeddings
import streamlit as st
 
#azure_api_key = "xxxxxxxxxxxx"
#azure_base_url = "https:xxxxxxxxxx.openai.azure.com/"
deployment_name = 'gpt-35-turbo-instruct'
 
os.environ['AZURE_OPENAI_API_KEY'] = azure_api_key
 
client = AzureOpenAI(
  api_key = azure_api_key,
  api_version = "2023-05-15",
  azure_endpoint = azure_base_url
)
 
# embedding = AzureOpenAIEmbeddings(
#   client=client,
#   model = deployment_name,
#   base_url = azure_base_url
# )
 
db = None
 
print('USBank GenAI Bootcamp Day2')
system_prompt = """
Your name is Mike (a Customer Service Agent) and you work for USBank.
Please ensure your respones are to the point and doesn't contain any extrenous details
only provide response to the user's latest request and do not provide responses to your own responses
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.
"""
 
qna = [
  {"system": system_prompt}
]
 
def write_to_vector_store(md_file_name:str):
  """
  data: str: The data to be written to the vector store
  """


  chunk_size = 2000
  embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
 
  loader = UnstructuredMarkdownLoader(md_file_name)
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=10)
  texts = text_splitter.split_documents(loader.load())
 
  print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
  db = Chroma.from_documents(texts, embedding_function, persist_directory="./vector_store")
 
  return db
 
def get_token_count(text:str):
  return len(text.split())
 
def do_qna(user_query:str,db):
  """
  user_query: str: The user query to be used for the completion
  """
  qna_as_text = ""
 
  for key, value in enumerate(qna):
    qna_as_text += f"{key}: {value}\n\n"
 
 
  # Fetch similar context from the vector store based on user_query
  similar_contexts = db.similarity_search_with_score(user_query)
 
  if similar_contexts:
      context = similar_contexts[0]  # Choosing the most similar context
      print(f"Found similar context: {context}")
     
      # Add the context to the prompt
      system_prompt_with_context = qna_as_text + "\n\n---\n\n" + context[0].page_content
      prompt = system_prompt_with_context + "\n\n---\n\nUser query:" + user_query
  else:
      # If no similar context found, use only the system prompt
      prompt = qna_as_text + "\n\n---\n\nUser query:" + user_query
 
  #prompt = qna_as_text+"\n\n---\n\nUser query:"+user_query
  length = get_token_count(prompt)
 
  response = client.completions.create(model=deployment_name,
                                       max_tokens=8192,
                                       temperature=0,
                                       seed=42,
                                       prompt=prompt)
 
  text = response.choices[0].text
  qna.append({"user": user_query})
  qna.append({"system": text})
  return text, length
 
 
def begin_qna(db):
  user_query = "start"
 
  while user_query != 'exit':
    user_query = input("User query: ")
    text, length = do_qna(user_query,db)
    print(f"\nContext Length: {length}\nResponse: {text}")


 
db = write_to_vector_store('data.md')
 
 
def get_answer(question, db):
    return do_qna(question, db)
 
 
def prepare_streamlit():
  st.title('QnA Activity')
 
  question = st.text_input('Enter your question here')
 
  if st.button('Get Answer'):
      answer, length = get_answer(question, db)
      answer = answer.strip().replace('\n', ' ')
      st.write(f'Answer: {answer}')
 
 
 
prepare_streamlit()
 