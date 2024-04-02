import os, json
from openai import AzureOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredFileLoader#UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import AzureOpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.embeddings import AzureOpenAIEmbeddings
 
azure_api_key = "168dfde591cd421ebd00ab8870defb3f"
azure_base_url = "https://usbank-bootcamp-2.openai.azure.com/"
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
  chunk_size = 5000
  embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
 
  
  loader = UnstructuredFileLoader(md_file_name)
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=10)
  texts = text_splitter.split_documents(loader.load())
 
 
  print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
  db = Chroma.from_documents(texts, embedding_function, persist_directory="./vector_store")
 
  return db
 
  # for text in texts:
  #   embedding_vector = embedding_function.embed_documents([text.page_content])
  #   print(f"Text: {text}\n\n embd_arr_len: {len(embedding_vector[0])} Embedding: {embedding_vector}\n")
  #   print("------------")
 
  # #query = "What is the purpose of life?"
  # output = db.similarity_search_with_score(query)
 
  # for entry in output:
  #   print(f"similar items to query: {query} = {entry}")
 
def get_token_count(text:str):
  return len(text.split())
 
def do_qna(user_query:str,db):
  """
  system_prompt: str: The system prompt to be used for the completion
  user_query: str: The user query to be used for the completion
  qna_so_far: str: The qna so far to be used for the completion
  """
  #global db
   
  # if db is None:
  #     print("Vector store is not initialized. Please initialize it first.")
  #     return
 
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
                                       temperature=0.3,
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
 
 
 
if __name__ == "__main__":
 
  db = write_to_vector_store(r'data.md')
  #db = write_to_vector_store(r'Altitude.md')
  begin_qna(db)
 