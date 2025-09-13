import os
from dotenv import load_dotenv
import pandas as pd
from langchain.chains import LLMChain  # Temporary, will update fully later
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint  # Correct import
from chromadb import Client
from chromadb.config import Settings
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer

# Load HuggingFace API token from .env
load_dotenv()

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parquet_path = os.path.normpath(os.path.join(base_dir, 'data', 'processed', 'argo_profiles.parquet'))
chroma_path = os.path.normpath(os.path.join(base_dir, 'db', 'chroma_db'))
collection_name = 'argo_summaries'
db_uri = 'postgresql://postgres:Aashi@1234@localhost:5432/argo_db'
test_query = "Show salinity profiles near equator in March 2025"

# Step 1: Populate ChromaDB (from vector_store.py logic)
print("=== STEP 1: Populating ChromaDB ===")
df = pd.read_parquet(parquet_path)
print(f"Loaded Parquet with shape: {df.shape}")

settings = Settings(persist_directory=chroma_path)
client = Client(settings=settings)
try:
    client.delete_collection(collection_name)
except:
    pass
collection = client.create_collection(name=collection_name)

model = SentenceTransformer('all-MiniLM-L6-v2')
profiles = df.groupby(['time', 'latitude', 'longitude'])
print(f"Found {len(profiles)} unique profiles")

added_count = 0
for group_key, group in profiles:
    time, lat, lon = group_key
    summary = (f"Argo profile at {lat:.2f} lat, {lon:.2f} lon on {time}. "
               f"Temperature range: {group['temperature'].min():.1f}-{group['temperature'].max():.1f}°C, "
               f"Salinity mean: {group['salinity'].mean():.1f} PSU")
    embedding = model.encode(summary).tolist()
    profile_id = str(hash(group_key))
    collection.add(documents=[summary], embeddings=[embedding], ids=[profile_id])
    added_count += 1

print(f"Added {added_count} profiles to collection")

# Verify in-session
count = collection.count()
print(f"In-session count: {count}")

# Step 2: RAG Query (from rag_pipeline.py logic)
print("\n=== STEP 2: Running RAG Query ===")
# Updated LLM setup
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
)

# Retrieve
results = collection.query(query_texts=[test_query], n_results=5)
context = "\n".join(results['documents'][0])
print(f"Retrieved context:\n{context}")

# Generate SQL
prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Use Model Context: Focus on ocean data, avoid hallucinations.\n"
             "Based on this context: {context}\n"
             "Translate to SQL for table argo_profiles (columns: time, latitude, longitude, pressure, temperature, salinity):\n"
             "User query: {query}\n"
             "SQL:"
)
chain = LLMChain(llm=llm, prompt=prompt)  # Temporary use; update to RunnableSequence later
sql = chain.run({"query": test_query, "context": context}).strip()
print(f"Generated SQL:\n{sql}")

# Execute SQL
engine = create_engine(db_uri)
df_result = pd.read_sql(sql, engine)
print(f"SQL result shape: {df_result.shape}")
print(f"Result head:\n{df_result.head().to_string()}")

# Summarize
summary_prompt = PromptTemplate(
    input_variables=["data"],
    template="Use Model Context: Focus on ocean data, avoid hallucinations.\n"
             "Summarize this oceanographic data in clear, concise language:\n{data}"
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
response = summary_chain.run({"data": df_result.to_string()})
print(f"Final response:\n{response}")

print("\n✅ Full pipeline test complete!")

#the below code is with a different llm model google/flan-t5-base
# test_fullpipeline.py
# import os
# import pandas as pd
# from langchain.chains import LLMChain  # Temporary, will update fully later
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint  # Correct import
# from chromadb import Client
# from chromadb.config import Settings
# from sqlalchemy import create_engine
# from sentence_transformers import SentenceTransformer

# # Set HuggingFace API token
# Example: os.environ['HUGGINGFACEHUB_API_TOKEN'] = '<your_token_here>'


# # Paths
# base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# parquet_path = os.path.normpath(os.path.join(base_dir, 'data', 'processed', 'argo_profiles.parquet'))
# chroma_path = os.path.normpath(os.path.join(base_dir, 'db', 'chroma_db'))
# collection_name = 'argo_summaries'
# db_uri = 'postgresql://postgres:Aashi@1234@localhost:5432/argo_db'
# test_query = "Show salinity profiles near equator in March 2025"

# # Step 1: Populate ChromaDB (from vector_store.py logic)
# print("=== STEP 1: Populating ChromaDB ===")
# df = pd.read_parquet(parquet_path)
# print(f"Loaded Parquet with shape: {df.shape}")

# settings = Settings(persist_directory=chroma_path)
# client = Client(settings=settings)
# try:
#     client.delete_collection(collection_name)
# except:
#     pass
# collection = client.create_collection(name=collection_name)

# model = SentenceTransformer('all-MiniLM-L6-v2')
# profiles = df.groupby(['time', 'latitude', 'longitude'])
# print(f"Found {len(profiles)} unique profiles")

# added_count = 0
# for group_key, group in profiles:
#     time, lat, lon = group_key
#     summary = (f"Argo profile at {lat:.2f} lat, {lon:.2f} lon on {time}. "
#                f"Temperature range: {group['temperature'].min():.1f}-{group['temperature'].max():.1f}°C, "
#                f"Salinity mean: {group['salinity'].mean():.1f} PSU")
#     embedding = model.encode(summary).tolist()
#     profile_id = str(hash(group_key))
#     collection.add(documents=[summary], embeddings=[embedding], ids=[profile_id])
#     added_count += 1

# print(f"Added {added_count} profiles to collection")

# # Verify in-session
# count = collection.count()
# print(f"In-session count: {count}")

# # Step 2: RAG Query (from rag_pipeline.py logic)
# print("\n=== STEP 2: Running RAG Query ===")
# # Updated LLM setup with a different model
# llm = HuggingFaceEndpoint(
#     repo_id="google/flan-t5-base",  # Changed to a publicly accessible model
#     task="text-generation",
#     max_new_tokens=512,
#     temperature=0.5,
#     huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN']
# )

# # Retrieve
# results = collection.query(query_texts=[test_query], n_results=5)
# context = "\n".join(results['documents'][0])
# print(f"Retrieved context:\n{context}")

# # Generate SQL
# prompt = PromptTemplate(
#     input_variables=["query", "context"],
#     template="Use Model Context: Focus on ocean data, avoid hallucinations.\n"
#              "Based on this context: {context}\n"
#              "Translate to SQL for table argo_profiles (columns: time, latitude, longitude, pressure, temperature, salinity):\n"
#              "User query: {query}\n"
#              "SQL:"
# )
# chain = LLMChain(llm=llm, prompt=prompt)  # Temporary use; update to RunnableSequence later
# sql = chain.run({"query": test_query, "context": context}).strip()
# print(f"Generated SQL:\n{sql}")

# # Execute SQL
# engine = create_engine(db_uri)
# df_result = pd.read_sql(sql, engine)
# print(f"SQL result shape: {df_result.shape}")
# print(f"Result head:\n{df_result.head().to_string()}")

# # Summarize
# summary_prompt = PromptTemplate(
#     input_variables=["data"],
#     template="Use Model Context: Focus on ocean data, avoid hallucinations.\n"
#              "Summarize this oceanographic data in clear, concise language:\n{data}"
# )
# summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
# response = summary_chain.run({"data": df_result.to_string()})
# print(f"Final response:\n{response}")

# print("\n✅ Full pipeline test complete!")