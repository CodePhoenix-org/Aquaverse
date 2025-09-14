import os
import pandas as pd
from langchain.chains import LLMChain  # Temporary
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from chromadb import Client
from chromadb.config import Settings
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
import re
from dotenv import load_dotenv
from db.database import DB_URI
# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv('OPENROUTER_API_KEY')
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment. Check .env file or set it manually.")

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parquet_path = os.path.normpath(os.path.join(base_dir, 'data', 'processed', 'argo_profiles.parquet'))
chroma_path = os.path.normpath(os.path.join(base_dir, 'db', 'chroma_db'))
collection_name = 'argo_summaries'
test_query = "Show salinity profiles near equator in March 2025"

# Step 1: Populating ChromaDB
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
count = collection.count()
print(f"In-session count: {count}")

# Step 2: RAG Query
print("\n=== STEP 2: Running RAG Query ===")
llm = ChatOpenAI(
    model_name="mistralai/mistral-7b-instruct:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=api_key,
    temperature=0.5,
    max_tokens=512
)

results = collection.query(query_texts=[test_query], n_results=5)
context = "\n".join(results['documents'][0])
print(f"Retrieved context:\n{context}")

# Refined prompt with context-based range
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an ocean data expert. Return ONLY the SQL query as a single line with no explanations or extra text. Use EXTRACT(MONTH FROM time) = <month> for TIMESTAMP columns or LIKE 'YYYY-MM%' for string time columns. Target PostgreSQL syntax. For 'near equator', use latitude BETWEEN -10 AND 10 unless context latitudes suggest a specific range, then use the minimum and maximum context latitudes ±0.1. Avoid multiplying latitude/longitude by large numbers."),
    ("user", "Based on this context: {context}\nTranslate to SQL for table argo_profiles (columns: time, latitude, longitude, pressure, temperature, salinity):\nUser query: {query}")
])
chain = LLMChain(llm=llm, prompt=prompt)
sql_response = chain.run({"query": test_query, "context": context}).strip()
print(f"Raw LLM SQL response:\n{sql_response}")

# Parse SQL from response
sql = re.search(r"SELECT.*?(?:;|$)", sql_response, re.IGNORECASE | re.DOTALL)
if sql:
    sql = sql.group(0).rstrip(';')
else:
    sql = sql_response.rstrip(';')
print(f"Parsed SQL:\n{sql}")

# Execute SQL with error handling
try:
    engine = create_engine(DB_URI)
    df_result = pd.read_sql(sql, engine)
    print(f"SQL result shape: {df_result.shape}")
    print(f"Result head:\n{df_result.head().to_string()}")
except Exception as e:
    print(f"SQL execution failed: {e}")
    print("Please ensure PostgreSQL is running and the argo_profiles table exists with the correct schema.")

# Summarize
if 'df_result' in locals():
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an ocean data expert. Return ONLY the summary as clear, concise text with no explanations or extra text."),
        ("user", "Summarize this oceanographic data in clear, concise language:\n{data}")
    ])
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    response = summary_chain.run({"data": df_result.to_string()}).strip()
    print(f"Final response:\n{response}")
else:
    print("Skipping summarization due to SQL failure.")

print("\n✅ Full pipeline test complete!")


