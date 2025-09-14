# import os
# import pandas as pd
# from langchain.chains import LLMChain  # Temporary
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from chromadb import Client
# from chromadb.config import Settings
# from sqlalchemy import create_engine
# from sentence_transformers import SentenceTransformer
# import re
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Verify API key
# api_key = os.getenv('OPENROUTER_API_KEY')
# if not api_key:
#     raise ValueError("OPENROUTER_API_KEY not found in environment. Check .env file or set it manually.")

# # Paths
# base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# parquet_path = os.path.normpath(os.path.join(base_dir, 'data', 'processed', 'argo_profiles.parquet'))
# chroma_path = os.path.normpath(os.path.join(base_dir, 'db', 'chroma_db'))
# collection_name = 'argo_summaries'
# db_uri = 'postgresql://postgres:Aashi%401234@localhost:5432/argo_db'

# # Step 1: Populating ChromaDB (run once at start)
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
# count = collection.count()
# print(f"In-session count: {count}")

# # Step 2: RAG Query (interactive loop)
# print("\n=== STEP 2: Running RAG Query (Interactive Mode) ===")
# print("Enter your query (type 'quit' to exit):")
# llm = ChatOpenAI(
#     model_name="mistralai/mistral-7b-instruct:free",
#     openai_api_base="https://openrouter.ai/api/v1",
#     openai_api_key=api_key,
#     temperature=0.5,
#     max_tokens=512
# )

# while True:
#     query = input("> ").strip()
#     if query.lower() == 'quit':
#         print("Exiting interactive mode. Goodbye!")
#         break

#     results = collection.query(query_texts=[query], n_results=5)
#     context = "\n".join(results['documents'][0])
#     print(f"Retrieved context:\n{context}")

#     # Generate SQL with diverse rules and examples
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are an ocean data expert. Return ONLY the SQL query as a single line with no explanations or extra text. Use EXTRACT(MONTH FROM time) = <month> for specific months, EXTRACT(YEAR FROM time) = EXTRACT(YEAR FROM CURRENT_DATE) for 'this year', or EXTRACT(MONTH FROM time) = EXTRACT(MONTH FROM CURRENT_DATE) AND EXTRACT(YEAR FROM time) = EXTRACT(YEAR FROM CURRENT_DATE) for 'right now'. Target PostgreSQL syntax. For 'near <lat> lat', use latitude BETWEEN <lat - 0.1> AND <lat + 0.1>. For 'near equator', use latitude BETWEEN -10 AND 10. For 'nearest floats to <lat>, <lon>', use latitude BETWEEN <lat - 0.1> AND <lat + 0.1> AND longitude BETWEEN <lon - 0.1> AND <lon + 0.1> ORDER BY ABS(latitude - <lat>) + ABS(longitude - <lon>) LIMIT 5. For 'right now', add time >= NOW() AND time < NOW() + INTERVAL '1 day'. For regional queries (e.g., Indian Ocean), use latitude BETWEEN -10 AND 10 AND longitude BETWEEN 60 AND 100; for Arabian Sea, use latitude BETWEEN 5 AND 25 AND longitude BETWEEN 45 AND 75. Use context to inform ranges, but default to these rules if context is unclear or conflicts. Examples: 1. What’s the temperature like near 8.9 lat in June 2025 -> SELECT temperature FROM argo_profiles WHERE EXTRACT(MONTH FROM time) = 6 AND latitude BETWEEN 8.8 AND 9.0. 2. Nearest floats to 8.9, 68.0 right now -> SELECT * FROM argo_profiles WHERE latitude BETWEEN 8.8 AND 9.0 AND longitude BETWEEN 67.9 AND 68.1 AND time >= NOW() AND time < NOW() + INTERVAL '1 day' ORDER BY ABS(latitude - 8.9) + ABS(longitude - 68.0) LIMIT 5. 3. Compare salinity in the Arabian Sea for the last 6 months -> SELECT salinity FROM argo_profiles WHERE latitude BETWEEN 5 AND 25 AND longitude BETWEEN 45 AND 75 AND time > CURRENT_DATE - INTERVAL '6 months'."),
#         ("user", "Based on this context: {context}\nTranslate to SQL for table argo_profiles (columns: time, latitude, longitude, pressure, temperature, salinity):\nUser query: {query}")
#     ])
#     chain = LLMChain(llm=llm, prompt=prompt)
#     sql_response = chain.run({"query": query, "context": context}).strip()
#     print(f"Raw LLM SQL response:\n{sql_response}")

#     # SQL Fixer with stricter enforcement
#     fixer_prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are a SQL expert. Return ONLY the SQL query as a single line, with no explanations, notes, or extra text. Fix for PostgreSQL syntax and ensure it is valid for table argo_profiles (columns: time, latitude, longitude, pressure, temperature, salinity). Do not alter the query intent (e.g., latitude or longitude ranges) unless there is a clear syntax error. Preserve the original intent from the raw query."),
#         ("user", "Fix this SQL: {sql_response}")
#     ])
#     fixer_chain = LLMChain(llm=llm, prompt=fixer_prompt)
#     sql_output = fixer_chain.run({"sql_response": sql_response}).strip()
#     # Use regex to extract only the SQL query
#     sql = re.search(r"SELECT\s+.*?(?:;|$)", sql_output, re.IGNORECASE | re.DOTALL).group(0).rstrip(';') if re.search(r"SELECT\s+.*?(?:;|$)", sql_output, re.IGNORECASE | re.DOTALL) else sql_output
#     print(f"Fixed SQL:\n{sql}")

#     # Execute SQL with error handling
#     try:
#         engine = create_engine(db_uri)
#         df_result = pd.read_sql(sql, engine)
#         print(f"SQL result shape: {df_result.shape}")
#         print(f"Result head:\n{df_result.head().to_string()}")
#     except Exception as e:
#         print(f"SQL execution failed: {e}")
#         print("Please ensure PostgreSQL is running and the argo_profiles table exists with the correct schema.")
#         continue

#     # Summarize (aggregate with chunking to handle large data)
#     if 'df_result' in locals():
#         if df_result.empty:
#             response = "No data found for this query."
#         else:
#             # Chunk data if large to stay within token limits
#             chunk_size = 100
#             chunks = [df_result[i:i + chunk_size] for i in range(0, len(df_result), chunk_size)]
#             summaries = []
#             for chunk in chunks:
#                 if not chunk.empty:
#                     summary_prompt = ChatPromptTemplate.from_messages([
#                         ("system", "You are an ocean data expert. Return ONLY the summary as clear, concise text with no explanations or extra text. Aggregate across all profiles in this chunk, reporting exact minimum, maximum, and average values for temperature to one decimal place if temperature is selected, or salinity if salinity is selected."),
#                         ("user", "Summarize this oceanographic data in clear, concise language:\n{data}")
#                     ])
#                     summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
#                     summary = summary_chain.run({"data": chunk.to_string()}).strip()
#                     summaries.append(summary)
#             response = "; ".join(summaries) if summaries else "No valid summary generated."
#         print(f"Final response:\n{response}")

#     print("\nEnter your next query (type 'quit' to exit):")

# print("\n✅ Interactive mode complete!")


# updated code for better summarisation but not yet tested. The above commented out code is working fine as well
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
db_uri = 'postgresql://postgres:Aashi%401234@localhost:5432/argo_db'

# Step 1: Populating ChromaDB (run once at start)
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

# Step 2: RAG Query (interactive loop)
print("\n=== STEP 2: Running RAG Query (Interactive Mode) ===")
print("Enter your query (type 'quit' to exit). Note: Free model limit (50 requests/day) may be reached. Consider https://x.ai/grok for more quota.")
llm = ChatOpenAI(
    model_name="mistralai/mistral-7b-instruct:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=api_key,
    temperature=0.5,
    max_tokens=512
)

while True:
    query = input("> ").strip()
    if query.lower() == 'quit':
        print("Exiting interactive mode. Goodbye!")
        break

    results = collection.query(query_texts=[query], n_results=5)
    context = "\n".join(results['documents'][0])
    print(f"Retrieved context:\n{context}")

    # Generate SQL with diverse rules and examples
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an ocean data expert. Return ONLY the SQL query as a single line with no explanations or extra text. Use EXTRACT(MONTH FROM time) = <month> for specific months, EXTRACT(YEAR FROM time) = EXTRACT(YEAR FROM CURRENT_DATE) for 'this year', or EXTRACT(MONTH FROM time) = EXTRACT(MONTH FROM CURRENT_DATE) AND EXTRACT(YEAR FROM time) = EXTRACT(YEAR FROM CURRENT_DATE) for 'right now'. Target PostgreSQL syntax. For 'near <lat> lat', use latitude BETWEEN <lat - 0.1> AND <lat + 0.1>. For 'near equator', use latitude BETWEEN -10 AND 10. For 'nearest floats to <lat>, <lon>', use latitude BETWEEN <lat - 0.1> AND <lat + 0.1> AND longitude BETWEEN <lon - 0.1> AND <lon + 0.1> ORDER BY ABS(latitude - <lat>) + ABS(longitude - <lon>) LIMIT 5. For 'right now', add time >= NOW() AND time < NOW() + INTERVAL '1 day'. For regional queries (e.g., Indian Ocean), use latitude BETWEEN -10 AND 10 AND longitude BETWEEN 60 AND 100; for Arabian Sea, use latitude BETWEEN 5 AND 25 AND longitude BETWEEN 45 AND 75. Use context to inform ranges, but default to these rules if context is unclear or conflicts. Examples: 1. What’s the temperature like near 8.9 lat in June 2025 -> SELECT temperature FROM argo_profiles WHERE EXTRACT(MONTH FROM time) = 6 AND latitude BETWEEN 8.8 AND 9.0. 2. Nearest floats to 8.9, 68.0 right now -> SELECT * FROM argo_profiles WHERE latitude BETWEEN 8.8 AND 9.0 AND longitude BETWEEN 67.9 AND 68.1 AND time >= NOW() AND time < NOW() + INTERVAL '1 day' ORDER BY ABS(latitude - 8.9) + ABS(longitude - 68.0) LIMIT 5. 3. Compare salinity in the Arabian Sea for the last 6 months -> SELECT salinity FROM argo_profiles WHERE latitude BETWEEN 5 AND 25 AND longitude BETWEEN 45 AND 75 AND time > CURRENT_DATE - INTERVAL '6 months'."),
        ("user", "Based on this context: {context}\nTranslate to SQL for table argo_profiles (columns: time, latitude, longitude, pressure, temperature, salinity):\nUser query: {query}")
    ])
    chain = LLMChain(llm=llm, prompt=prompt)
    sql_response = chain.run({"query": query, "context": context}).strip()
    print(f"Raw LLM SQL response:\n{sql_response}")

    # SQL Fixer with stricter enforcement
    fixer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a SQL expert. Return ONLY the SQL query as a single line, with no explanations, notes, or extra text. Fix for PostgreSQL syntax and ensure it is valid for table argo_profiles (columns: time, latitude, longitude, pressure, temperature, salinity). Do not alter the query intent (e.g., latitude or longitude ranges) unless there is a clear syntax error. Preserve the original intent from the raw query."),
        ("user", "Fix this SQL: {sql_response}")
    ])
    fixer_chain = LLMChain(llm=llm, prompt=fixer_prompt)
    sql_output = fixer_chain.run({"sql_response": sql_response}).strip()
    # Use regex to extract only the SQL query
    sql = re.search(r"SELECT\s+.*?(?:;|$)", sql_output, re.IGNORECASE | re.DOTALL).group(0).rstrip(';') if re.search(r"SELECT\s+.*?(?:;|$)", sql_output, re.IGNORECASE | re.DOTALL) else sql_output
    print(f"Fixed SQL:\n{sql}")

    # Execute SQL with error handling
    try:
        engine = create_engine(db_uri)
        df_result = pd.read_sql(sql, engine)
        print(f"SQL result shape: {df_result.shape}")
        print(f"Result head:\n{df_result.head().to_string()}")
    except Exception as e:
        print(f"SQL execution failed: {e}")
        print("Please ensure PostgreSQL is running and the argo_profiles table exists with the correct schema.")
        continue

    # Summarize (pre-aggregate to avoid token limit issues)
    if 'df_result' in locals():
        if df_result.empty:
            response = "No data found for this query."
        else:
            param = "temperature" if "temperature" in sql.lower() else "salinity" if "salinity" in sql.lower() else None
            if param:
                agg = df_result[param].agg(['min', 'max', 'mean']).round(1)
                summary_prompt = ChatPromptTemplate.from_messages([
                    ("system", f"You are an ocean data expert. Return ONLY the summary as clear, concise text with no explanations or extra text. Report exact minimum, maximum, and average values for {param} to one decimal place based on the provided stats."),
                    ("user", f"Summarize this oceanographic data: min {agg['min']}, max {agg['max']}, avg {agg['mean']}")
                ])
                summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
                response = summary_chain.run({}).strip()
            else:
                response = "No valid parameter (temperature or salinity) found for summarization."
        print(f"Final response:\n{response}")

    print("\nEnter your next query (type 'quit' to exit):")

print("\n✅ Interactive mode complete!")