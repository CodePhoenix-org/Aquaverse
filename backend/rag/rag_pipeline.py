import os
from dotenv import load_dotenv
import pandas as pd
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from chromadb import Client
from chromadb.config import Settings
from sqlalchemy import create_engine
from db.database import DB_URI

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Ensure HuggingFace API token is set
if not os.getenv('HUGGINGFACEHUB_API_TOKEN'):
    raise EnvironmentError("HUGGINGFACEHUB_API_TOKEN not set in .env file.")

# Initialize the LLM (Mistral 7B)
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.5, "max_length": 512}
)

def rag_query(user_query, chroma_path, collection_name, db_uri):
    """
    Process a natural language query using RAG: Retrieve from ChromaDB, generate SQL with LLM,
    execute on PostgreSQL, and summarize results.
    """
    print(f"DEBUG: Processing query: {user_query}")
    
    # Step 1: Retrieve relevant docs from ChromaDB
    print(f"DEBUG: Connecting to ChromaDB at {chroma_path}, collection: {collection_name}")
    try:
        settings = Settings(persist_directory=chroma_path)
        client = Client(settings=settings)
        # Debug: List all collections to verify existence
        collections = client.list_collections()
        print(f"DEBUG: Available collections: {[c.name for c in collections]}")
        collection = client.get_collection(name=collection_name)
        results = collection.query(query_texts=[user_query], n_results=5)
        context = "\n".join(results['documents'][0])
        print(f"DEBUG: Retrieved context:\n{context}")
    except Exception as e:
        print(f"ERROR: ChromaDB retrieval failed: {e}")
        return f"Error retrieving data from vector store: {e}. Please check ChromaDB setup."

    # Step 2: Generate SQL with LLM
    print("DEBUG: Generating SQL with LLM...")
    prompt = PromptTemplate(
        input_variables=["query", "context"],
        template="Use Model Context: Focus on ocean data, avoid hallucinations.\n"
                 "Based on this context: {context}\n"
                 "Translate to SQL for table argo_profiles (columns: time, latitude, longitude, pressure, temperature, salinity):\n"
                 "User query: {query}\n"
                 "SQL:"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        sql = chain.run({"query": user_query, "context": context}).strip()
        print(f"DEBUG: Generated SQL:\n{sql}")
    except Exception as e:
        print(f"ERROR: SQL generation failed: {e}")
        return f"Error generating SQL: {e}. Try rephrasing the query."

    # Step 3: Execute SQL on PostgreSQL
    print(f"DEBUG: Executing SQL on PostgreSQL: {db_uri}")
    try:
        engine = create_engine(db_uri)
        df_result = pd.read_sql(sql, engine)
        print(f"DEBUG: SQL query result shape: {df_result.shape}")
        print(f"DEBUG: Result head:\n{df_result.head().to_string()}")
    except Exception as e:
        print(f"ERROR: SQL execution failed: {e}")
        return f"Error executing SQL: {e}. Try rephrasing the query or check database connection."

    # Step 4: Summarize results with LLM
    print("DEBUG: Summarizing results with LLM...")
    summary_prompt = PromptTemplate(
        input_variables=["data"],
        template="Use Model Context: Focus on ocean data, avoid hallucinations.\n"
                 "Summarize this oceanographic data in clear, concise language:\n{data}"
    )
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    try:
        response = summary_chain.run({"data": df_result.to_string()})
        print(f"DEBUG: Final response:\n{response}")
    except Exception as e:
        print(f"ERROR: Summarization failed: {e}")
        return f"Error summarizing results: {e}"

    return response

# Test the function
if __name__ == "__main__":
    test_query = "Show salinity profiles near equator in March 2025"
    result = rag_query(test_query, 'C:/Users/Gagan/Desktop/Sea Sheperds/Aquaverse/db/chroma_db', 'argo_summaries', DB_URI)
    print(f"TEST: Query result:\n{result}")
