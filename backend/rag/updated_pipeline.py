import os
import re
import traceback
from typing import List, Dict, Tuple, Optional
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from db.database import DB_URI

# ------------------------------------------------------------
# Paths and Environment
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

load_dotenv(os.path.join(BACKEND_DIR, ".env"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PARQUET_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "argo_profiles.parquet")
CHROMA_PATH = os.path.join(PROJECT_ROOT, "db", "chroma_db")
COLLECTION_NAME = "argo_summaries"

if not DB_URI:
    raise ValueError("DB_URI not set in environment")

# ------------------------------------------------------------
# ChromaDB helpers
# ------------------------------------------------------------
def ensure_collection(chroma_path: str, collection_name: str):
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        return client.get_collection(collection_name)
    except Exception:
        return client.create_collection(collection_name)


def populate_chroma_if_empty(parquet_path: str, chroma_path: str, collection_name: str) -> int:
    collection = ensure_collection(chroma_path, collection_name)
    try:
        if collection.count() > 0:
            return collection.count()
    except:
        pass

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if df.empty:
        return 0

    model = SentenceTransformer("all-MiniLM-L6-v2")
    profiles = df.groupby(["time", "latitude", "longitude"])
    ids, docs = [], []

    for key, group in profiles:
        time, lat, lon = key
        summary = (
            f"Argo profile at {float(lat):.2f} lat, {float(lon):.2f} lon on {time}. "
            f"Temperature range: {float(group['temperature'].min()):.1f}-{float(group['temperature'].max()):.1f}°C, "
            f"Salinity mean: {float(group['salinity'].mean()):.1f} PSU"
        )
        ids.append(str(hash(key)))
        docs.append(summary)

    if docs:
        embeddings = model.encode(docs).tolist()
        collection.add(ids=ids, documents=docs, embeddings=embeddings)

    return len(docs)

# ------------------------------------------------------------
# SQL helper
# ------------------------------------------------------------
def run_sql(sql: str) -> pd.DataFrame:
    engine = create_engine(DB_URI)
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)

# ------------------------------------------------------------
# Summarization helpers
# ------------------------------------------------------------
def extract_numeric_stats(df: pd.DataFrame) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Extract numeric statistics from SQL results."""
    if df.empty:
        return "No data found.", {}

    lower_map = {c.lower(): c for c in df.columns}
    targets = ["temperature", "salinity"]
    stats_out: Dict[str, Dict[str, float]] = {}

    for target in targets:
        if target in lower_map:
            col = lower_map[target]
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if not series.empty:
                stats_out[target] = {
                    "min": round(float(series.min()), 1),
                    "max": round(float(series.max()), 1),
                    "mean": round(float(series.mean()), 1),
                }

    if not stats_out:
        return "No valid temperature or salinity data.", {}

    parts = [f"{name}: min {s['min']}, max {s['max']}, avg {s['mean']}" for name, s in stats_out.items()]
    return "; ".join(parts), stats_out


def extract_stats_from_docs(docs: List[str]) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """
    Parse temperature ranges and salinity means from retrieved document texts
    to build approximate stats when SQL results are not available.
    """
    if not docs:
        return "No data found.", {}

    temp_lows: List[float] = []
    temp_highs: List[float] = []
    sal_means: List[float] = []

    # Fixed regex patterns - removed double backslashes
    temp_re = re.compile(r"Temperature range:\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*°C", re.IGNORECASE)
    sal_re = re.compile(r"Salinity mean:\s*([0-9]+(?:\.[0-9]+)?)\s*PSU", re.IGNORECASE)

    for doc in docs:
        t = temp_re.search(doc)
        if t:
            try:
                temp_lows.append(float(t.group(1)))
                temp_highs.append(float(t.group(2)))
            except Exception:
                pass
        s = sal_re.search(doc)
        if s:
            try:
                sal_means.append(float(s.group(1)))
            except Exception:
                pass

    stats_map: Dict[str, Dict[str, float]] = {}
    parts: List[str] = []

    if temp_lows and temp_highs:
        t_min = min(temp_lows)
        t_max = max(temp_highs)
        t_mean = (sum(temp_lows + temp_highs) / (len(temp_lows) + len(temp_highs)))
        stats_map["temperature"] = {"min": round(t_min, 1), "max": round(t_max, 1), "mean": round(t_mean, 1)}
        t = stats_map["temperature"]
        parts.append(f"temperature: min {t['min']}, max {t['max']}, avg {t['mean']}")

    if sal_means:
        s_min = min(sal_means)
        s_max = max(sal_means)
        s_mean = sum(sal_means) / len(sal_means)
        stats_map["salinity"] = {"min": round(s_min, 1), "max": round(s_max, 1), "mean": round(s_mean, 1)}
        s = stats_map["salinity"]
        parts.append(f"salinity: min {s['min']}, max {s['max']}, avg {s['mean']}")

    if not parts:
        return "No valid temperature or salinity data.", {}

    return "; ".join(parts), stats_map


def create_simple_summary_from_stats(stats_text: str, user_query: str) -> str:
    """
    Create a simple summary when LLM is not available.
    """
    if "No valid" in stats_text or "No data" in stats_text:
        return "No oceanographic data found matching your query."
    
    # Parse the stats to create a more readable summary
    parts = []
    if "temperature:" in stats_text.lower():
        temp_match = re.search(r"temperature:\s*min\s*([\d.]+),\s*max\s*([\d.]+),\s*avg\s*([\d.]+)", stats_text.lower())
        if temp_match:
            min_t, max_t, avg_t = temp_match.groups()
            parts.append(f"Temperature ranges from {min_t}°C to {max_t}°C with an average of {avg_t}°C")
    
    if "salinity:" in stats_text.lower():
        sal_match = re.search(r"salinity:\s*min\s*([\d.]+),\s*max\s*([\d.]+),\s*avg\s*([\d.]+)", stats_text.lower())
        if sal_match:
            min_s, max_s, avg_s = sal_match.groups()
            if float(max_s) - float(min_s) < 2.0:  # Stable salinity
                parts.append(f"Salinity is relatively stable, averaging {avg_s} PSU")
            else:
                parts.append(f"Salinity varies from {min_s} to {max_s} PSU with an average of {avg_s} PSU")
    
    if parts:
        return ". ".join(parts) + "."
    return stats_text


def summarize_with_llm_from_stats(llm: ChatOpenAI, stats_text: str, user_query: str) -> str:
    """Generate a concise summary using the LLM."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an oceanographic data expert. Create concise, easy-to-understand summaries for non-experts. "
         "Focus on trends, patterns, and key insights. Avoid technical jargon and raw numbers where possible. "
         "Make it conversational and informative."),
        ("user",
         "User Query: {query}\n\n"
         "Computed oceanographic statistics: {stats}\n\n"
         "Instructions:\n"
         "- Provide a concise summary in 2-3 sentences\n"
         "- Focus on trends, ranges, and typical values\n"
         "- Use accessible language for non-experts\n"
         "- Do not repeat raw statistical data\n"
         "- Highlight any interesting patterns or insights")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"query": user_query, "stats": stats_text}).strip()
        # Clean up the result to ensure it's not too verbose
        if len(result) > 500:  # Truncate if too long
            sentences = result.split('.')
            result = '. '.join(sentences[:3]) + '.'
        return result
    except Exception as e:
        print(f"Warning: LLM summarization failed: {e}")
        # Fallback to simple summary
        return create_simple_summary_from_stats(stats_text, user_query)


# ------------------------------------------------------------
# Main RAG query function
# ------------------------------------------------------------
def rag_query(
    user_query: str,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    db_uri: str = DB_URI
) -> str:
    """
    Process a natural language query using RAG:
    - Retrieve relevant context from ChromaDB
    - Generate SQL via LLM (if available)
    - Execute SQL on PostgreSQL
    - Always return a summarized response, never raw documents
    """
    try:
        print(f"DEBUG: Processing query: {user_query}")
        
        # Step 1: Retrieve context from ChromaDB
        try:
            client = chromadb.PersistentClient(path=chroma_path)
            collection = client.get_collection(collection_name)
            results = collection.query(query_texts=[user_query], n_results=5)
            docs = results.get("documents", [[]])[0] if results else []
            context = "\n".join(docs) if docs else "No context found."
            print(f"DEBUG: Retrieved {len(docs)} documents from ChromaDB")
        except Exception as e:
            print(f"ERROR: ChromaDB retrieval failed: {e}")
            return f"Error accessing data store: {str(e)}"

        # Step 2: Initialize LLM if available
        llm = None
        if OPENROUTER_API_KEY:
            try:
                llm = ChatOpenAI(
                    model_name="mistralai/mistral-7b-instruct:free",
                    openai_api_base="https://openrouter.ai/api/v1",
                    openai_api_key=OPENROUTER_API_KEY,
                    temperature=0.5,
                    max_tokens=512
                )
                print("DEBUG: LLM initialized successfully")
            except Exception as e:
                print(f"WARNING: LLM initialization failed: {e}")
                llm = None

        # Step 3: Try to generate and execute SQL if LLM is available
        df_result = pd.DataFrame()
        if llm:
            try:
                # Generate SQL
                sql_prompt = ChatPromptTemplate.from_messages([
                    ("system", 
                     "You are a SQL expert for oceanographic data. Generate ONLY a valid PostgreSQL SELECT query. "
                     "Table: argo_profiles with columns: time, latitude, longitude, pressure, temperature, salinity. "
                     "Return only the SQL query, no explanations."),
                    ("user", 
                     "Context from vector search: {context}\n\n"
                     "User query: {query}\n\n"
                     "Generate a SQL SELECT query:")
                ])
                
                sql_chain = sql_prompt | llm | StrOutputParser()
                sql_response = sql_chain.invoke({"query": user_query, "context": context}).strip()
                
                # Extract clean SQL
                sql_match = re.search(r"SELECT\s+.*?(?:;|$)", sql_response, re.IGNORECASE | re.DOTALL)
                sql = (sql_match.group(0).rstrip(";") if sql_match else sql_response).strip()
                
                print(f"DEBUG: Generated SQL: {sql}")
                
                # Execute SQL
                engine = create_engine(db_uri)
                df_result = pd.read_sql(text(sql), engine)
                print(f"DEBUG: SQL executed successfully, got {len(df_result)} rows")
                
            except Exception as e:
                print(f"WARNING: SQL generation/execution failed: {e}")
                df_result = pd.DataFrame()

        # Step 4: Extract statistics and create summary
        # First try from SQL results
        stats_text, stats_map = extract_numeric_stats(df_result)
        
        # If SQL didn't work or returned no valid stats, use document parsing
        if not stats_map and docs:
            print("DEBUG: Falling back to document parsing for statistics")
            stats_text, stats_map = extract_stats_from_docs(docs)

        # Step 5: Always return a summarized response
        if stats_map:  # We have numerical data to summarize
            if llm:
                print("DEBUG: Creating LLM summary from statistics")
                return summarize_with_llm_from_stats(llm, stats_text, user_query)
            else:
                print("DEBUG: Creating simple summary from statistics (no LLM)")
                return create_simple_summary_from_stats(stats_text, user_query)
        
        # Step 6: If no numerical stats available, create a basic summary
        if docs:
            if llm:
                # Use LLM to summarize the document context
                try:
                    summary_prompt = ChatPromptTemplate.from_messages([
                        ("system", 
                         "Summarize this oceanographic data in 2-3 clear sentences for non-experts. "
                         "Focus on key insights and patterns, avoid listing raw data."),
                        ("user", 
                         "User asked: {query}\n\n"
                         "Available data: {context}\n\n"
                         "Provide a concise summary:")
                    ])
                    summary_chain = summary_prompt | llm | StrOutputParser()
                    return summary_chain.invoke({"query": user_query, "context": context}).strip()
                except Exception as e:
                    print(f"WARNING: LLM context summarization failed: {e}")
                    # Fall through to basic response
            
            # Basic response without LLM
            num_profiles = len(docs)
            return f"Found {num_profiles} relevant oceanographic profiles. The data includes various temperature and salinity measurements across different locations and times. For detailed analysis, please refine your query with specific parameters like location, time period, or measurement ranges."
        
        # Final fallback
        return "No relevant oceanographic data found for your query. Please try rephrasing or specifying location, time period, or measurement parameters."

    except Exception as e:
        print(f"ERROR in rag_query: {e}")
        traceback.print_exc()
        return f"An error occurred while processing your query: {str(e)}"


# ------------------------------------------------------------
# CLI interactive mode
# ------------------------------------------------------------
def main():
    try:
        inserted = populate_chroma_if_empty(PARQUET_PATH, CHROMA_PATH, COLLECTION_NAME)
        if inserted:
            print(f"Populated {inserted} documents into '{COLLECTION_NAME}'.")
        else:
            print(f"ChromaDB already contains data.")
    except Exception as e:
        print(f"ERROR: Failed to set up ChromaDB: {e}")
        return

    print("\n=== Interactive RAG CLI ===")
    print("Ask questions about oceanographic data. Type 'quit' to exit.")
    
    while True:
        try:
            query = input("\n> ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                break
            if not query:
                continue
                
            print("\nProcessing...")
            response = rag_query(query)
            print(f"\nResponse:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error processing query: {e}")

    print("✅ Interactive session ended.")


if __name__ == "__main__":
    main()