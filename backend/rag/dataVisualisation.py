import os
import re
import hashlib
import traceback
from typing import List, Dict, Tuple, Optional
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from db.database import DB_URI
from datetime import datetime

# ------------------------------------------------------------
# Global SQL cache
# ------------------------------------------------------------
sql_cache = {}

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
# DB helpers
# ------------------------------------------------------------
def get_available_dates(db_uri: str) -> list:
    """Get unique dates from argo_profiles table."""
    try:
        engine = create_engine(db_uri)
        with engine.connect() as conn:
            result = pd.read_sql("SELECT DISTINCT time::date FROM argo_profiles ORDER BY time", conn)
            return result['time'].astype(str).tolist()
    except Exception as e:
        print(f"WARNING: Failed to get available dates: {e}")
        return []


def get_dataset_stats(db_uri: str) -> str:
    """Get overall min/max for temperature and salinity from dataset."""
    try:
        sql = """
        SELECT 
            MIN(temperature) AS min_temp, MAX(temperature) AS max_temp,
            MIN(salinity) AS min_sal, MAX(salinity) AS max_sal
        FROM argo_profiles
        """
        engine = create_engine(db_uri)
        df = pd.read_sql(text(sql), engine)
        if not df.empty:
            min_temp = df['min_temp'].iloc[0]
            max_temp = df['max_temp'].iloc[0]
            min_sal = df['min_sal'].iloc[0]
            max_sal = df['max_sal'].iloc[0]
            return (
                f"temperatures ranging from ~{min_temp:.1f}°C to ~{max_temp:.1f}°C, "
                f"salinity from ~{min_sal:.1f} to ~{max_sal:.1f} PSU"
            )
        return "temperature and salinity data"
    except Exception as e:
        print(f"WARNING: Failed to get dataset stats: {e}")
        return "temperature and salinity data"


def run_sql(sql: str) -> pd.DataFrame:
    engine = create_engine(DB_URI)
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)

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
# Stats helpers
# ------------------------------------------------------------
def extract_numeric_stats(df: pd.DataFrame) -> Tuple[str, Dict[str, Dict[str, float]]]:
    if df.empty:
        return "No data found.", {}

    print(f"DEBUG: DataFrame shape: {df.shape}")
    print(f"DEBUG: DataFrame columns: {list(df.columns)}")
    print(f"DEBUG: DataFrame sample:\n{df.head()}")

    # Create a mapping of lowercase column names to actual column names
    lower_map = {c.lower(): c for c in df.columns}
    stats_out: Dict[str, Dict[str, float]] = {}
    
    # Define patterns to look for temperature and salinity columns
    temp_patterns = ["temperature", "min_temperature", "max_temperature", "avg_temperature", "mean_temperature"]
    sal_patterns = ["salinity", "min_salinity", "max_salinity", "avg_salinity", "mean_salinity"]
    
    # Handle temperature data
    temp_cols = [col for pattern in temp_patterns for col in df.columns if pattern.lower() in col.lower()]
    print(f"DEBUG: Found temperature columns: {temp_cols}")
    
    if temp_cols:
        temp_data = []
        for col in temp_cols:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            print(f"DEBUG: Column {col} has {len(series)} valid numeric values")
            if not series.empty:
                temp_data.extend(series.values)
        
        if temp_data:
            stats_out["temperature"] = {
                "min": round(float(min(temp_data)), 1),
                "max": round(float(max(temp_data)), 1),
                "mean": round(float(sum(temp_data) / len(temp_data)), 1),
            }
            print(f"DEBUG: Temperature stats: {stats_out['temperature']}")
    
    # Handle salinity data
    sal_cols = [col for pattern in sal_patterns for col in df.columns if pattern.lower() in col.lower()]
    print(f"DEBUG: Found salinity columns: {sal_cols}")
    
    if sal_cols:
        sal_data = []
        for col in sal_cols:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            print(f"DEBUG: Column {col} has {len(series)} valid numeric values")
            if not series.empty:
                sal_data.extend(series.values)
        
        if sal_data:
            stats_out["salinity"] = {
                "min": round(float(min(sal_data)), 1),
                "max": round(float(max(sal_data)), 1),
                "mean": round(float(sum(sal_data) / len(sal_data)), 1),
            }
            print(f"DEBUG: Salinity stats: {stats_out['salinity']}")

    if not stats_out:
        return "No valid temperature or salinity data.", {}

    parts = [f"{name}: min {s['min']}, max {s['max']}, avg {s['mean']}" for name, s in stats_out.items()]
    return "; ".join(parts), stats_out


def extract_stats_from_docs(docs: List[str]) -> Tuple[str, Dict[str, Dict[str, float]]]:
    if not docs:
        return "No data found.", {}

    temp_lows, temp_highs, sal_means = [], [], []
    temp_re = re.compile(r"Temperature range:\s*([0-9.]+)\s*-\s*([0-9.]+)\s*°C", re.I)
    sal_re = re.compile(r"Salinity mean:\s*([0-9.]+)\s*PSU", re.I)

    for doc in docs:
        t = temp_re.search(doc)
        if t:
            temp_lows.append(float(t.group(1)))
            temp_highs.append(float(t.group(2)))
        s = sal_re.search(doc)
        if s:
            sal_means.append(float(s.group(1)))

    stats_map, parts = {}, []
    if temp_lows and temp_highs:
        stats_map["temperature"] = {
            "min": round(min(temp_lows), 1),
            "max": round(max(temp_highs), 1),
            "mean": round(sum(temp_lows + temp_highs) / len(temp_lows + temp_highs), 1),
        }
        t = stats_map["temperature"]
        parts.append(f"temperature: min {t['min']}, max {t['max']}, avg {t['mean']}")
    if sal_means:
        stats_map["salinity"] = {
            "min": round(min(sal_means), 1),
            "max": round(max(sal_means), 1),
            "mean": round(sum(sal_means) / len(sal_means), 1),
        }
        s = stats_map["salinity"]
        parts.append(f"salinity: min {s['min']}, max {s['max']}, avg {s['mean']}")

    if not parts:
        return "No valid temperature or salinity data.", {}

    return "; ".join(parts), stats_map

# ------------------------------------------------------------
# Output cleaning + LLM summarizers
# ------------------------------------------------------------
def clean_llm_output(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    t = re.sub(r"^\s*(?:<s>\s*)?\[?OUT\]?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*(?:\[/?OUT\])\s*$", "", t, flags=re.I)
    t = re.sub(r"^\s*<s>\s*|\s*</s>\s*$", "", t, flags=re.I)
    return t.strip()


def summarize_with_llm_from_stats(llm, stats_text: str, query: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert oceanographer communicating with the general public. "
         "Interpret numeric oceanographic data in simple terms."),
        ("user", "User query: {query}\n\nStats: {stats}\n\nExplain conditions in 3–4 sentences.")
    ])
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"query": query, "stats": stats_text})
    return clean_llm_output(raw)


def create_context_summary_with_llm(llm, context: str, query: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an oceanographic expert. Summarize data patterns in 3–4 sentences."),
        ("user", "User query: {query}\n\nData:\n{context}")
    ])
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"query": query, "context": context})
    return clean_llm_output(raw)

# ------------------------------------------------------------
# SQL prompt + phenomenon prompt
# ------------------------------------------------------------
def create_phenomenon_focused_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", "Interpret stats to identify thermoclines, haloclines, mixing zones."),
        ("user", "Query: {query}\n\nStats: {stats}")
    ])


def create_enhanced_sql_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", "You are a SQL expert. Generate PostgreSQL queries for table argo_profiles. "
         "The table has columns: time, latitude, longitude, pressure, temperature, salinity, depth, oxygen, chlorophyll. "
         "Always include temperature and salinity columns in SELECT statements when they are relevant to the query."),
        ("user", "Context: {context}\n\nQuery: {query}\n\nGenerate SQL:")
    ])

# ------------------------------------------------------------
# Plot data helper
# ------------------------------------------------------------
def extract_plot_data(df: pd.DataFrame, query_type: str) -> dict:
    if df.empty:
        return None
    plot_data = {}
    
    # Check for temperature data with various column names
    temp_cols = [col for col in df.columns if 'temperature' in col.lower()]
    if temp_cols and "pressure" in df.columns:
        # Use the first temperature column found
        temp_col = temp_cols[0]
        depth_groups = df.groupby("pressure")[temp_col].agg(["mean", "min", "max"])
        plot_data["temperature_profile"] = {
            "depths": depth_groups.index.tolist(),
            "values": depth_groups["mean"].tolist(),
            "min": depth_groups["min"].tolist(),
            "max": depth_groups["max"].tolist()
        }
    
    # Check for salinity data with various column names
    sal_cols = [col for col in df.columns if 'salinity' in col.lower()]
    if sal_cols and "pressure" in df.columns:
        # Use the first salinity column found
        sal_col = sal_cols[0]
        depth_groups = df.groupby("pressure")[sal_col].agg(["mean", "min", "max"])
        plot_data["salinity_profile"] = {
            "depths": depth_groups.index.tolist(),
            "values": depth_groups["mean"].tolist(),
            "min": depth_groups["min"].tolist(),
            "max": depth_groups["max"].tolist()
        }
    
    return plot_data

# ------------------------------------------------------------
# Main RAG query pipeline
# ------------------------------------------------------------
def rag_query(
    user_query: str,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    db_uri: str = DB_URI,
    use_phenomenon_prompt: bool = False
) -> Tuple[str, Optional[Dict]]:
    try:
        print(f"DEBUG: Processing query: {user_query}")

        # Step 1: Retrieve context
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(collection_name)
        results = collection.query(query_texts=[user_query], n_results=5)
        docs = results.get("documents", [[]])[0] if results else []
        context = "\n".join(docs) if docs else "No context found."
        print(f"DEBUG: Retrieved {len(docs)} docs from ChromaDB")

        # Step 2: Initialize LLM
        llm = None
        if OPENROUTER_API_KEY:
            llm = ChatOpenAI(
                model_name="mistralai/mistral-7b-instruct:free",
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=OPENROUTER_API_KEY,
                temperature=0.1,
                max_tokens=512
            )
            print("DEBUG: LLM initialized")

        # Step 3: SQL generation (with caching)
        df_result = pd.DataFrame()
        sql_cache_key = hashlib.md5(user_query.encode()).hexdigest()
        if llm:
            if sql_cache_key in sql_cache:
                sql = sql_cache[sql_cache_key]
                print(f"DEBUG: Using cached SQL")
            else:
                sql_prompt = create_enhanced_sql_prompt()
                sql_chain = sql_prompt | llm | StrOutputParser()
                sql_response = sql_chain.invoke({"query": user_query, "context": context}).strip()
                sql_match = re.search(r"SELECT\s+.*?(?:;|$)", sql_response, re.I | re.S)
                sql = (sql_match.group(0).rstrip(";") if sql_match else sql_response).strip()
                sql_cache[sql_cache_key] = sql
                print(f"DEBUG: Generated SQL: {sql}")

            # Step 3b: Execute SQL
            try:
                df_result = run_sql(sql)
                print(f"DEBUG: SQL executed: {len(df_result)} rows")
            except Exception as e:
                print(f"WARNING: SQL failed: {e}")

        # Step 4: Visualization
        visualization_data = extract_plot_data(df_result, "profile") if not df_result.empty else None

        # Step 5: Summarization with enhanced oceanographic prompt
        stats_text, stats_map = extract_numeric_stats(df_result)
        print(f"DEBUG: Stats text: {stats_text}")

        if stats_text != "No data found." and stats_text != "No valid temperature or salinity data." and llm:
            # Use the enhanced oceanographic prompt that includes contextual summaries
              response_text = summarize_with_llm_from_stats(llm, stats_text, user_query)
        elif stats_text != "No data found." and stats_text != "No valid temperature or salinity data.":
            response_text = f"Summary: {stats_text}"
        else:
            response_text = "No matching data found."

        return response_text, visualization_data

    except Exception as e:
        traceback.print_exc()
        return f"Error: {e}", None