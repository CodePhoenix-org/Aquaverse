import os
import re
import traceback
from typing import List, Dict, Tuple
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
    Expected pattern in docs:
      "Temperature range: 3.0-28.2°C, Salinity mean: 35.4 PSU"
    """
    if not docs:
        return "No data found.", {}

    temp_lows: List[float] = []
    temp_highs: List[float] = []
    sal_means: List[float] = []

    temp_re = re.compile(r"Temperature range:\s*([0-9]+(?:\\.[0-9]+)?)\s*-\s*([0-9]+(?:\\.[0-9]+)?)\s*°C", re.IGNORECASE)
    sal_re = re.compile(r"Salinity mean:\s*([0-9]+(?:\\.[0-9]+)?)\s*PSU", re.IGNORECASE)

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


def summarize_with_llm_from_stats(llm: ChatOpenAI, stats_text: str, user_query: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an oceanographic data expert. Return concise summaries for non-experts."),
        ("user",
         "User Query: {query}\nComputed statistics: {stats}\n"
         "Instructions:\n- Summarize trends, ranges, typical values.\n- Do not repeat raw rows.")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": user_query, "stats": stats_text}).strip()


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
    - Generate SQL via LLM
    - Execute SQL on PostgreSQL
    - Summarize numeric stats using LLM
    """
    try:
        # Step 1: Retrieve context
        ensure_collection(chroma_path, collection_name)
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(collection_name)
        results = collection.query(query_texts=[user_query], n_results=5)
        docs = results.get("documents", [[]])[0] if results else []
        context = "\n".join(docs) or "No context found."

        # Step 2: Initialize LLM if API key available
        llm = ChatOpenAI(
            model_name="mistralai/mistral-7b-instruct:free",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=OPENROUTER_API_KEY,
            temperature=0.5,
            max_tokens=512
        ) if OPENROUTER_API_KEY else None

        # Step 3: Generate SQL if LLM available
        sql = None
        if llm:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Return only a valid PostgreSQL SELECT query."),
                ("user", "Context: {context}\nUser query: {query}")
            ])
            chain = prompt | llm | StrOutputParser()
            sql = chain.invoke({"query": user_query, "context": context}).strip()
            # Extract single SELECT
            m = re.search(r"SELECT\s+.*?(?:;|$)", sql, re.IGNORECASE | re.DOTALL)
            sql = (m.group(0).rstrip(";") if m else sql).strip()

        # Step 4: Execute SQL
        df_result = pd.DataFrame()
        if sql:
            try:
                engine = create_engine(db_uri)
                df_result = pd.read_sql(text(sql), engine)
            except Exception as e:
                print(f"SQL execution failed: {e}")

        # Step 5: Summarize numeric stats
        stats_text, stats_map = extract_numeric_stats(df_result)
        if df_result.empty or not stats_map:
            # Fallback: derive stats from retrieved docs and summarize with LLM if available
            stats_text_docs, stats_map_docs = extract_stats_from_docs(docs)
            if stats_map_docs:
                if llm:
                    return summarize_with_llm_from_stats(llm, stats_text_docs, user_query)
                return f"Summary (no LLM): {stats_text_docs}"
            return context

        if llm:
            return summarize_with_llm_from_stats(llm, stats_text, user_query)
        return f"Summary (no LLM): {stats_text}"

    except Exception as e:
        traceback.print_exc()
        return f"Error in rag_query: {e}"

# ------------------------------------------------------------
# CLI interactive mode
# ------------------------------------------------------------
def main():
    inserted = populate_chroma_if_empty(PARQUET_PATH, CHROMA_PATH, COLLECTION_NAME)
    if inserted:
        print(f"Populated {inserted} documents into '{COLLECTION_NAME}'.")

    print("\n=== Interactive RAG CLI ===")
    while True:
        query = input("> ").strip()
        if query.lower() == "quit":
            break
        response = rag_query(query)
        print(f"\nResponse:\n{response}\n")

    print("✅ Interactive session ended.")


if __name__ == "__main__":
    main()
