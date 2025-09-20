import hashlib

# Global SQL cache (in-memory dictionary)
sql_cache = {}

def get_available_dates(db_uri: str) -> list:
    """Get unique dates from argo_profiles table."""
    try:
        engine = create_engine(db_uri)
        with engine.connect() as conn:
            result = pd.read_sql("SELECT DISTINCT time::date FROM argo_profiles ORDER BY time", conn)
            return result['time'].astype(str).tolist()
    except Exception as e:
        print(f"WARNING: Failed to get available dates: {e}")
        return []  # Empty list if DB fails

def get_dataset_stats(db_uri: str) -> str:
    """Get overall min/max for temperature and salinity from the entire dataset."""
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
        return "temperature and salinity data"  # Generic fallback
    except Exception as e:
        print(f"WARNING: Failed to get dataset stats: {e}")
        return "temperature and salinity data"
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
from datetime import datetime

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
# Enhanced Summarization helpers
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


def create_enhanced_simple_summary_from_stats(stats_text: str, user_query: str) -> str:
    """
    Enhanced fallback summary when LLM is not available.
    Creates more interpretive summaries rather than just listing numbers.
    """
    if "No valid" in stats_text or "No data" in stats_text:
        return "No oceanographic data found matching your query."
    
    parts = []
    water_mass_indicators = []
    
    # Parse temperature data
    temp_match = re.search(r"temperature:\s*min\s*([\d.]+),\s*max\s*([\d.]+),\s*avg\s*([\d.]+)", stats_text.lower())
    if temp_match:
        min_t, max_t, avg_t = map(float, temp_match.groups())
        temp_range = max_t - min_t
        
        if temp_range > 15:
            parts.append(f"Water shows strong temperature stratification, ranging from {min_t}°C to {max_t}°C")
            water_mass_indicators.append("mixed water column")
        elif temp_range > 5:
            parts.append(f"Moderate temperature variation from {min_t}°C to {max_t}°C (avg: {avg_t}°C)")
            water_mass_indicators.append("some vertical mixing")
        else:
            parts.append(f"Relatively uniform temperature around {avg_t}°C")
            water_mass_indicators.append("stable water mass")
        
        # Temperature classification
        if avg_t > 25:
            water_mass_indicators.append("tropical surface water")
        elif avg_t > 15:
            water_mass_indicators.append("temperate water")
        elif avg_t > 5:
            water_mass_indicators.append("cool water mass")
        else:
            water_mass_indicators.append("cold/deep water")
    
    # Parse salinity data
    sal_match = re.search(r"salinity:\s*min\s*([\d.]+),\s*max\s*([\d.]+),\s*avg\s*([\d.]+)", stats_text.lower())
    if sal_match:
        min_s, max_s, avg_s = map(float, sal_match.groups())
        sal_range = max_s - min_s
        
        if sal_range < 0.5:
            parts.append(f"Salinity is very stable at {avg_s} PSU")
        elif sal_range < 2.0:
            parts.append(f"Salinity shows minor variation ({min_s}-{max_s} PSU)")
        else:
            parts.append(f"Salinity varies significantly from {min_s} to {max_s} PSU")
            water_mass_indicators.append("water mass mixing")
        
        # Salinity classification
        if avg_s > 35.5:
            water_mass_indicators.append("high-salinity oceanic water")
        elif avg_s > 34.5:
            water_mass_indicators.append("typical oceanic conditions")
        elif avg_s > 32:
            water_mass_indicators.append("coastal influence detected")
        else:
            water_mass_indicators.append("freshwater influence")
    
    # Combine into meaningful summary
    result = ". ".join(parts)
    if water_mass_indicators:
        unique_indicators = list(dict.fromkeys(water_mass_indicators))  # Remove duplicates while preserving order
        result += f". This suggests {', '.join(unique_indicators[:2])}"  # Limit to 2 key indicators
    
    return result + "."
# --- add this helper near the top with the other helpers ---
def clean_llm_output(text: str) -> str:
    """
    Remove common wrapper tokens produced by some LLM chains such as:
      <s> [OUT] ... [/OUT]
    Keeps the inner content and strips whitespace.
    """
    if not text:
        return text

    t = text.strip()

    # Remove opening wrapper like: <s> [OUT] or <s>[OUT] or [OUT]
    t = re.sub(r"^\s*(?:<s>\s*)?\[?OUT\]?\s*", "", t, flags=re.IGNORECASE)

    # Remove closing wrapper like: [/OUT] or [/out] or [OUT]
    t = re.sub(r"\s*(?:\[/?OUT\])\s*$", "", t, flags=re.IGNORECASE)

    # Also remove any stray leading/trailing <s> or </s>
    t = re.sub(r"^\s*<s>\s*|\s*</s>\s*$", "", t, flags=re.IGNORECASE)

    return t.strip()


# --- modify summarize_with_llm_from_stats to clean output ---
def summarize_with_llm_from_stats(llm, stats_text: str, query: str) -> str:
    """
    Enhanced prompt template for creating meaningful oceanographic summaries
    that focus on insights and interpretations rather than raw data repetition.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert oceanographer communicating with the general public. "
         "Your job is to interpret numeric oceanographic data and explain what it means "
         "in terms of ocean conditions and patterns.\n\n"
         "RULES:\n"
         "• Transform statistics into meaningful insights\n"
         "• Explain what temperature and salinity ranges indicate about water masses\n"
         "• Identify if conditions are typical, extreme, or noteworthy\n"
         "• Use accessible language - avoid technical jargon\n"
         "• Focus on 'what this tells us' rather than listing numbers\n"
         "• Keep response to 3-4 sentences maximum\n\n"
         "CONTEXT GUIDELINES:\n"
         "• Temperature ranges >10°C suggest mixing of different water masses\n"
         "• Salinity >35 PSU indicates oceanic water; <34 PSU suggests coastal influence\n"
         "• Stable salinity (variation <1 PSU) indicates uniform water mass\n"
         "• Cold temperatures (<5°C) suggest deep water or polar regions\n"
         "• Warm temperatures (>20°C) indicate surface tropical/subtropical water"),

        ("user",
         "USER QUESTION: {query}\n\n"
         "STATISTICAL DATA: {stats}\n\n"
         "Based on this data, provide a clear interpretation that answers:\n"
         "1. What do these temperature conditions tell us about the water?\n"
         "2. What does the salinity pattern indicate?\n"
         "3. What type of ocean environment or water mass does this represent?\n\n"
         "Write as if explaining to someone curious about ocean science but not an expert.")
    ])

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"query": query, "stats": stats_text})
    return clean_llm_output(raw).strip()


# --- modify create_context_summary_with_llm similarly ---
def create_context_summary_with_llm(llm, context: str, query: str) -> str:
    """
    Enhanced context summarization when no numerical stats are available.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an oceanographic expert. Analyze the provided oceanographic profile data "
         "and create a meaningful summary that identifies patterns, trends, and key insights. "
         "Focus on what the data reveals about ocean conditions rather than listing individual measurements.\n\n"
         "GUIDELINES:\n"
         "• Identify geographical patterns (if multiple locations)\n"
         "• Note temporal trends (if time series data)\n"
         "• Describe water mass characteristics\n"
         "• Highlight any unusual or notable conditions\n"
         "• Keep language accessible to non-experts\n"
         "• Limit to 3-4 sentences"),

        ("user",
         "USER QUERY: {query}\n\n"
         "OCEANOGRAPHIC DATA:\n{context}\n\n"
         "Provide a concise summary that explains what this data tells us about ocean conditions:")
    ])

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"query": query, "context": context})
    return clean_llm_output(raw).strip()




# ------------------------------------------------------------
# Alternative prompt templates for specific use cases
# ------------------------------------------------------------
def create_phenomenon_focused_prompt() -> ChatPromptTemplate:
    """
    Alternative prompt that focuses on identifying specific oceanographic phenomena.
    """
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are an oceanographic analyst. Interpret the statistical data to identify "
         "specific ocean phenomena and water mass characteristics.\n\n"
         
         "PHENOMENA TO IDENTIFY:\n"
         "• Thermoclines (sharp temperature gradients)\n"
         "• Haloclines (sharp salinity gradients)\n" 
         "• Water mass boundaries and mixing zones\n"
         "• Upwelling/downwelling signatures\n"
         "• Seasonal thermocline development\n"
         "• Coastal vs oceanic water characteristics\n\n"
         
         "RESPONSE FORMAT:\n"
         "1. Primary water mass type identified\n"
         "2. Key oceanographic process or phenomenon\n"
         "3. Confidence level and supporting evidence"),
        
        ("user",
         "Query: {query}\n"
         "Statistics: {stats}\n\n"
         "What oceanographic phenomenon does this data represent?")
    ])


def create_enhanced_sql_prompt() -> ChatPromptTemplate:
    """
    Enhanced SQL generation prompt with better oceanographic context.
    """
    return ChatPromptTemplate.from_messages([
        ("system", 
         "You are a SQL expert specializing in oceanographic data analysis. "
         "Generate efficient PostgreSQL queries for the argo_profiles table.\n\n"
         
         "TABLE SCHEMA:\n"
         "argo_profiles(time, latitude, longitude, pressure, temperature, salinity)\n\n"
         
         "QUERY OPTIMIZATION TIPS:\n"
         "• Use appropriate WHERE clauses for spatial/temporal filtering\n"
         "• Include statistical functions (AVG, MIN, MAX, STDDEV) for summaries\n"
         "• Consider pressure ranges for depth analysis\n"
         "• Use LIMIT for large result sets\n"
         "• Include ORDER BY for meaningful data organization\n\n"
         
         "Return ONLY the SQL query, no explanations."),
        
        ("user", 
         "Context from vector search: {context}\n\n"
         "User query: {query}\n\n"
         "Generate a PostgreSQL SELECT query:")
    ])

#fetching real data
# Add this function to extract plot data from SQL results
def extract_plot_data(df: pd.DataFrame, query_type: str) -> dict:
    """Extract data suitable for plotting from DataFrame"""
    if df.empty:
        return None
    
    plot_data = {}
    
    if "temperature" in df.columns and "pressure" in df.columns:
        # Group by pressure/depth and calculate stats
        depth_groups = df.groupby("pressure")["temperature"].agg(["mean", "min", "max"])
        plot_data["temperature_profile"] = {
            "depths": depth_groups.index.tolist(),
            "values": depth_groups["mean"].tolist(),
            "min": depth_groups["min"].tolist(),
            "max": depth_groups["max"].tolist()
        }
    
    if "salinity" in df.columns and "pressure" in df.columns:
        depth_groups = df.groupby("pressure")["salinity"].agg(["mean", "min", "max"])
        plot_data["salinity_profile"] = {
            "depths": depth_groups.index.tolist(),
            "values": depth_groups["mean"].tolist(),
            "min": depth_groups["min"].tolist(),
            "max": depth_groups["max"].tolist()
        }
    
    return plot_data

def rag_query(
    user_query: str,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    db_uri: str = DB_URI,
    use_phenomenon_prompt: bool = False
) -> Tuple[str, Optional[Dict]]:
    try:
        print(f"DEBUG: Processing query: {user_query}")
        
        # Step 1: ChromaDB retrieval (for SQL context only)
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(collection_name)
        results = collection.query(query_texts=[user_query], n_results=5)
        docs = results.get("documents", [[]])[0] if results else []
        context = "\n".join(docs) if docs else "No context found."
        print(f"DEBUG: Retrieved {len(docs)} documents from ChromaDB")

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
            print("DEBUG: LLM initialized successfully")

        # Step 3: SQL generation with caching
        df_result = pd.DataFrame()
        sql_cache_key = hashlib.md5(user_query.encode()).hexdigest()
        if llm:
            if sql_cache_key in sql_cache:
                sql = sql_cache[sql_cache_key]
                print(f"DEBUG: Using cached SQL: {sql}")
            else:
                sql_prompt = create_enhanced_sql_prompt()  # Use updated prompt below
                sql_chain = sql_prompt | llm | StrOutputParser()
                sql_response = sql_chain.invoke({"query": user_query, "context": context}).strip()
                sql_match = re.search(r"SELECT\s+.*?(?:;|$)", sql_response, re.IGNORECASE | re.DOTALL)
                sql = (sql_match.group(0).rstrip(";") if sql_match else sql_response).strip()
                # Adjust latitude for "near the equator" to match data (~8.8–9°N)
                if "equator" in user_query.lower():
                    sql = re.sub(r"latitude BETWEEN -5 AND 5", "latitude BETWEEN -10 AND 10", sql)
                sql_cache[sql_cache_key] = sql
                print(f"DEBUG: Generated and cached SQL: {sql}")

            # Fix invalid dates dynamically
            available_dates = get_available_dates(db_uri)
            if available_dates:
                date_pattern = r"\d{4}-\d{2}-\d{2}"
                dates_in_sql = re.findall(date_pattern, sql)
                for req_date in dates_in_sql:
                    if req_date not in available_dates:
                        print(f"DEBUG: No data for {req_date}; using available dates: {available_dates}")
                        dates_str = ", ".join(f"'{d}'" for d in available_dates)
                        sql = re.sub(
                            r"time::date = '\d{4}-\d{2}-\d{2}'|time BETWEEN '\d{4}-\d{2}-\d{2} 00:00:00' AND '\d{4}-\d{2}-\d{2} 23:59:59'",
                            f"time::date IN ({dates_str})",
                            sql
                        )
                        break
            
            try:
                df_result = run_sql(sql)
                print(f"DEBUG: SQL executed successfully, got {len(df_result)} rows")
                print(f"DEBUG: Columns: {df_result.columns.tolist()}")
            except Exception as e:
                print(f"WARNING: SQL execution failed: {e}")
                df_result = pd.DataFrame()

        # Step 4: Visualization data (only if SQL succeeded)
        visualization_data = None
        if not df_result.empty:
            if 'temperature' in df_result.columns and 'pressure' in df_result.columns:
                print(f"DEBUG: Temperature range: {df_result['temperature'].min()} to {df_result['temperature'].max()}")
                temp_by_depth = df_result.groupby('pressure')['temperature'].agg(['mean', 'min', 'max']).reset_index()
                visualization_data = {
                    "type": "temperature_profile",
                    "data": {
                        "depths": temp_by_depth['pressure'].tolist(),
                        "values": temp_by_depth['mean'].tolist(),
                        "min": temp_by_depth['min'].tolist(),
                        "max": temp_by_depth['max'].tolist()
                    },
                    "metadata": {"location": "Extracted from query", "parameter": "temperature"}
                }
            if 'salinity' in df_result.columns and 'pressure' in df_result.columns:
                print(f"DEBUG: Salinity range: {df_result['salinity'].min()} to {df_result['salinity'].max()}")
                sal_by_depth = df_result.groupby('pressure')['salinity'].agg(['mean', 'min', 'max']).reset_index()
                if visualization_data:
                    visualization_data["salinity_data"] = {
                        "depths": sal_by_depth['pressure'].tolist(),
                        "values": sal_by_depth['mean'].tolist(),
                        "min": sal_by_depth['min'].tolist(),
                        "max": sal_by_depth['max'].tolist()
                    }
                else:
                    visualization_data = {
                        "type": "salinity_profile",
                        "data": {
                            "depths": sal_by_depth['pressure'].tolist(),
                            "values": sal_by_depth['mean'].tolist(),
                            "min": sal_by_depth['min'].tolist(),
                            "max": sal_by_depth['max'].tolist()
                        },
                        "metadata": {"location": "Extracted from query", "parameter": "salinity"}
                    }

        # Step 5: Generate consistent response (only if SQL succeeded, no ChromaDB fallback for response)
        stats_text, stats_map = extract_numeric_stats(df_result)
        print(f"DEBUG: Stats text: {stats_text}")
        print(f"DEBUG: Stats map: {stats_map}")
        if stats_text != "No data found." and llm:
            # Add location context to LLM summary
            lat_range = (df_result['latitude'].min(), df_result['latitude'].max()) if 'latitude' in df_result.columns else (8.8, 9.0)
            lon_range = (df_result['longitude'].min(), df_result['longitude'].max()) if 'longitude' in df_result.columns else (66.8, 68.0)
            context_info = f"Data from tropical Indian Ocean (lat {lat_range[0]:.1f}–{lat_range[1]:.1f}°N, lon {lon_range[0]:.1f}–{lon_range[1]:.1f}°E)"
            response_text = summarize_with_llm_from_stats(llm, f"{stats_text}. {context_info}", user_query)
        elif stats_text != "No data found.":
            response_text = create_enhanced_simple_summary_from_stats(stats_text, user_query)
        else:
            # No fallback to ChromaDB; clear message instead
            available_dates = get_available_dates(db_uri)
            dates_desc = f" ({' and '.join(available_dates)})" if available_dates else ""
            response_text = (
                f"No data matching your query in the dataset. "
                f"Available data from 2025{dates_desc} shows {get_dataset_stats(db_uri)} in the tropical Indian Ocean (lat ~8.8–9°N, lon ~66.8–68°E). "
                "Try rephrasing with specific dates like March 11, 2025, or locations near 9°N 68°E."
            )
        
        if visualization_data:
            response_text += " Check the visualization for detailed profiles."
        else:
            response_text += " No visualization available due to lack of matching data."

        return response_text, visualization_data

    except Exception as e:
        print(f"ERROR in rag_query: {e}")
        traceback.print_exc()
        return f"An error occurred: {str(e)}", None

# ------------------------------------------------------------
# Enhanced CLI interactive mode
# ------------------------------------------------------------
def main():
    """
    Enhanced main function with better user interaction.
    """
    try:
        print("🌊 Setting up oceanographic data system...")
        inserted = populate_chroma_if_empty(PARQUET_PATH, CHROMA_PATH, COLLECTION_NAME)
        if inserted:
            print(f"✅ Populated {inserted} documents into '{COLLECTION_NAME}'.")
        else:
            print(f"✅ ChromaDB already contains data.")
    except Exception as e:
        print(f"❌ ERROR: Failed to set up ChromaDB: {e}")
        return

    print("\n🔬 === Interactive Oceanographic RAG System ===")
    print("Ask questions about ocean temperature, salinity, and water masses.")
    print("Example queries:")
    print("  • 'What are temperatures like in the tropical Pacific?'")
    print("  • 'Show me salinity data from the Atlantic Ocean'")
    print("  • 'Find cold water masses below 5 degrees'")
    print("\nCommands: 'quit' to exit, 'help' for more examples")
    print("-" * 60)
    
    while True:
        try:
            query = input("\n🌊 Query > ").strip()
            
            if query.lower() in ["quit", "exit", "q"]:
                break
            elif query.lower() == "help":
                print("\n📚 Example Queries:")
                print("  • Geographic: 'ocean data near 40°N 30°W'")
                print("  • Temperature: 'warm water above 20 degrees'")
                print("  • Salinity: 'high salinity regions'")
                print("  • Depth: 'surface water conditions'")
                print("  • Comparative: 'temperature differences in Atlantic vs Pacific'")
                print("  • Temporal: 'seasonal temperature changes'")
                continue
            elif not query:
                continue
                
            print("\n🔍 Processing...")
            response = rag_query(query)
            print(f"\n📊 Analysis:\n{response}\n")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error processing query: {e}")

    print("✅ Session ended. Thank you for exploring oceanographic data! 🌊")


if __name__ == "__main__":
    main()
