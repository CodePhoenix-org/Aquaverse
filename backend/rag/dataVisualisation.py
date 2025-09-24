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
from dotenv import load_dotenv
load_dotenv()

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
CHROMA_PATH = os.getenv("chromapath")
COLLECTION_NAME = os.getenv("collectioname")

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
    """Get overall min/max for all parameters from dataset."""
    try:
        sql = """
        SELECT 
            MIN(temperature) AS min_temp, MAX(temperature) AS max_temp,
            MIN(salinity) AS min_sal, MAX(salinity) AS max_sal,
            MIN(chlorophyll) AS min_chl, MAX(chlorophyll) AS max_chl,
            MIN(oxygen) AS min_o2, MAX(oxygen) AS max_o2,
            MIN(depth) AS min_depth, MAX(depth) AS max_depth
        FROM argo_profiles
        """
        engine = create_engine(db_uri)
        df = pd.read_sql(text(sql), engine)
        if not df.empty:
            stats = []
            if pd.notna(df['min_temp'].iloc[0]):
                stats.append(f"temperatures ranging from ~{df['min_temp'].iloc[0]:.1f}°C to ~{df['max_temp'].iloc[0]:.1f}°C")
            if pd.notna(df['min_sal'].iloc[0]):
                stats.append(f"salinity from ~{df['min_sal'].iloc[0]:.1f} to ~{df['max_sal'].iloc[0]:.1f} PSU")
            if pd.notna(df['min_chl'].iloc[0]):
                stats.append(f"chlorophyll from ~{df['min_chl'].iloc[0]:.3f} to ~{df['max_chl'].iloc[0]:.3f} mg/m³")
            if pd.notna(df['min_o2'].iloc[0]):
                stats.append(f"oxygen from ~{df['min_o2'].iloc[0]:.1f} to ~{df['max_o2'].iloc[0]:.1f} μmol/kg")
            if pd.notna(df['min_depth'].iloc[0]):
                stats.append(f"depth from ~{df['min_depth'].iloc[0]:.1f} to ~{df['max_depth'].iloc[0]:.1f} m")
            return ", ".join(stats) if stats else "oceanographic data"
        return "oceanographic data"
    except Exception as e:
        print(f"WARNING: Failed to get dataset stats: {e}")
        return "oceanographic data"

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
        summary_parts = [f"Argo profile at {float(lat):.2f} lat, {float(lon):.2f} lon on {time}."]
        
        if 'temperature' in group.columns:
            temp_range = f"Temperature range: {float(group['temperature'].min()):.1f}-{float(group['temperature'].max()):.1f}°C"
            summary_parts.append(temp_range)
        
        if 'salinity' in group.columns:
            sal_mean = f"Salinity mean: {float(group['salinity'].mean()):.1f} PSU"
            summary_parts.append(sal_mean)
        
        if 'chlorophyll' in group.columns and group['chlorophyll'].notna().any():
            chl_mean = f"Chlorophyll mean: {float(group['chlorophyll'].mean()):.3f} mg/m³"
            summary_parts.append(chl_mean)
        
        if 'oxygen' in group.columns and group['oxygen'].notna().any():
            o2_mean = f"Oxygen mean: {float(group['oxygen'].mean()):.1f} μmol/kg"
            summary_parts.append(o2_mean)
        
        if 'depth' in group.columns and group['depth'].notna().any():
            depth_range = f"Depth range: {float(group['depth'].min()):.1f}-{float(group['depth'].max()):.1f} m"
            summary_parts.append(depth_range)
        
        summary = " ".join(summary_parts)
        ids.append(str(hash(key)))
        docs.append(summary)

    if docs:
        embeddings = model.encode(docs).tolist()
        collection.add(ids=ids, documents=docs, embeddings=embeddings)

    return len(docs)

# ------------------------------------------------------------
# Parameter utilities
# ------------------------------------------------------------
def get_parameter_unit(parameter: str) -> str:
    """Return appropriate unit for each parameter"""
    units = {
        "temperature": "°C",
        "salinity": " PSU",
        "chlorophyll": " mg/m³",
        "oxygen": " μmol/kg",
        "pressure": " dbar",
        "depth": " m",
        "latitude": " degrees",
        "longitude": " degrees"
    }
    return units.get(parameter, "")

def detect_requested_parameter(query: str) -> List[str]:
    """Detect which parameters are being requested - returns list of parameters"""
    query_lower = query.lower()
    requested_params = []
    
    parameter_keywords = {
        "temperature": ["temperature", "temp", "sst", "sea surface temperature", "thermal"],
        "salinity": ["salinity", "sal", "salt", "psu"],
        "chlorophyll": ["chlorophyll", "chl", "chla", "chl-a", "phytoplankton"],
        "oxygen": ["oxygen", "o2", "dissolved oxygen", "do"],
        "pressure": ["pressure", "press", "depth pressure"],
        "depth": ["depth", "bathymetry"],
        "latitude": ["latitude", "lat"],
        "longitude": ["longitude", "lon", "long"]
    }
    
    for param, keywords in parameter_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            requested_params.append(param)
    
    if not requested_params:
        if "profile" in query_lower or "vertical" in query_lower:
            requested_params = ["temperature", "salinity", "chlorophyll", "oxygen", "depth"]
        else:
            requested_params = ["temperature", "salinity", "chlorophyll", "oxygen", "depth"]
    
    print(f"DEBUG: Detected parameters: {requested_params}")
    return requested_params

# ------------------------------------------------------------
# Enhanced Stats helpers
# ------------------------------------------------------------
def extract_numeric_stats(df: pd.DataFrame) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Enhanced stats extraction for all oceanographic parameters"""
    if df.empty:
        return "No data found.", {}

    print(f"DEBUG: DataFrame shape: {df.shape}")
    print(f"DEBUG: DataFrame columns: {list(df.columns)}")
    print(f"DEBUG: DataFrame sample:\n{df.head()}")

    stats_out: Dict[str, Dict[str, float]] = {}
    lower_map = {col.lower(): col for col in df.columns}
    
    parameter_patterns = {
        "temperature": ["temperature", "temp"],
        "salinity": ["salinity", "sal"],
        "chlorophyll": ["chlorophyll", "chl", "chla", "chl_a"],
        "oxygen": ["oxygen", "o2", "dissolved_oxygen"],
        "pressure": ["pressure", "press"],
        "depth": ["depth"],
        "latitude": ["latitude", "lat"],
        "longitude": ["longitude", "lon"]
    }
    
    for param_name, patterns in parameter_patterns.items():
        param_cols = []
        for pattern in patterns:
            param_cols.extend([col for col in df.columns if pattern.lower() in col.lower()])
        
        param_cols = list(dict.fromkeys(param_cols))
        print(f"DEBUG: Found {param_name} columns: {param_cols}")
        
        if param_cols:
            param_data = []
            for col in param_cols:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                print(f"DEBUG: Column {col} has {len(series)} valid numeric values")
                if not series.empty:
                    param_data.extend(series.values)
            
            if param_data:
                stats_out[param_name] = {
                    "min": round(float(min(param_data)), 3),
                    "max": round(float(max(param_data)), 3),
                    "mean": round(float(sum(param_data) / len(param_data)), 3),
                    "count": len(param_data)
                }
        else:
            agg_prefixes = ["avg_", "min_", "max_", "mean_"]
            for prefix in agg_prefixes:
                agg_col = f"{prefix}{param_name}"
                if agg_col in lower_map:
                    col = lower_map[agg_col]
                    series = pd.to_numeric(df[col], errors="coerce").dropna()
                    if not series.empty:
                        if param_name not in stats_out:
                            stats_out[param_name] = {}
                        if prefix == "min_":
                            stats_out[param_name]["min"] = round(float(series.min()), 1)
                        elif prefix == "max_":
                            stats_out[param_name]["max"] = round(float(series.max()), 1)
                        elif prefix in ["avg_", "mean_"]:
                            stats_out[param_name]["mean"] = round(float(series.mean()), 1)

    if not stats_out:
        return "No valid oceanographic data found.", {}

    parts = []
    for name, s in stats_out.items():
        if "min" in s and "max" in s and "mean" in s:
            parts.append(f"{name}: min {s['min']}, max {s['max']}, avg {s['mean']}")
        elif "min" in s and "max" in s:
            parts.append(f"{name}: min {s['min']}, max {s['max']}")
        elif "mean" in s:
            parts.append(f"{name}: avg {s['mean']}")

    return "; ".join(parts), stats_out

def extract_stats_from_docs(docs: List[str]) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Enhanced document stats extraction for all parameters"""
    if not docs:
        return "No data found.", {}

    temp_lows, temp_highs, sal_means, chl_means, o2_means, depth_lows, depth_highs = [], [], [], [], [], [], []
    
    temp_re = re.compile(r"Temperature range:\s*([0-9.]+)\s*-\s*([0-9.]+)\s*°C", re.I)
    sal_re = re.compile(r"Salinity mean:\s*([0-9.]+)\s*PSU", re.I)
    chl_re = re.compile(r"Chlorophyll mean:\s*([0-9.]+)\s*mg/m³", re.I)
    o2_re = re.compile(r"Oxygen mean:\s*([0-9.]+)\s*μmol/kg", re.I)
    depth_re = re.compile(r"Depth range:\s*([0-9.]+)\s*-\s*([0-9.]+)\s*m", re.I)

    for doc in docs:
        t = temp_re.search(doc)
        if t:
            temp_lows.append(float(t.group(1)))
            temp_highs.append(float(t.group(2)))
        
        s = sal_re.search(doc)
        if s:
            sal_means.append(float(s.group(1)))
        
        c = chl_re.search(doc)
        if c:
            chl_means.append(float(c.group(1)))
        
        o = o2_re.search(doc)
        if o:
            o2_means.append(float(o.group(1)))
        
        d = depth_re.search(doc)
        if d:
            depth_lows.append(float(d.group(1)))
            depth_highs.append(float(d.group(2)))

    stats_map, parts = {}, []
    
    if temp_lows and temp_highs:
        all_temps = temp_lows + temp_highs
        stats_map["temperature"] = {
            "min": round(min(temp_lows), 1),
            "max": round(max(temp_highs), 1),
            "mean": round(sum(all_temps) / len(all_temps), 1),
        }
        t = stats_map["temperature"]
        parts.append(f"temperature: min {t['min']}°C, max {t['max']}°C, avg {t['mean']}°C")
    
    if sal_means:
        stats_map["salinity"] = {
            "min": round(min(sal_means), 1),
            "max": round(max(sal_means), 1),
            "mean": round(sum(sal_means) / len(sal_means), 1),
        }
        s = stats_map["salinity"]
        parts.append(f"salinity: min {s['min']} PSU, max {s['max']} PSU, avg {s['mean']} PSU")
    
    if chl_means:
        stats_map["chlorophyll"] = {
            "min": round(min(chl_means), 3),
            "max": round(max(chl_means), 3),
            "mean": round(sum(chl_means) / len(chl_means), 3),
        }
        c = stats_map["chlorophyll"]
        parts.append(f"chlorophyll: min {c['min']} mg/m³, max {c['max']} mg/m³, avg {c['mean']} mg/m³")
    
    if o2_means:
        stats_map["oxygen"] = {
            "min": round(min(o2_means), 1),
            "max": round(max(o2_means), 1),
            "mean": round(sum(o2_means) / len(o2_means), 1),
        }
        o = stats_map["oxygen"]
        parts.append(f"oxygen: min {o['min']} μmol/kg, max {o['max']} μmol/kg, avg {o['mean']} μmol/kg")
    
    if depth_lows and depth_highs:
        all_depths = depth_lows + depth_highs
        stats_map["depth"] = {
            "min": round(min(depth_lows), 1),
            "max": round(max(depth_highs), 1),
            "mean": round(sum(all_depths) / len(all_depths), 1),
        }
        d = stats_map["depth"]
        parts.append(f"depth: min {d['min']} m, max {d['max']} m, avg {d['mean']} m")

    if not parts:
        return "No valid oceanographic data found.", {}

    return "; ".join(parts), stats_map

# ------------------------------------------------------------
# Output cleaning + Enhanced LLM summarizers
# ------------------------------------------------------------
def clean_llm_output(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    t = re.sub(r"^\s*(?:<s>\s*)?\[?OUT\]?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*(?:\[/?OUT\])\s*$", "", t, flags=re.I)
    t = re.sub(r"^\s*<s>\s*|\s*</s>\s*$", "", t, flags=re.I)
    return t.strip()

def summarize_parameter_specific_data(llm, stats_text: str, query: str, parameter_focus: str = None) -> str:
    """Enhanced summarization that focuses on the requested parameter"""
    if not parameter_focus:
        parameter_focus = detect_requested_parameter(query)
    
    if parameter_focus:
        system_message = f"""You are an expert oceanographer. The user asked specifically about {parameter_focus}. 
        Focus your response ONLY on {parameter_focus} data. Provide the specific numerical values with proper units and explain their oceanographic significance in 2-3 sentences.
        Do not mention other parameters unless directly relevant to understanding {parameter_focus}."""
    else:
        system_message = """You are an expert oceanographer. Analyze the provided oceanographic data and give a concise summary focusing on the most relevant parameters for the user's query. Use proper units and explain the oceanographic significance."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "User query: {query}\n\nData: {stats}\n\nProvide a focused response:")
    ])
    
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"query": query, "stats": stats_text})
    return clean_llm_output(raw)

def create_context_summary_with_llm(llm, context: str, query: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an oceanographic expert. Summarize data patterns in 3–4 sentences focusing on the user's specific request."),
        ("user", "User query: {query}\n\nData:\n{context}")
    ])
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"query": query, "context": context})
    return clean_llm_output(raw)

# ------------------------------------------------------------
# Enhanced SQL prompts
# ------------------------------------------------------------
def create_phenomenon_focused_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", "Interpret oceanographic stats to identify thermoclines, haloclines, mixing zones, and biological patterns."),
        ("user", "Query: {query}\n\nStats: {stats}")
    ])

def create_parameter_specific_sql_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", 
         "You are a SQL expert for oceanographic data. Generate PostgreSQL queries for table argo_profiles. "
         "The table has columns: time, latitude, longitude, pressure, temperature, salinity, depth, oxygen, chlorophyll. "
         "IMPORTANT RULES: "
         "1. When user asks for a specific parameter (like chlorophyll, oxygen, etc.), SELECT ONLY that parameter plus location/time columns "
         "2. Do NOT include temperature and salinity unless specifically requested "
         "3. For Indian Ocean: use latitude BETWEEN -40 AND 30 AND longitude BETWEEN 20 AND 120 "
         "4. For other oceans, use appropriate lat/lon bounds "
         "5. Always filter out NULL values for the requested parameter using WHERE column_name IS NOT NULL "
         "6. Include recent data by adding ORDER BY time DESC LIMIT clause when appropriate"),
        ("user", "Query: {query}\n\nContext: {context}\n\nGenerate SQL that focuses ONLY on the requested parameter:")
    ])

def create_enhanced_sql_prompt() -> ChatPromptTemplate:
    """Fallback SQL prompt for general queries"""
    return ChatPromptTemplate.from_messages([
        ("system", "You are a SQL expert. Generate PostgreSQL queries for table argo_profiles. "
         "The table has columns: time, latitude, longitude, pressure, temperature, salinity, depth, oxygen, chlorophyll. "
         "Focus on the parameters most relevant to the user's query."),
        ("user", "Context: {context}\n\nQuery: {query}\n\nGenerate SQL:")
    ])

# ------------------------------------------------------------
# Enhanced Plot data helper
# ------------------------------------------------------------
def extract_plot_data_enhanced(df: pd.DataFrame, query: str) -> dict:
    """Enhanced plot data extraction for all parameters"""
    if df.empty:
        return None
    
    plot_data = {}
    requested_params = detect_requested_parameter(query)
    target_params = [p for p in requested_params if p in ["temperature", "salinity", "chlorophyll", "oxygen", "depth"]]
    
    for param in target_params:
        param_cols = [col for col in df.columns if param.lower() in col.lower()]
        if param_cols and "pressure" in df.columns:
            param_col = param_cols[0]
            try:
                clean_data = df[['pressure', param_col]].dropna()
                if not clean_data.empty:
                    clean_data['pressure_rounded'] = clean_data['pressure'].round().astype(int)
                    profile = clean_data.groupby('pressure_rounded')[param_col].agg(['mean', 'min', 'max']).reset_index()
                    profile = profile[profile['count'] > 0]
                    if not profile.empty:
                        plot_data[param] = {
                            "depths": profile['pressure_rounded'].tolist(),
                            "values": profile['mean'].tolist(),
                            "min": profile['min'].tolist(),
                            "max": profile['max'].tolist(),
                            "title": f"{param.capitalize()} Profile - Indian Ocean",
                            "yLabel": f"{param.capitalize()} ({get_parameter_unit(param)})",
                            "color": {
                                "temperature": "#ff6b6b",
                                "salinity": "#4ecdc4",
                                "chlorophyll": "#2ecc71",
                                "oxygen": "#3498db",
                                "depth": "#9b59b6"
                            }.get(param, "#000000")
                        }
            except Exception as e:
                print(f"DEBUG: Failed to create {param} profile: {e}")
    
    return plot_data if plot_data else None

def extract_plot_data(df: pd.DataFrame, query_type: str) -> dict:
    """Legacy plot data function for backward compatibility"""
    return extract_plot_data_enhanced(df, query_type)

# ------------------------------------------------------------
# Main Enhanced RAG query pipeline
# ------------------------------------------------------------
def rag_query(
    user_query: str,
    chroma_path: str = None,
    collection_name: str = None,
    db_uri: str = None,
    use_phenomenon_prompt: bool = False
) -> Tuple[str, Optional[Dict]]:
    """Enhanced RAG query pipeline with parameter-specific handling"""
    try:
        print(f"DEBUG: Processing query: {user_query}")
        
        if chroma_path is None:
            chroma_path = CHROMA_PATH
        if collection_name is None:
            collection_name = COLLECTION_NAME
        if db_uri is None:
            db_uri = DB_URI
        
        print(f"DEBUG: Using chroma_path: {chroma_path}")
        print(f"DEBUG: Using collection_name: {collection_name}")
        print(f"DEBUG: Using db_uri: {db_uri}")

        requested_params = detect_requested_parameter(user_query)
        print(f"DEBUG: Detected parameters: {requested_params}")

        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(collection_name)
        results = collection.query(query_texts=[user_query], n_results=5)
        docs = results.get("documents", [[]])[0] if results else []
        context = "\n".join(docs) if docs else "No context found."
        print(f"DEBUG: Retrieved {len(docs)} docs from ChromaDB")

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

        df_result = pd.DataFrame()
        sql_cache_key = hashlib.md5(user_query.encode()).hexdigest()

        if requested_params:
            select_columns = ["pressure", "latitude", "longitude", "time"]
            for param in requested_params:
                if param in ["temperature", "salinity", "chlorophyll", "oxygen", "depth"]:
                    select_columns.append(param)
            select_clause = ", ".join(select_columns)

            where_conditions = ["1=1"]
            if "indian" in user_query.lower() or "india" in user_query.lower():
                where_conditions.append("longitude BETWEEN 20 AND 120")
                where_conditions.append("latitude BETWEEN -40 AND 30")

            month_mapping = {
                "march": 3, "april": 4, "may": 5, "june": 6,
                "july": 7, "august": 8, "september": 9, "october": 10,
                "november": 11, "december": 12, "january": 1, "february": 2
            }
            for month_name, month_num in month_mapping.items():
                if month_name in user_query.lower():
                    where_conditions.append(f"EXTRACT(MONTH FROM time) = {month_num}")
                    break

            for param in requested_params:
                if param in ["temperature", "salinity", "chlorophyll", "oxygen", "depth"]:
                    where_conditions.append(f"{param} IS NOT NULL")

            where_clause = " AND ".join(where_conditions)
            sql = f"""
            SELECT {select_clause}
            FROM argo_profiles
            WHERE {where_clause}
            ORDER BY time DESC, pressure
            LIMIT 2000
            """
            print(f"DEBUG: Generated SQL: {sql}")
        elif llm:
            if sql_cache_key in sql_cache:
                sql = sql_cache[sql_cache_key]
                print(f"DEBUG: Using cached SQL")
            else:
                sql_prompt = create_parameter_specific_sql_prompt()
                sql_chain = sql_prompt | llm | StrOutputParser()
                sql_response = sql_chain.invoke({"query": user_query, "context": context}).strip()
                sql_match = re.search(r"SELECT\s+.*?(?:;|$)", sql_response, re.I | re.S)
                sql = (sql_match.group(0).rstrip(";") if sql_match else sql_response).strip()
                sql_cache[sql_cache_key] = sql
                print(f"DEBUG: Generated SQL: {sql}")

        available_dates = get_available_dates(db_uri)
        if 'sql' in locals() and available_dates:
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

        visualization_data = None
        if not df_result.empty:
            plot_data = {}
            available_params = []
            print(f"DEBUG: DataFrame columns: {df_result.columns.tolist()}")
            print(f"DEBUG: DataFrame shape: {df_result.shape}")
            print(f"DEBUG: Pressure values sample: {df_result['pressure'].head(10).tolist() if 'pressure' in df_result.columns else 'No pressure column'}")

            # Process temperature data
            if 'temperature' in df_result.columns and 'pressure' in df_result.columns:
                temp_clean = df_result[['pressure', 'temperature']].dropna()
                if not temp_clean.empty:
                    temp_clean['pressure_rounded'] = temp_clean['pressure'].round().astype(int)
                    temp_profile = temp_clean.groupby('pressure_rounded')['temperature'].agg(['mean', 'count']).reset_index()
                    temp_profile = temp_profile[temp_profile['count'] > 0]
                    if not temp_profile.empty:
                        plot_data["temperature"] = {
                            "depths": temp_profile['pressure_rounded'].tolist(),
                            "values": temp_profile['mean'].tolist(),
                            "title": "Temperature Profile - Indian Ocean",
                            "yLabel": "Temperature (°C)",
                            "color": "#ff6b6b"
                        }
                        available_params.append("temperature")
                        print(f"DEBUG: Temperature profile created with {len(temp_profile)} depth points")

            # Process salinity data
            if 'salinity' in df_result.columns and 'pressure' in df_result.columns:
                sal_clean = df_result[['pressure', 'salinity']].dropna()
                if not sal_clean.empty:
                    sal_clean['pressure_rounded'] = sal_clean['pressure'].round().astype(int)
                    sal_profile = sal_clean.groupby('pressure_rounded')['salinity'].agg(['mean', 'count']).reset_index()
                    sal_profile = sal_profile[sal_profile['count'] > 0]
                    if not sal_profile.empty:
                        plot_data["salinity"] = {
                            "depths": sal_profile['pressure_rounded'].tolist(),
                            "values": sal_profile['mean'].tolist(),
                            "title": "Salinity Profile - Indian Ocean",
                            "yLabel": "Salinity (PSU)",
                            "color": "#4ecdc4"
                        }
                        available_params.append("salinity")
                        print(f"DEBUG: Salinity profile created with {len(sal_profile)} depth points")

            # Process chlorophyll data
            if 'chlorophyll' in df_result.columns and 'pressure' in df_result.columns:
                chl_clean = df_result[['pressure', 'chlorophyll']].dropna()
                if not chl_clean.empty:
                    chl_clean['pressure_rounded'] = chl_clean['pressure'].round().astype(int)
                    chl_profile = chl_clean.groupby('pressure_rounded')['chlorophyll'].agg(['mean', 'count']).reset_index()
                    chl_profile = chl_profile[chl_profile['count'] > 0]
                    if not chl_profile.empty:
                        plot_data["chlorophyll"] = {
                            "depths": chl_profile['pressure_rounded'].tolist(),
                            "values": chl_profile['mean'].tolist(),
                            "title": "Chlorophyll Profile - Indian Ocean",
                            "yLabel": "Chlorophyll (mg/m³)",
                            "color": "#2ecc71"
                        }
                        available_params.append("chlorophyll")
                        print(f"DEBUG: Chlorophyll profile created with {len(chl_profile)} depth points")

            # Process oxygen data
            if 'oxygen' in df_result.columns and 'pressure' in df_result.columns:
                o2_clean = df_result[['pressure', 'oxygen']].dropna()
                if not o2_clean.empty:
                    o2_clean['pressure_rounded'] = o2_clean['pressure'].round().astype(int)
                    o2_profile = o2_clean.groupby('pressure_rounded')['oxygen'].agg(['mean', 'count']).reset_index()
                    o2_profile = o2_profile[o2_profile['count'] > 0]
                    if not o2_profile.empty:
                        plot_data["oxygen"] = {
                            "depths": o2_profile['pressure_rounded'].tolist(),
                            "values": o2_profile['mean'].tolist(),
                            "title": "Oxygen Profile - Indian Ocean",
                            "yLabel": "Oxygen (μmol/kg)",
                            "color": "#3498db"
                        }
                        available_params.append("oxygen")
                        print(f"DEBUG: Oxygen profile created with {len(o2_profile)} depth points")

            # Process depth data
            if 'depth' in df_result.columns and 'pressure' in df_result.columns:
                depth_clean = df_result[['pressure', 'depth']].dropna()
                if not depth_clean.empty:
                    depth_clean['pressure_rounded'] = depth_clean['pressure'].round().astype(int)
                    depth_profile = depth_clean.groupby('pressure_rounded')['depth'].agg(['mean', 'count']).reset_index()
                    depth_profile = depth_profile[depth_profile['count'] > 0]
                    if not depth_profile.empty:
                        plot_data["depth"] = {
                            "depths": depth_profile['pressure_rounded'].tolist(),
                            "values": depth_profile['mean'].tolist(),
                            "title": "Depth Profile - Indian Ocean",
                            "yLabel": "Depth (m)",
                            "color": "#9b59b6"
                        }
                        available_params.append("depth")
                        print(f"DEBUG: Depth profile created with {len(depth_profile)} depth points")

            # Include latitude and longitude in metadata
            lat_lon_stats = {}
            if 'latitude' in df_result.columns and 'longitude' in df_result.columns:
                lat_lon_stats = {
                    "latitude": {
                        "min": round(float(df_result['latitude'].min()), 2),
                        "max": round(float(df_result['latitude'].max()), 2),
                        "mean": round(float(df_result['latitude'].mean()), 2)
                    },
                    "longitude": {
                        "min": round(float(df_result['longitude'].min()), 2),
                        "max": round(float(df_result['longitude'].max()), 2),
                        "mean": round(float(df_result['longitude'].mean()), 2)
                    }
                }

            print(f"DEBUG: Available parameters after processing: {available_params}")
            print(f"DEBUG: Plot data keys: {list(plot_data.keys())}")

            if plot_data:
                query_lower = user_query.lower()
                if any(p in query_lower for p in ["temperature", "salinity", "chlorophyll", "oxygen", "depth"]) and len(available_params) > 1:
                    visualization_data = {
                        "type": "multiple_profiles",
                        "available_params": available_params,
                        "all_data": plot_data,
                        "metadata": {
                            "query": user_query,
                            "parameters": available_params,
                            "timestamp": datetime.now().isoformat(),
                            "lat_lon_stats": lat_lon_stats,
                            "response_note": f"Profiles available for {', '.join(available_params)}"
                        }
                    }
                else:
                    primary_param = None
                    for param in ["chlorophyll", "oxygen", "depth", "temperature", "salinity"]:
                        if param in query_lower and param in available_params:
                            primary_param = param
                            break
                    if not primary_param and available_params:
                        primary_param = available_params[0]
                    if primary_param:
                        visualization_data = {
                            "type": f"{primary_param}_profile",
                            "data": plot_data[primary_param],
                            "available_params": available_params,
                            "all_data": plot_data,
                            "metadata": {
                                "query": user_query,
                                "parameter": primary_param,
                                "timestamp": datetime.now().isoformat(),
                                "lat_lon_stats": lat_lon_stats
                            }
                        }
                print(f"DEBUG: Final visualization_data structure: {visualization_data['type'] if visualization_data else 'None'}")

        stats_text, stats_map = extract_numeric_stats(df_result)
        print(f"DEBUG: Stats text: {stats_text}")

        if stats_text not in ["No data found.", "No valid oceanographic data found."] and llm:
            response_text = summarize_parameter_specific_data(llm, stats_text, user_query, requested_params)
        elif stats_text not in ["No data found.", "No valid oceanographic data found."]:
            response_text = f"Data summary: {stats_text}"
        else:
            if requested_params:
                response_text = f"No {', '.join(requested_params)} data found in the current dataset for the specified region/time period. The dataset may not contain these measurements for your query parameters."
            else:
                response_text = "No matching data found for your query in the current dataset."

        return response_text, visualization_data

    except Exception as e:
        traceback.print_exc()
        return f"Error processing query: {e}", None