# Updated Pipeline with Oxygen, Chlorophyll, Depth and LLM-safe summary
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
from langchain_core.prompts import ChatPromptTemplate
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
            f"Salinity mean: {float(group['salinity'].mean()):.1f} PSU, "
            f"Oxygen mean: {float(group['oxygen'].mean()):.1f} μmol/kg, "
            f"Chlorophyll mean: {float(group['chlorophyll'].mean()):.2f} mg/m³, "
            f"Depth range: {float(group['depth'].min()):.1f}-{float(group['depth'].max()):.1f} m"
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
    if df.empty:
        return "No data found.", {}

    lower_map = {c.lower(): c for c in df.columns}
    targets = ["temperature", "salinity", "oxygen", "chlorophyll", "depth"]
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
        return "No valid numeric data.", {}

    parts = [f"{name}: min {s['min']}, max {s['max']}, avg {s['mean']}" for name, s in stats_out.items()]
    return "; ".join(parts), stats_out

def extract_stats_from_docs(docs: List[str]) -> Tuple[str, Dict[str, Dict[str, float]]]:
    if not docs:
        return "No data found.", {}

    temp_lows, temp_highs, sal_means, oxy_means, chl_means, depth_mins, depth_maxs = [], [], [], [], [], [], []

    temp_re = re.compile(r"Temperature range:\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*°C", re.IGNORECASE)
    sal_re = re.compile(r"Salinity mean:\s*([0-9]+(?:\.[0-9]+)?)\s*PSU", re.IGNORECASE)
    oxy_re = re.compile(r"Oxygen mean:\s*([0-9]+(?:\.[0-9]+)?)\s*μmol/kg", re.IGNORECASE)
    chl_re = re.compile(r"Chlorophyll mean:\s*([0-9]+(?:\.[0-9]+)?)\s*mg/m", re.IGNORECASE)
    depth_re = re.compile(r"Depth range:\s*([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)\s*m", re.IGNORECASE)

    for doc in docs:
        t = temp_re.search(doc)
        if t:
            temp_lows.append(float(t.group(1)))
            temp_highs.append(float(t.group(2)))

        s = sal_re.search(doc)
        if s:
            sal_means.append(float(s.group(1)))

        o = oxy_re.search(doc)
        if o:
            oxy_means.append(float(o.group(1)))

        c = chl_re.search(doc)
        if c:
            chl_means.append(float(c.group(1)))

        d = depth_re.search(doc)
        if d:
            depth_mins.append(float(d.group(1)))
            depth_maxs.append(float(d.group(2)))

    stats_map: Dict[str, Dict[str, float]] = {}
    parts: List[str] = []

    if temp_lows and temp_highs:
        t_min, t_max = min(temp_lows), max(temp_highs)
        t_mean = sum(temp_lows + temp_highs) / (len(temp_lows) + len(temp_highs))
        stats_map["temperature"] = {"min": round(t_min, 1), "max": round(t_max, 1), "mean": round(t_mean, 1)}
        parts.append(f"temperature: min {t_min}, max {t_max}, avg {round(t_mean,1)}")

    if sal_means:
        s_min, s_max, s_mean = min(sal_means), max(sal_means), sum(sal_means)/len(sal_means)
        stats_map["salinity"] = {"min": round(s_min,1), "max": round(s_max,1), "mean": round(s_mean,1)}
        parts.append(f"salinity: min {s_min}, max {s_max}, avg {round(s_mean,1)}")

    if oxy_means:
        o_mean = sum(oxy_means)/len(oxy_means)
        stats_map["oxygen"] = {"mean": round(o_mean,1)}
        parts.append(f"oxygen: avg {round(o_mean,1)}")

    if chl_means:
        c_mean = sum(chl_means)/len(chl_means)
        stats_map["chlorophyll"] = {"mean": round(c_mean,2)}
        parts.append(f"chlorophyll: avg {round(c_mean,2)}")

    if depth_mins and depth_maxs:
        d_min, d_max = min(depth_mins), max(depth_maxs)
        stats_map["depth"] = {"min": round(d_min,1), "max": round(d_max,1)}
        parts.append(f"depth: min {d_min}, max {d_max}")

    if not parts:
        return "No valid numeric data.", {}
    
    return "; ".join(parts), stats_map

# ------------------------------------------------------------
# Simple enhanced summary (non-LLM)
# ------------------------------------------------------------
def create_enhanced_simple_summary_from_stats(stats_text: str, user_query: str) -> str:
    if "No valid" in stats_text or "No data" in stats_text:
        return "No oceanographic data found matching your query."

    parts = []
    water_mass_indicators = []

    # Temperature & salinity
    temp_match = re.search(r"temperature:\s*min\s*([\d.]+),\s*max\s*([\d.]+),\s*avg\s*([\d.]+)", stats_text.lower())
    sal_match = re.search(r"salinity:\s*min\s*([\d.]+),\s*max\s*([\d.]+),\s*avg\s*([\d.]+)", stats_text.lower())

    if temp_match:
        min_t, max_t, avg_t = map(float, temp_match.groups())
        temp_range = max_t - min_t
        if temp_range > 15:
            parts.append(f"Strong temperature stratification: {min_t}-{max_t}°C")
            water_mass_indicators.append("mixed water column")
        elif temp_range > 5:
            parts.append(f"Moderate temperature variation: {min_t}-{max_t}°C (avg {avg_t}°C)")
            water_mass_indicators.append("some vertical mixing")
        else:
            parts.append(f"Uniform temperature around {avg_t}°C")
            water_mass_indicators.append("stable water mass")

    if sal_match:
        min_s, max_s, avg_s = map(float, sal_match.groups())
        sal_range = max_s - min_s
        if sal_range < 0.5:
            parts.append(f"Salinity is very stable at {avg_s} PSU")
        elif sal_range < 2:
            parts.append(f"Salinity shows minor variation ({min_s}-{max_s} PSU)")
        else:
            parts.append(f"Salinity varies significantly from {min_s}-{max_s} PSU")
            water_mass_indicators.append("water mass mixing")

    # Oxygen
    oxy_match = re.search(r"oxygen: avg\s*([\d.]+)", stats_text.lower())
    if oxy_match:
        avg_o2 = float(oxy_match.group(1))
        if avg_o2 < 50:
            parts.append(f"Low oxygen ({avg_o2} μmol/kg) indicating hypoxic conditions")
        else:
            parts.append(f"Oxygen levels healthy at {avg_o2} μmol/kg")

    # Chlorophyll
    chl_match = re.search(r"chlorophyll: avg\s*([\d.]+)", stats_text.lower())
    if chl_match:
        avg_chl = float(chl_match.group(1))
        if avg_chl > 1:
            parts.append(f"High chlorophyll ({avg_chl} mg/m³) indicating productive waters")
        else:
            parts.append(f"Low chlorophyll ({avg_chl} mg/m³) indicating low productivity")

    # Depth
    depth_match = re.search(r"depth: min\s*([\d.]+),\s*max\s*([\d.]+)", stats_text.lower())
    if depth_match:
        min_d, max_d = map(float, depth_match.groups())
        parts.append(f"Depth ranges from {min_d}-{max_d} m")

    result = ". ".join(parts)
    if water_mass_indicators:
        unique_indicators = list(dict.fromkeys(water_mass_indicators))
        result += f". Suggests {', '.join(unique_indicators[:2])}"

    return result + "."

# ------------------------------------------------------------
# LLM summarization helper (FIXED for LangChain compatibility)
# ------------------------------------------------------------
def summarize_with_llm_from_stats(llm, stats_text: str, user_query: str) -> str:
    """
    Use an LLM to turn numeric stats into a readable natural language summary.
    Fixed to work with LangChain's invoke method.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an oceanography expert. Create concise, scientifically accurate summaries "
         "from oceanographic statistics. Focus on meaningful patterns and trends."),
        ("user", 
         f"User query: {user_query}\n\n"
         f"Statistics: {stats_text}\n\n"
         "Create a clear, concise summary suitable for oceanographers. Include key insights "
         "about temperature, salinity, oxygen, chlorophyll, and depth patterns.")
    ])
    
    try:
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({})
        return response.strip()
    except Exception as e:
        print(f"WARNING: LLM summarization failed: {e}")
        return create_enhanced_simple_summary_from_stats(stats_text, user_query)

# ------------------------------------------------------------
# Context summary with LLM (for fallback cases)
# ------------------------------------------------------------
def create_context_summary_with_llm(llm, context: str, user_query: str) -> str:
    """Create a summary from retrieved context documents when no numeric stats available."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an oceanography expert. Summarize relevant information from Argo float profiles "
         "to answer user questions about ocean conditions."),
        ("user", 
         f"User query: {user_query}\n\n"
         f"Available profile summaries:\n{context}\n\n"
         "Create a concise summary highlighting the most relevant oceanographic conditions "
         "from these profiles. Include location, time, and key measurements where available.")
    ])
    
    try:
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({})
        return response.strip()
    except Exception as e:
        print(f"WARNING: Context summarization failed: {e}")
        return f"Found relevant oceanographic profiles matching your query about {user_query}."

# ------------------------------------------------------------
# SQL prompt
# ------------------------------------------------------------
def create_enhanced_sql_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", 
         "You are a SQL expert specializing in oceanographic data analysis. "
         "Generate efficient PostgreSQL queries for the argo_profiles table.\n\n"
         "TABLE SCHEMA:\n"
         "argo_profiles(time, latitude, longitude, pressure, temperature, salinity, oxygen, chlorophyll, depth)\n\n"
         "Return ONLY the SQL query, no explanations or markdown formatting."),
        ("user", 
         "Context from vector search: {context}\n\n"
         "User query: {query}\n\n"
         "Generate a PostgreSQL SELECT query:")
    ])

# ------------------------------------------------------------
# Main RAG query function (COMPLETED)
# ------------------------------------------------------------
def rag_query(
    user_query: str,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    db_uri: str = DB_URI,
    use_phenomenon_prompt: bool = False
) -> str:
    """
    Process a natural language query using RAG with enhanced summarization:
    - Retrieve relevant context from ChromaDB
    - Generate SQL via LLM (if available)
    - Execute SQL on PostgreSQL
    - Always return a summarized response with meaningful insights
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
                # Use enhanced SQL prompt
                sql_prompt = create_enhanced_sql_prompt()
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
                if use_phenomenon_prompt:
                    # Note: create_phenomenon_focused_prompt() would need to be defined
                    # For now, fall back to standard summary
                    print("WARNING: Phenomenon prompt not implemented, using standard summary")
                    return summarize_with_llm_from_stats(llm, stats_text, user_query)
                else:
                    return summarize_with_llm_from_stats(llm, stats_text, user_query)
            else:
                print("DEBUG: Creating enhanced simple summary from statistics (no LLM)")
                return create_enhanced_simple_summary_from_stats(stats_text, user_query)
        
        # Step 6: If no numerical stats available, create a basic summary
        if docs:
            if llm:
                # Use enhanced context summarization
                try:
                    return create_context_summary_with_llm(llm, context, user_query)
                except Exception as e:
                    print(f"WARNING: LLM context summarization failed: {e}")
                    # Fall through to basic response
            
            # Enhanced basic response without LLM
            num_profiles = len(docs)
            # Try to extract some basic info from docs for better response
            locations = set()
            time_periods = set()
            
            for doc in docs:
                # Extract coordinates
                coord_match = re.search(r"([-+]?\d*\.?\d+)\s*lat,\s*([-+]?\d*\.?\d+)\s*lon", doc)
                if coord_match:
                    lat = float(coord_match.group(1))
                    # Classify region roughly
                    if abs(lat) < 30:
                        locations.add("tropical")
                    elif abs(lat) < 60:
                        locations.add("temperate")
                    else:
                        locations.add("polar")
                
                # Extract year if possible
                year_match = re.search(r"20\d{2}", doc)
                if year_match:
                    time_periods.add(year_match.group())
            
            region_desc = ", ".join(locations) if locations else "various"
            time_desc = f" from {min(time_periods)}-{max(time_periods)}" if len(time_periods) > 1 else f" from {list(time_periods)[0]}" if time_periods else ""
            
            return f"Found {num_profiles} oceanographic profiles from {region_desc} regions{time_desc}. The data includes temperature, salinity, oxygen, and chlorophyll measurements at various depths. For detailed analysis with specific statistics, please refine your query with parameters like location coordinates, depth ranges, or time periods."
        
        # Final fallback
        return "No relevant oceanographic data found for your query. Please try rephrasing or specifying location (lat/lon), time period, or measurement parameters (temperature/salinity/oxygen/chlorophyll ranges)."

    except Exception as e:
        print(f"ERROR in rag_query: {e}")
        traceback.print_exc()
        return f"An error occurred while processing your query: {str(e)}"

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
    print("Ask questions about ocean temperature, salinity, oxygen, chlorophyll, and water masses.")
    print("Example queries:")
    print("  • 'What are temperatures like in the tropical Pacific?'")
    print("  • 'Show me oxygen data from the Atlantic Ocean'")
    print("  • 'Find productive waters with high chlorophyll'")
    print("  • 'What are conditions at 200m depth?'")
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
                print("  • Salinity: 'high salinity regions above 35 PSU'")
                print("  • Oxygen: 'hypoxic conditions below 50 μmol/kg'")
                print("  • Chlorophyll: 'productive regions with chlorophyll > 1 mg/m³'")
                print("  • Depth: 'deep water conditions below 1000m'")
                print("  • Combined: 'warm, salty surface waters in summer'")
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