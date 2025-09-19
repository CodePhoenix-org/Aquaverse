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


# ------------------------------------------------------------
# Main RAG query function (updated)
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
                    phenomenon_prompt = create_phenomenon_focused_prompt()
                    chain = phenomenon_prompt | llm | StrOutputParser()
                    return chain.invoke({"query": user_query, "stats": stats_text}).strip()
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
                    lat, lon = float(coord_match.group(1)), float(coord_match.group(2))
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
            
            return f"Found {num_profiles} oceanographic profiles from {region_desc} regions{time_desc}. The data includes temperature and salinity measurements at various depths. For detailed analysis with specific statistics, please refine your query with parameters like location coordinates, depth ranges, or time periods."
        
        # Final fallback
        return "No relevant oceanographic data found for your query. Please try rephrasing or specifying location (lat/lon), time period, or measurement parameters (temperature/salinity ranges)."

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