from services.search_profiles import search_profiles
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer as STModel
from rag.vector_store import setup_chroma
import os
import xarray as xr
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
from db.database import DB_URI

# File paths
RAW_DATA_PATH = os.path.normpath(os.path.join('..', 'data', 'raw', '7902287_prof.nc'))
PROCESSED_PARQUET_PATH = os.path.normpath(os.path.join('..', 'data', 'processed', 'argo_profiles.parquet'))

# Vector DB config
CHROMA_PATH = os.path.normpath(os.path.join("..", "db", "chroma_db"))
CHROMA_COLLECTION_NAME = 'argo_summaries'


def process_argo_to_parquet(raw_path, parquet_path):
    """
    Reads the ARGO NetCDF file, processes it into a flat DataFrame,
    and saves it as a Parquet file.
    """
    print("DEBUG: Starting process_argo_to_parquet...")

    # Step 1: Check file existence
    print(f"DEBUG: Checking raw file at {raw_path}")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}. Please check the path.")
    ds = xr.open_dataset(raw_path)
    print("DEBUG: NetCDF dataset opened successfully. Variables:", list(ds.variables))

    # Step 2: Extract variables
    print("DEBUG: Extracting variables...")
    lat = ds["LATITUDE"].values
    lon = ds["LONGITUDE"].values
    juld = ds["JULD"].values
    pres = ds["PRES"].values
    temp = ds["TEMP"].values
    psal = ds["PSAL"].values
    print(f"DEBUG: Extracted shapes - LAT: {lat.shape}, LON: {lon.shape}, JULD: {juld.shape}, PRES: {pres.shape}, TEMP: {temp.shape}, PSAL: {psal.shape}")

    # Step 3: Convert JULD to datetime
    print("DEBUG: Converting JULD to datetime...")
    ref_date = datetime(1950, 1, 1)
    if np.issubdtype(juld.dtype, np.datetime64):
        time = pd.to_datetime(juld)
    else:
        juld = juld.astype(float)
        juld = np.where((np.isfinite(juld)) & (juld < 100000), juld, np.nan)
        print(f"DEBUG: JULD sample after filtering: {juld[:5]}")
        time = [
            ref_date + timedelta(days=float(t)) if not np.isnan(t) else pd.NaT
            for t in juld
        ]
    print(f"DEBUG: Time conversion done. Sample times: {time[:3]}")

    # Step 4: Flatten data
    print("DEBUG: Flattening data...")
    n_prof, n_levels = pres.shape
    print(f"DEBUG: Dimensions - Profiles: {n_prof}, Levels: {n_levels}")
    lat_expanded = np.repeat(lat, n_levels)
    lon_expanded = np.repeat(lon, n_levels)
    time_expanded = np.repeat(time, n_levels)
    print("DEBUG: Expansion done.")

    # Step 5: Create DataFrame
    print("DEBUG: Creating DataFrame...")
    df = pd.DataFrame({
        "time": time_expanded,
        "latitude": lat_expanded,
        "longitude": lon_expanded,
        "pressure": pres.flatten(),
        "temperature": temp.flatten(),
        "salinity": psal.flatten()
    })
    print(f"DEBUG: DataFrame created. Initial shape: {df.shape}")

    # Step 6: Drop missing values
    print("DEBUG: Dropping NaNs...")
    df = df.dropna()
    print(f"DEBUG: After dropna, shape: {df.shape}")

    if df.empty:
        raise ValueError("DataFrame is empty after dropna! Check for all NaN data.")

    # Step 7: Ensure output directory exists
    print(f"DEBUG: Ensuring output directory exists for {parquet_path}")
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    print("DEBUG: Directory ready.")

    # Step 8: Save to Parquet
    print("DEBUG: Saving to Parquet...")
    df.to_parquet(parquet_path, index=False, engine='pyarrow')
    print(f"SUCCESS: Processed data saved to {parquet_path}. Final shape: {df.shape} (rows, columns)")

    # Print head for verification
    print("DEBUG: DataFrame head:")
    print(df.head())

    return df


def load_to_postgres(parquet_path, db_uri=DB_URI):
    engine = create_engine(db_uri)
    df = pd.read_parquet(parquet_path)

    # Optional QC filter
    # if 'PRES_QC' in df.columns:
    #     df = df[df['PRES_QC'] == '1']

    df.to_sql('argo_profiles', engine, if_exists='replace', index=False)
    engine.dispose()
    print("✅ Data loaded into PostgreSQL table 'argo_profiles'")


if __name__ == "__main__":
    try:
        process_argo_to_parquet(RAW_DATA_PATH, PROCESSED_PARQUET_PATH)
        load_to_postgres(PROCESSED_PARQUET_PATH)

        # Vector DB setup
        setup_chroma(PROCESSED_PARQUET_PATH, CHROMA_PATH, CHROMA_COLLECTION_NAME)

        # Example search
        query = "profiles with highest salinity"
        results = search_profiles(query, top_k=3)
        print("\n🔍 Search Results:")
        for r in results:
            print(f"- {r['document']} (distance={r['distance']:.4f})")

    except Exception as e:
        print(f"ERROR: Failed with exception: {e}")
        import traceback
        traceback.print_exc()
