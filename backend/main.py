import os
import xarray as xr
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta

RAW_DATA_PATH = os.path.normpath(os.path.join('..', 'data', 'raw', '7902287_prof.nc'))
PROCESSED_PARQUET_PATH = os.path.normpath(os.path.join('..', 'data', 'processed', 'argo_profiles.parquet'))
DB_URI = 'postgresql://postgres:Aashi@1234@localhost:5432/argo_db'
CHROMA_PATH = '../db/chroma_db'
CHROMA_COLLECTION_NAME = 'argo_summaries'

def process_argo_to_parquet(raw_path, parquet_path):
    """
    This function reads the ARGO NetCDF file, processes it into a flat DataFrame,
    and saves it as a Parquet file.
    """
    print("DEBUG: Starting process_argo_to_parquet...")
    
    # Step 1: Check if raw file exists and open the NetCDF dataset
    print(f"DEBUG: Checking raw file at {raw_path}")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}. Please check the path.")
    ds = xr.open_dataset(raw_path)
    print("DEBUG: NetCDF dataset opened successfully. Variables:", list(ds.variables))
    
    # Step 2: Extract the key variables
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
        print(f"DEBUG: JULD sample after filtering: {juld[:5]}")  # First 5 for check
        time = [
            ref_date + timedelta(days=float(t)) if not np.isnan(t) else pd.NaT
            for t in juld
        ]
    print(f"DEBUG: Time conversion done. Sample times: {time[:3]}")  # First 3
    
    # Step 4: Flatten the data for the DataFrame
    print("DEBUG: Flattening data...")
    n_prof, n_levels = pres.shape
    print(f"DEBUG: Dimensions - Profiles: {n_prof}, Levels: {n_levels}")
    lat_expanded = np.repeat(lat, n_levels)
    lon_expanded = np.repeat(lon, n_levels)
    time_expanded = np.repeat(time, n_levels)
    print("DEBUG: Expansion done.")
    
    # Step 5: Create the Pandas DataFrame
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
    
    # Step 6: Drop rows with missing values
    print("DEBUG: Dropping NaNs...")
    df = df.dropna()
    print(f"DEBUG: After dropna, shape: {df.shape}")
    
    if df.empty:
        raise ValueError("DataFrame is empty after dropna! Check for all NaN data.")
    
    # Step 7: Create the output directory if it doesn't exist
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

if __name__ == "__main__":
    try:
        process_argo_to_parquet(RAW_DATA_PATH, PROCESSED_PARQUET_PATH)
    except Exception as e:
        print(f"ERROR: Failed with exception: {e}")
        import traceback
        traceback.print_exc()  # Full error trace