import os
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from dotenv import load_dotenv
from db.database import DB_URI
from rag.vector_store import setup_chroma
from services.search_profiles import search_profiles

load_dotenv()

# File paths
RAW_DATA_PATH = os.path.normpath(os.path.join('..', 'data', 'raw', '7902287_prof.nc'))
RAW_DATA_PATH2 = os.path.normpath(os.path.join('..', 'data', 'raw', '2902202_Sprof.nc'))
PROCESSED_PARQUET_PATH = os.path.normpath(os.path.join('..', 'data', 'processed', 'argo_profiles.parquet'))

# Vector DB config
CHROMA_PATH = os.getenv("chromapath")
CHROMA_COLLECTION_NAME = os.getenv('collectioname')


def process_argo_to_parquet(raw_path_main, raw_path_extra, parquet_path):
    """
    Reads ARGO NetCDF files, processes them into a flat DataFrame,
    and saves as a Parquet file. Correctly maps DOXY, CHLA, and PRES.
    Handles mismatched dimensions between files and vertical sampling.
    """
    print("DEBUG: Starting process_argo_to_parquet...")

    # Load main file
    if not os.path.exists(raw_path_main):
        raise FileNotFoundError(f"Main raw file not found: {raw_path_main}")
    ds_main = xr.open_dataset(raw_path_main)
    print("DEBUG: Main NetCDF opened. Variables:", list(ds_main.variables))
    
    # Get main file dimensions
    n_prof_main = len(ds_main.LATITUDE)
    if "PRES" in ds_main:
        n_levels_main = ds_main.PRES.shape[1] if len(ds_main.PRES.shape) > 1 else len(ds_main.PRES)
    else:
        n_levels_main = 0
    print(f"DEBUG: Main file dimensions - N_PROF: {n_prof_main}, N_LEVELS: {n_levels_main}")

    # Load extra file
    if not os.path.exists(raw_path_extra):
        raise FileNotFoundError(f"Extra raw file not found: {raw_path_extra}")
    ds_extra = xr.open_dataset(raw_path_extra)
    print("DEBUG: Extra NetCDF opened. Variables:", list(ds_extra.variables))
    
    # Get extra file dimensions
    n_prof_extra = len(ds_extra.LATITUDE) if "LATITUDE" in ds_extra else 0
    if "PRES" in ds_extra:
        n_levels_extra = ds_extra.PRES.shape[1] if len(ds_extra.PRES.shape) > 1 else len(ds_extra.PRES)
    else:
        n_levels_extra = 0
    print(f"DEBUG: Extra file dimensions - N_PROF: {n_prof_extra}, N_LEVELS: {n_levels_extra}")

    # Extract main variables
    lat = ds_main["LATITUDE"].values
    lon = ds_main["LONGITUDE"].values
    juld = ds_main["JULD"].values
    pres = ds_main["PRES"].values if "PRES" in ds_main else None
    temp = ds_main["TEMP"].values if "TEMP" in ds_main else None
    psal = ds_main["PSAL"].values if "PSAL" in ds_main else None

    # Extract extra variables (only if they exist)
    oxygen = ds_extra["DOXY"].values if "DOXY" in ds_extra else None
    chlorophyll = ds_extra["CHLA"].values if "CHLA" in ds_extra else None
    pres_extra = ds_extra["PRES"].values if "PRES" in ds_extra else None

    # Determine the reference profile count
    n_profiles = min(n_prof_main, n_prof_extra)
    print(f"DEBUG: Using {n_profiles} profiles for processing")

    # Limit arrays to matching number of profiles
    lat = lat[:n_profiles]
    lon = lon[:n_profiles]
    juld = juld[:n_profiles]
    
    if pres is not None:
        pres = pres[:n_profiles]
    if temp is not None:
        temp = temp[:n_profiles]
    if psal is not None:
        psal = psal[:n_profiles]
    
    if pres_extra is not None:
        pres_extra = pres_extra[:n_profiles]
    if oxygen is not None:
        oxygen = oxygen[:n_profiles]
    if chlorophyll is not None:
        chlorophyll = chlorophyll[:n_profiles]

    # Convert JULD to datetime
    ref_date = datetime(1950, 1, 1)
    if np.issubdtype(juld.dtype, np.datetime64):
        time = pd.to_datetime(juld)
    else:
        juld = juld.astype(float)
        juld = np.where((np.isfinite(juld)) & (juld < 100000), juld, np.nan)
        time = [ref_date + timedelta(days=float(t)) if not np.isnan(t) else pd.NaT for t in juld]

    # Create lists to store flattened data
    all_times = []
    all_lats = []
    all_lons = []
    all_presses = []
    all_temps = []
    all_psals = []
    all_depths = []
    all_oxygens = []
    all_chlorophylls = []

    # Process each profile
    for i in range(n_profiles):
        profile_time = time[i]
        profile_lat = lat[i]
        profile_lon = lon[i]
        
        print(f"DEBUG: Processing profile {i+1}/{n_profiles}")
        
        # === MAIN FILE PROCESSING ===
        # Get main file data for this profile
        if pres is not None:
            main_pres = pres[i]
            main_temp = temp[i] if temp is not None else None
            main_psal = psal[i] if psal is not None else None
            
            # Find valid levels in main file
            valid_main_levels = np.isfinite(main_pres)
            n_levels_main = np.sum(valid_main_levels)
            
            if n_levels_main == 0:
                print(f"DEBUG: Profile {i} has no valid pressure levels in main file, skipping")
                continue
                
            # Extract valid data from main file
            main_presses = main_pres[valid_main_levels]
            main_temps = main_temp[valid_main_levels] if main_temp is not None else np.full(n_levels_main, np.nan)
            main_psals = main_psal[valid_main_levels] if main_psal is not None else np.full(n_levels_main, np.nan)
        else:
            print(f"DEBUG: Profile {i} has no pressure data in main file, skipping")
            continue

        # === EXTRA FILE PROCESSING ===
        # Get extra file data for this profile (if available)
        if pres_extra is not None and oxygen is not None and chlorophyll is not None:
            extra_pres = pres_extra[i]
            extra_oxygen = oxygen[i]
            extra_chlorophyll = chlorophyll[i]
            
            # Find valid levels in extra file
            valid_extra_levels = np.isfinite(extra_pres)
            n_levels_extra = np.sum(valid_extra_levels)
            
            print(f"DEBUG: Profile {i} - Main levels: {n_levels_main}, Extra levels: {n_levels_extra}")
            
            if n_levels_extra > 0:
                # Method 1: Interpolate extra data to main pressure levels (simplified)
                # For now, we'll use the first N levels that match, or pad with NaN
                extra_presses = extra_pres[valid_extra_levels]
                extra_oxygens = extra_oxygen[valid_extra_levels]
                extra_chlorophylls = extra_chlorophyll[valid_extra_levels]
                
                # Take minimum of available levels or pad with NaN
                n_use_levels = min(n_levels_main, n_levels_extra)
                
                if n_use_levels > 0:
                    # Use the first n_use_levels from both
                    profile_oxygens = extra_oxygens[:n_use_levels]
                    profile_chlorophylls = extra_chlorophylls[:n_use_levels]
                    
                    # Pad if extra data is shorter
                    if n_use_levels < n_levels_main:
                        profile_oxygens = np.pad(profile_oxygens, (0, n_levels_main - n_use_levels), 
                                               mode='constant', constant_values=np.nan)
                        profile_chlorophylls = np.pad(profile_chlorophylls, (0, n_levels_main - n_use_levels), 
                                                    mode='constant', constant_values=np.nan)
                else:
                    # No overlapping valid levels
                    profile_oxygens = np.full(n_levels_main, np.nan)
                    profile_chlorophylls = np.full(n_levels_main, np.nan)
            else:
                # No valid extra data
                profile_oxygens = np.full(n_levels_main, np.nan)
                profile_chlorophylls = np.full(n_levels_main, np.nan)
        else:
            # No extra data available
            profile_oxygens = np.full(n_levels_main, np.nan)
            profile_chlorophylls = np.full(n_levels_main, np.nan)

        # === CREATE PROFILE DATA ===
        # Repeat profile-level data
        profile_times = np.full(n_levels_main, profile_time)
        profile_lats = np.full(n_levels_main, profile_lat)
        profile_lons = np.full(n_levels_main, profile_lon)
        
        # Use pressure as depth
        profile_depths = main_presses.copy()
        
        # Append to lists
        all_times.extend(profile_times)
        all_lats.extend(profile_lats)
        all_lons.extend(profile_lons)
        all_presses.extend(main_presses)
        all_temps.extend(main_temps)
        all_psals.extend(main_psals)
        all_depths.extend(profile_depths)
        all_oxygens.extend(profile_oxygens)
        all_chlorophylls.extend(profile_chlorophylls)

    print(f"DEBUG: Total data points after processing: {len(all_times)}")

    # Create DataFrame
    df = pd.DataFrame({
        "time": all_times,
        "latitude": all_lats,
        "longitude": all_lons,
        "pressure": all_presses,
        "temperature": all_temps,
        "salinity": all_psals,
        "depth": all_depths,
        "oxygen": all_oxygens,
        "chlorophyll": all_chlorophylls
    })

    # Remove rows with all NaN values (except time, lat, lon which should always be valid)
    df = df.dropna(subset=['pressure'], how='all')

    print(f"DEBUG: Final DataFrame shape: {df.shape}")
    print(f"DEBUG: Sample data:\n{df.head()}")

    # Save to Parquet
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    df.to_parquet(parquet_path, index=False, engine='pyarrow')
    print(f"✅ Processed data saved to {parquet_path}. Shape: {df.shape}")

    return df


def load_to_postgres(parquet_path, db_uri=DB_URI):
    engine = create_engine(db_uri)
    df = pd.read_parquet(parquet_path)
    df.to_sql('argo_profiles', engine, if_exists='replace', index=False)
    engine.dispose()
    print("✅ Data loaded into PostgreSQL table 'argo_profiles'")


if __name__ == "__main__":
    try:
        # Process files
        df = process_argo_to_parquet(RAW_DATA_PATH, RAW_DATA_PATH2, PROCESSED_PARQUET_PATH)
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