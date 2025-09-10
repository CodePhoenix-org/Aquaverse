import os
import xarray as xr
import pandas as pd
from sqlalchemy import create_engine
import chromadb
from sentence_transformers import SentenceTransformer

# Paths (relative to backend/ingestion/)
RAW_DATA_PATH = '../../data/raw/7902287_prof.nc'  # Adjust if needed
PROCESSED_PARQUET_PATH = '../../data/processed/argo_profiles.parquet'
DB_URI = 'postgresql://postgres:Aashi@1234@localhost:5432/argo_db'  # Update with your PostgreSQL credentials
CHROMA_PATH = '../db/chroma_db'
CHROMA_COLLECTION_NAME = 'argo_summaries'

def read_netcdf(file_path):
    """Read Argo profile NetCDF file using xarray."""
    ds = xr.open_dataset(file_path)
    print("Dataset variables:", list(ds.variables.keys()))
    return ds

def process_data(ds):
    """Filter and convert to Pandas DataFrame with fallback to raw variables if adjusted ones are missing."""

    # Utility function to pick adjusted if available, else raw
    def pick_var(adjusted, raw):
        if adjusted in ds.variables:
            return ds[adjusted]
        elif raw in ds.variables:
            return ds[raw]
        else:
            print(f"Warning: Neither {adjusted} nor {raw} found.")
            return None

    # Helper to handle QC flags (works for both adjusted and raw)
    def pick_qc(adjusted_qc, raw_qc):
        if adjusted_qc in ds.variables:
            arr = ds[adjusted_qc]
        elif raw_qc in ds.variables:
            arr = ds[raw_qc]
        else:
            print(f"Warning: Neither {adjusted_qc} nor {raw_qc} found.")
            return None

        # Convert QC values (bytes/str → int)
        return arr.astype(str).str.strip().astype(int)

    # Pick variables
    pres = pick_var('PRES_ADJUSTED', 'PRES')
    temp = pick_var('TEMP_ADJUSTED', 'TEMP')
    psal = pick_var('PSAL_ADJUSTED', 'PSAL')

    pres_qc = pick_qc('PRES_ADJUSTED_QC', 'PRES_QC')
    temp_qc = pick_qc('TEMP_ADJUSTED_QC', 'TEMP_QC')
    psal_qc = pick_qc('PSAL_ADJUSTED_QC', 'PSAL_QC')

    # Filter for good QC data (QC == 1 means good)
    good_data = ds.where(
        (pres_qc == 1) &
        (temp_qc == 1) &
        (psal_qc == 1),
        drop=True
    )

    # Convert to DataFrame (flatten arrays)
    df = pd.DataFrame({
        'profile_id': good_data.get('N_PROF', pd.Series(range(len(good_data['JULD'])))).values.flatten(),
        'time': good_data['JULD'].values.flatten(),
        'latitude': good_data['LATITUDE'].values.flatten(),
        'longitude': good_data['LONGITUDE'].values.flatten(),
        'pressure': pres.values.flatten() if pres is not None else None,
        'temperature': temp.values.flatten() if temp is not None else None,
        'salinity': psal.values.flatten() if psal is not None else None
    })

    df = df.dropna().reset_index(drop=True)
    print("Processed DataFrame shape:", df.shape)
    return df

def store_in_sql(df, db_uri):
    """Store DataFrame in PostgreSQL."""
    engine = create_engine(db_uri)
    df.to_sql('argo_profiles', engine, if_exists='replace', index=False)
    print("Data stored in PostgreSQL table 'argo_profiles'.")

def store_in_vector_db(df, chroma_path, collection_name):
    """Generate summaries and store in ChromaDB for RAG."""
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    collection = chroma_client.get_or_create_collection(name=collection_name)

    summaries, ids, metadatas = [], [], []
    for profile_id, group in df.groupby('profile_id'):
        avg_sal = group['salinity'].mean()
        avg_temp = group['temperature'].mean()
        lat = group['latitude'].iloc[0]
        lon = group['longitude'].iloc[0]
        time_str = group['time'].iloc[0]
        summary = (
            f"Argo profile {profile_id} at lat {lat:.2f}, lon {lon:.2f}, time {time_str}, "
            f"with avg temperature {avg_temp:.2f}°C and avg salinity {avg_sal:.2f} psu."
        )
        summaries.append(summary)
        ids.append(str(profile_id))
        metadatas.append({'profile_id': profile_id, 'source': '7902287_prof.nc', 'time': str(time_str)})

    embeddings = embedding_model.encode(summaries)
    collection.add(documents=summaries, embeddings=embeddings, ids=ids, metadatas=metadatas)
    print(f"Summaries stored in ChromaDB collection '{collection_name}' at {chroma_path}.")

def export_to_parquet(df, file_path):
    """Export to Parquet for structured storage."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_parquet(file_path)
    print(f"Data exported to Parquet: {file_path}")

if __name__ == "__main__":
    ds = read_netcdf(RAW_DATA_PATH)
    df = process_data(ds)
    export_to_parquet(df, PROCESSED_PARQUET_PATH)
    store_in_sql(df, DB_URI)
    store_in_vector_db(df, CHROMA_PATH, CHROMA_COLLECTION_NAME)
