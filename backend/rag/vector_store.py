import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os

def setup_chroma(parquet_path, chroma_path, collection_name):
    """
    Populate ChromaDB with ARGO profile summaries and embeddings.
    
    Args:
    - parquet_path: Path to argo_profiles.parquet
    - chroma_path: Path to ChromaDB storage
    - collection_name: Name of Chroma collection
    """
    print(f"DEBUG: Setting up ChromaDB at {chroma_path}, collection: {collection_name}")
    
    # Step 1: Check if Parquet file exists
    print(f"DEBUG: Checking Parquet file at {parquet_path}")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    # Step 2: Read Parquet file
    df = pd.read_parquet(parquet_path)
    print(f"DEBUG: Loaded Parquet with shape: {df.shape}")
    if df.empty:
        raise ValueError("Parquet file is empty!")
    print(f"DEBUG: Parquet head:\n{df.head().to_string()}")
    
    # Step 3: Initialize ChromaDB client with persistent storage
    os.makedirs(chroma_path, exist_ok=True)  # Ensure chroma_path exists
    print(f"DEBUG: Creating ChromaDB directory at {chroma_path}")
    try:
        # Use Settings for persistent storage (ChromaDB 1.0.x)
        settings = Settings(
            persist_directory=chroma_path,  # Correct parameter for 1.0.x
            is_persistent=True
        )
        client = chromadb.Client(settings=settings)
        print(f"DEBUG: ChromaDB client initialized with persistent storage")
    except Exception as e:
        print(f"ERROR: Failed to initialize persistent client: {e}")
        raise  # Re-raise to debug further if needed
    
    print(f"DEBUG: ChromaDB client initialized")
    
    # Step 4: Delete existing collection if it exists (fresh start for testing)
    try:
        client.delete_collection(collection_name)
        print(f"DEBUG: Deleted existing collection {collection_name} for fresh start")
    except:
        print(f"DEBUG: No existing collection {collection_name} to delete")
    
    collection = client.create_collection(name=collection_name)
    print(f"DEBUG: Created collection {collection_name}")
    
    # Step 5: Load sentence transformer for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"DEBUG: SentenceTransformer loaded")
    
    # Step 6: Group by profile (unique time/lat/lon)
    profiles = df.groupby(['time', 'latitude', 'longitude'])
    print(f"DEBUG: Found {len(profiles)} unique profiles")
    
    if len(profiles) == 0:
        raise ValueError("No profiles found after grouping! Check data.")
    
    # Step 7: Generate summaries and embeddings
    for group_key, group in profiles:
        time, lat, lon = group_key
        summary = (f"Argo profile at {lat:.2f} lat, {lon:.2f} lon on {time}. "
                   f"Temperature range: {group['temperature'].min():.1f}-{group['temperature'].max():.1f}°C, "
                   f"Salinity mean: {group['salinity'].mean():.1f} PSU")
        embedding = model.encode(summary).tolist()
        profile_id = str(hash(group_key))  # Unique ID for profile
        collection.add(
            documents=[summary],
            embeddings=[embedding],
            ids=[profile_id]
        )
        print(f"DEBUG: Added profile ID {profile_id} with summary: {summary}")
    
    # Step 8: Verify collection
    count = collection.count()
    print(f"SUCCESS: ChromaDB populated with {count} documents")
    
    # Optional: Peek at first few documents
    peek = collection.peek(2)  # Get first 2 documents
    print(f"DEBUG: Sample documents in collection:\n{peek['documents']}")

if __name__ == "__main__":
    try:
        # Use relative, system-independent paths
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        parquet_path = os.path.normpath(os.path.join(base_dir, 'data', 'processed', 'argo_profiles.parquet'))
        chroma_path = os.path.normpath(os.path.join(base_dir, 'db', 'chroma_db'))
        collection_name = 'argo_summaries'
        setup_chroma(
            parquet_path=parquet_path,
            chroma_path=chroma_path,
            collection_name=collection_name
        )
    except Exception as e:
        print(f"ERROR: Failed to setup ChromaDB: {e}")
        import traceback
        traceback.print_exc()