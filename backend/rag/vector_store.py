# vector_store.py
import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


def setup_chroma(parquet_path, chroma_path, collection_name):
    # Ensure directory exists
    os.makedirs(chroma_path, exist_ok=True)

    # Initialize Chroma client (persistent = survives restarts)
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(collection_name)

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load processed parquet data
    df = pd.read_parquet(parquet_path)

    # Group by profile (unique time/lat/lon)
    profiles = df.groupby(['time', 'latitude', 'longitude'])

    print(f"Populating Chroma with {len(profiles)} profiles...")

    for idx, (group_key, group) in enumerate(profiles, start=1):
        # Build a summary string for each profile
        summary = (
            f"Argo profile at {group_key[1]:.2f} lat, {group_key[2]:.2f} lon on {group_key[0]}. "
            f"Temp range: {group['temperature'].min():.1f}-{group['temperature'].max():.1f}°C, "
            f"Salinity avg: {group['salinity'].mean():.1f} PSU."
        )

        # Create embedding
        embedding = model.encode(summary).tolist()

        # Unique ID (stringified hash)
        doc_id = str(abs(hash(group_key)))  

        # Insert into Chroma
        collection.add(
            documents=[summary],
            embeddings=[embedding],
            ids=[doc_id]
        )

        if idx % 100 == 0:  # Progress log
            print(f"Inserted {idx} profiles...")

    print("✅ Vector DB populated successfully!")
