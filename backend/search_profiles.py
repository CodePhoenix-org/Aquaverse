from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Define paths/names consistently
CHROMA_PATH = "../db/chroma_db"  # Use the same as in main.py
CHROMA_COLLECTION_NAME = "argo_summaries"

# Load embedding model (same as in vector_store.py)
model = SentenceTransformer('all-MiniLM-L6-v2')

def search_profiles(query: str, top_k: int = 5):
    """
    Search stored Argo profiles using natural language.
    Args:
        query (str): User's search question (e.g., "profiles with highest salinity").
        top_k (int): Number of results to return.
    Returns:
        List of dicts with id, document, and distance.
    """
    # Connect to Chroma
    client = PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

    # Encode query into embedding
    query_embedding = model.encode(query).tolist()

    # Perform similarity search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Convert to cleaner structure
    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i]  # lower = more similar
        })
    return hits
