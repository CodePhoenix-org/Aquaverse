import chromadb
from chromadb.config import Settings

chroma_path = 'C:/Users/Dell/Desktop/Hackathons/SIH-25/db/chroma_db'
collection_name = 'argo_summaries'

settings = Settings(persist_directory=chroma_path)
client = chromadb.Client(settings=settings)
collections = client.list_collections()
print(f"Available collections: {[c.name for c in collections]}")
if collections:
    collection = client.get_collection(name=collection_name)
    count = collection.count()
    print(f"Collection {collection_name} has {count} documents")