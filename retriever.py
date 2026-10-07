import chromadb
from chromadb.utils import embedding_functions
import os

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_data")
EMBED_MODEL = "all-MiniLM-L6-v2"

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_collection(name="itil_kb", embedding_function=embed_fn)

def retrieve(query: str, k: int = 3):
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=k)
    
    found = []
    for i in range(len(results["ids"][0])):
        found.append({
            "id": results["ids"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "content": results["documents"][0][i],
            "score": results["distances"][0][i],
        })
    return found
