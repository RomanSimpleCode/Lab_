import json
import os
import chromadb
from chromadb.utils import embedding_functions

KB_DIR = os.path.join(os.path.dirname(__file__), "kb")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_data")
EMBED_MODEL = "all-MiniLM-L6-v2"

def load_articles():
    articles = []
    for filename in os.listdir(KB_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(KB_DIR, filename), "r", encoding="utf-8") as f:
                articles.append(json.load(f))
    print(f"Загружено статей: {len(articles)}")
    return articles

def build_index():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_or_create_collection(name="itil_kb", embedding_function=embed_fn)
    
    articles = load_articles()
    documents, metadatas, ids = [], [], []
    
    for article in articles:
        documents.append(article["content"])
        metadatas.append({"title": article["title"], "category": article["category"], "id": article["id"]})
        ids.append(article["id"])
        
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print("Индексация завершена.")

if __name__ == "__main__":
    build_index()
