import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

class VectorSearchEngine:
    def __init__(self, collection_name="market_intelligence"):
        self.client = chromadb.PersistentClient(path="./data/chroma_db")
        # Load Transformer model for deep learning embeddings
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
    def populate_from_json(self, json_path: str):
        with open(json_path, "r") as f:
            docs_data = json.load(f)
            
        docs = [d["text"] for d in docs_data]
        ids = [d["article_id"] for d in docs_data]
        metadatas = [{"company": d["company"], "title": d["title"]} for d in docs_data]
        
        # Generate PyTorch-backed Transformer embeddings
        embeddings = self.encoder.encode(docs).tolist()
        
        self.collection.add(
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Vector Database populated with {len(docs)} document embeddings.")

    def search(self, query: str, top_k: int = 2):
        query_embedding = self.encoder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        return results

if __name__ == "__main__":
    vdb = VectorSearchEngine()
    vdb.populate_from_json("data/raw_news.json")
    print("Test Search Query Output:")
    print(vdb.search("What is TechCorp doing with its pricing?"))