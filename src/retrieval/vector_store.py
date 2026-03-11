import json
import chromadb
import uuid
from chromadb.utils import embedding_functions
import os

class VectorStore:
    def __init__(self, db_path: str, collection_name: str = "omni_analyst_docs"):
        self.client = chromadb.PersistentClient(path=db_path)

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def ingest_jsonl(self, jsonl_path: str):
        documents = []
        metadatas = []
        ids = []

        with open(jsonl_path, 'r',encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                file_name = entry.get('file_name',"unknown_source")

                for i,doc in enumerate(entry.get("documents",[])):
                    text = doc.get("content","").strip()

                    if not text:
                        continue

                    metadata = doc.get("metadata",{})
                    metadata['source_file'] = file_name

                    documents.append(text)
                    metadatas.append(metadata)
                    ids.append(str(uuid.uuid4()))

        batch_size = 500
        total_chunks = len(documents)

        print(f"Preparing to ingest {total_chunks} chunks...")

        for i in range(0, total_chunks, batch_size):
            end = i + batch_size
            self.collection.add(
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
            print(f"Uploaded chunks {i} to {min(end, total_chunks)}")

        print(f"Ingestion complete. Total records in DB: {self.collection.count}")

    def query(self, prompt: str, n_results: int = 3):
        return self.collection.query(
            query_texts=[prompt],
            n_results=n_results
        )
    
if __name__ == "__main__":
    os.makedirs("data/vector_db",exist_ok=True)
    vs = VectorStore(db_path="data/vector_db")

    vs.ingest_jsonl("data/processed/master_index.jsonl")

    print("\n Testing Sematic Search:")
    query_text = "What are main findings in patient report?"
    results = vs.query(query_text,n_results=3)

    for idx, (doc, meta) in enumerate(zip(results["documents"][0],results["metadatas"][0])):
        print(f"\n ----Results {idx+1} (Source: {meta.get('source_file')}, Page: {meta.get('page')})-----")
        print(f"Content: {doc[:300]}...")