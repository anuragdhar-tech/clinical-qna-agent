import chromadb
import numpy as np
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embeddings import EmbeddingPipeline

class ChromaVectorStore:
    def __init__(self, persist_dir: str = "chroma_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("vectorstore")
        self.model = SentenceTransformer(embedding_model)
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        texts = [chunk.page_content for chunk in chunks]
        # metadatas = [{"text": text} for text in texts]
        metadatas = [
            {
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", -1),
            }
            for chunk in chunks
        ]
        self.add_embeddings(np.array(embeddings), metadatas, texts)
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any], texts: List[str]):
        ids = [str(i) for i in range(self.collection.count(), self.collection.count() + len(embeddings))]
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"[INFO] Added {len(embeddings)} vectors to Chroma collection.")

    def save(self):
        print(f"[INFO] ChromaDB auto-persists; no manual save needed.")

    def load(self):
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection("vectorstore")
        print(f"[INFO] Loaded Chroma collection from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
        )
        output = []
        # for i, idx, dist, doc in zip(results["ids"][0], results["distances"][0], results["documents"][0]):
        for i, (idx, dist, doc) in enumerate(zip(results["ids"][0], results["distances"][0], results["documents"][0])):    
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            output.append({
                "index": idx,
                "distance": dist,
                "text": doc,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", -1),
            })
            # output.append({"index": idx, "distance": dist, "metadata": {"text": doc}})
        return output

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype("float32")
        return self.search(query_emb, top_k=top_k)

# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents
    docs = load_all_documents("data")
    store = ChromaVectorStore("chroma_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is attention mechanism?", top_k=3))
