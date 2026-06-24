# from src.data_loader import load_all_documents
# from src.vectorstore_faiss import FaissVectorStore
# from src.vectorstore_chroma import ChromaVectorStore
# # from src.search import RAGSearch
# from src.embeddings import EmbeddingPipeline
# from src.reranker import Reranker
# from src.generator import generate
# import json

import sys
import json
from src.data_loader import load_all_documents
from src.vectorstore_chroma import ChromaVectorStore
from src.reranker import Reranker
from src.generator import generate

# Variables for the chroma and data directories
CHROMA_DIR = "chroma_store"
DATA_DIR = "data"


#Ingestion Pipeline: Load documents, chunk, embed, and store in vector store. Run Once.

def ingest():
    print("[INGEST] Loading documents...")
    docs = load_all_documents(DATA_DIR)
    
    print("[INGEST] Building vector store...")
    store = ChromaVectorStore(CHROMA_DIR)
    store.build_from_documents(docs)
    
    print(f"[INGEST] Done. {len(docs)} pages indexed into {CHROMA_DIR}/")

# Query Pipeline: Load vector store, retrieve, rerank, and generate answer. Run for each query.

def search(query: str):
    print(f"[SEARCH] Query: {query}\n")
    
    store = ChromaVectorStore(CHROMA_DIR)
    store.load()
    retrieved = store.query(query, top_k=5)
    
    reranker = Reranker()
    reranked = reranker.rerank(query, retrieved, top_k=3)
    
    result = generate(query, reranked, retrieved)
    
    print(json.dumps(result, indent=2))

# Main function for command-line execution.
    # First time or when you add new PDFs:
    # python3 app.py ingest

    # # Query:
    # python3 app.py search "Question?"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python app.py ingest")
        print("  python app.py search \"your question here\"")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        ingest()
    elif command == "search":
        if len(sys.argv) < 3:
            print("Error: provide a query. Example: python app.py search \"What is the treatment for TB?\"")
            sys.exit(1)
        search(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


# # Example usage
# if __name__ == "__main__":
    
#     # docs = load_all_documents("data")
#     # emb_pipe = EmbeddingPipeline()
#     # chunks = emb_pipe.chunk_documents(docs)
#     # chunks_vector = emb_pipe.embed_chunks(chunks)
#     # print("[INFO] Example embedding:", chunks_vector[0] if len(chunks_vector) > 0 else None)

#     ## FAISS VECTOR STOR
#     # store = FaissVectorStore("faiss_store")
#     # store.build_from_documents(docs)
#     # store = FaissVectorStore("faiss_store")
#     # store.load()
#     # print(store.query("What is diabetes?", top_k=3))

#     ## CHROMADB
#     # store = ChromaVectorStore("chroma_store")
#     # store.build_from_documents(docs)
#     store = ChromaVectorStore("chroma_store")
#     store.load()
#     # # print(store.query("What is Pneumonia?", top_k=3))
#     # results = store.query("How does diabetes affect TB treatment outcomes?", top_k = 3)

#     reranker = Reranker()
#     # re_ranked = reranker.rerank("How does diabetes affect TB treatment outcomes?", results, top_k=3)
#     # for r in re_ranked:
#     #     print(r["rerank_score"], r["source"], r["page"], r["text"][:100])


#     # store.load()
#     # #print(store.query("What is attention mechanism?", top_k=3))
#     # rag_search = RAGSearch()
#     # query = "What is attention mechanism?"
#     # summary = rag_search.search_and_summarize(query, top_k=3)
#     # print("Summary:", summary)
#     # print(docs)

#     query = "What is the recommended treatment for tuberculosis?"
#     retrieved = store.query(query, top_k=5)
#     reranked = reranker.rerank(query, retrieved, top_k=3)
#     result = generate(query, reranked, retrieved)
#     print(json.dumps(result, indent=2))