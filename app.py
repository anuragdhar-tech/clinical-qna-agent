from src.data_loader import load_all_documents
from src.vectorstore_faiss import FaissVectorStore
from src.vectorstore_chroma import ChromaVectorStore
# from src.search import RAGSearch
from src.embeddings import EmbeddingPipeline
from src.reranker import Reranker
from src.generator import generate
import json

# Example usage
if __name__ == "__main__":
    
    # docs = load_all_documents("data")
    # emb_pipe = EmbeddingPipeline()
    # chunks = emb_pipe.chunk_documents(docs)
    # chunks_vector = emb_pipe.embed_chunks(chunks)
    # print("[INFO] Example embedding:", chunks_vector[0] if len(chunks_vector) > 0 else None)

    ## FAISS VECTOR STOR
    # store = FaissVectorStore("faiss_store")
    # store.build_from_documents(docs)
    # store = FaissVectorStore("faiss_store")
    # store.load()
    # print(store.query("What is diabetes?", top_k=3))

    ## CHROMADB
    # store = ChromaVectorStore("chroma_store")
    # store.build_from_documents(docs)
    store = ChromaVectorStore("chroma_store")
    store.load()
    # # print(store.query("What is Pneumonia?", top_k=3))
    # results = store.query("How does diabetes affect TB treatment outcomes?", top_k = 3)

    reranker = Reranker()
    # re_ranked = reranker.rerank("How does diabetes affect TB treatment outcomes?", results, top_k=3)
    # for r in re_ranked:
    #     print(r["rerank_score"], r["source"], r["page"], r["text"][:100])


    # store.load()
    # #print(store.query("What is attention mechanism?", top_k=3))
    # rag_search = RAGSearch()
    # query = "What is attention mechanism?"
    # summary = rag_search.search_and_summarize(query, top_k=3)
    # print("Summary:", summary)
    # print(docs)

    query = "What is the recommended treatment for tuberculosis?"
    retrieved = store.query(query, top_k=5)
    reranked = reranker.rerank(query, retrieved, top_k=3)
    result = generate(query, reranked, retrieved)