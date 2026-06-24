# Clinical Q&A Agent

A production-grade RAG pipeline that answers clinical guideline questions grounded strictly in WHO, CDC, and NHS public documents. Every answer cites its source. Out-of-scope questions are refused explicitly — no fabrication, no guessing.

Built as a portfolio-grade implementation of Retrieval-Augmented Generation with CrossEncoder reranking, structured JSON outputs at every pipeline step, and a 28-case evaluation harness across five failure-mode categories.

---

## What It Does

- Answers questions grounded in loaded clinical guideline PDFs
- Returns structured JSON with answer, citations, retrieved chunks, and reranked chunks
- Refuses out-of-scope questions (personal medical advice, diagnosis, unrelated topics) before any LLM call
- Flags when a question is in-scope but the answer is not in the loaded documents — without fabricating

---

## Pipeline

```
INGEST (run once)
─────────────────────────────────────────
PDF files in data/
    ↓  data_loader.py       PyPDFLoader — preserves source + page metadata
    ↓  embeddings.py        RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
                            → SentenceTransformer all-MiniLM-L6-v2
    ↓  vectorstore_chroma.py  ChromaDB PersistentClient → chroma_store/

QUERY (run per question)
─────────────────────────────────────────
User query
    ↓  generator.py         is_in_scope() keyword gate
    ├── OUT-OF-SCOPE → refusal JSON returned immediately (no LLM call)
    └── IN-SCOPE ↓
    ↓  vectorstore_chroma.py  cosine similarity → top-5 chunks
    ↓  reranker.py            CrossEncoder ms-marco-MiniLM-L-6-v2 → top-3
    ↓  generator.py           LCEL chain: ChatPromptTemplate | ChatOpenAI | JsonOutputParser
    ↓  Structured JSON output
```

---

## Output Schema

Every query returns a structured JSON object:

```json
{
  "query": "What is the recommended treatment for tuberculosis?",
  "retrieved_chunks": [
    { "text": "...", "source": "TheUnion_DMTB_Guide.pdf", "score": 0.563 }
  ],
  "reranked_chunks": [
    { "text": "...", "source": "TheUnion_DMTB_Guide.pdf", "rerank_score": 7.977 }
  ],
  "answer": "The recommended treatment for new drug-susceptible TB is...",
  "citations": ["TheUnion_DMTB_Guide.pdf: page 62"],
  "is_in_scope": true,
  "refusal_reason": null
}
```

---

## Documents

Place clinical guideline PDFs in `data/`. The pipeline works with any publicly available clinical PDF. Suggested sources:

- WHO publications — who.int/publications
- CDC clinical guidance — cdc.gov
- NICE guidelines — nice.org.uk

Start with 3–5 documents across 2–3 distinct clinical domains — small enough to debug, varied enough to test cross-document retrieval.

---

## Project Structure

```
clinical-qna-agent/
├── data/                      # Clinical guideline PDFs (not committed)
├── chroma_store/              # Persisted vector index (not committed)
├── src/
│   ├── data_loader.py         # PDF loading with metadata preservation
│   ├── embeddings.py          # Chunking + SentenceTransformer embeddings
│   ├── vectorstore_chroma.py  # ChromaDB build, persist, load, query
│   ├── reranker.py            # CrossEncoder reranking
│   └── generator.py           # Scope gate, context builder, LCEL chain
├── eval/
│   ├── cases.json             # 28 test cases across 5 categories
│   └── eval.py                # Automated runner with per-category scoring
├── app.py                     # CLI — ingest and search pipelines
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/anuragdhar-tech/clinical-qna-agent.git
cd clinical-qna-agent
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
```

Add your own clinical guideline PDFs to `data/`. The pipeline works with any publicly available PDF — WHO, CDC, NHS, or similar. Start with 3–5 documents covering 2–3 distinct clinical domains.

---

## Running

**Ingest documents (run once, or when docs change):**
```bash
python app.py ingest
```

**Query the agent:**
```bash
python app.py search "What is the first-line treatment for drug-susceptible TB?"
```

---

## Evaluation

The eval harness runs 28 test cases across five failure-mode categories:

| Category | Cases | What It Tests |
|----------|-------|---------------|
| On-topic, answerable | 10 | Core retrieval + citation accuracy |
| On-topic, not in docs | 5 | Grounding — no fabrication when answer absent |
| Out-of-scope | 5 | Scope gate fires, no LLM call |
| Ambiguous query | 5 | Graceful handling, no hallucination |
| Cross-doc retrieval | 3 | Reranker surfaces correct source when docs overlap |

**Run the eval harness:**
```bash
python eval/eval.py
```

Results are logged per case with pass/fail and a per-category summary score. Overall percentage is reported but category breakdown is the signal — a 0/5 in one category masks a real gap that overall score hides.

---

## Key Design Decisions

**ChromaDB over FAISS** — Citation accuracy requires per-chunk source metadata. ChromaDB tracks this natively. FAISS requires a parallel data structure; extra failure surface for a citation-critical use case.

**Retrieve 5, rerank to 3** — Semantic similarity is not the same as relevance. Giving the CrossEncoder more candidates (top-5) produces better final context than retrieving 3 directly. The bi-encoder retrieves fast and approximate; the CrossEncoder reranks slow and precise.

**Scope gate before LLM** — Keyword check fires before any LLM call. Zero token cost for clearly out-of-scope queries. Deterministic for known categories. Prompt-only refusal still costs a full LLM call and can fail under adversarial input.

**Two hallucination defences** — Prompt instruction ("only use the provided context") plus an explicit "information not found in guidelines" fallback phrase. Single-layer prompt defence fails when context is insufficient and the LLM defaults to training knowledge.

**Structured JSON at every step** — Retrieved chunks, reranked chunks, answer, citations, and scope decision all emitted as structured JSON. Machine-readable audit trail, not just a text answer.

---

## Stack

- **Python 3.11+**
- **LangChain** — LCEL chains, document loaders, text splitters
- **ChromaDB** — persistent vector store
- **SentenceTransformers** — all-MiniLM-L6-v2 (embeddings), ms-marco-MiniLM-L-6-v2 (reranking)
- **OpenAI** — GPT-4o-mini via ChatOpenAI
- **PyPDF** — PDF loading

---

[linkedin.com/in/anuragdhar](https://linkedin.com/in/anuragdhar)