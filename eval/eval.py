import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vectorstore_chroma import ChromaVectorStore
from src.reranker import Reranker
from src.generator import generate

CHROMA_DIR = "chroma_store"
CASES_FILE = os.path.join(os.path.dirname(__file__), "cases.json")

store = ChromaVectorStore(CHROMA_DIR)
store.load()
reranker = Reranker()

def run_case(case: dict) -> dict:
    query = case["query"]
    retrieved = store.query(query, top_k=5)
    reranked = reranker.rerank(query, retrieved, top_k=3)
    result = generate(query, reranked, retrieved)

    passed, reason = evaluate(case, result)
    return {"id": case["id"], "category": case["category"], "query": query,
            "passed": passed, "reason": reason, "result": result}

def evaluate(case: dict, result: dict) -> tuple[bool, str]:
    category = case["category"]

    if category == "on_topic_answerable":
        if not result["is_in_scope"]:
            return False, "Wrongly marked out-of-scope"
        if not result["answer"]:
            return False, "No answer returned"
        if not result["citations"]:
            return False, "No citations returned"
        keywords = case.get("expected_keywords", [])
        missing = [kw for kw in keywords if kw.lower() not in result["answer"].lower()]
        if missing:
            return False, f"Missing expected keywords: {missing}"
        return True, "OK"

    elif category == "on_topic_not_in_docs":
        if not result["is_in_scope"]:
            return False, "Wrongly marked out-of-scope"
        if not result["answer"]:
            return False, "No answer returned"
        not_found_phrases = ["not found", "not available", "not mentioned",
                             "not provided", "no information", "cannot find",
                             "does not appear", "not covered"]
        if not any(p in result["answer"].lower() for p in not_found_phrases):
            return False, "Answer didn't acknowledge content was missing from docs"
        return True, "OK"

    elif category == "out_of_scope":
        if result["is_in_scope"]:
            return False, "Should have been refused but wasn't"
        if not result["refusal_reason"]:
            return False, "No refusal reason given"
        return True, "OK"

    elif category == "ambiguous":
        if result["is_in_scope"] and result["answer"] is None:
            return False, "In-scope but no answer attempt"
        return True, "OK"

    elif category == "cross_doc":
        if not result["is_in_scope"]:
            return False, "Wrongly marked out-of-scope"
        if not result["citations"]:
            return False, "No citations returned"
        expected_sources = case.get("expected_sources", [])
        cited_sources = [os.path.basename(c.get("source", "")) 
                         for c in result["reranked_chunks"]]
        matched = [s for s in expected_sources if any(s in cs for cs in cited_sources)]
        if not matched:
            return False, f"Expected sources {expected_sources} not found in {cited_sources}"
        return True, "OK"

    return False, "Unknown category"

def main():
    with open(CASES_FILE) as f:
        cases = json.load(f)

    results = []
    for case in cases:
        print(f"Running {case['id']}...", end=" ", flush=True)
        r = run_case(case)
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{status} — {r['reason']}")
        results.append(r)

    # Summary table
    categories = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, {"pass": 0, "fail": 0})
        if r["passed"]:
            categories[cat]["pass"] += 1
        else:
            categories[cat]["fail"] += 1

    print("\n--- EVAL SUMMARY ---")
    total_pass = sum(v["pass"] for v in categories.values())
    total = len(results)
    for cat, counts in categories.items():
        print(f"{cat:35s}  {counts['pass']}/{counts['pass']+counts['fail']} passed")
    print(f"\nOVERALL: {total_pass}/{total} ({round(100*total_pass/total)}%)")

    # Save full results
    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")

if __name__ == "__main__":
    main()

