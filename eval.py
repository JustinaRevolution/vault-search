#!/usr/bin/env python3
"""Evaluate vault-search against a JSONL set of {"query": ..., "gold": [path substrings]}.

A query counts as a hit at rank r if the r-th result's path contains any gold substring.
Reports hit@1, hit@5, hit@10 and MRR@10, plus the misses.

    python3 eval.py eval/household.jsonl [--k 10] [--show-misses]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vault_search as vs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("evalset")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--show-misses", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--thorough", action="store_true")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.evalset) if l.strip()]
    h1 = h5 = h10 = 0
    rr = 0.0
    misses = []
    import time
    t0 = time.time()
    for it in items:
        results = vs.search(it["query"], k=a.k, use_rerank=not a.no_rerank, thorough=a.thorough)
        rank = next((i + 1 for i, r in enumerate(results)
                     if any(g in r["path"] for g in it["gold"])), None)
        if rank:
            h1 += rank <= 1; h5 += rank <= 5; h10 += rank <= 10
            rr += 1 / rank
        if not rank or rank > 5:
            misses.append((it["query"], rank, [r["path"] for r in results[:3]]))
    n = len(items)
    print(f"[{'vector' if a.no_rerank else 'rerank'}] {(time.time()-t0)/n:.2f}s/query  n={n}  hit@1={h1/n:.2f}  hit@5={h5/n:.2f}  hit@10={h10/n:.2f}  MRR@10={rr/n:.3f}")
    if a.show_misses:
        for q, rank, top in misses:
            print(f"  MISS(rank={rank}) {q!r}\n      top: {top}")


if __name__ == "__main__":
    main()
