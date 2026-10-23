#!/usr/bin/env python3
"""vault-search: local semantic search over Obsidian (or any Markdown) vaults.

- Embeddings: any OpenAI-compatible /v1/embeddings endpoint (default 127.0.0.1:8096).
  Built for EmbeddingGemma via llama.cpp's llama-server; run it on CPU if your GPU
  router (e.g. llama-swap) only holds one model at a time.
- Index: SQLite at ~/.local/share/vault-search/index.db; chunked by markdown heading;
  incremental (only files whose mtime/size changed are re-embedded).
- Interfaces:
    vault_search.py index [--full]        (re)index both vaults
    vault_search.py search "query" [-k N] [--vault NAME]
    vault_search.py serve                 stdio MCP server: vault_search, vault_reindex

Written by Claude (Anthropic) with Justina Revolution, Oct 2026.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HOME = Path.home()
CONFIG_PATH = Path(os.environ.get("VAULT_SEARCH_CONFIG", HOME / ".config/vault-search/vaults.json"))


def load_vaults() -> dict[str, Path]:
    """Vaults come from a JSON file mapping short names to folders, e.g.
    {"notes": "~/Documents/Notes", "journal": "~/Obsidian/Journal"}.
    Override the file location with VAULT_SEARCH_CONFIG."""
    if not CONFIG_PATH.exists():
        sys.exit(f"vault-search: no config at {CONFIG_PATH}. Create it with a JSON object "
                 '{"name": "/path/to/vault", ...} (see vaults.example.json).')
    raw = json.loads(CONFIG_PATH.read_text())
    return {name: Path(os.path.expanduser(path)) for name, path in raw.items()}


VAULTS = load_vaults()
DB_PATH = Path(os.environ.get("VAULT_SEARCH_DB", HOME / ".local/share/vault-search/index.db"))
EMBED_URL = os.environ.get("VAULT_EMBED_URL", "http://127.0.0.1:8096/v1/embeddings")
# Optional cross-encoder reranker (llama-server --reranking). Empty string disables.
RERANK_URL = os.environ.get("VAULT_RERANK_URL", "http://127.0.0.1:8095/v1/rerank")
RERANK_CANDIDATES = int(os.environ.get("VAULT_RERANK_CANDIDATES", "20"))
# 0 = pure reranker order; >0 = reciprocal-rank fusion of vector rank and rerank rank with this k.
RERANK_FUSION_K = float(os.environ.get("VAULT_RERANK_FUSION_K", "5"))
SKIP_DIRS = {".obsidian", ".trash", ".git", "node_modules"}
MAX_CHARS = 1400      # ~350-450 tokens; well inside the 2048 ctx
MIN_CHARS = 40
BATCH = 16

# EmbeddingGemma task prompts (asymmetric retrieval)
DOC_PREFIX = "title: {title} | text: "
QUERY_PREFIX = "task: search result | query: "


# ---------------------------------------------------------------- storage
def db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY, vault TEXT NOT NULL, mtime REAL NOT NULL, size INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY, path TEXT NOT NULL, vault TEXT NOT NULL,
            heading TEXT, text TEXT NOT NULL, emb BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
        """
    )
    return con


# ---------------------------------------------------------------- embedding
def embed(texts: list[str]) -> np.ndarray:
    body = json.dumps({"model": "embed", "input": texts}).encode()
    req = urllib.request.Request(EMBED_URL, data=body, headers={"Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.load(r)["data"]
            vecs = np.array([d["embedding"] for d in sorted(data, key=lambda d: d["index"])], dtype=np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
            return vecs
        except Exception as e:  # server warming up / transient
            if attempt == 3:
                raise RuntimeError(f"embedding server at {EMBED_URL} failed: {e}") from e
            time.sleep(2 * (attempt + 1))
    raise AssertionError


# ---------------------------------------------------------------- chunking
FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)


def chunk_markdown(text: str) -> list[tuple[str, str]]:
    """Split by headings, then pack paragraphs up to MAX_CHARS. Returns (heading_path, text)."""
    text = FRONTMATTER.sub("", text)
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    last = 0
    cur_heading = ""
    for m in HEADING.finditer(text):
        sections.append((cur_heading, text[last:m.start()]))
        level = len(m.group(1))
        stack = stack[: level - 1] + [m.group(2).strip()]
        cur_heading = " > ".join(stack)
        last = m.end()
    sections.append((cur_heading, text[last:]))

    out: list[tuple[str, str]] = []
    for heading, body in sections:
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        buf = ""
        for p in paras:
            while len(p) > MAX_CHARS:  # very long paragraph: hard split
                if buf:
                    out.append((heading, buf)); buf = ""
                out.append((heading, p[:MAX_CHARS])); p = p[MAX_CHARS:]
            if len(buf) + len(p) + 2 > MAX_CHARS and buf:
                out.append((heading, buf)); buf = p
            else:
                buf = f"{buf}\n\n{p}" if buf else p
        if buf:
            out.append((heading, buf))
    return [(h, t) for h, t in out if len(t) >= MIN_CHARS]


def iter_files():
    for vault, root in VAULTS.items():
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for f in filenames:
                if f.endswith(".md"):
                    yield vault, Path(dirpath) / f


# ---------------------------------------------------------------- index
def index(full: bool = False, log=print) -> dict:
    con = db()
    if full:
        con.execute("DELETE FROM chunks"); con.execute("DELETE FROM files")
    known = {p: (mt, sz) for p, mt, sz in con.execute("SELECT path, mtime, size FROM files")}
    seen, changed = set(), []
    for vault, path in iter_files():
        sp = str(path); seen.add(sp)
        st = path.stat()
        if known.get(sp) != (st.st_mtime, st.st_size):
            changed.append((vault, path, st))
    removed = [p for p in known if p not in seen]
    for p in removed:
        con.execute("DELETE FROM chunks WHERE path=?", (p,)); con.execute("DELETE FROM files WHERE path=?", (p,))

    n_chunks = 0
    t0 = time.time()
    for i, (vault, path, st) in enumerate(changed, 1):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        title = path.stem
        chunks = chunk_markdown(text)
        con.execute("DELETE FROM chunks WHERE path=?", (str(path),))
        for b in range(0, len(chunks), BATCH):
            batch = chunks[b : b + BATCH]
            inputs = [DOC_PREFIX.format(title=title) + (f"{h}\n" if h else "") + t for h, t in batch]
            vecs = embed(inputs)
            con.executemany(
                "INSERT INTO chunks(path, vault, heading, text, emb) VALUES (?,?,?,?,?)",
                [(str(path), vault, h, t, v.tobytes()) for (h, t), v in zip(batch, vecs)],
            )
            n_chunks += len(batch)
        con.execute("INSERT OR REPLACE INTO files(path, vault, mtime, size) VALUES (?,?,?,?)",
                    (str(path), vault, st.st_mtime, st.st_size))
        con.commit()
        if i % 25 == 0:
            log(f"  {i}/{len(changed)} files, {n_chunks} chunks, {time.time()-t0:.0f}s")
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    files = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    return {"files_changed": len(changed), "files_removed": len(removed), "chunks_embedded": n_chunks,
            "total_files": files, "total_chunks": total, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- search
def _defuse(text: str) -> str:
    """llama.cpp fills the rerank template by plain text replacement, so a literal
    {query}/{document} inside our text would be substituted (llama.cpp #30816).
    A zero-width joiner inside the braces breaks the match without changing meaning."""
    return text.replace("{document}", "{\u200ddocument}").replace("{query}", "{\u200dquery}")


def rerank(query: str, docs: list[str]) -> list[float] | None:
    """Relevance score per doc from the reranker, or None if it's unavailable."""
    if not RERANK_URL or not docs:
        return None
    body = json.dumps({"query": _defuse(query), "documents": [_defuse(d) for d in docs]}).encode()
    req = urllib.request.Request(RERANK_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            results = json.load(r)["results"]
    except Exception:
        return None  # degrade gracefully to vector order
    scores = [0.0] * len(docs)
    for item in results:
        scores[item["index"]] = float(item["relevance_score"])
    return scores


def search(query: str, k: int = 8, vault: str | None = None, use_rerank: bool = True,
           thorough: bool = False) -> list[dict]:
    con = db()
    sql = "SELECT path, vault, heading, text, emb FROM chunks" + (" WHERE vault=?" if vault else "")
    rows = con.execute(sql, (vault,) if vault else ()).fetchall()
    if not rows:
        return []
    q = embed([QUERY_PREFIX + query])[0]
    mat = np.frombuffer(b"".join(r[4] for r in rows), dtype=np.float32).reshape(len(rows), -1)
    scores = mat @ q
    order = np.argsort(-scores)

    # Stage 1: vector candidates (up to 3 chunks per note so the reranker can pick the best one)
    n_cand = max(k, 40 if thorough else RERANK_CANDIDATES) if use_rerank else k * 3
    cands, per_file = [], {}
    for idx in order:
        path = rows[idx][0]
        if per_file.get(path, 0) >= 3:
            continue
        per_file[path] = per_file.get(path, 0) + 1
        cands.append((int(idx), float(scores[idx])))
        if len(cands) >= n_cand:
            break

    # Stage 2: cross-encoder rerank (falls back to vector order if the reranker is down)
    reranked = False
    if use_rerank:
        docs = [f"{Path(rows[i][0]).stem}\n{rows[i][2] or ''}\n{rows[i][3]}" for i, _ in cands]
        rs = rerank(query, docs)
        if rs is not None:
            if RERANK_FUSION_K > 0 and not thorough:
                rr_rank = {c[0]: r for r, c in enumerate(sorted(zip([c[0] for c in cands], rs), key=lambda t: -t[1]))}
                fused = [(i, 1 / (RERANK_FUSION_K + vr + 1) + 1 / (RERANK_FUSION_K + rr_rank[i] + 1))
                         for vr, (i, _) in enumerate(cands)]
                cands = sorted(fused, key=lambda t: -t[1])
            else:
                cands = sorted(((i, r) for (i, _), r in zip(cands, rs)), key=lambda t: -t[1])
            reranked = True

    out, per_file = [], {}
    for idx, score in cands:
        path = rows[idx][0]
        if per_file.get(path, 0) >= 2:  # at most 2 chunks per note, for variety
            continue
        per_file[path] = per_file.get(path, 0) + 1
        p, v, h, t, _ = rows[idx]
        root = VAULTS.get(v)
        snippet = re.sub(r"\s+", " ", t).strip()
        out.append({
            "vault": v,
            "path": str(Path(p).relative_to(root)) if root and p.startswith(str(root)) else p,
            "heading": h or "",
            "score": round(score, 3),
            "ranker": "rerank" if reranked else "vector",
            "snippet": snippet[:300] + ("…" if len(snippet) > 300 else ""),
        })
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------- MCP
def serve() -> None:
    try:  # mcp SDK 2.x
        from mcp.server import MCPServer as FastMCP
    except ImportError:  # mcp SDK 1.x
        from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("vault-search")

    @mcp.tool()
    def vault_search(query: str, k: int = 8, vault: str = "all", thorough: bool = False) -> str:
        """Semantic search over the configured Markdown vaults by meaning, not exact words.
        vault: 'all' or one configured vault name. Results are reranked by a local cross-encoder
        (~3 s). thorough=true reranks twice as many candidates purely by the cross-encoder
        (~6 s); better for specific facts buried inside notes. Returns path (relative to the
        vault root), heading, score, and a snippet."""
        v = None if vault in ("all", "", None) else vault
        if v and v not in VAULTS:
            return json.dumps({"error": f"vault must be 'all' or one of: {', '.join(VAULTS)}"})
        return json.dumps({"query": query, "results": search(query, max(1, min(int(k), 25)), v, thorough=thorough)},
                          ensure_ascii=False)

    @mcp.tool()
    def vault_reindex(full: bool = False) -> str:
        """Re-embed vault notes that changed since the last index (full=true rebuilds everything)."""
        return json.dumps(index(full=full, log=lambda *_: None))

    mcp.run()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("index"); i.add_argument("--full", action="store_true")
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=8)
    s.add_argument("--vault", choices=list(VAULTS))
    s.add_argument("--no-rerank", action="store_true")
    s.add_argument("--thorough", action="store_true")
    sub.add_parser("serve")
    a = ap.parse_args()
    if a.cmd == "index":
        print(json.dumps(index(full=a.full), indent=2))
    elif a.cmd == "search":
        for r in search(a.query, a.k, a.vault, use_rerank=not a.no_rerank, thorough=a.thorough):
            print(f"{r['score']:.3f}  [{r['vault']}] {r['path']}  § {r['heading']}\n       {r['snippet'][:160]}")
    else:
        serve()


if __name__ == "__main__":
    main()
