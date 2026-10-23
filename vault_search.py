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
def search(query: str, k: int = 8, vault: str | None = None) -> list[dict]:
    con = db()
    sql = "SELECT path, vault, heading, text, emb FROM chunks" + (" WHERE vault=?" if vault else "")
    rows = con.execute(sql, (vault,) if vault else ()).fetchall()
    if not rows:
        return []
    q = embed([QUERY_PREFIX + query])[0]
    mat = np.frombuffer(b"".join(r[4] for r in rows), dtype=np.float32).reshape(len(rows), -1)
    scores = mat @ q
    order = np.argsort(-scores)
    out, per_file = [], {}
    for idx in order:
        path = rows[idx][0]
        if per_file.get(path, 0) >= 2:  # at most 2 chunks per note, for variety
            continue
        per_file[path] = per_file.get(path, 0) + 1
        p, v, h, t, _ = rows[idx]
        root = VAULTS[v]
        snippet = re.sub(r"\s+", " ", t).strip()
        out.append({
            "vault": v,
            "path": str(Path(p).relative_to(root)) if p.startswith(str(root)) else p,
            "heading": h or "",
            "score": round(float(scores[idx]), 3),
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
    def vault_search(query: str, k: int = 8, vault: str = "all") -> str:
        """Semantic search over the configured Markdown vaults by meaning, not exact words.
        vault: 'all' or one configured vault name. Returns path (relative to the vault root),
        heading, similarity score, and a snippet."""
        v = None if vault in ("all", "", None) else vault
        if v and v not in VAULTS:
            return json.dumps({"error": f"vault must be 'all' or one of: {', '.join(VAULTS)}"})
        return json.dumps({"query": query, "results": search(query, max(1, min(int(k), 25)), v)}, ensure_ascii=False)

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
    sub.add_parser("serve")
    a = ap.parse_args()
    if a.cmd == "index":
        print(json.dumps(index(full=a.full), indent=2))
    elif a.cmd == "search":
        for r in search(a.query, a.k, a.vault):
            print(f"{r['score']:.3f}  [{r['vault']}] {r['path']}  § {r['heading']}\n       {r['snippet'][:160]}")
    else:
        serve()


if __name__ == "__main__":
    main()
