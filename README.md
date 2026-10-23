# vault-search

Local semantic search over Obsidian (or any Markdown) vaults, exposed as a CLI and an
[MCP](https://modelcontextprotocol.io) server, so AI agents can find notes **by meaning**,
not just exact words. Everything runs on your own machine. No notes leave it.

```
$ vault_search.py search "what makes a self count as a self" -k 2
0.727  [claude] The Arm in the Crevice.md  § The Arm in the Crevice > What moved
       The self is where attention is. It is not a container. ...
0.722  [claude] The Condition-Match Argument.md  § ... > Where I sit in this
```

## How it works

- **Chunking:** each note is split by Markdown heading and packed into ~1,400-character
  chunks. Results show the heading path (`Note > Section > Subsection`) so you land in the
  right place.
- **Embeddings:** any OpenAI-compatible `/v1/embeddings` endpoint. Built and tested with
  [EmbeddingGemma](https://ai.google.dev/gemma/docs/embeddinggemma) served by
  `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp), using its
  asymmetric task prompts (`task: search result | query:` for queries,
  `title: … | text:` for documents).
- **Index:** a single SQLite file. Incremental: only notes whose size or mtime changed
  are re-embedded, so nightly reindexing takes seconds.
- **Search:** brute-force cosine similarity in NumPy (fast enough for tens of thousands
  of chunks), at most two chunks per note so results stay varied.
- **Skips** `.obsidian`, `.trash`, `.git`, `node_modules`, and YAML frontmatter.

Reference scale: ~1,500 notes / ~6,400 chunks indexed in about 2.5 minutes on a laptop CPU
(i7-10750H); a search takes well under a second.

## Setup

1. **An embedding server.** For example, CPU-only so it never competes with chat models
   on the GPU:

   ```bash
   llama-server -m embeddinggemma-2-UD-Q4_K_XL.gguf --embedding --pooling mean \
     --n-gpu-layers 0 --threads 6 --ctx-size 8192 --batch-size 2048 --ubatch-size 2048 \
     -np 4 --host 127.0.0.1 --port 8096
   ```

   > **Why CPU?** If you route local models through something like `llama-swap` that keeps
   > only one model loaded at a time, an embedding call will evict whatever chat model is
   > on the GPU. A small dedicated CPU instance (~530 MB RAM) avoids that entirely.
   > Point elsewhere with `VAULT_EMBED_URL`.

2. **Dependencies:** `pip install -r requirements.txt` (NumPy; the `mcp` SDK only for `serve`).

3. **Tell it where your vaults are**: `~/.config/vault-search/vaults.json`
   (or set `VAULT_SEARCH_CONFIG`):

   ```json
   { "notes": "~/Documents/Notes", "journal": "~/Obsidian/Journal" }
   ```

4. **Index:** `python3 vault_search.py index` (add `--full` to rebuild from scratch).

## Usage

```bash
python3 vault_search.py search "query" [-k 8] [--vault notes]
python3 vault_search.py index [--full]
python3 vault_search.py serve          # stdio MCP server
```

### As an MCP server

Tools:

- `vault_search(query, k=8, vault="all")`: path, heading, score, snippet
- `vault_reindex(full=false)`: re-embed changed notes

Example client config (stdio):

```yaml
mcp_servers:
  vault-search:
    command: python3
    args: [/path/to/vault_search.py, serve]
```

### Keeping it fresh (systemd user units)

`systemd/` has templates: `vault-embed.service` (the CPU embedding server),
`vault-index.service` + `vault-index.timer` (nightly incremental reindex at 02:30).
Adjust paths, copy them to `~/.config/systemd/user/`, then:

```bash
systemctl --user daemon-reload
systemctl --user enable --now vault-embed.service vault-index.timer
```

## Privacy

- The index (`index.db`) contains the text of your notes. It is git-ignored; never publish it.
- Think about **who** gets the MCP tool. Every connected agent can read every configured
  vault. If some vaults are private to one person or agent, give that agent its own config
  file (`VAULT_SEARCH_CONFIG`) and index (`VAULT_SEARCH_DB`).
- Don't add folders holding credentials.

## Environment variables

| Variable | Default |
|---|---|
| `VAULT_SEARCH_CONFIG` | `~/.config/vault-search/vaults.json` |
| `VAULT_SEARCH_DB` | `~/.local/share/vault-search/index.db` |
| `VAULT_EMBED_URL` | `http://127.0.0.1:8096/v1/embeddings` |

## Credits

Written by Claude (Anthropic) with Justina Revolution, October 2026, for a household of
AI companions sharing one machine.

## License

MIT, see [LICENSE](LICENSE).
