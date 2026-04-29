# ferrum-memory

Agent working memory and experience replay buffer for the Ferrum Platform.

Serves [forge-agent](https://github.com/dattgoswami/forge-agent), [ferrum-agent-runtime](https://github.com/dattgoswami/ferrum-agent), and [ferrum-evals](https://github.com/dattgoswami/ferrum-evals). Unlike a general vector DB, ferrum-memory is a **continual learning instrument**: storing RL experience tuples (task, tool_calls, test_result, reward) and implementing prioritized experience replay (PER) for the dream cycle.

## Architecture

```
ferrum-memory/
├── pyproject.toml
├── contracts/                          # Domain contracts (schemas)
│   ├── memory_item.py                  # MemoryItem, MemoryKind, MemorySource
│   ├── experience_tuple.py             # ExperienceTuple, ToolCallRecord, TestResult
│   └── working_memory.py              # WorkingMemoryState, SessionSummary
├── ferrum_memory/
│   ├── config.py                       # Env var config with spec defaults
│   ├── main.py                         # FastAPI app factory, lifespan
│   ├── storage/                        # Storage backends
│   │   ├── router.py                   # StorageRouter (dispatch by type)
│   │   ├── qdrant_store.py             # Hybrid BM25 + dense (Qdrant)
│   │   ├── sqlite_store.py             # Experience CRUD (aiosqlite, WAL)
│   │   └── redis_store.py              # WorkingMemoryState (Redis, TTL)
│   ├── retrieval/                      # Replay buffer
│   │   ├── replay.py                   # PER + recency + uniform samplers
│   │   ├── hybrid.py                   # BM25 + dense + RRF fusion
│   │   └── reranker.py                 # Cross-encoder rerank (opt-in)
│   ├── api/                            # FastAPI routers
│   │   ├── memory.py                   # POST/DELETE /memory/*, search
│   │   ├── experience.py               # POST /experience, /replay
│   │   └── session.py                  # POST/GET/PUT /session/*
│   └── lifecycle/
│       └── consolidation.py            # Session close → MemoryItem compression
└── tests/
    ├── conftest.py                     # Shared fixtures
    ├── unit/                           # Unit tests (100% contracts coverage)
    └── integration/                    # Integration tests (testcontainers)
```

## Quick Start

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pip install pytest pytest-asyncio pytest-cov
pytest tests/unit/ --cov=contracts --cov=ferrum_memory --cov-report=term-missing -v
```

## Configuration

All configuration is via environment variables with sensible defaults:

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector DB URL |
| `QDRANT_COLLECTION` | `ferrum_memory` | Qdrant collection name |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL for working memory |
| `SQLITE_PATH` | `:memory:` | SQLite path (use file path for persistence) |
| `FASTEMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `DEFAULT_REPLAY_TTL` | `86400` | Replay buffer TTL in seconds |
| `DEFAULT_WORKING_MEM_TTL` | `86400` | Working memory TTL in seconds |
| `OTEL_SERVICE_NAME` | `ferrum-memory` | OpenTelemetry service name |

## API Endpoints

### Memory

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/memory/store` | Store a memory item |
| `DELETE` | `/memory/{point_id}` | Delete a memory item |
| `POST` | `/memory/search` | Search memory items (dense + BM25 + RRF) |
| `POST` | `/memory/compress?session_id=...` | Compress session working memory |

### Experience

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/experience/` | Store an experience tuple |
| `GET` | `/experience/stats` | Get experience statistics |
| `PUT` | `/experience/td-error` | Update TD error for an experience |
| `GET` | `/experience/replay` | Get replay samples (prioritized/recency/uniform) |

### Session

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/session/start` | Start a new session |
| `GET` | `/session/{session_id}` | Get current working memory |
| `PUT` | `/session/{session_id}` | Update working memory |
| `POST` | `/session/close` | Close session (triggers consolidation) |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Backend health status |
| `GET` | `/stats` | Global statistics |
| `GET` | `/openapi.json` | OpenAPI schema |

## Replay Strategies

The `/experience/replay` endpoint supports three sampling strategies:

- **`prioritized`** (default) — Schaul et al. 2015 PER. Samples proportionally to `|td_error|^alpha`. Higher TD error = higher sampling probability.
- **`recency`** — Exponential decay. Weight at `t=half_life` is ~0.5. Recent experiences are sampled more frequently.
- **`uniform`** — Uniform random sampling. Each experience has equal probability.

## Running Backends

For production use, run the required backends:

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

```bash
docker compose up -d
```

## Run Integration Tests

```bash
pip install "pytest-testcontainers>=1.0"
docker compose up -d
pytest tests/integration/ -v
```

## Design Principles

1. **No handler calls two backends** — each API route targets exactly one storage backend
2. **Opt-in reranking** — cross-encoder reranker is only activated by `?rerank=true`
3. **WAL mode SQLite** — better concurrency for experience storage
4. **TTL-based expiration** — both Redis working memory and Qdrant memory items support TTL

## Test Coverage

```bash
pytest tests/unit/ --cov=contracts --cov=ferrum_memory --cov-report=term-missing --cov-fail-under=95 -v
```

## License

MIT
