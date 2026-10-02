# langchain-hydradb

HydraDB integration for [LangChain](https://langchain.com): give your chains and
agents persistent, cross-session memory and retrieval powered by
[HydraDB](https://hydradb.com).

## Install

```bash
pip install langchain-hydradb   # once published; for local dev: pip install -e .
```

## Prerequisites

- Python >= 3.9
- A HydraDB account: API key + tenant ID ([hydradb.com](https://hydradb.com))
- `langchain-core >= 0.3`

## Quick start

```python
from langchain_hydradb import HydraDBClient, HydraDBRetriever

client = HydraDBClient(api_key="sk_live_...", tenant_id="your-tenant")

# Use HydraDB as a retriever in any chain
retriever = HydraDBRetriever(client=client, kind="both", max_results=6)
docs = retriever.invoke("what did we decide about auth?")

# Write durable memory
client.add_text("We deploy on Fridays only.", infer=True)
```

### As chat memory

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_hydradb import HydraDBChatMessageHistory

chain_with_memory = RunnableWithMessageHistory(
    chain,
    lambda session_id: HydraDBChatMessageHistory(client, session_id),
    input_messages_key="input",
    history_messages_key="history",
)
```

## What's included

| Class | Role |
|---|---|
| `HydraDBClient` | Thin HTTP client, public API only (`POST /query`, `POST /context/ingest`, `DELETE /context`) |
| `HydraDBRetriever` | `BaseRetriever` over memory, knowledge, or both |
| `HydraDBChatMessageHistory` | `BaseChatMessageHistory` that persists + recalls turns |

## Notes

- **Public API only.** Writes use the public knowledge path (`app_knowledge` via
  `/context/ingest`); the memory-family write route is not publicly exposed, so
  `add_text`/`add_conversation` store as knowledge. Recall `kind` may still be
  `"memory"`, `"knowledge"`, or `"both"` - default is `"knowledge"`.
- Ingestion is asynchronous; recall a source only after it finishes indexing.
- The client depends only on `requests`; no HydraDB SDK required.

## Test

```bash
pip install -e ".[test]"
pytest
```

## License

Apache-2.0 - Copyright (c) 2026 HydraDB
