import json

import responses
from langchain_core.messages import AIMessage, HumanMessage

from langchain_hydradb import HydraDBClient, HydraDBChatMessageHistory, HydraDBRetriever

BASE = "https://api.hydradb.com"


@responses.activate
def test_unified_both_retrieves_once_and_maps_distinct_ids():
    responses.add(responses.POST, BASE + "/query", json={"success": True, "data": {
        "chunks": [{"chunk_id": "chunk", "context_id": "context", "content": "body", "score": 0.0,
                    "enrichment": "facts", "received_at": "2026-11-19T00:00:00Z"}],
        "graph": [], "forceful_relations": [], "llm_prompt": "server prompt",
    }})
    retriever = HydraDBRetriever(client=HydraDBClient("k", "db", layout="unified"), kind="both")
    docs = retriever.invoke("question")
    assert len(responses.calls) == 1
    assert len(docs) == 1
    assert docs[0].page_content == "body"
    assert docs[0].metadata["id"] == "chunk"
    assert docs[0].metadata["context_id"] == "context"
    assert docs[0].metadata["score"] == 0.0
    assert docs[0].metadata["enrichment"] == "facts"
    assert docs[0].metadata["received_at"] == "2026-11-19T00:00:00Z"
    assert docs[0].metadata.get("source_title", "") == ""
    assert docs[0].metadata.get("source") not in ("memory", "knowledge")


@responses.activate
def test_unified_history_preserves_message_list_and_persists_conversation():
    responses.add(responses.POST, BASE + "/query", json={"success": True, "data": {
        "chunks": [{"chunk_id": "chunk", "context_id": "context", "content": "prior turn", "score": 0.7}]
    }})
    responses.add(responses.POST, BASE + "/context/ingest", json={"success": True, "data": {"success_count": 1}}, status=202)
    history = HydraDBChatMessageHistory(HydraDBClient("k", "db", layout="unified"), "session", infer=False)
    assert history.messages == [HumanMessage(content="prior turn")]
    history.add_message(HumanMessage(content="hello"))
    history.add_message(AIMessage(content="answer"))
    item = json.loads(responses.calls[1].request.body)["context"][0]
    assert item["conversation"] == [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "answer"}]
    assert item["enrich"] is False

