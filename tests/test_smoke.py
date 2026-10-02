"""Offline tests: verify wire format and mapping without hitting the network."""

import json

import responses

from langchain_hydradb import HydraDBClient, HydraDBRetriever

BASE = "https://api.hydradb.com"


def _client():
    return HydraDBClient(api_key="test-key", tenant_id="test_beam", sub_tenant_id="")


@responses.activate
def test_query_builds_body_and_normalises_chunks():
    responses.add(
        responses.POST,
        f"{BASE}/query",
        json={
            "success": True,
            "data": {
                "chunks": [
                    {
                        "chunk_uuid": "c1",
                        "chunk_content": "we deploy on Fridays",
                        "source_title": "CLAUDE.md",
                        "score": 0.91,
                    }
                ]
            },
            "error": None,
        },
        status=200,
    )
    chunks = _client().query("deploy?", kind="memory", max_results=3)
    assert len(chunks) == 1
    assert chunks[0].text == "we deploy on Fridays"
    assert chunks[0].id == "c1"
    assert chunks[0].source_title == "CLAUDE.md"

    sent = json.loads(responses.calls[0].request.body)
    assert sent["tenant_id"] == "test_beam"
    assert sent["kind"] == "memory"
    assert sent["max_results"] == 3


@responses.activate
def test_query_unwraps_content_envelope():
    envelope = json.dumps(
        {"id": "s1", "title": "int-check", "content": {"text": "the prose", "html_base64": ""}}
    )
    responses.add(
        responses.POST,
        f"{BASE}/query",
        json={
            "success": True,
            "data": {"chunks": [{"chunk_uuid": "s1_c0", "chunk_content": envelope, "relevancy_score": 0.7}]},
            "error": None,
        },
        status=200,
    )
    chunks = _client().query("q", kind="knowledge")
    assert chunks[0].text == "the prose"
    assert "html_base64" not in chunks[0].text
    assert chunks[0].score == 0.7


@responses.activate
def test_query_passes_through_non_json_content():
    responses.add(
        responses.POST,
        f"{BASE}/query",
        json={
            "success": True,
            "data": {"chunks": [{"chunk_uuid": "x", "chunk_content": '{"content": not json'}]},
            "error": None,
        },
        status=200,
    )
    assert _client().query("q")[0].text == '{"content": not json'


@responses.activate
def test_retriever_returns_documents_for_both_kinds():
    for _ in range(2):
        responses.add(
            responses.POST,
            f"{BASE}/query",
            json={
                "success": True,
                "data": {"chunks": [{"chunk_content": "x", "chunk_uuid": "i"}]},
                "error": None,
            },
            status=200,
        )
    retriever = HydraDBRetriever(client=_client(), kind="both", max_results=2)
    docs = retriever.invoke("q")
    assert len(docs) == 2
    assert {d.metadata["source"] for d in docs} == {"memory", "knowledge"}


@responses.activate
def test_add_text_posts_app_knowledge_to_context_ingest():
    responses.add(
        responses.POST,
        f"{BASE}/context/ingest",
        json={"success": True, "data": {"success_count": 1}, "error": None},
        status=200,
    )
    out = _client().add_text("remember pnpm", infer=False, title="pref")
    assert out == {"success_count": 1}
    body = responses.calls[0].request.body
    assert b"app_knowledge" in body
    assert b"memories" not in body
    assert b"test_beam" in body


@responses.activate
def test_delete_uses_delete_context_with_body():
    responses.add(
        responses.DELETE,
        f"{BASE}/context",
        json={"success": True, "data": {"deleted_count": 1}, "error": None},
        status=200,
    )
    out = _client().delete(["src-1"], kind="knowledge")
    assert out == {"deleted_count": 1}
    sent = json.loads(responses.calls[0].request.body)
    assert sent["ids"] == ["src-1"]
    assert sent["tenant_id"] == "test_beam"
