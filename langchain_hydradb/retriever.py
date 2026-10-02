"""LangChain retriever backed by HydraDB recall."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .client import HydraDBClient


class HydraDBRetriever(BaseRetriever):
    """Retrieve documents from HydraDB memory and/or knowledge.

    Example::

        retriever = HydraDBRetriever(
            client=HydraDBClient(api_key=..., tenant_id=...),
            kind="both",
            max_results=6,
        )
        docs = retriever.invoke("what did we decide about auth?")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: HydraDBClient
    kind: str = "knowledge"
    max_results: int = 6
    mode: str = "fast"
    graph_context: bool = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        kinds = ["memory", "knowledge"] if self.kind == "both" else [self.kind]
        docs: list[Document] = []
        for kind in kinds:
            chunks = self.client.query(
                query,
                kind=kind,
                mode=self.mode,
                max_results=self.max_results,
                graph_context=self.graph_context,
            )
            for chunk in chunks:
                metadata: dict[str, Any] = {
                    "source": kind,
                    "source_title": chunk.source_title,
                    "id": chunk.id,
                    **chunk.metadata,
                }
                if chunk.score is not None:
                    metadata["score"] = chunk.score
                docs.append(Document(page_content=chunk.text, metadata=metadata))
        return docs
