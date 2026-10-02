"""LangChain chat message history backed by HydraDB memory.

Persists conversation turns into HydraDB so they survive across sessions, and
recalls the most relevant past turns for the current thread. Use with
``RunnableWithMessageHistory`` or the classic memory wrappers.
"""

from __future__ import annotations

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .client import HydraDBClient


class HydraDBChatMessageHistory(BaseChatMessageHistory):
    """Chat history that writes turns to HydraDB and recalls relevant ones.

    Because HydraDB stores *distilled* memory rather than a raw transcript,
    ``messages`` returns the most relevant recalled turns for ``session_id``,
    not a verbatim log. Pass ``infer=False`` if you want literal storage.
    """

    def __init__(
        self,
        client: HydraDBClient,
        session_id: str,
        max_results: int = 10,
        infer: bool = True,
    ) -> None:
        self.client = client
        self.session_id = session_id
        self.max_results = max_results
        self.infer = infer
        self._pending_human: str | None = None

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        chunks = self.client.query(
            self.session_id,
            kind="knowledge",
            max_results=self.max_results,
        )
        return [HumanMessage(content=c.text) for c in chunks if c.text]

    def add_message(self, message: BaseMessage) -> None:
        if isinstance(message, HumanMessage):
            self._pending_human = _content_str(message)
            return
        if isinstance(message, AIMessage):
            assistant = _content_str(message)
            user = self._pending_human or ""
            self._pending_human = None
            self.client.add_conversation(user, assistant, infer=self.infer)
            return
        self.client.add_text(_content_str(message), infer=self.infer)

    def clear(self) -> None:
        """No-op: HydraDB has no per-session bulk delete; remove by id via the client."""
        return None


def _content_str(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)
