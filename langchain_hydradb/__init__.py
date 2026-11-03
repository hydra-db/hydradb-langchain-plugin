"""LangChain integration for HydraDB - persistent memory and retrieval."""

from .client import HydraChunk, HydraDBClient, HydraDBError, HydraQueryResult
from .memory import HydraDBChatMessageHistory
from .retriever import HydraDBRetriever

__all__ = [
    "HydraDBClient",
    "HydraChunk",
    "HydraQueryResult",
    "HydraDBError",
    "HydraDBRetriever",
    "HydraDBChatMessageHistory",
]

__version__ = "0.1.0"
