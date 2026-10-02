"""LangChain integration for HydraDB - persistent memory and retrieval."""

from .client import HydraChunk, HydraDBClient, HydraDBError
from .memory import HydraDBChatMessageHistory
from .retriever import HydraDBRetriever

__all__ = [
    "HydraDBClient",
    "HydraChunk",
    "HydraDBError",
    "HydraDBRetriever",
    "HydraDBChatMessageHistory",
]

__version__ = "0.1.0"
