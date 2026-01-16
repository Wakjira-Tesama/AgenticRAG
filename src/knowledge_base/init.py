"""
Knowledge base module for document processing and vector storage
"""
from .vector_store import VectorStoreManager
from .document_processor import DocumentProcessor

__all__ = [
    "VectorStoreManager",
    "DocumentProcessor"
]