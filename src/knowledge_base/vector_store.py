"""
Vector store management for document storage and retrieval
"""
import os
import logging
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages the vector store for document storage"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=config.get('embedding', {}).get('model', 'text-embedding-3-small')
        )
        
        # Setup vector store
        persist_dir = config.get('persist_directory', './data/chroma_db')
        collection_name = config.get('collection_name', 'research_documents')
        
        os.makedirs(persist_dir, exist_ok=True)
        
        self.vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
        logger.info(f"VectorStoreManager initialized with {self.get_document_count()} documents")
    
    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to vector store"""
        try:
            ids = self.vector_store.add_documents(documents)
            logger.info(f"Added {len(ids)} documents to vector store")
            return len(ids)
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return 0
    
    def search(self, query: str, k: int = 5, filter_dict: Dict = None) -> List[Document]:
        """Search for similar documents"""
        try:
            if filter_dict:
                docs = self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            else:
                docs = self.vector_store.similarity_search(query, k=k)
            
            logger.info(f"Found {len(docs)} documents for query: {query[:50]}...")
            return docs
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """Search for documents with similarity scores"""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} documents with scores")
            return results
        except Exception as e:
            logger.error(f"Error searching with scores: {e}")
            return []
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs"""
        try:
            self.vector_store.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get total number of documents in collection"""
        try:
            return self.vector_store._collection.count()
        except:
            return 0
    
    def clear_collection(self) -> bool:
        """Clear entire collection"""
        try:
            # Chroma doesn't have direct clear, so we delete collection
            self.vector_store.delete_collection()
            
            # Recreate empty collection
            self.vector_store = Chroma(
                persist_directory=self.vector_store._persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.vector_store._collection.name
            )
            
            logger.info("Collection cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False