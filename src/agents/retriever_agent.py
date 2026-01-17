import logging
import os
from typing import Dict, List, Any
from langchain.schema import Document

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None
    embedding_functions = None

logger = logging.getLogger(__name__)

if not chromadb:
    logger.warning("ChromaDB not installed. Retriever will run in fallback mode.")

class RetrieverAgent:
    """Agent responsible for retrieving relevant documents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.top_k = config.get('top_k', 5)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        
        # Initialize Vector DB
        try:
            # Persistent client
            persist_dir = os.environ.get('CHROMA_PERSIST_DIR', './data/chroma_db')
            self.client = chromadb.PersistentClient(path=persist_dir)
            
            # Embedding function
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get('OPENAI_API_KEY', 'dummy-key'),
                model_name=os.environ.get('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
            )
            
            # Get or create collection
            collection_name = os.environ.get('CHROMA_COLLECTION_NAME', 'research_papers')
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=openai_ef
            )
            
            logger.info(f"RetrieverAgent initialized with collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.collection = None
            # Fallback for testing/demo if DB fails
            logger.warning("Retriever running in fallback mode (no DB connection)")

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query
        """
        logger.info(f"Retrieving documents for: {query[:50]}...")
        
        if not self.collection:
            return self._get_fallback_docs(query)
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=self.top_k
            )
            
            documents = []
            if results['documents']:
                for i, doc_text in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    
                    # Convert to LangChain Document
                    documents.append(Document(
                        page_content=doc_text,
                        metadata=metadata
                    ))
            
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return self._get_fallback_docs(query)

    def _get_fallback_docs(self, query: str) -> List[Document]:
        """Return dummy documents for testing/fallback"""
        logger.info("Using fallback documents")
        return [
            Document(
                page_content="The Transformer is a deep learning architecture introduced in 2017. It relies on self-attention mechanisms.",
                metadata={"source": "Attention Is All You Need", "year": "2017"}
            ),
            Document(
                page_content="Artificial Intelligence (AI) refers to the simulation of human intelligence by machines.",
                metadata={"source": "AI Basics", "year": "2023"}
            ),
            Document(
                page_content="Machine Learning involves training algorithms to learn patterns from data.",
                metadata={"source": "ML Intro", "year": "2022"}
            )
        ]
