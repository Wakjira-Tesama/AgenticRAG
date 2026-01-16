"""
Document processing for knowledge base ingestion
"""
import os
import logging
from typing import List, Dict, Any
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader
)

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes documents for ingestion into knowledge base"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize text splitter
        chunking_config = config.get('chunking', {})
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking_config.get('size', 1000),
            chunk_overlap=chunking_config.get('overlap', 200),
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Supported file formats
        self.supported_formats = config.get('supported_formats', ['.pdf', '.txt', '.md', '.docx'])
        
        logger.info("DocumentProcessor initialized")
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """Load documents from various file formats"""
        all_documents = []
        
        for file_path in file_paths:
            try:
                docs = self._load_single_document(file_path)
                all_documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return all_documents
    
    def _load_single_document(self, file_path: str) -> List[Document]:
        """Load a single document based on file extension"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_ext == '.md':
            loader = UnstructuredMarkdownLoader(file_path)
        elif file_ext in ['.docx', '.doc']:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata['source'] = Path(file_path).name
            doc.metadata['file_path'] = file_path
            doc.metadata['file_type'] = file_ext
        
        return documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents by splitting into chunks"""
        if not documents:
            return []
        
        logger.info(f"Processing {len(documents)} documents...")
        
        # Split documents
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['total_chunks'] = len(chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """Process all documents in a directory"""
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        # Find all supported files
        file_paths = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_ext = Path(file).suffix.lower()
                if file_ext in self.supported_formats:
                    file_paths.append(os.path.join(root, file))
        
        if not file_paths:
            logger.warning(f"No supported files found in {directory_path}")
            return []
        
        logger.info(f"Found {len(file_paths)} files in {directory_path}")
        
        # Load and process all documents
        raw_documents = self.load_documents(file_paths)
        processed_documents = self.process_documents(raw_documents)
        
        return processed_documents
    
    def extract_metadata(self, document: Document) -> Dict[str, Any]:
        """Extract metadata from a document"""
        metadata = {
            'source': document.metadata.get('source', 'Unknown'),
            'page': document.metadata.get('page', 0),
            'file_type': document.metadata.get('file_type', 'Unknown'),
            'chunk_size': len(document.page_content)
        }
        
        # Try to extract title from content
        content = document.page_content
        lines = content.split('\n')
        
        if lines:
            # First non-empty line might be a title
            for line in lines:
                if line.strip():
                    if len(line.strip()) < 100 and line.strip().isprintable():
                        metadata['possible_title'] = line.strip()[:50]
                    break
        
        return metadata