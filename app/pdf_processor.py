"""
PDF Document Processor for AutoInsight AI
Handles PDF upload, text extraction, and document processing
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tempfile
import os
from typing import List


class PDFProcessor:
    """Handles PDF document processing and text extraction"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_pdf(self, uploaded_file) -> List[Document]:
        """
        Load and process a PDF file from Streamlit upload

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            List of Document objects with extracted text
        """
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # Load PDF using PyPDFLoader
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)

            return split_docs

        finally:
            # Clean up temporary file
            os.unlink(tmp_path)

    def extract_metadata(self, documents: List[Document]) -> dict:
        """
        Extract metadata from processed documents

        Args:
            documents: List of Document objects

        Returns:
            Dictionary with document metadata
        """
        total_pages = len(set(doc.metadata.get('page', 0) for doc in documents))
        total_chunks = len(documents)
        total_characters = sum(len(doc.page_content) for doc in documents)

        return {
            'total_pages': total_pages,
            'total_chunks': total_chunks,
            'total_characters': total_characters,
            'source_file': documents[0].metadata.get('source', 'Unknown') if documents else 'Unknown'
        }


def validate_pdf_file(uploaded_file) -> bool:
    """
    Validate if the uploaded file is a valid PDF

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        True if valid PDF, False otherwise
    """
    if uploaded_file is None:
        return False

    # Check file extension
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False

    # Check file size (max 50MB)
    if uploaded_file.size > 50 * 1024 * 1024:
        return False

    return True