from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from multi_agent import MultiAgentRAGSystem, create_multi_agent_crew
from evaluation import evaluate_rag_response
import os
from typing import List, Optional, Union, Dict, Any

load_dotenv()


class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with PDF support, multi-agent capabilities, and evaluation"""

    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.pdf_processor = PDFProcessor()
        self.multi_agent_system = MultiAgentRAGSystem(llm=self.llm)
        self.vectorstore = None
        self.retriever = None

    def load_documents(self, source: Union[str, List[Document]], source_type: str = "file") -> List[Document]:
        """
        Load documents from various sources

        Args:
            source: File path, uploaded file, or list of documents
            source_type: "file", "pdf", or "documents"

        Returns:
            List of Document objects
        """
        if source_type == "file":
            # Load from text file
            loader = TextLoader(source)
            documents = loader.load()
        elif source_type == "pdf":
            # Load from PDF upload
            documents = self.pdf_processor.load_pdf(source)
        elif source_type == "documents":
            # Use provided documents
            documents = source
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        return documents

    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """Create FAISS vectorstore from documents"""
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        return self.vectorstore

    def create_rag_chain(self):
        """Create the basic RAG chain"""
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {question}""")

        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain

    def query_basic(self, question: str) -> Dict[str, Any]:
        """Basic RAG query"""
        if not self.retriever:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore() first.")

        rag_chain = self.create_rag_chain()

        # Get context for evaluation
        context_docs = self.retriever.invoke(question)
        contexts = [doc.page_content for doc in context_docs]

        answer = rag_chain.invoke(question)

        return {
            'answer': answer,
            'contexts': contexts,
            'question': question,
            'method': 'basic_rag'
        }

    def query_multi_agent(self, question: str) -> Dict[str, Any]:
        """Multi-agent enhanced query"""
        if not self.retriever:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore() first.")

        # Get context
        context_docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in context_docs)
        contexts = [doc.page_content for doc in context_docs]

        # Use multi-agent system
        result = create_multi_agent_crew(question, context)

        return {
            'answer': result,
            'contexts': contexts,
            'question': question,
            'method': 'multi_agent'
        }

    def query_with_evaluation(self, question: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """Query with automatic evaluation"""
        if not self.retriever:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore() first.")

        # Get basic answer
        basic_result = self.query_basic(question)

        # Evaluate the response
        evaluation = evaluate_rag_response(
            question=basic_result['question'],
            answer=basic_result['answer'],
            contexts=basic_result['contexts'],
            ground_truth=ground_truth
        )

        return {
            **basic_result,
            'evaluation': evaluation,
            'method': 'evaluated_rag'
        }

    def get_document_metadata(self) -> Dict[str, Any]:
        """Get metadata about loaded documents"""
        if not self.vectorstore:
            return {'message': 'No documents loaded'}

        # This is a simplified metadata - in a real implementation,
        # you'd want to store document metadata during loading
        return {
            'vectorstore_size': len(self.vectorstore.docstore._dict),
            'embedding_model': 'gemini-embedding-001',
            'llm_model': 'gemini-2.5-flash'
        }


# Convenience functions for backward compatibility
def create_rag_pipeline(source_path: Optional[str] = None):
    """Create RAG pipeline - maintains backward compatibility"""
    pipeline = EnhancedRAGPipeline()

    if source_path:
        documents = pipeline.load_documents(source_path, "file")
    else:
        # Default to sample.txt
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, "data", "sample.txt")
        documents = pipeline.load_documents(data_path, "file")

    pipeline.create_vectorstore(documents)
    return pipeline