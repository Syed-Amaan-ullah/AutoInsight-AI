import streamlit as st
from rag_pipeline import EnhancedRAGPipeline
from pdf_processor import PDFProcessor, validate_pdf_file
from evaluation import get_evaluation_report
from memory import load_chats, save_chats, create_new_chat
from datasets import load_dataset
from langchain_core.documents import Document
from datetime import datetime
import os

# Check for optional dependencies
try:
    from multi_agent import CREWAI_AVAILABLE
    MULTI_AGENT_AVAILABLE = CREWAI_AVAILABLE
except ImportError:
    MULTI_AGENT_AVAILABLE = False

try:
    from evaluation import RAGAS_AVAILABLE
    EVALUATION_AVAILABLE = RAGAS_AVAILABLE
except ImportError:
    EVALUATION_AVAILABLE = False

st.set_page_config(
    page_title="AutoInsight AI - Enhanced RAG System",
    page_icon="",
    layout="wide"
)

st.title(" AutoInsight AI (Enhanced RAG System)")
st.markdown("*Powered by Gemini AI + PDF Support + Optional Multi-Agent & Evaluation*")

# Display feature availability
col_status1, col_status2, col_status3 = st.columns(3)
with col_status1:
    st.success("✅ PDF Upload" if True else "❌ PDF Upload")
with col_status2:
    st.success("✅ Multi-Agent" if MULTI_AGENT_AVAILABLE else "⚠️ Multi-Agent (CrewAI not available)")
with col_status3:
    st.success("✅ RAGAS Evaluation" if EVALUATION_AVAILABLE else "⚠️ Basic Evaluation (RAGAS not available)")

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = load_chats()
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# Initialize session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Sidebar for configuration
st.sidebar.title("⚙️ Configuration")

# Document source selection
doc_source = st.sidebar.radio(
    "Document Source:",
    ["ML-ArXiv-Papers Dataset", "Upload PDF"],
    help="Choose between using the ML-ArXiv-Papers dataset or uploading your own PDF"
)

# Query mode selection - only show available options
query_modes = ["Basic RAG"]
if MULTI_AGENT_AVAILABLE:
    query_modes.append("Multi-Agent Enhanced")
if EVALUATION_AVAILABLE:
    query_modes.append("With Evaluation")

query_mode = st.sidebar.radio(
    "Query Mode:",
    query_modes,
    help="Select the RAG processing mode"
)

# Chat History
st.sidebar.markdown("### 💬 Chat History")

if st.sidebar.button("➕ New Chat"):
    new_chat = create_new_chat()
    st.session_state.chats.append(new_chat)
    st.session_state.current_chat_id = new_chat['id']
    save_chats(st.session_state.chats)
    st.rerun()

chat_options = {chat['id']: f"{chat['name']} ({chat['date'][:10]})" for chat in st.session_state.chats}
if chat_options:
    selected_chat_id = st.sidebar.selectbox(
        "Select Chat",
        options=list(chat_options.keys()),
        format_func=lambda x: chat_options[x],
        key="chat_select"
    )
    if selected_chat_id != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected_chat_id
        st.rerun()

    if st.sidebar.button("Delete Chat") and st.session_state.current_chat_id:
        st.session_state.chats = [c for c in st.session_state.chats if c['id'] != st.session_state.current_chat_id]
        save_chats(st.session_state.chats)
        st.session_state.current_chat_id = None
        st.rerun()

# Initialize pipeline
if st.sidebar.button("Initialize/Load Documents"):
    with st.spinner("Loading documents..."):
        try:
            pipeline = EnhancedRAGPipeline()

            if doc_source == "Upload PDF":
                st.sidebar.info("Please upload a PDF file below")
            else:
                # Load ML-ArXiv-Papers dataset
                with st.spinner("Loading ML-ArXiv-Papers dataset..."):
                    ds = load_dataset("CShorten/ML-ArXiv-Papers")
                    documents = []
                    for item in ds['train']:  # Assuming 'train' split
                        content = f"Title: {item['title']}\n\nAbstract: {item['abstract']}\n\nAuthors: {', '.join(item.get('authors', []))}"
                        documents.append(Document(page_content=content, metadata={"source": "ML-ArXiv-Papers", "title": item['title']}))
                    pipeline.create_vectorstore(documents)
                    st.session_state.pipeline = pipeline
                    st.session_state.documents_loaded = True
                    st.sidebar.success(f"✅ Loaded {len(documents)} ML papers successfully!")

        except Exception as e:
            st.sidebar.error(f"❌ Error loading documents: {str(e)}")

# PDF Upload section
if doc_source == "Upload PDF":
    st.sidebar.markdown("### 📄 PDF Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF document",
        type=['pdf'],
        help="Upload a PDF file to analyze"
    )

    if uploaded_file and st.sidebar.button("Process PDF"):
        if validate_pdf_file(uploaded_file):
            with st.spinner("Processing PDF..."):
                try:
                    pipeline = EnhancedRAGPipeline()
                    documents = pipeline.load_documents(uploaded_file, "pdf")
                    pipeline.create_vectorstore(documents)

                    # Get metadata
                    metadata = pipeline.get_document_metadata()

                    st.session_state.pipeline = pipeline
                    st.session_state.documents_loaded = True

                    st.sidebar.success("✅ PDF processed successfully!")
                    st.sidebar.info(f"📊 {metadata.get('vectorstore_size', 0)} chunks processed")

                except Exception as e:
                    st.sidebar.error(f"❌ Error processing PDF: {str(e)}")
        else:
            st.sidebar.error("❌ Invalid PDF file. Please check file size (<50MB) and format.")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 Chat Interface")

    # Display current chat
    chat_container = st.container()
    with chat_container:
        current_chat = next((c for c in st.session_state.chats if c['id'] == st.session_state.current_chat_id), None)
        if current_chat:
            for msg in current_chat['messages']:
                with st.chat_message("user"):
                    st.write(msg['user'])
                with st.chat_message("assistant"):
                    st.write(msg['bot'])
        else:
            st.write("Select or create a new chat to start.")

    # Query input
    query = st.text_input("Ask a question about the documents:", key="query_input")

    if st.button(" Submit Query", type="primary") and query:
        if not st.session_state.documents_loaded or not st.session_state.pipeline:
            st.error("❌ Please initialize/load documents first!")
        else:
            with st.spinner(f"Processing with {query_mode}..."):
                try:
                    pipeline = st.session_state.pipeline

                    if query_mode == "Basic RAG":
                        result = pipeline.query_basic(query)
                        response = result['answer']
                    elif query_mode == "Multi-Agent Enhanced":
                        if not MULTI_AGENT_AVAILABLE:
                            st.error("❌ Multi-agent features not available")
                            st.stop()
                        result = pipeline.query_multi_agent(query)
                        response = result['answer']
                    elif query_mode == "With Evaluation":
                        result = pipeline.query_with_evaluation(query)
                        response = result['answer']
                        # Add evaluation info to response
                        eval_scores = result['evaluation']['scores']
                        response += f"\n\n📊 **Evaluation Scores:**\n"
                        for metric, score in eval_scores.items():
                            response += f"- {metric}: {score:.3f}\n"

                    # Add to current chat
                    if not st.session_state.current_chat_id:
                        chat_name = query[:50] + "..." if len(query) > 50 else query
                        new_chat = create_new_chat(name=chat_name)
                        st.session_state.chats.append(new_chat)
                        st.session_state.current_chat_id = new_chat['id']

                    current_chat = next((c for c in st.session_state.chats if c['id'] == st.session_state.current_chat_id), None)
                    current_chat['messages'].append({"user": query, "bot": response, "timestamp": datetime.now().isoformat()})
                    save_chats(st.session_state.chats)

                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")

with col2:
    st.markdown("### 📊 System Status")

    if st.session_state.documents_loaded and st.session_state.pipeline:
        metadata = st.session_state.pipeline.get_document_metadata()
        st.success("✅ Documents Loaded")
        st.info(f"📄 Chunks: {metadata.get('vectorstore_size', 'N/A')}")
        st.info(f"🤖 LLM: Gemini 2.5 Flash")
        st.info(f"🔍 Retriever: FAISS")
    else:
        st.warning("⚠️ No documents loaded")

    st.markdown("### 🎯 Query Mode")
    if query_mode == "Basic RAG":
        st.info("🔹 Standard retrieval-augmented generation")
    elif query_mode == "Multi-Agent Enhanced":
        st.info("🔹 Multi-agent analysis with CrewAI" if MULTI_AGENT_AVAILABLE else "❌ Multi-agent not available")
    elif query_mode == "With Evaluation":
        st.info("🔹 RAG with evaluation metrics" if EVALUATION_AVAILABLE else "🔹 RAG with basic evaluation")

    # Evaluation Report
    if st.button("📈 View Evaluation Report"):
        report = get_evaluation_report()
        st.markdown("### 📋 Evaluation Report")
        st.code(report, language="markdown")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit, LangChain, Gemini AI, and optional CrewAI & RAGAS*")

# Installation hints for missing features
if not MULTI_AGENT_AVAILABLE or not EVALUATION_AVAILABLE:
    st.markdown("### 🔧 Optional Features")
    if not MULTI_AGENT_AVAILABLE:
        st.info("💡 To enable multi-agent features: `pip install crewai`")
    if not EVALUATION_AVAILABLE:
        st.info("💡 To enable advanced evaluation: `pip install ragas datasets`")