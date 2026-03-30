# AutoInsight AI - Enhanced RAG System

An advanced Retrieval-Augmented Generation (RAG) system powered by Google's Gemini AI, featuring PDF document processing, optional multi-agent capabilities, and evaluation metrics.

## 🚀 Features

### Core RAG System
- **Gemini AI Integration**: Powered by Google's Gemini 2.0 Flash for high-quality responses
- **FAISS Vector Store**: Efficient similarity search and retrieval
- **Contextual Q&A**: Answers based on document content only

### Advanced Features
- **📄 PDF Upload Support**: Process and analyze PDF documents in real-time
- **🤖 Multi-Agent System**: Optional CrewAI-powered multi-agent analysis (requires CrewAI installation)
- **📊 RAG Evaluation**: Optional comprehensive evaluation metrics using RAGAS
- **💾 Chat Memory**: Persistent conversation history
- **🎨 Modern UI**: Clean Streamlit interface with sidebar configuration

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AutoInsight-AI
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install core dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional: Install advanced features**:
   ```bash
   # For multi-agent capabilities
   pip install crewai

   # For advanced evaluation metrics
   pip install ragas datasets
   ```

5. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```
   GOOGLE_API_KEY=your_google_ai_api_key_here
   ```

6. **Run the application**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## 📖 Usage

### Document Sources

1. **Default Sample**: Uses the included `sample.txt` with EV company information
2. **PDF Upload**: Upload your own PDF documents for analysis

### Query Modes

1. **Basic RAG**: Standard retrieval-augmented generation (always available)
2. **Multi-Agent Enhanced**: Uses CrewAI agents for document analysis (requires CrewAI)
3. **With Evaluation**: Includes evaluation metrics with each response (requires RAGAS)

### Interface Features

- **Sidebar Configuration**: Select document source and query mode
- **Chat Interface**: Interactive Q&A with persistent history
- **System Status**: Real-time information about loaded documents
- **Evaluation Reports**: View comprehensive performance metrics

## 🏗️ Architecture

```
app/
├── streamlit_app.py      # Main Streamlit application
├── rag_pipeline.py       # Enhanced RAG pipeline with multi-modal support
├── pdf_processor.py      # PDF document processing utilities
├── multi_agent.py        # Optional CrewAI multi-agent system
├── evaluation.py         # Optional RAGAS evaluation system
├── memory.py            # Chat history persistence
└── data/
    └── sample.txt       # Default document with EV company data
```

## 🔧 Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Your Google AI API key (required)

### PDF Processing
- **Max file size**: 50MB
- **Supported formats**: PDF only
- **Text extraction**: Automatic with chunking

### Vector Store
- **Embedding model**: Gemini Embedding 001
- **Similarity search**: FAISS with k=3 retrieval
- **Chunk size**: 1000 characters with 200 overlap

## 📊 Evaluation Metrics

The system provides comprehensive evaluation using RAGAS:

- **Faithfulness**: How well the answer is grounded in the context
- **Answer Relevancy**: How relevant the answer is to the question
- **Context Relevancy**: How relevant the retrieved context is
- **Context Recall**: Coverage of relevant information
- **Answer Correctness**: Accuracy compared to ground truth
- **Answer Similarity**: Semantic similarity to reference answers

## 🤖 Multi-Agent System

Powered by CrewAI with specialized agents:

- **Document Analyzer**: Extracts key insights and themes
- **Question Answerer**: Provides contextual responses
- **Evaluator**: Assesses response quality and provides feedback

## 📈 Performance Tips

1. **Document Quality**: Better documents = better responses
2. **Query Specificity**: Clear, specific questions get better answers
3. **PDF Processing**: Ensure PDFs have selectable text (not scanned images)
4. **Evaluation Mode**: Use evaluation mode to monitor and improve performance

## 🔒 Security & Privacy

- API keys stored securely in environment variables
- No data persistence except chat history
- Documents processed in-memory only
- Temporary files cleaned up automatically

## 🐛 Troubleshooting

### Common Issues

1. **"No documents loaded"**: Click "Initialize/Load Documents" first
2. **PDF processing fails**: Check file size (<50MB) and ensure it's a valid PDF
3. **API errors**: Verify GOOGLE_API_KEY is set correctly
4. **Import errors**: Run `pip install -r requirements.txt`

### Python 3.14 Compatibility

Some advanced features (CrewAI, RAGAS) may have limited compatibility with Python 3.14. The core RAG functionality with PDF support works with all supported Python versions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google AI**: For the Gemini models
- **LangChain**: For the RAG framework
- **CrewAI**: For multi-agent capabilities (optional)
- **RAGAS**: For evaluation metrics (optional)
- **Streamlit**: For the web interface