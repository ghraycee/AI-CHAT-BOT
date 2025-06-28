# Financial RAG Chatbot Setup Instructions

## 📋 Overview
This system creates a comprehensive Retrieval-Augmented Generation (RAG) chatbot for financial document analysis using PDF processing, vector databases, and Streamlit interface.

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Directory Structure
Create the following directory structure:
```
financial-rag-system/
├── data/
│   ├── pdfs/                 # Place your PDF files here
│   └── financial_chroma.db/  # Vector database (auto-created)
├── credentials.yml           # API keys
├── pdf_processor.py          # PDF processing script
├── financial_retrieval.py    # RAG retrieval system
├── financial_chatbot.py      # Streamlit app
└── requirements.txt          # Dependencies
```

### 3. Credentials Setup
Create `credentials.yml`:
```yaml
openai: "your-openai-api-key-here"
```

### 4. Add Financial Documents
- Place your PDF financial documents in the `data/pdfs/` directory
- Supported formats: Annual reports, financial statements, 10-K, 10-Q, etc.

### 5. Process Documents
```bash
python pdf_processor.py
```

### 6. Test RAG System
```bash
python financial_retrieval.py
```

### 7. Launch Streamlit App
```bash
streamlit run financial_chatbot.py
```

## 📦 Requirements.txt
```txt
langchain-community==0.0.13
langchain-core==0.1.12
langchain-openai==0.0.5
langchain-text-splitters==0.0.1
streamlit==1.29.0
plotly==5.17.0
pandas==2.1.4
numpy==1.24.3
PyPDF2==3.0.1
pypdf==3.17.4
chromadb==0.4.22
openai==1.6.1
PyYAML==6.0.1
tqdm==4.66.1
pathlib==1.0.1
pprint==0.1
```

## 🔧 Configuration Options

### PDF Processing Settings
- **Chunk Size**: 1000 characters (optimized for financial data)
- **Chunk Overlap**: 200 characters (ensures context continuity)
- **Embedding Model**: text-embedding-ada-002 (OpenAI)

### RAG System Settings
- **LLM Model**: gpt-4o-mini (cost-effective, high-quality)
- **Temperature**: 0.1 (low for factual responses)
- **Retrieval Count**: 4 chunks per query

### Streamlit Settings
- **Layout**: Wide mode for better visualization
- **Caching**: Enabled for RAG system initialization
- **Charts**: Plotly for interactive visualizations

## 📊 Features

### Document Processing
- ✅ Batch PDF processing
- ✅ Automatic metadata extraction
- ✅ Financial document classification
- ✅ Smart text chunking
- ✅ Vector embeddings creation

### RAG System
- ✅ Semantic search
- ✅ Context-aware responses
- ✅ Source attribution
- ✅ Financial terminology optimization
- ✅ Multi-document querying

### Streamlit Interface
- ✅ Real-time chat interface
- ✅ Financial data visualization
- ✅ Quick action buttons
- ✅ Follow-up question suggestions
- ✅ Chart generation
- ✅ Persistent chat history

## 🎯 Usage Examples

### Sample Questions
- "What is the company's total revenue for the latest period?"
- "What are the main sources of income and expenses?"
- "How has the financial performance changed over time?"
- "What are the key financial risks mentioned?"
- "What is the debt-to-equity ratio?"
- "Show me the cash flow analysis"

### Quick Actions
- Financial health overview
- Revenue breakdown analysis
- Expense category analysis
- Profit margin calculations
- Risk assessment summary

## 🛠️ Troubleshooting

### Common Issues

#### 1. "No PDF files found"
- Ensure PDF files are in `data/pdfs/` directory
- Check file permissions
- Verify files are not corrupted

#### 2. "OpenAI API key not found"
- Check `credentials.yml` format
- Verify API key is active
- Ensure sufficient API credits

#### 3. "RAG system not available"
- Run `pdf_processor.py` first
- Check if vector database was created
- Verify all dependencies are installed

#### 4. "Embedding creation failed"
- Check internet connection
- Verify OpenAI API limits
- Try reducing batch size

### Performance Tips
- Use smaller PDF files for faster processing
- Reduce chunk_size for memory constraints
- Enable caching for repeated queries
- Use GPU acceleration if available

## 🔐 Security Considerations
- Store API keys securely
- Don't commit credentials to version control
- Use environment variables in production
- Implement proper access controls

## 📈 Advanced Configuration

### Custom Document Types
Modify the document classification in `pdf_processor.py`:
```python
# Add custom document types
if 'custom_term' in content_lower:
    doc.metadata['document_type'] = 'Custom Document Type'
```

### Enhanced Visualizations
Add custom charts in `financial_chatbot.py`:
```python
def create_custom_chart(data):
    # Your custom visualization logic
    return fig
```

### Database Optimization
For large document collections:
- Increase chunk overlap
- Use hierarchical chunking
- Implement document filtering
- Add metadata indexing

## 🚀 Deployment

### Local Development
- Use the provided setup instructions
- Enable debug mode for development
- Test with sample documents

### Production Deployment
- Use Streamlit Cloud or similar service
- Implement proper logging
- Add error monitoring
- Use production-grade vector database

## 📞 Support
- Check the troubleshooting section first
- Review error messages carefully
- Ensure all requirements are met
- Test with simple queries first

## 🔄 Updates and Maintenance
- Regularly update dependencies
- Monitor API usage and costs
- Backup vector databases
- Update document classifications as needed
