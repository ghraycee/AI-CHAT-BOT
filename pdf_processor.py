# FINANCIAL PDF RAG SYSTEM
# ***

# Goal: Process PDF financial documents and create vector database for RAG

# LIBRARIES
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

import pandas as pd
import yaml
import os
from pathlib import Path
from pprint import pprint
import glob
from tqdm import tqdm

# OPENAI API SETUP


def load_openai_key():
    """Load OpenAI API key with error handling"""
    try:
        # Try multiple possible paths for credentials
        credential_paths = [
            'credentials.yml',  # Current directory first
            '../credentials.yml',
            './credentials.yml'
        ]

        for path in credential_paths:
            if os.path.exists(path):
                with open(path, 'r') as file:
                    credentials = yaml.safe_load(file)
                    return credentials['openai']

        # If no file found, check environment variable
        if 'OPENAI_API_KEY' in os.environ:
            return os.environ['OPENAI_API_KEY']

        raise FileNotFoundError(
            "No credentials file found and OPENAI_API_KEY not set")

    except Exception as e:
        print(f"Error loading OpenAI API key: {e}")
        print("Please ensure you have either:")
        print("1. A credentials.yml file with 'openai' key")
        print("2. OPENAI_API_KEY environment variable set")
        return None


# Load API key
api_key = load_openai_key()
if api_key:
    os.environ['OPENAI_API_KEY'] = api_key
else:
    print("WARNING: OpenAI API key not loaded. Please set it manually.")

# 1.0 PDF DATA LOADING


def load_pdf_documents(pdf_directory="data/pdfs"):
    """Load all PDF documents from directory"""
    try:
        # Ensure directory exists
        os.makedirs(pdf_directory, exist_ok=True)

        # Find all PDF files
        pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {pdf_directory}")
            print("Please add PDF files to the directory and try again.")
            return None

        print(f"Found {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files:
            print(f"  - {os.path.basename(pdf_file)}")

        # Load documents using DirectoryLoader for batch processing
        loader = DirectoryLoader(
            pdf_directory,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )

        documents = loader.load()
        print(f"\nLoaded {len(documents)} document pages total")

        return documents

    except Exception as e:
        print(f"Error loading PDF documents: {e}")
        return None


def process_financial_documents(documents):
    """Process and enhance financial documents with metadata"""
    if not documents:
        return None

    print("\nProcessing financial documents...")

    # Enhanced text splitter for financial documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunks for financial data
        chunk_overlap=200,  # Good overlap for context
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    # Split documents
    docs = text_splitter.split_documents(documents)
    print(f"Created {len(docs)} text chunks")

    # Enhance documents with financial context
    for i, doc in enumerate(docs):
        # Extract source file name
        source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
        page_num = doc.metadata.get('page', 0)

        # Add financial document context
        enhanced_content = f"""
Document: {source_file}
Page: {page_num + 1}
Content Type: Financial Document

{doc.page_content}
"""
        doc.page_content = enhanced_content.strip()

        # Add document type classification
        content_lower = doc.page_content.lower()
        if any(term in content_lower for term in ['balance sheet', 'assets', 'liabilities']):
            doc.metadata['document_type'] = 'Balance Sheet'
        elif any(term in content_lower for term in ['income statement', 'revenue', 'profit', 'loss']):
            doc.metadata['document_type'] = 'Income Statement'
        elif any(term in content_lower for term in ['cash flow', 'operating activities', 'investing activities']):
            doc.metadata['document_type'] = 'Cash Flow Statement'
        elif any(term in content_lower for term in ['annual report', '10-k', '10-q']):
            doc.metadata['document_type'] = 'Annual Report'
        else:
            doc.metadata['document_type'] = 'Financial Document'

    print("Documents enhanced with financial metadata")
    return docs


def create_vector_database(docs, db_path="data/financial_chroma.db"):
    """Create vector database from processed documents"""
    if not docs:
        print("No documents to process")
        return None

    try:
        # Initialize embeddings
        embedding_function = OpenAIEmbeddings(
            model='text-embedding-ada-002',
        )
        print("Embedding function initialized")

        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)

        # Check if database already exists
        if os.path.exists(db_path):
            print(f"\nLoading existing vector database from {db_path}")
            vectorstore = Chroma(
                embedding_function=embedding_function,
                persist_directory=db_path
            )

            # Add new documents if any
            if docs:
                print("Adding new documents to existing database...")
                vectorstore.add_documents(docs)
        else:
            print(f"\nCreating new vector database at {db_path}")
            vectorstore = Chroma.from_documents(
                docs,
                embedding=embedding_function,
                persist_directory=db_path
            )

        print("Vector database created/updated successfully")
        return vectorstore

    except Exception as e:
        print(f"Error creating vector database: {e}")
        return None


def test_similarity_search(vectorstore):
    """Test the vector database with sample financial queries"""
    if not vectorstore:
        print("No vector store available for testing")
        return

    test_queries = [
        "What is the company's revenue?",
        "Show me the balance sheet information",
        "What are the cash flow details?",
        "Tell me about the company's assets and liabilities",
        "What is the net income or profit?"
    ]

    print("\n" + "="*50)
    print("TESTING FINANCIAL DOCUMENT SEARCH")
    print("="*50)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)

        try:
            results = vectorstore.similarity_search(query, k=2)

            if results:
                for i, result in enumerate(results, 1):
                    print(f"Result {i}:")
                    print(
                        f"Source: {result.metadata.get('source', 'Unknown')}")
                    print(
                        f"Type: {result.metadata.get('document_type', 'Unknown')}")
                    print(f"Content: {result.page_content[:200]}...")
                    print()
            else:
                print("No results found")

        except Exception as e:
            print(f"Error in search: {e}")


def main():
    """Main execution function"""
    print("FINANCIAL PDF RAG SYSTEM")
    print("=" * 40)

    # Load PDF documents
    documents = load_pdf_documents()

    if documents:
        # Process documents
        processed_docs = process_financial_documents(documents)

        if processed_docs:
            # Create vector database
            vectorstore = create_vector_database(processed_docs)

            if vectorstore:
                # Test the system
                test_similarity_search(vectorstore)

                print("\n" + "="*50)
                print("SETUP COMPLETE!")
                print("="*50)
                print(f"✓ Processed {len(processed_docs)} document chunks")
                print("✓ Vector database created successfully")
                print("✓ Ready for RAG queries")
                print("\nNext steps:")
                print("1. Run the Streamlit app: streamlit run financial_chatbot.py")
                print("2. Start asking questions about your financial documents")

            else:
                print("Failed to create vector database")
        else:
            print("Failed to process documents")
    else:
        print("No documents loaded. Please add PDF files to data/pdfs/ directory")


if __name__ == "__main__":
    main()
