# FINANCIAL RETRIEVAL-AUGMENTED GENERATION (RAG)
# ***

# Goals: Financial Document Retrieval and Analysis
# - Load financial vector database
# - Create intelligent financial assistant
# - Provide accurate financial insights

# LIBRARIES
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import yaml
import os
from pprint import pprint

# Key Parameters
RAG_DATABASE = "data/financial_chroma.db"
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM = "gpt-4o-mini"

# OPENAI API KEY SETUP


def load_openai_key():
    """Load OpenAI API key with error handling"""
    try:
        credential_paths = [
            'credentials.yml',
            '../credentials.yml',
            './credentials.yml'
        ]

        for path in credential_paths:
            if os.path.exists(path):
                with open(path, 'r') as file:
                    credentials = yaml.safe_load(file)
                    return credentials['openai']

        if 'OPENAI_API_KEY' in os.environ:
            return os.environ['OPENAI_API_KEY']

        raise FileNotFoundError(
            "No credentials file found and OPENAI_API_KEY not set")

    except Exception as e:
        print(f"Error loading OpenAI API key: {e}")
        return None


# Load API key
api_key = load_openai_key()
if api_key:
    os.environ['OPENAI_API_KEY'] = api_key

# 1.0 CREATE FINANCIAL RETRIEVER


def initialize_financial_retriever():
    """Initialize the financial document retriever"""
    try:
        embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)

        vectorstore = Chroma(
            persist_directory=RAG_DATABASE,
            embedding_function=embedding_function
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Return top 4 most relevant chunks
        )

        print("Financial retriever initialized successfully")
        return retriever, vectorstore

    except Exception as e:
        print(f"Error initializing retriever: {e}")
        return None, None

# 2.0 FINANCIAL RAG CHAIN


def create_financial_rag_chain(retriever):
    """Create specialized financial RAG chain"""

    # Financial-specific prompt template
    financial_template = """You are a professional financial analyst assistant. Based on the provided financial documents, answer the user's question accurately and comprehensively.

Context from financial documents:
{context}

Instructions:
- Provide accurate financial analysis based only on the provided context
- If specific numbers are mentioned, include them in your response
- Identify the source document when possible
- If the information is not in the context, clearly state that
- Use professional financial terminology appropriately
- Structure your response clearly with key points
-If you are unsure about the answer, state that you cannot provide a definitive answer based on the available context. and 
do not make assumptions. just say "I cannot provide a definitive answer based on the available context." message Mubarak or Rachel : mubastudio@gmail.com

Question: {question}

Financial Analysis:"""

    prompt = ChatPromptTemplate.from_template(financial_template)

    # LLM with appropriate settings for financial analysis
    model = ChatOpenAI(
        model=LLM,
        temperature=0.1,  # Low temperature for factual financial responses
    )

    # Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain

# 3.0 FINANCIAL QUERY FUNCTIONS


def get_financial_response(question, rag_chain):
    """Get response for financial question"""
    try:
        result = rag_chain.invoke(question)
        return result
    except Exception as e:
        return f"Error processing question: {e}"


def get_baseline_response(question):
    """Get baseline response without RAG for comparison"""
    try:
        model = ChatOpenAI(model=LLM, temperature=0.1)
        result = model.invoke(question)
        return result.content
    except Exception as e:
        return f"Error getting baseline response: {e}"

# 4.0 TESTING AND DEMONSTRATION


def test_financial_rag():
    """Test the financial RAG system with sample questions"""

    print("FINANCIAL RAG SYSTEM TEST")
    print("=" * 50)

    # Initialize components
    retriever, vectorstore = initialize_financial_retriever()

    if not retriever:
        print("Failed to initialize retriever. Please run pdf_processor.py first.")
        return None, None

    rag_chain = create_financial_rag_chain(retriever)

    # Sample financial questions
    test_questions = [
        "What is the Tesla company's total revenue for the most recent period?",
        "What are the Microsoft's  main assets listed in the balance sheet?",
        "What is the net income or profit margin?",
        "What are the major expenses or costs?",
        "What is the cash flow from operations?",
        "What are the company's liabilities?",
        "What is the debt-to-equity ratio or financial leverage?",
        "What are the key financial risks mentioned?"
    ]

    print("\nTesting Financial RAG Responses:")
    print("=" * 50)

    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. QUESTION: {question}")
        print("-" * 40)

        # Get RAG response
        rag_response = get_financial_response(question, rag_chain)
        print("RAG RESPONSE:")
        print(rag_response)
        print("\n" + "="*50)

    return rag_chain, vectorstore

# 5.0 ADVANCED FINANCIAL QUERIES


def analyze_financial_health(rag_chain):
    """Perform comprehensive financial health analysis"""

    health_questions = [
        "What is the overall financial health of the company based on the available data?",
        "What are the key financial strengths and weaknesses?",
        "What trends can be identified in the financial performance?",
        "What are the main financial risks and opportunities?"
    ]

    print("\nFINANCIAL HEALTH ANALYSIS")
    print("=" * 50)

    for question in health_questions:
        print(f"\nAnalysis: {question}")
        print("-" * 40)
        response = get_financial_response(question, rag_chain)
        print(response)
        print("\n" + "="*30)


def search_specific_metrics(vectorstore, metrics):
    """Search for specific financial metrics"""

    print("\nSPECIFIC METRICS SEARCH")
    print("=" * 50)

    for metric in metrics:
        print(f"\nSearching for: {metric}")
        print("-" * 30)

        results = vectorstore.similarity_search(metric, k=2)

        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(
                f"Document: {os.path.basename(result.metadata.get('source', 'Unknown'))}")
            print(f"Type: {result.metadata.get('document_type', 'Unknown')}")
            print(f"Content: {result.page_content[:300]}...")
            print()

# 6.0 MAIN EXECUTION


def main():
    """Main function to run financial RAG system"""

    # Test the system
    rag_chain, vectorstore = test_financial_rag()

    if rag_chain and vectorstore:
        # Perform financial health analysis
        analyze_financial_health(rag_chain)

        # Search for specific metrics
        financial_metrics = [
            "revenue growth",
            "profit margin",
            "return on equity",
            "debt ratio",
            "current ratio"
        ]

        search_specific_metrics(vectorstore, financial_metrics)

        print("\n" + "="*50)
        print("FINANCIAL RAG SYSTEM READY")
        print("="*50)
        print("✓ Financial document retrieval working")
        print("✓ RAG chain operational")
        print("✓ Ready for Streamlit integration")

        return rag_chain, vectorstore

    else:
        print("Failed to initialize financial RAG system")
        return None, None


if __name__ == "__main__":
    main()
