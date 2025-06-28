# FINANCIAL CHATBOT STREAMLIT APPLICATION
# Run with: streamlit run financial_chatbot.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import os

# RAG Components
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Configuration
RAG_DATABASE = "data/financial_chroma.db"
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM = "gpt-4o-mini"

# Initialize Streamlit App
st.set_page_config(
    page_title="Financial AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .financial-metric {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "plots" not in st.session_state:
    st.session_state.plots = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Helper Functions


@st.cache_resource
def load_api_key():
    """Load OpenAI API key"""
    import yaml
    try:
        credential_paths = ['credentials.yml',
                            '../credentials.yml', './credentials.yml']
        for path in credential_paths:
            if os.path.exists(path):
                with open(path, 'r') as file:
                    credentials = yaml.safe_load(file)
                    return credentials['openai']

        if 'OPENAI_API_KEY' in os.environ:
            return os.environ['OPENAI_API_KEY']
        return None
    except:
        return None


@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with caching"""
    api_key = load_api_key()
    if not api_key:
        return None, None

    os.environ['OPENAI_API_KEY'] = api_key

    try:
        embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma(
            persist_directory=RAG_DATABASE,
            embedding_function=embedding_function
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # Financial-specific prompt
        template = """You are a professional financial analyst assistant. Based on the provided financial documents, answer the user's question accurately and comprehensively.

Context from financial documents:
{context}

Instructions:
- Provide accurate financial analysis based only on the provided context
- Include specific numbers and data points when available
- Identify the source document when possible
- If information is not in the context, clearly state that
- Use professional financial terminology
- Structure responses with clear key points
- Suggest relevant follow-up questions when appropriate

Question: {question}

Financial Analysis:"""

        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model=LLM, temperature=0.1)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        return rag_chain, vectorstore

    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None, None


def extract_financial_data(text):
    """Extract financial data from text for visualization"""
    data = {}

    # Extract monetary values
    money_pattern = r'\$[\d,]+\.?\d*[BMK]?'
    amounts = re.findall(money_pattern, text)

    # Extract percentages
    percent_pattern = r'\d+\.?\d*%'
    percentages = re.findall(percent_pattern, text)

    # Extract years
    year_pattern = r'\b(19|20)\d{2}\b'
    years = re.findall(year_pattern, text)

    data['amounts'] = amounts[:5]  # Limit to first 5
    data['percentages'] = percentages[:5]
    data['years'] = years[:5]

    return data


def create_financial_chart(data_type, response_text):
    """Create financial charts based on response content"""

    if data_type == "revenue_trend":
        # Sample revenue trend chart
        years = [2020, 2021, 2022, 2023, 2024]
        revenue = [100, 120, 135, 150, 165]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years,
            y=revenue,
            mode='lines+markers',
            name='Revenue (Millions)',
            line=dict(color='#1f77b4', width=3)
        ))
        fig.update_layout(
            title="Revenue Trend Analysis",
            xaxis_title="Year",
            yaxis_title="Revenue ($ Millions)",
            template="plotly_white"
        )
        return fig

    elif data_type == "expense_breakdown":
        # Sample expense breakdown
        categories = ['Operating Expenses',
                      'Cost of Goods Sold', 'R&D', 'Marketing', 'Other']
        values = [40, 35, 15, 7, 3]

        fig = go.Figure(data=[go.Pie(
            labels=categories,
            values=values,
            hole=0.3
        )])
        fig.update_layout(title="Expense Breakdown")
        return fig

    elif data_type == "financial_ratios":
        # Sample financial ratios
        ratios = ['Current Ratio', 'Debt-to-Equity', 'ROE', 'Profit Margin']
        values = [2.1, 0.8, 15.5, 12.3]

        fig = go.Figure(data=[go.Bar(
            x=ratios,
            y=values,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )])
        fig.update_layout(
            title="Key Financial Ratios",
            yaxis_title="Ratio Value"
        )
        return fig

    return None


def suggest_follow_up_questions(response_text):
    """Suggest relevant follow-up questions based on the response"""
    suggestions = []

    if "revenue" in response_text.lower():
        suggestions.extend([
            "What are the revenue growth trends?",
            "What are the main revenue streams?",
            "How does revenue compare to industry benchmarks?"
        ])

    if "expense" in response_text.lower() or "cost" in response_text.lower():
        suggestions.extend([
            "What are the largest expense categories?",
            "How have expenses changed over time?",
            "What cost-saving opportunities exist?"
        ])

    if "profit" in response_text.lower() or "income" in response_text.lower():
        suggestions.extend([
            "What is the profit margin trend?",
            "What factors affect profitability?",
            "How does profitability compare to competitors?"
        ])

    return suggestions[:3]  # Return top 3 suggestions


# Sidebar Configuration
st.sidebar.title("🏦 Financial Assistant")
st.sidebar.markdown("---")

# System Status
with st.sidebar:
    st.subheader("System Status")

    if st.session_state.rag_chain is None:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_chain, st.session_state.vectorstore = initialize_rag_system()

    if st.session_state.rag_chain:
        st.success("✅ RAG System Active")
        st.success("✅ Financial Documents Loaded")
    else:
        st.error("❌ RAG System Not Available")
        st.warning("Please ensure:")
        st.write("1. credentials.yml file exists")
        st.write("2. Financial documents are processed")
        st.write("3. Vector database is created")

# Quick Actions
st.sidebar.subheader("Quick Analysis")
quick_questions = [
    "What is the company's financial health?",
    "Show me the revenue breakdown",
    "What are the main expenses?",
    "What is the profit margin?",
    "What are the key financial risks?"
]

for question in quick_questions:
    if st.sidebar.button(question, key=f"quick_{question}"):
        st.session_state.messages.append({"role": "user", "content": question})

# Chart Options
st.sidebar.subheader("Financial Charts")
chart_options = {
    "Revenue Trend": "revenue_trend",
    "Expense Breakdown": "expense_breakdown",
    "Financial Ratios": "financial_ratios"
}

for chart_name, chart_type in chart_options.items():
    if st.sidebar.button(f"Show {chart_name}", key=f"chart_{chart_type}"):
        fig = create_financial_chart(chart_type, "")
        if fig:
            plot_index = len(st.session_state.plots)
            st.session_state.plots.append(fig)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Here's the {chart_name} analysis:",
                "plot_index": plot_index
            })

# Main App
st.markdown('<div class="main-header">💼 Financial AI Assistant</div>',
            unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        Ask questions about your financial documents and get intelligent insights
    </p>
</div>
""", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Display plot if it exists
        if "plot_index" in message:
            st.plotly_chart(
                st.session_state.plots[message["plot_index"]], use_container_width=True)

# Chat input
if prompt := st.chat_input("Ask about your financial documents..."):
    if not st.session_state.rag_chain:
        st.error("RAG system not available. Please check your setup.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing financial documents..."):
                try:
                    response = st.session_state.rag_chain.invoke(prompt)
                    st.write(response)

                    # Add assistant message
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response})

                    # Extract and visualize financial data if relevant
                    financial_data = extract_financial_data(response)

                    # Create visualization if financial data is found
                    if any(financial_data.values()):
                        st.markdown("### 📊 Financial Data Visualization")

                        col1, col2 = st.columns(2)

                        with col1:
                            if financial_data['amounts']:
                                st.markdown("**💰 Monetary Values Found:**")
                                for amount in financial_data['amounts']:
                                    st.markdown(f"- {amount}")

                        with col2:
                            if financial_data['percentages']:
                                st.markdown("**📈 Percentages Found:**")
                                for percent in financial_data['percentages']:
                                    st.markdown(f"- {percent}")

                    # Suggest follow-up questions
                    suggestions = suggest_follow_up_questions(response)
                    if suggestions:
                        st.markdown("### 🤔 Suggested Follow-up Questions:")
                        for i, suggestion in enumerate(suggestions):
                            if st.button(suggestion, key=f"suggestion_{len(st.session_state.messages)}_{i}"):
                                st.session_state.messages.append(
                                    {"role": "user", "content": suggestion})
                                st.experimental_rerun()

                except Exception as e:
                    error_msg = f"Error processing your question: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg})

# Footer with additional information
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📋 Features")
    st.markdown("""
    - PDF document processing
    - Intelligent financial analysis
    - Interactive visualizations
    - Follow-up suggestions
    """)

with col2:
    st.markdown("### 🔍 Capabilities")
    st.markdown("""
    - Revenue analysis
    - Expense breakdown
    - Financial ratios
    - Risk assessment
    """)

with col3:
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Be specific in your questions
    - Ask for comparisons
    - Request trend analysis
    - Inquire about specific metrics
    """)

# Debug section (only show in development)
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Debug Information")
    st.sidebar.write(f"Messages: {len(st.session_state.messages)}")
    st.sidebar.write(f"Plots: {len(st.session_state.plots)}")
    st.sidebar.write(f"RAG Active: {st.session_state.rag_chain is not None}")

    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.plots = []
        st.rerun()

# Sample questions section
st.markdown("---")
st.markdown("### 💭 Sample Questions to Get Started:")

sample_questions = [
    "What is the total revenue for the latest period?",
    "What are the main sources of income?",
    "What are the biggest expenses?",
    "How has the company's financial performance changed over time?",
    "What is the debt-to-equity ratio?",
    "What are the current assets and liabilities?",
    "What risks are mentioned in the financial documents?",
    "What is the profit margin and how does it compare to previous periods?"
]

cols = st.columns(2)
for i, question in enumerate(sample_questions):
    with cols[i % 2]:
        if st.button(question, key=f"sample_{i}"):
            st.session_state.messages.append(
                {"role": "user", "content": question})
            st.rerun()
