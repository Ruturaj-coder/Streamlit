import streamlit as st
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery

# --- 1. SETUP AND CONFIGURATION ---

# Load environment variables from .env file
load_dotenv()

# UI Configuration
st.set_page_config(page_title="Azure RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Advanced RAG Chatbot")

# --- 2. AZURE SERVICES INITIALIZATION ---

try:
    # Load credentials from .env file
    azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    azure_search_key = os.getenv("AZURE_SEARCH_KEY")
    azure_search_index = os.getenv("AZURE_SEARCH_INDEX")
    
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
    azure_openai_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

    # Check for missing credentials
    if not all([azure_search_endpoint, azure_search_key, azure_search_index,
                azure_openai_endpoint, azure_openai_key, azure_openai_deployment]):
        st.error("🔴 Critical Error: One or more environment variables are missing.")
        st.info("Please check your .env file and ensure all required credentials are set.")
        st.stop()

    # Initialize clients
    search_credential = AzureKeyCredential(azure_search_key)
    search_client = SearchClient(endpoint=azure_search_endpoint, index_name=azure_search_index, credential=search_credential)
    openai_client = AzureOpenAI(api_key=azure_openai_key, api_version="2024-02-01", azure_endpoint=azure_openai_endpoint)

except Exception as e:
    st.error(f"🔴 Failed to initialize Azure clients: {e}")
    st.stop()

# --- 3. HELPER FUNCTIONS ---

def retrieve_documents(question):
    """Performs hybrid search and returns the retrieved context and sources."""
    try:
        vector_query = VectorizableTextQuery(text=question, k_nearest_neighbors=5, fields="vector")
        
        results = search_client.search(
            search_text=question,
            vector_queries=[vector_query],
            select=["title", "chunk", "document_title", "author", "topic"],
            top=5
        )

        retrieved_context = ""
        sources = []
        for result in results:
            retrieved_context += result['chunk'] + "\n\n"
            sources.append({
                "title": result.get('title', 'N/A'),
                "document_title": result.get('document_title', 'N/A'),
                "author": result.get('author', 'N/A'),
                "relevance_score": result.get('@search.score', 0.0)
            })
        return retrieved_context, sources
    except Exception as e:
        st.error(f"An error occurred during document retrieval: {e}")
        return None, None

def stream_llm_response(question, context):
    """Streams the LLM response based on the provided question and context."""
    system_prompt = """
    You are a helpful AI assistant. Answer the user's question based ONLY on the context provided below.
    Format your answers clearly using markdown, including bullet points, bolding, and line breaks where appropriate.
    If the answer is not in the context, say 'I don't have enough information in the provided documents to answer that.'
    Do not make up information.
    """
    augmented_prompt = f"CONTEXT FROM DOCUMENTS:\n{context}\n\nQUESTION:\n{question}"

    try:
        stream = openai_client.chat.completions.create(
            model=azure_openai_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_prompt}
            ],
            temperature=0.2,
            max_tokens=1500,
            stream=True
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"An error occurred while generating the response: {e}"

# --- 4. SESSION STATE INITIALIZATION ---

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
if "latest_sources" not in st.session_state:
    st.session_state.latest_sources = []

# --- 5. UI RENDERING ---

# Main layout: Two columns for chat and sources
col1, col2 = st.columns([2, 1])

# Column 1: Chat Interface
with col1:
    st.header("Chat")

    # Display chat messages from history
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # Start generating the assistant's response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching for documents..."):
                context, sources = retrieve_documents(prompt)
                st.session_state.latest_sources = sources

            if context is None:
                st.error("Could not retrieve documents. Please check the connection to Azure AI Search.")
            else:
                # Use a placeholder for the streaming response
                response_placeholder = st.empty()
                full_response = ""
                # Stream the LLM response
                for chunk in stream_llm_response(prompt, context):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌") # Add a cursor effect
                response_placeholder.markdown(full_response) # Final response

        # Add the final assistant response to the message history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()

# Column 2: Sources and Controls
with col2:
    st.header("Sources & Controls")

    # Clear conversation button
    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
        st.session_state.latest_sources = []
        st.rerun()

    st.markdown("---")

    # Display sources for the latest response
    if st.session_state.latest_sources:
        st.subheader("Retrieved Documents")
        for i, source in enumerate(st.session_state.latest_sources):
            with st.container(border=True):
                st.markdown(f"**Source {i+1}:** {source.get('document_title', 'N/A')}")
                st.markdown(f"**Title:** {source.get('title', 'N/A')}")
                st.progress(source.get('relevance_score', 0.0), text=f"**Relevance:** {source.get('relevance_score', 0.0):.2f}")
    else:
        st.info("Sources for the latest answer will appear here.")
