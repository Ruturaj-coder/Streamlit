import streamlit as st
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery

# --- Load Environment Variables ---
# Load environment variables from .env file
load_dotenv() 

# --- UI Configuration ---
st.set_page_config(page_title="Azure RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Hybrid Search RAG Chatbot")

# --- Azure Credentials & Services Setup (from .env) ---
try:
    azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    azure_search_key = os.getenv("AZURE_SEARCH_KEY")
    azure_search_index = os.getenv("AZURE_SEARCH_INDEX")
    
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
    azure_openai_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

    # Check if all credentials are loaded
    if not all([azure_search_endpoint, azure_search_key, azure_search_index,
                azure_openai_endpoint, azure_openai_key, azure_openai_deployment]):
        st.error("🔴 Critical Error: One or more environment variables are missing.")
        st.error("Please check your .env file and make sure it contains all required credentials.")
        st.stop()

    # --- Initialize Clients ---
    search_credential = AzureKeyCredential(azure_search_key)
    search_client = SearchClient(endpoint=azure_search_endpoint, index_name=azure_search_index, credential=search_credential)
    
    openai_client = AzureOpenAI(
        api_key=azure_openai_key,
        api_version="2024-02-01", 
        azure_endpoint=azure_openai_endpoint
    )

except Exception as e:
    st.error(f"🔴 Failed to initialize Azure clients. Please check your credentials in the .env file. Error: {e}")
    st.stop()


# --- Backend RAG Logic (Updated with Hybrid Search) ---
def get_rag_response(question, search_client, openai_client, openai_deployment):
    """
    Performs hybrid search and generates a response using Azure OpenAI.
    Returns the response text and a list of sources.
    """
    try:
        # 1. RETRIEVAL: Perform hybrid search
        vector_query = VectorizableTextQuery(
            text=question, 
            k_nearest_neighbors=5, 
            fields="vector" # IMPORTANT: Assumes your vector field is named "vector"
        )

        search_args = {
            "search_text": question,
            "vector_queries": [vector_query],
            "select": [
                "title", "chunk", "parent_id", "chunk_id", "document_title", 
                "author", "topic" # Add other fields as needed
            ],
            "top": 5
        }
        
        results = search_client.search(**search_args)

        # 2. AUGMENTATION: Prepare context and track sources
        retrieved_context = ""
        sources = []
        for result in results:
            retrieved_context += result['chunk'] + "\n\n"
            sources.append({
                "title": result.get('title', 'N/A'),
                "document_title": result.get('document_title', 'N/A'),
                "chunk_id": result.get('chunk_id', 'N/A'),
                "author": result.get('author', 'N/A'),
                "topic": result.get('topic', 'N/A')
            })

        if not retrieved_context:
            return "I couldn't find any relevant information in the documents. Please try adjusting your query.", []

        # 3. GENERATION: Create a prompt and get a response from the LLM
        system_prompt = """
        You are a helpful AI assistant. Answer the user's question based ONLY on the context provided below.
        If the answer is not in the context, say 'I don't have enough information in the provided documents to answer that.'
        Do not make up information. Provide clear, concise answers.
        """
        
        augmented_prompt = f"CONTEXT FROM DOCUMENTS:\n{retrieved_context}\n\nQUESTION:\n{question}"

        response = openai_client.chat.completions.create(
            model=openai_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        return response.choices[0].message.content, sources

    except Exception as e:
        st.error(f"An error occurred during search: {e}")
        return "Sorry, I ran into an issue while processing your request.", []

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources if they exist in the message
        if "sources" in message and message["sources"]:
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"]):
                    st.write(f"**Source {i+1}:**")
                    st.write(f"- **Title:** {source.get('title', 'N/A')}")
                    st.write(f"- **Document:** {source.get('document_title', 'N/A')}")
                    st.write(f"- **Author:** {source.get('author', 'N/A')}")
                    st.write(f"- **Topic:** {source.get('topic', 'N/A')}")


if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            response_text, sources = get_rag_response(
                question=prompt,
                search_client=search_client,
                openai_client=openai_client,
                openai_deployment=azure_openai_deployment
            )
            
            st.markdown(response_text)

            if sources:
                with st.expander("Sources"):
                    for i, source in enumerate(sources):
                        st.write(f"**Source {i+1}:**")
                        st.write(f"- **Title:** {source.get('title', 'N/A')}")
                        st.write(f"- **Document:** {source.get('document_title', 'N/A')}")
                        st.write(f"- **Author:** {source.get('author', 'N/A')}")
                        st.write(f"- **Topic:** {source.get('topic', 'N/A')}")
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text, 
        "sources": sources
    })
