import streamlit as st
import os
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# --- UI Configuration ---
st.set_page_config(page_title="Azure RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 RAG Chatbot with Azure AI Search")

# --- Azure Credentials & Services Setup (Sidebar) ---
with st.sidebar:
    st.header("Azure Configuration ⚙️")

    # Input fields for Azure credentials
    azure_search_endpoint = st.text_input(
        "Azure AI Search Endpoint",
        type="password",
        placeholder="https://your-search-service.search.windows.net"
    )
    azure_search_key = st.text_input(
        "Azure AI Search Key",
        type="password"
    )
    azure_search_index = st.text_input(
        "Azure AI Search Index Name",
        placeholder="your-index-name"
    )
    
    st.markdown("---")

    azure_openai_endpoint = st.text_input(
        "Azure OpenAI Endpoint",
        type="password",
        placeholder="https://your-openai-service.openai.azure.com/"
    )
    azure_openai_key = st.text_input(
        "Azure OpenAI Key",
        type="password"
    )
    azure_openai_deployment = st.text_input(
        "Azure OpenAI Deployment Name",
        placeholder="your-gpt-deployment-name"
    )

    # Check if all credentials are provided
    all_credentials_provided = (
        azure_search_endpoint and azure_search_key and azure_search_index and
        azure_openai_endpoint and azure_openai_key and azure_openai_deployment
    )

    if not all_credentials_provided:
        st.warning("Please enter all Azure credentials to start the chatbot.")
        st.stop()

# --- Backend RAG Logic ---
def get_rag_response(question, search_client, openai_client, openai_deployment):
    """
    Retrieves context from Azure AI Search and generates a response using Azure OpenAI.
    """
    try:
        # 1. RETRIEVAL: Search for relevant documents
        # Assuming your search index has a "content" field with the text.
        # Adjust 'select' and the field name in the context loop as needed.
        search_results = search_client.search(
            search_text=question,
            top=3,  # Get the top 3 most relevant documents
            select="content"
        )

        # 2. AUGMENTATION: Build the context string from search results
        retrieved_context = ""
        for result in search_results:
            retrieved_context += f"\n\n---\n\n{result['content']}"

        if not retrieved_context:
            return "I couldn't find any relevant information in my knowledge base to answer your question."

        # 3. GENERATION: Create a prompt and get a response from the LLM
        system_prompt = """
        You are a helpful AI assistant. Answer the user's question based ONLY on the context provided below.
        If the answer is not found in the context, respond with "I couldn't find an answer in the provided documents."
        Do not make up information. Be concise and professional.
        """
        
        # Combine system prompt, context, and the user's question
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{retrieved_context}\n\nQuestion:\n{question}"}
        ]

        response = openai_client.chat.completions.create(
            model=openai_deployment,
            messages=messages,
            temperature=0.2, # Lower temperature for more factual answers
            max_tokens=500
        )
        
        return response.choices[0].message.content

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "Sorry, I ran into an issue while processing your request."

# --- Initialize Clients ---
try:
    search_credential = AzureKeyCredential(azure_search_key)
    search_client = SearchClient(endpoint=azure_search_endpoint, index_name=azure_search_index, credential=search_credential)
    
    openai_client = AzureOpenAI(
        api_key=azure_openai_key,
        api_version="2024-02-01", # Use a recent, stable API version
        azure_endpoint=azure_openai_endpoint
    )
except Exception as e:
    st.error(f"Failed to initialize Azure clients. Please check your credentials. Error: {e}")
    st.stop()


# --- Chat Interface ---
# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today based on my knowledge base?"}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get RAG response and display it
    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            response = get_rag_response(
                question=prompt,
                search_client=search_client,
                openai_client=openai_client,
                openai_deployment=azure_openai_deployment
            )
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
