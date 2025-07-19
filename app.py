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

# UI Configuration with enhanced styling
st.set_page_config(
    page_title="Azure RAG Assistant", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header with better styling
st.markdown("""
<div style='text-align: center; padding: 1rem 0; margin-bottom: 2rem;'>
    <h1 style='color: #1f77b4; margin-bottom: 0.5rem;'>🚀 Azure RAG Assistant</h1>
    <p style='color: #666; font-size: 1.1rem; margin: 0;'>Intelligent Document Search & Chat</p>
</div>
""", unsafe_allow_html=True)

# --- 2. AZURE SERVICES INITIALIZATION ---

@st.cache_resource
def initialize_azure_clients():
    """Initialize Azure clients with caching for better performance"""
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
            return None, None, "Missing environment variables"

        # Initialize clients
        search_credential = AzureKeyCredential(azure_search_key)
        search_client = SearchClient(
            endpoint=azure_search_endpoint, 
            index_name=azure_search_index, 
            credential=search_credential
        )
        openai_client = AzureOpenAI(
            api_key=azure_openai_key, 
            api_version="2024-02-01", 
            azure_endpoint=azure_openai_endpoint
        )
        
        return search_client, openai_client, None

    except Exception as e:
        return None, None, str(e)

# Initialize clients
search_client, openai_client, error = initialize_azure_clients()

if error:
    st.error(f"🔴 **Configuration Error:** {error}")
    st.info("💡 **Solution:** Please check your .env file and ensure all required credentials are properly set.")
    st.stop()

# Success indicator
st.success("✅ **Azure Services Connected Successfully**")

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
        st.error(f"❌ **Search Error:** {e}")
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
            model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
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
        yield f"❌ **Error generating response:** {e}"

# --- 4. SESSION STATE INITIALIZATION ---

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 **Hello!** I'm your Azure RAG Assistant. I can help you search through documents and answer questions based on the information I find. What would you like to know?"}
    ]
if "latest_sources" not in st.session_state:
    st.session_state.latest_sources = []

# --- 5. SIDEBAR CONFIGURATION ---

with st.sidebar:
    st.markdown("### 🎛️ **Chat Controls**")
    
    # Clear conversation button with better styling
    if st.button("🗑️ **Clear Chat History**", use_container_width=True, type="secondary"):
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 **Hello!** I'm your Azure RAG Assistant. I can help you search through documents and answer questions based on the information I find. What would you like to know?"}
        ]
        st.session_state.latest_sources = []
        st.rerun()
    
    st.markdown("---")
    
    # Chat statistics
    st.markdown("### 📊 **Chat Statistics**")
    user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
    assistant_messages = len([msg for msg in st.session_state.messages if msg["role"] == "assistant"])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Your Messages", user_messages)
    with col2:
        st.metric("AI Responses", assistant_messages)
    
    st.markdown("---")
    
    # Help section
    st.markdown("### ℹ️ **How to Use**")
    st.markdown("""
    1. **Ask Questions** - Type your question in the chat input
    2. **View Sources** - See which documents were used for answers
    3. **Clear History** - Start fresh anytime
    4. **Explore Topics** - Ask about different subjects in your knowledge base
    """)

# --- 6. MAIN CHAT INTERFACE ---

# Create main container for better spacing
with st.container():
    # Display chat messages with enhanced styling
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"**You:** {message['content']}")
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])

    # Chat input with enhanced prompt
    if prompt := st.chat_input("💬 Ask me anything about your documents..."):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"**You:** {prompt}")

        # Generate assistant response
        with st.chat_message("assistant", avatar="🤖"):
            # Create columns for loading indicator and content
            status_container = st.container()
            response_container = st.container()
            
            with status_container:
                with st.status("🔍 **Processing your request...**", expanded=False) as status:
                    st.write("🔎 Searching through documents...")
                    context, sources = retrieve_documents(prompt)
                    st.session_state.latest_sources = sources
                    
                    if context is None:
                        st.write("❌ Document search failed")
                        status.update(label="❌ **Search Failed**", state="error")
                    else:
                        st.write(f"✅ Found {len(sources)} relevant documents")
                        st.write("🤖 Generating response...")
                        status.update(label="✅ **Response Ready**", state="complete")

            with response_container:
                if context is None:
                    st.error("**Unable to retrieve documents.** Please check your Azure AI Search connection.")
                    full_response = "I apologize, but I couldn't access the document search service right now. Please try again later."
                else:
                    # Stream the response
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    for chunk in stream_llm_response(prompt, context):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()

# --- 7. SOURCES DISPLAY ---

if st.session_state.latest_sources:
    st.markdown("---")
    st.markdown("### 📚 **Source Documents**")
    st.markdown("*These documents were used to generate the response above*")
    
    # Create columns for better source layout
    cols = st.columns(min(len(st.session_state.latest_sources), 3))
    
    for i, source in enumerate(st.session_state.latest_sources):
        col_idx = i % len(cols)
        with cols[col_idx]:
            with st.container(border=True):
                st.markdown(f"**📄 {source.get('document_title', 'Unknown Document')}**")
                st.markdown(f"*{source.get('title', 'N/A')}*")
                
                # Enhanced relevance score display
                score = source.get('relevance_score', 0.0)
                if score >= 0.8:
                    st.success(f"🎯 **Relevance:** {score:.2f}")
                elif score >= 0.6:
                    st.info(f"✅ **Relevance:** {score:.2f}")
                else:
                    st.warning(f"⚠️ **Relevance:** {score:.2f}")
                
                if source.get('author', 'N/A') != 'N/A':
                    st.markdown(f"**Author:** {source.get('author')}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>🚀 Powered by Azure OpenAI & Azure AI Search</small>
</div>
""", unsafe_allow_html=True)
