import streamlit as st
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
import time

# — 1. SETUP AND CONFIGURATION —

# Load environment variables from .env file

load_dotenv()

# Custom CSS for enhanced aesthetics

st.markdown(”””

<style>
    /* Main app styling */
    .main > div {
        padding-top: 1rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Chat container styling */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
        border: 1px solid #e1e8ed;
    }
    
    /* Sources container styling */
    .sources-container {
        background: linear-gradient(145deg, #f8fafc, #e2e8f0);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid #cbd5e0;
    }
    
    /* Source card styling */
    .source-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .source-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .status-online {
        background: #d4edda;
        color: #155724;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
</style>

“””, unsafe_allow_html=True)

# UI Configuration

st.set_page_config(
page_title=“Azure RAG Chatbot”,
page_icon=“🤖”,
layout=“wide”,
initial_sidebar_state=“collapsed”
)

# Custom header

st.markdown(”””

<div class="main-header">
    <h1>🤖 Azure RAG Chatbot</h1>
    <p>Powered by Azure OpenAI & Cognitive Search | Intelligent Document Retrieval</p>
</div>
""", unsafe_allow_html=True)

# — 2. AZURE SERVICES INITIALIZATION —

@st.cache_resource
def initialize_azure_clients():
“”“Initialize Azure clients with caching for better performance”””
try:
# Load credentials from .env file
azure_search_endpoint = os.getenv(“AZURE_SEARCH_ENDPOINT”)
azure_search_key = os.getenv(“AZURE_SEARCH_KEY”)
azure_search_index = os.getenv(“AZURE_SEARCH_INDEX”)

```
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
    azure_openai_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

    # Check for missing credentials
    missing_vars = []
    if not azure_search_endpoint: missing_vars.append("AZURE_SEARCH_ENDPOINT")
    if not azure_search_key: missing_vars.append("AZURE_SEARCH_KEY")
    if not azure_search_index: missing_vars.append("AZURE_SEARCH_INDEX")
    if not azure_openai_endpoint: missing_vars.append("AZURE_OPENAI_ENDPOINT")
    if not azure_openai_key: missing_vars.append("AZURE_OPENAI_KEY")
    if not azure_openai_deployment: missing_vars.append("AZURE_OPENAI_CHAT_DEPLOYMENT")
    
    if missing_vars:
        return None, None, missing_vars

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
```

# Initialize clients

search_client, openai_client, error = initialize_azure_clients()

if error:
if isinstance(error, list):
st.markdown(”””
<div class="status-indicator status-error">
🔴 Configuration Error
</div>
“””, unsafe_allow_html=True)
st.error(“Missing environment variables:”)
for var in error:
st.write(f”• {var}”)
st.info(“💡 Please check your .env file and ensure all required credentials are set.”)
else:
st.markdown(”””
<div class="status-indicator status-error">
🔴 Connection Error
</div>
“””, unsafe_allow_html=True)
st.error(f”Failed to initialize Azure clients: {error}”)
st.stop()

# Show connection status

st.markdown(”””

<div class="status-indicator status-online">
    🟢 Azure Services Connected
</div>
""", unsafe_allow_html=True)

# — 3. HELPER FUNCTIONS —

def retrieve_documents(question):
“”“Performs hybrid search and returns the retrieved context and sources.”””
try:
vector_query = VectorizableTextQuery(text=question, k_nearest_neighbors=5, fields=“vector”)

```
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
            "topic": result.get('topic', 'N/A'),
            "relevance_score": result.get('@search.score', 0.0)
        })
    return retrieved_context, sources
except Exception as e:
    st.error(f"Document retrieval error: {e}")
    return None, None
```

def stream_llm_response(question, context):
“”“Streams the LLM response based on the provided question and context.”””
system_prompt = “””
You are an intelligent AI assistant specialized in providing accurate, well-structured answers.

```
Guidelines:
• Answer based ONLY on the provided context from documents
• Format responses clearly using markdown (headers, bullet points, bold text)
• Use line breaks and spacing for readability
• If information isn't in the context, clearly state: "I don't have sufficient information in the provided documents to answer that question."
• Be concise yet comprehensive
• Highlight key points and important information
"""

augmented_prompt = f"""
CONTEXT FROM RETRIEVED DOCUMENTS:
{context}

USER QUESTION:
{question}

Please provide a well-structured, informative response based on the context above.
"""

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
```

# — 4. SESSION STATE INITIALIZATION —

if “messages” not in st.session_state:
st.session_state.messages = [{
“role”: “assistant”,
“content”: “👋 **Welcome!** I’m your AI assistant powered by Azure’s advanced search and language models. Ask me anything about your documents!”
}]
if “latest_sources” not in st.session_state:
st.session_state.latest_sources = []
if “message_count” not in st.session_state:
st.session_state.message_count = 0

# — 5. UI RENDERING —

# Main layout: Two columns for chat and sources

col1, col2 = st.columns([2.2, 1], gap=“large”)

# Column 1: Chat Interface

with col1:
st.markdown(’<div class="chat-container">’, unsafe_allow_html=True)

```
# Chat header with metrics
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("""
    <div class="metric-card">
        <h4 style="margin:0; color:#667eea;">💬 Messages</h4>
        <h2 style="margin:0;">{}</h2>
    </div>
    """.format(len(st.session_state.messages)), unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="metric-card">
        <h4 style="margin:0; color:#667eea;">📚 Sources</h4>
        <h2 style="margin:0;">{}</h2>
    </div>
    """.format(len(st.session_state.latest_sources)), unsafe_allow_html=True)

with col_c:
    st.markdown("""
    <div class="metric-card">
        <h4 style="margin:0; color:#667eea;">🤖 Status</h4>
        <h2 style="margin:0; color:#10b981;">Online</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Display chat messages from history
for i, message in enumerate(st.session_state.messages):
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("💭 Ask me anything about your documents..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.message_count += 1
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Start generating the assistant's response
    with st.chat_message("assistant", avatar="🤖"):
        # Search phase
        with st.status("🔍 **Searching documents...**", expanded=False) as status:
            st.write("Analyzing your question...")
            time.sleep(0.5)
            st.write("Performing hybrid search...")
            context, sources = retrieve_documents(prompt)
            st.session_state.latest_sources = sources if sources else []
            time.sleep(0.5)
            st.write(f"✅ Found {len(sources) if sources else 0} relevant documents")
            status.update(label="🎯 **Search completed!**", state="complete")

        if context is None:
            st.error("❌ Could not retrieve documents. Please check the Azure AI Search connection.")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "❌ I'm having trouble accessing the document database. Please try again later."
            })
        else:
            # Generation phase
            with st.status("🧠 **Generating response...**", expanded=False) as gen_status:
                response_placeholder = st.empty()
                full_response = ""
                
                # Stream the LLM response
                for chunk in stream_llm_response(prompt, context):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                gen_status.update(label="✅ **Response generated!**", state="complete")

            # Add the final assistant response to the message history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
```

# Column 2: Sources and Controls

with col2:
st.markdown(’<div class="sources-container">’, unsafe_allow_html=True)

```
st.markdown("### 🎛️ **Controls**")

# Clear conversation button
if st.button("🧹 Clear Conversation", use_container_width=True):
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "👋 **Welcome back!** I'm ready to help you explore your documents again."
    }]
    st.session_state.latest_sources = []
    st.session_state.message_count = 0
    st.rerun()

st.markdown("---")

# Display sources for the latest response
st.markdown("### 📚 **Retrieved Sources**")

if st.session_state.latest_sources:
    for i, source in enumerate(st.session_state.latest_sources):
        st.markdown(f"""
        <div class="source-card">
            <h4 style="margin:0 0 0.5rem 0; color:#667eea;">📄 Source {i+1}</h4>
            <p style="margin:0.25rem 0;"><strong>Document:</strong> {source.get('document_title', 'N/A')}</p>
            <p style="margin:0.25rem 0;"><strong>Section:</strong> {source.get('title', 'N/A')}</p>
            <p style="margin:0.25rem 0;"><strong>Topic:</strong> {source.get('topic', 'N/A')}</p>
            <p style="margin:0.25rem 0;"><strong>Author:</strong> {source.get('author', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Relevance score with custom styling
        score = source.get('relevance_score', 0.0)
        st.progress(min(score, 1.0), text=f"**Relevance Score:** {score:.3f}")
        st.markdown("<br>", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="info-box">
        <p style="margin:0;"><strong>💡 Tip:</strong> Sources and relevance scores for retrieved documents will appear here after you ask a question.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
```

# Footer

st.markdown(”—”)
st.markdown(”””

<div style="text-align: center; opacity: 0.7; padding: 1rem;">
    <p>🚀 Powered by Azure OpenAI & Azure Cognitive Search | Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
