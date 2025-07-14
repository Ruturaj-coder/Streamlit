import streamlit as st
import requests

# ----------------- CONFIG -----------------
API_CHAT_URL = "http://localhost:7071/api/chat"
API_FILTERS_URL = "http://localhost:7071/api/filters"

# ----------------- UI SETUP -----------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 RAG Chatbot with Azure AI Search (Standalone Friendly)")

# ----------------- Filter Fetching -----------------
@st.cache_data(show_spinner=False)
def load_filter_values():
    try:
        response = requests.get(API_FILTERS_URL, timeout=3)
        response.raise_for_status()
        data = response.json()
        authors = sorted(data.get("authors", []))
        categories = sorted(data.get("categories", []))
    except Exception as e:
        st.sidebar.warning("⚠️ Backend not available. Using mock filters.")
        authors = ["Alice", "Bob", "Charlie"]
        categories = ["AI", "Cloud", "DevOps", "Security"]
    return authors, categories

# Load filters (real or mock)
authors, categories = load_filter_values()

# ----------------- SIDEBAR FILTERS -----------------
with st.sidebar:
    st.header("📂 Filters")
    selected_author = st.selectbox("📌 Document Author", [""] + authors)
    selected_category = st.selectbox("📁 Document Category", [""] + categories)
    selected_date = st.text_input("📅 Document Date (e.g. 2024-01-01)")

# ----------------- MAIN CHAT -----------------
query = st.text_input("🔍 Ask a question")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please type a question.")
    else:
        filters = {}
        if selected_author:
            filters["author"] = selected_author
        if selected_category:
            filters["category"] = selected_category
        if selected_date:
            filters["date"] = selected_date

        payload = {
            "query": query,
            "filters": filters
        }

        with st.spinner("Thinking..."):
            try:
                response = requests.post(API_CHAT_URL, json=payload, timeout=10)
                response.raise_for_status()
                data = response.json()

                st.success("✅ Answer")
                st.markdown(f"**{data['answer']}**")

                if "documents" in data and data["documents"]:
                    st.markdown("### 📄 Source Documents")
                    for i, doc in enumerate(data["documents"], 1):
                        st.markdown(f"**{i}. {doc['title']}**")
                        st.markdown(f"- **Author:** {doc['author']}")
                        st.markdown(f"- **Category:** {doc['category']}")
                        st.markdown(f"- **Date:** {doc['date']}")
                        st.markdown(f"> {doc['content'][:300]}...")
                        st.markdown("---")
                else:
                    st.info("No documents found.")

            except requests.exceptions.RequestException as e:
                st.error("❌ Could not connect to backend. Please start your Azure Function App.")
