import streamlit as st
import requests

API_CHAT_URL = "http://localhost:5000/chat"
API_FILTERS_URL = "http://localhost:5000/filters"

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 RAG Chatbot with Azure Filters")

# Load filters once
@st.cache_data(show_spinner=False)
def load_filter_values():
    response = requests.get(API_FILTERS_URL)
    data = response.json()
    return data.get("authors", []), data.get("categories", [])

authors, categories = load_filter_values()

# Sidebar for filters
with st.sidebar:
    st.header("📂 Filters")
    selected_author = st.selectbox("Document Author", [""] + authors)
    selected_category = st.selectbox("Document Category", [""] + categories)
    selected_date = st.text_input("Document Date (e.g. 2024-01-01)")

# User input
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
            response = requests.post(API_CHAT_URL, json=payload)
            if response.status_code == 200:
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
                st.error(f"Error: {response.text}")
