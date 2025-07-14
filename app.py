import azure.functions as func
import json
import logging
import os
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
import openai
from typing import Dict, List, Any, Optional

# Configure logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(**name**)

# Initialize Function App

app = func.FunctionApp()

# Global variables for clients

search_client = None
openai_client = None

def get_search_client():
“”“Initialize and return Azure Search client”””
global search_client
if search_client is None:
search_endpoint = os.getenv(“AZURE_SEARCH_ENDPOINT”)
search_key = os.getenv(“AZURE_SEARCH_KEY”)
search_index = os.getenv(“AZURE_SEARCH_INDEX”)

```
    if not all([search_endpoint, search_key, search_index]):
        raise ValueError("Missing required Azure Search configuration")
    
    credential = AzureKeyCredential(search_key)
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=search_index,
        credential=credential
    )
return search_client
```

def get_openai_client():
“”“Initialize and return OpenAI client”””
global openai_client
if openai_client is None:
openai_api_key = os.getenv(“OPENAI_API_KEY”)
if not openai_api_key:
raise ValueError(“Missing OpenAI API key”)

```
    openai_client = openai.OpenAI(api_key=openai_api_key)
return openai_client
```

def build_search_filter(filters: Dict[str, str]) -> Optional[str]:
“”“Build OData filter string from filters”””
filter_parts = []

```
for key, value in filters.items():
    if value and value.strip():
        # Escape single quotes in the value
        escaped_value = value.replace("'", "''")
        filter_parts.append(f"{key} eq '{escaped_value}'")

return " and ".join(filter_parts) if filter_parts else None
```

def search_documents(query: str, filters: Dict[str, str], top: int = 5) -> List[Dict[str, Any]]:
“”“Search documents using Azure AI Search”””
try:
client = get_search_client()

```
    # Build filter
    search_filter = build_search_filter(filters)
    
    # Perform search
    results = client.search(
        search_text=query,
        top=top,
        filter=search_filter,
        select=["id", "title", "content", "author", "category", "date"],
        search_mode="all"
    )
    
    documents = []
    for result in results:
        documents.append({
            "id": result.get("id", ""),
            "title": result.get("title", ""),
            "content": result.get("content", ""),
            "author": result.get("author", ""),
            "category": result.get("category", ""),
            "date": result.get("date", "")
        })
    
    return documents
    
except Exception as e:
    logger.error(f"Search error: {str(e)}")
    raise
```

def generate_answer(query: str, documents: List[Dict[str, Any]]) -> str:
“”“Generate answer using OpenAI with retrieved documents”””
try:
client = get_openai_client()

```
    # Prepare context from documents
    context = ""
    for i, doc in enumerate(documents, 1):
        context += f"Document {i}:\n"
        context += f"Title: {doc['title']}\n"
        context += f"Author: {doc['author']}\n"
        context += f"Category: {doc['category']}\n"
        context += f"Content: {doc['content']}\n\n"
    
    # Create prompt
    prompt = f"""Based on the following documents, please answer the user's question. 
```

If the documents don’t contain enough information to answer the question, please say so.

Context Documents:
{context}

User Question: {query}

Please provide a comprehensive answer based on the documents above:”””

```
    # Generate response
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documents."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.3
    )
    
    return response.choices[0].message.content
    
except Exception as e:
    logger.error(f"OpenAI error: {str(e)}")
    return f"Sorry, I couldn't generate an answer due to an error: {str(e)}"
```

def get_filter_values() -> Dict[str, List[str]]:
“”“Get unique values for filters from the search index”””
try:
client = get_search_client()

```
    # Get all documents (you might want to limit this in production)
    results = client.search(
        search_text="*",
        select=["author", "category"],
        top=1000
    )
    
    authors = set()
    categories = set()
    
    for result in results:
        if result.get("author"):
            authors.add(result["author"])
        if result.get("category"):
            categories.add(result["category"])
    
    return {
        "authors": sorted(list(authors)),
        "categories": sorted(list(categories))
    }
    
except Exception as e:
    logger.error(f"Filter values error: {str(e)}")
    # Return empty lists if there's an error
    return {"authors": [], "categories": []}
```

@app.route(route=“chat”, methods=[“POST”])
def chat_endpoint(req: func.HttpRequest) -> func.HttpResponse:
“”“Handle chat requests”””
try:
# Parse request
try:
req_body = req.get_json()
except ValueError:
return func.HttpResponse(
json.dumps({“error”: “Invalid JSON”}),
status_code=400,
mimetype=“application/json”
)

```
    if not req_body:
        return func.HttpResponse(
            json.dumps({"error": "Request body is required"}),
            status_code=400,
            mimetype="application/json"
        )
    
    query = req_body.get("query", "").strip()
    filters = req_body.get("filters", {})
    
    if not query:
        return func.HttpResponse(
            json.dumps({"error": "Query is required"}),
            status_code=400,
            mimetype="application/json"
        )
    
    # Search documents
    documents = search_documents(query, filters)
    
    # Generate answer
    answer = generate_answer(query, documents)
    
    # Return response
    response = {
        "answer": answer,
        "documents": documents,
        "query": query,
        "filters": filters
    }
    
    return func.HttpResponse(
        json.dumps(response),
        status_code=200,
        mimetype="application/json"
    )
    
except Exception as e:
    logger.error(f"Chat endpoint error: {str(e)}")
    return func.HttpResponse(
        json.dumps({"error": f"Internal server error: {str(e)}"}),
        status_code=500,
        mimetype="application/json"
    )
```

@app.route(route=“filters”, methods=[“GET”])
def filters_endpoint(req: func.HttpRequest) -> func.HttpResponse:
“”“Handle filter values requests”””
try:
filter_values = get_filter_values()

```
    return func.HttpResponse(
        json.dumps(filter_values),
        status_code=200,
        mimetype="application/json"
    )
    
except Exception as e:
    logger.error(f"Filters endpoint error: {str(e)}")
    return func.HttpResponse(
        json.dumps({"error": f"Internal server error: {str(e)}"}),
        status_code=500,
        mimetype="application/json"
    )
```

@app.route(route=“health”, methods=[“GET”])
def health_endpoint(req: func.HttpRequest) -> func.HttpResponse:
“”“Health check endpoint”””
return func.HttpResponse(
json.dumps({“status”: “healthy”}),
status_code=200,
mimetype=“application/json”
)
