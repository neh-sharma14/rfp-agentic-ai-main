import os
import re
import time
from typing import Optional, List, Any, Tuple, Union
import boto3
from pyparsing import Dict
import tiktoken
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from crewai.tools import BaseTool
import openai
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from requests.exceptions import ReadTimeout, ConnectTimeout, Timeout
from opensearchpy.exceptions import ConnectionTimeout, TransportError

class OpenSearchQueryTool(BaseTool):
    name: str = "OpenSearchQueryTool"
    description: str = "Synchronous semantic search on AWS OpenSearch using OpenAI embeddings with quality filtering."
    unique_id: Optional[str] = None
    filename : Optional[str] = None
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    index: Optional[str] = None
    tokenizer: Optional[Any] = None
    client: Optional[Any] = None
    openai_client: Optional[Any] = None
    min_score: float = 0.5  # Minimum score threshold for good results

    def __init__(self, min_score: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        self.index = os.getenv("OPENSEARCH_MAIN_INDEX_NAME")
        self.tokenizer = self._get_tokenizer(self.model)
        self.client = self._get_opensearch_client()
        # Support both OPENAI_API_KEY and OPENAI_KEY for flexibility
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        self.openai_client = openai.OpenAI(api_key=api_key)
        self.min_score = min_score

    def _get_opensearch_client(self):
        session = boto3.Session()
        credentials = session.get_credentials()
        region = os.getenv("AWS_REGION_NAME", "us-east-1")
        host = os.getenv("AWS_OPENSEARCH_SERVERLESS_COLLECTION_HOST")

        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            "aoss",
            session_token=credentials.token
        )

        return OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            retry_on_timeout=True,
            max_retries=5,  # Increased retries
            timeout=60,     # Increased timeout to 60 seconds
            retry_on_status=(502, 503, 504, 429),  # Retry on server errors and rate limits
        )

    def _retry_with_backoff(self, func, max_retries=3, base_delay=1.0, max_delay=30.0):
        """
        Retry a function with exponential backoff for handling timeout and connection errors.

        Args:
            func: Function to retry
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
        """
        for attempt in range(max_retries + 1):
            try:
                return func()
            except (ReadTimeout, ConnectTimeout, Timeout, ConnectionTimeout, TransportError) as e:
                if attempt == max_retries:
                    # Last attempt failed, raise the exception
                    raise e

                # Calculate delay with exponential backoff
                delay = min(base_delay * (2 ** attempt), max_delay)
                print(f"⚠️ OpenSearch connection attempt {attempt + 1} failed: {str(e)[:100]}...")
                print(f"🔄 Retrying in {delay:.1f} seconds... (attempt {attempt + 2}/{max_retries + 1})")
                time.sleep(delay)
            except Exception as e:
                # For other exceptions, don't retry
                raise e

    def _get_tokenizer(self, model_name: str):
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    # Chunking removed to keep outputs compact and reduce token usage

    def _categorize_score(self, score: float) -> str:
        """Categorize the similarity score"""
        if score >= 0.9:
            return "EXCELLENT"
        elif score >= 0.7:
            return "GOOD"
        elif score >= 0.5:
            return "MODERATE"
        else:
            return "POOR"


    def _run(self, query: str, mode: str = "hybrid") -> str:
        """
        Executes a search query against OpenSearch with support for:
        - "bm25": keyword search
        - "semantic": vector similarity search
        - "hybrid": combination of both
        """

        # Input validation and error handling
        if not isinstance(query, str):
            if isinstance(query, dict):
                # Handle common case where LLM passes a dictionary
                error_msg = (
                    f"ERROR: OpenSearchQueryTool received a dictionary instead of a string query. "
                    f"Please use only plain text queries like 'Section L proposal requirements' "
                    f"instead of JSON format. Received: {str(query)[:200]}..."
                )
                print(error_msg)
                return error_msg
            else:
                error_msg = f"ERROR: Query must be a string, received {type(query)}: {query}"
                print(error_msg)
                return error_msg

        if not query.strip():
            error_msg = "ERROR: Query cannot be empty. Please provide a descriptive search query."
            print(error_msg)
            return error_msg

        filename = self.filename
        batch_size = 20
        filters = []
        try:
            embedding = None
            if mode in ["semantic", "hybrid"]:
                # Add retry logic for embedding generation
                def generate_embedding():
                    return self.openai_client.embeddings.create(
                        input=query,
                        model=self.embedding_model
                    )

                print(f"🔗 Generating embeddings with retry logic...")
                embedding_response = self._retry_with_backoff(generate_embedding, max_retries=2, base_delay=1.0, max_delay=5.0)
                embedding = embedding_response.data[0].embedding
        except Exception as e:
            error_msg = f"❌ Failed to generate embedding after retries: {e}"
            print(error_msg)
            return error_msg

        should_clauses = []

        if mode in ["semantic", "hybrid"] and embedding:
            should_clauses.append({
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": batch_size
                    }
                }
            })

        if mode in ["bm25", "hybrid"]:
            should_clauses.append({
                "match": {
                    "text": {
                        "query": query,
                        "boost": 2.0  # prioritize lexical match
                    }
                }
            })

        if not should_clauses:
            return "❌ Invalid search mode. Choose from 'bm25', 'semantic', or 'hybrid'."

        if self.unique_id:
            filters.append({"term": {"unique_id": self.unique_id}})

        filename = self.filename # safely pull it from optional kwargs
        if filename:
            filters.append({"term": {"filename": filename}})

        query_body = {
            "size": batch_size,
            "_source": ["text", "filename", "unique_id"],
            "min_score": self.min_score,
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1,
                    "filter": filters
                }
            }
        }

        try:
            # Use retry logic for OpenSearch queries
            def execute_search():
                return self.client.search(index=self.index, body=query_body)

            print(f"🔍 Executing OpenSearch query with retry logic...")
            response = self._retry_with_backoff(execute_search, max_retries=3, base_delay=2.0, max_delay=15.0)

            hits = response["hits"]["hits"]
            filtered_hits = [hit for hit in hits if hit.get('_score', 0) >= self.min_score]

            if not filtered_hits:
                return f"⚠️ No high-quality documents found (minimum score: {self.min_score})."

            # Prepare compact top results
            top_k = min(8, len(filtered_hits))
            lines = []
            for i, hit in enumerate(filtered_hits[:top_k], 1):
                score = hit.get('_score', 0.0)
                score_category = self._categorize_score(score)
                text = hit['_source'].get('text', '') or ''
                filename = hit['_source'].get('filename', 'N/A')
                snippet = (text[:800] + '…') if len(text) > 800 else text
                lines.append(f"[{i}] {score_category} {score:.3f} | {filename}\n{snippet}")

            compact_context = "\n\n".join(lines)

            # Summarize using the chat model to reduce token size
            summary_text = ""
            try:
                # Also add retry logic for OpenAI API calls
                def execute_chat():
                    return self.openai_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You summarize search results into 5-10 terse bullet points with brief citations like [# filename]. Keep it under 200 words."},
                            {"role": "user", "content": f"Query: {query}\n\nTop results:\n{compact_context}"}
                        ],
                        temperature=0.2,
                        max_tokens=350,
                    )

                chat = self._retry_with_backoff(execute_chat, max_retries=2, base_delay=1.0, max_delay=5.0)
                summary_text = chat.choices[0].message.content.strip()
                #summary_text= compact_context
            except Exception as e:
                # Fallback to concatenated compact context if summarization fails
                summary_text = f"Summary unavailable due to error: {e}. Showing top snippets:\n\n{compact_context}"

            header = (
                f"✅ OpenSearch results for: '{query}' | mode={mode} | min_score={self.min_score}\n"
                f"Retrieved {len(filtered_hits)} matches. Showing top {top_k}.\n\n"
            )
            return header + summary_text

        except (ReadTimeout, ConnectTimeout, Timeout, ConnectionTimeout, TransportError) as e:
            error_msg = (
                f"❌ OpenSearch connection failed after multiple retries: {str(e)}\n"
                f"This may be due to network issues or AWS OpenSearch service being unavailable.\n"
                f"Please check your network connection and AWS service status."
            )
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"❌ OpenSearch query failed: {e}"
            print(error_msg)
            return error_msg



    def set_min_score(self, min_score: float):
        """Update the minimum score threshold"""
        self.min_score = min_score

