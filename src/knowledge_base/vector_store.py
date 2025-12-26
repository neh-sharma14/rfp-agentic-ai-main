import os
from typing import Optional, List, Dict, Any, BinaryIO
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import openai
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    Settings
)
from llama_index.vector_stores.opensearch import (
    OpensearchVectorStore,
    OpensearchVectorClient
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from io import BytesIO
from .text_extract import extract_text
import boto3
from urllib.parse import unquote

# Constants
MAIN_INDEX_NAME = os.environ.get("OPENSEARCH_MAIN_INDEX_NAME", "rfp_knowledge_base_test")
AWS_OPENSEARCH_SERVERLESS_COLLECTION_HOST = os.environ.get("AWS_OPENSEARCH_SERVERLESS_COLLECTION_HOST")
AWS_REGION_NAME = os.environ.get("AWS_REGION_NAME")
OPENAI_KEY = os.environ.get("OPENAI_KEY")

# Initialize OpenAI client
openai.api_key = OPENAI_KEY

# Initialize OpenSearch client with AWS4Auth
credentials = boto3.Session().get_credentials()
awsauth = AWSV4SignerAuth(credentials, AWS_REGION_NAME, "aoss")

opensearch_client = OpenSearch(
    hosts=[{
        'host': AWS_OPENSEARCH_SERVERLESS_COLLECTION_HOST,
        'port': 443
    }],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=30,  # Connection timeout in seconds
    retry_on_timeout=True,
    max_retries=3,
    retry_on_status=[408, 429, 500, 502, 503, 504]  # Retry on these status codes
)

# Default field names
TEXT_FIELD = "content"
EMBEDDING_FIELD = "embedding"

# Configure global settings
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-large",
    embed_batch_size=100,
    api_key=OPENAI_KEY
)
Settings.node_parser = SimpleNodeParser.from_defaults(
    chunk_size=1000,
    chunk_overlap=200
)

def ensure_main_index_exists():
    """Ensure the main index exists, create if it doesn't"""
    try:
        print(f"\nChecking if index {MAIN_INDEX_NAME} exists...")
        if not opensearch_client.indices.exists(index=MAIN_INDEX_NAME):
            print(f"Index {MAIN_INDEX_NAME} does not exist. Creating...")
            index_body = {
                "settings": {
                    "index.knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "refresh_interval": "30s"  # Increase refresh interval for better performance
                },
                "mappings": {
                    "properties": {
                        EMBEDDING_FIELD: {
                            "type": "knn_vector",
                            "dimension": 3072,  # OpenAI text-embedding-3-large uses 3072 dimensions
                            "method": {
                                "name": "hnsw",
                                "engine": "nmslib",
                                "space_type": "cosinesimil"
                            }
                        },
                        TEXT_FIELD: {"type": "text"},
                        "metadata": {"type": "object"}
                    }
                }
            }
            opensearch_client.indices.create(index=MAIN_INDEX_NAME, body=index_body)
            print(f"Successfully created index {MAIN_INDEX_NAME}")
        else:
            print(f"Index {MAIN_INDEX_NAME} already exists")
    except Exception as e:
        print(f"Error ensuring main index exists: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        raise

class KnowledgeBase:
    """A vector store implementation using LlamaIndex and OpenSearch"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnowledgeBase, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the vector store with a single index for all documents"""
        # Skip initialization if already done
        if self._initialized:
            return

        try:
            print("\nInitializing VectorStore...")
            self.index_name = MAIN_INDEX_NAME
            print(f"Using index: {self.index_name}")

            # Ensure main index exists
            ensure_main_index_exists()

            # Initialize node parser for document processing
            print("Initializing node parser...")
            self.node_parser = SimpleNodeParser.from_defaults(
                chunk_size=500,  # Reduced from 1000 to 500 for better granularity
                chunk_overlap=100  # Reduced from 200 to 100 to avoid too much overlap
            )
            print("Node parser initialized successfully")

            # Initialize OpenSearch vector client
            print("Initializing OpenSearch vector client...")
            self.vector_client = OpensearchVectorClient(
                endpoint=f"https://{AWS_OPENSEARCH_SERVERLESS_COLLECTION_HOST}",
                index=self.index_name,
                dim=3072,
                embedding_field=EMBEDDING_FIELD,
                text_field=TEXT_FIELD,
                http_auth=awsauth,
                connection_class=RequestsHttpConnection,
                timeout=30,
                retry_on_timeout=True,
                max_retries=3,
                retry_on_status=[408, 429, 500, 502, 503, 504]
            )
            print("OpenSearch vector client initialized successfully")

            # Initialize vector store with the client
            print("Initializing vector store...")
            self.vector_store = OpensearchVectorStore(self.vector_client)
            print("Vector store initialized successfully")

            # Initialize storage context
            print("Initializing storage context...")
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            print("Storage context initialized successfully")

            # Create a new index
            print("Creating new vector store index...")
            self.index = VectorStoreIndex(
                [],
                storage_context=self.storage_context
            )
            print("Vector store index created successfully")

            # Initialize retriever and query engine with higher top_k
            print("Initializing retriever and query engine...")
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=20  # Increased from 5 to 20
            )
            self.query_engine = RetrieverQueryEngine.from_args(
                retriever=self.retriever,
                similarity_top_k=20  # Added explicit top_k here as well
            )
            print("Retriever and query engine initialized successfully")
            print("VectorStore initialization completed successfully")

            # Mark as initialized
            self._initialized = True

        except Exception as e:
            print(f"\nError initializing VectorStore:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            raise

    def _get_document_id(self, metadata: Dict[str, Any]) -> str:
        """Generate a unique document ID based on metadata"""
        # Use solicitation_id and filename as unique identifier
        return f"{get('solicitation_id', 'unknown')}_{get('filename', 'unknown')}"

    def _document_exists(self, metadata: Dict[str, Any]) -> bool:
        """
        Check if a document with the same metadata already exists in the index.

        Args:
            metadata: Document metadata to check

        Returns:
            bool: True if document exists, False otherwise
        """
        try:
            print(f"\nChecking if document exists with metadata: {metadata}")
            # Search for documents with matching solicitation_id and filename
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"solicitation_id.keyword": get('solicitation_id')}},
                            {"term": {"filename.keyword": get('filename')}}
                        ]
                    }
                }
            }

            print(f"Executing search query: {query}")
            response = opensearch_client.search(
                index=self.index_name,
                body=query
            )

            exists = response['hits']['total']['value'] > 0
            print(f"Document exists check result: {exists}")
            if exists:
                print(f"Found existing document with {response['hits']['total']['value']} matches")
            return exists
        except Exception as e:
            print(f"Error checking document existence: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            return False

    def add_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Add or update a document in the vector store.

        Args:
            content: The text content of the document
            metadata: Document metadata including user_id, solicitation_id, and filename

        Returns:
            The ID of the added/updated document
        """
        try:
            print(f"\nAdding document with metadata: {metadata}")
            if not all(key in metadata for key in ['user_id', 'solicitation_id', 'filename']):
                raise ValueError("Metadata must include user_id, solicitation_id, and filename")

            # Remove redundant file_id if it exists
            if 'file_id' in metadata:
                del metadata['file_id']

            doc_id = self._get_document_id(metadata)
            print(f"Generated document ID: {doc_id}")

            # Check if document exists
            if self._document_exists(metadata):
                print(f"Document with solicitation_id={metadata['solicitation_id']} and filename={metadata['filename']} already exists")
                return doc_id

            # Add new document
            print("Creating new document...")
            document = Document(
                text=content,
                metadata=metadata,
                id_=doc_id
            )

            # Use the instance's node_parser
            print("Parsing document into nodes...")
            nodes = self.node_parser.get_nodes_from_documents([document])
            print(f"Created {len(nodes)} nodes from document")

            print("Inserting nodes into index...")
            self.index.insert_nodes(nodes)
            print(f"Successfully added document {doc_id}")

            return doc_id

        except Exception as e:
            print(f"\nError adding document:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            raise

    def add_file(self, file: BinaryIO, filename: str, metadata: Dict[str, Any]) -> str:
        """
        Process and add a file to the vector store.

        Args:
            file: File-like object containing the document
            filename: Name of the file
            metadata: Document metadata including user_id, solicitation_id, and filename

        Returns:
            The ID of the added document
        """
        try:
            content = extract_text(file, filename)
            if metadata is None:
                metadata = {}
            metadata["filename"] = filename
            # Remove file_id if it exists
            if 'file_id' in metadata:
                del metadata['file_id']
            return self.add_document(content, metadata)
        except Exception as e:
            raise Exception(f"Error processing file {filename}: {str(e)}")

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.

        Args:
            doc_id: Document ID

        Returns:
            Document content and metadata if found, None otherwise
        """
        try:
            node = self.index.docstore.docs.get(doc_id)
            if node:
                return {
                    "content": {
                        "text": node.text
                    },
                    "metadata": node.metadata
                }
            return None
        except Exception as e:
            raise Exception(f"Error retrieving document {doc_id}: {str(e)}")

    def update_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Update an existing document.

        Args:
            doc_id: ID of the document to update
            content: New content for the document
            metadata: New metadata for the document

        Returns:
            ID of the updated document
        """
        try:
            # Delete existing document
            self.index.delete_nodes([doc_id])

            # Add updated document
            return self.add_document(content, metadata)

        except Exception as e:
            raise Exception(f"Error updating document {doc_id}: {str(e)}")

    def delete_document(self, doc_id: str) -> None:
        """
        Delete a document from the vector store.

        Args:
            doc_id: ID of the document to delete
        """
        try:
            self.index.delete_nodes([doc_id])
        except Exception as e:
            raise Exception(f"Error deleting document {doc_id}: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary containing vector store statistics
        """
        try:
            # Get all documents
            all_docs = self.index.docstore.docs
            print(f"All docs: {all_docs}")

            # Count documents by solicitation_id
            solicitation_counts = {}
            for doc in all_docs.values():
                solicitation_id = doc.get('solicitation_id', 'unknown')
                solicitation_counts[solicitation_id] = solicitation_counts.get(solicitation_id, 0) + 1

            return {
                "index_name": self.index_name,
                "total_documents": len(all_docs),
                "documents_by_solicitation": solicitation_counts,
                "vector_store": self.vector_store.__class__.__name__,
                "embedding_model": Settings.embed_model.__class__.__name__,
                "chunk_size": self.node_parser.chunk_size,
                "chunk_overlap": self.node_parser.chunk_overlap
            }
        except Exception as e:
            raise Exception(f"Error getting vector store stats: {str(e)}")

    def query_with_filters(self, query_text: str, filters: Dict[str, Any] = None, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Query the vector store with filters.

        Args:
            query_text: The text to search for
            filters: Dictionary of filters to apply
            top_k: Number of results to return (default 20)

        Returns:
            List of matching documents with their scores
        """
        try:
            print(f"\nExecuting query with filters: {filters}")

            # Build filter query
            filter_query = {
                "bool": {
                    "must": []
                }
            }

            # Add filters if provided
            if filters:
                if 'solicitation_id' in filters:
                    filter_query["bool"]["must"].append({
                        "term": {"solicitation_id.keyword": filters['solicitation_id']}
                    })

                if 'user_id' in filters:
                    filter_query["bool"]["must"].append({
                        "term": {"user_id.keyword": filters['user_id']}
                    })

            # Create the search query with hybrid search
            search_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": self._get_embedding(query_text),
                                        "k": top_k
                                    }
                                }
                            },
                            {
                                "match": {
                                    "content": {
                                        "query": query_text,
                                        "boost": 0.5  # Lower boost for text matching
                                    }
                                }
                            }
                        ],
                        "filter": filter_query["bool"]["must"]
                    }
                },
                "size": top_k
            }

            # Execute search
            response = opensearch_client.search(
                index=self.index_name,
                body=search_query
            )

            # Process results
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'score': hit['_score'],
                    'content': hit['_source']['content'],
                    'metadata': hit['_source']['metadata']
                }
                results.append(result)

            print(f"Found {len(results)} results")
            return results

        except Exception as e:
            print(f"\nError executing query with filters:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            raise

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text using OpenAI's embedding model.

        Args:
            text: Text to get embedding for

        Returns:
            List of floats representing the embedding vector
        """
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_KEY)
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            raise

    def delete_documents_by_ids(self, doc_ids: List[str]) -> int:
        """
        Delete documents by their IDs.

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        try:
            print(f"\nDeleting documents with IDs: {doc_ids}")

            # Check if index exists
            if not opensearch_client.indices.exists(index=self.index_name):
                print(f"Index {self.index_name} does not exist. No documents to delete.")
                return 0

            # Build delete query
            delete_query = {
                "query": {
                    "ids": {
                        "values": doc_ids
                    }
                }
            }

            # Execute delete by query
            response = opensearch_client.delete_by_query(
                index=self.index_name,
                body=delete_query,
                refresh=True,
                wait_for_completion=True
            )

            deleted_count = response.get('deleted', 0)
            print(f"Successfully deleted {deleted_count} documents")
            return deleted_count

        except Exception as e:
            print(f"\nError deleting documents by IDs:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            if "index_not_found_exception" in str(e).lower():
                print(f"Index {self.index_name} does not exist. No documents to delete.")
                return 0
            raise

    def delete_documents_by_metadata(self, metadata_filters: Dict[str, Any]) -> int:
        """
        Delete all documents matching the given metadata filters.

        Args:
            metadata_filters: Dictionary of metadata filters to match
                Example: {
                    'solicitation_id': '123',
                    'user_id': '456'
                }

        Returns:
            Number of documents deleted
        """
        try:
            print(f"\nDeleting documents with metadata filters: {metadata_filters}")

            # Ensure index exists
            ensure_main_index_exists()

            # First, get all matching document IDs
            search_query = {
                "query": {
                    "bool": {
                        "must": []
                    }
                },
                "size": 10000,  # Large enough to get all documents
                "_source": False  # We only need IDs
            }

            # Add metadata filters to query
            for key, value in metadata_filters.items():
                search_query["query"]["bool"]["must"].append({
                    "term": {f"{key}.keyword": value}
                })

            # Execute search to get document IDs
            response = opensearch_client.search(
                index=self.index_name,
                body=search_query
            )

            # Extract and decode document IDs
            doc_ids = []
            for hit in response['hits']['hits']:
                try:
                    # URL decode the ID
                    doc_id = unquote(hit['_id'])
                    doc_ids.append(doc_id)
                except Exception as e:
                    print(f"Error decoding document ID {hit['_id']}: {str(e)}")
                    continue

            if not doc_ids:
                print("No documents found matching the filters")
                return 0

            print(f"Found {len(doc_ids)} documents to delete")

            # Delete documents one by one
            deleted_count = 0
            failed_count = 0
            for doc_id in doc_ids:
                try:
                    # Check if document exists before deleting
                    try:
                        opensearch_client.get(
                            index=self.index_name,
                            id=doc_id
                        )
                    except Exception as e:
                        if "not_found_exception" in str(e).lower():
                            print(f"Document {doc_id} no longer exists, skipping...")
                            continue
                        raise

                    # Delete the document
                    opensearch_client.delete(
                        index=self.index_name,
                        id=doc_id
                    )
                    deleted_count += 1
                    if deleted_count % 10 == 0:  # Progress update every 10 documents
                        print(f"Deleted {deleted_count} documents so far...")
                except Exception as e:
                    print(f"Error deleting document {doc_id}: {str(e)}")
                    failed_count += 1
                    continue

            print(f"Deletion complete. Successfully deleted {deleted_count} documents. Failed to delete {failed_count} documents.")
            return deleted_count

        except Exception as e:
            print(f"\nError deleting documents:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            if "index_not_found_exception" in str(e).lower():
                print(f"Index {self.index_name} does not exist. No documents to delete.")
                return 0
            raise

    def delete_all_documents(self) -> int:
        """
        Delete all documents from the index.

        Returns:
            Number of documents deleted
        """
        try:
            print("\nDeleting all documents from index")

            # Ensure index exists
            ensure_main_index_exists()

            # First, get all document IDs
            search_query = {
                "query": {
                    "match_all": {}
                },
                "size": 10000,
                "_source": False
            }

            # Execute search to get document IDs
            response = opensearch_client.search(
                index=self.index_name,
                body=search_query
            )

            # Extract document IDs
            doc_ids = [hit['_id'] for hit in response['hits']['hits']]

            if not doc_ids:
                print("No documents found to delete")
                return 0

            print(f"Found {len(doc_ids)} documents to delete")

            # Delete documents one by one
            deleted_count = 0
            for doc_id in doc_ids:
                try:
                    opensearch_client.delete(
                        index=self.index_name,
                        id=doc_id
                    )
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting document {doc_id}: {str(e)}")
                    continue

            print(f"Successfully deleted {deleted_count} documents")
            return deleted_count

        except Exception as e:
            print(f"\nError deleting all documents:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            if "index_not_found_exception" in str(e).lower():
                print(f"Index {self.index_name} does not exist. No documents to delete.")
                return 0
            raise

    def list_documents(self, filters: Dict[str, Any] = None, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """
        List all documents in the knowledge base with optional filters.

        Args:
            filters: Dictionary of filters to apply (e.g., {'solicitation_id': '123', 'user_id': '456'})
            limit: Maximum number of documents to return
            offset: Number of documents to skip

        Returns:
            Dictionary containing documents and metadata
        """
        try:
            print(f"\nListing documents with filters: {filters}")

            # Check if index exists
            if not opensearch_client.indices.exists(index=self.index_name):
                print(f"Index {self.index_name} does not exist. No documents to list.")
                return {
                    "total": 0,
                    "documents": [],
                    "limit": limit,
                    "offset": offset
                }

            # Build search query
            search_query = {
                "query": {
                    "bool": {
                        "must": []
                    }
                },
                "size": limit,
                "from": offset,
                "sort": [
                    {"_score": "desc"}
                ]
            }

            # Add filters if provided
            if filters:
                if 'solicitation_id' in filters:
                    search_query["query"]["bool"]["must"].append({
                        "term": {"solicitation_id.keyword": filters['solicitation_id']}
                    })
                if 'user_id' in filters:
                    search_query["query"]["bool"]["must"].append({
                        "term": {"user_id.keyword": filters['user_id']}
                    })

            # If no filters, use match_all
            if not search_query["query"]["bool"]["must"]:
                search_query["query"] = {"match_all": {}}

            # Execute search
            response = opensearch_client.search(
                index=self.index_name,
                body=search_query
            )

            # Process results
            documents = []
            for hit in response['hits']['hits']:
                doc = {
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'metadata': hit['_source'].get('metadata', {}),
                    'content_preview': hit['_source'].get('content', '')[:200] + '...' if len(hit['_source'].get('content', '')) > 200 else hit['_source'].get('content', '')
                }
                documents.append(doc)

            return {
                "total": response['hits']['total']['value'],
                "documents": documents,
                "limit": limit,
                "offset": offset
            }

        except Exception as e:
            print(f"\nError listing documents:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            if "index_not_found_exception" in str(e).lower():
                return {
                    "total": 0,
                    "documents": [],
                    "limit": limit,
                    "offset": offset
                }
            raise