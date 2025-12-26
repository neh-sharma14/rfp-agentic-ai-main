import os
import time
import fitz  # PyMuPDF
import openai
import boto3
from opensearchpy import OpenSearch, OpenSearchException, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from opensearchpy.helpers import bulk

# === Environment Variables ===
openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("AWS_EMBEDDING_MODEL_ARN")  # For OpenAI usage or switch if using Bedrock
EMBEDDING_DIM = 3072

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENSEARCH_HOST = os.getenv("AWS_OPENSEARCH_SERVERLESS_COLLECTION_HOST")

# === Set up IAM-based Auth for OpenSearch Serverless ===
session = boto3.Session()
credentials = session.get_credentials()
auth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    AWS_REGION,
    "aoss",  # service name for OpenSearch Serverless
    session_token=credentials.token
)

# === OpenSearch Client ===
opensearch_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)
INDEX_NAME= os.getenv("OPENSEARCH_MAIN_INDEX_NAME")
# === FUNCTIONS ===

def read_pdf(file_path):
    """Extracts all text from a PDF file."""
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text, max_len=2000, overlap=400):
    """
    Split text into chunks of max_len characters with optional overlap.

    Parameters:
    - text (str): The full input text.
    - max_len (int): Maximum length of each chunk.
    - overlap (int): Number of characters to overlap between chunks.

    Returns:
    - List[str]: List of text chunks.
    """
    if max_len <= overlap:
        raise ValueError("max_len must be greater than overlap")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        chunks.append(text[start:end])
        start += max_len - overlap

    return chunks


def embed_text(text):
    """Get embeddings from OpenAI (or replace with Bedrock)."""
    response = openai.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    embedding_vector = response.data[0].embedding
    print(f"Embedding length: {len(embedding_vector)}, type: {type(embedding_vector)}, sample: {embedding_vector[:5]}")
    return embedding_vector


def ensure_index_exists(index_name):
    """Check if the index exists, and create it if not."""
    if not opensearch_client.indices.exists(index=index_name):
        index_body = {
            "settings": {
                "index": {
                    "knn": True
                }
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_DIM
                    },
                    "filename": {"type": "keyword"},
                    "solicitation_id": {"type": "keyword"},
                    "unique_id": {"type": "keyword"}  # Add unique_id field
                }
            }
        }
        opensearch_client.indices.create(index=index_name, body=index_body)
        print(f"Created index: {index_name}")
    else:
        print(f"Index already exists: {index_name}")

# Call this at the start of your upload function
def upload_vectors_to_opensearch(files, solicitation_id,unique_id):
    ensure_index_exists(INDEX_NAME)
    print(opensearch_client.indices.get_mapping(index=INDEX_NAME))
    # Delete old documents with same solicitation_id

    bulk_actions = []

    for file_path in files:
        if not file_path.endswith(".pdf"):
            continue

        print(f"Processing {file_path}")
        text = read_pdf(file_path)
        chunks = chunk_text(text)

        for chunk in chunks:
            try:
                embedding = embed_text(chunk)
                if (
                    not embedding or
                    not isinstance(embedding, list) or
                    len(embedding) != EMBEDDING_DIM or
                    not all(isinstance(x, (float, int)) for x in embedding)
                ):
                    print("Skipping invalid embedding")
                    continue

                action = {
                    "_index": INDEX_NAME,
                    "_source": {
                        "text": chunk,
                        "embedding": embedding,
                        "filename": os.path.basename(file_path),
                        "solicitation_id": solicitation_id,
                        "unique_id": unique_id  # Use the unique_id passed to the function
                    }
                }
                bulk_actions.append(action)
            except Exception as e:
                print(f"Error embedding chunk: {e}")

    # === Bulk insert all collected docs ===
    if bulk_actions:
       batch_bulk_upload(opensearch_client, bulk_actions, batch_size=50)

    # === Confirm documents are uploaded and searchable ===
    query = {
        "query": {
            "term": {
                "unique_id": {
                    "value": unique_id
                }
            }
        }
    }

    while True:
        try:
            response = opensearch_client.search(index=INDEX_NAME, body=query)
            count = response['hits']['total']['value']
            print(f"Indexed and searchable documents for unique_id={unique_id}: {count}")

            if count >= len(bulk_actions):
                print("✅ All documents indexed and searchable.")
                break
            else:
                print(f"⚠️ WARNING: Expected {len(bulk_actions)} documents, but only {count} found.")
                time.sleep(5)  # Wait before next check
        except Exception as e:
            print(f"Search error: {e}")
            break


def batch_bulk_upload(client, actions, batch_size=50):
    for i in range(0, len(actions), batch_size):
        batch = actions[i:i+batch_size]
        try:
            success, _ = bulk(client, batch, request_timeout=120)
            print(f"Uploaded batch {i//batch_size + 1}: {success} docs")
        except Exception as e:
            print(f"Bulk batch {i//batch_size + 1} failed: {e}")
