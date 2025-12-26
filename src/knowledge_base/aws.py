if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

import os
import time
import boto3
import json
from typing import Optional, Tuple
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

AWS_OPENSEARCH_COLLECTION_ARN = os.environ.get("AWS_OPENSEARCH_COLLECTION_ARN")
AWS_KNOWLEDGE_BASE_ROLE_ARN = os.environ.get("AWS_KNOWLEDGE_BASE_ROLE_ARN")
AWS_EMBEDDING_MODEL_ARN = os.environ.get("AWS_EMBEDDING_MODEL_ARN")
AWS_EMBEDDING_MODEL_DIMENSIONS = int(os.environ.get("AWS_EMBEDDING_MODEL_DIMENSIONS", "1024"))
AWS_ENVIRONMENT = os.environ.get("AWS_ENVIRONMENT", "Development")
AWS_PROJECT = os.environ.get("AWS_PROJECT", "rfpai-agents")
AWS_OPENSEARCH_SERVERLESS_COLLECTION_HOST = os.environ.get("AWS_OPENSEARCH_SERVERLESS_COLLECTION_HOST")
AWS_REGION_NAME = os.environ.get("AWS_REGION_NAME")

s3_client = boto3.client("s3")
bedrock_client = boto3.client("bedrock-agent")
bedrock_runtime_client = boto3.client("bedrock-agent-runtime")


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
    connection_class=RequestsHttpConnection
)


def split_s3_uri(s3_uri: str) -> Tuple[str, str, Optional[str]]:
    """
    Takes an S3 URI and returns a tuple containing:
      - the bucket name,
      - the folder path (if any, with a trailing slash if non-empty),
      - the file name (if present, otherwise None).

    Examples:
      "s3://my-bucket/my-folder/my-file.txt" -> ("my-bucket", "my-folder/", "my-file.txt")
      "s3://my-bucket/my-folder/" -> ("my-bucket", "my-folder/", None)
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError("Not an S3 URI")

    # Remove the s3:// prefix
    path = s3_uri[5:]

    # Split into bucket and rest of path
    parts = path.split("/", 1)
    bucket = parts[0]

    if len(parts) == 1:
        # Only bucket provided
        return bucket, "", None

    # Handle the rest of the path
    rest = parts[1]
    if not rest:
        return bucket, "", None

    # Split the rest of the path into components
    path_parts = rest.split("/")

    # If the URI ends with a slash, it's a folder
    if s3_uri.endswith("/"):
        folder = rest
        if not folder.endswith("/"):
            folder += "/"
        return bucket, folder, None
    else:
        # Check if the last part has a file extension
        last_part = path_parts[-1]
        if "." in last_part:
            # It's a file
            folder = "/".join(path_parts[:-1])
            if folder:
                folder += "/"
            return bucket, folder, last_part
        else:
            # It's a folder without trailing slash
            folder = rest + "/"
            return bucket, folder, None


def aws_opensearch_create_index(index_name: str, retry: int=5, retry_delay: float=0.5):
    index_body = {
        "settings": {
            "index.knn": True
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": AWS_EMBEDDING_MODEL_DIMENSIONS,
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "space_type": "l2"
                    }
                },
                "content":   { "type": "text" },
                "metadata":  { "type": "keyword" }
            }
        }
    }
    while retry > 0:
        retry -= 1
        try:
            return opensearch_client.indices.create(index=index_name, body=index_body)
        except Exception:
            if retry == 0:
                raise
            else:
                time.sleep(retry_delay)


def aws_opensearch_delete_index(index_name: str, retry: int=5, retry_delay: float=0.5):
    while retry > 0:
        retry -= 1
        try:
            return opensearch_client.indices.delete(index=[index_name])
        except Exception:
            if retry == 0:
                raise
            else:
                time.sleep(retry_delay)


def aws_knowledge_base_create(knowledge_base_name: str, index_name: str, retry: int=20, retry_delay: float=1.0):
    knowledge_base_config = {
        "type": "VECTOR",
        "vectorKnowledgeBaseConfiguration": {
            "embeddingModelArn": AWS_EMBEDDING_MODEL_ARN,
            "embeddingModelConfiguration": {
                "bedrockEmbeddingModelConfiguration": {
                    "dimensions": AWS_EMBEDDING_MODEL_DIMENSIONS,
                    "embeddingDataType": "FLOAT32"
                }
            }
        }
    }

    storage_config = {
        "opensearchServerlessConfiguration": {
            "collectionArn": AWS_OPENSEARCH_COLLECTION_ARN,
            "fieldMapping": {
                "metadataField": "metadata",
                "textField": "content",
                "vectorField": "embedding"
            },
            "vectorIndexName": index_name
        },
        "type": "OPENSEARCH_SERVERLESS"
    }

    # The default retry count is high because it can take some time for the
    # opensearch index to be ready to be used by the knowledge base.
    while retry > 0:
        retry -= 1
        try:
            return bedrock_client.create_knowledge_base(
                name=knowledge_base_name,
                roleArn=AWS_KNOWLEDGE_BASE_ROLE_ARN,
                knowledgeBaseConfiguration=knowledge_base_config,
                storageConfiguration=storage_config,
                tags={
                    "AWS_ENVIRONMENT": AWS_ENVIRONMENT,
                    "AWS_PROJECT": AWS_PROJECT
                }
            )
        except Exception:
            if retry == 0:
                raise
            else:
                print("Retrying...")
                time.sleep(retry_delay)


def aws_knowledge_base_delete(knowledge_base_id: str, retry: int=5, retry_delay: float=0.5):
    while retry > 0:
        retry -= 1
        try:
            return bedrock_client.delete_knowledge_base(
                knowledgeBaseId=knowledge_base_id
            )
        except Exception:
            if retry == 0:
                raise
            else:
                time.sleep(retry_delay)


def aws_data_source_create(data_source_name: str, knowledge_base_id: str, bucket_name: str, folder_path: str, retry: int=5, retry_delay: float=0.5):
    # Ensure folder_path is not empty and properly formatted
    if not folder_path:
        folder_path = ""  # Use empty string for root folder
    elif not folder_path.endswith("/"):
        folder_path = f"{folder_path}/"

    data_source_config = {
        "type": "S3",
        "s3Configuration": {
            "bucketArn": f"arn:aws:s3:::{bucket_name}",
            "inclusionPrefixes": [folder_path]  # Use just the folder path
        }
    }

    vector_ingestion_config = {
        "chunkingConfiguration": {
            "chunkingStrategy": "FIXED_SIZE",
            "fixedSizeChunkingConfiguration": {
                "maxTokens": 512,
                "overlapPercentage": 10
            }
        }
    }

    while retry > 0:
        retry -= 1
        try:
            response = bedrock_client.create_data_source(
                name=data_source_name,
                knowledgeBaseId=knowledge_base_id,
                dataSourceConfiguration=data_source_config,
                vectorIngestionConfiguration=vector_ingestion_config
            )
            return response
        except Exception as e:
            if retry == 0:
                raise
            else:
                time.sleep(retry_delay)


def aws_s3_create_json(bucket_name: str, s3_key: str, data: dict, retry: int=5, retry_delay: float=0.5):
    while retry > 0:
        retry -= 1
        try:
            return s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=json.dumps(data)
            )
        except Exception:
            if retry == 0:
                raise
            else:
                time.sleep(retry_delay)


def aws_s3_get_json(bucket_name: str, s3_key: str, retry: int=5, retry_delay: float=0.5) -> dict:
    while retry > 0:
        retry -= 1
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            data = json.loads(response["Body"].read().decode("utf-8"))
            return data
        except Exception:
            if retry == 0:
                raise
            else:
                time.sleep(retry_delay)


def aws_s3_create_text(bucket_name: str, s3_key: str, text: str, retry: int=5, retry_delay: float=0.5):
    while retry > 0:
        retry -= 1
        try:
            return s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=text.encode("utf-8"),  # Ensure UTF-8 encoding
                ContentType="text/plain",  # Explicitly set MIME type
            )
        except Exception as e:
            if retry == 0:
                raise
            else:
                print(f"Retrying after error: {e}")
                time.sleep(retry_delay)


def aws_s3_get_text(bucket_name: str, s3_key: str, retry: int=5, retry_delay: float=0.5) -> str:
    while retry > 0:
        retry -= 1
        try:
            response = s3_client.get_object(
                Bucket=bucket_name,
                Key=s3_key
            )
            text = response["Body"].read().decode("utf-8")
            return text
        except Exception:
            if retry == 0:
                raise
            else:
                time.sleep(retry_delay)

def aws_opensearch_list_indices() -> list:
    """
    List all indices in OpenSearch.

    Returns:
        list: List of index names
    """
    try:
        response = opensearch_client.indices.get_alias().keys()
        return list(response)
    except Exception as e:
        print(f"Error listing indices: {str(e)}")
        return []

def aws_s3_delete_object(bucket_name: str, s3_key: str, retry: int=5, retry_delay: float=0.5):
    while retry > 0:
        retry -= 1
        try:
            return s3_client.delete_object(
                Bucket=bucket_name,
                Key=s3_key
            )
        except Exception:
            if retry == 0:
                raise
            else:
                time.sleep(retry_delay)


def aws_knowledge_base_ingest_s3_object(knowledge_base_id: str, data_source_id: str, bucket_name: str, s3_key: str, retry: int=5, retry_delay: float=0.5):
    s3_uri = f"s3://{bucket_name}/{s3_key}"
    print(f"Uploading document to {s3_uri}...")
    while retry > 0:
        retry -= 1
        try:
            return bedrock_client.ingest_knowledge_base_documents(
                dataSourceId=data_source_id,
                documents=[
                    {
                        "content": {
                            "dataSourceType": "S3",
                            "s3": {"s3Location": {"uri": s3_uri}},
                        }
                    }
                ],
                knowledgeBaseId=knowledge_base_id,
            )
        except Exception:
            if retry == 0:
                raise
            else:
                time.sleep(retry_delay)


def get_knowledge_base_documents(
    knowledge_base_id: str,
    data_source_id: str,
    bucket_name: str,
    s3_key: str,
    retry: int = 5,
    retry_delay: float = 0.5,
):
    s3_uri = f"s3://{bucket_name}/{s3_key}"
    while retry > 0:
        retry -= 1
        try:
            return bedrock_client.get_knowledge_base_documents(
                dataSourceId=data_source_id,
                documentIdentifiers=[
                    {
                        "dataSourceType": "S3",
                        "s3": {"uri": s3_uri},
                    },
                ],
                knowledgeBaseId=knowledge_base_id,
            )
        except Exception:
            if retry == 0:
                raise
            else:
                time.sleep(retry_delay)


def aws_knowledge_base_delete_document(knowledge_base_id: str, data_source_id: str, bucket_name: str, s3_key: str, retry: int=5, retry_delay: float=0.5):
    s3_uri = f"s3://{bucket_name}/{s3_key}"
    while retry > 0:
        retry -= 1
        try:
            return bedrock_client.delete_knowledge_base_documents(
                knowledgeBaseId=knowledge_base_id,
                dataSourceId=data_source_id,
                documentIdentifiers=[
                    {
                        "dataSourceType": "S3",
                        "s3": {
                            "uri": s3_uri
                        }
                    }
                ]
            )
        except Exception:
            if retry == 0:
                raise
            else:
                time.sleep(retry_delay)


def aws_knowledge_base_search(knowledge_base_id: str, query: str, limit: int, retry: int=5, retry_delay: float=0.5):
    while retry > 0:
        retry -= 1
        try:
            return bedrock_runtime_client.retrieve(
                knowledgeBaseId=knowledge_base_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {
                        "numberOfResults": limit
                    }
                }
            )
        except Exception:
            if retry == 0:
                raise
            else:
                time.sleep(retry_delay)


if __name__ == "__main__":
    # print(aws_opensearch_create_index("test-index"))

    # tries = 5
    # while tries > 0:
    #     tries -= 1
    #     try:
    #         print(aws_knowledge_base_create("test-knowledge-base", "test-index"))
    #         break
    #     except Exception as e:
    #         time.sleep(5)
    #         pass
    # aws_data_source_create("test-data-source", "ESWPGVXKVL", "lc-rfp-input", "s3://lc-rfp-input/rfp/test")
    # print(split_s3_uri("s3://lc-rfp-input/rfp/test-kb/"))
    pass
