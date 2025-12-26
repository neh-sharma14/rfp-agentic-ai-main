import os
from typing import Optional
from .text_extract import extract_text
from .aws import (
    split_s3_uri,
    aws_opensearch_create_index,
    aws_knowledge_base_create,
    aws_knowledge_base_delete,
    aws_data_source_create,
    aws_s3_create_json,
    aws_opensearch_delete_index,
    aws_s3_get_json,
    aws_s3_create_text,
    aws_s3_get_text,
    aws_s3_delete_object,
    aws_knowledge_base_ingest_s3_object,
    aws_knowledge_base_delete_document,
    aws_knowledge_base_search
)


AWS_S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME")
AWS_KB_PATH_URI = os.environ.get("AWS_KB_PATH_URI")


def make_name_from_uri(s3_uri: str) -> str:
    if not s3_uri.startswith("s3://"):
        raise ValueError("Not an S3 URI")
    return s3_uri[5:].lower().strip("/").replace("/", "-")


def create_knowledge_base(folder_s3_uri: str) -> dict:
    """
    Creates a knowledge base and an associated data source in AWS Bedrock for all
    documents located in a given S3 folder. It returns a dictionary containing:
      - knowledge_base_id
      - data_source_id
      - bucket_name
      - folder_path

    An index.json file with this mapping is uploaded to the same S3 folder.
    """
    search_index_id = None
    knowledge_base_id = None
    data_source_id = None

    bucket_name, folder_path, _ = split_s3_uri(folder_s3_uri)

    # Ensure folder_path ends with a slash
    if folder_path and not folder_path.endswith("/"):
        folder_path = f"{folder_path}/"

    # Generate names based on the folder path.
    postfix = make_name_from_uri(folder_s3_uri)
    kb_name = f"kb-{postfix}"
    index_name = f"index-{postfix}"
    ds_name = f"ds-{postfix}"

    try:
        search_index_response = aws_opensearch_create_index(index_name)
        search_index_id = search_index_response["index"]

        # Create the knowledge base.
        kb_response = aws_knowledge_base_create(kb_name, index_name)
        knowledge_base_id = kb_response["knowledgeBase"]["knowledgeBaseId"]

        # Create a data source for the specified S3 folder.
        ds_response = aws_data_source_create(ds_name, knowledge_base_id, bucket_name, folder_path)
        data_source_id = ds_response["dataSource"]["dataSourceId"]

        # Create and save index.json with the resource identifiers.
        index_data = {
            "knowledge_base_id": knowledge_base_id,
            "data_source_id": data_source_id,
            "index_name": index_name,
            "bucket_name": bucket_name,
            "folder_path": folder_path
        }
        index_key = os.path.join(folder_path, "index.json") if folder_path else "index.json"
        s3_response = aws_s3_create_json(AWS_S3_BUCKET_NAME, index_key, index_data)

        print(f"Knowledge base and data source created. IDs: {knowledge_base_id}, {data_source_id}")
        return index_data
    except Exception:
        if knowledge_base_id is not None:
            aws_knowledge_base_delete(knowledge_base_id)

        if search_index_id is not None:
            aws_opensearch_delete_index(index_name)

        raise


def get_index_json(folder_s3_uri: str) -> Optional[dict]:
    """
    Retrieves the index.json file from the S3 folder to get stored
    knowledge base and data source IDs.
    Returns the parsed JSON as a dictionary or None if not found.
    """
    bucket_name, folder_path, _ = split_s3_uri(folder_s3_uri)
    index_key = os.path.join(folder_path, "index.json") if folder_path else "index.json"
    try:
        print(bucket_name, index_key)
        return aws_s3_get_json(AWS_S3_BUCKET_NAME, index_key)
    except Exception as e:
        print(f"Error retrieving index.json: {e}")
        return None


def add_document(folder_s3_uri: str, file) -> None:
    """
    Extracts text from the provided file-like object, uploads the text to S3,
    and triggers knowledge base ingestion (placeholder for the actual ingestion API call).

    :param folder_s3_uri: The S3 folder where documents are stored.
    :param file: A file-like object that has a 'filename' attribute.
    """
    text = extract_text(file)
    if not text.strip():
        raise Exception("No text extracted from file.")

    base_name = os.path.splitext(file.filename)[0]
    txt_filename = base_name + ".txt"

    if not folder_s3_uri.endswith("/"):
        folder_s3_uri += "/"

    file_s3_uri = folder_s3_uri + txt_filename
    add_text(file_s3_uri, text)

def add_text(file_s3_uri: str, text: str) -> None:
    """
    Uploads text to S3 at the specified S3 URI and ingests
    the document into the knowledge base via AWS Bedrock.

    :param file_s3_uri: The full S3 URI where the document text will be stored.
                        e.g., "s3://my-bucket/my-folder/my-document.txt"
    :param text: The text content to upload and ingest.
    """
    # Upload the text to S3.
    bucket_name, folder_path, filename = split_s3_uri(file_s3_uri)
    if filename is None:
        raise ValueError("File name must be provided in the S3 URI")

    key = os.path.join(folder_path, filename)
    text_creation_res=aws_s3_create_text(AWS_S3_BUCKET_NAME, key, text)
    print(f"Text uploaded to S3: {text_creation_res}")
    print(f"Uploaded document '{filename}' to s3://{bucket_name}/{key}.")


    index_data = get_index_json(AWS_KB_PATH_URI)
    if not index_data:
        raise Exception("Index file not found in knowledge base folder; cannot ingest document.")

    knowledge_base_id = index_data["knowledge_base_id"]
    data_source_id = index_data["data_source_id"]
    s3_key = os.path.join(folder_path, filename) if folder_path else filename

    ingest_response = aws_knowledge_base_ingest_s3_object(knowledge_base_id, data_source_id, AWS_S3_BUCKET_NAME, s3_key)

    print(f"Document ingestion response: {ingest_response}")


def delete_document(file_s3_uri: str) -> None:
    """
    Deletes a document from the knowledge base.

    :param file_s3_uri: An S3 URI of the file to delete.
    """
    bucket_name, folder_path, filename = split_s3_uri(file_s3_uri)
    if filename is None:
        raise ValueError("File name must be provided in the S3 URI")

    index_data = get_index_json(file_s3_uri)
    if not index_data:
        raise Exception("Index file not found; cannot delete document.")

    knowledge_base_id = index_data["knowledge_base_id"]
    data_source_id = index_data["data_source_id"]
    s3_key = os.path.join(folder_path, filename) if folder_path else filename
    kb_response = aws_knowledge_base_delete_document(knowledge_base_id, data_source_id, AWS_S3_BUCKET_NAME, s3_key)
    s3_response = aws_s3_delete_object(AWS_S3_BUCKET_NAME, s3_key)
    print(f"Document {file_s3_uri} deletion response: {kb_response}")


def get_document_text(file_s3_uri: str) -> str:
    """
    Retrieves the text content of a document from its S3 URI.

    :param file_s3_uri: Full S3 URI where the document is stored.
    :return: The text content of the document.
    """
    bucket_name, folder_path, filename = split_s3_uri(file_s3_uri)
    if filename is None:
        raise ValueError("File name must be provided in the S3 URI")

    s3_key = os.path.join(folder_path, filename) if folder_path else filename
    return aws_s3_get_text(AWS_S3_BUCKET_NAME, s3_key)


def search(folder_s3_uri: str, query: str, limit: int = 5) -> dict:
    """
    Performs a vector search on the knowledge base using the given query.
    Retrieves the knowledge base ID from the index.json file stored in the S3 folder.

    :param folder_s3_uri: The S3 folder URI where index.json is located.
    :param query: The search query text.
    :param limit: Number of search results to return (default is 5).
    :return: The response from the Bedrock runtime client.
    """
    index_data = get_index_json(folder_s3_uri)
    if not index_data:
        raise Exception("Index file not found; cannot perform search.")

    knowledge_base_id = index_data["knowledge_base_id"]
    return aws_knowledge_base_search(knowledge_base_id, query, limit)


def delete_knowledge_base(folder_s3_uri: str) -> None:
    """
    Deletes a knowledge base and its associated resources (data source and OpenSearch index).

    :param folder_s3_uri: The S3 folder URI where the knowledge base is located.
    """
    index_data = get_index_json(folder_s3_uri)
    if not index_data:
        raise Exception("Index file not found; cannot delete knowledge base.")

    knowledge_base_id = index_data["knowledge_base_id"]
    index_name = index_data["index_name"]

    try:
        # Delete the knowledge base
        aws_knowledge_base_delete(knowledge_base_id)

        # Delete the OpenSearch index
        aws_opensearch_delete_index(index_name)

        # Delete the index.json file
        bucket_name, folder_path, _ = split_s3_uri(folder_s3_uri)
        index_key = os.path.join(folder_path, "index.json") if folder_path else "index.json"
        aws_s3_delete_object(bucket_name, index_key)
        print(f"Deleted index.json from s3://{bucket_name}/{index_key}")

    except Exception as e:
        print(f"Error deleting knowledge base: {e}")
        raise
