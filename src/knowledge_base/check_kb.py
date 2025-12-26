import os
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from dotenv import load_dotenv

load_dotenv()
from aws import get_knowledge_base_documents

AWS_OPENSEARCH_SERVERLESS_COLLECTION_HOST = os.environ.get(
    "AWS_OPENSEARCH_SERVERLESS_COLLECTION_HOST"
)
AWS_REGION_NAME = os.environ.get("AWS_REGION_NAME")

# Bedrock client for knowledge base management
bedrock_client = boto3.client("bedrock-agent")

# OpenSearch client for low-level index operations
credentials = boto3.Session().get_credentials()
awsauth = AWSV4SignerAuth(credentials, AWS_REGION_NAME, "aoss")

opensearch_client = OpenSearch(
    hosts=[{"host": AWS_OPENSEARCH_SERVERLESS_COLLECTION_HOST, "port": 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
)


def check_opensearch_index_exists(index_name: str) -> bool:
    """
    Check if an OpenSearch index exists.
    """
    try:
        exists = opensearch_client.indices.exists(index=index_name)
        return exists
    except Exception as e:
        print(f"Error checking index existence: {e}")
        return False

def list_indices():
    try:
        indices = opensearch_client.cat.indices(format="json")
        print("Indices in OpenSearch:")
        for idx in indices:
            print(idx["index"])
    except Exception as e:
        print(f"Error listing indices: {e}")

def delete_indices():
    try:
      indices_to_delete = [
      "index-lc-rfp-input-ats-kb",
      "index-rfpaiservice-ats-knowledge-base",
      "index-rfpaiservice-ats-test"
      ]
    except Exception as e:
      print(f"Error listing indices: {e}")

    for index in indices_to_delete:
        try:
            response = opensearch_client.indices.delete(index=index)
            print(f"Deleted index: {index} | Response: {response}")
        except Exception as e:
            print(f"Error deleting index {index}: {e}")

def search_all_docs(index_name: str):
    query = {"query": {"match_all": {}}}
    try:
        response = opensearch_client.search(index=index_name, body=query, size=100)
        total = response["hits"]["total"]["value"]
        print(f"Total documents in index '{index_name}': {total}")
        for hit in response["hits"]["hits"]:
            print(
                f"ID: {hit['_id']}, Source snippet: {str(hit['_source'])[:200]}"
            )  # print snippet
    except Exception as e:
        print(f"Search error: {e}")

def search_text_in_index(index_name: str, text: str):
    """
    Search the OpenSearch index for documents containing the specified text.
    """
    # Refresh index to make sure recent documents are searchable
    # try:
    #     opensearch_client.indices.refresh(index=index_name)
    # except Exception as e:
    #     print(f"Error refreshing index: {e}")

    query = {"query": {"match": {"content": {"query": text, "operator": "and"}}}}
    try:
        response = opensearch_client.search(index=index_name, body=query)
        hits = response["hits"]["hits"]
        total = response["hits"]["total"]
        total_hits = total["value"] if isinstance(total, dict) else total
        print(f"Found {total_hits} documents matching the query.")
        for hit in hits:
            print(f"Document ID: {hit['_id']}")
            print(f"Source: {hit['_source']}")
            print("---")
    except Exception as e:
        print(f"Error searching index: {e}")


if __name__ == "__main__":
    # print(list_indices())
    # print(delete_indices())
    # kb_status = get_knowledge_base_status(kb_id)
    # print(kb_status)

    # index_name = "index-lc-rfp-input-ats-kb"
    # print(search_all_docs(index_name))

    # status_res = get_knowledge_base_documents(
    #     knowledge_base_id="",
    #     data_source_id="",
    #     bucket_name="",
    #     s3_key="",
    # )
    # print(status_res, "status_res")
    # print(f"Does OpenSearch index '{index_name}' exist?")
    # index_exists = check_opensearch_index_exists(index_name)
    # print(index_exists)

    # if index_exists:
    #     # Replace this text with the snippet you want to verify was ingested
    #     text_to_search = (
    #        "SECTION I Letter to Interested GSA MAS 8(a) Small Business Contract Holders To Whom It May Concern: The General Services Administration (GSA), Office of the Chief Information Officer (OCIO), has identified a need for Digital Management Services. As such, this Request for Quotation (RFQ) is being issued to GSA MAS 8(a) small business holders. The RFQ summary is as follows: (1) Statement of Work for Digital Management Services. (2) Task Order Type: Labor Hour (3) Performance Period: 12-month base period and two (2) 12-month option periods. (4) Questions Due by NLT 10:00 AM ET, 24 Aug 2023 (5) Quotations Due By NLT 10:00 AM ET, 04 Sep 2023 "
    #     )
    #     print(f"Searching for text in index '{index_name}': \"{text_to_search}\"")
    #     search_text_in_index(index_name, text_to_search)
    pass