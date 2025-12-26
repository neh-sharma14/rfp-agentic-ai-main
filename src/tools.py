import os
from crewai.tools import BaseTool
from typing import Type, Any,Optional
from pydantic import BaseModel, Field
import boto3
import json
from crewai_tools import LlamaIndexTool
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from .knowledge_base.knowledge_base import search
import time

from .knowledge_base.vector_store import KnowledgeBase


class RFPKnowledgeBaseToolInput(BaseModel):
    query: str = Field(..., description="Search query text")

class RFPKnowledgeBaseTool(BaseTool):
    name: str = "RFP Semantic Search Tool"
    description: str = """A tool that uses advanced semantic search to find relevant information from RFP documents.
    It can search for specific requirements, evaluation criteria, proposal instructions, and other RFP-related content.
    The tool uses LlamaIndex's semantic search capabilities to understand the context and meaning of your query,
    not just keyword matching. This makes it particularly good at finding relevant sections even when the exact
    wording doesn't match."""
    args_schema: Type[BaseModel] = RFPKnowledgeBaseToolInput
    index_name: str = Field(..., description="OpenSearch index name")
    # limit: Optional[int] = 7  # Increased from 5 to 20 for more comprehensive results
    solicitation_id: Optional[str] = Field(None, description="Filter by solicitation ID")

    def _run(self, query: str, solicitation_id: Optional[str] = None, user_id: Optional[str] = None , limit :int = 7) -> str:
        try:
            # Use the solicitation_id from the tool if not provided in the query
            if not solicitation_id:
                solicitation_id = self.solicitation_id

            # print(f"\nExecuting semantic search query: {query}")
            # print(f"Using index: {self.index_name}")
            print(f"Result limit: {limit}")
            print(f"Using solicitation_id: {solicitation_id}")

            # Initialize KnowledgeBase
            kb = KnowledgeBase()

            # Build filters
            filters = {}
            if solicitation_id:
                filters['solicitation_id'] = solicitation_id
            if user_id:
                filters['user_id'] = user_id

            # Perform search with retries
            max_retries = 3
            retry_delay = 2  # seconds
            last_error = None

            for attempt in range(max_retries):
                try:
                    print(f"\nSearch attempt {attempt + 1}/{max_retries}")
                    # Use our custom query method
                    results = kb.query_with_filters(
                        query_text=query,
                        filters=filters,
                        top_k=limit
                    )
                    break
                except Exception as e:
                    last_error = e
                    print(f"Search attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
            else:
                raise last_error

            if not results:
                print("No results found")
                return "No relevant information found in the knowledge base."

            # Format the results into a readable response
            print(f"Found {len(results)} results")
            response_text = f"Found {len(results)} relevant sections:\n\n"

            # Group results by document
            doc_results = {}
            for result in results:
                metadata = result['metadata']
                doc_id = f"{metadata.get('solicitation_id', 'unknown')}_{metadata.get('filename', 'unknown')}"
                if doc_id not in doc_results:
                    doc_results[doc_id] = {
                        'filename': metadata.get('filename', 'unknown'),
                        'sections': []
                    }
                doc_results[doc_id]['sections'].append({
                    'content': result['content'],
                    'score': result['score']
                })

            # Format response by document
            for i, (doc_id, doc_info) in enumerate(doc_results.items(), 1):
                response_text += f"Document {i}: {doc_info['filename']}\n"
                response_text += "=" * 50 + "\n"

                # Sort sections by score
                sorted_sections = sorted(doc_info['sections'], key=lambda x: x['score'], reverse=True)

                for j, section in enumerate(sorted_sections, 1):
                    response_text += f"\nSection {j} (Relevance: {section['score']:.2f}):\n"
                    response_text += f"{section['content']}\n"
                    response_text += "-" * 30 + "\n"

                response_text += "\n"

            print("Query completed successfully")
            return response_text

        except Exception as e:
            error_msg = f"Error querying knowledge base: {str(e)}"
            print(f"\n{error_msg}")
            print(f"Error type: {type(e).__name__}")
            return error_msg


# TODO: Fully remove BedrockKnowledgeBaseSearchTool
class BedrockKnowledgeBaseSearchToolInput(BaseModel):
    query: str = Field(..., description="Search query text")


class BedrockKnowledgeBaseSearchTool(BaseTool):
    name: str = "Bedrock knowledge base search tool"
    description: str = "Searches AWS Bedrock knowledge base for query text and gets results"
    args_schema: Type[BaseModel] = BedrockKnowledgeBaseSearchToolInput
    limit: int = 3

    client: Any = boto3.client('bedrock-agent-runtime')
    knowledge_base_id: str

    def _run(self, query: str) -> str:
        response = self.client.retrieve(
            knowledgeBaseId = self.knowledge_base_id,
            retrievalQuery = {"text": query},
            retrievalConfiguration = {
                "vectorSearchConfiguration": {
                    "numberOfResults": self.limit
                }
            }
        )

        return json.dumps(response)


class KnowledgeBaseSearchToolInput(BaseModel):
    query: str = Field(..., description="Search query text")


class KnowledgeBaseSearchTool(BaseTool):
    name: str = "Knowledge base search tool"
    description: str = "Searches knowledge base for query text and gets results."
    args_schema: Type[BaseModel] = KnowledgeBaseSearchToolInput
    limit: int = 3

    client: Any = boto3.client('bedrock-agent-runtime')
    folder_s3_uri: str

    def _run(self, query: str) -> str:
        return json.dumps(search(folder_s3_uri, query, limit))

class LlamaIndexQueryTool:
    """
    Encapsulates the logic for reading knowledge documents, building an index,
    and creating a LlamaIndex query engine tool.
    """

    def __init__(
        self,
        knowledge_dir: str = "./knowledge",
        similarity_top_k: int = 4,
        aws_region: str = "us-east-1",
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        chunk_size: int = 2048,
        model_llm: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        model_embed: str = "amazon.titan-embed-text-v2:0",
    ):
        """
        Initializes the LlamaIndexQueryTool by setting up:
          - The LLM
          - The embedding model
          - The local documents and vectorstore index
          - The query engine
        """
        self.aws_region = aws_region or os.getenv("AWS_REGION_NAME")
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")

        # Set models and chunksize
        self.llm = Bedrock(
            model=model_llm,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region,
        )
        self.embed_model = BedrockEmbedding(
            model_name=model_embed,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region,
        )
        self.transformations = [SentenceSplitter(chunk_size=chunk_size)]

        # Embed documents and create query tool
        self.knowledge_reader = SimpleDirectoryReader(input_dir=knowledge_dir, recursive=True)
        self.knowledge_documents = self.knowledge_reader.load_data()
        self.knowledge_index = VectorStoreIndex.from_documents(
            self.knowledge_documents,
            embed_model=self.embed_model,
            transformations=self.transformations
        )
        self.knowledge_query_engine = self.knowledge_index.as_query_engine(
            llm=self.llm,
            similarity_top_k=similarity_top_k
        )
        self.knowledge_query_tool = LlamaIndexTool.from_query_engine(
            self.knowledge_query_engine,
            name="RFP Query Tool",
            description="Use this tool to look up RFP details",
        )

    def get_tool(self) -> LlamaIndexTool:
        """
        Returns the LlamaIndexTool instance.
        """
        return self.knowledge_query_tool

    def query(self, query_text: str) -> str:
        """
        Uses the query tool to run a query against the vectorstore.
        Returns the response as text.
        """
        try:
            response = self.knowledge_query_tool(query_text)
            if not response:
                raise ValueError("Received empty response from LLM query")
            return str(response)
        except Exception as e:
            # Log the error or raise with more context
            print(f"LLM query failed: {e}")
            raise RuntimeError(f"Query failed: {e}")