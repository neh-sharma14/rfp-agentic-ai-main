import os
import uuid
from dotenv import load_dotenv

# Disable CrewAI telemetry to avoid connection timeout errors
os.environ['OTEL_SDK_DISABLED'] = 'true'

from src.Common.s3_downloader import download_s3_files_new
from src.Common.vector_uploader import upload_vectors_to_opensearch
from src.compliance_matrix_crew.crew import kickoff_compliance_matrix_crew
load_dotenv()

import asyncio
from fastapi import FastAPI, UploadFile, HTTPException
from .compliance_crew.crew import kickoff_compliance_requirements_crew
from .proposal_outline_crew.crew import kickoff_proposal_outline_crew
from .refine_proposal_outline_crew.crew import kickoff_refine_proposal_outline_crew
from .proposal_outline_editor_crew.crew import kickoff_revise_proposal_outline_crew
from .pastperformance_crew.crew import kickoff_past_performance_crew
from .generic_proposal_content_crew.crew import kickoff_generic_proposal_content_crew
from .utils import download_s3_files, download_s3_files_as_text, prepare_vectordb_from_s3
from .models import CreateKnowledgeBaseRequest
from .knowledge_base.knowledge_base import create_knowledge_base, add_document, delete_document, delete_knowledge_base
from .types import (
    ComplianceMatrixRequest,
    ProposalOutlineRequest,
    RegenerateOutlineRequest,
    ReviseOutlineSubsectionRequest,
    GenericProposalContentRequest,
    PastPerformanceRequest,
    ComplianceRequirementsCrewRequest,
)
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
load_dotenv()
from .utils import download_s3_files, download_s3_files_as_text
from .knowledge_base.aws import (
    aws_opensearch_list_indices,
    aws_opensearch_delete_index
)
from .knowledge_base.vector_store import (
    KnowledgeBase,
)
import time
import json
from typing import Optional, Dict, Any

app = FastAPI(root_path="/rfpaiagents",debug=True)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAIN_INDEX_NAME = os.environ.get("OPENSEARCH_MAIN_INDEX_NAME", "rfp_knowledge_base_test")

@app.get("/")
async def index():
    return "index"

@app.post("/knowledge-base")
async def post_knowledge_base(request: CreateKnowledgeBaseRequest):
    loop  = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: create_knowledge_base(request.folder_s3_uri))
    return result

@app.delete("/knowledge-base/{folder_s3_uri:path}")
async def delete_knowledge_base_route(folder_s3_uri: str):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: delete_knowledge_base(folder_s3_uri))
    return result

@app.post("/knowledge-base/document/{folder_s3_uri:path}")
async def post_knowledge_base_document(folder_s3_uri: str, file: UploadFile):
    loop  = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: add_document(folder_s3_uri, file))
    return result

@app.delete("/knowledge-base/document/{file_s3_uri:path}")
async def delete_knowledge_base_document(file_s3_uri: str):
    loop  = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: delete_document(file_s3_uri))
    return result

@app.post("/compliance_requirements")
async def compliance_requirements(request: ComplianceRequirementsCrewRequest):
    try:
        inputs = {
            "solicitation_id": request.solicitation_id,
            "rfp_title": request.rfp_title,
            "index_name": MAIN_INDEX_NAME
        }

        # Kickoff crew using vector store
        result = await kickoff_compliance_requirements_crew(inputs=inputs)

        try:
            # If result.raw is already a dict, use it directly
            if isinstance(result.raw, dict):
                response = result.raw
            else:
                # Try to parse the string as JSON
                response = json.loads(result.raw)
            response["processing_metadata"] = {
                "solicitation_id": request.solicitation_id,
                "rfp_title": request.rfp_title
            }
            print(response)
            return response
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON response format: {e}"
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"\nError in compliance_requirements endpoint:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while running the crew: {str(e)}"
        )

@app.post("/proposal_outline")
async def proposal_outline(request: ProposalOutlineRequest):
    try:
        bucket = os.environ.get("RFP_S3_BUCKET")
        os.environ["AWS_S3_BUCKET_NAME"] = bucket
        local_knowledge_path = f"./knowledge_extract_text/{request.solicitation_id}"

        if  not request.old_unique_id:
            processed_files = download_s3_files_new(bucket_name=bucket,
                                             s3_prefix=f"rfp/{request.solicitation_id}",
                                             local_dir=local_knowledge_path)

            if not processed_files:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to process any PDF files"
                )
        vector_unique_id = ''



        if not request.old_unique_id:
            vector_unique_id = str(uuid.uuid4())
            upload_vectors_to_opensearch(files=processed_files,solicitation_id=request.solicitation_id,unique_id=vector_unique_id)
        else:
            vector_unique_id = request.old_unique_id
            print(f"Using existing unique_id: {vector_unique_id} for solicitation {request.solicitation_id}")
        inputs = {
            "solicitation_id": request.solicitation_id,
            "rfp_title": request.rfp_title,

            "index_name": MAIN_INDEX_NAME,
            "unique_id": vector_unique_id,
        }

        print(f"\nStarting crew with inputs: {inputs}")
        result = await kickoff_proposal_outline_crew(
            inputs=inputs
        )

        try:

            response = {
                "sections": result["sections"],
                "outline_title": request.rfp_title,
                "unique_id": vector_unique_id,
            }
            print(response)
            return response
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON response format: {e}"
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"\nError in proposal_outline endpoint:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while running the crew: {str(e)}"
        )

@app.post("/refine_proposal_outline")
async def refine_proposal_outline(request: RegenerateOutlineRequest):
    try:
        bucket = os.environ.get("RFP_S3_BUCKET")
        os.environ["AWS_S3_BUCKET_NAME"] = bucket

        processed_files, errors, local_knowledge_path = prepare_vectordb_from_s3(
            bucket=bucket,
            solicitation_id=request.solicitation_id,
            rfp_title=request.rfp_title
        )

        if not processed_files:
            raise HTTPException(
                status_code=500,
                detail="Failed to process any PDF files"
            )

        # Read the raw RFP text from the processed files
        raw_rfp_text = ""
        for file_path in processed_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_rfp_text += f.read() + "\n\n"
            except Exception as e:
                print(f"Warning: Could not read file {file_path}: {str(e)}")

        inputs = {
            "solicitation_id": request.solicitation_id,
            "rfp_title": request.rfp_title,
            "existing_outline": request.existing_outline.model_dump(),
            "refinement_prompt": request.refinement_prompt,
            "processed_files": processed_files,
            "index_name": MAIN_INDEX_NAME,
            "raw_rfp_text": raw_rfp_text
        }

        print(f"\nStarting refine proposal outline crew with inputs: {inputs}")
        result = await kickoff_refine_proposal_outline_crew(
            knowledge_dir=local_knowledge_path,
            inputs=inputs
        )

        try:
            raw_response = result.raw if isinstance(result.raw, str) else str(result.raw)
            last_brace = raw_response.rstrip().rfind('}')
            if last_brace != -1:
                json_str = raw_response[:last_brace + 1]
                response = json.loads(json_str)
            else:
                response = json.loads(raw_response)

            response["processing_metadata"] = {
                "solicitation_id": request.solicitation_id,
                "rfp_title": request.rfp_title,
                "errors": errors,
            }
            print(response)
            return response
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON response format: {e}"
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"\nError in refine_proposal_outline endpoint:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while running the crew: {str(e)}"
        )

@app.post("/generic_proposal_content")
async def generic_proposal_content(request: GenericProposalContentRequest):
    try:
        bucket = os.environ.get("RFP_S3_BUCKET")
        os.environ["AWS_S3_BUCKET_NAME"] = bucket

        processed_files, errors, local_knowledge_path = prepare_vectordb_from_s3(
            bucket=bucket,
            solicitation_id=request.solicitation_id,
            rfp_title=getattr(request, 'rfp_title', None)
        )

        if not processed_files:
            raise HTTPException(
                status_code=500,
                detail="Failed to process any PDF files"
            )

        inputs = {
            "solicitation_id": request.solicitation_id,
            "section": request.section,
            "subsection": request.subsection,
            "requirement": request.requirement,
            "section_purpose": request.section_purpose,
            "instructions_to_writer": request.instructions_to_writer,
            "source_mapping": request.source_mapping,
            "win_theme_alignment": request.win_theme_alignment,
            "refinement_prompt": request.refinement_prompt or "",
            "processed_files": processed_files,
            "index_name": MAIN_INDEX_NAME
        }

        print(f"\nStarting generic proposal content crew with inputs: {inputs}")
        result = await kickoff_generic_proposal_content_crew(
            knowledge_dir=local_knowledge_path,
            inputs=inputs
        )

        response = result.json_dict
        response["processing_metadata"] = {
            "solicitation_id": request.solicitation_id,
            "errors": errors,
        }
        print(response)
        return response
    except Exception as e:
        print(f"\nError in generic_proposal_content endpoint:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while running the crew: {str(e)}"
        )

@app.post("/revise_proposal_outline_section")
async def revise_proposal_outline_section(request: ReviseOutlineSubsectionRequest):

    inputs = {
        "user_prompt": request.user_prompt,
        "outline_section": request.outline_section.model_dump(),
        "proposal_outline": request.proposal_outline.model_dump()
    }

    # Download RFP package files from S3 to local knowledge dir
    local_knowledge_path = f"./knowledge_extract_text/{request.solicitation_id}"
    try:
        # S3 prefix is the solicitation id provided
        download_s3_files_as_text(
            bucket=os.environ.get("RFP_S3_BUCKET", "lc-rfp-input"),
            prefix=f"rfp/{request.solicitation_id}",
            solicitation_id=request.solicitation_id

        )
        # Kickoff crew using new knowledge dir
        result = await kickoff_revise_proposal_outline_crew(
            knowledge_dir=local_knowledge_path,
            inputs=inputs
        )
        # Get json output from crew result
        response = result.json_dict
        print(response)
        return response
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

@app.post("/pastperformance")
async def pastperformance(request: PastPerformanceRequest):
    try:
        result = await kickoff_past_performance_crew(request.model_dump())
        # Create response with input JSON and generated content
        response = request.model_dump()
        response["requirement"] = result.raw  # Assign generated content to requirement
        response["context"] = ""  # Empty context field
        return response
    except TimeoutError as te:
        raise HTTPException(status_code=504, detail=f"Request timed out: {str(te)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while running the crew: {str(e)}")

@app.post("/compliance-matrix")
async def compliance_matrix(request: ComplianceMatrixRequest):
    try:
        result = await kickoff_compliance_matrix_crew(inputs=request.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while running the crew: {str(e)}")

# Knowledge Base Routes
@app.get("/llama_knowledge_base/indices")
async def list_all_indices():
    """List all available knowledge base indices"""
    try:
        indices = aws_opensearch_list_indices()
        return {
            "indices": indices,
            "count": len(indices)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/llama_knowledge_base/indices/{index_name}")
async def delete_index(index_name: str):
    """Delete a specific index"""
    try:
        # Safety check - prevent deletion of main index
        if index_name == MAIN_INDEX_NAME:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete the main index {MAIN_INDEX_NAME}"
            )

        # Delete the index
        aws_opensearch_delete_index(index_name)
        return {
            "message": f"Index {index_name} deleted successfully",
            "index_name": index_name
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/llama_knowledge_base/stats")
async def get_llama_knowledge_base_stats():
    """Get statistics about the knowledge base"""
    try:
        kb = KnowledgeBase()
        stats = kb.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/llama_knowledge_base/search")
async def search_llama_knowledge_base(
    query: str,
    user_id: Optional[str] = None,
    solicitation_id: Optional[str] = None,
    limit: int = 5
):
    """Search the knowledge base with optional filters"""
    try:
        kb = KnowledgeBase()
        results = kb.search(
            query=query,
            user_id=user_id,
            solicitation_id=solicitation_id,
            limit=limit
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/llama_knowledge_base/documents")
async def list_documents(
    solicitation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List all documents in the knowledge base with optional filters"""
    try:
        kb = KnowledgeBase()

        # Build filters dictionary
        filters = {}
        if solicitation_id:
            filters['solicitation_id'] = solicitation_id
        if user_id:
            filters['user_id'] = user_id

        # Get documents using the new method
        result = kb.list_documents(filters=filters, limit=limit, offset=offset)

        return {
            "status": "success",
            **result  # Include all fields from the result
        }

    except Exception as e:
        # Handle 404 error gracefully
        if "index_not_found_exception" in str(e).lower() or "notfounderror" in str(e).lower():
            return {
                "status": "success",
                "total": 0,
                "documents": [],
                "limit": limit,
                "offset": offset,
                "message": "No documents found (index does not exist)"
            }
        # For other errors, raise HTTP exception
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llama_knowledge_base/documents")
async def add_document(
    content: str,
    user_id: str,
    solicitation_id: str,
    file_id: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Add or update a document in the knowledge base"""
    try:
        kb = KnowledgeBase()
        if metadata is None:
            metadata = {}
        metadata.update({
            "user_id": user_id,
            "solicitation_id": solicitation_id,
            "file_id": file_id
        })
        doc_id = kb.add_document(content, metadata)
        return {"doc_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/llama_knowledge_base/documents")
async def delete_documents(
    solicitation_id: Optional[str] = None,
    user_id: Optional[str] = None
):
    """Delete documents from the knowledge base based on metadata filters"""
    try:
        kb = KnowledgeBase()

        # Build metadata filters
        metadata_filters = {}
        if solicitation_id:
            metadata_filters['solicitation_id'] = solicitation_id
        if user_id:
            metadata_filters['user_id'] = user_id

        # If no filters provided, delete all documents
        if not metadata_filters:
            deleted_count = kb.delete_all_documents()
        else:
            deleted_count = kb.delete_documents_by_metadata(metadata_filters)

        return {
            "status": "success",
            "message": f"Successfully deleted {deleted_count} documents",
            "deleted_count": deleted_count
        }

    except Exception as e:
        # Handle 404 error gracefully
        raise HTTPException(status_code=500, detail=str(e))



    """
    Test endpoint to compare search performance between original and optimized tools
    """
    try:
        from src.Common.opensearch_query_tool import OpenSearchQueryTool

        import time

        unique_id = request.get("unique_id")
        query = request.get("query", "federal proposal requirements")
        mode = request.get("mode", "hybrid")

        results = {}

        # Test original tool
        start_time = time.time()
        original_tool = OpenSearchQueryTool(unique_id=unique_id, min_score=0.4)
        original_result = original_tool.run(query, mode=mode)
        original_time = time.time() - start_time

        # Test optimized tool
        start_time = time.time()
        optimized_tool = OpenSearchQueryTool(unique_id=unique_id, min_score=0.7)
        optimized_result = optimized_tool.run(query, mode=mode)
        optimized_time = time.time() - start_time

        # Test cached performance (second call)
        start_time = time.time()
        cached_result = optimized_tool.run(query, mode=mode)
        cached_time = time.time() - start_time

        results = {
            "query": query,
            "mode": mode,
            "performance_comparison": {
                "original_tool": {
                    "execution_time": round(original_time, 3),
                    "min_score": 0.4,
                    "result_length": len(original_result)
                },
                "optimized_tool": {
                    "execution_time": round(optimized_time, 3),
                    "min_score": 0.7,
                    "result_length": len(optimized_result),
                    "cache_stats": optimized_tool.get_cache_stats()
                },
                "cached_call": {
                    "execution_time": round(cached_time, 3),
                    "speedup_factor": round(optimized_time / max(cached_time, 0.001), 2)
                }
            },
            "results": {
                "original_tool_output": original_result[:500] + "..." if len(original_result) > 500 else original_result,
                "optimized_tool_output": optimized_result[:500] + "..." if len(optimized_result) > 500 else optimized_result
            }
        }

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search performance test failed: {str(e)}")