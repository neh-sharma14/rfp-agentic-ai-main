from pydantic import BaseModel
from typing import List, Optional


class ProposalOutlineRequest(BaseModel):
    """Request model for generating a proposal outline."""
    solicitation_id: str
    pdf_files: List[str]
    index_name: Optional[str] = None


class CreateKnowledgeBaseRequest(BaseModel):
    folder_s3_uri: str


class ComplianceRequirementsCrewRequest(BaseModel):
    file_s3_uri: str