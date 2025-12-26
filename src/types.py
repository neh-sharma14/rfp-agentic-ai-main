from typing import  Any, Dict, List ,Optional, Union
from pydantic import BaseModel

# Shared "base" models
class SubsectionBase(BaseModel):
    subsection_id: float
    subsection_title: str
    requirement: str
    section_purpose: str
    instructions_to_writer: str
    source_mapping: List[str]  # List of RFP references
    win_theme_alignment: List[str]  # List of win theme names/IDs

class SectionBase(BaseModel):
    section_id: float
    section_title: str
    section_purpose: str
    instructions_to_writer: str
    source_mapping: List[str]  # List of RFP references
    win_theme_alignment: List[str]  # List of win theme names/IDs

# Initial Outline models reuse the base classes
class InitialOutlineSection(SectionBase):
    subsections: List[SubsectionBase]

class InitialProposalOutline(BaseModel):
    sections: List[InitialOutlineSection]


# Final Outline models reuse the base classes + add fields
class OutlineSubsection(SubsectionBase):
    context: str

class OutlineSection(SectionBase):
    subsections: List[OutlineSubsection]


class ProposalOutline(BaseModel):
    sections: List[OutlineSection]

class OutlineSubsectionContent(BaseModel):
    content: str
    source_mapping: List[str]
    win_theme_alignment: List[str]

class ComplianceRequirement(BaseModel):
    sectionNo: str
    requirement: str

class ComplianceRequirementList(BaseModel):
    requirements: List[ComplianceRequirement]

class SectionAnalysis(BaseModel):
    task_type: str  # One of: technical, management, resume, past_performance, generic
    is_research: bool
    reasoning: str

class ProposalOutlineRequest(BaseModel):
    solicitation_id: str
    rfp_title: str
    old_unique_id: Optional[str] = None

class RegenerateOutlineRequest(BaseModel):
    solicitation_id: str
    rfp_title: str
    refinement_prompt: str
    existing_outline: InitialProposalOutline

class ReviseOutlineSubsectionRequest(BaseModel):
    solicitation_id: str
    user_prompt: str
    outline_section: OutlineSubsection
    proposal_outline: ProposalOutline

class GenericProposalContentRequest(BaseModel):
    solicitation_id: str
    section: str
    subsection: str
    requirement: str
    section_purpose: str
    instructions_to_writer: str
    source_mapping: List[str]
    win_theme_alignment: List[str]
    refinement_prompt: str

class PastPerformanceRequest(BaseModel):
    solicitation_id: str
    section: str
    subsection: str
    requirement: str
    context: str

class ComplianceRequirementsCrewRequest(BaseModel):
    solicitation_id: str
    rfp_title: str

class DeleteDocumentsRequest(BaseModel):
    solicitation_id: Optional[str] = None
    user_id: Optional[str] = None
    doc_ids: Optional[List[str]] = None

class ComplianceMatrixRequest(BaseModel):
    requirements: List[Dict[str, Union[str, float]]]
    outline: List[Dict[str, Any]]