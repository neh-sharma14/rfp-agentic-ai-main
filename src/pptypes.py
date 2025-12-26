from typing import Tuple, Union, Any, List
from pydantic import BaseModel


# Shared “base” models
class PPSubsectionBase(BaseModel):
    subsection_id: float
    subsection_title: str
    requirement: str

class PPSectionBase(BaseModel):
    section_id: float
    section_title: str

class PPOutlineBase(BaseModel):
    outline_title: str


# Initial Outline models reuse the base classes
class PPInitialOutlineSubsection(PPSubsectionBase):
    pass

class PPInitialOutlineSection(PPSectionBase):
    subsections: List[PPInitialOutlineSubsection]

class PPInitialTechnicalApproachOutline(PPOutlineBase):
    sections: List[PPInitialOutlineSection]


# Final Outline models reuse the base classes + add fields
class PPOutlineSubsection(PPSubsectionBase):
    context: str

class PPOutlineSection(PPSectionBase):
    subsections: List[PPOutlineSubsection]

class PPTechnicalApproachOutline(PPOutlineBase):
    sections: List[PPOutlineSection]