from zipfile import Path
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import NL2SQLTool
import os
import boto3
import openai
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from crewai.memory import ShortTermMemory
from src.Common.opensearch_query_tool import OpenSearchQueryTool
import json


def clean_and_parse_json(raw_content):
    """
    Clean and parse JSON content that might have markdown formatting.
    Handles ```json and ``` markdown code blocks.
    """
    if not isinstance(raw_content, str):
        return raw_content

    try:
        # Clean the JSON string if it contains markdown formatting
        cleaned_json = raw_content.strip()

        # Remove markdown code block formatting if present
        if cleaned_json.startswith('```json'):
            # Remove ```json from start and ``` from end
            cleaned_json = cleaned_json[7:]  # Remove '```json\n'
            if cleaned_json.endswith('```'):
                cleaned_json = cleaned_json[:-3]  # Remove '```'
            cleaned_json = cleaned_json.strip()
        elif cleaned_json.startswith('```'):
            # Remove generic ``` formatting
            cleaned_json = cleaned_json[3:]  # Remove '```'
            if cleaned_json.endswith('```'):
                cleaned_json = cleaned_json[:-3]  # Remove '```'
            cleaned_json = cleaned_json.strip()

        # Try to parse as JSON
        return json.loads(cleaned_json)

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Content preview: {raw_content[:500]}...")
        return {"error": "Invalid JSON", "raw_content": raw_content, "parse_error": str(e)}


@CrewBase
class ProposalOutlinePhase1Crew():
    """Phase 1 Crew for RFP Structure Analysis and Shredding"""

    agents_config = 'config/phase1_agents.yaml'
    tasks_config = 'config/phase1_tasks.yaml'

    def __init__(self, inputs: dict = {}):
        """
        Initializes the Phase 1 Crew for RFP structure analysis.
        """
        self.fileName = ''
        self.inputs = inputs
        self.unique_id = inputs.get("unique_id")
        self.rfp_document = inputs.get("rfp_document")
        self.llm = LLM(
            model=os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview'),
            api_key=os.getenv('OPENAI_KEY'),

        )

    @agent
    def rfp_document_analyzer(self) -> Agent:
        tools = []
        if self.unique_id:
            tools = [OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)]
        return Agent(
            config=self.agents_config['rfp_document_analyzer'],
            tools=tools,
            llm=self.llm,
        )

    @agent
    def section_l_m_extractor(self) -> Agent:
        tools = []
        if self.unique_id:
            tools = [OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)]
        return Agent(
            config=self.agents_config['section_l_m_extractor'],
            tools=tools,
            llm=self.llm,
        )

    @agent
    def compliance_structure_validator(self) -> Agent:
        tools = []
        if self.unique_id:
            tools = [OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)]
        return Agent(
            config=self.agents_config['compliance_structure_validator'],
            tools=tools,
            llm=self.llm,
        )

    @task
    def analyze_rfp_documents(self) -> Task:
        tools = []
        if self.unique_id:
            tools = [OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)]
        return Task(
            config=self.tasks_config['analyze_rfp_documents'],
            tools=tools,
            agent=self.rfp_document_analyzer(),
        )

    @task
    def extract_proposal_structure_requirements(self) -> Task:
        tools = []
        if self.unique_id:
            tools = [OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)]
        return Task(
            config=self.tasks_config['extract_proposal_structure_requirements'],
            tools=tools,
            agent=self.section_l_m_extractor(),
        )

    @task
    def validate_and_create_phase1_structure(self) -> Task:
        tools = []
        if self.unique_id:
            tools = [OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)]
        return Task(
            config=self.tasks_config['validate_and_create_phase1_structure'],
            tools=tools,
            agent=self.compliance_structure_validator(),
        )

    def convert_to_flat_list(self, phase1_structure):
        """
        Convert Phase 1 structure to a flat list of sections and subsections.
        Returns the structure as-is if already a list, or extracts from various formats.
        All sections must be discovered through vector search - no default generation.
        """
        try:
            # Parse if string
            if isinstance(phase1_structure, str):
                try:
                    structure = json.loads(phase1_structure)
                except json.JSONDecodeError:
                    print("Warning: Could not parse Phase 1 result as JSON")
                    print("Raw result:", phase1_structure[:500] + "..." if len(phase1_structure) > 500 else phase1_structure)
                    return []
            else:
                structure = phase1_structure

            flat_list = []

            # Handle different structure formats
            if isinstance(structure, dict):
                if 'sections' in structure:
                    # Extract from sections wrapper
                    sections = structure['sections']
                    print(f"Found sections wrapper with {len(sections)} sections")
                    for section in sections:
                        # Add main section
                        section_item = {**section, 'item_type': 'section'}
                        flat_list.append(section_item)

                        # Add subsections
                        subsections = section.get('subsections', [])
                        print(f"Section '{section.get('section_title', 'Unknown')}' has {len(subsections)} subsections")
                        for subsection in subsections:
                            subsection_item = {
                                **subsection,
                                'item_type': 'subsection',
                                'parent_section_id': section.get('section_id', ''),
                                'parent_section_title': section.get('section_title', '')
                            }
                            flat_list.append(subsection_item)

                elif 'volumes' in structure:
                    # Extract from volume structure
                    volumes = structure['volumes']
                    print(f"Found volume structure with {len(volumes)} volumes")
                    for volume in volumes:
                        volume_sections = volume.get('sections', [])
                        print(f"Volume '{volume.get('volume_title', 'Unknown')}' has {len(volume_sections)} sections")
                        for section in volume_sections:
                            # Add main section
                            section_item = {
                                **section,
                                'item_type': 'section',
                                'volume_id': volume.get('volume_id', ''),
                                'volume_title': volume.get('volume_title', '')
                            }
                            flat_list.append(section_item)

                            # Add subsections
                            subsections = section.get('subsections', [])
                            for subsection in subsections:
                                subsection_item = {
                                    **subsection,
                                    'item_type': 'subsection',
                                    'parent_section_id': section.get('section_id', ''),
                                    'parent_section_title': section.get('section_title', ''),
                                    'volume_id': volume.get('volume_id', ''),
                                    'volume_title': volume.get('volume_title', '')
                                }
                                flat_list.append(subsection_item)
                else:
                    # Single section object
                    if 'section_id' in structure or 'section_title' in structure:
                        print("Found single section structure")
                        section_item = {**structure, 'item_type': 'section'}
                        flat_list.append(section_item)

                        # Add subsections
                        subsections = structure.get('subsections', [])
                        print(f"Single section has {len(subsections)} subsections")
                        for subsection in subsections:
                            subsection_item = {
                                **subsection,
                                'item_type': 'subsection',
                                'parent_section_id': structure.get('section_id', ''),
                                'parent_section_title': structure.get('section_title', '')
                            }
                            flat_list.append(subsection_item)
                    else:
                        print("Warning: Unrecognized dict structure format")
                        print("Available keys:", list(structure.keys()))

            elif isinstance(structure, list):
                # Already a list - process each item
                print(f"Found list structure with {len(structure)} items")
                for item in structure:
                    if isinstance(item, dict):
                        # Check if it's a section with subsections
                        if 'subsections' in item:
                            # Add main section
                            section_item = {**item, 'item_type': 'section'}
                            flat_list.append(section_item)

                            # Add subsections
                            subsections = item.get('subsections', [])
                            for subsection in subsections:
                                subsection_item = {
                                    **subsection,
                                    'item_type': 'subsection',
                                    'parent_section_id': item.get('section_id', ''),
                                    'parent_section_title': item.get('section_title', '')
                                }
                                flat_list.append(subsection_item)
                        else:
                            # Assume it's already a flat item
                            item_type = 'subsection' if 'parent_section_id' in item or 'sub_section_id' in item else 'section'
                            flat_list.append({**item, 'item_type': item_type})
            else:
                print(f"Warning: Unrecognized structure type: {type(structure)}")
                return []

            # Validate that we have meaningful content
            sections_count = len([item for item in flat_list if item.get('item_type') == 'section'])
            subsections_count = len([item for item in flat_list if item.get('item_type') == 'subsection'])

            print(f"Converted Phase 1 result to flat list:")
            print(f"  - Total items: {len(flat_list)}")
            print(f"  - Sections: {sections_count}")
            print(f"  - Subsections: {subsections_count}")

            if len(flat_list) == 0:
                print("WARNING: No sections discovered through vector search!")
                print("This may indicate:")
                print("  - RFP documents don't contain clear section requirements")
                print("  - Vector search queries need refinement")
                print("  - Documents may not be properly indexed")

            return flat_list

        except Exception as e:
            print(f"Error converting to flat list: {e}")
            print(f"Structure type: {type(phase1_structure)}")
            print(f"Structure content: {str(phase1_structure)[:500]}...")
            return []

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.rfp_document_analyzer(),
                self.section_l_m_extractor(),
                self.compliance_structure_validator()
            ],
            tasks=[
                self.analyze_rfp_documents(),
                self.extract_proposal_structure_requirements(),
                self.validate_and_create_phase1_structure()
            ],
            process=Process.sequential,
            verbose=True,
            llm=self.llm,
            memory=True,
        )


@CrewBase
class ProposalOutlineCrew():
    """CrewaiJsonProposalOutlineForUsGovernmentRfps crew - Phase 2"""

    agents_config = 'config/phase2_agents.yaml'
    tasks_config = 'config/phase2_tasks.yaml'

    def __init__(self, inputs: dict = {}):
        """
        Initializes the ProposalOutlineCrew with the given inputs.
        Loads agents and tasks configuration from YAML files.
        """
        self.fileName =''
        self.inputs = inputs
        self.unique_id = inputs.get("unique_id")

        self.llm = LLM(
            model=os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview'),
            api_key=os.getenv('OPENAI_KEY'),
        )

    @agent
    def opensearchqueryagent(self) -> Agent:
        return Agent(
            config=self.agents_config['opensearchqueryagent'],
            tools=[OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)],
            llm=self.llm,
        )

    @agent
    def compliancemappingagent(self) -> Agent:
        return Agent(
            config=self.agents_config['compliancemappingagent'],
            tools=[OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)],
            llm=self.llm,
        )

    @agent
    def jsonstructureagent(self) -> Agent:
        return Agent(
            config=self.agents_config['jsonstructureagent'],
            tools=[OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)],
            llm=self.llm,
        )

    @task
    def extract_section_specific_content(self) -> Task:
        return Task(
            config=self.tasks_config['extract_section_specific_content'],
            tools=[OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)],
            agent=self.opensearchqueryagent(),
        )

    @task
    def map_compliance_and_evaluation_criteria(self) -> Task:
        return Task(
            config=self.tasks_config['map_compliance_and_evaluation_criteria'],
            tools=[OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)],
            agent=self.compliancemappingagent(),
        )

    @task
    def generate_detailed_json_outline(self) -> Task:
        return Task(
            config=self.tasks_config['generate_detailed_json_outline'],
            tools=[OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)],
            agent=self.jsonstructureagent(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.opensearchqueryagent(),
                self.compliancemappingagent(),
                self.jsonstructureagent()
            ],
            tasks=[
                self.extract_section_specific_content(),
                self.map_compliance_and_evaluation_criteria(),
                self.generate_detailed_json_outline()
            ],
            process=Process.sequential,
            verbose=True,
            llm=self.llm,
            memory=True,
        )


async def kickoff_phase1_crew(inputs: dict = {}):
    """
    Kicks off Phase 1 crew for RFP structure analysis and shredding.
    Returns parsed JSON output from LLM.
    """
    crew_instance = ProposalOutlinePhase1Crew(inputs=inputs)
    result = await crew_instance.crew().kickoff_async(inputs=inputs)

    # Parse the raw string content if it exists
    if hasattr(result, 'raw') and isinstance(result.raw, str):
        try:
            # Use the existing clean_and_parse_json function to parse the raw string
            parsed_result = clean_and_parse_json(result.raw)
            return parsed_result
        except Exception as e:
            print(f"Error parsing raw result: {e}")
            print(f"Raw content preview: {result.raw[:500]}...")
            return {"error": "Failed to parse JSON", "raw_content": result.raw}

    # Return the result as-is if no raw attribute or not a string
    return result


async def kickoff_phase2_crew_for_section(inputs: dict = {}):
    """
    Kicks off Phase 2 crew for a specific section or subsection.
    """
    crew_instance = ProposalOutlineCrew(inputs=inputs)
    result = await crew_instance.crew().kickoff_async(inputs=inputs)
    return result


def transform_to_desired_format(detailed_outline, outline_title="Proposal Outline"):
    """
    Transform the internal detailed outline format to the desired output format.
    """
    transformed_sections = []

    for section_data in detailed_outline.get("proposal_structure", {}).get("sections", []):
        # Extract section information
        original_section = section_data.get("original_section_data", {})
        section_outline = section_data.get("detailed_outline", {})

        # Build section structure
        section = {
            "section_title": (
                original_section.get("title") or
                original_section.get("section_title") or
                "Unknown Section"
            ),
            "section_id": str(section_data.get("section_id", "").replace("section_", "")),
            "subsections": []
        }

        # Add section-level outline data if available
        if isinstance(section_outline, dict) and not section_outline.get("error"):
            section.update({
                "section_purpose": section_outline.get("section_purpose", ""),
                "instructions_to_writer": section_outline.get("instructions_to_writer", ""),
                "source_mapping": section_outline.get("source_mapping", []),
                "win_theme_alignment": section_outline.get("win_theme_alignment", [])
            })

        # Process subsections
        for subsection_data in section_data.get("subsections", []):
            original_subsection = subsection_data.get("original_subsection_data", {})
            subsection_outline = subsection_data.get("detailed_outline", {})

            subsection = {
                "subsection_id": original_subsection.get("id", subsection_data.get("subsection_id", "").split("_")[-1]),
                "subsection_title": (
                    original_subsection.get("title") or
                    original_subsection.get("subsection_title") or
                    "Unknown Subsection"
                ),
                "requirement": original_subsection.get("requirement", ""),
                "context": original_subsection.get("context"),
                "parentId": original_subsection.get("parentId"),
                "subsections": original_subsection.get("subsections", []),
                "outlineSubSectionId": original_subsection.get("outlineSubSectionId"),
                "content": original_subsection.get("content"),
                "lastUpdatedDate": original_subsection.get("lastUpdatedDate"),
                "generatedContent": original_subsection.get("generatedContent", False)
            }

            # Add subsection-level outline data if available
            if isinstance(subsection_outline, dict) and not subsection_outline.get("error"):
                subsection.update({
                    "section_purpose": subsection_outline.get("section_purpose", ""),
                    "instructions_to_writer": subsection_outline.get("instructions_to_writer", ""),
                    "source_mapping": subsection_outline.get("source_mapping", []),
                    "win_theme_alignment": subsection_outline.get("win_theme_alignment", [])
                })

            section["subsections"].append(subsection)

        transformed_sections.append(section)

    return {
        "outline_title": outline_title,
        "sections": transformed_sections
    }


async def kickoff_proposal_outline_crew(inputs: dict = {}):
    """
    Main orchestration function that runs both phases:
    1. Phase 1: RFP structure analysis and shredding
    2. Phase 2: Generate detailed proposal outline for each section/subsection
    """
    print("Starting Phase 1: RFP Structure Analysis...")

    phase1_result = await kickoff_phase1_crew(inputs)

    # Extract sections from phase1_result.raw if it has the .raw attribute
    if hasattr(phase1_result, 'raw') and isinstance(phase1_result.raw, dict):
        sections_data = phase1_result.raw.get('sections', [])
        processing_metadata = phase1_result.raw.get('processing_metadata', {})
    elif isinstance(phase1_result, dict):
        sections_data = phase1_result.get('sections', [])
        processing_metadata = phase1_result.get('processing_metadata', {})
    else:
        print("Warning: Unexpected phase1_result format")
        return phase1_result

    print(f"Found {len(sections_data)} sections to process")

    # Build the final outline structure
    outline_title = processing_metadata.get('rfp_title', 'Proposal Outline')
    final_sections = []

    # Process each section
    for section_idx, section in enumerate(sections_data):
        print(f"Processing section {section_idx + 1}/{len(sections_data)}: {section.get('title', 'Unknown')}")

        # Process section itself
        section_inputs = {
            **inputs,
            'section_data': section,
            'item_type': 'section',
            'section_title': section.get('title', 'Unknown Section'),
            'subsection_title': '',  # Empty for section-level processing
            'section_context': section.get('page_limits', {}),
            'subsection_context': ''  # Empty for section-level processing
        }

        try:
            section_outline = await kickoff_phase2_crew_for_section(section_inputs)
            print(f"Generated outline for section: {section.get('title', 'Unknown')}")
        except Exception as e:
            print(f"Error generating outline for section {section.get('title', 'Unknown')}: {e}")
            section_outline = {"error": str(e)}

        # Build section structure
        final_section = {
            "section_title": section.get('title', 'Unknown Section'),
            "section_id": str(section_idx + 1),
            "subsections": [],
            "requirement": None,
            "context": None,
            "parentId": None,
            "outlineSectionId": section_idx + 1,
            "content": None,
            "lastUpdatedDate": "2025-08-18T00:00:00.000000Z",
        }

        # Add section-level outline data if available
        if isinstance(section_outline, dict) and not section_outline.get("error"):
            final_section.update({
                "section_purpose": section_outline.get("section_purpose", ""),
                "instructions_to_writer": section_outline.get("instructions_to_writer", ""),
                "source_mapping": section_outline.get("source_mapping", []),
                "win_theme_alignment": section_outline.get("win_theme_alignment", [])
            })

        # Process subsections
        subsections = section.get('subsections', [])
        print(f"Processing {len(subsections)} subsections for section: {section.get('title', 'Unknown')}")

        for subsection_idx, subsection in enumerate(subsections):
            print(f"  Processing subsection {subsection_idx + 1}/{len(subsections)}: {subsection.get('title', 'Unknown')}")

            # Process subsection
            subsection_inputs = {
                **inputs,
                'section_data': subsection,
                'item_type': 'subsection',
                'parent_section': section,
                'section_title': section.get('title', 'Unknown Section'),
                'subsection_title': subsection.get('title', 'Unknown Subsection'),
                'section_context': section.get('page_limits', {}),
                'subsection_context': subsection.get('page_limits', {})
            }

            try:
                subsection_outline = await kickoff_phase2_crew_for_section(subsection_inputs)
                print(f"  Generated outline for subsection: {subsection.get('title', 'Unknown')}")
            except Exception as e:
                print(f"  Error generating outline for subsection {subsection.get('title', 'Unknown')}: {e}")
                subsection_outline = {"error": str(e)}

            # Build subsection structure
            final_subsection = {
                "subsection_id": f"{section_idx + 1}.{subsection_idx + 1}",
                "subsection_title": subsection.get('title', 'Unknown Subsection'),
                "requirement": f"Detailed requirements for {subsection.get('title', 'this subsection')}.",
                "context": None,
                "parentId": section_idx + 1,
                "subsections": [],
                "outlineSubSectionId": int(f"{section_idx + 1}{subsection_idx + 1:02d}"),
                "content": None,
                "lastUpdatedDate": "2025-08-18T00:00:00.000000Z",
                "generatedContent": False
            }

            # Add subsection-level outline data if available
            if isinstance(subsection_outline, dict) and not subsection_outline.get("error"):
                final_subsection.update({
                    "section_purpose": subsection_outline.get("section_purpose", ""),
                    "instructions_to_writer": subsection_outline.get("instructions_to_writer", ""),
                    "source_mapping": subsection_outline.get("source_mapping", []),
                    "win_theme_alignment": subsection_outline.get("win_theme_alignment", [])
                })

            final_section["subsections"].append(final_subsection)

        final_sections.append(final_section)

    # Build final result
    final_result = {
        "outline_title": outline_title,
        "sections": final_sections,
        "unique_id": inputs.get("unique_id")
    }

    print(f"Generated complete proposal outline with {len(final_sections)} sections")
    return final_result



# Separate function for Phase 1 only (if needed independently)
async def get_rfp_structure_only(inputs: dict = {}):
    """
    Run only Phase 1 to get RFP structure without generating detailed outlines.
    """
    return await kickoff_phase1_crew(inputs)


# Separate function for Phase 2 only (if needed independently)
async def generate_outline_for_single_section(inputs: dict = {}):
    """
    Run only Phase 2 for a single section/subsection.
    """
    return await kickoff_phase2_crew_for_section(inputs)
