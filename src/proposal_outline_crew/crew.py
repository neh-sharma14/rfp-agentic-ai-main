from zipfile import Path
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import NL2SQLTool
import os
import boto3
import openai
from pathlib import Path as FilePath
import time
import asyncio
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from crewai.memory import ShortTermMemory
from src.Common.opensearch_query_tool import OpenSearchQueryTool
import json

"""
Proposal Outline Crew - Federal RFP Processing with LLM Caching

This module includes comprehensive LLM caching at multiple levels:
1. LLM-level caching: Enabled on all LLM instances for API cost reduction
2. Crew-level caching: Enabled on all crew instances for better performance
3. Memory persistence: Maintains context across crew executions

Caching benefits:
- Reduces OpenAI API costs by reusing similar responses
- Improves response times for repeated or similar queries
- Maintains consistency in proposal outline generation
- Optimizes performance for large federal RFP processing
"""


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


def save_phase1_result_to_file(phase1_result, unique_id: str = None):
    """
    Save Phase 1 result to a JSON file in the crew folder.
    Always overwrites if file exists.

    Args:
        phase1_result: The result from Phase 1 crew
        unique_id: Optional unique identifier for the file name
    """
    try:
        # Get the current file's directory (crew folder)
        current_dir = FilePath(__file__).parent

        # Create filename with unique_id if provided
        if unique_id:
            filename = f"phase1_result_{unique_id}.json"
        else:
            filename = "phase1_result.json"

        file_path = current_dir / filename

        # Prepare the data to save
        if hasattr(phase1_result, 'raw'):
            # If result has raw attribute, save the raw content
            if isinstance(phase1_result.raw, dict):
                data_to_save = phase1_result.raw
            elif isinstance(phase1_result.raw, str):
                try:
                    data_to_save = json.loads(phase1_result.raw)
                except json.JSONDecodeError:
                    data_to_save = {"raw_content": phase1_result.raw}
            else:
                data_to_save = {"raw_content": str(phase1_result.raw)}
        elif isinstance(phase1_result, dict):
            data_to_save = phase1_result
        else:
            data_to_save = {"result": str(phase1_result)}

        # Add metadata
        data_to_save["metadata"] = {
            "timestamp": int(time.time() * 1000),  # Unix timestamp in milliseconds
            "unique_id": unique_id,
            "file_path": str(file_path),
            "phase": "Phase 1 - RFP Structure Analysis"
        }

        # Write to file (always overwrite)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)

        print(f"✅ Phase 1 result saved to: {file_path}")
        return str(file_path)

    except Exception as e:
        print(f"❌ Error saving Phase 1 result to file: {e}")
        return None


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
             # Enable LLM caching for better performance and cost reduction
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
             # Enable crew-level caching for better performance
        )


@CrewBase
class ProposalOutlineCrew():
    """Federal RFP Proposal Outline Generation Crew - Phase 2"""

    agents_config = 'config/phase2_agents.yaml'
    tasks_config = 'config/phase2_tasks.yaml'

    def __init__(self, inputs: dict = {}):
        """
        Initializes the ProposalOutlineCrew with the given inputs.
        Loads agents and tasks configuration from YAML files for federal RFP processing.
        """
        self.fileName =''
        self.inputs = inputs
        self.unique_id = inputs.get("unique_id")

        self.llm = LLM(
            model=os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview'),
            api_key=os.getenv('OPENAI_KEY'),
             # Enable LLM caching for better performance and cost reduction
        )

    @agent
    def rfp_document_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['rfp_document_analyzer'],
            tools=[OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)],
            llm=self.llm,
        )

    @agent
    def structure_alignment_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['structure_alignment_agent'],
            tools=[OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)],
            llm=self.llm,
        )

    @agent
    def compliance_evaluation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['compliance_evaluation_agent'],
            tools=[OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)],
            llm=self.llm,
        )

    @agent
    def federal_json_outline_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['federal_json_outline_agent'],
            tools=[OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)],
            llm=self.llm,
        )

    @agent
    def outline_validation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['outline_validation_agent'],
            tools=[OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)],
            llm=self.llm,
        )

    @task
    def analyze_rfp_documents_and_extract_requirements(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_rfp_documents_and_extract_requirements'],
            tools=[OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)],
            agent=self.rfp_document_analyzer(),
        )

    @task
    def align_structure_to_rfp_format(self) -> Task:
        return Task(
            config=self.tasks_config['align_structure_to_rfp_format'],
            tools=[OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)],
            agent=self.structure_alignment_agent(),
        )

    @task
    def map_compliance_and_evaluation_criteria(self) -> Task:
        return Task(
            config=self.tasks_config['map_compliance_and_evaluation_criteria'],
            tools=[OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)],
            agent=self.compliance_evaluation_agent(),
        )

    @task
    def generate_detailed_json_outline(self) -> Task:
        return Task(
            config=self.tasks_config['generate_detailed_json_outline'],
            tools=[OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)],
            agent=self.federal_json_outline_agent(),
        )

    @task
    def validate_json_completeness_and_compliance(self) -> Task:
        return Task(
            config=self.tasks_config['validate_json_completeness_and_compliance'],
            tools=[OpenSearchQueryTool(unique_id=self.unique_id, min_score=0.4)],
            agent=self.outline_validation_agent(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.rfp_document_analyzer(),
                self.structure_alignment_agent(),
                self.compliance_evaluation_agent(),

                self.outline_validation_agent(),
                self.federal_json_outline_agent()
            ],
            tasks=[
                self.analyze_rfp_documents_and_extract_requirements(),
                self.align_structure_to_rfp_format(),
                self.map_compliance_and_evaluation_criteria(),

                self.validate_json_completeness_and_compliance(),
                self.generate_detailed_json_outline()
            ],
            process=Process.sequential,
            verbose=True,
            llm=self.llm,
            memory=True,
             # Enable crew-level caching for better performance
        )


async def kickoff_phase1_crew(inputs: dict = {}):
    """
    Kicks off Phase 1 crew for RFP structure analysis and shredding.
    Returns parsed JSON output from LLM.
    Includes retry logic with 3 attempts to handle potential errors.
    Saves result to JSON file automatically.
    """
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            print(f"\n🚀 Starting Phase 1 crew attempt {attempt + 1}/{max_retries}")
            crew_instance = ProposalOutlinePhase1Crew(inputs=inputs)
            result = await crew_instance.crew().kickoff_async(inputs=inputs)

            print(f"✅ Phase 1 crew completed successfully on attempt {attempt + 1}")
            print(f"📋 Result type: {type(result)}")
            print(f"📋 Result has 'raw' attribute: {hasattr(result, 'raw')}")

            # Parse the raw string content if it exists
            if hasattr(result, 'raw') and isinstance(result.raw, str):
                print(f"📋 Raw result length: {len(result.raw)} characters")
                print(f"📋 Raw result preview: {result.raw[:200]}...")
                try:
                    # Use the existing clean_and_parse_json function to parse the raw string
                    parsed_result = clean_and_parse_json(result.raw)

                    print(f"✅ Successfully parsed Phase 1 JSON result")
                    print(f"📋 Parsed result type: {type(parsed_result)}")
                    if isinstance(parsed_result, dict):
                        print(f"📋 Parsed result keys: {list(parsed_result.keys())}")
                        sections_count = len(parsed_result.get('sections', []))
                        print(f"📋 Sections found in parsed result: {sections_count}")

                    # Save parsed result to file
                    unique_id = inputs.get("unique_id")
                    save_phase1_result_to_file(parsed_result, unique_id)

                    return parsed_result
                except Exception as e:
                    print(f"❌ Error parsing raw result: {e}")
                    print(f"📋 Raw content preview: {result.raw[:500]}...")
                    error_result = {"error": "Failed to parse JSON", "raw_content": result.raw}

                    # Save error result to file
                    unique_id = inputs.get("unique_id")
                    save_phase1_result_to_file(error_result, unique_id)

                    return error_result

            # Handle case where result doesn't have raw attribute or raw is not a string
            print(f"📋 No raw string found, using result directly")
            if isinstance(result, dict):
                print(f"📋 Direct result keys: {list(result.keys())}")
                sections_count = len(result.get('sections', []))
                print(f"📋 Sections found in direct result: {sections_count}")

            # Save and return the result as-is if no raw attribute or not a string
            unique_id = inputs.get("unique_id")
            save_phase1_result_to_file(result, unique_id)
            return result

        except Exception as e:
            last_error = e
            print(f"❌ Phase 1 crew attempt {attempt + 1} failed: {str(e)}")
            print(f"📋 Error type: {type(e).__name__}")

            if attempt < max_retries - 1:
                print(f"🔄 Retrying... ({attempt + 2}/{max_retries})")
            else:
                print(f"💥 All {max_retries} attempts failed for Phase 1 crew")

    # If all retries failed, save error information to file
    error_result = {
        "error": f"Phase 1 crew failed after {max_retries} attempts",
        "last_error": str(last_error),
        "inputs": inputs
    }

    unique_id = inputs.get("unique_id")
    save_phase1_result_to_file(error_result, unique_id)

    return error_result


async def kickoff_phase2_crew_for_section(inputs: dict = {}):
    """
    Kicks off Phase 2 crew for a specific section or subsection.
    Uses section_crew() to return pure JSON instead of validation reports.
    Includes retry logic with 3 attempts to handle potential errors.
    """
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            crew_instance = ProposalOutlineCrew(inputs=inputs)
            # Use section_crew instead of full crew to get JSON outline directly
            result = await crew_instance.crew().kickoff_async(inputs=inputs)
            return result
        except Exception as e:
            last_error = e
            print(f"Phase 2 crew attempt {attempt + 1} failed: {str(e)}")

            if attempt < max_retries - 1:
                print(f"Retrying... ({attempt + 2}/{max_retries})")
            else:
                print(f"All {max_retries} attempts failed for Phase 2 crew")

    # If all retries failed, return error information
    return {
        "error": f"Phase 2 crew failed after {max_retries} attempts",
        "last_error": str(last_error),
        "inputs": inputs
    }

async def process_section_batch(section_idx: int, section: dict, inputs: dict):
    """
    Process a single section and all its subsections concurrently.
    Returns the complete section structure with all subsections processed.
    """
    print(f"Processing section {section_idx + 1}: {section.get('title', 'Unknown')}")

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

    # Build section structure with default values
    final_section = {
        "section_title": section.get('title', 'Unknown Section'),
        "section_id": str(section_idx + 1),
        "subsections": [],
        "section_purpose": "",
        "instructions_to_writer": "",
        "source_mapping": [],
        "win_theme_alignment": []
    }

    # Create tasks for concurrent processing
    tasks = []

    # Add section-level processing task
    tasks.append(kickoff_phase2_crew_for_section(section_inputs))

    # Add subsection processing tasks
    subsections = section.get('subsections', [])
    subsection_tasks = []

    for subsection_idx, subsection in enumerate(subsections):
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
        subsection_tasks.append((subsection_idx, subsection, kickoff_phase2_crew_for_section(subsection_inputs)))

    # Execute section task
    try:
        section_outline = await tasks[0]
        print(f"Generated outline for section: {section.get('title', 'Unknown')}")

        # Parse raw JSON if section_outline has raw attribute
        if hasattr(section_outline, 'raw') and isinstance(section_outline.raw, str):
            try:
                section_outline = json.loads(section_outline.raw)
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Failed to parse section_outline.raw as JSON: {e}")
                section_outline = {"error": "Failed to parse JSON"}

        # Update with actual data if successful, keep defaults if error
        if isinstance(section_outline, dict) and not section_outline.get("error"):
            final_section.update({
                "section_purpose": section_outline.get("section_purpose", ""),
                "instructions_to_writer": section_outline.get("instructions_to_writer", ""),
                "source_mapping": section_outline.get("source_mapping", []),
                "win_theme_alignment": section_outline.get("win_theme_alignment", [])
            })
        else:
            print(f"Warning: Section outline generation failed for '{section.get('title', 'Unknown')}', using defaults")
            if isinstance(section_outline, dict) and section_outline.get("error"):
                print(f"Error details: {section_outline.get('error')}")

    except Exception as e:
        print(f"Error generating outline for section {section.get('title', 'Unknown')}: {e}")

    # Process subsections concurrently if any exist
    if subsection_tasks:
        print(f"Processing {len(subsection_tasks)} subsections concurrently for section: {section.get('title', 'Unknown')}")

        # Execute all subsection tasks concurrently
        subsection_results = await asyncio.gather(
            *[task for _, _, task in subsection_tasks],
            return_exceptions=True
        )

        # Process results
        for (subsection_idx, subsection, _), result in zip(subsection_tasks, subsection_results):
            if isinstance(result, Exception):
                print(f"  Error generating outline for subsection {subsection.get('title', 'Unknown')}: {result}")
                subsection_outline = {"error": str(result)}
            else:
                subsection_outline = result
                print(f"  Generated outline for subsection: {subsection.get('title', 'Unknown')}")

            # Build subsection structure with default values
            final_subsection = {
                "subsection_id": f"{section_idx + 1}.{subsection_idx + 1}",
                "subsection_title": subsection.get('title', 'Unknown Subsection'),
                "requirement": f"Detailed requirements for {subsection.get('title', 'this subsection')}.",
                "section_purpose": "",
                "instructions_to_writer": "",
                "source_mapping": [],
                "win_theme_alignment": []
            }

            # Parse raw JSON if subsection_outline has raw attribute
            if hasattr(subsection_outline, 'raw') and isinstance(subsection_outline.raw, str):
                try:
                    subsection_outline = json.loads(subsection_outline.raw)
                except (json.JSONDecodeError, AttributeError) as e:
                    print(f"Failed to parse subsection_outline.raw as JSON: {e}")
                    subsection_outline = {"error": "Failed to parse JSON"}

            # Update with actual data if successful, keep defaults if error
            if isinstance(subsection_outline, dict) and not subsection_outline.get("error"):
                final_subsection.update({
                    "section_purpose": subsection_outline.get("section_purpose", ""),
                    "instructions_to_writer": subsection_outline.get("instructions_to_writer", ""),
                    "source_mapping": subsection_outline.get("source_mapping", []),
                    "win_theme_alignment": subsection_outline.get("win_theme_alignment", [])
                })
            else:
                print(f"Warning: Subsection outline generation failed for '{subsection.get('title', 'Unknown')}', using defaults")
                if isinstance(subsection_outline, dict) and subsection_outline.get("error"):
                    print(f"Error details: {subsection_outline.get('error')}")

            final_section["subsections"].append(final_subsection)

    print(f"Completed processing section: {section.get('title', 'Unknown')} with {len(final_section['subsections'])} subsections")
    return final_section


async def kickoff_proposal_outline_crew(inputs: dict = {}):
    """
    Main orchestration function that runs both phases with sequential section processing:
    1. Phase 1: RFP structure analysis and shredding
    2. Phase 2: Process sections sequentially, but subsections within each section concurrently
    """
    print("Starting Phase 1: RFP Structure Analysis...")

    phase1_result = await kickoff_phase1_crew(inputs)

    # Debug: Print Phase 1 result details
    print(f"\n🔍 DEBUG: Phase 1 Result Analysis:")
    print(f"📋 Result type: {type(phase1_result)}")
    print(f"📋 Result has 'raw' attribute: {hasattr(phase1_result, 'raw')}")

    if hasattr(phase1_result, 'raw'):
        print(f"📋 Raw type: {type(phase1_result.raw)}")
        print(f"📋 Raw content preview: {str(phase1_result.raw)[:500]}...")
    else:
        print(f"📋 Direct result preview: {str(phase1_result)[:500]}...")

    # Extract sections from phase1_result.raw if it has the .raw attribute
    if hasattr(phase1_result, 'raw') and isinstance(phase1_result.raw, dict):
        sections_data = phase1_result.raw.get('sections', [])
        processing_metadata = phase1_result.raw.get('processing_metadata', {})
        print(f"📋 Extracted from .raw attribute: {len(sections_data)} sections")
    elif isinstance(phase1_result, dict):
        sections_data = phase1_result.get('sections', [])
        processing_metadata = phase1_result.get('processing_metadata', {})
        print(f"📋 Extracted from direct dict: {len(sections_data)} sections")
    else:
        print("⚠️ WARNING: Unexpected phase1_result format - returning early")
        print(f"📋 Phase1 result: {phase1_result}")
        return phase1_result

    print(f"\n✅ Found {len(sections_data)} sections to process")

    # If no sections found, provide detailed error information
    if len(sections_data) == 0:
        print(f"\n❌ ERROR: No sections discovered in Phase 1!")
        print(f"📋 Processing metadata: {processing_metadata}")
        print(f"📋 Phase 1 complete result structure:")
        if hasattr(phase1_result, 'raw'):
            print(f"📋 Raw result keys: {list(phase1_result.raw.keys()) if isinstance(phase1_result.raw, dict) else 'Not a dict'}")
        else:
            print(f"📋 Direct result keys: {list(phase1_result.keys()) if isinstance(phase1_result, dict) else 'Not a dict'}")

        # Return an error structure instead of proceeding
        return {
            "error": "No sections found in RFP analysis",
            "outline_title": processing_metadata.get('rfp_title', 'Failed Analysis'),
            "sections": [],
            "unique_id": inputs.get("unique_id"),
            "debug_info": {
                "phase1_result_type": str(type(phase1_result)),
                "processing_metadata": processing_metadata,
                "sections_found": 0
            }
        }

    print("Starting Phase 2: Sequential Section Processing with Concurrent Subsections...")

    # Build the final outline structure
    outline_title = processing_metadata.get('rfp_title', 'Proposal Outline')
    processed_sections = []

    start_time = time.time()

    # Process sections sequentially (one at a time)
    for section_idx, section in enumerate(sections_data):
        print(f"\n{'='*60}")
        print(f"Processing Section {section_idx + 1}/{len(sections_data)}: {section.get('title', 'Unknown')}")
        print(f"{'='*60}")

        section_start_time = time.time()

        try:
            # Process this section and its subsections (subsections run concurrently within the section)
            section_result = await process_section_batch(section_idx, section, inputs)
            processed_sections.append(section_result)

            section_end_time = time.time()
            section_duration = section_end_time - section_start_time
            print(f"✅ Completed Section {section_idx + 1} in {section_duration:.2f} seconds")

        except Exception as e:
            print(f"❌ Error processing section {section_idx + 1}: {e}")
            # Create a fallback section structure
            fallback_section = {
                "section_title": section.get('title', f'Unknown Section {section_idx + 1}'),
                "section_id": str(section_idx + 1),
                "subsections": [],
                "section_purpose": f"Error processing section: {str(e)}",
                "instructions_to_writer": "Please review and complete this section manually.",
                "source_mapping": [],
                "win_theme_alignment": []
            }
            processed_sections.append(fallback_section)

            section_end_time = time.time()
            section_duration = section_end_time - section_start_time
            print(f"⚠️ Section {section_idx + 1} failed after {section_duration:.2f} seconds")

    end_time = time.time()
    total_duration = end_time - start_time

    # Build final result
    final_result = {
        "outline_title": outline_title,
        "sections": processed_sections,
        "unique_id": inputs.get("unique_id"),
        "processing_stats": {
            "total_sections": len(sections_data),
            "processing_time_seconds": total_duration,
            "sequential_section_processing": True,
            "concurrent_subsection_processing": True
        }
    }

    print(f"\n{'='*60}")
    print(f"🎉 COMPLETED: Generated proposal outline with {len(processed_sections)} sections")
    print(f"⏱️ Total processing time: {total_duration:.2f} seconds")
    if len(sections_data) > 0:
        print(f"📊 Average time per section: {total_duration/len(sections_data):.2f} seconds")
    else:
        print(f"📊 No sections found to process")
    print(f"{'='*60}")

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
    Includes caching for improved performance.
    """
    return await kickoff_phase2_crew_for_section(inputs)
