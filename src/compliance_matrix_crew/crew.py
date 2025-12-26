from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, task, crew
from typing import List, Dict, Any, Union
from pydantic import BaseModel
import asyncio
import warnings
import logging
import json
import os

# Suppress specific PyPDF warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pypdf._reader')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplianceRequirement(BaseModel):
    section_id: Union[str, float]  # Allow both string and float values
    requirement: str

class ComplianceMatrix(BaseModel):
    requirement: str
    status: str
    justification: str
    sectionNo: str

class ComplianceMatrixList(BaseModel):
    """Wrapper model for list of compliance matrices"""
    matrices: List[ComplianceMatrix]

class ComplianceMatrixCrew:
    """Crew for analyzing compliance between requirements and proposal outline"""

    def __init__(self):
        # Load configurations
        config_dir = os.path.join(os.path.dirname(__file__), 'config')
        self.agents_config = self._load_config(os.path.join(config_dir, 'agents.yaml'))
        self.tasks_config = self._load_config(os.path.join(config_dir, 'tasks.yaml'))

        # Validate configurations
        if not self.agents_config:
            logger.error("Failed to load agents configuration")
        if not self.tasks_config:
            logger.error("Failed to load tasks configuration")

    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration file"""
        try:
            import yaml
            if not os.path.exists(config_path):
                logger.error(f"Config file not found: {config_path}")
                return {}

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if not config:
                    logger.error(f"Empty config file: {config_path}")
                    return {}
                logger.info(f"Successfully loaded config from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            return {}

    def manager_agent(self) -> Agent:
        agent_config = self.agents_config.get('manager_agent', {})
        if not agent_config:
            logger.error("Manager agent configuration is empty")
        return Agent(
            config=agent_config,
            verbose=True,
        )

    def compliance_analyzer(self) -> Agent:
        agent_config = self.agents_config.get('compliance_analyzer', {})
        if not agent_config:
            logger.error("Compliance analyzer configuration is empty")
        return Agent(
            config=agent_config,
            verbose=True,
        )

    def analyze_compliance_task(self) -> Task:
        task_config = self.tasks_config.get('analyze_compliance_task', {})
        if not task_config:
            logger.error("Failed to load task configuration")
            return None

        description = task_config.get('description', '')
        expected_output = task_config.get('expected_output', '')

        if not description or not expected_output:
            logger.error("Missing required task configuration fields")
            logger.error(f"Description present: {bool(description)}")
            logger.error(f"Expected output present: {bool(expected_output)}")
            return None

        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.compliance_analyzer(),
            output_json=ComplianceMatrixList,  # Use the wrapper model
        )

    def crew(self) -> Crew:
        """Creates the ComplianceMatrix crew"""
        task = self.analyze_compliance_task()
        if not task:
            raise ValueError("Failed to create compliance analysis task")

        return Crew(
            agents=[
                self.manager_agent(),
                self.compliance_analyzer()
            ],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

    def _parse_crew_result(self, result) -> List[ComplianceMatrix]:
        """Parse the crew result into a list of ComplianceMatrix objects"""
        try:
            logger.info(f"Parsing crew result of type: {type(result)}")

            # If result is a string, try to parse it as JSON
            if isinstance(result, str):
                logger.info("Result is string, attempting to parse as JSON")
                data = json.loads(result)
            # If result has a json_dict attribute, use that
            elif hasattr(result, 'json_dict'):
                logger.info("Result has json_dict attribute")
                data = result.json_dict
            # If result is a dict, use it directly
            elif isinstance(result, dict):
                logger.info("Result is dictionary")
                data = result
            else:
                logger.error(f"Unexpected result type: {type(result)}")
                return []

            logger.info(f"Parsed data structure: {type(data)}")

            # Extract matrices from the data
            if isinstance(data, dict) and 'matrices' in data:
                logger.info("Found 'matrices' key in data")
                matrices_data = data['matrices']
            else:
                logger.info("Using data directly as matrices")
                matrices_data = data

            if not isinstance(matrices_data, list):
                logger.error(f"Matrices data is not a list: {type(matrices_data)}")
                return []

            # Convert to ComplianceMatrix objects
            matrices = []
            for idx, matrix_data in enumerate(matrices_data):
                if isinstance(matrix_data, dict):
                    try:
                        # Ensure all required fields are present
                        if not all(k in matrix_data for k in ['requirement', 'status', 'justification', 'sectionNo']):
                            logger.warning(f"Skipping matrix {idx} with missing fields: {matrix_data}")
                            continue

                        matrix = ComplianceMatrix(**matrix_data)
                        matrices.append(matrix)
                        logger.info(f"Successfully parsed matrix {idx}")
                    except Exception as e:
                        logger.error(f"Error parsing matrix {idx}: {str(e)}")
                        continue
                else:
                    logger.warning(f"Skipping non-dict matrix data at index {idx}: {type(matrix_data)}")

            logger.info(f"Successfully parsed {len(matrices)} matrices")
            return matrices
        except Exception as e:
            logger.error(f"Error parsing crew result: {str(e)}")
            return []

async def kickoff_compliance_matrix_crew(inputs: dict = {}):
    """
    Kicks off the compliance matrix crew with timeout handling and chunked processing.

    Args:
        inputs (dict): Dictionary containing:
            - requirements: List of compliance requirements
            - outline: Proposal outline to analyze against
    """
    try:
        # Initialize the crew instance
        crew_instance = ComplianceMatrixCrew()

        # Get requirements from inputs
        requirements = inputs.get('requirements', [])
        outline = inputs.get('outline', {})

        logger.info(f"Received {len(requirements)} requirements and outline with {len(outline)} sections")

        if not requirements:
            logger.warning("No requirements provided for analysis")
            return ComplianceMatrixList(matrices=[])

        if not outline:
            logger.warning("No outline provided for analysis")
            return ComplianceMatrixList(matrices=[])

        # Convert requirements to the expected format
        formatted_requirements = []
        for idx, req in enumerate(requirements):
            if not isinstance(req, dict):
                logger.warning(f"Skipping invalid requirement format at index {idx}: {req}")
                continue

            section_id = req.get('section_id')
            requirement = req.get('requirement')

            if section_id is None or requirement is None:
                logger.warning(f"Skipping requirement at index {idx} with missing fields: {req}")
                continue

            formatted_requirements.append({
                'section_id': section_id,
                'requirement': requirement
            })

        if not formatted_requirements:
            logger.warning("No valid requirements after formatting")
            return ComplianceMatrixList(matrices=[])

        # Process all requirements at once
        try:
            logger.info(f"Processing {len(formatted_requirements)} requirements")

            # Create inputs for all requirements
            chunk_inputs = {
                'requirements': formatted_requirements,
                'outline': outline
            }

            # Process with timeout
            logger.info("Starting crew processing with 5-minute timeout")
            result = await asyncio.wait_for(
                crew_instance.crew().kickoff_async(inputs=chunk_inputs),
                timeout=300  # 5 minute timeout
            )
            logger.info("Crew processing completed successfully")

            # Parse the result into ComplianceMatrix objects
            matrices = crew_instance._parse_crew_result(result)
            logger.info(f"Parsed {len(matrices)} matrices from result")

            # Verify all requirements were processed
            processed_sections = {m.sectionNo for m in matrices}
            missing_sections = {str(req['section_id']) for req in formatted_requirements} - processed_sections

            if missing_sections:
                logger.warning(f"Missing matrices for sections: {missing_sections}")
                # Add default matrices for missing sections
                for req in formatted_requirements:
                    if str(req['section_id']) in missing_sections:
                        matrices.append(ComplianceMatrix(
                            requirement=req['requirement'],
                            status="Fail",
                            justification="This requirement was not properly analyzed in the compliance matrix generation.",
                            sectionNo=str(req['section_id'])
                        ))

            logger.info(f"Completed processing. Generated {len(matrices)} matrices")
            return ComplianceMatrixList(matrices=matrices)

        except asyncio.TimeoutError:
            logger.warning("Timeout processing requirements")
            return ComplianceMatrixList(matrices=[])
        except Exception as e:
            logger.error(f"Error processing requirements: {str(e)}")
            return ComplianceMatrixList(matrices=[])

    except Exception as e:
        logger.error(f"Error in compliance matrix crew: {str(e)}")
        raise