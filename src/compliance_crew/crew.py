from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, task, crew
from ..tools import RFPKnowledgeBaseTool
from ..types import ComplianceRequirementList
import os
from typing import List, Dict, Any
import json
import re

class RequirementQueryStrategy:
    """Manages the query strategy for requirement extraction"""

    def __init__(self):
        # Define targeted queries for different requirement types
        self.requirement_queries = [
            # Core Requirement Terms
            {
                "name": "explicit_requirements",
                "queries": [
                    "shall must required to mandatory obligation",
                    "must comply with must adhere to must follow",
                    "shall provide shall submit shall include",
                    "shall demonstrate shall show shall prove",
                    "shall ensure shall maintain shall establish"
                ]
            },
            # Evaluation Requirements
            {
                "name": "evaluation_requirements",
                "queries": [
                    "evaluation factors evaluation criteria evaluation process",
                    "technical evaluation past performance price evaluation",
                    "scoring methodology evaluation methodology",
                    "evaluation team evaluation panel",
                    "evaluation weights evaluation points"
                ]
            },
            # Submission Requirements
            {
                "name": "submission_requirements",
                "queries": [
                    "submission requirements submission instructions",
                    "proposal format proposal structure",
                    "submission deadline due date",
                    "required documents required attachments",
                    "submission method submission process"
                ]
            },
            # Technical Requirements
            {
                "name": "technical_requirements",
                "queries": [
                    "technical approach technical solution",
                    "technical capabilities technical experience",
                    "technical requirements technical specifications",
                    "technical methodology technical process",
                    "technical team technical personnel"
                ]
            }
        ]

        # Results storage
        self.aggregated_results = {}
        self.processed_sections = set()

    def get_queries(self) -> List[Dict[str, Any]]:
        """Returns all queries to be executed"""
        return self.requirement_queries

    def normalize_requirement(self, text: str) -> str:
        """Normalize requirement text for deduplication"""
        return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".,;:")

    def aggregate_results(self, results: List[Dict[str, Any]], query_type: str) -> None:
        """Aggregates and deduplicates results from different queries"""
        for result in results:
            # Extract section number from metadata
            metadata = result.get('metadata', {})
            section = metadata.get('section', '')
            content = result.get('content', '').strip()

            if not content:
                continue

            # Normalize content for deduplication
            norm_content = self.normalize_requirement(content)

            # Skip if we've already processed this exact content
            if norm_content in self.processed_sections:
                continue

            # Store result with query type
            if section not in self.aggregated_results:
                self.aggregated_results[section] = {
                    'content': content,
                    'score': result.get('score', 0),
                    'query_types': [query_type],
                    'metadata': metadata,
                    'normalized': norm_content
                }
            else:
                # Update existing result if new one has higher score
                if result.get('score', 0) > self.aggregated_results[section]['score']:
                    self.aggregated_results[section].update({
                        'content': content,
                        'score': result.get('score', 0),
                        'query_types': [query_type],
                        'metadata': metadata,
                        'normalized': norm_content
                    })

            self.processed_sections.add(norm_content)

    def get_aggregated_results(self) -> List[Dict[str, Any]]:
        """Returns aggregated and sorted results"""
        # Convert to list and sort by section
        results = [
            {
                'section': section,
                **data
            }
            for section, data in self.aggregated_results.items()
        ]
        return sorted(results, key=lambda x: x['section'])

@CrewBase
class ComplianceExtractionCrew():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, inputs: dict = None):
        self.inputs = inputs or {}
        self.index_name = inputs.get('index_name')
        self.solicitation_id = inputs.get('solicitation_id')
        self.query_strategy = RequirementQueryStrategy()

        if not self.index_name:
            raise ValueError("OpenSearch index name is required")

        # Create the RFP Knowledge Base Tool with higher limit
        self.rfp_query_tool = RFPKnowledgeBaseTool(
            index_name=self.index_name,
            name="RFP Knowledge Base Tool",
            description="A tool to query the RFP knowledge base for specific requirements, evaluation criteria, and proposal instructions.",
            solicitation_id=self.solicitation_id,
            limit=30
        )

        # Set up LLM
        self.llm = LLM(
            model=os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview'),
            api_key=os.getenv('OPENAI_KEY'),
            max_tokens=15000,
            temperature=0.0,
            presence_penalty=0.0,
            frequency_penalty=0.3,
            n=1,
        )

    @agent
    def research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['research_agent'],
            tools=[self.rfp_query_tool],
            verbose=True
        )

    @agent
    def writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['writer_agent'],
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        # Modify task to use query strategy
        task_config = self.tasks_config['research_task'].copy()
        task_config['description'] += "\n\nUsing targeted query strategy to extract requirements."
        return Task(config=task_config)

    @task
    def writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['writing_task'],
            output_json=ComplianceRequirementList
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.research_agent(), self.writer_agent()],
            tasks=[self.research_task(), self.writing_task()],
            process=Process.sequential,
            verbose=True,
            LLM=self.llm
        )

    async def execute_queries(self) -> List[Dict[str, Any]]:
        """Executes all queries in the strategy and aggregates results"""
        for query_group in self.query_strategy.get_queries():
            query_type = query_group['name']
            for query in query_group['queries']:
                try:
                    print(f"\nExecuting query: {query}")
                    # Execute query with the tool using the research agent
                    research_agent = self.research_agent()
                    results = await research_agent.execute_task(
                        task=f"Search for requirements using this query: {query}",
                        context={"query": query, "solicitation_id": self.solicitation_id}
                    )

                    # Parse results and aggregate
                    if isinstance(results, str):
                        # Parse the formatted response back into structured data
                        results = self._parse_tool_response(results)

                    self.query_strategy.aggregate_results(results, query_type)
                    print(f"Found {len(results)} results for query: {query}")

                except Exception as e:
                    print(f"Error executing query '{query}': {str(e)}")
                    continue

        return self.query_strategy.get_aggregated_results()

    def _parse_tool_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse the tool's text response back into structured data"""
        results = []
        current_doc = None
        current_section = None

        for line in response_text.split('\n'):
            line = line.strip()

            # Document header
            if line.startswith('Document'):
                current_doc = line.split(':', 1)[1].strip()
                continue

            # Section content
            if line.startswith('Section') and '(Relevance:' in line:
                # Extract section number and score
                section_match = re.search(r'Section (\d+) \(Relevance: ([\d.]+)\)', line)
                if section_match:
                    current_section = {
                        'content': '',
                        'score': float(section_match.group(2)),
                        'metadata': {
                            'filename': current_doc,
                            'section': f"Section {section_match.group(1)}"
                        }
                    }
                continue

            # Content line
            if current_section and line and not line.startswith('=') and not line.startswith('-'):
                current_section['content'] += line + ' '

            # End of section
            if line.startswith('-' * 30) and current_section:
                current_section['content'] = current_section['content'].strip()
                results.append(current_section)
                current_section = None

        return results

# Kickoff function
async def kickoff_compliance_requirements_crew(inputs: dict = {}):
    """
    Kicks off the compliance requirements crew with enhanced query strategy.
    """
    crew_instance = ComplianceExtractionCrew(inputs=inputs)

    # Execute queries first
    print("\nExecuting targeted queries...")
    results = await crew_instance.execute_queries()

    # Update inputs with aggregated results
    inputs['aggregated_results'] = results

    # Run the crew
    crew = crew_instance.crew()
    result = await crew.kickoff_async(inputs=inputs)

    # Get and print usage metrics
    print("\nCrew Usage Metrics:")
    print(crew.usage_metrics)
    return result
