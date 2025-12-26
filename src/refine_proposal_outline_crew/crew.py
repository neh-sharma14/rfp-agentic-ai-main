### File: src/refine_proposal_outline_crew/crew.py

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, task, crew
from ..tools import LlamaIndexQueryTool
from ..types import ProposalOutline

@CrewBase
class RefineProposalOutlineCrew():
    """Crew for refining and expanding an existing proposal outline based on refinement prompt"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, knowledge_dir: str = "./knowledge"):
        self.knowledge_dir = knowledge_dir
        self.rfp_query_tool = LlamaIndexQueryTool(knowledge_dir=self.knowledge_dir).get_tool()

    @agent
    def outline_refiner(self) -> Agent:
        return Agent(
            config=self.agents_config['outline_refiner'],
            tools=[self.rfp_query_tool],
            verbose=True,
        )

    @task
    def refine_proposal_outline_task(self) -> Task:
        return Task(
            config=self.tasks_config['refine_proposal_outline_task'],
            output_json=ProposalOutline,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.outline_refiner(),
            ],
            tasks=[
                self.refine_proposal_outline_task(),
            ],
            process=Process.sequential,
            verbose=True,
        )

# Kickoff function
async def kickoff_refine_proposal_outline_crew(knowledge_dir: str = "./knowledge", inputs: dict = {}):
    crew_instance = RefineProposalOutlineCrew(knowledge_dir=knowledge_dir)
    result = await crew_instance.crew().kickoff_async(inputs=inputs)
    return result
