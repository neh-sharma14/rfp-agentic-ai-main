from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import boto3
import os

from ..tools import LlamaIndexQueryTool
from ..types import OutlineSubsection
import json


@CrewBase
class ProposalOutlineCrew():
	"""ProposalWriter crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def __init__(self, knowledge_dir: str = "./knowledge"):
		self.knowledge_dir = knowledge_dir

		# Create the LlamaIndexQueryTool with knowledge_dir parameter
		self.rfp_query_tool = LlamaIndexQueryTool(knowledge_dir=self.knowledge_dir).get_tool()

	@agent
	def proposal_outline_editor(self) -> Agent:
		return Agent(
			config=self.agents_config['proposal_outline_editor'],
			tools=[self.rfp_query_tool],
			verbose=True,
		)

	@task
	def revise_proposal_outline_task(self) -> Task:
		return Task(
			config=self.tasks_config['revise_proposal_outline_task'],
			output_json=OutlineSubsection,
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the ProposalWriter crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)


async def kickoff_revise_proposal_outline_crew(knowledge_dir: str = "./knowledge", inputs: dict = {}):
    """
    Kicks off the  approach outline crew,
    using the specified 'knowledge_dir'.
    """
    crew_instance = ProposalOutlineCrew(knowledge_dir=knowledge_dir)
    result = await crew_instance.crew().kickoff_async(inputs=inputs)
    return result
