import os, logging
import boto3
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from ..s3connectTool import S3WriterTool
# from ..pptypes import PPTechnicalApproachOutline, PPInitialTechnicalApproachOutline

@CrewBase
class PastPerformanceCrew():
	"""Past Performance Crew"""

	logging.basicConfig(level=logging.DEBUG)
	logger = logging.getLogger(__name__)

	s3_writer_tool = S3WriterTool()

	def __init__(self, solicitation_id: str):
		self.solicitation_id = solicitation_id
		# Get current working directory
		cwd = os.getcwd()

		# Load all files from knowledge directory
		knowledge_dir = os.path.join("knowledge", solicitation_id)

		# Get all files in the directory
		knowledge_files = []
		if os.path.exists(knowledge_dir):
			files = os.listdir(knowledge_dir)
			# Use paths without knowledge/ prefix
			knowledge_files = [os.path.join(solicitation_id, f) for f in files if f.lower().endswith('.pdf')]
		else:
			print(f"Directory does not exist: {knowledge_dir}")

		# Add default PDFs if they exist
		default_pdfs = [
			os.path.join("pastperformance", "k1.pdf"),
			os.path.join("pastperformance", "k3.pdf"),
			os.path.join("pastperformance", "resp_GSA.pdf")
		]
		# Check existence with full path but store relative path
		knowledge_files.extend([pdf for pdf in default_pdfs if os.path.exists(os.path.join("knowledge", pdf))])

		if not knowledge_files:
			raise ValueError(f"No PDF files found in {knowledge_dir} or default locations")

		self.pdf_source = PDFKnowledgeSource(file_paths=knowledge_files)

	# Set up LLM
	llm = LLM(
		model=os.getenv('MODEL'),
		aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
		aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
		aws_region_name=os.getenv('AWS_REGION_NAME')
	)

	# Boto3 session for Bedrock embedding
	boto3_session = boto3.Session(
		region_name=os.getenv('AWS_REGION_NAME'),
		aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
		aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
	)

	# Agent/task YAML configs
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def requirement_analyzer(self) -> Agent:
		return Agent(
			config=self.agents_config['requirement_analyzer'],
			verbose=True,
			knowledge_sources=[self.pdf_source],
			embedder={
				"provider": "bedrock",
				"config": {
					"session": self.boto3_session,
					"model": "amazon.titan-embed-text-v2:0",
					"vector_dimension": 1024
				}
			}
		)

	@agent
	def performance_review_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['performance_review_writer'],
			verbose=True,
			knowledge_sources=[self.pdf_source],
			tools=[self.s3_writer_tool],
			embedder={
				"provider": "bedrock",
				"config": {
					"session": self.boto3_session,
					"model": "amazon.titan-embed-text-v2:0",
					"vector_dimension": 1024
				}
			}
		)

	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
			verbose=True,
			output_file='report-data-extractor.md'
		)

	@task
	def generate_report_task(self) -> Task:
		return Task(
			config=self.tasks_config['generate_report_task'],
			verbose=True,
			output_file='report-writer.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the crew"""
		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True,
			memory=False,
			LLM=self.llm,
			embedder={
				"provider": "bedrock",
				"config": {
					"session": self.boto3_session,
					"model": "amazon.titan-embed-text-v2:0",
					"vector_dimension": 1024
				}
			}
		)

# Async kickoff method
async def kickoff_past_performance_crew(input_data: dict):
	inputs = {
		'section': input_data['section'],
		'subsection': input_data['subsection'],
		'requirement': input_data['requirement'],
		'context': input_data['context'],
		'solicitation_id': input_data['solicitation_id'],
		'my_bucket': os.getenv('pp_bucket')
	}
	result = await PastPerformanceCrew(solicitation_id=input_data['solicitation_id']).crew().kickoff_async(inputs=inputs)
	return result
