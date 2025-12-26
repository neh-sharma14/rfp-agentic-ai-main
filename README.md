<div align="center">
  
# 🚀 RFP Agentic AI Platform

 **Enterprise-Grade AI-Powered RFP Response Generation System**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.8-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![AWS](https://img.shields.io/badge/AWS-Bedrock-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/bedrock/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)


[![CrewAI](https://img.shields.io/badge/🤖_CrewAI-0.114.0-6C63FF?style=for-the-badge)](https://www.crewai.com/)
[![LlamaIndex](https://img.shields.io/badge/🦙_LlamaIndex-0.12.22-8B5CF6?style=for-the-badge)](https://www.llamaindex.ai/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![OpenSearch](https://img.shields.io/badge/OpenSearch-2.8.0-005EB8?style=for-the-badge&logo=opensearch&logoColor=white)](https://opensearch.org/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.23-FF6B6B?style=for-the-badge)](https://www.trychroma.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.10.6-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.3-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

---
</di>
<div align="left">
An intelligent, multi-agent AI platform that revolutionizes the Request for Proposal (RFP) response process using cutting-edge artificial intelligence, advanced natural language processing, and autonomous agent orchestration.

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Architecture](#-architecture)
- [Setup & Installation](#-setup--installation)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [Docker Deployment](#-docker-deployment)
- [API Documentation](#-api-documentation)

---

## 🎯 Overview

The **RFP Agentic AI Platform** leverages state-of-the-art AI technologies to automate and enhance the RFP response workflow. By utilizing multi-agent systems, advanced language models, and intelligent knowledge retrieval, this platform dramatically reduces the time and effort required to create high-quality, compliant proposal responses.

### What Makes This Platform Unique?

- **🤖 Multi-Agent Architecture**: Specialized AI agents work collaboratively on different aspects of proposal generation
- **🧠 Advanced RAG (Retrieval-Augmented Generation)**: Intelligent knowledge retrieval from your organization's past proposals and documentation
- **📊 Compliance Matrix Automation**: Automated compliance checking and matrix generation
- **🎨 Proposal Outline Generation**: AI-driven proposal structure and content organization
- **☁️ Cloud-Native Design**: Built for scalability with AWS integration
- **🔒 Enterprise-Ready**: Secure, scalable, and production-ready architecture

---

## ✨ Key Features

### 🎯 Core Capabilities

- **Intelligent Proposal Generation**: Automatically generate comprehensive proposal outlines and content
- **Compliance Analysis**: AI-powered compliance matrix creation and requirement analysis
- **Past Performance Integration**: Leverage historical project data to strengthen proposals
- **Multi-Model Support**: Integration with multiple LLM providers (Claude, GPT-4, Cohere)
- **Knowledge Base Management**: Advanced vector storage and semantic search capabilities
- **Real-time Collaboration**: RESTful API for seamless integration with existing workflows

### 🔧 Technical Features

- **Asynchronous Processing**: High-performance async operations for scalability
- **Vector Database Integration**: Advanced semantic search with OpenSearch and ChromaDB
- **Document Processing**: Intelligent parsing of PDFs, DOCX, and other document formats
- **Cloud Storage**: Seamless integration with AWS S3 for document management
- **Observability**: Built-in OpenTelemetry instrumentation for monitoring
- **Type Safety**: Full Pydantic v2 validation for data integrity

---

## 🛠 Technology Stack

Our platform is built on a foundation of cutting-edge technologies, carefully selected to deliver enterprise-grade performance, scalability, and reliability.

### **Core Framework & Runtime**

We leverage modern Python and high-performance web frameworks to ensure optimal speed and developer productivity.

| Technology | Version | Purpose & Benefits |
|------------|---------|-------------------|
| **Python** | 3.12 | Latest stable Python release with enhanced performance, improved type hints, and better error messages. Provides robust foundation for AI/ML workloads |
| **FastAPI** | 0.115.8 | One of the fastest Python web frameworks available. Automatic API documentation, async support, and built-in data validation ensure rapid development and production-ready APIs |
| **Uvicorn** | 0.34.0 | Lightning-fast ASGI server with excellent performance characteristics. Handles thousands of concurrent connections efficiently |
| **Pydantic** | 2.10.6 | Industry-leading data validation library. Ensures type safety, automatic data parsing, and comprehensive error handling across the entire application |

### **AI & Machine Learning Ecosystem**

Our AI stack represents the state-of-the-art in language models and agent orchestration, enabling sophisticated multi-agent workflows.

| Technology | Version | Purpose & Benefits |
|------------|---------|-------------------|
| **CrewAI** | 0.114.0 | **Revolutionary multi-agent orchestration framework**. Enables multiple AI agents to collaborate on complex tasks, mimicking human team dynamics for superior proposal generation |
| **LangChain** | 0.3.18 | **Industry-standard LLM framework**. Provides powerful chains, agents, and memory systems for building sophisticated AI applications with modular, reusable components |
| **LlamaIndex** | 0.12.22 | **Advanced data framework for LLMs**. Specializes in connecting custom data sources to language models, enabling context-aware responses from your organization's knowledge base |
| **OpenAI** | 1.63.2 | **Access to GPT-4 and GPT-4 Turbo**. Industry-leading language models for natural language understanding, generation, and reasoning tasks |
| **Anthropic** | 0.49.0 | **Claude 3.5 Sonnet integration**. Provides exceptional reasoning capabilities, extended context windows (200K tokens), and superior instruction following |
| **Cohere** | 5.13.12 | **Enterprise-grade LLM provider**. Offers specialized models for embeddings, classification, and generation with excellent multilingual support |
| **LiteLLM** | 1.60.2 | **Unified LLM interface**. Seamlessly switch between different LLM providers (OpenAI, Anthropic, Cohere) with a single API, ensuring flexibility and vendor independence |

### **AWS Cloud Services**

Enterprise-grade cloud infrastructure providing scalability, security, and reliability.

| Service | Purpose & Benefits |
|---------|-------------------|
| **Amazon Bedrock** | **Fully managed foundation model service**. Access to Claude 3.5 Sonnet and other leading models without managing infrastructure. Pay-per-use pricing and enterprise security |
| **AWS S3** | **Scalable object storage**. Stores documents, knowledge bases, and past performance data with 99.999999999% durability. Seamless integration with other AWS services |
| **Amazon OpenSearch Serverless** | **Managed vector database**. Serverless architecture eliminates infrastructure management. Provides fast, accurate semantic search across millions of documents |
| **AWS IAM** | **Enterprise security and access control**. Fine-grained permissions, role-based access, and comprehensive audit trails ensure data security and compliance |

### **Vector Databases & Semantic Search**

Multiple vector database options for optimal performance and flexibility in semantic search operations.

| Technology | Version | Purpose & Benefits |
|------------|---------|-------------------|
| **OpenSearch** | 2.8.0 | **Production-grade vector search**. Serverless deployment, automatic scaling, and sub-second query times across large document collections |
| **ChromaDB** | 0.5.23 | **Embedded vector database**. Perfect for development and testing. Simple API, fast queries, and minimal setup requirements |
| **LanceDB** | 0.19.0 | **High-performance embedded database**. Optimized for ML workloads with native support for multi-modal data and versioning |
| **Qdrant** | 1.13.2 | **Purpose-built vector search engine**. Exceptional performance for similarity search, filtering, and hybrid search scenarios |

### **Document Processing & Intelligence**

Comprehensive document processing pipeline supporting multiple formats with advanced extraction capabilities.

| Technology | Version | Purpose & Benefits |
|------------|---------|-------------------|
| **PyMuPDF** | 1.25.5 | **High-performance PDF processing**. Fast text extraction, image handling, and metadata parsing. Handles complex PDF structures with ease |
| **PDFPlumber** | 0.11.5 | **Advanced PDF data extraction**. Specialized in extracting tables, text positioning, and layout information for structured data extraction |
| **python-docx** | 1.1.2 | **Microsoft Word document processing**. Full support for .docx format including styles, tables, and embedded content |
| **BeautifulSoup4** | 4.13.3 | **HTML/XML parsing and extraction**. Robust parsing of web content and XML documents with flexible query capabilities |
| **LlamaParse** | 0.6.4 | **AI-powered document parsing**. Uses LLMs to intelligently extract and structure content from complex documents |

### **Data Processing & Analytics**

Industry-standard data processing tools for efficient data manipulation and analysis.

| Technology | Version | Purpose & Benefits |
|------------|---------|-------------------|
| **Pandas** | 2.2.3 | **Data manipulation powerhouse**. Fast, flexible data structures for analyzing structured data. Essential for processing proposal data and metrics |
| **NumPy** | 1.26.4 | **Numerical computing foundation**. Optimized array operations and mathematical functions. Powers data transformations and calculations |
| **SQLAlchemy** | 2.0.38 | **Enterprise ORM and database toolkit**. Database-agnostic ORM supporting PostgreSQL, MySQL, SQLite. Ensures data integrity and simplifies database operations |
| **Alembic** | 1.14.1 | **Database migration management**. Version control for database schemas. Enables safe, trackable database changes across environments |

### **Observability & Monitoring**

Production-grade monitoring and observability for maintaining system health and performance.

| Technology | Version | Purpose & Benefits |
|------------|---------|-------------------|
| **OpenTelemetry** | Latest | **Industry-standard observability framework**. Distributed tracing, metrics collection, and logging. Provides deep insights into system performance and bottlenecks |
| **Rich** | 13.9.4 | **Beautiful terminal output**. Enhanced logging with colors, tables, and progress bars. Improves developer experience and debugging efficiency |

### **Authentication & Security**

Enterprise-grade security infrastructure protecting sensitive proposal data and ensuring compliance.

| Technology | Version | Purpose & Benefits |
|------------|---------|-------------------|
| **Auth0** | 4.8.0 | **Enterprise identity platform**. Supports SSO, MFA, and social logins. Compliant with SOC 2, GDPR, and other security standards |
| **PyJWT** | 2.10.1 | **JSON Web Token implementation**. Secure, stateless authentication for API endpoints. Industry-standard token-based auth |
| **Cryptography** | 44.0.1 | **Comprehensive cryptographic library**. Provides encryption, hashing, and secure key management. Protects sensitive data at rest and in transit |
| **bcrypt** | 4.2.1 | **Password hashing**. Industry-standard algorithm for secure password storage. Resistant to brute-force attacks |

### **Development & DevOps**

Modern DevOps tools ensuring consistent deployments and development efficiency.

| Technology | Version | Purpose & Benefits |
|------------|---------|-------------------|
| **Docker** | Latest | **Containerization platform**. Ensures consistent environments across development, testing, and production. Simplifies deployment and scaling |
| **Kubernetes** | 32.0.0 | **Container orchestration**. Production-ready for cloud-native deployments. Auto-scaling, self-healing, and zero-downtime deployments |
| **pytest** | Latest | **Modern testing framework**. Comprehensive test coverage with fixtures, parametrization, and plugins. Ensures code quality and reliability |
| **mypy** | Latest | **Static type checking**. Catches type errors before runtime. Improves code quality and maintainability |

### **Additional Key Technologies**

| Technology | Version | Purpose & Benefits |
|------------|---------|-------------------|
| **NLTK** | 3.9.1 | **Natural Language Processing**. Advanced text processing, tokenization, and linguistic analysis for proposal content |
| **Instructor** | 1.7.2 | **Structured LLM outputs**. Ensures LLM responses conform to predefined schemas using Pydantic models |
| **Mem0** | 0.1.49 | **Memory management for AI agents**. Provides persistent memory across agent interactions for context retention |
| **Tenacity** | 9.0.0 | **Retry logic and resilience**. Handles transient failures gracefully with configurable retry strategies |
| **tiktoken** | 0.7.0 | **Token counting for LLMs**. Accurate token estimation for cost management and context window optimization |

---

## 🏗 Architecture

### Multi-Agent System Design

The platform employs a sophisticated multi-agent architecture where specialized AI agents collaborate to handle different aspects of proposal generation:

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                        │
│                  (Async REST API)                        │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼────┐           ┌─────▼─────┐
    │ Crews   │           │ Knowledge │
    │ Engine  │◄─────────►│   Base    │
    └────┬────┘           └─────┬─────┘
         │                      │
    ┌────┴─────────────┐       │
    │                  │       │
┌───▼───┐  ┌──────▼────────┐  │
│Outline│  │  Compliance   │  │
│ Crew  │  │  Matrix Crew  │  │
└───────┘  └───────────────┘  │
                               │
┌──────────────────────────────▼──────┐
│  Vector Stores (OpenSearch/Chroma)  │
└─────────────────────────────────────┘
```

### Agent Crews

1. **Proposal Outline Crew**: Generates structured proposal outlines
2. **Compliance Matrix Crew**: Analyzes requirements and creates compliance matrices
3. **Generic Proposal Content Crew**: Generates reusable proposal content
4. **Past Performance Crew**: Retrieves and formats relevant past performance data
5. **Refine Proposal Outline Crew**: Iteratively improves proposal quality

---

## 🚀 Setup & Installation

### Prerequisites

- **Python 3.12** (Required)
- **pip** (Python package manager)
- **AWS Account** (for cloud services)
- **Docker** (optional, for containerized deployment)

### Recommended: Python Version Management

We recommend using **pyenv** for managing Python versions:

```bash
# Install pyenv (if not already installed)
curl https://pyenv.run | bash

# Install Python 3.12
pyenv install 3.12

# Set Python 3.12 for this project
pyenv local 3.12
```

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd rfp-agentic-ai-main
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with the following configuration:

```bash
# ============================================
# LLM Configuration
# ============================================
MODEL=bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0

# OpenAI Configuration (Optional)
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_KEY=<your-openai-api-key>

# ============================================
# AWS Configuration
# ============================================
AWS_ACCESS_KEY_ID=<your-aws-access-key>
AWS_SECRET_ACCESS_KEY=<your-aws-secret-key>
AWS_REGION_NAME=us-east-1
AWS_DEFAULT_REGION=us-east-1

# ============================================
# AWS Services
# ============================================
# S3 Configuration
AWS_S3_BUCKET_NAME=<your-knowledge-base-bucket>
RFP_S3_BUCKET=<your-rfp-bucket>
RFP_S3_BASE_FOLDER=<base-folder-path>
pp_bucket=<past-performance-bucket>

# OpenSearch Configuration
AWS_OPENSEARCH_COLLECTION_ARN=<opensearch-collection-arn>
AWS_OPENSEARCH_SERVERLESS_COLLECTION_HOST=<opensearch-host>

# Bedrock Configuration
AWS_KNOWLEDGE_BASE_ROLE_ARN=<knowledge-base-role-arn>
AWS_EMBEDDING_MODEL_ARN=<embedding-model-arn>
AWS_EMBEDDING_MODEL_DIMENSIONS=1024

# ============================================
# Application Configuration
# ============================================
AWS_ENVIRONMENT=Development
AWS_PROJECT=rfpai-agents
```

### AWS Setup Requirements

1. **S3 Buckets**: Create buckets for knowledge base, RFPs, and past performance data
2. **OpenSearch Serverless**: Set up an OpenSearch Serverless collection
3. **IAM Roles**: Configure appropriate IAM roles with necessary permissions
4. **Bedrock Access**: Enable Amazon Bedrock in your AWS account

---

## 🎮 Running the Application

### Development Mode

Start the FastAPI development server with auto-reload:

```bash
fastapi dev src/serve.py
```

The API will be available at: `http://localhost:8000`

### Production Mode

Run with Uvicorn for production:

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 3000 --workers 4
```

### With Gunicorn (Recommended for Production)

```bash
gunicorn src.serve:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:3000 \
  --timeout 1200
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t rfp-agentic-ai:latest .
```

### Run Docker Container

```bash
docker run -d \
  --name rfp-ai-platform \
  -p 3000:3000 \
  --env-file .env \
  rfp-agentic-ai:latest
```

### Docker Compose (Optional)

Create a `docker-compose.yml` for easier deployment:

```yaml
version: '3.8'
services:
  rfp-ai:
    build: .
    ports:
      - "3000:3000"
    env_file:
      - .env
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

---

## 📚 API Documentation

Once the application is running, access the interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Key API Endpoints

- `POST /api/proposal/outline` - Generate proposal outline
- `POST /api/compliance/matrix` - Create compliance matrix
- `POST /api/past-performance` - Retrieve past performance data
- `GET /api/health` - Health check endpoint

---

## 🤝 Support & Contact

For questions, issues, or feature requests, please contact our team or open an issue in the repository.

---

## 📄 License

[Specify your license here]

---

<div align="center">

**Built with ❤️ using cutting-edge AI technology**

*Transforming RFP responses through intelligent automation*

</div>
