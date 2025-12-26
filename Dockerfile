# Use official Python image
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --force-reinstall -r requirements.txt
RUN pip install --upgrade "pydantic>=2" gunicorn uvicorn  # Use Pydantic v2 to match crewai

# Copy the project files
COPY . .

# Expose FastAPI default port
EXPOSE 3000

# Default command to run FastAPI with Gunicorn and multiple workers
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "4", "--timeout-keep-alive", "1200"]
