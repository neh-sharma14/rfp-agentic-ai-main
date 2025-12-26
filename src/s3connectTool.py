from typing import Type, Any
import os
import json

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import boto3
from botocore.exceptions import ClientError

class S3WriterToolInput(BaseModel):
    """Input schema for S3WriterTool."""
    file_path: str = Field(..., description="S3 file path (e.g., 's3://bucket-name/file-name')")
    content: Any = Field(..., description="Content to write to the file")

class S3WriterTool(BaseTool):
    name: str = "S3 Writer Tool"
    description: str = "Writes content to a file in Amazon S3 given an S3 file path"
    args_schema: Type[BaseModel] = S3WriterToolInput

    def _run(self, file_path: str, content: Any) -> str:
        try:
            bucket_name, object_key = self._parse_s3_path(file_path)

            s3 = boto3.client(
                's3',
                region_name=os.getenv('AWS_REGION_NAME'),
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )

            # Improved serialization method
            if isinstance(content, BaseModel):
                # If it's a Pydantic model, use model_dump_json()
                serialized_content = content.model_dump_json(indent=2)
            elif isinstance(content, (dict, list)):
                # If it's a dict or list, use json.dumps()
                serialized_content = json.dumps(content, indent=2)
            else:
                # For other types, convert to string
                serialized_content = str(content)

            s3.put_object(
                Bucket=bucket_name, 
                Key=object_key, 
                Body=serialized_content.encode('utf-8'),
                ContentType='application/json'
            )
            return f"Successfully wrote content to {file_path}"
        except Exception as e:
            return f"Error writing file to S3: {str(e)}"

    def _parse_s3_path(self, file_path: str) -> tuple:
        parts = file_path.replace("s3://", "").split("/", 1)
        return parts[0], parts[1]