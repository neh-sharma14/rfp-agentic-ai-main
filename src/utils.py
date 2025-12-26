### File: src/utils.py

import os
import shutil
import boto3
import botocore
from src.knowledge_base.text_extract import extract_text
from fastapi import UploadFile
import tempfile
from typing import Optional, List, Dict, Tuple, Union, Any
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from .knowledge_base.vector_store import KnowledgeBase

def download_s3_files(bucket: str, prefix: str, local_dir: str) -> None:
    """
    Download all files from the specified S3 bucket and prefix to a local directory.

    Args:
        bucket (str): The name of the S3 bucket.
        prefix (str): The S3 prefix under which files are stored.
        local_dir (str): The path of the local directory to save the downloaded files.

    Raises:
        FileNotFoundError: If the specified directory cannot be created.
        botocore.exceptions.ClientError: If any S3 operation fails.
        ValueError: If no files are found under the given prefix.
    """
    s3_client = boto3.client("s3")

    # If local directory exists, delete it
    if os.path.exists(local_dir):
        try:
            shutil.rmtree(local_dir)
        except OSError as e:
            raise FileNotFoundError(
                f"Failed to delete existing directory '{local_dir}': {e}"
            ) from e

    # Ensure the local directory exists
    try:
        os.makedirs(local_dir, exist_ok=True)
    except OSError as e:
        raise FileNotFoundError(
            f"Failed to create directory '{local_dir}': {e}"
        ) from e

    # List objects from S3
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    except botocore.exceptions.ClientError as e:
        # Catch any S3 listing errors
        raise botocore.exceptions.ClientError(
            {
                'Error': {
                    'Code': 'ListObjectsError',
                    'Message': (
                        f"Error listing objects from bucket '{bucket}' "
                        f"with prefix '{prefix}': {str(e)}"
                    )
                }
            },
            operation_name='ListObjectsV2'
        ) from e

    # Check if the prefix actually contains objects
    if 'Contents' not in response:
        raise ValueError(f"No files found in bucket '{bucket}' with prefix '{prefix}'.")

    # Download files one by one
    for obj in response.get('Contents', []):
        file_key = obj['Key']
        if file_key.endswith('/'):
          continue
        file_name = os.path.basename(file_key)
        local_path = os.path.join(local_dir, file_name)

        try:
            s3_client.download_file(bucket, file_key, local_path)
        except botocore.exceptions.ClientError as e:
            # Catch any S3 download errors
            raise botocore.exceptions.ClientError(
                {
                    'Error': {
                        'Code': 'DownloadFileError',
                        'Message': (
                            f"Error downloading file '{file_key}' "
                            f"to '{local_path}': {str(e)}"
                        )
                    }
                },
                operation_name='GetObject'
            ) from e

def download_s3_files_as_text(
    bucket: str,
    prefix: str,
    solicitation_id: str,
    max_workers: int = 4
) -> Dict[str, str]:
    """
    Download files from S3 and extract text from them.
    Downloads files to ./knowledge/<solicitation_id>/
    and saves text files to ./knowledge_extract_text/<solicitation_id>/

    Args:
        bucket (str): The name of the S3 bucket.
        prefix (str): The S3 prefix under which files are stored.
        solicitation_id (str): The ID of the solicitation/folder containing the files.
        max_workers (int): Maximum number of parallel processing workers. Defaults to 4.

    Returns:
        Dict[str, str]: A dictionary mapping original file names to their corresponding text file paths.

    Raises:
        FileNotFoundError: If the source directory doesn't exist.
        Exception: If text extraction fails for any file.
    """
    logger = logging.getLogger(__name__)
    processed_files = {}

    # Define source and destination directories
    source_dir = os.path.join("knowledge", solicitation_id)
    dest_dir = os.path.join("knowledge_extract_text", solicitation_id)

    # First download the files using download_s3_files
    try:
        download_s3_files(bucket, prefix, source_dir)
        logger.info(f"Successfully downloaded files to {source_dir}")
    except Exception as e:
        logger.error(f"Error downloading files: {e}")
        raise

    # Create destination directory if it doesn't exist
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory '{dest_dir}': {e}")
        raise FileNotFoundError(
            f"Failed to create directory '{dest_dir}': {e}"
        ) from e

    # Get list of files to process
    files_to_process = []
    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)
        if os.path.isfile(file_path):
            files_to_process.append(file_path)

    if not files_to_process:
        logger.warning(f"No files found in directory '{source_dir}'")
        raise ValueError(f"No files found in directory '{source_dir}'")

    logger.info(f"Found {len(files_to_process)} files to process")

    def process_single_file(file_path: str) -> tuple:
        """Process a single file and return (file_name, text_file_path) or (file_name, error)"""
        file_name = os.path.basename(file_path)
        temp_file = None

        try:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.close()

            # Copy the file to temporary location
            shutil.copy2(file_path, temp_file.name)

            # Create a FastAPI UploadFile object for text extraction
            with open(temp_file.name, 'rb') as f:
                upload_file = UploadFile(
                    filename=file_name,
                    file=f
                )

                # Extract text from the file
                text_content = extract_text(upload_file)

                # Save as .txt file
                txt_file_name = os.path.splitext(file_name)[0] + '.txt'
                txt_file_path = os.path.join(dest_dir, txt_file_name)

                with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text_content)

                logger.info(f"Successfully processed {file_name}")
                return file_name, txt_file_path

        except Exception as e:
            logger.error(f"Error processing file '{file_name}': {e}")
            return file_name, str(e)

        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except OSError as e:
                    logger.warning(f"Failed to delete temporary file {temp_file.name}: {e}")

    # Process files in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_file, file_path) for file_path in files_to_process]

        for future in tqdm(as_completed(futures), total=len(files_to_process), desc="Processing files"):
            file_name, result = future.result()
            if isinstance(result, str):  # Error occurred
                processed_files[file_name] = f"Error: {result}"
            else:  # Success
                processed_files[file_name] = result

    # Log summary
    success_count = sum(1 for result in processed_files.values() if not result.startswith("Error:"))
    error_count = len(processed_files) - success_count

    logger.info(f"Processing complete. Successfully processed {success_count} files, {error_count} errors.")

    return processed_files

def prepare_vectordb_from_s3(bucket, solicitation_id, rfp_title=None):
    """
    Download PDFs from S3, add them to the KnowledgeBase, and return processed_files, errors, and the local knowledge path.
    """
    local_knowledge_path = f"./knowledge/{solicitation_id}"
    os.makedirs(local_knowledge_path, exist_ok=True)

    # Download PDFs
    download_s3_files(
        bucket=bucket,
        prefix=f"/rfp/{solicitation_id}",
        local_dir=local_knowledge_path
    )

    # Add PDFs to KnowledgeBase
    kb = KnowledgeBase()
    pdf_files = [f for f in os.listdir(local_knowledge_path) if os.path.isfile(os.path.join(local_knowledge_path, f))]
    processed_files = []
    errors = []
    for pdf_file in pdf_files:
        try:
            file_path = os.path.join(local_knowledge_path, pdf_file)
            with open(file_path, 'rb') as f:
                metadata = {
                    "filename": pdf_file,
                    "solicitation_id": solicitation_id,
                    "rfp_title": rfp_title or "",
                    "user_id": "system",
                    "file_id": pdf_file,
                    "upload_timestamp": int(time.time()),
                    "document_type": "rfp",
                }
                doc_id = kb.add_file(f, pdf_file, metadata)
                processed_files.append({
                    "filename": pdf_file,
                    "doc_id": doc_id
                })
        except Exception as e:
            errors.append({
                "filename": pdf_file,
                "error": str(e)
            })
    return processed_files, errors, local_knowledge_path
