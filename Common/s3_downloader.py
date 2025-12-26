import boto3
import os

def download_s3_files_new(bucket_name, s3_prefix, local_dir):
    """
    Downloads all files from a given S3 prefix to a local directory.

    Args:
        bucket_name (str): S3 bucket name.
        s3_prefix (str): Path inside the bucket (can be empty string "").
        local_dir (str): Path to save the downloaded files.
        aws_access_key (str, optional): AWS access key.
        aws_secret_key (str, optional): AWS secret key.
        aws_region (str): AWS region.
    """
    os.makedirs(local_dir, exist_ok=True)

    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        s3 = boto3.client(
            's3',
            region_name=os.getenv("AWS_REGION_NAME"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
    else:
        s3 = boto3.client('s3')

    print(f"Listing files in s3://{bucket_name}/{s3_prefix}")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)

    if 'Contents' not in response:
        print("No files found.")
        return []

    downloaded_files = []

    for obj in response['Contents']:
        key = obj['Key']
        filename = os.path.basename(key)
        if not filename:
            continue  # skip folder keys
        local_path = os.path.join(local_dir, filename)
        print(f"Downloading {key} to {local_path}")
        s3.download_file(bucket_name, key, local_path)
        downloaded_files.append(local_path)

    return downloaded_files
