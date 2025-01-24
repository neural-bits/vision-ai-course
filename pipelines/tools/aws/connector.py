import os
from pathlib import Path

import boto3
import dotenv
from botocore.exceptions import ClientError
from loguru import logger

ROOT_PATH = Path(__file__).parent.parent
dotenv.load_dotenv(str(ROOT_PATH / ".env"))

# S3 Configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET", "raw-media-data")


class S3Connector:
    def __init__(self):
        # assert AWS_ACCESS_KEY or AWS_SECRET_ACCESS_KEY, "AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY is not set"
        self._s3 = boto3.resource("s3")
        self._bucket_name: str = f"amzn-s3-edgecv-{S3_BUCKET}"
        self._bucket = self._s3.Bucket(name=self._bucket_name)

        logger.info(f"Connected to S3 bucket: {self._bucket_name}")
        try:
            if self._bucket.creation_date:
                logger.info(f"Bucket {self._bucket_name} exists.")
            else:
                self._bucket.create(CreateBucketConfiguration={"LocationConstraint": "eu-west-3"})
                logger.info(f"Created demo bucket named {self._bucket.name}.")
        except ClientError as err:
            print(f"Tried and failed to create demo bucket {self._bucket}.")
            print(f"\t{err.response['Error']['Code']}:{err.response['Error']['Message']}")
            return

    def upload(self, local_file_path: str, remote_file_path: str) -> None:
        self._bucket.upload_file(local_file_path, remote_file_path)
        logger.info(f"Uploaded file: {local_file_path} to {remote_file_path}")

    def download(self):
        pass

    def close(self):
        pass
