import os
from pathlib import Path

import boto3
import dotenv
from loguru import logger

ROOT_PATH = Path(__file__).parent.parent
dotenv.load_dotenv(str(ROOT_PATH / ".env"))

# S3 Configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")


class S3Connector:
    def __init__(self):
        assert AWS_ACCESS_KEY or AWS_SECRET_ACCESS_KEY, "AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY is not set"
        self._session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        self._bucket = self._session.resource("s3").Bucket(S3_BUCKET)

    def upload(self, local_file_path: str, remote_file_path: str) -> None:
        pass

    def download(self):
        pass

    def close(self):
        pass
