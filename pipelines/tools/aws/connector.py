import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

import boto3
import dotenv
from botocore.exceptions import ClientError
from loguru import logger

ROOT_PATH = Path(__file__).parent.parent.parent
dotenv.load_dotenv(str(ROOT_PATH / ".env"))

logger = logger.bind(name="S3Connector")
# S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_DATALAKE_BUCKET = os.getenv("AWS_S3_DATALAKE_BUCKET")
AWS_S3_DATAWAREHOUSE_BUCKET = os.getenv("AWS_S3_DATAWAREHOUSE_BUCKET")


class S3BucketConnector(ABC):
    """Abstract base class for S3 bucket operations"""

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self._s3 = boto3.resource("s3")
        self._bucket = self._s3.Bucket(self.bucket_name)

    @abstractmethod
    def upload_fileobj(self, file_obj: BinaryIO, s3_key: str) -> None:
        """Upload file-like object to S3 bucket"""
        pass

    @abstractmethod
    def download_fileobj(self, s3_key: str) -> BinaryIO:
        """Download file from S3 bucket"""
        pass


class DataLakeConnector(S3BucketConnector):
    """Handles operations with the raw data lake bucket"""

    def __init__(self):
        super().__init__(AWS_S3_DATALAKE_BUCKET)

    def upload_fileobj(self, file_obj: BinaryIO, s3_key: str) -> None:
        try:
            self._bucket.upload_fileobj(
                Fileobj=file_obj, Key=s3_key, ExtraArgs={"ContentType": "application/octet-stream"}
            )
            logger.info(f"Uploaded raw file to datalake: s3://{self.bucket_name}/{s3_key}")
        except ClientError as e:
            logger.error(f"Failed to upload to datalake: {e}")
            raise

    def download_fileobj(self, s3_key: str) -> BinaryIO:
        # Implement download logic if needed
        pass


class DataWarehouseConnector(S3BucketConnector):
    """Handles operations with the processed data warehouse bucket"""

    def __init__(self):
        super().__init__(AWS_S3_DATAWAREHOUSE_BUCKET)

    def upload_fileobj(self, file_obj: BinaryIO, s3_key: str) -> None:
        try:
            self._bucket.upload_fileobj(
                Fileobj=file_obj, Key=s3_key, ExtraArgs={"ContentType": "application/octet-stream"}
            )
            logger.info(f"Uploaded processed file to warehouse: s3://{self.bucket_name}/{s3_key}")
        except ClientError as e:
            logger.error(f"Failed to upload to warehouse: {e}")
            raise

    def download_fileobj(self, s3_key: str) -> BinaryIO:
        # Implement download logic if needed
        pass


class S3DataManager:
    """Orchestrates operations between data lake and data warehouse"""

    def __init__(self):
        self.datalake = DataLakeConnector()
        self.warehouse = DataWarehouseConnector()
        self.s3_client = boto3.client("s3")

    def move_to_warehouse(self, source_key: str, dest_key: str) -> None:
        """Move file from datalake to warehouse using server-side copy"""
        try:
            copy_source = {"Bucket": self.datalake.bucket_name, "Key": source_key}
            self.s3_client.copy_object(CopySource=copy_source, Bucket=self.warehouse.bucket_name, Key=dest_key)
            logger.info(
                f"Moved s3://{self.datalake.bucket_name}/{source_key} to s3://{self.warehouse.bucket_name}/{dest_key}"
            )

            # Delete source file after successful copy
            self.s3_client.delete_object(Bucket=self.datalake.bucket_name, Key=source_key)
        except ClientError as e:
            logger.error(f"Failed to move file: {e}")
            raise

    def get_file(self, s3_key: str, bucket: str = "datalake") -> BinaryIO:
        """Get file from specified bucket"""
        connector = self.datalake if bucket == "datalake" else self.warehouse
        return connector.download_fileobj(s3_key)
