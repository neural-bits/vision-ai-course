import os
from datetime import datetime, timedelta
from pathlib import Path

import dotenv
import pymongo
from loguru import logger

from pipelines.common.models import RawMediaMongoDocument

ROOT_PATH = Path(__file__).parent.parent
dotenv.load_dotenv(str(ROOT_PATH / ".env"))


# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME")
RAW_DATA_COLLECTION = os.getenv("RAW_DATA_COLLECTION")
CLEAN_DATA_COLLECTION = os.getenv("CLEAN_DATA_COLLECTION")
DATASET_COLLECTION = os.getenv("DATASET_COLLECTION")


logger = logger.bind(name="MongoConnector")


class MongoConnector:
    def __init__(self):
        self._client = pymongo.MongoClient(MONGO_URI)
        if not self._client:
            raise ConnectionError("Could not connect to MongoDB")

        # Check for db or create it
        if DB_NAME not in self._client.list_database_names():
            self._db = self._client[DB_NAME]
            logger.info(f"Created MongoDB database: {DB_NAME}")

        # Check for collections or create them
        # TODO: move this to call only once
        self._db = self._client[DB_NAME]
        self._collections = self._db.list_collection_names()
        if RAW_DATA_COLLECTION not in self._collections:
            self._db.create_collection(RAW_DATA_COLLECTION)
            logger.info(f"Created MongoDB collection: {RAW_DATA_COLLECTION}")
        if CLEAN_DATA_COLLECTION not in self._collections:
            self._db.create_collection(CLEAN_DATA_COLLECTION)
            logger.info(f"Created MongoDB collection: {CLEAN_DATA_COLLECTION}")
        if DATASET_COLLECTION not in self._collections:
            self._db.create_collection(DATASET_COLLECTION)
            logger.info(f"Created MongoDB collection: {DATASET_COLLECTION}")

    def insert_raw_metadata(self, mongo_media_item: "RawMediaMongoDocument") -> None:
        collection = self._db[RAW_DATA_COLLECTION]
        collection.insert_one(mongo_media_item.model_dump(by_alias=True, exclude=["id"]))
        logger.info(f"Inserted data: {mongo_media_item}")

    def insert_clean_metadata(self, mongo_media_item: "RawMediaMongoDocument") -> None:
        collection = self._db[CLEAN_DATA_COLLECTION]
        collection.insert_one(mongo_media_item.model_dump(by_alias=True, exclude=["id"]))
        logger.info(f"Inserted data: {mongo_media_item}")

    def batch_read_all_on_date(self, batch_date: str) -> list["RawMediaMongoDocument"]:
        collection = self._db[RAW_DATA_COLLECTION]
        raw_data = collection.find({"batch_date": batch_date})
        return [raw for raw in raw_data]

    def batch_raw_records_by_subject_and_date(self, batch_date: str, subject: str) -> list["RawMediaMongoDocument"]:
        collection = self._db[RAW_DATA_COLLECTION]
        start_date = datetime.strptime(batch_date, "%Y-%m-%d")
        end_date = start_date + timedelta(days=1)

        query = {"created_at": {"$gte": start_date, "$lt": end_date}, "subject": subject}
        raw_data = collection.find(query)
        return [RawMediaMongoDocument.from_json(**raw) for raw in raw_data]

    def close(self):
        self.client.close()
        logger.info("Closed MongoDB connection")
