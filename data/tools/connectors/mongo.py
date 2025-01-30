from datetime import datetime, timedelta

import pymongo
from common import settings
from common.models import RawMediaMongoDocument
from loguru import logger

# MongoDB Configuration
config = settings.get_mongo_settings()

logger = logger.bind(name="MongoConnector")


class MongoConnector:
    def __init__(self, settings: settings.MongoSettings):
        self._client = pymongo.MongoClient(settings.mongo_uri)
        self._db_name = settings.mongo_db_name
        self._raw_colname = settings.raw_data_collection
        self._clean_colname = settings.clean_data_collection
        self._dataset_colname = settings.dataset_collection

        if not self._client:
            raise ConnectionError("Could not connect to MongoDB")

        self._db = self._client[self._db_name]
        if self._db_name not in self._client.list_database_names():
            self._db = self._client[settings.mon]
            logger.info(f"Created MongoDB database: {self._db_name}")
        self._collections = self._db.list_collection_names()

        if self._raw_colname not in self._collections:
            self._db.create_collection(self._raw_colname)
            logger.info(f"Created MongoDB collection: {self._raw_colname}")
        if self._clean_colname not in self._collections:
            self._db.create_collection(self._clean_colname)
            logger.info(f"Created MongoDB collection: {self._clean_colname}")
        if self._dataset_colname not in self._collections:
            self._db.create_collection(self._dataset_colname)
            logger.info(f"Created MongoDB collection: {self._dataset_colname}")

    def insert_raw_metadata(self, mongo_media_item: "RawMediaMongoDocument") -> None:
        collection = self._db[self._raw_colname]
        collection.insert_one(mongo_media_item.model_dump(by_alias=True, exclude=["id"]))
        logger.info(f"Inserted data: {mongo_media_item}")

    def insert_clean_metadata(self, mongo_media_item: "RawMediaMongoDocument") -> None:
        collection = self._db[self._raw_colname]
        collection.insert_one(mongo_media_item.model_dump(by_alias=True, exclude=["id"]))
        logger.info(f"Inserted data: {mongo_media_item}")

    def batch_read_all_on_date(self, batch_date: str) -> list["RawMediaMongoDocument"]:
        collection = self._db[self._raw_colname]
        raw_data = collection.find({"batch_date": batch_date})
        return [raw for raw in raw_data]

    def batch_raw_records_by_subject_and_date(self, batch_date: str, subject: str) -> list["RawMediaMongoDocument"]:
        collection = self._db[self._raw_colname]
        start_date = datetime.strptime(batch_date, "%Y-%m-%d")
        end_date = start_date + timedelta(days=1)

        query = {"created_at": {"$gte": start_date, "$lt": end_date}, "subject": subject}
        raw_data = collection.find(query).limit(10)
        parsed_data = [RawMediaMongoDocument.from_json(**raw) for raw in raw_data]
        logger.info(f"Fetched {len(parsed_data)} records for {subject} on {batch_date}")
        return parsed_data

    def close(self):
        self.client.close()
        logger.info("Closed MongoDB connection")
