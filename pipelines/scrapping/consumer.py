import asyncio
import uuid
from typing import TYPE_CHECKING

from loguru import logger
from models import RawMediaMongoDocument

from pipelines.tools.mongo.connector import MongoConnector

logger = logger.bind(name="APIConsumer")

if TYPE_CHECKING:
    from models import CommonMediaDocument


class ConsumerWorker:
    def __init__(self, queue: asyncio.Queue):
        self.consumer_id = uuid.uuid4()
        self.db_connector = MongoConnector()
        self.queue = queue
        logger.info(f"Consumer [{self.consumer_id}] created")

    async def process_item(self, item: "CommonMediaDocument") -> None:
        try:
            logger.info(f"Processing item: {item.image_url}")
            mongo_media_item = RawMediaMongoDocument.from_commondoc(item)
            self.db_connector.insert_raw_metadata(mongo_media_item)
            logger.info(f"Inserted item: {item.image_url}")
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return

    async def run(self) -> None:
        while True:
            item = await self.queue.get()
            logger.info(f"Consumer [{self.consumer_id}] received item - {item.image_url}")
            try:
                await self.process_item(item)
                self.queue.task_done()
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
            if self.queue.empty():
                break
