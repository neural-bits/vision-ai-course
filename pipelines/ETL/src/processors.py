import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import structlog
from aiohttp import ClientSession
from constants import PEXELS_API_URL, UNSPLASH_API_URL
from dotenv import load_dotenv
from models import MediaMetadata
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import ValidationError
from pydantic_settings import BaseSettings
from tenacity import before_log, retry, stop_after_attempt, wait_exponential

load_dotenv("./pipelines/ETL/.env")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()


# Configuration
class Config(BaseSettings):
    unsplash_api_key: str
    pexels_api_key: str
    mongo_uri: str

    class Config:
        env_file = ".env"


config = Config()

# Database and Blob Client
mongo_client = AsyncIOMotorClient(config.mongo_uri)
db = mongo_client["media_pipeline"]
collection = db["media_metadata"]


class BaseProcessor(ABC):
    def __init__(self, topic: str, queue: asyncio.Queue):
        self.topic = topic
        self.queue = queue

    @abstractmethod
    async def fetch_data(self) -> List[Dict[str, Any]]:
        pass

    async def run(self) -> None:
        while True:
            try:
                data = await self.fetch_data()
                for item in data:
                    await self.queue.put(item)
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__}: {e}")
                await asyncio.sleep(5)


class UnsplashProcessor(BaseProcessor):
    async def fetch_data(self) -> List[Dict[str, Any]]:
        async with ClientSession() as session:
            url = PEXELS_API_URL.format(topic=self.topic)
            headers = {"Authorization": f"Client-ID {config.unsplash_api_key}"}
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                return (await response.json()).get("results", [])


# Pexels Processor
class PexelsProcessor(BaseProcessor):
    async def fetch_data(self) -> List[Dict[str, Any]]:
        async with ClientSession() as session:
            url = PEXELS_API_URL.format(topic=self.topic)
            headers = {"Authorization": config.pexels_api_key}
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                return (await response.json()).get("photos", [])


# Consumer Worker
class ConsumerWorker:
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(5),
        before=before_log(logger, logging.INFO),
    )
    async def upload_to_blob_storage(self, url: str, file_name: str) -> str:
        async with ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                blob_client = container_client.get_blob_client(file_name)
                await blob_client.upload_blob(await response.read(), overwrite=True)
                logger.info(f"Uploaded {file_name} to blob storage")
                return f"{config.azure_container_name}/{file_name}"

    async def process_item(self, item: Dict[str, Any]) -> None:
        try:
            # Validate metadata
            metadata = MediaMetadata.from_dict(item)
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            return

        # Save metadata to MongoDB
        await collection.insert_one(metadata.as_dict())
        logger.info(f"Saved metadata for {metadata.id}")

    async def run(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                await self.process_item(item)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
            finally:
                self.queue.task_done()


# Main Entry Point
async def main():
    queue = asyncio.Queue()
    topics = ["bear"]

    # Create processors
    processors = [PexelsProcessor(topic, queue) for topic in topics]

    # Run processors and consumers
    consumers = [ConsumerWorker(queue) for _ in range(1)]  # Number of consumers

    tasks = [asyncio.create_task(processor.run()) for processor in processors] + [
        asyncio.create_task(consumer.run()) for consumer in consumers
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Shutting down gracefully...")
    finally:
        await queue.join()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted. Exiting...")
