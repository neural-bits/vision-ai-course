import asyncio
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import dotenv
import structlog
from aiohttp import ClientSession
from models import MediaMetadata
from motor.motor_asyncio import AsyncIOMotorClient
from tenacity import before_log, retry, stop_after_attempt, wait_exponential

ROOT = Path(__file__).parent.parent

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

# Environment variables
dotenv.load_dotenv(str(ROOT / "data" / ".env"))


class BaseProcessor(ABC):
    def __init__(self, subject: str, queue: asyncio.Queue):
        self.subject = subject
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


class FreepikProcessor(BaseProcessor):
    FREEPIK_API_URL = "https://api.freepik.com/v1/search?query={subject}"

    async def fetch_data(self) -> List[Dict[str, Any]]:
        async with ClientSession() as session:
            url = self.FREEPIK_API_URL.format(subject=self.subject)
            headers = {"Authorization": os.getenv("FREEPIK_API_KEY")}
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                return (await response.json()).get("photos", [])


class PixaBayProcessor(BaseProcessor):
    PIXABAY_API_URL = "https://pixabay.com/api/?key={key}&q={subject}"

    async def fetch_data(self) -> List[Dict[str, Any]]:
        async with ClientSession() as session:
            url = self.PIXABAY_API_URL.format(subject=self.subject)
            headers = {"Authorization": os.getenv("PIXABAY_API_KEY")}
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                return (await response.json()).get("photos", [])


class UnsplashProcessor(BaseProcessor):
    UNSPLASH_API_URL = "https://api.unsplash.com/search/photos?query={subject}"

    async def fetch_data(self) -> List[Dict[str, Any]]:
        async with ClientSession() as session:
            url = self.UNSPLASH_API_URL.format(subject=self.subject)
            headers = {"Authorization": f"Client-ID {os.getenv('UNSPLASH_API_KEY')}"}
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                return (await response.json()).get("results", [])


class PexelsProcessor(BaseProcessor):
    PEXELS_API_URL = "https://api.pexels.com/v1/search?query={subject}"

    async def fetch_data(self) -> List[Dict[str, Any]]:
        async with ClientSession() as session:
            url = self.PEXELS_API_URL.format(subject=self.subject)
            headers = {"Authorization": os.getenv("PEXELS_API_KEY")}
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                return (await response.json()).get("photos", [])


class ConsumerWorker:
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(5),
        before=before_log(logger, logging.INFO),
    )
    async def upload_to_blob_storage(self, url: str, file_name: str) -> str:
        # TODO: this should stay in another module, blob layer
        async with ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                blob_client = container_client.get_blob_client(file_name)
                await blob_client.upload_blob(await response.read(), overwrite=True)
                logger.info(f"Uploaded {file_name} to blob storage")
                return f"{config.azure_container_name}/{file_name}"

    async def process_item(self, item: Dict[str, Any]) -> None:
        try:
            metadata = MediaMetadata.from_dict(item)
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return

        # await collection.insert_one(metadata.as_dict())
        # logger.info(f"Saved metadata for {metadata.id}")

    async def run(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                await self.process_item(item)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
            finally:
                self.queue.task_done()


async def main():
    subject = "animals"
    queue = asyncio.Queue()
    processors = [
        # FreepikProcessor(subject, queue),
        # PixaBayProcessor(subject, queue),
        # UnsplashProcessor(subject, queue),
        PexelsProcessor(subject, queue),
    ]

    consumers = [ConsumerWorker(queue) for _ in range(1)]

    tasks = [asyncio.create_task(processor.run()) for processor in processors] + [
        asyncio.create_task(consumer.run()) for consumer in consumers
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Shutting down...")
    finally:
        await queue.join()


if __name__ == "__main__":
    # FIXME: i don't like this, adapt to fire in the future
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted. Exiting...")
