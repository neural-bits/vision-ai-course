import asyncio
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

import dotenv
from aiohttp import ClientSession
from loguru import logger
from models import CommonMediaDocument, PexelsItem, UnsplashItem
from tenacity import retry, stop_after_attempt, wait_exponential

ROOT = Path(__file__).parent.parent

# Environment variables
dotenv.load_dotenv(str(ROOT / ".env"))


class BaseProcessor(ABC):
    def __init__(
        self,
        subject: str,
        num_pages: int,
        num_items_per_page: int,
        queue: asyncio.Queue,
        kwargs: Dict[str, Any] = {},
    ):
        self.queue = queue
        self.subject = subject
        self.num_pages = num_pages
        self.num_items_per_page = num_items_per_page
        self.image_params = {
            "width": kwargs.get("width", 1920),
            "height": kwargs.get("height", 1080),
        }
        self.sleep_window = 5

    @abstractmethod
    async def fetch_data(self) -> List[CommonMediaDocument]:
        pass

    async def run(self) -> None:
        while True:
            try:
                data: List[CommonMediaDocument] = await self.fetch_data()
                for item in data:
                    await self.queue.put(item)
                await asyncio.sleep(self.sleep_window)
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__}: {e}")


class UnsplashProcessor(BaseProcessor):
    PROCESSOR_NAME = "Unsplash"
    UNSPLASH_API_URL = "https://api.unsplash.com/search/photos?query={subject}&width={im_width}&height={im_height}&page={num_pages}&per_page={num_items_per_page}"

    async def fetch_data(self) -> List[CommonMediaDocument]:
        async with ClientSession() as session:
            url = self.UNSPLASH_API_URL.format(
                subject=self.subject,
                im_width=self.image_params["width"],
                im_height=self.image_params["height"],
                num_pages=self.num_pages,
                num_items_per_page=self.num_items_per_page,
            )
            headers = {"Authorization": f"Client-ID {os.getenv('UNSPLASH_API_KEY')}"}
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                result = (await response.json()).get("results", [])

                documents = []
                if result:
                    logger.info(
                        f"Received {len(result)} items from {self.PROCESSOR_NAME}"
                    )
                    for res in result:
                        new_item = UnsplashItem.from_json(res)
                        common_item = CommonMediaDocument.from_unsplash(new_item)
                        documents.append(common_item)
                logger.info(f"Processed into {len(documents)}")
                return documents


class PexelsProcessor(BaseProcessor):
    PROCESSOR_NAME = "Pexels"
    PEXELS_API_URL = "https://api.pexels.com/v1/search?query={subject}&width={im_width}&height={im_height}&page={num_pages}&per_page={num_items_per_page}"

    async def fetch_data(self) -> CommonMediaDocument:
        async with ClientSession() as session:
            url = self.PEXELS_API_URL.format(
                subject=self.subject,
                num_pages=self.num_pages,
                num_items_per_page=self.num_items_per_page,
                im_width=self.image_params["width"],
                im_height=self.image_params["height"],
            )
            headers = {"Authorization": os.getenv("PEXELS_API_KEY")}
            async with session.get(
                url,
                headers=headers,
            ) as response:
                response.raise_for_status()
                result = await response.json()
                # TODO: implement error handling + group images by page in pydantic model

                documents = []
                if result:
                    logger.info(
                        f"Received {len(result)} items from {self.PROCESSOR_NAME}"
                    )

                    for res in result:
                        new_item = PexelsItem.from_json(res)
                        common_item = CommonMediaDocument.from_pexels(new_item)
                        documents.append(common_item)
                logger.info(f"Processed into {len(documents)}")
                return documents


class ConsumerWorker:
    def __init__(self, queue: asyncio.Queue):
        self.consumer_id = uuid.uuid4()
        self.queue = queue
        logger.info(f"Consumer [{self.consumer_id}] created")

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(2),
    )
    async def process_item(self, item: CommonMediaDocument) -> None:
        try:
            CommonMediaDocument.model_validate(item)
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return

    async def run(self) -> None:
        while True:
            item = await self.queue.get()
            logger.info(
                f"Consumer [{self.consumer_id}] received item - {item.image_url}"
            )
            try:
                await self.process_item(item)
                self.queue.task_done()
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
