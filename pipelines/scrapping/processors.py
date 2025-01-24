import asyncio
import os
import uuid
from abc import ABC, abstractmethod
from typing import List

from aiohttp import ClientSession
from loguru import logger
from models import CommonMediaDocument, PexelsItem, UnsplashItem


class BaseProcessor(ABC):
    def __init__(
        self,
        subject: str,
        start_page: int,
        num_pages: int,
        num_items_per_page: int,
        queue: asyncio.Queue,
    ):
        
        assert os.getenv("API_IMAGE_SUBJECT"), "API_IMAGE_SUBJECT is not set"
        self.queue = queue
        self._subject = subject
        self._last_page = start_page
        self._num_pages = num_pages
        self._num_items_per_page = num_items_per_page
        self._curr_page = start_page
    
    @abstractmethod
    async def fetch_data(self, page_num: int) -> List[CommonMediaDocument]:
        pass

    async def run(self) -> None:
        assert os.getenv("UNSPLASH_API_KEY") or os.getenv("PEXELS_API_KEY"), "API keys are not set"
        while self._curr_page <= self._last_page + self._num_pages:
            try:
                data: List[CommonMediaDocument] = await self.fetch_data(page_num=self._curr_page)
                for item in data:
                    await self.queue.put(item)
                self._curr_page += 1
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__}: {e}")


class UnsplashProcessor(BaseProcessor):
    PROCESSOR_NAME = "Unsplash"
    UNSPLASH_API_URL = "https://api.unsplash.com/search/photos?query={subject}&page={page}&per_page={num_items_per_page}"

    async def fetch_data(self, page_num: int) -> List[CommonMediaDocument]:
        async with ClientSession() as session:
            url = self.UNSPLASH_API_URL.format(
                subject=self._subject.replace(" ", "-"),
                page=page_num,
                num_items_per_page=self._num_items_per_page,
            )
            headers = {"Authorization": f"Client-ID {os.getenv('UNSPLASH_API_KEY')}"}
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                result = await response.json()

                documents = []
                if result:
                    items = result.get("results", [])
                    logger.info(
                        f"Received {len(items)} items from page={self._curr_page} from {self.PROCESSOR_NAME}"
                    )
                    for item in items:
                        new_item = UnsplashItem.from_json(item)
                        common_item = CommonMediaDocument.from_unsplash(item=new_item, subject=self._subject, page_number=self._curr_page)
                        documents.append(common_item)
                logger.info(f"Processed into {len(documents)}")
                return documents


class PexelsProcessor(BaseProcessor):
    PROCESSOR_NAME = "Pexels"
    PEXELS_API_URL = "https://api.pexels.com/v1/search?query={subject}&page={page}&per_page={num_items_per_page}"

    async def fetch_data(self, page_num: int) -> CommonMediaDocument:
        async with ClientSession() as session:
            url = self.PEXELS_API_URL.format(
                subject=self._subject.replace(" ", "%20"),
                page=page_num,
                num_items_per_page=self._num_items_per_page,
            )
            headers = {"Authorization": os.getenv("PEXELS_API_KEY")}
            async with session.get(
                url,
                headers=headers,
            ) as response:
                response.raise_for_status()
                result = await response.json()

                documents = []
                if result:
                    items = result.get("photos", [])
                    logger.info(
                        f"Received {len(items)} items from page={self._curr_page} from {self.PROCESSOR_NAME}"
                    )
                    for item in items:
                        new_item = PexelsItem.from_json(item)
                        common_item = CommonMediaDocument.from_pexels(item=new_item, subject=self._subject, page_number=self._curr_page)
                        documents.append(common_item)
                return documents
