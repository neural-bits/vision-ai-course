import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

import dotenv
from aiohttp import ClientSession
from common.models import CommonMediaDocument, PexelsItem, UnsplashItem
from loguru import logger
from prefect import flow, task
from tools.connectors.mongo import DataStage, MongoConnector

dotenv.load_dotenv(
    "/home/razvantalexandru/Documents/Projects/NeuralBits/vision-ai-course/orchestrator/.secrets/.env-scrapping.env"
)
dotenv.load_dotenv(
    "/home/razvantalexandru/Documents/Projects/NeuralBits/vision-ai-course/orchestrator/.secrets/.env-db.env"
)

print(os.environ)


class APIType(Enum):
    UNSPLASH = "UNSPLASH"
    PEXELS = "PEXELS"


API_TYPES = [APIType.UNSPLASH, APIType.PEXELS]


@dataclass
class APIEndpoint:
    api: APIType

    def construct_URL_and_headers(self, subject: str, page: int, num_items_per_page: int) -> str:
        if self.api == APIType.UNSPLASH:
            _url_template = (
                "https://api.unsplash.com/search/photos?query={subject}&page={page}&per_page={num_items_per_page}"
            )
            _auth_headers = {"Authorization": f"Client-ID {os.getenv('UNSPLASH_API_KEY')}"}
            return _url_template.format(
                subject=subject.replace(" ", "-"),
                page=page,
                num_items_per_page=num_items_per_page,
            ), _auth_headers
        elif self.api == APIType.PEXELS:
            _url_template = "https://api.pexels.com/v1/search?query={subject}&page={page}&per_page={num_items_per_page}"
            _auth_headers = {"Authorization": os.getenv("PEXELS_API_KEY")}
            return _url_template.format(
                subject=subject.replace(" ", "%20"),
                page=page,
                num_items_per_page=num_items_per_page,
            ), _auth_headers


@task(retries=3, retry_delay_seconds=60)
async def fetch_metadata_from_api(apitype: APIType, subject: str, page_num: int, num_items_per_page: int):
    api_endpoint = APIEndpoint(api=apitype)
    url, headers = api_endpoint.construct_URL_and_headers(subject, page_num, num_items_per_page)
    async with ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            result = await response.json()
            result["api"] = apitype.name
            return result


@task(retries=3, retry_delay_seconds=60)
def process_metadata_from_api(result: Dict[str, Any], subject: str, page_num: int) -> List[CommonMediaDocument]:
    documents = []
    if APIType(result["api"]) == APIType.PEXELS:
        if result and result.get("photos"):
            logger.info(f"Received [{APIType.PEXELS.name}] = {len(result.get('photos'))} entries.")
            for item in result["photos"]:
                try:
                    new_item = PexelsItem.from_json(item)
                    common_item = CommonMediaDocument.from_pexels(item=new_item, subject=subject, page_number=page_num)
                    documents.append(common_item)
                except Exception as e:
                    logger.error(f"Error processing Pexels item: {e}")
                    continue
    elif APIType(result["api"]) == APIType.UNSPLASH:
        if result and result.get("results"):
            logger.info(f"Received [{APIType.UNSPLASH.name}] = {len(result.get('results'))} entries.")
            for item in result["results"]:
                try:
                    new_item = UnsplashItem.from_json(item)
                    common_item = CommonMediaDocument.from_unsplash(
                        item=new_item, subject=subject, page_number=page_num
                    )
                    documents.append(common_item)
                except Exception as e:
                    logger.error(f"Error processing Unsplash item: {e}")
                    continue
    return documents


@task(retries=3, retry_delay_seconds=60)
def insert_raw_metadata_into_db(documents: List[CommonMediaDocument]):
    connector = MongoConnector()

    for document in documents:
        connector.insert_one_document(document, DataStage.RAW)


@task(retries=3, retry_delay_seconds=60)
def fetch_concurrently(subject: str, page_num: int, num_items_per_page: int):
    tasks = []
    for api in API_TYPES:
        tasks.append(fetch_metadata_from_api.submit(api, subject, page_num, num_items_per_page))
    return [t.result() for t in tasks]


@flow(retries=3, retry_delay_seconds=5, log_prints=True)
def run_flow():
    subject = "nature"
    num_pages = 2
    num_items_per_page = 2
    logger.info(f"Doing stuff: {subject}:{num_pages}:{num_items_per_page}")

    documents = []

    results = fetch_concurrently(subject, num_pages, num_items_per_page)
    logger.info("Got results from API")
    for result in results:
        processed = process_metadata_from_api(result, subject, num_pages)
        documents.extend(processed)

    insert_raw_metadata_into_db(documents)
    logger.info("Inserted raw metadata into MongoDB")


if __name__ == "__main__":
    run_flow()
