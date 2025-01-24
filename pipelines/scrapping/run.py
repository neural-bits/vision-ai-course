import asyncio
import os
from pathlib import Path

import dotenv
from loguru import logger
from processors import ConsumerWorker, PexelsProcessor, UnsplashProcessor

ROOT_PATH = Path(__file__).parent
dotenv.load_dotenv(str(ROOT_PATH / ".env"))

IMAGES_SUBJECT: str = os.getenv("API_IMAGE_SUBJECT")
NUM_PAGES: int = int(os.getenv("API_IMAGE_NUM_PAGES", 1))
NUM_ITEMS_PER_PAGE: int = int(os.getenv("API_IMAGE_NUM_ITEMS_PER_PAGE", 1))


async def main():
    shared_queue = asyncio.Queue()
    processors = [
        # UnsplashProcessor(
        #     subject=IMAGES_SUBJECT,
        #     num_pages=NUM_PAGES,
        #     num_items_per_page=NUM_ITEMS_PER_PAGE,
        #     queue=shared_queue,
        # ),
        PexelsProcessor(
            subject=IMAGES_SUBJECT,
            num_pages=NUM_PAGES,
            num_items_per_page=NUM_ITEMS_PER_PAGE,
            queue=shared_queue,
        ),
    ]

    consumers = [ConsumerWorker(queue=shared_queue) for _ in range(1)]

    tasks = [asyncio.create_task(processor.run()) for processor in processors] + [
        asyncio.create_task(consumer.run()) for consumer in consumers
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Shutting down...")
    finally:
        await shared_queue.join()


if __name__ == "__main__":
    # FIXME: i don't like this, adapt to fire in the future
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted. Exiting...")
        logger.info("Pipeline interrupted. Exiting...")
        logger.info("Pipeline interrupted. Exiting...")
