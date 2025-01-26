from io import BytesIO

import requests
from checks import transform_to_png, validate_image
from common import settings
from loguru import logger
from PIL import Image
from tools.aws.connector import S3DataManager
from tools.mongo.connector import MongoConnector

logger = logger.bind(name="StoragePipeline")

BATCH_DATE = "2025-01-24"
SUBJECT = "bottle"

# For specific topic, fetch all entries from mongo from that date
mongo = MongoConnector(settings=settings.get_mongo_settings())
s3man = S3DataManager(settings=settings.get_s3_settings())
records = mongo.batch_raw_records_by_subject_and_date(batch_date=BATCH_DATE, subject=SUBJECT)

if records:
    for entry in records:
        # Step 1: Check at header level
        is_image_valid = validate_image(entry.image_url)
        if not is_image_valid:
            continue
        # Step 2: Check at content level
        data = requests.get(entry.image_url).content

        try:
            img = Image.open(BytesIO(data))
            # TODO: verify closes the file descriptor, we need to reopen the image for transforming to png
            # img.verify()
            resized = img.resize([1080, 1920])
            as_png = transform_to_png(resized)

            # Upload to datalake
            s3man.datalake.upload_fileobj(as_png, f"raw_data/{SUBJECT}/{BATCH_DATE}/{entry.id}.png")
        except (IOError, SyntaxError) as e:
            raise ValueError(f"Corrupted image file: {e}")
else:
    logger.info(f"No records found for {SUBJECT} on {BATCH_DATE}")
