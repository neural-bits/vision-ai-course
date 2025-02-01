from io import BytesIO

import requests
import transforms
import validations
from loguru import logger
from PIL import Image
from tools.common import settings

from data.tools.connectors.aws import S3DataManager
from data.tools.connectors.mongo import MongoConnector

logger = logger.bind(name="StoragePipeline")

BATCH_DATE = "2025-01-24"
SUBJECT = "bottle"


def process_raw_records_by_subject_and_date(batch_date: str, subject: str):
    mongo = MongoConnector(settings=settings.get_mongo_settings())
    s3man = S3DataManager(settings=settings.get_s3_settings())
    records = mongo.batch_raw_records_by_subject_and_date(batch_date=batch_date, subject=subject)

    if records:
        for entry in records:
            is_valid_url = validations._is_valid_url(entry.image_url)
            if is_valid_url:
                url_header = requests.head(entry.image_url)
                is_valid_image_header = validations._has_valid_image_header(url_header)
                is_valid_image_size = validations._is_valid_image_size(url_header)
                if is_valid_image_header and is_valid_image_size:
                    data = requests.get(entry.image_url).content
                    image = transforms.deserialize_image(data=data)
                    image = transforms.resize_image(image=image, width=1080, height=1920)
                    image = transforms.serialize_image(image=image)
                    s3man.datalake.upload_fileobj(image, f"raw_data/{subject}/{batch_date}/{entry.id}.png")


# For specific topic, fetch all entries from mongo from that date
mongo = MongoConnector(settings=settings.get_mongo_settings())
s3man = S3DataManager(settings=settings.get_s3_settings())
records = mongo.batch_raw_records_by_subject_and_date(batch_date=BATCH_DATE, subject=SUBJECT)
