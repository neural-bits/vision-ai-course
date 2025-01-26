from io import BytesIO

import requests
from checks import transform_to_png, validate_image
from PIL import Image
from tools.aws.connector import S3DataManager
from tools.mongo.connector import MongoConnector

BATCH_DATE = "2025-01-24"
SUBJECT = "bottles"

# For specific topic, fetch all entries from mongo from that date
mongo = MongoConnector()
s3man = S3DataManager()
records = mongo.batch_raw_records_by_subject_and_date(batch_date=BATCH_DATE, subject=SUBJECT)

for entry in records:
    # Step 1: Check at header level
    is_image_valid = validate_image(entry["imageurl"])
    if not is_image_valid:
        continue
    # Step 2: Check at content level
    data = requests.get(entry["imageurl"]).content

    try:
        img = Image.open(BytesIO(data))
        img.verify()
        as_png = transform_to_png(data)

        # Upload to datalake
        s3man.datalake.upload_fileobj(as_png, f"raw_data/{SUBJECT}/{BATCH_DATE}/{entry['id']}.png")
    except (IOError, SyntaxError) as e:
        raise ValueError(f"Corrupted image file: {e}")
