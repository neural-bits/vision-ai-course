from io import BytesIO
from urllib.error import URLError
from urllib.parse import urlparse

import requests
from loguru import logger
from PIL import Image
from requests import Response

logger = logger.bind(name="StorageImageChecks")

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme in ["http", "https"], result.netloc])
    except URLError as e:
        raise ValueError(f"Invalid URL: {e}")


def has_valid_image_header(response: Response) -> bool:
    if not response.headers.get("Content-Type", "").startswith("image/"):
        raise ValueError("URL doesn't point to an image")
    return True


def is_valid_image_size(response: Response, maxsize: int = 10) -> bool:
    content_length = int(response.headers.get("Content-Length", 0))
    if content_length > maxsize * 1024 * 1024:
        raise ValueError(f"Image exceeds {maxsize}MB limit")
    return True


def validate_image(url: str) -> bool:
    checks = {"valid_url": False, "content_type": False, "file_size": False}
    try:
        response = requests.head(url)
        checks["valid_url"] = is_valid_url(url)
        checks["content_type"] = has_valid_image_header(response)
        checks["file_size"] = is_valid_image_size(response)

        return all(checks.values())

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return False


def transform_to_png(image: Image) -> bytes:
    try:
        png_buffer = BytesIO()

        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            img = image.convert("RGBA")
            format_options = {"optimize": True}
        else:
            img = image.convert("RGB")
            format_options = {"quality": 95, "optimize": True}

        img.save(png_buffer, format="PNG", **format_options)
        png_buffer.seek(0)
        return png_buffer.getvalue()
    except Exception as e:
        raise ValueError(f"Conversion failed: {str(e)}") from e
