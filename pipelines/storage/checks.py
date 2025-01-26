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


def transform_to_png(image: Image.Image) -> BytesIO:
    """Optimized PNG conversion with buffer reuse and minimal conversions"""
    try:
        # Reuse buffer for memory efficiency
        buffer = BytesIO()

        if image.mode in {"RGBA", "LA", "RGB"}:
            output_mode = image.mode
            params = {"optimize": True, "compress_level": 3}
        elif image.mode == "P" and "transparency" in image.info:
            output_mode = "RGBA"
            params = {"optimize": True}
        else:
            output_mode = "RGB"
            params = {"optimize": True}

        # Skip conversion if already in target mode
        if image.mode != output_mode:
            image = image.convert(output_mode)

        image.save(
            buffer,
            format="PNG",
            **params,
            dpi=image.info.get("dpi", (72, 72)),  # Preserve original DPI if available
        )

        buffer.seek(0)
        return buffer

    except Exception as e:
        raise ValueError(f"PNG conversion failed: {e}") from e
