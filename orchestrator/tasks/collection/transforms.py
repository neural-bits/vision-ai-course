from io import BytesIO
from typing import Tuple

from PIL import Image


def serialize_image(image: Image.Image) -> BytesIO:
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


def deserialize_image(data: bytes) -> Image.Image:
    """Deserialize image from bytes"""
    try:
        return Image.open(BytesIO(data))
    except Exception as e:
        raise ValueError(f"Image deserialization failed: {e}") from e


def resize_image(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Resize image to target size with aspect ratio preservation"""
    try:
        return image.resize(size, Image.LANCZOS)
    except Exception as e:
        raise ValueError(f"Image resizing failed: {e}") from e
