# logger_config.py
import sys

from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[script_name]: <10}</cyan> | "
    "<yellow>{message}</yellow>",
    level="INFO",
)

logger.add(
    "app.log",
    format="{time} {level} {message}",
    level="INFO",
    rotation="1 MB",
    retention="10 days",
)


def get_logger(script_name):
    return logger.bind(script_name=script_name)
