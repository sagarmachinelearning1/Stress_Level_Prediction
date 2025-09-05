import logging
from src.config import LOGGING_PATH
import os

os.makedirs(os.path.dirname(LOGGING_PATH),exist_ok=True)


logging.basicConfig(
    filename=LOGGING_PATH,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s"
)

logger=logging.getLogger("regression_pipeline")