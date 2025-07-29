import logging
import os


os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("docsearch_logger")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logs/app.log", encoding="utf-8")
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
file_handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(file_handler)
