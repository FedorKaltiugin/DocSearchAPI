import os
import gdown
import py7zr
from dotenv import load_dotenv
from datasets import load_dataset
from app.utils.logger import logger

load_dotenv()


def download_and_extract_models():
    os.makedirs("models", exist_ok=True)
    logger.info("The models folder was created")

    model_rag_id = os.getenv("model_rag")
    model_trained_id = os.getenv("model_trained")

    rag_archive = "models/flan-t5-lora-rag.7z"
    trained_archive = "models/flan-t5-lora-trained.7z"

    logger.info("Downloading RAG model...")
    gdown.download(f"https://drive.google.com/uc?id={model_rag_id}", rag_archive, quiet=False)

    logger.info("Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={model_trained_id}", trained_archive, quiet=False)

    logger.info("Unpacking the RAG model...")
    with py7zr.SevenZipFile(rag_archive, mode='r') as z:
        z.extractall(path="models")
    os.remove(rag_archive)
    logger.info("RAG model unpacked and archive deleted")

    logger.info("Unpacking model...")
    with py7zr.SevenZipFile(trained_archive, mode='r') as z:
        z.extractall(path="models")
    os.remove(trained_archive)
    logger.info("Model unpacked and archive deleted")


def load_dataset_from_hub():
    logger.info("Loading the rag-mini-wikipedia dataset from HuggingFace")
    dataset = load_dataset('rag-datasets/rag-mini-wikipedia', 'question-answer', split='test')
    logger.info("Dataset loaded successfully")
    return dataset
