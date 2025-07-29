from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel
from app.utils.logger import logger


def load_llm_model(base_path: str):
    logger.info(f"Loading base model and tokenizer from: {base_path}")

    try:
        model = T5ForConditionalGeneration.from_pretrained(base_path)
        logger.info("Base model loaded successfully")

        model = PeftModel.from_pretrained(model, base_path)
        logger.info("LoRA adapter loaded successfully")

        tokenizer = T5Tokenizer.from_pretrained(base_path)
        logger.info("Tokenizer loaded successfully")

        model.eval()
        logger.info("Model set to evaluation mode")

        return model, tokenizer

    except Exception as e:
        logger.exception(f"Failed to load model or tokenizer from {base_path}: {e}")
        raise
