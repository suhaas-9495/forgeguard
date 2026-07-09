"""Production Logging System for ForgeGuard
"""

import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Log Directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "inference.log"

# Logger
logger = logging.getLogger("ForgeGuard")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
# Logger Getter
def get_logger():
    return logger

#Inference Logger
def log_inference(result):
    logger.info("=" * 70)
    logger.info(f"Request ID: {result.request_id}")
    logger.info(f"Timestamp: {result.timestamp}")
    logger.info(f"Status: {result.status}")
    logger.info(f"Provider: {result.provider}")
    logger.info(f"Model: {result.model_name}")
    logger.info(f"Model Version: {result.model_version}")
    logger.info(f"Adapter: {result.adapter_name}")
    logger.info(f"Temperature: {result.temperature}")
    logger.info(f"Max Tokens: {result.max_new_tokens}")
    logger.info(f"Finish Reason: {result.finish_reason}")
    logger.info("")
    logger.info(f"Prompt: {result.prompt}")
    logger.info("")
    logger.info(f"Response: {result.response}")
    logger.info("")
    logger.info(f"Confidence: {result.sequence_confidence:.4f}")
    logger.info(f"Latency (ms): {result.latency_ms:.2f}")
    logger.info(f"Input Tokens: {result.input_tokens}")
    logger.info(f"Output Tokens: {result.output_tokens}")
    logger.info(f"Total Tokens: {result.total_tokens}")
    logger.info(f"Input Cost: ${result.input_cost:.8f}")
    logger.info(f"Output Cost: ${result.output_cost:.8f}")
    logger.info(f"Total Cost: ${result.total_cost:.8f}")
    logger.info("=" * 70)
#Error Logger
def log_error(request_id, error):
    logger.error("=" * 70)
    logger.error(f"Request ID: {request_id}")
    logger.error(f"Timestamp: {datetime.utcnow().isoformat()}")
    logger.error(f"Error: {str(error)}")
    logger.error("=" * 70)
#Warning Logger
def log_warning(message):

    logger.warning(message)