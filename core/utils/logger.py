import logging
import queue
from logging.handlers import QueueHandler, QueueListener
import os
import sys


def setup_logger(log_path: str, level: int = logging.INFO) -> QueueListener:
    """
    Set up logging with file and stream handlers using a queue.
    
    Uses QueueHandler/QueueListener pattern for thread-safe logging.
    Configurable log level for different environments (DEBUG, INFO, WARNING, ERROR).
    
    Args:
        log_path: Path to the log file
        level: Logging level (default: logging.INFO)
        
    Returns:
        QueueListener instance (call .stop() to clean up)
    """
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_queue = queue.Queue()
    queue_handler = QueueHandler(log_queue)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[queue_handler],
        force=True
    )
    
    file_handler = logging.FileHandler(log_path, mode="w")
    stream_handler = logging.StreamHandler(sys.stdout)

    listener = QueueListener(log_queue, file_handler, stream_handler)
    listener.start()
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    return listener
