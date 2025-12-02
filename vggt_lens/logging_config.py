import logging
import sys


def setup_logger(name: str = "VGGT-Lens") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if getattr(logger, "_vggt_lens_configured", False):
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    logger.handlers.clear()
    logger.addHandler(handler)
    logger._vggt_lens_configured = True
    return logger
