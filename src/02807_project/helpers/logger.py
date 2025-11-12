from loguru import logger

# Configure logger for console output
logger.remove()  # Remove default handler
logger.add(
    lambda msg: print(msg, end=""),  # noqa: T201
    level="INFO",
    colorize=True,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
)

__all__ = ["logger"]
