import logging
import pytz
from datetime import datetime

def setup_logger(name, log_file, level=logging.INFO):
    """
    Function to setup a logger that saves logs with timestamps in Regina, Canada timezone.

    Args:
        name (str): Name of the logger.
        log_file (str): File path to save the log.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    regina_tz = pytz.timezone('America/Regina')
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    class TimezoneFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, tz=pytz.utc).astimezone(regina_tz)
            return dt.strftime(datefmt or self.default_time_format)

    file_handler.setFormatter(TimezoneFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    logger.addHandler(file_handler)

    return logger

