import logging
from logging.config import dictConfig
from typing import Optional

# def get_logging_config(log_level: str = 'INFO', log_file: Optional[str] = None) -> dict:
#     """Generate logging configuration based on log level and optional log file."""
#     config = {
#         'version': 1,
#         'disable_existing_loggers': False,
#         'formatters': {
#             'default': {
#                 'format': '%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
#                 'datefmt': '%Y-%m-%d %H:%M:%S',
#             },
#         },
#         'handlers': {
#             'console': {
#                 'class': 'logging.StreamHandler',
#                 'formatter': 'default',
#                 'level': log_level,
#             },
#         },
#         'root': {
#             'handlers': ['console'],
#             'level': log_level,
#         },
#     }

#     if log_file:
#         config['handlers']['file'] = {
#             'class': 'logging.FileHandler',
#             'formatter': 'default',
#             'filename': log_file,
#             'level': log_level,
#             'mode': 'a',  # Append mode
#         }
#         config['root']['handlers'].append('file')

#     return config


class DummyLogger:
    def __init__(self):
        pass

    def info(self, msg, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        pass

    def debug(self, msg, *args, **kwargs):
        pass

    def exception(self, msg, *args, **kwargs):
        pass

    def log(self, msg, *args, **kwargs):
        pass


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None, rank: Optional[int] = 0) -> logging.Logger:
    if rank == 0:

        # Create a root logger
        logger = logging.getLogger()
        logger.setLevel(level.upper())

        # Create a console handler
        ch = logging.StreamHandler()
        ch.setLevel(level.upper())

        # Create a formatter and set it for the console handler
        formatter = logging.Formatter(
            fmt='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        ch.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(ch)

        # Hide default INFO log from httpx._client.py
        logging.getLogger('httpx').setLevel(logging.WARNING)

        # If a log file is provided, add a file handler
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(level.upper())
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger
    else:
        return DummyLogger()