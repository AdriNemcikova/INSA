import logging
import sys

FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s —" "%(funcName)s:%(lineno)d — %(message)s"
)


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout) # vypisovanie do konzoly
    console_handler.setFormatter(FORMATTER) # predpis ako vypisuje logy
    return console_handler
