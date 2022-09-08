"""

.. include:: ../README.md

## Tests and coverage

[See coverage](../coverage/index.html)

[See test report](../tests/report.html)

## Class diagram

![classes](./classes.png "Class diagram")

"""
import os
from pkg_resources import get_distribution
import logging

from rich.logging import RichHandler


__version__ = get_distribution(__name__).version

__author__ = "Y. de The"
__email__ = "ydethe@gmail.com"

logger = logging.getLogger("pyrat_logger")
logger.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())

stream_handler = RichHandler()
logger.addHandler(stream_handler)
