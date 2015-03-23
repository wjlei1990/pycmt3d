from __future__ import (absolute_import)
import logging.config

# setup the logger
logger = logging.getLogger("pycmt3d")
logger.setLevel(logging.DEBUG)
logger.propagate = 0

#ch = logging.StreamHandler()
ch = logging.FileHandler("log.txt")
# Add formatter
FORMAT = "%(name)s - %(levelname)s: %(message)s"
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
logger.addHandler(ch)
