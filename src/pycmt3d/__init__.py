from __future__ import (absolute_import)
import logging.config
import os

# setup the logger
logger = logging.getLogger("pycmt3d")
logger.setLevel(logging.INFO)
logger.propagate = 0

if os.path.exists("log.txt"):
    os.remove("log.txt")

# ch = logging.StreamHandler()
ch = logging.FileHandler("log.txt", mode='w')
# Add formatter
FORMAT = "%(name)s - %(levelname)s: %(message)s"
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
logger.addHandler(ch)
