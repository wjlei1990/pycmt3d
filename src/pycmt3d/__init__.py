from __future__ import (absolute_import)
import logging


# setup the logger
logger = logging.getLogger("pycmt3d")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
# Add formatter
#FORMAT = "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s"
#formatter = logging.Formatter(FORMAT)
#ch.setFormatter(formatter)
logger.addHandler(ch)