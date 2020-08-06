"""
   Manipulation of algebraic numbers
   Y. de The
"""
from pkg_resources import get_distribution
import logging
from datetime import datetime
import os


__version__ = get_distribution(__name__).version

__author__ = "Y. de The"
__email__ = "yann@johncloud.fr"


# création de l'objet logger qui va nous servir à écrire dans les logs
logger = logging.getLogger("algebraicnumber_logger")
# on met le niveau du logger à DEBUG, comme ça il écrit tout
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

# création d'un formateur qui va ajouter le temps, le niveau
# de chaque message quand on écrira un message dans le log
formatter = logging.Formatter("[%(levelname)s]%(message)s")
# création d'un handler qui va rediriger chaque écriture de log
# sur la console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# os.makedirs("logs", exist_ok=True)

# now = datetime.now()
# sd = now.strftime("%Y_%m_%d_%H_%M_%S")
# file_handler = logging.FileHandler(
# "logs/sc_%s.log" % sd, mode="w", encoding="utf-8", delay=False
# )
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
