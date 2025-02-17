import logging

logging.basicConfig(filename="newfile.log", format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ok=101
logger.debug(f"hety bro{ok}")