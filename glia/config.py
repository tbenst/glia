import multiprocessing
import logging

logger = logging.getLogger('glia')
logger.setLevel(logging.DEBUG)

processes = multiprocessing.cpu_count()
plot_directory = None


