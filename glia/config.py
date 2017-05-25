import multiprocessing
import logging

logger = logging.getLogger('glia')
logger.setLevel(logging.DEBUG)


class DuplicateFilter(object):
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv

dup_filter = DuplicateFilter()
logger.addFilter(dup_filter)

processes = multiprocessing.cpu_count()
plot_directory = None

channel_map = {
    1: (7,4),
    2: (8,4),
    3: (6,4),
    4: (5,4),
    5: (8,3),
    6: (7,3),
    7: (8,2),
    8: (6,3),
    9: (7,2),
    10: (7,1),
    11: (6,2),
    12: (6,1),
    13: (5,3),
    14: (5,2),
    15: (5,1),
    16: (4,1),
    17: (4,2),
    18: (4,3),
    19: (3,1),
    20: (3,2),
    21: (2,1),
    22: (2,2),
    23: (3,3),
    24: (1,2),
    25: (2,3),
    26: (1,3),
    27: (4,4),
    28: (3,4),
    29: (1,4),
    30: (2,4),
    31: (2,5),
    32: (1,5),
    33: (3,5),
    34: (4,5),
    35: (1,6),
    36: (2,6),
    37: (1,7),
    38: (3,6),
    39: (2,7),
    40: (2,8),
    41: (3,7),
    42: (3,8),
    43: (4,6),
    44: (4,7),
    45: (4,8),
    46: (5,8),
    47: (5,7),
    48: (5,6),
    49: (6,8),
    50: (6,7),
    51: (7,8),
    52: (7,7),
    53: (6,6),
    54: (8,7),
    55: (7,6),
    56: (8,6),
    57: (5,5),
    58: (6,5),
    59: (8,5),
    60: (7,5)
}