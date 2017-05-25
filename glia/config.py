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

# (x,y)
channel_map = {
    1: (4,7),
    2: (4,8),
    3: (4,6),
    4: (4,5),
    5: (3,8),
    6: (3,7),
    7: (2,8),
    8: (3,6),
    9: (2,7),
    10: (1,7),
    11: (2,6),
    12: (1,6),
    13: (3,5),
    14: (2,5),
    15: (1,5),
    16: (1,4),
    17: (2,4),
    18: (3,4),
    19: (1,3),
    20: (2,3),
    21: (1,2),
    22: (2,2),
    23: (3,3),
    24: (2,1),
    25: (3,2),
    26: (3,1),
    27: (4,4),
    28: (4,3),
    29: (4,1),
    30: (4,2),
    31: (5,2),
    32: (5,1),
    33: (5,3),
    34: (5,4),
    35: (6,1),
    36: (6,2),
    37: (7,1),
    38: (6,3),
    39: (7,2),
    40: (8,2),
    41: (7,3),
    42: (8,3),
    43: (6,4),
    44: (7,4),
    45: (8,4),
    46: (8,5),
    47: (7,5),
    48: (6,5),
    49: (8,6),
    50: (7,6),
    51: (8,7),
    52: (7,7),
    53: (6,6),
    54: (7,8),
    55: (6,7),
    56: (6,8),
    57: (5,5),
    58: (5,6),
    59: (5,8),
    60: (5,7)
}