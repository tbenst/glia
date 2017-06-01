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
    1: (3,6),
    2: (3,7),
    3: (3,5),
    4: (3,4),
    5: (2,7),
    6: (2,6),
    7: (1,7),
    8: (2,5),
    9: (1,6),
    10: (0,6),
    11: (1,5),
    12: (0,5),
    13: (2,4),
    14: (1,4),
    15: (0,4),
    16: (0,3),
    17: (1,3),
    18: (2,3),
    19: (0,2),
    20: (1,2),
    21: (0,1),
    22: (1,1),
    23: (2,2),
    24: (1,0),
    25: (2,1),
    26: (2,0),
    27: (3,3),
    28: (3,2),
    29: (3,0),
    30: (3,1),
    31: (4,1),
    32: (4,0),
    33: (4,2),
    34: (4,3),
    35: (5,0),
    36: (5,1),
    37: (6,0),
    38: (5,2),
    39: (6,1),
    40: (7,1),
    41: (6,2),
    42: (7,2),
    43: (5,3),
    44: (6,3),
    45: (7,3),
    46: (7,4),
    47: (6,4),
    48: (5,4),
    49: (7,5),
    50: (6,5),
    51: (7,6),
    52: (6,6),
    53: (5,5),
    54: (6,7),
    55: (5,6),
    56: (5,7),
    57: (4,4),
    58: (4,5),
    59: (4,7),
    60: (4,6)
}