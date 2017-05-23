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

channel_dict = {
    1: "El_47"
    2: "El_48"
    3: "El_46"
    4: "El_45"
    5: "El_38"
    6: "El_37"
    7: "El_28"
    8: "El_36"
    9: "El_27"
    10: "El_17"
    11: "El_26"
    12: "El_16"
    13: "El_35"
    14: "El_25"
    15: "El_15"
    16: "El_14"
    17: "El_24"
    18: "El_34"
    19: "El_13"
    20: "El_23"
    21: "El_12"
    22: "El_22"
    23: "El_33"
    24: "El_21"
    25: "El_32"
    26: "El_31"
    27: "El_44"
    28: "El_43"
    29: "El_41"
    30: "El_42"
    31: "El_52"
    32: "El_51"
    33: "El_53"
    34: "El_54"
    35: "El_61"
    36: "El_62"
    37: "El_71"
    38: "El_63"
    39: "El_72"
    40: "El_82"
    41: "El_73"
    42: "El_83"
    43: "El_64"
    44: "El_74"
    45: "El_84"
    46: "El_85"
    47: "El_75"
    48: "El_65"
    49: "El_86"
    50: "El_76"
    51: "El_87"
    52: "El_77"
    53: "El_66"
    54: "El_78"
    55: "El_67"
    56: "El_68"
    57: "El_55"
    58: "El_56"
    59: "El_58"
    60: "El_57"
}