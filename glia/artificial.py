import functools
from functools import reduce
from typing import Dict, Callable
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Any, Union
from .functional import f_filter, f_map, f_reduce
from scipy import signal
from glia.types import Unit
from tqdm import tqdm

def random_unit(total_time, retina_id, channel, unit_num):
    spike_train = []
    spike = np.random.random()
    while spike < total_time:
        spike_train.append(spike)
        spike += np.random.random()

    unit = Unit(retina_id, channel, unit_num)
    unit.spike_train = np.array(spike_train)
    unit.initialize_id()
    return unit

def hz_unit(total_time, hz, retina_id, channel, unit_num):
    spike_train = []
    timestep = 1/hz
    spike = timestep
    while spike < total_time:
        spike_train.append(spike)
        spike += timestep

    unit = Unit(retina_id, channel, unit_num)
    unit.spike_train = np.array(spike_train)
    unit.initialize_id()
    return unit