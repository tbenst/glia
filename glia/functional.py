import functools
from functools import reduce
from typing import Dict, Callable
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Any, Union

file = str
Dir = str
dat = str
Hz = int
SpikeUnits = List[np.ndarray]
SpikeUnits = List[np.ndarray]
Seconds = float
ms = float

# Must have these two keys: (stimulus, spikes)
# spikes is a list of spike times List[float]
Experiment = Dict
Experiments = List[Dict]
SpikeTrain = List[float]
# Mustbe: {"WAIT", "SOLID", "BAR", "GRATING"}
StimulusType = str
SpikeTrains = List[SpikeTrain]
Analytics = Dict[str,Any]


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)

def f_filter(function):
    def filter_dict(f,d):
        for key, val in d.items():
            if not f(key,val):
                continue
            yield key, val

    def anonymous(x):
        if type(x) is list:
            return list(filter(function, x))
        elif type(x) is dict:
            return {k:v for (k,v) in filter_dict(function,x)}
    
    return anonymous

def f_map(function):
    def anonymous(x):
        if type(x) is list:
            return list(map(function, x))
        elif type(x) is dict:
            return {key: function(val) for key, val in x.items()}
    
    return anonymous

def f_reduce(function, initial_value=None) -> Callable[[List[Experiment],Any], Any]:
    def anonymous(e):
        if initial_value is not None:
            return reduce(function, e, initial_value)
        else:
            return reduce(function, e)
    return anonymous
