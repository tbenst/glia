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
    return lambda x: list(filter(function, x))

def f_reduce(function, initial_value=None) -> Callable[[List[Experiment],Any], Any]:
    def anonymous(e):
        if initial_value is not None:
            return reduce(function, e, initial_value)
        else:
            return reduce(function, e)
    return anonymous

def f_has_stimulus_type(stimulus_type: Union[str]) -> Callable[[List[Experiment]], List[Experiment]]:
    if type(stimulus_type) is str:
        stimulus_type = [stimulus_type]
    def anonymous(e):
        # stimulus is from a javascript generator, so must look at value
        if (e["stimulus"]['value']["stimulusType"] in stimulus_type ):
            return True
        else:
            return False
    return f_filter(anonymous)
    

        
def f_group_by(stimulus_parameter) -> Callable[[List[Experiment]], Dict]:
    def anonymous(accumulator,experiment):
        parameter = experiment["stimulus"]["value"][stimulus_parameter]
        spikes = experiment["spikes"]
        new_accumulator = accumulator.copy()
        if parameter in new_accumulator:
            new_accumulator[parameter].append(spikes)
        else:
            new_accumulator[parameter] = [spikes]
        return new_accumulator
    return f_reduce(anonymous, {})

def f_group_by_stimulus() -> Callable[[List[Experiment]], Dict]:
    def anonymous(accumulator,experiment):
        parameter = str(experiment["stimulus"]["value"])
        spikes = experiment["spikes"]
        new_accumulator = accumulator.copy()
        if parameter in new_accumulator:
            new_accumulator[parameter].append(spikes)
        else:
            new_accumulator[parameter] = [spikes]
        return new_accumulator
    return f_reduce(anonymous, {})

def f_count_each_in_group() -> Callable[[Dict[str,List[SpikeTrain]]], Analytics]:
    def create_analytics(analytics):
        new = {}
        for k,v in analytics.items():
            # map count
            new[k] = list(map(lambda x: len(x),v))
        return new
    return create_analytics