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

def f_create_experiments(prepend_start_time=0, append_lifetime=0,
                         append_start_time=None):
    """
    Split spike train into individual experiments according to stimulus list.

    If append_start_time is given, ignore Lifetime and return experiments of duration
    append_start_time + prepend_start_time.
    """

    def create_experiments(spike_train: np.ndarray,
                           stimulus_list: List[Dict]) -> List[Dict]:
        experiments = [{"stimulus": s[1], "spikes": []} for s in stimulus_list]

        for i, stimulus in enumerate(stimulus_list):
            start_time = stimulus_list["start_time"] - prepend_start_time
            if append_start_time is not None:
                end_time = start_time + stimulus_list[1]["lifetime"]/120 + append_lifetime
            else:
                end_time = start_time + append_start_time

            bool_indices = (spike_train > start_time) and (spike_train < end_time)

            experiments[i]['spikes'] = spike_train[bool_indices]
        return experiments

    return create_experiments

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
        if (e["stimulus"]["stimulusType"] in stimulus_type ):
            return True
        else:
            return False
    return f_filter(anonymous)
    

        
def f_group_by(stimulus_parameter) -> Callable[[List[Experiment]], Dict]:
    def anonymous(accumulator,experiment):
        parameter = experiment["stimulus"][stimulus_parameter]
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
        parameter = str(experiment["stimulus"])
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