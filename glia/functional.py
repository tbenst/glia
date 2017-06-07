import functools
from functools import reduce, partial
from typing import Dict, Callable
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Any, Union
from multiprocessing import Pool
from .config import processes
from tqdm import tqdm

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


def _func(x,function=lambda x: x):
    k,v = x
    return (k,function(v))

def pmap(function, data, progress=False):
    """Parallel map that accepts lists or dictionaries.
    
    Use progress for interactive sessions."""
    
    pool = Pool(processes)
    length = len(data)
    

    if type(data)==list:
        if progress:
            gen = tqdm(iter(data), total=length)
        else:
            gen = iter(data)
        result = list(pool.imap(function, gen))
        pool.close()
        pool.join()
    elif type(data)==dict:
        if progress:
            gen = tqdm(data.items(), total=length)
        else:
            gen = data.items()

        length = len(data.keys())
        f = partial(_func,function=function)
        pre_result = list(pool.imap_unordered(f,
                                          gen))
        pool.close()
        pool.join()
        result = {k: v for k,v in pre_result}
    return result

def _group_by_helper(a,n,key,value):
    k = key(n)
    v = value(n)
    new_accumulator = a.copy()
    if k in a:
        new_accumulator[k].append(v)
    else:
        new_accumulator[k] = [v]
    return new_accumulator

def group_by(x: List[Any], key: Callable,
        value: Callable=lambda x: x) -> Dict[Any,List[Any]]:
    ""
    function = partial(_group_by_helper,key=key,value=value)
    return reduce(function,x,{})

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

flatten = f_reduce(lambda a,n: a+n,[])

def zip_dictionaries(*dictionaries, transform_yield=lambda v: v):
    "Iterate dictionaries and yield a tuple of their values, retaining order."
    # take keys that are in all dictionaries
    keys = set.intersection(*[set(d.keys()) for d in dictionaries])

    for key in keys:
        value = tuple((dictionary[key] for dictionary in dictionaries))
        to_yield = (key,value)
        yield transform_yield(to_yield)

def scanl(f, initial_value, mylist):
    """> scanl(operator.add, 0, range(1, 11))
    [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55]"""
    res = [initial_value]
    acc = initial_value
    for x in mylist:
     acc = f(acc, x)
     res += [acc]
    return res

def get_value(x,i=0):
    return list(x.values())[i]