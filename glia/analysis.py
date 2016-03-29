# not that parts of this library requires Matlab (2015b) and the Chronux
# package in the search path: http://chronux.org/

# import matlab.engine
# import matlab
import numpy as np
from typing import List
import pytest

file = str
Dir = str
dat = str
Hz = int
SpikeUnits = List[np.ndarray]


def last_spike_time(channels: SpikeUnits) -> (float):
    r"""
    Return time in seconds of last spike in channels array.

    Tests:
    >>> last_spike_time(read_mcs_dat('tests/sample_dat/'))
    19.970800000000001
    """
    last_spike_time = 0
    for channel in channels:
        if channel is not None and channel.size != 0:
            last_spike_time = max(last_spike_time, np.amax(channel))

    if last_spike_time <= 0:
        raise ValueError("Last spike time cannot be zero/negative")

    return(last_spike_time)


def moving_window_gen(a, freq, win_size, win_step):
    """Generator that yields (time, data_vector) that form a moving window.

    Time is picked to be the center of the window. Data_vector guaranteed to be
    win_size in length. Thus up to win_step indices may be truncatedas there is
    insufficient data to pass a lenght of win_size."""

    start = 0
    end = start + win_size
    length = len(a)
    while end <= length:
        middle = (start + end) / 2
        yield (middle / freq, a[start:end])
        start += win_step
        end += win_step


def calc_variance(vector, freq: Hz=20000, win_size=100, win_step=100):
    """Return a vector of (time, variance) using a moving window.

    Time is the middle of the window. Variance is calculated over
    said window."""

    voltage_var = []
    time = []

    for i in moving_window(vector, freq, win_size, win_step):
        time.append(i[0])
        voltage_var.append(np.var(i[1]))

    return (time, voltage_var)


def spike_summary(spike_units: SpikeUnits) -> (str):
    """Print num_units, num_spikes and last spike time for SpikeUnits."""
    num_units = len(spike_units)
    num_spikes = np.sum([len(u) for u in spike_units])

    last_spike_time = 0
    for unit in spike_units:
        if unit.size != 0:
            last_spike_time = max(last_spike_time, np.max(unit))

    return "Number of units: {}\nNumber of spikes: {}\n" \
           "Last spike: {} seconds".format(
               num_units, num_spikes, last_spike_time)

# Pytest


def setup_module(module):
    import files