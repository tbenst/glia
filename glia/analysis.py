# not that parts of this library requires Matlab (2015b) and the Chronux
# package in the search path: http://chronux.org/

# import matlab.engine
# import matlab
import numpy as np
from typing import List, Dict
# import pytest

file = str
Dir = str
dat = str
Hz = int
UnitSpikeTrains = List[Dict[str,np.ndarray]]
SpikeTrain = np.ndarray

def last_spike_time(channels: (UnitSpikeTrains)) -> (float):
    r"""
    Return time in seconds of last spike in channels array.

    Tests:
    >> last_spike_time(read_mcs_dat('tests/sample_dat/'))
    19.970800000000001

    (doctest disabled)
    """
    last_spike_time = 0
    for unit, spike_train in channels.items():
        if spike_train is not None and spike_train.size != 0:
            last_spike_time = max(last_spike_time, np.amax(spike_train))

    if last_spike_time <= 0:
        raise ValueError("Last spike time cannot be zero/negative")

    return(last_spike_time)


def construct_on_off_stimulus(analog):
    """Take analog TTL and output indices for on/off.

    Assumes first TTL is an on and second TTL is an off. Divide
    by the sampling rate to convert to seconds."""

    ttl_index = np.argwhere(np.diff(stimulus)>20000)[:,0] + 1
    on = []
    off = []
    for i,v in np.ndenumerate(ttl_index):
        if i[0]%2 == 1:
            on.append(v)
        else:
            off.append(v)
    return on, off

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


def spike_summary(unit_spike_trains: UnitSpikeTrains) -> (str):
    """Print num_units, num_spikes and last spike time for SpikeUnits."""
    num_units = len(unit_spike_trains)
    num_spikes = np.sum([len(u) for u in unit_spike_trains.keys()])

    last_spike_time = 0
    for unit,spike_train in unit_spike_trains.items():
        if spike_train.size != 0:
            last_spike_time = max(last_spike_time, np.max(spike_train))

    return "Number of units: {}\nNumber of spikes: {}\n" \
           "Last spike: {} seconds".format(
               num_units, num_spikes, last_spike_time)

def flatten(unit_spike_trains: UnitSpikeTrains) -> (np.ndarray):
    return np.hstack([c for c in unit_spike_trains.keys() if c is not None])

def plot_firing_rate (spike_train: SpikeTrain):
    """Take spike times of a particular spike unit and return a figure plotting 
    1) firing rate vs spike times (green) 2) 1-Dimensional Gaussian filter of firing rate vs spike times (magenta)
    Firing rate estimated by the interspike interval.
    Requires 'from scipy import ndimage.'"""

    y = np.diff(spike_train)
    x = spike_train
    firing_rate = 1/y
    #add 0 to the end of the firing_rate array to account for last spike where no firing_rate is calculated 
    #and to make x and y the same dimension
    firing_rate = np.append(firing_rate, 0)
    fig = plt.plot(x, firing_rate, 'green', linewidth=1)

    #sigma is standard deviation for Gaussian kernel
    sigma = 2
    x_g1d = ndimage.gaussian_filter1d(x, sigma)
    y_g1d = ndimage.gaussian_filter1d(firing_rate, sigma)

    fig = plt.plot(x_g1d, y_g1d, 'magenta', linewidth=1)

    return fig


def count_spikes(spike_train, x, window):
    """Count spikes that fall within bin starting at x and ending at x+window.
    Include spikes that fall on either bin edge"""
    
    #+1 to (x+window) because arange does not include the last bin edge value
    hist = np.histogram(spike_train, np.arange(x, x+window+1, window))
    count = hist[0]
    return count
    
    
def estimate_firing_rate(spike_train, window, step):
    """Estimate instantaneous firing rate using shifting, overlapping bins.
    Window is bin width. Step is amount bin is shifted.
    """
    firing_rate = []
    spike_counts = []
    #+1 - window is to to make the last x exactly 1 window length from the end of spike_train. +1 because amax does not include the last element
    for x in np.arange(0, np.amax(spike_train)+1-window, step):
        spike_counts = np.append(spike_counts, count_spikes(spike_train, x, window))
        firing_rate = np.array(spike_counts) / window
    return firing_rate

def sort_two_arrays_by_first(a,b):
    b = [y for (x,y) in sorted(zip(a,b))]
    a = sorted(a)
    return (a,b)
