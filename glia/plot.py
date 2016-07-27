import matplotlib.pyplot as plt
import numpy as np
from typing import List, Any
from .analysis import last_spike_time
# import pytest


Seconds = float
ms = float
SpikeUnits = List[np.ndarray]


def spike_histogram(channels: SpikeUnits, bin_width: Seconds=0.1,
                    time: (Seconds, Seconds)=(None, None), plot=True) -> (Any):
    """
    Plot histogram of summed spike counts across all channels.

    Args:
        channels: List of numpy arrays; each float represents spike times.
        bin_width: Size of histogram bin width in seconds.
        time: Tuple of (start_time, end_time) i

    Returns:
        matplotlib pyplot figure (can call .show() to display).

    Raises:
        ValueError for bad bin_width.
    """

    if bin_width <= 0:
        raise ValueError("bin_width must be greater than zero")

    # flatten array
    all_spikes = np.hstack([c for c in channels if c is not None])

    # filter for desired range
    start_time = time[0]
    end_time = time[1]
    if start_time is not None:
        all_spikes = all_spikes[all_spikes > start_time]
    else:
        start_time = 0

    if end_time is not None:
        all_spikes = all_spikes[all_spikes < end_time]
    else:
        end_time = last_spike_time(channels)

    # add bin_width to last_time so histogram includes final edge
    bins = np.arange(start_time, np.ceil(end_time) + bin_width, bin_width)

    # plot histogram
    if plot is True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(all_spikes, bins)
    else:
        return(np.histogram(all_spikes, bins))


def isi_histogram(channels: SpikeUnits, bin_width: Seconds=1/1000,
                  time: (Seconds, Seconds)=(0, 100/1000), average=True,
                  fig_size=(15, 30)) -> (Any):
    channels = [np.diff(c) for c in channels]
    # Unit is seconds so x is in ms for x/1000
    bins = np.arange(time[0], time[1], bin_width)
    fig = plt.figure(figsize=fig_size)

    if average:
        # flatten array
        all_isi = np.hstack([c for c in channels if c is not None])

        ax = fig.add_subplot(111)
        ax.hist(all_isi, bins)
    else:
        subp = subplot_generator(channels,5)
        for channel in channels:
            ax = fig.add_subplot(*next(subp))
            ax.hist(channel, bins)


def visualize_spikes(spike_units: SpikeUnits, fig_size=(30, 15)):
    fig = plt.figure(figsize=fig_size)

    # Draw each spike as black line
    ax = fig.add_subplot(211)
    for i, unit in enumerate(spike_units):
        ax.vlines(spike_units[i], i, i + 1)

    # Plot histogram
    ax2 = fig.add_subplot(212)
    ax2.plot(spike_histogram(spike_units))


# Helpers


def subplot_generator(n_charts, num_cols):
    """Generate arguments for matplotlib add_subplot.

    Must use * to unpack the returned tuple. For example,

    >>> fig = plt.figure()
    <matplotlib.figure.Figure at 0x10fdec7b8>
    >>> subp = subplot_generator(4,2)
    >>> fig.add_subplot(*next(subp))
    <matplotlib.axes._subplots.AxesSubplot at 0x112cee6d8>

    """

    if type(n_charts) is list:
        n_charts = len(n_charts)

    num_rows = n_charts // num_cols + (n_charts % num_cols != 0)
    n = 1
    while n <= n_charts:
        yield (num_rows, num_cols, n)
        n += 1


# @pytest.fixture(scope="module")
# def channels():
#     import files
#     return read_mcs_dat('tests/sample_dat/')