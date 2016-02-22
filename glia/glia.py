# not that parts of this library requires Matlab (2015b) and the Chronux
# package in the search path: http://chronux.org/

# import matlab.engine
# import matlab
import glob
import numpy as np
from typing import List, Any, Dict
import matplotlib.pyplot as plt
import pytest
import h5py
import warnings
import re

file = str
Dir = str
dat = str


def _lines_with_float(path: file):
    r"""Return generator yielding lines supporting float conversion.

    Note that generator returned could StopIteration without yielding.

    Tests:
    >>> x = _lines_with_float('tests/sample_dat/mcs_mea_recording_12.dat')
    >>> next(x)
    '19.93080\n'
    >>> next(x)
    '19.97080\n'
    >>> next(x)
    Traceback (most recent call last):
        ...
    StopIteration
    """
    with open(path, mode='r') as f:
        for line in f:
            try:
                float(line)
                yield(line)
            except:
                next
    return


def read_mcs_dat(my_path: Dir, only_channels: List[int]=None,
                 ignore_channels: List[int]=[],
                 channel_dict: Dict=None,
                 warn: bool=False) -> (List[np.ndarray]):
    """
    Take directory with MCS raw voltage for each channel, returns numpy array.

    This is intended to be used on DAT files exported from Multi Channel
    Systems spike sorting. Each number is interpreted as the time of a spike
    for a particular channel.

    Example .dat file:
        T[s]    24-Timestamp Unit1
        0.36848
        0.38512
        1.26852

    Args:
        my_path: string corresponding to directory containing MCS DAT files.
        channel_dict: Dict with keys as MCS label and values as desired
            channel index in returned List. To ignore a channel, set value
            to None.
        warn: if True, will warn when .dat files are not read because their
            label (two numbers prior to .dat) is not in channel_dict

    Returns:
        List of numpy arrays.

    Raises:
        ValueError: Files of unexpected format found.

    Tests:
    >>> channels = read_mcs_dat('tests/sample_dat/')
    >>> channels[14] is None
    True
    >>> channels[18]
    array([], dtype=float64)
    >>> len(channels)
    60
    >>> c = [x for x in channels if x is not None and x.size != 0]
    >>> c
    [array([ 19.9308,  19.9708])]

    """
    if channel_dict is None:
        # map of channel in HDF5 file to MCS Label. Seemingly arbitrary
        # can be found in "Data/Recording_0/AnalogStream/Stream_0/InfoChannel"
        channel_dict = {47: 0, 48: 1, 46: 2, 45: 3, 38: 4, 37: 5, 28: 6, 36: 7,
                        27: 8, 17: 9, 26: 10, 16: 11, 35: 12, 25: 13, 15: 14,
                        14: 15, 24: 16, 34: 17, 13: 18, 23: 19, 12: 20, 22: 21,
                        33: 22, 21: 23, 32: 24, 31: 25, 44: 26, 43: 27, 41: 28,
                        42: 29, 52: 30, 51: 31, 53: 32, 54: 33, 61: 34, 62: 35,
                        71: 36, 63: 37, 72: 38, 82: 39, 73: 40, 83: 41, 64: 42,
                        74: 43, 84: 44, 85: 45, 75: 46, 65: 47, 86: 48, 76: 49,
                        87: 50, 77: 51, 66: 52, 78: 53, 67: 54, 68: 55, 55: 56,
                        56: 57, 58: 58, 57: 59}

    if only_channels is not None:
        for k, v in channel_dict.items():
            if v not in only_channels:
                del channel_dict[k]

    # remove channels from mapping & ignore
    for k, v in channel_dict.items():
        if v in ignore_channels:
            channel_dict[k] = None

    dat_files = glob.glob(my_path + '/*.dat')

    if not dat_files:
        raise ValueError("No .dat files found")

    # initialize list; None will remain for channel_dict
    channels = [None] * len(channel_dict)

    # generator that filters out lines unable to be converted to a float

    for dat in dat_files:
        # read dat channel label
        match = re.search(r'(\d\d)\.dat$', dat)
        index = int(match.group(1))
        # verify we have mapping for label
        if index not in channel_dict:
            if warn is True:
                warnings.warn(
                    "Ignored '{f}': Not in channel_dict".format(f=dat))
            continue

        # For consistency, we map MCS Label to channel number read in by HDF5
        index = channel_dict[index]

        # skip bad channels while leaving initialized None as placeholder
        if index is None:
            if warn is True:
                warnings.warn(
                    "Ignored '{f}': index is None".format(f=dat))
            continue

        filtered = _lines_with_float(dat)
        try:
            # ignore npyio.py:891: UserWarning: loadtxt: Empty input file:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", lineno=891)
                channel = np.loadtxt(filtered, float)
        except StopIteration:
            # if generator is empty, this will save numpy from crashing
            if warn is True:
                warnings.warn(
                    "No spikes found in '{f}'".format(f=dat))
            channel = np.array([])

        channels[index] = channel

    if not channels:
        raise ValueError("Unexpected format in .dat files")

    return(channels)


def read_hdf5_voltages(file: file) -> (np.ndarray):
    r"""
    Read HDF5 file from MCS and return Numpy Array.

    Args:
        file: file ending in .h5

    Returns:
        2D ndarray, where first dimension is the number of channels (expected
        to be 60) and the second dimension is voltage for a given sampling
        frequency (by default 40kHz)

    Tests:
    >>> dset = read_hdf5_voltages('tests/sample-mcs-mea-recording.h5')
    >>> dset.shape
    (60, 2)
    >>> dset[0,:]
    array([86, 39], dtype=int32)
    """
    # verify extension matches .hdf, .h4, .hdf4, .he2, .h5, .hdf5, .he5
    if re.search(r'\.h[de]?f?[f245]$', file) is None:
        raise ValueError("Must supply HDF5 file (.h5)")

    recording = h5py.File(file, 'r')
    return np.array(recording[
        "Data/Recording_0/AnalogStream/Stream_0/ChannelData"], dtype='int32')


def last_spike_time(channels: List[np.ndarray]) -> (float):
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


def test_spike_histogram(channels):
    # final bin should have two spikes
    assert spike_histogram(channels)[0][199] == 2


def spike_histogram(channels: List[np.ndarray], bin_width: float=0.1) -> (Any):
    """
    Plot histogram of summed spike counts across all channels.

    Args:
        channels: List of numpy arrays; each float represents spike times.
        bin_width: Size of histogram bin width in seconds.

    Returns:
        matplotlib pyplot figure (can call .show() to display).

    Raises:
        ValueError for bad bin_width.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if bin_width <= 0:
        raise ValueError("bin_width must be greater than zero")

    last_time = last_spike_time(channels)

    # add bin_width to last_time so histogram includes final edge
    bins = np.arange(0, np.ceil(last_time) + bin_width, bin_width)
    # flatten List[ndarray] -> ndarray
    all_spikes = np.hstack([c for c in channels if c is not None])
    # plot histogram
    ax.hist(all_spikes, bins)
    return(np.histogram(all_spikes, bins))


# Pytest modules


@pytest.fixture(scope="module")
def channels():
    return read_mcs_dat('tests/sample_dat/')
