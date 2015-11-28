# not that this library requires Matlab installed (2015b) and the Chronux
# package in the search path: http://chronux.org/

import matlab.engine
import matlab
import glob
import numpy as np
from typing import List, Any, Dict, Tuple
import matplotlib.pyplot as plt
import pytest
import h5py
import warnings
# import pandas as pd
import re

file = str
Dir = str
dat = str
m = matlab.engine.start_matlab()


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


def read_mcs_dat(my_path: Dir) -> (List[np.ndarray]):
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

    Returns:
        List of numpy arrays.

    Raises:
        ValueError: Files of unexpected format found.

    Tests:
    >>> channels = read_mcs_dat('tests/sample_dat/')
    >>> len(channels)
    2
    >>> channels
    [array([ 19.9308,  19.9708]), array([], dtype=float64)]
    """
    dat_files = glob.glob(my_path + '/*.dat')

    if not dat_files:
        raise ValueError("No .dat files found")

    channels = []

    # generator that filters out lines unable to be converted to a float

    for dat in dat_files:
        filtered = _lines_with_float(dat)
        try:
            # ignore npyio.py:891: UserWarning: loadtxt: Empty input file:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", lineno=891)
                channel = np.loadtxt(filtered, float)
        except StopIteration:
            # if generator is empty, this will save numpy from crashing
            channel = np.array([])

        channels.append(channel)

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
        if channel.size != 0:
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
    # flatten List[ndarray] -> ndarray & plot histogram
    all_spikes = np.hstack(channels)

    ax.hist(all_spikes, bins)
    # commented out so Jupyter does not plot twice
    return(np.histogram(all_spikes, bins))

# Functions for working with MATLAB below


def test_ndarray_to_matlab():
    assert matlab.double([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]]) \
        == ndarray_to_matlab(np.reshape(np.arange(12), (2, 3, 2)))


def ndarray_to_matlab(nd: np.ndarray) -> (matlab.double):
    r"""
    Convert ndarray to matlab matrix of float.

    Args:
        nd: ndarray of dtype int or float

    Returns:
        MATLAB matrix of integers.

    Raises:
        N/A

    Tests:
    >>> ndarray_to_matlab(np.arange(3))
    matlab.double([[0.0,1.0,2.0]])
    >>> ndarray_to_matlab(np.reshape(np.arange(6),(3,2)))
    matlab.double([[0.0,1.0],[2.0,3.0],[4.0,5.0]])
    """
    # create generator
    nditer = np.nditer(nd, order='F')
    # create list with comprehension then convert to MATLAB double
    row = matlab.double([float(i) for i in nditer])
    # turn row in matrix
    row.reshape(nd.shape)
    return row


def plot_voltage_spectrogram(file: file, time: Tuple=None, channel: int=None,
                             tapers: int=4,
                             window_size: float=0.5, window_step: float=0.125,
                             params: Dict={"tapers": matlab.double([4, 7]),
                                           "Fs": 40000.0,
                                           "fpass": matlab.double([1, 100]),
                                           "pad": 1.0,
                                           "trialave": 1.0}) -> (Any):
    r"""
    Use multitaper method to plot spectrogram.

    Expects HDF5 file exported from MCS. Calls mtspecgramc for computation.

    Args:

    Returns:
        None; plots with matplotlib.

    Raises:
        N/A
    """
    S = mtspecgramc(
        file, time, channel, tapers, window_size, window_step, params)
    ax = plt.subplot('111')
    # extent will change the axis labels, with y reversed
    implot = ax.imshow(10 * np.log(np.transpose(S)), aspect='auto',
                       extent=[time[0], time[1], 100, 0])

    # invert y axis so 0 Hz is at bottom
    plt.gca().invert_yaxis()
    # show colorbar to see numeric values for intensity
    plt.colorbar(implot)
    plt.title("Spectrogram")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")


def mtspecgramc(file: file, time: Tuple=None, channel: int=None, tapers: int=4,
                window_size: float=0.5, window_step: float=0.125,
                params: Dict={"tapers": matlab.double([4, 7]),
                              "Fs": 40000.0,
                              "fpass": matlab.double([1, 100]),
                              "pad": 1.0,
                              "trialave": 1.0}) -> (np.ndarray):
    r"""
    Multi-taper time-frequency spectrum - continuous process.

    A wrapper for the Chronux function mtspecgramc. Usage of this function
    requires Matlab--tested with 2015b. See 'The Chronux Manual'
    for further information on this function.

    Args:
        file: must be HDF5
        time: a tuple of form (start, end) in seconds
        channel: specify an individual channel (e.g. 0-59), else use all
        window_size: size in seconds of window
        window_step: size in seconds of window step (often 25-50% of
            window_size)
        tapers: construct params with this number of tapers. Do not use with
            params.
        params: for Chronux mtspecgramc

    Returns:
        S: power

    Raises:
        N/A
    """
    # construct string for matlab index of time
    if time is None:
        time_str = ':'
    else:
        time_str = '{start}:{end}'.format(start=time[0] * params['Fs'] + 1,
                                          end=time[1] * params['Fs'])

    # construct string for matlab index of channel
    if channel is None:
        channel_str = ':'
    else:
        channel_str = str(channel)

    # TODO tapers follows suggested values for
    if tapers is not None:
        params['tapers'] = matlab.double([tapers, tapers * 2 - 1])

    m.workspace['params'] = params
    m.workspace['win'] = matlab.double([window_size, window_step])
    m.eval("""recording = double(hdf5read('{hdf5}',...
        '/Data/Recording_0/AnalogStream/Stream_0/ChannelData'));""".format(
        hdf5=file), nargout=0)

    S = m.eval("""mtspecgramc(recording({time_str},{channel_str}),...
               win,params);""".format(time_str=time_str,
                                      channel_str=channel_str), nargout=1)
    m.eval('clearvars params win recording', nargout=0)
    return np.array(S)

# Pytest modules


@pytest.fixture(scope="module")
def channels():
    return read_mcs_dat('tests/sample_dat/')


if __name__ == "__main__":
    pytest.main([__file__, '--doctest-modules'])
