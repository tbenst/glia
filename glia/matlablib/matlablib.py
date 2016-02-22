# The following functions require a working MATLAB engine for Python
# http://www.mathworks.com/help/matlab/matlab-engine-for-python.html

# Old MATLAB versions are not supported (this is Mathworks's fault)
# Suggest using Python 3.4 or later

import matlab.engine
import matlab
import numpy as np
from typing import List, Any, Dict, Tuple
import matplotlib.pyplot as plt

file = str
Dir = str
dat = str

m = matlab.engine.start_matlab()


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


def plot_voltage_spectrogram(file: file, time: Tuple=None, tapers: int=4,
                             average: bool=True,
                             channels: Any=[x for x in range(60)],
                             ignore_channels: List[int]=None,
                             window_size: float=0.5, window_step: float=0.125,
                             params: Dict={"tapers": matlab.double([4, 7]),
                                           "Fs": 40000.0,
                                           "fpass": matlab.double([1, 100]),
                                           "pad": 1.0,
                                           "trialave": 1.0}) -> (Any):
    r"""
    Use multitaper method to plot spectrogram.

    Expects HDF5 file exported from MCS. Calls mtspecgramc for computation.

    Returns:
        None; plots with matplotlib.

    Raises:
        N/A
    """
    if average is True or type(channels) is int:
        params['trialave'] = 1.0
        S = mtspecgramc(
            file, time, channels, tapers, window_size, window_step,
            ignore_channels, params)
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

    else:
        params['trialave'] = 0.0
        # filter ignored channels
        if ignore_channels is not None:
            channels = list(
                filter(lambda x: x not in ignore_channels, channels))

        num_cols = 5
        # determine number of rows
        if len(channels) % num_cols == 0:
            num_rows = len(channels) // num_cols
        else:
            num_rows = len(channels) // num_cols + 1

        plt.rcParams['figure.figsize'] = (15.0, 2.5 * num_rows)

        for i, c in enumerate(channels):
            S = mtspecgramc(
                file, time, c, tapers, window_size, window_step, None, params)
            ax = plt.subplot(num_rows, num_cols, i + 1)
            implot = ax.imshow(10 * np.log(np.transpose(S)), aspect='auto',
                               extent=[time[0], time[1], 100, 0])
            plt.gca().invert_yaxis()
            plt.colorbar(implot)
            plt.title("Channel " + str(c))

        plt.rcParams['figure.figsize'] = (6.0, 4.0)


def mtspecgramc(file: file, time: Tuple=None,
                channels: Any=[x for x in range(60)], tapers: int=5,
                window_size: float=0.5,
                window_step: float=0.125,
                ignore_channels: List[int]=None,
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
        channels: specify an individual channel (e.g. 0-59) or a List of
            channels, else use all
        window_size: size in seconds of window
        window_step: size in seconds of window step (often 25-50% of
            window_size)
        tapers: construct params with this number of tapers. Do not use with
            params.
        ignore_channels: exclude these channels from calculation
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
    # filter ignored channels
    if ignore_channels is not None:
        channels = list(filter(lambda x: x not in ignore_channels, channels))
    # construct string for matlab index of channel
    # adjust for 1-index
    if channels is None:
        channel_str = ':'
    elif type(channels) is list:
        channels = list(map(lambda x: x + 1, channels))
        channel_str = '[' + ','.join(map(str, channels)) + ']'
    elif type(channels) is int:
        channel_str = str(channels + 1)
    else:
        raise ValueError(
            "Unexpected type {t} for channels".format(t=type(channels)))

    if tapers is not None:
        params['tapers'] = matlab.double([(tapers + 1) / 2, tapers])

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


def mtspecgrampt(channel: np.ndarray, window_size: float=0.5,
                 window_step: float=0.5, tapers: int=5,
                 params: Dict={"tapers": matlab.double([4, 7]),
                               "Fs": 40000.0,
                               "fpass": matlab.double([1, 100]),
                               "pad": 1.0,
                               "trialave": 1.0}) -> (np.ndarray):
    r"""
    Multi-taper time-frequency spectrum - point process.
    """
    if tapers is not None:
        params['tapers'] = matlab.double([(tapers + 1) / 2, tapers])
    m.workspace['params'] = params
    m.workspace['win'] = matlab.double([window_size, window_step])
    m.workspace['channel'] = ndarray_to_matlab(channel)

    S = m.eval("mtspecgrampt(channel,win,params);", nargout=1)

    return np.array(S)
