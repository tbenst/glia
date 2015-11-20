# import matlab.engine
import glob
import numpy as np
from typing import List, Any
import matplotlib.pyplot as plt
import pytest
# import pandas as pd
# import re


file = str
Dir = str
dat = str
m = 1  # matlab.engine.start_matlab()


def voltage_spectrogram(file: str, num_tapers: int=5, eng=m) -> None:
    """Create spectrogram plot of voltages with reasonable parameters."""
    {"tapers": [4, 7], "Fs": 40000, "fpass": [1, 100], "pad": 1, }

    # m.eval
    #     params = struct('tapers', [4 7], 'Fs', 40000, 'fpass', [1 100],...
    #        'pad', 1, 'trialave', 1);
    #     t_start = 15; % seconds
    #     t_end = 30; % seconds
    #     [S,t,f] = mtspecgramc(melrd{1}(t_start*40000:t_end*40000,:),...
    #         [.5  .125], params);
    #     h1 = figure;
    #     imagesc(t,flip(f),10*log(S')) % convert to decibel, transpose & flip
    #     set(gca,'YTickLabel',flipud(get(gca,'YTickLabel')));
    #     xlabel('time [sec]'); ylabel('Frequency');
    #     title('MelRd');
    #     colorbar
    #      nargout=0)


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
            channel = np.loadtxt(filtered, float)
        except StopIteration:
            # if generator is empty, this will save numpy from crashing
            channel = np.array([])

        channels.append(channel)
        # try:
        #    channel = np.loadtxt(_filtered_file_generator(dat), skiprows=1)
        # except:
        #    channel = np.array([])

    if not channels:
        raise ValueError("Unexpected format in .dat files")

    return(channels)


def test_last_spike_time():
    assert False


def last_spike_time(channels: List[np.ndarray]) -> (float):
    r"""
    Return time in seconds of last spike in channels array.

    Tests:
    >>> last_spike_time(read_mcs_dat('tests/sample_dat/'))
    False
    """
    pass


def test_spike_histogram():
    assert(False)


def spike_histogram(channels: List[np.ndarray], bin_width: float=0.1) -> (Any):
    """
    Plot histogram of summed spike counts across all channels.

    Args:
        channels: List of numpy arrays; each float represents spike times.
        bin_width: Size of histogram bin width in seconds.

    Returns:
        matplotlib pyplot figure (can call .show() to display)

    Raises:
        N/A
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if bin_width <= 0:
        raise ValueError("bin_width must be greater than zero")

    last_spike_time = 0
    for channel in channels:
        if channel.size != 0:
            last_spike_time = max(last_spike_time, np.amax(channel))

    if last_spike_time <= 0:
        raise ValueError("Last spike time cannot be zero/negative")

    bins = np.arange(0, int(last_spike_time), bin_width)
    ax.hist([item for sublist in channels for item in sublist], bins)
    return(fig)


# Wrappers for Chronux functions below

# need to check if in function (def snipped) can get rid of superflous ':' when
# final parameter is of format x: str
# by using backreferences


@pytest.fixture(scope="module")
def channels():
    return read_mcs_dat('tests/sample_dat/')


if __name__ == "__main__":
    pytest.main([__file__, '--doctest-modules'])
    channels = read_mcs_dat('tests/sample_dat/')
    fig = spike_histogram(channels)
    fig.show()
