import numpy as np
import re
from typing import List, Dict
# import pytest
import glob
import warnings
import h5py
import os
import csv
from uuid import uuid4
from .classes import Unit

file = str
Dir = str
dat = str
UnitSpikeTrains = List[Dict[str,np.ndarray]]

# VOLTAGE DATA


def read_raw_voltage(raw_filename):
    """Read in a raw file exported from MCS datatool."""
    header, offset = get_header(raw_filename)
    channel_start = re.search('\nStreams = ', header).span()[1]
    channel_str = header[channel_start:-7] + ';'
    channels = re.findall('(.._..);', channel_str)
    num_cols = len(channels)
    num_rows = int(np.memmap(raw_filename, offset=offset,
                             dtype='int16').shape[0] / num_cols)

    return np.memmap(raw_filename, shape=(num_rows, num_cols),
                     offset=offset,
                     dtype='int16')

def read_plexon_txt_file(filepath, retina_id):
    """Assumes export format of Channel,Unit,timestamp exported per waveform."""
    units = {}
    unit_dictionary = {}
    with open(filepath) as file:
        for row in csv.reader(file, delimiter=','):
            channel = int(row[0])
            unit_num = int(row[1])
            spike_time = float(row[2])

            if (channel,unit_num) in unit_dictionary:
                unit_id = unit_dictionary[(channel,unit_num)]
            else:
                # initialize key for both dictionaries
                unit = Unit(retina_id, channel)
                unit_id = unit.id
                units[unit_id] = unit
                unit_dictionary[(channel,unit_num)] = unit.id
            
            units[unit_id].spike_train.append(spike_time)

    return units




def get_header(filename: file) -> (str):
    """Read the header from a MCS raw file."""
    header = ""
    header_end = b'EOH\r\n'
    num_bytes = 0
    with open(filename, mode='rb') as file:
        for line in file:
            num_bytes += len(line)
            header += line.decode("Windows-1252", errors='ignore')
            if line == header_end:
                break
            if num_bytes > 2000:
                raise Exception('error reading header')
    return header, num_bytes


def get_result_path(filename: file) -> (file):
    """Return path based on spyking circus output to subfolder."""
    directory, name = os.path.split(filename)
    name, ext = os.path.splitext(name)

    return os.path.join(directory, name, name + '.result.hdf5')


def sampling_rate(filename: file) -> (int):
    """Read the sampling rate from a MCS raw file."""
    header = get_header(filename)[0]
    return int(re.search("Sample rate = (\d+)", header).group(1))


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
    >>> dset = read_hdf5_voltages('tests/data/sample-mcs-mea-recording.h5')
    >>> dset.shape
    (60, 2)
    >>> dset[0,:]
    array([86, 39], dtype=int32)

    (doctest disabled)
    """
    # verify extension matches .hdf, .h4, .hdf4, .he2, .h5, .hdf5, .he5
    if re.search(r'\.h[de]?f?[f245]$', file) is None:
        raise ValueError("Must supply HDF5 file (.h5)")

    recording = h5py.File(file, 'r')
    return np.array(recording[
        "Data/Recording_0/AnalogStream/Stream_0/ChannelData"], dtype='int32')


def merge_mcs_raw_files(files_to_merge: List[str], output_file_name: str
                        ) -> (bool):
    """Take multiple MCS raw files with header and combine into one file.

    Copies header from first file. Assumes exported as int16 from MCS
    DataTool."""

    # copy header
    with open(files_to_merge[0], mode='rb') as file:
        with open(output_file_name, 'wb') as newfile:
            # read header from file then write to newfile
            for line in file:
                newfile.write(line)
                if line == b"EOH\r\n":
                    break

            # write new data to file
            for data in files_to_merge:
                volts = read_raw_voltage(data)
                volts.tofile(newfile)

# SPIKE DATA


def read_mcs_dat(my_path: Dir, only_channels: List[int]=None,
                 ignore_channels: List[int]=[],
                 channel_dict: Dict=None,
                 warn: bool=False) -> (UnitSpikeTrains):
    """
    Take directory with MCS dat files for each channel, returns numpy array.

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
    >>> channels = read_mcs_dat('tests/data/sample_dat/')
    >>> channels[14] is None
    True
    >>> channels[18]
    array([], dtype=float64)
    >>> len(channels)
    60
    >>> c = [x for x in channels if x is not None and x.size != 0]
    >>> c
    [array([ 19.9308,  19.9708])]

    (doctest disabled)
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


def read_spyking_results(filepath: str, sampling_rate: int) -> (
        UnitSpikeTrains):
    """Read the results from Spyking Circus spike sorting
    and gives a list of arrays."""

    result_regex = re.compile('.*\.result.hdf5$')
    if not re.match(result_regex, filepath):
        raise ValueError('Filepath must end in "result.hdf5"')

    result_h5 = h5py.File(filepath, 'r')

    spike_units = {}
    for unit in result_h5["spiketimes"]:
        spikes = np.array(
            result_h5["spiketimes"][unit], dtype='int32') / sampling_rate
        spike_units[unit] = spikes

    return spike_units


# HELPER FUNCTIONS


def _lines_with_float(path: file):
    r"""Return generator yielding lines supporting float conversion.

    Note that generator returned could StopIteration without yielding.

    Tests:
    >>> x = _lines_with_float('tests/data/sample_dat/mcs_mea_recording_12.dat')
    >>> next(x)
    '19.93080\n'
    >>> next(x)
    '19.97080\n'
    >>> next(x)
    Traceback (most recent call last):
        ...
    StopIteration
    
    (doctest disabled)
    """
    with open(path, mode='r') as f:
        for line in f:
            try:
                float(line)
                yield(line)
            except:
                next
    return
