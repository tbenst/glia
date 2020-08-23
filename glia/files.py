import numpy as np
import re, av
from typing import List, Dict
# import pytest
from glob import glob
import warnings
from warnings import warn
import h5py
import os
import csv
from .types import Unit
from .io.mdaio import readmda
from tqdm import tqdm
from glia.config import logger

file = str
Dir = str
dat = str
UnitSpikeTrains = List[Dict[str, np.ndarray]]

# VOLTAGE DATA


def match_filename(start,ext='txt'):
    files = glob(start + "*." + ext)
    if len(files)==1:
        return files[0]
    else:
        raise(ValueError("Could not match file, try specifying full filename"))


def find_notebook(directory):
    notebooks = glob(os.path.join(directory, 'lab*.yml')) + \
        glob(os.path.join(directory, 'lab*.yaml'))
    if len(notebooks)==0:
        raise ValueError("no lab notebooks (.yml) were found. Either add to directory," \
            "or specify file path with -n.")
    elif len(notebooks)>1:
        logger.warning(f"""Found multiple possible lab notebooks.
        Using {notebooks[0]}. If wrong, try manually specifying""")
    return notebooks[0]

def read_raw_voltage(raw_filename):
    """Read in a raw file exported from MCS datatool."""
    header, offset, adc_zero, El = get_header(raw_filename)
    channel_start = re.search('\nStreams = ', header).span()[1]
    channel_str = header[channel_start:-7] + ';'
    channels = re.findall('(.._..);', channel_str)
    num_cols = len(channels)
    num_rows = int(np.memmap(raw_filename, offset=offset,
                             dtype='int16').shape[0] / num_cols)

    return (np.memmap(raw_filename, shape=(num_rows, num_cols),
                     offset=offset,
                     dtype='int16') - 0) * El


def read_plexon_txt_file(filepath, retina_id, channel_map=None):
    """Assume export format of Channel,Unit,timestamp exported per waveform."""
    unit_dictionary = {}
    with open(filepath) as file:
        row_count = 0
        for row in csv.reader(file, delimiter=','):
            # skip header
            if re.match('Channel',row[0]): continue
            row_count+=1
            # match raw or number
            m = re.match("adc(\d+)",row[0])
            if m:
                channel = int(m.group(1))
            else:
                channel = int(row[0])
            unit_num = int(row[1]) - 1
            if channel_map:
                c = channel_map[channel]
            else:
                c = channel

            spike_time = float(row[2])

            if (c, unit_num) not in unit_dictionary:
                # initialize key for both dictionaries
                unit = Unit(retina_id, c, unit_num)
                unit_dictionary[(c, unit_num)] = unit

            unit_dictionary[(c, unit_num)].spike_train.append(spike_time)


    for uid in unit_dictionary.keys():
        unit_dictionary[uid] = unit_dictionary[uid]
        unit_dictionary[uid].spike_train = np.array(unit_dictionary[uid].spike_train)

    total_spike = 0
    for v in unit_dictionary.values():
        total_spike+=len(v.spike_train)
        # total_spike+=v.spike_train.shape[0]
    try:
        assert total_spike==row_count
    except:
        print(total_spike,row_count)
        print(unit_dictionary.keys())
        raise

    return {unit.id: unit for k,unit in unit_dictionary.items()}

def read_mda_file(filepath, retina_id, channel_map=None, sampling_rate=20000):
    """Read MountainSort file.

    Assume export format of (channel,time,unit) x #events
    """
    unit_dictionary = {}
    mda = readmda(filepath)
    for col in tqdm(mda.T):
        channel = int(col[0])
        spike_time = col[1]/sampling_rate
        unit_num = int(col[2])

        if channel_map:
            c = channel_map[channel]
        else:
            c = channel

        if (c, unit_num) not in unit_dictionary:
            # initialize key for both dictionaries
            unit = Unit(retina_id, c, unit_num)
            unit_dictionary[(c, unit_num)] = unit

        unit_dictionary[(c, unit_num)].spike_train.append(spike_time)


    for uid in unit_dictionary.keys():
        unit_dictionary[uid] = unit_dictionary[uid]
        unit_dictionary[uid].spike_train = np.array(unit_dictionary[uid].spike_train)

    return {unit.id: unit for k,unit in unit_dictionary.items()}

def read_3brain_analog(analog_file):
    with h5py.File(analog_file, 'r') as file:
        analog = np.array(file["3BData"]["Raw"])
    return analog

def read_3brain_spikes(filepath, retina_id, channel_map=None, truncate=False):
    """Read spikes detected by 3brain in a .bxr file.

    If channel_map is None, read from file, else use arg. Units will
    be sorted by order of first spike. For easy debugging, only read in a subset
    of spikes by setting truncate=True."""
    unit_dictionary = {}
    assert os.path.splitext(filepath)[1]==".bxr"

    with h5py.File(filepath, 'r') as h5_3brain_spikes:
        # read into memory by using [()]
        spike_channel_ids = h5_3brain_spikes["3BResults"]["3BChEvents"]["SpikeChIDs"][()]
        spike_unit_ids = h5_3brain_spikes["3BResults"]["3BChEvents"]["SpikeUnits"][()]
        spike_times = h5_3brain_spikes["3BResults"]["3BChEvents"]["SpikeTimes"][()]
        tot_spikes = spike_times.shape[0]
        spikes = zip(spike_channel_ids, spike_unit_ids, spike_times)

        if channel_map is None:
            channel_map = h5_3brain_spikes["3BRecInfo"]["3BMeaStreams"]["Raw"]["Chs"][()]

        sampling_rate = float(h5_3brain_spikes["3BRecInfo"]["3BRecVars"]["SamplingRate"][0])

        # map 3brain unit num
        # numbers typically from -4 to 9000
        # where negative numbers appear across multiple channels
        # and thus are presumably bad units...?
        # positive-numbered units appear on one channel
        unit_id_2_num = {}

        n_unit_nums = 0
        for chunk in iter_chunks(h5_3brain_spikes['3BResults/3BChEvents/SpikeUnits'], 10000):
            n_unit_nums = max(n_unit_nums, chunk.max())
        
        unit_map = {}
        channel_unit_count = {}
        #TODO map absolute unit number to relative unit number
        # read in spikes
        print("assign each spike to a channel & unit")
        # for rapid debugging, cap number of spikes read
        nnn = 0
        if truncate:
            logger.warn("PARTIAL READ OF SPIKES DEBUG!!!!")
        for channel, unit_id, spike_time in tqdm(spikes, total=tot_spikes):
            nnn+=1
            if truncate and nnn>1000000:
                break
            c = channel_map[channel]
            # convert to tuple
            # account for 1-indexing
            c = (c[0]-1,c[1]-1)

            # count num units on channel
            # first check if we've seen this channel before
            if not c in channel_unit_count:
                # if not, initialize channel_unit_count for the channel
                channel_unit_count[c] = 1
                unit_num = 0
                # add unit
                unit_id_2_num[unit_id] = unit_num
            else:
                
                # then check if we've seen this unit before
                if not unit_id in unit_id_2_num:
                    # if not, assign unit_num for this new unit
                    unit_num = channel_unit_count[c]
                    unit_id_2_num[unit_id] = unit_num
                    channel_unit_count[c] += 1
                else:
                    # otherwise, look it up
                    unit_num = unit_id_2_num[unit_id]

            t = spike_time / sampling_rate
            # next, we add unit to our dictionary if not already there
            if (c, unit_num) not in unit_dictionary:
                # initialize key for both dictionaries
                unit = Unit(retina_id, c, unit_num)
                unit_dictionary[(c, unit_num)] = unit
            # finally, we add the spike
            unit_dictionary[(c, unit_num)].spike_train.append(t)

 
    # convert each list to numpy array
    for uid in unit_dictionary.keys():
        # unit_dictionary[uid] = unit_dictionary[uid]
        unit_dictionary[uid].spike_train = np.array(unit_dictionary[uid].spike_train)

    return {unit.id: unit for k,unit in unit_dictionary.items()}

def combine_units_by_channel(units):
    new_units = {}
    for unit in units.values():
        channel = unit.channel
        if channel in new_units:
            new_units[channel] = merge_units(new_units[channel], unit)
        else:
            new_units[channel] = unit
    return {unit.id: unit for unit in new_units.values()}

def merge_units(a,b):
    channel = a.channel
    assert channel == b.channel
    retina_id = a.retina_id
    assert retina_id == b.retina_id
    spike_train = merge_spike_trains(a.spike_train,b.spike_train)
    # we use unit_num of 0 since other code may use this for indexing
    new = Unit(retina_id, channel, 0, spike_train)
    return new

def merge_spike_trains(a,b):
    return np.array(sorted(np.hstack((a,b))))

def get_header(filename: file) -> (str):
    """Read the header from a MCS raw file."""
    header = ""
    header_end = b'EOH\r\n'
    num_bytes = 0
    adc_zero = None
    El = None
    with open(filename, mode='rb') as file:
        for line in file:
            num_bytes += len(line)
            decoded_line = line.decode("Windows-1252", errors='ignore')
            header += decoded_line
            if decoded_line[:11]=="ADC zero = ":
                adc_zero = int(decoded_line[11:])
                print(f"ADC zero: {adc_zero}")
            if decoded_line[:5]=="El = ":
                El = float(decoded_line[5:11])
                print(f"El: {El}")
            if line == header_end:
                break
            if num_bytes > 2000:
                raise Exception('error reading header')
    return header, num_bytes, adc_zero, El


def get_result_path(filename: file) -> (file):
    """Return path based on spyking circus output to subfolder."""
    directory, name = os.path.split(filename)
    name, ext = os.path.splitext(name)

    return os.path.join(directory, name, name + '.result.hdf5')


def sampling_rate(filename: file) -> (int):
    """Read the sampling rate from a MCS raw file or 3brain .brw"""
    if filename[-4:]==".brw":
        with h5py.File(filename, 'r') as file:
            return file["3BRecInfo"]["3BRecVars"]["SamplingRate"][0]

    header = get_header(filename)[0]
    return int(re.search("Sample rate = (\d+)", header).group(1))

def iter_chunks(array, max_chunk, verbose=False):
    "Iterate chunks on first dim of arrary."
    end_idx = array.shape[0]
    gen = range(0,end_idx,max_chunk)
    if verbose:
        gen = tqdm(gen)
    for s in gen:
        e = min(s+max_chunk, end_idx)
        yield array[s:e]


def read_spyking_results(filepath: str, retina_id, sampling_rate: int) -> (
        UnitSpikeTrains):
    """Read the results from Spyking Circus spike sorting
    and gives a list of arrays."""

    result_regex = re.compile('.*\.result.hdf5$')
    if not re.match(result_regex, filepath):
        raise ValueError('Filepath must end in "result.hdf5"')

    result_h5 = h5py.File(filepath, 'r')

    spike_units = {}
    for template in result_h5["spiketimes"]:
        unit = Unit(retina_id, None)
        unit.spike_train = np.array(
            result_h5["spiketimes"][template], dtype='int32') / sampling_rate
        spike_units[unit.id] = unit

    return spike_units


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

    dat_files = glob(my_path + '/*.dat')

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

def print_h5py_attrs(name, obj):
    shift = name.count('/') * '    '
    if type(obj)==h5py._hl.dataset.Dataset:
        prefix = "o "
        postfix = ": " + str(obj)
    elif type(obj)==h5py._hl.group.Group:
        prefix = "- "
        postfix = ""
    print(shift + prefix + name + postfix)
    for key, val in obj.attrs.items():
        print(shift + '    ' + f"{key}: {val}", end="")
        if type(val)==h5py._hl.dataset.Dataset:
            print(val)
        else:
            print("")

def describe_h5py(h5):
    h5.visititems(print_h5py_attrs)


# These may vary from recording to recording so skip...
to_skip = [ "3BRecInfo/3BRecVars/NRecFrames"
        , "3BData/Raw"
        , "3BData/3BInfo/3BNoise/StdMean"
        , "3BData/3BInfo/3BNoise/ValidChs"
        , "3BRecInfo/3BMeaStreams/Spikes/Params"
        , "3BUserInfo/ChsGroups"
        , "3BUserInfo/ExpMarkers"
        , "3BUserInfo/ExpNotes"
        , "3BUserInfo/MeaLayersInfo"
        , "3BUserInfo/TimeIntervals"
        ]
def copy_metadata(name, obj, copy_to, to_skip=to_skip):
    "copy {name: obj} to copy_to"
    if name in to_skip:
        pass
    else:
        if type(obj) is h5py._hl.group.Group:
            copy_to.require_group(name)
        else:
            if obj.shape[0] > 0:
                copy_to[name] = obj[()]

def get_images_from_vid(stimulus_list, frame_log, video_file):
    # sometimes we drop a frame on the first render of a stimuli, so take the last frame to be conservative
    # this index in frame-space
    last_frame_idx_of_stimuli = np.concatenate([
        np.where(np.diff(frame_log.stimulusIndex))[0],
        [frame_log.stimulusIndex.iloc[-1]]
    ])

    image_stim = list(filter(
        lambda s: s['stimulus']['stimulusType']=='IMAGE',
        stimulus_list))
    image_stim_idx = [s['stimulus']['stimulusIndex']
                            for s in image_stim]
    image_classes = [s['stimulus']['metadata']['class']
                            for s in image_stim]

    # # take last frame of stimulus as image is static
    image_stim_frame_idx = last_frame_idx_of_stimuli[image_stim_idx]

    # get first frame from video so we know the size
    container = av.open(video_file)
    for frame in container.decode(video=0):
        first_frame = frame.to_ndarray(format='rgb24')
        break

    frame_indices = list(image_stim_frame_idx)
    next_frame_idx = frame_indices.pop(0)

    # iterate through frames, and match to each entry in frame_indices
    # seeking by time would be faster, but does not map 1:1 with frame index
    frames = []
    for n,frame in tqdm(enumerate(container.decode(video=0)),total=frame_indices[-1]):
        if n==(next_frame_idx):
            frames.append(frame.to_ndarray(format='rgb24'))
            if len(frame_indices)!=0:
                next_frame_idx = frame_indices.pop(0)
            else:
                break


    if len(frames) != len(image_classes):
        raise ValueError(f"Different number of frames and classes (number of stimuli) found: {len(frames)} and {len(image_classes)}")
    return frames, image_classes