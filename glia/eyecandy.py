import re
import numpy as np
import asyncio
import requests
import json
import pickle
import yaml
import concurrent.futures
from typing import List, Dict
from functools import reduce
import os
from uuid import uuid4, UUID
from scipy.ndimage import filters
from scipy import signal
from warnings import warn
from webcolors import html5_parse_legacy_color
from .files import sampling_rate, read_raw_voltage, read_3brain_analog
from .config import logger
import matplotlib.pyplot as plt

file = str
Dir = str
dat = str
Hz = int
Seconds = float
ms = float
UnitSpikeTrains = List[Dict[str,np.ndarray]]

# TODO thread with async/await

async def get_stimulus(url):
    done = False
    stimuli = []
    while(not done):
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:

            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(
                    executor,
                    requests.get,
                    url
                )
                for i in range(20)
            ]
            for response in await asyncio.gather(*futures):
                try:
                    json = response.json()
                except:
                    # requests beyond last stimuli will return error
                    continue
                if json['done']==True:
                    done = True
                else:
                    value = json["value"]
                    value["stimulusIndex"] = json["stimulusIndex"]
                    stimuli.append(value)
    return iter(sorted(stimuli, key=lambda x: x['stimulusIndex']))


def create_epl_gen_v2(program, epl, window_width, window_height, seed,
        eyecandy_url):
    # Create program from eyecandy YAML.
    r = requests.post(eyecandy_url + '/analysis/run-program',
                      data={'program': program,
                           "epl": epl,
                           'windowHeight': window_height,
                           'windowWidth': window_width,
                           "seed": seed})
    try:
        response = r.json()
    except:
        print("couldn't parse json from ", r)
        raise

    stimuli = []
    for json in response['stimulusList']:
        value = json["value"]
        value["stimulusIndex"] = json["stimulusIndex"]
        stimuli.append(value)

    metadata = response["metadata"]
    assert type(metadata)==dict
    return (metadata,
        iter(sorted(stimuli, key=lambda x: x['stimulusIndex'])))

def open_lab_notebook(filepath):
    """Take a filepath for YAML lab notebook and return dictionary."""
    with open( filepath, 'r') as f:
        y = list(yaml.safe_load_all(f))
    try:
        assert "epl" in y[0]
    except:
        raise(ValueError(f"{filepath} is not in lab notebook format"))
    return y

def get_experiment_protocol(lab_notebook_yaml, name):
    """Given lab notebook, return protocol matching name."""
    for experiment in lab_notebook_yaml:
        logger.info(f"inside get_experiment_protocol: {experiment['filename']},"
        f" {name}")
        if experiment["filename"]==name:
            return experiment

    raise(Exception(f"Could not find matching experiment name."))

def get_epl_from_experiment(experiment):
    try:
        ret = (experiment["program"],experiment["epl"],
                        experiment["windowWidth"],experiment["windowHeight"],
                        experiment["seed"])
    except:
        raise(ValueError(f"""Could not find required field. Here's what we found
        {experiment}"""))
    return ret


def dump_stimulus(stimulus_list, file_path):
    ".stimulus file"
    pickle.dump(stimulus_list, open(file_path, "wb"))

def load_stimulus(file_path):
    return pickle.load(open(file_path, "rb"))

def save_stimulus(stimuli, file_path):
    ".stim file"
    with open(file_path, 'w') as outfile:
        yaml.dump(stimuli, outfile)

def read_stimulus(file_path):
    with open(file_path, 'r') as file:
        ret = yaml.safe_load(file)
    return (ret["metadata"], list(ret["stimulus_list"]), ret['method'])

def get_threshold(analog_file, nsigma=3):
    analog = read_raw_voltage(analog_file)
    mean, sigma = analog.mean(), analog.std(ddof=1)
    return mean+nsigma*sigma


def maybe_average(a,b):
    if not a:
        return float(b)
    elif not b:
        return float(a)
    else:
        return float((a+b)/2)


def fill_missing_stimulus_times(stimulus_list, reverse=False):
    ret = []
    if reverse:
        previous_start_time = None
        stimulus_list=reversed(stimulus_list)
    else:
        predicted_start_time = None

    for s in stimulus_list:
        stimulus = s['stimulus']
        start_time = s['start_time']

        try:
            predicted_duration = np.ceil(stimulus["lifespan"])
        except:
            print("malformed stimulus: {}".format(stimulus))
            raise

        if start_time == None:
            # try to estimate start_time
            if reverse:
                if previous_start_time != None:
                    start_time = previous_start_time - predicted_duration
            else:
                if predicted_start_time != None:
                    start_time = predicted_start_time

        # prepare for the next loop
        if start_time != None:
            if reverse:
                previous_start_time = start_time
            else:
                predicted_start_time = start_time + predicted_duration

        ret.append({'stimulus': stimulus, 'start_time': start_time})
    if reverse:
        return reversed(ret)
    else:
        return ret

def create_stimuli(analog_file, stimulus_file, lab_notebook_fp,
        data_name, eyecandy_url, analog_idx=1, ignore_extra=False, calibration='auto',
        within_threshold=None):
    """Uses stimulus index modulo 3 Strategy.

    calibration determines the mean in linear light space for each stimulus
    index"""
    logger.debug(f"create_stimuli analog_file: {analog_file}")
    if analog_file[-4:]==".brw":
        analog = read_3brain_analog(analog_file)
    else:
        analog = read_raw_voltage(analog_file)[:,analog_idx]
    if calibration=='auto':
        data_directory, name = os.path.split(stimulus_file)
        calibration = auto_calibration(analog, data_directory)
    else:
        calibration = np.array(calibration)

    sampling = sampling_rate(analog_file)
    lab_notebook = open_lab_notebook(lab_notebook_fp)
    experiment_protocol = get_experiment_protocol(lab_notebook, data_name)
    filtered = assign_stimulus_index_to_analog(analog,calibration)

    program, epl,window_width,window_height,seed = get_epl_from_experiment(
        experiment_protocol)
    metadata, stimulus_gen = create_epl_gen_v2(program, epl, window_width,
            window_height, seed, eyecandy_url)

    print('getting start times')
    start_times = get_stimulus_index_start_times(filtered,sampling,stimulus_gen,0.8)
    total = len(start_times)
    number_missing = len(list(filter(lambda x: x['start_time'] is None,
        start_times)))
    print(f'{number_missing}/{total} start times were missing. Estimating missing times')
    stimulus_list = estimate_missing_start_times(start_times)

    try:
        print('validating stimulus list')
        validate_stimulus_list(stimulus_list,stimulus_gen,ignore_extra, within_threshold)
    except:
        data_directory, name = os.path.split(analog_file)
        analog_histogram(analog, data_directory)
        save_stimulus(start_times, stimulus_file+'.debug')
        raise
    # create the.stimulus file
    stimuli = {"metadata": metadata, "stimulus_list": stimulus_list,
               "method": 'analog-flicker'}
    save_stimulus(stimuli, stimulus_file)

    return (metadata, stimulus_list)

def analog_histogram(analog, data_directory):
    fig,ax = plt.subplots()
    bins = np.linspace(np.min(analog),np.max(analog),128)
    histogram = ax.hist(analog,bins)
    fig.savefig(os.path.join(data_directory, "analog_histogram.png"))

def auto_calibration(analog, data_directory):
    bins = np.linspace(np.min(analog),np.max(analog),128)
    histogram = np.histogram(analog,bins)
    idx_local_maxima = np.logical_and(
        np.r_[True, histogram[0][1:] > histogram[0][:-1]],
        np.r_[histogram[0][:-1] > histogram[0][1:], True]
    )
    idx_above_thresh = histogram[0]>np.percentile(histogram[0],80)

    idx = np.where(np.logical_and(idx_local_maxima, idx_above_thresh))[0]
    v = histogram[1]
    try:
        assert(idx.size==6)
    except:
        logger.warning("found candidate values " \
            f"{np.round(v[idx])}. See analog_histogram.png for guidance in" \
            "manually setting the calibration such that it contains most values")
        analog_histogram(analog, data_directory)
        raise(ValueError("Autocalibration failed"))

    calibration = np.array([
        np.floor(v[max(0, idx[0]-5)]), np.ceil(v[min(v.size-1, idx[1]+5)]),
        np.floor(v[max(0, idx[2]-5)]), np.ceil(v[min(v.size-1, idx[3]+5)]),
        np.floor(v[max(0, idx[4]-5)]), np.ceil(v[min(v.size-1, idx[5]+5)])
    ])
    print(f"analog autocalibration of {calibration}")
    return calibration

def create_stimuli_without_analog(stimulus_file, lab_notebook_fp,
        data_name, eyecandy_url):
    lab_notebook = open_lab_notebook(lab_notebook_fp)
    experiment_protocol = get_experiment_protocol(lab_notebook, data_name)

    program, epl,window_width,window_height,seed = get_epl_from_experiment(
        experiment_protocol)
    metadata, stimulus_gen = create_epl_gen_v2(program, epl, window_width,
            window_height, seed, eyecandy_url)

    stimulus_list = []
    start_time = 0
    for stimulus in stimulus_gen:
        new = {'start_time': start_time,
            'stimulus': stimulus}
        stimulus_list.append(new)
        start_time += stimulus['lifespan']


    validate_stimulus_list(stimulus_list,stimulus_gen,False, 0.01)
    # create the.stimulus file
    stimuli = {"metadata": metadata, "stimulus_list": stimulus_list,
               "method": 'eyecandy-timing'}
    save_stimulus(stimuli, stimulus_file)

    return (metadata, stimulus_list)

#
# def get_index_near_value(analog, value, distance):
#     floor = value - distance
#     ceiling = value + distance
#     indices = np.where((analog > floor) & (analog < ceiling))
#     return indices

# deprecate
# def get_stimulus_values_from_analog(analog, calibration=(0.88,0.55,0.24)):
#     # according to flickr 0.4 linear lightspace
#     maximum = np.max(analog)
#     minimum = np.min(analog)
#     # mean value{light intensity}, refers to stimulus modulus
#     stimulus_0 = float(calibration[0]*(maximum-minimum)+minimum)
#     stimulus_1 = float(calibration[1]*(maximum-minimum)+minimum)
#     stimulus_2 = float(calibration[2]*(maximum-minimum)+minimum)
#     return (stimulus_0, stimulus_1, stimulus_2)

def assign_stimulus_index_to_analog(analog,calibration):
    # eyecandy flicker start in the middle, then bottom, then top, repeat
    # not logical, but there for legacy reasons
    stimulus_1 = np.where(
        np.logical_and(
            analog>calibration[0],
            analog<calibration[1]
        )
    )[0]

    stimulus_0 = np.where(
        np.logical_and(
            analog>calibration[2],
            analog<calibration[3]
        )
    )[0]

    stimulus_2 = np.where(
        np.logical_and(
            analog>calibration[4],
            analog<calibration[5]
        )
    )[0]
    filtered = np.full(analog.shape,-1,dtype="int8")
    filtered[stimulus_0] = 0
    filtered[stimulus_1] = 1
    filtered[stimulus_2] = 2
    return filtered

def state_lasts_full_frame(state,index,filtered, sampling_rate, percentage_threshold):
    "Validate that state lasts for at least a frame (8ms)."
    frame_length = int(np.ceil(sampling_rate*48/1000))
    frame = filtered[index:index+frame_length]
    state_count = (frame==state).sum()
#     print("full frame",state_count,frame_length)

    # percentage_threshold==0.5: at least half of these values must have correct state
    if state_count>frame_length*percentage_threshold:
        return True
    else:
        return False


# good optimization candidate..
def get_stimulus_index_start_times(filtered,sampling_rate, stimulus_gen, percentage_threshold):
    m = filtered.size
    previous_state = -1
    next_stimulus = next(stimulus_gen)
    stimulus_list = []
    for i,state in np.ndenumerate(filtered):
        i = i[0]

        #
        if state==-1:
            # invalid state
            pass
        elif previous_state==state:
            # still in same stimulus
            pass
        elif not state_lasts_full_frame(state,i,filtered,sampling_rate,percentage_threshold):
            # light level does not persist for full frame, likely transient
            pass
        elif previous_state==-1:
            # need to account for initial condition of previous is None
            # eg state==2 yields 0,1
            # print(state)
            for s in range(0,state):
                # account for missing
                stimulus_list.append({"start_time": None,
                                      "stimulus": next_stimulus})
                # should handle StopIteration TODO
                next_stimulus = next(stimulus_gen)

            stimulus_list.append({"start_time": float(i/sampling_rate),
                                 "stimulus": next_stimulus})
            next_stimulus = next(stimulus_gen)
            previous_state = state

        else:
            # new stimulus detected--transition states
            try:
                # print(next_stimulus)
                # flickering encodes modulus
                next_stimulus_index = next_stimulus["stimulusIndex"]%3

                if state==next_stimulus_index:
                    # state transition happened with no errors
                    # print(next_stimulus_index)
                    stimulus_list.append({"start_time": i/sampling_rate,
                                         "stimulus": next_stimulus})
                    next_stimulus = next(stimulus_gen)
                    previous_state = state

                else:
                    # current state does not match anticipated next
                    logger.warning(f"""got state {state} but expected
                        {next_stimulus_index} at index {i} and
                        stimulus index {next_stimulus['stimulusIndex']}""")
                    if state==(next_stimulus_index+1)%3:
                        # recover if we only missed one
                        stimulus_list.append({"start_time": None,
                                          "stimulus": next_stimulus})
                        next_stimulus = next(stimulus_gen)
                        stimulus_list.append({"start_time": i/sampling_rate,
                                             "stimulus": next_stimulus})
                        next_stimulus = next(stimulus_gen)
                        previous_state = state

                    else:
                        raise Exception("unrecoverable: missing 2 stimuli",i,state,stimulus_list)
            except StopIteration:
                return stimulus_list

    return stimulus_list

def estimate_missing_start_times(ret):
    "Takes a stimulus list, replaces start_time==none With an estimate"
    forward_stimulus_list = fill_missing_stimulus_times(ret)
    backward_stimulus_list = fill_missing_stimulus_times(ret, reverse=True)
    stimulus_list = []
    for forward, backward, r in zip(forward_stimulus_list,backward_stimulus_list,ret):
        forward_start = forward["start_time"]
        backward_start = backward["start_time"]
        assert forward["stimulus"]==backward["stimulus"]
        stimulus = forward["stimulus"]
        stimulus_list.append({'start_time': maybe_average(forward_start,backward_start),
            'stimulus': stimulus})

    return stimulus_list


def next_state(current_state):
    if current_state==0:
        return (1,2)
    elif current_state==1:
        return (2,0)
    elif current_state==2:
        return (0,1)
    else:
        raise ValueError("Invalid state")

def validate_stimulus_list(stimulus_list,stimulus_gen,ignore_extra=True,
                            within_threshold=None):
    # probably should modify to compare with filtered
    try:
        print(next(stimulus_gen),
            f"previous start time: {stimulus_list[-1]['start_time']}")
        if not ignore_extra:
            raise ValueError("More stimuli than start times detected." \
                "Perhaps experiment ended early?" \
                "Use --ignore-extra to ignore (At your own risk!)")
    except StopIteration as e:
        pass

    predicted_start_time = stimulus_list[0]["start_time"] + stimulus_list[0]["stimulus"]["lifespan"]
    previous_lifespan = 0
    for s in stimulus_list[1:]:
        start_time = s["start_time"]
        stimulus = s["stimulus"]
        lifespan = stimulus["lifespan"]
        try:
            if within_threshold is not None:
                np.abs(start_time - predicted_start_time) < within_threshold
            elif previous_lifespan>10:
                # temporary hack while eye-candy is frame based
                assert np.abs(start_time - predicted_start_time) < previous_lifespan/20
            else:
                assert np.abs(start_time - predicted_start_time) < 0.5
        except Exception as e:
            print(previous_lifespan,lifespan)
            print("malformed stimulus list--try a different trigger, adjusting the threshold, or --fix-missing.")
            print("Expected start time of {} but found {} for {}".format(
                predicted_start_time,start_time, stimulus))
            # for s in stimulus_list:
            #     print(s['stimulus']['lifespan'], s['start_time'])
            raise
        predicted_start_time =  start_time + lifespan
        previous_lifespan = lifespan

def hex_to_linear(i):
    # i in [0,255]
    # account for gamma compression
    linear = (i/255)**2.2
    return linear

def color_to_linear(color):
    return hex_to_linear(sum(html5_parse_legacy_color(color))/3)

def checkerboard_contrast(stimulus):
    color = color_to_linear(stimulus['color'])
    alternateColor = color_to_linear(stimulus['alternateColor'])
    return np.abs(color - alternateColor)

def bar_contrast(stimulus):
    color = color_to_linear(stimulus['barColor'])
    alternateColor = color_to_linear(stimulus['backgroundColor'])
    return np.abs(color - alternateColor)

grating_contrast = bar_contrast
