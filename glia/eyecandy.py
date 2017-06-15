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

from .files import sampling_rate, read_raw_voltage
from .config import logger
# import pytest

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




def create_epl_gen(program, epl, window_width, window_height, seed,
        eyecandy_url):
    # Create program from eyecandy YAML. 
    r = requests.post(eyecandy_url + '/analysis/start-program',
                      data={'program': program,
                           "epl": epl,
                           'windowHeight': window_height,
                           'windowWidth': window_width,
                           "seed": seed})
    sid = r.text
    url = eyecandy_url + '/analysis/program/{}'.format(sid)

    loop = asyncio.get_event_loop()
    generator = get_stimulus(url)
    a = loop.run_until_complete(generator)
    print(a)
    return a

def open_lab_notebook(filepath):
    """Take a filepath for YAML lab notebook and return dictionary."""
    with open( filepath, 'r') as f:
        y = list(yaml.safe_load_all(f))
    return y
    
def get_experiment_protocol(lab_notebook_yaml, name):
    """Given lab notebook, return protocol matching name."""
    for experiment in lab_notebook_yaml:
        if experiment["filename"]==name:
            return experiment

def get_epl_from_experiment(experiment):
    try:
        ret = (experiment["program"],experiment["epl"],
                        experiment["windowWidth"],experiment["windowHeight"],
                        experiment["seed"])
    except:
        raise(ValueError("Must use older version glia with frame-based lifespan."))
    return ret


def dump_stimulus(stimulus_list, file_path):
    ".stimulus file"
    pickle.dump(stimulus_list, open(file_path, "wb"))

def load_stimulus(file_path):
    return pickle.load(open(file_path, "rb"))

def save_stimulus(stimulus_list, file_path):
    ".stim file"
    with open(file_path, 'w') as outfile:
        yaml.dump(stimulus_list, outfile)

def read_stimulus(file_path):
    with open(file_path, 'r') as file:
        ret = list(yaml.safe_load(file))
    return ret

def get_threshold(analog_file, nsigma=3):
    analog = read_raw_voltage(analog_file)
    mean, sigma = analog.mean(), analog.std(ddof=1)
    return mean+nsigma*sigma


def maybe_average(a,b):
    if not a:
        return b
    elif not b:
        return a
    else:
        return (a+b)/2


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

def create_stimulus_list(analog_file, stimulus_file, lab_notebook_fp,
        data_name, eyecandy_url, ignore_extra=False, calibration=(0.55,0.24,0.88),
        distance=1100, within_threshold=None):
    """Uses stimulus index modulo 3 Strategy.

    calibration determines the mean in linear light space for each stimulus index"""
    analog = read_raw_voltage(analog_file)[:,1]
    sampling = sampling_rate(analog_file)
    lab_notebook = open_lab_notebook(lab_notebook_fp)
    experiment_protocol = get_experiment_protocol(lab_notebook, data_name)
    stimulus_values = get_stimulus_values_from_analog(analog,calibration)
    filtered = assign_stimulus_index_to_analog(analog,stimulus_values,distance)

    program, epl,window_width,window_height,seed = get_epl_from_experiment(
        experiment_protocol)
    stimulus_gen = create_epl_gen(program, epl, window_width,
            window_height, seed, eyecandy_url)
        
    start_times = get_stimulus_index_start_times(filtered,sampling,stimulus_gen,0.8)
    stimulus_list = estimate_missing_start_times(start_times)

    validate_stimulus_list(stimulus_list,stimulus_gen,ignore_extra, within_threshold)
    # create the.stimulus file
    dump_stimulus(stimulus_list, stimulus_file)

    return stimulus_list

def create_stimulus_list_without_analog(stimulus_file, lab_notebook_fp,
        data_name, eyecandy_url):
    lab_notebook = open_lab_notebook(lab_notebook_fp)
    experiment_protocol = get_experiment_protocol(lab_notebook, data_name)

    program, epl,window_width,window_height,seed = get_epl_from_experiment(
        experiment_protocol)
    stimulus_gen = create_epl_gen(program, epl, window_width,
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
    save_stimulus(stimulus_list, stimulus_file)

    return stimulus_list


def get_index_near_value(analog, value, distance):
    floor = value - distance
    ceiling = value + distance
    indices = np.where((analog > floor) & (analog < ceiling))
    return indices

def get_stimulus_values_from_analog(analog, calibration=(0.88,0.55,0.24)):
    # according to flickr 0.4 linear lightspace
    maximum = np.max(analog)
    minimum = np.min(analog)
    # mean value{light intensity}, refers to stimulus modulus
    stimulus_0 = calibration[0]*(maximum-minimum)+minimum
    stimulus_1 = calibration[1]*(maximum-minimum)+minimum
    stimulus_2 = calibration[2]*(maximum-minimum)+minimum
    return (stimulus_0, stimulus_1, stimulus_2)

def assign_stimulus_index_to_analog(analog,stimulus_values, distance=1200):
    stimulus_0 = get_index_near_value(analog,stimulus_values[0], distance)
    stimulus_1 = get_index_near_value(analog,stimulus_values[1], distance)
    stimulus_2 = get_index_near_value(analog,stimulus_values[2], distance)
    filtered = np.full(analog.shape,-1,dtype="int")
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
    

# @jit
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

            stimulus_list.append({"start_time": i/sampling_rate, 
                                 "stimulus": next_stimulus})
            next_stimulus = next(stimulus_gen)
            previous_state = state
           
        else:
            try:
                # new stimulus detected--transition states
    #             print(next_stimulus)
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
        print(next(stimulus_gen))
        if not ignore_extra:
            raise ValueError("More stimuli than start times detected." \
                "Use --ignore-extra to ignore.")
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
