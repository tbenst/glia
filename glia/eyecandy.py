import re
import numpy as np
import requests
import json
import pickle
import yaml
from typing import List, Dict
from functools import reduce
import os
from uuid import uuid4, UUID
from scipy.ndimage import filters
from scipy import signal
from warnings import warn

from .files import sampling_rate, read_raw_voltage
# import pytest

file = str
Dir = str
dat = str
Hz = int
Seconds = float
ms = float
UnitSpikeTrains = List[Dict[str,np.ndarray]]


def create_eyecandy_gen(program_yaml, eyecandy_url):
    # Create program from eyecandy YAML. 
    r = requests.post(eyecandy_url + '/analysis/start-program',
                      data={'programYAML': program_yaml,
                           'windowHeight': 1140,
                           'windowWidth': 912})
    sid = r.text
    def eyecandy_gen():
        done = False
        while (True):
            # Get the next stimulus using program sid
            json = requests.get(eyecandy_url + '/analysis/program/{}'.format(sid)).json()
            done = json["done"]
            if done:
                break
            else:
                yield json["value"]

    return eyecandy_gen()

def open_lab_notebook(filepath):
    """Take a filepath for YAML lab notebook and return dictionary."""
    with open( filepath, 'r') as f:
        y = yaml.load(f)
    return y
def get_experiment_protocol(lab_notebook_yaml, name):
    """Given lab notebook, return protocol matching name."""
    study_data = lab_notebook_yaml['study']['data']
    for mouse in study_data:
        for retina in mouse["retinas"]:
            for protocol in retina["experiments"]:
                if protocol["name"]==name:
                    return protocol

def get_stimulus_from_protocol( protocol ):
    """Get stimulus text from protocol suitable for eye-candy."""
    return yaml.dump(protocol['stimulus']['program'])

def get_stimulus_start_times(analog_file, threshold=10000):
    """
    Given the input signal `y` with samples at times `t`,
    find the times where `y` increases through the value `threshold`.

    `t` and `y` must be 1-D numpy arrays.

    Linear interpolation is used to estimate the time `t` between
    samples at which the transitions occur.
    """
    #sampling rate
    fs = sampling_rate(analog_file)
    # Note: this is specific to an arbitrary Trigger 2 - TODO
    analog = read_raw_voltage(analog_file)[:,1]
    nsamples = analog.shape[0]
    T = nsamples/fs
    #t is times
    t = np.linspace(0, T, nsamples, endpoint=False)
    
    
    # Find where analog crosses the threshold (increasing).
    lower = (analog < threshold) & (analog > 1000)
    higher = analog >= threshold
    transition_indices = np.where(lower[:-1] & higher[1:])[0]

    # Linearly interpolate the time values where the transition occurs.
    t0 = t[transition_indices]
    t1 = t[transition_indices + 1]
    analog0 = analog[transition_indices]
    analog1 = analog[transition_indices + 1]
    slope = (analog1 - analog0) / (t1 - t0)
    transition_times = t0 + (threshold - analog0) / slope

    return list(transition_times)


def validate_stimulus_times(stimulus_list,start_times, stimulus_gen, ignore_extra=False):
    stimulus_length = len(stimulus_list)
    start_length = len(start_times)
    
    if stimulus_length == 0:
        raise ValueError("No stimulus start times detected in analog file. " \
                         "Try lowering the threshold or changing --trigger".format(start_length,stimulus_length))
    if stimulus_length == 1:
        warn("Only one stimulus detected")
    elif start_length > stimulus_length:
        raise ValueError("start_times ({}) is longer than stimulus_list ({}). " \
                         "Try raising the threshold".format(start_length,stimulus_length))
    # elif start_length < stimulus_length:
    #     raise ValueError("start_times ({}) is shorter than stimulus_list ({}). " \
    #                      "Try lowering the threshold".format(start_length,stimulus_length))

    try:
        print(next(stimulus_gen))
        if not ignore_extra:
            raise ValueError("More stimuli than start times detected." \
                "Use --ignore_extra to ignore.")
    except StopIteration as e:
        pass

    predicted_start_time = stimulus_list[0]["start_time"] + stimulus_list[0]["stimulus"]["lifespan"]/120
    for s in stimulus_list[1:]:
        start_time = s["start_time"]
        stimulus = s["stimulus"]
        lifespan = stimulus["lifespan"]/120
        try:
            assert np.abs(start_time - predicted_start_time) < 0.5
        except Exception as e:
            print("malformed stimulus list--try a different trigger, adjusting the threshold, or --fix-missing.")
            print("Expected start time of {} but found {} for {}".format(
                predicted_start_time,start_time, stimulus))
            # for s in stimulus_list:
            #     print(s['stimulus']['lifespan'], s['start_time'])
            raise
        predicted_start_time =  start_time + lifespan


    
def get_stimulus_from_eyecandy(start_times, eyecandy_gen, fix_missing=False):
    """Return list of tuples (start time, corresponding eyecandy_gen)"""
    # compensate for weird analog behavior at end of recording
    # start_times.pop()
    if fix_missing:
        ret = [{'start_time': start_times[0], 'stimulus': next(eyecandy_gen)}]
        predicted_start_time = ret[0]["stimulus"]["lifespan"]/120+start_times[0]
        for time in start_times[1:]:

            # only assign start times that roughly matched the prediction
            while(time > predicted_start_time):
                # this will eventually cause an error when the generator runs out
                if np.abs(time - predicted_start_time) < 0.5:
                    break

                try:
                    new = next(eyecandy_gen)
                except:
                    for r in ret:
                        print(r['start_time'],r['stimulus']['stimulusType'])
                    raise
                predicted_start_time += new['lifespan']/120
                ret.append({"stimulus": new, "start_time": None})
                print(time,predicted_start_time)

            try:
                new = next(eyecandy_gen)
            except:
                for r in ret:
                    print(r['start_time'],r['stimulus']['stimulusType'])
                raise
            predicted_start_time = time + new['lifespan']/120
            ret.append({"stimulus": new, "start_time": time})

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
            # print(r["stimulus"]["stimulusType"],forward_start,backward_start,r["start_time"])
        return stimulus_list
    else:
        return list((map(lambda x: {'start_time': x, 'stimulus': next(eyecandy_gen)}, start_times)))

def dump_stimulus(stimulus_list, file_path):
    pickle.dump(stimulus_list, open(file_path, "wb"))

def load_stimulus(file_path):
    return pickle.load(open(file_path, "rb"))

    
def create_experiments(unit: np.ndarray, stimulus_list,
                       #is this supposed to return a list of dictionaries?
                       duration: Seconds=100/1000) -> List[List[float]]:
    """Split firing train into experiments based on start times (e.g. on time).
    stimulus_list: List[(float, dict)]"""
    num_experiments = len(stimulus_list)
    #s[1] takes the next(eyecandy_gen) from the tuples in stimulus_list
    experiments = [{"stimulus": s[1], "spikes": []} for s in stimulus_list]
    i = 0
    #generates a list of spiketimes that are normalized by the stimulus start_time
    for spike in unit:
        #accounts for if duration is such that the spike time belongs to the next stimulus start_time
        #checks if at last experiment
        if i +1 < num_experiments and spike >= stimulus_list[i+1][0]:
            i += 1
            #
            if spike < stimulus_list[i][0] + duration:
                experiments[i]["spikes"].append(spike - stimulus_list[i][0])
        #checks if spiketime is between the start time and the duration we're counting for
        elif (spike >= stimulus_list[i][0]) and (spike < stimulus_list[i][0] + duration):
            # subtract start_time to normalize
            experiments[i]["spikes"].append(spike - stimulus_list[i][0])
    #a list of dictionaries for a given   
    return experiments

#Analytics is a dictionary of dictionaries. The keys are the stimulus types. 
# The values are dictionaries containing the information in stimulus (spike counts, angle, lifespan)
def split_by_experiment_type(experiments: List[List[float]], stimulus_type: str,
                   group_by: str) -> Dict[str, List[float]]:
    """"""
    analytics = {  "WAIT":  {},  "SOLID": {},  "BAR": {},  "GRATING": {} }
    # only bars and gratings have angles. only waits and solids have a lifespan
    has_angle=re.compile(r"^(BAR|GRATING)$")
    has_time=re.compile(r"^(WAIT|SOLID)$")
    #stimulus is a dictionary in experiments
    for stimulus in experiments:
        the_stimulus = stimulus["stimulus"]
        spike_count = len(stimulus["spikes"])
        stimulus_type = the_stimulus[ "stimulusType"]
        if has_angle.match(stimulus_type):
            angle = the_stimulus[ "angle"]
            try:
                analytics[stimulus_type][angle].append(spike_count)
            except:
                analytics[stimulus_type][angle]=[spike_count]
        elif has_time.match(stimulus_type):
            the_time = the_stimulus[ "lifespan"]
            try:
                analytics[stimulus_type][the_time].append(the_time)
            except:
                analytics[stimulus_type][the_time] = [the_time]
    return analytics


def get_threshold(analog_file, nsigma=3):
    analog = read_raw_voltage(analog_file)
    mean, sigma = analog.mean(), analog.std(ddof=1)
    return mean+nsigma*sigma


def get_flash_times(analog,sampling_rate,nsigma):
    "Returns start times when light detector records light above threshold."
    b,a = signal.butter(3,0.001)
    filtered = signal.filtfilt(b,a,analog)
    threshold = nsigma*np.std(filtered)
    transitions = np.diff((filtered > threshold).astype(int))
    return np.where(transitions==1)[0]/sampling_rate

def maybe_average(a,b):
    if not a:
        return b
    elif not b:
        return a
    else:
        return (a+b)/2

def get_start_times_of_stimulus(stimulus_type, stimulus_list):
    r = re.compile(r'^SOLID$')
    ret = []
    for (t,s) in stimulus_list:
        stimulus_type = s["stimulusType"]
        if r.match(stimulus_type):
            ret.append(t)
    return ret

def legacy_create_stimulus_list_from_flicker(analog_file, stimulus_file, lab_notebook_fp,
        data_name, eyecandy_url, ignore_extra=False, nsigma=3):
    # this will catch if the .stimulus file does not exist
    threshold = get_threshold(analog_file, nsigma)
    start_times = get_stimulus_start_times(analog_file, threshold)
    lab_notebook = open_lab_notebook(lab_notebook_fp)
    experiment_protocol = get_experiment_protocol(lab_notebook, data_name)
    stimulus_protocol = get_stimulus_from_protocol(experiment_protocol)
    stimulus_gen = create_eyecandy_gen(stimulus_protocol, eyecandy_url)
    stimulus_list = get_stimulus_from_eyecandy(start_times,stimulus_gen)
    validate_stimulus_times(stimulus_list, start_times, stimulus_gen, ignore_extra)
    # create the.stimulus file
    dump_stimulus(stimulus_list, stimulus_file)

    return stimulus_list

def create_stimulus_list_from_flicker(analog_file, stimulus_file, lab_notebook_fp,
        data_name, eyecandy_url, ignore_extra=False, nsigma=3, fix_missing=False):
    # this will catch if the .stimulus file does not exist
    threshold = get_threshold(analog_file, nsigma)
    analog = read_raw_voltage(analog_file)[:,1]
    sampling =sampling_rate(analog_file)
    start_times = get_flash_times(analog, sampling, nsigma)
    lab_notebook = open_lab_notebook(lab_notebook_fp)
    experiment_protocol = get_experiment_protocol(lab_notebook, data_name)
    stimulus_protocol = get_stimulus_from_protocol(experiment_protocol)
    stimulus_gen = create_eyecandy_gen(stimulus_protocol, eyecandy_url)
    stimulus_list = get_stimulus_from_eyecandy(start_times,stimulus_gen, fix_missing)
    validate_stimulus_times(stimulus_list, start_times, stimulus_gen, ignore_extra)
    # create the.stimulus file
    dump_stimulus(stimulus_list, stimulus_file)

    return stimulus_list

def create_stimulus_list_from_SOLID(analog_file, stimulus_file, lab_notebook_fp,
        data_name, eyecandy_url, ignore_extra=False, nsigma=3):
    "using the solid time as ground truth, construct estimates of intermediate stimulus time going forwards and backwards"
    threshold = get_threshold(analog_file, nsigma)
    lab_notebook = open_lab_notebook(lab_notebook_fp)
    experiment_protocol = get_experiment_protocol(lab_notebook, data_name)
    stimulus_protocol = get_stimulus_from_protocol(experiment_protocol)
    stimulus_gen = create_eyecandy_gen(stimulus_protocol, eyecandy_url)
    sample_rate = sampling_rate(analog_file)
    # specific to Trigger 2 - TODO
    analog = read_raw_voltage(analog_file)[:,1]

    stimuli = []
    for s in stimulus_gen:
        stimuli.append(s)

    flash_start_times = get_flash_times(analog,sample_rate)
    # check if number of detected flashes is equal to number of solid
    assert len(flash_start_times)==len([s for s in stimuli if s["stimulusType"]=="SOLID"])

    # using the solid time as ground truth, construct estimates of intermediate stimulus time going forwards and backwards
    solid_gen = iter(flash_start_times)
    forward_start_times = []

    # TODO: make a separate function, DRY
    start_time = None
    previous_duration = None
    for stimulus in stimuli:
        if stimulus["stimulusType"]=="SOLID":
            start_time = next(solid_gen)
            forward_start_times.append(start_time)
        else:
            if start_time:
                start_time += previous_duration
            forward_start_times.append(start_time)
        try:
            previous_duration = np.ceil(stimulus["lifespan"])/120
        except:
            print("malformed stimulus: {}".format(stimulus))
            raise

    solid_gen = reversed(flash_start_times)
    backward_start_times = []
    previous_duration = None
    start_time = None
    for stimulus in reversed(stimuli):
        if stimulus["stimulusType"]=="SOLID":
            start_time = next(solid_gen)
            backward_start_times.append(start_time)
        else:
            if start_time:
                start_time -= np.ceil(stimulus["lifespan"])/120
            backward_start_times.append(start_time)

    backward_start_times.reverse()

    # we now average forward and backwards estimate to construct the stimulus start time

    stimulus_list = []

    for forward, backward, stimulus in zip(forward_start_times,backward_start_times,stimuli):
        stimulus_list.append({'start_time': maybe_average(forward,backward),'stimulus': stimulus})
    
    validate_stimulus_times(stimulus_list, stimulus_list, stimulus_gen, ignore_extra)

    # create the.stimulus file
    dump_stimulus(stimulus_list, stimulus_file)

    return stimulus_list

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
            predicted_duration = np.ceil(stimulus["lifespan"])/120
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

# def histogram_of_stimulus(stimulus, experiments, bins = np.arange(0,1,10/1000)):
#     analytics = {  "WAIT":  {},  "SOLID": {},  "BAR": {},  "GRATING": {} }
#     has_time=re.compile(r"^{}$".format(stimulus))
#     for stimulus in experiments:
#         the_stimulus = stimulus["stimulus"]['value']
#         stimulus_type = the_stimulus[ "stimulusType"]
#         if has_angle.match(stimulus_type):
#             angle = the_stimulus[ "angle"]
#             try:
#                 analytics[stimulus_type][angle].append(spike_count)
#             except:
#                 analytics[stimulus_type][angle]=[spike_count]
#         elif has_time.match(stimulus_type):
#             the_time = the_stimulus[ "lifespan"]
#             try:
#                 analytics[stimulus_type][the_time].append(the_time)
#             except:
#                 analytics[stimulus_type][the_time] = [the_time]
#     return analytics