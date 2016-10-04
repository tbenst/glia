import glia
import click
import os
import re

import glia
import numpy as np
from functools import reduce
import os
from uuid import uuid4, UUID
import re
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.ndimage import filters
from scipy import signal
from matplotlib.ticker import FuncFormatter, MultipleLocator
from datetime import datetime
from datetime import timedelta
from scipy import stats
import reprlib
from warnings import warn

def load_flicker(data_file, data_directory):


	units = glia.read_plexon_txt_file(data_file, uuid4())
	try:
	    stimulus_list = glia.load_stimulus(stimulus_file)
	except OSError:
	    # this will catch if the .stimulus file does not exist
	    threshold = glia.get_threshold(analog_file)
	    start_times = glia.get_stimulus_start_times(analog_file, threshold)
	    lab_notebook = glia.open_lab_notebook(lab_notebook_fp)
	    experiment_protocol = glia.get_experiment_protocol(lab_notebook, data_name)
	    stimulus_protocol = glia.get_stimulus_from_protocol(experiment_protocol)
	    stimulus_gen = glia.create_eyecandy_gen(stimulus_protocol, eyecandy_url)
	    stimulus_list = glia.get_stimulus_from_eyecandy(start_times,stimulus_gen)
	    glia.validate_stimulus_times(stimulus_list, start_times)
	    # create the.stimulus file
	    glia.dump_stimulus(stimulus_list, stimulus_file)

	return (units, stimulus_list)

def load_direct(data_file, data_directory):
	"using the solid time as ground truth, construct estimates of intermediate stimulus time going forwards and backwards"
	units = glia.read_plexon_txt_file(data_file, uuid4())
	try:
	    stimulus_list = glia.load_stimulus(stimulus_file)
	except OSError:
	    # this will catch if the .stimulus file does not exist

	    threshold = glia.get_threshold(analog_file)
	    analog = glia.read_raw_voltage(analog_file)[:,1]
	    # start_times = glia.get_stimulus_start_times(analog_file, threshold)
	    lab_notebook = glia.open_lab_notebook(lab_notebook_fp)
	    experiment_protocol = glia.get_experiment_protocol(lab_notebook, data_name)
	    stimulus_protocol = glia.get_stimulus_from_protocol(experiment_protocol)
	    stimulus_gen = glia.create_eyecandy_gen(stimulus_protocol, eyecandy_url)
	    sample_rate = glia.sampling_rate(analog_file)
	    stimuli = []
	    for s in stimulus_gen:
	        stimuli.append(s['value'])

	    flash_start_times = get_flash_times(analog,sample_rate)
	    # check if number of detected flashes is equal to number of solid
	    assert len(flash_start_times)==len([s for s in stimuli if s["stimulusType"]=="SOLID"])

	    # using the solid time as ground truth, construct estimates of intermediate stimulus time going forwards and backwards
	    solid_gen = iter(flash_start_times)
	    forward_start_times = []

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

	        previous_duration = np.ceil(stimulus["lifespan"])/120

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
	    
	    glia.validate_stimulus_times(stimulus_list, start_times)

	    # create the.stimulus file
	    glia.dump_stimulus(stimulus_list, stimulus_file)

def get_flash_times(analog,sampling_rate):
    "Returns start times when light detector records light above threshold."
    b,a = signal.butter(3,0.001)
    filtered = signal.filtfilt(b,a,analog)
    threshold = 2*np.std(filtered)
    transitions = np.diff((filtered > threshold).astype(int))
    return np.where(transitions==1)[0]/sampling_rate

def maybe_average(a,b):
    if not a:
        return b
    elif not b:
        return a
    else:
        return (a+b)/2