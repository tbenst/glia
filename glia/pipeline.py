import functools
from functools import reduce
from typing import Dict, Callable
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Any, Union
from .functional import f_filter, f_map, f_reduce
from scipy import signal
from glia.classes import Unit
from tqdm import tqdm


file = str
Dir = str
dat = str
Hz = int
SpikeUnits = List[np.ndarray]
SpikeUnits = List[np.ndarray]
Seconds = float
ms = float

# Must have these two keys: (stimulus, spikes)
# spikes is a list of spike times List[float]
Experiment = Dict
Experiments = List[Dict]
SpikeTrain = List[float]
# Mustbe: {"WAIT", "SOLID", "BAR", "GRATING"}
StimulusType = str
SpikeTrains = List[SpikeTrain]
Analytics = Dict[str,Any]


def apply_pipeline(pipeline, units):
    print("Applying pipeline")
    return {k: (pipeline(v.spike_train) if (type(v) is Unit) \
        else pipeline(v)) for k,v in tqdm(units.items())}

def f_create_experiments(stimulus_list: List[Dict], prepend_start_time=0, append_lifespan=0,
                         append_start_time=None):
    """
    Split spike train into individual experiments according to stimulus list.

    If append_start_time is given, ignore lifespan and return experiments of duration
    append_start_time + prepend_start_time.
    """

    def create_experiments(spike_train: np.ndarray) -> List[Dict]:
        experiments = [{"stimulus": s["stimulus"], "spikes": []} for s in stimulus_list]

        for i, stimulus in enumerate(stimulus_list):
            start_time = stimulus["start_time"] - prepend_start_time
            if append_start_time is not None:
                end_time = start_time + append_start_time
            else:
                end_time = start_time + stimulus["stimulus"]["lifespan"]/120 + append_lifespan + prepend_start_time

            bool_indices = (spike_train > start_time) & (spike_train < end_time)

            raw_spike_train = spike_train[bool_indices]

            # normalize to start of stimulus
            experiments[i]['spikes'] = raw_spike_train - start_time
        return experiments

    return create_experiments

def f_has_stimulus_type(stimulus_type: Union[str]) -> Callable[[List[Experiment]], List[Experiment]]:
    if type(stimulus_type) is str:
        stimulus_type = [stimulus_type]
    def anonymous(e):
        if (e["stimulus"]["stimulusType"] in stimulus_type ):
            return True
        else:
            return False
    return f_filter(anonymous)
    
def f_group_by(stimulus_parameter) -> Callable[[List[Experiment]], Dict]:
    def anonymous(accumulator,experiment):
        parameter = experiment["stimulus"][stimulus_parameter]
        spikes = experiment["spikes"]
        new_accumulator = accumulator.copy()
        if parameter in new_accumulator:
            new_accumulator[parameter].append(spikes)
        else:
            new_accumulator[parameter] = [spikes]
        return new_accumulator
    return f_reduce(anonymous, {})

def f_group_by_stimulus() -> Callable[[List[Experiment]], Dict]:
    def anonymous(accumulator,experiment):
        experiment["stimulus"].pop('stimulusIndex', None)
        parameter = str(experiment["stimulus"])
        spikes = experiment["spikes"]
        new_accumulator = accumulator.copy()
        if parameter in new_accumulator:
            new_accumulator[parameter].append(spikes)
        else:
            new_accumulator[parameter] = [spikes]
        return new_accumulator
    return f_reduce(anonymous, {})

def f_count_each_in_group() -> Callable[[Dict[str,List[SpikeTrain]]], Analytics]:
    def create_analytics(analytics):
        new = {}
        for k,v in analytics.items():
            # map count
            new[k] = list(map(lambda x: len(x),v))
        return new
    return create_analytics



def f_calculate_firing_rate_by_stimulus():
    def create_analytics(analytics):
        new = {}
        for stimulus_key, spike_trains in analytics.items():
            stimulus = eval(stimulus_key)
            duration = stimulus["lifespan"]/120
            new[stimulus_key] = list(map(lambda x: len(x) / duration,
                                       spike_trains))
        return new
    return create_analytics



def f_calculate_firing_rate_by_waveperiod():
    def create_analytics(analytics):
        new = {}
        for stimulus_key, spike_trains in analytics.items():
            stimulus = eval(stimulus_key)
            duration =  stimulus["wavelength"]/stimulus["speed"]
            new[stimulus_key] = list(map(lambda x: len(x) / duration,
                                       spike_trains))
        return new
    return create_analytics

def f_split_by_wavelength(skip_initial=False, screen_width=None,
        screen_height=None, number_of_wavelengths=1):
    "given a spike train, return a list of spike trains for each waveperiod."
    def split(spike_train, wavelength, speed):
        "return a list of relative-time spike_trains."
        wave_period = wavelength/speed
        if skip_initial:
            initial_wait_time = np.sqrt(screen_height**2 + screen_width**2)/speed - wave_period
        else:
            initial_wait_time = 0
        new_samples = [[]]
        i = 0
        period_start_time = initial_wait_time
        period_end_time = period_start_time + wave_period

        # TODO can make this much faster
        for spike in spike_train:
            if spike < period_start_time:
                continue
            elif (spike >= period_end_time):
                # add an empty array as no spike in this period.
                while (spike >= period_end_time):
                    period_start_time = period_end_time
                    period_end_time += wave_period
                    i += 1
                    new_samples.append([])
            if (spike >= period_start_time) and (spike < period_end_time):
                relative_spike_time = spike - period_start_time
                new_samples[i].append(relative_spike_time)
        return [np.array(s) for s in new_samples]
    
    def split_each(e):
        spike_train = e["spikes"]
        stimulus = e["stimulus"]
        wavelength = stimulus["wavelength"]
        speed = stimulus["speed"]
        return {"stimulus": stimulus,
            "train_list": split(spike_train,wavelength*number_of_wavelengths,speed)}

    def create_analytics(experiments):
        return list(map(split_each,experiments))
    return create_analytics

def list_average(l):
    accumulated = 0
    for item in l:
        accumulated+=item
    return accumulated/len(l)


def group_by_contrast(dictionary_by_stimulus):
    contrasts = {}
    
    for stimulus, value in dictionary_by_stimulus.items():
        # convert from hex and drop the #.
        try: 
            background_color = int(eval(stimulus)["backgroundColor"][1:], 16)
        except: 
            background_color = int(eval(stimulus)["backgroundColor"][0][1:], 16)
        bar_color = int(eval(stimulus)["barColor"][1:], 16)
        contrast = (bar_color - background_color)/(bar_color + background_color)
        if contrast in contrasts:
            raise ValueError("function not designed for multiple stimuli with the same contrast")
        contrasts[contrast] = value
        
    return contrasts

def count_spikes(spike_train):
    return f_map(lambda x: len(x))(spike_train)

def sort_two_arrays_by_first(a,b):
    b = [y for (x,y) in sorted(zip(a,b))]
    a = sorted(a)
    return (a,b)
assert(sort_two_arrays_by_first([ 0,   1,   1,    0,   1,   2,   2,   0,   1], \
                               ["a", "b", "c", "d", "e", "f", "g", "h", "i"]) == \
       ([0,0,0,1,1,1,1,2,2],["a", "d", "h", "b", "c", "e", "i", "f", "g"]))

def f_normalize_dictionary_values(factor="max"):
    def normalize_dictionary_values(dictionary):
        new = {}
        contrasts = sorted(dictionary.keys()) 
        if factor == "max":
            normalization_factor = dictionary[contrasts[-1]]
        elif factor == "min":
            normalization_factor = dictionary[contrasts[0]]

        if normalization_factor == 0: 
            return None 
        for key, value in dictionary.items():
            new[key] = value/normalization_factor

        return new
    return normalize_dictionary_values

def ISI(experiments):
    new_experiments = []
    for experiment in experiments:
        stimulus = experiment["stimulus"]
        new_experiment = {}
        spike_train = experiment["spikes"]
        new_experiment["spikes"] = np.diff(spike_train)
        new_experiment["stimulus"] = stimulus
        new_experiments.append(new_experiment)
    return new_experiments


def f_instantaneous_firing_rate(bandwidth, bin_width=0.001, sigma=6):
    "Reasonable values for bandwidth are 50-300ms"      
    def anonymous(experiments): 
        new_experiments = []
        for experiment in experiments:
            stimulus = experiment["stimulus"]
            spike_train = experiment["spikes"]
            end_time = np.ceil(stimulus["lifespan"]/120)
            
            new_experiment = {}
            new_experiment["spikes"] = IFR(spike_train,end_time,bandwidth,bin_width,sigma)
            new_experiment["stimulus"] = stimulus
            new_experiments.append(new_experiment)
        return new_experiments
    return anonymous

def IFR(spike_train, end_time, bandwidth, bin_width=0.001, sigma=6):
    transformed_sigma = bandwidth/bin_width
    bins = np.arange(0,end_time+bin_width, bin_width)
    spike_train_to_indices = np.vectorize(lambda x: np.digitize(x, bins))
    if len(spike_train)==0:
        ifr = np.zeros(bins.size)
    else:
        spike_bins = np.zeros(bins.size)
        if type(spike_train) is list:
            for train in spike_train:                
                indices = spike_train_to_indices(train)
                spike_bins[indices] += 1
        else:
            indices = spike_train_to_indices(spike_train)
            spike_bins[indices] = 1
        # multiplied by two to adjust distribution on both sides of window
        window = signal.gaussian(2*sigma*transformed_sigma, std=transformed_sigma)
        ifr = signal.convolve(spike_bins, window,mode="same")
    return ifr

def get_unit(units):
    return next(iter(units.items()))


def by_speed_width_then_angle(unit):
    # we will accumulate by angle in this dictionary and then divide
    by_speed_width_then_angle = {}
    
    for key,rate_list in unit.items():
        stimulus = eval(key)
#         speed_width = "speed:%s, width:%s" % (stimulus["speed"], stimulus["width"])
        speed_width = (stimulus["speed"], stimulus["width"])
        angle = stimulus["angle"]
        for rate in rate_list:
            if speed_width not in by_speed_width_then_angle:
                by_speed_width_then_angle[speed_width] = {}
            if angle in by_speed_width_then_angle[speed_width]:
                by_speed_width_then_angle[speed_width][angle]["accumulator"]+=rate
                by_speed_width_then_angle[speed_width][angle]["number_total"]+=1
            else:
                by_speed_width_then_angle[speed_width][angle] = {}
                by_speed_width_then_angle[speed_width][angle]["accumulator"]=rate
                by_speed_width_then_angle[speed_width][angle]["number_total"]=1
    
    ret = {}
    
    for speed_width,angles in by_speed_width_then_angle.items():
        for angle,analytics in angles.items():
            if speed_width not in ret:
                    ret[speed_width] = {}
            ret[speed_width][angle] = analytics["accumulator"]/analytics["number_total"]
        
    return ret

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def cartesian2polar(array):
    r = np.sqrt(array[0]**2 + array[1]**2)
    theta = np.arctan2(array[1], array[0])
    return np.array([r,theta])

def polar2cartesian(array):
    x = array[0] * np.cos(array[1])
    y = array[0] * np.sin(array[1])
    return np.array([x,y])

assert cartesian2polar(polar2cartesian(np.array([1, np.pi])))[0] == 1
assert cartesian2polar(polar2cartesian(np.array([1, np.pi])))[1] == np.pi

def calculate_dsi_by_speed_width(unit):
    to_return = {}
    overall_direction_preference = np.array([0,0])
    overall_spikes = 0
    
    for speed_width,angles in unit.items():
        direction_preference_vector = np.array([0,0])
        total_spikes = 0
        
        for angle,average_spike_count in angles.items():
            direction_preference_vector = direction_preference_vector + \
                polar2cartesian(np.array([average_spike_count,angle]))
            total_spikes += average_spike_count
            # divide r / total_spikes
        to_return[speed_width] = cartesian2polar(direction_preference_vector)[0]/total_spikes
        
        overall_direction_preference = overall_direction_preference + direction_preference_vector
        overall_spikes += total_spikes
    
    to_return["overall"] = cartesian2polar(overall_direction_preference)[0]/overall_spikes
        
    return to_return

def osi(a,b,sum_total_response):
    return np.sqrt(a**2+b**2)/sum_total_response

# new based on swindale
def calculate_osi_by_speed_width(unit):
    to_return = {}
    overall_a = 0
    overall_b= 0
    overall_spikes = 0
    
    for speed_width,angles in unit.items():
        a = 0
        b = 0
        total_spikes = 0
        
        for angle,average_spike_count in angles.items():
            a += average_spike_count * np.sin(2*angle)
            b += average_spike_count * np.cos(2*angle)
            total_spikes += average_spike_count
            
            overall_a += average_spike_count * np.sin(2*angle)
            overall_b += average_spike_count * np.cos(2*angle)
        # divide r / total_spikes
        to_return[speed_width] = osi(a, b, total_spikes)
        
        overall_spikes += total_spikes
    
    
    to_return["overall"] = osi(overall_a, overall_b, overall_spikes)
        
    return to_return

def f_get_key(i):
    return lambda item: item[i]

def f_calculate_peak_ifr_by_stimulus(bandwidth=0.15, bin_width=0.001, sigma=6):
    def create_analytics(analytics):
        new = {}
        for stimulus_key, spike_trains in analytics.items():
            stimulus = eval(stimulus_key)
            duration = stimulus["lifespan"]/120
            new[stimulus_key] = list(map(lambda x: np.max(IFR(x,duration,bandwidth,bin_width,sigma)),
                                       spike_trains))
        return new
    return create_analytics

def concatenate_by_stimulus(unit):
    return {stimulus: np.sort(np.hstack(values)) for stimulus, values in unit.items()}
