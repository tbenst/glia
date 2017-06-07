from uuid import uuid4
from hashlib import md5
import base64
import glia.humanhash as humanhash
import numpy as np
import glia.config as config
from collections import namedtuple as nt
from warnings import warn


SpikeTrain = np.ndarray
StartTime = int
ExperimentIFR = nt("ExperimentIFR", ["ifr", "start_time", "stimulus"])

Experiment = nt("Experiment", ["spike_train", "start_time", "stimulus"])

class Analytic(object):
    """docstring for Analytic"""
    def __init__(self, experiment: Experiment):
        super(Analytic, self).__init__()
        self.start_time = experiment.start_time
        self.stimulus = experiment.stimulus

        

class Mouse:
    def __init__(self, mouse_line, dob, gender):
        self.id = uuid4()
        self.mouse_line = mouse_line
        self.dob = dob
        self.gender = gender


class Retina:
    def __init__(self, mouse_id, name, location):
        self.id = uuid4()
        self.mouse_id = mouse_id
        self.name = name
        self.location = location

# use md5 as key to retrieve unit_name

class Unit:
    
    # store humanized names as value with unit_id as key
    lookup = {}

    def __init__(self, retina_id, channel, unit_num, spike_train=None):
        # id will be URL safe MD5 hash of spike_train
        my_id =  retina_id + '_' + str(channel) + "_"+str(unit_num)
        self.id = my_id
        self.retina_id = retina_id
        self.channel = channel
        self.unit_num = unit_num
        if spike_train is None:
            self.spike_train = []
        else:
            self.spike_train = spike_train

        Unit.lookup[my_id] = self

get_lifespan = lambda e: e["stimulus"]["lifespan"]