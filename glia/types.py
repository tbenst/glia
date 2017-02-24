from uuid import uuid4
from hashlib import md5
import base64
import glia.humanhash as humanhash
import numpy as np
import glia.config as config
from collections import namedtuple as nt

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
    name_lookup = {}

    def __init__(self, retina_id, channel, unit_num):
        # id will be URL safe MD5 hash of spike_train
        self._id = None
        self._name = None
        self.retina_id = retina_id
        self.channel = channel
        self.unit_num = unit_num
        self.spike_train = None
    

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    def initialize_id(self):
        "Initialize id and name after adding Spike train."
        if self.spike_train.size != 0:
            md5_hash = md5(self.spike_train.tostring())
            self._id = str(base64.urlsafe_b64encode(md5_hash.digest()),"utf8")
            name = "{}-{}_".format(self.channel, self.unit_num) + \
                humanhash.humanize(md5_hash.hexdigest())
            self._name = name
            Unit.name_lookup[self._id] = name
        else:
            self._id = "no-spikes"
            self._name = "no-spikes"
            Unit.name_lookup[self._id] = self._name

class PlotFunction():
    def __init__(self, plot_function, **kwargs):
        "Sets kwargs as attributes."
        self.plot_function = plot_function
        self.__dict__.update(kwargs)

    def __call__(self, ax_gen, data):
        return self.plot_function(ax_gen, data)