from uuid import uuid4
from hashlib import md5
import base64
import humanhash

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

class Unit:
    def __init__(self, retina_id, channel, unit_num):
    	# id will be URL safe MD5 hash of spike_train
        self.id = None
        self.name = None
        self.retina_id = retina_id
        self.channel = channel
        self.unit_num
        self.spike_train = None

    @property
    def spike_train(self):
        return self._spike_train

    @spike_train.setter
    def spike_train(self, value):
    	unit_id = base64.urlsafe_b64encode(md5(value.tostring()).digest())
		self.id = unit_id
		self.name = humanhash.humanizer(unit_id)
        self._spike_train = value

