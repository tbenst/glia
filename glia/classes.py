from uuid import uuid4

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
	def __init__(self, retina_id, channel):
		self.id = uuid4()
		self.retina_id = retina_id
		self.channel = channel
		self.spike_train = []
