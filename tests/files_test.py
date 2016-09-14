import glia
import re
from uuid import uuid4

# def read_spyking_results_test(unit_spike_trains, sampling_rate):
# 	assert unit_spike_trains['temp_16'][0] == float(9886 / sampling_rate)
# 	assert len(unit_spike_trains) == 96

# def read_plexon_txt_file_test(plexon_txt_filepath):
# 	units = glia.read_plexon_txt_file(plexon_txt_filepath, uuid4())

# 	all_units = set()
# 	pattern = re.compile("^(\d+),(\d+),.*$")
# 	with open(plexon_txt_filepath) as file:
# 	    for line in file:
# 	        m = pattern.match(line)
# 	        all_units.add(m.groups())
# 	assert len(all_units) == len(units.keys())
