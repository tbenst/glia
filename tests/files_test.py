def read_spyking_results_test(unit_spike_trains, sampling_rate):
	assert unit_spike_trains['temp_16'][0] == float(9886 / sampling_rate)
	assert len(unit_spike_trains) == 96

def read_plexon_txt_file_test(units):
	assert True