import pytest
import glia
from data.stimulus_list import gratings_stimulus_list

def assert_within(a,b,within=1):
	assert abs(a-b) < within

@pytest.fixture
def sampling_rate():
	return 25000


@pytest.fixture
def unit_spike_trains():
	return glia.read_spyking_results("tests/data/gratings.result.hdf5", sampling_rate())

@pytest.fixture
def stimulus_start_times():
	return glia.get_stimulus_start_times("tests/data/gratings.analog")


@pytest.fixture
def stimulus_list():
	return gratings_stimulus_list


def read_spyking_results_test(unit_spike_trains):
	assert unit_spike_trains['temp_16'][0] == float(9886 / sampling_rate())
	assert len(unit_spike_trains) == 96

def get_stimulus_start_times_test(stimulus_start_times,stimulus_list):
	assert len(stimulus_start_times) == 143
	
	previous_time=stimulus_start_times[0]
	for i in range(1,len(stimulus_start_times)-1):
		duration = stimulus_start_times[i] - stimulus_start_times[i-1]
		assert_within(duration,gratings_stimulus_list[i-1][0],0.1)
	assert len(stimulus_start_times) == len(gratings_stimulus_list)
	assert_within(stimulus_start_times[0]+2,stimulus_start_times[1],0.1)
	assert_within(stimulus_start_times[1]+1,stimulus_start_times[2],0.1)
	assert_within(stimulus_start_times[2]+3,stimulus_start_times[3],0.1)
	assert_within(stimulus_start_times[3]+50,stimulus_start_times[4],0.1)

