import pytest
import glia
from data.stimulus_list import gratings_stimulus_list

def assert_within(a,b,within=1):
	assert abs(a-b) < within

@pytest.fixture(scope="module")
def sampling_rate():
	return 25000


@pytest.fixture(scope="module")
def unit_spike_trains():
	return glia.read_spyking_results("tests/data/gratings.result.hdf5", sampling_rate())

@pytest.fixture(scope="module")
def stimulus_start_times():
	return glia.get_stimulus_start_times("tests/data/gratings.analog")


@pytest.fixture(scope="module")
def stimulus_list():
	return gratings_stimulus_list
