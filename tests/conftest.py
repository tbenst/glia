import pytest
import glia
import numpy as np
from uuid import uuid4
# from data.stimulus_list import gratings_stimulus_list

def assert_within(a,b,within=1):
	assert abs(a-b) <= within

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

@pytest.fixture(scope="module")
def spike_train(unit_spike_trains):
	return unit_spike_trains["temp_16"]


@pytest.fixture(scope="module")
def units():
	return read_plexon_txt_file("tests/data/E1_R1_DAD_45min_movingbar.txt", uuid4())

@pytest.fixture(scope="module")
def plexon_txt_filepath():
	return "tests/data/E1_R1_DAD_45min_movingbar.txt"

@pytest.fixture(scope="module")
def units():
	simulated_units = {}
	for i in range(0,100):
	    unit = glia.Unit(uuid4(),1)
	    unit.spike_train = np.arange(0,2200,1)
	    simulated_units[unit.id] = unit

	return simulated_units

@pytest.fixture(scope="module")
def unit():
	return next(iter(units().values()))

@pytest.fixture(scope="module")
def spike_train():
	return next(iter(units().values())).spike_train

@pytest.fixture(scope="module")
def stimulus_list():
    return glia.load_stimulus("tests/data/160615/E1_R1_DAD_55min_contrastgratings.stimulus")