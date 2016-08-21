import pytest
import glia

@pytest.fixture
def unit_spike_trains():
	return glia.read_spyking_results("tests/data/example.result.hdf5", 25000)


def read_spyking_results_test(unit_spike_trains):
	assert True
	# assert unit_spike_trains[0]