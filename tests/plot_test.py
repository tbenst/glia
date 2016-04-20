import pytest
import glia

def test_spike_histogram(channels):
    # final bin should have two spikes
    hist = glia.spike_histogram(channels, bin_width=0.1,
                    time=(None, None), plot=False)
    assert hist[0][199] == 2



@pytest.fixture(scope="module")
def channels():
    return glia.read_mcs_dat('tests/data/sample_dat/')
