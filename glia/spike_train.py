import numpy as np
from scipy import signal

def IFR(spike_train, end_time, bandwidth, bin_width=0.001, sigma=6):
    transformed_sigma = bandwidth/bin_width
    bins = np.arange(0,end_time+bin_width, bin_width)
    spike_train_to_indices = np.vectorize(lambda x: np.digitize(x, bins))
    if len(spike_train)==0:
        ifr = np.zeros(bins.size)
    else:
        indices = spike_train_to_indices(spike_train)
        spike_bins = np.zeros(bins.size)
        spike_bins[indices] = 1
        # multiplied by two to adjust distribution on both sides of window
        window = signal.gaussian(2*sigma*transformed_sigma, std=transformed_sigma)
        ifr = signal.convolve(spike_bins, window)
    return ifr
