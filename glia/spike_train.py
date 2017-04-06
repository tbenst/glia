import glia
import numpy as np
from functools import reduce
import os
from uuid import uuid4, UUID
import re
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from scipy.sparse import csc_matrix
from scipy.ndimage import filters
from scipy import signal
from matplotlib.ticker import FuncFormatter, MultipleLocator
from datetime import datetime
from datetime import timedelta
from scipy import stats
from functools import update_wrapper, partial
import reprlib
from warnings import warn
import sklearn.metrics as metrics
from sklearn import manifold
from sklearn.metrics import euclidean_distances
import itertools
import elephant
from neo.core import SpikeTrain
import quantities

victor_purpura = lambda v: elephant.spike_train_dissimilarity.victor_purpura_dist(
            list(map(experiment_to_SpikeTrain,v)))


def experiment_to_SpikeTrain(experiment):
    return SpikeTrain(experiment["spikes"]*quantities.s,experiment["stimulus"]["lifespan"]/120)


def vp(a,b,q=1):
    """Victor-Purpura distance between two SpikeTrains."""
    return elephant.spike_train_dissimilarity.victor_purpura_dist([a,b])[0][1]


def IFR(spike_train, end_time, bandwidth=0.15, bin_width=0.001, sigma=6):
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
