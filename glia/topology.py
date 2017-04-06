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

def distance_by_row(similarity):
    return list(map(lambda x: np.std(x), similarity))

def truncate_experiment(max_time, experiment):
    new = experiment.copy()
    train = experiment["spikes"]
    new["spikes"] = train[np.where(train<max_time)]
    return new

# i = glia.compose(
#     glia.f_create_experiments(stimulus_list),
#     glia.f_has_stimulus_type(["SOLID"]),
#     partial(filter,lambda x: x["stimulus"]["lifespan"]==60),
#     lambda x: list(x),
# )

icentroid = glia.compose(
    distance_by_row,
    np.min
)