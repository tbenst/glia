import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glia
from sklearn import datasets, svm, metrics, neighbors
from sklearn.neighbors import KNeighborsClassifier
import sklearn
import os
from functools import reduce, partial

from scipy.stats import binom

def px_to_logmar(px,px_per_deg=7.5):
    minutes = px/px_per_deg*60
    return np.log10(minutes)

def get_sizes(stimulus_list):
    f = glia.compose(
        glia.f_filter(lambda x: x["stimulus"]['stimulusType']=='CHECKERBOARD'),
        partial(glia.group_by,
                key=lambda x: x["stimulus"]["size"])
        )
    sizes = sorted(list(f(stimulus_list).keys()))
    assert len(sizes)>0
    return sizes

def svm_helper(training_data, training_target, validation_data, validation_target):
    # Create a classifier: a support vector classifier
    classifier = svm.SVC()
    classifier.fit(training_data, training_target)

    predicted = classifier.predict(validation_data)
    expected = validation_target

    return metrics.accuracy_score(expected, predicted)

def main(data, stimulus_list, plot_directory):
    print("plotting checkerboard classification accuracy.")

    sizes = get_sizes(stimulus_list)

    shape = data["training_data"].shape
    (nsizes, n_training, timesteps, n_x, n_y, n_units) = shape

    # turn the data in a (samples, feature) matrix from 100ms time bins:
    new_steps = int(timesteps/100)
    training_100ms = np.sum(data["training_data"].reshape(
                        (nsizes, n_training, new_steps, 100, n_x,n_y,n_units)),
                    axis=3).reshape(
                        (nsizes, n_training, new_steps*n_x*n_y*n_units))
    training_sum = np.sum(data["training_data"],axis=2).reshape((nsizes, n_training, n_x*n_y*n_units))

    n_validation = data["validation_data"].shape[1]
    validation_100ms = np.sum(data["validation_data"].reshape(
                        (nsizes, n_validation, new_steps, 100, n_x,n_y,n_units)),
                    axis=3).reshape(
                        (nsizes, n_validation, new_steps*n_x*n_y*n_units))
    validation_sum = np.sum(data["validation_data"],axis=2).reshape((nsizes, n_validation, n_x*n_y*n_units))

    # convert target to one hot vector
    # training_target = np.eye(2)[data["training_target"]]
    training_target = data["training_target"]
    # validation_target = np.eye(2)[data["validation_target"]]
    validation_target = data["validation_target"]

    # nclasses = training_target.shape[2]
    nclasses = 2
    nfeatures = training_100ms.shape[2]

    accuracy = np.full((nsizes), 0, dtype=np.float)
    for size in range(nsizes):
        accuracy[size] = svm_helper(training_100ms[size], training_target[size],
                   validation_100ms[size], validation_target[size])
    a100 = accuracy

    accuracy = np.full((nsizes), 0, dtype=np.float)
    for size in range(nsizes):
        accuracy[size] = svm_helper(training_sum[size], training_target[size],
                   validation_sum[size], validation_target[size])
    a = accuracy

    logmar = list(map(px_to_logmar,sizes))


    # In[55]:

    sig5 = np.repeat(binom.ppf(0.95, n_validation, 0.5)/n_validation, len(sizes))
    sig1 = np.repeat(binom.ppf(0.99, n_validation, 0.5)/n_validation, len(sizes))


    # In[58]:
    fig, ax = plt.subplots()
    ax.plot(logmar, a100, 'b', marker='o', label='100ms bins')
    ax.plot(logmar, a, 'g', marker='o',label='Avg firing rate')
    ax.plot(logmar, sig1, 'k--', label='1% significance')
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("logMAR")

    x_major_ticks = np.arange(2, 3.6, 0.2)                                              
    y_major_ticks = np.arange(0, 101, 20)                                              
    x_minor_ticks = np.arange(0, 101, 5)                                               
    y_minor_ticks = np.arange(0, 101, 5)                                               
    ax.set_xticks(x_major_ticks)                                                       
    # ax.set_xticks(minor_ticks, minor=True)                                           
    # ax.set_yticks(major_ticks)                                                       
    # ax.set_yticks(minor_ticks, minor=True)

    ax.grid(which='both')  

    ax.set_ylim(0.35,1.05)
    ax.set_xlim(1.9,3.65)
    ax.legend(loc=(0.5,0.1))
    ax.set_title('Checkerboard classification via Support Vector Clustering')
    fig.savefig(os.path.join(plot_directory,"checkerboard_acuity.png"))