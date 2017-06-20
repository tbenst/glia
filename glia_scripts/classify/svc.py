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

def get_checkerboard_sizes(stimulus_list):
    f = glia.compose(
        glia.f_filter(lambda x: x["stimulus"]['stimulusType']=='CHECKERBOARD'),
        partial(glia.group_by,
                key=lambda x: x["stimulus"]["size"])
        )
    sizes = sorted(list(f(stimulus_list).keys()))
    assert len(sizes)>0
    return sizes

def get_checkerboard_contrasts(stimulus_list):
    f = glia.compose(
        glia.f_filter(lambda x: x["stimulus"]['stimulusType']=='CHECKERBOARD'),
        partial(glia.group_by,
                key=lambda x: glia.checkerboard_contrast(x["stimulus"]))
        )
    contrasts = sorted(list(f(stimulus_list).keys()))
    assert len(contrasts)>0
    return contrasts

def get_grating_sizes(stimulus_list):
    f = glia.compose(
        glia.f_filter(lambda x: x["stimulus"]['stimulusType']=='GRATING'),
        partial(glia.group_by,
                key=lambda x: x["stimulus"]["width"])
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

def main(data, stimulus_list, plot_directory, name,
    sizes, contrasts):
    print(f"plotting {name} classification accuracy.")

    shape = data["training_data"].shape
    (ncontrasts, nsizes, n_training, timesteps, n_x, n_y, n_units) = shape

    # turn the data in a (samples, feature) matrix from 100ms time bins:
    new_steps = int(timesteps/100)
    training_100ms = np.sum(data["training_data"].reshape(
                        (ncontrasts, nsizes,
                            n_training, new_steps, 100, n_x,n_y,n_units)),
                    axis=4).reshape(
                        (ncontrasts, nsizes, \
                            n_training, new_steps*n_x*n_y*n_units))
    training_sum = np.sum(data["training_data"],axis=3).reshape(
        (ncontrasts, nsizes, n_training, n_x*n_y*n_units))

    n_validation = data["validation_data"].shape[2]
    validation_100ms = np.sum(data["validation_data"].reshape(
                        (ncontrasts, nsizes,
                            n_validation, new_steps, 100, n_x,n_y,n_units)),
                    axis=4).reshape(
                        (ncontrasts, nsizes,
                            n_validation, new_steps*n_x*n_y*n_units))
    validation_sum = np.sum(data["validation_data"],axis=3).reshape(
        (ncontrasts, nsizes,
            n_validation, n_x*n_y*n_units))

    # convert target to one hot vector
    # training_target = np.eye(2)[data["training_target"]]
    training_target = data["training_target"]
    # validation_target = np.eye(2)[data["validation_target"]]
    validation_target = data["validation_target"]

    # nclasses = training_target.shape[2]
    nclasses = 2
    nfeatures = training_100ms.shape[2]

    accuracy_100 = np.full((ncontrasts, nsizes), 0, dtype=np.float)
    for contrast in range(ncontrasts):
        for size in range(nsizes):
            accuracy_100[contrast, size] = svm_helper(
                training_100ms[contrast, size], training_target[contrast, size],
                validation_100ms[contrast, size], validation_target[contrast, size])

    logmar = list(map(px_to_logmar,sizes))


    # In[55]:

    sig5 = np.repeat(binom.ppf(0.95, n_validation, 0.5)/n_validation, len(sizes))
    sig1 = np.repeat(binom.ppf(0.99, n_validation, 0.5)/n_validation, len(sizes))


    # In[58]:
    fig, ax = plt.subplots()
    for contrast in range(ncontrasts):
        ax.plot(logmar, accuracy_100[contrast], marker='o',
            label=f'{contrasts[contrast]}')
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
    ax.set_title(f'{name} classification by contrast')
    fig.savefig(os.path.join(plot_directory, f"{name}_acuity.png"))


def plot_diff_nsamples(data, stimulus_list, plot_directory, name, 
    sizes, nsamples=10):
    # TODO broken, maybe delete?
    print(f"plotting {name} classification accuracy.")

    shape = data["training_data"].shape
    (nsizes, n_training, timesteps, n_x, n_y, n_units) = shape

    # turn the data in a (samples, feature) matrix from 100ms time bins:
    new_steps = int(timesteps/100)
    training_100ms = np.sum(data["training_data"].reshape(
                        (nsizes, n_training, new_steps, 100, n_x,n_y,n_units)),
                    axis=4).reshape(
                        (nsizes, n_training, new_steps*n_x*n_y*n_units))
    training_sum = np.sum(data["training_data"],axis=3).reshape((nsizes, n_training, n_x*n_y*n_units))

    n_validation = data["validation_data"].shape[2]
    validation_100ms = np.sum(data["validation_data"].reshape(
                        (nsizes, n_validation, new_steps, 100, n_x,n_y,n_units)),
                    axis=4).reshape(
                        (nsizes, n_validation, new_steps*n_x*n_y*n_units))
    validation_sum = np.sum(data["validation_data"],axis=3).reshape((nsizes, n_validation, n_x*n_y*n_units))

    # convert target to one hot vector
    # training_target = np.eye(2)[data["training_target"]]
    training_target = data["training_target"]
    # validation_target = np.eye(2)[data["validation_target"]]
    validation_target = data["validation_target"]

    # nclasses = training_target.shape[2]
    nclasses = 2
    nfeatures = training_100ms.shape[2]

    sample_end = list(map(lambda x: int(np.round(x)),
        np.linspace(0,n_training,nsamples+1)))[1:]
    print(f'using samples of {sample_end}, actual shape: {shape}')
    accuracy = np.full((nsizes,nsamples), 0, dtype=np.float)
    for size in range(nsizes):
        for i in range(nsamples):
            end = sample_end[i]
            accuracy[size,i] = svm_helper(
                training_100ms[size,0:end], training_target[size,0:end],
                validation_100ms[size,0:end], validation_target[size,0:end])
    a100 = accuracy

    logmar = list(map(px_to_logmar,sizes))


    # In[55]:

    sig5 = np.repeat(binom.ppf(0.95, n_validation, 0.5)/n_validation, len(sizes))
    sig1 = np.repeat(binom.ppf(0.99, n_validation, 0.5)/n_validation, len(sizes))


    # In[58]:
    fig, ax = plt.subplots()

    for i in range(nsamples):
        ax.plot(logmar, a100[:,i], marker='o', label=f'{sample_end[i]} samples')
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
    ax.set_title(f'{name} classification via Support Vector Clustering')
    fig.savefig(os.path.join(plot_directory, f"{name}_nsample_acuity.png"))


def checkerboard_svc(data, stimulus_list, plot_directory, nsamples):
    sizes = get_checkerboard_sizes(stimulus_list)
    contrasts = get_checkerboard_contrasts(stimulus_list)
    if nsamples>0:
        plot_diff_nsamples(data, stimulus_list, plot_directory, 
            "checkerboard", sizes, contrasts, nsamples)
    else:
        main(data, stimulus_list, plot_directory, "checkerboard",
            sizes, contrasts)

def grating_svc(data, stimulus_list, plot_directory, nsamples):
    sizes = get_grating_sizes(stimulus_list)
    # TODO function not written
    # contrasts = get_grating_contrasts(stimulus_list)
    if nsamples>0:
        plot_diff_nsamples(data, stimulus_list, plot_directory,
            "grating", sizes, contrasts, nsamples)
    else:
        main(data, stimulus_list, plot_directory, "grating".
            sizes, contrasts)
