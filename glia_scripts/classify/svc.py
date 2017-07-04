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

def get_checkerboard_durations(stimulus_list):
    f = glia.compose(
        glia.f_filter(lambda x: x["stimulus"]['stimulusType']=='CHECKERBOARD'),
        partial(glia.group_by,
            key=lambda x: x["stimulus"]["lifespan"])
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

def get_grating_durations(stimulus_list):
    f = glia.compose(
        glia.f_filter(lambda x: x["stimulus"]['stimulusType']=='GRATING'),
        partial(glia.group_by,
                key=lambda x: x["stimulus"]["lifespan"])
        )
    durations = sorted(list(f(stimulus_list).keys()))
    assert len(durations)>0
    return durations

def svm_helper(training_data, training_target, validation_data, validation_target):
    # Create a classifier: a support vector classifier
    classifier = svm.SVC()
    classifier.fit(training_data, training_target)

    predicted = classifier.predict(validation_data)
    expected = validation_target

    return metrics.accuracy_score(expected, predicted)

def error_bars_for_condition(data, target, ndraws=20):
    accuracy_score = []
    n = data.shape[1]
    (ntrain, nvalid, _) = glia.tvt_by_percentage(n,60,40,0)
    indices = np.indices(n)
    for i in range(ndraws):
        indices = np.random.shuffle(indices)
        training_ind = indices[0:ntrain]
        validation_ind = indices[ntrain:]

        training_data = data[training_ind]
        training_target = target[training_ind]
        validation_data = data[validation_ind]
        validation_target = target[validation_ind]

        accuracy_score.append(svm_helper(training_data, training_target,
            validation_data, validation_target))
    return (mean,below,above)


def bin_100ms(data):
    # turn the data in a (samples, feature) matrix from 100ms time bins:
    new_steps = int(timesteps/100)
    (nconditions, nsizes, n_training, timesteps, n_x, n_y, n_units) = data.shape
    return np.sum(data.reshape(
                        (nconditions, nsizes,
                            n_training, new_steps, 100, n_x,n_y,n_units)),
                    axis=4).reshape(
                        (nconditions, nsizes, \
                            n_training, new_steps*n_x*n_y*n_units))

def bin_sum(data):
    (nconditions, nsizes, n_training, timesteps, n_x, n_y, n_units) = data.shape
    return np.sum(data["training_data"],axis=3).reshape(
        (nconditions, nsizes, n_training, n_x*n_y*n_units))

def plot_acuity(logmar, accuracy, yerror,
                n_validation, name, conditions, condition_name):
    sig5 = np.repeat(binom.ppf(0.95, n_validation, 0.5)/n_validation, len(sizes))
    sig1 = np.repeat(binom.ppf(0.99, n_validation, 0.5)/n_validation, len(sizes))

    fig, ax = plt.subplots()
    for condition in range(nconditions):
        ax.errorbar(logmar, accuracy[condition], marker='o',
            yerr=yerror, label=f'{conditions[condition]}')
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
    if condition_name is None:
        ax.set_title(f'{name} classification by binning technique')
        fig.savefig(os.path.join(plot_directory, f"{name}_acuity.png"))
    else:
        ax.set_title(f'{name} classification by {condition_name}')
        fig.savefig(os.path.join(plot_directory, f"{name}-{condition_name}_acuity.png"))

def acuity(training_data, training_target, validation_data, validation_target,
            stimulus_list, plot_directory, name,
            sizes, conditions, condition_name):
    print(f"plotting {name} classification accuracy.")

    shape = training_data.shape
    (nconditions, nsizes, n_training, timesteps, n_x, n_y, n_units) = shape
    n_validation = validation_data.shape[2]

    nclasses = 2

    accuracy_100 = np.full((nconditions, nsizes), 0, dtype=np.float)
    yerror = np.full((nconditions,2,nsizes),0, dtype=np.float)
    for condition in range(nconditions):
        for size in range(nsizes):
            data = np.vstack([training_data,validation_data])
            target = np.vstack([training_target,validation_target])
            (mean,below,above) = error_bars_for_condition(data,target)
            accuracy_100[condition, size] = svm_helper(data)
            yerror[condition, :, size] = [below,above]

    logmar = list(map(px_to_logmar,sizes))

    plot_acuity(logmar, accuracy_100, yerror, n_validation,
                name, conditions, condition_name)
#
# def plot_diff_nsamples(data, stimulus_list, plot_directory, name,
#     sizes, nsamples=10):
#     # TODO broken, maybe delete?
#     print(f"plotting {name} classification accuracy.")
#
#     shape = data["training_data"].shape
#     (nsizes, n_training, timesteps, n_x, n_y, n_units) = shape
#
#     # turn the data in a (samples, feature) matrix from 100ms time bins:
#     new_steps = int(timesteps/100)
#     training_100ms = np.sum(data["training_data"].reshape(
#                         (nsizes, n_training, new_steps, 100, n_x,n_y,n_units)),
#                     axis=4).reshape(
#                         (nsizes, n_training, new_steps*n_x*n_y*n_units))
#     training_sum = np.sum(data["training_data"],axis=3).reshape((nsizes, n_training, n_x*n_y*n_units))
#
#     n_validation = data["validation_data"].shape[2]
#     validation_100ms = np.sum(data["validation_data"].reshape(
#                         (nsizes, n_validation, new_steps, 100, n_x,n_y,n_units)),
#                     axis=4).reshape(
#                         (nsizes, n_validation, new_steps*n_x*n_y*n_units))
#     validation_sum = np.sum(data["validation_data"],axis=3).reshape((nsizes, n_validation, n_x*n_y*n_units))
#
#     # convert target to one hot vector
#     # training_target = np.eye(2)[data["training_target"]]
#     training_target = data["training_target"]
#     # validation_target = np.eye(2)[data["validation_target"]]
#     validation_target = data["validation_target"]
#
#     # nclasses = training_target.shape[2]
#     nclasses = 2
#     nfeatures = training_100ms.shape[2]
#
#     sample_end = list(map(lambda x: int(np.round(x)),
#         np.linspace(0,n_training,nsamples+1)))[1:]
#     print(f'using samples of {sample_end}, actual shape: {shape}')
#     accuracy = np.full((nsizes,nsamples), 0, dtype=np.float)
#     for size in range(nsizes):
#         for i in range(nsamples):
#             end = sample_end[i]
#             accuracy[size,i] = svm_helper(
#                 training_100ms[size,0:end], training_target[size,0:end],
#                 validation_100ms[size,0:end], validation_target[size,0:end])
#     a100 = accuracy
#
#     logmar = list(map(px_to_logmar,sizes))
#
#
#     # In[55]:
#
#     sig5 = np.repeat(binom.ppf(0.95, n_validation, 0.5)/n_validation, len(sizes))
#     sig1 = np.repeat(binom.ppf(0.99, n_validation, 0.5)/n_validation, len(sizes))
#
#
#     # In[58]:
#     fig, ax = plt.subplots()
#
#     for i in range(nsamples):
#         ax.plot(logmar, a100[:,i], marker='o', label=f'{sample_end[i]} samples')
#     ax.plot(logmar, sig1, 'k--', label='1% significance')
#     ax.set_ylabel("Accuracy")
#     ax.set_xlabel("logMAR")
#
#     x_major_ticks = np.arange(2, 3.6, 0.2)
#     y_major_ticks = np.arange(0, 101, 20)
#     x_minor_ticks = np.arange(0, 101, 5)
#     y_minor_ticks = np.arange(0, 101, 5)
#     ax.set_xticks(x_major_ticks)
#     # ax.set_xticks(minor_ticks, minor=True)
#     # ax.set_yticks(major_ticks)
#     # ax.set_yticks(minor_ticks, minor=True)
#
#     ax.grid(which='both')
#
#     ax.set_ylim(0.35,1.05)
#     ax.set_xlim(1.9,3.65)
#     ax.legend(loc=(0.5,0.1))
#     ax.set_title(f'{name} classification via Support Vector Clustering')
#     fig.savefig(os.path.join(plot_directory, f"{name}_nsample_acuity.png"))


def checkerboard_svc(data, metadata, stimulus_list, lab_notebook, plot_directory,
                nsamples):
    sizes = get_checkerboard_sizes(stimulus_list)
    if metadata["name"]=='checkerboard-contrasts':
        training_data = bin_100ms(data["training_data"])
        validation_data = bin_100ms(data["validation_data"])
        training_target = data["training_target"]
        validation_target = data["validation_target"]

        conditions = get_checkerboard_contrasts(stimulus_list)
        condition_name = "contrast"
    elif metadata["name"]=="checkerboard-durations":
        training_data = bin_100ms(data["training_data"])
        validation_data = bin_100ms(data["validation_data"])
        training_target = data["training_target"]
        validation_target = data["validation_target"]

        conditions = get_checkerboard_durations(stimulus_list)
        condition_name = "duration"
    elif metadata["name"]=="checkerboard":
        training_100ms = bin_100ms(data["training_data"])
        training_sum = bin_sum(data["training_data"])
        training_data = [training_100ms, training_sum]
        validation_100ms = bin_100ms(data["validation_data"])
        validation_sum = bin_sum(data["validation_data"])
        validation_data = [validation_100ms, validation_sum]
        training_target = data["training_target"]
        validation_target = data["validation_target"]

        conditions = ['100ms bins', 'spike count']
        condition_name = None
    else:
        raise(ValueError(f'Unknown experiment {name}'))
    if nsamples>0:
        plot_diff_nsamples(data, stimulus_list, plot_directory,
            "checkerboard", sizes, conditions, condition_name)
    else:
        acuity(training_data, training_target, validation_data, validation_target,
            stimulus_list, plot_directory, "checkerboard",
            sizes, conditions, condition_name)

def grating_svc(data, metadata, stimulus_list, lab_notebook, plot_directory,
                nsamples):
    sizes = get_grating_sizes(stimulus_list)
    durations = get_grating_durations(stimulus_list)
    # TODO function not written
    # contrasts = get_grating_contrasts(stimulus_list)
    if nsamples>0:
        plot_diff_nsamples(data, stimulus_list, plot_directory,
            "grating", sizes, durations, nsamples)
    else:
        acuity(data, stimulus_list, plot_directory, "grating",
            sizes, durations)
