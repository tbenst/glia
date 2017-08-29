import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glia
from sklearn import datasets, svm, metrics, neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import sklearn
import os
from functools import reduce, partial
from glia import logger

from scipy.stats import binom

def px_to_logmar(px,px_per_deg=7.5):
    minutes = px/px_per_deg*60
    return np.log10(minutes)

def get_stimulus_parameters(stimulus_list, stimulus_type, parameter):
    f = glia.compose(
        glia.f_filter(lambda x: x["stimulus"]['stimulusType']==stimulus_type),
        partial(glia.group_by,
                key=lambda x: x["stimulus"][parameter])
        )
    parameters = sorted(list(f(stimulus_list).keys()))
    assert len(parameters)>0
    return parameters

def get_checkerboard_contrasts(stimulus_list):
    f = glia.compose(
        glia.f_filter(lambda x: x["stimulus"]['stimulusType']=='CHECKERBOARD'),
        partial(glia.group_by,
                key=lambda x: glia.checkerboard_contrast(x["stimulus"]))
        )
    contrasts = [float(x) for x in sorted(list(f(stimulus_list).keys()))]
    assert len(contrasts)>0
    return contrasts

def svm_helper(training_data, training_target, validation_data, validation_target):
    # Create a classifier: a support vector classifier
    classifier = svm.SVC()
    classifier.fit(training_data, training_target)

    predicted = classifier.predict(validation_data)
    expected = validation_target

    return metrics.accuracy_score(expected, predicted)


def classifier_helper(classifier, training, validation):
    training_data, training_target = training
    validation_data, validation_target = validation

    classifier.fit(training_data, training_target)
    predicted = classifier.predict(validation_data)
    expected = validation_target

    report = metrics.classification_report(expected, predicted)
    confusion = confusion_matrix(expected, predicted)
    return (report, confusion)

letter_map = {'K': 4, 'C': 1, 'V': 9, 'N': 5, 'R': 7, 'H': 3, 'O': 6, 'Z': 10, 'D': 2, 'S': 8, 'BLANK': 0}
letter_classes = list(map(lambda x: x[0],
                   sorted(list(letter_map.items()),
                          key=lambda x: x[1])))

def confusion_matrix(expected, predicted, classes=letter_classes):
    m = metrics.confusion_matrix(expected, predicted)

    return pd.DataFrame(data=m, index=classes,
                              columns=classes)

def error_bars(data, target, ndraws=20):
    n = data.shape[0]
    accuracy = np.full((ndraws,), 0)
    (ntrain, nvalid, _) = glia.tvt_by_percentage(n,60,40,0)
    indices = np.arange(n)
    for i in range(ndraws):
        np.random.shuffle(indices)
        training_ind = indices[0:ntrain]
        validation_ind = indices[ntrain:]

        training_data = data[training_ind]
        training_target = target[training_ind]
        validation_data = data[validation_ind]
        validation_target = target[validation_ind]

        accuracy[i] = svm_helper(training_data, training_target,
            validation_data, validation_target)
    std = np.std(accuracy)
    return (np.mean(accuracy),std,std)


def bin_100ms(data):
    # turn the data in a (samples, feature) matrix from 100ms time bins:
    (nconditions, nsizes, n_training, timesteps, n_x, n_y, n_units) = data.shape
    new_steps = int(timesteps/100)
    return np.sum(data.reshape(
                        (nconditions, nsizes,
                            n_training, new_steps, 100, n_x,n_y,n_units)),
                    axis=4).reshape(
                        (nconditions, nsizes, \
                            n_training, new_steps*n_x*n_y*n_units))

def bin_sum(data):
    (nconditions, nsizes, n_training, timesteps, n_x, n_y, n_units) = data.shape
    return np.sum(data,axis=3).reshape(
        (nconditions, nsizes, n_training, n_x*n_y*n_units))

def plot_acuity(logmar, accuracy, yerror,
                n_validation, name, conditions, condition_name, plot_directory):
    print(f"plotting {name} {condition_name} classification accuracy.")
    sig5 = np.repeat(binom.ppf(0.95, n_validation, 0.5)/n_validation, len(logmar))
    sig1 = np.repeat(binom.ppf(0.99, n_validation, 0.5)/n_validation, len(logmar))

    fig, ax = plt.subplots()
    nconditions = len(conditions)
    for condition in range(nconditions):
        if type(conditions[condition]) is float:
            label = f'{conditions[condition]:.2f}'
        else:
            label = f'{conditions[condition]}'

        ax.errorbar(logmar, accuracy[condition], marker='o', markersize=4, capsize=4,
            yerr=yerror[condition], label=label)
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
    print(f"training classifiers.")
    # polymorphic over ndarray or list for conditions
    nconditions = len(training_data)
    assert nconditions==len(training_target)
    assert nconditions==len(validation_data)
    assert nconditions==len(validation_target)
    nsizes = training_data[0].shape[0]
    assert nsizes==training_target[0].shape[0]
    assert nsizes==validation_data[0].shape[0]
    assert nsizes==validation_target[0].shape[0]
    n_validation = validation_data[0].shape[1]

    nclasses = 2
    accuracy_100 = np.full((nconditions, nsizes), 0, dtype=np.float)
    yerror = np.full((nconditions,2,nsizes),0, dtype=np.float)
    for condition in range(nconditions):
        for size in range(nsizes):
            data = np.concatenate(
                [training_data[condition][size],validation_data[condition][size]])
            target = np.concatenate(
                [training_target[condition][size],validation_target[condition][size]])
            (mean,below,above) = error_bars(data,target)
            accuracy_100[condition, size] = mean
            yerror[condition, :, size] = [below,above]

    logmar = list(map(px_to_logmar,sizes))

    plot_acuity(logmar, accuracy_100, yerror, n_validation,
                name, conditions, condition_name, plot_directory)


def checkerboard_svc(data, metadata, stimulus_list, lab_notebook, plot_directory,
                nsamples):
    sizes = get_stimulus_parameters(stimulus_list, "CHECKERBOARD", 'size')
    name = metadata["name"]
    if name=='checkerboard-contrast':
        training_data = bin_100ms(data["training_data"])
        validation_data = bin_100ms(data["validation_data"])
        training_target = data["training_target"]
        validation_target = data["validation_target"]

        conditions = get_checkerboard_contrasts(stimulus_list)
        condition_name = "contrast"
    elif name=="checkerboard-durations":
        training_data = bin_100ms(data["training_data"])
        validation_data = bin_100ms(data["validation_data"])
        training_target = data["training_target"]
        validation_target = data["validation_target"]

        conditions = get_stimulus_parameters(stimulus_list,
            "CHECKERBOARD", 'lifespan')
        condition_name = "durations"
    elif name=="checkerboard":
        training_100ms = bin_100ms(data["training_data"])[0]
        training_sum = bin_sum(data["training_data"])[0]
        training_data = [training_100ms, training_sum]
        validation_100ms = bin_100ms(data["validation_data"])[0]
        validation_sum = bin_sum(data["validation_data"])[0]
        validation_data = [validation_100ms, validation_sum]
        tt = data["training_target"][0]
        training_target = [tt,tt]
        vt = data["validation_target"][0]
        validation_target = [vt,vt]

        conditions = ['100ms bins', 'spike count']
        condition_name = None
    else:
        raise(ValueError(f'Unknown experiment {metadata["name"]}'))
    if nsamples>0:
        plot_diff_nsamples(data, stimulus_list, plot_directory,
            "checkerboard", sizes, conditions, condition_name)
    else:
        acuity(training_data, training_target, validation_data, validation_target,
            stimulus_list, plot_directory, "checkerboard",
            sizes, conditions, condition_name)


def grating_svc(data, metadata, stimulus_list, lab_notebook, plot_directory,
                nsamples):
    sizes = get_stimulus_parameters(stimulus_list, "GRATING", "width")
    if metadata["name"]=='grating-contrast':
        training_data = bin_100ms(data["training_data"])
        validation_data = bin_100ms(data["validation_data"])
        training_target = data["training_target"]
        validation_target = data["validation_target"]

        conditions = get_grating_contrasts(stimulus_list)
        condition_name = "contrast"
    elif metadata["name"]=="grating-durations":
        training_data = bin_100ms(data["training_data"])
        validation_data = bin_100ms(data["validation_data"])
        training_target = data["training_target"]
        validation_target = data["validation_target"]

        conditions = get_stimulus_parameters(stimulus_list, "GRATING", 'lifespan')
        condition_name = "durations"
    elif metadata["name"]=="grating-speeds":
        training_data = bin_100ms(data["training_data"])
        validation_data = bin_100ms(data["validation_data"])
        training_target = data["training_target"]
        validation_target = data["validation_target"]

        conditions = get_stimulus_parameters(stimulus_list, "GRATING", 'speed')
        condition_name = "speeds"
    elif metadata["name"]=="grating":
        training_100ms = bin_100ms(data["training_data"])[0]
        training_sum = bin_sum(data["training_data"])[0]
        training_data = [training_100ms, training_sum]
        validation_100ms = bin_100ms(data["validation_data"])[0]
        validation_sum = bin_sum(data["validation_data"])[0]
        validation_data = [validation_100ms, validation_sum]
        tt = data["training_target"][0]
        training_target = [tt,tt]
        vt = data["validation_target"][0]
        validation_target = [vt,vt]

        conditions = ['100ms bins', 'spike count']
        condition_name = None
    else:
        raise(ValueError(f'Unknown experiment {metadata["name"]}'))
    if nsamples>0:
        plot_diff_nsamples(data, stimulus_list, plot_directory,
            "grating", sizes, conditions, condition_name)
    else:
        acuity(training_data, training_target, validation_data, validation_target,
            stimulus_list, plot_directory, "grating",
            sizes, conditions, condition_name)


def letter_svc(data, metadata, stimulus_list, lab_notebook, plot_directory,
                nsamples):
    print("Classifying Letters")
    sizes = get_stimulus_parameters(stimulus_list, "LETTER", 'size')
    name = metadata["name"]
    if name=="letters":
        # n_sizes, n_training, n_steps, n_x, n_y, n_units = data["training_data"].shape
        logger.debug(data["training_data"].shape)
        # add nconditions dim
        training_100ms = bin_100ms(np.expand_dims(data["training_data"],0))
        validation_100ms = bin_100ms(np.expand_dims(data["validation_data"],0))
        logger.debug(f'training_100ms shape {training_100ms.shape}')
        logger.debug(f'sizes {sizes}')
        for i, size in enumerate(sizes):
            print(f'SVC for size {size}')
            # note: no expand dims, hardcoded 1 ncondition
            training_target = data["training_target"][i]
            validation_target = data["validation_target"][i]
            logger.debug(np.size(training_target))
            svr = svm.SVC()
            parameters = {'C': [1, 10, 100, 1000],
                          'gamma': [0.001, 0.0001]},
            clf = GridSearchCV(svr, parameters, n_jobs=12)
            report, confusion = classifier_helper(clf,
                (training_100ms[0,i], training_target),
                (validation_100ms[0,i], validation_target))
            with open(f"{plot_directory}/letter-{size}.txt", "w") as f:
                f.write(report+'\n')
                f.write(str(confusion))

    else:
        raise(ValueError(f'Unknown experiment {metadata["name"]}'))
