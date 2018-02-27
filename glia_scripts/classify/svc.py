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
from glia import logger, config
from scipy import stats

from scipy.stats import binom

def svm_grid(training_data, training_target, test_data, test_target, n_jobs=config.processes):
    svr = svm.SVC()
    parameters = {'C': [1, 10, 100, 1000],
                  'gamma': [0.001, 0.0001]},
    clf = GridSearchCV(svr, parameters, n_jobs=12)
    clf.fit(training_data, training_target)

    predicted = clf.predict(test_data)
    expected = test_target

    return metrics.accuracy_score(expected, predicted)


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
    ax.plot(logmar, sig1, 'k--')
    # ax.plot(logmar, sig1, 'k--', label='p<0.01')
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("logMAR")

    x_major_ticks = np.arange(1.6, 3.2, 0.2)
    ax.set_xticks(x_major_ticks)
    ax.set_xlim(1.5,3.25)
    # ax.set_xticks(minor_ticks, minor=True)
    # ax.set_yticks(major_ticks)
    # ax.set_yticks(minor_ticks, minor=True)

    ax.grid(which='both')

    ax.set_ylim(0.35,1.05)
    ax.legend(loc=(0.7,0.1))
    if condition_name is None:
        ax.set_title(f'{name} classification by binning technique')
        fig.savefig(os.path.join(plot_directory, f"{name}_acuity.png"))
    else:
        ax.set_title(f'{name} classification by {condition_name}')
        fig.savefig(os.path.join(plot_directory, f"{name}-{condition_name}_acuity.png"))

def plot_sensitivity(logmar, accuracy, yerror,
                n_validation, name, conditions, condition_name, plot_directory):
    raise "Not implemented"
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
    ax.plot(logmar, sig1, 'k--')
    # ax.plot(logmar, sig1, 'k--', label='p<0.01')
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("logMAR")

    x_major_ticks = np.arange(1.6, 3.2, 0.2)
    ax.set_xticks(x_major_ticks)
    ax.set_xlim(1.5,3.25)
    # ax.set_xticks(minor_ticks, minor=True)
    # ax.set_yticks(major_ticks)
    # ax.set_yticks(minor_ticks, minor=True)

    ax.grid(which='both')

    ax.set_ylim(0.35,1.05)
    ax.legend(loc=(0.7,0.1))
    if condition_name is None:
        ax.set_title(f'{name} classification by binning technique')
        fig.savefig(os.path.join(plot_directory, f"{name}_acuity.png"))
    else:
        ax.set_title(f'{name} classification by {condition_name}')
        fig.savefig(os.path.join(plot_directory, f"{name}-{condition_name}_acuity.png"))

def acuity(training_data, training_target, validation_data, validation_target,
            stimulus_list, plot_directory, name,
            sizes, conditions, condition_name, n_draws=30):
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
    accuracy = np.full((nconditions, nsizes), 0, dtype=np.float)
    yerror = np.full((nconditions,nsizes),0, dtype=np.float)
    ntrain = training_data[0].shape[1]
    for condition in range(nconditions):
        for size in range(nsizes):
            data = np.concatenate(
                [training_data[condition][size],validation_data[condition][size]])
            target = np.concatenate(
                [training_target[condition][size],validation_target[condition][size]])
            acc = glia.mccv(svm_grid, data,target,n_draws,ntrain)
            accuracy[condition, size] = np.mean(acc)
            yerror[condition, size] = stats.sem(acc)

    logmar = list(map(glia.px_to_logmar,sizes))

    plot_acuity(logmar, accuracy, yerror, n_validation,
                name, conditions, condition_name, plot_directory)
    # plot_sensitivity(logmar, accuracy, yerror, n_validation,
    #             name, conditions, condition_name, plot_directory)


def checkerboard_svc(data, metadata, stimulus_list, lab_notebook, plot_directory,
                nsamples, n_draws=30):
    sizes = glia.get_stimulus_parameters(stimulus_list, "CHECKERBOARD", 'size')
    name = metadata["name"]
    if name=='checkerboard-contrast':
        training_data = glia.bin_100ms(data["training_data"])
        validation_data = glia.bin_100ms(data["validation_data"])
        training_target = data["training_target"]
        validation_target = data["validation_target"]

        conditions = glia.get_checkerboard_contrasts(stimulus_list)
        condition_name = "contrast"
    elif name=="checkerboard-durations":
        training_data = glia.bin_100ms(data["training_data"])
        validation_data = glia.bin_100ms(data["validation_data"])
        training_target = data["training_target"]
        validation_target = data["validation_target"]

        conditions = glia.get_stimulus_parameters(stimulus_list,
            "CHECKERBOARD", 'lifespan')
        condition_name = "durations"
    elif name=="checkerboard":
        training_100ms = glia.bin_100ms(data["training_data"])[0]
        training_sum = glia.bin_sum(data["training_data"])[0]
        training_data = [training_100ms, training_sum]
        validation_100ms = glia.bin_100ms(data["validation_data"])[0]
        validation_sum = glia.bin_sum(data["validation_data"])[0]
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
            sizes, conditions, condition_name, n_draws)


def grating_svc(data, metadata, stimulus_list, lab_notebook, plot_directory,
                nsamples,n_draws=30):
    sizes = glia.get_stimulus_parameters(stimulus_list, "GRATING", "width")
    if metadata["name"]=='grating-contrast':
        training_data = glia.bin_100ms(data["training_data"])
        validation_data = glia.bin_100ms(data["validation_data"])
        training_target = data["training_target"]
        validation_target = data["validation_target"]

        conditions = get_grating_contrasts(stimulus_list)
        condition_name = "contrast"
    elif metadata["name"]=="grating-durations":
        training_data = glia.bin_100ms(data["training_data"])
        validation_data = glia.bin_100ms(data["validation_data"])
        training_target = data["training_target"]
        validation_target = data["validation_target"]

        conditions = glia.get_stimulus_parameters(stimulus_list, "GRATING", 'lifespan')
        condition_name = "durations"
    elif metadata["name"]=="grating-speeds":
        training_data = glia.bin_100ms(data["training_data"])
        validation_data = glia.bin_100ms(data["validation_data"])
        training_target = data["training_target"]
        validation_target = data["validation_target"]

        conditions = glia.get_stimulus_parameters(stimulus_list, "GRATING", 'speed')
        condition_name = "speeds"
    elif metadata["name"]=="grating":
        training_100ms = glia.bin_100ms(data["training_data"])[0]
        training_sum = glia.bin_sum(data["training_data"])[0]
        training_data = [training_100ms, training_sum]
        validation_100ms = glia.bin_100ms(data["validation_data"])[0]
        validation_sum = glia.bin_sum(data["validation_data"])[0]
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
            sizes, conditions, condition_name, n_draws)


def letter_svc(data, metadata, stimulus_list, lab_notebook, plot_directory,
                nsamples):
    print("Classifying Letters - warning not using latest acuity function")
    # TODO
    sizes = glia.get_stimulus_parameters(stimulus_list, "LETTER", 'size')
    name = metadata["name"]
    if name=="letters":
        # n_sizes, n_training, n_steps, n_x, n_y, n_units = data["training_data"].shape
        logger.debug(data["training_data"].shape)
        # add nconditions dim
        training_100ms = glia.bin_100ms(np.expand_dims(data["training_data"],0))
        validation_100ms = glia.bin_100ms(np.expand_dims(data["validation_data"],0))
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
            report, confusion = glia.classifier_helper(clf,
                (training_100ms[0,i], training_target),
                (validation_100ms[0,i], validation_target))
            with open(f"{plot_directory}/letter-{size}.txt", "w") as f:
                f.write(report+'\n')
                f.write(str(confusion))

    else:
        raise(ValueError(f'Unknown experiment {metadata["name"]}'))

def tiled_letter_svc(data, metadata, stimulus_list, lab_notebook, plot_directory,
                nsamples):
    print("Classifying Letters - warning not using latest acuity function")
    # TODO
    sizes = glia.get_stimulus_parameters(stimulus_list, "TILED_LETTER", 'size')
    name = metadata["name"]
    # n_sizes, n_training, n_steps, n_x, n_y, n_units = data["training_data"].shape
    logger.debug(data["training_data"].shape)
    # add nconditions dim
    training_100ms = glia.bin_100ms(np.expand_dims(data["training_data"],0))
    validation_100ms = glia.bin_100ms(np.expand_dims(data["validation_data"],0))
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
        report, confusion = glia.classifier_helper(clf,
            (training_100ms[0,i], training_target),
            (validation_100ms[0,i], validation_target))
        with open(f"{plot_directory}/letter-{size}.txt", "w") as f:
            f.write(report+'\n')
            f.write(str(confusion))

def image_svc(data, metadata, stimulus_list, lab_notebook, plot_directory,
                nsamples):
    print("Classifying Letters - warning not using latest acuity function")
    # TODO
    sizes = glia.get_image_parameters(stimulus_list)
    name = metadata["name"]
    # n_sizes, n_training, n_steps, n_x, n_y, n_units = data["training_data"].shape
    logger.debug(data["training_data"].shape)
    # add nconditions dim
    training_100ms = glia.bin_100ms(np.expand_dims(data["training_data"],0))
    validation_100ms = glia.bin_100ms(np.expand_dims(data["validation_data"],0))
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
        report, confusion = glia.classifier_helper(clf,
            (training_100ms[0,i], training_target),
            (validation_100ms[0,i], validation_target))
        with open(f"{plot_directory}/letter-{size}.txt", "w") as f:
            f.write(report+'\n')
            f.write(str(confusion))
