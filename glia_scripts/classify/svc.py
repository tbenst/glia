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

def plot_acuity(logmar, accuracy, yerror,
                n_test, name, conditions, condition_name, plot_directory):
    print(f"plotting {name} {condition_name} classification accuracy.")
    sig5 = np.repeat(binom.ppf(0.95, n_test, 0.5)/n_test, len(logmar))
    sig1 = np.repeat(binom.ppf(0.99, n_test, 0.5)/n_test, len(logmar))

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

def acuity(training_data, training_target, test_data, test_target,
            stimulus_list, plot_directory, name,
            sizes, conditions, condition_name):
    print(f"training classifiers.")
    # polymorphic over ndarray or list for conditions
    nconditions = len(training_data)
    assert nconditions==len(training_target)
    assert nconditions==len(test_data)
    assert nconditions==len(test_target)
    nsizes = training_data[0].shape[0]
    assert nsizes==training_target[0].shape[0]
    assert nsizes==test_data[0].shape[0]
    assert nsizes==test_target[0].shape[0]
    n_test = test_data[0].shape[1]

    nclasses = 2
    accuracy_100 = np.full((nconditions, nsizes), 0, dtype=np.float)
    yerror = np.full((nconditions,2,nsizes),0, dtype=np.float)
    for condition in range(nconditions):
        for size in range(nsizes):
            data = np.concatenate(
                [training_data[condition][size],test_data[condition][size]])
            target = np.concatenate(
                [training_target[condition][size],test_target[condition][size]])
            (mean,below,above) = glia.error_bars(data,target)
            accuracy_100[condition, size] = mean
            yerror[condition, :, size] = [below,above]

    logmar = list(map(glia.px_to_logmar,sizes))

    plot_acuity(logmar, accuracy_100, yerror, n_test,
                name, conditions, condition_name, plot_directory)


def checkerboard_svc(data, metadata, stimulus_list, lab_notebook, plot_directory,
                nsamples):
    sizes = glia.get_stimulus_parameters(stimulus_list, "CHECKERBOARD", 'size')
    name = metadata["name"]
    if name=='checkerboard-contrast':
        training_data = glia.bin_100ms(data["training_data"])
        test_data = glia.bin_100ms(data["test_data"])
        training_target = data["training_target"]
        test_target = data["test_target"]

        conditions = glia.get_checkerboard_contrasts(stimulus_list)
        condition_name = "contrast"
    elif name=="checkerboard-durations":
        training_data = glia.bin_100ms(data["training_data"])
        test_data = glia.bin_100ms(data["test_data"])
        training_target = data["training_target"]
        test_target = data["test_target"]

        conditions = glia.get_stimulus_parameters(stimulus_list,
            "CHECKERBOARD", 'lifespan')
        condition_name = "durations"
    elif name=="checkerboard":
        training_100ms = glia.glia.bin_100ms(data["training_data"])[0]
        training_sum = glia.bin_sum(data["training_data"])[0]
        training_data = [training_100ms, training_sum]
        test_100ms = glia.bin_100ms(data["test_data"])[0]
        test_sum = glia.bin_sum(data["test_data"])[0]
        test_data = [test_100ms, test_sum]
        tt = data["training_target"][0]
        training_target = [tt,tt]
        vt = data["test_target"][0]
        test_target = [vt,vt]

        conditions = ['100ms bins', 'spike count']
        condition_name = None
    else:
        raise(ValueError(f'Unknown experiment {metadata["name"]}'))
    if nsamples>0:
        plot_diff_nsamples(data, stimulus_list, plot_directory,
            "checkerboard", sizes, conditions, condition_name)
    else:
        acuity(training_data, training_target, test_data, test_target,
            stimulus_list, plot_directory, "checkerboard",
            sizes, conditions, condition_name)


def grating_svc(data, metadata, stimulus_list, lab_notebook, plot_directory,
                nsamples):
    sizes = glia.get_stimulus_parameters(stimulus_list, "GRATING", "width")
    if metadata["name"]=='grating-contrast':
        training_data = glia.bin_100ms(data["training_data"])
        test_data = glia.bin_100ms(data["test_data"])
        training_target = data["training_target"]
        test_target = data["test_target"]

        conditions = get_grating_contrasts(stimulus_list)
        condition_name = "contrast"
    elif metadata["name"]=="grating-durations":
        training_data = glia.bin_100ms(data["training_data"])
        test_data = glia.bin_100ms(data["test_data"])
        training_target = data["training_target"]
        test_target = data["test_target"]

        conditions = glia.get_stimulus_parameters(stimulus_list, "GRATING", 'lifespan')
        condition_name = "durations"
    elif metadata["name"]=="grating-speeds":
        training_data = glia.bin_100ms(data["training_data"])
        test_data = glia.bin_100ms(data["test_data"])
        training_target = data["training_target"]
        test_target = data["test_target"]

        conditions = glia.get_stimulus_parameters(stimulus_list, "GRATING", 'speed')
        condition_name = "speeds"
    elif metadata["name"]=="grating":
        training_100ms = glia.glia.bin_100ms(data["training_data"])[0]
        training_sum = glia.bin_sum(data["training_data"])[0]
        training_data = [training_100ms, training_sum]
        test_100ms = glia.bin_100ms(data["test_data"])[0]
        test_sum = glia.bin_sum(data["test_data"])[0]
        test_data = [test_100ms, test_sum]
        tt = data["training_target"][0]
        training_target = [tt,tt]
        vt = data["test_target"][0]
        test_target = [vt,vt]

        conditions = ['100ms bins', 'spike count']
        condition_name = None
    else:
        raise(ValueError(f'Unknown experiment {metadata["name"]}'))
    if nsamples>0:
        plot_diff_nsamples(data, stimulus_list, plot_directory,
            "grating", sizes, conditions, condition_name)
    else:
        acuity(training_data, training_target, test_data, test_target,
            stimulus_list, plot_directory, "grating",
            sizes, conditions, condition_name)


def letter_svc(data, metadata, stimulus_list, lab_notebook, plot_directory,
                nsamples):
    print("Classifying Letters")
    sizes = glia.get_stimulus_parameters(stimulus_list, "LETTER", 'size')
    name = metadata["name"]
    if name=="letters":
        # n_sizes, n_training, n_steps, n_x, n_y, n_units = data["training_data"].shape
        logger.debug(data["training_data"].shape)
        # add nconditions dim
        training_100ms = glia.bin_100ms(np.expand_dims(data["training_data"],0))
        test_100ms = glia.bin_100ms(np.expand_dims(data["test_data"],0))
        logger.debug(f'training_100ms shape {training_100ms.shape}')
        logger.debug(f'sizes {sizes}')
        for i, size in enumerate(sizes):
            print(f'SVC for size {size}')
            # note: no expand dims, hardcoded 1 ncondition
            training_target = data["training_target"][i]
            test_target = data["test_target"][i]
            logger.debug(np.size(training_target))
            svr = svm.SVC()
            parameters = {'C': [1, 10, 100, 1000],
                          'gamma': [0.001, 0.0001]},
            clf = GridSearchCV(svr, parameters, n_jobs=12)
            report, confusion = glia.classifier_helper(clf,
                (training_100ms[0,i], training_target),
                (test_100ms[0,i], test_target))
            with open(f"{plot_directory}/letter-{size}.txt", "w") as f:
                f.write(report+'\n')
                f.write(str(confusion))

    else:
        raise(ValueError(f'Unknown experiment {metadata["name"]}'))

def tiled_letter_svc(data, metadata, stimulus_list, lab_notebook, plot_directory,
                nsamples):
    print("Classifying Letters")
    sizes = glia.get_stimulus_parameters(stimulus_list, "TILED_LETTER", 'size')
    name = metadata["name"]
    # n_sizes, n_training, n_steps, n_x, n_y, n_units = data["training_data"].shape
    logger.debug(data["training_data"].shape)
    # add nconditions dim
    training_100ms = glia.glia.bin_100ms(np.expand_dims(data["training_data"],0))
    validation_100ms = glia.glia.bin_100ms(np.expand_dims(data["validation_data"],0))
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
    print("Classifying Letters")
    sizes = glia.get_image_parameters(stimulus_list)
    name = metadata["name"]
    # n_sizes, n_training, n_steps, n_x, n_y, n_units = data["training_data"].shape
    logger.debug(data["training_data"].shape)
    # add nconditions dim
    training_100ms = glia.bin_100ms(np.expand_dims(data["training_data"],0))
    test_100ms = glia.bin_100ms(np.expand_dims(data["test_data"],0))
    logger.debug(f'training_100ms shape {training_100ms.shape}')
    logger.debug(f'sizes {sizes}')
    for i, size in enumerate(sizes):
        print(f'SVC for size {size}')
        # note: no expand dims, hardcoded 1 ncondition
        training_target = data["training_target"][i]
        test_target = data["test_target"][i]
        logger.debug(np.size(training_target))
        svr = svm.SVC()
        parameters = {'C': [1, 10, 100, 1000],
                      'gamma': [0.001, 0.0001]},
        clf = GridSearchCV(svr, parameters, n_jobs=12)
        report, confusion = glia.classifier_helper(clf,
            (training_100ms[0,i], training_target),
            (test_100ms[0,i], test_target))
        with open(f"{plot_directory}/letter-{size}.txt", "w") as f:
            f.write(report+'\n')
            f.write(str(confusion))
