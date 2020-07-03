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

def _svm_grid(training_data, training_target, test_data, test_target, n_jobs=config.processes):
    svr = svm.SVC()
    parameters = {'C': [1, 10, 100, 1000],
                  'gamma': [0.001, 0.0001]},
    clf = GridSearchCV(svr, parameters, n_jobs=12, cv=5, iid=False)
    clf.fit(training_data, training_target)

    predicted = clf.predict(test_data)
    
    return clf, predicted


def svm_grid(training_data, training_target, test_data, test_target, n_jobs=config.processes):
    clf, predicted = _svm_grid(training_data, training_target, test_data,
                               test_target, n_jobs)
    expected = test_target

    return metrics.accuracy_score(expected, predicted)

def svm_grid_with_report(training_data, training_target, test_data, test_target, classes, n_jobs=config.processes):
    clf, predicted = _svm_grid(training_data, training_target, test_data,
                               test_target, n_jobs)

    expected = test_target
    
    report, confusion = glia.classifier_helper(clf,
        (training_data, training_target),
        (test_data, test_target), classes=classes)

    return report, confusion, predicted


def plot_acuity(logmar, accuracy, yerror,
                n_validation, name, conditions, condition_name, plot_directory,
                unit_label="logMAR"):
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
    ax.set_xlabel(unit_label)
    if unit_label=="logMAR":
        x_major_ticks = np.arange(1.6, 3.2, 0.2)
        ax.set_xticks(x_major_ticks)
        ax.set_xlim(1.5,3.25)
    elif unit_label=="cpd":
        pass
    # ax.set_xticks(minor_ticks, minor=True)
    # ax.set_yticks(major_ticks)
    # ax.set_yticks(minor_ticks, minor=True)

    ax.grid(which='both')

    ax.set_ylim(0.35,1.05)
    ax.legend(loc=(1,0.1))
    if condition_name is None:
        ax.set_title(f'{name} classification by binning technique')
        fig.tight_layout()
        fig.savefig(os.path.join(plot_directory, f"{name}_acuity.png"))
    else:
        ax.set_title(f'{name} classification by {condition_name}')
        fig.tight_layout()
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
            stimulus_list, plot_directory, name, sizes, conditions,
            condition_name, n_draws=30, units="logmar", px_per_deg=12.524):
    """
    Units is in ['logmar', 'cpd']"""
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

    if units=="logmar":
        unit_func = partial(glia.px_to_logmar, px_per_deg=px_per_deg)
        unit_label = "logMAR"
    elif units=="cpd":
        unit_func = partial(glia.px_to_cpd, px_per_deg=px_per_deg)
        unit_label = "cycles per degree (cpd)"
    logmar = list(map(unit_func,sizes))

    plot_acuity(logmar, accuracy, yerror, n_validation,
                name, conditions, condition_name, plot_directory,
                unit_label=unit_label)
    # plot_sensitivity(logmar, accuracy, yerror, n_validation,
    #             name, conditions, condition_name, plot_directory)


def checkerboard_svc(data, metadata, stimulus_list, lab_notebook, plot_directory,
                nsamples, n_draws=30, px_per_deg=10.453):
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
            sizes, conditions, condition_name, n_draws, units="cpd",
            px_per_deg=px_per_deg)


def grating_svc(data, metadata, stimulus_list, lab_notebook, plot_directory,
    nsamples, n_draws=30, sinusoid=False, px_per_deg=10.453):
    if sinusoid:
        stimulus_type = "SINUSOIDAL_GRATING"
        plot_name = "Sinusoidal grating"
    else:
        stimulus_type = 'GRATING'
        plot_name = "grating"

    sizes = glia.get_stimulus_parameters(stimulus_list, stimulus_type, "width")
    if metadata["name"] in \
        ['grating-contrast', 'grating-sinusoidal-contrast']:
        training_data = glia.bin_100ms(data["training_data"])
        validation_data = glia.bin_100ms(data["validation_data"])
        training_target = data["training_target"]
        validation_target = data["validation_target"]

        conditions = glia.get_grating_contrasts(stimulus_list, stimulus_type)
        condition_name = "contrast"
    elif metadata["name"] in \
        ["grating-durations", "grating-sinusoidal-durations"]:
        training_data = glia.bin_100ms(data["training_data"])
        validation_data = glia.bin_100ms(data["validation_data"])
        training_target = data["training_target"]
        validation_target = data["validation_target"]

        conditions = glia.get_stimulus_parameters(stimulus_list, stimulus_type, 'lifespan')
        condition_name = "durations"
    elif metadata["name"] in \
        ["grating-speeds", "grating-sinusoidal-speeds"]:
        training_data = glia.bin_100ms(data["training_data"])
        validation_data = glia.bin_100ms(data["validation_data"])
        training_target = data["training_target"]
        validation_target = data["validation_target"]

        conditions = glia.get_stimulus_parameters(stimulus_list, stimulus_type, 'speed')
        condition_name = "speeds"
    elif metadata["name"] in \
        ["grating", "grating-sinusoidal"]:
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
            plot_name, sizes, conditions, condition_name)
    else:
        acuity(training_data, training_target, validation_data, validation_target,
            stimulus_list, plot_directory, plot_name,
            sizes, conditions, condition_name, n_draws, units="cpd",
            px_per_deg=px_per_deg)


def letter_svc(data, metadata, stimulus_list, lab_notebook, plot_directory,
                nsamples, px_per_deg=10.453):
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


def generic_image_classify(data, metadata, stimulus_list, lab_notebook,
                           plot_directory, nsamples, target_mappers,
                           mapper_classes, mapper_names):
    '''
    
    target_mappers is a list of dictionaries that map classes. For example,
    suppose you have classes=np.arange(4), but 0,1 are images of the same person
    not smiling and smiling. Then, you could have {0:0,1:0,2:1,3:1} as a mapper
    to classify by person rather than by image.'''
    # n_sizes, n_training, n_steps, n_x, n_y, n_units = data["training_data"].shape
    logger.debug(data["training_data"].shape)
    # hack for dimension assumptions in bin_100ms
    training_100ms = glia.bin_100ms(data["training_data"][None,None])[0,0]
    validation_100ms = glia.bin_100ms(data["validation_data"][None,None])[0,0]   
    logger.debug(f'training_100ms shape {training_100ms.shape}')
    
    training_target = data["training_target"]
    validation_target = data["validation_target"]
    logger.debug(np.size(training_target))

    mappers_iter = zip(target_mappers, mapper_classes, mapper_names)
    for target_map, map_classes, map_name in mappers_iter:
        print(f"classifying {map_name}")
        mapped_train_target = np.array([target_map[n] for n in training_target])
        mapped_valid_target = np.array([target_map[n] for n in validation_target])
        # for each training target split
        report, confusion, predicted = svm_grid_with_report(
            training_100ms, mapped_train_target, validation_100ms,
            mapped_valid_target, map_classes)
        with open(f"{plot_directory}/{map_name}_report.txt", "w") as f:
            f.write(report+'\n')
            f.write(str(confusion))
        confusion.columns = confusion.columns.astype(str)
        confusion.to_parquet(
            f"{plot_directory}/{map_name}_confusion.parquet")
        np.save(
            f"{plot_directory}/{map_name}_predictions.npy", predicted)
