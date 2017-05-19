import glia
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from warnings import warn
from collections import namedtuple
import logging
from copy import deepcopy

logger = logging.getLogger('glia')

letter_map = {'K': 4, 'C': 1, 'V': 9, 'N': 5, 'R': 7, 'H': 3, 'O': 6, 'Z': 10, 'D': 2, 'S': 8, 'BLANK': 0}


group_contains_letter = partial(glia.group_contains, "LETTER")
group_contains_checkerboard = partial(glia.group_contains, "CHECKERBOARD")
# SizeDuration = namedtuple('SizeDuration', ['size', 'duration'])
f_flatten = glia.f_reduce(lambda a,n: a+n,[])

def adjust_lifespan(experiment,adjustment=0.5):
    e = deepcopy(experiment)
    if 'stimulus' in experiment:
        e["stimulus"]["lifespan"] = experiment["stimulus"]["lifespan"]+adjustment*120
    else:
        e["lifespan"] = experiment["lifespan"]+adjustment*120
    return e

def truncate(experiment,adjustment=0.5):
    e = deepcopy(experiment)
    lifespan = e["stimulus"]["lifespan"] #-adjustment*120
#     e["stimulus"]["lifespan"] = lifespan
    e["spikes"] = e["spikes"][np.where(e["spikes"]<lifespan/120)] 
    return e

# training_validation_test = TVT(6,3,3)


def letter_class(stimulus):
    if "letter" in stimulus:
        return letter_map[stimulus["letter"]]
    else:
        return letter_map["BLANK"]

def checker_class(stimulus):
    checker = stimulus["metadata"]["class"]
    if checker=='A':
        return 0
    elif checker=='B':
        return 1
    else:
        raise ValueError

def balance_blanks(cohort):
    """Remove 90% of blanks."""
    new = []
    includes_blank = False
    for e in cohort:
        if 'letter' in e["stimulus"]:
            new.append(e)
        else:
            if not includes_blank:
                new.append(e)
                includes_blank = True
    return new


def save_eyechart_npz(units, stimulus_list, name):
    print("Saving eyechart NPZ file.")

    # TODO add blanks
    get_letters = glia.compose(
        partial(glia.create_experiments,
            stimulus_list=stimulus_list,append_lifespan=0.5),
        partial(glia.group_by,
                key=lambda x: x["metadata"]["group"]),
        glia.group_dict_to_list,
        glia.f_filter(group_contains_letter),
        glia.f_map(lambda x: adjust_lifespan(x[1])),
        partial(glia.group_by,
                key=lambda x: x["size"]),
        glia.f_map(partial(glia.group_by,
                key=lambda x: x[1]["stimulus"]["metadata"]["cohort"])),
        glia.f_map(glia.f_map(f_flatten)),
    )
    letters = get_letters(units)
    # TODO account for cohorts
    sizes = sorted(list(letters.keys()))
    nsizes = len(sizes)
    ncohorts = len(list(letters.values())[0])
    ex_letters = glia.get_a_value(list(letters.values())[0])
    nletters = len(ex_letters)
    print("nletters",nletters)
    duration = ex_letters[0]["lifespan"]
    d = int(np.ceil(duration/120*1000)) # 1ms bins
    nunits = len(units.keys())
    tvt = glia.tvt_by_percentage(ncohorts,60,40,0)
    training_data = np.full((nsizes,tvt.training,d,nunits),0,dtype='int8')
    training_target = np.full((nsizes,tvt.training),0,dtype='int8')
    validation_data = np.full((nsizes,tvt.validation,d,nunits),0,dtype='int8')
    validation_target = np.full((nsizes,tvt.validation),0,dtype='int8')
    test_data = np.full((nsizes,tvt.test,d,nunits),0,dtype='int8')
    test_target = np.full((nsizes,tvt.test),0,dtype='int8')

    size_map = {s: i for i,s in enumerate(sizes)}
    for size, experiments in letters.items():
        split = glia.f_split_dict(tvt)
        flatten_cohort = glia.compose(
            glia.group_dict_to_list,
            f_flatten
        )
        X = glia.tvt_map(split(experiments), flatten_cohort)

        td, tt = glia.experiments_to_ndarrays(X.training, letter_class)
        size_index = size_map[size]
        training_data[size_index] = td
        training_target[size_index] = tt

        td, tt = glia.experiments_to_ndarrays(X.validation, letter_class)
        validation_data[size_index] = td
        validation_target[size_index] = tt

        td, tt = glia.experiments_to_ndarrays(X.test, letter_class)
        test_data[size_index] = td
        test_target[size_index] = tt

    np.savez(name, training_data=training_data, training_target=training_target,
         validation_data=validation_data, validation_target=validation_target)
          # test_data=test_data, test_target=test_target)


def save_letter_npz(units, stimulus_list, name):
    print("Saving letters NPZ file.")

    get_letters = glia.compose(
        glia.f_create_experiments(stimulus_list,append_lifespan=0.5),
        partial(glia.group_by,
                key=lambda x: x["stimulus"]["metadata"]["group"]),
        glia.group_dict_to_list,
        glia.f_filter(group_contains_letter),
        glia.f_map(lambda x: x[0:2]),
        glia.f_map(lambda x: [truncate(x[0]), adjust_lifespan(x[1])]),
        partial(glia.group_by,
                key=lambda x: x[1]["stimulus"]["metadata"]["cohort"]),
        glia.f_map(f_flatten),
        glia.f_map(balance_blanks)
    )
    letters = glia.apply_pipeline(get_letters,units)

    n = len(list(glia.get_unit(letters)[1].keys()))
    training_validation_test = glia.tvt_by_percentage(n,60,20,20)

    tvt_letters = glia.apply_pipeline(glia.f_split_dict(training_validation_test),
                                      letters)
    training_letters = glia.apply_pipeline(
        glia.compose(
            lambda x: x.training,
            glia.group_dict_to_list,
            f_flatten
        ),
        tvt_letters, progress=True)

    validation_letters = glia.apply_pipeline(
        glia.compose(
            lambda x: x.validation,
            glia.group_dict_to_list,
            f_flatten
        ),
        tvt_letters)

    test_letters = glia.apply_pipeline(
        glia.compose(
            lambda x: x.test,
            glia.group_dict_to_list,
            f_flatten
        ),
        tvt_letters)

    training_data, training_target = glia.units_to_ndarrays(training_letters, letter_class)
    validation_data, validation_target = glia.units_to_ndarrays(validation_letters, letter_class)
    test_data, test_target = glia.units_to_ndarrays(test_letters, letter_class)

    np.savez(name, training_data=training_data, training_target=training_target,
         validation_data=validation_data, validation_target=validation_target,
          test_data=test_data, test_target=test_target)


def save_checkerboard_npz(units, stimulus_list, name):
    print("Saving checkerboard NPZ file.")

    # TODO make generic by filter stimulus_list
    # & make split by cohort (currently by list due to
    # eyecandy mistake)

    get_checkers = glia.compose(
        partial(glia.create_experiments,
            stimulus_list=stimulus_list,append_lifespan=0.5),
        partial(glia.group_by,
                key=lambda x: x["metadata"]["group"]),
        glia.group_dict_to_list,
        glia.f_filter(group_contains_checkerboard),
        glia.f_map(lambda x: adjust_lifespan(x[1])),
        partial(glia.group_by,
                key=lambda x: x["size"]),
    )
    checkers = get_checkers(units)

    sizes = sorted(list(checkers.keys()))
    nsizes = len(sizes)
    ncheckers = len(list(checkers.values())[0])
    # print(list(checkers.values()))
    duration = list(checkers.values())[0][0]["lifespan"]
    d = int(np.ceil(duration/120*1000)) # 1ms bins
    nunits = len(units.keys())
    tvt = glia.tvt_by_percentage(ncheckers,60,40,0)
    training_data = np.full((nsizes,tvt.training,d,nunits),0,dtype='int8')
    training_target = np.full((nsizes,tvt.training),0,dtype='int8')
    validation_data = np.full((nsizes,tvt.validation,d,nunits),0,dtype='int8')
    validation_target = np.full((nsizes,tvt.validation),0,dtype='int8')
    # test_data = np.full((nsizes,tvt.test,d,nunits),0,dtype='int8')
    # test_target = np.full((nsizes,tvt.test),0,dtype='int8')

    size_map = {s: i for i,s in enumerate(sizes)}
    for size, experiments in checkers.items():
        split = glia.f_split_list(tvt)
        n = len(experiments)
        half = int(n/2)
        sorted_exp = sorted(experiments, key=checker_class)
        class_a = sorted_exp[0:half]
        class_b = sorted_exp[half:n]
        # careful to split evenly across classes
        balanced_experiments = [None]*n
        balanced_experiments[::2] = class_a
        balanced_experiments[1::2] = class_b
        X = split(balanced_experiments)
        td, tt = glia.experiments_to_ndarrays(X.training, checker_class)
        size_index = size_map[size]
        training_data[size_index] = td
        training_target[size_index] = tt

        td, tt = glia.experiments_to_ndarrays(X.validation, checker_class)
        validation_data[size_index] = td
        validation_target[size_index] = tt

        # td, tt = glia.experiments_to_ndarrays(X.test, checker_class)
        # test_data[size_index] = td
        # test_target[size_index] = tt

    np.savez(name, training_data=training_data, training_target=training_target,
         validation_data=validation_data, validation_target=validation_target)
          # test_data=test_data, test_target=test_target)


def save_integrity_npz(units, stimulus_list, name):
    print("Saving integrity NPZ file.")

    get_solid = glia.compose(
        glia.f_create_experiments(stimulus_list),
        glia.filter_integrity,
        partial(glia.group_by,
            key=lambda x: x["stimulus"]["metadata"]["group"]),
        glia.group_dict_to_list,
        partial(sorted,key=lambda x: x[0]["stimulus"]["stimulusIndex"])
        )

    response = glia.apply_pipeline(get_solid,units, progress=True)

    integrity_stimuli = list(filter(lambda x: "label" in x["metadata"] and \
        x["metadata"]["label"]=="integrity", l))
    # TODO not finished
    training_validation_test = glia.TVT(120,40,40)

    tvt_letters = glia.apply_pipeline(glia.f_split_dict(training_validation_test),
                                      letters)
    training_letters = glia.apply_pipeline(
        glia.compose(
            lambda x: x.training,
            glia.group_dict_to_list,
            f_flatten
        ),
        tvt_letters, progress=True)

    validation_letters = glia.apply_pipeline(
        glia.compose(
            lambda x: x.validation,
            glia.group_dict_to_list,
            f_flatten
        ),
        tvt_letters)

    test_letters = glia.apply_pipeline(
        glia.compose(
            lambda x: x.test,
            glia.group_dict_to_list,
            f_flatten
        ),
        tvt_letters)

    training_data, training_target = glia.units_to_ndarrays(training_letters, letter_class)
    validation_data, validation_target = glia.units_to_ndarrays(validation_letters, letter_class)
    test_data, test_target = glia.units_to_ndarrays(test_letters, letter_class)

    np.savez(name, training_data=training_data, training_target=training_target,
         validation_data=validation_data, validation_target=validation_target,
          test_data=test_data, test_target=test_target)
