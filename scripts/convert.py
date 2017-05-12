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
    e["stimulus"]["lifespan"] = experiment["stimulus"]["lifespan"]+adjustment*120
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


def save_letter_npz(units, stimulus_list, name):
    print("Saving letters NPZ file.")

    # TODO 
    training_validation_test = glia.TVT(120,40,40)


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

    training_validation_test = glia.TVT(40,40,0)


    get_checkers = glia.compose(
        glia.f_create_experiments(stimulus_list,append_lifespan=0.5),
        partial(glia.group_by,
                key=lambda x: x["stimulus"]["metadata"]["group"]),
        glia.group_dict_to_list,
        glia.f_filter(group_contains_checkerboard),
        glia.f_map(lambda x: adjust_lifespan(x[1])),
        partial(glia.group_by,
                key=lambda x: x["stimulus"]["size"]),
    )
    checkers = glia.pure_retina_spikes(glia.apply_pipeline(get_checkers,units))

    sizes = sorted(list(checkers.keys()))
    nsizes = len(sizes)
    ncheckers = len(checkers.values())
    nunits = len(unit.keys())

    training_data = np.full((nsizes,ncheckers,nunits),0,dtype='int8')
    training_target = np.full((nsizes,ncheckers),0,dtype='int8')
    validation_data = np.full((nsizes,ncheckers,nunits),0,dtype='int8')
    validation_target = np.full((nsizes,ncheckers),0,dtype='int8')
    test_data = np.full((nsizes,ncheckers,nunits),0,dtype='int8')
    test_target = np.full((nsizes,ncheckers),0,dtype='int8')

    size_map = {s: i for i,s in enumerate(sizes)}
    split = glia.f_split_list(training_validation_test)
    for size,experiments in checkers.items():
        tvt = split(experiments)
        td, tt = glia.units_to_ndarrays(tvt.training)
        training_data[size] = td
        training_target[size] = tt

        td, tt = glia.units_to_ndarrays(tvt.validation)
        validation_data[size] = td
        validation_target[size] = tt

        td, tt = glia.units_to_ndarrays(tvt.test)
        test_data[size] = td
        test_target[size] = tt

    # TODO UNTESTED!!!!!!!!!!!!!!!!
    np.savez(name, training_data=training_data, training_target=training_target,
         validation_data=validation_data, validation_target=validation_target,
          test_data=test_data, test_target=test_target)


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
