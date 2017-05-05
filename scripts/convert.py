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

def save_letter_npz(units, stimulus_list, name):
    print("Saving NPZ file.")

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
        glia.f_map(f_flatten)
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
            lambda x: x.validation,
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
