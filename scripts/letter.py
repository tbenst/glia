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

def group_contains(stimulus_type, group):
    for experiment in group:
        if experiment["stimulus"]["stimulusType"]==stimulus_type:
            return True
    return False

group_contains_letter = partial(group_contains, "LETTER")
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

TVT = namedtuple("TVT", ['training', "validation", "test"])
# training_validation_test = TVT(6,3,3)

    
def f_split_dict(tvt):
    """Subset dict into training, validation, & test."""
    def anonymous(dictionary):
        i = 0
        split = TVT({},{},{})
        for k,v in dictionary.items():
            if i < tvt.training:
                split.training[k] = v
            elif i < tvt.validation + tvt.training:
                split.validation[k] = v
            elif i < tvt.test + tvt.validation + tvt.training:
                split.test[k] = v
            else:
                raise(ValueError, 'bad training, validation & test split.')
            i += 1
        assert i == tvt.training+tvt.validation+tvt.test
        return split
            
    return anonymous
    
def f_split_list(tvt):
    """Subset list into training, validation, & test."""
    def anonymous(my_list):
        split = TVT([],[],[])
        for i,v in enumerate(my_list):
            if i < tvt.training:
                split.training.append(v)
            elif i < tvt.validation + tvt.training:
                split.validation.append(v)
            elif i < tvt.test + tvt.validation + tvt.training:
                split.test.append(v)
            else:
                raise(ValueError, 'bad training, validation & test split.')
        assert len(my_list.keys()) == tvt.training+tvt.validation+tvt.test
        return split
            
    return anonymous

def units_to_ndarray(units):
    "Units::Dict[unit_id,List[Experiment]] -> (ndarray, ndarray)"
    key_map = {k: i for i,k in enumerate(sorted(list(units.keys())))}
    unitListE = glia.get_unit(units)[1]
    duration = unitListE[0]['stimulus']['lifespan']
    for l in unitListE:
        assert duration==l['stimulus']['lifespan']
    d = int(np.ceil(duration/120*1000)) # 1ms bins
    nE = len(unitListE)
    data = np.full((nE,len(key_map.keys()),d), 0, dtype=np.int16)
    target = np.full(nE, np.nan, dtype=np.int8)

    for unit_id, listE in units.items():
        u = key_map[unit_id]
        for i,e in enumerate(listE):
            for spike in e['spikes']:
                s = int(np.floor(spike*1000))
                data[i,u,s] = 1
            stimulus = e["stimulus"]
            if "letter" in stimulus:
                t = letter_map[stimulus["letter"]]
                target[i] = t
            else:
                t = letter_map["BLANK"]
                target[i] = t

    return (data, target)
                


def save_npz(units, stimulus_list, name):
    print("Saving NPZ file.")

    # TODO 
    training_validation_test = TVT(120,40,40)


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


    tvt_letters = glia.apply_pipeline(f_split_dict(training_validation_test),
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

    training_data, training_target = units_to_ndarray(training_letters)
    validation_data, validation_target = units_to_ndarray(validation_letters)
    test_data, test_target = units_to_ndarray(test_letters)

    np.savez(name, training_data=training_data, training_target=training_target,
         validation_data=validation_data, validation_target=validation_target,
          test_data=test_data, test_target=test_target)