from collections import namedtuple
from .pipeline import get_unit
from .types import Unit
from .functional import f_map, pmap, flatten, compose
from .pipeline import group_dict_to_list
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
from functools import partial
import logging
logger = logging.getLogger('glia')


TVT = namedtuple("TVT", ['training', "test", "validation"])

def tvt_by_percentage(n, training=60, validation=20,testing=20):
    summed = training+validation+testing
    assert summed==100
    train = int(np.floor(n*training/100))
    valid = int(np.ceil(n*validation/100))
    test = n - valid - train
    return TVT(train, test, valid)

def f_split_dict(tvt):
    """Subset dict into training, validation, & test."""
    def anonymous(dictionary):
        i = 0
        split = TVT({},{},{})
        # for k,v in dictionary.items():
        for k in sorted(list(dictionary.keys())):
            v = dictionary[k]
            if i < tvt.training:
                split.training[k] = v
            elif i < tvt.test + tvt.training:
                split.test[k] = v
            elif i < tvt.validation + tvt.test + tvt.training:
                split.validation[k] = v
            else:
                raise(ValueError, 'bad training, test & validation split.')
            i += 1
        assert i == tvt.training+tvt.test+tvt.validation
        return split

    return anonymous

training_cohorts = compose(
        lambda x: x.training,
        group_dict_to_list,
        flatten
    )

validation_cohorts = compose(
        lambda x: x.validation,
        group_dict_to_list,
        flatten
    )

test_cohorts = compose(
        lambda x: x.test,
        group_dict_to_list,
        flatten
    )

def tvt_map(tvt, f):
    return TVT(f(tvt.training), f(tvt.test), f(tvt.validation))

def f_split_list(tvt, get_list=lambda x: x):
    """Subset list into training, validation, & test."""
    def anonymous(x):
        split = TVT([],[],[])
        my_list = get_list(x)
        for i,v in enumerate(my_list):
            if i < tvt.training:
                split.training.append(v)
            elif i < tvt.test + tvt.training:
                split.test.append(v)
            elif i < tvt.validation + tvt.test + tvt.training:
                split.validation.append(v)
            else:
                raise(ValueError, 'bad training, test & validation split.')
        try:
            assert len(my_list) == tvt.training+tvt.test+tvt.validation
        except Exception as e:
            print(len(my_list), tvt.training+tvt.test+tvt.validation)
            raise e
        return split

    return anonymous


def units_to_ndarrays(units, get_class, get_list=lambda x: x):
    """Units::Dict[unit_id,List[Experiment]] -> (ndarray, ndarray)

    get_class is a function"""
    key_map = {k: i for i,k in enumerate(sorted(list(units.keys())))}
    unitListE = get_unit(units)[1]
    duration = unitListE[0]['stimulus']['lifespan']
    for l in unitListE:
        assert duration==l['stimulus']['lifespan']
    d = int(np.ceil(duration*1000)) # 1ms bins
    nE = len(unitListE)
    data = np.full((nE,d,len(key_map.keys())), 0, dtype=np.int8)
    classes = np.full(nE, np.nan, dtype=np.int8)

    for unit_id, value in units.items():
        listE = get_list(value)
        u = key_map[unit_id]
        for i,e in enumerate(listE):
            for spike in e['spikes']:
                s = int(np.floor(spike*1000))
                data[i,s,u] = 1
            stimulus = e["stimulus"]
            classes[i] = get_class(stimulus)


    return (data, classes)

def spike_train_to_sparse(experiment, key_map, shape):
    array = np.full(shape, 0, dtype=np.int8)
    for unit_id, spikes in experiment['units'].items():
        # TODO check row/column
        (row, column, unit_num) = key_map[unit_id]
        # if len(spikes)==0:
        #     print("empty spike train..?", experiment)
        for spike in spikes:
            s = int(np.floor(spike*1000))
            # if s>1000:
            #     print('>1000',spikes, e)
            #     raise ValueError()
            array[s,row,column,unit_num] = 1
    return array


def experiments_to_ndarrays(experiments, get_class=lambda x: x['metadata']['class'],
    progress=False):
    """

    get_class is a function"""
    nE = len(experiments)
    logger.info(f"number of experiments to convert to ndarray: {nE}")
    print("converting to ndarray")

    key_map = {}
    for k in experiments[0]['units'].keys():
        u = Unit.lookup[k]
        (row,column) = u.channel
        unit_num = u.unit_num
        key_map[k] = (row,column,unit_num)
    duration = experiments[0]['lifespan']
    for l in experiments:
        try:
            assert duration==l['lifespan']
        except:
            logger.info(f"duration: {duration} != {l['lifespan']}, for {l}" )
            raise
    d = int(np.ceil(duration*1000)) # 1ms bins
    # TODO hardcoded 64 channel x 10 unit
    shape = (nE,d,8,8,10)
    data = np.full(shape, 0, dtype=np.int8)
    classes = np.full(nE, np.nan, dtype=np.int8)

    # accumulate indices for value 1
    # easy to parallelize accumulation & then single-threaded mutation
    sparse = []

    classes = np.array(f_map(get_class)(experiments), dtype=np.int8)
    assert classes.shape==(nE,)

    to_sparse = partial(spike_train_to_sparse, shape=shape[1:], key_map=key_map)
    # arrays = f_map(to_sparse)(experiments)
    arrays = pmap(to_sparse, experiments, progress=True)
    # data[indices] = 1
    for i,array in enumerate(arrays):
        data[i] = array

    # for idx in indices:
        # data[idx] = 1
    # for i,e in gen:
    #     for unit_id, spikes in e['units'].items():
    #         (row, column, unit_num) = key_map[unit_id]
    #         for spike in spikes:
    #             s = int(np.floor(spike*1000))
    #             if s>1000:
    #                 print('>1000',spikes, e)
    #                 raise ValueError()
    #             data[i,s,row,column,unit_num] = 1
    #     classes[i] = get_class(e)


    return (data, classes)
