from collections import namedtuple
from .pipeline import get_unit
from .types import Unit
from .functional import f_map, pmap, flatten, compose, f_filter, group_by
from .pipeline import group_dict_to_list
from .eyecandy import checkerboard_contrast
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
from functools import partial
import logging
from sklearn import datasets, svm, metrics, neighbors
import pandas as pd
logger = logging.getLogger('glia')


TVT = namedtuple("TVT", ['training', "validation", "test"])

def tvt_by_percentage(n, training=60, validation=20,testing=20):
    summed = training+validation+testing
    assert summed==100
    train = int(np.floor(n*training/100))
    valid = int(np.ceil(n*validation/100))
    test = n - valid - train
    return TVT(train, valid, test)

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
    return TVT(f(tvt.training), f(tvt.validation), f(tvt.test))

def f_split_list(tvt, get_list=lambda x: x):
    """Subset list into training, validation, & test."""
    def anonymous(x):
        split = TVT([],[],[])
        my_list = get_list(x)
        for i,v in enumerate(my_list):
            if i < tvt.training:
                split.training.append(v)
            elif i < tvt.validation + tvt.training:
                split.validation.append(v)
            elif i < tvt.test + tvt.validation + tvt.training:
                split.test.append(v)
            else:
                raise(ValueError, 'bad training, validation & test split.')
        try:
            assert len(my_list) == tvt.training+tvt.validation+tvt.test
        except Exception as e:
            print(len(my_list), tvt.training+tvt.validation+tvt.test)
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

    return (data, classes)


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

letter_map = {'K': 4, 'C': 1, 'V': 9, 'N': 5, 'R': 7, 'H': 3, 'O': 6, 'Z': 10, 'D': 2, 'S': 8, 'BLANK': 0}
letter_classes = list(map(lambda x: x[0],
                   sorted(list(letter_map.items()),
                          key=lambda x: x[1])))

def classifier_helper(classifier, training, validation, classes=letter_classes):
    training_data, training_target = training
    validation_data, validation_target = validation

    classifier.fit(training_data, training_target)
    predicted = classifier.predict(validation_data)
    expected = validation_target

    report = metrics.classification_report(expected, predicted)
    confusion = confusion_matrix(expected, predicted, classes)
    return (report, confusion)


def confusion_matrix(expected, predicted, classes=letter_classes):
    m = metrics.confusion_matrix(expected, predicted)

    return pd.DataFrame(data=m, index=classes,
                              columns=classes)


def px_to_logmar(px,px_per_deg=12.524):
    minutes = px/px_per_deg*60
    return np.log10(minutes)

def get_stimulus_parameters(stimulus_list, stimulus_type, parameter):
    f = compose(
        f_filter(lambda x: x["stimulus"]['stimulusType']==stimulus_type),
        partial(group_by,
                key=lambda x: x["stimulus"][parameter])
        )
    parameters = sorted(list(f(stimulus_list).keys()))
    logger.debug(f"Parameters: {parameters}")
    assert len(parameters)>0
    return parameters

def get_image_parameters(stimulus_list):
    f = compose(
        f_filter(lambda x: x["stimulus"]['stimulusType']=='IMAGE'),
        partial(group_by,
                key=lambda x: x["stimulus"]["metadata"]["parameter"])
        )
    parameters = sorted(list(f(stimulus_list).keys()))
    logger.debug(f"Parameters: {parameters}")
    assert len(parameters)>0
    return parameters


def get_checkerboard_contrasts(stimulus_list):
    f = compose(
        f_filter(lambda x: x["stimulus"]['stimulusType']=='CHECKERBOARD'),
        partial(group_by,
                key=lambda x: checkerboard_contrast(x["stimulus"]))
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
