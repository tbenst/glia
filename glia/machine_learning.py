from collections import namedtuple
from .pipeline import get_unit
import numpy as np

TVT = namedtuple("TVT", ['training', "validation", "test"])


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
        assert len(my_list) == tvt.training+tvt.validation+tvt.test
        return split
            
    return anonymous


def units_to_ndarrays(units, get_class):
    """Units::Dict[unit_id,List[Experiment]] -> (ndarray, ndarray)

    get_class is a function"""
    key_map = {k: i for i,k in enumerate(sorted(list(units.keys())))}
    unitListE = get_unit(units)[1]
    duration = unitListE[0]['stimulus']['lifespan']
    for l in unitListE:
        assert duration==l['stimulus']['lifespan']
    d = int(np.ceil(duration/120*1000)) # 1ms bins
    nE = len(unitListE)
    data = np.full((nE,len(key_map.keys()),d), 0, dtype=np.int16)
    classes = np.full(nE, np.nan, dtype=np.int8)

    for unit_id, listE in units.items():
        u = key_map[unit_id]
        for i,e in enumerate(listE):
            for spike in e['spikes']:
                s = int(np.floor(spike*1000))
                data[i,u,s] = 1
            stimulus = e["stimulus"]
            classes[i] = get_class(stimulus)
            

    return (data, classes)
                
