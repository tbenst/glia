import glia
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from warnings import warn
from collections import namedtuple
from copy import deepcopy
import logging
logger = logging.getLogger('glia')

letter_map = {'K': 4, 'C': 1, 'V': 9, 'N': 5, 'R': 7, 'H': 3, 'O': 6, 'Z': 10, 'D': 2, 'S': 8, 'BLANK': 0}


group_contains_letter = partial(glia.group_contains, "LETTER")
group_contains_checkerboard = partial(glia.group_contains, "CHECKERBOARD")
# SizeDuration = namedtuple('SizeDuration', ['size', 'duration'])
f_flatten = glia.f_reduce(lambda a,n: a+n,[])

def adjust_lifespan(experiment,adjustment=0.5):
    e = deepcopy(experiment)
    if 'stimulus' in experiment:
        e["stimulus"]["lifespan"] = experiment["stimulus"]["lifespan"]+adjustment
    else:
        e["lifespan"] = experiment["lifespan"]+adjustment
    return e

def truncate(experiment,adjustment=0.5):
    e = deepcopy(experiment)
    if 'stimulus' in experiment:
        lifespan = e["stimulus"]["lifespan"] #-adjustment
    #     e["stimulus"]["lifespan"] = lifespan
        e["spikes"] = e["spikes"][np.where(e["spikes"]<lifespan)] 
    else:
        lifespan = e['lifespan']
        e['units'] = glia.f_map(lambda x: x[np.where(x<lifespan)])(e['units'])
    return e


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

def grating_class(stimulus):
    grating = stimulus["metadata"]["class"]
    if grating=='FORWARD':
        return 0
    elif grating=='REVERSE':
        return 1
    else:
        raise ValueError

def checker_discrimination_class(stimulus):
    checker = stimulus["metadata"]["target"]
    if checker=='SAME':
        return 0
    elif checker=='DIFFERENT':
        return 1
    else:
        raise ValueError

def balance_blanks(cohort):
    """Remove 90% of blanks (all but first)."""
    new = []
    includes_blank = False
    for e in cohort:
        if "stimulus" in e:
            letter = lambda x: 'letter' in e["stimulus"]
        else:
            letter = lambda x: 'letter' in e

        if letter(e):
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
    d = int(np.ceil(duration*1000)) # 1ms bins
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
    print("Saving letter NPZ file.")

    # TODO TEST!!!
    get_letters = glia.compose(
        partial(glia.create_experiments,
            # stimulus_list=stimulus_list,progress=True),
            stimulus_list=stimulus_list,append_lifespan=0.5,progress=True),
        partial(glia.group_by,
                key=lambda x: x["metadata"]["group"]),
        glia.group_dict_to_list,
        glia.f_filter(group_contains_letter),
        glia.f_map(lambda x: x[0:2]),
        # glia.f_map(lambda x: x[1]),
        glia.f_map(lambda x: [truncate(x[0]), adjust_lifespan(x[1])]),
        partial(glia.group_by,
                key=lambda x: x[1]["metadata"]["cohort"]),
                # key=lambda x: x["metadata"]["cohort"]),
        glia.f_map(f_flatten),
        glia.f_map(balance_blanks)
    )
    letters = get_letters(units)


    ncohorts = len(letters.keys())
    training_validation_test = glia.tvt_by_percentage(ncohorts,60,20,20)
    tvt_letters = glia.f_split_dict(training_validation_test)(letters)
    
    training_letters = glia.compose(
            lambda x: x.training,
            glia.group_dict_to_list,
            f_flatten
        )(tvt_letters)
    # print("#34",training_letters[34])
    validation_letters = glia.compose(
            lambda x: x.validation,
            glia.group_dict_to_list,
            f_flatten
        )(tvt_letters)

    test_letters = glia.compose(
            lambda x: x.test,
            glia.group_dict_to_list,
            f_flatten
        )(tvt_letters)

    training_data, training_target = glia.experiments_to_ndarrays(
        training_letters, letter_class, progress=True)
    validation_data, validation_target = glia.experiments_to_ndarrays(
        validation_letters, letter_class, progress=True)
    test_data, test_target = glia.experiments_to_ndarrays(
        test_letters, letter_class, progress=True)

    np.savez(name, training_data=training_data, training_target=training_target,
         validation_data=validation_data, validation_target=validation_target,
          test_data=test_data, test_target=test_target)




def save_checkerboard_npz(units, stimulus_list, name):
    "Psychophysics discrimination checkerboard 0.2.0"
    print("Saving checkerboard NPZ file.")

    # TODO make generic by filter stimulus_list
    # & make split by cohort (currently by list due to
    # eyecandy mistake)

    get_checkers = glia.compose(
        partial(glia.create_experiments, progress=True,
            stimulus_list=stimulus_list,append_lifespan=0.5),
        partial(glia.group_by,
                key=lambda x: x["metadata"]["group"]),
        glia.group_dict_to_list,
        glia.f_filter(group_contains_checkerboard),
        glia.f_map(lambda x: [x[1],adjust_lifespan(x[2])]),
        glia.f_map(glia.merge_experiments),
        partial(glia.group_by,
                key=lambda x: glia.checkerboard_contrast(x)),
        glia.f_map(partial(glia.group_by,
                key=lambda x: x["size"])),
        glia.f_map(glia.f_map(partial(glia.group_by,
                key=lambda x: x["metadata"]["cohort"])))
    )
    checkers = get_checkers(units)

    contrasts = sorted(list(checkers.keys()))
    ncontrasts = len(contrasts)
    example_contrast = glia.get_value(checkers)
    sizes = sorted(list(example_contrast.keys()))
    nsizes = len(sizes)

    example_size = glia.get_value(example_contrast)
    ncohorts = len(example_size)
    # print(list(checkers.values()))
    duration = glia.get_value(example_size)[0]["lifespan"]
    d = int(np.ceil(duration*1000)) # 1ms bins
    tvt = glia.tvt_by_percentage(ncohorts,60,40,0)
    # (TODO?) 2 dims for first checkerboard and second checkerboard
    # 4 per cohort
    training_data = np.full((ncontrasts,nsizes,
        tvt.training*4,d,8,8,10),0,dtype='int8')
    training_target = np.full((ncontrasts,nsizes,
        tvt.training*4),0,dtype='int8')
    validation_data = np.full((ncontrasts,nsizes,
        tvt.validation*4,d,8,8,10),0,dtype='int8')
    validation_target = np.full((ncontrasts,nsizes,
        tvt.validation*4),0,dtype='int8')
    # test_data = np.full((nsizes,tvt.test,d,nunits),0,dtype='int8')
    # test_target = np.full((nsizes,tvt.test),0,dtype='int8')

    contrast_map = {c: i for i,c in enumerate(contrasts)}
    size_map = {s: i for i,s in enumerate(sizes)}
    for contrast, sizes in checkers.items():
        for size, cohorts in sizes.items():
            X = glia.f_split_dict(tvt)(cohorts)

            td, tt = glia.experiments_to_ndarrays(glia.training_cohorts(X), checker_discrimination_class)
            contrast_index = contrast_map[contrast]
            size_index = size_map[size]
            training_data[contrast_index, size_index] = td
            training_target[contrast_index, size_index] = tt

            td, tt = glia.experiments_to_ndarrays(glia.validation_cohorts(X), checker_discrimination_class)
            validation_data[contrast_index, size_index] = td
            validation_target[contrast_index, size_index] = tt

            # td, tt = glia.experiments_to_ndarrays(X.test, checker_class)
            # test_data[size_index] = td
            # test_target[size_index] = tt
    print('saving to ',name)
    np.savez(name, training_data=training_data, training_target=training_target,
         validation_data=validation_data, validation_target=validation_target)
          # test_data=test_data, test_target=test_target)



def save_grating_npz(units, stimulus_list, name):
    print("Saving grating NPZ file.")

    get_gratings = glia.compose(
        partial(glia.create_experiments,
            stimulus_list=stimulus_list,append_lifespan=0.5,progress=True),
        glia.f_map(lambda x: adjust_lifespan(x)),
        glia.f_filter(lambda x: x['stimulusType']=='GRATING'),
        partial(glia.group_by,
                key=lambda x: x["width"]),
        glia.f_map(partial(glia.group_by,
                key=lambda x: x["metadata"]["cohort"]))    )
    gratings = get_gratings(units)

    gratings = get_gratings(units)

    sizes = sorted(list(gratings.keys()))
    nsizes = len(sizes)
    ncohorts = len(glia.get_value(gratings))
    duration = glia.get_value(glia.get_value(gratings))[0]["lifespan"]
    d = int(np.ceil(duration*1000)) # 1ms bins
    tvt = glia.tvt_by_percentage(ncohorts,60,40,0)
    # 2 per cohort
    training_data = np.full((nsizes,tvt.training*2,d,8,8,10),0,dtype='int8')
    training_target = np.full((nsizes,tvt.training*2),0,dtype='int8')
    validation_data = np.full((nsizes,tvt.validation*2,d,8,8,10),0,dtype='int8')
    validation_target = np.full((nsizes,tvt.validation*2),0,dtype='int8')

    size_map = {s: i for i,s in enumerate(sizes)}
    for size, cohorts in gratings.items():
        X = glia.f_split_dict(tvt)(cohorts)

        td, tt = glia.experiments_to_ndarrays(glia.training_cohorts(X), grating_class)
        size_index = size_map[size]
        training_data[size_index] = td
        training_target[size_index] = tt

        td, tt = glia.experiments_to_ndarrays(glia.validation_cohorts(X), grating_class)
        validation_data[size_index] = td
        validation_target[size_index] = tt

    print('saving to ',name)
    np.savez(name, training_data=training_data, training_target=training_target,
         validation_data=validation_data, validation_target=validation_target)