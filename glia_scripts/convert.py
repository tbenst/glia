import glia
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from warnings import warn
import warnings
from collections import namedtuple
from copy import deepcopy
import logging, tables
logger = logging.getLogger('glia')
from glia.types import Unit
from tables import NaturalNameWarning
import json
warnings.filterwarnings("ignore", category=NaturalNameWarning)

letter_map = {'K': 4, 'C': 1, 'V': 9, 'N': 5, 'R': 7, 'H': 3, 'O': 6, 'Z': 10, 'D': 2, 'S': 8, 'BLANK': 0}


group_contains_letter = partial(glia.group_contains, "LETTER")
group_contains_tiled_letter = partial(glia.group_contains, "TILED_LETTER")
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

def truncate(experiment,lifespan):
    e = deepcopy(experiment)
    if 'stimulus' in experiment:
        e["spikes"] = e["spikes"][np.where(e["spikes"]<lifespan)]
        e['lifespan'] = lifespan
    else:
        e['units'] = glia.f_map(lambda x: x[np.where(x<lifespan)])(e['units'])
        e['lifespan'] = lifespan
    return e


def letter_class(stimulus):
    if "letter" in stimulus:
        return letter_map[stimulus["letter"]]
    else:
        return letter_map["BLANK"]

def acuity_image_class(stimulus):
    metadata = stimulus["metadata"]
    if "target" in metadata:
        return letter_map[metadata["target"]]
    else:
        return letter_map["BLANK"]
    
def get_classes_from_stimulus_list(stimulus_list):
    "use `stimulus.metadata.class` as (JSONstring) key, and resolve to a number"
    class_resolver = dict()
    for s in stimulus_list:
        metadata = s['stimulus']['metadata']
        if "class" in metadata:
            c = json.dumps(metadata['class'])
            l = json.dumps(metadata['classLabels'])
            class_resolver[c] = l
            
    label = glia.get_value(class_resolver)
    for i,k in enumerate(class_resolver.keys()):
        assert class_resolver[k] == label # we only support one classLabel type
        class_resolver[k] = i
    return class_resolver

def get_image_class_from_stim(stimulus, class_resolver):
    "Based on precedent in 20faces.js"
    metadata = stimulus["metadata"]
    return class_resolver(metadata["class"])
        

def checker_class(stimulus):
    checker = stimulus["metadata"]["class"]
    if checker=='A':
        return 0
    elif checker=='B':
        return 1
    else:
        raise ValueError

def get_grating_class_from_stim(stimulus):
    grating = stimulus["metadata"]["class"]
    if grating=='FORWARD':
        return 0
    elif grating=='REVERSE':
        return 1
    else:
        raise ValueError

def get_checker_discrimination_class(stimulus):
    checker = stimulus["metadata"]["target"]
    if checker=='SAME':
        return 0
    elif checker=='DIFFERENT':
        return 1
    else:
        raise ValueError

def get_checker_quad_discrimination_class(stimulus):
    checker = stimulus["metadata"]["target"]
    # frame A or frame B
    first_class = stimulus["metadata"]["class"]
    if checker=='SAME':
        if first_class=="A":
            return 0
        elif first_class=='B':
            return 2
        else:
            raise(Error('bad class'))
    elif checker=='DIFFERENT':
        if first_class=="A":
            return 1
        elif first_class=='B':
            return 3
        else:
            raise(Error('bad class'))
    else:
        raise ValueError

def balance_blanks(cohort, key='letter'):
    """Remove 90% of blanks (all but first)."""
    new = []
    includes_blank = False
    for e in cohort:
        if "stimulus" in e:
            letter = lambda x: key in e["stimulus"]
        else:
            letter = lambda x: key in e

        if letter(e):
            new.append(e)
        else:
            if not includes_blank:
                new.append(e)
                includes_blank = True
    return new


def save_eyechart_npz(units, stimulus_list, name, append=0.5):
    print("Saving eyechart NPZ file.")

    # TODO add blanks
    get_letters = glia.compose(
        partial(glia.create_experiments,
            stimulus_list=stimulus_list,append_lifespan=append),
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
    logger.info(f"{tvt}, {ncohorts}")
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

        td, tt = glia.experiments_to_ndarrays(X.training, letter_class, append)
        size_index = size_map[size]
        training_data[size_index] = td
        training_target[size_index] = tt

        td, tt = glia.experiments_to_ndarrays(X.validation, letter_class, append)
        validation_data[size_index] = td
        validation_target[size_index] = tt

        td, tt = glia.experiments_to_ndarrays(X.test, letter_class, append)
        test_data[size_index] = td
        test_target[size_index] = tt

    np.savez(name, training_data=training_data, training_target=training_target,
         validation_data=validation_data, validation_target=validation_target)
          # test_data=test_data, test_target=test_target)

def save_letter_npz(units, stimulus_list, name, append):
    print("Saving letter NPZ file. Warning: not including Off response--performance can be improved!")
    # TODO use merge_experiment
    # TODO add TEST!!!
    get_letters = glia.compose(
        partial(glia.create_experiments,
            stimulus_list=stimulus_list,progress=True,append_lifespan=append),
        partial(glia.group_by,
                key=lambda x: x["metadata"]["group"]),
        glia.group_dict_to_list,
        glia.f_filter(group_contains_letter),
        glia.f_map(lambda x: x[0:2]),
        partial(glia.group_by,
                key=lambda x: x[1]["size"]),
        glia.f_map(partial(glia.group_by,
                key=lambda x: x[1]["metadata"]["cohort"])),
        glia.f_map(glia.f_map(f_flatten)),
        glia.f_map(glia.f_map(balance_blanks))
    )
    letters = get_letters(units)
    sizes = sorted(list(letters.keys()))
    nsizes = len(sizes)
    ncohorts = len(list(letters.values())[0])
    ex_letters = glia.get_value(list(letters.values())[0])
    nletters = len(ex_letters)
    print("nletters",nletters)
    duration = ex_letters[0]["lifespan"]


    d = int(np.ceil(duration*1000)) # 1ms bins
    nunits = len(units.keys())
    tvt = glia.tvt_by_percentage(ncohorts,60,40,0)
    logger.info(f"{tvt}, ncohorts: {ncohorts}")

    experiments_per_cohort = 11
    training_data = np.full((nsizes,
        tvt.training*experiments_per_cohort,d,Unit.nrow,Unit.ncol,Unit.nunit),0,dtype='int8')
    training_target = np.full((nsizes,
        tvt.training*experiments_per_cohort),0,dtype='int8')
    validation_data = np.full((nsizes,
        tvt.validation*experiments_per_cohort,d,Unit.nrow,Unit.ncol,Unit.nunit),0,dtype='int8')
    validation_target = np.full((nsizes,
        tvt.validation*experiments_per_cohort),0,dtype='int8')

    size_map = {s: i for i,s in enumerate(sizes)}
    for size, cohorts in letters.items():
        X = glia.f_split_dict(tvt)(cohorts)
        logger.info(f"ncohorts: {len(cohorts)}")
        td, tt = glia.experiments_to_ndarrays(glia.training_cohorts(X),
                    letter_class, append)
        logger.info(td.shape)
        missing_duration = d - td.shape[1]
        pad_td = np.pad(td,
            ((0,0),(0,missing_duration),(0,0),(0,0),(0,0)),
            mode='constant')
        size_index = size_map[size]
        training_data[size_index] = pad_td
        training_target[size_index] = tt

        td, tt = glia.experiments_to_ndarrays(glia.validation_cohorts(X),
                    letter_class, append)
        pad_td = np.pad(td,
            ((0,0),(0,missing_duration),(0,0),(0,0),(0,0)),
            mode='constant')
        validation_data[size_index] = pad_td
        validation_target[size_index] = tt

    np.savez(name, training_data=training_data, training_target=training_target,
         validation_data=validation_data, validation_target=validation_target)
    #   test_data=test_data, test_target=test_target)

def save_letters_npz(units, stimulus_list, append, name, contains=group_contains_tiled_letter):

    get_letters = glia.compose(
        partial(glia.create_experiments,
            stimulus_list=stimulus_list,progress=True, append_lifespan=append),
        partial(glia.group_by,
                key=lambda x: x["metadata"]["group"]),
        glia.group_dict_to_list,
        glia.f_filter(contains),
        glia.f_map(lambda x: x[0:2]),
        partial(glia.group_by,
                key=lambda x: x[1]["size"]),
        glia.f_map(partial(glia.group_by,
                key=lambda x: x[1]["metadata"]["cohort"])),
        glia.f_map(glia.f_map(f_flatten)),
        glia.f_map(glia.f_map(balance_blanks))
    )
    letters = get_letters(units)
    sizes = sorted(list(letters.keys()))
    nsizes = len(sizes)
    ncohorts = len(list(letters.values())[0])
    ex_letters = glia.get_value(list(letters.values())[0])
    nletters = len(ex_letters)
    print("nletters",nletters)
    duration = ex_letters[0]["lifespan"]

    # small hack to fix bug in letters 0.2.0
    letter_duration = ex_letters[1]['lifespan']
    if duration!=letter_duration:
        new_letters = {}
        for size, cohorts in letters.items():
            new_letters[size] = {}
            for cohort, stimuli in cohorts.items():
                new_letters[size][cohort] = list(map(lambda s: truncate(s, letter_duration), stimuli))
        letters = new_letters


    d = int(np.ceil(duration*1000)) # 1ms bins
    nunits = len(units.keys())
    tvt = glia.tvt_by_percentage(ncohorts,60,40,0)
    logger.info(f"{tvt}, ncohorts: {ncohorts}")

    experiments_per_cohort = 11
    training_data = np.full((nsizes,
        tvt.training*experiments_per_cohort,d,Unit.nrow,Unit.ncol,Unit.nunit),0,dtype='int8')
    training_target = np.full((nsizes,
        tvt.training*experiments_per_cohort),0,dtype='int8')
    validation_data = np.full((nsizes,
        tvt.validation*experiments_per_cohort,d,Unit.nrow,Unit.ncol,Unit.nunit),0,dtype='int8')
    validation_target = np.full((nsizes,
        tvt.validation*experiments_per_cohort),0,dtype='int8')

    size_map = {s: i for i,s in enumerate(sizes)}
    for size, cohorts in letters.items():
        X = glia.f_split_dict(tvt)(cohorts)
        logger.info(f"ncohorts: {len(cohorts)}")
        td, tt = glia.experiments_to_ndarrays(glia.training_cohorts(X),
                    letter_class, append)
        logger.info(td.shape)
        missing_duration = d - td.shape[1]
        pad_td = np.pad(td,
            ((0,0),(0,missing_duration),(0,0),(0,0),(0,0)),
            mode='constant')
        size_index = size_map[size]
        training_data[size_index] = pad_td
        training_target[size_index] = tt

        td, tt = glia.experiments_to_ndarrays(glia.validation_cohorts(X),
                    letter_class, append)
        pad_td = np.pad(td,
            ((0,0),(0,missing_duration),(0,0),(0,0),(0,0)),
            mode='constant')
        validation_data[size_index] = pad_td
        validation_target[size_index] = tt

    np.savez(name, training_data=training_data, training_target=training_target,
         validation_data=validation_data, validation_target=validation_target)
    #   test_data=test_data, test_target=test_target)

def save_acuity_image_npz(units, stimulus_list, name, append):
    "Assumes metadata includes a parameter to group by, as well as a blank image"

    get_letters = glia.compose(
        partial(glia.create_experiments,
            stimulus_list=stimulus_list,progress=True, append_lifespan=append),
        partial(glia.group_by,
                key=lambda x: x["metadata"]["group"]),
        glia.group_dict_to_list,
        glia.f_filter(partial(glia.group_contains, "IMAGE")),
        glia.f_map(lambda x: x[0:2]),
        partial(glia.group_by,
                key=lambda x: x[1]["metadata"]["parameter"]),
        glia.f_map(partial(glia.group_by,
                key=lambda x: x[1]["metadata"]["cohort"])),
        glia.f_map(glia.f_map(f_flatten)),
        glia.f_map(glia.f_map(partial(balance_blanks, key='image')))
    )
    letters = get_letters(units)
    sizes = sorted(list(letters.keys()))
    nsizes = len(sizes)
    ncohorts = len(list(letters.values())[0])
    ex_letters = glia.get_value(list(letters.values())[0])
    nletters = len(ex_letters)
    print("nletters",nletters)
    duration = ex_letters[0]["lifespan"]

    # small hack to fix bug in letters 0.2.0
    letter_duration = ex_letters[1]['lifespan']
    if duration!=letter_duration:
        new_letters = {}
        for size, cohorts in letters.items():
            new_letters[size] = {}
            for cohort, stimuli in cohorts.items():
                new_letters[size][cohort] = list(map(lambda s: truncate(s, letter_duration), stimuli))
        letters = new_letters


    d = int(np.ceil(duration*1000)) # 1ms bins
    nunits = len(units.keys())
    tvt = glia.tvt_by_percentage(ncohorts,60,40,0)
    logger.info(f"{tvt}, ncohorts: {ncohorts}")

    experiments_per_cohort = 11
    training_data = np.full((nsizes,
        tvt.training*experiments_per_cohort,d,Unit.nrow,Unit.ncol,Unit.nunit),0,dtype='int8')
    training_target = np.full((nsizes,
        tvt.training*experiments_per_cohort),0,dtype='int8')
    validation_data = np.full((nsizes,
        tvt.validation*experiments_per_cohort,d,Unit.nrow,Unit.ncol,Unit.nunit),0,dtype='int8')
    validation_target = np.full((nsizes,
        tvt.validation*experiments_per_cohort),0,dtype='int8')

    size_map = {s: i for i,s in enumerate(sizes)}
    for size, cohorts in letters.items():
        X = glia.f_split_dict(tvt)(cohorts)
        logger.info(f"ncohorts: {len(cohorts)}")
        td, tt = glia.experiments_to_ndarrays(glia.training_cohorts(X),
                    acuity_image_class, append)
        logger.info(td.shape)
        missing_duration = d - td.shape[1]
        pad_td = np.pad(td,
            ((0,0),(0,missing_duration),(0,0),(0,0),(0,0)),
            mode='constant')
        size_index = size_map[size]
        training_data[size_index] = pad_td
        training_target[size_index] = tt

        td, tt = glia.experiments_to_ndarrays(glia.validation_cohorts(X),
                    acuity_image_class, append)
        pad_td = np.pad(td,
            ((0,0),(0,missing_duration),(0,0),(0,0),(0,0)),
            mode='constant')
        validation_data[size_index] = pad_td
        validation_target[size_index] = tt

    np.savez(name, training_data=training_data, training_target=training_target,
         validation_data=validation_data, validation_target=validation_target)
    #   test_data=test_data, test_target=test_target)


def save_images_h5(units, stimulus_list, name, frame_log, 
        video_file, append):
    """Assumes each group is three stimuli with image in second position.
    
    Concatenate second stimuli with first 0.5s of third stimuli"""
    # open first so if there's a problem we don't waste time
    compression_level = 3
    dset_filter = tables.filters.Filters(complevel=compression_level,
        complib='blosc:zstd')
    with tables.open_file(name + ".h5", 'w') as h5:
        class_resolver = get_classes_from_stimulus_list(stimulus_list)
        nclasses = len(class_resolver)
        frames, image_classes = glia.get_images_from_vid(stimulus_list, frame_log, video_file)

        # serialize as JSON
        image_class_num = list(map(lambda x: class_resolver[json.dumps(x)],
            image_classes))
        idx_sorted_order = np.argsort(image_class_num)

        # save mapping of class_num target to class metadata
        # this way h5.root.image_classes[n] will give the class metadata string
        logger.info("create class_resolver with max string of 256")
        resolver = h5.create_carray(h5.root, "image_classes",
                        tables.StringAtom(itemsize=256),(nclasses,))
        img_class_array = np.array([json.dumps(c) for c in image_classes],
                                   dtype="S256")[idx_sorted_order]
        for i, image_class in enumerate(img_class_array):
            resolver[i] = image_class
        
        atom = tables.Atom.from_dtype(frames[0].dtype)
        images = h5.create_carray(h5.root, "images", atom,
            (nclasses, *frames[0].shape),
            filters=dset_filter)
        
        frames = np.array(frames)
        nFrames = len(frames)
        for i, idx in enumerate(idx_sorted_order):
            if idx >= nFrames:
                logger.warn(f"skipping class {image_classes[idx]} as no accompanying frame. This should only occur if experiment stopped early.")
                continue
            images[i] = frames[idx]

        print("finished saving images")
        get_image_responses = glia.compose(
            # returns a list
            partial(glia.create_experiments,
                stimulus_list=stimulus_list,progress=True, append_lifespan=append),
            partial(glia.group_by,
                key=lambda x: x["metadata"]["group"]),
            glia.group_dict_to_list,
            glia.f_filter(partial(glia.group_contains, "IMAGE")),
            # truncate to 0.5s
            glia.f_map(lambda x: [x[1], truncate(x[2],0.5)]),
            glia.f_map(glia.merge_experiments),
            partial(glia.group_by,
                    key=lambda x: x["metadata"]["cohort"]),
            # glia.f_map(f_flatten)
        )
        
        image_responses = get_image_responses(units)
        ncohorts = len(image_responses)
        ex_cohort = glia.get_value(image_responses)
        images_per_cohort = len(ex_cohort)
        print("images_per_cohort",images_per_cohort)
        duration = ex_cohort[0]["lifespan"]

        d = int(np.ceil(duration*1000)) # 1ms bins
        logger.info(f"ncohorts: {ncohorts}")
        # import pdb; pdb.set_trace()
        
        logger.info(f"nclasses: {nclasses}")
        if nclasses < 256:
            class_dtype = np.dtype('uint8')
        else: 
            class_dtype = np.dtype('uint16')
            
        class_resolver_func = lambda c: class_resolver[str(c)]

        # determine shape
        experiments = glia.flatten_group_dict(image_responses)
        nE = len(experiments)
        d = int(np.ceil(duration*1000)) # 1ms bins
        data_shape = (nE,d,Unit.nrow,Unit.ncol,Unit.nunit)


        print(f"writing to {name}.h5 with zstd compression...")
        data = h5.create_carray("/", "data",
            tables.Atom.from_dtype(np.dtype('uint8')), shape=data_shape,
            filters=dset_filter)
        target = h5.create_carray("/", "target",
            tables.Atom.from_dtype(class_dtype), shape=(nE,),
            filters=dset_filter)

        glia.experiments_to_h5(experiments,
            data, target,
            partial(get_image_class_from_stim, class_resolver=class_resolver_func),
            append, class_dtype=class_dtype)


def save_checkerboard_npz(units, stimulus_list, name, append, group_by, quad=False):
    "Psychophysics discrimination checkerboard 0.2.0"
    print("Saving checkerboard NPZ file.")

    get_checkers = glia.compose(
        partial(glia.create_experiments, progress=True,append_lifespan=append,
            # stimulus_list=stimulus_list,append_lifespan=0.5),
            stimulus_list=stimulus_list),
        partial(glia.group_by,
                key=lambda x: x["metadata"]["group"]),
        glia.group_dict_to_list,
        glia.f_filter(group_contains_checkerboard),
        glia.f_map(lambda x: [x[1],x[2]]),
        glia.f_map(glia.merge_experiments),
        partial(glia.group_by,
                key=group_by),
        glia.f_map(partial(glia.group_by,
                key=lambda x: x["size"])),
        glia.f_map(glia.f_map(partial(glia.group_by,
                key=lambda x: x["metadata"]["cohort"])))
    )
    checkers = get_checkers(units)

    max_duration = 0.0
    for condition, sizes in checkers.items():
        for size, cohorts in sizes.items():
            for cohort, experiments in cohorts.items():
                max_duration = max(max_duration,
                    experiments[0]['lifespan'])
    max_duration += append

    conditions = sorted(list(checkers.keys()))
    print("Conditions:", name, conditions)
    nconditions = len(conditions)
    example_condition = glia.get_value(checkers)
    sizes = sorted(list(example_condition.keys()))
    nsizes = len(sizes)

    example_size = glia.get_value(example_condition)
    ncohorts = len(example_size)
    # print(list(checkers.values()))
    d = int(np.ceil(max_duration*1000)) # 1ms bins

    tvt = glia.tvt_by_percentage(ncohorts,60,40,0)
    logger.info(f"{tvt}, {ncohorts}")
    # (TODO?) 2 dims for first checkerboard and second checkerboard
    # 4 per cohort
    if quad:
        ntraining = tvt.training*4
        nvalid = tvt.validation*4
    else:
        ntraining = tvt.training*2
        nvalid = tvt.validation*2

    training_data = np.full((nconditions,nsizes,
        ntraining,d,Unit.nrow,Unit.ncol,Unit.nunit),0,dtype='int8')
    training_target = np.full((nconditions,nsizes,
        ntraining),0,dtype='int8')
    validation_data = np.full((nconditions,nsizes,
        nvalid,d,Unit.nrow,Unit.ncol,Unit.nunit),0,dtype='int8')
    validation_target = np.full((nconditions,nsizes,
        nvalid),0,dtype='int8')
    # test_data = np.full((nsizes,tvt.test,d,nunits),0,dtype='int8')
    # test_target = np.full((nsizes,tvt.test),0,dtype='int8')

    if quad:
        get_class = get_checker_quad_discrimination_class
    else:
        get_class = get_checker_discrimination_class
    condition_map = {c: i for i,c in enumerate(conditions)}
    size_map = {s: i for i,s in enumerate(sizes)}
    for condition, sizes in checkers.items():
        for size, cohorts in sizes.items():
            X = glia.f_split_dict(tvt)(cohorts)

            td, tt = glia.experiments_to_ndarrays(glia.training_cohorts(X),
                        get_class, append)
            logger.info(td.shape)
            missing_duration = d - td.shape[1]
            pad_td = np.pad(td,
                ((0,0),(0,missing_duration),(0,0),(0,0),(0,0)),
                mode='constant')
            condition_index = condition_map[condition]
            size_index = size_map[size]
            training_data[condition_index, size_index] = pad_td
            training_target[condition_index, size_index] = tt

            td, tt = glia.experiments_to_ndarrays(glia.validation_cohorts(X),
                        get_class, append)
            pad_td = np.pad(td,
                ((0,0),(0,missing_duration),(0,0),(0,0),(0,0)),
                mode='constant')
            validation_data[condition_index, size_index] = pad_td
            validation_target[condition_index, size_index] = tt

    print('saving to ',name)
    np.savez(name, training_data=training_data, training_target=training_target,
         validation_data=validation_data, validation_target=validation_target)
          # test_data=test_data, test_target=test_target)

# TODO refactor all of these functions. DRY!!
def save_checkerboard_flicker_npz(units, stimulus_list, name, append, group_by, quad=False):
    "Psychophysics discrimination checkerboard 0.2.0"
    print("Saving checkerboard NPZ file.")

    get_checkers = glia.compose(
        partial(glia.create_experiments, progress=True, append_lifespan=append,
            # stimulus_list=stimulus_list,append_lifespan=0.5),
            stimulus_list=stimulus_list),
        partial(glia.group_by,
                key=lambda x: x["metadata"]["group"]),
        glia.group_dict_to_list,
        glia.f_filter(group_contains_checkerboard),
        glia.f_map(glia.f_filter(lambda x: x['stimulusType']=='CHECKERBOARD')),
        glia.f_map(glia.merge_experiments),
        partial(glia.group_by,
                key=group_by),
        glia.f_map(partial(glia.group_by,
                key=lambda x: x["size"])),
        glia.f_map(glia.f_map(partial(glia.group_by,
                key=lambda x: x["metadata"]["cohort"])))
    )
    checkers = get_checkers(units)

    max_duration = 0.0
    for condition, sizes in checkers.items():
        for size, cohorts in sizes.items():
            for cohort, experiments in cohorts.items():
                max_duration = max(max_duration,
                    experiments[0]['lifespan'])
    max_duration += append
    print(f"max_duration: {max_duration}")

    conditions = sorted(list(checkers.keys()))
    print("Conditions:", name, conditions)
    nconditions = len(conditions)
    example_condition = glia.get_value(checkers)
    sizes = sorted(list(example_condition.keys()))
    nsizes = len(sizes)
    # TODO remove
    if max_duration<9:
        print(example_condition)

    example_size = glia.get_value(example_condition)
    ncohorts = len(example_size)
    # print(list(checkers.values()))
    d = int(np.ceil(max_duration*1000)) # 1ms bins

    tvt = glia.tvt_by_percentage(ncohorts,60,40,0)
    logger.info(f"{tvt}, {ncohorts}")
    # (TODO?) 2 dims for first checkerboard and second checkerboard
    # 4 per cohort
    if quad:
        ntraining = tvt.training*4
        nvalid = tvt.validation*4
    else:
        ntraining = tvt.training*2
        nvalid = tvt.validation*2

    training_data = np.full((nconditions,nsizes,
        ntraining,d,Unit.nrow,Unit.ncol,Unit.nunit),0,dtype='int8')
    training_target = np.full((nconditions,nsizes,
        ntraining),0,dtype='int8')
    validation_data = np.full((nconditions,nsizes,
        nvalid,d,Unit.nrow,Unit.ncol,Unit.nunit),0,dtype='int8')
    validation_target = np.full((nconditions,nsizes,
        nvalid),0,dtype='int8')
    # test_data = np.full((nsizes,tvt.test,d,nunits),0,dtype='int8')
    # test_target = np.full((nsizes,tvt.test),0,dtype='int8')

    if quad:
        get_class = get_checker_quad_discrimination_class
    else:
        get_class = get_checker_discrimination_class
    condition_map = {c: i for i,c in enumerate(conditions)}
    size_map = {s: i for i,s in enumerate(sizes)}
    for condition, sizes in checkers.items():
        for size, cohorts in sizes.items():
            X = glia.f_split_dict(tvt)(cohorts)

            td, tt = glia.experiments_to_ndarrays(glia.training_cohorts(X),
                        get_class, append)
            logger.info(td.shape)
            missing_duration = d - td.shape[1]
            pad_td = np.pad(td,
                ((0,0),(0,missing_duration),(0,0),(0,0),(0,0)),
                mode='constant')
            condition_index = condition_map[condition]
            size_index = size_map[size]
            training_data[condition_index, size_index] = pad_td
            training_target[condition_index, size_index] = tt

            td, tt = glia.experiments_to_ndarrays(glia.validation_cohorts(X),
                        get_class, append)
            pad_td = np.pad(td,
                ((0,0),(0,missing_duration),(0,0),(0,0),(0,0)),
                mode='constant')
            validation_data[condition_index, size_index] = pad_td
            validation_target[condition_index, size_index] = tt

    print('saving to ',name)
    np.savez(name, training_data=training_data, training_target=training_target,
         validation_data=validation_data, validation_target=validation_target)
          # test_data=test_data, test_target=test_target)



def save_grating_npz(units, stimulus_list, name, append, group_by, sinusoid=False):
    "Psychophysics discrimination grating 0.2.0"
    print("Saving grating NPZ file.")
    if sinusoid:
        stimulus_type = "SINUSOIDAL_GRATING"
    else:
        stimulus_type = 'GRATING'
    get_gratings = glia.compose(
            partial(glia.create_experiments,
                stimulus_list=stimulus_list, append_lifespan=append),
            glia.f_filter(lambda x: x['stimulusType']==stimulus_type),
            partial(glia.group_by,
                    key=group_by),
            glia.f_map(partial(glia.group_by,
                    key=lambda x: x["width"])),
            glia.f_map(glia.f_map(partial(glia.group_by,
                    key=lambda x: x["metadata"]["cohort"])))
        )
    gratings = get_gratings(units)

    max_duration = 0.0
    for condition, sizes in gratings.items():
        for size, cohorts in sizes.items():
            for cohort, experiments in cohorts.items():
                max_duration = max(max_duration,
                    experiments[0]['lifespan'])
    max_duration += append

    conditions = sorted(list(gratings.keys()))
    print("Conditions:", name, conditions)
    nconditions = len(conditions)
    example_condition = glia.get_value(gratings)
    sizes = sorted(list(example_condition.keys()))
    print("Sizes:", sizes)
    nsizes = len(sizes)

    example_size = glia.get_value(example_condition)
    ncohorts = len(example_size)
    # print(list(gratings.values()))
    d = int(np.ceil(max_duration*1000)) # 1ms bins
    tvt = glia.tvt_by_percentage(ncohorts,60,40,0)
    # 2 per cohort
    training_data = np.full((nconditions,nsizes,
        tvt.training*2,d,Unit.nrow,Unit.ncol,Unit.nunit),0,dtype='int8')
    training_target = np.full((nconditions,nsizes,
        tvt.training*2),0,dtype='int8')
    validation_data = np.full((nconditions,nsizes,
        tvt.validation*2,d,Unit.nrow,Unit.ncol,Unit.nunit),0,dtype='int8')
    validation_target = np.full((nconditions,nsizes,
        tvt.validation*2),0,dtype='int8')

    condition_map = {c: i for i,c in enumerate(conditions)}
    size_map = {s: i for i,s in enumerate(sizes)}
    for condition, sizes in gratings.items():
        for size, cohorts in sizes.items():
            X = glia.f_split_dict(tvt)(cohorts)

            td, tt = glia.experiments_to_ndarrays(glia.training_cohorts(X),
                        get_grating_class_from_stim, append)
            missing_duration = d - td.shape[1]
            pad_td = np.pad(td,
                ((0,0),(0,missing_duration),(0,0),(0,0),(0,0)),
                mode='constant')
            condition_index = condition_map[condition]
            size_index = size_map[size]
            training_data[condition_index, size_index] = pad_td
            training_target[condition_index, size_index] = tt

            td, tt = glia.experiments_to_ndarrays(glia.validation_cohorts(X),
                        get_grating_class_from_stim, append)
            pad_td = np.pad(td,
                ((0,0),(0,missing_duration),(0,0),(0,0),(0,0)),
                mode='constant')
            validation_data[condition_index, size_index] = pad_td
            validation_target[condition_index, size_index] = tt

    print('saving to ',name)
    np.savez(name, training_data=training_data, training_target=training_target,
         validation_data=validation_data, validation_target=validation_target)
          # test_data=test_data, test_target=test_target)
