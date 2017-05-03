import glia
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from warnings import warn
import logging
from copy import deepcopy
from sklearn import svm, metrics

logger = logging.getLogger('glia')

def plot_psth(fig, axis_gen, data,prepend_start_time=1,append_lifespan=1,bin_width=0.1):
    for s,spike_train in data.items():
        ax = next(ax_gen)
        stimulus = eval(s)
        lifespan = stimulus["lifespan"]/120
        # if lifespan > 5:
        #     print("skipping stimulus longer than 5 seconds")
        #     return None
        duration = prepend_start_time+lifespan+append_lifespan
        ax.hist(spike_train,bins=np.arange(0,duration,bin_width),linewidth=None,ec="none")
        ax.axvspan(0,prepend_start_time,facecolor="gray", edgecolor="none", alpha=0.1)
        ax.axvspan(prepend_start_time+lifespan,duration,facecolor="gray", edgecolor="none", alpha=0.1)
        ax.set_title("Post-stimulus Time Histogram of SOLID")
        ax.set_xlabel("relative time (s)")
        ax.set_ylabel("spike count")


def plot_spike_trains(fig, axis_gen, data,prepend_start_time=1,append_lifespan=1):
    colors = set()
    for e in data:
        color = e["stimulus"]["backgroundColor"]
        colors.add(color)

    sorted_colors = sorted(list(colors),reverse=True)

    for color in sorted_colors:
        ax = next(axis_gen)
        filtered_data = list(filter(lambda x: x["stimulus"]["backgroundColor"]==color,
            data))
        trial = 0

        for v in filtered_data:
            # print(type(v))
            stimulus, spike_train = (v["stimulus"], v["spikes"])
            lifespan = stimulus['lifespan'] / 120
            if lifespan > 20:
                print("skipping stimulus longer than 20 seconds")
                continue
            if spike_train.size>0:
                glia.draw_spikes(ax, spike_train, ymin=trial+0.3,ymax=trial+1)
            
            stimulus_end = prepend_start_time + lifespan
            duration = stimulus_end + append_lifespan
            ax.fill([0,prepend_start_time,prepend_start_time,0],
                    [trial,trial,trial+1,trial+1],
                    facecolor="gray", edgecolor="none", alpha=0.1)
            ax.fill([stimulus_end,duration,duration,stimulus_end],
                    [trial,trial,trial+1,trial+1],
                    facecolor="gray", edgecolor="none", alpha=0.1)
            trial += 1
            
        ax.set_title("Unit spike train per SOLID ({})".format(color))
        ax.set_xlabel("time (s)")
        ax.set_ylabel("trials")

def plot_spike_trains_vFail(fig, axis_gen, data):
    # remove this function asap
    ax = next(axis_gen)

    trial = 0
    # forgot to change group id, so iterate triplets
    for i in range(0, len(data)-1, 3):
        # x offset for row
        offset = 0

        for v in data[i:i+3]:
            stimulus, spike_train = (v["stimulus"], v["spikes"])
            lifespan = stimulus['lifespan'] / 120
            end_time = lifespan + offset
            if lifespan > 20:
                logger.warning("skipping stimulus longer than 20 seconds")
                continue
            if spike_train.size>0:
                glia.draw_spikes(ax, spike_train+offset, ymin=trial+0.3,
                    ymax=trial+1)
            
            ax.fill([offset,end_time,end_time,offset],
                    [trial,trial,trial+1,trial+1],
                    facecolor=stimulus["backgroundColor"], edgecolor="none", alpha=0.1)
            offset = end_time
        trial += 1

        
    ax.set_title("Unit spike train per SOLID group")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("trials")

get_lifespan = lambda e: e["stimulus"]["lifespan"]

def plot_spike_train_triplet(fig, axis_gen, data):
    # 
    ax = next(axis_gen)
    trial = 0
    # hardcoded 2 must correspond to pivot
    longest_group = max(map(lambda x: get_lifespan(x[1]),
        data))/120 + 2
    for group in data:
        # x offset for row
        offset = 0
        for i,v in enumerate(group):
            stimulus, spike_train = (v["stimulus"], v["spikes"])
            lifespan = stimulus['lifespan'] / 120

            if i==0:
                # only show last second before middle stimuli
                pivot = lifespan-1
                if pivot<0:
                    logger.error("first stimuli is too short--must be >1s")
                    pivot = 0
                spike_train = spike_train[spike_train>pivot] - pivot
                end_time = 1
            elif i==1:
                end_time = lifespan + offset
            elif i==2:
                # only show last second before middle stimuli
                spike_train = spike_train[spike_train<1]
                end_time = 1 + offset

            if lifespan > 20:
                logger.warning("skipping stimulus longer than 20 seconds")
                continue
            if spike_train.size>0:
                glia.draw_spikes(ax, spike_train+offset, ymin=trial+0.3,
                    ymax=trial+1)
            
            ax.fill([offset,end_time,end_time,offset],
                    [trial,trial,trial+1,trial+1],
                    facecolor=stimulus["backgroundColor"],
                    edgecolor="none", alpha=0.1)
            offset = end_time

        if offset<longest_group:
            ax.fill([offset,longest_group,longest_group,offset],
                    [trial,trial,trial+1,trial+1],
                    facecolor="black",
                    edgecolor="none", alpha=1)

        trial += 1

        
    ax.set_title("Unit spike train per SOLID group")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("trials")
    ax.set_xlim((0,longest_group))
    ax.set_ylim((0,trial))

def save_unit_psth(units, stimulus_list, c_unit_fig, c_add_retina_figure, prepend, append):
    print("Creating solid unit PSTH")

    get_psth = glia.compose(
        glia.f_create_experiments(stimulus_list,prepend_start_time=prepend,append_lifespan=append),
        glia.f_has_stimulus_type(["SOLID"]),
        glia.f_group_by_stimulus(),
        glia.concatenate_by_stimulus
    )
    psth = glia.apply_pipeline(get_psth,units, progress=True)
    plot_function = partial(plot_psth,prepend_start_time=prepend,append_lifespan=append)
    result = glia.plot_units(partial(plot_function,bin_width=0.01),psth,ax_xsize=10, ax_ysize=5)
    c_unit_fig(result)
    glia.close_figs([fig for the_id,fig in result])


def save_unit_spike_trains(units, stimulus_list, c_unit_fig, c_add_retina_figure, prepend, append):
    print("Creating solid unit spike trains")
    
    get_solid = glia.compose(
        glia.f_create_experiments(stimulus_list,prepend_start_time=prepend,append_lifespan=append),
        glia.f_has_stimulus_type(["SOLID"]),
    )
    response = glia.apply_pipeline(get_solid,units, progress=True)
    plot_function = partial(plot_spike_trains,prepend_start_time=prepend,append_lifespan=append)
    result = glia.plot_units(plot_function,response,ncols=1,ax_xsize=10, ax_ysize=5)
    c_unit_fig(result)
    glia.close_figs([fig for the_id,fig in result])

def filter_time(l):
    return list(filter(lambda x: x["stimulus"]["lifespan"]==60, l))

def save_integrity_chart(units, stimulus_list, c_unit_fig, c_add_retina_figure):
    print("Creating integrity chart")
    
    get_solid = glia.compose(
        glia.f_create_experiments(stimulus_list,prepend_start_time=1,append_lifespan=2),
        glia.f_has_stimulus_type(["SOLID"]),
        filter_time
    )
    response = glia.apply_pipeline(get_solid,units, progress=True)
    plot_function = partial(plot_spike_trains,prepend_start_time=1,append_lifespan=2)
    glia.plot_units(plot_function,c_unit_fig,response,ncols=1,ax_xsize=10, ax_ysize=5,
                             figure_title="Integrity Test (5 Minute Spacing)")


def integrity_spike_counts(cohort):
    "Takes a list of integrity groups (of 3 experiments), and return List[Int]"
    dark_experiments = []
    on_experiments = []
    off_experiments = []

    for integrity_group in cohort:
        dark_experiment = integrity_group[0]
        on_experiment = integrity_group[1]
        off_experiment = integrity_group[2]
        dark_lifespan = dark_experiment["stimulus"]['lifespan']
        on_lifespan = on_experiment["stimulus"]["lifespan"]
        off_lifespan = off_experiment["stimulus"]["lifespan"]

        if dark_lifespan >=120 and on_lifespan==60 and off_lifespan >=60:
            d = dark_experiment["spikes"]
            dark_experiments.append(d[(d>0.5) & (d<=1)].size)
            f = off_experiment['spikes']
            off_experiments.append(f[f<=0.5].size)
            on_experiments.append(on_experiment['spikes'].size)
        elif dark_lifespan == on_lifespan and on_lifespan==off_lifespan:
            on_experiments.append(on_experiment['spikes'].size)
            off_experiments.append(off_experiment['spikes'].size)
            dark_experiments.append(dark_experiment['spikes'].size)
        else:
            logger.error("unexpected integrity group lifespans")
            print(on_experiment["stimulus"]['lifespan'],
                 off_experiment["stimulus"]['lifespan'])
            raise ValueError
    return (dark_experiments, on_experiments, off_experiments)

def ideal_classification_accuracy(cohort):
    """Intended for Integrity, sorted by cohort. Return classification accuracy

    for dark vs on and dark vs off"""
    dark_experiments, on_experiments, off_experiments = integrity_spike_counts(cohort)

    on_X = np.array(dark_experiments + on_experiments).reshape((-1,1))
#     print(on_X.shape)
    on_Y = np.hstack([np.full(len(dark_experiments), 0,dtype='int8'),
                  np.full(len(on_experiments),1,dtype='int8')])
    on = svm.SVC(kernel='linear')
    on.fit(on_X, on_Y)
    on_predicted = on.predict(on_X)

    off_X = np.array(dark_experiments + off_experiments).reshape((-1,1))
    off_Y = np.hstack([np.full(len(dark_experiments), 0,dtype='int8'),
                  np.full(len(off_experiments),1,dtype='int8')])
    off = svm.SVC(kernel='linear')
    off.fit(off_X, off_Y)
    off_predicted = on.predict(off_X)
    return {"on": float(metrics.accuracy_score(on_Y, on_predicted)),
                         "off": float(metrics.accuracy_score(off_Y, off_predicted))}
#         quality[unit] = {"on": metrics.accuracy_score(on_Y, on_predicted),
#                          "off": metrics.accuracy_score(off_Y, off_predicted)}
#     return quality

def plot_units_accuracy(units_accuracy):
    on_dist = [v['on'] for v in units_accuracy.values()]
    off_dist = [v['off'] for v in units_accuracy.values()]
    bins = np.linspace(0.5,1,11)
    fig, ax = plt.subplots(2)
    ax[0].hist(on_dist, bins=bins)
    ax[0].set_title("Classification accuracy by ON response")
    ax[0].set_xlim(.5,1)
    ax[0].set_ylabel("Accuracy")
    ax[0].set_ylabel("Number of units")
    ax[1].hist(off_dist, bins=bins)
    ax[1].set_title("Classification accuracy by OFF response")
    ax[1].set_xlim(.5,1)
    ax[1].set_ylabel("Number of units")
    ax[1].set_xlabel("Accuracy")
    fig.tight_layout()

    return fig

def save_integrity_chart_v2(units, stimulus_list, c_unit_fig, c_add_retina_figure):
    print("Creating integrity chart")
    get_solid = glia.compose(
        glia.f_create_experiments(stimulus_list),
        filter_integrity,
        partial(glia.group_by,
            key=lambda x: x["stimulus"]["metadata"]["group"]),
        glia.group_dict_to_list,
        partial(sorted,key=lambda x: x[0]["stimulus"]["stimulusIndex"])
        )

    response = glia.apply_pipeline(get_solid,units, progress=True)
    plot_function = partial(glia.raster_group)
    # c = partial(c_unit_fig,"kinetics-{}".format(i))

    glia.plot_units(plot_function,c_unit_fig,response,ncols=1,ax_xsize=10, ax_ysize=5,
                             figure_title="Integrity Test (5 Minute Spacing)")

    units_accuracy = glia.pmap(ideal_classification_accuracy, response)
    c_add_retina_figure("Integrity Accuracy",plot_units_accuracy(units_accuracy))

def save_unit_wedges(units, stimulus_list, c_unit_fig, c_add_retina_figure, prepend, append):
    print("Creating solid unit wedges")
    
    get_solid = glia.compose(
        glia.f_create_experiments(stimulus_list,prepend_start_time=prepend,append_lifespan=append),
        glia.f_has_stimulus_type(["SOLID"]),
        partial(sorted,key=lambda x: x["stimulus"]["lifespan"])
    )
    response = glia.apply_pipeline(get_solid,units, progress=True)

    colors = set()
    for solid in glia.get_unit(response)[1]:
        colors.add(solid["stimulus"]["backgroundColor"])
    ncolors = len(colors)

    plot_function = partial(plot_spike_trains,prepend_start_time=prepend,
        append_lifespan=append)
    glia.plot_units(plot_function,c_unit_fig,response,nplots=ncolors,
        ncols=min(ncolors,5),ax_xsize=10, ax_ysize=5)

def filter_integrity(l):
    return list(filter(lambda x: "label" in x["stimulus"]["metadata"] and \
        x["stimulus"]["metadata"]["label"]=="integrity", l))

def integrity_fix_hack(listlistE):
    "Hack for mistake where all integrity in same group."
    old = listlistE[0]
    new = []
    for i in range(0,len(old),3):
        new.append(old[i:i+3])
    return new

def save_integrity_chart_vFail(units, stimulus_list, c_unit_fig, c_add_retina_figure):
    print("Creating integrity chart")
    get_solid = glia.compose(
        glia.f_create_experiments(stimulus_list),
        filter_integrity,
        partial(glia.group_by,
            key=lambda x: x["stimulus"]["metadata"]["group"]),
        glia.group_dict_to_list,
        integrity_fix_hack,
        partial(sorted,key=lambda x: x[0]["stimulus"]["stimulusIndex"])
        )

    response = glia.apply_pipeline(get_solid,units, progress=True)
    plot_function = partial(glia.raster_group)
    # c = partial(c_unit_fig,"kinetics-{}".format(i))

    glia.plot_units(plot_function,c_unit_fig,response,ncols=1,ax_xsize=10, ax_ysize=5,
                             figure_title="Integrity Test (5 Minute Spacing)")

    units_accuracy = glia.pmap(ideal_classification_accuracy, response)
    c_add_retina_figure("Integrity Accuracy",plot_units_accuracy(units_accuracy))

# def save_integrity_chart_vFail(units, stimulus_list, c_unit_fig, c_add_retina_figure):
#     print("Creating integrity chart")
    
#     get_solid = glia.compose(
#         glia.f_create_experiments(stimulus_list),
#         filter_integrity
#     )
#     response = glia.apply_pipeline(get_solid,units, progress=True)
#     plot_function = partial(plot_spike_trains_vFail)
#     glia.plot_units(plot_function,c_unit_fig,response,ncols=1,ax_xsize=10, ax_ysize=5,
#                              figure_title="Integrity Test (5 Minute Spacing)")

def save_unit_wedges_v2(units, stimulus_list, c_unit_fig, c_add_retina_figure):
    print("Creating solid unit wedges")

    get_solid = glia.compose(
        glia.f_create_experiments(stimulus_list),
        glia.f_has_stimulus_type(["SOLID","WAIT"]),
        partial(glia.group_by,
            key=lambda x: x["stimulus"]["metadata"]["group"]),
        glia.group_dict_to_list,
        partial(sorted,key=lambda x: get_lifespan(x[1]))
    )
    response = glia.apply_pipeline(get_solid,units, progress=True)

    glia.plot_units(plot_spike_train_triplet,c_unit_fig,response,nplots=1,
        ncols=1,ax_xsize=10, ax_ysize=5)

def save_unit_kinetics_v1(units, stimulus_list, c_unit_fig, c_add_retina_figure):
    print("Creating solid unit kinetics")
    

    for i in range(5):
        s = i*150
        e = (i+1)*150
        get_solid = glia.compose(
            glia.f_create_experiments(stimulus_list),
            lambda x: x[s:e],
            partial(glia.group_by,
                key=lambda x: x["stimulus"]["metadata"]["group"]),
            glia.group_dict_to_list,
            partial(sorted,key=lambda x: get_lifespan(x[2]))
        )
        response = glia.apply_pipeline(get_solid,units, progress=True)
        c = partial(c_unit_fig,"kinetics-{}".format(i))
        glia.plot_units(glia.raster_group,c,response,nplots=1,
            ncols=1,ax_xsize=10, ax_ysize=5)

def save_unit_kinetics(units, stimulus_list, c_unit_fig, c_add_retina_figure):
    print("Creating solid unit kinetics")
    
    get_solid = glia.compose(
        glia.f_create_experiments(stimulus_list),
        partial(glia.group_by,
            key=lambda x: x["stimulus"]["metadata"]["group"]),
        glia.group_dict_to_list,
        partial(sorted,key=lambda x: get_lifespan(x[2]))
    )
    response = glia.apply_pipeline(get_solid,units, progress=True)

    # glia.plot_units(plot_group_spike_train,c_unit_fig,response,nplots=1,
    #     ncols=1,ax_xsize=10, ax_ysize=5)
    glia.plot_units(glia.raster_group,c_unit_fig,response,nplots=1,
        ncols=1,ax_xsize=10, ax_ysize=5)
