import glia
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from warnings import warn
import logging
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
    psth = glia.apply_pipeline(get_psth,units)
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
    response = glia.apply_pipeline(get_solid,units)
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
    response = glia.apply_pipeline(get_solid,units)
    plot_function = partial(plot_spike_trains,prepend_start_time=1,append_lifespan=2)
    glia.plot_units(plot_function,c_unit_fig,response,ncols=1,ax_xsize=10, ax_ysize=5,
                             figure_title="Integrity Test (5 Minute Spacing)")

def save_unit_wedges(units, stimulus_list, c_unit_fig, c_add_retina_figure, prepend, append):
    print("Creating solid unit wedges")
    
    get_solid = glia.compose(
        glia.f_create_experiments(stimulus_list,prepend_start_time=prepend,append_lifespan=append),
        glia.f_has_stimulus_type(["SOLID"]),
        partial(sorted,key=lambda x: x["stimulus"]["lifespan"])
    )
    response = glia.apply_pipeline(get_solid,units)

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


def save_integrity_chart_vFail(units, stimulus_list, c_unit_fig, c_add_retina_figure):
    print("Creating integrity chart")
    
    get_solid = glia.compose(
        glia.f_create_experiments(stimulus_list),
        filter_integrity
    )
    response = glia.apply_pipeline(get_solid,units)
    plot_function = partial(plot_spike_trains_vFail)
    glia.plot_units(plot_function,c_unit_fig,response,ncols=1,ax_xsize=10, ax_ysize=5,
                             figure_title="Integrity Test (5 Minute Spacing)")

def flatten_groups(dictionary):
    to_return = []
    for k,v in dictionary.items():
        if v==None:
            logger.warning("got a value of None for group {}".format(
                k))
        else:
            to_return.append(v)
    return to_return

def save_unit_wedges_v2(units, stimulus_list, c_unit_fig, c_add_retina_figure):
    print("Creating solid unit wedges")
    
    group_length = lambda x: sum(map(get_lifespan, x))

    get_solid = glia.compose(
        glia.f_create_experiments(stimulus_list),
        glia.f_has_stimulus_type(["SOLID"]),
        partial(glia.group_by,
            key=lambda x: x["stimulus"]["metadata"]["group"]),
        flatten_groups,
        # lambda x: print('yo',x[1][1]["stimulus"]["lifespan"]),

        partial(sorted,key=lambda x: get_lifespan(x[1]))
    )
    response = glia.apply_pipeline(get_solid,units)

    glia.plot_units(plot_spike_train_triplet,c_unit_fig,response,nplots=1,
        ncols=1,ax_xsize=10, ax_ysize=5)
