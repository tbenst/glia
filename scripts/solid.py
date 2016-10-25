import glia
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from warnings import warn

def plot_psth(ax_gen,data,prepend_start_time=1,append_lifespan=1,bin_width=0.1):
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


def plot_spike_trains(axis_gen,data,prepend_start_time=1,append_lifespan=1):
    ax = next(axis_gen)
    trial = 0
    for v in data:
        # print(type(v))
        stimulus, spike_train = (v["stimulus"], v["spikes"])
        lifespan = stimulus['lifespan'] / 120
        if lifespan > 5:
            print("skipping stimulus longer than 5 seconds")
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
        
    ax.set_title("Unit spike train per SOLID")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("trial # (lower is earlier)")


def save_unit_psth(units, stimulus_list, c_add_unit_figures, c_add_retina_figure):
    print("Creating solid unit PSTH")

    get_psth = glia.compose(
        glia.f_create_experiments(stimulus_list,prepend_start_time=1,append_lifespan=1),
        glia.f_has_stimulus_type(["SOLID"]),
        glia.f_group_by_stimulus(),
        glia.concatenate_by_stimulus
    )
    psth = glia.apply_pipeline(get_psth,units)
    result = glia.plot_units(partial(plot_psth,bin_width=0.01),psth,ax_xsize=10, ax_ysize=5)
    c_add_unit_figures(result)
    glia.close_figs([fig for the_id,fig in result])


def save_unit_spike_trains(units, stimulus_list, c_add_unit_figures, c_add_retina_figure):
    print("Creating solid unit spike trains")
    
    get_solid = glia.compose(
        glia.f_create_experiments(stimulus_list,prepend_start_time=1,append_lifespan=1),
        glia.f_has_stimulus_type(["SOLID"]),
    )
    response = glia.apply_pipeline(get_solid,units)
    result = glia.plot_units(plot_spike_trains,response,ncols=1,ax_xsize=10, ax_ysize=5)
    c_add_unit_figures(result)
    glia.close_figs([fig for the_id,fig in result])
