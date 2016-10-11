import glia
import matplotlib.pyplot as plt
import numpy as np

def f_plot_psth(prepend_start_time,append_lifespan,bin_width):
    def plot(ax,unit_id,value):
        stimulus = eval(value[0])
        spike_train = value[1]
        lifespan = stimulus["lifespan"]/120
        if lifespan > 5:
            print("skipping stimulus longer than 5 seconds")
            return None
        duration = prepend_start_time+lifespan+append_lifespan
        ax.hist(spike_train,bins=np.arange(0,duration,bin_width),linewidth=None,ec="none")
        ax.axvspan(0,prepend_start_time,facecolor="gray", edgecolor="none", alpha=0.2)
        ax.axvspan(prepend_start_time+lifespan,duration,facecolor="gray", edgecolor="none", alpha=0.2)
        ax.set_title(unit_id)
        ax.set_xlabel("relative time (s)")
        ax.set_ylabel("spike count")
    return plot

def f_plot_spike_trains(prepend_start_time,append_lifespan):
    # plot_a_roster of spikes relative to stimulus on time
    def plot(axis_gen,unit_id,value):
        ax = next(axis_gen)
        trial = 0
        for v in value:
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
                    facecolor="gray", edgecolor="none", alpha=0.2)
            ax.fill([stimulus_end,duration,duration,stimulus_end],
                    [trial,trial,trial+1,trial+1],
                    facecolor="gray", edgecolor="none", alpha=0.2)
            trial += 1
            
        ax.set_title(unit_id)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("trial # (lower is earlier)")
    return plot


def save_unit_psth(output_file, units, stimulus_list):
    print("Creating solid unit PSTH")
    # for stimulus in stimulus_list:

    get_psth = glia.compose(
        glia.f_create_experiments(stimulus_list,prepend_start_time=1,append_lifespan=1),
        glia.f_has_stimulus_type(["SOLID"]),
        glia.f_group_by_stimulus(),
        glia.concatenate_by_stimulus
    )
    psth = glia.apply_pipeline(get_psth,units)
    fig_psth = glia.plot_each_by_unit(f_plot_psth(1,1,0.01),psth,ax_xsize=10, ax_ysize=5)
    fig_psth.savefig(output_file)


def save_unit_spike_trains(output_file, units, stimulus_list):
    print("Creating solid unit spike trains")
    get_solid = glia.compose(
        glia.f_create_experiments(stimulus_list,prepend_start_time=1,append_lifespan=1),
        glia.f_has_stimulus_type(["SOLID"]),
    )
    response = glia.apply_pipeline(get_solid,units)
    figures = glia.plot_units(f_plot_spike_trains(1,1),response,ncols=2,ax_xsize=10, ax_ysize=5)
    filenames = [#TODO]
    save_figs(figures, output_file)
