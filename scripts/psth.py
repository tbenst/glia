import glia
import matplotlib.pyplot as plt
import numpy as np

def f_plot_solid_psth(on_time,off_time,duration,bin_width):
    def plot(ax,unit_id,value):
        number_of_stimuli = len(value.values())
        color=iter(plt.cm.rainbow(np.linspace(0,1,number_of_stimuli)))
        for stimulus, spike_train in value.items():
            ax.hist(spike_train,bins=np.arange(0,duration,bin_width),linewidth=None,ec="none",color=next(color))
        ax.axvspan(0,on_time,facecolor="gray", edgecolor="none", alpha=0.2)
        ax.axvspan(off_time,duration,facecolor="gray", edgecolor="none", alpha=0.2)
        ax.set_title(unit_id)
        ax.set_xlabel("relative time (s)")
        ax.set_ylabel("spike count")
    return plot

def f_plot_solid_spike_trains(on_time,off_time,duration,bin_width):
    def plot(ax,unit_id,value):
        for stimulus, spike_trains in value.items():
            # this does not handle the case of multiple stimuli
            for i, spike_train in enumerate(spike_trains):
                if spike_train.size>0:
                    glia.draw_spikes(ax, spike_train, ymin=i+0.3,ymax=i+1)
        ax.axvspan(0,on_time,facecolor="gray", edgecolor="none", alpha=0.2)
        ax.axvspan(off_time,duration,facecolor="gray", edgecolor="none", alpha=0.2)
        ax.set_title(unit_id)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("trial # (lower is earlier)")
    return plot


def save_solid_unit_psth(output_file, units, stimulus_list):
	print("Creating solid unit PSTH")
	get_solid_psth = glia.compose(
	    glia.f_create_experiments(stimulus_list,prepend_start_time=1,append_start_time=3),
	    glia.f_has_stimulus_type(["SOLID"]),
	    glia.f_group_by_stimulus(),
	    glia.concatenate_by_stimulus
	)
	solid_psth = glia.apply_pipeline(get_solid_psth,units)
	fig_psth = glia.plot_units(f_plot_solid_psth(1,2,3,0.01),solid_psth,ncols=2,ax_xsize=10, ax_ysize=5, xlim=(0,3))
	fig_psth.savefig(output_file)


def save_solid_unit_spike_trains(output_file, units, stimulus_list):
	print("Creating solid unit spike trains")
	get_solid = glia.compose(
	    glia.f_create_experiments(stimulus_list,prepend_start_time=1,append_start_time=3),
	    glia.f_has_stimulus_type(["SOLID"]),
	    glia.f_group_by_stimulus(),
	)
	solid_response = glia.apply_pipeline(get_solid,units)

	fig = glia.plot_units(f_plot_solid_spike_trains(1,2,3,0.01),solid_response,ncols=2,ax_xsize=10, ax_ysize=5, xlim=(0,3))
	fig.savefig(output_file)