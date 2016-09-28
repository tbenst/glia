import matplotlib.pyplot as plt
import numpy as np
from typing import List, Any, Dict
from .analysis import last_spike_time
# import pytest


Seconds = float
ms = float
UnitSpikeTrains = List[Dict[str,np.ndarray]]


def axis_generator(ax):
    for handle in ax.reshape(-1):
        yield(handle)    



def spike_histogram(unit_spike_trains: UnitSpikeTrains, bin_width: Seconds=0.1,
                    time: (Seconds, Seconds)=(None, None), plot=True) -> (Any):
    """
    Plot histogram of summed spike counts across all channels.

    Args:
        channels: List of numpy arrays; each float represents spike times.
        bin_width: Size of histogram bin width in seconds.
        time: Tuple of (start_time, end_time) i

    Returns:
        matplotlib pyplot figure (can call .show() to display).

    Raises:
        ValueError for bad bin_width.
    """
    channels = unit_spike_trains.keys()
    if bin_width <= 0:
        raise ValueError("bin_width must be greater than zero")

    # flatten array
    all_spikes = np.hstack([c for c in channels if c is not None])

    # filter for desired range
    start_time = time[0]
    end_time = time[1]
    if start_time is not None:
        all_spikes = all_spikes[all_spikes > start_time]
    else:
        start_time = 0

    if end_time is not None:
        all_spikes = all_spikes[all_spikes < end_time]
    else:
        end_time = last_spike_time(channels)

    # add bin_width to last_time so histogram includes final edge
    bins = np.arange(start_time, np.ceil(end_time) + bin_width, bin_width)

    # plot histogram
    if plot is True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(all_spikes, bins)
    else:
        return(np.histogram(all_spikes, bins))


def isi_histogram(unit_spike_trains: UnitSpikeTrains, bin_width: Seconds=1/1000,
                  time: (Seconds, Seconds)=(0, 100/1000), average=True,
                  fig_size=(15, 30)) -> (Any):
    channels = unit_spike_trains.keys()

    channels = [np.diff(c) for c in channels]
    # Unit is seconds so x is in ms for x/1000
    bins = np.arange(time[0], time[1], bin_width)
    fig = plt.figure(figsize=fig_size)

    if average:
        # flatten array
        all_isi = np.hstack([c for c in channels if c is not None])

        ax = fig.add_subplot(111)
        ax.hist(all_isi, bins)
    else:
        subp = subplot_generator(channels,5)
        for channel in channels:
            ax = fig.add_subplot(*next(subp))
            ax.hist(channel, bins)


def visualize_spikes(unit_spike_trains: UnitSpikeTrains, fig_size=(30, 15)):
    fig = plt.figure(figsize=fig_size)

    channels = unit_spike_trains.keys()

    # Draw each spike as black line
    ax = fig.add_subplot(211)
    for i, unit in enumerate(channels):
        ax.vlines(channels[i], i, i + 1)

    # Plot histogram
    ax2 = fig.add_subplot(212)
    ax2.plot(spike_histogram(channels))


def draw_spikes(ax, spike_train, ymin=0,ymax=1,color="black",alpha=0.3):
    "Draw each spike as black line."
    draw_spike = np.vectorize(lambda s: ax.vlines(s, ymin, ymax,colors=color,alpha=alpha))
    for spike in spike_train:
        draw_spike(spike)

# Helpers

def subplot_generator(n_charts, num_cols):
    """Generate arguments for matplotlib add_subplot.

    Must use * to unpack the returned tuple. For example,

    >> fig = plt.figure()
    <matplotlib.figure.Figure at 0x10fdec7b8>
    >> subp = subplot_generator(4,2)
    >> fig.add_subplot(*next(subp))
    <matplotlib.axes._subplots.AxesSubplot at 0x112cee6d8>
    (NOTE doctest not running)
    """

    if type(n_charts) is list:
        n_charts = len(n_charts)

    num_rows = n_charts // num_cols + (n_charts % num_cols != 0)
    n = 1
    while n <= n_charts:
        yield (num_rows, num_cols, n)
        n += 1

def create_polar_scatterplot(stimulus_analytics: dict, ax = None):
    if ax is None:
        fig =plt.figure()
        ax = fig.add_subplot(111, projection='polar')
    else:
        fig = None
    angles = {k:count_items(v) for k,v in stimulus_analytics.items()}
    theta = []
    r = []
    area = []
    for a, counts in angles.items():
        for spike_count, number_of_occurrences  in counts.items(): 
            theta.append(a)
            r.append(spike_count)
            area.append(number_of_occurrences^3)
    c = ax.scatter(theta, r, area)
    c.set_alpha(0.75)
    if fig is not None:
        return fig


def count_items(my_list):
    to_return={}
    for i in my_list:
        try:
            to_return[i]+=1
        except:
            to_return[i]=1
    return to_return

def plot_ifr(ax_gen, unit, ylim=None, legend=False):
    color=iter(plt.cm.rainbow(np.linspace(0,1,20))) #this is a hack, should not be 20
    
    for stimulus, ifr_list in unit.items():
        c = next(color)
        stim = eval(stimulus)
        l = "speed:"+ str(stim["speed"]) + ", width:" + str(stim["width"]) + ", bar_color:"+str(stim["barColor"])
        for trial in ifr_list:
            ax = next(ax_gen)
            ax.plot(trial, color=c)
            ax.set_title(l)
            if ylim is not None:
                ax.set_ylim(ylim)

def plot_direction_selectively(ax, unit_id, bar_firing_rate, bar_dsi, legend=False):
    """Plot the average for each speed and width."""
    # we will accumulate by angle in this dictionary and then divide
    analytics = by_speed_width_then_angle(bar_firing_rate[unit_id])
    speed_widths = analytics.keys()
    speeds = sorted(list(set([speed for speed,width in speed_widths])))
    widths = sorted(list(set([width for speed,width in speed_widths])))
    color=iter(plt.cm.rainbow(np.linspace(0,1,len(speeds))))
    w = iter(np.linspace(1,5,len(widths)))
    speed_style = {speed: next(color) for speed in speeds}
    width_style = {width: next(w) for width in widths}
    
    for speed_width, angle_dictionary in analytics.items():
        speed, width = speed_width
        line_angle = []
        line_radius = []
        for angle, average_number_spikes in angle_dictionary.items():
            line_angle.append(angle)
            line_radius.append(average_number_spikes)
        # connect the line
        line_angle, line_radius = sort_two_arrays_by_first(line_angle,line_radius)
        line_angle.append(line_angle[0])
        line_radius.append(line_radius[0])
        ax.plot(line_angle,line_radius, linewidth=width_style[width], color=speed_style[speed], label=speed_width)
        
    ax.set_title('Unit: '+str(unit_id))
    ax.set_ylabel("Firing rate (Hz)")
    speed_style["overall"] = "white"
    unique_speed_width = sorted(speed_widths, key=f_get_key(1))
    unique_speed_width = sorted(unique_speed_width, key=f_get_key(0))
    
    columns = unique_speed_width + ["overall"]
    colors = [speed_style[speed] for speed,width in unique_speed_width] + ['white']
    cells = [["{:1.3f}".format(bar_dsi[unit_id][speed_width]) for speed_width in columns], \
             ["{:1.3f}".format(bar_osi[unit_id][speed_width]) for speed_width in columns]]
    
    table = ax.table(cellText=cells,
                      rowLabels=['DSI',"OSI"],
                      colLabels=columns,
                     colColours=colors,
                        loc='bottom', bbox = [0,-0.2,1,0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    if legend is True:
        ax.legend()

# @pytest.fixture(scope="module")
# def channels():
#     import files
#     return read_mcs_dat('tests/sample_dat/')