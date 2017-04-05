import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from tqdm import tqdm
from typing import List, Any, Dict


from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool, Semaphore
from functools import partial
import traceback
# import pytest

import glia.config as config
from .analysis import last_spike_time
from .pipeline import get_unit
from .functional import zip_dictionaries
from .types import Unit
from glia.config import logger

Seconds = float
ms = float
UnitSpikeTrains = List[Dict[str,np.ndarray]]


def axis_generator(ax,transpose=False):
    if isinstance(ax,matplotlib.axes.Axes):
        yield(ax)
    else:
        if transpose:
            axes = np.transpose(ax)
        else:
            axes = ax
        for handle in axes.reshape(-1):
            yield(handle)

def multiple_figures(nfigs, nplots, ncols=4, nrows=None, ax_xsize=4, ax_ysize=4, subplot_kw=None):
    figures = []
    axis_generators = []

    if not nrows:
        nrows = int(np.ceil(nplots/ncols))

    for i in range(nfigs):
        fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*ax_xsize,nrows*ax_ysize), subplot_kw=subplot_kw)
        axis = axis_generator(ax)
        figures.append(fig)
        axis_generators.append(axis)

    return (figures, axis_generators)

def subplots(nplots, ncols=4, nrows=None, ax_xsize=4, ax_ysize=4, transpose=False,subplot_kw=None):
    if not nrows:
        nrows = int(np.ceil(nplots/ncols))

    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*ax_xsize,nrows*ax_ysize), subplot_kw=subplot_kw)
    axis = axis_generator(ax,transpose=transpose)

    return fig, axis

def plot_pdf_path(directory,name):
    return os.path.join(directory,name+".pdf")

def open_pdfs(plot_directory, unit_ids,unit_name_lookup=None):
    if unit_name_lookup is not None:
        return {unit_id: PdfPages(plot_pdf_path(plot_directory,unit_name_lookup[unit_id])) for unit_id in unit_ids}
    else:    
        return {unit_id: PdfPages(plot_pdf_path(plot_directory, unit_id)) for unit_id in unit_ids}

def add_figure_to_unit_pdf(fig,unit_id,unit_pdfs):
    unit_pdfs[unit_id].savefig(fig)
    return (unit_id, fig)

def add_to_unit_pdfs(id_figure_tuple,unit_pdfs):
    print("Adding figures to PDFs")
    for unit_id,fig in tqdm(id_figure_tuple):
        add_figure_to_unit_pdf(fig,unit_id,unit_pdfs)

def close_pdfs(unit_pdfs):
    print("saving PDFs")
    for unit_id,pdf in tqdm(unit_pdfs.items()):
        pdf.close()

def close_figs(figures):
    for fig in figures:
        plt.close(fig)

def save_figs(figures, filenames):
    for fig,name in zip(figures,filenames):
        fig.savefig(name)

def save_fig(filename, unit_id, fig):
    logger.debug("Saving {} for {}".format(filename,unit_id))
    name = Unit.name_lookup[unit_id]
    fig.savefig(os.path.join(config.plot_directory,name,filename+".jpg"))

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


def draw_spikes(ax, spike_train, ymin=0,ymax=1,color="black",alpha=0.5):
    "Draw each spike as black line."
    # draw_spike = np.vectorize(lambda s: ax.vlines(s, ymin, ymax,colors=color,alpha=alpha))
    # for spike in spike_train:
    #     draw_spike(spike)
    ax.vlines(spike_train, ymin, ymax,colors=color,alpha=alpha)


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

def plot_units(unit_plot_function, c_unit_fig, *units_data, nplots=1, ncols=1,
               nrows=None, ax_xsize=2, ax_ysize=2, figure_title=None, transpose=False,
               subplot_kw=None):
    """Create a giant figure with one or more plots per unit.
    
    c_unit_fig determines what happens to fig when produced
    transpose==True will yield axes by column 

    Must supply an even number of arguments that alternate function, units. If one pair is provided,
    ncols will determine the number of columns. Otherwise, each unit will get one row."""
    logger.info("plotting units")
    number_of_units = len(units_data[0].keys())

    processes = config.processes

    all_data = zip_dictionaries(*units_data)

    def data_generator():
        for unit_id, data in all_data:
            yield (unit_id, data)
                # ax_xsize, ax_ysize, figure_title, subplot_kw, semaphore)

    pool = Pool(processes)
    logger.info("passing tasks to pool")
    plot_worker = partial(_plot_worker, unit_plot_function, c_unit_fig, nplots,
                ncols, nrows, ax_xsize, ax_ysize, figure_title, transpose, subplot_kw)
    list(pool.imap_unordered(plot_worker, tqdm(data_generator(), total=number_of_units)))
    pool.close()
    pool.join()


def _plot_worker(plot_function, c_unit_fig, nplots, ncols, nrows, ax_xsize,
        ax_ysize, figure_title, transpose, subplot_kw, args):
    "Use with functools.partial() so function only takes args."

    logger.debug("plot worker")
    
    unit_id, data = args
        # ax_ysize, figure_title, subplot_kw, semaphore) = args
    if len(data)==1:
        data = data[0]
    fig = plot(plot_function, data, nplots, ncols=ncols, nrows=nrows,
        ax_xsize=ax_xsize, ax_ysize=ax_ysize,
        figure_title=figure_title, transpose=transpose, subplot_kw=subplot_kw)

    logger.debug("Plot worker successful for {}".format(unit_id))
    c_unit_fig(unit_id,fig)
    plt.close(fig)


def plot_each_by_unit(unit_plot_function, units, ax_xsize=2, ax_ysize=2,
               subplot_kw=None):
    "Iterate each value by unit and pass to the plot function."
    # number of units
    # number of values
    number_of_plots = len(get_unit(units)[1].values())
    number_of_units = len(list(units.keys()))
    
    # if nrows*ncols > 100:
    #     nrows = int(np.floor(100/ncols))
    #     print("only plotting first {} units".format(nrows))
    
    # fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*ax_xsize,nrows*ax_ysize), subplot_kw=subplot_kw)
    # axis = axis_generator(ax)
    
    multiple_figures(number_of_units, number_of_plots)
    i = 0
    for unit_id, value in units.items():
        if i>=100:
            break
        else:
            i+=1
        
        if type(value) is dict:
            gen = value.items()
        else:
            gen = value
        for v in gen:
            cur_ax = next(axis)
            unit_plot_function(cur_ax,unit_id,v)
    return fig


def plot_from_generator(plot_function, data_generator, nplots, ncols=4, ax_xsize=7, ax_ysize=10, ylim=None, xlim=None, subplot_kw=None):
    "plot each data in list_of_data using plot_function(ax, data)."
    nrows = int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*ax_xsize,nrows*ax_ysize), subplot_kw=subplot_kw)
    axis = axis_generator(ax)

    for data in data_generator():
        cur_ax = next(axis)
        plot_function(cur_ax, data)
        if ylim is not None:
            cur_ax.set_ylim(ylim)
        if xlim is not None:
            cur_ax.set_xlim(xlim)

    return fig


def plot(plot_function, data, nplots=1, ncols=1, nrows=None, ax_xsize=4,
            ax_ysize=4, figure_title=None, transpose=False, subplot_kw=None):
    fig, axes = subplots(nplots, ncols=ncols, nrows=nrows, ax_xsize=ax_xsize, ax_ysize=ax_ysize,
                        transpose=transpose, subplot_kw=subplot_kw)
    try:
        plot_function(fig, axes, data)
    except Exception as e:
        print("Plot function ({}) failed: ".format(plot_function),e)
        traceback.print_tb(e.__traceback__)
    if figure_title is not None:
        fig.suptitle(figure_title)
    return fig

def plot_each_for_unit(unit_plot_function, unit, subplot_kw=None):
    "Single unit version of plot_each_by_unit."
    unit_id = unit[0]
    value = unit[1]
    ncols = len(value.values())
    fig, ax = plt.subplots(1, ncols, subplot_kw=subplot_kw)
    axis = axis_generator(ax)
#     axis = iter([ax])
    
    
    if type(value) is dict:
        gen = value.items()
    else:
        gen = value
    for v in gen:
        cur_ax = next(axis)
        unit_plot_function(cur_ax,unit_id,v)
    return fig

def c_plot_solid(ax):
    ax.set_title("Unit spike train per SOLID")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("trials")

def _axis_continuation_helper(continuation,axis):
    continuation(axis)
    return axis

def axis_continuation(function):
    """Takes a lambda that modifies the axis object, and returns the axis.

    This enables a pipeline of functions that modify the axis object"""

    return partial(_axis_continuation_helper,function)

def plot_spike_trains(fig, axis_gen, data,prepend_start_time=0,append_lifespan=0,
                      continuation=c_plot_solid, ymap=None):
    ax = next(axis_gen)
    for i,v in enumerate(data):
        stimulus, spike_train = (v["stimulus"], v["spikes"])
            
        # use ymap to keep charts aligned if a comparable stimulus was not run
        # i.e., if each row is a different bar width but one chart is sparsely sampled
        if ymap:
            trial = ymap(stimulus)
        else:
            trial = i

        lifespan = stimulus['lifespan'] / 120
        logger.debug("plot_spike_trains ({}) iteration: {}, lifespan: {}".format(
            stimulus["stimulusType"],trial,lifespan))
        if lifespan > 120:
            logger.debug("skipping stimulus longer than 120 seconds")
        
        if spike_train.size>0:
            draw_spikes(ax, spike_train, ymin=trial+0.3,ymax=trial+1)
        
        stimulus_end = prepend_start_time + lifespan
        duration = stimulus_end + append_lifespan
        if stimulus_end!=duration:
            # this is for solid
            ax.fill([0,prepend_start_time,prepend_start_time,0],
                    [trial,trial,trial+1,trial+1],
                    facecolor="gray", edgecolor="none", alpha=0.1)
            ax.fill([stimulus_end,duration,duration,stimulus_end],
                    [trial,trial,trial+1,trial+1],
                    facecolor="gray", edgecolor="none", alpha=0.1)
        else:
            # draw all gray for all others
            ax.fill([0,lifespan,lifespan,0],
                    [trial,trial,trial+1,trial+1],
                    facecolor="gray", edgecolor="none", alpha=0.1)


    continuation(ax)

# @pytest.fixture(scope="module")
# def channels():
#     import files
#     return read_mcs_dat('tests/sample_dat/')


def group_lifespan(group):
    return sum(list(map(lambda x: x["stimulus"]["lifespan"]/120, group)))


def raster_group(fig, axis_gen, data):
    """Plot all spike trains for group on same row.

    Expects data to be a list of lists """
    ax = next(axis_gen)
    trial = 0
    longest_group = max(map(group_lifespan, data))

    for group in data:
        # x offset for row, ie relative start_time
        offset = 0
        end_time = 0
        for i,v in enumerate(group):
            stimulus, spike_train = (v["stimulus"], v["spikes"])
            lifespan = stimulus['lifespan'] / 120
            end_time += lifespan


            if lifespan > 60:
                logger.warning("skipping stimulus longer than 60 seconds")
                continue
            if spike_train.size>0:
                draw_spikes(ax, spike_train+offset, ymin=trial+0.3,
                    ymax=trial+1)

            kwargs = {"facecolor": stimulus["backgroundColor"],
                      "edgecolor": "none",
                      "alpha": 0.1}
            if stimulus['stimulusType']=='BAR':
                kwargs['hatch'] = '/'
            ax.fill([offset,end_time,end_time,offset],
                    [trial,trial,trial+1,trial+1],
                    **kwargs)
            offset = end_time

        if offset<longest_group:
            ax.fill([offset,longest_group,longest_group,offset],
                    [trial,trial,trial+1,trial+1],
                    facecolor="black",
                    edgecolor="none", alpha=1)

        trial += 1
        
    ax.set_title("Unit spike train per group")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("trials")
    ax.set_xlim((0,longest_group))
    ax.set_ylim((0,trial))
