import glia
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import update_wrapper, partial
# from tests.conftest import display_top, tracemalloc
import logging
logger = logging.getLogger('glia')

# def plot_motion_sensitivity(axis_gen,data):
def c_plot_bar(ax, title):
    # continuation
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("trials by descending bar width")

def plot_solid_versus_bar(fig, axis_gen, data, prepend, append):
    solids,bars_by_speed = data
    for speed in sorted(list(bars_by_speed.keys())):
        bars = bars_by_speed[speed]
        max_lifespan = max(bars,
            key=lambda e: e["stimulus"]["lifespan"])["stimulus"]["lifespan"]
        lifespans = set()
        for e in bars:
            # need to calculate duration of light over a particular point
            width = e["stimulus"]["width"]
            light_duration = int(np.ceil(width/speed))
            lifespans.add(light_duration)

        light_wedge = glia.compose(
            partial(filter,lambda x: x["stimulus"]["lifespan"] in lifespans),
            partial(sorted,key=lambda x: x["stimulus"]["lifespan"])
        )
        
        sorted_solids = light_wedge(solids)
        sorted_bars = sorted(bars,key=lambda x: x["stimulus"]["width"])


        xlim = glia.axis_continuation(lambda axis: axis.set_xlim(0,max_lifespan))
        bar_text = glia.axis_continuation(partial(c_plot_bar,
            title="Bars with speed: {}".format(speed)))

        glia.plot_spike_trains(axis_gen,sorted_solids,prepend,append,continuation=xlim)
        glia.plot_spike_trains(axis_gen,sorted_bars,
            continuation=glia.compose(xlim,bar_text))

# for v2
def plot_solid_versus_bar_for_speed(fig, axis_gen, data, prepend, append, speed):
    # also assumes 5 contrasts
    logger.debug("plot solid versus bar for speed")
    
    
    solids,bars_by_speed = data
    bars = bars_by_speed[speed]
    max_lifespan = max(bars,
        key=lambda e: e["stimulus"]["lifespan"])["stimulus"]["lifespan"]
    lifespans = set()
    widths = set()
    colors = set()
    for e in bars:
        width = e["stimulus"]["width"]
        widths.add(width)
        color = e["stimulus"]["barColor"]
        # need to calculate duration of light over a particular point
        light_duration = int(np.ceil(width/speed))

        lifespans.add(light_duration)
        colors.add(color)

    logger.debug("lifespans {}, widths {}".format(len(lifespans),len(widths)))
    # WARNING
    # assert len(lifespans)==len(widths)

    # keep charts aligned by row
    bar_ymap = {w: i for i,w in enumerate(sorted(list(widths)))}
    solid_ymap = {l: i for i,l in enumerate(sorted(list(lifespans)))}
    
    # used to map stimulus to proper row
    c_bar_ymap = lambda s: bar_ymap[s["width"]]
    c_solid_ymap = lambda s: solid_ymap[s["lifespan"]]

    sorted_colors = sorted(list(colors),reverse=True)

    ntrials = len(lifespans)
    xlim = glia.axis_continuation(lambda axis: axis.set_xlim(0,max_lifespan))
    ylim = glia.axis_continuation(lambda axis: axis.set_ylim(0,ntrials))
    color_text = lambda x: glia.axis_continuation(partial(c_plot_bar,
            title="Color: {}".format(x)))
    solid_continuation = lambda x: glia.compose(ylim,xlim,color_text(x))
    
    bar_text = glia.axis_continuation(partial(c_plot_bar,
            title="Bar"))
    bar_continuation = glia.compose(ylim,xlim,bar_text)


    # we plot the solid first so they are all on the same row
    for color in sorted_colors:
        logger.debug('plotting SOLID for {}'.format(color))
        light_wedge = glia.compose(
            partial(filter,lambda x: x["stimulus"]["lifespan"] in lifespans),
            partial(filter,lambda x: x["stimulus"]["backgroundColor"]==color),
            partial(sorted,key=lambda x: x["stimulus"]["lifespan"])
        )
        
        sorted_solids = light_wedge(solids)
        logger.debug('Solid lifespans are: ' + ",".join([str(s["stimulus"]["lifespan"]) for s in sorted_solids]))
        glia.plot_spike_trains(axis_gen,sorted_solids,prepend,append,
            continuation=solid_continuation(color), ymap=c_solid_ymap)

    for color in sorted_colors:
        filtered_bars = glia.compose(
            partial(filter,lambda x: x["stimulus"]["barColor"]==color),
            partial(sorted,key=lambda x: x["stimulus"]["width"])
            )
        sorted_bars = filtered_bars(bars)
        logger.debug('plotting BAR for {}'.format(color))
        logger.debug('Bar widths are: ' + ",".join([str(s["stimulus"]["width"]) for s in sorted_bars]))
        glia.plot_spike_trains(axis_gen,sorted_bars,
            continuation=bar_continuation, ymap=c_bar_ymap)

def plot_acuity_v3(fig, axis_gen, data, prepend, append, speed):
    logger.debug("plot solid versus bar for speed")    
    
    ax = next(axis_gen)
    solids,bars_by_speed = data
    bars = bars_by_speed[speed]
    max_lifespan = max(bars,
        key=lambda e: e["stimulus"]["lifespan"])["stimulus"]["lifespan"]
    lifespans = set()
    widths = set()
    angles = set()
    for e in bars:
        width = e["stimulus"]["width"]
        angle = e["stimulus"]["angle"]
        widths.add(width)
        angles.add(angle)
        # need to calculate duration of light over a particular point
        light_duration = int(np.ceil(width/speed))

        lifespans.add(light_duration)

    logger.debug("lifespans {}, widths {}".format(len(lifespans),len(widths)))
    # WARNING
    # assert len(lifespans)==len(widths)

    # keep charts aligned by row
    angle_width = []
    for w in sorted(widths):
        for a in sorted(angles):
            angle_width.append((a,w))

    bar_ymap = {aw: i for i,aw in enumerate(angle_width)}
    
    # used to map stimulus to proper row
    c_bar_ymap = lambda s: bar_ymap[(s["angle"],s["width"])]

    nangles = len(angles)
    nwidths = len(widths)
    ny = nangles*nwidths

    xlim = glia.axis_continuation(lambda axis: axis.set_xlim(0,max_lifespan))
    ylim = glia.axis_continuation(lambda axis: axis.set_ylim(0,ny))

    bar_text = glia.axis_continuation(partial(c_plot_bar,
            title="{} angles x {} widths at {} px/s".format(
                nangles,nwidths,speed)))
    bar_continuation = glia.compose(ylim,xlim,bar_text)

    for e in bars:
        spikes = e["spikes"]
        lifespan = e["stimulus"]["lifespan"]
        angle = e["stimulus"]["angle"]
        width = e["stimulus"]["width"]
        y = bar_ymap[(angle, width)]
        glia.draw_spikes(ax,spikes,y+0.2,y+0.8)
        ax.fill([0,lifespan,lifespan,0],
                [y,y,y+1,y+1],
                facecolor="gray", edgecolor="none", alpha=0.1)

    ax.yaxis.set_ticks(np.arange(0,nwidths*nangles,nangles))
    ax.yaxis.set_ticklabels(sorted(list(widths)))
    ax.set_ylabel("Bar Width in pixels (angle changes each row)")
    ax.set_xlabel("Time in seconds")

def plot_dissimilarity(fig, axis_gen, data, prepend, append, speed):
    logger.debug("plot dissimilarity")    
    
    ax = next(axis_gen)
    solids,bars_by_speed = data
    bars = bars_by_speed[speed]
    max_lifespan = 0
    min_lifespan = np.inf
    widths = set()
    angles = set()
    for e in bars:
        lifespan = e["stimulus"]["lifespan"]
        widths.add(e["stimulus"]["width"])
        angles.add(e["stimulus"]["angle"])
        if lifespan > max_lifespan:
            max_lifespan = lifespan
        elif lifespan < min_lifespan:
            min_lifespan = lifespan
    logger.warning("truncating bars to {} seconds (max is {})".format(
        min_lifespan, max_lifespan))
    normalized_bars = list(map(partial(glia.truncate_experiment,min_lifespan),
        bars))

    # normalized by dividing by seconds
    dissimilarity = glia.victor_purpura(normalized_bars)/min_lifespan
    cax = ax.imshow(dissimilarity, cmap='viridis')

    nangles = len(angles)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, nangles))
    ax.xaxis.set_ticklabels(sorted(list(widths)))
    ax.set_xlabel("{} angles per bar width (pixels)".format(nangles))
    ax.yaxis.set_ticks(np.arange(start, end, nangles))
    ax.yaxis.set_ticklabels(sorted(list(widths)))
    ax.set_ylabel("{} angles per bar width (pixels)".format(nangles))
    fig.colorbar(cax)

def plot_motion_sensitivity(fig, axis_gen, data,nwidths,speeds,prepend, append):
    solids,bars_by_speed = data    
    # We assume each band of widths is length 8 & first band is 2-16
    # each subsequent band is twice the width
    
    # the list is sorted so the final bar has longest lifespan
    max_lifespan = bars_by_speed[speeds[0]][nwidths-1]["stimulus"]["lifespan"]
    
    for i,speed in enumerate(speeds):
        base_width = 2**(i+1)
        next_width = base_width
        widths = {base_width*(n+1) for n in range(8)}
        bars_to_plot = list(filter(lambda x: x["stimulus"]["width"] in widths,
            bars_by_speed[speed]))
        bars_to_plot.sort(key=lambda x: x["stimulus"]["width"])
        try:
            assert len(bars_to_plot)==8
        except:
            print("width",widths)
            print([bar["stimulus"] for bar in bars_to_plot])
            raise

        xlim = glia.axis_continuation(lambda axis: axis.set_xlim(0,max_lifespan))
        ylim = glia.axis_continuation(lambda axis: axis.set_ylim(0,8))
        bar_text = glia.axis_continuation(partial(c_plot_bar,
            title="Bars with speed: {} & base width: {}".format(
                speed,base_width)))
        glia.plot_spike_trains(axis_gen,bars_to_plot,
            continuation=glia.compose(xlim,bar_text,ylim))
        
        if i==len(speeds)-1:
            # After the last speed, Plot solid
            lifespans = set()
            for w in widths:
                light_duration = int(np.ceil(w/speed))
                lifespans.add(light_duration)
                
            light_wedge = glia.compose(
                partial(filter,lambda x: x["stimulus"]["lifespan"] in lifespans),
                partial(sorted,key=lambda x: x["stimulus"]["lifespan"])
            )
            
            sorted_solids = light_wedge(solids)
            glia.plot_spike_trains(axis_gen,sorted_solids,prepend,append,continuation=xlim)
 

def save_acuity_chart(units, stimulus_list, c_unit_fig,
                         c_add_retina_figure, prepend, append):
    "Compare SOLID light wedge to BAR response in corresponding ascending width."

    print("Creating acuity chart v3.")
    get_solids = glia.compose(
        glia.f_create_experiments(stimulus_list,prepend_start_time=prepend,
                                  append_lifespan=append),
        glia.f_has_stimulus_type(["SOLID"]),
    )
    solids = glia.apply_pipeline(get_solids,units, progress=True)

    # offset to avoid diamond pixel artifacts
    get_bars_by_speed = glia.compose(
        glia.f_create_experiments(stimulus_list),
        glia.f_has_stimulus_type(["BAR"]),
        partial(sorted,key=lambda x: x["stimulus"]["angle"]),
        partial(sorted,key=lambda x: x["stimulus"]["width"]),
        partial(glia.group_by,key=lambda x: x["stimulus"]["speed"])
    )
    bars_by_speed = glia.apply_pipeline(get_bars_by_speed,units, progress=True)

    speeds = list(glia.get_unit(bars_by_speed)[1].keys())

    for speed in sorted(speeds):
        print("Plotting acuity for speed {}".format(speed))
        plot_function = partial(plot_acuity_v3,
                            prepend=prepend,append=append,speed=speed)
        filename = "acuity-{}".format(speed)
        result = glia.plot_units(plot_function,partial(c_unit_fig,filename),solids,bars_by_speed,
                                 nplots=1,ncols=1,ax_xsize=5, ax_ysize=15,
                                 figure_title="Bars with speed {}".format(speed))
        
        plot_function = partial(plot_dissimilarity,
                            prepend=prepend,append=append,speed=speed)
        filename = "dissimilarity-{}".format(speed)
        result = glia.plot_units(plot_function,partial(c_unit_fig,filename),solids,bars_by_speed,
                                 nplots=1,ncols=1,ax_xsize=7, ax_ysize=7,
                                 figure_title="Dissimilarity matrix for bars with speed {}".format(speed))
