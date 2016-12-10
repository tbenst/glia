import glia
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import update_wrapper, partial

# def plot_motion_sensitivity(axis_gen,data):
def c_plot_bar(ax, title):
    # continuation
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("trials by descending bar width")

def plot_solid_versus_bar(axis_gen,data, prepend, append):
    solids,bars_by_speed = data
    for speed in sorted(list(bars_by_speed.keys())):
        bars = bars_by_speed[speed]
        max_lifespan = max(bars,
            key=lambda e: e["stimulus"]["lifespan"])["stimulus"]["lifespan"]/120
        lifespans = set()
        for e in bars:
            # need to calculate duration of light over a particular point
            width = e["stimulus"]["width"]
            light_duration = int(np.ceil(width/speed*120))
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

def plot_motion_sensitivity():
    pass

def save_acuity_chart(units, stimulus_list, c_add_unit_figures,
                      c_add_retina_figure, prepend, append):
    "Compare SOLID light wedge to BAR response in corresponding ascending width."

    print("Creating acuity chart.")
    # for each speed, create two charts--SOLID & BAR
    # the solid should only include lifespans corresponding to each width

    # pseudocode
    # create dictionary of speeds
    # for each speed:
    #     lifespans = widths["lifespan"]
    #     f_select_experiments(widths)
    #     solids = f_select(solid like spent)
    #     plot_solid_wedge(solids)
    #     plot_spike_trains(solids)

    get_solids = glia.compose(
        glia.f_create_experiments(stimulus_list,prepend_start_time=prepend,
                                  append_lifespan=append),
        glia.f_has_stimulus_type(["SOLID"]),
    )
    solids = glia.apply_pipeline(get_solids,units)

    get_bars_by_speed = glia.compose(
        glia.f_create_experiments(stimulus_list),
        glia.f_has_stimulus_type(["BAR"]),
        partial(glia.group_by,key=lambda x: x["stimulus"]["speed"])
    )
    bars_by_speed = glia.apply_pipeline(get_bars_by_speed,units)

    nspeeds = len(glia.get_unit(bars_by_speed)[1].keys())

    # we plot bar and solid for each speed
    plot_function = partial(plot_solid_versus_bar,prepend=prepend,append=append)
    result = glia.plot_units(plot_function,solids,bars_by_speed,
                             nplots=nspeeds*2,ncols=2,ax_xsize=10, ax_ysize=5)
    c_add_unit_figures(result)
    glia.close_figs([fig for the_id,fig in result])

