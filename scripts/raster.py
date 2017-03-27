import glia
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from warnings import warn
import logging
logger = logging.getLogger('glia')

def save_raster(units, stimulus_list, c_unit_fig, c_add_retina_figure,
        sort_by=glia.group_lifespan):
    print("Creating spike train raster plot")
    
    get_solid = glia.compose(
        glia.f_create_experiments(stimulus_list),
        partial(glia.group_by,
            key=lambda x: x["stimulus"]["metadata"]["group"]),
        glia.group_dict_to_list,
        partial(sorted,key=sort_by)
    )
    response = glia.apply_pipeline(get_solid,units)
    glia.plot_units(glia.raster_group,c_unit_fig,response,nplots=1,
        ncols=1,ax_xsize=15, ax_ysize=10)
