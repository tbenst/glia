#!/usr/bin/env python
#%%
import matplotlib
matplotlib.use("agg")
matplotlib.rcParams['figure.max_open_warning'] = 250
matplotlib.rcParams['font.size'] = 16

import glia
from glia import match_filename
from fnmatch import fnmatch
import click
import os, av, pandas as pd
from functools import reduce
from pathlib import Path
import sys
import re, h5py
import glia_scripts.solid as solid
import glia_scripts.bar as bar
import glia_scripts.acuity as acuity
import glia_scripts.grating as grating
import glia_scripts.raster as raster
import glia_scripts.convert as convert
import glia_scripts.video as video
import errno
from glia_scripts.classify import svc
import traceback, pdb
import glia.config as config
from glia.config import logger, logging, channel_map
from functools import update_wrapper, partial
from matplotlib import animation, cm, gridspec
# from tests.conftest import display_top, tracemalloc

from tqdm.auto import tqdm
from glob import glob
from glia.types import Unit
from matplotlib.backends.backend_pdf import PdfPages
from random import randint
import numpy as np
import yaml, csv
import cProfile
from glia import iter_chunks, read_3brain_analog

#%%
def plot_function(f):
    @click.pass_context
    def new_func(ctx, *args, **kwargs):
        context_object = ctx.obj
        return ctx.invoke(f, ctx.obj["units"], ctx.obj["stimulus_list"],
            ctx.obj["metadata"], ctx.obj["c_unit_fig"], ctx.obj["c_retina_fig"],
            *args[3:], **kwargs)
    return update_wrapper(new_func, f)

def analysis_function(f):
    @click.pass_context
    def new_func(ctx, *args, **kwargs):
        context_object = ctx.obj
        return ctx.invoke(f, ctx.obj["units"], ctx.obj["stimulus_list"],
            ctx.obj["metadata"], ctx.obj['filename'],
            *args[2:], **kwargs)
    return update_wrapper(new_func, f)

def vid_analysis_function(f):
    @click.pass_context
    def new_func(ctx, *args, **kwargs):
        context_object = ctx.obj
        return ctx.invoke(f, ctx.obj["units"], ctx.obj["stimulus_list"],
            ctx.obj["metadata"], ctx.obj['filename'],
            ctx.obj['frame_log'], ctx.obj['video_file'],
            *args[2:], **kwargs)
    return update_wrapper(new_func, f)

def video_function(f):
    @click.pass_context
    def new_func(ctx, *args, **kwargs):
        context_object = ctx.obj
        return ctx.invoke(f, ctx.obj["units"], ctx.obj["stimulus_list"],
            ctx.obj["metadata"], ctx.obj["c_unit_fig"], ctx.obj["c_retina_fig"],
            ctx.obj['frame_log'], ctx.obj['video_file'],
            *args[2:], **kwargs)
    return update_wrapper(new_func, f)


def plot_path(directory,plot_name):
    return os.path.join(directory,plot_name+".png")

@click.group()
def main():
    pass


@main.command()
@click.argument('filename', type=str)
def header(filename):
    """Print the header length from a Multichannel systems binary file.

    Commonly used for import into spike sorting software"""

    if not os.path.isfile(filename):
        filename = glia.match_filename(filename,'voltages')
    try:
        print('header length: ', glia.get_header(filename)[1])
    except:
        raise(ValueError, "Could not get header, are you sure it's a MCD binary export?")

generate_choices = ["random","hz"]
@main.group(chain=True)
@click.argument('filename', type=str)
@click.option("--notebook", "-n", type=click.Path(exists=True))
@click.option("--eyecandy", "-e", default="http://localhost:3000")
@click.option('generate_method', "-m",
    type=click.Choice(generate_choices), default="random",
    help="Type of test data.")
@click.option('number', "-n",
    type=int, default=2,
    help="Number of channels.")
@click.option('nunits', "-u",
    type=int, default=2,
    help="Number of channels.")
@click.option('stimulus', "-s",
    is_flag=True,
    help="Create .stim file without analog")
@click.pass_context
def generate(ctx, filename, eyecandy, generate_method, notebook, number,
    nunits, stimulus):
    data_directory, data_name = os.path.split(filename)
    if data_directory=='':
        data_directory=os.getcwd()

    if not notebook:
        notebook = glia.find_notebook(data_directory)

    lab_notebook = glia.open_lab_notebook(notebook)
    name, ext = os.path.splitext(filename)

    ctx.obj = {'filename': generate_method+"_"+name}

    stimulus_file = os.path.join(data_directory, name + ".stim")
    try:
        metadata, stimulus_list, method = glia.read_stimulus(stimulus_file)
        print('found .stim file')
    except:
        print('creating .stim file.')
        metadata, stimulus_list = glia.create_stimuli_without_analog(stimulus_file,
            notebook, name, eyecandy)

    ctx.obj["stimulus_list"] = stimulus_list
    ctx.obj["metadata"] = metadata
    # total_time = sum(map(lambda x: x['stimulus']['lifespan'], stimulus_list))
    last_stim = stimulus_list[-1]
    total_time = last_stim['start_time']+last_stim['stimulus']['lifespan']
    units = {}
    retina_id = f'{generate_method}_{name}'
    print('generating test data')
    for channel_x in range(number):
        for channel_y in range(number):
            # for unit_j in range(randint(1,5)):
            for unit_j in range(nunits):
                if generate_method=='random':
                    u = glia.random_unit(total_time, retina_id,
                        (channel_x, channel_y), unit_j)
                elif generate_method=="hz":
                    # hz = randint(1,90)
                    hz = 60
                    u = glia.hz_unit(total_time, hz, retina_id,
                        (channel_x, channel_y), unit_j)
                else:
                    raise(ValueError(f"Undefined generate_method: {generate_method}"))

                units[u.id] = u
    ctx.obj["units"] = units

    # prepare_output
    plot_directory = os.path.join(data_directory, f"{retina_id}-plots")
    config.plot_directory = plot_directory

    os.makedirs(plot_directory, exist_ok=True)
    os.chmod(plot_directory, 0o777)

    logger.debug("Outputting png")
    ctx.obj["c_unit_fig"] = glia.save_unit_fig
    ctx.obj["c_retina_fig"] = glia.save_retina_fig
    os.makedirs(os.path.join(plot_directory,"00-all"), exist_ok=True)

    for unit_id in ctx.obj["units"].keys():
        name = unit_id
        os.makedirs(os.path.join(plot_directory,name), exist_ok=True)

    try:
        lab_notebook_notype = glia.open_lab_notebook(notebook, convert_types=False)
        protocol_notype = glia.get_experiment_protocol(lab_notebook_notype,
                                                                  name)
        date_prefix = os.path.join(data_directory,
            protocol_notype['date'].replace(':','_'))
        frames_file = date_prefix + "_eyecandy_frames.log"
        video_file = date_prefix + "_eyecandy.mkv"
        frame_log = pd.read_csv(frames_file)
        frame_log = frame_log[:-1] # last frame is not encoded for some reason
        ctx.obj["frame_log"] = frame_log
        ctx.obj["video_file"] = video_file
    except Exception as e:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        print(e)
        print("Attempting to continue without frame log...")


@main.command()
@click.argument('filename', type=str)
@click.option("--analog-idx", "-i", type=int, help="Channel of light detector", default=1)
def calibrate(filename, analog_idx):
    """Find calibration for frame detection from analog channel of light flicker.
    """
    if not os.path.isfile(filename):
        filename = glia.match_filename(filename,"analog")
    analog = glia.read_raw_voltage(filename)[:,analog_idx]
    data_directory, fn = os.path.split(filename)
    calibration = glia.auto_calibration(analog, data_directory)
    glia.analog_histogram(analog, data_directory)
    print(f"saving analog histogram to {data_directory}/analog_histogram.png")
    with open(os.path.join(data_directory,"config.yml"), 'w') as outfile:
        yaml.dump({"analog_calibration": calibration.tolist()}, outfile)
    print(f"saving suggested config to {data_directory}/config.yml")



def init_logging(name, processes, verbose, debug):
    #### LOGGING CONFIGURATION
    fh = logging.FileHandler(name + '.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    if verbose:
        ch.setLevel(logging.INFO)
        # tracemalloc.start()
    elif debug:
        ch.setLevel(logging.DEBUG)

    else:
        ch.setLevel(logging.WARNING)
    if processes!=None:
        config.processes = processes
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s', '%H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Verbose logging on")
    logger.debug("Debug logging on")


@main.group(chain=True)
@click.argument('filename', type=str)
# @click.argument('filename', type=click.Path(exists=True))
@click.option("--notebook", "-n", type=click.Path(exists=True))
@click.option("--eyecandy", "-e", default="http://localhost:3000")
@click.option("--processes", "-p", type=int, help="Number of processors")
@click.option("--analog-idx", "-i", type=int, help="Channel of light detector", default=1)
# @click.option("--calibration", "-c", default="auto", help="""Sets the analog value
#     for each stimulus index. Should be dimension (3,2)""")
@click.option("--configuration", "-c", type=click.Path(exists=True), help="""Use
    configuration file for analog calibration, etc.""")
@click.option("--output", "-o", type=click.Choice(["png","pdf"]), default="png")
@click.option("--ignore-extra",  is_flag=True, help="Ignore extra stimuli if stimulus list is longer than detected start times in analog file.")
@click.option("--fix-missing",  is_flag=True, help="Attempt to fill in missing start times, use with --ignore-extra.")
@click.option("--threshold", "-r", type=float, default=9, help="Set the threshold for flicker")
@click.option("--integrity-filter", "-w", type=float, default=0.0,
    help="Only include units where classification percentage exceeds the specified amount.")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--by-channel", "-C", is_flag=True, help="Combine units by channel")
@click.option("--default-channel-map", "-m", is_flag=True, help="Use default channel map instead of reading from 3brain file")
@click.option("--debug", "-vv", is_flag=True)
@click.option("--dev", is_flag=True)
@click.option("--trigger", "-t", type=click.Choice(["flicker", 'detect-solid', "legacy", "ttl"]), default="flicker",
    help="""Use flicker if light sensor was on the eye candy flicker, solid if the light sensor detects the solid stimulus,
    or ttl if there is a electrical impulse for each stimulus.
    """)
@click.pass_context

def analyze(ctx, filename, trigger, threshold, eyecandy, ignore_extra=False,
        fix_missing=False, output=None, notebook=None,
        configuration=None, verbose=False, debug=False,processes=None,
        by_channel=False, integrity_filter=0.0, analog_idx=1,
        default_channel_map=False, dev=False):
    """Analyze data recorded with eyecandy.
    
    This command/function preprocesses the data & aligns stimuli to ephys
    recording.
    """
    print("version 0.5.1")
    init_logging(filename, processes, verbose, debug)
    #### FILEPATHS
    logger.debug(str(filename) + "   " + str(os.path.curdir))
    if not os.path.isfile(filename):
        try:
            filename = glia.match_filename(filename,"txt")
        except:
            try:
                filename = glia.match_filename(filename,"bxr")
            except:
                filename = glia.match_filename(filename,"csv")
            
    data_directory, data_name = os.path.split(filename)
    name, extension = os.path.splitext(data_name)
    # ignore first of two extensions (if applicable)
    name, _ = os.path.splitext(name)
    analog_file = os.path.join(data_directory, name +'.analog')
    if not os.path.isfile(analog_file):
        # use 3brain analog file
        analog_file = os.path.join(data_directory, name +'.analog.brw')

    if not os.path.isfile(analog_file):
        # Tyler's format; used if files were split for example
        analog_file = os.path.join(data_directory, name +'.analog.npz')

    stimulus_file = os.path.join(data_directory, name + ".stim")
    ctx.obj = {"filename": os.path.join(data_directory,name)}
    print(f"Analyzing {name}")

    if configuration!=None:
        with open(configuration, 'r') as f:
            user_config = yaml.safe_load(f)
        config.user_config = user_config
        if "analog_calibration" in user_config:
            config.analog_calibration = user_config["analog_calibration"]
        if "notebook" in user_config:
            notebook = user_config["notebook"]
        if "eyecandy" in user_config:
            eyecandy = user_config["eyecandy"]
        if "processes" in user_config:
            processes = user_config["processes"]
        if "integrity_filter" in user_config:
            integrity_filter = user_config["integrity_filter"]
        if "by_channel" in user_config:
            by_channel = user_config["by_channel"]

    if not notebook:
        notebook = glia.find_notebook(data_directory)

    lab_notebook = glia.open_lab_notebook(notebook)
    logger.info(f"{name=}")
    experiment_protocol = glia.get_experiment_protocol(lab_notebook, name)
    flicker_version = experiment_protocol["flickerVersion"]


    #### LOAD STIMULUS
    try:
        metadata, stimulus_list, method = glia.read_stimulus(stimulus_file)
        ctx.obj["stimulus_list"] = stimulus_list
        ctx.obj["metadata"] = metadata
        # assert method=='analog-flicker'
    except:
        print("No .stim file found. Creating from .analog file.".format(trigger))
        if flicker_version==0.3:
            metadata, stimulus_list = glia.create_stimuli(
                analog_file, stimulus_file, notebook, name, eyecandy, analog_idx, ignore_extra,
                config.analog_calibration, threshold)
            ctx.obj["stimulus_list"] = stimulus_list
            ctx.obj["metadata"] = metadata
            print('finished creating .stim file')
        elif trigger == "ttl":
            raise ValueError('not implemented')
        else:
            raise ValueError("invalid trigger: {}".format(trigger))
    
    # look for .frames file
    try:
        lab_notebook_notype = glia.open_lab_notebook(notebook, convert_types=False)
        protocol_notype = glia.get_experiment_protocol(lab_notebook_notype,
                                                                  name)
        date_prefix = os.path.join(data_directory,
            protocol_notype['date'].replace(':','_'))
        frames_file = date_prefix + "_eyecandy_frames.log"
        video_file = date_prefix + "_eyecandy.mkv"
        frame_log = pd.read_csv(frames_file)
        frame_log = frame_log[:-1] # last frame is not encoded for some reason
        ctx.obj["frame_log"] = frame_log
        ctx.obj["video_file"] = video_file
    except Exception as e:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        print(e)
        ctx.obj["frame_log"] = None
        ctx.obj["video_file"] = None
        print("Attempting to continue without frame log...")
    
    #### LOAD SPIKES
    spyking_regex = re.compile('.*\.result.hdf5$')
    eye = experiment_protocol['eye']
    experiment_n = experiment_protocol['experimentNumber']

    date = experiment_protocol['date'].date().strftime("%y%m%d")

    retina_id = date+'_R'+eye+'_E'+experiment_n
    if extension == ".txt":
        ctx.obj["units"] = glia.read_plexon_txt_file(filename,retina_id, channel_map)
    elif extension == ".bxr":
        if default_channel_map:
            channel_map_3brain = config.channel_map_3brain
        else:
            channel_map_3brain = None
        ctx.obj["units"] = glia.read_3brain_spikes(filename, retina_id,
            channel_map_3brain, truncate=dev)
    elif extension == ".csv":
        ctx.obj["units"] = glia.read_csv_spikes(filename, retina_id)        
    elif re.match(spyking_regex, filename):
        ctx.obj["units"] = glia.read_spyking_results(filename)
    else:
        raise ValueError(f'could not read {extension=}. Is it a plexon or spyking circus file?')

    #### DATA MUNGING OPTIONS
    if integrity_filter>0.0:
        good_units = solid.filter_units_by_accuracy(
            ctx.obj["units"], ctx.obj['stimulus_list'], integrity_filter)
        filter_good_units = glia.f_filter(lambda u,v: u in good_units)
        ctx.obj["units"] = filter_good_units(ctx.obj["units"])

    if by_channel:
        ctx.obj["units"] = glia.combine_units_by_channel(ctx.obj["units"])


    # prepare_output
    plot_directory = os.path.join(data_directory, name+"-plots")
    config.plot_directory = plot_directory

    os.makedirs(plot_directory, exist_ok=True)
    os.chmod(plot_directory, 0o777)

    if output == "pdf":
        logger.debug("Outputting pdf")
        ctx.obj["retina_pdf"] = PdfPages(glia.plot_pdf_path(plot_directory, "retina"))
        ctx.obj["unit_pdfs"] = glia.open_pdfs(plot_directory, list(ctx.obj["units"].keys()), Unit.name_lookup())
        # c connotes 'continuation' for continuation passing style
        ctx.obj["c_unit_fig"] = partial(glia.add_to_unit_pdfs,
            unit_pdfs=ctx.obj["unit_pdfs"])
        ctx.obj["c_retina_fig"] = lambda x: ctx.obj["retina_pdf"].savefig(x)

    elif output == "png":
        logger.debug("Outputting png")
        ctx.obj["c_unit_fig"] = glia.save_unit_fig
        ctx.obj["c_retina_fig"] = glia.save_retina_fig
        os.makedirs(os.path.join(plot_directory,"00-all"), exist_ok=True)

        for unit_id in ctx.obj["units"].keys():
            name = unit_id
            os.makedirs(os.path.join(plot_directory,name), exist_ok=True)


@analyze.resultcallback()
@click.pass_context
def cleanup(ctx, results, filename, trigger, threshold, eyecandy,
        ignore_extra=False, fix_missing=False, output=None, notebook=None,
        configuration=None, version=None, verbose=False, debug=False,
        processes=None, by_channel=False, integrity_filter=0.0, analog_idx=1,
        default_channel_map=None, dev=None):
    if output == "pdf":
        ctx.obj["retina_pdf"].close()
        glia.close_pdfs(ctx.obj["unit_pdfs"])

    print("Finished")


def create_cover_page(ax_gen, data):
    ax = next(ax_gen)
    unit_id = data
    ax.text(0.5, 0.5,unit_id, horizontalalignment='center',
        verticalalignment='center',
        fontsize=32)
    ax.set_axis_off()

@analyze.command()
@plot_function
def cover(units, stimulus_list, metadata, c_unit_fig, c_retina_fig):
    "Add cover page."
    data = {k:v.name for k,v in units.items()}
    result = glia.plot_units(create_cover_page,data,ax_xsize=10, ax_ysize=5)
    c_unit_fig(result)
    glia.close_figs([fig for the_id,fig in result])

@analyze.command()
@click.pass_context
def all(ctx):
    "Run all analyses."
    ctx.forward(solid_cmd)
    ctx.forward(bar_cmd)
    ctx.forward(grating_cmd)

@analyze.command("sta")
@video_function
def sta_cmd(units, stimulus_list, metadata, c_unit_fig, c_retina_fig,
        frame_log, video_file):
    binary_noise_frame_idx = video.get_binary_noise_frame_idx(stimulus_list,
                                                        frame_log)
    # # single unit
    # row_to_id = {}
    # for i,key in enumerate(units.keys()):
    #     row_to_id[i] = key
    #     # spikes = units[key].spike_train[units[key].spike_train<9]
    # unit_id = row_to_id[37]
    # print("unit_id", unit_id)
    # spike_train = units[unit_id].spike_train
    
    # sta = video.calc_sta(spike_train,  binary_noise_frame_idx, frame_log,
    #                    video_file)
    # fig = video.plot_sta(sta)
    plot_func = partial(video.sta_unit_plot_function,
                        frame_indices=binary_noise_frame_idx,
                        frame_log=frame_log,
                        video_file=video_file,
                        nsamples_before=25)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    glia.plot_units(plot_func, partial(c_unit_fig,"spatial_filter"), units, nplots=2, ncols = 1, GridSpec = gs)
    

@analyze.command("solid")
@click.option("--prepend", "-p", type=float, default=1,
    help="plot (seconds) before SOLID start time")
@click.option("--append", "-a", type=float, default=1,
    help="plot (seconds) after SOLID end time")
@click.option("--chronological", "-c", type=float, default=False,
    help="plot chronological order")
@plot_function
def solid_cmd(units, stimulus_list, metadata, c_unit_fig, c_retina_fig,
        prepend, append, chronological):
    "Create PTSH and raster of spikes in response to solid."
    # safe_run(solid.save_unit_psth,
    #     (units, stimulus_list, c_unit_fig, c_retina_fig, prepend, append))
    name = metadata['name']
    if chronological:
        solid.save_unit_spike_trains(units, stimulus_list, c_unit_fig, c_retina_fig, prepend, append)
    elif name=="wedge":
        solid.save_unit_wedges_v2(
            units, stimulus_list, partial(c_unit_fig,"wedge"), c_retina_fig)
    elif name=="kinetics":
        solid.save_unit_kinetics(
            units, stimulus_list, partial(c_unit_fig,"kinetics"), c_retina_fig)
    else:
        raise(ValueError(f"No match for {name}"))

generate.add_command(solid_cmd)

@analyze.command("bar")
@click.option("--by", "-b", type=click.Choice(["angle", "width","acuity"]), default="angle")
@plot_function
def bar_cmd(units, stimulus_list, metadata, c_unit_fig, c_retina_fig, by):
    # if all_methods or "direction" in methods:
    if by=="angle":
        bar.save_unit_response_by_angle(
            units, stimulus_list, c_unit_fig, c_retina_fig)
    elif by=="acuity":
        bar.save_acuity_direction(
            units, stimulus_list, partial(c_unit_fig,"acuity"),
                c_retina_fig)


@analyze.command("stim")
def stim_cmd():
    "Create .stim file without running other commands."
    pass

def debug_lambda(x,f):
    print('debug: ', x)
    return f(x)

generate.add_command(bar_cmd)

@analyze.command("convert")
@click.option("--quad", "-q", is_flag=True,
    help="use four classes for checkerboard")
@click.option("--append", "-a", type=float, default=0,
    help="add time (seconds) after stimulus end time")
@vid_analysis_function
def convert_cmd(units, stimulus_list, metadata, filename, frame_log,
        video_file, append, version=2, quad=False):
    name = metadata['name']
    if name=='letters':
        convert.save_letter_npz(
            units, stimulus_list, filename, append)
    elif name=='letters-tiled':
        print("Saving letters-tiled NPZ file.")
        convert.save_letters_npz(
            units, stimulus_list, filename, append,
            partial(glia.group_contains, "TILED_LETTER"))
    elif name=='10faces':
        # there's a bug in this program and a single WAIT is missing metadata
        raise ValueError("Deprecated--use glia commit from before 2020-08-20")
    elif ('faces' in name) or ('ffhq' in name):
        print("Saving faces h5 file.")
        convert.save_images_h5(
            units, stimulus_list, filename,
            frame_log, video_file, append)
    elif name=='eyechart-saccade':
        print("Saving eyechart-saccade NPZ file.")
        convert.save_acuity_image_npz(
            units, stimulus_list, filename, append)
    elif name=='letters-saccade':
        print("Saving letters-saccade NPZ file.")
        convert.save_acuity_image_npz(
            units, stimulus_list, filename, append)
    elif name=='checkerboard':
        convert.save_checkerboard_npz(
            units, stimulus_list, filename, append,
            lambda x: 'ONE CONDITION',
            quad)
            # partial(debug_lambda, f=lambda x: 'ONE CONDITION'))
    elif name=='checkerboard-contrast':
        convert.save_checkerboard_npz(
            units, stimulus_list, filename, append,
            lambda x: glia.checkerboard_contrast(x),
            quad)
    elif name=='checkerboard-flicker':
        convert.save_checkerboard_flicker_npz(
            units, stimulus_list, filename, append,
            lambda x: glia.checkerboard_contrast(x),
            quad)
    elif name=='checkerboard-durations':
        convert.save_checkerboard_npz(
            units, stimulus_list, filename, append,
            lambda x: x["lifespan"],
            quad)
    elif name=='grating':
        convert.save_grating_npz(
            units, stimulus_list, filename, append,
            lambda x: 'ONE CONDITION')
    elif name=='grating-speeds':
        convert.save_grating_npz(
            units, stimulus_list, filename, append,
            lambda x: x["speed"])
    elif name=='grating-contrast':
        convert.save_grating_npz(
            units, stimulus_list, filename, append,
            lambda x: glia.bar_contrast(x))
    elif name=='grating-durations':
        convert.save_grating_npz(
            units, stimulus_list, filename, append,
            lambda x: x["lifespan"])
    elif name=='grating-sinusoidal':
        convert.save_grating_npz(
            units, stimulus_list, filename, append,
            lambda x: 'ONE CONDITION',
            sinusoid=True)
    elif name=='grating-sinusoidal-speeds':
        convert.save_grating_npz(
            units, stimulus_list, filename, append,
            lambda x: x["speed"],
            sinusoid=True)
    elif name=='grating-sinusoidal-contrast':
        convert.save_grating_npz(
            units, stimulus_list, filename, append,
            lambda x: glia.bar_contrast(x),
            sinusoid=True)
    elif name=='grating-sinusoidal-durations':
        convert.save_grating_npz(
            units, stimulus_list, filename, append,
            lambda x: x["lifespan"],
            sinusoid=True)
    elif name=='eyechart':
        if append==0:
            append = 0.5
        convert.save_eyechart_npz(
            units, stimulus_list, filename, append)
    else:
        raise(ValueError(f'Unknown name {name}'))

generate.add_command(convert_cmd)

def strip_generated(name, choices=generate_choices):
    for prefix in generate_choices:
        length = len(prefix)
        if prefix==name[0:length]:
            # strip `prefix_`
            return name[length+1:]
    return name


@main.command("process")
@click.argument('filename', type=str, default=None)
@click.option("--configuration", "-c", type=click.Path(exists=True), help="""Use
    configuration file for analog calibration, etc.""")
@click.option("--analog-idx", "-i", type=int, help="Channel of light detector", default=1)
@click.option("--threshold", "-r", type=float, default=9, help="Set the threshold for flicker")
@click.option("--ignore-extra",  is_flag=True, help="Ignore extra stimuli if stimulus list is longer than detected start times in analog file.")
@click.option("--notebook", "-n", type=click.Path(exists=True))
@click.option("--eyecandy", "-e", default="http://localhost:3000")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--debug", "-vv", is_flag=True)
@click.pass_context
def process_cmd(ctx, filename, notebook, eyecandy, configuration=None, debug=False,
        analog_idx=1, threshold=9, ignore_extra=False, verbose=False):
    "Process analog + frames log to align stim/ephys in .frames file."
    # TODO: deduplicate code from analyze command..
    # problem: hard to do this due to click ctx object...perhaps can pass to function...?
    init_logging(filename, processes=None, verbose=verbose, debug=debug)
    #### FILEPATHS
    logger.debug(str(filename) + "   " + str(os.path.curdir))
    if not os.path.isfile(filename):
        try:
            filename = glia.match_filename(filename,"txt")
        except:
            try:
                filename = glia.match_filename(filename,"bxr")
            except:
                filename = glia.match_filename(filename,"csv")
    data_directory, data_name = os.path.split(filename)
    name, extension = os.path.splitext(data_name)
    analog_file = os.path.join(data_directory, name +'.analog')
    if not os.path.isfile(analog_file):
        # use 3brain analog file
        analog_file = os.path.join(data_directory, name +'.analog.brw')
        
    if not os.path.isfile(analog_file):
        # Tyler's format; used if files were split for example
        analog_file = os.path.join(data_directory, name +'.analog.npz')

    stimulus_file = os.path.join(data_directory, name + ".stim")
    ctx.obj = {"filename": os.path.join(data_directory,name)}
    
    if configuration!=None:
        with open(configuration, 'r') as f:
            user_config = yaml.safe_load(f)
        config.user_config = user_config
        if "analog_calibration" in user_config:
            config.analog_calibration = user_config["analog_calibration"]
        if "notebook" in user_config:
            notebook = user_config["notebook"]
        if "eyecandy" in user_config:
            eyecandy = user_config["eyecandy"]
        if "processes" in user_config:
            processes = user_config["processes"]
        if "integrity_filter" in user_config:
            integrity_filter = user_config["integrity_filter"]
        if "by_channel" in user_config:
            by_channel = user_config["by_channel"]

    if not notebook:
        notebook = glia.find_notebook(data_directory)

    lab_notebook = glia.open_lab_notebook(notebook)
    logger.info(f"{name=}")
    experiment_protocol = glia.get_experiment_protocol(lab_notebook, name)
    flicker_version = experiment_protocol["flickerVersion"]


    #### LOAD STIMULUS
    try:
        metadata, stimulus_list, method = glia.read_stimulus(stimulus_file)
        ctx.obj["stimulus_list"] = stimulus_list
        ctx.obj["metadata"] = metadata
        # assert method=='analog-flicker'
    except:
        trigger = "flicker"
        print("No .stim file found. Creating from .analog file.".format(trigger))
        if flicker_version==0.3:
            metadata, stimulus_list = glia.create_stimuli(
                analog_file, stimulus_file, notebook, name, eyecandy, analog_idx, ignore_extra,
                config.analog_calibration, threshold)
            ctx.obj["stimulus_list"] = stimulus_list
            ctx.obj["metadata"] = metadata
            print('finished creating .stim file')
        elif trigger == "ttl":
            raise ValueError('not implemented')
        else:
            raise ValueError("invalid trigger: {}".format(trigger))

    if not notebook:
        notebook = glia.find_notebook(data_directory)
        
    ## END of code duplication
    
    lab_notebook = glia.open_lab_notebook(notebook, convert_types=False)
    experiment_protocol = glia.get_experiment_protocol(lab_notebook, name)
    
    date_prefix = (data_directory + experiment_protocol['date']).replace(':','_')
    frame_log_file = date_prefix + "_eyecandy_frames.log"
    video_file = date_prefix + "_eyecandy.mkv"

    
    container = av.open(str(video_file))
    n_video_frames = 0
    for _ in container.decode(video=0):
        n_video_frames += 1
    
    
    stimulus_list = glia.read_stimulus(stimulus_file)

    if analog_file[-4:]==".brw":
        analog = glia.read_3brain_analog(analog_file)
    elif analog_file[-11:]==".analog.npz":
        analog = np.load(analog_file)["analog"]
    else:
        analog = glia.read_raw_voltage(analog_file)[:,1]
    sampling_rate = glia.sampling_rate(analog_file)
    
    analog_std = 0.5*analog.std() + analog.min()
    # beginning of experiment
    # TODO: after clustering, should subtract known / estimated latency for better frame time..?
    # for maximum temporal accuracy, frame should begin at start of slope
    approximate_start_idx = np.where(analog > analog_std)[0][0]
    baseline_offset = int(sampling_rate/10) # rise started before experiment_start_idx
    # we add 3sigma of baseline to baseline.max() to create threshold for end of experiment
    baseline_thresh = np.max(analog[:approximate_start_idx-baseline_offset]) \
                    + np.std(analog[:approximate_start_idx-baseline_offset])*3
    experiment_start_idx = np.where(analog > baseline_thresh)[0][0]
    
    frame_log = pd.read_csv(frame_log_file)
    
    nframes_in_log = len(frame_log)
    if np.abs(n_video_frames - nframes_in_log) > 1:
        logger.warn(f"found {n_video_frames} video frames, but {nframes_in_log} frames in log")
    assert np.abs(n_video_frames - nframes_in_log) < 2
    # assert n_video_frames == nframes_in_log or n_video_frames + 1 == nframes_in_log 
    # gross adjustment for start time
    frame_log.time = (frame_log.time - frame_log.time[0])/1000 + experiment_start_idx/sampling_rate

    # finer piecewise linear adjustments for each stimulus start frame
    # I've seen this off by 50ms after a 4.4s bar stimulus!
    newStimFrames = np.where(frame_log.stimulusIndex.diff())[0]
    stim_start_diff = np.abs(frame_log.iloc[newStimFrames].time.diff()[1:] - np.diff(np.array(list(map(lambda s: s["start_time"], stimulus_list[1])))))
    max_time_diff = stim_start_diff.max()
    print("stimulus start sum_time_diff", max_time_diff)
    print("stimulus start mean_time_diff", stim_start_diff.mean())
    assert len(newStimFrames) == len(stimulus_list[1])
    for n,stim in enumerate(stimulus_list[1]):
        flickerTime = stim["start_time"]
        frameNum = newStimFrames[n]
        if n +1 < len(stimulus_list[1]):
            nextFrameNum = newStimFrames[n+1]
            # we adjust all frames in a stimulus
            loc = (frame_log.framenum >= frameNum) & (frame_log.framenum < nextFrameNum)
        else:
            loc = (frame_log.framenum >= frameNum)
        frame_log_time = frame_log.loc[frameNum].time
        time_delta = flickerTime - frame_log_time
        frame_log.loc[loc, 'time'] += time_delta

    stim_start_diff = np.abs(frame_log.iloc[newStimFrames].time.diff()[1:] - np.diff(np.array(list(map(lambda s: s["start_time"], stimulus_list[1])))))
    max_time_diff = stim_start_diff.max()
    print("post alignment stimulus start sum_time_diff", max_time_diff)
    assert max_time_diff < 0.001
    # frame_log.head()
    
    name, _ = os.path.splitext(frame_log_file)
    frame_log.to_csv(name + ".frames", index=False)
    print(f"Saved to {name + '.frames'}")
    

@main.command("classify")
@click.argument('filename', type=str, default=None)
@click.option('--nsamples', "-n", type=int, default=0,
    help="Show results for n different sample numbers.")
# @click.option("--letter", default=False, is_flag=True,
#     help="")
# @click.option("--integrity", default=False, is_flag=True,
#     help="")
@click.option("--notebook", "-n", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True)
@click.option("--debug", "-vv", is_flag=True)
@click.option("--processes", "-p", type=int, help="Number of processors")
@click.option('--skip', "-s", default=False, is_flag=True,
    help="Skip method assertion (for testing)")
@click.option("--n-draws", "-d", type=int, help="Number of draws for Monte Carlo Cross-validation", default=30)
@click.option("--px-per-deg", "-p", type=float, default=10.453,
              help="Measured pixels per degree")
# @click.option("--eyechart", default=False, is_flag=True,
#     help="")
# @click.option("--letter", default=False, is_flag=True,
#     help="Output npz for letter classification")
def classify_cmd(filename, nsamples, notebook, skip, debug=False, verbose=False,
                 version=2, processes=None, n_draws=30, px_per_deg=10.453):
    "Classify using converted NPZ"

    if not os.path.isfile(filename):
        filename = glia.match_filename(filename, 'npz')
    try:
        assert filename[-4:]==".npz"
    except:
        print("Please specify a npz file.")
        raise

    data_directory, data_name = os.path.split(filename)
    if data_directory=='':
        data_directory=os.getcwd()

    if not notebook:
        notebook = glia.find_notebook(data_directory)

    lab_notebook = glia.open_lab_notebook(notebook)

    name, extension = os.path.splitext(data_name)
    init_logging(name, processes, verbose, debug)
    stim_name = strip_generated(name)
    stimulus_file = os.path.join(data_directory, stim_name + ".stim")
    metadata, stimulus_list, method = glia.read_stimulus(stimulus_file)
    if not skip:
        assert method=='analog-flicker' # if failed, delete .stim

    data = np.load(filename)
    shape = np.shape(data['training_data'])
    logger.debug(f"Data dim: {shape}")

    plots_directory = os.path.join(data_directory, name+"-plots")
    os.makedirs(plots_directory, exist_ok=True)
    plot_directory = os.path.join(plots_directory,"00-all")
    os.makedirs(plot_directory, exist_ok=True)


    # if letter:
    #     safe_run(classify.save_letter_npz_v2,
    #         (units, stimulus_list, name))
    # elif integrity:
    #     safe_run(convert.save_integrity_npz,
    #         (units, stimulus_list, name))
    name = metadata['name']
    if re.match('checkerboard',name):
        svc.checkerboard_svc(
            data, metadata, stimulus_list, lab_notebook, plot_directory,
             nsamples, n_draws, px_per_deg=px_per_deg)
    elif re.match('grating-sinusoidal',name):
        svc.grating_svc(
            data, metadata, stimulus_list, lab_notebook, plot_directory,
             nsamples, n_draws, sinusoid=True, px_per_deg=px_per_deg)
    elif re.match('grating',name):
        svc.grating_svc(
            data, metadata, stimulus_list, lab_notebook, plot_directory,
            nsamples, n_draws, px_per_deg=px_per_deg)
    elif 'faces' in name:
        # dict with entries like '[37, False, True]': 0
        class_resolver = data['class_resolver'].item()
        nclasses = np.array(list(class_resolver.values())).max()+1
        id_map = {i: i for i in np.arange(nclasses)}
        num2gender = {}
        num2smiling = {}
        num2person = {}
        for class_str, num in class_resolver.items():
            temp = class_str[1:-1].split(", ")
        #     print(temp)
            image_class = [int(temp[0]), temp[1]=="True", temp[2]=="True"]
            num2gender[num] = image_class[1]
            num2smiling[num] = image_class[2]
            num2person[num] = image_class[0]
        target_mappers = [id_map, num2gender, num2person, num2smiling]
        mapper_classes = [np.arange(nclasses),
                         ["Female", "Male"],
                         np.arange(20),
                         ["Not smiling", "Smiling"]]
        mapper_names = ["by_image", "is_male", "by_person",
                        "is_smiling"]
        svc.generic_image_classify(
            data, metadata, stimulus_list, lab_notebook, plot_directory,
             nsamples, target_mappers, mapper_classes, mapper_names)
    elif 'letters-tiled'==name:
        svc.tiled_letter_svc(
            data, metadata, stimulus_list, lab_notebook, plot_directory,
             nsamples, px_per_deg=px_per_deg)
    elif 'eyechart-saccade'==name:
        svc.image_svc(
            data, metadata, stimulus_list, lab_notebook, plot_directory,
             nsamples, px_per_deg=px_per_deg)
    elif 'letters-saccade'==name:
        svc.image_svc(
            data, metadata, stimulus_list, lab_notebook, plot_directory,
             nsamples, px_per_deg=px_per_deg)
    elif re.match('letter',name):
        svc.letter_svc(
            data, metadata, stimulus_list, lab_notebook, plot_directory,
             nsamples, px_per_deg=px_per_deg)
    else:
        raise(ValueError(f"unknown name: {name}"))
    # elif eyechart:
    #     safe_run(convert.save_eyechart_npz,
    #         (units, stimulus_list, name))

generate.add_command(convert_cmd)


@analyze.command("raster")
@plot_function
def raster_cmd(units, stimulus_list, metadata, c_unit_fig, c_retina_fig):
    raster.save_raster(
        units, stimulus_list, partial(c_unit_fig,"raster"), c_retina_fig)

@analyze.command("integrity")
@click.option("--version", "-v", type=str, default="2")
@plot_function
def integrity_cmd(units, stimulus_list, metadata, c_unit_fig, c_retina_fig, version=1):
    if version=="1":
        solid.save_integrity_chart(
            units, stimulus_list, partial(c_unit_fig,"integrity"),
                c_retina_fig)
    elif version=="2":
        solid.save_integrity_chart_v2(
            units, stimulus_list, partial(c_unit_fig,"integrity"),
                c_retina_fig)
    elif version=="fail":
        solid.save_integrity_chart_vFail(
            units, stimulus_list, partial(c_unit_fig,"integrity"),
                c_retina_fig)

generate.add_command(integrity_cmd)


@analyze.command("grating")
@click.option("-w", "--width", type=int,
    help="Manually provide screen width for old versions of Eyecandy")
@click.option("-h", "--height", type=int,
    help="Manually provide screen height for old versions of Eyecandy")
@plot_function
def grating_cmd(units, stimulus_list, metadata, c_unit_fig, c_retina_fig, width, height):
    grating.save_unit_spike_trains(
        units, stimulus_list, c_unit_fig, c_retina_fig, width, height)


@analyze.command("acuity")
@click.option("--prepend", "-p", type=float, default=1,
    help="plot (seconds) before SOLID start time")
@click.option("--append", "-a", type=float, default=1,
    help="plot (seconds) after SOLID end time")
@click.option("--version", "-v", type=float, default=3)
@plot_function
def acuity_cmd(units, stimulus_list, metadata, c_unit_fig, c_retina_fig,
        prepend, append, version):
    acuity.save_acuity_chart(
        units, stimulus_list, c_unit_fig, c_retina_fig,
            prepend, append)

generate.add_command(acuity_cmd)

#%%
@main.command()
@click.argument('files', type=click.Path(exists=True), nargs=-1)
@click.argument('output', type=str, nargs=1)
def combine(files, output):
    """Combine multiple .brw files into a single .brw file.

    Useful to spike sort multiple recordings as if continuous.
    """
    # read all files
    bxrs = [h5py.File(f,'r') for f in files]
    # some paths we might care about & will copy
    metadata_paths = [
        '3BRecInfo/3BRecVars/MaxVolt',
        '3BRecInfo/3BRecVars/MinVolt',
        '3BRecInfo/3BRecVars/BitDepth',
        '3BRecInfo/3BRecVars/SignalInversion',
        '3BRecInfo/3BRecVars/SamplingRate',
        '3BRecInfo/3BRecVars/ExperimentType',
        '3BRecInfo/3BMeaChip/NRows',
        '3BRecInfo/3BMeaChip/NCols',
        '3BRecInfo/3BMeaChip/Layout',
        '3BRecInfo/3BMeaChip/MeaType',
        '3BRecInfo/3BMeaSystem/FwVersion',
        '3BRecInfo/3BMeaSystem/HwVersion',
        '3BRecInfo/3BMeaSystem/System'
    ]

    # count n_frames, n_samples from each file
    # also verify that key metadata matches
    n_frames = bxrs[0]['3BRecInfo/3BRecVars/NRecFrames'][0]
    n_samples = [bxrs[0]['3BData/Raw'].shape[0]]
    sampling_rate = bxrs[0]['3BRecInfo/3BRecVars/SamplingRate'][0]
    print("checking that all brw files have matching metadata")
    for b in bxrs[1:]:
        for m in metadata_paths:
            try:
                if len(bxrs[0][m])==1:
                    assert bxrs[0][m][:] == b[m][:]
                else:
                    assert np.all(bxrs[0][m][:] == b[m][:])
            except Exception as E:
                logger.warn(f"""metadata does not match for {m}:
                found {bxrs[0][m]} and {b[m]}
                """)
        n_frames += b['3BRecInfo/3BRecVars/NRecFrames'][0]
        n_samples.append(b["3BData/Raw"].shape[0])
    print(f"combined duration: {n_frames/sampling_rate/60:.2f} minutes")

    out_bxr = h5py.File(output, "w")
    # copy metadata
    bxrs[0].visititems(partial(glia.copy_metadata, copy_to=out_bxr))

    # copy data
    out_bxr['3BRecInfo/3BRecVars/NRecFrames'] = [n_frames]
    out_bxr['nSamplesPerRecording'] = n_samples
    tot_samples = sum(n_samples)
    assert np.isclose(tot_samples/n_frames, 4096) #4096 channels
    
    # copy raw data
    raw_dtype = bxrs[0]["3BData/Raw"].dtype
    dset = out_bxr.create_dataset("3BData/Raw", (tot_samples,),
        dtype=raw_dtype)
    start_sample = 0
    max_chunk = int(1e8) # <1GiB 
    for i, b in enumerate(bxrs):
        print(f"Copying {files[i]}")
        end_sample = start_sample+n_samples[i]
        for s in tqdm(range(0,n_samples[i],max_chunk)):
            e = min(s+max_chunk, end_sample)
            dset[start_sample+s:start_sample+e] = b["3BData/Raw"][s:e]
        start_sample = end_sample

    # cleanup
    out_bxr.close()
    [b.close() for b in bxrs]

@main.command()
@click.argument('analog_file', type=click.Path(exists=True), nargs=1)
def length(analog_file):
    """Get length (in samples) of a 3brain analog file.
    
    Useful for getting nsamples as argument for `glia split`."""
    if analog_file[-10:] == 'analog.brw':
        with h5py.File(analog_file, 'r') as file:
            print(len(file["3BData"]["Raw"]))
    else:
        raise NotImplementedError("Only for use with *analog.brw files")
    

@main.command()
@click.argument('filepath', type=click.Path(exists=True), nargs=1)
@click.argument('nsamples', type=int, nargs=-1)
def split(filepath, nsamples):
    """Split single .brw or .bxr file into multiple .brw or .bxr files.
    
    Can pass multiple NSAMPLES, i.e. pass 5 integers to split into 5 files.

    Useful if multiple experiments were spike-sorted together and now you want
    to run through glia.
    
    Creates a _spikes.csv for each split, as well as a single _channel_map.npy
    """
    start = np.cumsum([0] + list(nsamples[:-1]))
    if filepath[-10:] == 'analog.brw':
        filename = filepath[:-10]
        analog = read_3brain_analog(filepath)
        for i, (s,n) in enumerate(zip(start, nsamples)):
            name = f"{filename}_part_{i}_analog.npz"
            print(f"Saving {name}")
            sampling_rate = glia.sampling_rate(filepath)
            np.savez(name, analog=analog[s:s+n],
                     sampling_rate=sampling_rate)
    elif filepath[-4:] == ".bxr":
        filename = filepath[:-4]
        # split spike-sorted data
        with h5py.File(filepath, 'r') as h5:
            # shared setup for the concatenated arrays
            sampling_rate = float(h5["3BRecInfo"]["3BRecVars"]["SamplingRate"][0])
            channel_map = h5["3BRecInfo"]["3BMeaStreams"]["Raw"]["Chs"][()]
            
            # map 3brain unit num
            # numbers typically from -4 to 9000
            # where negative numbers appear across multiple channels
            # and thus are presumably bad units...?
            # positive-numbered units appear on one channel
            unit_id_2_num = {}

            n_unit_nums = 0
            for chunk in iter_chunks(h5['3BResults/3BChEvents/SpikeUnits'], 10000):
                n_unit_nums = max(n_unit_nums, chunk.max())
            
            unit_map = {}
            channel_unit_count = {}


            # operate on each of the concatenated arrays, one at a time
            for i, (s,n) in enumerate(zip(start, nsamples)):
                startTime = s / sampling_rate
                first_idx = None
                for chunk in iter_chunks(h5['3BResults/3BChEvents/SpikeTimes'], 10000):
                    valid_idxs = np.argwhere(h5["3BResults/3BChEvents/SpikeTimes"] > s)
                    if len(valid_idxs) > 0:
                        first_idx = valid_idxs[0][0]
                        break
                assert not first_idx is None
                print(f"identified start idx of {first_idx}.")

                # for simplicity, we just iterate again, could have faster implementation
                last_idx = len(h5['3BResults/3BChEvents/SpikeTimes'])
                chunk_size = 10000
                for j, chunk in enumerate(iter_chunks(h5['3BResults/3BChEvents/SpikeTimes'], chunk_size)):
                    invalid_idxs = np.argwhere(chunk > s + n)
                    if len(invalid_idxs) > 0:
                        last_idx = invalid_idxs[0][0] + j*chunk_size
                        break
                print(f"identified stop idx of {last_idx}.")
                
                spike_channel_ids = h5["3BResults"]["3BChEvents"]["SpikeChIDs"][first_idx:last_idx]
                spike_unit_ids = h5["3BResults"]["3BChEvents"]["SpikeUnits"][first_idx:last_idx]
                # poorly named; time is in units of 1/sampling_rate
                # aka sample number
                # subtract to adjust start time
                spike_times = h5["3BResults"]["3BChEvents"]["SpikeTimes"][first_idx:last_idx] - s
                                

                    
                csv_name = f'{filename}_part_{i}_spikes.csv'
                spikes = zip(spike_channel_ids, spike_unit_ids, spike_times)
                tot_spikes = spike_times.shape[0]
                print(f"creating {csv_name} ...")
                with open(csv_name, 'w', newline='') as csvfile:
                    fieldnames = ['channel_i', 'channel_j', 'unit', "spike_time"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writeheader()
                    for channel, unit_id, spike_time in tqdm(spikes,
                                                             total=tot_spikes):
                        c = channel_map[channel]
                        # convert to tuple
                        # account for 1-indexing
                        c = (c[0]-1,c[1]-1)
                        
                        # count num units on channel
                        # first check if we've seen this channel before
                        if not c in channel_unit_count:
                            # if not, initialize channel_unit_count for the channel
                            channel_unit_count[c] = 1
                            unit_num = 0
                            # add unit
                            unit_id_2_num[unit_id] = unit_num
                        else:
                            
                            # then check if we've seen this unit before
                            if not unit_id in unit_id_2_num:
                                # if not, assign unit_num for this new unit
                                unit_num = channel_unit_count[c]
                                unit_id_2_num[unit_id] = unit_num
                                channel_unit_count[c] += 1
                            else:
                                # otherwise, look it up
                                unit_num = unit_id_2_num[unit_id]
                                
                                
                        t = spike_time / sampling_rate
                        writer.writerow({"channel_i": c[0],
                            "channel_j": c[1],
                            "unit": unit_num,
                            "spike_time": t})
                        
                np.save(f"{filename}_channel_map.npy", channel_map)


if __name__ == '__main__':
    try:
        main()
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
