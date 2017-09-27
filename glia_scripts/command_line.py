#!/usr/bin/env python

import matplotlib
matplotlib.use("agg")
matplotlib.rcParams['figure.max_open_warning'] = 250

import glia
from fnmatch import fnmatch
import click
import os
import sys
import re
import glia_scripts.solid as solid
import glia_scripts.bar as bar
import glia_scripts.acuity as acuity
import glia_scripts.grating as grating
import glia_scripts.raster as raster
import glia_scripts.convert as convert
import errno
from glia_scripts.classify import svc
import traceback
import glia.config as config
from glia.config import logger, logging, channel_map
from functools import update_wrapper, partial
# from tests.conftest import display_top, tracemalloc


from glob import glob
from glia.types import Unit
from matplotlib.backends.backend_pdf import PdfPages
from random import randint
import numpy as np
import yaml
import cProfile


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



def safe_run(function, args):
    try:
        function(*args)
    except Exception as exception:
        # traceback.print_tb(exception.__traceback__)
        # traceback.print_exception(exception)
        logger.exception("Error running {}. Skipping".format(str(function)))

def plot_path(directory,plot_name):
    return os.path.join(directory,plot_name+".png")

@click.group()
def main():
    pass

def match_filename(start,ext='txt'):
    files = glob(start + "*." + ext)
    if len(files)==1:
        return files[0]
    else:
        raise(ValueError("Could not match file, try specifying full filename"))

@main.command()
@click.argument('filename', type=str)
def header(filename):
    """Print the header length from a Multichannel systems binary file.

    Commonly used for import into spike sorting software"""

    if not os.path.isfile(filename):
        filename = match_filename(filename,'voltages')
    try:
        print('header length: ', glia.get_header(filename)[1])
    except:
        raise(ValueError, "Could not get header, are you sure it's a MCD binary export?")

generate_choices = ["random","hz"]
@main.group(chain=True)
@click.argument('filename', type=str)
@click.option("--notebook", "-n", type=click.Path(exists=True))
@click.option("--eyecandy", "-e", default="http://localhost:3000")
@click.option('method', "-m",
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
def generate(ctx, filename, eyecandy, method, notebook, number,
    nunits, stimulus):
    data_directory, data_name = os.path.split(filename)
    if data_directory=='':
        data_directory=os.getcwd()

    if not notebook:
        notebook = find_notebook(data_directory)

    lab_notebook = glia.open_lab_notebook(notebook)
    if not os.path.isfile(filename):
        filename = match_filename(filename,"txt")
    name, ext = os.path.splitext(filename)

    ctx.obj = {'filename': method+"_"+name}

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
    retina_id = f'{method}_{name}'
    print('generating test data')
    for channel_x in range(number):
        for channel_y in range(number):
            # for unit_j in range(randint(1,5)):
            for unit_j in range(nunits):
                if method=='random':
                    u = glia.random_unit(total_time, retina_id,
                        (channel_x, channel_y), unit_j)
                elif method=="hz":
                    # hz = randint(1,90)
                    hz = 60
                    u = glia.hz_unit(total_time, hz, retina_id,
                        (channel_x, channel_y), unit_j)

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

def find_notebook(directory):
    notebooks = glob(os.path.join(directory, 'lab*.yml')) + \
        glob(os.path.join(directory, 'lab*.yaml'))
    if len(notebooks)==0:
        raise ValueError("no lab notebooks (.yml) were found. Either add to directory," \
            "or specify file path with -n.")
    elif len(notebooks)>1:
        logger.warning(f"""Found multiple possible lab notebooks.
        Using {notebooks[0]}. If wrong, try manually specifying""")
    return notebooks[0]

def init_logging(name, data_directory, processes, verbose, debug):
    #### LOGGING CONFIGURATION
    fh = logging.FileHandler(os.path.join(data_directory,name + '.log'))
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


@main.group(chain=True)
@click.argument('filename', type=str)
# @click.argument('filename', type=click.Path(exists=True))
@click.option("--notebook", "-n", type=click.Path(exists=True))
@click.option("--eyecandy", "-e", default="http://localhost:3000")
@click.option("--processes", "-p", type=int, help="Number of processors")
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
@click.option("--debug", "-vv", is_flag=True)
@click.option("--trigger", "-t", type=click.Choice(["flicker", 'detect-solid', "legacy", "ttl"]), default="flicker",
    help="""Use flicker if light sensor was on the eye candy flicker, solid if the light sensor detects the solid stimulus,
    or ttl if there is a electrical impulse for each stimulus.
    """)
@click.pass_context
def analyze(ctx, filename, trigger, threshold, eyecandy, ignore_extra=False,
        fix_missing=False, output=None, notebook=None,
        configuration=None, verbose=False, debug=False,processes=None,
        by_channel=False, integrity_filter=0.0):
    """Analyze data recorded with eyecandy.
    """
    #### FILEPATHS
    if not os.path.isfile(filename):
        filename = match_filename(filename,"txt")
    data_directory, data_name = os.path.split(filename)
    name, extension = os.path.splitext(data_name)
    analog_file = os.path.join(data_directory, name +'.analog')
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
        notebook = find_notebook(data_directory)

    init_logging(name, data_directory, processes, verbose, debug)

    lab_notebook = glia.open_lab_notebook(notebook)
    logger.info(name)
    experiment_protocol = glia.get_experiment_protocol(lab_notebook, name)
    flicker_version = experiment_protocol["flickerVersion"]


    #### LOAD STIMULUS
    try:
        metadata, stimulus_list, method = glia.read_stimulus(stimulus_file)
        ctx.obj["stimulus_list"] = stimulus_list
        ctx.obj["metadata"] = metadata
        assert method=='analog-flicker'
    except:
        print("No .stim file found. Creating from .analog file.".format(trigger))
        if flicker_version==0.3:
            metadata, stimulus_list = glia.create_stimuli(
                analog_file, stimulus_file, notebook, name, eyecandy, ignore_extra,
                config.analog_calibration, threshold)
            ctx.obj["stimulus_list"] = stimulus_list
            ctx.obj["metadata"] = metadata
            print('finished creating .stim file')
        elif trigger == "ttl":
            raise ValueError('not implemented')
        else:
            raise ValueError("invalid trigger: {}".format(trigger))

    #### LOAD SPIKES
    spyking_regex = re.compile('.*\.result.hdf5$')
    eye = experiment_protocol['eye']
    experiment_n = experiment_protocol['experimentNumber']

    date = experiment_protocol['date'].date().strftime("%y%m%d")

    retina_id = date+'_R'+eye+'_E'+experiment_n
    if extension == ".txt":
        ctx.obj["units"] = glia.read_plexon_txt_file(filename,retina_id, channel_map)
    elif re.match(spyking_regex, filename):
        ctx.obj["units"] = glia.read_spyking_results(filename)
    else:
        raise ValueError('could not read {}. Is it a plexon or spyking circus file?')

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
        # c connotes 'continuation'
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
def cleanup(ctx, results, filename, trigger, threshold, eyecandy, ignore_extra=False,
        fix_missing=False, output=None, notebook=None,
        configuration=None, version=None, verbose=False, debug=False,processes=None,
        by_channel=False, integrity_filter=0.0):
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
@analysis_function
def convert_cmd(units, stimulus_list, metadata, filename, version=2):
    name = metadata['name']
    if name=='letters':
        convert.save_letter_npz(
            units, stimulus_list, filename)
    elif name=='letters-tiled':
        convert.save_letters_tiled_npz(
            units, stimulus_list, filename)
    elif name=='checkerboard':
        convert.save_checkerboard_npz(
            units, stimulus_list, filename,
            lambda x: 'ONE CONDITION')
            # partial(debug_lambda, f=lambda x: 'ONE CONDITION'))
    elif name=='checkerboard-contrast':
        convert.save_checkerboard_npz(
            units, stimulus_list, filename,
            lambda x: glia.checkerboard_contrast(x))
    elif name=='checkerboard-durations':
        convert.save_checkerboard_npz(
            units, stimulus_list, filename,
            lambda x: x["lifespan"])
    elif name=='grating':
        convert.save_grating_npz(
            units, stimulus_list, filename,
            lambda x: 'ONE CONDITION')
    elif name=='grating-speeds':
        convert.save_grating_npz(
            units, stimulus_list, filename,
            lambda x: x["speed"])
    elif name=='grating-durations':
        convert.save_grating_npz(
            units, stimulus_list, filename,
            lambda x: x["lifespan"])
    elif name=='eyechart':
        convert.save_eyechart_npz(
            units, stimulus_list, filename)
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
# @click.option("--eyechart", default=False, is_flag=True,
#     help="")
# @click.option("--letter", default=False, is_flag=True,
#     help="Output npz for letter classification")
def classify_cmd(filename, nsamples, notebook, skip, debug=False,
                 verbose=False, version=2, processes=None):
    "Classify using converted NPZ"

    if not os.path.isfile(filename):
        filename = match_filename(filename, 'npz')

    data_directory, data_name = os.path.split(filename)
    if data_directory=='':
        data_directory=os.getcwd()

    if not notebook:
        notebook = find_notebook(data_directory)

    lab_notebook = glia.open_lab_notebook(notebook)

    name, extension = os.path.splitext(data_name)
    init_logging(name, data_directory, processes, verbose, debug)
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
             nsamples)
    elif re.match('grating',name):
        svc.grating_svc(
            data, metadata, stimulus_list, lab_notebook, plot_directory,
             nsamples)
    elif 'letters-tiled'==name:
        svc.tiled_letter_svc(
            data, metadata, stimulus_list, lab_notebook, plot_directory,
             nsamples)
    elif re.match('letter',name):
        svc.letter_svc(
            data, metadata, stimulus_list, lab_notebook, plot_directory,
             nsamples)
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

if __name__ == '__main__':
    main()
