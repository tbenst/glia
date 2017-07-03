#!/usr/bin/env python

import matplotlib
matplotlib.use("agg")
matplotlib.rcParams['figure.max_open_warning'] = 250



import glia
import fnmatch
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



def plot_function(f):
    @click.pass_context
    def new_func(ctx, *args, **kwargs):
        context_object = ctx.obj
        return ctx.invoke(f, ctx.obj["units"], ctx.obj["stimulus_list"],
            ctx.obj["c_unit_fig"], ctx.obj["c_retina_fig"],
            *args[3:], **kwargs)
    return update_wrapper(new_func, f)

def analysis_function(f):
    @click.pass_context
    def new_func(ctx, *args, **kwargs):
        context_object = ctx.obj
        return ctx.invoke(f, ctx.obj["units"], ctx.obj["stimulus_list"],
            ctx.obj['filename'],
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
    help="Create .stim file")
@click.pass_context
def generate(ctx, filename, eyecandy, method, notebook, number,
    nunits, stimulus):
    data_directory, data_name = os.path.split(filename)
    if data_directory=='':
        data_directory=os.getcwd()

    if not notebook:
        notebook = find_notebook(data_directory)

    lab_notebook = glia.open_lab_notebook(notebook)
    name=None
    for doc in lab_notebook:
        n= doc['filename']
        if fnmatch.fnmatch(n , filename+'*'):
            name=n
            break
    assert name is not None

    ctx.obj = {'filename': method+"_"+name}

    if stimulus:
        stimulus_file = os.path.join(data_directory, name + ".stim")
        try:
            stimulus_list = glia.read_stimulus(stimulus_file)
            print('found .stim file')
        except:
            print('creating .stim file.')
            stimulus_list = glia.create_stimulus_list_without_analog(stimulus_file,
                notebook, name, eyecandy)
    else:
        stimulus_file = os.path.join(data_directory, name + ".stimulus")
        stimulus_list = glia.load_stimulus(stimulus_file)

    ctx.obj["stimulus_list"] = stimulus_list
    # total_time = sum(map(lambda x: x['stimulus']['lifespan'], stimulus_list))
    last_stim = stimulus_list[-1]
    total_time = last_stim['start_time']+last_stim['stimulus']['lifespan']
    units = {}
    retina_id = 'test'
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
    plot_directory = os.path.join(data_directory, name+"-plots")
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
    notebooks = glob(os.path.join(directory, '*.yml')) + \
        glob(os.path.join(directory, '*.yaml'))
    if len(notebooks)==0:
        raise ValueError("no lab notebooks (.yml) were found. Either add to directory," \
            "or specify file path with -n.")
    return notebooks[0]

def find_stim(name):
    notebooks = glob(os.path.join(directory, '*.yml')) + \
        glob(os.path.join(directory, '*.yaml'))
    if len(notebooks)==0:
        raise ValueError("no lab notebooks (.yml) were found. Either add to directory," \
            "or specify file path with -n.")
    return notebooks[0]

@main.group(chain=True)
@click.argument('filename', type=str)
# @click.argument('filename', type=click.Path(exists=True))
@click.option("--notebook", "-n", type=click.Path(exists=True))
@click.option("--eyecandy", "-e", default="http://localhost:3000")
@click.option("--processes", "-p", type=int, help="Number of processors")
@click.option("--calibration", "-c", default=(0.55,0.24,0.88), help="Sets the analog value for each stimulus index.")
@click.option("--distance", "-d", default=1100, help="Sets the distance from calibration for detecting stimulus index.")
@click.option("--output", "-o", type=click.Choice(["png","pdf"]), default="png")
@click.option("--ignore-extra",  is_flag=True, help="Ignore extra stimuli if stimulus list is longer than detected start times in analog file.")
@click.option("--fix-missing",  is_flag=True, help="Attempt to fill in missing start times, use with --ignore-extra.")
@click.option("--threshold", "-r", type=float, default=9, help="Set the threshold for flicker")
@click.option("--window-height", "-h", type=int, help="Manually set the window resolution. Only applies to legacy eyecandy")
@click.option("--window-width", "-w", type=int)
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
        fix_missing=False, window_height=None, window_width=None, output=None, notebook=None,
        calibration=None, distance=None, verbose=False, debug=False,processes=None,
        by_channel=False, integrity_filter=0.0):
    """Analyze data recorded with eyecandy.
    """
    #### FILEPATHS
    if not os.path.isfile(filename):
        filename = match_filename(filename)
    data_directory, data_name = os.path.split(filename)
    name, extension = os.path.splitext(data_name)
    analog_file = os.path.join(data_directory, name +'.analog')
    stimulus_file = os.path.join(data_directory, name + ".stimulus")
    ctx.obj = {"filename": os.path.join(data_directory,name)}

    if not notebook:
        notebook = find_notebook(data_directory)

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

    lab_notebook = glia.open_lab_notebook(notebook)
    experiment_protocol = glia.get_experiment_protocol(lab_notebook, name)
    flicker_version = experiment_protocol["flickerVersion"]


    #### LOAD STIMULUS
    try:
        ctx.obj["stimulus_list"] = glia.load_stimulus(stimulus_file)
    except OSError:
        print("No .stimulus file found. Attempting to create from .analog file.".format(trigger))
        if flicker_version==0.3:
            ctx.obj["stimulus_list"] = glia.create_stimulus_list(
                analog_file, stimulus_file, notebook, name, eyecandy, ignore_extra,
                calibration, distance, threshold)
            print('finished creating stimulus list')
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
        fix_missing=False, window_height=None, window_width=None, output=None, notebook=None,
        calibration=None, distance=None, version=None, verbose=False, debug=False,processes=None,
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
def cover(units, stimulus_list, c_unit_fig, c_retina_fig):
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
@click.option("--wedge/--no-wedge", default=False,
    help="Sort by flash duration")
@click.option("--kinetics", default=False, is_flag=True,
    help="Sort by third stimuli duration (WAIT)")
@click.option("--version", "-v", type=str, default="2")
@plot_function
def solid_cmd(units, stimulus_list, c_unit_fig, c_retina_fig,
        prepend, append, wedge, kinetics, version=1):
    "Create PTSH and raster of spikes in response to solid."
    # safe_run(solid.save_unit_psth,
    #     (units, stimulus_list, c_unit_fig, c_retina_fig, prepend, append))
    if wedge:
        if version=="1":
            safe_run(solid.save_unit_wedges,
                (units, stimulus_list, partial(c_unit_fig,"wedge"), c_retina_fig, prepend, append))
        elif version=="2":
            safe_run(solid.save_unit_wedges_v2,
                (units, stimulus_list, partial(c_unit_fig,"wedge"), c_retina_fig))
    if kinetics:
        if version=="1":
            safe_run(solid.save_unit_kinetics_v1,
                (units, stimulus_list, c_unit_fig, c_retina_fig))
        elif version=="2":
            safe_run(solid.save_unit_kinetics,
                (units, stimulus_list, partial(c_unit_fig,"kinetics"), c_retina_fig))

    else:
        safe_run(solid.save_unit_spike_trains,
            (units, stimulus_list, c_unit_fig, c_retina_fig, prepend, append))

generate.add_command(solid_cmd)

@analyze.command("bar")
@click.option("--by", "-b", type=click.Choice(["angle", "width","acuity"]), default="angle")
@plot_function
def bar_cmd(units, stimulus_list, c_unit_fig, c_retina_fig, by):
    # if all_methods or "direction" in methods:
    if by=="angle":
        safe_run(bar.save_unit_response_by_angle,
            (units, stimulus_list, c_unit_fig, c_retina_fig))
    elif by=="acuity":
        safe_run(bar.save_acuity_direction,
            (units, stimulus_list, partial(c_unit_fig,"acuity"),
                c_retina_fig))

generate.add_command(bar_cmd)

@analyze.command("convert")
@click.option("--letter", '-l', default=False, is_flag=True,
    help="Output npz for letter classification")
@click.option("--integrity", default=False, is_flag=True,
    help="Output npz for integrity classification")
@click.option("--grating", '-g', default=False, is_flag=True,
    help="Output npz for grating classification")
@click.option("--checkerboard", '-c', default=False, is_flag=True,
    help="Output npz for checkerboard classification")
@click.option("--eyechart", default=False, is_flag=True,
    help="Output npz for eyechart classification")
# @click.option("--letter", default=False, is_flag=True,
#     help="Output npz for letter classification")
@analysis_function
def convert_cmd(units, stimulus_list, filename, letter, integrity, checkerboard,
    eyechart, grating, version=2):
    if letter:
        safe_run(convert.save_letter_npz,
            (units, stimulus_list, filename))
    elif checkerboard:
        safe_run(convert.save_checkerboard_npz,
            (units, stimulus_list, filename))
    elif grating:
        safe_run(convert.save_grating_npz,
            (units, stimulus_list, filename))
    elif eyechart:
        safe_run(convert.save_eyechart_npz,
            (units, stimulus_list, filename))

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
@click.option('stimulus', "-s",
    is_flag=True,
    help="Use .stim file")
# @click.option("--letter", default=False, is_flag=True,
#     help="")
# @click.option("--integrity", default=False, is_flag=True,
#     help="")
@click.option("--checkerboard", '-c', default=False, is_flag=True,
    help="")
@click.option("--grating", '-g', default=False, is_flag=True,
    help="")
# @click.option("--eyechart", default=False, is_flag=True,
#     help="")
# @click.option("--letter", default=False, is_flag=True,
#     help="Output npz for letter classification")
def classify_cmd(filename, stimulus, grating, #letter, integrity, eyechart,
    checkerboard, version=2):
    "Classify using converted NPZ"
    if not os.path.isfile(filename):
        filename = match_filename(filename, 'npz')

    data_directory, data_name = os.path.split(filename)
    name, extension = os.path.splitext(data_name)
    stim_name = strip_generated(name)
    if stimulus:
        stimulus_file = os.path.join(data_directory, stim_name + ".stim")
        stimulus_list = glia.read_stimulus(stimulus_file)
    else:
        stimulus_file = os.path.join(data_directory, stim_name + ".stimulus")
        stimulus_list = glia.load_stimulus(stimulus_file)

    data = np.load(filename)

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
    if checkerboard:
        safe_run(svc.checkerboard_svc,
            (data, stimulus_list, plot_directory))
    if grating:
        safe_run(svc.grating_svc,
            (data, stimulus_list, plot_directory))
    # elif eyechart:
    #     safe_run(convert.save_eyechart_npz,
    #         (units, stimulus_list, name))

generate.add_command(convert_cmd)


@analyze.command("raster")
@plot_function
def raster_cmd(units, stimulus_list, c_unit_fig, c_retina_fig):
    safe_run(raster.save_raster,
        (units, stimulus_list, partial(c_unit_fig,"raster"), c_retina_fig))

@analyze.command("integrity")
@click.option("--version", "-v", type=str, default="2")
@plot_function
def integrity_cmd(units, stimulus_list, c_unit_fig, c_retina_fig, version=1):
    if version=="1":
        safe_run(solid.save_integrity_chart,
            (units, stimulus_list, partial(c_unit_fig,"integrity"),
                c_retina_fig))
    elif version=="2":
        safe_run(solid.save_integrity_chart_v2,
            (units, stimulus_list, partial(c_unit_fig,"integrity"),
                c_retina_fig))
    elif version=="fail":
        safe_run(solid.save_integrity_chart_vFail,
            (units, stimulus_list, partial(c_unit_fig,"integrity"),
                c_retina_fig))

generate.add_command(integrity_cmd)


@analyze.command("grating")
@click.option("-w", "--width", type=int,
    help="Manually provide screen width for old versions of Eyecandy")
@click.option("-h", "--height", type=int,
    help="Manually provide screen height for old versions of Eyecandy")
@plot_function
def grating_cmd(units, stimulus_list, c_unit_fig, c_retina_fig, width, height):
    safe_run(grating.save_unit_spike_trains,
        (units, stimulus_list, c_unit_fig, c_retina_fig, width, height))


@analyze.command("acuity")
@click.option("--prepend", "-p", type=float, default=1,
    help="plot (seconds) before SOLID start time")
@click.option("--append", "-a", type=float, default=1,
    help="plot (seconds) after SOLID end time")
@click.option("--version", "-v", type=float, default=3)
@plot_function
def acuity_cmd(units, stimulus_list, c_unit_fig, c_retina_fig,
        prepend, append, version):
    safe_run(acuity.save_acuity_chart,
        (units, stimulus_list, c_unit_fig, c_retina_fig,
            prepend, append))

generate.add_command(acuity_cmd)


# @main.command()
# @click.argument('filename', type=click.Path(exists=True))
# def fix_header(filename):
#     """Change MCS .raw header encoding from Windows-1252 to UTF-8.

#     A new .data file will be created. To launch, type 'glia' on the command
#     line."""

#     name, ext = os.path.splitext(filename)

#     header_end = "EOH\r\n".encode("Windows-1252")
#     newfile_name = name + '.data'

#     with open(filename, mode='rb') as file:
#         with open(newfile_name, 'wb') as newfile:
#             for line in file:
#                 newline = line.decode("Windows-1252")
#                 newline = re.sub('µ', 'u', newline)
#                 newfile.write(newline.encode("utf8"))
#                 if line == header_end:
#                     break

#             next_chunk = "placeholder"
#             while next_chunk != b'':
#                 next_chunk = file.read(2**28)
#                 newfile.write(next_chunk)

#     return newfile_name


# @main.command()
# @click.argument('filename', type=click.Path(exists=True))
# def convert_mcd(filename):
#     """Convert .mcd to .raw and .analog binary files.

#     Currently setup for 60 channel only. To switch to more channels,
#     change the argument following '-s' to 'El' and 'An'
#     """

#     name, ext = os.path.splitext(filename)

#     electrical = re.sub('%input_filename%', filename, mc_elec_str)
#     electrical = re.sub('%output_filename%', name + '.raw', electrical)
#     os.system(electrical)

#     analog = re.sub('%input_filename%', filename, mc_analog_str)
#     analog = re.sub('%output_filename%', name + '.analog', analog)
#     os.system(analog)


# @main.command()
# @click.argument('filename', type=click.Path(exists=True))
# def gen_params(filename):
#     """Create params file for Spyking Circus with default settings."""

#     name = os.path.splitext(filename)[0]
#     sampling_rate = glia.sampling_rate(filename)
#     params = re.sub('\%rate\%', str(sampling_rate), spyking_params)
#     with open(name + ".params", 'w') as file:
#         file.write(params)

# spyking_circus_method_help = """by default, all 4 steps of the algorithm are \
# performed, but a subset x,y can be done, using the syntax -m x,y. Steps are:
#     - filtering
#     - whitening
#     - clustering
#     - fitting
#     - (extra) merging [meta merging]
#     - (extra) gathering [to collect results]
#     - (extra) extracting [templates from spike times]
#     - (extra) converting [to export to phy format]
#     - (extra) benchmarking [with -o and -t]
# """


# @main.command()
# @click.argument('filename', type=click.Path(exists=True))
# @click.option('--method', '-m',
#               default="filtering,whitening,clustering,fitting",
#               help=spyking_circus_method_help)
# @click.option('-c', default=8, help="Number of CPU")
# @click.option('-g', default=0, help="Number of GPU")
# def spyking_circus(filename, method, c, g):
#     """Run spyking circus."""
#     command = 'spyking-circus "{}" -m {} -c {} -g {}'.format(filename,
#                                                              method,
#                                                              c,
#                                                              g)
#     os.system(command)


# @main.command()
# @click.argument('filename', type=click.Path(exists=True))
# @click.option('--method', '-m',
#               default="filtering,whitening,clustering,fitting",
#               help=spyking_circus_method_help)
# @click.option('-c', default=8, help="Number of CPU")
# @click.option('-g', default=0, help="Number of GPU")
# @click.pass_context
# def run_all(context, filename, method, c, g):
#     """Create data file, create params & run Spyking Circus."""
#     # call fix_header and gen_params
#     context.invoke(fix_header, filename=filename)
#     context.invoke(gen_params, filename=filename)
#     context.forward(spyking_circus)

# # Strings


# spyking_params = """[data]
# data_offset    = MCS                    # Length of the header ('MCS' is auto for MCS file)
# mapping        = /Users/tyler/Dropbox/Science/notebooks/mea/data/mcs_60.prb     # Mapping of the electrode (see http://spyking-circus.rtfd.ord)
# suffix         =                        # Suffix to add to generated files
# data_dtype     = int16                 # Type of the data
# dtype_offset   = auto                   # Padding for data (if auto: uint16 is 32767, uint8 is 127, int16 is 0, ...)
# spike_thresh   = 6                      # Threshold for spike detection
# skip_artefact  = False                  # Skip part of signals with large fluctuations
# sampling_rate  = %rate%                  # Sampling rate of the data [Hz]
# N_t            = 5                      # Width of the templates [in ms]
# stationary     = True                   # Should be False for long recordings: adaptive thresholds
# radius         = auto                   # Radius [in um] (if auto, read from the prb file)
# alignment      = True                   # Realign the waveforms by oversampling
# global_tmp     = False                   # should be False if local /tmp/ has enough space (better for clusters)
# multi-files    = False                  # If several files mydata_0,1,..,n.dat should be processed together (see documentation

# [filtering]
# cut_off        = 500       # Cut off frequency for the butterworth filter [Hz]
# filter         = True      # If True, then a low-pass filtering is performed

# [whitening]
# chunk_size     = 60        # Size of the data chunks [in s]
# safety_time    = 1         # Temporal zone around which templates are isolated [in ms]
# temporal       = False     # Perform temporal whitening
# spatial        = True      # Perform spatial whitening
# max_elts       = 10000     # Max number of events per electrode (should be compatible with nb_elts)
# nb_elts        = 0.8       # Fraction of max_elts that should be obtained per electrode [0-1]
# output_dim     = 5         # Can be in percent of variance explain, or num of dimensions for PCA on waveforms

# [clustering]
# extraction     = median-raw # Can be either median-raw (default), median-pca, mean-pca, mean-raw, or quadratic
# safety_space   = True       # If True, we exclude spikes in the vicinity of a selected spikes
# safety_time    = 1       # Temporal zone around which templates are isolated [in ms]
# max_elts       = 10000      # Max number of events per electrode (should be compatible with nb_elts)
# nb_elts        = 0.8        # Fraction of max_elts that should be obtained per electrode [0-1]
# nclus_min      = 0.01       # Min number of elements in a cluster (given in percentage)
# max_clusters   = 10         # Maximal number of clusters for every electrodes
# nb_repeats     = 3          # Number of passes used for the clustering
# smart_search   = 0          # Parameter for the smart search [0-1]. The higher, the more strict
# sim_same_elec  = 3          # Distance within clusters under which they are re-merged
# cc_merge       = 0.975      # If CC between two templates is higher, they are merged
# noise_thr      = 0.8        # Minimal amplitudes are such than amp*min(templates) < noise_thr*threshold in [0-1]
# make_plots     = png        # Generate sanity plots of the clustering [Nothing or None if no plots]
# remove_mixture = True       # At the end of the clustering, we remove mixtures of templates

# [fitting]
# chunk          = 1         # Size of chunks used during fitting [in second]
# gpu_only       = True      # Use GPU for computation of b's AND fitting
# amp_limits     = (0.3, 30)  # Amplitudes for the templates during spike detection
# amp_auto       = True      # True if amplitudes are adjusted automatically for every templates
# refractory     = 0         # Refractory period, in ms [0 is None]
# max_chunk      = inf       # Fit only up to max_chunk

# [merging]
# cc_overlap     = 0.5       # Only templates with CC higher than cc_overlap may be merged
# cc_bin         = 2         # Bin size for computing CC [in ms]

# [extracting]
# safety_time    = 1         # Temporal zone around which spikes are isolated [in ms]
# max_elts       = 10000      # Max number of collected events per templates
# nb_elts        = 0.8       # Fraction of max_elts that should be obtained per electrode [0-1]
# output_dim     = 5         # Percentage of variance explained while performing PCA
# cc_merge       = 0.975     # If CC between two templates is higher, they are merged
# noise_thr      = 0.8       # Minimal amplitudes are such than amp*min(templates) < noise_thr*threshold

# [noedits]
# filter_done    = False     # Will become True automatically after filtering.

# """

# mc_elec_str = r'wine ~/.wine/drive_c/Program\ Files/Multi\ Channel\ Systems/MC_DataTool/MC_DataTool.com -bin -i "%input_filename%" -o "%output_filename%" -s "Electrode Raw Data:21" -s "Electrode Raw Data:31" -s "Electrode Raw Data:41" -s "Electrode Raw Data:51" -s "Electrode Raw Data:61" -s "Electrode Raw Data:71" -s "Electrode Raw Data:12" -s "Electrode Raw Data:22" -s "Electrode Raw Data:32" -s "Electrode Raw Data:42" -s "Electrode Raw Data:52" -s "Electrode Raw Data:62" -s "Electrode Raw Data:72" -s "Electrode Raw Data:82" -s "Electrode Raw Data:13" -s "Electrode Raw Data:23" -s "Electrode Raw Data:33" -s "Electrode Raw Data:43" -s "Electrode Raw Data:53" -s "Electrode Raw Data:63" -s "Electrode Raw Data:73" -s "Electrode Raw Data:83" -s "Electrode Raw Data:14" -s "Electrode Raw Data:24" -s "Electrode Raw Data:34" -s "Electrode Raw Data:44" -s "Electrode Raw Data:54" -s "Electrode Raw Data:64" -s "Electrode Raw Data:74" -s "Electrode Raw Data:84" -s "Electrode Raw Data:15" -s "Electrode Raw Data:25" -s "Electrode Raw Data:35" -s "Electrode Raw Data:45" -s "Electrode Raw Data:55" -s "Electrode Raw Data:65" -s "Electrode Raw Data:75" -s "Electrode Raw Data:85" -s "Electrode Raw Data:16" -s "Electrode Raw Data:26" -s "Electrode Raw Data:36" -s "Electrode Raw Data:46" -s "Electrode Raw Data:56" -s "Electrode Raw Data:66" -s "Electrode Raw Data:76" -s "Electrode Raw Data:86" -s "Electrode Raw Data:17" -s "Electrode Raw Data:27" -s "Electrode Raw Data:37" -s "Electrode Raw Data:47" -s "Electrode Raw Data:57" -s "Electrode Raw Data:67" -s "Electrode Raw Data:77" -s "Electrode Raw Data:87" -s "Electrode Raw Data:28" -s "Electrode Raw Data:38" -s "Electrode Raw Data:48" -s "Electrode Raw Data:58" -s "Electrode Raw Data:68" -s "Electrode Raw Data:78" -WriteHeader -ToSigned'
# mc_analog_str = r'wine ~/.wine/drive_c/Program\ Files/Multi\ Channel\ Systems/MC_DataTool/MC_DataTool.com -bin -i "%input_filename%" -o "%output_filename%" -s "Analog Raw Data:A1" -WriteHeader -ToSigned'
