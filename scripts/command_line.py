#!/usr/bin/env python

import matplotlib
matplotlib.use("agg")
matplotlib.rcParams['figure.max_open_warning'] = 250

import glia
import click
import os
import re
import scripts.solid as solid
import scripts.bar as bar
import errno
import traceback
from functools import update_wrapper, partial


from glob import glob
from glia.classes import Unit
from matplotlib.backends.backend_pdf import PdfPages

def analysis_function(f):
    @click.pass_context
    def new_func(ctx, *args, **kwargs):
        context_object = ctx.obj
        return ctx.invoke(f, ctx.obj["units"], ctx.obj["stimulus_list"], 
            ctx.obj["c_add_unit_figures"], ctx.obj["c_add_retina_figure"],
            *args[3:], **kwargs)
    return update_wrapper(new_func, f)



def safe_run(function, args):
    try:
        function(*args)
    except Exception as exception:
        traceback.print_tb(exception.__traceback__)
        print(exception)
        print("Error running {}. Skipping".format(function, ))

def plot_path(directory,plot_name):
    return os.path.join(directory,plot_name+".png")

@click.group()
def main():
    pass


@main.group(chain=True)
@click.argument('filename', type=click.Path(exists=True))
@click.option("--notebook", "-n", type=click.Path(exists=True))
@click.option("--eyecandy", "-e", default="http://eyecandy:3000")
@click.option("--output", "-o", type=click.Choice(["pdf"]), default="pdf")
@click.option("--trigger", "-t", type=click.Choice(["flicker", 'detect-solid', "ttl"]), default="flicker",
    help="""Use flicker if light sensor was on the eye candy flicker, solid if the light sensor detects the solid stimulus,
    or ttl if there is a electrical impulse for each stimulus.
    """)
@click.pass_context
def analyze(ctx, filename, trigger, eyecandy, output=None, notebook=None):
    """Analyze data recorded with eyecandy.
    """
    ctx.obj = {}
    data_directory, data_name = os.path.split(filename)
    name, extension = os.path.splitext(data_name)
    analog_file = os.path.join(data_directory, name +'.analog')
    stimulus_file = os.path.join(data_directory, name + ".stimulus")

    spyking_regex = re.compile('.*\.result.hdf5$')
    if extension == ".txt":
        ctx.obj["units"] = glia.read_plexon_txt_file(filename,filename)
    elif re.match(spyking_regex, filename):
        ctx.obj["units"] = glia.read_spyking_results(filename)
    else:
        raise ValueError('could not read {}. Is it a plexon or spyking circus file?')

    if not notebook:
        notebooks = glob(os.path.join(data_directory, '*.yml'))
        if len(notebooks)==0:
            raise ValueError("no lab notebooks (.yml) were found. Either add to directory," \
                "or specify file path with -n.")
        notebook=notebooks[0]


    try:
        ctx.obj["stimulus_list"] = glia.load_stimulus(stimulus_file)
    except OSError:
        print("No .stimulus file found. Attempting to create from .analog file via {}".format(trigger))
        if trigger == "flicker":
            ctx.obj["stimulus_list"] = glia.create_stimulus_list_from_flicker(
                analog_file, stimulus_file, notebook, name, eyecandy)
        elif trigger == "detect-solid":
            ctx.obj["stimulus_list"] = glia.create_stimulus_list_from_SOLID(
                analog_file, stimulus_file, notebook, name, eyecandy)
        elif trigger == "ttl":
            raise ValueError('not implemented')
        else:
            raise ValueError("invalid trigger: {}".format(trigger))

    # prepare_output
    plot_directory = os.path.join(data_directory, name+"-plots")
    try:
        os.makedirs(plot_directory)
        os.chmod(plot_directory, 0o777)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    if output == "pdf":
        ctx.obj["retina_pdf"] = PdfPages(glia.plot_pdf_path(plot_directory, "retina"))
        ctx.obj["unit_pdfs"] = glia.open_pdfs(plot_directory, list(ctx.obj["units"].keys()), Unit.name_lookup())
        # c connotes 'continuation'
        ctx.obj["c_add_unit_figures"] = partial(glia.add_to_unit_pdfs,
            unit_pdfs=ctx.obj["unit_pdfs"])
        ctx.obj["c_add_retina_figure"] = lambda x: ctx.obj["retina_pdf"].savefig(x)
        
    elif output == "png":
        raise ValueError("not implemented")


@analyze.resultcallback()
@click.pass_context
def cleanup(ctx, results, filename, trigger, eyecandy, output=None, notebook=None):
    if output == "pdf":
        ctx.obj["retina_pdf"].close()
        glia.close_pdfs(ctx.obj["unit_pdfs"])

    print("Finished")


@analyze.command()
@click.pass_context
def all(ctx):
    "Run all analyses."
    ctx.forward(solid_cmd)
    ctx.forward(bar_cmd)


@analyze.command("solid")
@analysis_function
def solid_cmd(units, stimulus_list, c_add_unit_figures, c_add_retina_figure):
    "Create PTSH and raster of spikes in response to solid."
    print("in solid")
    safe_run(solid.save_unit_psth,
        (units, stimulus_list, c_add_unit_figures, c_add_retina_figure))
    safe_run(solid.save_unit_spike_trains,
        (units, stimulus_list, c_add_unit_figures, c_add_retina_figure))

@analyze.command("bar")
# @click.option("--method", "-m", type=click.Choice(["direction"]), default="all")
@analysis_function
def bar_cmd(units, stimulus_list, c_add_unit_figures, c_add_retina_figure):
    # if all_methods or "direction" in methods:
    safe_run(bar.save_unit_response_by_angle,
        (units, stimulus_list, c_add_unit_figures, c_add_retina_figure))
    safe_run(bar.save_unit_spike_trains,
        (units, stimulus_list, c_add_unit_figures, c_add_retina_figure))

@analyze.command("grating")
@analysis_function
def grating_cmd(units, stimulus_list, c_add_unit_figures, c_add_retina_figure):
    pass

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
