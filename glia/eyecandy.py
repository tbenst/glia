def create_eyecandy_gen(program_yaml, eyecandy_url):
    # Create program from eyecandy YAML. 
    r = requests.post(eyecandy_url + '/analysis/start-program',
                      data={'programYAML': program_yaml,
                           'windowHeight': 1140,
                           'windowWidth': 912})
    sid = r.text
    def eyecandy_gen():
        done = False
        while (True):
            # Get the next stimulus using program sid
            json = requests.get(base_url + '/analysis/program/{}'.format(sid)).json()
            done = json["done"]
            if done:
                break
            else:
                yield json 

    return eyecandy_gen()

def open_lab_notebook(filepath):
    """Take a filepath for YAML lab notebook and return dictionary."""
    with open( filepath, 'r') as f:
        y = yaml.load(f)
    return y

def get_experiment_protocol(lab_notebook_yaml, name):
    """Given lab notebook, return protocol matching name."""
    protocol_list = lab_notebook_yaml['study']['data'][0]['retinas'][0]['experiments']
    for protocol in protocol_list:
        if protocol["name"]==name:
            return protocol

def get_stimulus_from_protocol( protocol ):
    """Get stimulus text from protocol suitable for eye-candy."""
    return yaml.dump(protocol['stimulus']['program'])

def find_transition_times(analog_file, threshold=10000):
    """
    Given the input signal `y` with samples at times `t`,
    find the times where `y` increases through the value `threshold`.

    `t` and `y` must be 1-D numpy arrays.

    Linear interpolation is used to estimate the time `t` between
    samples at which the transitions occur.
    """
    #sampling rate
    fs = sampling_rate(analog_file)
    analog = read_raw_voltage(analog_file)[:,1]
    nsamples = analog.shape[0]
    T = nsamples/fs
    #t is times
    t = np.linspace(0, T, nsamples, endpoint=False)
    
    
    # Find where analog crosses the threshold (increasing).
    lower = (analog < threshold) & (analog > 1000)
    higher = analog >= threshold
    transition_indices = np.where(lower[:-1] & higher[1:])[0]

    # Linearly interpolate the time values where the transition occurs.
    t0 = t[transition_indices]
    t1 = t[transition_indices + 1]
    analog0 = analog[transition_indices]
    analog1 = analog[transition_indices + 1]
    slope = (analog1 - analog0) / (t1 - t0)
    transition_times = t0 + (threshold - analog0) / slope

    return list(transition_times)

    
def get_stimulus_from_eyecandy(start_times, eyecandy_gen):
    """Return list of tuples (start time, corresponding eyecandy_gen)"""
    # compensate for weird analog behavior at end of recording
    start_times.pop()
    return list((map(lambda x: (x, next(eyecandy_gen)), start_times)))
    
def create_experiments(unit: np.ndarray, stimulus_list,
                       #is this supposed to return a list of dictionaries?
                       duration: Seconds=100/1000) -> List[List[float]]:
    """Split firing train into experiments based on start times (e.g. on time).
    stimulus_list: List[(float, dict)]"""
    num_experiments = len(stimulus_list)
    #s[1] takes the next(eyecandy_gen) from the tuples in stimulus_list
    experiments = [{"stimulus": s[1], "spikes": []} for s in stimulus_list]
    i = 0
    #generates a list of spiketimes that are normalized by the stimulus start_time
    for spike in unit:
        #accounts for if duration is such that the spike time belongs to the next stimulus start_time
        #checks if at last experiment
        if i +1 < num_experiments and spike >= stimulus_list[i+1][0]:
            i += 1
            #
            if spike < stimulus_list[i][0] + duration:
                experiments[i]["spikes"].append(spike - stimulus_list[i][0])
        #checks if spiketime is between the start time and the duration we're counting for
        elif (spike >= stimulus_list[i][0]) and (spike < stimulus_list[i][0] + duration):
            # subtract start_time to normalize
            experiments[i]["spikes"].append(spike - stimulus_list[i][0])
    #a list of dictionaries for a given   
    return experiments

#Analytics is a dictionary of dictionaries. The keys are the stimulus types. 
# The values are dictionaries containing the information in stimulus (spike counts, angle, lifespan)
def unit_analytics(experiments):
    analytics = {  "WAIT":  {},  "SOLID": {},  "BAR": {},  "GRATING": {} }
    # only bars and gratings have angles. only waits and solids have a lifespan
    has_angle=re.compile(r"^(BAR|GRATING)$")
    has_time=re.compile(r"^(WAIT|SOLID)$")
    #stimulus is a dictionary in experiments
    for stimulus in experiments:
        the_stimulus = stimulus["stimulus"]['value']
        spike_count = len(stimulus["spikes"])
        stimulus_type = the_stimulus[ "stimulusType"]
        if has_angle.match(stimulus_type):
            angle = the_stimulus[ "angle"]
            try:
                analytics[stimulus_type][angle].append(spike_count)
            except:
                analytics[stimulus_type][angle]=[spike_count]
        elif has_time.match(stimulus_type):
            the_time = the_stimulus[ "lifespan"]
            try:
                analytics[stimulus_type][the_time].append(the_time)
            except:
                analytics[stimulus_type][the_time] = [the_time]
    return analytics


def get_start_times_of_stimulus(stimulus_type, stimulus_list):
    r = re.compile(r'^SOLID$')
    ret = []
    for (t,s) in stimulus_list:
        stimulus_type = s['value']["stimulusType"]
        if r.match(stimulus_type):
            ret.append(t)
    return ret

# def histogram_of_stimulus(stimulus, experiments, bins = np.arange(0,1,10/1000)):
#     analytics = {  "WAIT":  {},  "SOLID": {},  "BAR": {},  "GRATING": {} }
#     has_time=re.compile(r"^{}$".format(stimulus))
#     for stimulus in experiments:
#         the_stimulus = stimulus["stimulus"]['value']
#         stimulus_type = the_stimulus[ "stimulusType"]
#         if has_angle.match(stimulus_type):
#             angle = the_stimulus[ "angle"]
#             try:
#                 analytics[stimulus_type][angle].append(spike_count)
#             except:
#                 analytics[stimulus_type][angle]=[spike_count]
#         elif has_time.match(stimulus_type):
#             the_time = the_stimulus[ "lifespan"]
#             try:
#                 analytics[stimulus_type][the_time].append(the_time)
#             except:
#                 analytics[stimulus_type][the_time] = [the_time]
#     return analytics