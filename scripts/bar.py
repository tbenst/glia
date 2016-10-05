import glia
import matplotlib.pyplot as plt
import numpy as np

def f_get_key(i):
    return lambda item: item[i]


def get_fr_dsi_osi(units, stimulus_list):

	get_bar_firing_rate = glia.compose(
	    glia.f_create_experiments(stimulus_list),
	    glia.f_has_stimulus_type(["BAR"]),
	    glia.f_group_by_stimulus(),
	    glia.f_calculate_peak_ifr_by_stimulus(),
	)
	bar_firing_rate = glia.apply_pipeline(get_bar_firing_rate,units)

	get_bar_dsi = glia.compose(
	    glia.by_speed_width_then_angle,
	    glia.calculate_dsi_by_speed_width
	)
	bar_dsi = glia.apply_pipeline(get_bar_dsi,bar_firing_rate)

	get_bar_osi = glia.compose(
	    glia.by_speed_width_then_angle,
	    glia.calculate_osi_by_speed_width
	)
	bar_osi = glia.apply_pipeline(get_bar_osi,units)


	return (bar_firing_rate, bar_dsi, bar_osi)


def plot_unit_response_by_angle(ax, data):
    """Plot the average for each speed and width."""
    # we will accumulate by angle in this dictionary and then divide
    unit_id, bar_firing_rate, bar_dsi, bar_osi, first = data
    analytics = glia.by_speed_width_then_angle(bar_firing_rate)
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
        line_angle, line_radius = glia.sort_two_arrays_by_first(line_angle,line_radius)
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
    cells = [["{:1.3f}".format(bar_dsi[speed_width]) for speed_width in columns], \
             ["{:1.3f}".format(bar_osi[speed_width]) for speed_width in columns]]
    
    table = ax.table(cellText=cells,
                      rowLabels=['DSI',"OSI"],
                      colLabels=columns,
                     colColours=colors,
                        loc='bottom', bbox = [0,-0.2,1,0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    if first is True:
        ax.legend()

def plot_population_dsi_osi(ax,data):
	bar_dsi, bar_osi = data

	population_dsi = pd.DataFrame.from_dict(bar_dsi, orient="index")
	population_osi = pd.DataFrame.from_dict(bar_osi, orient="index")

	number_of_units = len(bar_dsi.keys())

	def f_histogram(bins, the_range):
	    return lambda x: np.histogram(x,bins,the_range)

	dsi_population = population_dsi.apply(f_histogram(10,(0,1))).apply(lambda counts_bins: (counts_bins[0]/number_of_units, counts_bins[1]))
	osi_population = population_osi.apply(f_histogram(10,(0,1))).apply(lambda counts_bins: (counts_bins[0]/number_of_units, counts_bins[1]))

	bins = np.arange(0,1,0.1)
	dh = pd.DataFrame(index=bins)
	oh = pd.DataFrame(index=bins)

	for item in dsi_population.iteritems():
	    parameter = item[0]
	    values, bins = item[1]
	    dh[parameter] = values
	    
	for item in osi_population.iteritems():
	    parameter = item[0]
	    values, bins = item[1]
	    oh[parameter] = values

	fig,ax = plt.subplots(2,1)
	dh.plot.bar(figsize=(20,20), ax=ax[0])
	ax[0].set_title("DSI: "+data_directory+data_name)
	oh.plot.bar(figsize=(20,20), ax=ax[1])
	ax[1].set_title("OSI: "+data_directory+data_name)


def save_unit_responses_by_angle(output_files, units, stimulus_list):
	bar_firing_rate, bar_dsi, bar_osi = get_fr_dsi_osi(units, stimulus_list)
	
	def data_generator():
		first = True
		for unit_id in bar_firing_rate.keys():
		    yield (unit_id, bar_firing_rate[unit_id], bar_dsi[unit_id],
		    	bar_osi[unit_id], first)
		    first = False
    nplots = len(bar_firing_rate.keys()) + 1

    fig_unit_response = glia.plot_from_generator(plot_unit_response_by_angle,data_generator,nplots)
    fig_unit_response.savefig(output_files[0])
    fig_population,ax = plt.subplots()
    glia.plot_population_dsi_osi(ax, (bar_firing_rate, bar_dsi, bar_osi))
    fig_population.savefig(output_files[1])
