from conftest import assert_within
import glia

def compose_test():
	test = glia.compose(lambda x: 3*x, lambda y: y**2, lambda z: z+1)
	assert test(4) == 145

# def f_create_experiments_test(spike_train, sampling_rate, stimulus_list):
# 	create = glia.f_create_experiments(stimulus_list)
# 	experiments = create(spike_train)
# 	assert len(stimulus_list)==len(experiments)
# 	assert experiments[3]["spikes"][0] == 220742 / sampling_rate
# 	assert experiments[3]["stimulus"]["stimulusType"] == "GRATING"
# 	assert len(experiments[3]["spikes"]) == 32

def f_create_experiments_test(spike_train, stimulus_list):
	create = glia.f_create_experiments(stimulus_list)
	experiments = create(spike_train)
	for e in experiments:
		time = e["stimulus"]["lifespan"]/120
		spikes = e["spikes"]
		number_of_spikes = len(spikes)
		assert time==number_of_spikes
		assert spikes[0] <= 1

def f_has_stimulus_type_test(stimulus_list):
	pipeline = glia.f_has_stimulus_type("GRATING")
	filtered = pipeline(stimulus_list)
	for each in filtered:
		assert each["stimulus"]["stimulusType"]=="GRATING"

def simulated_test(units, stimulus_list):
	assert len(next(iter(units.values())).spike_train)==2200

# def f_flow_test(spike_train, stimulus_list):
	# get_wait_firing_rate = glia.compose(
	#     glia.f_create_experiments(stimulus_list),
	#     glia.f_has_stimulus_type("WAIT"),
	#     glia.f_group_by_stimulus(),
	#     glia.f_calculate_firing_rate_by_stimulus(),
	# )
	# wait_firing_rate = glia.apply_pipeline(get_wait_firing_rate,
	#                                        spike_trains_by_unit)
