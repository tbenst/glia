from conftest import assert_within
import glia

def compose_test():
	test = glia.compose(lambda x: 3*x, lambda y: y**2, lambda z: z+1)
	assert test(4) == 145

def f_create_experiments_test(spike_train, sampling_rate, stimulus_list):
	create = glia.f_create_experiments()
	experiments = create(spike_train, stimulus_list)
	assert experiments[0][0] == 53420 / sampling_rate
	assert len(experiments[0])==1
