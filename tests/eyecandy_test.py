from conftest import assert_within


def get_stimulus_start_times_test(stimulus_start_times,stimulus_list):
	assert len(stimulus_start_times) == 143
	
	previous_time=stimulus_start_times[0]
	for i in range(1,len(stimulus_start_times)-1):
		duration = stimulus_start_times[i] - stimulus_start_times[i-1]
		assert_within(duration,stimulus_list[i-1][1]['lifespan']/120,0.1)
	assert len(stimulus_start_times) == len(stimulus_list)
	assert_within(stimulus_start_times[0]+2,stimulus_start_times[1],0.1)
	assert_within(stimulus_start_times[1]+1,stimulus_start_times[2],0.1)
	assert_within(stimulus_start_times[2]+3,stimulus_start_times[3],0.1)
	assert_within(stimulus_start_times[3]+50,stimulus_start_times[4],0.1)

