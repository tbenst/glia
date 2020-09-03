import tables, numpy as np, matplotlib.pyplot as plt, av, seaborn as sns
from tqdm.auto import tqdm

def frame_idx_before_time(frame_log,time,nBefore=2):
    # get previous frame by subtracting 1, or else we get
    # the insertion index
    return frame_log.time.searchsorted(time,'left')-nBefore

def get_binary_noise_frame_idx(stimulus_list, frame_log):
    binary_noise_stim = list(filter(
        lambda s: s['stimulus']['stimulusType']=='IMAGE'
            and 'label' in s['stimulus']['metadata']
            and s['stimulus']['metadata']['label']=='celltype',
        stimulus_list))
    binary_noise_stim_idx = [s['stimulus']['stimulusIndex']
                            for s in binary_noise_stim]
    
    binary_noise_frame_idx = np.array(frame_log[
        (frame_log.stimulusIndex >= binary_noise_stim_idx[0]) &
        (frame_log.stimulusIndex <= binary_noise_stim_idx[-1])].framenum)
    
    return binary_noise_frame_idx

def unit_lag_sta(video_file, for_frame_add_to_lag,nsamples_before=25,
                 progress=False):

    last_frame_idx = min(for_frame_add_to_lag.keys()) + nsamples_before
    container = av.open(video_file)
    if progress:
        gen = tqdm(enumerate(container.decode(video=0)),total=last_frame_idx)
    else:
        gen = enumerate(container.decode(video=0))
    for n,frame in gen:
        if n==0:
            # initialize lag_sta
            ndframe = frame.to_ndarray(format='bgr24')
            lag_sta = np.zeros([nsamples_before, *ndframe.shape])
            
        if n in for_frame_add_to_lag:
            ndframe = frame.to_ndarray(format='bgr24')
            for lag in for_frame_add_to_lag[n]:
                lag_sta[lag] += ndframe
        elif n > last_frame_idx:
            break
    return lag_sta

def unit_spikes_to_frame_lag(unit_spike_frame_idx, frame_indices,
                             nsamples_before=25):
    filtered_frame_spike_idx = [s for s in unit_spike_frame_idx if s in frame_indices]
    # e.g. k: v means for frame k, add this to lag [0, 1, 2, 5]
    # note that lag 5 is actually 5 frames prior to spike
    for_frame_add_to_lag = {}
    for z_idx in filtered_frame_spike_idx:
        for t in range(0,nsamples_before):
            idx = z_idx - t
            if idx not in for_frame_add_to_lag:
                for_frame_add_to_lag[idx] = [t]
            else:
                for_frame_add_to_lag[idx].append(t)
    return for_frame_add_to_lag

def plot_sta(spatial_filter, ax):
    spatial_filter -= np.mean(spatial_filter)
    spatial_filter /= spatial_filter.std()
    maxval = np.max(np.abs(spatial_filter))
    sns.heatmap(spatial_filter,cmap='seismic_r',vmin=-maxval,vmax=maxval, ax=ax)
    ax.set_title("Spatial filter (std)")
    
def sta_unit_plot_function(fig, axes, unit,
        frame_indices, frame_log, video_file, nsamples_before=25):
    sta = calc_sta(unit.spike_train,  frame_indices, frame_log,
                       video_file, nsamples_before, progress=False)
    ax = next(axes) # single axis in generator
    plot_sta(sta, ax)

def calc_sta(spike_train, frame_indices, frame_log, video_file, nsamples_before=25, progress=False):

    unit_spike_frame_idx = np.array([frame_idx_before_time(frame_log,t) for t in spike_train])
    
    for_frame_add_to_lag = unit_spikes_to_frame_lag(unit_spike_frame_idx,
                                                    frame_indices)
    lag_sta = unit_lag_sta(video_file, for_frame_add_to_lag,nsamples_before, progress=progress)

    spatial_filter = lag_sta[...,0].std(0)
    spatial_filter -= np.mean(spatial_filter)
    return spatial_filter
