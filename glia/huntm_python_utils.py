import matplotlib.pyplot as plt
from scipy.signal import chirp
import numpy as np
import glia
from functools import reduce, partial
from scipy import signal
import math
import sklearn
import seaborn as sns
import matplotlib.gridspec as gridspec
import pandas as pd
import umap
import pickle
from os import path

def get_stimulus_plot_list(experiment_LOD):
    """This function does what it says: takes in an experiment_LOD and returns a list of times with
    associated stimuli. With the thing returned we will be able to plot the stimulus and """

    def make_plot_list_segment(stim,start,stop):
        stim_type = stim['stimulus']['stimulusType']
        temp_stimulus = np.zeros(int(stop*Hz-start*Hz))
        temp_time = np.arange(start,stop,1/Hz)
#         temp_time = np.insert(temp_time,0,last_point[0])
#         temp_stimulus = np.insert(temp_stimulus,0,last_point[1])
        duration = temp_stimulus.shape[0]
        if stim_type == 'WAIT':
            temp_stimulus[:] = 0
            color = 'k'
        elif stim_type == 'SOLID':
            if stim['stimulus']['backgroundColor'] == "white":
                temp_stimulus[:] = 1
                color = 'k'
            elif stim['stimulus']['backgroundColor'] == "gray":
                temp_stimulus[:] = 0.5
                color = 'k'
            elif stim['stimulus']['backgroundColor'] == "green":
                temp_stimulus[:] = 1
                color = 'g'
            elif stim['stimulus']['backgroundColor'] == "blue":
                temp_stimulus[:] = 1
                color = 'b'
            elif stim['stimulus']['backgroundColor'] == "red":
                temp_stimulus[:] = 1
                color = 'r'
        elif stim_type == "CHIRP": 
            f0 = stim['stimulus']['f0'] 
            f1 = stim['stimulus']['f1'] 
            t1 = stim['stimulus']['t1']
            t = np.arange(0,t1,1/Hz)
            if f0==f1:
                chirp_train = chirp(t, f0=f0, f1=f1, t1=t1, method='linear')
                temp_stimulus[:] = ((chirp_train)*np.arange(0,t1,1/Hz)/t1+1)/2
            else:
                chirp_train = chirp(t, f0=f0, f1=f1, t1=t1, method='linear')
                temp_stimulus[:] = (chirp_train+1)/2
            color = 'k'
#         last_point = [temp_time[-1],temp_stimulus[-1]]

        print(stim_type)
        return temp_stimulus,temp_time,color

    Hz = 1000
    #1. Get the total time
    total_time = 0

    for i in range(len(experiment_LOD)):
        stim = experiment_LOD[i]
        total_time += stim['stimulus']['lifespan']

    #note we don't actually use the following
    stimulus_segment_list = []
    time_segment_list = []
    color_list = []

    time = 0
    last_point = [0,0]
    for i in range(len(experiment_LOD)):
        stim = experiment_LOD[i]
        timestart = time
        time += stim['stimulus']['lifespan']
        start = timestart*1000
        stop = time*1000
        temp_stimulus,temp_time,color = make_plot_list_segment(stim,timestart,time)
        stimulus_segment_list.append(temp_stimulus)
        time_segment_list.append(temp_time)
        color_list.append(color)
    stimulus_plot_list = [time_segment_list,stimulus_segment_list,color_list]
    fullfield_stim_duration=time
    return stimulus_plot_list,fullfield_stim_duration

def plot_single_unit(unit):
    """unit is in teh experiment_LOD form"""
    spike_train = np.array([])
    start_time = 0
    for stim in unit:
        spikes = stim["spikes"] + start_time # clever.
        spike_train = np.concatenate([spike_train, spikes])
        start_time += stim["stimulus"]["lifespan"]
    plt.figure(figsize=(20,4))
    plt.vlines(spike_train, 0, 1,colors='k',alpha=0.1)
    # plt.fill([1,1.5,1.5,1], [-0.1,-0.1,1.1,1.1], alpha=0.2)
    # plt.xlim(0,3.5)
    plt.title("Single unit spikes")
    plt.xlabel("time (s)")
    plt.xlim(0,start_time)
    # plt.xlim(0,46)
    plt.show()

def plot_stimulus_and_unit_response(unit):
    """unit is the experiment_LOD format"""
    stimulus_plot_list,_ = get_stimulus_plot_list(unit)
    plot_from_stimulus_plot_list(stimulus_plot_list)
    plot_single_unit(unit)

def plot_from_stimulus_plot_list(plist):
    """plots our big stimulus on the provided ax
    inputs: plist is a list of lists [time_segment_list,stimulus_segment_list,color_list]"""
    plt.figure(figsize=(20,4))
    time_segment_list,stimulus_segment_list,color_list = plist[0],plist[1],plist[2]
    last_point=[0,0]
    for i in range(len(plist[0])):
        x_plot = np.insert(time_segment_list[i],0,last_point[0])
        y_plot = np.insert(stimulus_segment_list[i],0,last_point[1])
        plt.plot(x_plot,y_plot,color = color_list[i])
        last_point=[time_segment_list[i][-1],stimulus_segment_list[i][-1]]
    plt.show()


#next we'll get "chop points", which will be the points to cut teh stimulus train for clustering. We'll chop
# at the midpoint of the WAIT blocks.
def get_wait_chop_points(experiment_LOD,Hz = 1000):
    """takes in experiment_LOD and returns the chop points for the fullfield stim.
    chop points are the mid-timepoints of the WAIT blocks, except in the case that the first and/or last
    block is wait. Then we group that with the prior"""
    total_time = 0
    chop_times = []
    last_block_ind = len(experiment_LOD)-1
    for i in range(len(experiment_LOD)):
        stim = experiment_LOD[i]
        temp_time = stim['stimulus']['lifespan']
        if stim['stimulus']['stimulusType'] == "WAIT":
            if i != 0 and i != last_block_ind:
                chop_times.append(int((total_time+0.5*temp_time)*Hz))
        total_time += temp_time
    return chop_times

def split_stimulus_array(ifr,unit):
    """a unit as an experiment_LOD, which we use assuming all units in ifr underwent same stimulus train
    returns a list containing the segments of our ifr array, and also the chopped_plot_list, which is the
    stimulus train chopped into the correct pieces
    chopped_plot_list = [chopped_time,chopped_stim,colors

    This is getting A little bit messy. We will run get_stimulus_plot_list on this to get the stimulus list
    which we then will split and output. this is bc the stim split is different from our previous plotting
    split"""
    stimulus_plot_list,_ = get_stimulus_plot_list(unit)
    time_segment_list,stimulus_segment_list,color_list = stimulus_plot_list[0],stimulus_plot_list[1],stimulus_plot_list[2]
    plot_list = [time_segment_list,stimulus_segment_list,color_list] 

    chop_times = get_wait_chop_points(unit)
    print(f"chop times are {chop_times}")
    ifr_segment_list = []
    chopped_plot_list = []
    for i in range(len(chop_times)):
        if i == 0:
            t1 = 0
            t2 = chop_times[i]
        else:
            t1 = chop_times[i-1]
            t2 = chop_times[i]
        ifr_segment_list.append(ifr[:,t1:t2])
    ifr_segment_list.append(ifr[:,t2:])
    print(f"the length of sum of split arrays is {sum([x.shape[1] for x in ifr_segment_list])}")
    return ifr_segment_list,chop_times

def split_plot_list_by_chops(plot_list,chop_times):
    """This function takes a plot list, which is separated by unique stimulus, and converts it into a
    separation by chop times. handling color is getting really ugly. We are making a list the same number of
    elements, and then we'll say if there's any color other than 'k',we plot that color...

    Plot_list: this is the output of the get_stimulus_plot_list
    Note this is a mess. We had to append the last_p;oint to the front of each element there in order to plot
    it. It should probably be refactored anyway. """
    color_expanded = []
    chopped_colors = [] #this is managed in manage_color
    for i in range(len(plot_list[0])):
        color = plot_list[-1][i]
        color_exp = [color for i in range(len(plot_list[0][i])-1)]
        color_expanded+=color_exp
    time_merge = np.concatenate(plot_list[0])
    stim_merge = np.concatenate(plot_list[1])
    chopped_time_list=[]
    chopped_stim_list=[]
    for i in range(len(chop_times)):
        if i == 0:
            t1 = 0
            t2 = chop_times[i]
        else:
            t1 = chop_times[i-1]
            t2 = chop_times[i]
        chopped_time_list.append(time_merge[t1:t2])
        chopped_stim_list.append(stim_merge[t1:t2])
        chopped_colors.append(manage_color([t1,t2],color_expanded))
    chopped_time_list.append(time_merge[t2:])
    chopped_stim_list.append(stim_merge[t2:])
    chopped_colors.append(manage_color([t2,len(stim_merge)],color_expanded))
    chopped_plot_list = [chopped_time_list,chopped_stim_list,chopped_colors]
    return chopped_plot_list

def manage_color(range_t, color_expanded):
    t1 = range_t[0]
    t2 = range_t[1]
    for t in range(t1,t2):
        if color_expanded[t] != 'k':
            print(color_expanded[t])
            return color_expanded[t]
    return 'k'

def plot_average_firing_rate(ifr_segment_list,chopped_plt_list):
    fig,ax = plt.subplots(2,1,figsize = (20,4))
    for i in range(len(chopped_plt_list[0])):
        ax[0].plot(chopped_plt_list[0][i],chopped_plt_list[1][i],color = chopped_plt_list[2][i])
        ax[0].set_title('plotted from choped stim')
        x_plot = chopped_plt_list[0][i] 
        mean = np.mean(ifr_segment_list[i],axis=0)
        std = np.std(ifr_segment_list[i],axis=0)
        ax[1].plot(x_plot,mean)
        ax[1].fill_between(x_plot,mean-std,mean+std,alpha = 0.5)
#         ax[1].set_ylim(bottom=y_min,top=None)
        ax[1].set_title(f"Mean Firing Rate of All Cells")

def plot_firing_rate_stats(ifr_segment_list,chopped_plt_list,bins=100):
    if type(ifr_segment_list)!=list:
        ifr_segment_list=[ifr_segment_list]
    rows = len(ifr_segment_list)
    if rows == 1:
        rows += 1
    #first plot the aggregate firing
    plot_average_firing_rate(ifr_segment_list,chopped_plt_list)
    fig,ax = plt.subplots(rows,3,figsize = (15,15))
    for i in range(len(ifr_segment_list)):
        if chopped_plt_list is not None:
            signal_info = [pl[i] for pl in chopped_plt_list]
            ax[i][0].plot(signal_info[0],signal_info[1],color = signal_info[2])
            ax[i][0].set_title("Signal")
        mean_fr = list(np.mean(ifr_segment_list[i],axis=1))
        max_fr = list(np.amax(ifr_segment_list[i],axis=1))
        ax[i][1].hist(mean_fr,bins = bins)
        ax[i][1].set_title("Mean Firing Rate")
        ax[i][1].set_xlabel("Firing Rate")
        ax[i][1].set_ylabel("Cell Count")        
        ax[i][2].hist(max_fr,bins = bins)
        ax[i][2].set_title("Max Firing Rate")
        ax[i][2].set_xlabel("Firing Rate")
        ax[i][2].set_ylabel("Cell Count")
    fig.tight_layout(pad=1.5)

def filter_nonfiring_cells(ifr,kappas=None,cut_lowest_n_bins = 1):
    mean = np.mean(ifr,axis = 1)
    n,bins,patches = plt.hist(mean,bins=100)
    threshold = bins[cut_lowest_n_bins] #meaning we cut the first bin only
    above_threshold = np.where(mean>threshold)
    filtered_ifr = ifr[above_threshold[0],:]
    if kappas is not None:
        f_kappas = kappas[above_threshold[0]]
    else:
        f_kappas=None
    return filtered_ifr,f_kappas

def normalize_cell_firing(ifr):
    means = np.mean(ifr,axis=1)
    ifr_norm = np.divide(ifr,means[:,np.newaxis])
    return ifr_norm


#####################################################################################################

"""clustering functions"""
# 
"""def divisive_clustering(ifr_segment_list,chopped_plt_list,nested_cluster_dict):
#    The one fn to rule them all. Take in the big ifr_segment_list and nested cluster dicts and does
    cluster_dict = nested_cluster_dict
    labels_list = cluster_pipeline_chopped(ifr_segment_list,chopped_plt_list,cluster_dict)
    labels_array = labels_list[0]
    kappa = cluster_dict['kappas_mat']
    subcluster_dict = make_subcluster_dict(ifr_segment_list,labels_array,kappa)

    clusters_to_split = cluster_dict['clusters_to_split']# this is how the recursive structrue takes place
    for cluster_of_interest in clusters_to_split:
        print(f"Plotting cluster_{cluster_of_interest}")
        ifr_subsegs = subcluster_dict[cluster_of_interest]['ifr']
        kps = subcluster_dict[cluster_of_interest]['kappas']

        cluster_dict = nested_cluster_dict['subcluster_dicts'][f"cluster_{cluster_of_interest}"]
        if cluster_dict['kappas_mat'] is not None:
            cluster_dict['kappas_mat'] = kps
        labels_list = cluster_pipeline_chopped(ifr_subsegs,chopped_plt_list,cluster_dict)
        # plot the entire stimulus, even if we only clustered over part of it
        if len(cluster_dict['segments_to_cluster'])<6: #less than full stim
            print(f"plotting full cluster traces, even tho we clustered on only {len(cluster_dict['segments_to_cluster'])} of 6.")
            traces_dict = {'overlap_stim_on_plot':True, 'plot_unclustered':True,}
            plot_cluster_traces_nopd(labels_list[0],np.concatenate(ifr_subsegs,axis=1),chopped_plt_list,traces_dict)

        labels_array = labels_list[0]
        scd = make_subcluster_dict(ifr_subsegs,labels_array,kps)
        subcluster_dict[cluster_of_interest]['subclusters']=scd
    return subcluster_dict
"""

def recursive_clustering(ifr_segment_list,chopped_plt_list,original_positions,nested_cluster_dict,subcluster_dict=None):
    """The one fn to rule them all. Take in the big ifr_segment_list and nested cluster dicts and does
    clustering and further subclustering.
    nested_cluster_dict: {'level':1,'parent_cluster':None,...rest_of_dictionary...

    returns subcluster_dict {cluster#:{'ifr':[ifr_segment_list==cluster#],'kappas':kappa==cluster#,
                                        'subclusters':{cluster#:{'ifr':[...},cluster#:...}},
                            cluster#:{}}

    original_positions is just a vector like np.arange(ifr_segment_list[0].shape[0])

    Note: I may need to return the labels as well somehow.
    """
    cluster_dict = nested_cluster_dict
    labels_list = cluster_pipeline_chopped(ifr_segment_list,chopped_plt_list,cluster_dict)
    if cluster_dict.get('plot_full_traces',False):
        if len(cluster_dict['segments_to_cluster'])<6: #less than full stim
            print(f"plotting full cluster traces, even tho we clustered on only {len(cluster_dict['segments_to_cluster'])} of 6.")
            traces_dict = {'overlap_stim_on_plot':True,'plot_unclustered':True,'global_palette':cluster_dict.get('global_palette')}
            plot_cluster_traces_nopd(labels_list[0],np.concatenate(ifr_segment_list,axis=1),chopped_plt_list,traces_dict)

    if cluster_dict.get('global_palette'):
        shorten_palette(cluster_dict,labels_list)

    labels_array = labels_list[0]
    kappa = cluster_dict['kappas_mat']
    subcluster_dict = make_subcluster_dict(ifr_segment_list,labels_array,original_positions,kappa)
    if 'clusters_to_split' in cluster_dict: #Recurse
        clusters_to_split = cluster_dict['clusters_to_split']# this is how the recursive structrue takes place
        for cluster_of_interest in clusters_to_split:
            print(f"Plotting cluster_{cluster_of_interest}")
            ifr_subsegs = subcluster_dict[cluster_of_interest]['ifr']
            kps = subcluster_dict[cluster_of_interest]['kappas']
            original_positions = subcluster_dict[cluster_of_interest]['original_positions']
            global_palette = cluster_dict.get('global_palette')
            cluster_dict = nested_cluster_dict['subcluster_dicts'][f"cluster_{cluster_of_interest}"]
            cluster_dict.update({'global_palette':global_palette})
            if cluster_dict['kappas_mat'] is not None:
                cluster_dict['kappas_mat'] = kps
            scd = recursive_clustering(ifr_subsegs,chopped_plt_list,original_positions,cluster_dict)
            subcluster_dict[cluster_of_interest]['subclusters']=scd
    return subcluster_dict

def shorten_palette(cluster_dict,labels_list):
    for i in range(np.amax(labels_list)+2):
        popped = cluster_dict['global_palette'].pop(0)
        print(f"popped {popped} from the palette")

def make_subcluster_dict(ifr_segment_list,labels,original_positions,kappas = None):
    """uses the labels to return a dict: label_# --> list_of_segments from corresponding label
    original_positions is a vector giving each cell's position in the original ifr matrix."""
    subcluster_dict = {}
    nclust = max(labels)
    for c in range(-1,nclust+1):
        # We are also including -1 unclustered.
        subcluster_dict[c] = {'ifr':[],'original_positions':original_positions[labels == c]}
        for ifr_segment in ifr_segment_list:
            subcluster_dict[c]['ifr'].append(ifr_segment[labels == c,:])
            if kappas is not None:
                subcluster_dict[c]['kappas'] = kappas[labels==c]
            else:
                subcluster_dict[c]['kappas'] = None

    return subcluster_dict


def examine_subcluster_dict(subcluster_dict,level=0,parent=None,return_cluster=None):
    """just prints the shapes of the subclusters and such. Also returns a chosen cluster by cluster string.
    return_cluster, you feed in the cluster string you want, like '4.1.-1' and that will give you all of the
    ifr_segments, kappa, and child clusters for cluster 4, subcluster 1, subcluster -1 (unclustered) cells"""
#     print(f"Level {level} clusters")
    tab_str = ''.join(['\t' for i in range(level)])
    for k,v in subcluster_dict.items():
        if parent is None:
            cluster_str = str(k)
        else:
            cluster_str = f"{parent}.{k}"
        print(tab_str+f"Cluster {cluster_str}")
        print(tab_str+str([ifr_seg.shape for ifr_seg in v['ifr']]))

        if cluster_str == return_cluster:
            cluster_to_return = v
            return cluster_to_return
        else:
            cluster_to_return = None

        if 'subclusters' in v:
            print(f"For cluster {k}, we have subclusters...")
            cluster_to_return = examine_subcluster_dict(subcluster_dict[k]['subclusters'],level=level+1,parent=cluster_str,return_cluster = return_cluster)
    return cluster_to_return

def assign_subcluster_labels(scd,sc_labels_out,parent = None):
    """recurse thru the subcluster dict and make a subcluster_labels_list which is the shape of our original
    ifr, and has the subcluster label at each position. The subcluster label is somthing like 4.0.2 for
    cluster 4, subcluster 0, subcluster 2"""
    for k,v in scd.items():
        if parent is None:
            cluster_str = str(k)
        else:
            cluster_str = f"{parent}.{k}"
        if 'subclusters' in v:
            assign_subcluster_labels(v['subclusters'],sc_labels_out,parent=cluster_str)
        else:
            print(cluster_str)
            original_positions = v['original_positions']
            for i in original_positions:
                sc_labels_out[i] = cluster_str


def plot_subcluster_labels_on_original(subcluster_dict,original_dim_reduction,marker_dict={}):
    """we plot our subcluster labels onto our dimensionality-reduced plot. If the dim reduction is umap, then
    its type will be string and we'll use a saved and loaded reduction, so we don't make mistakes trying to
    remake it"""
    if type(original_dim_reduction) == str:
        print(f"loading the last embeddings from {original_dim_reduction}")
        embedding_dict = pickle.load(open(original_dim_reduction,'rb'))
        lowdim_data = embedding_dict['lowdim_data']

    original_ifr_length = sum([v['original_positions'].shape[0] for v in subcluster_dict.values()])
    sc_labels_out = ['Fill' for x in range(original_ifr_length)]

    assign_subcluster_labels(subcluster_dict,sc_labels_out,parent = None)
    cluster_set = sorted(set(sc_labels_out))
    palette = sns.color_palette("muted", len(cluster_set))
    print(len(palette))

    plt.figure(figsize=(20,20))
    for klass, color in zip(cluster_set, palette):
        Xk = np.array([lowdim_data[i] for i in range(len(sc_labels_out)) if sc_labels_out[i] == klass])
        plt.scatter(Xk[:, 0], Xk[:, 1], color=color, label=klass, marker=marker_dict.get(klass,'.'), alpha=0.8)
    plt.legend()
    plt.show()
#     plt.plot(X[labels_1 == -1, pc_plot[0]], X[labels_1 == -1, pc_plot[1]], 'k+', alpha=0.1)
#     plt.set_title(f'Clustering at {ep} epsilon cut\nDBSCAN')
#     plt.set_xlabel(f"PC {pc_plot[0]}")
#     plt.set_ylabel(f"PC {pc_plot[1]}")




"""Note: eventually I should make a plot_subcluster_dict function"""



def cluster_pipeline_chopped(ifr_segment_list,chopped_plt_list,cluster_dict):
    """This runs teh big clustering pipeline"""
    if type(cluster_dict['n_components']) == list:
        assert len(cluster_dict['n_components']) == len(cluster_dict['segments_to_cluster'])
        join = False
    else:
        join = True
    segments_to_cluster = cluster_dict['segments_to_cluster']
    ifr_segs,plists = get_chopped_segments_to_cluster(ifr_segment_list,chopped_plt_list,segments_to_cluster,join=join)

    labels_list, _ = cluster_and_plot_traces(ifr_segs,cluster_dict,plists)
    return labels_list

def get_chopped_segments_to_cluster(ifr_segment_list,chopped_plt_list,segments,join = True):
    """This function returns the ifr_segments we will cluster over and the corresponding stimulus signal
    segments. These then get passed into the function to cluster and plot
    segments is a list of numbers [1,2,4], that pulsl the segments we want to cluster over.

    If a list like range of number of segments in entire stim is passed in, we will cluster over the
    aggregated stimulus, as if it were not ever chopped."""
    l = []
    plists = [[],[],[]]
    for s_index in segments:
        l.append(ifr_segment_list[s_index])
        plists[0].append(chopped_plt_list[0][s_index])
        plists[1].append(chopped_plt_list[1][s_index])
        plists[2].append(chopped_plt_list[2][s_index])
    if join == True:
        ifr_segment = np.concatenate(l,axis=1)
    else:
        ifr_segment = l
    return ifr_segment,plists


def plot_umap(data,umap_parameters):
    from sklearn.decomposition import PCA
    up = umap_parameters
    reducer = umap.UMAP(n_neighbors = up['n_neighbors'],
                       min_dist = up['min_dist'],
                       n_components = up['n_components'],
                       metric = up['metric'])
    if up['normalize_firing']:
        data = normalize_cell_firing(data)
    if up['pre_pca'] is not False:
        data = PCA(up['pre_pca']).fit_transform(data)
    n_f_ifr_embedded = reducer.fit_transform(data)
    def scatter(data_in):
        plt.figure(figsize=(15,15))
        plt.scatter(data_in[:,0],data_in[:,1],s=3)
        plt.show()
    scatter(n_f_ifr_embedded)

def reduce_dim(ifr_segment,num_components,cluster_dict):
    umap_params = cluster_dict.get('UMAP',False)
    if cluster_dict.get('UMAP',False):
        up = umap_params
        if up.get('save_load_umap',False) and up.get('overwrite',False)==False and path.exists(up['save_load_umap']):
            print(f"loading the last embeddings from {up['save_load_umap']}")
            embedding_dict = pickle.load(open(up['save_load_umap'],'rb'))
            if embedding_dict['umap_dict'] == cluster_dict['UMAP'] and embedding_dict.get('segments_to_cluster') == cluster_dict['segments_to_cluster']:
                lowdim_data = embedding_dict['lowdim_data']
                make_new_embedding = False
            else:
                print("There was a change in the umap parameters since last time. Creating new embedding")
                make_new_embedding = True
        else:
            make_new_embedding = True
        if make_new_embedding == True:
            reducer = umap.UMAP(n_neighbors = up['n_neighbors'],
                               min_dist = up['min_dist'],
                               n_components = up['n_components'],
                               metric = up['metric'])
            if up['pre_pca'] is not False:
                ifr_segment = sklearn.decomposition.PCA(up['pre_pca']).fit_transform(ifr_segment)
            lowdim_data = reducer.fit_transform(ifr_segment)
            if up.get('save_load_umap',False):
                embedding_dict = {'lowdim_data':lowdim_data,'umap_dict':cluster_dict['UMAP'],'segments_to_cluster':cluster_dict['segments_to_cluster']}
                pickle.dump(embedding_dict,open(up['save_load_umap'],'wb'))
                print(f"saved the embedding_dict to {up['save_load_umap']}")
    else:
        pca = sklearn.decomposition.PCA(num_components)
        lowdim_data = pca.fit_transform(ifr_segment)

    return lowdim_data

def cluster_and_plot_traces(ifr_segment,cluster_dict,plist):
    """ example input: cluster_dict = {'n_components':5,
                    'kappas_mat':None,
                    'xi':0.05,
                    'min_samples':50,
                    'metric':"cosine",
                    'pc_plot'=[0,1]}
        plist has a very ugly format
        [pl[seg_index] for pl in chopped_plt_list]
        The n_components can be list or int. If its a list, then ifr segment will also be a list and the
        different segments will be pca-ed separately and clustered together after concat of the pca
        dimensions. Plotting of the irf_segment always occurs after joining. """

    if 'normalize_firing' in cluster_dict:
        normalize = cluster_dict['normalize_firing']
        print(f"normalizing = {normalize}")
    else:
        normalize = False
    cd = cluster_dict
    if type(ifr_segment)==list:
        print("PCA-ing each chopped section separately, and clustering together")
        lowdim_data = []
        for i in range(len(ifr_segment)):
            ifr_seg = ifr_segment[i]
            n_components = cluster_dict['n_components'][i]
            if normalize == True:
                pca_data = normalize_cell_firing(ifr_seg)
            else:
                pca_data = ifr_seg
            lowdim_data.append(reduce_dim(pca_data,n_components,cluster_dict))
        lowdim_data = np.concatenate(lowdim_data,axis=1)
        ifr_segment = np.concatenate(ifr_segment,axis=1)
    else:
         if normalize == True:
            pca_data = normalize_cell_firing(ifr_segment)
         else:
            pca_data = ifr_segment
         n_components = cluster_dict['n_components']
         lowdim_data = reduce_dim(pca_data,n_components,cluster_dict)

    if cluster_dict['kappas_mat'] is not None:
        lowdim_data = np.concatenate([lowdim_data, cluster_dict['kappas_mat'][:,np.newaxis]],axis=1)

    # The following saves a bunch of time by caching the optics results. Note that we automatically overwrite
    # if there has been a change in  the cluster_dict or the umap dict inside of the cluster_dict
    """TO DO: Extract this to a load_state() function"""
    def compare_optics_dict(cluster_dict,loaded_dict,excluded_keys):
        cd = {k:v for k,v in cluster_dict.items() if k not in excluded_keys}
        ld = {k:v for k,v in loaded_dict.items() if k not in excluded_keys}
        return cd == ld

    if cluster_dict.get('save_load_optics',False) and cluster_dict.get('overwrite_optics',False)==False and path.exists(cluster_dict['save_load_optics']):
        print(f"Loading optics clustering results from {cluster_dict['save_load_optics']}")
        optics_dict = pickle.load(open(cluster_dict['save_load_optics'],'rb'))
        if compare_optics_dict(cluster_dict,optics_dict['cluster_dict'],excluded_keys = ['global_palette','suppress_plots','clusters_to_split','subcluster_dicts','eps','overlap_stim_on_plot','plot_unclustered','plot_optics','pc_plot']):
            make_new_optics = False
            optics = optics_dict['data']
        else:
            print(f"There was a change in the cluster dict since last time, so we are going to make a new optics_dict and overwrite")
            make_new_optics = True
    else:
        make_new_optics = True
    if make_new_optics == True:
        optics = sklearn.cluster.OPTICS(min_samples=cd['min_samples'], xi=cd['xi'] ,metric=cd['metric'])
        optics.fit(lowdim_data)
        if cluster_dict.get('save_load_optics',False):
            print(f"saving optics clustering results to {cluster_dict['save_load_optics']}")
            optics_dict = {'data':optics,'cluster_dict':cluster_dict}
            pickle.dump(optics_dict,open(cluster_dict['save_load_optics'],'wb'))

    labels = optics.labels_
    nclust = max(labels)+1
    labels_list = plot_clusters_and_reachability(lowdim_data,optics,nclust,cluster_dict)
    if not cluster_dict.get('suppress_plots',False):
        if 'plot_optics' in cluster_dict:
            if cluster_dict['plot_optics']==True:
                plot_cluster_traces_nopd(labels,ifr_segment,plist,cluster_dict)
        if labels_list != []:
            for i in range(len(labels_list)):
                plot_cluster_traces_nopd(labels_list[i],ifr_segment,plist,cluster_dict)
    return labels_list, nclust

def plot_clusters_and_reachability(lowdim_data,optics,nclust,cluster_dict):
    labels_list = []#used for optics-->dbscan conversion if we're doing that
    if 'eps' in cluster_dict:
        eps_list = []
        for ep in cluster_dict['eps']:
            eps_list.append(ep)
#             eps1=cluster_dict['eps'][0]
#         eps2=cluster_dict['eps'][1]
    pc_plot = [0,1]
    if cluster_dict.get('global_palette') is None:
        palette = sns.color_palette("colorblind", nclust)
    else:
        palette = cluster_dict['global_palette']
    fig = plt.figure(figsize=(15, 10))
    G = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, 0])
    ax3 = plt.subplot(G[1, 1])
    ax4 = plt.subplot(G[1, 2])

    # Reachability plot
    X = lowdim_data
    space = np.arange(len(X))
    reachability = optics.reachability_[optics.ordering_]
    reach_labels = optics.labels_[optics.ordering_]
    for klass, color in zip(range(0, nclust), palette):
        Xk = space[reach_labels == klass]
        Rk = reachability[reach_labels == klass]
        ax1.plot(Xk, Rk, color=color, marker='.', linestyle='None', alpha=0.3)
    ax1.plot(space[reach_labels == -1], reachability[reach_labels == -1], 'k.', alpha=0.3)
    ax1.set_ylabel('Reachability')
    ax1.set_title('Reachability Plot')
    if 'ylim' in cluster_dict:
        ylim = cluster_dict['ylim']
        ax1.set_ylim(bottom = ylim[0],top=ylim[1])
    
    if 'eps' in cluster_dict:
        for ep in eps_list:
            ax1.plot(space, np.full_like(space, ep, dtype=float), 'k-', alpha=0.5)
#         ax1.plot(space, np.full_like(space, eps1, dtype=float), 'k-.', alpha=0.5)

    # OPTICS
    for klass, color in zip(range(0, nclust), palette):
        Xk = X[optics.labels_ == klass]
        ax2.plot(Xk[:, pc_plot[0]], Xk[:, pc_plot[1]], color=color, linestyle='None', marker='.', alpha=0.3)
    ax2.plot(X[optics.labels_ == -1, pc_plot[0]], X[optics.labels_ == -1, pc_plot[1]], 'k+', alpha=0.1)
    ax2.set_title('Automatic Clustering\nOPTICS')
    ax2.set_xlabel(f"PC {pc_plot[0]}")
    ax2.set_ylabel(f"PC {pc_plot[1]}")

    #OPTICS to DBSCAN conversion
    if 'eps' in cluster_dict:
        labels_list = []
        for ep in eps_list:
            labels_1 = sklearn.cluster.cluster_optics_dbscan(reachability=optics.reachability_,
                                           core_distances=optics.core_distances_,
                                           ordering=optics.ordering_, eps=ep)
#             labels_2 = sklearn.cluster.cluster_optics_dbscan(reachability=optics.reachability_,
#                                            core_distances=optics.core_distances_,
#                                            ordering=optics.ordering_, eps=eps2)
            # DBSCAN at eps1
            for klass, color in zip(range(0, max(labels_1)+1), palette):
                Xk = X[labels_1 == klass]
                ax3.plot(Xk[:, pc_plot[0]], Xk[:, pc_plot[1]], color=color, linestyle='None', marker='.', alpha=0.3)
            ax3.plot(X[labels_1 == -1, pc_plot[0]], X[labels_1 == -1, pc_plot[1]], 'k+', alpha=0.1)
            ax3.set_title(f'Clustering at {ep} epsilon cut\nDBSCAN')
            ax3.set_xlabel(f"PC {pc_plot[0]}")
            ax3.set_ylabel(f"PC {pc_plot[1]}")
#             # DBSCAN at eps2.
#             for klass, color in zip(range(0, max(labels_2)+1), palette):
#                 Xk = X[labels_2 == klass]
#                 ax4.plot(Xk[:, -2], Xk[:, -1], color=color, linestyle='None', marker='.', alpha=0.3)
#             ax4.plot(X[labels_2 == -1, -2], X[labels_2 == -1, -1], 'k+', alpha=0.1)
#             ax4.set_title(f'Clustering at {eps2} epsilon cut\nDBSCAN')
#             ax4.set_xlabel("PC 1")
#             ax4.set_ylabel("PC 2")
            labels_list.append(labels_1)
    fig.tight_layout()
    plt.show()
    return labels_list

def cluster_grid_search(n_components_list,xi_list,min_samples_list,metric_list):
        performance_dict = {}
        for n_components in n_components_list:
            for xi in xi_list:
                for min_samples in min_samples_list:
                    for metric in metric_list:
                        key_string = f"c={n_components}_xi={xi}_ms={min_samples}_metric={metric}"
#                         try:
                        labels,nclust,optics_cells,optics_cluster_traces = cluster_and_plot_traces(n_components,[0,1],xi,min_samples,metric)
                        num_unclustered = sum(labels == -1)
                        performance_dict[key_string] = {'nclust':nclust,'labels':labels,
                                                        'unclustered':num_unclustered}
                        optics_cluster_traces.suptitle(key_string,y=.2)
                        optics_cluster_traces.savefig(f"../../data/figures/{key_string}.png")
                        print(key_string)
                        print(f"nclust = {nclust}, unclustered = {num_unclustered}")
                        del optics_cells
                        del optics_cluster_traces
#                         except:
#                             print(key_string)
#                             print("couldn't' be run for some reason. Moving along")
        return performance_dict

def plot_cluster_traces_nopd(labels,ifr_segment,plist,cluster_dict):
    """same as below but only uses numpy arrays for speed i think
    plist is the chopped_plot_list that has information for the stim_train. 
    It's really of teh format [pl[seg_index] for pl in chopped_plt_list]
    If we want to plot multiple signal_chop segments at once,then we need to put the plists segments in list
    format and this function will handle it as well"""

    if 'overlap_stim_on_plot' in cluster_dict:
        overlap = cluster_dict['overlap_stim_on_plot']
    else:
        overlap = False
    cells = []
    ncells = []
    x_length = ifr_segment.shape[1]
    x_plot = np.arange(x_length)
    nclust = max(labels)+1
    print(nclust)
    if nclust > 15:
        return
    # for c in range(-1,nclust):
    # ignore unclustered
    y_min = 0
    fig_length = 10
    fig,ax = plt.subplots(nclust+2,1,figsize=(fig_length,(fig_length+3)/4*(nclust+1)))
    ax = ax.flatten()
    if type(plist[0])!=list: #if we don't plot multiple segments...
        stim_x = [plist[0]]
        stim_y = [plist[1]]
        color = [plist[2]]
    else:
        stim_x = plist[0]
        stim_y = plist[1]
        color = plist[2]

    for i in range(len(stim_x)):
        ax[0].plot(stim_x[i],stim_y[i],color=color[i])
    ax[0].set_title("Stimulus Train Chop") 
    x = []
    x_plot = [t for seg_times in stim_x for t in seg_times]

    if cluster_dict.get('global_palette') is None:
        palette = sns.color_palette("colorblind", nclust+1)
    else:
        palette = cluster_dict['global_palette']

    for c in range(-1,nclust):
        if c > -1:
            cells_to_plot = ifr_segment[labels==c,:]
#             plot_trace(ax,cells_to_plot,palette,c,y_min,stim_x,stim_y,cluster_dict)
            mean = np.mean(cells_to_plot,axis =0)
            std = np.std(cells_to_plot,axis=0)
            ax[c+1].plot(x_plot,mean,color=palette[c])
            ax[c+1].fill_between(x_plot,mean-std,mean+std,alpha = 0.5,color=palette[c])
            ax[c+1].set_ylim(bottom=y_min,top=None)
            ax[c+1].set_title(f"Cluster {c}: {cells_to_plot.shape[0]} cells")
            if cluster_dict.get('overlap_stim_on_plot'): #if we want to overlap, we can plot the max(mean)*stim height
                for i in range(len(stim_x)):
                    ax[c+1].plot(stim_x[i],stim_y[i]*np.amax(mean),color=color[i],alpha=0.2)
    if cluster_dict['plot_unclustered']==True:
        cells_to_plot = ifr_segment[labels==-1,:]
        if cluster_dict.get('global_palette') is None:
            trace_color = palette[-1]
        else:
            trace_color = palette[nclust]
        mean = np.mean(cells_to_plot,axis =0)
        std = np.std(cells_to_plot,axis=0)
        ax[-1].plot(x_plot,mean,color=trace_color)
        ax[-1].fill_between(x_plot,mean-std,mean+std,alpha = 0.5,color=trace_color)
        ax[-1].set_ylim(bottom=y_min,top=None)
        ax[-1].set_title(f"Unclustered cells: {cells_to_plot.shape[0]} cells")
        if cluster_dict.get('overlap_stim_on_plot'): #if we want to overlap, we can plot the max(mean)*stim height
            for i in range(len(stim_x)):
                ax[-1].plot(stim_x[i],stim_y[i]*np.amax(mean),color=color[i],alpha=0.2)
    fig.tight_layout(pad=2)
    plt.show()


def plot_trace(ax,cells_to_plot,palette,c,y_min,stim_x,stim_y,cluster_dict):
    mean = np.mean(cells_to_plot,axis =0)
    std = np.std(cells_to_plot,axis=0)
    ax[c+1].plot(x_plot,mean,color=palette[c])
    ax[c+1].fill_between(x_plot,mean-std,mean+std,alpha = 0.5,color=palette[c])
    ax[c+1].set_ylim(bottom=y_min,top=None)
    ax[c+1].set_title(f"Cluster {c}: {cells_to_plot.shape[0]} cells")
    if cluster_dict.get('overlap_stim_on_plot'): #if we want to overlap, we can plot the max(mean)*stim height
        for i in range(len(stim_x)):
            ax[c+1].plot(stim_x[i],stim_y[i]*np.amax(mean),color=color[i],alpha=0.2)

# 
 
"""The following are just abstracted functions from the jupyter notebook"""
def separate_spikes_into_celltyping_responses(stimulus_list,spikes,parameters = {}):
    """This is just a bunch of actions happening in the notebook that I'm abstracting to here to make things
    cleaner"""
    def filter_celltyping(l):
        """I think this function filters out any of the elements within a single unit's experiment list of dicts
        that do not have the 'label':'celltype' k,v pair in the metadata k,v pair. IDK in which situtaions this arises."""
        return list(filter(lambda x: "label" in x["stimulus"]["metadata"] and \
            x["stimulus"]["metadata"]["label"]=="celltype", l))

    def remove_stimulus(l, stimulusTypes):
        return list(filter(lambda x: not x["stimulus"]["stimulusType"] in stimulusTypes, l))

    def remove_stimulusIndex_range(l, startIdx, stopIdx):
        return list(filter(lambda x: x["stimulus"]["stimulusIndex"] < startIdx or x["stimulus"]["stimulusIndex"] > stopIdx, l))

    if 'Idxs' not in parameters:
        parameters['Idxs'] = [9,54]

    # select cell
    get_celltyping = glia.compose(
        glia.f_create_experiments(stimulus_list), # create_experiemnts returns a function that is used to make an experiment
        # log for each spike_train using the stimulus_list information. Running apply_pipeline using get_celltyping will run the compose of these functions on the spiketrain adn return a dictionary
        # of lists of dicts that contains the experiemtn information for each cell unit in the experiment.
        filter_celltyping,
        )

    get_fullfield = glia.compose(
        partial(remove_stimulus, stimulusTypes=["IMAGE", "BAR","WHITE_NOISE"]), #creates a partial function that takes a
        # remove WAITs in between BAR (stimulusIndex 9-54)
        # WARNING: if celltyping is updated, these hardcoded numbers may need to change
        partial(remove_stimulusIndex_range, startIdx=parameters['Idxs'][0], stopIdx=parameters['Idxs'][1])
    )
    celltyping_responses = glia.apply_pipeline(get_celltyping,spikes, progress=True)
    """Applies create_experiemnts function, and filter celltyping.
    Input: spikes (and implicitly simulus_list) --> output: dict with k's = all_unit id's, v's = experiment list-of_dict for each unit.
    1. create_experiemtns: For each unit in spikes-->segregate the spiketrain into the appropriate portion of experiement and return the
    experiment list-of-dicts for each unit (thus we have a dict(list(dicts)))

    2. filter_celltyping: I think for each unit, just remove each portion of teh experiments list that does not have
    'label':'celltype' in its metadata dictionary. At least for first set of data, all have 'label':'celltype'"""


    celltyping_responses_fullfield = glia.apply_pipeline(get_fullfield, celltyping_responses)
    """Applies get_fullfield-->dict k's = all-units, v's = experimetn list-of-dict for each unit.
    1. first partial: takes each unit's experiment list-of-dicts and removes all 'BAR' and 'IMAGE' stimulus types
        Note that the stimulusType is associated with the experiment information, and is thus attached to each unit's portion of each experiment.
    2. second partial: takes unit's experiment list-of-dicts and removes all stimulusIndex's that fall w/in
    the range 9-54. For this first dataset it happens that the WAIT stimuli between BAR (i think) are all
    within 9-54 in terms of index."""
    if parameters.get('skip_noise',False):
        celltyping_responses_noise = {}
    else:
        celltyping_responses_noise = glia.apply_pipeline(glia.f_has_stimulus_type("IMAGE"), celltyping_responses)
    """
    glia.f_has_stimulus_type is going to return a function. That function is the f_filter(anonymous) function
    Oh, I think the overall effect is that celltyping_responses_noise is going to be all of the experimental sections from each
    unit that are stimulusType = 'Image'.
    """

    """See the below cells for illustrations of each of these."""
    return celltyping_responses,celltyping_responses_fullfield,celltyping_responses_noise

def preprocess_celltyping_bar(celltyping_responses):
    """celltyping_responses is an experiment LOD"""

    celltyping_responses_bar = glia.apply_pipeline(
        glia.compose(glia.f_has_stimulus_type("BAR"), #keep only the poritions of each unit's experiment list-of-dicts that has "BAR" type
                     glia.f_group_by("angle"),
                     glia.f_map(lambda x: x[0]), # only 1 stimuli in group, no reps

                     glia.count_spikes),

        celltyping_responses)
    bar_spike_count = {k: sum(v.values()) for k,v in celltyping_responses_bar.items()}
    def fit_von_mises_from_angle_dict(angle_dict, stabilize=True, fscale=1):
        """Take a dict with angle (0 to 2pi) as key and spike count as value. Return Von Mises Kappa.

        If stabilize=True, we add 1 spike for each angle for numerical stability
        (else a single spike to one direction will have a high kappa)

        TODO: test if scipy handles non-zero mean properly. May need to mean center at 0.
        Need to specify fscale=1 else get bad fits sometimes......
        https://stackoverflow.com/questions/39020222/python-scipy-how-to-fit-a-von-mises-distribution
        https://github.com/scipy/scipy/issues/8878"""
        samples = []
        for angle, count in angle_dict.items():
            samples += [angle - np.pi] * count
        if stabilize:
            samples += list(angle_dict.keys())
        if len(samples)==0:
            # no response to any moving bar is a uniform response
            # thus, kappa is 0
            return 0
        if not fscale is None:
            kappa, loc, scale = vonmises.fit(samples, fscale=fscale)
        else:
            kappa, loc, scale = vonmises.fit(samples)
        return kappa, loc, scale
    try:
        import pickle
        kappas = pickle.load(open('../../data/kappas_dict_07_13_2020','rb'))
        print("kappa dict was loaded")
    except:
        print("no kappa dict on record. Making one")
        kappas = glia.pmap(fit_von_mises_from_angle_dict, celltyping_responses_bar, progress=True)
    most_selective_units = sorted([(k,sc,kappa[0]) for k, (sc,kappa) in glia.zip_dictionaries(bar_spike_count, kappas)], key=lambda x: x[2])
    return most_selective_units,kappas,celltyping_responses_bar


def make_rgc_irf_kappa_arrays(celltyping_responses_fullfield,kappas,fullfield_stim_duration,bin_width=0.001,bandwidth=0.05,sigma=6):
    """kappas is a dict"""
    # bin spikes each 1ms for signal processing
    nCells = len(celltyping_responses_fullfield)
    rgcs = np.zeros((nCells, int(fullfield_stim_duration * 1000)), dtype=np.uint8) # 1 ms bins
    bins = np.arange(0, fullfield_stim_duration+0.001, 0.001)
    kappa = np.zeros(nCells)
    for i, (name,un) in enumerate(celltyping_responses_fullfield.items()):
        spike_train = np.array([])
        start_time = 0
        for stim in un: # this goes thru each unit and concat the spike trains and then uses a histogram to give a value for
            # the number of spikes at each timebin (or rather I think there's just one spike at each time bin, but maybe that's not for sure. )
            un_spikes = stim["spikes"] + start_time
            spike_train = np.concatenate([spike_train, un_spikes])
            start_time += stim["stimulus"]["lifespan"]
        un_spike_train = np.histogram(spike_train, bins)[0] # histogram approach is smart. gets zeros most places.
        rgcs[i] = un_spike_train
        if kappas is not None:
            kappa[i] = kappas[name][0]


    if kappas is None:
        kappa = None
    spike_counts = np.sum(rgcs,1)
    # estimate firing rate using gaussian smoothing
    transformed_sigma = bandwidth/bin_width
    window = signal.gaussian(2*sigma*transformed_sigma, std=transformed_sigma)
    window /= bandwidth
    window /= math.sqrt(2*math.pi)
    # Ah, right. If your area = 1 in units of seconds for the x-axis, your std is in units of seconds (which we have at first)
    # In this case, transformed_sigma is in terms of # bins, which are 0.001 seconds.
    # okay but when we integrate under the curve, we still treat our std [=] seconds, and we use the appropriate area terms
    # for our Riemann sums.

    # instantaneous firing rate (acausal)
    ifr = np.array([signal.convolve(unit, window,mode="same") for unit in rgcs])
    def check_window(window,bin_width):
        window_riemann = window * bin_width
        print(f"the area under the gaussian is indeed {sum(window_riemann)}")
        
    check_window(window, bin_width)
    return rgcs,ifr,kappa

def load_celltyping_responses(data_dir,filename,params={}):
    import os
    from pathlib import Path

    if not params:
        params = {'Idxs':[9,71],'skip_noise':True}

    fig_dict = {}
    fig_dict["plot_name"] = filename
    fig_dict["filename"] = data_dir + filename
    fig_dict["notebook"] = data_dir + "lab notebook.yml"
    name = fig_dict["plot_name"]
    filename = fig_dict["filename"]
    notebook = fig_dict["notebook"]

    data_directory, data_name = os.path.split(filename)
    lab_notebook = glia.open_lab_notebook(fig_dict["notebook"])

    name, extension = os.path.splitext(data_name)
    stimulus_file = os.path.join(data_directory, name + ".stim")
    metadata, stimulus_list, method = glia.read_stimulus(stimulus_file)

    channel_map = glia.config.channel_map_3brain
    spikes = glia.read_3brain_spikes(Path(data_directory) / (name +".bxr"), name, channel_map)
    celltyping_responses,celltyping_responses_fullfield,celltyping_responses_noise = separate_spikes_into_celltyping_responses(stimulus_list,spikes,params)

    return celltyping_responses,celltyping_responses_fullfield,celltyping_responses_noise

def load_and_merge(params):
    """takes in some filenames and a few other parameters. Returns basically the full preprocessing so you
    can move on to clustering"""

    data_dir = params['data_dir']
    celltyping_responses,celltyping_responses_fullfield,celltyping_responses_noise  = {},{},{}
    for filename in params['filenames']:
        if params.get('preloaded_celltype_responses',False):
            if filename in params['preloaded_celltype_responses']:
                cr = params['preloaded_celltype_responses'][filename][0]
                crf= params['preloaded_celltype_responses'][filename][1]
                crn= params['preloaded_celltype_responses'][filename][2]
            else:
                cr,crf,crn = load_celltyping_responses(data_dir,filename,params)
        else:
            cr,crf,crn = load_celltyping_responses(data_dir,filename,params)
        celltyping_responses.update(cr)
        celltyping_responses_fullfield.update(crf)
        celltyping_responses_noise.update(crn)

    #Show the Stimulus and an example cell
    k = sorted(celltyping_responses_fullfield.keys())[0]
    unit = celltyping_responses_fullfield[k]
    stimulus_plot_list,fullfield_stim_duration = get_stimulus_plot_list(unit)
    plot_stimulus_and_unit_response(unit)

    #Make the firing rate arrays and filtered version
    """Should make kappas possible as a parameter"""
    kappas = None
    rgcs,ifr,kappa = make_rgc_irf_kappa_arrays(celltyping_responses_fullfield,kappas,fullfield_stim_duration,bin_width = 0.001, bandwidth = 0.05,sigma = 6)
    f_ifr,f_kappa = filter_nonfiring_cells(ifr,kappa)

    #Chop arrays and the stimulus info at each "wait" block
    unit = celltyping_responses_fullfield[k]
    ifr_segment_list,chop_times = split_stimulus_array(ifr,unit)
    f_ifr_segment_list,chop_times = split_stimulus_array(f_ifr,unit)
    chopped_plt_list = split_plot_list_by_chops(stimulus_plot_list,chop_times)
    plt.figure(figsize=(20,4))
    for i in range(len(chopped_plt_list[0])):
        plt.plot(chopped_plt_list[0][i],chopped_plt_list[1][i],color = chopped_plt_list[2][i])
        plt.title('plotted from choped stim')

    output_dict = {'celltyping_responses_fullfield':celltyping_responses_fullfield,
                   'celltyping_responses_noise':celltyping_responses_noise,
                   'celltyping_responses':celltyping_responses,
                   'rgcs':rgcs,
                   'ifr':ifr,
                   'f_ifr':f_ifr,
                   'kappa':kappa,
                   'f_kappa':f_kappa,
                   'ifr_segment_list':ifr_segment_list,
                   'f_ifr_segment_list':f_ifr_segment_list,
                   'chopped_plt_list':chopped_plt_list,
                   'chop_times':chop_times,}
    return output_dict
