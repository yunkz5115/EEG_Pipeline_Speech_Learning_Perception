# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 14:28:48 2025

@author: yu028288
"""
import numpy as np
import os
import scipy.io as scio
import glob
from EEG_MVPA_Package import (
    tg_permutation_standard,
    tg_transfer_permutation_standard,
    )
from EEG_RSA_Package import (
    all_rsa,
    )
from scipy.stats import t
import matplotlib.pyplot as plt

def get_epoch_path(root_path,sub_index,blk,type):
    participant_list = sorted(os.listdir(root_path))
    data_path = glob.glob(root_path+participant_list[sub_index]+'/'+'*'+blk+'_'+type+'_'+'-epo.fif')[0]
    return data_path

def load_catch_index_for_MVPA(file_path):
    file = scio.loadmat(file_path)
    stim_index = file['participant_log']['stim_index'][0][0]
    pre_index = stim_index[stim_index[:,4]==1,3]
    train_index = stim_index[stim_index[:,4]==2,3]
    post_index = stim_index[stim_index[:,4]==3,3]
    pre_post_NS_index = np.zeros(stim_index.shape[0])
    NS_locker = np.where(stim_index[:,1]==1)[0]
    for loc in NS_locker:
        pre_post_NS_index[(loc-3):loc] = 1
        pre_post_NS_index[(loc+1):(loc+4)] = 2
        
    pre_post_NS_index = pre_post_NS_index[60:60+420]
    
    return pre_index,train_index,post_index,pre_post_NS_index

#Set epoch loading parameters
blk_list = ['blk-1','blk-2','blk-3']
type_list = ['lowband','highband']

#Set MVPA parameters
permutation_time = 5000
input_ch = None
num_participants = 19
thr = -t.ppf(0.05, num_participants-1)
n_jobs = -1
re_sfreq = 25
plot_extent = [-0.2, 0.8, -0.2, 0.8]
reject_thr=0.0001
n_fold = 8
crop_t = [0,0.8]
tlim = [-0.2,0.8]

#Set load path
root_path = 'Data/Inf_ref/'
participant_log_path = 'Behaviour'
save_path='MVPA_result/'
log_file_list = sorted(os.listdir(participant_log_path))

epoch_path_list = []
normal_trial_pre_list = []
normal_trial_post_list = []
normal_trial_NS_list = []

participants_index = np.arange(num_participants)
n_participants = len(participants_index)
num_participants = n_participants
thr = -t.ppf(0.05, num_participants-1)

for i in participants_index: #How many participants
    epoch_path_list = epoch_path_list + [get_epoch_path(root_path,i,blk_list[1],type_list[1])]
    _,train_index,_,pre_post_NS_index = load_catch_index_for_MVPA(participant_log_path+'/'+log_file_list[i])
    normal_trial_pre_NS_index = np.where((train_index==0) * (pre_post_NS_index==1))[0]
    normal_trial_post_NS_index = np.where((train_index==0) * (pre_post_NS_index==2))[0]
    normal_trial_NS_index = np.where((train_index==0) * (pre_post_NS_index==0))[0]
    normal_trial_pre_list = normal_trial_pre_list + [normal_trial_pre_NS_index]
    normal_trial_post_list = normal_trial_post_list + [normal_trial_post_NS_index]
    normal_trial_NS_list = normal_trial_NS_list + [normal_trial_NS_index]
    
normal_trial_pre_early_list = []
normal_trial_pre_late_list = []
normal_trial_post_early_list = []
normal_trial_post_late_list = []
for i in range(len(normal_trial_pre_list)):
    normal_trial_pre_early_list = normal_trial_pre_early_list + [normal_trial_pre_list[i][:60]]
    normal_trial_pre_late_list = normal_trial_pre_late_list + [normal_trial_pre_list[i][60:]]
    normal_trial_post_early_list = normal_trial_post_early_list + [normal_trial_post_list[i][:60]]
    normal_trial_post_late_list = normal_trial_post_late_list + [normal_trial_post_list[i][60:]]

normal_trial_early_list = normal_trial_pre_early_list + normal_trial_post_early_list
normal_trial_late_list = normal_trial_pre_late_list + normal_trial_post_late_list
none_picks = num_participants*[None]

    
###############################################################################
# Pre vs Post full band
scores,significant_mask,cluster,good_cluster,t_map,cluster_p_list = tg_permutation_standard(permutation_time = permutation_time,
                                                                        threshold=thr,
                                                                        chance=0.5,
                                                                        epoch1_path_list = epoch_path_list,
                                                                        epoch2_path_list = epoch_path_list,
                                                                        filter_band = None,
                                                                        label_1 = None,label_2 = None,
                                                                        picks_1 = normal_trial_pre_list,picks_2 = normal_trial_post_list,
                                                                        tlim = tlim,
                                                                        n_jobs = n_jobs,
                                                                        re_sfreq=re_sfreq,
                                                                        n_fold=n_fold,
                                                                        score_type='roc_auc',
                                                                        input_ch=None,
                                                                        reject_thr = reject_thr,crop_t = crop_t)

name = 'pre_vs_post_MVPA.mat'    
scio.savemat(save_path+name,{
    'scores':scores,
    'significant_mask':significant_mask,
    'cluster':cluster,
    'good_cluster':good_cluster,
    't_map':t_map,
    'cluster_p_list':cluster_p_list
                    })

fig1, ax = plt.subplots(1, 1)
im = plt.imshow(
    np.mean(scores,axis=0),
    interpolation="lanczos",
    origin="lower",
    cmap="RdBu_r",
    extent=plot_extent,
    vmin=0.4,
    vmax=0.6,
)
ax.axvline(0, color="k")
ax.axhline(0, color="k")
ax.set_title(name)
cbar = plt.colorbar(im, ax=ax)

###############################################################################
#pre vs post erp band early
scores,significant_mask,cluster,good_cluster,t_map,cluster_p_list = tg_permutation_standard(permutation_time = permutation_time,
                                                                        threshold=thr,
                                                                        chance=0.5,
                                                                        epoch1_path_list = epoch_path_list,
                                                                        epoch2_path_list = epoch_path_list,
                                                                        filter_band = None,
                                                                        label_1 = None,label_2 = None,
                                                                        picks_1 = normal_trial_pre_early_list,picks_2 = normal_trial_post_early_list,
                                                                        tlim = tlim,
                                                                        n_jobs = n_jobs,
                                                                        re_sfreq=re_sfreq,
                                                                        n_fold=n_fold,
                                                                        score_type='roc_auc',
                                                                        input_ch=None,
                                                                        reject_thr = reject_thr,crop_t = crop_t)

name = 'early_post_vs_pre_MVPA_erpband.mat'    
scio.savemat(save_path+name,{
    'scores':scores,
    'significant_mask':significant_mask,
    'cluster':cluster,
    'good_cluster':good_cluster,
    't_map':t_map,
    'cluster_p_list':cluster_p_list
                    })

fig2, ax = plt.subplots(1, 1)
im = plt.imshow(
    np.mean(scores,axis=0),
    interpolation="lanczos",
    origin="lower",
    cmap="RdBu_r",
    extent=plot_extent,
    vmin=0.4,
    vmax=0.6,
)
ax.axvline(0, color="k")
ax.axhline(0, color="k")
ax.set_title(name)
cbar = plt.colorbar(im, ax=ax)
###############################################################################
#pre vs post erp band late
scores,significant_mask,cluster,good_cluster,t_map,cluster_p_list = tg_permutation_standard(permutation_time = permutation_time,
                                                                        threshold=thr,
                                                                        chance=0.5,
                                                                        epoch1_path_list = epoch_path_list,
                                                                        epoch2_path_list = epoch_path_list,
                                                                        filter_band = None,
                                                                        label_1 = None,label_2 = None,
                                                                        picks_1 = normal_trial_pre_late_list,picks_2 = normal_trial_post_late_list,
                                                                        tlim = tlim,
                                                                        n_jobs = n_jobs,
                                                                        re_sfreq=re_sfreq,
                                                                        n_fold=n_fold,
                                                                        score_type='roc_auc',
                                                                        input_ch=None,
                                                                        reject_thr = reject_thr,crop_t = crop_t)

name = 'late_post_vs_pre_MVPA_erpband.mat'    
scio.savemat(save_path+name,{
    'scores':scores,
    'significant_mask':significant_mask,
    'cluster':cluster,
    'good_cluster':good_cluster,
    't_map':t_map,
    'cluster_p_list':cluster_p_list
                    })

fig2, ax = plt.subplots(1, 1)
im = plt.imshow(
    np.mean(scores,axis=0),
    interpolation="lanczos",
    origin="lower",
    cmap="RdBu_r",
    extent=plot_extent,
    vmin=0.4,
    vmax=0.6,
)
ax.axvline(0, color="k")
ax.axhline(0, color="k")
ax.set_title(name)
cbar = plt.colorbar(im, ax=ax)
