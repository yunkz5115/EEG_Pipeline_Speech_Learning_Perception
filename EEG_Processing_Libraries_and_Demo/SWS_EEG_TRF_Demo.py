# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 15:14:26 2025

@author: yu028288
"""

import numpy as np
import os
import scipy.io as scio
import glob
from EEG_TRF_Analysis_Package import (
    sws_all_trf_standard,
    sws_all_full_epoch_trf_standard,
    )
from scipy.stats import t
import matplotlib.pyplot as plt

def get_epoch_path(root_path,sub_index,blk,type):
    participant_list = sorted(os.listdir(root_path))
    data_path = glob.glob(root_path+participant_list[sub_index]+'/'+'*'+blk+'_'+type+'_'+'-epo.fif')[0]
    return data_path
def load_catch_index_for_TRF(file_path):
    file = scio.loadmat(file_path)
    stim_index = file['participant_log']['stim_index'][0][0]
    pre_index = stim_index[stim_index[:,4]==1,3]
    train_index = stim_index[stim_index[:,4]==2,3]
    post_index = stim_index[stim_index[:,4]==3,3]
    pre_post_NS_index = np.zeros(stim_index.shape[0])
    NS_locker = np.where(stim_index[:,1]==1)[0]
    feature_index = []
    for loc in NS_locker:
        pre_post_NS_index[(loc-3):loc] = 1
        pre_post_NS_index[(loc+1):(loc+4)] = 2
        feature_index = feature_index + list(stim_index[(loc-3):loc,0].astype('int16')-1)
        feature_index = feature_index + [int(stim_index[loc,0]-1)]
        feature_index = feature_index + list(stim_index[(loc+1):(loc+4),0].astype('int16')-1)
        
    pre_post_NS_index = pre_post_NS_index[60:60+420]
    
    return pre_index,train_index,post_index,pre_post_NS_index,np.array(feature_index)

#Set epoch loading parameters
blk_list = ['blk-1','blk-2','blk-3']
type_list = ['lowband','highband']

#Set parameters
num_participants = 19
thr = -t.ppf(0.05, num_participants-1)
n_jobs = -1
base_re_sfreq = 50
allband_re_sfreq = 100
sws_thr_navg = [0.0001,3] # 100uV threshold, 3 epochs averaged pseudo epoch
ns_thr_navg = None
score_type = 'corrcoef'
t_min,t_max = [0,0.6]


#Set load path
root_path = 'DemoData/Inf_ref/'
participant_log_path = 'Behavior'
save_path='Pre_vs_Post_Individual_Feature/'

Envelop_feature_path = 'TRF_Features/Envelop/'
Cochleagram_feature_path = 'TRF_Features/Cochleagram/'
Modulation_feature_path = 'TRF_Features/Modulation/'
log_file_list = sorted(os.listdir(participant_log_path))

epoch_path_list = []
normal_trial_pre_list = []
normal_trial_post_list = []
normal_trial_NS_list = []
feature_index_list = []
feature_index_list_NS = []

participants_index = np.arange(num_participants)
n_participants = len(participants_index)

for i in participants_index: #How many participants
    epoch_path_list = epoch_path_list + [get_epoch_path(root_path,i,blk_list[1],type_list[1])]
    _,train_index,_,pre_post_NS_index,feature_index = load_catch_index_for_TRF(participant_log_path+'/'+log_file_list[i])
    normal_trial_pre_NS_index = np.where((train_index==0) * (pre_post_NS_index==1))[0]
    normal_trial_post_NS_index = np.where((train_index==0) * (pre_post_NS_index==2))[0]
    normal_trial_NS_index = np.where((train_index==0) * (pre_post_NS_index==0))[0]
    normal_trial_pre_list = normal_trial_pre_list + [normal_trial_pre_NS_index]
    normal_trial_post_list = normal_trial_post_list + [normal_trial_post_NS_index]
    normal_trial_NS_list = normal_trial_NS_list + [normal_trial_NS_index]
    
    feature_index_list = feature_index_list + [feature_index[(train_index==0) * (pre_post_NS_index==1)]]
    feature_index_list_NS = feature_index_list_NS + [feature_index[(train_index==0) * (pre_post_NS_index==0)]]
    
normal_trial_pre_early_list = []
normal_trial_pre_late_list = []
normal_trial_post_early_list = []
normal_trial_post_late_list = []
feature_index_early_list = []
feature_index_late_list = []

for i in range(len(normal_trial_pre_list)):
    normal_trial_pre_early_list = normal_trial_pre_early_list + [normal_trial_pre_list[i][:60]]
    normal_trial_pre_late_list = normal_trial_pre_late_list + [normal_trial_pre_list[i][60:]]
    normal_trial_post_early_list = normal_trial_post_early_list + [normal_trial_post_list[i][:60]]
    normal_trial_post_late_list = normal_trial_post_late_list + [normal_trial_post_list[i][60:]]
    feature_index_early_list = feature_index_early_list + [feature_index_list[i][:60]]
    feature_index_late_list = feature_index_late_list + [feature_index_list[i][60:]]
    
none_picks = num_participants*[None]

###############################################################################
# Full Band
# Envelop feature
score_all_list,coef_all_list = sws_all_trf_standard(epoch1_path_list = epoch_path_list,epoch2_path_list = epoch_path_list,
                                                    feature_path = Envelop_feature_path,feature1_index = feature_index_list_NS,feature2_index = feature_index_list_NS,
                                                    filter_band = [None,None],
                                                    label_1 = None,label_2 = None,
                                                    picks_1 = normal_trial_pre_list,picks_2 = normal_trial_post_list,
                                                    tlim = [-0.2,0.8],thr_navg1 = sws_thr_navg,thr_navg2 = sws_thr_navg,
                                                    n_jobs = n_jobs,
                                                    re_sfreq=allband_re_sfreq,
                                                    score_type=score_type,input_ch=None,if_abs = False,reject_thr=None,
                                                    t_min = t_min,t_max = t_max)
name = 'Envelop_trf.mat'    
scio.savemat(save_path+name,{
    'scores':np.array(score_all_list)*np.array(score_all_list),
    'conn_ref':np.array(coef_all_list)})

# Cochleagram feature
score_all_list,coef_all_list = sws_all_trf_standard(epoch1_path_list = epoch_path_list,epoch2_path_list = epoch_path_list,
                                                    feature_path = Cochleagram_feature_path,feature1_index = feature_index_list_NS,feature2_index = feature_index_list_NS,
                                                    filter_band = [None,None],
                                                    label_1 = None,label_2 = None,
                                                    picks_1 = normal_trial_pre_list,picks_2 = normal_trial_post_list,
                                                    tlim = [-0.2,0.8],thr_navg1 = sws_thr_navg,thr_navg2 = sws_thr_navg,
                                                    n_jobs = n_jobs,
                                                    re_sfreq=allband_re_sfreq,
                                                    score_type=score_type,input_ch=None,if_abs = False,reject_thr=None,
                                                    t_min = t_min,t_max = t_max)
name = 'Cochleagram_trf.mat'    
scio.savemat(save_path+name,{
    'scores':np.array(score_all_list)*np.array(score_all_list),
    'conn_ref':np.array(coef_all_list)})

# Modulation feature
score_all_list,coef_all_list = sws_all_trf_standard(epoch1_path_list = epoch_path_list,epoch2_path_list = epoch_path_list,
                                                    feature_path = Modulation_feature_path,feature1_index = feature_index_list_NS,feature2_index = feature_index_list_NS,
                                                    filter_band = [None,None],
                                                    label_1 = None,label_2 = None,
                                                    picks_1 = normal_trial_pre_list,picks_2 = normal_trial_post_list,
                                                    tlim = [-0.2,0.8],thr_navg1 = sws_thr_navg,thr_navg2 = sws_thr_navg,
                                                    n_jobs = n_jobs,
                                                    re_sfreq=allband_re_sfreq,
                                                    score_type=score_type,input_ch=None,if_abs = False,reject_thr=None,
                                                    t_min = t_min,t_max = t_max)
name = 'Modulation_trf.mat'    
scio.savemat(save_path+name,{
    'scores':np.array(score_all_list)*np.array(score_all_list),
    'conn_ref':np.array(coef_all_list)})
