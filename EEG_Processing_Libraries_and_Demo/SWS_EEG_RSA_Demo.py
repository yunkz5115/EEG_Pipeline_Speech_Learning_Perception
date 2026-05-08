# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 13:42:01 2025

@author: yu028288
"""

import numpy as np
import os
import scipy.io as scio
import glob
from EEG_RSA_Package import (
    all_rsa,
    time_similarity_correlation_to_step
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

#Set hyper-parameters
input_ch = None
num_participants = 19
n_jobs = -1
base_re_sfreq = 50
allband_re_sfreq = 100
sws_thr_navg = [0.0001,3] # 100uV threshold, 3 epochs averaged pseudo epoch
ns_thr_navg = None

#Set load path
root_path = 'DemoData/Inf_ref/'
participant_log_path = 'Behaviour'
save_path='RSA_result/'
log_file_list = sorted(os.listdir(participant_log_path))

epoch_path_list = []
normal_trial_pre_list = []
normal_trial_post_list = []
normal_trial_NS_list = []

participants_index = np.arange(num_participants)
n_participants = len(participants_index)

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
    
none_picks = num_participants*[None]

# ###############################################################################
#RSA pre vs post all bands power
scores,conn_ref = all_rsa(permutation=False,current_epoch_list = None,return_epoch = False,
                          epoch1_path_list = epoch_path_list,
                          epoch2_path_list = epoch_path_list,
                          filter_band = [None,None],
                          label_1 = None,label_2 = None,
                          picks_1 = normal_trial_pre_list,picks_2 = normal_trial_post_list,
                          tlim = [-0.2,0.8], thr_navg1 = sws_thr_navg, thr_navg2 = sws_thr_navg,
                          n_jobs = n_jobs,
                          re_sfreq=allband_re_sfreq,
                          r_type='r2',input_ch=None,if_normalize=True,if_phase=False)
 
name = 'pre_vs_post_RSA_allband.mat'    
scio.savemat(save_path+name,{
    'scores':scores,
    'conn_ref':conn_ref})
# ###############################################################################
#RSA pre vs post all bands phase
scores,conn_ref = all_rsa(permutation=False,current_epoch_list = None,return_epoch = False,
                          epoch1_path_list = epoch_path_list,
                          epoch2_path_list = epoch_path_list,
                          filter_band = [None,None],
                          label_1 = None,label_2 = None,
                          picks_1 = normal_trial_pre_list,picks_2 = normal_trial_post_list,
                          tlim = [-0.2,0.8], thr_navg1 = sws_thr_navg, thr_navg2 = sws_thr_navg,
                          n_jobs = n_jobs,
                          re_sfreq=allband_re_sfreq,
                          r_type='r2',input_ch=None,if_normalize=True,if_phase=True)
 
name = 'pre_vs_post_RSA_allband_phase.mat'    
scio.savemat(save_path+name,{
    'scores':scores,
    'conn_ref':conn_ref})

# ###############################################################################
# Demo for LMM fit and visualization (without p-correction)
x_ticks_index = np.hstack([np.arange(0,50,10),49])
x_ticks_label = [-0.2,0,0.2,0.4,0.6,0.8]
t_list,p_list,distance_matrix,r_list,d_list,ci_low_list,ci_up_list = time_similarity_correlation_to_step(scores,conn_ref,col0_repeat=1.0,col1_repeat=1.0,if_permutation=False,if_trend_up=False,if_norm = True,baseline_fix = [-0.2,0])
p_corrected = p_list
fig, ax = plt.subplots()
ax.plot(np.arange(p_corrected.shape[0]),r_list,color='gray')
ax.plot([10,10],[-20,20],color='k')
ax.plot([0,50],[0,0],color='k')
significant_index = np.where(p_corrected<=0.05)[0]
ax.scatter(significant_index,np.ones(significant_index.shape),color='k',marker='*',s=10)
ax.set_xticks(x_ticks_index,x_ticks_label)
ax.set_ylim([-0.1,0.1])
ax.set_ylabel('Coef. ± Std. Err.')