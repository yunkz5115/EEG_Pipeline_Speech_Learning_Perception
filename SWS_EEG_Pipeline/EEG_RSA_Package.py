# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 13:57:01 2025

@author: yu028288
"""

import numpy as np
import scipy.io as scio
import mne
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from mne.decoding import (
    SlidingEstimator,
    GeneralizingEstimator,
    cross_val_multiscore,
    get_coef,
    LinearModel,
)
from EEG_MVPA_Package import (
    mvpa_cluster
    )
from mne.viz import (plot_topomap as plot_topo,
                     plot_brain_colorbar)

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

import os
from scipy.stats import ttest_1samp
from scipy.stats import pearsonr as pear_r
from tqdm import trange
import tqdm
from joblib import Parallel, delayed
import scipy.io as scio
import pandas as pd
import statsmodels.api as sm
from tqdm import trange
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from scipy.signal import hilbert

t_start = -0.2
trial_length = 1
###############################################################################
#Standard Ant-Neuro Preprocessed .fif loader
def create_pseudo_trial_by_thr(data,thr,n_avg):
    data_pseudo_trial = np.zeros([int(data.shape[0]/n_avg),data.shape[1],data.shape[2]])
    for i in range(data_pseudo_trial.shape[0]):
        non_rej_count = 0
        for j in range(n_avg):
            if np.max(np.abs(data[n_avg*i+j,:,:]))<=thr:
                non_rej_count = non_rej_count + 1
                data_pseudo_trial[i,:,:] = data_pseudo_trial[i,:,:] + data[n_avg*i+j,:,:]
                
        if non_rej_count!=0:
            data_pseudo_trial[i,:,:] = data_pseudo_trial[i,:,:]/non_rej_count
        else:
            data_pseudo_trial[i,:,:] = np.mean(data[n_avg*i:n_avg*(i+1),:,:],axis=0)
            print('\nCannot reject trials!\n')
        
    return data_pseudo_trial
        
def tg_fif_epoch_loader(epoch1_path,epoch2_path,
                        filter_band = [None,None],
                        label_1 = None,label_2 = None,
                        picks_1 = None,picks_2 = None,
                        tlim = None,thr_navg1 = None,thr_navg2 = None,
                        n_jobs = 8,
                        if_abs = False,if_phase=False,reject_thr = None):
    if label_1 is not None:
        epoch1 = mne.read_epochs(epoch1_path)[label_1]
    else:
        epoch1 = mne.read_epochs(epoch1_path)
    if label_2 is not None:
        epoch2 = mne.read_epochs(epoch2_path)[label_2]
    else:
        epoch2 = mne.read_epochs(epoch2_path)
    
    if filter_band[1] is not None:
        epoch1 = epoch1.filter(h_freq=filter_band[1], l_freq=None,n_jobs=n_jobs,method='fir')
        epoch2 = epoch2.filter(h_freq=filter_band[1], l_freq=None,n_jobs=n_jobs,method='fir')
    if filter_band[0] is not None:
        epoch1 = epoch1.filter(h_freq=None, l_freq=filter_band[0],n_jobs=n_jobs,method='fir')
        epoch2 = epoch2.filter(h_freq=None, l_freq=filter_band[0],n_jobs=n_jobs,method='fir')
    
    if tlim is not None:
        epoch1.crop(tmin=tlim[0], tmax=tlim[1])
        epoch2.crop(tmin=tlim[0], tmax=tlim[1])
        
    if picks_1 is not None:
        epoch1 = epoch1[picks_1]
    if picks_2 is not None:
        epoch2 = epoch2[picks_2]
        
    data1 = epoch1.get_data()
    data2 = epoch2.get_data()
    
    if if_abs:
        data1 = np.abs(hilbert(data1,axis=-1))
        data2 = np.abs(hilbert(data2,axis=-1))
    
    if reject_thr is not None:
        data1_reject_index = np.where(np.max(np.max(data1,axis=-1),axis=-1)>reject_thr)[0]
        data2_reject_index = np.where(np.max(np.max(data2,axis=-1),axis=-1)>reject_thr)[0]
        data1 = np.delete(data1,data1_reject_index,axis=0)
        data2 = np.delete(data2,data2_reject_index,axis=0)
        
        
    if thr_navg1 is not None:
        data1 = create_pseudo_trial_by_thr(data1,thr_navg1[0],thr_navg1[1])
    if thr_navg2 is not None:
        data2 = create_pseudo_trial_by_thr(data2,thr_navg2[0],thr_navg2[1])
        
    if (not if_abs)&(if_phase):
        data1 = np.unwrap(np.angle(hilbert(data1,axis=-1)))
        data2 = np.unwrap(np.angle(hilbert(data2,axis=-1)))
        
    data = np.vstack([data1,data2])
    print('\n\nReject thr = ',reject_thr,'  Data1 shape = ',data1.shape[0],'  Data2 shape = ',data2.shape[0],'\n\n')
    events = np.zeros([data.shape[0],3])
    events[:,0] = np.arange(data.shape[0])
    events[:,2] = np.hstack([np.zeros(data1.shape[0]),np.ones(data2.shape[0])])
    events = events.astype('int64')
    event_id = {"event1": 0, "event2": 1}
    
    t_start = epoch1.baseline[0]
    tg_epoch = mne.EpochsArray(data, info=epoch1.info, events=events,tmin=epoch1.tmin,event_id=event_id,baseline=epoch1.baseline)
    tg_epoch.apply_baseline()
    return tg_epoch
###############################################################################
def epoch_normalize_by_ch(X):
    #Note X shape: n_epochs, n_eeg_channels, n_times
    X1 = X.copy()
    normalize_factor = np.max(abs(X),axis=1)
    for i in range(X.shape[1]):
        X1[:,i,:] = X[:,i,:]/normalize_factor
    return X1

def erp_rsa(erp1,erp2,r_type='r',input_ch = None,n_jobs=-1):
    time_series = np.zeros(erp1.shape[1])
    for i in range(len(time_series)):
        if r_type=='r':
            time_series[i],_ = pear_r(erp1[:,i],erp2[:,i])
        elif r_type=='r2':
            r,_ = pear_r(erp1[:,i],erp2[:,i])
            time_series[i] = r*r
        elif r_type=='euclidean':
            time_series[i] = np.sqrt((erp1[:,i]-erp2[:,i])**2)
    return time_series

def epoch_erp_rsa(epochs,re_sfreq=25,permutation=False,if_normalize = False,
                  r_type='r',input_ch = None,n_jobs=8):
    epochs = epochs.resample(sfreq=re_sfreq)
    X = epochs.get_data()  # EEG signals: n_epochs, n_eeg_channels, n_times
    if not input_ch is None:
        X = X[:,input_ch,:]
    y = epochs.events[:, 2]  # target: auditory left vs visual left
    
    if permutation:
        np.random.shuffle(y)
    
    if if_normalize:
        X = epoch_normalize_by_ch(X)
        
    index_0 = np.where(y==np.unique(y)[0])[0]
    index_1 = np.where(y==np.unique(y)[1])[0]
    scores = np.zeros([len(index_0)*len(index_1),X.shape[2]])
    conn_ref = np.zeros([len(index_0)*len(index_1),2])
    kk = 0
    for i in trange(len(index_0)):
        for j in range(len(index_1)):
            scores[kk,:] = erp_rsa(X[index_0[i],:,:],X[index_0[j],:,:],r_type=r_type,input_ch = input_ch,n_jobs=n_jobs)
            conn_ref[kk,0] = i
            conn_ref[kk,1] = j
            kk+=1
    print('\n\nRSM calculation finished.\n\n')
    return scores,conn_ref

def epoch_erp_rsa_parallel(epochs,re_sfreq=25,permutation=False,if_normalize = False,
                  r_type='r',input_ch = None,n_jobs=8):
    epochs = epochs.resample(sfreq=re_sfreq)
    X = epochs.get_data()  # EEG signals: n_epochs, n_eeg_channels, n_times
    if not input_ch is None:
        X = X[:,input_ch,:]
    y = epochs.events[:, 2]  # target: auditory left vs visual left
    
    if permutation:
        np.random.shuffle(y)
    
    if if_normalize:
        X = epoch_normalize_by_ch(X)
        
    index_0 = np.where(y==np.unique(y)[0])[0]
    index_1 = np.where(y==np.unique(y)[1])[0]
    # Define a helper function to compute RSA for a single pair (i, j)
    def _compute_rsa_for_pair(i_idx, j_idx, X_data, r_type_val, input_ch_val):
        """
        Helper function to be called in parallel.
        Computes ERP RSA for a single pair of epochs.
        """
        # Ensure that X_data[i_idx,:,:] and X_data[j_idx,:,:] are passed directly
        # to erp_rsa. The 'input_ch' parameter for erp_rsa is essentially handled
        # by the slicing of X *before* this function is called, so erp_rsa's
        # input_ch can be ignored or removed if it's only for X slicing.
        # For clarity, I'm passing None to erp_rsa's input_ch since X is already sliced.
        scores_ij = erp_rsa(X_data[i_idx, :, :], X_data[j_idx, :, :],
                            r_type=r_type_val, input_ch=input_ch_val) # Pass original input_ch if erp_rsa uses it internally
        return scores_ij, i_idx, j_idx

    print(f"Starting parallel RSA calculation with {len(index_0) * len(index_1)} pairs...")

    # Create a list of delayed calls
    # We iterate over the indices and create tasks for each (i, j) pair
    # Using 'tqdm' with 'Parallel' can be tricky, so we'll collect results and then assemble.
    tasks = []
    for i in index_0:
        for j in index_1:
            tasks.append(
                delayed(_compute_rsa_for_pair)(
                    i, j, X, r_type, input_ch # Pass X directly here
                )
            )

    # Execute tasks in parallel
    # The `backend='loky'` is typically good for CPU-bound tasks
    results_iterator = Parallel(n_jobs=n_jobs, backend='loky')(tasks)
    print('Parallel iterator is set up...')
    # Wrap the iterator with tqdm to display a progress bar
    # `total` is important for tqdm to know the total number of items
    # `desc` adds a description to the progress bar
    # `leave=True` keeps the progress bar visible after completion
    results_list = list(tqdm.tqdm(results_iterator, total=len(tasks), desc="RSA Pairs", leave=True))
    #print('Progress bar is set up...')
    # Initialize scores and conn_ref arrays based on the total number of pairs
    total_pairs = len(index_0) * len(index_1)
    scores = np.zeros([total_pairs, X.shape[2]])
    conn_ref = np.zeros([total_pairs, 2], dtype=int)

    kk = 0
    # Process results from parallel execution
    for scores_ij, original_i, original_j in results_list:
        # Need to find the original relative indices for conn_ref
        # `index_0.tolist().index(original_i)` gives the position of `original_i` within `index_0`
        conn_ref[kk, 0] = index_0.tolist().index(original_i)
        conn_ref[kk, 1] = index_1.tolist().index(original_j)
        scores[kk, :] = scores_ij
        kk += 1

    print('\n\nRSM calculation finished.\n\n')
    return scores,conn_ref


def all_rsa(permutation=False,current_epoch_list = None,return_epoch = False,
                        epoch1_path_list = [],epoch2_path_list = [],
                        filter_band = None,
                        label_1 = '',label_2 = '',
                        picks_1 = None,picks_2 = None,
                        tlim = None,thr_navg1 = None,thr_navg2 = None,
                        n_jobs = 8,
                        re_sfreq=None,
                        r_type='r2',input_ch=None,if_normalize=True,if_abs=False,if_phase=False,reject_thr=None):
    
    if current_epoch_list is None:
        current_epoch_list = []
        scores_all = []
        for score_count,[epoch1_path,epoch2_path] in enumerate(zip(epoch1_path_list,epoch2_path_list)):
            
            epochs = tg_fif_epoch_loader(epoch1_path,epoch2_path,
                                         filter_band=filter_band,
                                         label_1=label_1,label_2=label_2,
                                         picks_1=picks_1[score_count],picks_2=picks_2[score_count],
                                         tlim = tlim,thr_navg1 = thr_navg1,thr_navg2 = thr_navg2,n_jobs = n_jobs,if_abs=if_abs,if_phase=if_phase,reject_thr=reject_thr)
            scores,conn_ref = epoch_erp_rsa_parallel(epochs,re_sfreq=re_sfreq,permutation=permutation,if_normalize = if_normalize,r_type=r_type,input_ch = input_ch,n_jobs=n_jobs)
            scores_all = scores_all + [scores]
            if return_epoch:
                current_epoch_list = current_epoch_list + [epochs]
    else:
        scores_all = []
        for score_count,epochs in enumerate(current_epoch_list):
            scores,conn_ref = epoch_erp_rsa_parallel(epochs,re_sfreq=re_sfreq,permutation=permutation,if_normalize = if_normalize,r_type=r_type,input_ch = input_ch,n_jobs=n_jobs)
            scores_all = scores_all + [scores]
    if return_epoch:
        return np.array(scores_all),current_epoch_list,conn_ref
    else:
        return np.array(scores_all),conn_ref   
    
def return_avg_rsm(scores,conn_ref,time_locker):
    matrix_row = len(np.unique(conn_ref[:,0]))
    matrix_col = len(np.unique(conn_ref[:,1]))
    matrix = np.nan*np.zeros([matrix_row,matrix_col])
    scores_avg = np.mean(scores[:,time_locker[0]:time_locker[1]],axis=1)
    for i in range(conn_ref.shape[0]):
        matrix[conn_ref[i,0],conn_ref[i,1]] = scores_avg[i]

    return matrix

def rsm_plot(time_window = [-0.2,0.8],
             path = ' ',window_size = 0.1,sample_r = 50,if_fisher_z = False,if_square = False,if_norm = False,layout = [2,5],baseline_fix = None,if_norm_plot = False,
             cmap='viridis',vmin=None,vmax=None,figsize=None,title=' ',shrink=0.8,pad=0.1,fontsize=8,if_colorbar=True,interpolation='auto'):
    time_locker = []
    for i in range(int((time_window[1]-time_window[0])/window_size)):
        time_locker = time_locker + [[int(i*window_size*sample_r),int((i+1)*window_size*sample_r)]]
    rsa_result = scio.loadmat(path)
    scores = rsa_result['scores']
    if if_fisher_z:
        scores = np.arctanh(scores)
    if if_square:
        scores = scores**2
    if baseline_fix is not None:
        if len(baseline_fix)==2:
            start_idx = int(scores.shape[0]*(baseline_fix[0]+t_start)/trial_length)
            end_idx = int(scores.shape[0]*(baseline_fix[1]+t_start)/trial_length)
            for participant_idx in range(scores.shape[0]):
                current_score = scores[participant_idx,:,:]
                fixer = np.mean(current_score[:,start_idx:end_idx],axis=-1)
                for i in range(scores.shape[2]):
                    scores[participant_idx,:,i] = scores[participant_idx,:,i]-fixer
    scores = np.mean(scores,0)
    if if_norm:
        for i in range(scores.shape[1]):
            scores[:,i] = 2*(scores[:,i]-np.min(scores[:,i])/(np.max(scores[:,i])-np.min(scores[:,i])))-1
    conn_ref = rsa_result['conn_ref']
    
    figure_rsa,axis = plt.subplots(layout[0],layout[1],figsize=figsize)
    for i,ax in enumerate(axis.flat):
        matrix_plot = return_avg_rsm(scores,conn_ref,time_locker[i])
        if if_norm_plot:
            matrix_plot = 2*(matrix_plot-np.min(matrix_plot)/(np.max(matrix_plot)-np.min(matrix_plot)))-1
        ax.imshow(matrix_plot,origin="lower",cmap=cmap,vmin=vmin,vmax=vmax,interpolation = interpolation)
        ax.set_title(f'{np.round(float((time_locker[i][0])/sample_r+time_window[0])*1000,-1):.0f} - {np.round(float((time_locker[i][1])/sample_r+time_window[0])*1000,-1):.0f}',fontsize = fontsize)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([0,matrix_plot.shape[1]],['Trial 1','Trial N'])
        ax.set_yticks([0,matrix_plot.shape[0]],['Trial 1','Trial n'])
    plt.tight_layout()
    return figure_rsa

def rsm_plot_no_label(time_window = [-0.2,0.8],
             path = ' ',window_size = 0.1,sample_r = 50,if_fisher_z = False,if_square = False,if_norm = False,layout = [2,5],baseline_fix = None,if_norm_plot = False,
             cmap='viridis',vmin=None,vmax=None,figsize=None,title=' ',shrink=0.8,pad=0.1,fontsize=8,if_colorbar=True,interpolation='auto'):
    time_locker = []
    for i in range(int((time_window[1]-time_window[0])/window_size)):
        time_locker = time_locker + [[int(i*window_size*sample_r),int((i+1)*window_size*sample_r)]]
    rsa_result = scio.loadmat(path)
    scores = rsa_result['scores']
    if if_fisher_z:
        scores = np.arctanh(scores)
    if if_square:
        scores = scores**2
    if baseline_fix is not None:
        if len(baseline_fix)==2:
            start_idx = int(scores.shape[0]*(baseline_fix[0]+t_start)/trial_length)
            end_idx = int(scores.shape[0]*(baseline_fix[1]+t_start)/trial_length)
            for participant_idx in range(scores.shape[0]):
                current_score = scores[participant_idx,:,:]
                fixer = np.mean(current_score[:,start_idx:end_idx],axis=-1)
                for i in range(scores.shape[2]):
                    scores[participant_idx,:,i] = scores[participant_idx,:,i]-fixer
    scores = np.mean(scores,0)
    if if_norm:
        for i in range(scores.shape[1]):
            scores[:,i] = 2*(scores[:,i]-np.min(scores[:,i])/(np.max(scores[:,i])-np.min(scores[:,i])))-1
    conn_ref = rsa_result['conn_ref']
    
    figure_rsa,axis = plt.subplots(layout[0],layout[1],figsize=figsize)
    for i,ax in enumerate(axis.flat):
        matrix_plot = return_avg_rsm(scores,conn_ref,time_locker[i])
        if if_norm_plot:
            matrix_plot = 2*(matrix_plot-np.min(matrix_plot)/(np.max(matrix_plot)-np.min(matrix_plot)))-1
        ax.imshow(matrix_plot,origin="lower",cmap=cmap,vmin=vmin,vmax=vmax,interpolation = interpolation)
        #ax.set_title(f'{np.round(float((time_locker[i][0])/sample_r+time_window[0])*1000,-1):.0f} - {np.round(float((time_locker[i][1])/sample_r+time_window[0])*1000,-1):.0f}',fontsize = fontsize)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    return figure_rsa

def return_sws_trial_similairty(scores,conn_ref):
    matrix_row = len(np.unique(conn_ref[:,0]))
    matrix_col = len(np.unique(conn_ref[:,1]))
    trial_similairty = np.nan*np.zeros([scores.shape[0],matrix_row,scores.shape[2]])
    for participant_i in range(scores.shape[0]):
        matrix = np.nan*np.zeros([matrix_row,matrix_col,scores.shape[2]])
        for i in range(conn_ref.shape[0]):
            matrix[conn_ref[i,0],conn_ref[i,1],:] = scores[participant_i,i,:]
        trial_similairty[participant_i,:,:] = np.mean(matrix,axis=1)

    return trial_similairty
    
def return_ns_trial_similairty(scores,conn_ref):
    matrix_row = len(np.unique(conn_ref[:,0]))
    matrix_col = len(np.unique(conn_ref[:,1]))
    trial_similairty = np.nan*np.zeros([scores.shape[0],matrix_col,scores.shape[2]])
    for participant_i in range(scores.shape[0]):
        matrix = np.nan*np.zeros([matrix_row,matrix_col,scores.shape[2]])
        for i in range(conn_ref.shape[0]):
            matrix[conn_ref[i,0],conn_ref[i,1],:] = scores[participant_i,i,:]
        trial_similairty[participant_i,:,:] = np.mean(matrix,axis=0)

    return trial_similairty

def rsm_plot_sws_ns_avg(time_window = [-0.2,0.8],
                     path = ' ',window_size = 0.1,sample_r = 50,if_fisher_z = False,if_square = False,
                     cmap='viridis',vmin=None,vmax=None,figsize=None,title=' ',shrink=0.8,pad=0.1,fontsize=8,if_colorbar=True):
    rsa_result = scio.loadmat(path)
    scores = rsa_result['scores']
    if if_fisher_z:
        scores = np.arctanh(scores)
    if if_square:
        scores = scores**2
    conn_ref = rsa_result['conn_ref']
    trial_similairty = return_sws_trial_similairty(scores,conn_ref)
    print(trial_similairty.shape)
    avg_trial_similairty = np.mean(trial_similairty,axis=0)
    fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.imshow(avg_trial_similairty,origin="lower",cmap=cmap,aspect='auto',vmin=vmin,vmax=vmax)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    time_vec = np.arange(0,60,10)
    time_vec[-1] = time_vec[-1]-1
    time_label = np.round(np.arange(0,60,10)*(time_window[1]-time_window[0])/sample_r+time_window[0],1)
    ax.set_yticks([0,avg_trial_similairty.shape[0]],['Trial 1','Trial N'])
    ax.set_ylabel('Trial Index')
    ax.set_xticks(time_vec,time_label)
    ax.set_xlabel('Time')
    plt.tight_layout()
    return fig

def rsm_plotline_sws_ns_avg(time_window = [-0.2,0.8],
                     path = ' ',window_size = 0.1,sample_r = 50,if_fisher_z = False,if_square = False,
                     cmap='viridis',vmin=None,vmax=None,figsize=None,title=' ',shrink=0.8,pad=0.1,fontsize=8,if_colorbar=True):
    rsa_result = scio.loadmat(path)
    scores = rsa_result['scores']
    if if_fisher_z:
        scores = np.arctanh(scores)
    if if_square:
        scores = scores**2
    conn_ref = rsa_result['conn_ref']
    trial_similairty = return_sws_trial_similairty(scores,conn_ref)
    print(trial_similairty.shape)
    avg_trial_similairty = np.mean(trial_similairty,axis=0)
    fig,ax = plt.subplots(1,1,figsize=figsize)
    for i in range(avg_trial_similairty.shape[1]):
        ax.plot(avg_trial_similairty[:,i],alpha=0.3)
    plt.tight_layout()
    return fig


def rsm_plot_ns_sws_avg(time_window = [-0.2,0.8],
                     path = ' ',window_size = 0.1,sample_r = 50,if_fisher_z = False,if_square = False,
                     cmap='viridis',vmin=None,vmax=None,figsize=None,title=' ',shrink=0.8,pad=0.1,fontsize=8,if_colorbar=True):
    rsa_result = scio.loadmat(path)
    scores = rsa_result['scores']
    if if_fisher_z:
        scores = np.arctanh(scores)
    if if_square:
        scores = scores**2
    conn_ref = rsa_result['conn_ref']
    trial_similairty = return_ns_trial_similairty(scores,conn_ref)
    print(trial_similairty.shape)
    avg_trial_similairty = np.mean(trial_similairty,axis=0)
    fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.imshow(avg_trial_similairty,origin="lower",cmap=cmap,aspect='auto',vmin=vmin,vmax=vmax)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    time_vec = np.arange(0,60,10)
    time_vec[-1] = time_vec[-1]-1
    time_label = np.round(np.arange(0,60,10)*(time_window[1]-time_window[0])/sample_r+time_window[0],1)
    ax.set_yticks([0,avg_trial_similairty.shape[0]],['Trial 1','Trial N'])
    ax.set_ylabel('Trial Index')
    ax.set_xticks(time_vec,time_label)
    ax.set_xlabel('Time')
    plt.tight_layout()
    return fig

def build_rsa_trial_df(scores,conn_ref,t=0,col0_repeat=3.0,col1_repeat=3.0,if_trend_up=True,if_trend_sqrt=True):
    distance_vector = np.zeros(conn_ref.shape[0])
    matrix_row = len(np.unique(conn_ref[:,0]))
    matrix_col = len(np.unique(conn_ref[:,1]))
    distance_matrix = np.nan*np.zeros([matrix_row,matrix_col,scores.shape[2]])
    for i in range(conn_ref.shape[0]):
        if if_trend_up:
            if if_trend_sqrt:
                distance_vector[i] = np.sqrt((np.floor(np.min(conn_ref[:,0])/col0_repeat)-np.floor(conn_ref[i,0]/col0_repeat))**2+(np.floor(np.min(conn_ref[:,1])/col1_repeat)-np.floor(conn_ref[i,1]/col1_repeat))**2)
            else:
                distance_vector[i] = (np.floor(np.min(conn_ref[:,0])/col0_repeat)-np.floor(conn_ref[i,0]/col0_repeat))**2+(np.floor(np.min(conn_ref[:,1])/col1_repeat)-np.floor(conn_ref[i,1]/col1_repeat))**2
        else:
            if if_trend_sqrt:
                distance_vector[i] = np.sqrt((np.floor(np.max(conn_ref[:,0])/col0_repeat)-np.floor(conn_ref[i,0]/col0_repeat))**2+(np.floor(np.max(conn_ref[:,1])/col1_repeat)-np.floor(conn_ref[i,1]/col1_repeat))**2)
            else:
                distance_vector[i] = (np.floor(np.max(conn_ref[:,0])/col0_repeat)-np.floor(conn_ref[i,0]/col0_repeat))**2+(np.floor(np.max(conn_ref[:,1])/col1_repeat)-np.floor(conn_ref[i,1]/col1_repeat))**2
        distance_matrix[conn_ref[i,0],conn_ref[i,1],:] = distance_vector[i]
    data = []
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            data.append({'SubjectID': i, 'Step': distance_vector[j], 'Score': scores[i,j,t]})

    df = pd.DataFrame(data)
    return df,distance_matrix
    
def time_similarity_correlation_to_step(scores,conn_ref,if_permutation,if_fisher_z = False,if_square = False,if_norm = False,baseline_fix = None,**kwargs):
    #Get p value, z value, correlation and std of corr
    p_list = np.zeros([scores.shape[2]])
    t_list = np.zeros([scores.shape[2]])
    r_list = np.zeros([scores.shape[2]])
    d_list = np.zeros([scores.shape[2]])
    if if_fisher_z:
        scores = np.arctanh(scores)
    if if_square:
        scores = scores**2
    _,distance_matrix = build_rsa_trial_df(scores,conn_ref,t=0,**kwargs)
    if baseline_fix is not None:
        if len(baseline_fix)==2:
            start_idx = int(scores.shape[0]*(baseline_fix[0]+t_start)/trial_length)
            end_idx = int(scores.shape[0]*(baseline_fix[1]+t_start)/trial_length)
            for participant_idx in range(scores.shape[0]):
                current_score = scores[participant_idx,:,:]
                fixer = np.mean(current_score[:,start_idx:end_idx],axis=-1)
                for i in range(scores.shape[2]):
                    scores[participant_idx,:,i] = scores[participant_idx,:,i]-fixer
    if if_norm:
        for i in range(scores.shape[2]):
            scores[:,:,i] = (scores[:,:,i]-np.min(scores[:,:,i]))/(np.max(scores[:,:,i])-np.min(scores[:,:,i]))*np.max(distance_matrix)
    if if_permutation:
        distance_matrix_flatten = distance_matrix.flatten()
        np.random.shuffle(distance_matrix_flatten)
        distance_matrix = distance_matrix_flatten.reshape(distance_matrix.shape)
    for t in trange(scores.shape[2]):
        df,_ = build_rsa_trial_df(scores,conn_ref,t=t,**kwargs)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model = sm.MixedLM.from_formula("Score ~ Step", data=df, groups=df["SubjectID"])
            result = model.fit()
        
        p_list[t] = result.pvalues['Step']
        t_list[t] = result.tvalues['Step']
        r_list[t] = result.fe_params['Step']
        d_list[t] = result.bse['Step']
    return t_list,p_list,distance_matrix,r_list,d_list

def build_rsa_trial_2waydf(scores,scores1,conn_ref,conn_ref1,t=0,col0_repeat=3.0,col1_repeat=3.0,col0_1_repeat=3.0,col1_1_repeat=3.0):
    distance_vector = np.zeros(conn_ref.shape[0])
    matrix_row = len(np.unique(conn_ref[:,0]))
    matrix_col = len(np.unique(conn_ref[:,1]))
    distance_matrix = np.nan*np.zeros([matrix_row,matrix_col,scores.shape[2]])
    for i in range(conn_ref.shape[0]):
        distance_vector[i] = (np.floor(np.max(conn_ref[:,0])/col0_repeat)-np.floor(conn_ref[i,0]/col0_repeat))**2+(np.floor(np.max(conn_ref[:,1])/col1_repeat)-np.floor(conn_ref[i,1]/col1_repeat))**2
        distance_matrix[conn_ref[i,0],conn_ref[i,1],:] = distance_vector[i]
        
    distance_vector1 = np.zeros(conn_ref1.shape[0])
    matrix_row = len(np.unique(conn_ref1[:,0]))
    matrix_col = len(np.unique(conn_ref1[:,1]))
    distance_matrix1 = np.nan*np.zeros([matrix_row,matrix_col,scores1.shape[2]])
    for i in range(conn_ref1.shape[0]):
        distance_vector1[i] = (np.floor(np.max(conn_ref1[:,0])/col0_1_repeat)-np.floor(conn_ref1[i,0]/col0_1_repeat))**2+(np.floor(np.max(conn_ref1[:,1])/col1_1_repeat)-np.floor(conn_ref1[i,1]/col1_1_repeat))**2
        distance_matrix1[conn_ref1[i,0],conn_ref1[i,1],:] = distance_vector1[i]
        
    data = []
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            data.append({'SubjectID': i, 'Step': distance_vector[j], 'Score': scores[i,j,t], 'Type':0})
            data.append({'SubjectID': i, 'Step': distance_vector[j], 'Score': scores1[i,j,t], 'Type':1})

    df = pd.DataFrame(data)
    df['Type'] = pd.Categorical(df['Type'])
    return df,distance_matrix,distance_matrix1
    
def time_similarity_correlation_to_step_anova(scores,scores1,conn_ref,conn_ref1,if_permutation,if_fisher_z = False,if_square = False,**kwargs):
    p_list = np.zeros([scores.shape[2]])
    t_list = np.zeros([scores.shape[2]])
    if if_fisher_z:
        scores = np.arctanh(scores)
    if if_square:
        scores = scores**2
    _,distance_matrix,distance_matrix1 = build_rsa_trial_df(scores,conn_ref,t=0,**kwargs)
    if if_permutation:
        distance_matrix_flatten = distance_matrix.flatten()
        np.random.shuffle(distance_matrix_flatten)
        distance_matrix = distance_matrix_flatten.reshape(distance_matrix.shape)
        distance_matrix_flatten = distance_matrix1.flatten()
        np.random.shuffle(distance_matrix_flatten)
        distance_matrix1 = distance_matrix_flatten.reshape(distance_matrix1.shape)
    
    for t in trange(scores.shape[2]):
        df,_,_ = build_rsa_trial_2waydf(scores,scores1,conn_ref,conn_ref1,t=t,**kwargs)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model = smf.mixedlm("Score ~ Step*C(Type)", data=df, groups=df["SubjectID"],re_formula="~ Step")
            result = model.fit()
        
        p_list[t] = result.pvalues[3]
        t_list[t] = result.tvalues[3]
    return t_list,p_list,distance_matrix,distance_matrix


def temporal_cluster(t_list,threshold):
    t_map_mask = t_list[t_list<=threshold]
    cluster,clu_num = mvpa_cluster(t_map_mask)
    return t_list,t_map_mask,cluster,clu_num

def all_rsa_time_step_correlation(scores,conn_ref,permutation_time,threshold,n_jobs=-1,if_fisher_z = False,if_square = False,**kwargs):
    t_list,_,_ = time_similarity_correlation_to_step(scores,conn_ref,False,if_fisher_z = False,if_square = False,**kwargs)
    t_map,t_map_mask,cluster,clu_num = temporal_cluster(t_list,threshold)
    permutation_max_t = np.zeros(permutation_time)
    for i in range(permutation_time):
        permutation_t_list,_,_ = time_similarity_correlation_to_step(scores,conn_ref,True,if_fisher_z = False,if_square = False,**kwargs)
        permutation_t_map,permutation_t_map_mask,permutation_cluster,permutation_clu_num = temporal_cluster(permutation_t_list,threshold)
        if permutation_clu_num>0:
            permutation_t_list = np.zeros(permutation_clu_num)
            for j in range(permutation_clu_num):
                permutation_t_list[j] = np.sum((np.abs(permutation_t_map)-threshold)*permutation_cluster[j])
            permutation_max_t[i] = np.max(permutation_t_list)
        else:
            permutation_max_t[i] = 0
        
    good_cluster = []
    cluster_p_list = []
    for i in range(clu_num):
        sum_t = np.sum((np.abs(t_map)-threshold)*cluster[i])
        cluster_p = np.sum(permutation_max_t>sum_t)/len(permutation_max_t)
        cluster_p_list = cluster_p_list + [cluster_p]
        if cluster_p<=0.05:
            good_cluster = good_cluster + [i]
    
    return t_map,t_map_mask,cluster,good_cluster,np.array(cluster_p_list)
    
def all_rsa_time_step_correlation_parallel(scores, conn_ref, permutation_time, threshold, n_jobs=-1,if_fisher_z = False,if_square = False, **kwargs):
    """
    Performs RSA time-step correlation analysis with parallelized permutations using joblib.
    """
    print("Calculating observed t-map...")
    if if_fisher_z:
        scores = np.arctanh(scores)
    if if_square:
        scores = scores**2
    t_list, _, _ = time_similarity_correlation_to_step(scores, conn_ref, False,if_fisher_z = False,if_square = False, **kwargs)
    t_map, t_map_mask, cluster, clu_num = temporal_cluster(t_list, threshold)

    print(f"Starting {permutation_time} permutation analyses in parallel with {n_jobs} jobs...")
    permutation_max_t = np.zeros(permutation_time)

    # Define a helper function to run a single permutation iteration
    def _single_permutation_iteration(iteration_index, scores_data, conn_ref_data, threshold_val, **kwargs_val):
        """
        Executes one iteration of the permutation analysis.
        This function will be called in parallel.
        """
        permutation_t_list, _, _ = time_similarity_correlation_to_step(scores_data, conn_ref_data, True,if_fisher_z = False,if_square = False, **kwargs_val)
        permutation_t_map, _, permutation_cluster, permutation_clu_num = temporal_cluster(permutation_t_list, threshold_val)

        current_permutation_max_t = 0
        if permutation_clu_num > 0:
            # Re-calculating permutation_t_list as sum of t-values in clusters
            # Note: `permutation_t_list` here is being reused to store sums, which might be confusing.
            # Let's rename it to `cluster_t_sums` for clarity.
            cluster_t_sums = np.zeros(permutation_clu_num)
            for j in range(permutation_clu_num):
                # Ensure element-wise multiplication for array * array if permutation_cluster[j] is an array
                # `permutation_cluster[j]` is a boolean mask or 0/1 array for the cluster.
                cluster_t_sums[j] = np.sum((np.abs(permutation_t_map) - threshold_val) * permutation_cluster[j])
            current_permutation_max_t = np.max(cluster_t_sums)
        else:
            current_permutation_max_t = 0 # No significant clusters found for this permutation

        return current_permutation_max_t, iteration_index # Return max t-value and original index

    # Create a list of delayed calls for each permutation
    tasks = [
        delayed(_single_permutation_iteration)(
            i, scores, conn_ref, threshold, **kwargs
        )
        for i in range(permutation_time)
    ]

    # Execute tasks in parallel and wrap the *results iterator* with tqdm for the progress bar
    results_iterator = Parallel(n_jobs=n_jobs, backend='loky')(tasks)

    # Wrap the iterator with tqdm to display a progress bar
    processed_results = list(tqdm.tqdm(results_iterator, total=permutation_time, desc="Permutations", leave=True))

    # Populate the final permutation_max_t array using the results
    for max_t_val, original_idx in processed_results:
        permutation_max_t[original_idx] = max_t_val

    print('\n\nPermutation analysis finished.\n\n')

    # --- Post-permutation analysis (sequential, as it's typically fast) ---
    good_cluster = []
    cluster_p_list = []
    for i in range(clu_num):
        sum_t = np.sum((np.abs(t_map) - threshold) * cluster[i])
        # Calculate cluster p-value based on the distribution of permutation_max_t
        cluster_p = np.sum(permutation_max_t > sum_t) / len(permutation_max_t)
        cluster_p_list.append(cluster_p) # Use .append for lists

        if cluster_p <= 0.05: # Significance level
            good_cluster.append(i) # Use .append for lists

    print('\n\nCluster analysis finished.\n\n')

    return t_map, t_map_mask, cluster, good_cluster, np.array(cluster_p_list)