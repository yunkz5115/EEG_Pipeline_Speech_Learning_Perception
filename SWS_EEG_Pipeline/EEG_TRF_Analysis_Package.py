# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 11:32:02 2025

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
    ReceptiveField
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

random_seed_index = 0
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
            print('\n\n\nCannot reject trials!\n\n\n')
        
    return data_pseudo_trial

def load_feature_for_trf(feature_path,feature1_index,feature2_index):
    file_list = sorted(os.listdir(feature_path))
    feature1_list = []
    feature2_list = []
    for i in feature1_index:
        feature1 = np.loadtxt(feature_path + '/' + file_list[i])
        if feature1.ndim==1:
            feature1 = feature1[:, np.newaxis]
        feature1_list = feature1_list + [feature1]
    for i in feature2_index:
        feature2 = np.loadtxt(feature_path + '/' + file_list[i])
        if feature2.ndim==1:
            feature2 = feature2[:, np.newaxis]
        feature2_list = feature2_list + [feature2]
    return feature1_list,feature2_list
    

def tg_fif_epoch_loader_with_feature(epoch1_path,epoch2_path,feature_path,feature1_index,feature2_index,
                                     filter_band = [None,None],
                                     label_1 = None,label_2 = None,
                                     picks_1 = None,picks_2 = None,
                                     tlim = None,thr_navg1 = None,thr_navg2 = None,
                                     n_jobs = 8,
                                     if_abs = False,reject_thr = None):
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
        
    data = np.vstack([data1,data2])
    print('\n\nReject thr = ',reject_thr,'  Data1 shape = ',data1.shape[0],'  Data2 shape = ',data2.shape[0],'\n\n')
    events = np.zeros([data.shape[0],3])
    events[:,0] = np.arange(data.shape[0])
    events[:,2] = np.hstack([np.zeros(data1.shape[0]),np.ones(data2.shape[0])])
    events = events.astype('int64')
    event_id = {"event1": 0, "event2": 1}
    
    tg_epoch = mne.EpochsArray(data, info=epoch1.info, events=events,tmin=epoch1.tmin,event_id=event_id,baseline=epoch1.baseline)
    tg_epoch.apply_baseline()
    feature1_list,feature2_list = load_feature_for_trf(feature_path,feature1_index,feature2_index)
    return tg_epoch,feature1_list,feature2_list

def single_epoch_trf(epochs,feature_list,re_sfreq,t_min=-0.2,t_max=0.4,input_ch = None,n_jobs=None,score_type='r2'):
    epochs = epochs.resample(sfreq=re_sfreq)
    rf = ReceptiveField(t_min, t_max, re_sfreq, feature_names=None, estimator=1.0, scoring=score_type, n_jobs = n_jobs)
    score_list = []
    coef_list = []
    X = epochs.get_data() # EEG signals: n_epochs, n_eeg_channels, n_times
    if not input_ch is None:
        X = X[:,input_ch,:]
    X = np.swapaxes(X, 1, 2)
    y = epochs.events[:,2] # target: auditory left vs visual left
    X1 = X[y==0,:,:]
    X2 = X[y==1,:,:]
    if (X1.shape[0]!=X2.shape[0])&(X2.shape[0]!=len(feature_list)):
        raise ValueError("Pre - Post SWS and feature not match")
    for i,feature in enumerate(feature_list):
        rf.fit(feature, X2[i,:,:])
        score = rf.score(feature, X1[i,:,:])
        coef = rf.coef_
        score_list = score_list + [score]
        coef_list = coef_list + [coef]
    
    return score_list,coef_list

def sws_all_trf_standard(epoch1_path_list = [],epoch2_path_list = [],feature_path = [],feature1_index = [],feature2_index = [],
                        filter_band = None,
                        label_1 = '',label_2 = '',
                        picks_1 = None,picks_2 = None,
                        tlim = None,thr_navg1 = None,thr_navg2 = None,
                        n_jobs = 8,
                        re_sfreq=None,
                        score_type='r2',input_ch=None,if_abs = False,reject_thr=None,
                        t_min = -0.2,t_max = 0.4):
    
    score_all_list = []
    coef_all_list = []
    
    for score_count,[epoch1_path,epoch2_path,feature1_list,feature2_list] in enumerate(zip(epoch1_path_list,epoch2_path_list,feature1_index,feature2_index)):
        
        epochs,feature1_list,feature2_list = tg_fif_epoch_loader_with_feature(epoch1_path,epoch2_path,feature_path,feature1_list,feature2_list,
                                                    filter_band=filter_band,
                                                    label_1=label_1,label_2=label_2,
                                                    picks_1=picks_1[score_count],picks_2=picks_2[score_count],
                                                    tlim = tlim,thr_navg1 = thr_navg1,thr_navg2 = thr_navg2,n_jobs = n_jobs,if_abs = if_abs,reject_thr=reject_thr)

        scores,coefs = single_epoch_trf(epochs,feature1_list,re_sfreq=re_sfreq,t_min=t_min,t_max=t_max,n_jobs=n_jobs,score_type=score_type,input_ch=None)
        score_all_list = score_all_list + [np.array(scores)]
        coef_all_list = coef_all_list + [np.array(coefs)]

    return score_all_list,coef_all_list

def full_epoch_trf(epochs,feature_list,re_sfreq,t_min=-0.2,t_max=0.4,input_ch = None,n_jobs=None,score_type='r2'):
    epochs = epochs.resample(sfreq=re_sfreq)
    rf = ReceptiveField(t_min, t_max, re_sfreq, feature_names=None, estimator=1.0, scoring=score_type, n_jobs = n_jobs)
    score_list = []
    coef_list = []
    X = epochs.get_data() # EEG signals: n_epochs, n_eeg_channels, n_times
    if not input_ch is None:
        X = X[:,input_ch,:]
    X = np.swapaxes(X, 1, 2)
    y = epochs.events[:,2] # target: auditory left vs visual left
    X1 = X[y==0,:,:]
    X2 = X[y==1,:,:]
    if (X1.shape[0]!=X2.shape[0])&(X2.shape[0]!=len(feature_list)):
        raise ValueError("Pre - Post SWS and feature not match")
        
    data_for_train = np.swapaxes(X2, 0, 1)
    feature_for_train = np.swapaxes(np.array(feature_list), 0, 1)
    rf.fit(feature_for_train, data_for_train)
    for i,feature in enumerate(feature_list):
        score = rf.score(feature, X1[i,:,:])
        coef = rf.coef_
        score_list = score_list + [score]
        coef_list = coef_list + [coef]
    
    return score_list,coef_list

def sws_all_full_epoch_trf_standard(epoch1_path_list = [],epoch2_path_list = [],feature_path = [],feature1_index = [],feature2_index = [],
                        filter_band = None,
                        label_1 = '',label_2 = '',
                        picks_1 = None,picks_2 = None,
                        tlim = None,thr_navg1 = None,thr_navg2 = None,
                        n_jobs = 8,
                        re_sfreq=None,
                        score_type='r2',input_ch=None,if_abs = False,reject_thr=None,
                        t_min = -0.2,t_max = 0.4):
    
    score_all_list = []
    coef_all_list = []
    
    for score_count,[epoch1_path,epoch2_path,feature1_list,feature2_list] in enumerate(zip(epoch1_path_list,epoch2_path_list,feature1_index,feature2_index)):
        
        epochs,feature1_list,feature2_list = tg_fif_epoch_loader_with_feature(epoch1_path,epoch2_path,feature_path,feature1_list,feature2_list,
                                                    filter_band=filter_band,
                                                    label_1=label_1,label_2=label_2,
                                                    picks_1=picks_1[score_count],picks_2=picks_2[score_count],
                                                    tlim = tlim,thr_navg1 = thr_navg1,thr_navg2 = thr_navg2,n_jobs = n_jobs,if_abs = if_abs,reject_thr=reject_thr)

        scores,coefs = full_epoch_trf(epochs,feature1_list,re_sfreq=re_sfreq,t_min=t_min,t_max=t_max,n_jobs=n_jobs,score_type=score_type,input_ch=None)
        score_all_list = score_all_list + [np.array(scores)]
        coef_all_list = coef_all_list + [np.array(coefs)]

    return score_all_list,coef_all_list

def full_epoch_trf_feature_inbalance(epochs,feature_list1,feature_list2,re_sfreq,t_min=-0.2,t_max=0.4,input_ch = None,n_jobs=None,score_type='r2'):
    epochs = epochs.resample(sfreq=re_sfreq)
    rf = ReceptiveField(t_min, t_max, re_sfreq, feature_names=None, estimator=1.0, scoring=score_type, n_jobs = n_jobs)
    score_list = []
    coef_list = []
    X = epochs.get_data() # EEG signals: n_epochs, n_eeg_channels, n_times
    if not input_ch is None:
        X = X[:,input_ch,:]
    X = np.swapaxes(X, 1, 2)
    y = epochs.events[:,2] # target: auditory left vs visual left
    X1 = X[y==0,:,:]
    X2 = X[y==1,:,:]
    if (X1.shape[0]!=len(feature_list1))&(X2.shape[0]!=len(feature_list2)):
        raise ValueError("Pre - Post SWS and feature not match")
        
    data_for_train = np.swapaxes(X2, 0, 1)
    feature_for_train = np.swapaxes(np.array(feature_list2), 0, 1)
    rf.fit(feature_for_train, data_for_train)
    for i,feature in enumerate(feature_list1):
        score = rf.score(feature, X1[i,:,:])
        coef = rf.coef_
        score_list = score_list + [score]
        coef_list = coef_list + [coef]
    
    return score_list,coef_list

def sws_all_full_epoch_trf_feature_inbalance(epoch1_path_list = [],epoch2_path_list = [],feature_path = [],feature1_index = [],feature2_index = [],
                        filter_band = None,
                        label_1 = '',label_2 = '',
                        picks_1 = None,picks_2 = None,
                        tlim = None,thr_navg1 = None,thr_navg2 = None,
                        n_jobs = 8,
                        re_sfreq=None,
                        score_type='r2',input_ch=None,if_abs = False,reject_thr=None,
                        t_min = -0.2,t_max = 0.4):
    
    score_all_list = []
    coef_all_list = []
    
    for score_count,[epoch1_path,epoch2_path,feature1_list,feature2_list] in enumerate(zip(epoch1_path_list,epoch2_path_list,feature1_index,feature2_index)):
        
        epochs,feature1_list,feature2_list = tg_fif_epoch_loader_with_feature(epoch1_path,epoch2_path,feature_path,feature1_list,feature2_list,
                                                    filter_band=filter_band,
                                                    label_1=label_1,label_2=label_2,
                                                    picks_1=picks_1[score_count],picks_2=picks_2[score_count],
                                                    tlim = tlim,thr_navg1 = thr_navg1,thr_navg2 = thr_navg2,n_jobs = n_jobs,if_abs = if_abs,reject_thr=reject_thr)

        scores,coefs = full_epoch_trf_feature_inbalance(epochs,feature1_list,feature2_list,re_sfreq=re_sfreq,t_min=t_min,t_max=t_max,n_jobs=n_jobs,score_type=score_type,input_ch=None)
        score_all_list = score_all_list + [np.array(scores)]
        coef_all_list = coef_all_list + [np.array(coefs)]

    return score_all_list,coef_all_list
        
    
    
def single_epoch_train_test_all_trf(epochs,feature_list,re_sfreq,t_min=-0.2,t_max=0.4,input_ch = None,n_jobs=None,score_type='r2'):
    epochs = epochs.resample(sfreq=re_sfreq)
    rf = ReceptiveField(t_min, t_max, re_sfreq, feature_names=None, estimator=1.0, scoring=score_type, n_jobs = n_jobs)
    score_list = []
    coef_list = []
    X = epochs.get_data() # EEG signals: n_epochs, n_eeg_channels, n_times
    if not input_ch is None:
        X = X[:,input_ch,:]
    X = np.swapaxes(X, 1, 2)
    y = epochs.events[:,2] # target: auditory left vs visual left
    X1 = X[y==0,:,:]
    X2 = X[y==1,:,:]
    if (X1.shape[0]!=X2.shape[0])&(X2.shape[0]!=len(feature_list)):
        raise ValueError("Pre - Post SWS and feature not match")
    for i,feature in enumerate(feature_list):
        rf.fit(feature, X2[i,:,:])
        for j in range(X1.shape[0]):
            if j==0:
                score = rf.score(feature, X1[j,:,:])
            else:
                score = np.vstack([score,rf.score(feature, X1[j,:,:])])
        coef = rf.coef_
        score_list = score_list + [score]
        coef_list = coef_list + [coef]
    
    return score_list,coef_list

def sws_all_trf_test_all_standard(epoch1_path_list = [],epoch2_path_list = [],feature_path = [],feature1_index = [],feature2_index = [],
                        filter_band = None,
                        label_1 = '',label_2 = '',
                        picks_1 = None,picks_2 = None,
                        tlim = None,thr_navg1 = None,thr_navg2 = None,
                        n_jobs = 8,
                        re_sfreq=None,
                        score_type='r2',input_ch=None,if_abs = False,reject_thr=None,
                        t_min = -0.2,t_max = 0.4):
    
    score_all_list = []
    coef_all_list = []
    
    for score_count,[epoch1_path,epoch2_path,feature1_list,feature2_list] in enumerate(zip(epoch1_path_list,epoch2_path_list,feature1_index,feature2_index)):
        
        epochs,feature1_list,feature2_list = tg_fif_epoch_loader_with_feature(epoch1_path,epoch2_path,feature_path,feature1_list,feature2_list,
                                                    filter_band=filter_band,
                                                    label_1=label_1,label_2=label_2,
                                                    picks_1=picks_1[score_count],picks_2=picks_2[score_count],
                                                    tlim = tlim,thr_navg1 = thr_navg1,thr_navg2 = thr_navg2,n_jobs = n_jobs,if_abs = if_abs,reject_thr=reject_thr)

        scores,coefs = single_epoch_train_test_all_trf(epochs,feature1_list,re_sfreq=re_sfreq,t_min=t_min,t_max=t_max,n_jobs=n_jobs,score_type=score_type,input_ch=None)
        score_all_list = score_all_list + [np.array(scores)]
        coef_all_list = coef_all_list + [np.array(coefs)]

    return score_all_list,coef_all_list
