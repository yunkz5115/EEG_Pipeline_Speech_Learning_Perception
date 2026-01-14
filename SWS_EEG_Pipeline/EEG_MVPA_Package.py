# -*- coding: utf-8 -*-
"""
Created on Sep Jul  1 00:31:00 2023

@author: yunkz
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
from mne.viz import (plot_topomap as plot_topo,
                     plot_brain_colorbar)

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

import os
from scipy.stats import ttest_1samp
from tqdm import trange
from scipy.signal import hilbert

random_seed_index = 0
t_start = -0.2
###############################################################################
#Data loader tools

def load_ID(path='C:/'):
    
    return 0

def erp_loader(id=3,phase=1,stim='sws',trial_type='single',ch_num=96,
               root_path='C:/'):
    print('trial type is: ',trial_type,'.\n')
    if trial_type=='single':
        path = root_path+'/'+'Participant'+str(id).zfill(4)+'_ph'+str(phase)+'_'+stim+'_single_trials.mat'
    elif trial_type=='avg':
        path = root_path+'/'+'Participant'+str(id).zfill(4)+'_ph'+str(phase)+'_'+stim+'_aref.mat'
    print('Load : ',path)
    erp = scio.loadmat(path)
    channels_name = []
    for key in erp.keys():
        if key[0]=='C':
            channels_name = channels_name + [key]
            
    ch_num = ch_num
    if len(channels_name)>ch_num:
        del channels_name[ch_num:]
    
    loc = np.zeros([ch_num,6])
    for i in range(ch_num):
        loc[i,0]=erp['Channels'][0,i][0][0][0]
        loc[i,1]=erp['Channels'][0,i][1][0][0]
    
    return erp,channels_name,loc

def pol_to_car(loc):
    new_loc = np.zeros(loc.shape)
    for i in range(len(loc)):
        new_loc[i,0] = loc[i,1]*np.cos(loc[i,0]/180*np.pi)/1000
        new_loc[i,1] = loc[i,1]*np.sin(loc[i,0]/180*np.pi)/1000
        
    return new_loc

def tg_epoch_builder(id1=3,phase1=1,phase2=2,stim1='sws',stim2='sws',
                     root_path='C:/'):
    erp1,ch1,loc1=erp_loader(id=id1,phase=phase1,stim=stim1,root_path=root_path)
    erp2,ch2,loc2=erp_loader(id=id1,phase=phase2,stim=stim2,root_path=root_path)
    trial_num1 = erp1[ch1[0]].shape[0]
    trial_num2 = erp2[ch2[0]].shape[0]
    
    loc1 = pol_to_car(loc1)
    loc2 = pol_to_car(loc2)
    
    trial_num = trial_num1+trial_num2
    
    event_list = np.hstack([np.zeros(trial_num1),np.ones(trial_num2)])
    events = np.vstack([np.arange(trial_num)+1,np.zeros(trial_num),event_list]).T.astype('int64')
    event_id = {"event1": 0, "event2": 1}
    
    data_array = np.zeros([trial_num,len(ch1),erp1[ch1[0]].shape[1]])
    
    for i in range(trial_num):
        for j in range(len(ch1)):
            if i<trial_num1:
                data_array[i,j,:]=erp1[ch1[j]][i,:]
            else:
                data_array[i,j,:]=erp2[ch2[j]][i-trial_num1,:]
            
    #tmin, tmax = -0.2, 0.8
    ch_types = []
    for i in range(len(ch1)):
        ch_types = ch_types + ['eeg']
    
    info = mne.create_info(ch_names=ch1, sfreq=erp1['SampleRate'],ch_types=ch_types)
    
    for i in range(len(info['chs'])):
        info['chs'][i]['loc'][:6]=loc1[i,:]
    
    tg_epoch = mne.EpochsArray(data_array, info=info, events=events,tmin=-0.2,event_id=event_id,baseline=(None,0))
    
    return tg_epoch

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

def tg_fif_epoch_loader(epoch1_path,epoch2_path,
                        filter_band = [None,None],
                        label_1 = None,label_2 = None,
                        picks_1 = None,picks_2 = None,
                        tlim = None,thr_navg1 = None,thr_navg2 = None,
                        n_jobs = 8,
                        if_abs = False,reject_thr = 0.0001):
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
    
    epoch1.apply_baseline()
    epoch2.apply_baseline()
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
    
    if tlim[0] < 0:
        t_start = tlim[0]
        tg_epoch = mne.EpochsArray(data, info=epoch1.info, events=events,tmin=epoch1.tmin,event_id=event_id,baseline=epoch1.baseline)
    else:
        t_start = 0
        tg_epoch = mne.EpochsArray(data, info=epoch1.info, events=events,tmin=epoch1.tmin,event_id=event_id,baseline=[None,0])
    #tg_epoch.apply_baseline()
    return tg_epoch
        
    
###############################################################################

def sws_weight(epochs,re_sfreq=25,n_fold=3,num_jobs=5,score_type='roc_auc',permutation=False,
               input_ch = None):
    epochs = epochs.resample(sfreq=re_sfreq)
    label = np.arange(epochs.events[:, 2].shape[0])
    np.random.seed(random_seed_index) 
    np.random.shuffle(label)
    X = epochs.get_data()[label,:,:]  # MEG signals: n_epochs, n_meg_channels, n_times
    Y = epochs.events[label, 2]  # target: auditory left vs visual left
    if permutation:
        np.random.shuffle(Y)
    feature_coef = np.zeros([n_fold,X.shape[1],X.shape[2]])
    if not input_ch is None:
        X = X[:,input_ch,:]
    kf = KFold(n_splits=n_fold)
    k=0
    for train, test in kf.split(Y):
        clf = make_pipeline(StandardScaler(), LinearModel(SVC(kernel="linear", gamma='auto', C=1)))
        time_gen = GeneralizingEstimator(clf, n_jobs=num_jobs, scoring=score_type, verbose=True)
        X_train, y_train = X[train], Y[train]
        time_gen.fit(X_train, y_train)
        if not input_ch is None:
            feature_coef[k,input_ch,:] = get_coef(time_gen,attr='patterns_')
        else:
            feature_coef[k,:,:] = get_coef(time_gen,attr='patterns_')
        k=k+1

    
    return np.mean(feature_coef,0)

def sws_tg_transfer(epochs_train,epochs_test,re_sfreq=25,n_fold=3,num_jobs=5,score_type='roc_auc',permutation=False,
                input_ch = None):
    epochs_train = epochs_train.resample(sfreq=re_sfreq)
    epochs_test = epochs_test.resample(sfreq=re_sfreq)
    
    if epochs_train.events[:, 2].shape[0]<epochs_test.events[:, 2].shape[0]:
        label = np.arange(epochs_train.events[:, 2].shape[0])
    else:
        label = np.arange(epochs_test.events[:, 2].shape[0])
    
    np.random.shuffle(label)
    X_train = epochs_train.get_data()[label,:,:]  # MEG signals: n_epochs, n_meg_channels, n_times
    Y_train = epochs_train.events[label, 2]  # target: auditory left vs visual left
    
    np.random.shuffle(label)
    X_test = epochs_test.get_data()[label,:,:]  # MEG signals: n_epochs, n_meg_channels, n_times
    Y_test = epochs_test.events[label, 2]  # target: auditory left vs visual left
    
    if permutation:
        np.random.shuffle(Y_train)
        np.random.shuffle(Y_test)
    # feature_coef = np.zeros([n_fold,X.shape[1],X.shape[2]])
    if not input_ch is None:
        X_train = X_train[:,input_ch,:]
        X_test = X_test[:,input_ch,:]
    kf = KFold(n_splits=n_fold)
    k=0
    score_matrix = np.zeros([n_fold,X_train.shape[2],X_train.shape[2]])
    for train, test in kf.split(Y_train):
        clf = make_pipeline(StandardScaler(), LinearModel(SVC(kernel="linear", gamma='auto', C=1)))
        time_gen = GeneralizingEstimator(clf, n_jobs=num_jobs, scoring=score_type, verbose=True)
        x_train, y_train = X_train[train], Y_train[train]
        x_test, y_test = X_test, Y_test
        time_gen.fit(x_train, y_train)
        score_matrix[k,:,:] = time_gen.score(x_test,y_test)
        k=k+1

    
    return score_matrix
    
    
def epoch_normalize_by_ch(X):
    #Note X shape: n_epochs, n_eeg_channels, n_times
    X1 = X.copy()
    normalize_factor = np.max(abs(X),axis=1)
    for i in range(X.shape[1]):
        X1[:,i,:] = X[:,i,:]/normalize_factor
    return X1

def sws_tg(epochs,re_sfreq=25,n_fold=3,num_jobs=5,score_type='roc_auc',permutation=False,
           input_ch = None, if_normalize = False):
    epochs = epochs.resample(sfreq=re_sfreq)
    label = np.arange(epochs.events[:, 2].shape[0])
    np.random.seed(random_seed_index) 
    np.random.shuffle(label)
    X = epochs.get_data()[label,:,:]  # EEG signals: n_epochs, n_eeg_channels, n_times
    if not input_ch is None:
        X = X[:,input_ch,:]
    y = epochs.events[label, 2]  # target: auditory left vs visual left
    
    if permutation:
        np.random.shuffle(y)
    
    if if_normalize:
        X = epoch_normalize_by_ch(X)
    
    clf = make_pipeline(StandardScaler(), SVC(kernel="linear", gamma='auto', C=0.5))
    # define the Temporal generalization object
    time_gen = GeneralizingEstimator(clf, n_jobs=num_jobs, scoring=score_type, verbose=True)
    # again, cv=3 just for speed
    scores = cross_val_multiscore(time_gen, X, y, cv=n_fold, n_jobs=num_jobs)
    
    return scores

def sws_ts(epochs,re_sfreq=25,n_fold=3,num_jobs=5,score_type='roc_auc',permutation=False,
           input_ch = None, if_normalize = False):
    epochs = epochs.resample(sfreq=re_sfreq)
    label = np.arange(epochs.events[:, 2].shape[0])
    np.random.seed(random_seed_index) 
    np.random.shuffle(label)
    X = epochs.get_data()[label,:,:]  # EEG signals: n_epochs, n_eeg_channels, n_times
    if not input_ch is None:
        X = X[:,input_ch,:]
    y = epochs.events[label, 2]  # target: auditory left vs visual left
    
    if permutation:
        np.random.shuffle(y)
        
    if if_normalize:
        X = epoch_normalize_by_ch(X)
    
    clf = make_pipeline(StandardScaler(), SVC(kernel="linear", gamma='auto', C=1))
    # define the Temporal generalization object
    time_gen = SlidingEstimator(clf, n_jobs=num_jobs, scoring=score_type, verbaose=True)
    # again, cv=3 just for speed
    scores = cross_val_multiscore(time_gen, X, y, cv=n_fold, n_jobs=num_jobs)
    
    return scores

def sws_all_tg_tranfer(ID_list=load_ID(),re_sfreq=25,n_fold=3,num_jobs=5,score_type='roc_auc',permutation=False,
               phase1_train=1,phase2_train=2,stim1_train='sws',stim2_train='sws',input_ch=None,
               phase1_test=2,phase2_test=3,stim1_test='cont',stim2_test='cont',
                                    root_path='C:/'):
    score_count = 0
    for ID in ID_list:
        score_count = score_count + 1
        epochs_train = tg_epoch_builder(id1=ID,phase1=phase1_train,phase2=phase2_train,stim1=stim1_train,stim2=stim2_train,root_path=root_path)
        epochs_test = tg_epoch_builder(id1=ID,phase1=phase1_test,phase2=phase2_test,stim1=stim1_test,stim2=stim2_test,root_path=root_path)
        if score_count==1:
            scores = sws_tg_transfer(epochs_train,epochs_test,re_sfreq=re_sfreq,n_fold=n_fold,num_jobs=num_jobs,score_type=score_type,permutation=permutation,input_ch=None)
        else:
            scores = np.concatenate((scores,
                                    sws_tg_transfer(epochs_train,epochs_test,re_sfreq=re_sfreq,n_fold=n_fold,num_jobs=num_jobs,score_type=score_type,permutation=permutation,input_ch=None)),
                                    axis=0)
    return scores

def sws_all_tg_tranfer_standard(permutation=False,epoch_train_list = None,epoch_test_list = None,return_epoch = False,
                                epoch_train_path_list = [],epoch_test_path_list = [],
                                filter_band = None,
                                epoch1_label = ['SWS_Pre','SWS_Pre'],epoch2_label = ['SWS_Post','SWS_Post'],
                                epoch1_picks = (None,None),epoch2_picks = (None,None),
                                tlim = None,thr_navg1 = None,thr_navg2 = None,
                                n_jobs = 8,
                                re_sfreq=None,n_fold=3,
                                score_type='roc_auc',input_ch=None,if_abs = False,reject_thr=None):
    
    if (epoch_train_list is None)|(epoch_test_list is None):
        epoch_train_list = []
        epoch_test_list = [] 
        for score_count,[epoch1_path,epoch2_path] in enumerate(zip(epoch_train_path_list,epoch_test_path_list)):
            
            epochs_train = tg_fif_epoch_loader(epoch1_path,epoch2_path,
                                         filter_band=filter_band,
                                         label_1=epoch1_label[0],label_2=epoch1_label[1],
                                         picks_1=epoch1_picks[0][score_count],picks_2=epoch1_picks[1][score_count],
                                         tlim = tlim,thr_navg1 = thr_navg1,thr_navg2 = thr_navg2,n_jobs = n_jobs,if_abs = if_abs,reject_thr=reject_thr)
            epochs_test = tg_fif_epoch_loader(epoch1_path,epoch2_path,
                                         filter_band=filter_band,
                                         label_1=epoch2_label[0],label_2=epoch2_label[1],
                                         picks_1=epoch2_picks[0][score_count],picks_2=epoch2_picks[1][score_count],
                                         tlim = tlim,thr_navg1 = thr_navg1,thr_navg2 = thr_navg2,n_jobs = n_jobs,if_abs = if_abs,reject_thr=reject_thr)
            if score_count==0:
                scores = sws_tg_transfer(epochs_train,epochs_test,re_sfreq=re_sfreq,n_fold=n_fold,num_jobs=n_jobs,score_type=score_type,permutation=permutation,input_ch=input_ch)
            else:
                scores = np.concatenate((scores,
                                        sws_tg_transfer(epochs_train,epochs_test,re_sfreq=re_sfreq,n_fold=n_fold,num_jobs=n_jobs,score_type=score_type,permutation=permutation,input_ch=input_ch)),
                                        axis=0)
            epoch_train_list = epoch_train_list + [epochs_train]
            epoch_test_list = epoch_test_list + [epochs_test]
    else:
        for score_count,[epochs_train,epochs_test] in enumerate(zip(epoch_train_list,epoch_test_list)):
            
            if score_count==0:
                scores = sws_tg_transfer(epochs_train,epochs_test,re_sfreq=re_sfreq,n_fold=n_fold,num_jobs=n_jobs,score_type=score_type,permutation=permutation,input_ch=input_ch)
            else:
                scores = np.concatenate((scores,
                                        sws_tg_transfer(epochs_train,epochs_test,re_sfreq=re_sfreq,n_fold=n_fold,num_jobs=n_jobs,score_type=score_type,permutation=permutation,input_ch=input_ch)),
                                        axis=0)
    if return_epoch:
        return scores,epoch_train_list,epoch_test_list
    else:
        return scores

def sws_all_tg(ID_list=load_ID(),re_sfreq=25,n_fold=3,num_jobs=5,score_type='roc_auc',permutation=False,
               phase1=1,phase2=2,stim1='sws',stim2='sws',input_ch=None,
                                    root_path='C:/'):
    score_count = 0
    for ID in ID_list:
        score_count = score_count + 1
        epochs = tg_epoch_builder(id1=ID,phase1=phase1,phase2=phase2,stim1=stim1,stim2=stim2,root_path=root_path)
        if score_count==1:
            scores = sws_tg(epochs,re_sfreq=re_sfreq,n_fold=n_fold,num_jobs=num_jobs,score_type=score_type,permutation=permutation,input_ch=None)
        else:
            scores = np.concatenate((scores,
                                    sws_tg(epochs,re_sfreq=re_sfreq,n_fold=3,num_jobs=5,score_type=score_type,permutation=permutation,input_ch=None)),
                                    axis=0)
    return scores

def sws_all_tg_standard(permutation=False,current_epoch_list = None,return_epoch = False,
                        epoch1_path_list = [],epoch2_path_list = [],
                        filter_band = None,
                        label_1 = '',label_2 = '',
                        picks_1 = None,picks_2 = None,
                        tlim = None,thr_navg1 = None,thr_navg2 = None,
                        n_jobs = 8,
                        re_sfreq=None,n_fold=3,
                        score_type='roc_auc',input_ch=None,if_abs = False,reject_thr=None):
    
    if current_epoch_list is None:
        current_epoch_list = []
        for score_count,[epoch1_path,epoch2_path] in enumerate(zip(epoch1_path_list,epoch2_path_list)):
            
            epochs = tg_fif_epoch_loader(epoch1_path,epoch2_path,
                                         filter_band=filter_band,
                                         label_1=label_1,label_2=label_2,
                                         picks_1=picks_1[score_count],picks_2=picks_2[score_count],
                                         tlim = tlim,thr_navg1 = thr_navg1,thr_navg2 = thr_navg2,n_jobs = n_jobs,if_abs = if_abs,reject_thr=reject_thr)
            if score_count==0:
                scores = sws_tg(epochs,re_sfreq=re_sfreq,n_fold=n_fold,num_jobs=n_jobs,score_type=score_type,permutation=permutation,input_ch=None)
            else:
                scores = np.concatenate((scores,
                                        sws_tg(epochs,re_sfreq=re_sfreq,n_fold=n_fold,num_jobs=n_jobs,score_type=score_type,permutation=permutation,input_ch=None)),
                                        axis=0)
            if return_epoch:
                current_epoch_list = current_epoch_list + [epochs]
    else:
        for score_count,epochs in enumerate(current_epoch_list):
            
            if score_count==0:
                scores = sws_tg(epochs,re_sfreq=re_sfreq,n_fold=n_fold,num_jobs=n_jobs,score_type=score_type,permutation=permutation,input_ch=None)
            else:
                scores = np.concatenate((scores,
                                        sws_tg(epochs,re_sfreq=re_sfreq,n_fold=n_fold,num_jobs=n_jobs,score_type=score_type,permutation=permutation,input_ch=None)),
                                        axis=0)
    if return_epoch:
        return scores,current_epoch_list
    else:
        return scores
    
def sws_all_weight_standard(permutation=False,current_epoch_list = None,return_epoch = False,
                        epoch1_path_list = [],epoch2_path_list = [],
                        filter_band = None,
                        label_1 = '',label_2 = '',
                        picks_1 = None,picks_2 = None,
                        tlim = None,thr_navg1 = None,thr_navg2 = None,
                        n_jobs = 8,
                        re_sfreq=None,n_fold=3,
                        score_type='roc_auc',input_ch=None,if_abs = False,reject_thr=None):
    
    if current_epoch_list is None:
        current_epoch_list = []
        for score_count,[epoch1_path,epoch2_path] in enumerate(zip(epoch1_path_list,epoch2_path_list)):
            
            epochs = tg_fif_epoch_loader(epoch1_path,epoch2_path,
                                         filter_band=filter_band,
                                         label_1=label_1,label_2=label_2,
                                         picks_1=picks_1[score_count],picks_2=picks_2[score_count],
                                         tlim = tlim,thr_navg1 = thr_navg1,thr_navg2 = thr_navg2,n_jobs = n_jobs,if_abs = if_abs,reject_thr=reject_thr)
            if score_count==0:
                scores = np.expand_dims(sws_weight(epochs,re_sfreq=re_sfreq,n_fold=n_fold,num_jobs=n_jobs,score_type=score_type,permutation=permutation,input_ch=None), axis = 0)
            else:
                scores = np.concatenate((scores,
                                        np.expand_dims(sws_weight(epochs,re_sfreq=re_sfreq,n_fold=n_fold,num_jobs=n_jobs,score_type=score_type,permutation=permutation,input_ch=None), axis = 0)),
                                        axis=0)
            if return_epoch:
                current_epoch_list = current_epoch_list + [epochs]
    
    return scores

def sws_all_ts(ID_list=load_ID(),re_sfreq=25,n_fold=3,num_jobs=5,score_type='roc_auc',permutation=False,
               phase1=1,phase2=2,stim1='sws',stim2='sws',input_ch=None,
                                    root_path='C:/'):
    score_count = 0
    for ID in ID_list:
        score_count = score_count + 1
        epochs = tg_epoch_builder(id1=ID,phase1=phase1,phase2=phase2,stim1=stim1,stim2=stim2,root_path=root_path)
        if score_count==1:
            scores = sws_ts(epochs,re_sfreq=re_sfreq,n_fold=3,num_jobs=5,score_type=score_type,permutation=permutation,input_ch=None)
        else:
            scores = np.concatenate((scores,
                                    sws_ts(epochs,re_sfreq=re_sfreq,n_fold=3,num_jobs=5,score_type=score_type,permutation=permutation,input_ch=None)),
                                    axis=0)
    return scores

def mvpa_cluster(t_map_mask):
    map = t_map_mask.copy()
    cluster = np.array([False*np.ones(map.shape)])
    num_cluster = 0
    if len(map.shape)==2:
        print('A 2D cluster')
        clu_list = []
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                if map[i,j]>0:
                    #print('Cluster found')
                    num_cluster = num_cluster + 1
                    cluster = np.concatenate((cluster,np.array([False*np.ones(map.shape)])),axis=0)
                    clu_list = [[i,j]]
                    while len(clu_list)>0:
                        x = clu_list[-1][0]
                        y = clu_list[-1][1]
                        map[x,y]=0
                        cluster[-1][x,y]=True
                        del(clu_list[-1])
                        if map[max(x-1,0),y]>0:
                            map[max(x-1,0),y]=0
                            cluster[-1][max(x-1,0),y]=True
                            clu_list = clu_list + [[max(x-1,0),y]]
                        if map[min(x+1,map.shape[0]-1),y]>0:
                            map[min(x+1,map.shape[0]-1),y]=0
                            cluster[-1][min(x+1,map.shape[0]-1),y]=True
                            clu_list = clu_list + [[min(x+1,map.shape[0]-1),y]]
                        if map[x,max(y-1,0)]>0:
                            map[x,max(y-1,0)]=0
                            cluster[-1][x,max(y-1,0)]=True
                            clu_list = clu_list + [[x,max(y-1,0)]]
                        if map[x,min(y+1,map.shape[1]-1)]>0:
                            map[x,min(y+1,map.shape[1]-1)]=0
                            cluster[-1][x,min(y+1,map.shape[1]-1)]=True
                            clu_list = clu_list + [[x,min(y+1,map.shape[1]-1)]]
                            
    elif len(map.shape)==1:
        print('A 1D cluster')
        clu_list = []
        for i in range(map.shape[0]):
            if map[i]>0:
                #print('Cluster found')
                num_cluster = num_cluster + 1
                cluster = np.concatenate((cluster,np.array([False*np.ones(map.shape)])),axis=0)
                clu_list = [i]
                while len(clu_list)>0:
                    x = clu_list[-1]
                    map[x]=0
                    cluster[-1][x]=True
                    del(clu_list[-1])
                    if map[max(x-1,0)]>0:
                        map[max(x-1,0)]=0
                        cluster[-1][max(x-1,0)]=True
                        clu_list = clu_list + [max(x-1,0)]
                    if map[min(x+1,map.shape[0]-1)]>0:
                        map[min(x+1,map.shape[0]-1)]=0
                        cluster[-1][min(x+1,map.shape[0]-1)]=True
                        clu_list = clu_list + [min(x+1,map.shape[0]-1)]
                
    cluster = np.delete(cluster,0,axis=0)
    print('Clustering finished')
    return cluster,num_cluster

def tg_stat_map(scores,chance=0.5,threshold=0.5,crop_t = None):
    t_map = np.zeros(np.mean(scores,0).shape)
    t_map_mask = np.zeros(np.mean(scores,0).shape)
    if crop_t is not None:
        exclude_mask = np.ones(t_map_mask.shape, dtype=bool)
        crop_index = np.arange((crop_t[0]-t_start)*t_map_mask.shape[-1],(crop_t[1]-t_start)*t_map_mask.shape[-1]).astype('int32')
        include_idx = np.ix_(crop_index, crop_index)
        exclude_mask[include_idx] = False
    for i in range(scores.shape[1]):
        for j in range(scores.shape[2]):
            t_map[i,j] = ttest_1samp(scores[:,i,j],chance,alternative='greater').statistic
            if t_map[i,j]>threshold:
                t_map_mask[i,j]=1
    if crop_t is not None:
        t_map[exclude_mask] = 0
        t_map_mask[exclude_mask] = 0
    cluster,clu_num = mvpa_cluster(t_map_mask)
    return t_map,t_map_mask,cluster,clu_num

def ts_stat_map(scores,chance=0.5,threshold=0.5):
    t_map = np.zeros(np.mean(scores,0).shape)
    t_map_mask = np.zeros(np.mean(scores,0).shape)
    for i in range(scores.shape[1]):
            t_map[i] = ttest_1samp(scores[:,i],chance,alternative='greater').statistic
            if t_map[i]>threshold:
                t_map_mask[i]=1
    cluster,clu_num = mvpa_cluster(t_map_mask)
    
    return t_map,t_map_mask,cluster,clu_num

def tg_permutation(ID_list=load_ID(),permutation_time = 10,re_sfreq=25,n_fold=3,num_jobs=5,score_type='roc_auc',threshold=0.8,chance=0.5,
                   phase1=1,phase2=2,stim1='sws',stim2='sws',input_ch=None,
                                    root_path='C:/',crop_t = None):
    scores = sws_all_tg(ID_list=ID_list,re_sfreq=re_sfreq,n_fold=n_fold,num_jobs=num_jobs,score_type=score_type,
                        phase1=phase1,phase2=phase2,stim1=stim1,stim2=stim2,input_ch=None,
                                             root_path=root_path)
    
    t_map,t_map_mask,cluster,clu_num = tg_stat_map(scores,chance=chance,threshold=threshold,crop_t = None)
    significant_mask = t_map
    permutation_max_t = np.zeros(permutation_time)
    for i in range(permutation_time):
        permutation_score = sws_all_tg(ID_list=ID_list,re_sfreq=re_sfreq,n_fold=n_fold,num_jobs=num_jobs,score_type=score_type,permutation=True,
                                       phase1=phase1,phase2=phase2,stim1=stim1,stim2=stim2,input_ch=None,
                                                            root_path=root_path)
        permutation_t_map,permutation_t_map_mask,permutation_cluster,permutation_clu_num = tg_stat_map(permutation_score,chance=chance,threshold=threshold,crop_t = None)
        if permutation_clu_num>0:
            permutation_t_list = np.zeros(permutation_clu_num)
            for j in range(permutation_clu_num):
                permutation_t_list[j] = np.sum((permutation_t_map-threshold)*permutation_cluster[j])
            permutation_max_t[i] = np.max(permutation_t_list)
        else:
            permutation_max_t[i] = 0
        
    good_cluster = []
    for i in range(clu_num):
        sum_t = np.sum((t_map-threshold)*cluster[i])
        cluster_p = np.sum(permutation_max_t>sum_t)/len(permutation_max_t)
        if cluster_p<=0.05:
            good_cluster = good_cluster + [i]
    
    return scores,significant_mask,cluster,good_cluster

def tg_permutation_standard(permutation_time = 10,threshold=0.8,chance=0.5,crop_t = None,**kwargs):
    scores,epoch_list = sws_all_tg_standard(permutation=False,current_epoch_list=None,return_epoch=True,**kwargs)
    
    t_map,t_map_mask,cluster,clu_num = tg_stat_map(scores,chance=chance,threshold=threshold,crop_t = crop_t)
    significant_mask = t_map
    permutation_max_t = np.zeros(permutation_time)
    for i in range(permutation_time):
        permutation_score = sws_all_tg_standard(permutation=True,current_epoch_list=epoch_list,return_epoch=False,**kwargs)
        
        permutation_t_map,permutation_t_map_mask,permutation_cluster,permutation_clu_num = tg_stat_map(permutation_score,chance=chance,threshold=threshold,crop_t = crop_t)
        if permutation_clu_num>0:
            permutation_t_list = np.zeros(permutation_clu_num)
            for j in range(permutation_clu_num):
                permutation_t_list[j] = np.sum((permutation_t_map-threshold)*permutation_cluster[j])
            permutation_max_t[i] = np.max(permutation_t_list)
        else:
            permutation_max_t[i] = 0
        
    good_cluster = []
    cluster_p_list = []
    for i in range(clu_num):
        sum_t = np.sum((t_map-threshold)*cluster[i])
        cluster_p = np.sum(permutation_max_t>sum_t)/len(permutation_max_t)
        cluster_p_list = cluster_p_list + [cluster_p]
        if cluster_p<=0.05:
            good_cluster = good_cluster + [i]
    
    return scores,significant_mask,cluster,good_cluster,t_map,np.array(cluster_p_list)

def tg_transfer_permutation_standard(permutation_time = 10,threshold=0.8,chance=0.5,crop_t = None,**kwargs):
    scores,epoch_train_list,epoch_test_list = sws_all_tg_tranfer_standard(permutation=False,epoch_train_list=None,epoch_test_list=None,return_epoch=True,**kwargs)
    
    t_map,t_map_mask,cluster,clu_num = tg_stat_map(scores,chance=chance,threshold=threshold,crop_t = crop_t)
    significant_mask = t_map
    permutation_max_t = np.zeros(permutation_time)
    for i in range(permutation_time):
        permutation_score = sws_all_tg_tranfer_standard(permutation=True,epoch_train_list=epoch_train_list,epoch_test_list=epoch_test_list,return_epoch=False,**kwargs)
        
        permutation_t_map,permutation_t_map_mask,permutation_cluster,permutation_clu_num = tg_stat_map(permutation_score,chance=chance,threshold=threshold,crop_t = crop_t)
        if permutation_clu_num>0:
            permutation_t_list = np.zeros(permutation_clu_num)
            for j in range(permutation_clu_num):
                permutation_t_list[j] = np.sum((permutation_t_map-threshold)*permutation_cluster[j])
            permutation_max_t[i] = np.max(permutation_t_list)
        else:
            permutation_max_t[i] = 0
        
    good_cluster = []
    cluster_p_list = []
    for i in range(clu_num):
        sum_t = np.sum((t_map-threshold)*cluster[i])
        cluster_p = np.sum(permutation_max_t>sum_t)/len(permutation_max_t)
        cluster_p_list = cluster_p_list + [cluster_p]
        if cluster_p<=0.05:
            good_cluster = good_cluster + [i]
    
    return scores,significant_mask,cluster,good_cluster,t_map,np.array(cluster_p_list)

def ts_permutation(ID_list=load_ID(),permutation_time = 10,re_sfreq=25,n_fold=3,num_jobs=5,score_type='roc_auc',threshold=0.8,chance=0.5,
                   phase1=1,phase2=2,stim1='sws',stim2='sws',input_ch=None,
                                    root_path='C:/'):
    scores = sws_all_ts(ID_list=ID_list,re_sfreq=re_sfreq,n_fold=n_fold,num_jobs=num_jobs,score_type=score_type,
                        phase1=phase1,phase2=phase2,stim1=stim1,stim2=stim2,input_ch=None,
                                             root_path=root_path)
    
    t_map,t_map_mask,cluster,clu_num = ts_stat_map(scores,chance=chance,threshold=threshold)
    significant_mask = t_map
    permutation_max_t = np.zeros(permutation_time)
    for i in range(permutation_time):
        permutation_score = sws_all_ts(ID_list=ID_list,re_sfreq=re_sfreq,n_fold=n_fold,num_jobs=num_jobs,score_type=score_type,permutation=True,
                                       phase1=phase1,phase2=phase2,stim1=stim1,stim2=stim2,input_ch=None,
                                                            root_path=root_path)
        permutation_t_map,permutation_t_map_mask,permutation_cluster,permutation_clu_num = ts_stat_map(permutation_score,chance=chance,threshold=threshold)
        if permutation_clu_num>0:
            permutation_t_list = np.zeros(permutation_clu_num)
            for j in range(permutation_clu_num):
                permutation_t_list[j] = np.sum((permutation_t_map-threshold)*permutation_cluster[j])
            permutation_max_t[i] = np.max(permutation_t_list)
        else:
            permutation_max_t[i] = 0
        
    good_cluster = []
    for i in range(clu_num):
        sum_t = np.sum((t_map-threshold)*cluster[i])
        cluster_p = np.sum(permutation_max_t>sum_t)/len(permutation_max_t)
        if cluster_p<=0.05:
            good_cluster = good_cluster + [i]
    
    return scores,significant_mask,cluster,good_cluster

def cluster_boundry(mask,kernal_size=3):
    rad = int(np.floor(kernal_size/2))
    new_mask = np.zeros(mask.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            new_mask[i,j] = np.min(mask[np.max([i-rad,0]):np.min([i+rad,mask.shape[0]-1]),
                                        np.max([j-rad,0]):np.min([j+rad,mask.shape[1]-1])])
            
    new_mask = mask-new_mask
    return new_mask

def tg_plot(scores,
            time_window=[-0.2,0.8,-0.2,0.8],
            seg = 0.2,
            phase1=1,phase2=2,
            type1='sws',type2='sws',cmap='viridis',color='w',
            mask=None,
            figsize=[5.2,5],vmin=0.5,vmax=0.6,vmiddle=0.55,
            pad=0.1,shrink=0.8,if_scatter=True,
            fontsize=10,label_size=10,title=None,if_colorbar=True):
    plt.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(1, 1,figsize=figsize)
    
                        
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xlim(time_window[0],time_window[1],emit=False)
    ax.set_ylim(time_window[2],time_window[3],emit=False)
    ax.set_xticks(np.round(np.arange(time_window[0],time_window[1]+seg,seg),1))
    ax.set_yticks(np.round(np.arange(time_window[2],time_window[3]+seg,seg),1))
    ax.set_xticklabels(ax.get_xticklabels(),fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(),fontsize=fontsize)
    ax.set_xlabel("Testing Time (s)",fontsize=label_size)
    ax.set_ylabel("Training Time (s)",fontsize=label_size)
    if title is None:
        ax.set_title("TG "+"Phase "+str(phase1)+" "+type1+" vs "+"Phase "+str(phase2)+" "+type2,fontsize=label_size)
    else:
        ax.set_title(title,fontsize=label_size)
    ax.axvline(0, color=color,lw=1)
    ax.axhline(0, color=color,lw=1)
    
    if len(scores.shape)>2:
        scores = np.mean(scores,0)
    
    if not if_scatter:
        if not mask is None:
            for i in range(scores.shape[0]):
                for j in range(scores.shape[1]):
                    if (mask[i,j]<1):
                        scores[i,j]=0.5
        else:
            for i in range(scores.shape[0]):
                for j in range(scores.shape[1]):
                        scores[i,j]=0.5
        
    im = ax.imshow(
        scores,
        interpolation="lanczos",
        origin="lower",
        cmap=cmap,
        extent=np.around(time_window,1),
        vmin=vmin,
        vmax=vmax,
    )
    print(np.max(scores))
    base=-time_window[0]*scores.shape[0]/(time_window[1]-time_window[0])
    if if_scatter:
        if not mask is None:
            for i in range(scores.shape[0]):
                for j in range(scores.shape[1]):
                    if (mask[i,j]==1) & (i>base) & (j>base):
                        ax.scatter(j/scores.shape[0]+time_window[0],i/scores.shape[1]+time_window[2],
                                       s=0.2,marker='.',c=color)
    
    if if_colorbar:
        cbar = plt.colorbar(im, ax=ax,shrink=shrink,pad=pad)
        cbar.set_label("AUC",fontsize=label_size)
        cbar.set_ticks([vmin,vmiddle,vmax])
    
    return fig

def mat_tg_plot(
            time_window=[-0.2,0.8,-0.2,0.8],
            phase1=1,phase2=2,if_scatter=True,vmin=0.5,vmax=0.6,vmiddle=0.55,fontsize=10,label_size=10,
            type1='sws',type2='sws',cmap='viridis',color='w',figsize=[5.2,5],shrink=0.8,pad=0.1,title=None,if_colorbar=True,
            root_path = 'C:/'):
    name = 'type1_'+type1+'_type2_'+type2+'_phase1_'+str(phase1)+'_phase2_'+str(phase2)+'.mat'    
    score_mat = scio.loadmat(root_path + name)
    scores = np.mean(score_mat['scores'],0)
    if score_mat['good_cluster'].shape[1]>1:
        mask=np.sum(score_mat['cluster'][np.squeeze(score_mat['good_cluster']),:,:],axis=0)
    elif score_mat['good_cluster'].shape[1]==1:
        mask=score_mat['cluster'][np.squeeze(score_mat['good_cluster']),:,:]
    else:
        mask=None
    
    fig = tg_plot(scores=scores,time_window=time_window,cmap=cmap,color=color,
                  phase1=phase1,phase2=phase2,type1=type1,type2=type2,
                  mask=mask,figsize=figsize,shrink=shrink,pad=pad,if_scatter=if_scatter,
                  vmin=vmin,vmax=vmax,vmiddle=vmiddle,fontsize=fontsize,label_size=label_size,title=title,if_colorbar=if_colorbar)
    
    return fig,mask

def tg_plot_standard(scores,
            time_window=[-0.2,1.0,-0.2,1.0],
            seg = 0.2,
            cmap='viridis',color='w',
            mask=None,
            figsize=[5.2,5],vmin=0.5,vmax=0.6,vmiddle=0.55,
            pad=0.1,shrink=0.8,if_scatter=True,
            fontsize=10,label_size=10,title=None,if_colorbar=True):
    plt.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(1, 1,figsize=figsize)
    
                        
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xlim(time_window[0],time_window[1],emit=False)
    ax.set_ylim(time_window[2],time_window[3],emit=False)
    ax.set_xticks(np.round(np.arange(time_window[0],time_window[1]+seg,seg),1))
    ax.set_yticks(np.round(np.arange(time_window[2],time_window[3]+seg,seg),1))
    ax.set_xticklabels(ax.get_xticklabels(),fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(),fontsize=fontsize)
    ax.set_xlabel("Testing Time (s)",fontsize=label_size)
    ax.set_ylabel("Training Time (s)",fontsize=label_size)
    if title is None:
        ax.set_title(' ',fontsize=label_size)
    else:
        ax.set_title(title,fontsize=label_size)
    ax.axvline(0, color=color,lw=1)
    ax.axhline(0, color=color,lw=1)
    
    if len(scores.shape)>2:
        scores = np.mean(scores,0)
    
    if not if_scatter:
        if not mask is None:
            for i in range(scores.shape[0]):
                for j in range(scores.shape[1]):
                    if (mask[i,j]<1):
                        scores[i,j]=0.5
        else:
            for i in range(scores.shape[0]):
                for j in range(scores.shape[1]):
                        scores[i,j]=0.5
        
    im = ax.imshow(
        scores,
        interpolation="lanczos",
        origin="lower",
        cmap=cmap,
        extent=np.around(time_window,1),
        vmin=vmin,
        vmax=vmax,
    )
    print(np.max(scores))
    base=-time_window[0]*scores.shape[0]/(time_window[1]-time_window[0])
    if if_scatter:
        if not mask is None:
            for i in range(scores.shape[0]):
                for j in range(scores.shape[1]):
                    if (mask[i,j]==1) & (i>base) & (j>base):
                        ax.scatter(j/scores.shape[0]+time_window[0],i/scores.shape[1]+time_window[2],
                                       s=0.2,marker='.',c=color)
    
    if if_colorbar:
        cbar = plt.colorbar(im, ax=ax,shrink=shrink,pad=pad)
        cbar.set_label("AUC",fontsize=label_size)
        cbar.set_ticks([vmin,vmiddle,vmax])
    
    return fig

def mat_tg_plot_standard(
            time_window=[-0.2,1.0,-0.2,1.0],
            path = '',if_scatter=True,vmin=0.5,vmax=0.6,vmiddle=0.55,fontsize=10,label_size=10,
            cmap='viridis',color='w',figsize=[5.2,5],shrink=0.8,pad=0.1,title=None,if_colorbar=True):
    score_mat = scio.loadmat(path)
    scores = np.mean(score_mat['scores'],0)
    if score_mat['good_cluster'].shape[1]>1:
        mask=np.sum(score_mat['cluster'][np.squeeze(score_mat['good_cluster']),:,:],axis=0)
    elif score_mat['good_cluster'].shape[1]==1:
        mask=score_mat['cluster'][np.squeeze(score_mat['good_cluster']),:,:]
    else:
        mask=None
    
    fig = tg_plot(scores=scores,time_window=time_window,cmap=cmap,color=color,
                  mask=mask,figsize=figsize,shrink=shrink,pad=pad,if_scatter=if_scatter,
                  vmin=vmin,vmax=vmax,vmiddle=vmiddle,fontsize=fontsize,label_size=label_size,title=title,if_colorbar=if_colorbar)
    
    return fig,mask
    

def load_avg_score(phase1=1,phase2=2,type1='sws',type2='sws',
                        h_mask=np.ones([25,25]),mask=None,
                        root_path='C:/'):
    path = root_path + '/type1_' + type1 + '/type2_' + type2 + '/phase1_' + str(phase1) + '/phase2_' + str(phase2) + '/'
    name_list = os.listdir(path)
    avg_score = np.zeros(h_mask.shape)
    for name in name_list:
        avg_score = avg_score + np.loadtxt(path + name)
        
    avg_score = avg_score*h_mask/len(name_list)
    
    fig = tg_plot(avg_score,[-0.2,0.8,-0.2,0.8],phase1=phase1,phase2=phase2,type1=type1,type2=type2,mask=mask)
    
    return avg_score,fig

def load_avg_weight(phase1=1,phase2=2,type1='sws',type2='sws',vmin=-0.2,vmax=0.2,
                        h_mask=np.ones([96,25]),window_size=0.025,re_sfreq=25,plot_time=-0.2,
                        fontsize=8,extrapolate='head',n_row=2,n_col=8,figsize=[10,2.5],axis_pos=[0.92, 0.2, 0.01, 0.6],
                        root_path='C:/'):
    path = root_path + '/type1_' + type1 + '/type2_' + type2 + '/phase1_' + str(phase1) + '/phase2_' + str(phase2) + '/'
    name_list = os.listdir(path)
    avg_score = np.zeros(h_mask.shape)
    for name in name_list:
        avg_score = avg_score + np.loadtxt(path + name)
        
    avg_score = avg_score*h_mask/len(name_list)
    
    erp,ch,loc = erp_loader()
    loc = pol_to_car(loc)
    
    fig = multi_window_topo_plot(avg_score,loc,window_size=window_size,re_sfreq=re_sfreq,
                                 plot_type='weight',plot_time=plot_time,vmin=vmin,vmax=vmax,
                                 fontsize=fontsize,extrapolate=extrapolate,n_row=n_row,n_col=n_col,figsize=figsize,
                                 axis_pos=axis_pos)
    
    return avg_score,fig 

def score_plot(scores):
    fig, ax = plt.subplots(1, 1)
    im = ax.plot(
        np.diagonal(scores),
    )
    plt.ylim([0.4,1])

def load_epoch(id1=3,phase1=1,stim1='sws',trial_type='avg',
               root_path='C:/'):
    print('trial type is: ',trial_type,'.\n')
    erp1,ch1,loc1=erp_loader(id=id1,phase=phase1,stim=stim1,root_path=root_path,trial_type=trial_type)
    trial_num = erp1[ch1[0]].shape[0]
    event_list = np.zeros(trial_num)
    events = np.vstack([np.arange(trial_num)+1,np.zeros(trial_num),event_list]).T.astype('int64')
    data_array = np.zeros([trial_num,len(ch1),erp1[ch1[0]].shape[1]])
    event_id = {"event1": 0}
    
    for i in range(trial_num):
        for j in range(len(ch1)):
            data_array[i,j,:]=erp1[ch1[j]][i,:]
            
    #tmin, tmax = -0.2, 0.8
    ch_types = []
    for i in range(len(ch1)):
        ch_types = ch_types + ['eeg']
    
    info = mne.create_info(ch_names=ch1, sfreq=erp1['SampleRate'],ch_types=ch_types)
    loc1 = pol_to_car(loc1)
    for i in range(len(info['chs'])):
        info['chs'][i]['loc'][:6]=loc1[i,:]
    epochs = mne.EpochsArray(data_array, info=info, events=events,tmin=-0.2,event_id=event_id,baseline=(None,0))
    
    return epochs


def anova_2wrm_set_loader(ID_list = load_ID(),phase1=1,phase2=2,type1='sws',type2='cont',trial_type='avg',
                      root_path='C:/'):
    epochs = load_epoch(ID_list[0],phase1,type1,trial_type=trial_type,root_path=root_path)
    data = epochs.get_data()
    anova_set = np.zeros([len(ID_list),4,data.shape[1],data.shape[2]])
    for i in range(len(ID_list)):
        anova_set[i,0,:,:] = np.mean(load_epoch(ID_list[i],phase1,type1,trial_type=trial_type,root_path=root_path).get_data(),0)
        anova_set[i,1,:,:] = np.mean(load_epoch(ID_list[i],phase1,type2,trial_type=trial_type,root_path=root_path).get_data(),0)
        anova_set[i,2,:,:] = np.mean(load_epoch(ID_list[i],phase2,type1,trial_type=trial_type,root_path=root_path).get_data(),0)
        anova_set[i,3,:,:] = np.mean(load_epoch(ID_list[i],phase2,type2,trial_type=trial_type,root_path=root_path).get_data(),0)
    
    return anova_set
    
def multi_window_topo_plot(data,loc,window_size=0.05,sr=500,start_time=-0.2,plot_time=-0.2,re_sfreq=False,
                           vmin=-0.4,vmax=0.4,vmiddle=0,mask=None,plot_type=None,
                           colormap='RdBu_r',figsize=np.array([10,2.5]),size=1,
                           channel_omit = np.array([96]),channel_set_0 = np.array([]),
                           n_row=2,n_col=8,fontsize=20,
                           axis_pos=[0.92, 0.2, 0.01, 0.6],extrapolate='head',markersize=2,if_threshold=False,sensors=True,
                           mask_sensor='k.',
                           ):
    plt.rcParams.update({'font.size': fontsize})
    
    if len(channel_set_0)>0:
        data[channel_set_0,:] = np.zeros([len(channel_set_0),data.shape[1]])
    
    if channel_omit is not None:
        if (channel_omit.shape[0]!=0):
            channel_omit = channel_omit-1
            data = np.delete(data, channel_omit, axis=0)
            if isinstance(loc, np.ndarray):
                loc = np.delete(loc, channel_omit, axis=0)
            if not mask is None:
                mask = np.delete(mask, channel_omit, axis=0)
            
        
    if re_sfreq:
        new_data = np.zeros([data.shape[0],int(data.shape[1]*sr/re_sfreq)])
        for i in range(new_data.shape[1]):
            new_data[:,i] = data[:,int(i*re_sfreq/sr)]
        data=new_data
        
    win_num = int(np.round((data.shape[1]-sr*(plot_time-start_time))/window_size/sr))
    if plot_type=='weight':
        label='Feature Weight'
        win_mask=np.zeros([data.shape[0],win_num])
    elif plot_type is None:
        label=' '
        if not mask is None:
            mask = ~np.isnan(mask)
            win_mask = np.zeros([data.shape[0],win_num])!=0
            for i in range(win_num):
                win_mask[:,i] = np.sum(mask[:,int(((plot_time-start_time)+i*window_size)*sr):int(((plot_time-start_time)+(i+1)*window_size)*sr)],1)!=0
        else:
            win_mask=np.zeros([data.shape[0],win_num])
    elif plot_type=='F stat':
        label='F Score'
        if not mask is None:
            mask = ~np.isnan(mask)
            win_mask = np.zeros([data.shape[0],win_num])!=0
            for i in range(win_num):
                win_mask[:,i] = np.sum(mask[:,int(((plot_time-start_time)+i*window_size)*sr):int(((plot_time-start_time)+(i+1)*window_size)*sr)],1)!=0
        else:
            win_mask=np.zeros([data.shape[0],win_num])
    elif plot_type=='T stat':
        label='T Score'
        if not mask is None:
            mask = ~np.isnan(mask)
            win_mask = np.zeros([data.shape[0],win_num])!=0
            for i in range(win_num):
                win_mask[:,i] = np.sum(mask[:,int(((plot_time-start_time)+i*window_size)*sr):int(((plot_time-start_time)+(i+1)*window_size)*sr)],1)!=0
        else:
            win_mask=np.zeros([data.shape[0],win_num])
    print(win_num,' Windows are plotted.')
    
    if (isinstance(sensors,str))&(sensors==mask_sensor):
        mask_params = dict(marker=sensors[-1], markerfacecolor=sensors[:-1], markeredgecolor=sensors[:-1],
            linewidth=0, markersize=markersize)
    else:
        mask_params = dict(marker=mask_sensor[-1], markerfacecolor=mask_sensor[:-1], markeredgecolor=mask_sensor[:-1],
            linewidth=0, markersize=markersize)

    fig, ax = plt.subplots(n_row,n_col,figsize=figsize)
    num = 0
    
    for i in range(n_row):
        for j in range(n_col):
            head_t = plot_time+num*window_size
            if int((plot_time+(num+1)*window_size)*sr)<data.shape[1]:
                data_plot = np.mean(data[:,int(((plot_time-start_time)+num*window_size)*sr):int(((plot_time-start_time)+(num+1)*window_size)*sr)],1)
                #print(data_plot.shape)
            else:
                data_plot = np.zeros([data.shape[0],window_size*sr])
                #print(data_plot.shape)
            
            if if_threshold:
                for count_threshold in range(data_plot.shape[0]):
                    if not win_mask[count_threshold,num]:
                        if plot_type=='F stat':
                            data_plot[count_threshold]=vmin
                        if plot_type=='T stat':
                            data_plot[count_threshold]=vmiddle
                        if plot_type=='Feature Weight':
                            data_plot[count_threshold]=vmiddle
                            
                    win_mask[count_threshold,num]=True
            
            
            if isinstance(loc, np.ndarray):
                if n_row>1:    
                    plot_topo(data_plot, loc[:,0:2],ch_type='eeg',size=size,contours=0,sensors=sensors,
                              mask=win_mask[:,num],mask_params = mask_params,axes = ax[i,j],vlim=(vmin,vmax),
                              cmap=colormap,show=False,extrapolate=extrapolate)
                    ax[i,j].set_xlabel(f"{int(np.round((head_t)*1000,4))}-{int(np.round((head_t+window_size),4)*1000)} ",loc='center',
                                       fontsize=fontsize)
                elif (n_row==1)&(n_col>1):
                    plot_topo(data_plot, loc[:,0:2],ch_type='eeg',size=size,contours=0,sensors=sensors,
                              mask=win_mask[:,num],mask_params = mask_params,axes = ax[j],vlim=(vmin,vmax),
                              cmap=colormap,show=False,extrapolate=extrapolate)
                    ax[j].set_xlabel(f"{int(np.round((head_t)*1000,4))}-{int(np.round((head_t+window_size),4)*1000)} ",loc='center',
                                       fontsize=fontsize)
                elif (n_row==1)&(n_col==1):
                    plot_topo(data_plot, loc[:,0:2],ch_type='eeg',size=size,contours=0,sensors=sensors,
                              mask=win_mask[:,num],mask_params = mask_params,axes = ax,vlim=(vmin,vmax),
                              cmap=colormap,show=False,extrapolate=extrapolate)
                    ax.set_xlabel(f"{int(np.round((head_t)*1000,4))}-{int(np.round((head_t+window_size),4)*1000)} ",loc='center',
                                       fontsize=fontsize)
            else:
                if n_row>1:    
                    plot_topo(data_plot, loc,ch_type='eeg',size=size,contours=0,sensors=sensors,
                              mask=win_mask[:,num],mask_params = mask_params,axes = ax[i,j],vlim=(vmin,vmax),
                              cmap=colormap,show=False,extrapolate=extrapolate)
                    ax[i,j].set_xlabel(f"{int(np.round((head_t)*1000,4))}-{int(np.round((head_t+window_size),4)*1000)} ",loc='center',
                                       fontsize=fontsize)
                elif (n_row==1)&(n_col>1):
                    plot_topo(data_plot, loc,ch_type='eeg',size=size,contours=0,sensors=sensors,
                              mask=win_mask[:,num],mask_params = mask_params,axes = ax[j],vlim=(vmin,vmax),
                              cmap=colormap,show=False,extrapolate=extrapolate)
                    ax[j].set_xlabel(f"{int(np.round((head_t)*1000,4))}-{int(np.round((head_t+window_size),4)*1000)} ",loc='center',
                                       fontsize=fontsize)
                elif (n_row==1)&(n_col==1):
                    plot_topo(data_plot, loc,ch_type='eeg',size=size,contours=0,sensors=sensors,
                              mask=win_mask[:,num],mask_params = mask_params,axes = ax,vlim=(vmin,vmax),
                              cmap=colormap,show=False,extrapolate=extrapolate)
                    ax.set_xlabel(f"{int(np.round((head_t)*1000,4))}-{int(np.round((head_t+window_size),4)*1000)} ",loc='center',
                                       fontsize=fontsize)
                
            num=num+1
    ax1 =  fig.add_axes(axis_pos)
    clim={'kind':'value',
          'lims':np.array([vmin,vmiddle,vmax])}
    plot_brain_colorbar(ax1,clim,colormap=colormap,label=label,transparent=False)
    
    fig.show()

    return fig
        
    
def edit_colomap(colormap,add_color=None,add_location=np.arange(15)):
    
    if add_color is None:
        sample_cmap = mpl.colormaps['RdBu_r'].resampled(256)(np.linspace(0, 1, 256))
        add_color = (sample_cmap[127,:]+sample_cmap[128,:])/2
    
    color = mpl.colormaps[colormap].resampled(256)
    newcolors = color(np.linspace(0, 1, 256))
    for i in add_location:
        newcolors[i,:] = add_color*(np.max(add_location)-i)/np.max(add_location)+newcolors[i,:]*(i)/np.max(add_location)
    
    newcmp = ListedColormap(newcolors)
    
    return newcmp

###############################################################################