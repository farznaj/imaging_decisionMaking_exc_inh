# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:15:20 2017

@author: farznaj
"""

#%%

mice = 'fni16', 'fni17', 'fni18', 'fni19'

corrTrained = 1
ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
useEqualTrNums = 1
shflTrsEachNeuron = 0  # Set to 0 for normal SVM training. # Shuffle trials in X_svm (for each neuron independently) to break correlations between neurons in each trial.
thTrained = 10#10 # number of trials of each class used for svm training, min acceptable value to include a day in analysis

trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
execfile("defFuns.py")

regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.


#%% Set days (all days and good days) for each mouse

days_allMice, numDaysAll, daysGood_allMice, dayinds_allMice, mn_corr_allMice = svm_setDays_allMice(mice, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, regressBins, useEqualTrNums, shflTrsEachNeuron, thTrained)


#%%

nowStr_allMice= []
eventI_ds_allMice = []
classErr_allN_allDays_allMice = []
classErr_inh_allDays_allMice = []
classErr_exc_allDays_allMice = []
eventI_ds_allDays_allMice = []


#%%    
for im in range(len(mice)):
        
    mousename = mice[im] # mousename = 'fni16' #'fni17'
    
    
    #%% Set svm_stab_testing mat file name

    imagingDir = setImagingAnalysisNamesP(mousename)

    fname = os.path.join(imagingDir, 'analysis')    
    if not os.path.exists(fname):
        print 'creating folder'
        os.makedirs(fname)    
        
    finame = os.path.join(fname, 'svm_testEachDecoderOnAllTimes_[0-9]*.mat')

    stabTestDecodeName = glob.glob(finame)
    stabTestDecodeName = sorted(stabTestDecodeName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    stabTestDecodeName = stabTestDecodeName[0]
    print stabTestDecodeName
    
    nowStr = stabTestDecodeName[-17:-4]
    nowStr_allMice.append(nowStr)


    #%% load stab_testing vars

    Data = scio.loadmat(stabTestDecodeName)
    
    classErr_allN_allDays = Data.pop('classErr_allN_allDays').flatten() # nDays # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    classErr_inh_allDays = Data.pop('classErr_inh_allDays').flatten() # nDays # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    classErr_exc_allDays = Data.pop('classErr_exc_allDays') # nDays x nExcSamps # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    eventI_ds_allDays = Data.pop('eventI_ds_allDays').flatten().astype(int)
#    eventI_allDays = Data.pop('eventI_allDays').flatten()
#    lastTimeBinMissed_allDays = Data.pop('lastTimeBinMissed_allDays').flatten()
                           
    numExcSamps = classErr_exc_allDays[0].shape[0]
    
    
    ##%% Plot each day    
    '''
    iday = 0                               
    classErrAveSamps = np.mean(classErr_allN_allDays[iday], axis=1) # trainedFrs x testingFrs
    
    nFrs = classErr_allN_allDays[iday].shape[-1]
    
    plt.figure()
    plt.imshow(classErrAveSamps); plt.colorbar()
    plt.axhline(eventI_ds_allDays[iday], 0, nFrs)
    plt.axvline(eventI_ds_allDays[iday], 0, nFrs)
    '''


    ############### ############### ############### ############### 
    #%% Keep vars of all mice

    classErr_allN_allDays_allMice.append(classErr_allN_allDays) # nMice # each mouse: nDays # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    classErr_inh_allDays_allMice.append(classErr_inh_allDays) # nMice # each mouse:  nDays # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    classErr_exc_allDays_allMice.append(classErr_exc_allDays) # nMice # each mouse:  nDays x nExcSamps # each day: trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
    eventI_ds_allDays_allMice.append(eventI_ds_allDays)
    
    


#%%
############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### 
############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### ############### 

#%% Plots of each mouse

for im in range(len(mice)):
    
    classErr_allN_allDays = classErr_allN_allDays_allMice[im] # numDays
    classErr_inh_allDays = classErr_inh_allDays_allMice[im] # numDays
    classErr_exc_allDays = classErr_exc_allDays_allMice[im] # numDays x nExcSamps
   
   
    #%% For each day average class accur across samps
    
    # subtract 100 to turn it into class accur
    classAcc_allN_aveSamps = 100-np.array([np.mean(classErr_allN_allDays[iday], axis=1) for iday in range(numDaysAll[im])]) # numDays; each day: trainedFrs x testingFrs
    classAcc_inh_aveSamps = 100-np.array([np.mean(classErr_inh_allDays[iday], axis=1) for iday in range(numDaysAll[im])]) # numDays; each day: trainedFrs x testingFrs
    classAcc_exc_aveSamps = 100-np.array([([np.mean(classErr_exc_allDays[iday][iexc], axis=1) for iexc in range(numExcSamps)]) for iday in range(numDaysAll[im])]) # numDays x nExcSamps; each day: trainedFrs x testingFrs



    ############### Align samp-averaged traces of all days ###############

    #%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
    # By common eventI, we  mean the index on which all traces will be aligned.
    
    time_aligned, nPreMin, nPostMin = set_nprepost(classAcc_allN_aveSamps, eventI_ds_allDays, mn_corr_allMice[im], thTrained, regressBins) # av_test_data_inh

    eventI_ds_allMice.append(nPreMin) # you need this var to align traces of all mice
    
    
    #%% Align traces of all days

    classAcc_allN_aveSamps_alig = alTrace_frfr(classAcc_allN_aveSamps, eventI_ds_allDays, nPreMin, nPostMin) # days x trainedAlFrs x testedAlFrs
    classAcc_inh_aveSamps_alig = alTrace_frfr(classAcc_inh_aveSamps, eventI_ds_allDays, nPreMin, nPostMin) # days x trainedAlFrs x testedAlFrs
    classAcc_exc_aveSamps_alig = np.array([alTrace_frfr(classAcc_exc_aveSamps[:,iexc], eventI_ds_allDays, nPreMin, nPostMin) for iexc in range(numExcSamps)]) # excSamps x days x alFrs x alFrs
    

    #%% Average above aligned traces across days

    classAcc_allN_aveSamps_alig_avD = np.mean(classAcc_allN_aveSamps_alig, axis=0)   # trainedAlFrs x testedAlFrs
    classAcc_inh_aveSamps_alig_avD = np.mean(classAcc_inh_aveSamps_alig, axis=0)   # trainedAlFrs x testedAlFrs
    classAcc_exc_aveSamps_alig_avD = np.mean(classAcc_exc_aveSamps_alig, axis=1)   # excSamps trainedAlFrs x testedAlFrs
    
    nFrs = classAcc_allN_aveSamps_alig_avD.shape[0]
    
    
    #%%
    
    plt.figure() #(figsize=(7,7))
    
    extent = setExtent_imshow(time_aligned)
    
    plt.imshow(classAcc_allN_aveSamps_alig_avD, extent=extent)

    plt.axhline(nPreMin, 0, nFrs)
    plt.axvline(nPreMin, 0, nFrs)
    
    plt.ylabel('Time at which decoder was trained (rel. choice, ms)')
    plt.xlabel('Time at which decoder was tested (rel. choice, ms)')
    plt.colorbar(label='Class accuracy')
    
    makeNicePlots(plt.gca())
    
  
    
    remember about the diagonal! ... no testing training data... so it makes sense to remove it
    also use extent for all stability figures
    
    
    
    
    