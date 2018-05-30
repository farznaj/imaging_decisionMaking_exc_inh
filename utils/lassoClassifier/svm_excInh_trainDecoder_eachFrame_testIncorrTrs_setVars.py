#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Load mat vars saved in svm_excInh_trainDecoder_eachFrame.py with testIncorr=1... does it for all days.
It is like svm_excInh_trainDecoder_eachFrame_stabTestTimes_setVars.py, except there classErr vars are computed by manually projecting the incorr-trial traces on the corr-trial decoder.

Created on Sun May 27 22:58:04 2018
@author: farznaj
"""

#%% Change the following vars:

mice = 'fni16', 'fni17', 'fni18', 'fni19'

saveResults = 1

testIncorr = 1 # if 1, load svm file in which the decoder trained on correct trials was used to test how it does on incorrect trials, i.e. on predicting labels of incorrect trials (using incorr trial neural traces)
    
doAllN = 1 # plot allN, instead of allExc

outcome2ana = 'corr' # '', corr', 'incorr' # trials to use for SVM training (all, correct or incorrect trials) # outcome2ana will be used if trialHistAnalysis is 0. When it is 1, by default we are analyzing past correct trials. If you want to change that, set it in the matlab code.        
corrTrained = 1
doIncorr = 0
loadWeights = 0
loadYtest = 0
trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 

execfile("defFuns.py")

chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]

import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.
        
useEqualTrNums = 1
shflTrsEachNeuron = 0  # Set to 0 for normal SVM training. # Shuffle trials in X_svm (for each neuron independently) to break correlations between neurons in each trial.
  
from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')


    
#%%
# im = 0
for im in range(len(mice)):
        
    #%%
    print '====================================================================================\n====================================================================================\n====================================================================================\n'     
    mousename = mice[im] # mousename = 'fni16' #'fni17'
    if mousename == 'fni18': #set one of the following to 1:
        allDays = 1# all 7 days will be used (last 3 days have z motion!)
        noZmotionDays = 0 # 4 days that dont have z motion will be used.
        noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!
        noExtraStimDays = np.nan
    elif mousename == 'fni19':    
        allDays = 1
        noExtraStimDays = 0   
        noZmotionDays = np.nan
        noZmotionDays_strict = np.nan
    else:
        allDays = np.nan
        noZmotionDays = np.nan
        noZmotionDays_strict = np.nan
        noExtraStimDays = np.nan
        
#    execfile("svm_plots_setVars_n.py")      
    days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays)
#    numDaysAll[im] = len(days)
        
        
    #%% Loop over days
    
    classErr_allN_allDays = []
    classErr_inh_allDays = []
    classErr_exc_allDays = []

    classErr_allN_shfl_allDays = []
    classErr_inh_shfl_allDays = []
    classErr_exc_shfl_allDays = []
        
    eventI_ds_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
    eventI_allDays = np.full((len(days)), np.nan)
    lastTimeBinMissed_allDays = np.full((len(days)), np.nan)
    

    #%% iday = 25   
    for iday in range(len(days)): 
    
        #%%            
        print '___________________'
        imagingFolder = days[iday][0:6]; #'151013'
        mdfFileNumber = map(int, (days[iday][7:]).split("-")); #[1,2] 
            
        ##%% Set .mat file names
        pnev2load = [] #[] [3] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.
        signalCh = [2] # since gcamp is channel 2, should be always 2.
        postNProvided = 1; # If your directory does not contain pnevFile and instead it contains postFile, set this to 1 to get pnevFileName
        
        # from setImagingAnalysisNamesP import *
        
        imfilename, pnevFileName, dataPath = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load, postNProvided=postNProvided)
        
        postName = os.path.join(os.path.dirname(pnevFileName), 'post_'+os.path.basename(pnevFileName))
        moreName = os.path.join(os.path.dirname(pnevFileName), 'more_'+os.path.basename(pnevFileName))
        
        print(os.path.basename(imfilename))
    
    
        ###########################################################################################################################################
    
        #%% Load SVM vars : loadSVM_excInh

        perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, \
        w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN, trsExcluded, \
        perClassErrorTest_data_inh_incorr, perClassErrorTest_shfl_inh_incorr, perClassErrorTest_data_allExc_incorr, perClassErrorTest_shfl_allExc_incorr, perClassErrorTest_data_exc_incorr, perClassErrorTest_shfl_exc_incorr, \
        = loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0, loadYtest=loadYtest, testIncorr=testIncorr)

#        perClassErrorTest_data_inh_incorr # samps x frames
#        perClassErrorTest_data_exc_incorr # numShufflesExc x numSamples x numFrames
        
        ### load eventI_ds
        d = scio.loadmat(svmName_allN, variable_names=['eventI_ds'])
        eventI_ds = d.pop('eventI_ds').flatten()
        
        
        #%% Set var sizesso they match classErr vars (as they used to be computed in svm_excInh_trainDecoder_eachFrame_stabTestTimes_setVars.py by manually projecting the incorr-trial traces on to the decoder) 
        
        classErr_allN = perClassErrorTest_data_allExc_incorr.T # frames x samps
        classErr_inh = perClassErrorTest_data_inh_incorr.T
        classErr_exc = np.transpose(perClassErrorTest_data_exc_incorr,(0,2,1)) # nExcSamps x frs x nSamps
        
        classErr_allN_shfl = perClassErrorTest_shfl_allExc_incorr.T # frames x samps
        classErr_inh_shfl = perClassErrorTest_shfl_inh_incorr.T
        classErr_exc_shfl = np.transpose(perClassErrorTest_shfl_exc_incorr,(0,2,1)) # nExcSamps x frs x nSamps



        ############################################################%%    
        #%% Keep vars of all days
        
        classErr_allN_allDays.append(classErr_allN)
        classErr_inh_allDays.append(classErr_inh)        
        classErr_exc_allDays.append(classErr_exc.tolist())  

        classErr_allN_shfl_allDays.append(classErr_allN_shfl)
        classErr_inh_shfl_allDays.append(classErr_inh_shfl)            
        classErr_exc_shfl_allDays.append(classErr_exc_shfl.tolist())  
        
        eventI_ds_allDays[iday] = eventI_ds.astype(int)
#        eventI_allDays[iday] = eventI #.astype(int)
#        lastTimeBinMissed_allDays[iday] = lastTimeBinMissed

                
    #%% Done with all days
    ####################################################################################################################################
    ############################################ Save results as .mat files in a folder named svm ####################################################################################################################################
    ####################################################################################################################################
    
    #%% Save SVM results
    
    if saveResults:
        print 'Saving results ....'
        
        fname = os.path.join(os.path.dirname(os.path.dirname(imfilename)), 'analysis')    
        if not os.path.exists(fname):
            print 'creating folder'
            os.makedirs(fname)    
            
        nts = ''    
        nw = ''
        
        if testIncorr:
            sn = 'svm_testDecoderOnIncorrTrs'
        else:
            sn = 'svm_testEachDecoderOnAllTimes'
            
        finame = os.path.join(fname, ('%s_%s%s%s.mat' %(sn, nts, nw, nowStr)))
        
        
        scio.savemat(finame, {'eventI_ds_allDays':eventI_ds_allDays,                              
                             'classErr_allN_allDays':classErr_allN_allDays,
                             'classErr_inh_allDays':classErr_inh_allDays,
                             'classErr_exc_allDays':classErr_exc_allDays,
                             'classErr_allN_shfl_allDays':classErr_allN_shfl_allDays,
                             'classErr_inh_shfl_allDays':classErr_inh_shfl_allDays,
                             'classErr_exc_shfl_allDays':classErr_exc_shfl_allDays}) # 'lastTimeBinMissed_allDays':lastTimeBinMissed_allDays, 'eventI_allDays':eventI_allDays,
    
    else:
        print 'Not saving .mat file'

                
    #%% Keep vars of all mice

#    classErr_allN_allMice.append(classErr_allN_allDays)
#    classErr_inh_allMice.append(classErr_inh_allDays)        
#    classErr_exc_allMice.append(classErr_exc_allDays)        
#    eventI_ds_allDays_allMice.append(eventI_ds_allDays)    
#    eventI_allDays_allMice.append(eventI_allDays)    


#classErr_allN_allMice = np.array(classErr_allN_allMice)
#classErr_inh_allMice = np.array(classErr_inh_allMice)        
#classErr_exc_allMice = np.array(classErr_exc_allMice)                
#eventI_ds_allDays_allMice = np.array(eventI_ds_allDays_allMice)
#eventI_allDays_allMice = np.array(eventI_allDays_allMice)
#eventI_ds_allDays_allMice = np.array([[int(i) for i in eventI_ds_allDays_allMice[im]] for im in range(len(mice))]) # convert eventIs to int
#numDaysAll = numDaysAll.astype(int)







