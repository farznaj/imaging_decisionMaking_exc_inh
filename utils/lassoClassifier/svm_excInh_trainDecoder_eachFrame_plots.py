# -*- coding: utf-8 -*-
"""
Plot time course of class accuracy: svm trained on non-overlapping time windows  (outputs of file svm_eachFrame.py)
 ... svm trained to decode choice on choice-aligned or stimulus-aligned traces.
 
 
Remember for fni18 there are 2 svm_eachFrame mat files, the earlier file is using all trials (unequal HR, LR, like how you've done all your analysis). 
The later mat file is with equal number of hr and lr trials (subselecting trials)... this helped with 151209 class accur trace which was weird in the earlier mat file.
 
Created on Sun Mar 12 15:12:29 2017
@author: farznaj
"""     

#%% Change the following vars:

mousename = 'fni19'

shflTrsEachNeuron = 0  # Set to 0 for normal SVM training. # Shuffle trials in X_svm (for each neuron independently) to break correlations between neurons in each trial.
addNs_roc = 1 # if 1 do the following analysis: add neurons 1 by 1 to the decoder based on their tuning strength to see how the decoder performance increases.
do_excInhHalf = 0 # 0: Load vars for inh,exc,allExc, 1: Load exc,inh SVM vars for excInhHalf (ie when the population consists of half exc and half inh) and allExc2inhSize (ie when populatin consists of allExc but same size as 2*inh size)
loadYtest = 0 # get svm performance for different trial strength # to load the svm files that include testTrInds_allSamps, etc vars (ie in addition to percClassError, they include the Y_hat for each trial)

savefigs = 0
doAllN = 1 # matters only when do_excInhHalf =0; plot allN, instead of allExc


corrTrained = 1
doIncorr = 0

thTrained = 10  # number of trials of each class used for svm training, min acceptable value to include a day in analysis
loadWeights = 0
useEqualTrNums = 1

import numpy as np
allDays = np.nan
noZmotionDays = np.nan
noZmotionDays_strict = np.nan
noExtraStimDays = np.nan

if mousename == 'fni18': #set one of the following to 1:
    allDays = 1# all 7 days will be used (last 3 days have z motion!)
    noZmotionDays = 0 # 4 days that dont have z motion will be used.
    noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!
elif mousename == 'fni19':    
    allDays = 1    
    noExtraStimDays = 0   

    
ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. #chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
#loadInhAllexcEqexc = 1 # if 1, load 2nd run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc separately; also for all days the new vector inhRois_pix was used (not the old inhRois)       
trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
superimpose = 1 # the averaged aligned traces of testing and shuffled will be plotted on the same figure
thStimStrength = 3 # needed if loadYtest=1 # 2; # threshold of stim strength for defining hard, medium and easy trials.
plotSingleSessions = 0 # plot individual sessions when addNs_roc=0
    
chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]

if do_excInhHalf:
    doAllN=0
    labInh = 'InhExcHalf'
else:
    labInh = 'inh'

if loadYtest:
    doAllN=0
        
if doAllN==1:
    labAll = 'allN'
else:
    labAll = 'allExc'
       

#if loadInhAllexcEqexc==1:
if addNs_roc:   
    dnow = '/excInh_trainDecoder_eachFrame_addNs1by1ROC/'+mousename+'/'
else:
    dnow = '/excInh_trainDecoder_eachFrame/'+mousename+'/'
if do_excInhHalf:
    dnow = dnow+'InhExcHalf/'
if shflTrsEachNeuron:
    dnow = dnow+'trsShfledPerNeuron/'
#else: # old svm files
#    dnow = '/excInh_trainDecoder_eachFrame/'+mousename+'/inhRois/'
if doAllN==1:
    smallestC = 0 # Identify best c: if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
    if smallestC==1:
        print 'bestc = smallest c whose cv error is less than 1se of min cv error'
    else:
        print 'bestc = c that gives min cv error'

execfile("defFuns.py")
#execfile("svm_plots_setVars_n.py")  
days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays, loadYtest)
# remove some days if needed:
#days = np.delete(days, [14])

frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.
#lastTimeBinMissed = 0 #1# if 0, things were ran fine; if 1: by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')

   
#%% 
'''
#####################################################################################################################################################   
#####################################################################################################################################################
'''            
eventI_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
eventI_ds_allDays = np.full((len(days)), np.nan)
perClassErrorTest_data_inh_all = []
perClassErrorTest_shfl_inh_all = []
perClassErrorTest_chance_inh_all = []
perClassErrorTest_data_allExc_all = []
perClassErrorTest_shfl_allExc_all = []
perClassErrorTest_chance_allExc_all = []
perClassErrorTest_data_exc_all = []
perClassErrorTest_shfl_exc_all = []
perClassErrorTest_chance_exc_all = []
numInh = np.full((len(days)), np.nan)
numAllexc = np.full((len(days)), np.nan)
corr_hr_lr = np.full((len(days),2), np.nan) # number of hr, lr correct trials for each day
perClassErrorTest_data_easy_inh_all = []
perClassErrorTest_data_hard_inh_all = []
perClassErrorTest_data_medium_inh_all = []
perClassErrorTest_data_easy_allExc_all = []
perClassErrorTest_data_hard_allExc_all = []
perClassErrorTest_data_medium_allExc_all = []
perClassErrorTest_data_easy_exc_all = []
perClassErrorTest_data_hard_exc_all = []
perClassErrorTest_data_medium_exc_all = []
perClassErrorTest_shfl_easy_inh_all = []
perClassErrorTest_shfl_hard_inh_all = []
perClassErrorTest_shfl_medium_inh_all = []
perClassErrorTest_shfl_easy_allExc_all = []
perClassErrorTest_shfl_hard_allExc_all = []
perClassErrorTest_shfl_medium_allExc_all = []
perClassErrorTest_shfl_easy_exc_all = []
perClassErrorTest_shfl_hard_exc_all = []
perClassErrorTest_shfl_medium_exc_all = []
num_ehm_inh_all = []
num_ehm_allExc_all = []
num_ehm_exc_all = []

#%% Loop over days    

# iday = 0  
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
    
    imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load, postNProvided=postNProvided)
    
    postName = os.path.join(os.path.dirname(pnevFileName), 'post_'+os.path.basename(pnevFileName))
    moreName = os.path.join(os.path.dirname(pnevFileName), 'more_'+os.path.basename(pnevFileName))
    
    print(os.path.basename(imfilename))


    #%% Get number of hr, lr trials that were used for svm training
    
    svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, [1,0,0], regressBins, useEqualTrNums, corrTrained, shflTrsEachNeuron)[0]   
    
    corr_hr, corr_lr = set_corr_hr_lr(postName, svmName)

    corr_hr_lr[iday,:] = [corr_hr, corr_lr]        
    
    
    #%% Load matlab vars to set eventI_ds (downsampled eventI)

    eventI, eventI_ds = setEventIds(postName, chAl, regressBins=3, trialHistAnalysis=0)
    
    eventI_allDays[iday] = eventI
    eventI_ds_allDays[iday] = eventI_ds    

    
    #%% Load SVM vars

    if addNs_roc:    
        # number of neurons in the decoder x nSamps x nFrs
#        perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, w_data_inh, w_data_allExc, b_data_inh, b_data_allExc, svmName_excInh = loadSVM_excInh_addNs1by1(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, loadWeights, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0)
#        perClassErrorTest_data_exc = 0; perClassErrorTest_shfl_exc = 0; perClassErrorTest_chance_exc = 0; 
        perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh = loadSVM_excInh_addNs1by1(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, loadWeights, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0)
        
    elif do_excInhHalf:
        # numShufflesExc x numSamples x numFrames
        perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, w_data_inh, w_data_allExc, b_data_inh, b_data_allExc, svmName_excInh = loadSVM_excInh_excInhHalf(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, loadWeights, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0)
        perClassErrorTest_data_exc = 0; perClassErrorTest_shfl_exc = 0; perClassErrorTest_chance_exc = 0; 

    else:
        perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN, testTrInds_allSamps_inh, Ytest_allSamps_inh, Ytest_hat_allSampsFrs_inh, testTrInds_allSamps_allExc, Ytest_allSamps_allExc, Ytest_hat_allSampsFrs_allExc, testTrInds_allSamps_exc, Ytest_allSamps_exc, Ytest_hat_allSampsFrs_exc = loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0, loadYtest=loadYtest)
    
    ##%% Get number of inh and exc        
    if loadWeights==1:
        numInh[iday] = w_data_inh.shape[1]
        numAllexc[iday] = w_data_allExc.shape[1]
        


    #%% Set classification error for easy, medium, and hard trials.
    
    if loadYtest:

        thMinEMH = -1 #1 #3 ... makes sense to make sure each day is contributing to all easy,med,hard, but if I make this above -1, fni16,18,19 will have almost no days!
        ###########%% Set stimrate for testing trials of each sample         
        
        Data = scio.loadmat(postName, variable_names=['stimrate', 'cb'])
        stimrate = np.array(Data.pop('stimrate')).flatten().astype('float')
        cb = np.array(Data.pop('cb')).flatten().astype('float')        
        
        # inh
        data = scio.loadmat(svmName_excInh[0], variable_names=['trsExcluded', 'trsnow_allSamps'])
        trsExcluded = np.array(data.pop('trsExcluded')).flatten().astype('bool') 
        trsnow_allSamps_inh = np.array(data.pop('trsnow_allSamps')).astype('int') # index of trials after picking random hr (or lr) in order to make sure both classes have the same number in the final Y (on which svm was run)
        
        # allExc
        data = scio.loadmat(svmName_excInh[1], variable_names=['trsnow_allSamps'])
        trsnow_allSamps_allExc = np.array(data.pop('trsnow_allSamps')).astype('int') 
        
        # exc
        data = scio.loadmat(svmName_excInh[2], variable_names=['trsnow_allSamps'])
        trsnow_allSamps_exc = np.array(data.pop('trsnow_allSamps')).astype('int') 

        
        stimrate = stimrate[~trsExcluded] # length= length of Y0 (ie before making hr and lr the same number in Y)
        len_test = Ytest_allSamps_inh.shape[1] # numSamples x numTestTrs
        numSamps = testTrInds_allSamps_inh.shape[0]
        nFrs = Ytest_hat_allSampsFrs_inh.shape[1]
        numShufflesExc = trsnow_allSamps_exc.shape[0]
        
#        stimrate_testTrs_allSamps = np.full((numSamps, len_test), np.nan)                
        perClassEr_easy_inh = np.full((numSamps, nFrs), np.nan)                
        perClassEr_hard_inh = np.full((numSamps, nFrs), np.nan)
        perClassEr_medium_inh = np.full((numSamps, nFrs), np.nan)
        perClassEr_easy_allExc = np.full((numSamps, nFrs), np.nan)
        perClassEr_hard_allExc = np.full((numSamps, nFrs), np.nan)
        perClassEr_medium_allExc = np.full((numSamps, nFrs), np.nan)
        perClassEr_easy_exc = np.full((numShufflesExc, numSamps, nFrs), np.nan)
        perClassEr_hard_exc = np.full((numShufflesExc, numSamps, nFrs), np.nan)
        perClassEr_medium_exc = np.full((numShufflesExc, numSamps, nFrs), np.nan)
        perClassEr_easy_shfl_inh = np.full((numSamps, nFrs), np.nan)                
        perClassEr_hard_shfl_inh = np.full((numSamps, nFrs), np.nan)
        perClassEr_medium_shfl_inh = np.full((numSamps, nFrs), np.nan)
        perClassEr_easy_shfl_allExc = np.full((numSamps, nFrs), np.nan)
        perClassEr_hard_shfl_allExc = np.full((numSamps, nFrs), np.nan)
        perClassEr_medium_shfl_allExc = np.full((numSamps, nFrs), np.nan)
        perClassEr_easy_shfl_exc = np.full((numShufflesExc, numSamps, nFrs), np.nan)
        perClassEr_hard_shfl_exc = np.full((numShufflesExc, numSamps, nFrs), np.nan)
        perClassEr_medium_shfl_exc = np.full((numShufflesExc, numSamps, nFrs), np.nan)        
        
        num_ehm_inh = np.full((numSamps, 3), np.nan) # number of easy, hard and medium testing trials for the inhibitory-population decoder
        num_ehm_allExc = np.full((numSamps, 3), np.nan)
        num_ehm_exc = np.full((numShufflesExc, numSamps, 3), np.nan)
        
        for isamp in range(numSamps):            
            
            ###########%% set stimrate for testing trials            
            # inh
            trsnow = trsnow_allSamps_inh[isamp]
            testTrInds = testTrInds_allSamps_inh[isamp]                
            testTrInds_outOfY0 = trsnow[testTrInds] # index of testing trials out of Y0 (not Y!) (that will be used in svm below)
            stimrateTestTrs = stimrate[testTrInds_outOfY0] # stimrate of testing trials    
            # set easy, medium, hard trials
            s = stimrateTestTrs-cb # how far is the stimulus rate from the category boundary?        
            str2ana_easy_inh = (abs(s) >= (max(abs(s)) - thStimStrength))        
            str2ana_hard_inh = (abs(s) <= thStimStrength)        
            str2ana_medium_inh = ((abs(s) > thStimStrength) & (abs(s) < (max(abs(s)) - thStimStrength)))
            num_ehm_inh[isamp,:] = [sum(str2ana_easy_inh), sum(str2ana_hard_inh), sum(str2ana_medium_inh)]
    
            # allExc
            trsnow = trsnow_allSamps_allExc[isamp]
            testTrInds = testTrInds_allSamps_allExc[isamp]                
            testTrInds_outOfY0 = trsnow[testTrInds] # index of testing trials out of Y0 (not Y!) (that will be used in svm below)
            stimrateTestTrs = stimrate[testTrInds_outOfY0] # stimrate of testing trials    
            # set easy, medium, hard trials
            s = stimrateTestTrs-cb # how far is the stimulus rate from the category boundary?        
            str2ana_easy_allExc = (abs(s) >= (max(abs(s)) - thStimStrength))        
            str2ana_hard_allExc = (abs(s) <= thStimStrength)        
            str2ana_medium_allExc = ((abs(s) > thStimStrength) & (abs(s) < (max(abs(s)) - thStimStrength)))
            num_ehm_allExc[isamp,:] = [sum(str2ana_easy_allExc), sum(str2ana_hard_allExc), sum(str2ana_medium_allExc)]

            # exc
            str2ana_easy_exc = []
            str2ana_hard_exc = []
            str2ana_medium_exc = []
            for ii in range(numShufflesExc): # loop through exc subsamples   
                trsnow = trsnow_allSamps_exc[ii,isamp] # numShufflesExc x numSamples x numTrs
                testTrInds = testTrInds_allSamps_exc[ii,isamp]                
                testTrInds_outOfY0 = trsnow[testTrInds] # index of testing trials out of Y0 (not Y!) (that will be used in svm below)
                stimrateTestTrs = stimrate[testTrInds_outOfY0] # stimrate of testing trials    
                # set easy, medium, hard trials
                s = stimrateTestTrs-cb # how far is the stimulus rate from the category boundary?        
                str2ana_easy_exc.append(abs(s) >= (max(abs(s)) - thStimStrength))        
                str2ana_hard_exc.append(abs(s) <= thStimStrength)        
                str2ana_medium_exc.append((abs(s) > thStimStrength) & (abs(s) < (max(abs(s)) - thStimStrength)))
            num_ehm_exc[:,isamp,:] = np.array([np.sum(str2ana_easy_exc, axis=1), np.sum(str2ana_hard_exc, axis=1), np.sum(str2ana_medium_exc, axis=1)]).T



            ###########%% set fraction of classification error for each trial strength 
            Ytest_inh = Ytest_allSamps_inh[isamp] 
            Ytest_allExc = Ytest_allSamps_allExc[isamp]             
            
            for ifr in range(nFrs): # each frame has a Ytest_hat                
                # inh
                if np.all(num_ehm_inh[isamp,:]>thMinEMH): # len(Ytest_inh) > thMinEMH:
                    Ytest_hat = Ytest_hat_allSampsFrs_inh[isamp,ifr]
                    if type(Ytest_hat)==np.float64:
                        Ytest_hat = np.array([Ytest_hat])                        
                    perClassEr_easy_inh[isamp,ifr] = perClassError(Ytest_inh[str2ana_easy_inh], Ytest_hat[str2ana_easy_inh])
                    perClassEr_hard_inh[isamp,ifr] = perClassError(Ytest_inh[str2ana_hard_inh], Ytest_hat[str2ana_hard_inh])
                    perClassEr_medium_inh[isamp,ifr] = perClassError(Ytest_inh[str2ana_medium_inh], Ytest_hat[str2ana_medium_inh])                
                    Ytest_hat = Ytest_hat[rng.permutation(len_test)] # shuffle Ytest_hat
                    perClassEr_easy_shfl_inh[isamp,ifr] = perClassError(Ytest_inh[str2ana_easy_inh], Ytest_hat[str2ana_easy_inh])
                    perClassEr_hard_shfl_inh[isamp,ifr] = perClassError(Ytest_inh[str2ana_hard_inh], Ytest_hat[str2ana_hard_inh])
                    perClassEr_medium_shfl_inh[isamp,ifr] = perClassError(Ytest_inh[str2ana_medium_inh], Ytest_hat[str2ana_medium_inh])                                
                # allExc
                if np.all(num_ehm_allExc[isamp,:]>thMinEMH): # len(Ytest_allExc) > thMinEMH:
                    Ytest_hat = Ytest_hat_allSampsFrs_allExc[isamp,ifr]
                    if type(Ytest_hat)==np.float64:
                        Ytest_hat = np.array([Ytest_hat])
                    perClassEr_easy_allExc[isamp,ifr] = perClassError(Ytest_allExc[str2ana_easy_allExc], Ytest_hat[str2ana_easy_allExc])
                    perClassEr_hard_allExc[isamp,ifr] = perClassError(Ytest_allExc[str2ana_hard_allExc], Ytest_hat[str2ana_hard_allExc])
                    perClassEr_medium_allExc[isamp,ifr] = perClassError(Ytest_allExc[str2ana_medium_allExc], Ytest_hat[str2ana_medium_allExc])    
                    Ytest_hat = Ytest_hat[rng.permutation(len_test)] # shuffle Ytest_hat
                    perClassEr_easy_shfl_allExc[isamp,ifr] = perClassError(Ytest_inh[str2ana_easy_allExc], Ytest_hat[str2ana_easy_allExc])
                    perClassEr_hard_shfl_allExc[isamp,ifr] = perClassError(Ytest_inh[str2ana_hard_allExc], Ytest_hat[str2ana_hard_allExc])
                    perClassEr_medium_shfl_allExc[isamp,ifr] = perClassError(Ytest_inh[str2ana_medium_allExc], Ytest_hat[str2ana_medium_allExc])                                
                
                # exc
                for ii in range(numShufflesExc):
                    Ytest_exc = Ytest_allSamps_exc[ii,isamp] # numShufflesExc x numSamples x numTestTrs 
                    Ytest_hat = Ytest_hat_allSampsFrs_exc[ii,isamp,ifr]
                    if type(Ytest_hat)==np.float64:
                        Ytest_hat = np.array([Ytest_hat])
                    if np.all(num_ehm_exc[ii,isamp,:]>thMinEMH): # len(Ytest_exc) > thMinEMH:
                        perClassEr_easy_exc[ii,isamp,ifr] = perClassError(Ytest_exc[str2ana_easy_exc[ii]], Ytest_hat[str2ana_easy_exc[ii]])
                        perClassEr_hard_exc[ii,isamp,ifr] = perClassError(Ytest_exc[str2ana_hard_exc[ii]], Ytest_hat[str2ana_hard_exc[ii]])
                        perClassEr_medium_exc[ii,isamp,ifr] = perClassError(Ytest_exc[str2ana_medium_exc[ii]], Ytest_hat[str2ana_medium_exc[ii]])
                        Ytest_hat = Ytest_hat[rng.permutation(len_test)] # shuffle Ytest_hat
                        perClassEr_easy_shfl_exc[ii,isamp,ifr] = perClassError(Ytest_exc[str2ana_easy_exc[ii]], Ytest_hat[str2ana_easy_exc[ii]])
                        perClassEr_hard_shfl_exc[ii,isamp,ifr] = perClassError(Ytest_exc[str2ana_hard_exc[ii]], Ytest_hat[str2ana_hard_exc[ii]])
                        perClassEr_medium_shfl_exc[ii,isamp,ifr] = perClassError(Ytest_exc[str2ana_medium_exc[ii]], Ytest_hat[str2ana_medium_exc[ii]])
                    
                
    
    #%% Once done with all frames, save vars for all days
    
    perClassErrorTest_data_inh_all.append(perClassErrorTest_data_inh) # each day: samps x numFrs    # if addNs_roc : number of neurons in the decoder x nSamps x nFrs    
    perClassErrorTest_shfl_inh_all.append(perClassErrorTest_shfl_inh)
    perClassErrorTest_chance_inh_all.append(perClassErrorTest_chance_inh)
    perClassErrorTest_data_allExc_all.append(perClassErrorTest_data_allExc) # each day: samps x numFrs    # if addNs_roc : number of neurons in the decoder x nSamps x nFrs
    perClassErrorTest_shfl_allExc_all.append(perClassErrorTest_shfl_allExc)
    perClassErrorTest_chance_allExc_all.append(perClassErrorTest_chance_allExc) 
    if do_excInhHalf==0: #np.logical_and(do_excInhHalf==0, addNs_roc==0):
        perClassErrorTest_data_exc_all.append(perClassErrorTest_data_exc) # each day: numShufflesExc x numSamples x numFrames   # if addNs_roc: # numShufflesExc x number of neurons in the decoder x numSamples x nFrs 
        perClassErrorTest_shfl_exc_all.append(perClassErrorTest_shfl_exc)
        perClassErrorTest_chance_exc_all.append(perClassErrorTest_chance_exc)
    if loadYtest:
        perClassErrorTest_data_easy_inh_all.append(perClassEr_easy_inh)
        perClassErrorTest_data_hard_inh_all.append(perClassEr_hard_inh)
        perClassErrorTest_data_medium_inh_all.append(perClassEr_medium_inh)
        perClassErrorTest_data_easy_allExc_all.append(perClassEr_easy_allExc)
        perClassErrorTest_data_hard_allExc_all.append(perClassEr_hard_allExc)
        perClassErrorTest_data_medium_allExc_all.append(perClassEr_medium_allExc)
        perClassErrorTest_data_easy_exc_all.append(perClassEr_easy_exc)
        perClassErrorTest_data_hard_exc_all.append(perClassEr_hard_exc)
        perClassErrorTest_data_medium_exc_all.append(perClassEr_medium_exc)
        perClassErrorTest_shfl_easy_inh_all.append(perClassEr_easy_shfl_inh)
        perClassErrorTest_shfl_hard_inh_all.append(perClassEr_hard_shfl_inh)
        perClassErrorTest_shfl_medium_inh_all.append(perClassEr_medium_shfl_inh)
        perClassErrorTest_shfl_easy_allExc_all.append(perClassEr_easy_shfl_allExc)
        perClassErrorTest_shfl_hard_allExc_all.append(perClassEr_hard_shfl_allExc)
        perClassErrorTest_shfl_medium_allExc_all.append(perClassEr_medium_shfl_allExc)
        perClassErrorTest_shfl_easy_exc_all.append(perClassEr_easy_shfl_exc)
        perClassErrorTest_shfl_hard_exc_all.append(perClassEr_hard_shfl_exc)
        perClassErrorTest_shfl_medium_exc_all.append(perClassEr_medium_shfl_exc)
        num_ehm_inh_all.append(num_ehm_inh) # each day: samps x 3: easy, hard, medium
        num_ehm_allExc_all.append(num_ehm_allExc)
        num_ehm_exc_all.append(num_ehm_exc)
        
    # Delete vars before starting the next day    
    del perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc
    if loadWeights==1:
        del w_data_inh, w_data_exc, w_data_allExc
        

eventI_allDays = eventI_allDays.astype(int)   
eventI_ds_allDays = eventI_ds_allDays.astype(int)
numD = len(perClassErrorTest_data_inh_all)


#%%    
######################################################################################################################################################    
######################################################################################################################################################          
######################################################################################################################################################    
######################################################################################################################################################   

#%% Average and st error of class accuracies across CV samples ... for each day
# in the function below we turn class error to class accuracy (by subtracting them from 100)

if addNs_roc:
    fr2an = -1 # relative to eventI_ds_allDays
    # av_test_data_inh : # numDays; each day: number of neurons in the decoder
#    numSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc \
#        = av_se_CA_trsamps_addNs1by1(numD, perClassErrorTest_data_inh_all, perClassErrorTest_shfl_inh_all, perClassErrorTest_chance_inh_all, perClassErrorTest_data_allExc_all, perClassErrorTest_shfl_allExc_all, perClassErrorTest_chance_allExc_all, fr2an, eventI_ds_allDays)
    numSamples, numExcSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, av_test_data_exc, sd_test_data_exc, av_test_shfl_exc, sd_test_shfl_exc, av_test_chance_exc, sd_test_chance_exc, av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc \
        = av_se_CA_trsamps_addNs1by1(numD, perClassErrorTest_data_inh_all, perClassErrorTest_shfl_inh_all, perClassErrorTest_chance_inh_all, perClassErrorTest_data_exc_all, perClassErrorTest_shfl_exc_all, perClassErrorTest_chance_exc_all, perClassErrorTest_data_allExc_all, perClassErrorTest_shfl_allExc_all, perClassErrorTest_chance_allExc_all, fr2an, eventI_ds_allDays)
    
elif do_excInhHalf:
    numSamples, numExcSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc \
        = av_se_CA_trsamps_excInhHalf(numD, perClassErrorTest_data_inh_all, perClassErrorTest_shfl_inh_all, perClassErrorTest_chance_inh_all, perClassErrorTest_data_allExc_all, perClassErrorTest_shfl_allExc_all, perClassErrorTest_chance_allExc_all)

else:
    numSamples, numExcSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, av_test_data_exc, sd_test_data_exc, av_test_shfl_exc, sd_test_shfl_exc, av_test_chance_exc, sd_test_chance_exc, av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc \
        = av_se_CA_trsamps(numD, perClassErrorTest_data_inh_all, perClassErrorTest_shfl_inh_all, perClassErrorTest_chance_inh_all, perClassErrorTest_data_exc_all, perClassErrorTest_shfl_exc_all, perClassErrorTest_chance_exc_all, perClassErrorTest_data_allExc_all, perClassErrorTest_shfl_allExc_all, perClassErrorTest_chance_allExc_all)
 
    if loadYtest:
        # data ... easy, hard, medium
        _,_,av_test_data_inh_easy, sd_test_data_inh_easy, av_test_data_inh_hard, sd_test_data_inh_hard, av_test_data_inh_medium, sd_test_data_inh_medium, \
        av_test_data_exc_easy, sd_test_data_exc_easy, av_test_data_exc_hard, sd_test_data_exc_hard, av_test_data_exc_medium, sd_test_data_exc_medium, \
        av_test_data_allExc_easy, sd_test_data_allExc_easy, av_test_data_allExc_hard, sd_test_data_allExc_hard, av_test_data_allExc_medium, sd_test_data_allExc_medium \
        = av_se_CA_trsamps(numD, perClassErrorTest_data_easy_inh_all, perClassErrorTest_data_hard_inh_all, perClassErrorTest_data_medium_inh_all, \
                           perClassErrorTest_data_easy_exc_all, perClassErrorTest_data_hard_exc_all, perClassErrorTest_data_medium_exc_all, \
                           perClassErrorTest_data_easy_allExc_all, perClassErrorTest_data_hard_allExc_all, perClassErrorTest_data_medium_allExc_all)
        # shfl ... easy, hard, medium
        _,_,av_test_shfl_inh_easy, sd_test_shfl_inh_easy, av_test_shfl_inh_hard, sd_test_shfl_inh_hard, av_test_shfl_inh_medium, sd_test_shfl_inh_medium, \
        av_test_shfl_exc_easy, sd_test_shfl_exc_easy, av_test_shfl_exc_hard, sd_test_shfl_exc_hard, av_test_shfl_exc_medium, sd_test_shfl_exc_medium, \
        av_test_shfl_allExc_easy, sd_test_shfl_allExc_easy, av_test_shfl_allExc_hard, sd_test_shfl_allExc_hard, av_test_shfl_allExc_medium, sd_test_shfl_allExc_medium \
        = av_se_CA_trsamps(numD, perClassErrorTest_shfl_easy_inh_all, perClassErrorTest_shfl_hard_inh_all, perClassErrorTest_shfl_medium_inh_all, perClassErrorTest_shfl_easy_exc_all, perClassErrorTest_shfl_hard_exc_all, perClassErrorTest_shfl_medium_exc_all, perClassErrorTest_shfl_easy_allExc_all, perClassErrorTest_shfl_hard_allExc_all, perClassErrorTest_shfl_medium_allExc_all)


        ######################################## Exclude days that have too few non-nan samples! ######################################## 
        # number of non-nan days ... 
#        easy_nValidDays_inh = [~np.isnan(av_test_data_inh_easy[iday][0]) for iday in range(len(days))]
#        hard_nValidDays_inh = [~np.isnan(av_test_data_inh_hard[iday][0]) for iday in range(len(days))]
        
        thMinSamps = 10        
        
        ####### inh        
        nValidSamps_easy_inh = np.array([sum(~np.isnan(perClassErrorTest_data_easy_inh_all[iday][:,0])) for iday in range(len(days))])
        nValidSamps_hard_inh = np.array([sum(~np.isnan(perClassErrorTest_data_hard_inh_all[iday][:,0])) for iday in range(len(days))])
        nValidSamps_medium_inh = np.array([sum(~np.isnan(perClassErrorTest_data_medium_inh_all[iday][:,0])) for iday in range(len(days))])       
        # set to nan        
        for iday in range(len(days)):
            if nValidSamps_easy_inh[iday] < thMinSamps:
                av_test_data_inh_easy[iday] = np.full((len(av_test_data_inh_easy[iday])),np.nan)
                sd_test_data_inh_easy[iday] = np.full((len(av_test_data_inh_easy[iday])),np.nan)
            if nValidSamps_hard_inh[iday] < thMinSamps:
                av_test_data_inh_hard[iday] = np.full((len(av_test_data_inh_hard[iday])),np.nan)
                sd_test_data_inh_hard[iday] = np.full((len(av_test_data_inh_hard[iday])),np.nan)        
            if nValidSamps_medium_inh[iday] < thMinSamps:
                av_test_data_inh_medium[iday] = np.full((len(av_test_data_inh_medium[iday])),np.nan)
                sd_test_data_inh_medium[iday] = np.full((len(av_test_data_inh_medium[iday])),np.nan)

        ####### exc ... here if number of valid samples averaged across excSamps and trSumSamps is < 10/50 we exclude the day
        nValidSamps_easy_exc = np.array([np.mean(~np.isnan(perClassErrorTest_data_easy_exc_all[iday][:,:,0]), axis=(0,1)) for iday in range(len(days))])
        nValidSamps_hard_exc = np.array([np.mean(~np.isnan(perClassErrorTest_data_hard_exc_all[iday][:,:,0]), axis=(0,1)) for iday in range(len(days))])
        nValidSamps_medium_exc = np.array([np.mean(~np.isnan(perClassErrorTest_data_medium_exc_all[iday][:,:,0]), axis=(0,1)) for iday in range(len(days))])       
        # set to nan        
        for iday in range(len(days)):
            if nValidSamps_easy_exc[iday] < thMinSamps/float(numSamples):
                av_test_data_exc_easy[iday] = np.full((len(av_test_data_exc_easy[iday])),np.nan)
                sd_test_data_exc_easy[iday] = np.full((len(av_test_data_exc_easy[iday])),np.nan)
            if nValidSamps_hard_exc[iday] < thMinSamps/float(numSamples):
                av_test_data_exc_hard[iday] = np.full((len(av_test_data_exc_hard[iday])),np.nan)
                sd_test_data_exc_hard[iday] = np.full((len(av_test_data_exc_hard[iday])),np.nan)        
            if nValidSamps_medium_exc[iday] < thMinSamps/float(numSamples):
                av_test_data_exc_medium[iday] = np.full((len(av_test_data_exc_medium[iday])),np.nan)
                sd_test_data_exc_medium[iday] = np.full((len(av_test_data_exc_medium[iday])),np.nan)                

        ####### allExc
        nValidSamps_easy_allExc = np.array([sum(~np.isnan(perClassErrorTest_data_easy_allExc_all[iday][:,0])) for iday in range(len(days))])
        nValidSamps_hard_allExc = np.array([sum(~np.isnan(perClassErrorTest_data_hard_allExc_all[iday][:,0])) for iday in range(len(days))])
        nValidSamps_medium_allExc = np.array([sum(~np.isnan(perClassErrorTest_data_medium_allExc_all[iday][:,0])) for iday in range(len(days))])       
        # set to nan        
        for iday in range(len(days)):
            if nValidSamps_easy_allExc[iday] < thMinSamps:
                av_test_data_allExc_easy[iday] = np.full((len(av_test_data_allExc_easy[iday])),np.nan)
                sd_test_data_allExc_easy[iday] = np.full((len(av_test_data_allExc_easy[iday])),np.nan)
            if nValidSamps_hard_allExc[iday] < thMinSamps:
                av_test_data_allExc_hard[iday] = np.full((len(av_test_data_allExc_hard[iday])),np.nan)
                sd_test_data_allExc_hard[iday] = np.full((len(av_test_data_allExc_hard[iday])),np.nan)        
            if nValidSamps_medium_allExc[iday] < thMinSamps:
                av_test_data_allExc_medium[iday] = np.full((len(av_test_data_allExc_medium[iday])),np.nan)
                sd_test_data_allExc_medium[iday] = np.full((len(av_test_data_allExc_medium[iday])),np.nan)
                
                
        # Print number of valid sessions
        print '%d %d %d :inh easy, hard, medium: n valid sessions' %(sum(nValidSamps_easy_inh >= thMinSamps), sum(nValidSamps_hard_inh >= thMinSamps), sum(nValidSamps_medium_inh >= thMinSamps))
        print '%d %d %d :exc easy, hard, medium: n valid sessions' %(sum(nValidSamps_easy_exc >= thMinSamps/float(numSamples)), sum(nValidSamps_hard_exc >= thMinSamps/float(numSamples)), sum(nValidSamps_medium_exc >= thMinSamps/float(numSamples)))
        print '%d %d %d :allExc easy, hard, medium: n valid sessions' %(sum(nValidSamps_easy_allExc >= thMinSamps), sum(nValidSamps_hard_allExc >= thMinSamps), sum(nValidSamps_medium_allExc >= thMinSamps))
        


                
#%% Keep vars for chAl and stAl

if shflTrsEachNeuron:
    eventI_allDays_shflTrsEachN = eventI_allDays + 0    
    av_test_data_inh_shflTrsEachN =  av_test_data_inh + 0
    sd_test_data_inh_shflTrsEachN = sd_test_data_inh + 0 
    av_test_shfl_inh_shflTrsEachN = av_test_shfl_inh + 0
    sd_test_shfl_inh_shflTrsEachN = sd_test_shfl_inh + 0
    av_test_chance_inh_shflTrsEachN = av_test_chance_inh + 0
    sd_test_chance_inh_shflTrsEachN = sd_test_chance_inh + 0    
    av_test_data_allExc_shflTrsEachN = av_test_data_allExc + 0
    sd_test_data_allExc_shflTrsEachN = sd_test_data_allExc + 0
    av_test_shfl_allExc_shflTrsEachN = av_test_shfl_allExc + 0
    sd_test_shfl_allExc_shflTrsEachN = sd_test_shfl_allExc + 0
    av_test_chance_allExc_shflTrsEachN = av_test_chance_allExc + 0
    sd_test_chance_allExc_shflTrsEachN = sd_test_chance_allExc + 0
    av_test_data_exc_shflTrsEachN = av_test_data_exc + 0
    sd_test_data_exc_shflTrsEachN = sd_test_data_exc + 0
    av_test_shfl_exc_shflTrsEachN = av_test_shfl_exc + 0
    sd_test_shfl_exc_shflTrsEachN = sd_test_shfl_exc + 0
    av_test_chance_exc_shflTrsEachN = av_test_chance_exc + 0
    sd_test_chance_exc_shflTrsEachN = sd_test_chance_exc + 0
#    perClassErrorTest_data_inh_all_shflTrsEachN = perClassErrorTest_data_inh_all
#    perClassErrorTest_data_exc_all_shflTrsEachN = perClassErrorTest_data_exc_all
#    perClassErrorTest_data_allExc_all_shflTrsEachN = perClassErrorTest_data_allExc_all
    

if chAl==1:    
    eventI_allDays_ch = eventI_allDays + 0    
    av_test_data_inh_ch =  av_test_data_inh + 0
    sd_test_data_inh_ch = sd_test_data_inh + 0 
    av_test_shfl_inh_ch = av_test_shfl_inh + 0
    sd_test_shfl_inh_ch = sd_test_shfl_inh + 0
    av_test_chance_inh_ch = av_test_chance_inh + 0
    sd_test_chance_inh_ch = sd_test_chance_inh + 0    
    av_test_data_allExc_ch = av_test_data_allExc + 0
    sd_test_data_allExc_ch = sd_test_data_allExc + 0
    av_test_shfl_allExc_ch = av_test_shfl_allExc + 0
    sd_test_shfl_allExc_ch = sd_test_shfl_allExc + 0
    av_test_chance_allExc_ch = av_test_chance_allExc + 0
    sd_test_chance_allExc_ch = sd_test_chance_allExc + 0    
    if do_excInhHalf==0: #np.logical_and(do_excInhHalf==0, addNs_roc==0):
        av_test_data_exc_ch = av_test_data_exc + 0
        sd_test_data_exc_ch = sd_test_data_exc + 0
        av_test_shfl_exc_ch = av_test_shfl_exc + 0
        sd_test_shfl_exc_ch = sd_test_shfl_exc + 0
        av_test_chance_exc_ch = av_test_chance_exc + 0
        sd_test_chance_exc_ch = sd_test_chance_exc + 0
'''
else:        
    eventI_allDays_st = eventI_allDays + 0    
    av_l2_train_d_st = av_l2_train_d + 0
    sd_l2_train_d_st = sd_l2_train_d + 0    
    av_l2_test_d_st = av_l2_test_d + 0
    sd_l2_test_d_st = sd_l2_test_d + 0    
    av_l2_test_s_st = av_l2_test_s + 0
    sd_l2_test_s_st = sd_l2_test_s + 0    
    av_l2_test_c_st = av_l2_test_c + 0
    sd_l2_test_c_st = sd_l2_test_c + 0
'''    

no    

#%% Decide what days to analyze: exclude days with too few trials used for training SVM, also exclude incorr from days with too few incorr trials.

# th for min number of trs of each class
'''
thTrained = 30 #25; # 1/10 of this will be the testing tr num! and 9/10 was used for training
thIncorr = 4 #5
'''
mn_corr = np.min(corr_hr_lr,axis=1) # number of trials of each class. 90% of this was used for training, and 10% for testing.
#mn_corr = np.delete(mn_corr, [46]) # #np.append(mn_corr,[12,12])
print '%d days will be excluded: too few trials for svm training' %(sum(mn_corr < thTrained))

print np.array(days)[mn_corr < thTrained]



#%% Plots of adding neurons 1 by 1
##################################################################################################
##################################################################################################
##################################################################################################

if addNs_roc:   

    alph = .5
    labs = ['allExc','inh', 'exc']
    colors = ['k','r','b']
    colors_shflTrsEachN = 'darkgray', 'darksalmon', 'deepskyblue'        

    if np.array_equal(av_test_data_inh_shflTrsEachN , av_test_data_inh)==0:
        compWith_shflTrsEachN = 1 # codes are run for both values of shflTrsEachNeuron to compare CA traces before and after breaking noise correlations
        dnow = '/excInh_trainDecoder_eachFrame_addNs1by1ROC/'+mousename+'/'
    else:
        compWith_shflTrsEachN = 0

    
    ########### add nan for days with num neurons < mxNumNeur, so you can get an average across days
    def sameMaxNs(CAav_alig, mxNumNeur, nanEnd=1): 
        CAav_alig_sameMaxN = []
        for iday in range(len(CAav_alig)):
            # create a nan array
            a = np.full((mxNumNeur), np.nan)
            if nanEnd:# fill in the begining part of a with CAav_alig vals, so the end part of a includes nans
                a[0: min(mxNumNeur, CAav_alig[iday].shape[0])] = CAav_alig[iday][0: min(mxNumNeur, CAav_alig[iday].shape[0])]
            else: # fill in the end part of a with CAav_alig values, so the begining part of a includes nans
                a[-min(mxNumNeur, CAav_alig[iday].shape[0]):] = CAav_alig[iday][0: min(mxNumNeur, CAav_alig[iday].shape[0])]
            
            CAav_alig_sameMaxN.append(a) # numDays x max_numNeurs_used_for_training x nFrs    
            
        CAav_alig_sameMaxN = np.array(CAav_alig_sameMaxN)
        return CAav_alig_sameMaxN 



    #%% Make the traces for all days the same length in order to set average and se across days and plot them!
    
    ########## Define mxNumNeur for the function sameMaxNs
    numInh = np.array([len(av_test_data_inh[iday]) for iday in range(numD)])
    numExc = np.array([len(av_test_data_allExc[iday]) for iday in range(numD)])
#    numExcSubsamp = np.array([len(av_test_data_exc_ch[iday]) for iday in range(numD)]) # here exc was first subsampled (to take n exc, n=numInh), then it was ordered based on ROC    
    pv = 100 # add nan for days with num neurons < 20th percentile of nInh (or nExc)... so you can get an average across days (with max number of neurons in the decoder = 20th percentile (instead of min) number of inh neurons across days)
    pInh = np.percentile(numInh,pv).astype(int)
    pExc = np.percentile(numExc,pv).astype(int)
#    pInh = max(numInh[mn_corr>=thTrained])
#    pExc = max(numExc[mn_corr>=thTrained])    
    
    for nanEnd in [0,1]:  # nanEnd = 0 # nanEnd = 1
        # if 0, nans will be added at the begining of the traces (do this when you want to see how going from max to 1 nN in the decoder changes CA). 
        # if 1, nans will be added at the end (do this when you want to see how going from 1 to max nN in the decoder changes CA)
    
        ########### Make the traces for all days the same length (remember days have different number of neurons in the decoder): add nan for days with num neurons < pInh (eg 20th percentile of nInh (or nExc))... so you can get an average across days (with max number of neurons in the decoder = 20th percentile (instead of min) number of inh neurons across days)            
        av_test_data_inh_samePN = sameMaxNs(av_test_data_inh[mn_corr>=thTrained], pInh, nanEnd) # numDays x nNs in the decoder (20th perc number neurons across days)
        av_test_data_allExc_samePN = sameMaxNs(av_test_data_allExc[mn_corr>=thTrained], pExc, nanEnd)
        av_test_data_exc_samePN = sameMaxNs(av_test_data_exc[mn_corr>=thTrained], pInh, nanEnd)
        
        
        ##### for each value of number of neurons in the decoder how many days contributed (ie had non-nan values): (days that did not contribute are the ones whose numNs was less than 20th percentile of nNs across days)
        nDaysPinh = np.array([np.sum(~np.isnan(av_test_data_inh_samePN[:,iN])) for iN in range(av_test_data_inh_samePN.shape[1])])
        nDaysPexc = np.array([np.sum(~np.isnan(av_test_data_allExc_samePN[:,iN])) for iN in range(av_test_data_allExc_samePN.shape[1])])
        nDaysPexcSubsamp = np.array([np.sum(~np.isnan(av_test_data_exc_samePN[:,iN])) for iN in range(av_test_data_exc_samePN.shape[1])])
        
        if pv==100: #### If you went with pv=100 (ie you went up to max nN across days), then make sure you set to nan (in all days) those x values (nN in the deocder) that have too few non-nan days ... so those x values don't  contribute to nanmean across days.        
            
            thds = 5            
            thD = min(numDays, thds) #10
            av_test_data_inh_samePN[:,nDaysPinh<thD] = np.nan ##### For all days, set to nan nNs in decoder (x values) which are so large that only very few days (<thD) have that number and hence will contribute to average across days
            av_test_data_allExc_samePN[:,nDaysPexc<thD] = np.nan
            av_test_data_exc_samePN[:,nDaysPexcSubsamp<thD] = np.nan    
        
        ######################### Average and se across days #########################
        avD_av_test_data_inh_samePN = np.nanmean(av_test_data_inh_samePN, axis=0) # nInh (20th perc)
        seD_av_test_data_inh_samePN = np.nanstd(av_test_data_inh_samePN, axis=0) / np.sqrt(nDaysPinh) # nInh (20th perc)
        
        avD_av_test_data_allExc_samePN = np.nanmean(av_test_data_allExc_samePN, axis=0) # nExc (20th perc)
        seD_av_test_data_allExc_samePN = np.nanstd(av_test_data_allExc_samePN, axis=0) / np.sqrt(nDaysPexc)
    
        avD_av_test_data_exc_samePN = np.nanmean(av_test_data_exc_samePN, axis=0) # nExc (20th perc)
        seD_av_test_data_exc_samePN = np.nanstd(av_test_data_exc_samePN, axis=0) / np.sqrt(nDaysPexcSubsamp)    
            
        
        #################################################################################################################
        ######################### No noise correlations bc trials were shuffled for each neuron #########################
        if compWith_shflTrsEachN:
            ########### Make the traces for all days the same length (remember days have different number of neurons in the decoder): add nan for days with num neurons < pInh (eg 20th percentile of nInh (or nExc))... so you can get an average across days (with max number of neurons in the decoder = 20th percentile (instead of min) number of inh neurons across days)            
            av_test_data_inh_samePN_shflTrsEachN = sameMaxNs(av_test_data_inh_shflTrsEachN[mn_corr>=thTrained], pInh, nanEnd) # numDays x nNs in the decoder (20th perc number neurons across days)
            av_test_data_allExc_samePN_shflTrsEachN = sameMaxNs(av_test_data_allExc_shflTrsEachN[mn_corr>=thTrained], pExc, nanEnd)
            av_test_data_exc_samePN_shflTrsEachN = sameMaxNs(av_test_data_exc_shflTrsEachN[mn_corr>=thTrained], pInh, nanEnd)
            
            if pv==100: #### If you went with pv=100 (ie you went up to max nN across days), then make sure you set to nan (in all days) those x values (nN in the deocder) that have too few non-nan days ... so those x values don't  contribute to nanmean across days.        
                thD = min(numDays, thds) #10
                av_test_data_inh_samePN_shflTrsEachN[:,nDaysPinh<thD] = np.nan ##### For all days, set to nan nNs in decoder (x values) which are so large that only very few days (<thD) have that number and hence will contribute to average across days
                av_test_data_allExc_samePN_shflTrsEachN[:,nDaysPexc<thD] = np.nan
                av_test_data_exc_samePN_shflTrsEachN[:,nDaysPexcSubsamp<thD] = np.nan    
            
            ######################### Average and se across days #########################
            avD_av_test_data_inh_samePN_shflTrsEachN = np.nanmean(av_test_data_inh_samePN_shflTrsEachN, axis=0) # nInh (20th perc)
            seD_av_test_data_inh_samePN_shflTrsEachN = np.nanstd(av_test_data_inh_samePN_shflTrsEachN, axis=0) / np.sqrt(nDaysPinh) # nInh (20th perc)
            
            avD_av_test_data_allExc_samePN_shflTrsEachN = np.nanmean(av_test_data_allExc_samePN_shflTrsEachN, axis=0) # nExc (20th perc)
            seD_av_test_data_allExc_samePN_shflTrsEachN = np.nanstd(av_test_data_allExc_samePN_shflTrsEachN, axis=0) / np.sqrt(nDaysPexc)
        
            avD_av_test_data_exc_samePN_shflTrsEachN = np.nanmean(av_test_data_exc_samePN_shflTrsEachN, axis=0) # nExc (20th perc)
            seD_av_test_data_exc_samePN_shflTrsEachN = np.nanstd(av_test_data_exc_samePN_shflTrsEachN, axis=0) / np.sqrt(nDaysPexcSubsamp)    
        
        
            # P value across all neuron numbers (averaged across days): class accuracy being different between shflTrsEachNeuron and the actual case.
            _,p_exc = stats.ttest_ind(avD_av_test_data_exc_samePN.transpose(), avD_av_test_data_exc_samePN_shflTrsEachN.transpose(), nan_policy= 'omit') # p value of class accuracy being different between shflTrsEachNeuron and the actual case.
            _,p_inh = stats.ttest_ind(avD_av_test_data_inh_samePN.transpose(), avD_av_test_data_inh_samePN_shflTrsEachN.transpose(), nan_policy= 'omit') # p value of class accuracy being different between shflTrsEachNeuron and the actual case.
            _,p_allExc = stats.ttest_ind(avD_av_test_data_allExc_samePN.transpose(), avD_av_test_data_allExc_samePN_shflTrsEachN.transpose(), nan_policy= 'omit') # p value of class accuracy being different between shflTrsEachNeuron and the actual case.
            
            # P value for each neuron nuber across days : class accuracy being different between shflTrsEachNeuron and the actual case.
            _,p_exc_eachN = stats.ttest_ind(av_test_data_exc_samePN, av_test_data_exc_samePN_shflTrsEachN, axis=0, nan_policy= 'omit')
            _,p_inh_eachN = stats.ttest_ind(av_test_data_inh_samePN, av_test_data_inh_samePN_shflTrsEachN, axis=0, nan_policy= 'omit')
            _,p_allExc_eachN = stats.ttest_ind(av_test_data_allExc_samePN, av_test_data_allExc_samePN_shflTrsEachN, axis=0, nan_policy= 'omit')




        ########################################################################################################################################   
        ######################### Plot average and se across days: SVM performance vs number of neurons in the decoder #########################
        ########################################################################################################################################
        plt.figure(figsize=(4.5,3))
        
        ##### allExc #####
        i = 0
        av = avD_av_test_data_allExc_samePN
        sd = seD_av_test_data_allExc_samePN
        x = np.arange(1, len(av)+1)     
        if nanEnd==0: # reverse the traces so we go from max number of N to fewest number of N in the decoder (for the x axis)
            av = av[::-1]
            sd = sd[::-1]
            xtix = np.full(len(nDaysPexc), np.nan)
            xtix[0:sum(nDaysPexc>=thD)] = np.arange(1, sum(nDaysPexc>=thD)+1)[::-1]        
        plt.fill_between(x, av-sd, av+sd, alpha=alph, edgecolor=colors[i], facecolor=colors[i])
        plt.plot(x, av, colors[i], label=labs[i]) #, marker='.')
    
        ax1 = plt.gca()           
        xti = ax1.get_xticks().astype(int)
        if nanEnd==0:
            ax2 = ax1.twiny()    
            ax2.set_xticks(xti)
            xti = np.delete(xti, np.argwhere(xti>len(nDaysPexc)))
            xtx = xtix[xti].astype(int)
#            xtx = [str(x) for x in xtx[:-1]] + ['']
            ax2.set_xticklabels(xtx)
        else:
            xtix = x
            plt.xticks(xti, xtix[xti])          

        ##### shflTrsEachN : allExc
        if compWith_shflTrsEachN:    
            av = avD_av_test_data_allExc_samePN_shflTrsEachN
            sd = seD_av_test_data_allExc_samePN_shflTrsEachN
            x = np.arange(1, len(av)+1)     
            if nanEnd==0: # reverse the traces so we go from max number of N to fewest number of N in the decoder (for the x axis)
                av = av[::-1]
                sd = sd[::-1]
                xtix = np.full(len(nDaysPexc), np.nan)
                xtix[0:sum(nDaysPexc>=thD)] = np.arange(1, sum(nDaysPexc>=thD)+1)[::-1]        
            plt.fill_between(x, av-sd, av+sd, alpha=alph, edgecolor=colors_shflTrsEachN[i], facecolor=colors_shflTrsEachN[i])
            plt.plot(x, av, colors_shflTrsEachN[i]) #, marker='.')         
            if nanEnd==0:
                yt = 1.15
            else:
                yt = 1
            plt.title('P_actVSshfl:\nallExc %.2f, inh %.2f, exc %.2f' %(p_inh, p_exc, p_allExc), y=yt)
            # mark neuron numbers that are significantly different after and before shflTrsEachN
            yl = plt.gca().get_ylim()
            pp = p_allExc_eachN+0
            pp[pp>.05] = np.nan
            pp[pp<=.05] = yl[1]-np.diff(yl)/20
            plt.plot(x, pp, marker='*',color=colors[i], markeredgecolor=colors[i], linestyle='', markersize=1)
        
        
        ##### inh #####
        i = 1
        av = avD_av_test_data_inh_samePN
        sd = seD_av_test_data_inh_samePN
        x = np.arange(1, len(av)+1).astype(float)    
        if nanEnd==0: # reverse the traces so we go from max number of N to fewest number of N in the decoder (for the x axis)
            av = av[::-1]
            sd = sd[::-1]
            xtix = np.full(len(nDaysPinh), np.nan)
            xtix[0:sum(nDaysPinh>=thD)] = np.arange(1, sum(nDaysPinh>=thD)+1)[::-1]
        ax1.fill_between(x, av-sd, av+sd, alpha=alph, edgecolor=colors[i], facecolor=colors[i])
        ax1.plot(x, av, colors[i], label=labs[i]) #, marker='.')

        ##### shflTrsEachN : inh       
        if compWith_shflTrsEachN:
            av = avD_av_test_data_inh_samePN_shflTrsEachN
            sd = seD_av_test_data_inh_samePN_shflTrsEachN
            x = np.arange(1, len(av)+1).astype(float)    
            if nanEnd==0: # reverse the traces so we go from max number of N to fewest number of N in the decoder (for the x axis)
                av = av[::-1]
                sd = sd[::-1]
                xtix = np.full(len(nDaysPinh), np.nan)
                xtix[0:sum(nDaysPinh>=thD)] = np.arange(1, sum(nDaysPinh>=thD)+1)[::-1]
            ax1.fill_between(x, av-sd, av+sd, alpha=alph, edgecolor=colors_shflTrsEachN[i], facecolor=colors_shflTrsEachN[i])
            ax1.plot(x, av, colors_shflTrsEachN[i]) #, marker='.')
            # mark neuron numbers that are significantly different after and before shflTrsEachN
            yl = plt.gca().get_ylim()
            pp = p_inh_eachN+0
            pp[pp>.05] = np.nan
            pp[pp<=.05] = max(av+sd)+.5 #yl[1]-np.diff(yl)/20
            plt.plot(x, pp, marker='*',color=colors[i], markeredgecolor=colors[i], linestyle='', markersize=1)        
        
        ##### exc #####
        i = 2
        av = avD_av_test_data_exc_samePN
        sd = seD_av_test_data_exc_samePN
        x = np.arange(1, len(av)+1)    
        if nanEnd==0: # reverse the traces so we go from max number of N to fewest number of N in the decoder (for the x axis)
            av = av[::-1]
            sd = sd[::-1]
        ax1.fill_between(x, av-sd, av+sd, alpha=alph, edgecolor=colors[i], facecolor=colors[i])
        ax1.plot(x, av, colors[i], label=labs[i]) #, marker='.')    
        if nanEnd==0:
            xti = np.arange(0,sum(nDaysPinh>=thD),20) #len(x),20) #plt.gca().get_xticks().astype(int)        
            ax1.set_xticks(xti)
            ax1.set_xticklabels(xtix[xti].astype(int))

        #####  shflTrsEachN : exc
        if compWith_shflTrsEachN:
            av = avD_av_test_data_exc_samePN_shflTrsEachN
            sd = seD_av_test_data_exc_samePN_shflTrsEachN
            x = np.arange(1, len(av)+1)    
            if nanEnd==0: # reverse the traces so we go from max number of N to fewest number of N in the decoder (for the x axis)
                av = av[::-1]
                sd = sd[::-1]
            ax1.fill_between(x, av-sd, av+sd, alpha=alph, edgecolor=colors_shflTrsEachN[i], facecolor=colors_shflTrsEachN[i])
            ax1.plot(x, av, colors_shflTrsEachN[i]) #, marker='.')
            # mark neuron numbers that are significantly different after and before shflTrsEachN
            yl = plt.gca().get_ylim()
            pp = p_exc_eachN+0
            pp[pp>.05] = np.nan
            pp[pp<=.05] = max(av+sd)+.5 #yl[1]-np.diff(yl)/20
            plt.plot(x, pp, marker='*',color=colors[i], markeredgecolor=colors[i], linestyle='', markersize=1)            
        
        
        ax1.set_xlim([-5, sum(nDaysPexc>=thD)+4])   #pExc
        ax1.set_xlabel('Numbers of neurons in the decoder')
        ax1.set_ylabel('% Class accuracy')    
        ax1.legend(loc='center left', bbox_to_anchor=(.6, .2), frameon=False)         
        if nanEnd==0:
            ax2.set_xlim([-5, sum(nDaysPexc>=thD)+4])  
            ax1.tick_params(direction='out')    
            ax2.tick_params(direction='out')    
            
            ax1.spines['right'].set_visible(False)       
            ax2.spines['right'].set_visible(False)           
            ax1.yaxis.set_ticks_position('left')
            ax2.yaxis.set_ticks_position('left')
        else:
            makeNicePlots(plt.gca())      
        
        
        
        if savefigs:       
            if nanEnd==0:
                nN = 'dropNs_thDays%d_' %(thds)
            else:
                nN = 'addNs_thDays%d_' %(thds)
            if compWith_shflTrsEachN:
                nc = 'VSshflTrsEachN_'
            else:
                nc = ''
            if chAl==1:
                dd = 'chAl_'+nc+'avSeDays_'+nN+'addNsROC_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            else:
                dd = 'stAl_'+nc+'avSeDays_'+nN+'addNsROC_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
                
            d = os.path.join(svmdir+dnow)
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)
                    
            fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
            
    
    
        #%%
        ######################### Plot diff of ave +/- se across days ... to better see when it plateaus #########################        
#        if nanEnd==0:
#            ai = avD_av_test_data_inh_samePN[::-1]
#            aa = avD_av_test_data_allExc_samePN[::-1]
#            ae = avD_av_test_data_exc_samePN[::-1]            
#        else:
        if nanEnd:
            ai = avD_av_test_data_inh_samePN
            aa = avD_av_test_data_allExc_samePN
            ae = avD_av_test_data_exc_samePN
            
            di = np.diff(ai)
            de = np.diff(aa)
            des = np.diff(ae)
            
            
            plt.figure(figsize=(4.5,3))
            plt.plot(de, color='k', label='allExc')
            plt.plot(di, color='r', label='inh')
            plt.plot(des, color='b', label='exc')
            plt.xlim([-5, pExc+4])  
            makeNicePlots(plt.gca())  
            plt.xlabel('Numbers Neurons in the decoder')
            plt.ylabel('% Change in class accuracy')    
            plt.legend(loc='center left', bbox_to_anchor=(.7, .7), frameon=False) 
            
            
            if savefigs: 
                if nanEnd==0:
                    nN = 'dropNs_thDays%d_' %(thds)
                else:
                    nN = 'addNs_thDays%d_' %(thds)

                if chAl==1:
                    dd = 'chAl_diff_avSeDays_'+nN+'addNsROC_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
                else:
                    dd = 'stAl_diff_avSeDays_'+nN+'addNsROC_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
                    
                d = os.path.join(svmdir+dnow)
                if not os.path.exists(d):
                    print 'creating folder'
                    os.makedirs(d)
                        
                fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
             
                plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
                


    #%%
    ######################### Plots of each day #########################
    ##%% Plot each day: how choice prediction varies by increasing the population size, compare exc vs inh
   
    plt.figure(figsize=(3, 3.5*numD))    
    cnt = 0
    for iday in range(numD):    
        if mn_corr[iday] >= thTrained:
            cnt = cnt+1
                   
            plt.subplot(numD,1,cnt)
            
            # allExc
            i = 0            
            av = av_test_data_allExc[iday]
            sd = sd_test_data_allExc[iday]            
            x = np.arange(1, len(av)+1)
            plt.fill_between(x, av-sd, av+sd, alpha=alph, edgecolor=colors[i], facecolor=colors[i])
            plt.plot(x, av, colors[i], label=labs[i])
            
            # inh
            i = 1
            av = av_test_data_inh[iday]
            sd = sd_test_data_inh[iday]            
            x = np.arange(1, len(av)+1)
            plt.fill_between(x, av-sd, av+sd, alpha=alph, edgecolor=colors[i], facecolor=colors[i])
            plt.plot(x, av, colors[i], label=labs[i])            
            
            # exc
            i = 2
            av = av_test_data_exc[iday]
            sd = sd_test_data_exc[iday]            
            x = np.arange(1, len(av)+1)
            plt.fill_between(x, av-sd, av+sd, alpha=alph, edgecolor=colors[i], facecolor=colors[i])
            plt.plot(x, av, colors[i], label=labs[i])
                        
            
            plt.title(days[iday])        
            plt.xlim([-5, len(av_test_data_allExc[iday])+4])  
            makeNicePlots(plt.gca())
            plt.subplots_adjust(wspace=.5, hspace=.5)
            if cnt==1:
                plt.xlabel('Numbers neurons in the decoder')
                plt.ylabel('% Class accuracy')
            
            
    ##%% Save the figure    
    if savefigs:               
        if chAl==1:
            dd = 'chAl_eachDay_addNsROC_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_eachDay_addNsROC_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
     
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
        
        
        
        
    #%% For each day compute CA when all neurons are in the deocder (do separately for inh, exc and allExc)
    
    # get vals at nN in decoder = max nN        
    av_test_data_inh_maxNinDecode = np.array([av_test_data_inh[iday][-1] for iday in range(numD)])
    av_test_data_allExc_maxNinDecode = np.array([av_test_data_allExc[iday][-1] for iday in range(numD)])
    av_test_data_exc_maxNinDecode = np.array([av_test_data_exc[iday][-1] for iday in range(numD)])
    
    # set ave across days and se
    aa = av_test_data_allExc_maxNinDecode.mean()
    ai = av_test_data_inh_maxNinDecode.mean()
    ae = av_test_data_exc_maxNinDecode.mean()
    
    sa = av_test_data_allExc_maxNinDecode.std() / np.sqrt(numD)
    si = av_test_data_inh_maxNinDecode.std() / np.sqrt(numD)
    se = av_test_data_exc_maxNinDecode.std() / np.sqrt(numD)
    
    ######### Plot
    plt.figure(figsize=(2,3))
    
    plt.errorbar(0, aa, sa, fmt='o', label='allExc', color='k')
    plt.errorbar(1, ai, si, fmt='o', label='inh', color='r')
    plt.errorbar(2, ae, se, fmt='o', label='exc', color='b')
    
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1)#, frameon=False) 
    #plt.xlabel('Population', fontsize=11)
    plt.ylabel('Classification accuracy (%)\n [-97 0] ms rel. choice', fontsize=11)
    plt.xlim([-.2,3-1+.2])
    plt.xticks(range(3), ['allExc','inh','exc'])
    ax = plt.gca()
    makeNicePlots(ax)
    yl = ax.get_ylim()
    plt.ylim([yl[0]-2, yl[1]])
    
   
    if savefigs:       
        if nanEnd==0:
            nN = 'dropNs_thDays%d_' %(thds)
        else:
            nN = 'addNs_thDays%d_' %(thds)
        if chAl==1:
            dd = 'chAl_avSeDays_time-1_allNs_'+nN+'addNsROC_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_avSeDays_time-1_allNs_'+nN+'addNsROC_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])         
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    



##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
#%%
if addNs_roc==0:    
    ##################################################################################################
    ############## Align class accur traces of all days to make a final average trace ##############
    ################################################################################################## 
      
    ##%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
    # By common eventI, we  mean the index on which all traces will be aligned.
    
    time_aligned, nPreMin, nPostMin = set_nprepost(av_test_data_inh, eventI_ds_allDays, mn_corr, thTrained, regressBins)
    # I commented the codes below and instead used the one above.
    
    """        
    nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
    for iday in range(numDays):
        nPost[iday] = (len(av_test_data_inh_ch[iday]) - eventI_ds_allDays[iday] - 1)
    
    nPreMin = min(eventI_ds_allDays) # number of frames before the common eventI, also the index of common eventI. 
    nPostMin = min(nPost)
    print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)
    
    
    #% Set the time array for the across-day aligned traces
    
    '''
    if corrTrained==0: # remember below has issues...
        # old svm files were ran using the old downsampling method
        a = -(np.asarray(frameLength*regressBins) * range(nPreMin+1)[::-1])
        b = (np.asarray(frameLength*regressBins) * range(1, nPostMin+1))
        time_aligned = np.concatenate((a,b))
    else:
    '''
    totLen = nPreMin + nPostMin +1
    time_aligned = set_time_al(totLen, min(eventI_allDays), lastTimeBinMissed)
    
    print time_aligned
    """
    
    
    #%% Align traces of all days on the common eventI (use nan for days with few trianed svm trials)
    # not including days with too few svm trained trials.
    
    def alTrace(trace, eventI_ds_allDays, nPreMin, nPostMin):    #same as alTrace in defFuns, except here the output is an array of size frs x days, there the output is a list of size days x frs
    
        trace_aligned = np.ones((nPreMin + nPostMin + 1, trace.shape[0])) + np.nan # frames x days, aligned on common eventI (equals nPreMin)     
        for iday in range(trace.shape[0]):
            if mn_corr[iday] >= thTrained: # dont include days with too few svm trained trials.
                trace_aligned[:, iday] = trace[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]    
        return trace_aligned
        
    
    av_test_data_inh_aligned = alTrace(av_test_data_inh, eventI_ds_allDays, nPreMin, nPostMin)
    av_test_shfl_inh_aligned = alTrace(av_test_shfl_inh, eventI_ds_allDays, nPreMin, nPostMin)
    av_test_chance_inh_aligned = alTrace(av_test_chance_inh, eventI_ds_allDays, nPreMin, nPostMin)    
    if do_excInhHalf==0:
        av_test_data_exc_aligned = alTrace(av_test_data_exc, eventI_ds_allDays, nPreMin, nPostMin)
        av_test_shfl_exc_aligned = alTrace(av_test_shfl_exc, eventI_ds_allDays, nPreMin, nPostMin)
        av_test_chance_exc_aligned = alTrace(av_test_chance_exc, eventI_ds_allDays, nPreMin, nPostMin)    
    av_test_data_allExc_aligned = alTrace(av_test_data_allExc, eventI_ds_allDays, nPreMin, nPostMin)
    av_test_shfl_allExc_aligned = alTrace(av_test_shfl_allExc, eventI_ds_allDays, nPreMin, nPostMin)
    av_test_chance_allExc_aligned = alTrace(av_test_chance_allExc, eventI_ds_allDays, nPreMin, nPostMin)
    if loadYtest:
        av_test_data_inh_easy_aligned = alTrace(av_test_data_inh_easy, eventI_ds_allDays, nPreMin, nPostMin)
        av_test_data_inh_hard_aligned = alTrace(av_test_data_inh_hard, eventI_ds_allDays, nPreMin, nPostMin)
        av_test_data_inh_medium_aligned = alTrace(av_test_data_inh_medium, eventI_ds_allDays, nPreMin, nPostMin)            
        av_test_shfl_inh_easy_aligned = alTrace(av_test_shfl_inh_easy, eventI_ds_allDays, nPreMin, nPostMin)
        av_test_shfl_inh_hard_aligned = alTrace(av_test_shfl_inh_hard, eventI_ds_allDays, nPreMin, nPostMin)
        av_test_shfl_inh_medium_aligned = alTrace(av_test_shfl_inh_medium, eventI_ds_allDays, nPreMin, nPostMin)

        av_test_data_exc_easy_aligned = alTrace(av_test_data_exc_easy, eventI_ds_allDays, nPreMin, nPostMin)
        av_test_data_exc_hard_aligned = alTrace(av_test_data_exc_hard, eventI_ds_allDays, nPreMin, nPostMin)
        av_test_data_exc_medium_aligned = alTrace(av_test_data_exc_medium, eventI_ds_allDays, nPreMin, nPostMin)            
        av_test_shfl_exc_easy_aligned = alTrace(av_test_shfl_exc_easy, eventI_ds_allDays, nPreMin, nPostMin)
        av_test_shfl_exc_hard_aligned = alTrace(av_test_shfl_exc_hard, eventI_ds_allDays, nPreMin, nPostMin)
        av_test_shfl_exc_medium_aligned = alTrace(av_test_shfl_exc_medium, eventI_ds_allDays, nPreMin, nPostMin)
        
        av_test_data_allExc_easy_aligned = alTrace(av_test_data_allExc_easy, eventI_ds_allDays, nPreMin, nPostMin)
        av_test_data_allExc_hard_aligned = alTrace(av_test_data_allExc_hard, eventI_ds_allDays, nPreMin, nPostMin)
        av_test_data_allExc_medium_aligned = alTrace(av_test_data_allExc_medium, eventI_ds_allDays, nPreMin, nPostMin)            
        av_test_shfl_allExc_easy_aligned = alTrace(av_test_shfl_allExc_easy, eventI_ds_allDays, nPreMin, nPostMin)
        av_test_shfl_allExc_hard_aligned = alTrace(av_test_shfl_allExc_hard, eventI_ds_allDays, nPreMin, nPostMin)
        av_test_shfl_allExc_medium_aligned = alTrace(av_test_shfl_allExc_medium, eventI_ds_allDays, nPreMin, nPostMin)

        
    ##%% Average and standard DEVIATION across days (each day includes the average class accuracy across samples.)
    
    av_av_test_data_inh_aligned = np.nanmean(av_test_data_inh_aligned, axis=1)
    sd_av_test_data_inh_aligned = np.nanstd(av_test_data_inh_aligned, axis=1)
    av_av_test_shfl_inh_aligned = np.nanmean(av_test_shfl_inh_aligned, axis=1)
    sd_av_test_shfl_inh_aligned = np.nanstd(av_test_shfl_inh_aligned, axis=1)
    av_av_test_chance_inh_aligned = np.nanmean(av_test_chance_inh_aligned, axis=1)
    sd_av_test_chance_inh_aligned = np.nanstd(av_test_chance_inh_aligned, axis=1)    
    if do_excInhHalf==0:
        av_av_test_data_exc_aligned = np.nanmean(av_test_data_exc_aligned, axis=1)
        sd_av_test_data_exc_aligned = np.nanstd(av_test_data_exc_aligned, axis=1)    
        av_av_test_shfl_exc_aligned = np.nanmean(av_test_shfl_exc_aligned, axis=1)
        sd_av_test_shfl_exc_aligned = np.nanstd(av_test_shfl_exc_aligned, axis=1)    
        av_av_test_chance_exc_aligned = np.nanmean(av_test_chance_exc_aligned, axis=1)
        sd_av_test_chance_exc_aligned = np.nanstd(av_test_chance_exc_aligned, axis=1)    
    av_av_test_data_allExc_aligned = np.nanmean(av_test_data_allExc_aligned, axis=1)
    sd_av_test_data_allExc_aligned = np.nanstd(av_test_data_allExc_aligned, axis=1)
    av_av_test_shfl_allExc_aligned = np.nanmean(av_test_shfl_allExc_aligned, axis=1)
    sd_av_test_shfl_allExc_aligned = np.nanstd(av_test_shfl_allExc_aligned, axis=1)
    av_av_test_chance_allExc_aligned = np.nanmean(av_test_chance_allExc_aligned, axis=1)
    sd_av_test_chance_allExc_aligned = np.nanstd(av_test_chance_allExc_aligned, axis=1)
    if loadYtest: # compute standard error across days
        av_av_test_data_inh_easy_aligned = np.nanmean(av_test_data_inh_easy_aligned, axis=1)
        sd_av_test_data_inh_easy_aligned = np.nanstd(av_test_data_inh_easy_aligned, axis=1) / np.sqrt(sum(nValidSamps_easy_inh >= thMinSamps))
        av_av_test_data_inh_hard_aligned = np.nanmean(av_test_data_inh_hard_aligned, axis=1)
        sd_av_test_data_inh_hard_aligned = np.nanstd(av_test_data_inh_hard_aligned, axis=1) / np.sqrt(sum(nValidSamps_hard_inh >= thMinSamps))
        av_av_test_data_inh_medium_aligned = np.nanmean(av_test_data_inh_medium_aligned, axis=1)
        sd_av_test_data_inh_medium_aligned = np.nanstd(av_test_data_inh_medium_aligned, axis=1) / np.sqrt(sum(nValidSamps_medium_inh >= thMinSamps))   
        av_av_test_data_exc_easy_aligned = np.nanmean(av_test_data_exc_easy_aligned, axis=1)
        sd_av_test_data_exc_easy_aligned = np.nanstd(av_test_data_exc_easy_aligned, axis=1) / np.sqrt(sum(nValidSamps_easy_exc >= thMinSamps/float(numSamples)))
        av_av_test_data_exc_hard_aligned = np.nanmean(av_test_data_exc_hard_aligned, axis=1)
        sd_av_test_data_exc_hard_aligned = np.nanstd(av_test_data_exc_hard_aligned, axis=1) / np.sqrt(sum(nValidSamps_hard_exc >= thMinSamps/float(numSamples)))
        av_av_test_data_exc_medium_aligned = np.nanmean(av_test_data_exc_medium_aligned, axis=1)
        sd_av_test_data_exc_medium_aligned = np.nanstd(av_test_data_exc_medium_aligned, axis=1) / np.sqrt(sum(nValidSamps_medium_exc >= thMinSamps/float(numSamples)))       
        av_av_test_data_allExc_easy_aligned = np.nanmean(av_test_data_allExc_easy_aligned, axis=1)
        sd_av_test_data_allExc_easy_aligned = np.nanstd(av_test_data_allExc_easy_aligned, axis=1) / np.sqrt(sum(nValidSamps_easy_allExc >= thMinSamps))
        av_av_test_data_allExc_hard_aligned = np.nanmean(av_test_data_allExc_hard_aligned, axis=1)
        sd_av_test_data_allExc_hard_aligned = np.nanstd(av_test_data_allExc_hard_aligned, axis=1) / np.sqrt(sum(nValidSamps_hard_allExc >= thMinSamps))
        av_av_test_data_allExc_medium_aligned = np.nanmean(av_test_data_allExc_medium_aligned, axis=1)
        sd_av_test_data_allExc_medium_aligned = np.nanstd(av_test_data_allExc_medium_aligned, axis=1) / np.sqrt(sum(nValidSamps_medium_allExc >= thMinSamps))        
        # shfl
        av_av_test_shfl_inh_easy_aligned = np.nanmean(av_test_shfl_inh_easy_aligned, axis=1)
        sd_av_test_shfl_inh_easy_aligned = np.nanstd(av_test_shfl_inh_easy_aligned, axis=1) / np.sqrt(sum(nValidSamps_easy_inh >= thMinSamps))
        av_av_test_shfl_inh_hard_aligned = np.nanmean(av_test_shfl_inh_hard_aligned, axis=1)
        sd_av_test_shfl_inh_hard_aligned = np.nanstd(av_test_shfl_inh_hard_aligned, axis=1) / np.sqrt(sum(nValidSamps_hard_inh >= thMinSamps))
        av_av_test_shfl_inh_medium_aligned = np.nanmean(av_test_shfl_inh_medium_aligned, axis=1)
        sd_av_test_shfl_inh_medium_aligned = np.nanstd(av_test_shfl_inh_medium_aligned, axis=1) / np.sqrt(sum(nValidSamps_medium_inh >= thMinSamps))   
        av_av_test_shfl_exc_easy_aligned = np.nanmean(av_test_shfl_exc_easy_aligned, axis=1)
        sd_av_test_shfl_exc_easy_aligned = np.nanstd(av_test_shfl_exc_easy_aligned, axis=1) / np.sqrt(sum(nValidSamps_easy_exc >= thMinSamps/float(numSamples)))
        av_av_test_shfl_exc_hard_aligned = np.nanmean(av_test_shfl_exc_hard_aligned, axis=1)
        sd_av_test_shfl_exc_hard_aligned = np.nanstd(av_test_shfl_exc_hard_aligned, axis=1) / np.sqrt(sum(nValidSamps_hard_exc >= thMinSamps/float(numSamples)))
        av_av_test_shfl_exc_medium_aligned = np.nanmean(av_test_shfl_exc_medium_aligned, axis=1)
        sd_av_test_shfl_exc_medium_aligned = np.nanstd(av_test_shfl_exc_medium_aligned, axis=1) / np.sqrt(sum(nValidSamps_medium_exc >= thMinSamps/float(numSamples)))       
        av_av_test_shfl_allExc_easy_aligned = np.nanmean(av_test_shfl_allExc_easy_aligned, axis=1)
        sd_av_test_shfl_allExc_easy_aligned = np.nanstd(av_test_shfl_allExc_easy_aligned, axis=1) / np.sqrt(sum(nValidSamps_easy_allExc >= thMinSamps))
        av_av_test_shfl_allExc_hard_aligned = np.nanmean(av_test_shfl_allExc_hard_aligned, axis=1)
        sd_av_test_shfl_allExc_hard_aligned = np.nanstd(av_test_shfl_allExc_hard_aligned, axis=1) / np.sqrt(sum(nValidSamps_hard_allExc >= thMinSamps))
        av_av_test_shfl_allExc_medium_aligned = np.nanmean(av_test_shfl_allExc_medium_aligned, axis=1)
        sd_av_test_shfl_allExc_medium_aligned = np.nanstd(av_test_shfl_allExc_medium_aligned, axis=1) / np.sqrt(sum(nValidSamps_medium_allExc >= thMinSamps))                
        
        
    #_,pcorrtrace0 = stats.ttest_1samp(av_l2_test_d_aligned.transpose(), 50) # p value of class accuracy being different from 50    
    if do_excInhHalf==0:
        _,pcorrtrace = stats.ttest_ind(av_test_data_exc_aligned.transpose(), av_test_data_inh_aligned.transpose(), nan_policy= 'omit') # p value of class accuracy being different from 50
    else:
        _,pcorrtrace = stats.ttest_ind(av_test_data_allExc_aligned.transpose(), av_test_data_inh_aligned.transpose(), nan_policy= 'omit') # p value of class accuracy being different from 50            
            
    
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%
    ######################## PLOTS ########################
    
    #%% Compare number of inh vs exc for each day
    
    if loadWeights:
        plt.figure(figsize=(4.5,3)) #plt.figure()
        
        plt.plot(numInh, label='inh', color='r')
        plt.plot(numAllexc, label=labAll, color='k')
        if doAllN:
            plt.title('average: inh=%d allN=%d inh/exc=%.2f' %(np.round(numInh.mean()), np.round(numAllexc.mean()), np.mean(numInh) / np.mean(numAllexc)))
        else:
            plt.title('average: inh=%d allExc=%d inh/exc=%.2f' %(np.round(numInh.mean()), np.round(numAllexc.mean()), np.mean(numInh) / np.mean(numAllexc)))    
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False) 
        plt.xlabel('Days')
        plt.ylabel('# Neurons')
        plt.xticks(range(len(days)), [days[i][0:6] for i in range(len(days))], rotation='vertical')
        
        makeNicePlots(plt.gca(),1)
        
        
        if savefigs:#% Save the figure
            if chAl==1:
                dd = 'chAl_numNeurons_days_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            else:
                dd = 'stAl_numNeurons_days_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' +nowStr
                
            d = os.path.join(svmdir+dnow)
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)
                    
            fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        
    
        
    #%% Plot class accur trace for all days (only use days with svm trained trials above thTrained) on top of each other
    
    if do_excInhHalf:
        Cols = 'm','b'
    else:
        Cols = 'r','k'
        
        
    plt.figure()
    
    for iday in range(len(days)):    
    
        if mn_corr[iday] >= thTrained:    
            totLen = len(av_test_data_inh[iday])
            # old downsampling method.
            '''
            if corrTrained==0: # remember below has problems...but if downsampling in svm_eachFrame was done in the old way, you have to use below.
                nPre = eventI_allDays[iday] # number of frames before the common eventI, also the index of common eventI. 
                nPost = (len(av_test_data_inh_ch[iday]) - eventI_allDays[iday] - 1)
                
                a = -(np.asarray(frameLength*regressBins) * range(nPre+1)[::-1])
                b = (np.asarray(frameLength*regressBins) * range(1, nPost+1))
                time_al = np.concatenate((a,b))
            else:
            '''    
            lastTimeBinMissed = 0
            time_al = set_time_al(totLen, eventI_allDays[iday], lastTimeBinMissed)    
            if len(time_al) != len(av_test_data_allExc[iday]):
                lastTimeBinMissed = 1
                time_al = set_time_al(totLen, eventI_allDays[iday], lastTimeBinMissed)    
                
    #        plt.figure()
            plt.subplot(221)
            if do_excInhHalf==0:
                plt.errorbar(time_al, av_test_data_exc[iday], yerr = sd_test_data_exc[iday], label='exc', color='b')
            plt.errorbar(time_al, av_test_data_inh[iday], yerr = sd_test_data_inh[iday], label=labInh, color=Cols[0])    
            plt.errorbar(time_al, av_test_data_allExc[iday], yerr = sd_test_data_allExc[iday], label=labAll, color=Cols[1])
    #        plt.title(days[iday])
        
            plt.subplot(222)
            if do_excInhHalf==0:
                plt.plot(time_al, av_test_shfl_exc[iday], label=' ', color='b')        
            plt.plot(time_al, av_test_shfl_inh[iday], label=' ', color=Cols[0])    
            plt.plot(time_al, av_test_shfl_allExc[iday], label=' ', color=Cols[1])        
            
            
            plt.subplot(223)
            if do_excInhHalf==0:
                h0,=plt.plot(time_al, av_test_chance_exc[iday], label='exc', color='b')        
            h1,=plt.plot(time_al, av_test_chance_inh[iday], label=labInh, color=Cols[0])    
            h2,=plt.plot(time_al, av_test_chance_allExc[iday], label=labAll, color=Cols[1])        
            
        #    plt.subplot(223)
        #    plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 
        #    plt.show()
    
    ##%%
    plt.subplot(221)
    plt.title('testing')
    if chAl==1:
        plt.xlabel('Time since choice onset (ms)', fontsize=13)
    else:
        plt.xlabel('Time since stim onset (ms)', fontsize=13)
    plt.ylabel('Classification accuracy (%)', fontsize=13)
    ax = plt.gca()
    makeNicePlots(ax,1)
    
    plt.subplot(222)
    plt.title('shfl')
    ax = plt.gca()
    makeNicePlots(ax,1)
    
    plt.subplot(223)
    plt.title('chance')
    ax = plt.gca()
    makeNicePlots(ax,1)
    if do_excInhHalf==0:
        plt.legend(handles=[h0,h1,h2], loc='center left', bbox_to_anchor=(1, .7), frameon=False) 
    else:
        plt.legend(handles=[h1,h2], loc='center left', bbox_to_anchor=(1, .7), frameon=False) 
    
    #plt.show()
    plt.subplots_adjust(hspace=0.75)
    
    
    ##%% Save the figure    
    if savefigs:
        if chAl==1:
            dd = 'chAl_allDays_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_allDays_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
     
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
    
        
    
       
    #%% Plot a few example sessions (exc, inh, allExc superimposed)      
    
    if plotSingleSessions:
        # fni17, 151015: example day, excShfl 3
        ####### pick an example exc shfl
        if do_excInhHalf==0:
            iexcshfl = rng.permutation(perClassErrorTest_data_exc_all[iday].shape[0])[0]
            print iexcshfl
            
            av_test_data_exc_1shfl = np.array([100-np.nanmean(perClassErrorTest_data_exc_all[iday][iexcshfl], axis=0,) for iday in range(numD)]) # numDays
            sd_test_data_exc_1shfl = np.array([np.nanstd(perClassErrorTest_data_exc_all[iday][iexcshfl], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  
            
            av_test_shfl_exc_1shfl = np.array([100-np.nanmean(perClassErrorTest_shfl_exc_all[iday][iexcshfl], axis=0) for iday in range(numD)]) # numDays
            sd_test_shfl_exc_1shfl = np.array([np.nanstd(perClassErrorTest_shfl_exc_all[iday][iexcshfl], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  
            
            av_test_chance_exc_1shfl = np.array([100-np.nanmean(perClassErrorTest_chance_exc_all[iday][iexcshfl], axis=0) for iday in range(numD)]) # numDays
            sd_test_chance_exc_1shfl = np.array([np.nanstd(perClassErrorTest_chance_exc_all[iday][iexcshfl], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  
        
        
        ####### plot
        for iday in range(len(days)): #[30,35]:#,43]:
            plt.figure() #(figsize=(4.5,3))
            totLen = len(av_test_data_inh[iday])
            time_al = set_time_al(totLen, eventI_allDays[iday], lastTimeBinMissed)    
            #        plt.errorbar(time_al, av_test_data_exc_ch[iday], yerr = sd_test_data_exc_ch[iday], label='exc', color='b')
                    
            #### testing data
            plt.subplot(221)
            # exc
            if do_excInhHalf==0:
                plt.fill_between(time_al, av_test_data_exc_1shfl[iday] - sd_test_data_exc_1shfl[iday], av_test_data_exc_1shfl[iday] + sd_test_data_exc_1shfl[iday], alpha=0.5, edgecolor='b', facecolor='b')
                plt.plot(time_al, av_test_data_exc_1shfl[iday], 'b', label='exc')
            # inh
            plt.fill_between(time_al, av_test_data_inh[iday] - sd_test_data_inh[iday], av_test_data_inh[iday] + sd_test_data_inh[iday], alpha=0.5, edgecolor=Cols[0], facecolor=Cols[0])
            plt.plot(time_al, av_test_data_inh[iday], color=Cols[0], label=labInh)
            # allExc
            plt.fill_between(time_al, av_test_data_allExc[iday] - sd_test_data_allExc[iday], av_test_data_allExc[iday] + sd_test_data_allExc[iday], alpha=0.5, edgecolor=Cols[1], facecolor=Cols[1])
            plt.plot(time_al, av_test_data_allExc[iday], color=Cols[1], label=labAll)
            
            if chAl==1:
                plt.xlabel('Time relative to choice onset (ms)', fontsize=11)
            else:
                plt.xlabel('Time relative to stim onset (ms)', fontsize=11)
            plt.ylabel('Classification accuracy (%)', fontsize=11)
            if superimpose==0:    
                plt.title('Testing data')
            #plt.title('SVM trained on non-overlapping %.2f ms windows' %(regressBins*frameLength), fontsize=13)
            #plt.legend()
            ax = plt.gca()
            makeNicePlots(ax,1,1)
            # Plot a dot for significant time points
            ymin, ymax = ax.get_ylim()
            #pp = pcorrtrace+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
            #plt.plot(time_al, pp, color='k')
            
            
            #### shfl
            
            # exc
            if do_excInhHalf==0:
                plt.fill_between(time_al, av_test_shfl_exc_1shfl[iday] - sd_test_shfl_exc_1shfl[iday], av_test_shfl_exc_1shfl[iday] + sd_test_shfl_exc_1shfl[iday], alpha=0.3, edgecolor='b', facecolor='b')
                plt.plot(time_al, av_test_shfl_exc_1shfl[iday], 'b')
            # inh
            plt.fill_between(time_al, av_test_shfl_inh[iday] - sd_test_shfl_inh[iday], av_test_shfl_inh[iday] + sd_test_shfl_inh[iday], alpha=0.3, edgecolor=Cols[0], facecolor=Cols[0])
            plt.plot(time_al, av_test_shfl_inh[iday], color=Cols[0])
            # allExc
            plt.fill_between(time_al, av_test_shfl_allExc[iday] - sd_test_shfl_allExc[iday], av_test_shfl_allExc[iday] + sd_test_shfl_allExc[iday], alpha=0.3, edgecolor=Cols[1], facecolor=Cols[1])
            plt.plot(time_al, av_test_shfl_allExc[iday], color=Cols[1])
            
            ax = plt.gca()
            makeNicePlots(ax,1,1)
            
            
            plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False) 
        
        
            #xmin, xmax = ax.get_xlim()
            #plt.xlim([-1400,500])
            
            ##%% Save the figure    
            if savefigs:            
                if do_excInhHalf==0:
                    n0 = '_exShfl' + str(iexcshfl)
                else:
                    n0 = ''
                    
                if chAl==1:
                    dd = 'chAl_day' + days[iday][0:6] + n0 + '_' + nowStr
                else:
                    dd = 'stAl_day' + days[iday][0:6] + n0 + '_' + nowStr
            
                if superimpose==1:        
                    dd = dd+'_sup'
                    
                d = os.path.join(svmdir+dnow+'CA_exampDay')
                if not os.path.exists(d):
                    print 'creating folder'
                    os.makedirs(d)
                        
                fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])
             
                plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
            #    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+'svg')
            #    plt.savefig(fign, dpi=300, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        
        
    
        
    #%%#####################%% Plot the average of aligned traces across all days (exc, inh, allExc superimposed)
    # only days with enough svm trained trials are used.
           
    plt.figure() #(figsize=(4.5,3))
    
    #### testing data
    plt.subplot(221)
    # exc
    if do_excInhHalf==0:
        plt.fill_between(time_aligned, av_av_test_data_exc_aligned - sd_av_test_data_exc_aligned, av_av_test_data_exc_aligned + sd_av_test_data_exc_aligned, alpha=0.5, edgecolor='b', facecolor='b')
        plt.plot(time_aligned, av_av_test_data_exc_aligned, 'b', label='exc')
    # inh
    plt.fill_between(time_aligned, av_av_test_data_inh_aligned - sd_av_test_data_inh_aligned, av_av_test_data_inh_aligned + sd_av_test_data_inh_aligned, alpha=0.5, edgecolor=Cols[0], facecolor=Cols[0])
    plt.plot(time_aligned, av_av_test_data_inh_aligned, color=Cols[0], label=labInh)
    # allExc
    plt.fill_between(time_aligned, av_av_test_data_allExc_aligned - sd_av_test_data_allExc_aligned, av_av_test_data_allExc_aligned + sd_av_test_data_allExc_aligned, alpha=0.5, edgecolor=Cols[1], facecolor=Cols[1])
    plt.plot(time_aligned, av_av_test_data_allExc_aligned, color=Cols[1], label=labAll)
    
    if chAl==1:
        plt.xlabel('Time relative to choice onset (ms)', fontsize=11)
    else:
        plt.xlabel('Time relative to stim onset (ms)', fontsize=11)
    plt.ylabel('Classification accuracy (%)', fontsize=11)
    ax = plt.gca()
    if superimpose==0:    
        plt.title('Testing data')
        #plt.title('SVM trained on non-overlapping %.2f ms windows' %(regressBins*frameLength), fontsize=13)
        #plt.legend()
        makeNicePlots(ax,1,1)
    # Plot a dot for significant time points
    ymin, ymax = ax.get_ylim()
    pp = pcorrtrace+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
    plt.plot(time_aligned, pp, color=Cols[1])
    
    
    #### shfl
    if superimpose==0:
        plt.subplot(222)
    # exc
    if do_excInhHalf==0:
        plt.fill_between(time_aligned, av_av_test_shfl_exc_aligned - sd_av_test_shfl_exc_aligned, av_av_test_shfl_exc_aligned + sd_av_test_shfl_exc_aligned, alpha=0.3, edgecolor='b', facecolor='b')
        plt.plot(time_aligned, av_av_test_shfl_exc_aligned, 'b')
    # inh
    plt.fill_between(time_aligned, av_av_test_shfl_inh_aligned - sd_av_test_shfl_inh_aligned, av_av_test_shfl_inh_aligned + sd_av_test_shfl_inh_aligned, alpha=0.3, edgecolor=Cols[0], facecolor=Cols[0])
    plt.plot(time_aligned, av_av_test_shfl_inh_aligned, color=Cols[0])
    # allExc
    plt.fill_between(time_aligned, av_av_test_shfl_allExc_aligned - sd_av_test_shfl_allExc_aligned, av_av_test_shfl_allExc_aligned + sd_av_test_shfl_allExc_aligned, alpha=0.3, edgecolor=Cols[1], facecolor=Cols[1])
    plt.plot(time_aligned, av_av_test_shfl_allExc_aligned, color=Cols[1])
    if superimpose==0:    
        plt.title('Shuffled Y')
    ax = plt.gca()
    makeNicePlots(ax,1,1)
    
    
    #### chance
    if superimpose==0:    
        plt.subplot(223)
        # exc
        if do_excInhHalf==0:
            plt.fill_between(time_aligned, av_av_test_chance_exc_aligned - sd_av_test_chance_exc_aligned, av_av_test_chance_exc_aligned + sd_av_test_chance_exc_aligned, alpha=0.5, edgecolor='b', facecolor='b')
            plt.plot(time_aligned, av_av_test_chance_exc_aligned, 'b')
        # inh
        plt.fill_between(time_aligned, av_av_test_chance_inh_aligned - sd_av_test_chance_inh_aligned, av_av_test_chance_inh_aligned + sd_av_test_chance_inh_aligned, alpha=0.5, edgecolor=Cols[0], facecolor=Cols[0])
        plt.plot(time_aligned, av_av_test_chance_inh_aligned, color=Cols[0])
        # allExc
        plt.fill_between(time_aligned, av_av_test_chance_allExc_aligned - sd_av_test_chance_allExc_aligned, av_av_test_chance_allExc_aligned + sd_av_test_chance_allExc_aligned, alpha=0.5, edgecolor=Cols[1], facecolor=Cols[1])
        plt.plot(time_aligned, av_av_test_chance_allExc_aligned, color=Cols[1])
        plt.title('Chance Y')
        ax = plt.gca()
        makeNicePlots(ax,1,1)
    
        plt.subplots_adjust(hspace=0.65)
        
        
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False) 
    
    
    #xmin, xmax = ax.get_xlim()
    #plt.xlim([-1400,500])
    
    ##%% Save the figure    
    if savefigs:
        if chAl==1:
            dd = 'chAl_aveSdDays_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_aveSdDays_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
    
        if superimpose==1:        
            dd = dd+'_sup'
            
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
     
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
    #    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+'svg')
    #    plt.savefig(fign, dpi=300, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
    
    
    
    


    #%% easy, hard, medium: Average class accuracy for testing trials of different stimulus strength: easy, hard, medium
    
    ######################%% Plot the average of aligned traces across all days (exc, inh, allExc superimposed)
    # only days with enough svm trained trials are used.
            
    if loadYtest:            
               
        plt.figure(figsize=(8,5))
        
        #### testing data
        plt.subplot(221) #### exc 
        # easy    
        plt.fill_between(time_aligned, av_av_test_data_exc_easy_aligned - sd_av_test_data_exc_easy_aligned, av_av_test_data_exc_easy_aligned + sd_av_test_data_exc_easy_aligned, alpha=0.5, edgecolor='b', facecolor='b')
        plt.plot(time_aligned, av_av_test_data_exc_easy_aligned, 'b', label='easy')
        # hard
        plt.fill_between(time_aligned, av_av_test_data_exc_hard_aligned - sd_av_test_data_exc_hard_aligned, av_av_test_data_exc_hard_aligned + sd_av_test_data_exc_hard_aligned, alpha=0.5, edgecolor='b', facecolor='b')
        plt.plot(time_aligned, av_av_test_data_exc_hard_aligned, 'b', label='hard', linestyle='-.')
        # medium
        plt.fill_between(time_aligned, av_av_test_data_exc_medium_aligned - sd_av_test_data_exc_medium_aligned, av_av_test_data_exc_medium_aligned + sd_av_test_data_exc_medium_aligned, alpha=0.5, edgecolor='b', facecolor='b')
        plt.plot(time_aligned, av_av_test_data_exc_medium_aligned, 'b', label='medium', linestyle=':')
        if chAl==1:
            plt.xlabel('Time relative to choice onset (ms)', fontsize=11)
        else:
            plt.xlabel('Time relative to stim onset (ms)', fontsize=11)
        plt.ylabel('Class accuracy (%)', fontsize=11)
        makeNicePlots(plt.gca(),1,1)
        plt.title('exc')
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)         
        
            
        plt.subplot(223) #### inh
        # easy    
        plt.fill_between(time_aligned, av_av_test_data_inh_easy_aligned - sd_av_test_data_inh_easy_aligned, av_av_test_data_inh_easy_aligned + sd_av_test_data_inh_easy_aligned, alpha=0.5, edgecolor=Cols[0], facecolor=Cols[0])
        plt.plot(time_aligned, av_av_test_data_inh_easy_aligned, color=Cols[0], label=labInh)
        # hard
        plt.fill_between(time_aligned, av_av_test_data_inh_hard_aligned - sd_av_test_data_inh_hard_aligned, av_av_test_data_inh_hard_aligned + sd_av_test_data_inh_hard_aligned, alpha=0.5, edgecolor=Cols[0], facecolor=Cols[0])
        plt.plot(time_aligned, av_av_test_data_inh_hard_aligned, color=Cols[0], label=labInh, linestyle='-.')
        # medium
        plt.fill_between(time_aligned, av_av_test_data_inh_medium_aligned - sd_av_test_data_inh_medium_aligned, av_av_test_data_inh_medium_aligned + sd_av_test_data_inh_medium_aligned, alpha=0.5, edgecolor=Cols[0], facecolor=Cols[0])
        plt.plot(time_aligned, av_av_test_data_inh_medium_aligned, color=Cols[0], label=labInh, linestyle=':')
        if chAl==1:
            plt.xlabel('Time relative to choice onset (ms)', fontsize=11)
        else:
            plt.xlabel('Time relative to stim onset (ms)', fontsize=11)
        plt.ylabel('Class accuracy (%)', fontsize=11)
        makeNicePlots(plt.gca(),1,1)
        plt.title('inh')
        
        
        plt.subplot(222) #### allExc 
        # easy    
        plt.fill_between(time_aligned, av_av_test_data_allExc_easy_aligned - sd_av_test_data_allExc_easy_aligned, av_av_test_data_allExc_easy_aligned + sd_av_test_data_allExc_easy_aligned, alpha=0.5, edgecolor=Cols[1], facecolor=Cols[1])
        plt.plot(time_aligned, av_av_test_data_allExc_easy_aligned, color=Cols[1], label=labAll)
        # hard
        plt.fill_between(time_aligned, av_av_test_data_allExc_hard_aligned - sd_av_test_data_allExc_hard_aligned, av_av_test_data_allExc_hard_aligned + sd_av_test_data_allExc_hard_aligned, alpha=0.5, edgecolor=Cols[1], facecolor=Cols[1])
        plt.plot(time_aligned, av_av_test_data_allExc_hard_aligned, color=Cols[1], label=labAll, linestyle='-.')
        # medium
        plt.fill_between(time_aligned, av_av_test_data_allExc_medium_aligned - sd_av_test_data_allExc_medium_aligned, av_av_test_data_allExc_medium_aligned + sd_av_test_data_allExc_medium_aligned, alpha=0.5, edgecolor=Cols[1], facecolor=Cols[1])
        plt.plot(time_aligned, av_av_test_data_allExc_medium_aligned, color=Cols[1], label=labAll, linestyle=':')       
        if chAl==1:
            plt.xlabel('Time relative to choice onset (ms)', fontsize=11)
        else:
            plt.xlabel('Time relative to stim onset (ms)', fontsize=11)
        plt.ylabel('Class accuracy (%)', fontsize=11)
        makeNicePlots(plt.gca(),1,1)
        plt.title('allExc')
        # Plot a dot for significant time points
#        ymin, ymax = ax.get_ylim()
#        pp = pcorrtrace+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
#        plt.plot(time_aligned, pp, color=Cols[1])
        
        
        
        #### shfl
        plt.subplot(221) #### exc 
        # easy    
#        plt.fill_between(time_aligned, av_av_test_shfl_exc_easy_aligned - sd_av_test_shfl_exc_easy_aligned, av_av_test_shfl_exc_easy_aligned + sd_av_test_shfl_exc_easy_aligned, alpha=0.5, edgecolor='b', facecolor='b')
        plt.plot(time_aligned, av_av_test_shfl_exc_easy_aligned, 'b', label='exc')
        # hard
#        plt.fill_between(time_aligned, av_av_test_shfl_exc_hard_aligned - sd_av_test_shfl_exc_hard_aligned, av_av_test_shfl_exc_hard_aligned + sd_av_test_shfl_exc_hard_aligned, alpha=0.5, edgecolor='b', facecolor='b')
        plt.plot(time_aligned, av_av_test_shfl_exc_hard_aligned, 'b', label='exc', linestyle='-.')
        # medium
#        plt.fill_between(time_aligned, av_av_test_shfl_exc_medium_aligned - sd_av_test_shfl_exc_medium_aligned, av_av_test_shfl_exc_medium_aligned + sd_av_test_shfl_exc_medium_aligned, alpha=0.5, edgecolor='b', facecolor='b')
        plt.plot(time_aligned, av_av_test_shfl_exc_medium_aligned, 'b', label='exc', linestyle=':')
        
            
        plt.subplot(223) #### inh
        # easy    
#        plt.fill_between(time_aligned, av_av_test_shfl_inh_easy_aligned - sd_av_test_shfl_inh_easy_aligned, av_av_test_shfl_inh_easy_aligned + sd_av_test_shfl_inh_easy_aligned, alpha=0.5, edgecolor=Cols[0], facecolor=Cols[0])
        plt.plot(time_aligned, av_av_test_shfl_inh_easy_aligned, color=Cols[0], label=labInh)
        # hard
#        plt.fill_between(time_aligned, av_av_test_shfl_inh_hard_aligned - sd_av_test_shfl_inh_hard_aligned, av_av_test_shfl_inh_hard_aligned + sd_av_test_shfl_inh_hard_aligned, alpha=0.5, edgecolor=Cols[0], facecolor=Cols[0])
        plt.plot(time_aligned, av_av_test_shfl_inh_hard_aligned, color=Cols[0], label=labInh, linestyle='-.')
        # medium
#        plt.fill_between(time_aligned, av_av_test_shfl_inh_medium_aligned - sd_av_test_shfl_inh_medium_aligned, av_av_test_shfl_inh_medium_aligned + sd_av_test_shfl_inh_medium_aligned, alpha=0.5, edgecolor=Cols[0], facecolor=Cols[0])
        plt.plot(time_aligned, av_av_test_shfl_inh_medium_aligned, color=Cols[0], label=labInh, linestyle=':')
        
        
        plt.subplot(222) #### allExc 
        # easy    
#        plt.fill_between(time_aligned, av_av_test_shfl_allExc_easy_aligned - sd_av_test_shfl_allExc_easy_aligned, av_av_test_shfl_allExc_easy_aligned + sd_av_test_shfl_allExc_easy_aligned, alpha=0.5, edgecolor=Cols[1], facecolor=Cols[1])
        plt.plot(time_aligned, av_av_test_shfl_allExc_easy_aligned, color=Cols[1], label=labAll)
        # hard
#        plt.fill_between(time_aligned, av_av_test_shfl_allExc_hard_aligned - sd_av_test_shfl_allExc_hard_aligned, av_av_test_shfl_allExc_hard_aligned + sd_av_test_shfl_allExc_hard_aligned, alpha=0.5, edgecolor=Cols[1], facecolor=Cols[1])
        plt.plot(time_aligned, av_av_test_shfl_allExc_hard_aligned, color=Cols[1], label=labAll, linestyle='-.')
        # medium
#        plt.fill_between(time_aligned, av_av_test_shfl_allExc_medium_aligned - sd_av_test_shfl_allExc_medium_aligned, av_av_test_shfl_allExc_medium_aligned + sd_av_test_shfl_allExc_medium_aligned, alpha=0.5, edgecolor=Cols[1], facecolor=Cols[1])
        plt.plot(time_aligned, av_av_test_shfl_allExc_medium_aligned, color=Cols[1], label=labAll, linestyle=':')       
        
        
        
        plt.subplots_adjust(wspace=1.4, hspace=.8)  
        #xmin, xmax = ax.get_xlim()
        #plt.xlim([-1400,500])
        
        ##%% Save the figure    
        if savefigs:
            if chAl==1:
                dd = 'chAl_aveSeDays_easyHardMedTrs_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            else:
                dd = 'stAl_aveSeDays_easyHardMedTrs_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        
                
            d = os.path.join(svmdir+dnow)
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)
                    
            fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
         
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        #    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+'svg')
        #    plt.savefig(fign, dpi=300, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        
        
        
    
    
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%% # For each day plot dist of event times (match it with class accur traces). Do it for both stim-aligned and choice-aligned data 
    # For this section you need to run eventTimesDist.py; also run the codes above once with chAl=0 and once with chAl=1.
    """
    if 'eventI_allDays_ch' in locals() and 'eventI_allDays_st' in locals():
        execfile("eventTimesDist.py")
        
        
        for iday in range(len(days)):    
        
            #%%
            # relative to stim onset
            timeStimOffset0_all_relOn = np.nanmean(timeStimOffset0_all[iday] - timeStimOnset_all[iday])
            timeStimOffset_all_relOn = np.nanmean(timeStimOffset_all[iday] - timeStimOnset_all[iday])
            timeCommitCL_CR_Gotone_all_relOn = np.nanmean(timeCommitCL_CR_Gotone_all[iday] - timeStimOnset_all[iday])
            time1stSideTry_all_relOn = np.nanmean(time1stSideTry_all[iday] - timeStimOnset_all[iday])
        
            # relative to choice onset
            timeStimOnset_all_relCh = np.nanmean(timeStimOnset_all[iday] - time1stSideTry_all[iday])    
            timeStimOffset0_all_relCh = np.nanmean(timeStimOffset0_all[iday] - time1stSideTry_all[iday])
            timeStimOffset_all_relCh = np.nanmean(timeStimOffset_all[iday] - time1stSideTry_all[iday])
            timeCommitCL_CR_Gotone_all_relCh = np.nanmean(timeCommitCL_CR_Gotone_all[iday] - time1stSideTry_all[iday])
            
            
            #%% Plot class accur trace for each day
            
            plt.figure()
            
            ######## stAl
            colors = 'k','r','b','m'
            nPre = eventI_allDays_st[iday] # number of frames before the common eventI, also the index of common eventI. 
            nPost = (len(av_l2_test_d_st[iday]) - eventI_allDays_st[iday] - 1)
            
            a = -(np.asarray(frameLength*regressBins) * range(nPre+1)[::-1])
            b = (np.asarray(frameLength*regressBins) * range(1, nPost+1))
            time_al = np.concatenate((a,b))
            
            plt.subplot(223)
            plt.errorbar(time_al, av_l2_test_d_st[iday], yerr = sd_l2_test_d_st[iday])    
        #    plt.title(days[iday])
            #plt.errorbar(range(len(av_l2_test_d[iday])), av_l2_test_d[iday], yerr = sd_l2_test_d[iday])
        
            # mark event times relative to stim onset
            plt.plot([0, 0], [50, 100], color='g')
            plt.plot([timeStimOffset0_all_relOn, timeStimOffset0_all_relOn], [50, 100], color=colors[0])
            plt.plot([timeStimOffset_all_relOn, timeStimOffset_all_relOn], [50, 100], color=colors[1])
            plt.plot([timeCommitCL_CR_Gotone_all_relOn, timeCommitCL_CR_Gotone_all_relOn], [50, 100], color=colors[2])
            plt.plot([time1stSideTry_all_relOn, time1stSideTry_all_relOn], [50, 100], color=colors[3])
            plt.xlabel('Time relative to stim onset (ms)')
            plt.ylabel('Classification accuracy (%)') #, fontsize=13)
            
            ax = plt.gca(); xl = ax.get_xlim(); makeNicePlots(ax,1)
            
            
            ###### Plot hist of event times
            stimOffset0_st = timeStimOffset0_all[iday] - timeStimOnset_all[iday]
            stimOffset_st = timeStimOffset_all[iday] - timeStimOnset_all[iday]
            goTone_st = timeCommitCL_CR_Gotone_all[iday] - timeStimOnset_all[iday]
            sideTry_st = time1stSideTry_all[iday] - timeStimOnset_all[iday]
        
            stimOffset0_st = stimOffset0_st[~np.isnan(stimOffset0_st)]
            stimOffset_st = stimOffset_st[~np.isnan(stimOffset_st)]
            goTone_st = goTone_st[~np.isnan(goTone_st)]
            sideTry_st = sideTry_st[~np.isnan(sideTry_st)]
            
            labs = 'stimOffset_1rep', 'stimOffset', 'goTone', '1stSideTry'
            a = stimOffset0_st, stimOffset_st, goTone_st, sideTry_st
            binEvery = 100
            
            bn = np.arange(np.min(np.concatenate((a))), np.max(np.concatenate((a))), binEvery)
            bn[-1] = np.max(a) # unlike digitize, histogram doesn't count the right most value
            
            plt.subplot(221)
            # set hists
            for i in range(len(a)):
                hist, bin_edges = np.histogram(a[i], bins=bn)
            #    hist = hist/float(np.sum(hist))     # use this if you want to get fraction of trials instead of number of trials
                hb = plt.bar(bin_edges[0:-1], hist, binEvery, alpha=.4, color=colors[i], label=labs[i]) 
            
        #    plt.xlabel('Time relative to stim onset (ms)')
            plt.ylabel('Number of trials')
            plt.legend(handles=[hb], loc='center left', bbox_to_anchor=(.7, .7)) 
            
            plt.title(days[iday])
            plt.xlim(xl)    
            ax = plt.gca(); makeNicePlots(ax,1)
        
        
        
            #%% chAl
            colors = 'g','k','r','b'
            nPre = eventI_allDays_ch[iday] # number of frames before the common eventI, also the index of common eventI. 
            nPost = (len(av_l2_test_d_ch[iday]) - eventI_allDays_ch[iday] - 1)
            
            a = -(np.asarray(frameLength*regressBins) * range(nPre+1)[::-1])
            b = (np.asarray(frameLength*regressBins) * range(1, nPost+1))
            time_al = np.concatenate((a,b))
            
            plt.subplot(224)
            plt.errorbar(time_al, av_l2_test_d_ch[iday], yerr = sd_l2_test_d_ch[iday])    
        #    plt.title(days[iday])
        
            # mark event times relative to choice onset
            plt.plot([timeStimOnset_all_relCh, timeStimOnset_all_relCh], [50, 100], color=colors[0])    
            plt.plot([timeStimOffset0_all_relCh, timeStimOffset0_all_relCh], [50, 100], color=colors[1])
            plt.plot([timeStimOffset_all_relCh, timeStimOffset_all_relCh], [50, 100], color=colors[2])
            plt.plot([timeCommitCL_CR_Gotone_all_relCh, timeCommitCL_CR_Gotone_all_relCh], [50, 100], color=colors[3])
            plt.plot([0, 0], [50, 100], color='m')
            plt.xlabel('Time relative to choice onset (ms)')
            ax = plt.gca(); xl = ax.get_xlim(); makeNicePlots(ax,1)
            
            
            ###### Plot hist of event times
            stimOnset_ch = timeStimOnset_all[iday] - time1stSideTry_all[iday]
            stimOffset0_ch = timeStimOffset0_all[iday] - time1stSideTry_all[iday]
            stimOffset_ch = timeStimOffset_all[iday] - time1stSideTry_all[iday]
            goTone_ch = timeCommitCL_CR_Gotone_all[iday] - time1stSideTry_all[iday]
            
            stimOnset_ch = stimOnset_ch[~np.isnan(stimOnset_ch)]
            stimOffset0_ch = stimOffset0_ch[~np.isnan(stimOffset0_ch)]
            stimOffset_ch = stimOffset_ch[~np.isnan(stimOffset_ch)]
            goTone_ch = goTone_ch[~np.isnan(goTone_ch)]    
            
            labs = 'stimOnset','stimOffset_1rep', 'stimOffset', 'goTone'
            a = stimOnset_ch, stimOffset0_ch, stimOffset_ch, goTone_ch
            binEvery = 100
            
            bn = np.arange(np.min(np.concatenate((a))), np.max(np.concatenate((a))), binEvery)
            bn[-1] = np.max(a) # unlike digitize, histogram doesn't count the right most value
            
            plt.subplot(222)
            # set hists
            for i in range(len(a)):
                hist, bin_edges = np.histogram(a[i], bins=bn)
            #    hist = hist/float(np.sum(hist))     # use this if you want to get fraction of trials instead of number of trials
                plt.bar(bin_edges[0:-1], hist, binEvery, alpha=.4, color=colors[i], label=labs[i]) 
            
        #    plt.xlabel('Time relative to choice onset (ms)')
        #    plt.ylabel('Number of trials')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 
            
            plt.xlim(xl)    
            ax = plt.gca(); makeNicePlots(ax,1)
            
            plt.subplots_adjust(wspace=1, hspace=.5)
            
            
            '''
            plt.subplot(223)
            a = timeStimOffset0_all[iday] - timeStimOnset_all[iday]
            plt.hist(a[~np.isnan(a)], color='b')
            
            a = timeStimOffset_all[iday] - timeStimOnset_all[iday]
            plt.hist(a[~np.isnan(a)], color='r')
        
            a = timeCommitCL_CR_Gotone_all[iday] - timeStimOnset_all[iday]
            plt.hist(a[~np.isnan(a)], color='k')
        
            a = time1stSideTry_all[iday] - timeStimOnset_all[iday]
            plt.hist(a[~np.isnan(a)], color='m')
            
            plt.xlabel('Time relative to stim onset (ms)')
            plt.ylabel('# trials')           
            '''
        
            
        ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
