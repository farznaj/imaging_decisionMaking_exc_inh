# -*- coding: utf-8 -*-
"""
# results get saved in /home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/SVM/excInh_trainDecoder_eachFrame/

Summary of all mice: 
Plots class accuracy of frame -1 (outputs of file svm_eachFrame.py)
 ... svm trained to decode choice on choice-aligned or stimulus-aligned traces.
 
 
Remember for fni18 there are 2 svm_eachFrame mat files, the earlier file is using all trials (unequal HR, LR, like how you've done all your analysis). 
The later mat file is with equal number of hr and lr trials (subselecting trials)... this helped with 151209 class accur trace which was weird in the earlier mat file.
 
Created on Sun Mar 12 15:12:29 2017
@author: farznaj
"""


#%%
mice = 'fni16', 'fni17', 'fni18', 'fni19' # if you want to use only one mouse, make sure you put comma at the end; eg. mice = 'fni19',

do_excInhHalf = 0 # 0: Load vars for inh,exc,allExc, 1: Load exc,inh SVM vars for excInhHalf (ie when the population consists of half exc and half inh) and allExc2inhSize (ie when populatin consists of allExc but same size as 2*inh size)
decodeStimCateg = 0 #1
noExtraStimDayAnalysis = 0 # 1 # if 1, exclude days with extraStim from analysis
use_both_remCorr_testing = [1,0,0] # if decodeStimCate = 1, decide which one (testing, remCorr, or Both) you want to use for the rest of this code
shflTrsEachNeuron = 0  # Set to 0 for normal SVM training. # Shuffle trials in X_svm (for each neuron independently) to break correlations between neurons in each trial.

savefigs = 1

ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. #chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces.  #chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
trialHistAnalysis = 0 # 0
testIncorr = 1 # 1 # set to 0 for trialHistoryAnalysis # load svm file that includes testing corr as well as testing incorr trials.
corrTrained = 1 # 1 # set to 0 for trialHistoryAnalysis 


doAllN = 1 # plot allN, instead of allExc
time2an = -1; # relative to eventI, look at classErr in what time stamp.
thTrained = 10 # number of trials of each class used for svm training, min acceptable value to include a day in analysis
doIncorr = 0

loadWeights = 0
num2AnInhAllexcEqexc = 3 # if 3, all 3 types (inh, allExc, exc) will be analyzed. If 2, inh and allExc will be analyzed. if 1, inh will be analyzed. # if you have all svm files saved, set it to 3.
iTiFlg = 2 # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
useEqualTrNums = 1

import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.
chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]

if do_excInhHalf:
    doAllN = 0
    labInh = 'InhExcHalf'
    num2AnInhAllexcEqexc = 2
else:
    labInh = 'inh'
    
if doAllN==1:
    labAll = 'allN'
else:
    labAll = 'allExc'    
    
dnow0 = '/excInh_trainDecoder_eachFrame/'
if do_excInhHalf:
    dnow0 = dnow0+'InhExcHalf/'
#dnow0 = '/excInh_trainDecoder_eachFrame/frame'+str(time2an)+'/'
#loadInhAllexcEqexc = 1 # if 1, load 2nd run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc separately; also for all days the new vector inhRois_pix was used (not the old inhRois)        
from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')

#if doAllN==1:
#    smallestC = 0 # Identify best c: if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
#    if smallestC==1:
#        print 'bestc = smallest c whose cv error is less than 1se of min cv error'
#    else:
#        print 'bestc = c that gives min cv error'


execfile("defFuns.py") # Define common funcitons
loadYtest = 0

if decodeStimCateg:
    corrTrained=0
    
if corrTrained==0:
    corrn = 'allOutcomeTrained_'
else:
    corrn = 'corrTrained_'
    
if decodeStimCateg:
    corrn = 'decodeStimCateg_' + corrn

if noExtraStimDayAnalysis:
    corrn = corrn + 'noExtraStim_'
    
    

#%%
av_test_data_inh_allMice = []
sd_test_data_inh_allMice = []
av_test_data_exc_allMice = []
sd_test_data_exc_allMice = []
av_test_data_allExc_allMice = []
sd_test_data_allExc_allMice = []

av_test_shfl_inh_allMice = []
sd_test_shfl_inh_allMice = []
av_test_shfl_exc_allMice = []
sd_test_shfl_exc_allMice = []
av_test_shfl_allExc_allMice = []
sd_test_shfl_allExc_allMice = []

av_test_chance_inh_allMice = []
sd_test_chance_inh_allMice = []
av_test_chance_exc_allMice = []
sd_test_chance_exc_allMice = []
av_test_chance_allExc_allMice = []
sd_test_chance_allExc_allMice = []

corr_hr_lr_allMice = []
numGoodDays_allMice = np.full(np.shape(mice), 0.)

#%% Loop through mice
# im = 0
for im in range(len(mice)):
     
    #%%       
    mousename = mice[im] # mousename = 'fni16' #'fni17'
    if mousename == 'fni18': #set one of the following to 1:
        allDays = 1# all 7 days will be used (last 3 days have z motion!)
        noZmotionDays = 0 # 4 days that dont have z motion will be used.
        noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!
    elif mousename == 'fni19':    
        allDays = 1
        noExtraStimDays = 0   
    else:
        import numpy as np
        allDays = np.nan
        noZmotionDays = np.nan
        noZmotionDays_strict = np.nan
        noExtraStimDays = np.nan

    
#    execfile("svm_plots_setVars_n.py")      
    days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays, loadYtest, decodeStimCateg)
#    execfile("svm_plots_setVars.py")  
    # remove some days if needed:
#    if mousename == 'fni17':
#        days = np.delete(days, [10])
    
    #%% 
#    if loadInhAllexcEqexc==1:
    dnow = '/excInh_trainDecoder_eachFrame/'+mousename+'/'
    if do_excInhHalf:
        dnow = dnow+'InhExcHalf/'    
    if shflTrsEachNeuron:
        dnow = dnow+'trsShfledPerNeuron/'
#    else:
#        dnow = '/excInh_trainDecoder_eachFrame/'+mousename+'/inhRois/' # old inhibit ROI identification... 
        
    '''    
    if loadInhAllexcEqexc==1:
        dnow = '/excInh_trainDecoder_eachFrame/frame'+str(time2an)+'/'+mousename+'/'
    else:
        dnow = '/excInh_trainDecoder_eachFrame/frame'+str(time2an)+'/'+mousename+'/inhRois/'
    '''
                
    #%% Loop over days    

    if loadWeights==1:            
        numInh = np.full((len(days)), np.nan)
        numAllexc = np.full((len(days)), np.nan)

    perClassErrorTest_data_inh_all = []
    perClassErrorTest_shfl_inh_all = []
    perClassErrorTest_chance_inh_all = []
    perClassErrorTest_data_allExc_all = []
    perClassErrorTest_shfl_allExc_all = []
    perClassErrorTest_chance_allExc_all = []
    perClassErrorTest_data_exc_all = []
    perClassErrorTest_shfl_exc_all = []
    perClassErrorTest_chance_exc_all = []
    corr_hr_lr = np.full((len(days),2), np.nan) # number of hr, lr correct trials for each day
    eventI_ds_allDays = np.full((len(days)), np.nan)    
    eventI_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)       
    fractTrs_50msExtraStim = np.full(len(days), np.nan)    

    #%%
#    iday = 0
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
    
        
        #%% Identify days with extra stim
        
        d = scio.loadmat(postName, variable_names=['timeStimOffset', 'timeSingleStimOffset', 'timeStimOnset'])
        timeStimOnset = d.pop('timeStimOnset').flatten()
        timeStimOffset = d.pop('timeStimOffset').flatten()
        timeSingleStimOffset = d.pop('timeSingleStimOffset').flatten()
        fractTrs_50msExtraStim[iday] = np.mean((timeStimOffset - timeSingleStimOffset)>50)   

        
        #%% Set number of hr, lr trials that were used for svm training
        
        svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, [1,0,0], regressBins, useEqualTrNums, corrTrained, shflTrsEachNeuron)[0]   
        
        corr_hr, corr_lr = set_corr_hr_lr(postName, svmName)    
        corr_hr_lr[iday,:] = [corr_hr, corr_lr]        

    
        #%% Set eventI_ds (downsampled eventI)

        eventI, eventI_ds = setEventIds(postName, chAl, regressBins=3, trialHistAnalysis=0)
        
        eventI_allDays[iday] = eventI
        eventI_ds_allDays[iday] = eventI_ds    

        
        #%% Load SVM vars
        
        if decodeStimCateg:
            
            perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, \
            perClassErrorTestRemCorr_inh, perClassErrorTestRemCorr_shfl_inh, perClassErrorTestRemCorr_chance_inh, \
            perClassErrorTestBoth_inh, perClassErrorTestBoth_shfl_inh, perClassErrorTestBoth_chance_inh, \
            perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, \
            perClassErrorTestRemCorr_allExc, perClassErrorTestRemCorr_shfl_allExc, perClassErrorTestRemCorr_chance_allExc, \
            perClassErrorTestBoth_allExc, perClassErrorTestBoth_shfl_allExc, perClassErrorTestBoth_chance_allExc, \
            perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, \
            perClassErrorTestRemCorr_exc, perClassErrorTestRemCorr_shfl_exc, perClassErrorTestRemCorr_chance_exc, \
            perClassErrorTestBoth_exc, perClassErrorTestBoth_shfl_exc, perClassErrorTestBoth_chance_exc, \
            w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN, trsExcluded, \
            testTrInds_allSamps_inh, Ytest_allSamps_inh, Ytest_hat_allSampsFrs_inh, trsnow_allSamps_inh, \
            testTrInds_allSamps_allExc, Ytest_allSamps_allExc, Ytest_hat_allSampsFrs_allExc, trsnow_allSamps_allExc, \
            testTrInds_allSamps_exc, Ytest_allSamps_exc, Ytest_hat_allSampsFrs_exc, trsnow_allSamps_exc, \
            = loadSVM_excInh_decodeStimCateg(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0, loadYtest=loadYtest)        
    
            ############################################
            ###### average across nRandCorr samps ######
            ############################################
            perClassErrorTest_data_inh = np.mean(perClassErrorTest_data_inh, axis=-2)
            perClassErrorTest_shfl_inh = np.mean(perClassErrorTest_shfl_inh, axis=-2)
            perClassErrorTest_chance_inh = np.mean(perClassErrorTest_chance_inh, axis=-2)
            perClassErrorTestRemCorr_inh = np.mean(perClassErrorTestRemCorr_inh, axis=-2)
            perClassErrorTestRemCorr_shfl_inh = np.mean(perClassErrorTestRemCorr_shfl_inh, axis=-2)
            perClassErrorTestRemCorr_chance_inh = np.mean(perClassErrorTestRemCorr_chance_inh, axis=-2)
            perClassErrorTestBoth_inh = np.mean(perClassErrorTestBoth_inh, axis=-2)
            perClassErrorTestBoth_shfl_inh = np.mean(perClassErrorTestBoth_shfl_inh, axis=-2)
            perClassErrorTestBoth_chance_inh = np.mean(perClassErrorTestBoth_chance_inh, axis=-2)
    
            perClassErrorTest_data_allExc = np.mean(perClassErrorTest_data_allExc, axis=-2)
            perClassErrorTest_shfl_allExc = np.mean(perClassErrorTest_shfl_allExc, axis=-2)
            perClassErrorTest_chance_allExc = np.mean(perClassErrorTest_chance_allExc, axis=-2)
            perClassErrorTestRemCorr_allExc = np.mean(perClassErrorTestRemCorr_allExc, axis=-2)
            perClassErrorTestRemCorr_shfl_allExc = np.mean(perClassErrorTestRemCorr_shfl_allExc, axis=-2)
            perClassErrorTestRemCorr_chance_allExc = np.mean(perClassErrorTestRemCorr_chance_allExc, axis=-2)
            perClassErrorTestBoth_allExc = np.mean(perClassErrorTestBoth_allExc, axis=-2)
            perClassErrorTestBoth_shfl_allExc = np.mean(perClassErrorTestBoth_shfl_allExc, axis=-2)
            perClassErrorTestBoth_chance_allExc = np.mean(perClassErrorTestBoth_chance_allExc, axis=-2)
            
            perClassErrorTest_data_exc = np.mean(perClassErrorTest_data_exc, axis=-2)
            perClassErrorTest_shfl_exc = np.mean(perClassErrorTest_shfl_exc, axis=-2)
            perClassErrorTest_chance_exc = np.mean(perClassErrorTest_chance_exc, axis=-2)
            perClassErrorTestRemCorr_exc = np.mean(perClassErrorTestRemCorr_exc, axis=-2)
            perClassErrorTestRemCorr_shfl_exc = np.mean(perClassErrorTestRemCorr_shfl_exc, axis=-2)
            perClassErrorTestRemCorr_chance_exc = np.mean(perClassErrorTestRemCorr_chance_exc, axis=-2)
            perClassErrorTestBoth_exc = np.mean(perClassErrorTestBoth_exc, axis=-2)
            perClassErrorTestBoth_shfl_exc = np.mean(perClassErrorTestBoth_shfl_exc, axis=-2)
            perClassErrorTestBoth_chance_exc = np.mean(perClassErrorTestBoth_chance_exc, axis=-2)
        
            
        elif do_excInhHalf:

            # numShufflesExc x numSamples x numFrames
            perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, w_data_inh, w_data_allExc, b_data_inh, b_data_allExc, svmName_excInh = loadSVM_excInh_excInhHalf(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, loadWeights, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0)
            perClassErrorTest_data_exc = 0; perClassErrorTest_shfl_exc = 0; perClassErrorTest_chance_exc = 0; 

            
        else: # regular case
            perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, \
            w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN, trsExcluded, \
            perClassErrorTest_data_inh_incorr, perClassErrorTest_shfl_inh_incorr, perClassErrorTest_data_allExc_incorr, perClassErrorTest_shfl_allExc_incorr, perClassErrorTest_data_exc_incorr, perClassErrorTest_shfl_exc_incorr, \
            = loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0, loadYtest=0, testIncorr=testIncorr)            

#            perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN, trsExcluded, testTrInds_allSamps_inh, Ytest_allSamps_inh, Ytest_hat_allSampsFrs_inh, trsnow_allSamps_inh, testTrInds_allSamps_allExc, Ytest_allSamps_allExc, Ytest_hat_allSampsFrs_allExc, trsnow_allSamps_allExc, testTrInds_allSamps_exc, Ytest_allSamps_exc, Ytest_hat_allSampsFrs_exc, trsnow_allSamps_exc \
#            = loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0, loadYtest=loadYtest, testIncorr=testIncorr)
            
#            perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN = loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron)
#            perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN = loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron)
        
        ##%% Get number of inh and exc        
        if loadWeights==1:
            numInh[iday] = w_data_inh.shape[1]
            numAllexc[iday] = w_data_allExc.shape[1]

        
        #%% Get class error values at a specific time bin: right before the choice 

        if decodeStimCateg:
            
            perClassErrorTest_data_inh = perClassErrorTestBoth_inh[:,eventI_ds+time2an].squeeze() # numSamps
            perClassErrorTest_shfl_inh = perClassErrorTestBoth_shfl_inh[:,eventI_ds+time2an].squeeze() # numSamps
            perClassErrorTest_chance_inh = perClassErrorTestBoth_chance_inh[:,eventI_ds+time2an].squeeze() # numSamps
            
            perClassErrorTest_data_allExc = perClassErrorTestBoth_allExc[:,eventI_ds+time2an].squeeze() # numSamps
            perClassErrorTest_shfl_allExc = perClassErrorTestBoth_shfl_allExc[:,eventI_ds+time2an].squeeze() # numSamps
            perClassErrorTest_chance_allExc = perClassErrorTestBoth_chance_allExc[:,eventI_ds+time2an].squeeze() # numSamps
    
            perClassErrorTest_data_exc = perClassErrorTestBoth_exc[:,:,eventI_ds+time2an].squeeze() # numShufflesExc x numSamps
            perClassErrorTest_shfl_exc = perClassErrorTestBoth_shfl_exc[:,:,eventI_ds+time2an].squeeze() # numShufflesExc x numSamps
            perClassErrorTest_chance_exc = perClassErrorTestBoth_chance_exc[:,:,eventI_ds+time2an].squeeze() # numShufflesExc x numSamps
                
        elif do_excInhHalf:
            perClassErrorTest_data_inh = perClassErrorTest_data_inh[:,:,eventI_ds+time2an].squeeze() # numShufflesExc x numSamps
            perClassErrorTest_shfl_inh = perClassErrorTest_shfl_inh[:,:,eventI_ds+time2an].squeeze() # numShufflesExc x numSamps
            perClassErrorTest_chance_inh = perClassErrorTest_chance_inh[:,:,eventI_ds+time2an].squeeze() # numShufflesExc x numSamps
                
            perClassErrorTest_data_allExc = perClassErrorTest_data_allExc[:,:,eventI_ds+time2an].squeeze() # numShufflesExc x numSamps
            perClassErrorTest_shfl_allExc = perClassErrorTest_shfl_allExc[:,:,eventI_ds+time2an].squeeze() # numShufflesExc x numSamps
            perClassErrorTest_chance_allExc = perClassErrorTest_chance_allExc[:,:,eventI_ds+time2an].squeeze() # numShufflesExc x numSamps            
        
        else:
            perClassErrorTest_data_inh = perClassErrorTest_data_inh[:,eventI_ds+time2an].squeeze() # numSamps
            perClassErrorTest_shfl_inh = perClassErrorTest_shfl_inh[:,eventI_ds+time2an].squeeze() # numSamps
            perClassErrorTest_chance_inh = perClassErrorTest_chance_inh[:,eventI_ds+time2an].squeeze() # numSamps
            
            perClassErrorTest_data_allExc = perClassErrorTest_data_allExc[:,eventI_ds+time2an].squeeze() # numSamps
            perClassErrorTest_shfl_allExc = perClassErrorTest_shfl_allExc[:,eventI_ds+time2an].squeeze() # numSamps
            perClassErrorTest_chance_allExc = perClassErrorTest_chance_allExc[:,eventI_ds+time2an].squeeze() # numSamps
    
            if num2AnInhAllexcEqexc == 3:
                perClassErrorTest_data_exc = perClassErrorTest_data_exc[:,:,eventI_ds+time2an].squeeze() # numShufflesExc x numSamps
                perClassErrorTest_shfl_exc = perClassErrorTest_shfl_exc[:,:,eventI_ds+time2an].squeeze() # numShufflesExc x numSamps
                perClassErrorTest_chance_exc = perClassErrorTest_chance_exc[:,:,eventI_ds+time2an].squeeze() # numShufflesExc x numSamps
    
            
            
        #%% Keep vars for all days
        
        perClassErrorTest_data_inh_all.append(perClassErrorTest_data_inh) # days x samps
        perClassErrorTest_shfl_inh_all.append(perClassErrorTest_shfl_inh)
        perClassErrorTest_chance_inh_all.append(perClassErrorTest_chance_inh)

        perClassErrorTest_data_allExc_all.append(perClassErrorTest_data_allExc) # days x samps
        perClassErrorTest_shfl_allExc_all.append(perClassErrorTest_shfl_allExc)
        perClassErrorTest_chance_allExc_all.append(perClassErrorTest_chance_allExc) 

        if num2AnInhAllexcEqexc == 3:
            perClassErrorTest_data_exc_all.append(perClassErrorTest_data_exc) # days x numShufflesExc x numSamples
            perClassErrorTest_shfl_exc_all.append(perClassErrorTest_shfl_exc)
            perClassErrorTest_chance_exc_all.append(perClassErrorTest_chance_exc)

        # Delete vars before starting the next day    
        if loadWeights==1:
            if num2AnInhAllexcEqexc == 3:
                del perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc
            else:
                del perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, w_data_inh, w_data_allExc

        else:
            if num2AnInhAllexcEqexc == 3:
                del perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc
            else:
                del perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc




    #%% Done with all days
    ######################################################################################################################################################
    ######################################################################################################################################################        
    ######################################################################################################################################################
    ######################################################################################################################################################        
    
    #%% Keep original days before removing low tr days.
    
    numDays0 = numDays
    days0 = days    
    
    ##%% Decide what days to analyze: exclude days with too few trials used for training SVM, also exclude incorr from days with too few incorr trials.    
    # th for min number of trs of each class
    '''
    thTrained = 30 #25; # 1/10 of this will be the testing tr num! and 9/10 was used for training
    thIncorr = 4 #5
    '''
    mn_corr = np.min(corr_hr_lr,axis=1) # number of trials of each class. 90% of this was used for training, and 10% for testing.
    
    print 'num days to be excluded with few svm-trained trs:', sum(mn_corr < thTrained)    
    print np.array(days)[mn_corr < thTrained]
    
    
    ####### Set days without extraStim
    #    days=np.array(days); days[fractTrs_50msExtraStim<.1]
    if noExtraStimDayAnalysis: # dont analyze days with extra stim
        days2an_noExtraStim = np.logical_and(fractTrs_50msExtraStim < .1 , mn_corr >= thTrained)
    else:
        days2an_noExtraStim = mn_corr >= thTrained #fractTrs_50msExtraStim < 2 # include all days
        
    days2an_final = days2an_noExtraStim

    
    
    numDays = sum(days2an_final)
    days = np.array(days)[days2an_final]
    
    numGoodDays_allMice[im] = numDays

    
    #%%
    eventI_allDays = eventI_allDays.astype('int')
    eventI_ds_allDays = eventI_ds_allDays.astype('int')
    
    perClassErrorTest_data_inh_all = np.array(perClassErrorTest_data_inh_all) # days x samps (if do_excInhHalf=1, then size will be : # days x numShufflesExc x numSamples)
    perClassErrorTest_shfl_inh_all = np.array(perClassErrorTest_shfl_inh_all) 
    perClassErrorTest_chance_inh_all = np.array(perClassErrorTest_chance_inh_all)
    
    perClassErrorTest_data_allExc_all = np.array(perClassErrorTest_data_allExc_all) # days x samps
    perClassErrorTest_shfl_allExc_all = np.array(perClassErrorTest_shfl_allExc_all)
    perClassErrorTest_chance_allExc_all = np.array(perClassErrorTest_chance_allExc_all)

    if num2AnInhAllexcEqexc == 3:
        perClassErrorTest_data_exc_all = np.array(perClassErrorTest_data_exc_all) # days x numShufflesExc x numSamples
        perClassErrorTest_shfl_exc_all = np.array(perClassErrorTest_shfl_exc_all)
        perClassErrorTest_chance_exc_all = np.array(perClassErrorTest_chance_exc_all)

   
    
    #%% For each day set average and std of class accuracies across CV samples in timebin time2an 
    # For exc average across both cv samps and exc shufls.
    #### ONLY include days with enough trs used for svm training!
    
    if do_excInhHalf==0:
        numSamples = np.shape(perClassErrorTest_data_inh_all[0])[0]
        if num2AnInhAllexcEqexc == 3:
            numExcSamples = np.shape(perClassErrorTest_data_exc_all[0])[0]
        
        av_test_data_inh = 100-np.mean(perClassErrorTest_data_inh_all[days2an_final,:], axis=1) # numDays
        sd_test_data_inh = np.std(perClassErrorTest_data_inh_all[days2an_final,:], axis=1) / np.sqrt(numSamples)
        av_test_shfl_inh = 100-np.mean(perClassErrorTest_shfl_inh_all[days2an_final,:], axis=1)
        sd_test_shfl_inh = np.std(perClassErrorTest_shfl_inh_all[days2an_final,:], axis=1) / np.sqrt(numSamples)
        av_test_chance_inh = 100-np.mean(perClassErrorTest_chance_inh_all[days2an_final,:], axis=1)
        sd_test_chance_inh = np.std(perClassErrorTest_chance_inh_all[days2an_final,:], axis=1) / np.sqrt(numSamples)
        
        av_test_data_allExc = 100-np.mean(perClassErrorTest_data_allExc_all[days2an_final,:], axis=1) # numDays
        sd_test_data_allExc = np.std(perClassErrorTest_data_allExc_all[days2an_final,:], axis=1) / np.sqrt(numSamples)
        av_test_shfl_allExc = 100-np.mean(perClassErrorTest_shfl_allExc_all[days2an_final,:], axis=1)
        sd_test_shfl_allExc = np.std(perClassErrorTest_shfl_allExc_all[days2an_final,:], axis=1) / np.sqrt(numSamples)
        av_test_chance_allExc = 100-np.mean(perClassErrorTest_chance_allExc_all[days2an_final,:], axis=1)
        sd_test_chance_allExc = np.std(perClassErrorTest_chance_allExc_all[days2an_final,:], axis=1) / np.sqrt(numSamples)
        
        if num2AnInhAllexcEqexc == 3:
            av_test_data_exc = 100-np.mean(perClassErrorTest_data_exc_all[days2an_final,:,:], axis=(1,2)) # numDays  # average across cv samples and excShuffles
            sd_test_data_exc = np.std(perClassErrorTest_data_exc_all[days2an_final,:,:], axis=(1,2)) / np.sqrt(numSamples+numExcSamples)
            av_test_shfl_exc = 100-np.mean(perClassErrorTest_shfl_exc_all[days2an_final,:,:], axis=(1,2))
            sd_test_shfl_exc = np.std(perClassErrorTest_shfl_exc_all[days2an_final,:,:], axis=(1,2)) / np.sqrt(numSamples+numExcSamples)
            av_test_chance_exc = 100-np.mean(perClassErrorTest_chance_exc_all[days2an_final,:,:], axis=(1,2))
            sd_test_chance_exc = np.std(perClassErrorTest_chance_exc_all[days2an_final,:,:], axis=(1,2)) / np.sqrt(numSamples+numExcSamples)
    
    else:
        numSamples = np.shape(perClassErrorTest_data_inh_all[0])[1]
        numExcSamples = np.shape(perClassErrorTest_data_inh_all[0])[0]
        
        ######### excInhHalf
        av_test_data_inh = 100-np.mean(perClassErrorTest_data_inh_all[days2an_final,:,:], axis=(1,2)) # numDays  # average across cv samples and excShuffles
        sd_test_data_inh = np.std(perClassErrorTest_data_inh_all[days2an_final,:,:], axis=(1,2)) / np.sqrt(numSamples+numExcSamples)
        av_test_shfl_inh = 100-np.mean(perClassErrorTest_shfl_inh_all[days2an_final,:,:], axis=(1,2))
        sd_test_shfl_inh = np.std(perClassErrorTest_shfl_inh_all[days2an_final,:,:], axis=(1,2)) / np.sqrt(numSamples+numExcSamples)
        av_test_chance_inh = 100-np.mean(perClassErrorTest_chance_inh_all[days2an_final,:,:], axis=(1,2))
        sd_test_chance_inh = np.std(perClassErrorTest_chance_inh_all[days2an_final,:,:], axis=(1,2)) / np.sqrt(numSamples+numExcSamples)
        
        ######### allExc2inhSize    
        av_test_data_allExc = 100-np.mean(perClassErrorTest_data_allExc_all[days2an_final,:,:], axis=(1,2)) # numDays  # average across cv samples and excShuffles
        sd_test_data_allExc = np.std(perClassErrorTest_data_allExc_all[days2an_final,:,:], axis=(1,2)) / np.sqrt(numSamples+numExcSamples)
        av_test_shfl_allExc = 100-np.mean(perClassErrorTest_shfl_allExc_all[days2an_final,:,:], axis=(1,2))
        sd_test_shfl_allExc = np.std(perClassErrorTest_shfl_allExc_all[days2an_final,:,:], axis=(1,2)) / np.sqrt(numSamples+numExcSamples)
        av_test_chance_allExc = 100-np.mean(perClassErrorTest_chance_allExc_all[days2an_final,:,:], axis=(1,2))
        sd_test_chance_allExc = np.std(perClassErrorTest_chance_allExc_all[days2an_final,:,:], axis=(1,2)) / np.sqrt(numSamples+numExcSamples)

    
    # print p values... across days, are exc and inh CAs different?
    _,pcorrtrace = stats.ttest_ind(av_test_data_inh, av_test_data_allExc) # p value of class accuracy being different from 50
    print pcorrtrace 
    if num2AnInhAllexcEqexc == 3:
        _,pcorrtrace = stats.ttest_ind(av_test_data_inh, av_test_data_exc) # p value of class accuracy being different from 50
        print pcorrtrace     
        
        
    #%% Keep values of all mice (cvSample-averages for each session)
    
    corr_hr_lr_allMice.append(corr_hr_lr)
    
    av_test_data_inh_allMice.append(av_test_data_inh)
    sd_test_data_inh_allMice.append(sd_test_data_inh)
    av_test_shfl_inh_allMice.append(av_test_shfl_inh)
    sd_test_shfl_inh_allMice.append(sd_test_shfl_inh)
    av_test_chance_inh_allMice.append(av_test_chance_inh)
    sd_test_chance_inh_allMice.append(sd_test_chance_inh)

    av_test_data_allExc_allMice.append(av_test_data_allExc)
    sd_test_data_allExc_allMice.append(sd_test_data_allExc)
    av_test_shfl_allExc_allMice.append(av_test_shfl_allExc)
    sd_test_shfl_allExc_allMice.append(sd_test_shfl_allExc)
    av_test_chance_allExc_allMice.append(av_test_chance_allExc)
    sd_test_chance_allExc_allMice.append(sd_test_chance_allExc)

    if num2AnInhAllexcEqexc == 3:
        av_test_data_exc_allMice.append(av_test_data_exc)
        sd_test_data_exc_allMice.append(sd_test_data_exc)
        av_test_shfl_exc_allMice.append(av_test_shfl_exc)
        sd_test_shfl_exc_allMice.append(sd_test_shfl_exc)
        av_test_chance_exc_allMice.append(av_test_chance_exc)
        sd_test_chance_exc_allMice.append(sd_test_chance_exc)


    #%%
    ######################## PLOTS (each mouse) ########################
    
    #%% Plot class accuracy in the frame before the choice for each day
    # Class accuracy vs day number
    
    plt.figure(figsize=(6,10))
    

    ### data - shuffle
    plt.subplot(311)
    plt.errorbar(range(numDays), av_test_data_inh - av_test_shfl_inh, sd_test_data_inh, label=labInh, color='r')
    plt.errorbar(range(numDays), av_test_data_allExc - av_test_shfl_allExc, sd_test_data_allExc, label=labAll, color='k')
    if num2AnInhAllexcEqexc == 3:
        plt.errorbar(range(numDays), av_test_data_exc - av_test_shfl_exc, sd_test_data_exc, label='exc', color='b')
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)
#    plt.xticks(range(numDays), [days[i][0:6] for i in range(len(days))], rotation='vertical')    
#    plt.xlabel('Days')
    plt.ylabel('Classification accuracy (%)\ndata-shfl')
    makeNicePlots(plt.gca())
    plt.xlim([-1,len(days)-1+1])


    ### data
    plt.subplot(312)
    plt.errorbar(range(numDays), av_test_data_inh, sd_test_data_inh, label=labInh, color='r')
    plt.errorbar(range(numDays), av_test_data_allExc, sd_test_data_allExc, label=labAll, color='k')
    if num2AnInhAllexcEqexc == 3:
        plt.errorbar(range(numDays), av_test_data_exc, sd_test_data_exc, label='exc', color='b')        
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)        
#    plt.xticks(range(numDays), [days[i][0:6] for i in range(len(days))], rotation='vertical')    
#    plt.xlabel('Days')
    plt.ylabel('Classification accuracy (%)\ndata')
    makeNicePlots(plt.gca())
    plt.xlim([-1,len(days)-1+1])


    ### shuffle
    plt.subplot(313)
    plt.errorbar(range(numDays), av_test_shfl_inh, sd_test_data_inh, label=labInh, color='r')
    plt.errorbar(range(numDays), av_test_shfl_allExc, sd_test_data_allExc, label=labAll, color='k')
    if num2AnInhAllexcEqexc == 3:
        plt.errorbar(range(numDays), av_test_shfl_exc, sd_test_data_exc, label='exc', color='b')
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)
    plt.xticks(range(numDays), [days[i][0:6] for i in range(len(days))], rotation='vertical')    
#    plt.xlabel('Days')
    plt.ylabel('Classification accuracy (%)\nshfl')
    makeNicePlots(plt.gca())
    plt.xlim([-1,len(days)-1+1])


    #### Save fig
    if savefigs:#% Save the figure
        if chAl==1:
            dd = 'chAl_'+ corrn + 'eachDay_time' + str(time2an) + '_aveSeSamps_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_'+ corrn + 'eachDay_time' + str(time2an) + '_aveSeSamps_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)






#%% Done with all mice
######################################################################################################################################################        
######################################################################################################################################################        
######################################################################################################################################################        
######################################################################################################################################################        

#%% each element is ave and se across cv samps. (we dont really use se across cv samps below... below we plot ave and sd of cv-samped averages across sessions)
    
av_test_data_inh_allMice = np.array(av_test_data_inh_allMice) #numMice; each mouse: numDays
sd_test_data_inh_allMice = np.array(sd_test_data_inh_allMice)
av_test_shfl_inh_allMice = np.array(av_test_shfl_inh_allMice)
sd_test_shfl_inh_allMice = np.array(sd_test_shfl_inh_allMice)
av_test_chance_inh_allMice = np.array(av_test_chance_inh_allMice)
sd_test_chance_inh_allMice = np.array(sd_test_chance_inh_allMice)

av_test_data_allExc_allMice = np.array(av_test_data_allExc_allMice)
sd_test_data_allExc_allMice = np.array(sd_test_data_allExc_allMice)
av_test_shfl_allExc_allMice = np.array(av_test_shfl_allExc_allMice)
sd_test_shfl_allExc_allMice = np.array(sd_test_shfl_allExc_allMice)
av_test_chance_allExc_allMice = np.array(av_test_chance_allExc_allMice)
sd_test_chance_allExc_allMice = np.array(sd_test_chance_allExc_allMice)

if num2AnInhAllexcEqexc == 3:
    av_test_data_exc_allMice = np.array(av_test_data_exc_allMice)
    sd_test_data_exc_allMice = np.array(sd_test_data_exc_allMice)
    av_test_shfl_exc_allMice = np.array(av_test_shfl_exc_allMice)
    sd_test_shfl_exc_allMice = np.array(sd_test_shfl_exc_allMice)
    av_test_chance_exc_allMice = np.array(av_test_chance_exc_allMice)
    sd_test_chance_exc_allMice = np.array(sd_test_chance_exc_allMice)


#%% Set p value: data vs shfl, for each mouse
    
p_ds_allExc = np.full((len(mice)), np.nan)
p_ds_exc = np.full((len(mice)), np.nan)
p_ds_inh = np.full((len(mice)), np.nan)
for im in range(len(mice)):
    _,p_ds_allExc[im] = stats.ttest_ind(av_test_data_allExc_allMice[im], av_test_shfl_allExc_allMice[im])    
    _,p_ds_exc[im] = stats.ttest_ind(av_test_data_exc_allMice[im], av_test_shfl_exc_allMice[im])    
    _,p_ds_inh[im] = stats.ttest_ind(av_test_data_inh_allMice[im], av_test_shfl_inh_allMice[im])    
    
    
# for each mouse compute ttest p value between exc and inh (CA in last time bin before the choice, each averaged across samps)    
pei_allMice = np.full(len(mice), np.nan)    
for im in range(len(mice)):
    if num2AnInhAllexcEqexc == 3:
        _,pei = stats.ttest_ind(av_test_data_inh_allMice[im] , av_test_data_exc_allMice[im])
    else:
        _,pei = stats.ttest_ind(av_test_data_inh_allMice[im] , av_test_data_allExc_allMice[im])
    pei_allMice[im] = pei
print pei_allMice    



#%% Average and se of classErr across sessions for each mouse

numMice = len(mice)
#numGoodDays_allMice0 = numGoodDays_allMice
#numGoodDays_allMice = [1,1,1,1] # if you want sd and not se

av_av_test_data_inh_allMice = np.array([np.nanmean(av_test_data_inh_allMice[im], axis=0) for im in range(numMice)]) # numMice; each mouse: numDays
sd_av_test_data_inh_allMice = np.array([np.nanstd(av_test_data_inh_allMice[im], axis=0)/np.sqrt(numGoodDays_allMice[im]) for im in range(numMice)])
av_av_test_shfl_inh_allMice = np.array([np.nanmean(av_test_shfl_inh_allMice[im], axis=0) for im in range(numMice)])
sd_av_test_shfl_inh_allMice = np.array([np.nanstd(av_test_shfl_inh_allMice[im], axis=0)/np.sqrt(numGoodDays_allMice[im]) for im in range(numMice)])
av_av_test_chance_inh_allMice = np.array([np.nanmean(av_test_chance_inh_allMice[im], axis=0) for im in range(numMice)])
sd_av_test_chance_inh_allMice = np.array([np.nanstd(av_test_chance_inh_allMice[im], axis=0)/np.sqrt(numGoodDays_allMice[im]) for im in range(numMice)])

av_av_test_data_allExc_allMice = np.array([np.nanmean(av_test_data_allExc_allMice[im], axis=0) for im in range(numMice)])
sd_av_test_data_allExc_allMice = np.array([np.nanstd(av_test_data_allExc_allMice[im], axis=0)/np.sqrt(numGoodDays_allMice[im]) for im in range(numMice)])
av_av_test_shfl_allExc_allMice = np.array([np.nanmean(av_test_shfl_allExc_allMice[im], axis=0) for im in range(numMice)])
sd_av_test_shfl_allExc_allMice = np.array([np.nanstd(av_test_shfl_allExc_allMice[im], axis=0)/np.sqrt(numGoodDays_allMice[im]) for im in range(numMice)])
av_av_test_chance_allExc_allMice = np.array([np.nanmean(av_test_chance_allExc_allMice[im], axis=0) for im in range(numMice)])
sd_av_test_chance_allExc_allMice = np.array([np.nanstd(av_test_chance_allExc_allMice[im], axis=0)/np.sqrt(numGoodDays_allMice[im]) for im in range(numMice)])

if num2AnInhAllexcEqexc == 3:
    av_av_test_data_exc_allMice = np.array([np.nanmean(av_test_data_exc_allMice[im], axis=0) for im in range(numMice)])
    sd_av_test_data_exc_allMice = np.array([np.nanstd(av_test_data_exc_allMice[im], axis=0)/np.sqrt(numGoodDays_allMice[im]) for im in range(numMice)])
    av_av_test_shfl_exc_allMice = np.array([np.nanmean(av_test_shfl_exc_allMice[im], axis=0) for im in range(numMice)])
    sd_av_test_shfl_exc_allMice = np.array([np.nanstd(av_test_shfl_exc_allMice[im], axis=0)/np.sqrt(numGoodDays_allMice[im]) for im in range(numMice)])
    av_av_test_chance_exc_allMice = np.array([np.nanmean(av_test_chance_exc_allMice[im], axis=0) for im in range(numMice)])
    sd_av_test_chance_exc_allMice = np.array([np.nanstd(av_test_chance_exc_allMice[im], axis=0)/np.sqrt(numGoodDays_allMice[im]) for im in range(numMice)])


#%% Plot classErr (averaged across all sessions) for each mouse

#, markerfacecolor=colors[0], markeredgecolor=colors[0], markersize=4
gp = .2
x = np.arange(0,numMice)

plt.figure(figsize=(1.7,3))

# testing data
plt.errorbar(x, av_av_test_data_allExc_allMice, sd_av_test_data_allExc_allMice, fmt='o', label=labAll, color='k', markeredgecolor='k', markersize=4)
plt.errorbar(x+gp, av_av_test_data_inh_allMice, sd_av_test_data_inh_allMice, fmt='o', label=labInh, color='r', markeredgecolor='r', markersize=4)
if num2AnInhAllexcEqexc == 3:
    plt.errorbar(x+2*gp, av_av_test_data_exc_allMice, sd_av_test_data_exc_allMice, fmt='o', label='exc', color='b', markeredgecolor='b', markersize=4)

# shfl
plt.errorbar(x, av_av_test_shfl_allExc_allMice, sd_av_test_shfl_allExc_allMice, color='k', fmt='o', alpha=.3, markeredgecolor='k', markersize=4)
plt.errorbar(x+gp, av_av_test_shfl_inh_allMice, sd_av_test_shfl_inh_allMice, color='r', fmt='o', alpha=.3, markeredgecolor='r', markersize=4)
if num2AnInhAllexcEqexc == 3:
    plt.errorbar(x+2*gp, av_av_test_shfl_exc_allMice, sd_av_test_shfl_exc_allMice, color='b', fmt='o', alpha=.3, markeredgecolor='b', markersize=4)


plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1)#, frameon=False) 
plt.xlabel('Mice', fontsize=11)
plt.ylabel('Classification accuracy (%)\n [-97 0] ms rel. choice', fontsize=11)
plt.xlim([-.2, len(mice)-1+.2+2*gp])
plt.xticks(x+gp, mice)
ax = plt.gca()
makeNicePlots(ax)
yl = ax.get_ylim()
plt.ylim([yl[0]-2, yl[1]])

### mark mice that are sig (data vs shfl)
pp_allExc = np.full(len(mice), np.nan)
pp_allExc[p_ds_allExc <=.05] = yl[1]
pp_inh = np.full(len(mice), np.nan)
pp_inh[p_ds_inh <=.05] = yl[1]
pp_exc = np.full(len(mice), np.nan)
pp_exc[p_ds_exc <=.05] = yl[1]

plt.plot(x, pp_allExc, 'k*', markeredgecolor='k', markersize=2)
plt.plot(x+gp, pp_inh, 'r*', markeredgecolor='r', markersize=2)
plt.plot(x+2*gp, pp_exc, 'b*', markeredgecolor='b', markersize=2)

plt.ylim([yl[0]-2, yl[1]+2])


##%%
if savefigs:#% Save the figure
    if shflTrsEachNeuron:
        sn = 'shflTrsPerN_'
    else:
        sn = ''
        
    if chAl==1:
        dd = 'chAl_'+ corrn + 'aveSeDays_time' + str(time2an) + '_' + labAll + '_' + sn + '_'.join(mice) + '_' + nowStr # + days[0][0:6] + '-to-' + days[-1][0:6]
    else:
        dd = 'stAl_'+ corrn + 'aveSeDays_time' + str(time2an) + '_' + labAll + '_' + sn + '_'.join(mice) + '_' + nowStr # + days[0][0:6] + '-to-' + days[-1][0:6]
        
    d = os.path.join(svmdir+dnow0)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])

    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)




