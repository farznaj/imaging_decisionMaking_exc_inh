# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:05:28 2017

@author: farznaj
"""

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

savefigs = 0
doAllN = 1 # plot allN, instead of allExc
time2an = -1; # relative to eventI, look at classErr in what time stamp.
thTrained = 10#10 # number of trials of each class used for svm training, min acceptable value to include a day in analysis
corrTrained = 1
doIncorr = 0

ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. #chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces.  #chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
loadWeights = 0
num2AnInhAllexcEqexc = 3 # if 3, all 3 types (inh, allExc, exc) will be analyzed. If 2, inh and allExc will be analyzed. if 1, inh will be analyzed. # if you have all svm files saved, set it to 3.
trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
useEqualTrNums = 1

import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.
chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]
if doAllN==1:
    labAll = 'allN'
else:
    labAll = 'allExc'    
dnow0 = '/excInh_trainDecoder_eachFrame/'
#dnow0 = '/excInh_trainDecoder_eachFrame/frame'+str(time2an)+'/'
#loadInhAllexcEqexc = 1 # if 1, load 2nd run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc separately; also for all days the new vector inhRois_pix was used (not the old inhRois)        
from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')

if doAllN==1:
    smallestC = 0 # Identify best c: if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
    if smallestC==1:
        print 'bestc = smallest c whose cv error is less than 1se of min cv error'
    else:
        print 'bestc = c that gives min cv error'


execfile("defFuns.py") # Define common funcitons


#%%
    
for shflTrsEachNeuron in [1,0]:  # Set to 0 for normal SVM training. # Shuffle trials in X_svm (for each neuron independently) to break correlations between neurons in each trial.
    
    ###
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
    
    av_test_dms_inh_allMice = []
    sd_test_dms_inh_allMice = []
    av_test_dms_exc_allMice = []
    sd_test_dms_exc_allMice = []
    av_test_dms_allExc_allMice = []
    sd_test_dms_allExc_allMice = []    

    ###    
    perClassErrorTest_data_inh_all_allMice = []
    perClassErrorTest_shfl_inh_all_allMice = []
    perClassErrorTest_chance_inh_all_allMice = []
    test_dms_inh_allMice = []
    
    perClassErrorTest_data_allExc_all_allMice = []
    perClassErrorTest_shfl_allExc_all_allMice = []
    perClassErrorTest_chance_allExc_all_allMice = []
    test_dms_allExc_allMice = []
    
    perClassErrorTest_data_exc_all_allMice = []
    perClassErrorTest_shfl_exc_all_allMice = []
    perClassErrorTest_chance_exc_all_allMice = []    
    test_dms_exc_allMice = []

    ###
    corr_hr_lr_allMice = []    
    eventI_ds_allDays_allMice = []
    eventI_allDays_allMice = []
    
    
    #%% Loop through mice
    
    for im in range(len(mice)):
            
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
0   
        
#        execfile("svm_plots_setVars_n.py")      
        days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays)


        #%% Loop over days    
    
        if loadWeights==1:            
            numInh = np.full((len(days)), np.nan)
            numAllexc = np.full((len(days)), np.nan)
    
        eventI_ds_allDays = np.full((len(days)), np.nan)    
        eventI_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
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
        
            
            #%% Set number of hr, lr trials that were used for svm training
            
            svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, [1,0,0], regressBins, useEqualTrNums, corrTrained, shflTrsEachNeuron)[0]   
            
            corr_hr, corr_lr = set_corr_hr_lr(postName, svmName)    
            corr_hr_lr[iday,:] = [corr_hr, corr_lr]        
    
        
            #%% Load matlab vars to set eventI_ds (downsampled eventI)
    
            eventI, eventI_ds = setEventIds(postName, chAl, regressBins=3, trialHistAnalysis=0)
            
            eventI_allDays[iday] = eventI
            eventI_ds_allDays[iday] = eventI_ds    
    
            
            #%% Load SVM vars

            perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, \
            w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN, trsExcluded, \
            perClassErrorTest_data_inh_incorr, perClassErrorTest_shfl_inh_incorr, perClassErrorTest_data_allExc_incorr, perClassErrorTest_shfl_allExc_incorr, perClassErrorTest_data_exc_incorr, perClassErrorTest_shfl_exc_incorr, \
            = loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0, loadYtest=0, testIncorr=0)
            
#            perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN ,\
#            = loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron)
            
            ##%% Get number of inh and exc        
            if loadWeights==1:
                numInh[iday] = w_data_inh.shape[1]
                numAllexc[iday] = w_data_allExc.shape[1]
    
            
            #%% Get class error values at a specific time bin: right before the choice 
            
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
    
    
    
        ######################################################################################################################################################
        ######################################################################################################################################################        
        #%% Done with all days
        
        eventI_allDays = eventI_allDays.astype('int')
        eventI_ds_allDays = eventI_ds_allDays.astype('int')
        
        perClassErrorTest_data_inh_all = np.array(perClassErrorTest_data_inh_all) # days x samps
        perClassErrorTest_shfl_inh_all = np.array(perClassErrorTest_shfl_inh_all) 
        perClassErrorTest_chance_inh_all = np.array(perClassErrorTest_chance_inh_all)
        
        perClassErrorTest_data_allExc_all = np.array(perClassErrorTest_data_allExc_all) # days x samps
        perClassErrorTest_shfl_allExc_all = np.array(perClassErrorTest_shfl_allExc_all)
        perClassErrorTest_chance_allExc_all = np.array(perClassErrorTest_chance_allExc_all)
    
        if num2AnInhAllexcEqexc == 3:
            perClassErrorTest_data_exc_all = np.array(perClassErrorTest_data_exc_all) # days x numShufflesExc x numSamples
            perClassErrorTest_shfl_exc_all = np.array(perClassErrorTest_shfl_exc_all)
            perClassErrorTest_chance_exc_all = np.array(perClassErrorTest_chance_exc_all)
        
        
        #%% Keep original days before removing low tr days.
        
        numDays0 = numDays
        days0 = days
        
        
        #%% Decide what days to analyze: exclude days with too few trials used for training SVM, also exclude incorr from days with too few incorr trials.
        
        # th for min number of trs of each class
        '''
        thTrained = 30 #25; # 1/10 of this will be the testing tr num! and 9/10 was used for training
        thIncorr = 4 #5
        '''
        mn_corr = np.min(corr_hr_lr,axis=1) # number of trials of each class. 90% of this was used for training, and 10% for testing.
        
        print 'num days to be excluded with few svm-trained trs:', sum(mn_corr < thTrained)    
        print np.array(days)[mn_corr < thTrained]
        
        numDays = sum(mn_corr>=thTrained)
        days = np.array(days)[mn_corr>=thTrained]
        
        
        #%% For each day set average and std of class accuracies across CV samples in timebin time2an 
        # For exc average across both cv samps and exc shufls.
        #### ONLY include days with enough trs used for svm training!
        
        numSamples = np.shape(perClassErrorTest_data_inh_all[0])[0]
        if num2AnInhAllexcEqexc == 3:
            numExcSamples = np.shape(perClassErrorTest_data_exc_all[0])[0]
        
        av_test_data_inh = 100-np.mean(perClassErrorTest_data_inh_all[mn_corr >= thTrained,:], axis=1) # numDays
        sd_test_data_inh = np.std(perClassErrorTest_data_inh_all[mn_corr >= thTrained,:], axis=1) / np.sqrt(numSamples)
        av_test_shfl_inh = 100-np.mean(perClassErrorTest_shfl_inh_all[mn_corr >= thTrained,:], axis=1)
        sd_test_shfl_inh = np.std(perClassErrorTest_shfl_inh_all[mn_corr >= thTrained,:], axis=1) / np.sqrt(numSamples)
        av_test_chance_inh = 100-np.mean(perClassErrorTest_chance_inh_all[mn_corr >= thTrained,:], axis=1)
        sd_test_chance_inh = np.std(perClassErrorTest_chance_inh_all[mn_corr >= thTrained,:], axis=1) / np.sqrt(numSamples)
        test_dms_inh = (100-perClassErrorTest_data_inh_all[mn_corr >= thTrained,:].T - av_test_shfl_inh).T # subtract samp-ave shfl from each samp of data # dimensions: just like perClassErrorTest_data_inh_all
        av_test_dms_inh = np.mean(test_dms_inh, axis=1) 
        sd_test_dms_inh = np.std(test_dms_inh, axis=1) / np.sqrt(numSamples)  # same as sd_test_data_inh bc sd_test_dms_inh = sd(test_data_inh - ave(test_shfl_inh))
                
        av_test_data_allExc = 100-np.mean(perClassErrorTest_data_allExc_all[mn_corr >= thTrained,:], axis=1) # numDays
        sd_test_data_allExc = np.std(perClassErrorTest_data_allExc_all[mn_corr >= thTrained,:], axis=1) / np.sqrt(numSamples)
        av_test_shfl_allExc = 100-np.mean(perClassErrorTest_shfl_allExc_all[mn_corr >= thTrained,:], axis=1)
        sd_test_shfl_allExc = np.std(perClassErrorTest_shfl_allExc_all[mn_corr >= thTrained,:], axis=1) / np.sqrt(numSamples)
        av_test_chance_allExc = 100-np.mean(perClassErrorTest_chance_allExc_all[mn_corr >= thTrained,:], axis=1)
        sd_test_chance_allExc = np.std(perClassErrorTest_chance_allExc_all[mn_corr >= thTrained,:], axis=1) / np.sqrt(numSamples)
        test_dms_allExc = (100-perClassErrorTest_data_allExc_all[mn_corr >= thTrained,:].T - av_test_shfl_allExc).T # subtract samp-ave shfl from each samp of data
        av_test_dms_allExc = np.mean(test_dms_allExc, axis=1) 
        sd_test_dms_allExc = np.std(test_dms_allExc, axis=1) / np.sqrt(numSamples) 
        
        if num2AnInhAllexcEqexc == 3:
            av_test_data_exc = 100-np.mean(perClassErrorTest_data_exc_all[mn_corr >= thTrained,:,:], axis=(1,2)) # numDays  # average across cv samples and excShuffles
            sd_test_data_exc = np.std(perClassErrorTest_data_exc_all[mn_corr >= thTrained,:,:], axis=(1,2)) / np.sqrt(numSamples+numExcSamples)
            av_test_shfl_exc = 100-np.mean(perClassErrorTest_shfl_exc_all[mn_corr >= thTrained,:,:], axis=(1,2))
            sd_test_shfl_exc = np.std(perClassErrorTest_shfl_exc_all[mn_corr >= thTrained,:,:], axis=(1,2)) / np.sqrt(numSamples+numExcSamples)
            av_test_chance_exc = 100-np.mean(perClassErrorTest_chance_exc_all[mn_corr >= thTrained,:,:], axis=(1,2))
            sd_test_chance_exc = np.std(perClassErrorTest_chance_exc_all[mn_corr >= thTrained,:,:], axis=(1,2)) / np.sqrt(numSamples+numExcSamples)
            test_dms_exc = (100-perClassErrorTest_data_exc_all[mn_corr >= thTrained,:].T - av_test_shfl_exc).T # subtract samp-ave shfl from each samp of data
            av_test_dms_exc = np.mean(test_dms_exc, axis=(1,2)) 
            sd_test_dms_exc = np.std(test_dms_exc, axis=(1,2)) / np.sqrt(numSamples) 
        
        
        #%% Keep values of all mice (cvSample-averages for each session)
        
        corr_hr_lr_allMice.append(corr_hr_lr)
        eventI_ds_allDays_allMice.append(eventI_ds_allDays)
        eventI_allDays_allMice.append(eventI_allDays)
        
        av_test_data_inh_allMice.append(av_test_data_inh)
        sd_test_data_inh_allMice.append(sd_test_data_inh)
        av_test_shfl_inh_allMice.append(av_test_shfl_inh)
        sd_test_shfl_inh_allMice.append(sd_test_shfl_inh)
        av_test_chance_inh_allMice.append(av_test_chance_inh)
        sd_test_chance_inh_allMice.append(sd_test_chance_inh)
        av_test_dms_inh_allMice.append(av_test_dms_inh)
        sd_test_dms_inh_allMice.append(sd_test_dms_inh)
        
        av_test_data_allExc_allMice.append(av_test_data_allExc)
        sd_test_data_allExc_allMice.append(sd_test_data_allExc)
        av_test_shfl_allExc_allMice.append(av_test_shfl_allExc)
        sd_test_shfl_allExc_allMice.append(sd_test_shfl_allExc)
        av_test_chance_allExc_allMice.append(av_test_chance_allExc)
        sd_test_chance_allExc_allMice.append(sd_test_chance_allExc)
        av_test_dms_allExc_allMice.append(av_test_dms_allExc)
        sd_test_dms_allExc_allMice.append(sd_test_dms_allExc)
        
        if num2AnInhAllexcEqexc == 3:
            av_test_data_exc_allMice.append(av_test_data_exc)
            sd_test_data_exc_allMice.append(sd_test_data_exc)
            av_test_shfl_exc_allMice.append(av_test_shfl_exc)
            sd_test_shfl_exc_allMice.append(sd_test_shfl_exc)
            av_test_chance_exc_allMice.append(av_test_chance_exc)
            sd_test_chance_exc_allMice.append(sd_test_chance_exc)
            av_test_dms_exc_allMice.append(av_test_dms_exc)
            sd_test_dms_exc_allMice.append(sd_test_dms_exc)    
    

        #
        perClassErrorTest_data_inh_all_allMice.append(perClassErrorTest_data_inh_all) # days x samps
        perClassErrorTest_shfl_inh_all_allMice.append(perClassErrorTest_shfl_inh_all) 
        perClassErrorTest_chance_inh_all_allMice.append(perClassErrorTest_chance_inh_all)
        test_dms_inh_allMice.append(test_dms_inh) # remember dms already includes good days: [mn_corr >= thTrained,ies,:]
        
        perClassErrorTest_data_allExc_all_allMice.append(perClassErrorTest_data_allExc_all) # days x samps
        perClassErrorTest_shfl_allExc_all_allMice.append(perClassErrorTest_shfl_allExc_all)
        perClassErrorTest_chance_allExc_all_allMice.append(perClassErrorTest_chance_allExc_all)
        test_dms_allExc_allMice.append(test_dms_allExc)
        
        if num2AnInhAllexcEqexc == 3:
            perClassErrorTest_data_exc_all_allMice.append(perClassErrorTest_data_exc_all) # days x numShufflesExc x numSamples
            perClassErrorTest_shfl_exc_all_allMice.append(perClassErrorTest_shfl_exc_all)
            perClassErrorTest_chance_exc_all_allMice.append(perClassErrorTest_chance_exc_all)
            test_dms_exc_allMice.append(test_dms_exc)            
            
            
        #%% keep shflTrsPerN values vs non shuffled.
    
        if shflTrsEachNeuron:
            #
            av_test_data_inh_allMice_shflTrsPerN = av_test_data_inh_allMice
            sd_test_data_inh_allMice_shflTrsPerN = sd_test_data_inh_allMice
            av_test_shfl_inh_allMice_shflTrsPerN = av_test_shfl_inh_allMice
            sd_test_shfl_inh_allMice_shflTrsPerN = sd_test_shfl_inh_allMice
            av_test_chance_inh_allMice_shflTrsPerN = av_test_chance_inh_allMice
            sd_test_chance_inh_allMice_shflTrsPerN = sd_test_chance_inh_allMice
            av_test_dms_inh_allMice_shflTrsPerN = av_test_dms_inh_allMice
            sd_test_dms_inh_allMice_shflTrsPerN = sd_test_dms_inh_allMice
            
            av_test_data_allExc_allMice_shflTrsPerN = av_test_data_allExc_allMice
            sd_test_data_allExc_allMice_shflTrsPerN = sd_test_data_allExc_allMice
            av_test_shfl_allExc_allMice_shflTrsPerN = av_test_shfl_allExc_allMice
            sd_test_shfl_allExc_allMice_shflTrsPerN = sd_test_shfl_allExc_allMice
            av_test_chance_allExc_allMice_shflTrsPerN = av_test_chance_allExc_allMice
            sd_test_chance_allExc_allMice_shflTrsPerN = sd_test_chance_allExc_allMice
            av_test_dms_allExc_allMice_shflTrsPerN = av_test_dms_allExc_allMice
            sd_test_dms_allExc_allMice_shflTrsPerN = sd_test_dms_allExc_allMice
            
            if num2AnInhAllexcEqexc == 3:
                av_test_data_exc_allMice_shflTrsPerN = av_test_data_exc_allMice
                sd_test_data_exc_allMice_shflTrsPerN = sd_test_data_exc_allMice
                av_test_shfl_exc_allMice_shflTrsPerN = av_test_shfl_exc_allMice
                sd_test_shfl_exc_allMice_shflTrsPerN = sd_test_shfl_exc_allMice
                av_test_chance_exc_allMice_shflTrsPerN = av_test_chance_exc_allMice
                sd_test_chance_exc_allMice_shflTrsPerN = sd_test_chance_exc_allMice
                av_test_dms_exc_allMice_shflTrsPerN = av_test_dms_exc_allMice
                sd_test_dms_exc_allMice_shflTrsPerN = sd_test_dms_exc_allMice                

            
            #
            perClassErrorTest_data_inh_all_allMice_shflTrsPerN = perClassErrorTest_data_inh_all_allMice
            perClassErrorTest_shfl_inh_all_allMice_shflTrsPerN = perClassErrorTest_shfl_inh_all_allMice
            perClassErrorTest_chance_inh_all_allMice_shflTrsPerN = perClassErrorTest_chance_inh_all_allMice
            test_dms_inh_allMice_shflTrsPerN = test_dms_inh_allMice
            
            perClassErrorTest_data_allExc_all_allMice_shflTrsPerN = perClassErrorTest_data_allExc_all_allMice
            perClassErrorTest_shfl_allExc_all_allMice_shflTrsPerN = perClassErrorTest_shfl_allExc_all_allMice
            perClassErrorTest_chance_allExc_all_allMice_shflTrsPerN = perClassErrorTest_chance_allExc_all_allMice
            test_dms_allExc_allMice_shflTrsPerN = test_dms_allExc_allMice
            
            if num2AnInhAllexcEqexc == 3:
                perClassErrorTest_data_exc_all_allMice_shflTrsPerN = perClassErrorTest_data_exc_all_allMice
                perClassErrorTest_shfl_exc_all_allMice_shflTrsPerN = perClassErrorTest_shfl_exc_all_allMice
                perClassErrorTest_chance_exc_all_allMice_shflTrsPerN = perClassErrorTest_chance_exc_all_allMice
                test_dms_exc_allMice_shflTrsPerN = test_dms_exc_allMice
                
                        
            
######################################################################################################################################################        
######################################################################################################################################################        
#     Done with all mice
######################################################################################################################################################                
            
#%% Compute pvalue for each day of each mouse (using all samps, time bin -1)

pwmt_data_inh_allMice = []
pwmt_data_exc_allMice = []
pwmt_data_allExc_allMice = []

pwmt_shfl_inh_allMice = []
pwmt_shfl_exc_allMice = []
pwmt_shfl_allExc_allMice = []

pwmt_chance_inh_allMice = []
pwmt_chance_exc_allMice = []
pwmt_chance_allExc_allMice = []

pwmt_dms_inh_allMice = []
pwmt_dms_exc_allMice = []
pwmt_dms_allExc_allMice = []
                
for im in range(len(mice)):
    corr_hr_lr = corr_hr_lr_allMice[im]
    mn_corr = np.min(corr_hr_lr,axis=1) # number of trials of each class. 90% of this was used for training, and 10% for testing.
    
    ### data ###
    pwmt_data_inh_allMice.append(pwmt_frs(perClassErrorTest_data_inh_all_allMice[im][mn_corr >= thTrained,:].T, perClassErrorTest_data_inh_all_allMice_shflTrsPerN[im][mn_corr >= thTrained,:].T)) # nMice # each mouse: 3 x days  
    pwmt_data_allExc_allMice.append(pwmt_frs(perClassErrorTest_data_allExc_all_allMice[im][mn_corr >= thTrained,:].T,  perClassErrorTest_data_allExc_all_allMice_shflTrsPerN[im][mn_corr >= thTrained,:].T))
    # compute p for each excShfl, then compute ave across excShfl (do this for each day separately)
    c = np.mean([pwmt_frs(perClassErrorTest_data_exc_all_allMice[im][mn_corr >= thTrained,ies,:].T,  perClassErrorTest_data_exc_all_allMice_shflTrsPerN[im][mn_corr >= thTrained,ies,:].T) for ies in range(numExcSamples)], axis=0)
    pwmt_data_exc_allMice.append(c)
    
    ### shfl ###
    pwmt_shfl_inh_allMice.append(pwmt_frs(perClassErrorTest_shfl_inh_all_allMice[im][mn_corr >= thTrained,:].T,  perClassErrorTest_shfl_inh_all_allMice_shflTrsPerN[im][mn_corr >= thTrained,:].T))            
    pwmt_shfl_allExc_allMice.append(pwmt_frs(perClassErrorTest_shfl_allExc_all_allMice[im][mn_corr >= thTrained,:].T,  perClassErrorTest_shfl_allExc_all_allMice_shflTrsPerN[im][mn_corr >= thTrained,:].T))
    # compute p for each excShfl, then compute ave across excShfl (do this for each day separately)
    c = np.mean([pwmt_frs(perClassErrorTest_shfl_exc_all_allMice[im][mn_corr >= thTrained,ies,:].T,  perClassErrorTest_shfl_exc_all_allMice_shflTrsPerN[im][mn_corr >= thTrained,ies,:].T) for ies in range(numExcSamples)], axis=0)
    pwmt_shfl_exc_allMice.append(c)            

    ### chance ###
    pwmt_chance_inh_allMice.append(pwmt_frs(perClassErrorTest_chance_inh_all_allMice[im][mn_corr >= thTrained,:].T,  perClassErrorTest_chance_inh_all_allMice_shflTrsPerN[im][mn_corr >= thTrained,:].T))            
    pwmt_chance_allExc_allMice.append(pwmt_frs(perClassErrorTest_chance_allExc_all_allMice[im][mn_corr >= thTrained,:].T,  perClassErrorTest_chance_allExc_all_allMice_shflTrsPerN[im][mn_corr >= thTrained,:].T))
    # compute p for each excShfl, then compute ave across excShfl (do this for each day separately)
    c = np.mean([pwmt_frs(perClassErrorTest_chance_exc_all_allMice[im][mn_corr >= thTrained,ies,:].T,  perClassErrorTest_chance_exc_all_allMice_shflTrsPerN[im][mn_corr >= thTrained,ies,:].T) for ies in range(numExcSamples)], axis=0)
    pwmt_chance_exc_allMice.append(c)
    
    ### dms ### (remember dms already includes good days: [mn_corr >= thTrained,ies,:])
    pwmt_dms_inh_allMice.append(pwmt_frs(test_dms_inh_allMice[im].T, test_dms_inh_allMice_shflTrsPerN[im].T))            
    pwmt_dms_allExc_allMice.append(pwmt_frs(test_dms_allExc_allMice[im].T, test_dms_allExc_allMice_shflTrsPerN[im].T))
    # compute p for each excShfl, then compute ave across excShfl (do this for each day separately)
    c = np.mean([pwmt_frs(test_dms_exc_allMice[im][:,ies,:].T,  test_dms_exc_allMice_shflTrsPerN[im][:,ies,:].T) for ies in range(numExcSamples)], axis=0)
    pwmt_dms_exc_allMice.append(c)


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


#
av_test_data_inh_allMice_shflTrsPerN = np.array(av_test_data_inh_allMice_shflTrsPerN) #numMice; each mouse: numDays
sd_test_data_inh_allMice_shflTrsPerN = np.array(sd_test_data_inh_allMice_shflTrsPerN)
av_test_shfl_inh_allMice_shflTrsPerN = np.array(av_test_shfl_inh_allMice_shflTrsPerN)
sd_test_shfl_inh_allMice_shflTrsPerN = np.array(sd_test_shfl_inh_allMice_shflTrsPerN)
av_test_chance_inh_allMice_shflTrsPerN = np.array(av_test_chance_inh_allMice_shflTrsPerN)
sd_test_chance_inh_allMice_shflTrsPerN = np.array(sd_test_chance_inh_allMice_shflTrsPerN)

av_test_data_allExc_allMice_shflTrsPerN = np.array(av_test_data_allExc_allMice_shflTrsPerN)
sd_test_data_allExc_allMice_shflTrsPerN = np.array(sd_test_data_allExc_allMice_shflTrsPerN)
av_test_shfl_allExc_allMice_shflTrsPerN = np.array(av_test_shfl_allExc_allMice_shflTrsPerN)
sd_test_shfl_allExc_allMice_shflTrsPerN = np.array(sd_test_shfl_allExc_allMice_shflTrsPerN)
av_test_chance_allExc_allMice_shflTrsPerN = np.array(av_test_chance_allExc_allMice_shflTrsPerN)
sd_test_chance_allExc_allMice_shflTrsPerN = np.array(sd_test_chance_allExc_allMice_shflTrsPerN)

if num2AnInhAllexcEqexc == 3:
    av_test_data_exc_allMice_shflTrsPerN = np.array(av_test_data_exc_allMice_shflTrsPerN)
    sd_test_data_exc_allMice_shflTrsPerN = np.array(sd_test_data_exc_allMice_shflTrsPerN)
    av_test_shfl_exc_allMice_shflTrsPerN = np.array(av_test_shfl_exc_allMice_shflTrsPerN)
    sd_test_shfl_exc_allMice_shflTrsPerN = np.array(sd_test_shfl_exc_allMice_shflTrsPerN)
    av_test_chance_exc_allMice_shflTrsPerN = np.array(av_test_chance_exc_allMice_shflTrsPerN)
    sd_test_chance_exc_allMice_shflTrsPerN = np.array(sd_test_chance_exc_allMice_shflTrsPerN)
  
eventI_ds_allDays_allMice = np.array(eventI_ds_allDays_allMice)
eventI_allDays_allMice = np.array(eventI_allDays_allMice)


#%%
######################## PLOTS ########################

# Plot class accuracy in the frame before the choice onset for each day
# Class accuracy vs day number
def plotnow(top, tops, e, es, lab, labs, ylab, col, p):
    numDays = len(top)
    plt.errorbar(range(numDays), top, e, label=lab, color=col)
    plt.errorbar(range(numDays), tops, es, label=labs, color='g')#, linestyle='--')
    plt.xlabel('Days')
    plt.ylabel('Classification accuracy (%%)\n%s' %(ylab))
    plt.xlim([-1,numDays-1+1])
    ax = plt.gca()
    makeNicePlots(ax)
#    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)

    pp = p
    pp[p>.05] = np.nan
    yl = np.max(np.concatenate((top,tops))) #ax.get_ylim()[1]
    pp[p<=.05] = yl
    plt.plot(range(numDays), pp, marker='*',color=col, linestyle='')
    
    
#%% Plots of individual mice

pn = 2 # which p value to use: wilcoxon, MW, ttest  

for im in range(len(mice)):

    ##%%
    av_test_data_inh = av_test_data_inh_allMice[im]
    sd_test_data_inh = sd_test_data_inh_allMice[im] 
    av_test_shfl_inh = av_test_shfl_inh_allMice[im]
    sd_test_shfl_inh = sd_test_shfl_inh_allMice[im]
    av_test_chance_inh = av_test_chance_inh_allMice[im]
    sd_test_chance_inh = sd_test_chance_inh_allMice[im]
    av_test_dms_inh = av_test_dms_inh_allMice[im]
    sd_test_dms_inh = sd_test_dms_inh_allMice[im]
        
    av_test_data_allExc = av_test_data_allExc_allMice[im]
    sd_test_data_allExc = sd_test_data_allExc_allMice[im]
    av_test_shfl_allExc = av_test_shfl_allExc_allMice[im]
    sd_test_shfl_allExc = sd_test_shfl_allExc_allMice[im]
    av_test_chance_allExc = av_test_chance_allExc_allMice[im]
    sd_test_chance_allExc = sd_test_chance_allExc_allMice[im]
    av_test_dms_allExc = av_test_dms_allExc_allMice[im]
    sd_test_dms_allExc = sd_test_dms_allExc_allMice[im]
    
    if num2AnInhAllexcEqexc == 3:
        av_test_data_exc = av_test_data_exc_allMice[im]
        sd_test_data_exc = sd_test_data_exc_allMice[im]
        av_test_shfl_exc = av_test_shfl_exc_allMice[im]
        sd_test_shfl_exc = sd_test_shfl_exc_allMice[im]
        av_test_chance_exc = av_test_chance_exc_allMice[im]
        sd_test_chance_exc = sd_test_chance_exc_allMice[im]
        av_test_dms_exc = av_test_dms_exc_allMice[im]
        sd_test_dms_exc = sd_test_dms_exc_allMice[im]
                
    # _shflTrsPerN
    av_test_data_inh_shflTrsPerN = av_test_data_inh_allMice_shflTrsPerN[im]
    sd_test_data_inh_shflTrsPerN = sd_test_data_inh_allMice_shflTrsPerN[im]
    av_test_shfl_inh_shflTrsPerN = av_test_shfl_inh_allMice_shflTrsPerN[im]
    sd_test_shfl_inh_shflTrsPerN = sd_test_shfl_inh_allMice_shflTrsPerN[im]
    av_test_chance_inh_shflTrsPerN = av_test_chance_inh_allMice_shflTrsPerN[im]
    sd_test_chance_inh_shflTrsPerN = sd_test_chance_inh_allMice_shflTrsPerN[im]
    av_test_dms_inh_shflTrsPerN = av_test_dms_inh_allMice_shflTrsPerN[im]
    sd_test_dms_inh_shflTrsPerN = sd_test_dms_inh_allMice_shflTrsPerN[im]
    
    av_test_data_allExc_shflTrsPerN = av_test_data_allExc_allMice_shflTrsPerN[im]
    sd_test_data_allExc_shflTrsPerN = sd_test_data_allExc_allMice_shflTrsPerN[im]
    av_test_shfl_allExc_shflTrsPerN = av_test_shfl_allExc_allMice_shflTrsPerN[im]
    sd_test_shfl_allExc_shflTrsPerN = sd_test_shfl_allExc_allMice_shflTrsPerN[im]
    av_test_chance_allExc_shflTrsPerN = av_test_chance_allExc_allMice_shflTrsPerN[im]
    sd_test_chance_allExc_shflTrsPerN = sd_test_chance_allExc_allMice_shflTrsPerN[im]
    av_test_dms_allExc_shflTrsPerN = av_test_dms_allExc_allMice_shflTrsPerN[im]
    sd_test_dms_allExc_shflTrsPerN = sd_test_dms_allExc_allMice_shflTrsPerN[im]
    
    if num2AnInhAllexcEqexc == 3:
        av_test_data_exc_shflTrsPerN = av_test_data_exc_allMice_shflTrsPerN[im]
        sd_test_data_exc_shflTrsPerN = sd_test_data_exc_allMice_shflTrsPerN[im]
        av_test_shfl_exc_shflTrsPerN = av_test_shfl_exc_allMice_shflTrsPerN[im]
        sd_test_shfl_exc_shflTrsPerN = sd_test_shfl_exc_allMice_shflTrsPerN[im]
        av_test_chance_exc_shflTrsPerN = av_test_chance_exc_allMice_shflTrsPerN[im]
        sd_test_chance_exc_shflTrsPerN = sd_test_chance_exc_allMice_shflTrsPerN[im]
        av_test_dms_exc_shflTrsPerN = av_test_dms_exc_allMice_shflTrsPerN[im]
        sd_test_dms_exc_shflTrsPerN = sd_test_dms_exc_allMice_shflTrsPerN[im]


    # p(act,shfl) at time2an
    pwmt_data_inh_t2an = pwmt_data_inh_allMice[im][pn,:].squeeze()
    pwmt_shfl_inh_t2an = pwmt_shfl_inh_allMice[im][pn,:].squeeze()
    pwmt_chance_inh_t2an = pwmt_chance_inh_allMice[im][pn,:].squeeze()    
    pwmt_dms_inh_t2an = pwmt_dms_inh_allMice[im][pn,:].squeeze()    

    pwmt_data_exc_t2an = pwmt_data_exc_allMice[im][pn,:].squeeze()
    pwmt_shfl_exc_t2an = pwmt_shfl_exc_allMice[im][pn,:].squeeze()
    pwmt_chance_exc_t2an = pwmt_chance_exc_allMice[im][pn,:].squeeze()    
    pwmt_dms_exc_t2an = pwmt_dms_exc_allMice[im][pn,:].squeeze()    

    pwmt_data_allExc_t2an = pwmt_data_allExc_allMice[im][pn,:].squeeze()
    pwmt_shfl_allExc_t2an = pwmt_shfl_allExc_allMice[im][pn,:].squeeze()
    pwmt_chance_allExc_t2an = pwmt_chance_allExc_allMice[im][pn,:].squeeze()    
    pwmt_dms_allExc_t2an = pwmt_dms_allExc_allMice[im][pn,:].squeeze()    

    
    ###############
    ##%% Compare class accuracies of actual with trsShuffledPerNeuron
    # Plot class accuracy in the frame before the choice onset for each day
    # Class accuracy vs day number

    plt.figure(figsize=(6,20))           
    ylab = 'data-shfl , shfl, data'    
    
    ############ data - shuffle ############ 
    ##### inh
    top = av_test_dms_inh #av_test_data_inh - av_test_shfl_inh
    tops = av_test_dms_inh_shflTrsPerN #av_test_data_inh_shflTrsPerN - av_test_shfl_inh_shflTrsPerN
    e = sd_test_dms_inh # same as sd_test_data_inh bc sd_test_dms_inh = sd(test_data_inh - ave(test_shfl_inh))
    es = sd_test_dms_inh_shflTrsPerN
    p = pwmt_dms_inh_t2an + 0
    lab = 'inh'
    labs = 'inh_shflTrsPerN'    
    col = 'r'
    plt.subplot(311)
    plotnow(top, tops, e, es, lab, labs, ylab, col,p)
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)


    ##### exc
    top = av_test_dms_exc #av_test_data_exc - av_test_shfl_exc
    tops = av_test_dms_exc_shflTrsPerN #av_test_data_exc_shflTrsPerN - av_test_shfl_exc_shflTrsPerN
    e = sd_test_dms_exc #sd_test_data_exc
    es = sd_test_dms_exc_shflTrsPerN #sd_test_data_exc_shflTrsPerN
    p = pwmt_dms_exc_t2an + 0
    lab = 'exc'
    labs = 'exc_shflTrsPerN'
    col = 'b'
    plt.subplot(312)
    plotnow(top, tops, e, es, lab, labs, ylab, col,p)
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)


    ##### allN
    top = av_test_dms_allExc #av_test_data_allExc - av_test_shfl_allExc
    tops = av_test_dms_allExc_shflTrsPerN #av_test_data_allExc_shflTrsPerN - av_test_shfl_allExc_shflTrsPerN
    e = sd_test_dms_allExc
    es = sd_test_dms_allExc_shflTrsPerN
    p = pwmt_dms_allExc_t2an + 0
    lab = 'allN'
    labs = 'allN_shflTrsPerN'
    col = 'k'    
    plt.subplot(313)
    plotnow(top, tops, e, es, lab, labs, ylab, col,p)
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)



    ############ data ############ 
#    ylab = 'data'        
    ##### inh
    top = av_test_data_inh
    tops = av_test_data_inh_shflTrsPerN
    e = sd_test_data_inh
    es = sd_test_data_inh_shflTrsPerN
    p = pwmt_data_inh_t2an + 0
    lab = 'inh'
    labs = 'inh_shflTrsPerN'    
    col = 'r'
    plt.subplot(311)
    plotnow(top, tops, e, es, lab, labs, ylab, col,p)
    ax = plt.gca()
    ylim = ax.get_ylim()
    plt.ylim([ylim[0], ylim[1]+5])    

    ##### exc
    top = av_test_data_exc
    tops = av_test_data_exc_shflTrsPerN
    e = sd_test_data_exc
    es = sd_test_data_exc_shflTrsPerN
    p = pwmt_data_exc_t2an + 0
    lab = 'exc'
    labs = 'exc_shflTrsPerN'
    col = 'b'
    plt.subplot(312)
    plotnow(top, tops, e, es, lab, labs, ylab, col,p)
    ax = plt.gca()
    ylim = ax.get_ylim()
    plt.ylim([ylim[0], ylim[1]+5])    


    ##### allN
    top = av_test_data_allExc
    tops = av_test_data_allExc_shflTrsPerN
    e = sd_test_data_allExc
    es = sd_test_data_allExc_shflTrsPerN
    p = pwmt_data_allExc_t2an + 0
    lab = 'allN'
    labs = 'allN_shflTrsPerN'
    col = 'k'    
    plt.subplot(313)
    plotnow(top, tops, e, es, lab, labs, ylab, col,p)
    ax = plt.gca()
    ylim = ax.get_ylim()
    plt.ylim([ylim[0], ylim[1]+5])    



    ############ shfl ############ 
#    ylab = 'shfl'        
    ##### inh
    top = av_test_shfl_inh
    tops = av_test_shfl_inh_shflTrsPerN
    e = sd_test_shfl_inh
    es = sd_test_shfl_inh_shflTrsPerN
    p = pwmt_shfl_inh_t2an + 0
    lab = 'inh'
    labs = 'inh_shflTrsPerN'    
    col = 'r'
    plt.subplot(311)
    plotnow(top, tops, e, es, lab, labs, ylab, col,p)
    

    ##### exc
    top = av_test_shfl_exc
    tops = av_test_shfl_exc_shflTrsPerN
    e = sd_test_shfl_exc
    es = sd_test_shfl_exc_shflTrsPerN
    p = pwmt_shfl_exc_t2an + 0
    lab = 'exc'
    labs = 'exc_shflTrsPerN'
    col = 'b'
    plt.subplot(312)
    plotnow(top, tops, e, es, lab, labs, ylab, col,p)
    

    ##### allN
    top = av_test_shfl_allExc
    tops = av_test_shfl_allExc_shflTrsPerN
    e = sd_test_shfl_allExc
    es = sd_test_shfl_allExc_shflTrsPerN
    p = pwmt_shfl_allExc_t2an + 0
    lab = 'allN'
    labs = 'allN_shflTrsPerN'
    col = 'k'   
    plt.subplot(313)
    plotnow(top, tops, e, es, lab, labs, ylab, col,p)
    


    ############ Save fig ############
    if savefigs:#% Save the figure
        mousename = mice[im]
        dnow = '/excInh_trainDecoder_eachFrame/'+mousename+'/trsShfledPerNeuron/'        
            
        if chAl==1:
            dd = 'chAl_eachDay_time' + str(time2an) + '_shflVSact_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_eachDay_time' + str(time2an) + '_shflVSact_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        




#%%
###################### Plots of all mice ######################   
#%% Compute p values among sessions (samp aved) for each mouse

pwm_data_inh_mouse = np.full((3,len(mice)), np.nan)
pwm_data_exc_mouse = np.full((3,len(mice)), np.nan)
pwm_data_allExc_mouse = np.full((3,len(mice)), np.nan)
pwm_shfl_inh_mouse = np.full((3,len(mice)), np.nan)
pwm_shfl_exc_mouse = np.full((3,len(mice)), np.nan)
pwm_shfl_allExc_mouse = np.full((3,len(mice)), np.nan)
pwm_chance_inh_mouse = np.full((3,len(mice)), np.nan)
pwm_chance_exc_mouse = np.full((3,len(mice)), np.nan)
pwm_chance_allExc_mouse = np.full((3,len(mice)), np.nan)
pwm_dms_inh_mouse = np.full((3,len(mice)), np.nan)
pwm_dms_exc_mouse = np.full((3,len(mice)), np.nan)
pwm_dms_allExc_mouse = np.full((3,len(mice)), np.nan)

for im in range(len(mice)):
    pwm_data_inh_mouse[:,im] = pwm(av_test_data_inh_allMice[im], av_test_data_inh_allMice_shflTrsPerN[im])
    pwm_data_exc_mouse[:,im] = pwm(av_test_data_exc_allMice[im], av_test_data_exc_allMice_shflTrsPerN[im])
    pwm_data_allExc_mouse[:,im] = pwm(av_test_data_allExc_allMice[im], av_test_data_allExc_allMice_shflTrsPerN[im])
    pwm_shfl_inh_mouse[:,im] = pwm(av_test_shfl_inh_allMice[im], av_test_shfl_inh_allMice_shflTrsPerN[im])
    pwm_shfl_exc_mouse[:,im] = pwm(av_test_shfl_exc_allMice[im], av_test_shfl_exc_allMice_shflTrsPerN[im])
    pwm_shfl_allExc_mouse[:,im] = pwm(av_test_shfl_allExc_allMice[im], av_test_shfl_allExc_allMice_shflTrsPerN[im])
    pwm_chance_inh_mouse[:,im] = pwm(av_test_chance_inh_allMice[im], av_test_chance_inh_allMice_shflTrsPerN[im])
    pwm_chance_exc_mouse[:,im] = pwm(av_test_chance_exc_allMice[im], av_test_chance_exc_allMice_shflTrsPerN[im])
    pwm_chance_allExc_mouse[:,im] = pwm(av_test_chance_allExc_allMice[im], av_test_chance_allExc_allMice_shflTrsPerN[im])
    pwm_dms_inh_mouse[:,im] = pwm(av_test_dms_inh_allMice[im], av_test_dms_inh_allMice_shflTrsPerN[im])
    pwm_dms_exc_mouse[:,im] = pwm(av_test_dms_exc_allMice[im], av_test_dms_exc_allMice_shflTrsPerN[im])
    pwm_dms_allExc_mouse[:,im] = pwm(av_test_dms_allExc_allMice[im], av_test_dms_allExc_allMice_shflTrsPerN[im])


#%% Average and sd (not se) of classErr across sessions for each mouse

numMice = len(mice)
numDays_allMice = np.array([len(av_test_data_inh_allMice[im]) for im in range(numMice)])

av_av_test_data_inh_allMice = np.array([np.nanmean(av_test_data_inh_allMice[im], axis=0) for im in range(numMice)]) # numMice
sd_av_test_data_inh_allMice = np.array([np.nanstd(av_test_data_inh_allMice[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
av_av_test_shfl_inh_allMice = np.array([np.nanmean(av_test_shfl_inh_allMice[im], axis=0) for im in range(numMice)])
sd_av_test_shfl_inh_allMice = np.array([np.nanstd(av_test_shfl_inh_allMice[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
av_av_test_chance_inh_allMice = np.array([np.nanmean(av_test_chance_inh_allMice[im], axis=0) for im in range(numMice)])
sd_av_test_chance_inh_allMice = np.array([np.nanstd(av_test_chance_inh_allMice[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
av_av_test_dms_inh_allMice = np.array([np.nanmean(av_test_dms_inh_allMice[im], axis=0) for im in range(numMice)])
sd_av_test_dms_inh_allMice = np.array([np.nanstd(av_test_dms_inh_allMice[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
#av_test_dms_inh_allMice = np.array([av_test_data_inh_allMice[im] - np.mean(av_av_test_shfl_inh_allMice[im]) for im in range(numMice)]) # for each mouse subtract day-ave-shfl from individual day data


av_av_test_data_allExc_allMice = np.array([np.nanmean(av_test_data_allExc_allMice[im], axis=0) for im in range(numMice)])
sd_av_test_data_allExc_allMice = np.array([np.nanstd(av_test_data_allExc_allMice[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
av_av_test_shfl_allExc_allMice = np.array([np.nanmean(av_test_shfl_allExc_allMice[im], axis=0) for im in range(numMice)])
sd_av_test_shfl_allExc_allMice = np.array([np.nanstd(av_test_shfl_allExc_allMice[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
av_av_test_chance_allExc_allMice = np.array([np.nanmean(av_test_chance_allExc_allMice[im], axis=0) for im in range(numMice)])
sd_av_test_chance_allExc_allMice = np.array([np.nanstd(av_test_chance_allExc_allMice[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
av_av_test_dms_allExc_allMice = np.array([np.nanmean(av_test_dms_allExc_allMice[im], axis=0) for im in range(numMice)])
sd_av_test_dms_allExc_allMice = np.array([np.nanstd(av_test_dms_allExc_allMice[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])

if num2AnInhAllexcEqexc == 3:
    av_av_test_data_exc_allMice = np.array([np.nanmean(av_test_data_exc_allMice[im], axis=0) for im in range(numMice)])
    sd_av_test_data_exc_allMice = np.array([np.nanstd(av_test_data_exc_allMice[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
    av_av_test_shfl_exc_allMice = np.array([np.nanmean(av_test_shfl_exc_allMice[im], axis=0) for im in range(numMice)])
    sd_av_test_shfl_exc_allMice = np.array([np.nanstd(av_test_shfl_exc_allMice[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
    av_av_test_chance_exc_allMice = np.array([np.nanmean(av_test_chance_exc_allMice[im], axis=0) for im in range(numMice)])
    sd_av_test_chance_exc_allMice = np.array([np.nanstd(av_test_chance_exc_allMice[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
    av_av_test_dms_exc_allMice = np.array([np.nanmean(av_test_dms_exc_allMice[im], axis=0) for im in range(numMice)])    
    sd_av_test_dms_exc_allMice = np.array([np.nanstd(av_test_dms_exc_allMice[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
    

##### shflTrsPerNeur
av_av_test_data_inh_allMice_shflTrsPerN = np.array([np.nanmean(av_test_data_inh_allMice_shflTrsPerN[im], axis=0) for im in range(numMice)]) # numMice
sd_av_test_data_inh_allMice_shflTrsPerN = np.array([np.nanstd(av_test_data_inh_allMice_shflTrsPerN[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
av_av_test_shfl_inh_allMice_shflTrsPerN = np.array([np.nanmean(av_test_shfl_inh_allMice_shflTrsPerN[im], axis=0) for im in range(numMice)])
sd_av_test_shfl_inh_allMice_shflTrsPerN = np.array([np.nanstd(av_test_shfl_inh_allMice_shflTrsPerN[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
av_av_test_chance_inh_allMice_shflTrsPerN = np.array([np.nanmean(av_test_chance_inh_allMice_shflTrsPerN[im], axis=0) for im in range(numMice)])
sd_av_test_chance_inh_allMice_shflTrsPerN = np.array([np.nanstd(av_test_chance_inh_allMice_shflTrsPerN[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
av_av_test_dms_inh_allMice_shflTrsPerN = np.array([np.nanmean(av_test_dms_inh_allMice_shflTrsPerN[im], axis=0) for im in range(numMice)])
sd_av_test_dms_inh_allMice_shflTrsPerN = np.array([np.nanstd(av_test_dms_inh_allMice_shflTrsPerN[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])


av_av_test_data_allExc_allMice_shflTrsPerN = np.array([np.nanmean(av_test_data_allExc_allMice_shflTrsPerN[im], axis=0) for im in range(numMice)])
sd_av_test_data_allExc_allMice_shflTrsPerN = np.array([np.nanstd(av_test_data_allExc_allMice_shflTrsPerN[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
av_av_test_shfl_allExc_allMice_shflTrsPerN = np.array([np.nanmean(av_test_shfl_allExc_allMice_shflTrsPerN[im], axis=0) for im in range(numMice)])
sd_av_test_shfl_allExc_allMice_shflTrsPerN = np.array([np.nanstd(av_test_shfl_allExc_allMice_shflTrsPerN[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
av_av_test_chance_allExc_allMice_shflTrsPerN = np.array([np.nanmean(av_test_chance_allExc_allMice_shflTrsPerN[im], axis=0) for im in range(numMice)])
sd_av_test_chance_allExc_allMice_shflTrsPerN = np.array([np.nanstd(av_test_chance_allExc_allMice_shflTrsPerN[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
av_av_test_dms_allExc_allMice_shflTrsPerN = np.array([np.nanmean(av_test_dms_allExc_allMice_shflTrsPerN[im], axis=0) for im in range(numMice)])
sd_av_test_dms_allExc_allMice_shflTrsPerN = np.array([np.nanstd(av_test_dms_allExc_allMice_shflTrsPerN[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])

if num2AnInhAllexcEqexc == 3:
    av_av_test_data_exc_allMice_shflTrsPerN = np.array([np.nanmean(av_test_data_exc_allMice_shflTrsPerN[im], axis=0) for im in range(numMice)])
    sd_av_test_data_exc_allMice_shflTrsPerN = np.array([np.nanstd(av_test_data_exc_allMice_shflTrsPerN[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
    av_av_test_shfl_exc_allMice_shflTrsPerN = np.array([np.nanmean(av_test_shfl_exc_allMice_shflTrsPerN[im], axis=0) for im in range(numMice)])
    sd_av_test_shfl_exc_allMice_shflTrsPerN = np.array([np.nanstd(av_test_shfl_exc_allMice_shflTrsPerN[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
    av_av_test_chance_exc_allMice_shflTrsPerN = np.array([np.nanmean(av_test_chance_exc_allMice_shflTrsPerN[im], axis=0) for im in range(numMice)])
    sd_av_test_chance_exc_allMice_shflTrsPerN = np.array([np.nanstd(av_test_chance_exc_allMice_shflTrsPerN[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
    av_av_test_dms_exc_allMice_shflTrsPerN = np.array([np.nanmean(av_test_dms_exc_allMice_shflTrsPerN[im], axis=0) for im in range(numMice)])
    sd_av_test_dms_exc_allMice_shflTrsPerN = np.array([np.nanstd(av_test_dms_exc_allMice_shflTrsPerN[im], axis=0)/np.sqrt(numDays_allMice[im]) for im in range(numMice)])
    

#%% Plot classErr (averaged across all sessions) for each mouse
# compare actual with shflTrsPerNeuron

x = np.arange(numMice)*2.5
a = .7

plt.figure(figsize=(4.8,3))


############### data and shfl ###############
plt.subplot(121)

# testing data
plt.errorbar(x, av_av_test_data_inh_allMice, sd_av_test_data_inh_allMice, fmt='o', label='inh', color='r')
plt.errorbar(x, av_av_test_data_exc_allMice, sd_av_test_data_exc_allMice, fmt='o', label='exc', color='b')
plt.errorbar(x, av_av_test_data_allExc_allMice, sd_av_test_data_allExc_allMice, fmt='o', label=labAll, color='k')

# shfl
plt.errorbar(x, av_av_test_shfl_inh_allMice, sd_av_test_shfl_inh_allMice, color='r', fmt='o', alpha=.3)
plt.errorbar(x, av_av_test_shfl_exc_allMice, sd_av_test_shfl_exc_allMice, color='b', fmt='o', alpha=.3)
plt.errorbar(x, av_av_test_shfl_allExc_allMice, sd_av_test_shfl_allExc_allMice, color='k', fmt='o', alpha=.3)


## shflTrsPerNeuron
plt.errorbar(x+a, av_av_test_data_inh_allMice_shflTrsPerN, sd_av_test_data_inh_allMice_shflTrsPerN, fmt='o', color='r', markerfacecolor='w', markeredgecolor='r') #, label='inh'
plt.errorbar(x+a, av_av_test_data_exc_allMice_shflTrsPerN, sd_av_test_data_exc_allMice_shflTrsPerN, fmt='o', color='b', markerfacecolor='w', markeredgecolor='b') # , label='exc'
plt.errorbar(x+a, av_av_test_data_allExc_allMice_shflTrsPerN, sd_av_test_data_allExc_allMice_shflTrsPerN, fmt='o', color='k', markerfacecolor='w', markeredgecolor='k') # , label=labAll

# shfl
plt.errorbar(x+a, av_av_test_shfl_inh_allMice_shflTrsPerN, sd_av_test_shfl_inh_allMice_shflTrsPerN, color='r', fmt='o', alpha=.3, markerfacecolor='w', markeredgecolor='r')
plt.errorbar(x+a, av_av_test_shfl_exc_allMice_shflTrsPerN, sd_av_test_shfl_exc_allMice_shflTrsPerN, color='b', fmt='o', alpha=.3, markerfacecolor='w', markeredgecolor='b')
plt.errorbar(x+a, av_av_test_shfl_allExc_allMice_shflTrsPerN, sd_av_test_shfl_allExc_allMice_shflTrsPerN, color='k', fmt='o', alpha=.3, markerfacecolor='w', markeredgecolor='k')

#plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1)#, frameon=False) 
plt.xlabel('Mice', fontsize=11)
plt.ylabel('Classification accuracy (%)\n[-97 0] ms rel. choice', fontsize=11)
plt.xlim([x[0]-a, x[-1]+a+a])
plt.xticks(x+a/2.,mice)
ax = plt.gca()
makeNicePlots(ax)

# p values: data (act vs shftTrsPerN)
print pwm_data_inh_mouse, pwm_data_exc_mouse, pwm_data_allExc_mouse
# p values: shfl (act vs shftTrsPerN)
print pwm_shfl_inh_mouse, pwm_shfl_exc_mouse, pwm_shfl_allExc_mouse
#### Mark mice with sig p (inh: act vs shflTrsPerN)
b = 0#.1
# inh
pp = np.full(len(pwm_data_inh_mouse[pn,:]), np.nan)
pp[pwm_data_inh_mouse[pn,:]<=.05] = av_av_test_data_inh_allMice_shflTrsPerN[pwm_data_inh_mouse[pn,:]<=.05]
plt.plot(x+a+b, pp, 'g', marker='.', linestyle='')

# exc
pp = np.full(len(pwm_data_exc_mouse[pn,:]), np.nan)
pp[pwm_data_exc_mouse[pn,:]<=.05] = av_av_test_data_exc_allMice_shflTrsPerN[pwm_data_exc_mouse[pn,:]<=.05]
plt.plot(x+a+b, pp, 'g', marker='.', linestyle='')

# allExc
pp = np.full(len(pwm_data_allExc_mouse[pn,:]), np.nan)
pp[pwm_data_allExc_mouse[pn,:]<=.05] = av_av_test_data_allExc_allMice_shflTrsPerN[pwm_data_allExc_mouse[pn,:]<=.05]
plt.plot(x+a+b, pp, 'g', marker='.', linestyle='')


    
############### data - shfl ###############
plt.subplot(122)

# testing data
plt.errorbar(x, av_av_test_dms_inh_allMice , sd_av_test_dms_inh_allMice, fmt='o', label='inh', color='r')
plt.errorbar(x, av_av_test_dms_exc_allMice , sd_av_test_dms_exc_allMice, fmt='o', label='exc', color='b')
plt.errorbar(x, av_av_test_dms_allExc_allMice, sd_av_test_dms_allExc_allMice, fmt='o', label=labAll, color='k')
#plt.errorbar(x, av_av_test_data_inh_allMice - av_av_test_shfl_inh_allMice , sd_av_test_data_inh_allMice, fmt='o', label='inh', color='r')
#plt.errorbar(x, av_av_test_data_exc_allMice - av_av_test_shfl_exc_allMice , sd_av_test_data_exc_allMice, fmt='o', label='exc', color='b')
#plt.errorbar(x, av_av_test_data_allExc_allMice - av_av_test_shfl_allExc_allMice, sd_av_test_data_allExc_allMice, fmt='o', label=labAll, color='k')


## shflTrsPerNeuron
plt.errorbar(x+a, av_av_test_dms_inh_allMice_shflTrsPerN , sd_av_test_dms_inh_allMice_shflTrsPerN, fmt='o', label='inh', color='r', markerfacecolor='w', markeredgecolor='r')
plt.errorbar(x+a, av_av_test_dms_exc_allMice_shflTrsPerN , sd_av_test_dms_exc_allMice_shflTrsPerN, fmt='o', label='exc', color='b', markerfacecolor='w', markeredgecolor='b')
plt.errorbar(x+a, av_av_test_dms_allExc_allMice_shflTrsPerN, sd_av_test_dms_allExc_allMice_shflTrsPerN, fmt='o', label=labAll, color='k', markerfacecolor='w', markeredgecolor='k')
#plt.errorbar(x+a, av_av_test_data_inh_allMice_shflTrsPerN - av_av_test_shfl_inh_allMice_shflTrsPerN, sd_av_test_data_inh_allMice_shflTrsPerN, fmt='o', color='r', markerfacecolor='w', markeredgecolor='r') #, label='inh'
#plt.errorbar(x+a, av_av_test_data_exc_allMice_shflTrsPerN - av_av_test_shfl_exc_allMice_shflTrsPerN, sd_av_test_data_exc_allMice_shflTrsPerN, fmt='o', color='b', markerfacecolor='w', markeredgecolor='b') # , label='exc'
#plt.errorbar(x+a, av_av_test_data_allExc_allMice_shflTrsPerN - av_av_test_shfl_allExc_allMice_shflTrsPerN, sd_av_test_data_allExc_allMice_shflTrsPerN, fmt='o', color='k', markerfacecolor='w', markeredgecolor='k') # , label=labAll

plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1)#, frameon=False) 
plt.xlabel('Mice', fontsize=11)
#plt.ylabel('Classification accuracy (%)\n[-97 0] ms rel. choice', fontsize=11)
plt.title('Data - Shfl')
plt.xlim([x[0]-a, x[-1]+a+a])
plt.xticks(x+a/2.,mice)
ax = plt.gca()
makeNicePlots(ax)

# p values: data-shfl (act vs shftTrsPerN)
print pwm_dms_inh_mouse, pwm_dms_exc_mouse, pwm_dms_allExc_mouse
#### Mark mice with sig p (inh: act vs shflTrsPerN)
# inh
pp = np.full(len(pwm_dms_inh_mouse[pn,:]), np.nan)
pp[pwm_dms_inh_mouse[pn,:]<=.05] = av_av_test_dms_inh_allMice_shflTrsPerN[pwm_dms_inh_mouse[pn,:]<=.05]
plt.plot(x+a+b, pp, 'g', marker='.', linestyle='')

# exc
pp = np.full(len(pwm_dms_exc_mouse[pn,:]), np.nan)
pp[pwm_dms_exc_mouse[pn,:]<=.05] = av_av_test_dms_exc_allMice_shflTrsPerN[pwm_dms_exc_mouse[pn,:]<=.05]
plt.plot(x+a+b, pp, 'g', marker='.', linestyle='')

# allExc
pp = np.full(len(pwm_dms_allExc_mouse[pn,:]), np.nan)
pp[pwm_dms_allExc_mouse[pn,:]<=.05] = av_av_test_dms_allExc_allMice_shflTrsPerN[pwm_dms_allExc_mouse[pn,:]<=.05]
plt.plot(x+a+b, pp, 'g', marker='.', linestyle='')



plt.subplots_adjust(wspace=0.5)    


if savefigs:#% Save the figure
    sn = '_shflTrsPerN'

    if chAl==1:
        dd = 'chAl_aveDays_time' + str(time2an) + sn + '_shflVSact_' + '_'.join(mice) + '_' + nowStr # + days[0][0:6] + '-to-' + days[-1][0:6]
    else:
        dd = 'stAl_aveDays_time' + str(time2an) + sn + '_shflVSact_' + '_'.join(mice) + '_' + nowStr # + days[0][0:6] + '-to-' + days[-1][0:6]
        
    d = os.path.join(svmdir+dnow0)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])

    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)




