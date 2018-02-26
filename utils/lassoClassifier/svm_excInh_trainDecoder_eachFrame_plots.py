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

mousename = 'fni18' #'fni17'

addNs_roc = 1 # if 1 do the following analysis: add neurons 1 by 1 to the decoder based on their tuning strength to see how the decoder performance increases.
do_excInhHalf = 0 # 0: Load vars for inh,exc,allExc, 1: Load exc,inh SVM vars for excInhHalf (ie when the population consists of half exc and half inh) and allExc2inhSize (ie when populatin consists of allExc but same size as 2*inh size)
shflTrsEachNeuron = 0  # Set to 0 for normal SVM training. # Shuffle trials in X_svm (for each neuron independently) to break correlations between neurons in each trial.
savefigs = 1

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

chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]

if do_excInhHalf:
    doAllN=0
    labInh = 'InhExcHalf'
else:
    labInh = 'inh'
        
if doAllN==1:
    labAll = 'allN'
else:
    labAll = 'allExc'

#if loadInhAllexcEqexc==1:
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
days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays)
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
#% 
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
        perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, w_data_inh, w_data_allExc, b_data_inh, b_data_allExc, svmName_excInh = loadSVM_excInh_addNs1by1(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, loadWeights, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0)
        perClassErrorTest_data_exc = 0; perClassErrorTest_shfl_exc = 0; perClassErrorTest_chance_exc = 0; 
        
    elif do_excInhHalf:
        # numShufflesExc x numSamples x numFrames
        perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, w_data_inh, w_data_allExc, b_data_inh, b_data_allExc, svmName_excInh = loadSVM_excInh_excInhHalf(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, loadWeights, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0)
        perClassErrorTest_data_exc = 0; perClassErrorTest_shfl_exc = 0; perClassErrorTest_chance_exc = 0; 

    else:
        perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN = loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron)
    
    ##%% Get number of inh and exc        
    if loadWeights==1:
        numInh[iday] = w_data_inh.shape[1]
        numAllexc[iday] = w_data_allExc.shape[1]
        
        
    #%% Once done with all frames, save vars for all days
    
    perClassErrorTest_data_inh_all.append(perClassErrorTest_data_inh) # each day: samps x numFrs    # if addNs_roc : number of neurons in the decoder x nSamps x nFrs    
    perClassErrorTest_shfl_inh_all.append(perClassErrorTest_shfl_inh)
    perClassErrorTest_chance_inh_all.append(perClassErrorTest_chance_inh)
    perClassErrorTest_data_allExc_all.append(perClassErrorTest_data_allExc) # each day: samps x numFrs    # if addNs_roc : number of neurons in the decoder x nSamps x nFrs
    perClassErrorTest_shfl_allExc_all.append(perClassErrorTest_shfl_allExc)
    perClassErrorTest_chance_allExc_all.append(perClassErrorTest_chance_allExc) 
    if np.logical_and(do_excInhHalf==0, addNs_roc==0):
        perClassErrorTest_data_exc_all.append(perClassErrorTest_data_exc) # each day: numShufflesExc x numSamples x numFrames    
        perClassErrorTest_shfl_exc_all.append(perClassErrorTest_shfl_exc)
        perClassErrorTest_chance_exc_all.append(perClassErrorTest_chance_exc)

    # Delete vars before starting the next day    
    del perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc
    if loadWeights==1:
        del w_data_inh, w_data_exc, w_data_allExc
        

eventI_allDays = eventI_allDays.astype(int)   
eventI_ds_allDays = eventI_ds_allDays.astype(int)
numD = len(eventI_allDays)


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
    numSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc \
        = av_se_CA_trsamps_addNs1by1(numD, perClassErrorTest_data_inh_all, perClassErrorTest_shfl_inh_all, perClassErrorTest_chance_inh_all, perClassErrorTest_data_allExc_all, perClassErrorTest_shfl_allExc_all, perClassErrorTest_chance_allExc_all, fr2an, eventI_ds_allDays)
    
elif do_excInhHalf:
    numSamples, numExcSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc \
        = av_se_CA_trsamps_excInhHalf(numD, perClassErrorTest_data_inh_all, perClassErrorTest_shfl_inh_all, perClassErrorTest_chance_inh_all, perClassErrorTest_data_allExc_all, perClassErrorTest_shfl_allExc_all, perClassErrorTest_chance_allExc_all)

else:
    numSamples, numExcSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, av_test_data_exc, sd_test_data_exc, av_test_shfl_exc, sd_test_shfl_exc, av_test_chance_exc, sd_test_chance_exc, av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc \
        = av_se_CA_trsamps(numD, perClassErrorTest_data_inh_all, perClassErrorTest_shfl_inh_all, perClassErrorTest_chance_inh_all, perClassErrorTest_data_exc_all, perClassErrorTest_shfl_exc_all, perClassErrorTest_chance_exc_all, perClassErrorTest_data_allExc_all, perClassErrorTest_shfl_allExc_all, perClassErrorTest_chance_allExc_all)


#%% Keep vars for chAl and stAl

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
    
    if np.logical_and(do_excInhHalf==0, addNs_roc==0):
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


#%% Decide what days to analyze: exclude days with too few trials used for training SVM, also exclude incorr from days with too few incorr trials.

# th for min number of trs of each class
'''
thTrained = 30 #25; # 1/10 of this will be the testing tr num! and 9/10 was used for training
thIncorr = 4 #5
'''
mn_corr = np.min(corr_hr_lr,axis=1) # number of trials of each class. 90% of this was used for training, and 10% for testing.

print '%d days will be excluded: too few trials for svm training' %(sum(mn_corr < thTrained))

print np.array(days)[mn_corr < thTrained]



#%%
##################################################################################################
##################################################################################################
##################################################################################################
# Plots of adding neurons 1 by 1:

#%% Plot each day: how choice prediction varies by increasing the population size, compare exc vs inh

if addNs_roc:
    
    dnow = '/excInh_trainDecoder_eachFrame_addNs1by1ROC/'+mousename+'/'
    alph = .5
    colors = ['b','r']
    labs = ['exc','inh']
    
    plt.figure(figsize=(3, 3.5*numDays))    
    cnt = 0
    for iday in range(numDays):    
        if mn_corr[iday] >= thTrained:
            cnt = cnt+1
                   
            plt.subplot(numDays,1,cnt)
            i = 0
            x = np.arange(1, len(av_test_data_allExc_ch[iday])+1)
            av = av_test_data_allExc_ch[iday]
            sd = sd_test_data_allExc_ch[iday]
            
            plt.fill_between(x, av-sd, av+sd, alpha=alph, edgecolor=colors[i], facecolor=colors[i])
            plt.plot(np.arange(1, len(av_test_data_allExc_ch[iday])+1), av_test_data_allExc_ch[iday], colors[i], label=labs[i])
            
            
            i = 1
            x = np.arange(1, len(av_test_data_inh_ch[iday])+1)
            av = av_test_data_inh_ch[iday]
            sd = sd_test_data_inh_ch[iday]
            
            plt.fill_between(x, av-sd, av+sd, alpha=alph, edgecolor=colors[i], facecolor=colors[i])
            plt.plot(np.arange(1, len(av_test_data_inh_ch[iday])+1), av_test_data_inh_ch[iday], colors[i], label=labs[i])
            
            plt.title(days[iday])        
            plt.xlim([-5, len(av_test_data_allExc_ch[iday])+4])  
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
        
        
        
    #%% Plot average and se across days: SVM performance vs number of neurons in the decoder
    
    ########### add nan for days with num neurons < 20th percentile of nInh (or nExc)... so you can get an average across days
    numInh = np.array([len(av_test_data_inh_ch[iday]) for iday in range(numDays)])
    numExc = np.array([len(av_test_data_allExc_ch[iday]) for iday in range(numDays)])
    
    mnInh = min(numInh[mn_corr>=thTrained])
    mnExc = min(numExc[mn_corr>=thTrained])
    
    pInh = np.percentile(numInh,20).astype(int)
    pExc = np.percentile(numExc,20).astype(int)
    
    #_, inh_nPreMin, inh_nPostMin = set_nprepost(av_test_data_inh_ch, nInhPerDay, mn_corr, thTrained, regressBins)
    #_, exc_nPreMin, exc_nPostMin = set_nprepost(av_test_data_allExc_ch, nExcPerDay, mn_corr, thTrained, regressBins)    
    #avDays_av_test_data_inh_ch = np.array([av_test_data_inh_ch[iday][0:mnInh+1] for iday in range(numDays)])
    
    def sameMaxNs(CAav_alig, mxNumNeur):
        CAav_alig_sameMaxN = []
        for iday in range(len(days)):
            a = np.full((mxNumNeur), np.nan)
            a[0: min(mxNumNeur, CAav_alig[iday].shape[0])] = CAav_alig[iday][0: min(mxNumNeur, CAav_alig[iday].shape[0])]
            CAav_alig_sameMaxN.append(a) # numDays x max_numNeurs_used_for_training x nFrs    
        CAav_alig_sameMaxN = np.array(CAav_alig_sameMaxN)
        return CAav_alig_sameMaxN 
            
            
    av_test_data_inh_samePN = sameMaxNs(av_test_data_inh, pInh) # numDays x nNs in the decoder
    av_test_data_allExc_samePN = sameMaxNs(av_test_data_allExc, pExc)
    
    # for each value of number of neurons in the decoder how many days contributed (ie had non-nan values):
    nDaysPinh = np.array([np.sum(~np.isnan(av_test_data_inh_samePN[:,iN])) for iN in range(av_test_data_inh_samePN.shape[1])])
    nDaysPexc = np.array([np.sum(~np.isnan(av_test_data_allExc_samePN[:,iN])) for iN in range(av_test_data_allExc_samePN.shape[1])])
    
    # Average and se across days 
    avD_av_test_data_inh_samePN = np.nanmean(av_test_data_inh_samePN[mn_corr>=thTrained], axis=0) # nInh (20th perc)
    seD_av_test_data_inh_samePN = np.nanstd(av_test_data_inh_samePN[mn_corr>=thTrained], axis=0) / np.sqrt(nDaysPinh) # nInh (20th perc)
    
    avD_av_test_data_allExc_samePN = np.nanmean(av_test_data_allExc_samePN[mn_corr>=thTrained], axis=0) # nExc (20th perc)
    seD_av_test_data_allExc_samePN = np.nanstd(av_test_data_allExc_samePN[mn_corr>=thTrained], axis=0) / np.sqrt(nDaysPexc)
    
    
    
    ###################### plot ave +/- se across days
    plt.figure(figsize=(4.5,3))
    
    i = 0
    x = np.arange(1, len(avD_av_test_data_allExc_samePN)+1)
    av = avD_av_test_data_allExc_samePN
    sd = seD_av_test_data_allExc_samePN
    
    plt.fill_between(x, av-sd, av+sd, alpha=alph, edgecolor=colors[i], facecolor=colors[i])
    plt.plot(np.arange(1, len(avD_av_test_data_allExc_samePN)+1), avD_av_test_data_allExc_samePN, colors[i], label=labs[i]) #, marker='.')
    
    
    i = 1
    x = np.arange(1, len(avD_av_test_data_inh_samePN)+1)
    av = avD_av_test_data_inh_samePN
    sd = seD_av_test_data_inh_samePN
    
    plt.fill_between(x, av-sd, av+sd, alpha=alph, edgecolor=colors[i], facecolor=colors[i])
    plt.plot(np.arange(1, len(avD_av_test_data_inh_samePN)+1), avD_av_test_data_inh_samePN, colors[i], label=labs[i])
    
    plt.xlim([-5, pExc+4])  
    makeNicePlots(plt.gca())      
    plt.xlabel('Numbers Neurons in the decoder')
    plt.ylabel('% Class accuracy')    
    plt.legend(loc='center left', bbox_to_anchor=(.7, .2), frameon=False) 
    
    
    if savefigs:               
        if chAl==1:
            dd = 'chAl_avSeDays_addNsROC_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_avSeDays_addNsROC_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
     
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
        


    ###################### plot diff of ave +/- se across days ... to better see when it plateaus
    di = np.diff(avD_av_test_data_inh_samePN)
    de = np.diff(avD_av_test_data_allExc_samePN)
    
    plt.figure(figsize=(4.5,3))
    plt.plot(de, color='b', label='exc')
    plt.plot(di, color='r', label='inh')
    plt.xlim([-5, pExc+4])  
    makeNicePlots(plt.gca())  
    plt.xlabel('Numbers Neurons in the decoder')
    plt.ylabel('% Chanbe in class accuracy')    
    plt.legend(loc='center left', bbox_to_anchor=(.7, .7), frameon=False) 
    
    
    if savefigs:               
        if chAl==1:
            dd = 'chAl_diff_avSeDays_addNsROC_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_diff_avSeDays_addNsROC_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
     
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
        



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
    
    time_aligned, nPreMin, nPostMin = set_nprepost(av_test_data_inh_ch, eventI_ds_allDays, mn_corr, thTrained, regressBins)
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
    
    plt.figure()
    
    for iday in range(len(days)):    
    
        if mn_corr[iday] >= thTrained:    
            totLen = len(av_test_data_inh_ch[iday])
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
            if len(time_al) != len(av_test_data_allExc_ch[iday]):
                lastTimeBinMissed = 1
                time_al = set_time_al(totLen, eventI_allDays[iday], lastTimeBinMissed)    
                
    #        plt.figure()
            plt.subplot(221)
            if do_excInhHalf==0:
                plt.errorbar(time_al, av_test_data_exc_ch[iday], yerr = sd_test_data_exc_ch[iday], label='exc', color='b')
            plt.errorbar(time_al, av_test_data_inh_ch[iday], yerr = sd_test_data_inh_ch[iday], label=labInh, color='r')    
            plt.errorbar(time_al, av_test_data_allExc_ch[iday], yerr = sd_test_data_allExc_ch[iday], label=labAll, color='k')
    #        plt.title(days[iday])
        
            plt.subplot(222)
            if do_excInhHalf==0:
                plt.plot(time_al, av_test_shfl_exc_ch[iday], label=' ', color='b')        
            plt.plot(time_al, av_test_shfl_inh_ch[iday], label=' ', color='r')    
            plt.plot(time_al, av_test_shfl_allExc_ch[iday], label=' ', color='k')        
            
            
            plt.subplot(223)
            if do_excInhHalf==0:
                h0,=plt.plot(time_al, av_test_chance_exc_ch[iday], label='exc', color='b')        
            h1,=plt.plot(time_al, av_test_chance_inh_ch[iday], label=labInh, color='r')    
            h2,=plt.plot(time_al, av_test_chance_allExc_ch[iday], label=labAll, color='k')        
            
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
       
    '''   
    # fni17, 151015: example day, excShfl 3
    ####### pick an example exc shfl
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
        plt.fill_between(time_al, av_test_data_exc_1shfl[iday] - sd_test_data_exc_1shfl[iday], av_test_data_exc_1shfl[iday] + sd_test_data_exc_1shfl[iday], alpha=0.5, edgecolor='b', facecolor='b')
        plt.plot(time_al, av_test_data_exc_1shfl[iday], 'b', label='exc')
        # inh
        plt.fill_between(time_al, av_test_data_inh[iday] - sd_test_data_inh[iday], av_test_data_inh[iday] + sd_test_data_inh[iday], alpha=0.5, edgecolor='r', facecolor='r')
        plt.plot(time_al, av_test_data_inh[iday], 'r', label='inh')
        # allExc
        plt.fill_between(time_al, av_test_data_allExc[iday] - sd_test_data_allExc[iday], av_test_data_allExc[iday] + sd_test_data_allExc[iday], alpha=0.5, edgecolor='k', facecolor='k')
        plt.plot(time_al, av_test_data_allExc[iday], 'k', label=labAll)
        
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
        plt.fill_between(time_al, av_test_shfl_exc_1shfl[iday] - sd_test_shfl_exc_1shfl[iday], av_test_shfl_exc_1shfl[iday] + sd_test_shfl_exc_1shfl[iday], alpha=0.3, edgecolor='b', facecolor='b')
        plt.plot(time_al, av_test_shfl_exc_1shfl[iday], 'b')
        # inh
        plt.fill_between(time_al, av_test_shfl_inh[iday] - sd_test_shfl_inh[iday], av_test_shfl_inh[iday] + sd_test_shfl_inh[iday], alpha=0.3, edgecolor='r', facecolor='r')
        plt.plot(time_al, av_test_shfl_inh[iday], 'r')
        # allExc
        plt.fill_between(time_al, av_test_shfl_allExc[iday] - sd_test_shfl_allExc[iday], av_test_shfl_allExc[iday] + sd_test_shfl_allExc[iday], alpha=0.3, edgecolor='k', facecolor='k')
        plt.plot(time_al, av_test_shfl_allExc[iday], 'k')
        
        ax = plt.gca()
        makeNicePlots(ax,1,1)
        
        
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False) 
    
    
        #xmin, xmax = ax.get_xlim()
        #plt.xlim([-1400,500])
        
        ##%% Save the figure    
        if savefigs:
    
            if chAl==1:
                dd = 'chAl_day' + days[iday][0:6] + '_exShfl' + str(iexcshfl) + '_' + nowStr
            else:
                dd = 'stAl_day' + days[iday][0:6] + '_exShfl' + str(iexcshfl) + '_' + nowStr
        
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
    
    '''
    
        
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
    plt.fill_between(time_aligned, av_av_test_data_inh_aligned - sd_av_test_data_inh_aligned, av_av_test_data_inh_aligned + sd_av_test_data_inh_aligned, alpha=0.5, edgecolor='r', facecolor='r')
    plt.plot(time_aligned, av_av_test_data_inh_aligned, 'r', label=labInh)
    # allExc
    plt.fill_between(time_aligned, av_av_test_data_allExc_aligned - sd_av_test_data_allExc_aligned, av_av_test_data_allExc_aligned + sd_av_test_data_allExc_aligned, alpha=0.5, edgecolor='k', facecolor='k')
    plt.plot(time_aligned, av_av_test_data_allExc_aligned, 'k', label=labAll)
    
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
    plt.plot(time_aligned, pp, color='k')
    
    
    #### shfl
    if superimpose==0:
        plt.subplot(222)
    # exc
    if do_excInhHalf==0:
        plt.fill_between(time_aligned, av_av_test_shfl_exc_aligned - sd_av_test_shfl_exc_aligned, av_av_test_shfl_exc_aligned + sd_av_test_shfl_exc_aligned, alpha=0.3, edgecolor='b', facecolor='b')
        plt.plot(time_aligned, av_av_test_shfl_exc_aligned, 'b')
    # inh
    plt.fill_between(time_aligned, av_av_test_shfl_inh_aligned - sd_av_test_shfl_inh_aligned, av_av_test_shfl_inh_aligned + sd_av_test_shfl_inh_aligned, alpha=0.3, edgecolor='r', facecolor='r')
    plt.plot(time_aligned, av_av_test_shfl_inh_aligned, 'r')
    # allExc
    plt.fill_between(time_aligned, av_av_test_shfl_allExc_aligned - sd_av_test_shfl_allExc_aligned, av_av_test_shfl_allExc_aligned + sd_av_test_shfl_allExc_aligned, alpha=0.3, edgecolor='k', facecolor='k')
    plt.plot(time_aligned, av_av_test_shfl_allExc_aligned, 'k')
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
        plt.fill_between(time_aligned, av_av_test_chance_inh_aligned - sd_av_test_chance_inh_aligned, av_av_test_chance_inh_aligned + sd_av_test_chance_inh_aligned, alpha=0.5, edgecolor='r', facecolor='r')
        plt.plot(time_aligned, av_av_test_chance_inh_aligned, 'r')
        # allExc
        plt.fill_between(time_aligned, av_av_test_chance_allExc_aligned - sd_av_test_chance_allExc_aligned, av_av_test_chance_allExc_aligned + sd_av_test_chance_allExc_aligned, alpha=0.5, edgecolor='k', facecolor='k')
        plt.plot(time_aligned, av_av_test_chance_allExc_aligned, 'k')
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
            dd = 'chAl_aveDays_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_aveDays_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
    
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
