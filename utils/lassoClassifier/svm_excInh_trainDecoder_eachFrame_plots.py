# -*- coding: utf-8 -*-
"""
Plot time course of class accuracy: svm trained on non-overlapping time windows  (outputs of file svm_eachFrame.py)
 ... svm trained to decode choice on choice-aligned or stimulus-aligned traces.
 
 
Remember for fni18 there are 2 svm_eachFrame mat files, the earlier file is using all trials (unequal HR, LR, like how you've done all your analysis). 
The later mat file is with equal number of hr and lr trials (subselecting trials)... this helped with 151209 class accur trace which was weird in the earlier mat file.
 
Created on Sun Mar 12 15:12:29 2017
@author: farznaj
"""     

# Use the following script for the plots of all mice (change in CA after breaking noise correslations):
# svm_excInh_trainDecoder_eachFrame_shflTrsEachN_sumAllMice_plots.py

#%% Change the following vars:

mousename = 'fni16'

do_Ctrace = 0 # if 1, use C trace (denoised fluorescent trace), instead of S (inferred spikes)
same_HRLR_acrossDays = 0 # normally 0, but if 1, svm will be trained on n trials, where n is the minimum number of trials across all sessions of a mouse (this variable is already computed and saved in the analysis folder of each mouse) # set to 1 to balance the number of trials across training days to make sure svm performance is not affected by that.
decodeOutcome = 1
decodeStimCateg = 0
noExtraStimDayAnalysis = 0 # if 1, exclude days with extraStim from analysis; if 0, include all days
shflTrsEachNeuron = 0 # 1st set to 1, then to 0 to compare how decoder changes after removing noise corrs. # Set to 0 for normal SVM training. # Shuffle trials in X_svm (for each neuron independently) to break correlations between neurons in each trial.
addNs_roc = 0 # if 1 do the following analysis: add neurons 1 by 1 to the decoder based on their tuning strength to see how the decoder performance increases.
h2l = 1 # if 1, load svm file in which neurons were added from high to low AUC. If 0: low to high AUC.
do_excInhHalf = 0 # 0: Load vars for inh,exc,allExc, 1: Load exc,inh SVM vars for excInhHalf (ie when the population consists of half exc and half inh) and allExc2inhSize (ie when populatin consists of allExc but same size as 2*inh size)
loadYtest = 0 # get svm performance for different trial strength # to load the svm files that include testTrInds_allSamps, etc vars (ie in addition to percClassError, they include the Y_hat for each trial)

ch_st_goAl = [-1,0,0] # [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. #chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
trialHistAnalysis = 0 # 1 # 
testIncorr = 1 # 0 # set to 0 for trialHistoryAnalysis  #set it to 1 to load the latest svm file even if you dont want to analyze incorr trials. # load svm file that includes testing corr as well as testing incorr trials.
corrTrained = 1 # 0 # set to 0 for trialHistoryAnalysis # svm was trained on what trial outcomes: if corr, set to 1; if all outcomes, set to 0)

doAllN = 1 # matters only when do_excInhHalf =0; plot allN, instead of allExc
savefigs = 0 #1
plotSingleSessions = 1 # plot individual sessions when addNs_roc=0

if decodeStimCateg:
    use_both_remCorr_testing = [1,0,0] # [1,0,0]: use this for decodeStimCateg  # if decodeStimCateg = 1, decide which one (testing, remCorr, or Both) you want to use for the rest of this code
elif decodeOutcome:
    use_both_remCorr_testing = [0,0,1] #[0,0,1] #[0,0,1]: use this for decodeOutcome; since remCorr included only correct trials, CA of remCorr is similar for both shfl and data... so we only use the testing data!


if shflTrsEachNeuron:
    testIncorr = 0 # it was ran in the past, so there is no testIncorr in the file name    
    doAllN = 0 # matters only when do_excInhHalf =0; plot allN, instead of allExc

if do_Ctrace:
    testIncorr = 0
    
thTrained = 10  # number of trials of each class used for svm training, min acceptable value to include a day in analysis; i.e. if a day has <10 hr trials and <10 lr trials, it wont be included in the analysis.
loadWeights = 0
useEqualTrNums = 1
doIncorr = 0
import numpy as np
allDays = np.nan
noZmotionDays = np.nan
noZmotionDays_strict = np.nan
noExtraStimDays = np.nan

if mousename == 'fni18': #set one of the following to 1:
    allDays = 1 # all 7 days will be used (last 3 days have z motion!)
    noZmotionDays = 0 # 4 days that dont have z motion will be used.
    noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!
    if do_Ctrace:
        allDays = 0
        noZmotionDays = 1
elif mousename == 'fni19':    
    allDays = 1    
    noExtraStimDays = 0   

#loadInhAllexcEqexc = 1 # if 1, load 2nd run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc separately; also for all days the new vector inhRois_pix was used (not the old inhRois)           
iTiFlg = 2 # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
superimpose = 1 # the averaged aligned traces of testing and shuffled will be plotted on the same figure
thStimStrength = 3 # needed if loadYtest=1 # 2; # threshold of stim strength for defining hard, medium and easy trials.
    
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

if decodeStimCateg or decodeOutcome:
    corrTrained=0
    
if corrTrained==0:
    corrn = 'allOutcomeTrained_'
else:
    corrn = 'corrTrained_'
    
if decodeStimCateg:
    corrn = 'decodeStimCateg_' + corrn

if noExtraStimDayAnalysis:
    corrn = corrn + 'noExtraStim_'

if decodeOutcome:
    corrn = 'decodeOutcome_' + corrn    

if same_HRLR_acrossDays:
    thTrained = 20 # use this if you want to analyze days that svm was run on 20HR and 20LR trials (these are the lateset SVM_sameTrNum files)... if a day doesn't have these latest files, the ealier svm_sameTr file will be loaded, but when we check thTrained we exclude these days.
    corrn = 'sameTrNumAllDays_20Trs_' + corrn    
    testIncorr = 0

if do_Ctrace:
    corrn = 'Ctrace_' + corrn

if mousename=='fni18' and noZmotionDays:
    corrn = corrn + 'noZmotionDays_'

    
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
days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays, loadYtest, decodeStimCateg, decodeOutcome)
# remove some days if needed:
#days = np.delete(days, [14])

frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.
#lastTimeBinMissed = 0 #1# if 0, things were ran fine; if 1: by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')


if same_HRLR_acrossDays:    
    dataPath = '/home/farznaj/Shares/Churchland_nlsas_data/data/'
    if mousename=='fni18' or mousename=='fni19':
        dataPath = '/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/'
        
    a = os.path.join(dataPath, mousename, 'imaging', 'analysis', 'SVM_minHRLR_acrossDays_choiceDecode_chAl.mat')
    data = scio.loadmat(a) #, {'mnHRLR_acrossDays':mnHRLR_acrossDays, 'mn_corr':mn_corr, 'days2an_heatmap':days2an_heatmap})
    
    aa = scio.loadmat(a, variable_names=['days2an_heatmap'])
    days2an_heatmap = aa.pop('days2an_heatmap').flatten()
    
    days = list(np.array(days)[days2an_heatmap.astype(bool)])

print 'number of days to be analyzed: ', len(days)


#%% Initiate vars
'''
#####################################################################################################################################################   
#####################################################################################################################################################
'''

days2an = np.arange(0, len(days)) # range(len(days)) # 

if (mousename=='fni17' and shflTrsEachNeuron==1 and corrTrained==0):
    days2an = np.delete(days2an, 33) # fni17 151013 doesnt have the mat file: excInh_SVMtrained_eachFrame_shflTrsPerN_eqExc_currChoice_chAl_ds3_eqTrs_171016-*


numInh = np.full(len(days2an), np.nan)
numAllexc = np.full(len(days2an), np.nan)
if decodeOutcome:
    corr_hr_lr = np.full((len(days2an),4), np.nan) # number of hr, lr correct trials, and hr, lr incorrect trials for each day
else:
    corr_hr_lr = np.full((len(days2an),2), np.nan) # number of hr, lr correct trials for each day

eventI_allDays = np.full(len(days2an), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
eventI_ds_allDays = np.full(len(days2an), np.nan)
perClassErrorTest_data_inh_all = []
perClassErrorTest_shfl_inh_all = []
perClassErrorTest_chance_inh_all = []
perClassErrorTest_data_allExc_all = []
perClassErrorTest_shfl_allExc_all = []
perClassErrorTest_chance_allExc_all = []
perClassErrorTest_data_exc_all = []
perClassErrorTest_shfl_exc_all = []
perClassErrorTest_chance_exc_all = []
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
perClassErrorTestBoth_inh_all = []
perClassErrorTestBoth_shfl_inh_all = []
perClassErrorTestBoth_chance_inh_all = []
perClassErrorTestRemCorr_inh_all = []
perClassErrorTestRemCorr_shfl_inh_all = []
perClassErrorTestRemCorr_chance_inh_all = []
perClassErrorTestBoth_allExc_all = []
perClassErrorTestBoth_shfl_allExc_all = []
perClassErrorTestBoth_chance_allExc_all = []
perClassErrorTestRemCorr_allExc_all = []
perClassErrorTestRemCorr_shfl_allExc_all = []
perClassErrorTestRemCorr_chance_allExc_all = []
perClassErrorTestBoth_exc_all = []
perClassErrorTestBoth_shfl_exc_all = []
perClassErrorTestBoth_chance_exc_all = []
perClassErrorTestRemCorr_exc_all = []
perClassErrorTestRemCorr_shfl_exc_all = []
perClassErrorTestRemCorr_chance_exc_all = []      
fractTrs_50msExtraStim = np.full(len(days2an), np.nan)    


#%% Loop over days    

cntd = -1
# iday = 0
for iday in days2an: # np.arange(3, len(days)): # range(len(days)): # 
        
    #%%            
    cntd = cntd+1
    
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
    fractTrs_50msExtraStim[cntd] = np.mean((timeStimOffset - timeSingleStimOffset)>50)    


    #%% Get number of hr, lr trials that were used for svm training

#    svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, [1,0,0], regressBins, useEqualTrNums, corrTrained, shflTrsEachNeuron)[0]   
    svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, 1, [1,0,0], regressBins, useEqualTrNums, corrTrained, shflTrsEachNeuron)[0]   

    if decodeOutcome:
        [corr_hr, corr_lr, incorr_hr, incorr_lr] = set_corr_incorr_hr_lr(postName, svmName)
        corr_hr_lr[cntd,:] = [corr_hr, corr_lr, incorr_hr, incorr_lr]
    else:
        corr_hr, corr_lr = set_corr_hr_lr(postName, svmName)
        corr_hr_lr[cntd,:] = [corr_hr, corr_lr]        
    
    
    #%% Load matlab vars to set eventI_ds (downsampled eventI)

    eventI, eventI_ds = setEventIds(postName, chAl, regressBins=3, trialHistAnalysis=0)
    
    eventI_allDays[cntd] = eventI
    eventI_ds_allDays[cntd] = eventI_ds

    
    #%% Load SVM vars

    if addNs_roc:    
        # number of neurons in the decoder x nSamps x nFrs
#        perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, w_data_inh, w_data_allExc, b_data_inh, b_data_allExc, svmName_excInh = loadSVM_excInh_addNs1by1(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, loadWeights, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0)
#        perClassErrorTest_data_exc = 0; perClassErrorTest_shfl_exc = 0; perClassErrorTest_chance_exc = 0; 
        perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh = \
        loadSVM_excInh_addNs1by1(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, loadWeights, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0, h2l=h2l)
        
    elif do_excInhHalf:
        # numShufflesExc x numSamples x numFrames
        perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, w_data_inh, w_data_allExc, b_data_inh, b_data_allExc, svmName_excInh = \
        loadSVM_excInh_excInhHalf(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, loadWeights, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0)
        perClassErrorTest_data_exc = 0; perClassErrorTest_shfl_exc = 0; perClassErrorTest_chance_exc = 0; 
        
    elif decodeStimCateg or decodeOutcome:
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
        = loadSVM_excInh_decodeStimCateg(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0, loadYtest=loadYtest, decodeOutcome=decodeOutcome)

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
        
        if len(perClassErrorTest_data_exc)>0:
            perClassErrorTest_data_exc = np.mean(perClassErrorTest_data_exc, axis=-2)
            perClassErrorTest_shfl_exc = np.mean(perClassErrorTest_shfl_exc, axis=-2)
            perClassErrorTest_chance_exc = np.mean(perClassErrorTest_chance_exc, axis=-2)
            perClassErrorTestRemCorr_exc = np.mean(perClassErrorTestRemCorr_exc, axis=-2)
            perClassErrorTestRemCorr_shfl_exc = np.mean(perClassErrorTestRemCorr_shfl_exc, axis=-2)
            perClassErrorTestRemCorr_chance_exc = np.mean(perClassErrorTestRemCorr_chance_exc, axis=-2)
            perClassErrorTestBoth_exc = np.mean(perClassErrorTestBoth_exc, axis=-2)
            perClassErrorTestBoth_shfl_exc = np.mean(perClassErrorTestBoth_shfl_exc, axis=-2)
            perClassErrorTestBoth_chance_exc = np.mean(perClassErrorTestBoth_chance_exc, axis=-2)
        
    else: # regular case
        if loadYtest==0:
            perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, \
            w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN, trsExcluded, \
            perClassErrorTest_data_inh_incorr, perClassErrorTest_shfl_inh_incorr, perClassErrorTest_data_allExc_incorr, perClassErrorTest_shfl_allExc_incorr, perClassErrorTest_data_exc_incorr, perClassErrorTest_shfl_exc_incorr, \
            = loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0, loadYtest=0, testIncorr=testIncorr, same_HRLR_acrossDays=same_HRLR_acrossDays, do_Ctrace=do_Ctrace)
        else:
            perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, \
            w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN, trsExcluded, \
            testTrInds_allSamps_inh, Ytest_allSamps_inh, Ytest_hat_allSampsFrs_inh, trsnow_allSamps_inh, \
            testTrInds_allSamps_allExc, Ytest_allSamps_allExc, Ytest_hat_allSampsFrs_allExc, trsnow_allSamps_allExc, \
            testTrInds_allSamps_exc, Ytest_allSamps_exc, Ytest_hat_allSampsFrs_exc, trsnow_allSamps_exc, \
            = loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0, loadYtest=1, testIncorr=testIncorr)
            
            
#        perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN, testTrInds_allSamps_inh, Ytest_allSamps_inh, Ytest_hat_allSampsFrs_inh, testTrInds_allSamps_allExc, Ytest_allSamps_allExc, Ytest_hat_allSampsFrs_allExc, testTrInds_allSamps_exc, Ytest_allSamps_exc, Ytest_hat_allSampsFrs_exc = loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0, loadYtest=loadYtest)
#        perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN, trsExcluded, testTrInds_allSamps_inh, Ytest_allSamps_inh, Ytest_hat_allSampsFrs_inh, trsnow_allSamps_inh, testTrInds_allSamps_allExc, Ytest_allSamps_allExc, Ytest_hat_allSampsFrs_allExc, trsnow_allSamps_allExc, testTrInds_allSamps_exc, Ytest_allSamps_exc, Ytest_hat_allSampsFrs_exc, trsnow_allSamps_exc = \
#        loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0, loadYtest=loadYtest, testIncorr=testIncorr)
    
    ##%% Get number of inh and exc        
    if loadWeights==1:
        numInh[cntd] = w_data_inh.shape[1]
        numAllexc[cntd] = w_data_allExc.shape[1]
        


    #%% Set classification error for easy, medium, and hard trials.
    
    if loadYtest:
        execfile("set_classErr_easyMedHard.py")
                    
    
    #%% Once done with all frames, save vars for all days
    
    perClassErrorTest_data_inh_all.append(perClassErrorTest_data_inh) # each day: samps x numFrs    # if addNs_roc : number of neurons in the decoder x nSamps x nFrs    
    perClassErrorTest_shfl_inh_all.append(perClassErrorTest_shfl_inh)
    perClassErrorTest_chance_inh_all.append(perClassErrorTest_chance_inh)
    if decodeStimCateg or decodeOutcome:
        perClassErrorTestBoth_inh_all.append(perClassErrorTestBoth_inh) # each day: samps x numFrs    # if addNs_roc : number of neurons in the decoder x nSamps x nFrs    
        perClassErrorTestBoth_shfl_inh_all.append(perClassErrorTestBoth_shfl_inh)
        perClassErrorTestBoth_chance_inh_all.append(perClassErrorTestBoth_chance_inh)
        perClassErrorTestRemCorr_inh_all.append(perClassErrorTestRemCorr_inh) # each day: samps x numFrs    # if addNs_roc : number of neurons in the decoder x nSamps x nFrs    
        perClassErrorTestRemCorr_shfl_inh_all.append(perClassErrorTestRemCorr_shfl_inh)
        perClassErrorTestRemCorr_chance_inh_all.append(perClassErrorTestRemCorr_chance_inh)   
    
    perClassErrorTest_data_allExc_all.append(perClassErrorTest_data_allExc) # each day: samps x numFrs    # if addNs_roc : number of neurons in the decoder x nSamps x nFrs
    perClassErrorTest_shfl_allExc_all.append(perClassErrorTest_shfl_allExc)
    perClassErrorTest_chance_allExc_all.append(perClassErrorTest_chance_allExc) 
    if decodeStimCateg or decodeOutcome:
        perClassErrorTestBoth_allExc_all.append(perClassErrorTestBoth_allExc) # each day: samps x numFrs    # if addNs_roc : number of neurons in the decoder x nSamps x nFrs    
        perClassErrorTestBoth_shfl_allExc_all.append(perClassErrorTestBoth_shfl_allExc)
        perClassErrorTestBoth_chance_allExc_all.append(perClassErrorTestBoth_chance_allExc)
        perClassErrorTestRemCorr_allExc_all.append(perClassErrorTestRemCorr_allExc) # each day: samps x numFrs    # if addNs_roc : number of neurons in the decoder x nSamps x nFrs    
        perClassErrorTestRemCorr_shfl_allExc_all.append(perClassErrorTestRemCorr_shfl_allExc)
        perClassErrorTestRemCorr_chance_allExc_all.append(perClassErrorTestRemCorr_chance_allExc)   

    if do_excInhHalf==0: #np.logical_and(do_excInhHalf==0, addNs_roc==0):
        perClassErrorTest_data_exc_all.append(perClassErrorTest_data_exc) # each day: numShufflesExc x numSamples x numFrames   # if addNs_roc: # numShufflesExc x number of neurons in the decoder x numSamples x nFrs 
        perClassErrorTest_shfl_exc_all.append(perClassErrorTest_shfl_exc)
        perClassErrorTest_chance_exc_all.append(perClassErrorTest_chance_exc)
        if decodeStimCateg or decodeOutcome:
            perClassErrorTestBoth_exc_all.append(perClassErrorTestBoth_exc) # each day: samps x numFrs    # if addNs_roc : number of neurons in the decoder x nSamps x nFrs    
            perClassErrorTestBoth_shfl_exc_all.append(perClassErrorTestBoth_shfl_exc)
            perClassErrorTestBoth_chance_exc_all.append(perClassErrorTestBoth_chance_exc)
            perClassErrorTestRemCorr_exc_all.append(perClassErrorTestRemCorr_exc) # each day: samps x numFrs    # if addNs_roc : number of neurons in the decoder x nSamps x nFrs    
            perClassErrorTestRemCorr_shfl_exc_all.append(perClassErrorTestRemCorr_shfl_exc)
            perClassErrorTestRemCorr_chance_exc_all.append(perClassErrorTestRemCorr_chance_exc)   

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

print '_____________________ Done with all days _____________________'

#no
#%%    
######################################################################################################################################################    
######################################################################################################################################################          
######################################################################################################################################################    
######################################################################################################################################################   

#%% Average and st error of class accuracies across CV samples ... for each day
# in the function below we turn class error to class accuracy (by subtracting them from 100)

if addNs_roc: # remember in this analysis you got the CAs only for time bin -1 
    fr2an = -1 # relative to eventI_ds_allDays
    # av_test_data_inh : # numDays; each day: number of neurons in the decoder
#    numSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc \
#        = av_se_CA_trsamps_addNs1by1(numD, perClassErrorTest_data_inh_all, perClassErrorTest_shfl_inh_all, perClassErrorTest_chance_inh_all, perClassErrorTest_data_allExc_all, perClassErrorTest_shfl_allExc_all, perClassErrorTest_chance_allExc_all, fr2an, eventI_ds_allDays)
    numSamples, numExcSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, av_test_data_exc, sd_test_data_exc, av_test_shfl_exc, sd_test_shfl_exc, av_test_chance_exc, sd_test_chance_exc, av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc \
        = av_se_CA_trsamps_addNs1by1(numD, perClassErrorTest_data_inh_all, perClassErrorTest_shfl_inh_all, perClassErrorTest_chance_inh_all, perClassErrorTest_data_exc_all, perClassErrorTest_shfl_exc_all, perClassErrorTest_chance_exc_all, perClassErrorTest_data_allExc_all, perClassErrorTest_shfl_allExc_all, perClassErrorTest_chance_allExc_all, fr2an, eventI_ds_allDays)
    
elif do_excInhHalf:
    numSamples, numExcSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc \
        = av_se_CA_trsamps_excInhHalf(numD, perClassErrorTest_data_inh_all, perClassErrorTest_shfl_inh_all, perClassErrorTest_chance_inh_all, perClassErrorTest_data_allExc_all, perClassErrorTest_shfl_allExc_all, perClassErrorTest_chance_allExc_all)

elif decodeStimCateg or decodeOutcome:
    numSamples, numExcSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, \
    av_test_remCorr_data_inh, sd_test_remCorr_data_inh, av_test_remCorr_shfl_inh, sd_test_remCorr_shfl_inh, av_test_remCorr_chance_inh, sd_test_remCorr_chance_inh, \
    av_test_both_data_inh, sd_test_both_data_inh, av_test_both_shfl_inh, sd_test_both_shfl_inh, av_test_both_chance_inh, sd_test_both_chance_inh, \
    av_test_data_exc, sd_test_data_exc, av_test_shfl_exc, sd_test_shfl_exc, av_test_chance_exc, sd_test_chance_exc, \
    av_test_remCorr_data_exc, sd_test_remCorr_data_exc, av_test_remCorr_shfl_exc, sd_test_remCorr_shfl_exc, av_test_remCorr_chance_exc, sd_test_remCorr_chance_exc, \
    av_test_both_data_exc, sd_test_both_data_exc, av_test_both_shfl_exc, sd_test_both_shfl_exc, av_test_both_chance_exc, sd_test_both_chance_exc, \
    av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc, \
    av_test_remCorr_data_allExc, sd_test_remCorr_data_allExc, av_test_remCorr_shfl_allExc, sd_test_remCorr_shfl_allExc, av_test_remCorr_chance_allExc, sd_test_remCorr_chance_allExc, \
    av_test_both_data_allExc, sd_test_both_data_allExc, av_test_both_shfl_allExc, sd_test_both_shfl_allExc, av_test_both_chance_allExc, sd_test_both_chance_allExc, \
        = av_se_CA_trsamps_decodeStimCateg(numD, perClassErrorTest_data_inh_all, perClassErrorTest_shfl_inh_all, perClassErrorTest_chance_inh_all, \
        perClassErrorTestRemCorr_inh_all, perClassErrorTestRemCorr_shfl_inh_all, perClassErrorTestRemCorr_chance_inh_all, \
        perClassErrorTestBoth_inh_all, perClassErrorTestBoth_shfl_inh_all, perClassErrorTestBoth_chance_inh_all, \
        perClassErrorTest_data_exc_all, perClassErrorTest_shfl_exc_all, perClassErrorTest_chance_exc_all, \
        perClassErrorTestRemCorr_exc_all, perClassErrorTestRemCorr_shfl_exc_all, perClassErrorTestRemCorr_chance_exc_all, \
        perClassErrorTestBoth_exc_all, perClassErrorTestBoth_shfl_exc_all, perClassErrorTestBoth_chance_exc_all, \
        perClassErrorTest_data_allExc_all, perClassErrorTest_shfl_allExc_all, perClassErrorTest_chance_allExc_all, \
        perClassErrorTestRemCorr_allExc_all, perClassErrorTestRemCorr_shfl_allExc_all, perClassErrorTestRemCorr_chance_allExc_all, \
        perClassErrorTestBoth_allExc_all, perClassErrorTestBoth_shfl_allExc_all, perClassErrorTestBoth_chance_allExc_all)


    if len(perClassErrorTestRemCorr_inh_all[0]) and np.isnan(perClassErrorTestRemCorr_inh_all[0]): # when you decoded outcome and traces were outcome aligned, you sent remCorr and _both vars to nan (instead of those huge matrices), bc anyway you were going to only use the testing data
        ignore = 1
    else:
        ignore = 0

    # for each day run ttest across samples between data and shfl, to see if classifier performance (in each frame) is significantly different from shuffled case or not.
    ttest_pval_allExc = np.array([stats.ttest_ind(perClassErrorTest_data_allExc_all[iday], perClassErrorTest_shfl_allExc_all[iday], axis=0)[1] for iday in range(numD)])
    ttest_pval_inh = np.array([stats.ttest_ind(perClassErrorTest_data_inh_all[iday], perClassErrorTest_shfl_inh_all[iday], axis=0)[1] for iday in range(numD)])    
    ttest_pval_exc = [] # = np.full((perClassErrorTest_data_exc_all[iday].shape[-1], numExcSamples), np.nan)
    for iexc in range(numExcSamples):
        ttest_pval_exc_e = []
        for iday in range(numD):
            ttest_pval_exc_e.append(stats.ttest_ind(perClassErrorTest_data_exc_all[iday][iexc], perClassErrorTest_shfl_inh_all[iday][iexc], axis=0)[1])
        ttest_pval_exc.append(ttest_pval_exc_e) 
    ttest_pval_exc = np.array(ttest_pval_exc) # samps # days # frs


    if ignore==0:
        ttest_pval_allExc_remCorr = np.array([stats.ttest_ind(perClassErrorTestRemCorr_allExc_all[iday], perClassErrorTestRemCorr_shfl_allExc_all[iday], axis=0)[1] for iday in range(numD)])
        ttest_pval_inh_remCorr = np.array([stats.ttest_ind(perClassErrorTestRemCorr_inh_all[iday], perClassErrorTestRemCorr_shfl_inh_all[iday], axis=0)[1] for iday in range(numD)])    
        ttest_pval_exc_remCorr = [] # = np.full((perClassErrorTestRemCorr_data_exc_all[iday].shape[-1], numExcSamples), np.nan)
        for iexc in range(numExcSamples):
            ttest_pval_exc_e = []
            for iday in range(numD):
                ttest_pval_exc_e.append(stats.ttest_ind(perClassErrorTestRemCorr_exc_all[iday][iexc], perClassErrorTestRemCorr_shfl_inh_all[iday][iexc], axis=0)[1])
            ttest_pval_exc_remCorr.append(ttest_pval_exc_e) 
        ttest_pval_exc_remCorr = np.array(ttest_pval_exc_remCorr) # samps # days # frs


    ttest_pval_allExc_both = np.array([stats.ttest_ind(perClassErrorTestBoth_allExc_all[iday], perClassErrorTestBoth_shfl_allExc_all[iday], axis=0)[1] for iday in range(numD)])
    ttest_pval_inh_both = np.array([stats.ttest_ind(perClassErrorTestBoth_inh_all[iday], perClassErrorTestBoth_shfl_inh_all[iday], axis=0)[1] for iday in range(numD)])    
    ttest_pval_exc_both = [] # = np.full((perClassErrorTestBoth_data_exc_all[iday].shape[-1], numExcSamples), np.nan)
    for iexc in range(numExcSamples):
        ttest_pval_exc_e = []
        for iday in range(numD):
            ttest_pval_exc_e.append(stats.ttest_ind(perClassErrorTestBoth_exc_all[iday][iexc], perClassErrorTestBoth_shfl_inh_all[iday][iexc], axis=0)[1])
        ttest_pval_exc_both.append(ttest_pval_exc_e) 
    ttest_pval_exc_both = np.array(ttest_pval_exc_both) # samps # days # frs
  
    
    
    
else:
    numSamples, numExcSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, av_test_data_exc, sd_test_data_exc, av_test_shfl_exc, sd_test_shfl_exc, av_test_chance_exc, sd_test_chance_exc, av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc \
        = av_se_CA_trsamps(numD, perClassErrorTest_data_inh_all, perClassErrorTest_shfl_inh_all, perClassErrorTest_chance_inh_all, perClassErrorTest_data_exc_all, perClassErrorTest_shfl_exc_all, perClassErrorTest_chance_exc_all, perClassErrorTest_data_allExc_all, perClassErrorTest_shfl_allExc_all, perClassErrorTest_chance_allExc_all)

    # for each day run ttest across samples between data and shfl, to see if classifier performance (in each frame) is significantly different from shuffled case or not.
    ttest_pval_allExc = np.array([stats.ttest_ind(perClassErrorTest_data_allExc_all[iday], perClassErrorTest_shfl_allExc_all[iday], axis=0)[1] for iday in range(numD)])
    ttest_pval_inh = np.array([stats.ttest_ind(perClassErrorTest_data_inh_all[iday], perClassErrorTest_shfl_inh_all[iday], axis=0)[1] for iday in range(numD)])    
    ttest_pval_exc = [] # = np.full((perClassErrorTest_data_exc_all[iday].shape[-1], numExcSamples), np.nan)
    for iexc in range(numExcSamples):
        ttest_pval_exc_e = []
        for iday in range(numD):
            ttest_pval_exc_e.append(stats.ttest_ind(perClassErrorTest_data_exc_all[iday][iexc], perClassErrorTest_shfl_inh_all[iday][iexc], axis=0)[1])
        ttest_pval_exc.append(ttest_pval_exc_e) 
    ttest_pval_exc = np.array(ttest_pval_exc) # samps # days # frs
    
    
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

if shflTrsEachNeuron: # if you want to compare data vs shflTrsEachNeuron, first run the codes with shflTrsEachNeuron = 1, and then run it with shflTrsEachNeuron = 0.
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
    
    perClassErrorTest_data_inh_all_shflTrsEachN = perClassErrorTest_data_inh_all
    perClassErrorTest_data_allExc_all_shflTrsEachN = perClassErrorTest_data_allExc_all
    perClassErrorTest_data_exc_all_shflTrsEachN = perClassErrorTest_data_exc_all
"""
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
"""
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
if decodeOutcome:
    mn_corr = 2*mn_corr # bc in decodeOutcome, corr_hr_lr shows hr and lr for each class, so the total number of trials per class will be twice the mn_corr.
#mn_corr = np.delete(mn_corr, [46]) # #np.append(mn_corr,[12,12])
print '%d days will be excluded: < %d trials per choice for svm training' %(sum(mn_corr < thTrained), thTrained)
print np.array(days)[mn_corr < thTrained]

numDaysGood = sum(mn_corr>=thTrained)
print 'Final number of days: %d' %(numDaysGood)

###%% NOTE: Do you want to do this?    
#days = np.array(days)
#days = days[days2an]
mn_corr0 = mn_corr + 0 # will be used for the heatmaps of CA for all days; We need to exclude 151023 from fni16, this day had issues! ...  in the mat file of stabTestTrainTimes, its class accur is very high ... and in the mat file of excInh_trainDecoder_eachFrame it is very low ...ANYWAY I ended up removing it from the heatmap of CA for all days!!! but it is included in other analyses!!
if (mousename=='fni16' and decodeStimCateg==0 and same_HRLR_acrossDays==0 and decodeOutcome==0): #np.logical_and(mousename=='fni16', corrTrained==1):
    mn_corr0[days.index('151023_1')] = thTrained - 1 # see it will be excluded from analysis!    
days2an_heatmap = mn_corr0 >= thTrained     


####### Set days without extraStim
#    days=np.array(days); days[fractTrs_50msExtraStim<.1]
if noExtraStimDayAnalysis: # dont analyze days with extra stim
    days2an_noExtraStim = np.logical_and(fractTrs_50msExtraStim < .1 , mn_corr >= thTrained)
else:
    days2an_noExtraStim = mn_corr >= thTrained #fractTrs_50msExtraStim < 2 # include all days
    
days2an_heatmap = np.logical_and(days2an_heatmap , days2an_noExtraStim)

# across all days that went into the learning analysis, the mim number of HR (or LR) trials was mnHRLR_acrossDays
mnHRLR_acrossDays = min(mn_corr[days2an_heatmap]) # min(mn_corr[mn_corr>=thTrained]) 
#print mnHRLR_acrossDays

# save mn_corr etc vars in the analysis folder for each mouse
'''
a = os.path.join(dataPath, mousename, 'imaging', 'analysis', 'SVM_minHRLR_acrossDays_choiceDecode_chAl.mat')
scio.savemat(a, {'mnHRLR_acrossDays':mnHRLR_acrossDays, 'mn_corr':mn_corr, 'days2an_heatmap':days2an_heatmap})

aa = scio.loadmat(a, variable_names=['mnHRLR_acrossDays'])
aaa = aa.pop('mnHRLR_acrossDays')[0,:]
aaa, mnHRLR_acrossDays
'''
#no

#%% Load behavioral performance vars (for plots of svm performance vs behavioral performance)

# set svm_stab mat file name that contains behavioral and class accuracy vars
imagingDir = setImagingAnalysisNamesP(mousename)
fname = os.path.join(imagingDir, 'analysis')    
# set svmStab mat file name that contains behavioral and class accuracy vars

if trialHistAnalysis==0:
    finame = os.path.join(fname, 'svm_stabilityBehCA_*.mat')
else:
    finame = os.path.join(fname, 'svm_trialHist_stabilityBehCA_*.mat')
    
stabBehName = glob.glob(finame)
stabBehName = sorted(stabBehName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
stabBehName = stabBehName[0]

# load beh vars    
Data = scio.loadmat(stabBehName)
behCorr_all = Data.pop('behCorr_all').flatten()[days2an] # the following comment is for the above mat file: I didnt save _all vars... so what is saved is only for one day! ... have to reset these 3 vars again here!
behCorrHR_all = Data.pop('behCorrHR_all').flatten()[days2an]
behCorrLR_all = Data.pop('behCorrLR_all').flatten()[days2an]
#    classAccurTMS_inh = Data.pop('classAccurTMS_inh') # the following comment is for the above mat file: there was also problem in setting these vars, so you need to reset these 3 vars here
#    classAccurTMS_exc = Data.pop('classAccurTMS_exc')
#    classAccurTMS_allExc = Data.pop('classAccurTMS_allExc')
   
if np.logical_and(decodeStimCateg==1, mousename=='fni18'): # first day ('151209_1') is not included
    behCorr_all = np.delete(behCorr_all, 0)
    behCorrHR_all = np.delete(behCorrHR_all, 0)
    behCorrLR_all = np.delete(behCorrLR_all, 0)


##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################


#%% Go to line 1900 for addNs_roc plots. 

if addNs_roc==0:    

    alph = .001 # to find significancy (data vs shuffle)
    
    ##################################################################################################
    ############## Align class accur traces of all days to make a final average trace ##############
    ################################################################################################## 
      
    ##%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
    # By common eventI, we  mean the index on which all traces will be aligned.
    
    time_aligned, nPreMin, nPostMin = set_nprepost(av_test_data_inh, eventI_ds_allDays, mn_corr, thTrained, regressBins)
#    nPreMin = np.nanmin(eventI_ds_allDays[mn_corr >= thTrained]).astype('int')
    
    
    #%% Align traces of all days on the common eventI (use nan for days with few trianed svm trials)
    # not including days with too few svm trained trials.
    
    def alTrace(trace, eventI_ds_allDays, nPreMin, nPostMin):    #same as alTrace in defFuns, except here the output is an array of size frs x days, there the output is a list of size days x frs
    
        trace_aligned = np.ones((nPreMin + nPostMin + 1, trace.shape[0])) + np.nan # frames x days, aligned on common eventI (equals nPreMin)     
        for iday in range(trace.shape[0]):
            if mn_corr[iday] >= thTrained: # dont include days with too few svm trained trials.
                trace_aligned[:, iday] = trace[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]    
        return trace_aligned
        
    # Align traces
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
    
    if decodeStimCateg or decodeOutcome:
        if ignore==0:
            av_test_remCorr_data_inh_aligned = alTrace(av_test_remCorr_data_inh, eventI_ds_allDays, nPreMin, nPostMin)
            av_test_remCorr_shfl_inh_aligned = alTrace(av_test_remCorr_shfl_inh, eventI_ds_allDays, nPreMin, nPostMin)
            av_test_remCorr_chance_inh_aligned = alTrace(av_test_remCorr_chance_inh, eventI_ds_allDays, nPreMin, nPostMin)    
            av_test_both_data_inh_aligned = alTrace(av_test_both_data_inh, eventI_ds_allDays, nPreMin, nPostMin)
            av_test_both_shfl_inh_aligned = alTrace(av_test_both_shfl_inh, eventI_ds_allDays, nPreMin, nPostMin)
            av_test_both_chance_inh_aligned = alTrace(av_test_both_chance_inh, eventI_ds_allDays, nPreMin, nPostMin)    
        
            if do_excInhHalf==0:
                av_test_remCorr_data_exc_aligned = alTrace(av_test_remCorr_data_exc, eventI_ds_allDays, nPreMin, nPostMin)
                av_test_remCorr_shfl_exc_aligned = alTrace(av_test_remCorr_shfl_exc, eventI_ds_allDays, nPreMin, nPostMin)
                av_test_remCorr_chance_exc_aligned = alTrace(av_test_remCorr_chance_exc, eventI_ds_allDays, nPreMin, nPostMin)    
                av_test_both_data_exc_aligned = alTrace(av_test_both_data_exc, eventI_ds_allDays, nPreMin, nPostMin)
                av_test_both_shfl_exc_aligned = alTrace(av_test_both_shfl_exc, eventI_ds_allDays, nPreMin, nPostMin)
                av_test_both_chance_exc_aligned = alTrace(av_test_both_chance_exc, eventI_ds_allDays, nPreMin, nPostMin)    
                
            av_test_remCorr_data_allExc_aligned = alTrace(av_test_remCorr_data_allExc, eventI_ds_allDays, nPreMin, nPostMin)
            av_test_remCorr_shfl_allExc_aligned = alTrace(av_test_remCorr_shfl_allExc, eventI_ds_allDays, nPreMin, nPostMin)
            av_test_remCorr_chance_allExc_aligned = alTrace(av_test_remCorr_chance_allExc, eventI_ds_allDays, nPreMin, nPostMin)        
            av_test_both_data_allExc_aligned = alTrace(av_test_both_data_allExc, eventI_ds_allDays, nPreMin, nPostMin)
            av_test_both_shfl_allExc_aligned = alTrace(av_test_both_shfl_allExc, eventI_ds_allDays, nPreMin, nPostMin)
            av_test_both_chance_allExc_aligned = alTrace(av_test_both_chance_allExc, eventI_ds_allDays, nPreMin, nPostMin)        
    
        # keep a copy of testing data vars:
        av_test_data_inh_aligned0 = av_test_data_inh_aligned
        av_test_shfl_inh_aligned0 = av_test_shfl_inh_aligned
        av_test_chance_inh_aligned0 = av_test_chance_inh_aligned
        if do_excInhHalf==0:
            av_test_data_exc_aligned0 = av_test_data_exc_aligned
            av_test_shfl_exc_aligned0 = av_test_shfl_exc_aligned
            av_test_chance_exc_aligned0 = av_test_chance_exc_aligned
        av_test_data_allExc_aligned0 = av_test_data_allExc_aligned
        av_test_shfl_allExc_aligned0 = av_test_shfl_allExc_aligned
        av_test_chance_allExc_aligned0 = av_test_chance_allExc_aligned


    
    # Align p values of all days
    ttest_pval_inh_aligned = alTrace(ttest_pval_inh, eventI_ds_allDays, nPreMin, nPostMin)
    ttest_pval_allExc_aligned = alTrace(ttest_pval_allExc, eventI_ds_allDays, nPreMin, nPostMin)
    ttest_pval_exc_aligned = []
    for iexc in range(numExcSamples):
        ttest_pval_exc_aligned.append(alTrace(ttest_pval_exc[iexc], eventI_ds_allDays, nPreMin, nPostMin))
    ttest_pval_exc_aligned = np.array(ttest_pval_exc_aligned) # excSamps x frs x days
    
    if decodeStimCateg or decodeOutcome:
        if ignore==0:
            ttest_pval_inh_aligned_remCorr = alTrace(ttest_pval_inh_remCorr, eventI_ds_allDays, nPreMin, nPostMin)
            ttest_pval_allExc_aligned_remCorr = alTrace(ttest_pval_allExc_remCorr, eventI_ds_allDays, nPreMin, nPostMin)
            ttest_pval_exc_aligned_remCorr = []
            for iexc in range(numExcSamples):
                ttest_pval_exc_aligned_remCorr.append(alTrace(ttest_pval_exc_remCorr[iexc], eventI_ds_allDays, nPreMin, nPostMin))
            ttest_pval_exc_aligned_remCorr = np.array(ttest_pval_exc_aligned_remCorr) # excSamps x frs x days        
     
            ttest_pval_inh_aligned_both = alTrace(ttest_pval_inh_both, eventI_ds_allDays, nPreMin, nPostMin)
            ttest_pval_allExc_aligned_both = alTrace(ttest_pval_allExc_both, eventI_ds_allDays, nPreMin, nPostMin)
            ttest_pval_exc_aligned_both = []
            for iexc in range(numExcSamples):
                ttest_pval_exc_aligned_both.append(alTrace(ttest_pval_exc_both[iexc], eventI_ds_allDays, nPreMin, nPostMin))
            ttest_pval_exc_aligned_both = np.array(ttest_pval_exc_aligned_both) # excSamps x frs x days        
        
        # keep a copy of testing data vars:        
        ttest_pval_inh_aligned0 = ttest_pval_inh_aligned
        ttest_pval_allExc_aligned0 = ttest_pval_allExc_aligned
        ttest_pval_exc_aligned0 = ttest_pval_exc_aligned
        
        
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

    
    #%% In decodeStimCateg, decide which one (testing, remCorr, or Both) you want to use for the rest of this code

    if decodeStimCateg or decodeOutcome:    
        if use_both_remCorr_testing[0]==1: # use Both
            av_test_data_inh_aligned = av_test_both_data_inh_aligned
            av_test_shfl_inh_aligned = av_test_both_shfl_inh_aligned
            av_test_chance_inh_aligned = av_test_both_chance_inh_aligned
            if do_excInhHalf==0:
                av_test_data_exc_aligned = av_test_both_data_exc_aligned
                av_test_shfl_exc_aligned = av_test_both_shfl_exc_aligned
                av_test_chance_exc_aligned = av_test_both_chance_exc_aligned
            av_test_data_allExc_aligned = av_test_both_data_allExc_aligned
            av_test_shfl_allExc_aligned = av_test_both_shfl_allExc_aligned
            av_test_chance_allExc_aligned = av_test_both_chance_allExc_aligned        
            # pval
            ttest_pval_inh_aligned = ttest_pval_inh_aligned_both
            ttest_pval_allExc_aligned = ttest_pval_allExc_aligned_both
            ttest_pval_exc_aligned = ttest_pval_exc_aligned_both
    
            
        elif use_both_remCorr_testing[1]==1: # use remCorr
            av_test_data_inh_aligned = av_test_remCorr_data_inh_aligned
            av_test_shfl_inh_aligned = av_test_remCorr_shfl_inh_aligned
            av_test_chance_inh_aligned = av_test_remCorr_chance_inh_aligned
            if do_excInhHalf==0:
                av_test_data_exc_aligned = av_test_remCorr_data_exc_aligned
                av_test_shfl_exc_aligned = av_test_remCorr_shfl_exc_aligned
                av_test_chance_exc_aligned = av_test_remCorr_chance_exc_aligned
            av_test_data_allExc_aligned = av_test_remCorr_data_allExc_aligned
            av_test_shfl_allExc_aligned = av_test_remCorr_shfl_allExc_aligned
            av_test_chance_allExc_aligned = av_test_remCorr_chance_allExc_aligned
            # pval
            ttest_pval_inh_aligned = ttest_pval_inh_aligned_remCorr
            ttest_pval_allExc_aligned = ttest_pval_allExc_aligned_remCorr
            ttest_pval_exc_aligned = ttest_pval_exc_aligned_remCorr
                
        elif use_both_remCorr_testing[2]==1: # use testing data (cv)
            av_test_data_inh_aligned = av_test_data_inh_aligned0
            av_test_shfl_inh_aligned = av_test_shfl_inh_aligned0
            av_test_chance_inh_aligned = av_test_chance_inh_aligned0
            if do_excInhHalf==0:
                av_test_data_exc_aligned = av_test_data_exc_aligned0
                av_test_shfl_exc_aligned = av_test_shfl_exc_aligned0
                av_test_chance_exc_aligned = av_test_chance_exc_aligned0
            av_test_data_allExc_aligned = av_test_data_allExc_aligned0
            av_test_shfl_allExc_aligned = av_test_shfl_allExc_aligned0
            av_test_chance_allExc_aligned = av_test_chance_allExc_aligned0
            # pval
            ttest_pval_inh_aligned = ttest_pval_inh_aligned0
            ttest_pval_allExc_aligned = ttest_pval_allExc_aligned0
            ttest_pval_exc_aligned = ttest_pval_exc_aligned0
            
            
    #%% Save aligned vars (CVsample-averaged CA vars for each day)
    """
    if decodeStimCateg:
        nnow = 'decodeStimCateg_'
    else:
        nnow = 'decodeChoice_'
    finamen = os.path.join(fname, 'svm_CA_' + nnow + nowStr+'.mat')
        
    scio.savemat(finamen, {'av_test_data_allExc_aligned':av_test_data_allExc_aligned, 'av_test_data_exc_aligned':av_test_data_exc_aligned, 'av_test_data_inh_aligned':av_test_data_inh_aligned, 'nPreMin':nPreMin, 'days2an_heatmap':days2an_heatmap, 'behCorr_all':behCorr_all})
    """
        
#    no
        
    #%% Define function to find the onset of emergence of choice signal
     
    def sigOnset(top, th1):
        solns_allD = []
        for iday in range(len(top)):
            values = top[iday]        
            searchval = np.ones((th1))
            
            N = len(searchval)
            possibles = np.where(values == searchval[0])[0]
            
            solns = []
            for p in possibles:
                check = values[p:p+N]
                if np.all(check == searchval):
                    solns.append(p)
            if len(solns)==0:
                solns = [np.nan]        
            solns_allD.append(solns)
    
        # find the first time bin ... as the onset of emergece of choice signal.
        firstFrame = np.array([solns_allD[iday][0] for iday in range(len(top))])
        # compute onset timing of choice signal relative to animal's choice time
        firstFrame = (firstFrame - nPreMin)*frameLength*regressBins
        
        return firstFrame
    
    
    #%% For each day find the onset of emergence of choice signal. Then see how the onset of choice signal varies with training days. Also do a scatter plot of the onset time vs. behavioral performance.
    
    sigDur = 500 # ms # we need 500ms continuous significancy for CA of data vs shuffle, to count it as choice signal emergence.
    # choice signal onset will be set to nan if the above criteria is not met (ie there is not 500ms continuous sig CA data vs shfl)
    
    te = ttest_pval_exc_aligned[rng.permutation(numExcSamples)[0]].T  < alph  # a random exc samp    #    te = np.mean(ttest_pval_exc_aligned, axis=0).T[days2an_heatmap]  < alph  # ave exc samps       
    topall = ttest_pval_allExc_aligned.T < alph , te , ttest_pval_inh_aligned.T < alph
    
    th1 = np.round(sigDur / (frameLength*regressBins))

    # find all time bins that are followed by 500ms of sig CA.
    for iplot in range(3): # iplot = 0    
        top = topall[iplot]                    
        firstFrame = sigOnset(top, th1)

        if iplot==0:
            firstFrame_allN = firstFrame        
        if iplot==1:
            firstFrame_exc = firstFrame
        if iplot==2:
            firstFrame_inh = firstFrame
            
    firstFrame_exc_randSamp = firstFrame_exc
    
    # find firstFrame for each exc samp, then take an average across the exc samps!
    a = np.transpose((ttest_pval_exc_aligned<alph), (0,2,1)) # nExcSamps x days x frames
    firstFrame = [] # nExcSamps x nDays
    for isamp in range(numExcSamples):
        firstFrame.append(sigOnset(a[isamp], th1))    
    # average across excSamps
    firstFrame_exc_avSamps = np.nanmean(firstFrame, axis=0)

            

    #%% Save vars
    """
    if corrTrained:
        corrn0 = 'corr_'
    else:
        corrn0 = corrn
        
    scio.savemat(fname + '/svm_' + corrn0 + 'classAcc_nNsPlateau_' + nowStr, {'mousename': mousename, 'numD':numD, 'mn_corr':mn_corr, 'thTrained':thTrained, 'eventI_ds_allDays':eventI_ds_allDays,
      'av_test_data_allExc_aligned':av_test_data_allExc_aligned,
      'av_test_data_exc_aligned':av_test_data_exc_aligned,
      'av_test_data_inh_aligned':av_test_data_inh_aligned,
      'firstFrame_allN':firstFrame_allN,
      'firstFrame_inh':firstFrame_inh,
      'firstFrame_exc_randSamp':firstFrame_exc_randSamp,
      'firstFrame_exc_avSamps':firstFrame_exc_avSamps})
    """
     
            
    #%% Average and standard DEVIATION across days (each day includes the average class accuracy across samples.)
       
    doSd = 0 # if 1, compute standarad deviation; if 0, compute standard erro
    if doSd:
        nsa = 1
    else:
#        nsa = np.sqrt(numDaysGood)
        nsa = np.sqrt(sum(days2an_noExtraStim))
        
    av_av_test_data_inh_aligned = np.nanmean(av_test_data_inh_aligned[:,days2an_noExtraStim], axis=1)
    sd_av_test_data_inh_aligned = np.nanstd(av_test_data_inh_aligned[:,days2an_noExtraStim], axis=1) / nsa
    av_av_test_shfl_inh_aligned = np.nanmean(av_test_shfl_inh_aligned[:,days2an_noExtraStim], axis=1)
    sd_av_test_shfl_inh_aligned = np.nanstd(av_test_shfl_inh_aligned[:,days2an_noExtraStim], axis=1) / nsa
    av_av_test_chance_inh_aligned = np.nanmean(av_test_chance_inh_aligned[:,days2an_noExtraStim], axis=1)
    sd_av_test_chance_inh_aligned = np.nanstd(av_test_chance_inh_aligned[:,days2an_noExtraStim], axis=1) / nsa
    if do_excInhHalf==0:
        av_av_test_data_exc_aligned = np.nanmean(av_test_data_exc_aligned[:,days2an_noExtraStim], axis=1)
        sd_av_test_data_exc_aligned = np.nanstd(av_test_data_exc_aligned[:,days2an_noExtraStim], axis=1) / nsa    
        av_av_test_shfl_exc_aligned = np.nanmean(av_test_shfl_exc_aligned[:,days2an_noExtraStim], axis=1)
        sd_av_test_shfl_exc_aligned = np.nanstd(av_test_shfl_exc_aligned[:,days2an_noExtraStim], axis=1) / nsa    
        av_av_test_chance_exc_aligned = np.nanmean(av_test_chance_exc_aligned[:,days2an_noExtraStim], axis=1)
        sd_av_test_chance_exc_aligned = np.nanstd(av_test_chance_exc_aligned[:,days2an_noExtraStim], axis=1) / nsa    
    av_av_test_data_allExc_aligned = np.nanmean(av_test_data_allExc_aligned[:,days2an_noExtraStim], axis=1)
    sd_av_test_data_allExc_aligned = np.nanstd(av_test_data_allExc_aligned[:,days2an_noExtraStim], axis=1) / nsa
    av_av_test_shfl_allExc_aligned = np.nanmean(av_test_shfl_allExc_aligned[:,days2an_noExtraStim], axis=1)
    sd_av_test_shfl_allExc_aligned = np.nanstd(av_test_shfl_allExc_aligned[:,days2an_noExtraStim], axis=1) / nsa
    av_av_test_chance_allExc_aligned = np.nanmean(av_test_chance_allExc_aligned[:,days2an_noExtraStim], axis=1)
    sd_av_test_chance_allExc_aligned = np.nanstd(av_test_chance_allExc_aligned[:,days2an_noExtraStim], axis=1) / nsa
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
            

    
#    no
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PLOTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    #%% Compare number of inh vs exc for each day
    
    if loadWeights:
        plt.figure(figsize=(4.5,3)) 
        
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
                dd = 'chAl_' + corrn + 'numNeurons_days_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            else:
                dd = 'stAl_' + corrn + 'numNeurons_days_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' +nowStr
                
            d = os.path.join(svmdir+dnow)
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)
                    
            fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        

    
    
    #%% Heatmaps of class accuracy showing all days

    sepCB = 1 # if 1: plot inh and exc on a separate c axis scale than allN    
    asp = 'auto' #2
    cmap ='jet'
    
    if chAl==1:
        xl = 'Time since choice onset (ms)'
    else:
        xl = 'Time since stimulus onset (ms)'
        
    for iplot in [-1,0,1,2]: # iplot = 0   

        if iplot==-1: ### plot class accuracy for each day: data - shfl
            if do_excInhHalf==0:
                topall = av_test_data_inh_aligned.T[days2an_heatmap] - av_test_shfl_inh_aligned.T[days2an_heatmap], av_test_data_exc_aligned.T[days2an_heatmap] - av_test_shfl_exc_aligned.T[days2an_heatmap], av_test_data_allExc_aligned.T[days2an_heatmap] - av_test_shfl_allExc_aligned.T[days2an_heatmap]
            else:
                topall = av_test_data_inh_aligned.T[days2an_heatmap] - av_test_shfl_inh_aligned.T[days2an_heatmap], av_test_data_allExc_aligned.T[days2an_heatmap] - av_test_shfl_allExc_aligned.T[days2an_heatmap]
            namp = 'classAccur_dataMshfl'
            cblab = 'Class accuracy (%) (data-shfl)'
        
        if iplot==0: ### plot class accuracy for each day: data
            if do_excInhHalf==0:
                topall = av_test_data_inh_aligned.T[days2an_heatmap], av_test_data_exc_aligned.T[days2an_heatmap], av_test_data_allExc_aligned.T[days2an_heatmap]
            else:
                topall = av_test_data_inh_aligned.T[days2an_heatmap], av_test_data_allExc_aligned.T[days2an_heatmap]
            namp = 'classAccur_'
            cblab = 'Class accuracy (%)'
 
        if iplot==1: ### plot significancy of CA for each day (data vs shfl)
            if do_excInhHalf==0:
                te = ttest_pval_exc_aligned[rng.permutation(numExcSamples)[0]].T[days2an_heatmap]  < alph  # a random exc samp
    #            te = np.mean(ttest_pval_exc_aligned, axis=0).T[days2an_heatmap]  < alph  # ave exc samps       
                topall = ttest_pval_inh_aligned.T[days2an_heatmap] < alph, te, ttest_pval_allExc_aligned.T[days2an_heatmap] < alph
            else:
                topall = ttest_pval_inh_aligned.T[days2an_heatmap] < alph, ttest_pval_allExc_aligned.T[days2an_heatmap] < alph
            namp = 'sigDataShfl_'
            cblab = 'Sig class accuracy (data vs shfl)'
            
        if iplot==2: ### plot p val of CA for each day (data vs shfl)
            if do_excInhHalf==0:
                te = ttest_pval_exc_aligned[rng.permutation(numExcSamples)[0]].T[days2an_heatmap]  # a random exc samp
#                te = np.mean(ttest_pval_exc_aligned, axis=0).T[days2an_heatmap]  # ave exc samps       
                topall = ttest_pval_inh_aligned.T[days2an_heatmap], te, ttest_pval_allExc_aligned.T[days2an_heatmap]
            else:
                topall = ttest_pval_inh_aligned.T[days2an_heatmap], ttest_pval_allExc_aligned.T[days2an_heatmap]
            namp = 'pDataShfl_'
            cblab = 'P class accuracy (data vs shfl)'

        topi = topall[0]
        tope = topall[1]
        if do_excInhHalf==0:
            topa = topall[2]
        else:
            topa = topall[1]
            tope = topall[0] # when exc is not run (ie when you are waiting for it to be done), just make it like inh so the code doesnt give error            
        
        
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7,5))
        
        if sepCB: # plot inh and exc on a different c axis scale
            cminc = np.floor(np.min([np.nanmin(topi), np.nanmin(tope)]))
            cmaxc = np.ceil(np.max([np.nanmax(topi), np.nanmax(tope)]))
            cbn = 'sepColorbar_'
        else:
            cminc = np.floor(np.min([np.nanmin(topi), np.nanmin(tope), np.nanmin(topa)]))
            cmaxc = np.ceil(np.max([np.nanmax(topi), np.nanmax(tope), np.nanmax(topa)]))            
            cbn = ''
        
        lab = 'inh'
        top = topi
        ax = axes.flat[0]
        img = plotStabScore(top, lab, cminc, cmaxc, cmap, cblab, ax, xl)
        ax.set_aspect(asp)
        makeNicePlots(ax,1)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()        
     
       
        lab = 'exc'
        top = tope
        ax = axes.flat[1]
        img = plotStabScore(top, lab, cminc, cmaxc, cmap, cblab, ax, xl)
    #    y=-1; pp = np.full(ps.shape, np.nan); pp[pc<=.05] = y
    #    ax.plot(range(len(time_aligned)), pp, color='r', lw=2)
        ax.set_aspect(asp)
        makeNicePlots(ax,1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)        
        if sepCB: # plot inh and exc on a different c axis scale
            cb_ax = fig.add_axes([0, 0.15, 0.02, 0.72])
            cbar = fig.colorbar(img, cax=cb_ax, label='')
        
        
        ### allN
        if sepCB:
            cminc = np.floor(np.min([np.nanmin(topi), np.nanmin(tope), np.nanmin(topa)]))
            cmaxc = np.ceil(np.max([np.nanmax(topi), np.nanmax(tope), np.nanmax(topa)]))
        
        lab = labAll
        top = topa
        ax = axes.flat[2]
        img = plotStabScore(top, lab, cminc, cmaxc, cmap, cblab, ax, xl)
        #plt.colorbar(label=cblab) #,fraction=0.14, pad=0.04); #plt.clim(cmins, cmaxs)
        #add_colorbar(img)
        cb_ax = fig.add_axes([0.92, 0.15, 0.02, 0.72])
        cbar = fig.colorbar(img, cax=cb_ax, label=cblab)
        #cbar = fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.95)
        #fig.colorbar(img, ax=axes.ravel().tolist())
        # Make an axis for the colorbar on the right side
        #from mpl_toolkits.axes_grid1 import make_axes_locatable
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)
        #fig.colorbar(img, cax=cax)
    #    y=-1; pp = np.full(ps.shape, np.nan); pp[pca<=.05] = y
    #    ax.plot(range(len(time_aligned)), pp, color='r', lw=2)
        ax.set_aspect(asp)
        makeNicePlots(ax,1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
     
        
        plt.subplots_adjust(wspace=.4)
        
        
        ##%% Save the figure    
        if savefigs:
            # _sepColorbar
            d = os.path.join(svmdir+dnow) #,mousename)       
    #            daysnew = (np.array(days))[dayinds]
            if chAl==1:
                dd = 'chAl_' + corrn + 'eachDay_heatmap_' + cbn + namp + labAll+'_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            else:
                dd = 'stAl_' + corrn + 'eachDay_heatmap_' + cbn + namp + labAll+'_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr       
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)            
            fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
            
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)



    #%% 
    #########################################################################################################
    ########################## Plots : svm performance vs behavioral performance ############################
    #########################################################################################################
#    
#    perc_thb = [10,90] #[20,80] #  # percentiles of behavioral performance for determining low and high performance.
#    perc_thb = [15,85]
    perc_thb = [20,80]
#    perc_thb = [20,70]
#    perc_thb = [10,80]    
    
    #%% Plot CA vs. day, also compare CA for days with low vs high behavioral performance   
    
    plt.figure(figsize=(8,6))
    
    gs = gridspec.GridSpec(2, 4)#, width_ratios=[2, 1])  #    h1 = gs[0,0:2]
    h2 = gs[0,2:3]

    #################################### Plot CA vs. day ###########################################################
    plt.subplot(221)
    plt.plot(av_test_data_allExc_aligned[nPreMin-1, days2an_heatmap], marker='.', color='k', label='allN')
    plt.plot(av_test_data_exc_aligned[nPreMin-1, days2an_heatmap], marker='.', color='b', label='exc')
    plt.plot(av_test_data_inh_aligned[nPreMin-1, days2an_heatmap], marker='.', color='r', label='inh')
    plt.xlabel('Training day')
    plt.ylabel('Class accuracy (%)')
    plt.legend(loc=0, frameon=False, bbox_to_anchor=(.5, .7))
    makeNicePlots(plt.gca())
    
    #################################### Plot beh performance vs. day ##############################################
    plt.subplot(223)
    plt.plot(behCorr_all[days2an_heatmap])
    plt.plot(behCorrHR_all[days2an_heatmap], linestyle=':', label='HR trs')
    plt.plot(behCorrLR_all[days2an_heatmap], linestyle=':', label='LR trs')
    plt.xlabel('Training day')
    plt.ylabel('Behavior (fract. correct, easy trs)')
    plt.legend(loc=0, frameon=False, bbox_to_anchor=(.5, .7))
    makeNicePlots(plt.gca())
    
    
    #################################### Compare CA for days with low vs high behavioral performance ###############
    a = behCorr_all[days2an_heatmap]    
    thb = np.percentile(a, perc_thb)
#    mn = np.min(a);     mx = np.max(a);     thb = mn+.1, mx-.1    
    
    loBehCorrDays = (a <= thb[0])    
    hiBehCorrDays = (a >= thb[1])
    print sum(loBehCorrDays), sum(hiBehCorrDays), ': num days with low and high beh performance'
    
    
    aa = av_test_data_allExc_aligned[nPreMin-1, days2an_heatmap] 
    bb = av_test_data_inh_aligned[nPreMin-1, days2an_heatmap] 
    cc = av_test_data_exc_aligned[nPreMin-1, days2an_heatmap] 
    if same_HRLR_acrossDays: # we subtract the shuffle and then add 50 to it, this is bc when same_HRLR_acrossDays, shfl is not really at 50 (due to low trial num)
        aa = aa - av_test_shfl_allExc_aligned[nPreMin-1, days2an_heatmap] + 50 
        bb = bb - av_test_shfl_inh_aligned[nPreMin-1, days2an_heatmap] + 50
        cc = cc - av_test_shfl_exc_aligned[nPreMin-1, days2an_heatmap] + 50        
    a = aa[loBehCorrDays]; ah = aa[hiBehCorrDays]
    b = bb[loBehCorrDays]; bh = bb[hiBehCorrDays]
    c = cc[loBehCorrDays]; ch = cc[hiBehCorrDays]    
    
    ####### errorbar: mean and st error: # compare CA for days with low vs high behavioral performance        
#    plt.figure(figsize=(1.5,2.5)        
    plt.subplot(h2)
    plt.errorbar([0], [ah.mean()], [ah.std()/np.sqrt(len(ah))], marker='o', color='k', fmt='.', markeredgecolor='k', label='high beh')
    plt.errorbar([0], [a.mean()], [a.std()/np.sqrt(len(a))], marker='o', color='gray', fmt='.', markeredgecolor='gray', label='low beh')    
    plt.errorbar([1], [ch.mean()], [ch.std()/np.sqrt(len(ch))], marker='o', color='b', fmt='.', markeredgecolor='b')
    plt.errorbar([1], [c.mean()], [c.std()/np.sqrt(len(c))], marker='o', color='lightblue', fmt='.', markeredgecolor='lightblue')       
    plt.errorbar([2], [bh.mean()], [bh.std()/np.sqrt(len(bh))], marker='o',color='r', fmt='.', markeredgecolor='r')        
    plt.errorbar([2], [b.mean()], [b.std()/np.sqrt(len(b))], marker='o',color='lightsalmon', fmt='.', markeredgecolor='lightsalmon')        
    
    plt.xlim([-1,3])    
    plt.xticks([0,1,2], ('allN', 'exc', 'inh')) # , rotation='vertical'
    plt.ylabel('Class accur (%)')    
    yl = plt.gca().get_ylim()
    r = np.diff(yl)
    plt.ylim([yl[0], yl[1]+r/10.])
    plt.legend(loc=0, frameon=False, numpoints=1, bbox_to_anchor=(.5, 1))
    makeNicePlots(plt.gca(),0,1)
    
    plt.subplots_adjust(hspace=.4, wspace=.4)
    
    ##%% Save the figure    
    if savefigs:
        dayinds = np.arange(len(days))
        dayinds = np.delete(dayinds, np.argwhere(mn_corr < thTrained))
        daysGood = np.array(days)[dayinds]
        
        d = os.path.join(svmdir+dnow) #,mousename)       
    #            daysnew = (np.array(days))[dayinds]
        if chAl==1:
            dd = 'chAl_' + corrn + 'classAccur_vs_days_loHiBeh_'+'perc'+str(perc_thb)+'_inhExc'+labAll+'_'  + daysGood[0][0:6] + '-to-' + daysGood[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_' + corrn + 'classAccur_vs_days_loHiBeh_'+'perc'+str(perc_thb)+'_inhExc'+labAll+'_'  + daysGood[0][0:6] + '-to-' + daysGood[-1][0:6] + '_' + nowStr
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)            
        fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
    
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,) 

        
    #%% Define function to do scatter plots : svm performance vs behavioral performance
    
    def plotscat(x, y, xlab, ylab, clab):
        
        plt.scatter(x, y, c=np.arange(len(y)), cmap=cm.jet, edgecolors='face', s=8) #, label='class accuracy (% correct testing trials)')
        
        plt.xlabel(xlab)#'beh (fract corr, all trs)') #'behavior (Fraction correct, all trials)'
        plt.ylabel(ylab)    
        plt.colorbar(label='days')
    #    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False, numpoints=1)     
        ccc = np.corrcoef(x,y)[0, 1]
    #    ccs = np.corrcoef(x,y2)[0, 1]
        plt.title('r(%s,beh) = %.2f' %(clab, ccc), position=[.5,1.05])#, ccs))
        makeNicePlots(plt.gca(),1,1)
    
    
    #%% Scatter plots, svm performance vs behavioral performance

    from matplotlib import cm
 
   # plot vs. beh performance on all trials (half hr, half lr)    
    plt.figure(figsize=(9, 12))
        
    clab = 'accur'    
    
    #### allN 
    y = av_test_data_allExc_aligned[nPreMin-1, days2an_heatmap] # CA at time bin -1
    ylab = 'AllN class accur(%)'
    
    a = plt.subplot(431)   
    x = behCorr_all[days2an_heatmap] #Mice[im]
    xlab = 'beh (fract corr, all trs)'    
    plotscat(x, y, xlab, ylab, clab)    
    
    plt.subplot(432)   
    x = behCorrHR_all[days2an_heatmap] #Mice[im]
    xlab = 'beh (fract corr, HR trs)'    
    plotscat(x, y, xlab, ylab, clab)
    
    plt.subplot(433)   
    x = behCorrLR_all[days2an_heatmap] #Mice[im]
    xlab = 'beh (fract corr, LR trs)'    
    plotscat(x, y, xlab, ylab, clab)
    

    #### exc
    y = av_test_data_exc_aligned[nPreMin-1, days2an_heatmap] # CA at time bin -1
    ylab = 'Exc class accur(%)'
    
    a = plt.subplot(434)   
    x = behCorr_all[days2an_heatmap] #Mice[im]
    xlab = 'beh (fract corr, all trs)'    
    plotscat(x, y, xlab, ylab, clab)    
    
    plt.subplot(435)   
    x = behCorrHR_all[days2an_heatmap] #Mice[im]
    xlab = 'beh (fract corr, HR trs)'    
    plotscat(x, y, xlab, ylab, clab)
    
    plt.subplot(436)   
    x = behCorrLR_all[days2an_heatmap] #Mice[im]
    xlab = 'beh (fract corr, LR trs)'    
    plotscat(x, y, xlab, ylab, clab)


    #### inh
    y = av_test_data_inh_aligned[nPreMin-1, days2an_heatmap] # CA at time bin -1
    ylab = 'Inh class accur(%)'    
    
    a = plt.subplot(437)   
    x = behCorr_all[days2an_heatmap] #Mice[im]
    xlab = 'beh (fract corr, all trs)'    
    plotscat(x, y, xlab, ylab, clab)    
    
    plt.subplot(438)   
    x = behCorrHR_all[days2an_heatmap] #Mice[im]
    xlab = 'beh (fract corr, HR trs)'    
    plotscat(x, y, xlab, ylab, clab)
    
    plt.subplot(439)   
    x = behCorrLR_all[days2an_heatmap] #Mice[im]
    xlab = 'beh (fract corr, LR trs)'    
    plotscat(x, y, xlab, ylab, clab)


    plt.subplots_adjust(wspace=.8, hspace=1)
    
    
    ##%% Save the figure    
    if savefigs:
        
        d = os.path.join(svmdir+dnow) #,mousename)       
    #            daysnew = (np.array(days))[dayinds]
        if chAl==1:
            dd = 'chAl_' + corrn + 'classAccur_beh_scatter_inhExc'+labAll+'_'  + daysGood[0][0:6] + '-to-' + daysGood[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_' + corrn + 'classAccur_beh_scatter_inhExc'+labAll+'_'  + daysGood[0][0:6] + '-to-' + daysGood[-1][0:6] + '_' + nowStr
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)            
        fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
    
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)           
        


    
    #%% 
    ##############################################################################################################
    ###################################### Plots of choice signal onset ##########################################
    ##############################################################################################################

    #%% Only get good days
    
    firstFrame_allN = firstFrame_allN[days2an_heatmap] # ('150903_1') ... it is one of the early days and it is significant all over the trial!!! (rerun svm for this day!)        
    firstFrame_inh = firstFrame_inh[days2an_heatmap]
    firstFrame_exc_randSamp = firstFrame_exc_randSamp[days2an_heatmap]
    firstFrame_exc_avSamps = firstFrame_exc_avSamps[days2an_heatmap]
        
        
    #%% NOTE: do you want to do the following??
    
    if np.logical_and(mousename=='fni16', corrTrained==1):
        firstFrame_allN[days.index('150903_1')] = np.nan # ('150903_1') ... it is one of the early days and it is significant all over the trial!!! (rerun svm for this day!)        
        firstFrame_inh[days.index('150903_1')] = np.nan
        firstFrame_exc_randSamp[days.index('150903_1')] = np.nan
        firstFrame_exc_avSamps[days.index('150903_1')] = np.nan
        
        
    #%% Plot choice signal onset vs. day, also compare choice signal onset for days with low vs high behavioral performance #########%%       

#    perc_thb = [10,90] #[20,80] # perc_thb = [15,85] # percentiles of behavioral performance for determining low and high performance.
    exc_av = 1 # if 1 use firstFrame_exc_avSamps # use firstFrame found for a random exc samp or found for each exc samp and then averaged!
    
    if exc_av:
        firstFrame_exc = firstFrame_exc_avSamps
    else:
        firstFrame_exc = firstFrame_exc_randSamp
        

    plt.figure(figsize=(8,6)) # plt.figure(figsize=(4,3))
    gs = gridspec.GridSpec(2, 4)#, width_ratios=[2, 1])  #    h1 = gs[0,0:2]
    h2 = gs[0,2:3]    
    
    plt.subplot(221)    
    plt.plot(firstFrame_allN, marker='.', color='k')
    plt.plot(firstFrame_exc, marker='.', color='b')
    plt.plot(firstFrame_inh, marker='.', color='r')
    plt.xlabel('Training day')
    plt.ylabel('Choice signal onset (ms)\nrel. animal choice')
    makeNicePlots(plt.gca())


    #########%% Compare choice signal onset for days with low vs high behavioral performance #########%%      
    #### allN
    a = behCorr_all[days2an_heatmap][~np.isnan(firstFrame_allN)] 
    thb = np.percentile(a, perc_thb)
#    mn = np.min(a);    mx = np.max(a);    thb = mn+.1, mx-.1        
    
    loBehCorrDays_allN = (a <= thb[0])     # bc different days are valid for allN, inh and exc, we have to find low and high beh performance days separately for each population.
    hiBehCorrDays_allN = (a >= thb[1])
    print sum(loBehCorrDays_allN), sum(hiBehCorrDays_allN), ': num days with low and high beh performance'

    #### inh
    a = behCorr_all[days2an_heatmap][~np.isnan(firstFrame_inh)]     
    thb = np.percentile(a, perc_thb)
    
    loBehCorrDays_inh = (a <= thb[0])    
    hiBehCorrDays_inh = (a >= thb[1])
    print sum(loBehCorrDays_inh), sum(hiBehCorrDays_inh), ': num days with low and high beh performance'

    #### exc
    a = behCorr_all[days2an_heatmap][~np.isnan(firstFrame_exc)] 
    thb = np.percentile(a, perc_thb)
    
    loBehCorrDays_exc = (a <= thb[0])    
    hiBehCorrDays_exc = (a >= thb[1])
    print sum(loBehCorrDays_exc), sum(hiBehCorrDays_exc), ': num days with low and high beh performance'

    
    aa = firstFrame_allN[~np.isnan(firstFrame_allN)]
    bb = firstFrame_inh[~np.isnan(firstFrame_inh)]
    cc = firstFrame_exc[~np.isnan(firstFrame_exc)]
    a = aa[loBehCorrDays_allN]; ah = aa[hiBehCorrDays_allN]
    b = bb[loBehCorrDays_inh]; bh = bb[hiBehCorrDays_inh]
    c = cc[loBehCorrDays_exc]; ch = cc[hiBehCorrDays_exc]    
    
    ################ errorbar: mean and st error: # compare CA for days with low vs high behavioral performance        
#    plt.figure(figsize=(1.5,2.5))
    plt.subplot(h2)
    plt.errorbar([0], [ah.mean()], [ah.std()/np.sqrt(len(ah))], marker='o', color='k', fmt='.', markeredgecolor='k', label='high beh')
    plt.errorbar([0], [a.mean()], [a.std()/np.sqrt(len(a))], marker='o', color='gray', fmt='.', markeredgecolor='gray', label='low beh')    
    plt.errorbar([1], [ch.mean()], [ch.std()/np.sqrt(len(ch))], marker='o', color='b', fmt='.', markeredgecolor='b')
    plt.errorbar([1], [c.mean()], [c.std()/np.sqrt(len(c))], marker='o', color='lightblue', fmt='.', markeredgecolor='lightblue')       
    plt.errorbar([2], [bh.mean()], [bh.std()/np.sqrt(len(bh))], marker='o',color='r', fmt='.', markeredgecolor='r')        
    plt.errorbar([2], [b.mean()], [b.std()/np.sqrt(len(b))], marker='o',color='lightsalmon', fmt='.', markeredgecolor='lightsalmon')        
    
    plt.xlim([-1,3])    
    plt.xticks([0,1,2], ('allN', 'exc', 'inh')) # , rotation='vertical'
#    plt.ylabel('Choice signal onset (ms)\nrel. animal choice')    
    yl = plt.gca().get_ylim()
    r = np.diff(yl)
    plt.ylim([yl[0], yl[1]+r/10.])
    plt.legend(frameon=False, numpoints=1, bbox_to_anchor=(2, 1))
    makeNicePlots(plt.gca(),0,1)
    
    plt.subplots_adjust(hspace=.4, wspace=.4)


    ##%% Save the figure    
    if savefigs:
        d = os.path.join(svmdir+dnow) #,mousename)       
    #            daysnew = (np.array(days))[dayinds]
        if chAl==1:
            dd = 'chAl_' + corrn + 'onsetChoice_vs_days_inhExc'+labAll+'_'  + daysGood[0][0:6] + '-to-' + daysGood[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_' + corrn + 'onsetChoice_vs_days_inhExc'+labAll+'_'  + daysGood[0][0:6] + '-to-' + daysGood[-1][0:6] + '_' + nowStr
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)            
        fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
    
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)           
   
    
    #%% Scatter plot of choice signal onset vs behavioral performance

    # plot vs. beh performance on all trials (half hr, half lr)    
    plt.figure(figsize=(9, 12))    
    clab = 'onset'    
    
    ##### allN
    valD = ~np.isnan(firstFrame_allN)
    y = firstFrame_allN[valD]
    ylab = 'AllN choice signal onset (ms)\nrel. animal choice'    
    a = plt.subplot(431)   
    x = behCorr_all[days2an_heatmap][valD] #Mice[im]
    xlab = 'beh (fract corr, all trs)'    
    plotscat(x, y, xlab, ylab, clab)    
    
    plt.subplot(432)   
    ylab = ''
    x = behCorrHR_all[days2an_heatmap][valD] #Mice[im]
    xlab = 'beh (fract corr, HR trs)'    
    plotscat(x, y, xlab, ylab, clab)
    
    plt.subplot(433)   
    ylab = ''
    x = behCorrLR_all[days2an_heatmap][valD] #Mice[im]
    xlab = 'beh (fract corr, LR trs)'    
    plotscat(x, y, xlab, ylab, clab)
    

    ##### exc
    valD = ~np.isnan(firstFrame_exc)
    y = firstFrame_exc[valD]
    ylab = 'Exc choice signal onset (ms)\nrel. animal choice'    
    a = plt.subplot(434)   
    x = behCorr_all[days2an_heatmap][valD] #Mice[im]
    xlab = 'beh (fract corr, all trs)'    
    plotscat(x, y, xlab, ylab, clab)    
    
    plt.subplot(435)   
    ylab = ''
    x = behCorrHR_all[days2an_heatmap][valD] #Mice[im]
    xlab = 'beh (fract corr, HR trs)'    
    plotscat(x, y, xlab, ylab, clab)
    
    plt.subplot(436)   
    ylab = ''
    x = behCorrLR_all[days2an_heatmap][valD] #Mice[im]
    xlab = 'beh (fract corr, LR trs)'    
    plotscat(x, y, xlab, ylab, clab)


    ##### inh
    valD = ~np.isnan(firstFrame_inh)
    y = firstFrame_inh[valD]
    ylab = 'Inh choice signal onset (ms)\nrel. animal choice'    
    a = plt.subplot(437)   
    x = behCorr_all[days2an_heatmap][valD] #Mice[im]
    xlab = 'beh (fract corr, all trs)'    
    plotscat(x, y, xlab, ylab, clab)    
    
    plt.subplot(438)   
    ylab = ''
    x = behCorrHR_all[days2an_heatmap][valD] #Mice[im]
    xlab = 'beh (fract corr, HR trs)'    
    plotscat(x, y, xlab, ylab, clab)
    
    plt.subplot(439)   
    ylab = ''
    x = behCorrLR_all[days2an_heatmap][valD] #Mice[im]
    xlab = 'beh (fract corr, LR trs)'    
    plotscat(x, y, xlab, ylab, clab)

    plt.subplots_adjust(wspace=.8, hspace=1)
    
    
    ##%% Save the figure    
    if savefigs:
        d = os.path.join(svmdir+dnow) #,mousename)       
    #            daysnew = (np.array(days))[dayinds]
        if chAl==1:
            dd = 'chAl_' + corrn + 'onsetChoice_beh_scatter_inhExc'+labAll+'_'  + daysGood[0][0:6] + '-to-' + daysGood[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_' + corrn + 'onsetChoice_beh_scatter_inhExc'+labAll+'_'  + daysGood[0][0:6] + '-to-' + daysGood[-1][0:6] + '_' + nowStr
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)            
        fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])    
    
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)           
   


    
    #%% 
    #############################################################################################
    ############ Line plots of class accuracy (each day, and average of days) ###################
    #############################################################################################
    
    if do_excInhHalf:
        Cols = 'm','b'
    else:
        Cols = 'r','k'
        
               
    #%% Plot class accur trace for all days (only use days with svm trained trials above thTrained) on top of each other
    
    plt.figure()
    
    for iday in range(len(days2an)):    # perhaps only plot days2an_heatmap (not all days!)
    
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
    elif chAl==0:
        plt.xlabel('Time since stim onset (ms)', fontsize=13)
    elif chAl==-1:
        plt.xlabel('Time since outcome onset (ms)', fontsize=13)        
        
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
            dd = 'chAl_' + corrn + 'allDays_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_' + corrn + 'allDays_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            
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
        iday = -1
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
            elif chAl==0:
                plt.xlabel('Time relative to stim onset (ms)', fontsize=11)
            elif chAl==-1:
                plt.xlabel('Time relative to outcome onset (ms)', fontsize=11)
                
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
                    dd = 'chAl_' + corrn + 'day' + days[iday][0:6] + n0 + '_' + nowStr
                else:
                    dd = 'stAl_' + corrn + 'day' + days[iday][0:6] + n0 + '_' + nowStr
            
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
        
        
    
        
    #%% Plot the average of aligned traces across all days (exc, inh, allExc superimposed)
#    superimpose=0
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
    elif chAl==0:
        plt.xlabel('Time relative to stim onset (ms)', fontsize=11)
    elif chAl==-1:
        plt.xlabel('Time relative to outcome onset (ms)', fontsize=11)
        
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
    
    
    ###### mark frames that are different between data and shfl
    _,p_inh = stats.ttest_ind(av_test_data_inh_aligned[:,days2an_noExtraStim], av_test_shfl_inh_aligned[:,days2an_noExtraStim], axis=-1, nan_policy='omit')
    _,p_exc = stats.ttest_ind(av_test_data_exc_aligned[:,days2an_noExtraStim], av_test_shfl_exc_aligned[:,days2an_noExtraStim], axis=-1, nan_policy='omit')
    _,p_allExc = stats.ttest_ind(av_test_data_allExc_aligned[:,days2an_noExtraStim], av_test_shfl_allExc_aligned[:,days2an_noExtraStim], axis=-1, nan_policy='omit')
    
    pp_inh = np.full(len(p_inh), np.nan)
    pp_inh[p_inh<=.05] = 1
    
    pp_exc = np.full(len(p_inh), np.nan)
    pp_exc[p_exc<=.05] = 1
    
    pp_allExc = np.full(len(p_inh), np.nan)
    pp_allExc[p_allExc<=.05] = 1
    
    yl = plt.gca().get_ylim()
#    yy = np.full(len(p_inh), yl[1])
    # exc
    if do_excInhHalf==0:
        plt.plot(time_aligned, pp_exc*yl[1], 'b.', markersize=3)
    # inh
    plt.plot(time_aligned, pp_inh*yl[1]+1, color=Cols[0], marker='.', linestyle='', markersize=3)
    # allExc
    plt.plot(time_aligned, pp_allExc*yl[1]+2, color=Cols[1], marker='.', linestyle='', markersize=3)
    ax = plt.gca()
    makeNicePlots(ax,1,1)
    plt.ylim([yl[0], yl[1]+3])
    
        
    
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
        
        
    
    if superimpose:
        ##%% data - shfl    
        plt.subplot(222)
        # exc
        if do_excInhHalf==0:
            av = av_av_test_data_exc_aligned - av_av_test_shfl_exc_aligned
            plt.fill_between(time_aligned, av - sd_av_test_data_exc_aligned, av + sd_av_test_data_exc_aligned, alpha=0.5, edgecolor='b', facecolor='b')
            plt.plot(time_aligned, av, 'b', label='exc')
        # inh
        av = av_av_test_data_inh_aligned - av_av_test_shfl_inh_aligned
        plt.fill_between(time_aligned, av - sd_av_test_data_inh_aligned, av + sd_av_test_data_inh_aligned, alpha=0.5, edgecolor=Cols[0], facecolor=Cols[0])
        plt.plot(time_aligned, av, color=Cols[0], label=labInh)
        # allExc
        av = av_av_test_data_allExc_aligned - av_av_test_shfl_allExc_aligned
        plt.fill_between(time_aligned, av - sd_av_test_data_allExc_aligned, av + sd_av_test_data_allExc_aligned, alpha=0.5, edgecolor=Cols[1], facecolor=Cols[1])
        plt.plot(time_aligned, av, color=Cols[1], label=labAll)
        
#        if chAl==1:
#            plt.xlabel('Time relative to choice onset (ms)', fontsize=11)
#        else:
#            plt.xlabel('Time relative to stim onset (ms)', fontsize=11)
#        plt.ylabel('Classification accuracy (%)', fontsize=11)
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
        ax = plt.gca()
        makeNicePlots(ax,1,1)
        plt.title('data-shfl')
    
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False) 
    #xmin, xmax = ax.get_xlim()
    #plt.xlim([-1400,500])
       
    
    ##%% Save the figure    
    if savefigs:
        if doSd:
            sdn = 'aveSdDays_'
        else:
            sdn = 'aveSeDays_'
        if chAl==1:
            dd = 'chAl_' + corrn + sdn + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_' + corrn + sdn + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
    
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

    
    #%% Easy, hard, medium: Average class accuracy for testing trials of different stimulus strength: easy, hard, medium
    
    # Plot the average of aligned traces across all days (exc, inh, allExc superimposed)
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
        elif chAl==0:
            plt.xlabel('Time relative to stim onset (ms)', fontsize=11)
        elif chAl==-1:
            plt.xlabel('Time relative to outcome onset (ms)', fontsize=11)
        
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
        elif chAl==0:
            plt.xlabel('Time relative to stim onset (ms)', fontsize=11)
        elif chAl==-1:
            plt.xlabel('Time relative to outcome onset (ms)', fontsize=11)
        
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
        elif chAl==0:
            plt.xlabel('Time relative to stim onset (ms)', fontsize=11)
        elif chAl==-1:
            plt.xlabel('Time relative to outcome onset (ms)', fontsize=11)
            
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
                dd = 'chAl_' + corrn + 'aveSeDays_easyHardMedTrs_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            else:
                dd = 'stAl_' + corrn + 'aveSeDays_easyHardMedTrs_' + labAll + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        
                
            d = os.path.join(svmdir+dnow)
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)
                    
            fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
         
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        #    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+'svg')
        #    plt.savefig(fign, dpi=300, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        
        

#%%   
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% Plots of adding neurons 1 by 1
        
##################################################################################################
##################################################################################################
##################################################################################################

if addNs_roc:   

    alphFig = .5
    labs = ['allExc','inh', 'exc']
    colors = ['k','r','b']
    colors_shflTrsEachN = 'darkgray', 'darksalmon', 'deepskyblue'        
    if h2l==1: # high to low ROC
        h2ln = 'hi2loROC_'
    else: # low to high ROC
        h2ln = 'lo2hiROC_'

        
    if 'av_test_data_inh_shflTrsEachN' in locals():
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


    ###%% Average CAs across cv samps, do this separately for each exc samp    
    if compWith_shflTrsEachN:
        av_test_data_exc_shflTrsEachN_excSamp = []
        for iday in range(numD):
            # numShufflesExc x number of neurons in the decoder x nFrs 
            av_test_data_exc_shflTrsEachN_excSamp.append(100 - np.mean(perClassErrorTest_data_exc_all_shflTrsEachN[iday][:,:,:,eventI_ds_allDays[iday]+fr2an], axis=2)) # numShufflesExc x number of neurons in the decoder
    
    av_test_data_exc_excSamp = []
    av_test_shfl_exc_excSamp = []
    for iday in range(numD):
        av_test_data_exc_excSamp.append(100 - np.mean(perClassErrorTest_data_exc_all[iday][:,:,:,eventI_ds_allDays[iday]+fr2an], axis=2)) # numShufflesExc x number of neurons in the decoder
        av_test_shfl_exc_excSamp.append(100 - np.mean(perClassErrorTest_shfl_exc_all[iday][:,:,:,eventI_ds_allDays[iday]+fr2an], axis=2)) # numShufflesExc x number of neurons in the decoder            
            
            
    if compWith_shflTrsEachN:
        # we have to do the following otherwise we can't save them as mat files
        # numDays; each day: # nNeurons x nSamps        
        perClassErrorTest_data_allExc_all_shflTrsEachNn = np.array([perClassErrorTest_data_allExc_all_shflTrsEachN[iday][:,:,eventI_ds_allDays[iday]+fr2an] for iday in range(numD)])        
        # numDays; each day: # nNeurons x nSamps        
        perClassErrorTest_data_inh_all_shflTrsEachNn = np.array([perClassErrorTest_data_inh_all_shflTrsEachN[iday][:,:,eventI_ds_allDays[iday]+fr2an] for iday in range(numD)])
        # numDays; each day: # nNeurons x nExcSamps x nSamps
        perClassErrorTest_data_exc_all_shflTrsEachNn = np.array([np.transpose(perClassErrorTest_data_exc_all_shflTrsEachN[iday][:,:,:,eventI_ds_allDays[iday]+fr2an], (1,0,2)) for iday in range(numD)])

    perClassErrorTest_data_allExc_alln = np.array([perClassErrorTest_data_allExc_all[iday][:,:,eventI_ds_allDays[iday]+fr2an] for iday in range(numD)])
    perClassErrorTest_data_inh_alln = np.array([perClassErrorTest_data_inh_all[iday][:,:,eventI_ds_allDays[iday]+fr2an] for iday in range(numD)])   
    perClassErrorTest_data_exc_alln = np.array([np.transpose(perClassErrorTest_data_exc_all[iday][:,:,:,eventI_ds_allDays[iday]+fr2an], (1,0,2)) for iday in range(numD)])
    perClassErrorTest_shfl_exc_alln = np.array([np.transpose(perClassErrorTest_shfl_exc_all[iday][:,:,:,eventI_ds_allDays[iday]+fr2an], (1,0,2)) for iday in range(numD)])

    """
    scio.savemat(fname+'/svm_addNsROC_classErr_shflTrsEachN_'+nowStr, {'mousename': mousename, 'numD':numD, 'mn_corr':mn_corr, 'thTrained':thTrained, 'fr2an':fr2an, 'eventI_ds_allDays':eventI_ds_allDays,
      'perClassErrorTest_data_allExc_all_shflTrsEachNn':perClassErrorTest_data_allExc_all_shflTrsEachNn, 'av_test_data_allExc_shflTrsEachN':av_test_data_allExc_shflTrsEachN,
      'perClassErrorTest_data_inh_all_shflTrsEachNn':perClassErrorTest_data_inh_all_shflTrsEachNn, 'av_test_data_inh_shflTrsEachN':av_test_data_inh_shflTrsEachN,
      'perClassErrorTest_data_exc_all_shflTrsEachNn':perClassErrorTest_data_exc_all_shflTrsEachNn, 'av_test_data_exc_shflTrsEachN':av_test_data_exc_shflTrsEachN,
      'perClassErrorTest_data_allExc_alln':perClassErrorTest_data_allExc_alln, 'av_test_data_allExc':av_test_data_allExc, 'av_test_shfl_allExc':av_test_shfl_allExc,
      'perClassErrorTest_data_inh_alln':perClassErrorTest_data_inh_alln, 'av_test_data_inh':av_test_data_inh, 'av_test_shfl_inh':av_test_shfl_inh,
      'perClassErrorTest_data_exc_alln':perClassErrorTest_data_exc_alln, 'av_test_data_exc':av_test_data_exc, 'av_test_shfl_exc':av_test_shfl_exc,
      'perClassErrorTest_shfl_exc_alln':perClassErrorTest_shfl_exc_alln})

    scio.savemat(fname+'/svm_addNsROC_shflClassErr_'+nowStr, {'av_test_shfl_allExc':av_test_shfl_allExc, 'av_test_shfl_inh':av_test_shfl_inh, 'av_test_shfl_exc':av_test_shfl_exc, 
      'perClassErrorTest_shfl_exc_alln':perClassErrorTest_shfl_exc_alln})
    """
            
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
        plt.fill_between(x, av-sd, av+sd, alpha=alphFig, edgecolor=colors[i], facecolor=colors[i])
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
            plt.fill_between(x, av-sd, av+sd, alpha=alphFig, edgecolor=colors_shflTrsEachN[i], facecolor=colors_shflTrsEachN[i])
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
        ax1.fill_between(x, av-sd, av+sd, alpha=alphFig, edgecolor=colors[i], facecolor=colors[i])
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
            ax1.fill_between(x, av-sd, av+sd, alpha=alphFig, edgecolor=colors_shflTrsEachN[i], facecolor=colors_shflTrsEachN[i])
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
        ax1.fill_between(x, av-sd, av+sd, alpha=alphFig, edgecolor=colors[i], facecolor=colors[i])
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
            ax1.fill_between(x, av-sd, av+sd, alpha=alphFig, edgecolor=colors_shflTrsEachN[i], facecolor=colors_shflTrsEachN[i])
            ax1.plot(x, av, colors_shflTrsEachN[i]) #, marker='.')
            # mark neuron numbers that are significantly different after and before shflTrsEachN
            yl = plt.gca().get_ylim()
            pp = p_exc_eachN+0
            pp[pp>.05] = np.nan
            pp[pp<=.05] = max(av+sd)+.5 #yl[1]-np.diff(yl)/20
            plt.plot(x, pp, marker='*',color=colors[i], markeredgecolor=colors[i], linestyle='', markersize=1)            
        
        
        ax1.set_xlim([-5, sum(nDaysPexc>=thD)+4])   #pExc
        ax1.set_xlabel('Number of neurons in the decoder')
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
                dd = 'chAl_' + corrn + nc+'avSeDays_'+nN+'addNsROC_' + h2ln + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            else:
                dd = 'stAl_' + corrn + nc+'avSeDays_'+nN+'addNsROC_' + h2ln + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
                
            d = os.path.join(svmdir+dnow)
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)
                    
            fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
            
    
    
        #%% Plot diff of ave +/- se across days ... to better see when it plateaus
        
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
                    dd = 'chAl_' + corrn + 'diff_avSeDays_'+nN+'addNsROC_' + h2ln + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
                else:
                    dd = 'stAl_' + corrn + 'diff_avSeDays_'+nN+'addNsROC_' + h2ln + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
                    
                d = os.path.join(svmdir+dnow)
                if not os.path.exists(d):
                    print 'creating folder'
                    os.makedirs(d)
                        
                fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
             
                plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
                

            
    #%% Compute the change in CA from the actual case to shflTrsEachNeuron case ... aveaged across those population sizes that are significantly different between actual and shflTrsEachN
    # for each day do ttest across samples for each of the population sizes to see if shflTrsEachN is differnt fromt eh eactual case (ie neuron numbers in the decoder)
    
    if compWith_shflTrsEachN:

        lastPopSize = 0 # if 1, compute CA changes on the last population size (ie when all neurons are included in the population). If 0, average change in CA across all population sizes
        onlySigPops = 1 # if 1, only use population sizes that show sig difference between original and shuffled. If 0, include all population sizes, regardless of significancy

        dav_allExc, dav_inh, dav_exc_av, dav_exc = changeCA_shflTrsEachN(lastPopSize, onlySigPops)
        
        
    #%% Plots of change in CA after breaking noise correaltions
    # Use the following script for the plots of all mice (change in CA after breaking noise correslations):
    # svm_excInh_trainDecoder_eachFrame_shflTrsEachN_sumAllMice_plots.py
    
    if compWith_shflTrsEachN:
        
        perc_thb = [20,80]
#        perc_thb = [15,85] # percentiles of behavioral performance for determining low and high performance.
        
        plt.figure(figsize=(4.4, 4.5)) # plt.figure(figsize=(3,2))        
        gs = gridspec.GridSpec(2,4)#, width_ratios=[2, 1])  #    h1 = gs[0,0:2]
        h1 = gs[0,0:3]
        h2 = gs[1,0:1]
        h3 = gs[1,2:3]
    
        ############ Plot change in CA vs training day ############
        plt.subplot(h1)
        plt.plot(dav_allExc, 'k.-', label='allExc')
        plt.plot(dav_inh, 'r.-', label='inh')
        plt.plot(dav_exc_av, 'b.-', label='exc')
        makeNicePlots(plt.gca())
        plt.ylabel('Change in CA', fontsize=11)        
        plt.xlabel('Training day', fontsize=11)        
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1, frameon=False)         
    #    plt.hist(dav_exc[~np.isnan(dav_exc)])
    #    plt.hist(dav_inh[~np.isnan(dav_inh)])
        
        
        ############%% Plot ave and se of change in CA across days ############
        aa = np.nanmean(dav_allExc)
        ai = np.nanmean(dav_inh)
        ae = np.nanmean(dav_exc_av)
        # se across days
        sa = np.nanstd(dav_allExc) / np.sqrt(sum(~np.isnan(dav_allExc)))
        si = np.nanstd(dav_inh) / np.sqrt(sum(~np.isnan(dav_inh)))
        se = np.nanstd(dav_exc_av) / np.sqrt(sum(~np.isnan(dav_exc_av)))       
        
#        plt.figure(figsize=(1,2))
        plt.subplot(h2)        
        plt.errorbar(0, aa, sa, fmt='o', label='allExc', color='k', markeredgecolor='k')
        plt.errorbar(1, ai, si, fmt='o', label='inh', color='r', markeredgecolor='r')
        plt.errorbar(2, ae, se, fmt='o', label='exc', color='b', markeredgecolor='b')        
#        plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1, frameon=False)         
        plt.ylabel('Change in CA', fontsize=11)        
        plt.xticks(range(3), ['allExc','inh','exc'], rotation=70)
        plt.xlim([-.5, 2+.5])        
        ax = plt.gca()
        makeNicePlots(ax)
        yl = ax.get_ylim()
        plt.ylim([yl[0]-2, yl[1]])
#        plt.ylim([-4, 10])
       

        ################## Compare change in CA (after removing noise corr) for days with low vs high behavioral performance ##################
        a = behCorr_all[days2an_heatmap]    
#        thb = np.percentile(a, [10,90])
        thb = np.percentile(a, perc_thb)
        
        loBehCorrDays = (a <= thb[0])    
        hiBehCorrDays = (a >= thb[1])
        print sum(loBehCorrDays), sum(hiBehCorrDays), ': num days with low and high beh performance'
        
        aa = dav_allExc[days2an_heatmap]
        bb = dav_inh[days2an_heatmap]
        cc = dav_exc_av[days2an_heatmap]
        a = aa[loBehCorrDays]; ah = aa[hiBehCorrDays]
        b = bb[loBehCorrDays]; bh = bb[hiBehCorrDays]
        c = cc[loBehCorrDays]; ch = cc[hiBehCorrDays]    
        
        ahs = np.nanstd(ah)/ np.sqrt(sum(~np.isnan(ah)))        
        chs = np.nanstd(ch)/ np.sqrt(sum(~np.isnan(ch)))
        bhs = np.nanstd(bh)/ np.sqrt(sum(~np.isnan(bh)))
        as0 = np.nanstd(a) / np.sqrt(sum(~np.isnan(a)))
        cs0 = np.nanstd(c) / np.sqrt(sum(~np.isnan(c)))
        bs0 = np.nanstd(b) / np.sqrt(sum(~np.isnan(b)))
        
        ## errorbar: mean and st error: # compare change in CA (after removing noise corr) for days with low vs high behavioral performance        
    #    plt.figure(figsize=(1.5,2.5)        
        plt.subplot(h3)
        plt.errorbar(0, np.nanmean(ah), ahs, marker='o', color='k', fmt='.', markeredgecolor='k', label='high beh')
        plt.errorbar(0, np.nanmean(a), as0, marker='o', color='gray', fmt='.', markeredgecolor='gray', label='low beh')    
        plt.errorbar(1, np.nanmean(ch), chs, marker='o', color='b', fmt='.', markeredgecolor='b')
        plt.errorbar(1, np.nanmean(c), cs0, marker='o', color='lightblue', fmt='.', markeredgecolor='lightblue')       
        plt.errorbar(2, np.nanmean(bh), bhs, marker='o',color='r', fmt='.', markeredgecolor='r')        
        plt.errorbar(2, np.nanmean(b), bs0, marker='o',color='lightsalmon', fmt='.', markeredgecolor='lightsalmon')                
        plt.xlim([-.5, 2+.5]) # plt.xlim([-1,3])    
        plt.xticks([0,1,2], ('allExc', 'exc', 'inh'), rotation=70) # 'vertical'
        plt.ylabel('Change in CA (%)')    
        yl = plt.gca().get_ylim()
        r = np.diff(yl)
        plt.ylim([yl[0], yl[1]+r/10.])
        plt.legend(loc=0, frameon=False, numpoints=1, bbox_to_anchor=(.7, .73))
        makeNicePlots(plt.gca(),0,1)
#        plt.ylim([4, 15])
        
        plt.subplots_adjust(hspace=.5, wspace=.4)
        

        if savefigs:
            if lastPopSize:
                ppn = 'lastPop_'
            else:
                ppn = 'avePop_'
            if onlySigPops:
                spn = 'onlySig_'
            else:
                spn = 'all_'
            
            if chAl==1:
                dd = 'chAl_' + corrn + 'VSshflTrsEachN_avSeDays_CAchange_' + ppn + spn + 'addNsROC_' + h2ln +days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            else:
                dd = 'stAl_' + corrn + 'VSshflTrsEachN_avSeDays_CAchange_' + ppn + spn + 'addNsROC_' + h2ln +days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
                
            d = os.path.join(svmdir+dnow)
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)
                    
            fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])         
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
            
    

    #%%        
    ###########################################################################
    ###########################################################################
    
    #%% Compute number of neurons to reach plateau for each day
    
    if h2l:
        nNsContHiCA = 1 #3 # if there is a gap of >=3 continuous n Ns with high CA (CA > thp percentile of CA across all Ns), call the first number of N as the point of plateuou.
        alph = .05 # only set plateau if CA is sig different from chance
        
        platN_allExc, platN_inh, platN_exc, platN_exc_excSamp = numNeurPlateau()
           
        
        #%% Plot number of neurons to plateau vs. training day
        
        plt.figure(figsize=(3,2))    
        plt.plot(platN_allExc, 'k.-', label='allExc')
        plt.plot(platN_inh, 'r.-', label='inh')
        plt.plot(platN_exc, 'b.-', label='exc')
        makeNicePlots(plt.gca())
        plt.ylabel('# neurons to plateau', fontsize=11)        
        plt.xlabel('Training day', fontsize=11)        
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1, frameon=False)  
    
            
        #%% Plot ave and se (across days) : number of neurons to plateau
    
        aa = np.nanmean(platN_allExc)
        ai = np.nanmean(platN_inh)
        ae = np.nanmean(platN_exc)
        sa = np.nanstd(platN_allExc) / np.sqrt(sum(mn_corr>=thTrained))
        si = np.nanstd(platN_inh) / np.sqrt(sum(mn_corr>=thTrained))
        se = np.nanstd(platN_exc) / np.sqrt(sum(mn_corr>=thTrained))
    
    
        plt.figure(figsize=(1,2))
        
        plt.errorbar(0, aa, sa, fmt='o', label='allExc', color='k', markeredgecolor='k')
        plt.errorbar(1, ai, si, fmt='o', label='inh', color='r', markeredgecolor='r')
        plt.errorbar(2, ae, se, fmt='o', label='exc', color='b', markeredgecolor='b')
        
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1)#, frameon=False) 
        #plt.xlabel('Population', fontsize=11)
        plt.ylabel('# neurons to plateau', fontsize=11)
        plt.xlim([-.2,3-1+.2])    
        plt.xticks(range(3), ['allExc','inh','exc'])
        ax = plt.gca()
        makeNicePlots(ax)
        yl = ax.get_ylim()
        plt.ylim([yl[0]-2, yl[1]])
    #    plt.ylim([8, 80])
       
        if savefigs:
            if chAl==1:
                dd = 'chAl_' + corrn + 'avSeDays_numNeurPlateau_addNsROC_' + h2ln +days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            else:
                dd = 'stAl_' + corrn + 'avSeDays_numNeurPlateau_addNsROC_' + h2ln +days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
                
            d = os.path.join(svmdir+dnow)
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)
                    
            fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])         
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
        
        

    #%%        
    ###########################################################################
    ###########################################################################            
    
    #%% Plots of each day : Plot each day: how choice prediction varies by increasing the population size, compare exc vs inh
   
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
            plt.fill_between(x, av-sd, av+sd, alpha=alphFig, edgecolor=colors[i], facecolor=colors[i])
            plt.plot(x, av, colors[i], label=labs[i])
            if compWith_shflTrsEachN:
                av = av_test_data_allExc_shflTrsEachN[iday]
                sd = sd_test_data_allExc_shflTrsEachN[iday]            
                x = np.arange(1, len(av)+1)
                plt.fill_between(x, av-sd, av+sd, alpha=alphFig, edgecolor=colors_shflTrsEachN[i], facecolor=colors_shflTrsEachN[i])
                plt.plot(x, av, colors_shflTrsEachN[i], label=labs[i])            
#            if ttest2(av_test_data_allExc[iday], av_test_data_allExc_shflTrsEachN[iday]) <= alph:
#                plt.gca().text(50,50, str(np.mean(av_test_data_allExc_shflTrsEachN[iday] - av_test_data_allExc[iday])))
                
            
            
            # inh
            i = 1
            av = av_test_data_inh[iday]
            sd = sd_test_data_inh[iday]            
            x = np.arange(1, len(av)+1)
            plt.fill_between(x, av-sd, av+sd, alpha=alphFig, edgecolor=colors[i], facecolor=colors[i])
            plt.plot(x, av, colors[i], label=labs[i])            
            if compWith_shflTrsEachN:
                av = av_test_data_inh_shflTrsEachN[iday]
                sd = sd_test_data_inh_shflTrsEachN[iday]            
                x = np.arange(1, len(av)+1)
                plt.fill_between(x, av-sd, av+sd, alpha=alphFig, edgecolor=colors_shflTrsEachN[i], facecolor=colors_shflTrsEachN[i])
                plt.plot(x, av, colors_shflTrsEachN[i], label=labs[i])            
#            if ttest2(av_test_data_inh[iday], av_test_data_inh_shflTrsEachN[iday]) <= alph:
#                plt.gca().text(50,50, str(np.mean(av_test_data_inh_shflTrsEachN[iday] - av_test_data_inh[iday])))
#                plt.gca().text(60,50, str(dav_inh[iday]))
                
                
            # exc
            i = 2
            av = av_test_data_exc[iday]
            sd = sd_test_data_exc[iday]            
            x = np.arange(1, len(av)+1)
            plt.fill_between(x, av-sd, av+sd, alpha=alphFig, edgecolor=colors[i], facecolor=colors[i])
            plt.plot(x, av, colors[i], label=labs[i])
            if compWith_shflTrsEachN:
                av = av_test_data_exc_shflTrsEachN[iday]
                sd = sd_test_data_exc_shflTrsEachN[iday]            
                x = np.arange(1, len(av)+1)
                plt.fill_between(x, av-sd, av+sd, alpha=alphFig, edgecolor=colors_shflTrsEachN[i], facecolor=colors_shflTrsEachN[i])
                plt.plot(x, av, colors_shflTrsEachN[i], label=labs[i])                        
#            if ttest2(av_test_data_exc[iday], av_test_data_exc_shflTrsEachN[iday]) <= alph:
#                plt.gca().text(50,50, str(np.mean(av_test_data_exc_shflTrsEachN[iday] - av_test_data_exc[iday])))
#                plt.gca().text(50,50, str(dav_exc[iday]))
            
#            if compWith_shflTrsEachN:
    #            list1 = [np.round(dav_allExc[iday]), np.round(dav_inh[iday]), np.round(dav_exc_av[iday])]
    #            str1 = ' '.join(str(e) for e in list1) 
    #            plt.gca().text(50,50, str1)
            
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
            dd = 'chAl_' + corrn + 'eachDay_addNsROC_' + h2ln +days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_' + corrn + 'eachDay_addNsROC_' + h2ln +days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        
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
    plt.figure(figsize=(1,2))
    
    plt.errorbar(0, aa, sa, fmt='o', label='allExc', color='k')
    plt.errorbar(1, ai, si, fmt='o', label='inh', color='r')
    plt.errorbar(2, ae, se, fmt='o', label='exc', color='b')
    
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1)#, frameon=False) 
    #plt.xlabel('Population', fontsize=11)
    if chAl==1:
        plt.ylabel('Classification accuracy (%)\n [-97 0] ms rel. choice', fontsize=11)
    elif chAl==0:
        plt.ylabel('Classification accuracy (%)\n [-97 0] ms rel. stimulus', fontsize=11)
        
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
            dd = 'chAl_' + corrn + 'avSeDays_time-1_allNs_'+nN+'addNsROC_' + h2ln + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_' + corrn + 'avSeDays_time-1_allNs_'+nN+'addNsROC_' + h2ln + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])         
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    




#%%
"""    
    #%% ########## work with each exc samp (instead of their average) ############

    ########### Make the traces for all days the same length (remember days have different number of neurons in the decoder): add nan for days with num neurons < pInh (eg 20th percentile of nInh (or nExc))... so you can get an average across days (with max number of neurons in the decoder = 20th percentile (instead of min) number of inh neurons across days)                
    av_test_data_exc_samePN_shflTrsEachN_excSamp = [] 
    av_test_data_exc_samePN_excSamp = []
    for iday in range(numD):
        if mn_corr[iday] >= thTrained:
            av_test_data_exc_samePN_shflTrsEachN_excSamp.append(sameMaxNs(av_test_data_exc_shflTrsEachN_excSamp[iday], pInh, nanEnd)) # nExcSamp x nNeurons
            av_test_data_exc_samePN_excSamp.append(sameMaxNs(av_test_data_exc_excSamp[iday], pInh, nanEnd))
    
    av_test_data_exc_samePN_shflTrsEachN_excSamp = np.array(av_test_data_exc_samePN_shflTrsEachN_excSamp) # ndays x nExcSamp x nNeurons
    av_test_data_exc_samePN_excSamp = np.array(av_test_data_exc_samePN_excSamp) # ndays x nExcSamp x nNeurons
    
    
    #%% Compute difference between CA of shflTrsEachN and the actual data, averaged across neuron numbers above a certain number (p) 
    # how breaking noise correlations helps with improving classification accuracy?
      
    q = 1
    
    diff_shflTrsEachN_allExc = np.full((numDaysGood), np.nan)
    diff_shflTrsEachN_inh = np.full((numDaysGood), np.nan)
    diff_shflTrsEachN_exc = np.full((numDaysGood), np.nan)
    
    for iday in range(numDaysGood):
        
        #### allExc
        av = av_test_data_allExc_samePN[iday]        
        p = np.nanpercentile(av, q)
        avs = av_test_data_allExc_samePN_shflTrsEachN[iday]        
        diff_shflTrsEachN_allExc[iday] = np.nanmean(avs[av >= p] - av[av >= p])        

        #### inh
        av = av_test_data_inh_samePN[iday]        
        p = np.nanpercentile(av, q)
        avs = av_test_data_inh_samePN_shflTrsEachN[iday]        
        diff_shflTrsEachN_inh[iday] = np.nanmean(avs[av >= p] - av[av >= p])
        
        #### exc
        av = av_test_data_exc_samePN[iday]        
        p = np.nanpercentile(av, q)
        avs = av_test_data_exc_samePN_shflTrsEachN[iday]        
        diff_shflTrsEachN_exc[iday] = np.nanmean(avs[av >= p] - av[av >= p])

    print ttest2(diff_shflTrsEachN_exc, diff_shflTrsEachN_inh)



    ##%% Set diff of shflTrsEachN and the actual case for each exc samp
    
    diff_shflTrsEachN_exc_excSamp = np.full((numDaysGood, numExcSamples), np.nan)
    for iday in range(numDaysGood):
        for iexc in range(numExcSamples):  #### exc; each exc samp
            av = av_test_data_exc_samePN_excSamp[iday, iexc] 
            p = np.nanpercentile(av, q)
            avs = av_test_data_exc_samePN_shflTrsEachN_excSamp[iday, iexc]        
            diff_shflTrsEachN_exc_excSamp[iday, iexc] = np.nanmean(avs[av >= p] - av[av >= p])

    # ave across exc samps
#    diff_shflTrsEachN_exc_aveSamps = np.nanmean(diff_shflTrsEachN_exc_excSamp, axis=1)
    diff_shflTrsEachN_exc_aveSamps = np.nanmedian(diff_shflTrsEachN_exc_excSamp, axis=1)
     # just pick a random exc samp for each day
#    diff_shflTrsEachN_exc_aveSamps = diff_shflTrsEachN_exc_excSamp[:, rng.permutation(numExcSamples)][:,0]    
    print ttest2(diff_shflTrsEachN_exc_aveSamps, diff_shflTrsEachN_inh)
    
    ##%%
    plt.figure()
    plt.subplot(221)
    plt.plot(diff_shflTrsEachN_allExc, 'k')
    plt.subplot(222)
    plt.plot(diff_shflTrsEachN_inh, 'r')
    plt.subplot(223)
    plt.plot(diff_shflTrsEachN_exc, 'c')
    plt.subplot(224)
    plt.plot(diff_shflTrsEachN_exc_aveSamps, 'b')
    
    
    
    #%%    
    a = diff_shflTrsEachN_exc_excSamp.flatten()
    b = diff_shflTrsEachN_inh.flatten()
    binEvery = 5
    bn = np.arange(np.min(np.concatenate((a,b))), np.max(np.concatenate((a,b))), binEvery)
    bn[-1] = np.max([np.max(a),np.max(b)]) # unlike digitize, histogram doesn't count the right most value
    
    # plt hist of a
    hist, bin_edges = np.histogram(a, bins=bn)
    hist = hist/float(np.sum(hist))    
    plt.plot(bin_edges[0:-1]+binEvery/2., hist, color='b') 
    # plt hist of b
    hist, bin_edges = np.histogram(b, bins=bn)
    hist = hist/float(np.sum(hist))        
    plt.plot(bin_edges[0:-1]+binEvery/2., hist, color='r') 
    m = np.nanmean(diff_shflTrsEachN_exc_excSamp, axis=(0,1))
    s = np.nanstd(diff_shflTrsEachN_exc_excSamp, axis=(0,1))
    plt.plot(m, 0, 'co')
    plt.plot(m+s, 0, 'yo')
    plt.plot(m-s, 0, 'yo')
    plt.plot(diff_shflTrsEachN_inh.flatten().mean(), 0, 'ro')
    
    
    
    #%%               
#    diff_shflTrsEachN_exc_excSamp, diff_shflTrsEachN_inh
    for iday in range(numDaysGood):
        plt.figure()
        plt.hist(diff_shflTrsEachN_exc_excSamp[iday])
        m = diff_shflTrsEachN_exc_excSamp[iday].mean()
        s = diff_shflTrsEachN_exc_excSamp[iday].std()
        plt.plot(m, 0, 'co')
        plt.plot(m+s, 0, 'yo')
        plt.plot(m-s, 0, 'yo')
        plt.plot(diff_shflTrsEachN_inh[iday], 0, 'ro')
    
   
    plt.hist(diff_shflTrsEachN_exc_excSamp.flatten())
    plt.hist(diff_shflTrsEachN_inh.flatten())
    m = np.nanmean(diff_shflTrsEachN_exc_excSamp, axis=(0,1))
    s = np.nanstd(diff_shflTrsEachN_exc_excSamp, axis=(0,1))
    plt.plot(m, 0, 'co')
    plt.plot(m+s, 0, 'yo')
    plt.plot(m-s, 0, 'yo')
    plt.plot(diff_shflTrsEachN_inh.flatten().mean(), 0, 'ro')

    
    #%% Plot ave and se across days

    aa = np.nanmean(diff_shflTrsEachN_allExc)
    ai = np.nanmean(diff_shflTrsEachN_inh)
#    ae = np.nanmean(diff_shflTrsEachN_exc)
    ae = np.nanmean(diff_shflTrsEachN_exc_aveSamps)
#    ae = np.nanmean(diff_shflTrsEachN_exc_excSamp, axis=(0,1))
    
    sa = np.nanstd(diff_shflTrsEachN_allExc) / np.sqrt(numDaysGood)
    si = np.nanstd(diff_shflTrsEachN_inh) / np.sqrt(numDaysGood)
#    se = np.nanstd(diff_shflTrsEachN_exc) / np.sqrt(numDaysGood)
    se = np.nanstd(diff_shflTrsEachN_exc_aveSamps) / np.sqrt(numDaysGood)
#    se = np.nanstd(diff_shflTrsEachN_exc_excSamp, axis=(0,1)) / np.sqrt(numDaysGood*numExcSamples)


    plt.figure(figsize=(1,2))
    
    plt.errorbar(0, aa, sa, fmt='o', label='allExc', color='k', markeredgecolor='k')
    plt.errorbar(1, ai, si, fmt='o', label='inh', color='r', markeredgecolor='r')
    plt.errorbar(2, ae, se, fmt='o', label='exc', color='b', markeredgecolor='b')
    
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1)#, frameon=False) 
    #plt.xlabel('Population', fontsize=11)
    plt.ylabel('Change in CA', fontsize=11)
    plt.xlim([-.2,3-1+.2])    
    plt.xticks(range(3), ['allExc','inh','exc'])
    ax = plt.gca()
    makeNicePlots(ax)
    yl = ax.get_ylim()
    plt.ylim([yl[0]-2, yl[1]])
#    plt.ylim([8, 80])
"""

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
