# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 17:23:36 2017

@author: farznaj
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:52:31 2017

@author: farznaj
"""

#%% Change the following vars:

mice = 'fni16', 'fni17', 'fni18', 'fni19'

saveResults = 1

testIncorr = 1 # if 1, use the decoder trained on correct trials to test how it does on incorrect trials, i.e. on predicting labels of incorrect trials (using incorr trial neural traces)
doTestingTrs = 1 # if 1 compute classifier performance only on testing trials; otherwise on all trials
    
normWeights = 0 #1 # if 1, weights will be normalized to unity length. ### NOTE: you figured if you do np.dot(x,w) using normalized w, it will not match the output of svm (perClassError)
zscoreX = 1 # z-score X (just how SVM was performed) before computing FRs
doAllN = 1 # plot allN, instead of allExc

outcome2ana = 'corr' # '', corr', 'incorr' # trials to use for SVM training (all, correct or incorrect trials) # outcome2ana will be used if trialHistAnalysis is 0. When it is 1, by default we are analyzing past correct trials. If you want to change that, set it in the matlab code.        
corrTrained = 1
doIncorr = 0
loadWeights = 1    

#lastTimeBinMissed = 0 #I think it should be 0 bc the exc,inh svm data was run that way. # if 0, things were ran fine; if 1: by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
softNorm = 1 # if 1, no neurons will be excluded, bc we do soft normalization of FRs, so non-active neurons wont be problematic. if softNorm = 0, NsExcluded will be found
thAct = 5e-4 #5e-4; # 1e-5 # neurons whose average activity during ep is less than thAct will be called non-active and will be excluded.
strength2ana = 'all' # 'all', easy', 'medium', 'hard' % What stim strength to use for training?
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

if doAllN==1:
    smallestC = 0 # Identify best c: if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
    if smallestC==1:
        print 'bestc = smallest c whose cv error is less than 1se of min cv error'
    else:
        print 'bestc = c that gives min cv error'
        
from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')

normX = 0
useEqualTrNums = 1
shflTrsEachNeuron = 0  # Set to 0 for normal SVM training. # Shuffle trials in X_svm (for each neuron independently) to break correlations between neurons in each trial.
if testIncorr:
    doTestingTrs = 0
if doTestingTrs:
    loadYtest = 1
else:
    loadYtest = 0
#    doAllN = 0
    
    
#%% Define function to compute class accuracy on all frames
    
def testDecode(y, x, w, b, th=0):
    nFrs = x.shape[0]
#    numSamples = w.shape[0]
#    classErr = np.full((numSamples, nFrs, nFrs), np.nan)  # nSamps x testingFrs x trainedFrs
    classErr = [] # trainedFrs x nSamps x testingFrs
    
    for ifr in range(nFrs): # this is the trainedFr # get the decoder of this frame
        
        ww = w[:,:,ifr] # nSamps x neurons
        bb = b[:,ifr]
        # x: nFrs x units x trials
        
        # test the decoder of frame ifr (trainedFr) on all other time points (testingFrs)
        # projecting neural activity of each frame onto the decoder of frame ifr
        yhat = np.dot(ww, x) + bb[:, np.newaxis, np.newaxis] # nSamps x testingFrs x trials
        # predict trial labels                     
        yhat[yhat<th] = 0
        yhat[yhat>th] = 1  
        
        # Now compute difference between actual and predicted y
        if np.ndim(y) == 1:
            d = yhat - y  # nSamps x testingFrs x trials
        else:  # if y is samps x trials (ie if y is shuffled)
#            test this
            d = np.transpose(np.transpose(yhat, (1,0,2)) - y, (1,0,2))  # nSamps x testingFrs x trials
        
        # average across trials to get classification error 
        classErr.append(np.mean(abs(d), axis=-1) * 100) # nSamps x testingFrs 

#        classErr[:,:,ifr] = np.mean(abs(yhat - Y_svm), axis=-1) * 100 # nSamps x testingFrs x trainedFrs (how well decoder ifr does in predicting choice on each frame of frs)

    return classErr  # trainedFrs x nSamps x testingFrs



#%% Same as above but set class accuracy NOT on all trials, but instead on testing trials (done for each sample separately using w and testTrInds of that sample)
    
def testDecode_testingTrs(y, x, w, b, testTrInds_outOfY0, th=0):
#    x # frames x neurons x trials
#    w # samps x neurons x frames
#    b # samps x frames       
#    testTrInds_outOfY0_inh # samps x numTestingTrs      
    nFrs = x.shape[0]
    nSamps = w.shape[0]
    classErr = np.full((nFrs, nSamps, nFrs), np.nan)  # trainedFrs x nSamps x testingFrs
    
    for isamp in range(nSamps):
        # Get x and y for each sample        
        xx = x[:,:,testTrInds_outOfY0[isamp]] # frames x neurons x testingTrials

        if np.ndim(y) == 1: # normal case (not shuffled)
            yy = y[testTrInds_outOfY0[isamp]]                
        else: # shuffled y: samps x trials
            yy = y[isamp, testTrInds_outOfY0[isamp]] # for each sample a different shuffled order of trial labels is used to compute fractError        

        # Loop over frames: test the decoder of frame ifr (trainedFr) on all other time points (testingFrs)            
        for ifr in range(nFrs): # this is the trainedFr bc we are getting the decoder (w and b) from this frame            
            ww = w[isamp,:,ifr] # nNeurons
            bb = b[isamp,ifr]             
            
            # Project population activity of each frame onto the decoder of frame ifr
            yhat = np.dot(ww, xx) + bb # testingFrs x testing trials
            # predict trial labels                     
            yhat[yhat<th] = 0 # testingFrs x testing trials
            yhat[yhat>th] = 1  
            
            d = yhat - yy  # testing Frs x nTesting Trials # difference between actual and predicted y           
            # average across trials to get classification error 
            classErr[ifr,isamp,:] = np.mean(abs(d), axis=-1) * 100 # testingFrs     

    return classErr  # trainedFrs x nSamps x testingFrs


#%% For each frame, project the activity onto the decoder of that frame, to get predicted activity for each trial: samps x trials
    
def testDecode_sameFr(Y_svm, x, w, b, th=0):
    
    classErr = np.full((x.shape[0], w.shape[0]), np.nan) # frs x samps
    for ifr in range(x.shape[0]): 
        xx = x[ifr]
        ww = w[:,:,ifr] # nSamps x neurons
        bb = b[:,ifr]
                    
        yhat = np.dot(ww,xx) + bb[:,np.newaxis] # samps x trs
        yhat[yhat<th] = 0
        yhat[yhat>th] = 1                       
        # For each sampe compute average across trials: percent trials predicted incorrectly
        classErr[ifr,:] = np.mean(abs(yhat - Y_svm), axis=-1) * 100 # samps
        
    return classErr
  
    
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
        testTrInds_allSamps_inh, Ytest_allSamps_inh, Ytest_hat_allSampsFrs_inh, trsnow_allSamps_inh, testTrInds_allSamps_allExc, Ytest_allSamps_allExc, Ytest_hat_allSampsFrs_allExc, trsnow_allSamps_allExc, testTrInds_allSamps_exc, Ytest_allSamps_exc, Ytest_hat_allSampsFrs_exc, trsnow_allSamps_exc \
        = loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0, loadYtest=loadYtest)


        #%% Set X_svm and Y_svm ... to project onto the decoders loaded above.
        
        ###########################################################################################################################################
        ###########################################################################################################################################
        ###########################################################################################################################################
        ###########################################################################################################################################
        ###########################################################################################################################################
        ###########################################################################################################################################
        ###########################################################################################################################################
        ###########################################################################################################################################

        #%% Load matlab variables: event-aligned traces, inhibitRois, outcomes,  choice, etc
        #     - traces are set in set_aligned_traces.m matlab script.
        
        ################# Load outcomes and choice (allResp_HR_LR) for the current trial
        # if trialHistAnalysis==0:
        Data = scio.loadmat(postName, variable_names=['outcomes', 'allResp_HR_LR'])
        outcomes = (Data.pop('outcomes').astype('float'))[0,:]
        # allResp_HR_LR = (Data.pop('allResp_HR_LR').astype('float'))[0,:]
        allResp_HR_LR = np.array(Data.pop('allResp_HR_LR')).flatten().astype('float')
        choiceVecAll = allResp_HR_LR+0;  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
        # choiceVecAll = np.transpose(allResp_HR_LR);  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
        #print 'Current outcome: %d correct choices; %d incorrect choices' %(sum(outcomes==1), sum(outcomes==0))
        
        
        if trialHistAnalysis:
            # Load trialHistory structure to get choice vector of the previous trial
            Data = scio.loadmat(postName, variable_names=['trialHistory'],squeeze_me=True,struct_as_record=False)
            choiceVec0All = Data['trialHistory'].choiceVec0.astype('float')
        
        
            
        ################## Set trials strength and identify trials with stim strength of interest
        if trialHistAnalysis==0:
            Data = scio.loadmat(postName, variable_names=['stimrate', 'cb'])
            stimrate = np.array(Data.pop('stimrate')).flatten().astype('float')
            cb = np.array(Data.pop('cb')).flatten().astype('float')
        
            s = stimrate-cb; # how far is the stimulus rate from the category boundary?
            if strength2ana == 'easy':
                str2ana = (abs(s) >= (max(abs(s)) - thStimStrength));
            elif strength2ana == 'hard':
                str2ana = (abs(s) <= thStimStrength);
            elif strength2ana == 'medium':
                str2ana = ((abs(s) > thStimStrength) & (abs(s) < (max(abs(s)) - thStimStrength))); 
            else:
                str2ana = np.full((1, np.shape(outcomes)[0]), True, dtype=bool).flatten();
        
            print 'Number of trials with stim strength of interest = %i' %(str2ana.sum())
            print 'Stim rates for training = {}'.format(np.unique(stimrate[str2ana]))
        
            '''
            # Set to nan those trials in outcomes and allRes that are nan in traces_al_stim
            I = (np.argwhere((~np.isnan(traces_al_stim).sum(axis=0)).sum(axis=1)))[0][0] # first non-nan neuron
            allTrs2rmv = np.argwhere(sum(np.isnan(traces_al_stim[:,I,:])))
            print(np.shape(allTrs2rmv))
        
            outcomes[allTrs2rmv] = np.nan
            allResp_HR_LR[allTrs2rmv] = np.nan
            '''   
            
        
        #%% Load inhibitRois
        
        Data = scio.loadmat(moreName, variable_names=['inhibitRois_pix'])
        inhibitRois = Data.pop('inhibitRois_pix')[0,:]    
        print '%d inhibitory, %d excitatory; %d unsure class' %(np.sum(inhibitRois==1), np.sum(inhibitRois==0), np.sum(np.isnan(inhibitRois)))
        
            
        ###########################################################################################################################################
        #%% Set choiceVec0  (Y: the response vector)
            
        if trialHistAnalysis:
            choiceVec0 = choiceVec0All[:,iTiFlg] # choice on the previous trial for short (or long or all) ITIs
            choiceVec0S = choiceVec0All[:,0]
            choiceVec0L = choiceVec0All[:,1]
        else: # set choice for the current trial
            choiceVec0 = allResp_HR_LR;  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
            # choiceVec0 = np.transpose(allResp_HR_LR);  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
            if outcome2ana == 'corr':
                choiceVec0[outcomes!=1] = np.nan; # analyze only correct trials.
            elif outcome2ana == 'incorr':
                choiceVec0[outcomes!=0] = np.nan; # analyze only incorrect trials.   
            
            choiceVec0[~str2ana] = np.nan   
            # Y = choiceVec0
            # print(choiceVec0.shape)
        print '%d correct trials; %d incorrect trials' %((outcomes==1).sum(), (outcomes==0).sum())
        
        
        #%% Set Y for incorrect trials
        
        # set Y_incorr: vector of choices for incorrect trials
        Y_incorr0 = choiceVecAll+0
        Y_incorr0[outcomes!=0] = np.nan; # analyze only incorrect trials.
        print '\tincorrect trials: %d HR; %d LR' %((Y_incorr0==1).sum(), (Y_incorr0==0).sum())
        
        
        #%% Load spikes and time traces to set X for training SVM
        
        if chAl==1:    #%% Use choice-aligned traces 
            # Load 1stSideTry-aligned traces, frames, frame of event of interest
            # use firstSideTryAl_COM to look at changes-of-mind (mouse made a side lick without committing it)
            Data = scio.loadmat(postName, variable_names=['firstSideTryAl'],squeeze_me=True,struct_as_record=False)
            traces_al_1stSide = Data['firstSideTryAl'].traces.astype('float')
            time_aligned_1stSide = Data['firstSideTryAl'].time.astype('float')
            eventI = Data['firstSideTryAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
            #eventI_ch = Data['firstSideTryAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
            # print(np.shape(traces_al_1stSide))
            
            trsExcluded = (np.isnan(np.sum(traces_al_1stSide, axis=(0,1))) + np.isnan(choiceVec0)) != 0
            
            X_svm = traces_al_1stSide[:,:,~trsExcluded]  
            
            time_trace = time_aligned_1stSide            
        
            ## incorrect trials
            trsExcluded_incorr = (np.isnan(np.sum(traces_al_1stSide, axis=(0,1))) + np.isnan(Y_incorr0)) != 0
            X_svm_incorr = traces_al_1stSide[:,:,~trsExcluded_incorr] 
            
        print 'frs x units x trials', X_svm.shape    
        print 'frs x units x trials (incorrect trials)', X_svm_incorr.shape
        ##%%
#        corr_hr = sum(np.logical_and(allResp_HR_LR==1 , ~trsExcluded)).astype(int)
#        corr_lr = sum(np.logical_and(allResp_HR_LR==0 , ~trsExcluded)).astype(int)           
        
        
        #%% Set Y for training SVM   
        
        Y_svm = choiceVec0[~trsExcluded]
        print 'Y size: ', Y_svm.shape

        Y_svm_incorr = Y_incorr0[~trsExcluded_incorr]
        print 'Y_incorr size: ', Y_svm_incorr.shape
        
        
        ## print number of hr,lr trials after excluding trials
        if outcome2ana == 'corr':
            print '\tcorrect trials: %d HR; %d LR' %((Y_svm==1).sum(), (Y_svm==0).sum())
        elif outcome2ana == 'incorr':
            print '\tincorrect trials: %d HR; %d LR' %((Y_svm==1).sum(), (Y_svm==0).sum())
        else:
            print '\tall trials: %d HR; %d LR' %((Y_svm==1).sum(), (Y_svm==0).sum())
            
        print '\tincorrect trials: %d HR; %d LR' %((Y_svm_incorr==1).sum(), (Y_svm_incorr==0).sum())
    
        # Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
        hr_trs = (Y_svm==1)
        lr_trs = (Y_svm==0)            
        print '%d HR trials; %d LR trials' %(sum(hr_trs), sum(lr_trs))

    
        ##%% I think we need at the very least 3 trials of each class to train SVM. So exit the analysis if this condition is not met!
        
    #    if min((Y_svm==1).sum(), (Y_svm==0).sum()) < 3:
    #        sys.exit('Too few trials to do SVM training! HR=%d, LR=%d' %((Y_svm==1).sum(), (Y_svm==0).sum()))
        
           
        #%% Save a copy of X_svm before downsampling 
        
        X_svm_o = X_svm
        time_trace_o = time_trace
        X_svm_incorr_o = X_svm_incorr
        

        #%% Downsample X: average across multiple times (downsampling, not a moving average. we only average every regressBins points.)
        
        lastTimeBinMissed = 0
        X_svm, time_trace, eventI_ds = downsampXsvmTime(X_svm_o, time_trace_o, eventI, regressBins, lastTimeBinMissed)
        X_svm_incorr, _, _ = downsampXsvmTime(X_svm_incorr_o, [], eventI, regressBins, lastTimeBinMissed)
        
        # stupid issue with lastTimeBinMissed, some decoders are computed with lastTimeBinMissed=1 ... so we make sure x and w have the same size
        if X_svm.shape[0] == w_data_allExc.shape[-1]+1:
            lastTimeBinMissed = 1
            X_svm, time_trace, eventI_ds = downsampXsvmTime(X_svm_o, time_trace_o, eventI, regressBins, lastTimeBinMissed)
            X_svm_incorr, _, _ = downsampXsvmTime(X_svm_incorr_o, [], eventI, regressBins, lastTimeBinMissed)
            
            
        #%% Perhaps (not done in svm_excInh_trainDecoder_eachFrame.py) : After downsampling normalize X_svm so each neuron's max is at 1 (you do this in matlab for S traces before downsampling... so it makes sense to again normalize the traces After downsampling so max peak is at 1)                
        
        if normX:
            m = np.max(X_svm,axis=(0,2))   # find the max of each neurons across all trials and frames # max(X_svm.flatten())            
            X_svm = np.transpose(np.transpose(X_svm,(0,2,1)) / m, (0,2,1))
        
        
        #%% Set NsExcluded : Identify neurons that did not fire in any of the trials (during ep) and then exclude them. Otherwise they cause problem for feature normalization.
        # thAct and thTrsWithSpike are parameters that you can play with.
        
        if softNorm==0:    
            if trialHistAnalysis==0:
                # X_svm
                T1, N1, C1 = X_svm.shape
                X_svm_N = np.reshape(X_svm.transpose(0 ,2 ,1), (T1*C1, N1), order = 'F') # (frames x trials) x units
                
                # Define NsExcluded as neurons with low stdX
                stdX_svm = np.std(X_svm_N, axis = 0);
                NsExcluded = stdX_svm < thAct
                # np.sum(stdX < thAct)
                
            print '%d = Final # non-active neurons' %(sum(NsExcluded))
            # a = size(spikeAveEp0,2) - sum(NsExcluded);
            print 'Using %d out of %d neurons; Fraction excluded = %.2f\n' %(np.shape(X_svm)[1]-sum(NsExcluded), np.shape(X_svm)[1], sum(NsExcluded)/float(np.shape(X_svm)[1]))
            
            """
            print '%i, %i, %i: #original inh, excit, unsure' %(np.sum(inhibitRois==1), np.sum(inhibitRois==0), np.sum(np.isnan(inhibitRois)))
            # Check what fraction of inhibitRois are excluded, compare with excitatory neurons.
            if neuronType==2:    
                print '%i, %i, %i: #excluded inh, excit, unsure' %(np.sum(inhibitRois[NsExcluded]==1), np.sum(inhibitRois[NsExcluded]==0), np.sum(np.isnan(inhibitRois[NsExcluded])))
                print '%.2f, %.2f, %.2f: fraction excluded inh, excit, unsure\n' %(np.sum(inhibitRois[NsExcluded]==1)/float(np.sum(inhibitRois==1)), np.sum(inhibitRois[NsExcluded]==0)/float(np.sum(inhibitRois==0)), np.sum(np.isnan(inhibitRois[NsExcluded]))/float(np.sum(np.isnan(inhibitRois))))
            """
            ##%% Exclude non-active neurons from X and set inhRois (ie neurons that don't fire in any of the trials during ep)
            
            X_svm = X_svm[:,~NsExcluded,:]
            print 'choice-aligned traces: ', np.shape(X_svm)
            
        else:
            print 'Using soft normalization; so all neurons will be included!'
            NsExcluded = np.zeros(np.shape(X_svm)[1]).astype('bool')  
        
        
        # Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)
        inhRois = inhibitRois[~NsExcluded]
        print 'Number: inhibit = %d, excit = %d, unsure = %d' %(np.sum(inhRois==1), np.sum(inhRois==0), np.sum(np.isnan(inhRois)))
        #    print 'Fraction: inhibit = %.2f, excit = %.2f, unsure = %.2f' %(fractInh, fractExc, fractUn)
        
            
        #%% Print some varas

        numTrials, numNeurons = X_svm.shape[2], X_svm.shape[1]
        print '%d trials; %d neurons' %(numTrials, numNeurons)
        # print ' The data has %d frames recorded from %d neurons at %d trials' %Xt.shape            
           
            
        #%% Center and normalize X: feature normalization and scaling: to remove effects related to scaling and bias of each neuron, we need to zscore data (i.e., make data mean 0 and variance 1 for each neuron) 
        
        if zscoreX:
            #% Keep a copy of X_svm before normalization
            X_svm00 = X_svm + 0        
            X_svm_incorr00 = X_svm_incorr + 0
            
            ##%%
            # Normalize all frames to the same value (problem: some frames FRs may span a much larger range than other frames, hence svm solution will become slow)
            """
            # X_svm
            T1, N1, C1 = X_svm.shape
            X_svm_N = np.reshape(X_svm.transpose(0 ,2 ,1), (T1*C1, N1), order = 'F') # (frames x trials) x units
                
            meanX_svm = np.mean(X_svm_N, axis = 0);
            stdX_svm = np.std(X_svm_N, axis = 0);
            
            # normalize X
            X_svm_N = (X_svm_N - meanX_svm) / stdX_svm;
            X_svm = np.reshape(X_svm_N, (T1, C1, N1), order = 'F').transpose(0 ,2 ,1)
            """
            
            # Normalize each frame separately (do soft normalization)
            X_svm_N = np.full(np.shape(X_svm), np.nan)
            X_svm_incorr_N = np.full(np.shape(X_svm_incorr00), np.nan)
            meanX_fr = []
            stdX_fr = []
            for ifr in range(np.shape(X_svm)[0]):
                if testIncorr: # use both corr and incorr traces to set mean and std for each neuron
                    xtot = np.concatenate((X_svm00, X_svm_incorr00),axis=2)
                    xf = xtot[ifr,:,:]
                else:
                    xf = X_svm[ifr,:,:]
        
                m = np.mean(xf, axis=1)
                s = np.std(xf, axis=1)   
                meanX_fr.append(m) # frs x neurons
                stdX_fr.append(s)       
                
                if softNorm==1: # soft normalziation : neurons with sd<thAct wont have too large values after normalization
                    s = s+thAct     
            
                ##### do normalization
                X_svm_N[ifr,:,:] = ((X_svm[ifr,:,:].T - m) / s).T
                X_svm_incorr_N[ifr,:,:] = ((X_svm_incorr00[ifr,:,:].T - m) / s).T  
                
            meanX_fr = np.array(meanX_fr) # frames x neurons
            stdX_fr = np.array(stdX_fr) # frames x neurons
            
            
            X_svm = X_svm_N
            X_svm_incorr = X_svm_incorr_N


        #%%
        ###########################################################################################################################################
        ###########################################################################################################################################
        ###########################################################################################################################################
        ###########################################################################################################################################
        ###########################################################################################################################################
        ###########################################################################################################################################
        ###########################################################################################################################################
        ###########################################################################################################################################
        
        #%% Set X_svm for inh and allExc neurons
        
        Xinh = X_svm[:, inhRois==1,:]              
        XallExc = X_svm[:, inhRois==0,:]            
        
        Xinh_incorr = X_svm_incorr[:, inhRois==1,:]              
        XallExc_incorr = X_svm_incorr[:, inhRois==0,:]   
        

        #%% Set Xexc for each excShfl

        Datae = scio.loadmat(svmName_excInh[1], variable_names=['excNsEachSamp'])
        excNsEachSamp = Datae.pop('excNsEachSamp')
        
 
        numShufflesExc = w_data_exc.shape[0]
        lenInh = (inhRois==1).sum()
        excI = np.argwhere(inhRois==0)        
        XexcEq = []
        XexcEq_incorr = []
        #excNsEachSamp = []
        for ii in range(numShufflesExc): 
            en = excNsEachSamp[ii].squeeze() # n randomly selected exc neurons.    
            Xexc = X_svm[:, en,:]
            XexcEq.append(Xexc)    
            Xexc_incorr = X_svm_incorr[:, en,:]   
            XexcEq_incorr.append(Xexc_incorr)    
            
        #    excNsEachSamp.append(en) # indeces of exc neurons (our of X_svm) used for svm training in each exc shfl (below).... you need this if you want to get svm projections for a particular exc shfl (eg w_data_exc[nExcShfl,:,:,:])
        XexcEq = np.array(XexcEq) # nExcShfl x frames x units x trials
        XexcEq_incorr = np.array(XexcEq_incorr)

        
        #%% Normalize weights for each samp
        
        if normWeights: ##%% normalize weights of each sample        
            normw = sci.linalg.norm(w_data_inh, axis=1);   # numSamps x numFrames
            w_data_inh_normed = np.transpose(np.transpose(w_data_inh,(1,0,2))/normw, (1,0,2)) # numSamples x numNeurons x numFrames; normalize weights so weights (of each frame) have length 1
        
            normw = sci.linalg.norm(w_data_exc, axis=2);   # numShufflesExc x numSamps x numFrames
            w_data_exc_normed = np.transpose(np.transpose(w_data_exc,(2,0,1,3))/normw, (1,2,0,3)) # numShufflesExc x numSamples x numNeurons x numFrames; normalize weights so weights (of each frame) have length 1
            
            normw = sci.linalg.norm(w_data_allExc, axis=1);   # numSamps x numFrames
            w_data_allN_normed = np.transpose(np.transpose(w_data_allExc,(1,0,2))/normw, (1,0,2)) # numSamples x numNeurons x numFrames; normalize weights so weights (of each frame) have length 1
        else:
            w_data_inh_normed = w_data_inh
            w_data_exc_normed = w_data_exc
            w_data_allN_normed = w_data_allExc


        ##%% Take average across samples (this is like getting a single decoder across all trial subselects... so I guess bagging)... (not ave of abs... the idea is that ave of shuffles represents the neurons weights for the decoder given the entire dataset (all trial subselects)... if a w switches signs across samples then it means this neuron is not very consistently contributing to the choice... so i think we should average its w and not its abs(w))
        '''
        winhAve = np.mean(w_data_inh_normed, axis=0) # neurons x frames
        wexcAve = np.mean(w_data_exc_normed, axis=0)
        wallNAve = np.mean(w_data_allN_normed, axis=0)
            
        if normWeights: ##%% normalize averaged weights
#        if useAllNdecoder==0:
            normw = sci.linalg.norm(winhAve, axis=0)
            winhAve = winhAve / normw
                    
            normw = sci.linalg.norm(wexcAve, axis=0)
            wexcAve = wexcAve / normw

            normw = sci.linalg.norm(wallNAve, axis=0)
            wallNAve = wallNAve / normw
        '''

        
            
        #%% Now predict trial labels on all time points from decoders trained at at time point, and compute their classification accuracy
            
        # what to do with b... can i just use b of each sample... why to average across samples....?
        
        # project the trace onto decoder (of each sample), then add the decoders b.
        # at the end you get multiple predictions (nSamps)... take an average across them...
        
        nSamps = w_data_allN_normed.shape[0] #len(trsnow_allSamps_inh)                   
        if testIncorr:
            Y_svm = Y_svm_incorr


        ########### All neurons ###########    
        w = w_data_allN_normed # samps x neurons x frames
        b = b_data_allExc # samps x frames   
        
        if testIncorr:
            x = X_svm_incorr            
            classErr_allN = testDecode_sameFr(Y_svm, x, w, b) # frames x samps
            
        else:
            x = X_svm # frames x neurons x trials               
            if doTestingTrs:
                testTrInds_outOfY0 = np.array([trsnow_allSamps_allExc[isamp][testTrInds_allSamps_allExc[isamp]] for isamp in range(nSamps)]) # samps x numTestingTrs          
                classErr_allN = testDecode_testingTrs(Y_svm, x, w, b, testTrInds_outOfY0, th=0)  # trainedFrs x nSamps x testingFrs
            else:            
                classErr_allN = testDecode(Y_svm, x, w, b) # trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)

        
        ########### inh neurons ###########                
        w = w_data_inh_normed
        b = b_data_inh
        if testIncorr:
            x = Xinh_incorr
            classErr_inh = testDecode_sameFr(Y_svm, x, w, b) # frames x samps
            
        else:
            x = Xinh            
            if doTestingTrs:
                testTrInds_outOfY0 = np.array([trsnow_allSamps_inh[isamp][testTrInds_allSamps_inh[isamp]] for isamp in range(nSamps)]) # samps x numTestingTrs          
                classErr_inh = testDecode_testingTrs(Y_svm, x, w, b, testTrInds_outOfY0, th=0)  # trainedFrs x nSamps x testingFrs
            else:                    
                classErr_inh = testDecode(Y_svm, x, w, b) # trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
        

        ########### exc neurons ###########        
        classErr_exc = []
        for iexc in range(numShufflesExc):
            w = w_data_exc_normed[iexc]
            b = b_data_exc[iexc]
            if testIncorr:
                x = XexcEq_incorr[iexc]
                classErr_exc0 = testDecode_sameFr(Y_svm, x, w, b) # frames x samps
                
            else:
                x = XexcEq[iexc]            
                if doTestingTrs:
                    testTrInds_outOfY0 = np.array([trsnow_allSamps_exc[iexc,isamp][testTrInds_allSamps_exc[iexc,isamp]] for isamp in range(nSamps)]) # samps x numTestingTrs          
                    classErr_exc0 = testDecode_testingTrs(Y_svm, x, w, b, testTrInds_outOfY0, th=0)  # trainedFrs x nSamps x testingFrs
#                    classErr_exc.append(classErr_exc0) # nExcSamps x trainedFrs x nSamps x testingFrs
                else:    
                    classErr_exc0 = testDecode(Y_svm, x, w, b)
#                    classErr_exc.append(classErr_exc0) # nExcSamps x trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
            # collect var for each exc samp
            classErr_exc.append(classErr_exc0) # nExcSamps x frs x nSamps
        
        
        #%% ############%% Sanity checks: make sure classErr when testing and training frames are the same is the same as the loaded variable perClassErrors        
        
        # NOTE: with normWeights=0, below becomes mostly correct, but still it fails for some cases ... and I don't know why. But it doesn't change the results
        # of stabTestTimes because we are doing the same for all frames ... also now with the cross validation CAs near the diagonal are not higher than the diagonal.
        # so the main purpose of doing cross validation is achieved!
        
        if doTestingTrs:
            eqy_allN = np.full(X_svm.shape[0], np.nan)
            eqy_inh = np.full(X_svm.shape[0], np.nan)
            eqy_exc = np.full((X_svm.shape[0],numShufflesExc), np.nan)
            for ifr in range(X_svm.shape[0]):                    
                eqy_allN[ifr] = np.mean(np.equal(perClassErrorTest_data_allExc[:,ifr], classErr_allN[ifr,:,ifr]))
                eqy_inh[ifr] = np.mean(np.equal(perClassErrorTest_data_inh[:,ifr], classErr_inh[ifr,:,ifr]))
                for iexc in range(numShufflesExc):
                    eqy_exc[ifr,iexc] = np.mean(np.equal(perClassErrorTest_data_exc[iexc,:,ifr], classErr_exc[iexc][ifr,:,ifr]))                
            
            if ~((np.mean(eqy_allN)==1) + (np.mean(eqy_inh)==1) + (np.mean(eqy_exc)==1)):
                sys.exit('Error: classErr when testing and training frames are the same is NOT the same as the loaded variable perClassErrors!!!')
    
            '''
            classErrAveSamps = np.mean(classErr_allN, axis=1) # average across samples
            # classErrAveSamps[testFr, trainedFr]
            # classErrAveSamps[:, trainedFr]
            
            plt.figure()
            plt.imshow(classErrAveSamps); plt.colorbar()
            plt.axhline(eventI_ds, 0, nFrs)
            plt.axvline(eventI_ds, 0, nFrs)
            '''

        
        #%% Same as above but now set class accuracy for shuffled trial labels    
        
        # Set shuffled Y_svm
#        nSamps = w_data_allN_normed.shape[0]
        
        Y_shfl = np.full((nSamps,len(Y_svm)), np.nan) # samps x trials
        for isamp in range(nSamps):
            Y_shfl[isamp,:] = Y_svm[rng.permutation(len(Y_svm))]  # for each sample a different shuffled order of trial labels is used to compute fractError         
            
            
            
        ########### All neurons ###########    
        w = w_data_allN_normed # samps x neurons x frames
        b = b_data_allExc # samps x frames   
        
        if testIncorr:
            x = X_svm_incorr            
            classErr_allN_shfl = testDecode_sameFr(Y_shfl, x, w, b) # frames x samps
            
        else:
            x = X_svm # frames x neurons x trials               
            if doTestingTrs:
                testTrInds_outOfY0 = np.array([trsnow_allSamps_allExc[isamp][testTrInds_allSamps_allExc[isamp]] for isamp in range(nSamps)]) # samps x numTestingTrs          
                classErr_allN_shfl = testDecode_testingTrs(Y_shfl, x, w, b, testTrInds_outOfY0, th=0)  # trainedFrs x nSamps x testingFrs
            else:            
                classErr_allN_shfl = testDecode(Y_shfl, x, w, b) # trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)

        
        ########### inh neurons ###########                
        w = w_data_inh_normed
        b = b_data_inh
        if testIncorr:
            x = Xinh_incorr
            classErr_inh_shfl = testDecode_sameFr(Y_shfl, x, w, b) # frames x samps
            
        else:
            x = Xinh            
            if doTestingTrs:
                testTrInds_outOfY0 = np.array([trsnow_allSamps_inh[isamp][testTrInds_allSamps_inh[isamp]] for isamp in range(nSamps)]) # samps x numTestingTrs          
                classErr_inh_shfl = testDecode_testingTrs(Y_shfl, x, w, b, testTrInds_outOfY0, th=0)  # trainedFrs x nSamps x testingFrs
            else:                    
                classErr_inh_shfl = testDecode(Y_shfl, x, w, b) # trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
        

        ########### exc neurons ###########        
        classErr_exc_shfl = []
        for iexc in range(numShufflesExc):
            w = w_data_exc_normed[iexc]
            b = b_data_exc[iexc]
            if testIncorr:
                x = XexcEq_incorr[iexc]
                classErr_exc0 = testDecode_sameFr(Y_shfl, x, w, b) # frames x samps
                
            else:
                x = XexcEq[iexc]            
                if doTestingTrs:
                    testTrInds_outOfY0 = np.array([trsnow_allSamps_exc[iexc,isamp][testTrInds_allSamps_exc[iexc,isamp]] for isamp in range(nSamps)]) # samps x numTestingTrs          
                    classErr_exc0 = testDecode_testingTrs(Y_shfl, x, w, b, testTrInds_outOfY0, th=0)  # trainedFrs x nSamps x testingFrs
#                    classErr_exc.append(classErr_exc0) # nExcSamps x trainedFrs x nSamps x testingFrs
                else:    
                    classErr_exc0 = testDecode(Y_shfl, x, w, b)
#                    classErr_exc.append(classErr_exc0) # nExcSamps x trainedFrs x nSamps x testingFrs (how well decoder ifr does in predicting choice on each frame of frs)
            # collect var for each exc samp
            classErr_exc_shfl.append(classErr_exc0) # nExcSamps x frs x nSamps




        ############################################################%%    
        #%% Keep vars of all days
        
        classErr_allN_allDays.append(classErr_allN)
        classErr_inh_allDays.append(classErr_inh)        
        classErr_exc_allDays.append(classErr_exc)  

        classErr_allN_shfl_allDays.append(classErr_allN_shfl)
        classErr_inh_shfl_allDays.append(classErr_inh_shfl)            
        classErr_exc_shfl_allDays.append(classErr_exc_shfl)  
        
        eventI_ds_allDays[iday] = eventI_ds.astype(int)
        eventI_allDays[iday] = eventI #.astype(int)
        lastTimeBinMissed_allDays[iday] = lastTimeBinMissed

                
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
        
        if doTestingTrs:
            nts = 'testingTrs_'
        else:
            nts = ''
        
        if normWeights:
            nw = 'wNormed_'
        else:
            nw = 'wNotNormed_'
        
        if testIncorr:
            sn = 'svm_testDecoderOnIncorrTrs'
        else:
            sn = 'svm_testEachDecoderOnAllTimes'
            
        finame = os.path.join(fname, ('%s_%s%s%s.mat' %(sn, nts, nw, nowStr)))
        
        
        scio.savemat(finame, {'lastTimeBinMissed_allDays':lastTimeBinMissed_allDays,
                              'eventI_ds_allDays':eventI_ds_allDays,
                              'eventI_allDays':eventI_allDays,
                              'classErr_allN_allDays':classErr_allN_allDays,
                              'classErr_inh_allDays':classErr_inh_allDays,
                              'classErr_exc_allDays':classErr_exc_allDays,
                              'classErr_allN_shfl_allDays':classErr_allN_shfl_allDays,
                              'classErr_inh_shfl_allDays':classErr_inh_shfl_allDays,
                              'classErr_exc_shfl_allDays':classErr_exc_shfl_allDays})
    
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







