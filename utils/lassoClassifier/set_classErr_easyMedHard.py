#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Called inside svm_excInh_trainDecoder_eachFrame_plots.py

Created on Thu May  3 18:35:06 2018
@author: farznaj
"""

thMinEMH = -1 #1 #3 ... makes sense to make sure each day is contributing to all easy,med,hard, but if I make this above -1, fni16,18,19 will have almost no days!
###########%% Set stimrate for testing trials of each sample         

Data = scio.loadmat(postName, variable_names=['stimrate', 'cb'])
stimrate = np.array(Data.pop('stimrate')).flatten().astype('float')
cb = np.array(Data.pop('cb')).flatten().astype('float')        
'''
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
'''


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

for isamp in range(numSamps): # isamp = 0
    
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
            
