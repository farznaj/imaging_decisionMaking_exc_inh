# -*- coding: utf-8 -*-
"""
# Set the following variables, then run svm_notebook_setVars() which will call mainSVM_notebook:
# There are more variables in mainSVM_notebook_setVars that you may want to reset.

mousename = 'fni17'
imagingFolder = '151013'
mdfFileNumber = [1,2] 

trialHistAnalysis = 0;
roundi = 1; # For the same dataset we run the code multiple times, each time we select a random subset of neurons (of size n, n=.95*numTrials)
    

# Optional inputs:
iTiFlg = 2; # 0: short ITI, 1: long ITI, 2: all ITIs.
numSamples = 100; # number of iterations for finding the best c (inverse of regularization parameter)
neuronType = 2; # 0: excitatory, 1: inhibitory, 2: all types.    
saveResults = 1; # save results in mat file.

mainSVM_notebook_setVars(mousename, imagingFolder, mdfFileNumber, trialHistAnalysis, roundi)
#mainSVM_notebook_setVars(mousename, imagingFolder, mdfFileNumber, trialHistAnalysis, roundi, iTiFlg=2, numSamples=100, neuronType=2, saveResults=1)

Created on Fri Oct 28 12:48:43 2016
@author: farznaj
"""

#allExc = 0 # use allexc and inh neurons for decoding (but not unsure neurons)                       
#eqExcInh = 1 # use equal number of exc and inh neurons for decoding... numSamps of these populations will be made.
def svm_notebook_setVars6(mousename, imagingFolder, mdfFileNumber, minTrPerChoice=15, earliestGoTone=50, numSamples=100, trialHistAnalysis=0, iTiFlg=2):

    stimAligned = 1 # use stimulus-aligned traces for analyses?
    doSVR = 1 # decode stimulus?
    epAllStim = 1 # when doing svr on stim-aligned traces, if 1, svr will computed on X_svr (ave pop activity during the entire stim presentation); Otherwise it will be computed on X0 (ave pop activity during ep)
    doBagging = 0 # BAGGING will be performed (in addition to l2) for svm/svr    
    
    regType = 'l2' # 'l2' : regularization type
    kfold = 10;

    neuronType=2
    saveResults=1	 
    setNsExcluded=1
    
#    doNsRand = 0; # if 1, a random set of neurons will be selected to make sure we have fewer neurons than trials. 
#    compExcInh = 0 # if 1, analyses will be run to compare exc inh neurons.
#    setNsExcluded = 0; # if 1, NsExcluded will be set even if it is already saved.
    doPlots = 0; # Whether to make plots or not.
    saveHTML = 0; # whether to save the html file of notebook with all figures or not.
    
    if trialHistAnalysis==1: # more parameters are specified in popClassifier_trialHistory.m
#        iTiFlg = 1; # 0: short ITI, 1: long ITI, 2: all ITIs.
        epEnd_rel2stimon_fr = 0 # 3; # -2 # epEnd = eventI + epEnd_rel2stimon_fr
    else:
        # not needed to set ep_ms here, later you define it as [choiceTime-300 choiceTime]ms # we also go 30ms back to make sure we are not right on the choice time!
        if mousename=='fni17':
            ep_ms = [809, 1109] #[425, 725] # [1000, 1300]; #[700, 900]; # [500, 700]; # training epoch relative to stimOnset % we want to decode animal's upcoming choice by traninig SVM for neural average responses during ep ms after stimulus onset.
        # outcome2ana will be used if trialHistAnalysis is 0. When it is 1, by default we are analyzing past correct trials. If you want to change that, set it in the matlab code.
        outcome2ana = 'corr' # '', corr', 'incorr' # trials to use for SVM training (all, correct or incorrect trials)
        strength2ana = 'all' # 'all', easy', 'medium', 'hard' % What stim strength to use for training?
        thStimStrength = 3; # 2; # threshold of stim strength for defining hard, medium and easy trials.
        th_stim_dur = 800; # min stim duration to include a trial in timeStimOnset
    
    trs4project = 'trained' # 'trained', 'all', 'corr', 'incorr' # trials that will be used for projections and the class accuracy trace; if 'trained', same trials that were used for SVM training will be used. "corr" and "incorr" refer to current trial's outcome, so they don't mean much if trialHistAnalysis=1. 
    windowAvgFlg = 1 # if 0, data points during ep wont be averaged when setting X (for SVM training), instead each frame of ep will be treated as a separate datapoint. It helps with increasing number of datapoints, but will make data mor enoisy.
    
    thAct = 5e-4 #5e-4; # 1e-5 # neurons whose average activity during ep is less than thAct will be called non-active and will be excluded.
    thTrsWithSpike = 1; # 3 % remove neurons that are active in <thSpTr trials.
    
    pnev2load = [] #[] [3] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.


    #%% Run mainSVM_notebook
#    execfile("svm_notebook.py")
#    execfile("svm_excInh_cPath.py")
#    execfile("svm_excInh_trainDecoder.py")
#    execfile("svm_diffNumNeurons.py")    
#    execfile("svm_stability.py")
#    execfile("svm_stability_sr_chAl.py")
#    execfile("svm_corr_incorr.py")
#    execfile("svm_excInh_cPath_chAl.py")
    execfile("svr_1choice.py")
