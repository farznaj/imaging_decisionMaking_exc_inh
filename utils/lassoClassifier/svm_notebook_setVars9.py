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


doInhAllexcEqexc = [0,1,0]
#    1st element: analyze inhibitory neurons (train SVM for numSamples for each value of C)
#    2nd element: analyze all excitatory neurons (train SVM for numSamples for each value of C)   
#    3rd element: analyze excitatory neurons, equal number to inhibitory neurons (train SVM for numSamples for each value of C, repeat this numShufflesExc times (each time subselecting n exc neurons))
    
Created on Fri Oct 28 12:48:43 2016
@author: farznaj
"""


#allExc = 0 # use allexc and inh neurons for decoding (but not unsure neurons)                       
#eqExcInh = 1 # use equal number of exc and inh neurons for decoding... numSamps of these populations will be made.
def svm_notebook_setVars9(mousename, imagingFolder, mdfFileNumber, chAl, doInhAllexcEqexc, numSamples=50, numShufflesExc=50, trialHistAnalysis=0, iTiFlg=2):

    shflTrsEachNeuron = 1  # Set to 0 for normal SVM training. # Shuffle trials in X_svm (for each neuron independently) to break correlations between neurons in each trial.
    outcome2ana = 'corr' # '', corr', 'incorr' # trials to use for SVM training (all, correct or incorrect trials) # outcome2ana will be used if trialHistAnalysis is 0. When it is 1, by default we are analyzing past correct trials. If you want to change that, set it in the matlab code.        
#    chAl = 0 # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
    softNorm = 1 # if 1, no neurons will be excluded, bc we do soft normalization of FRs, so non-active neurons wont be problematic. if softNorm = 0, NsExcluded will be found
    useEqualTrNums = 1 # Make sure both classes have the same number of trials when training the classifier
    winLen = 100 # ms, length of window for downsampling; svm will be trained in non-overlapping windows of size winLen ms.

    regType = 'l2' # 'l2' : regularization type
    kfold = 10;

    saveResults=1
    doPlots = 0; # Whether to make plots or not.  

    if trialHistAnalysis==0:
        # not needed to set ep_ms here, later you define it as [choiceTime-300 choiceTime]ms # we also go 30ms back to make sure we are not right on the choice time!
#        ep_ms = [809, 1109] #[425, 725] # optional, it will be set according to min choice time if not provided.# training epoch relative to stimOnset % we want to decode animal's upcoming choice by traninig SVM for neural average responses during ep ms after stimulus onset. [1000, 1300]; #[700, 900]; # [500, 700]; 
        strength2ana = 'all' # 'all', easy', 'medium', 'hard' % What stim strength to use for training?
        thStimStrength = 3; # 2; # threshold of stim strength for defining hard, medium and easy trials.
        th_stim_dur = 800; # min stim duration to include a trial in timeStimOnset

    thAct = 5e-4 #5e-4; # 1e-5 # neurons whose average activity during ep is less than thAct will be called non-active and will be excluded.
#    thTrsWithSpike = 1; # 3 % remove neurons that are active in <thSpTr trials.
    pnev2load = [] #[] [3] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.
    neuronType = 2    	 
    setNsExcluded = 1
    saveHTML = 0; # whether to save the html file of notebook with all figures or not.


    #%% Run mainSVM_notebook
#    execfile("svm_notebook.py")
#    execfile("svm_excInh_cPath.py")
#    execfile("svm_excInh_trainDecoder.py")
#    execfile("svm_diffNumNeurons.py")    
#    execfile("svm_stability.py")
#    execfile("svm_stability_sr_chAl.py")
#    execfile("svm_corr_incorr.py")
#    execfile("svm_excInh_cPath_chAl.py")
#    execfile("svr_1choice.py")
#    execfile("svm_decodeStimCateg.py")
#    execfile("svm_eachFrame.py")
    execfile("svm_excInh_trainDecoder_eachFrame.py")
    
    
    