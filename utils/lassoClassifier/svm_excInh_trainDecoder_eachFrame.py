# -*- coding: utf-8 -*-
# NOTE: add for saving meanX, stdX and _chAl mean and std.

"""
# Trains decoder on each frame (time window) using only inh or only exc neurons after finding bestc for each decoder separately.


Train SVM on each frame (time windows of 100ms)... Also find best c for each frame.
Decode choice using:
 choice-aligned traces
 all trials (corr, incorr)
 soft normalization (including all neurons)

Created on Fri Feb 10 20:20:08 2017
@author: farznaj

"""

'''
############ below is the gist of the codes for svm training (done in linearSVM.py) ############
import sklearn.svm as svm                        

linear_svm = svm.LinearSVC(C = cbest_allExc[12], loss='squared_hinge', penalty='l2', dual=True) # get c for a frame, eg frame 12.
linear_svm.fit(X_svm[12,:,:].transpose(), np.squeeze(Y_svm)) # x should be in trials x units

linear_svm.predict(XTest)
linear_svm.coef_
linear_svm.intercept_

###
# def perClassError(Y, Yhat):
#    import numpy as np
#    perClassEr = np.sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
#    return perClassEr

# perClassErrorTest = perClassError(YTest, linear_svm.predict(XTest)); 

# summary.model = linear_svm;
'''


##%%
####### NOTE ########
#print 'Decide which one to load: inhibitRois_pix or inhibitRois!!'

import sys
import os
import numpy as np
from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')


#%% For Jupiter: Add the option to toggle on/off the raw code. Copied from http://stackoverflow.com/questions/27934885/how-to-hide-code-from-cells-in-ipython-notebook-visualized-with-nbviewer

if 'ipykernel' in sys.modules:
    from IPython.display import HTML

    HTML('''<script>
    code_show=true; 
    function code_toggle() {
     if (code_show){
     $('div.input').hide();
     } else {
     $('div.input').show();
     }
     code_show = !code_show
    } 
    $( document ).ready(code_toggle);
    </script>
    <form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')


#%% Set vars if not calling this script from svm_notebook_setVars9.py

# Only run the following section if you are running the code in jupyter or in spyder, but not on the cluster!
if ('ipykernel' in sys.modules) or any('SPYDER' in name for name in os.environ):
    
    # Set these variables:
    mousename = 'fni16' #'fni19' #'fni16' #''fni19' #'fni16' #fni16' #
    imagingFolder = '151001' #'151005' #'150918' #'150818' #'151102' #'151019' #'151006' #'151023' #'151023' #'151001' 
    mdfFileNumber = [1] #[1] #

    do_Ctrace = 0 # if 1, use C trace (denoised fluorescent trace), instead of S (inferred spikes)
    same_HRLR_acrossDays = 0 # normally 0, but if 1, svm will be trained on n trials, where n is the minimum number of trials across all sessions of a mouse (this variable is already computed and saved in the analysis folder of each mouse) # set to 1 to balance the number of trials across training days to make sure svm performance is not affected by that.    
    cbestKnown = 0 #1 # if cbest is already saved, set this to 1, to load it instead of running svm on multiple c values to find the optimum one.
    shflTrsEachNeuron = 0  # Set to 0 for normal SVM training. # Shuffle trials in X_svm (for each neuron independently) to break correlations in neurons FRs across trials.
    shflTrLabs = 0 # svm is already run on the actual data, so now load bestc, and run it on trial-label shuffles.    

    outcome2ana = 'corr' #'corr' # 'all' # '', 'corr', 'incorr' # trials to use for SVM training (all, correct or incorrect trials) # outcome2ana will be used if trialHistAnalysis is 0. When it is 1, by default we are analyzing past correct trials. If you want to change that, set it in the matlab code.            
    trialHistAnalysis = 0
    chAl = 1 # If 1, use choice-aligned traces; if 0 use stim-aligned traces for trainign SVM; if -1 use reward and puunishment (outcome) aligned traces!         
    doInhAllexcEqexc = [1,0,0,-3] # [1,0,0,0] # [1,0,0,-2] # [0,0,1,2]  # 
    #    1st element: analyze inhibitory neurons (train SVM for numSamples for each value of C)
    #    2nd element: if 1: analyze all excitatory neurons (train SVM for numSamples for each value of C)   
                    # if 2: analyze all neurons (exc, inh, unsure) ... this is like code svm_eachFrame.py   
    #    3rd element: if 1: analyze excitatory neurons, equal number to inhibitory neurons (train SVM for numSamples for each value of C, repeat this numShufflesExc times (each time subselecting n exc neurons))
                    # if 2: take half exc, half inh, and run svm
                    # if 3: take lenInh*2 of only exc and run svm. 
    # if there is a 4th element (eg [0,1,0,1]), the following analysis will be done; otherwise we will decode choice. 
        # If 4th element is: 
        # -3, train svm to decode current choice, using half corr and half incorr trials for each choice (this is the right way of decoding choice so the effects of choice and stimulus are separated.)
        # -2, train svm to decode trial outcome (corr vs incorr)
        # -1, test decoder on incorrect trials (in addition to correct trials.)     
        # 0, train svm to decode stimulus category (HR vs LR) on equal number of corr and incorr trials.
        # 1: add neurons based on their ROC choice tuning : sort from highest to lowest choice tuning. ##(if eqExc, exc ns are first subsampled then they are sorted based on ROC; and then svm is run) 
        # 2: add neurons based on their ROC choice tuning : sort from lowest to highest choice tuning. ##(if eqExc, exc ns are first subsampled then they are sorted based on ROC; and then svm is run)
        # 3: add neurons randomly
    
    
    doPlots = 0 # Whether to make plots or not.
    saveResults = 0 # save results in mat file.

    numSamples = 3 #50#100; # number of iterations for finding the best c (inverse of regularization parameter)
    numShufflesExc = 2 #50# we choose 10 sets of n random excitatory neurons.    
    nRandCorrSel = 2

    useEqualTrNums = 1 # Make sure both classes have the same number of trials when training the classifier    
    softNorm = 1 # if 1, no neurons will be excluded, bc we do soft normalization of FRs, so non-active neurons wont be problematic. if softNorm = 0, NsExcluded will be found    
    winLen = 100 # ms, length of window for downsampling; svm will be trained in non-overlapping windows of size winLen ms.


    regType = 'l2' # 'l2' : regularization type
    kfold = 10    
    iTiFlg = 2 # Only needed if trialHistAnalysis=1 and outcome2ana = 'corr'; short ITI, 1: long ITI, 2: all ITIs.

#    if trialHistAnalysis==1: # more parameters are specified in popClassifier_trialHistory.m
    #        iTiFlg = 1; # 0: short ITI, 1: long ITI, 2: all ITIs.
#        epEnd_rel2stimon_fr = 0 # 3; # -2 # epEnd = eventI + epEnd_rel2stimon_fr
    if trialHistAnalysis==0:
        # not needed to set ep_ms here, later you define it as [choiceTime-300 choiceTime]ms # we also go 30ms back to make sure we are not right on the choice time!
#        ep_ms = [809, 1109] #[425, 725] # optional, it will be set according to min choice time if not provided.# training epoch relative to stimOnset % we want to decode animal's upcoming choice by traninig SVM for neural average responses during ep ms after stimulus onset. [1000, 1300]; #[700, 900]; # [500, 700]; 
        strength2ana = 'all' # 'all', easy', 'medium', 'hard' % What stim strength to use for training?
        thStimStrength = 3; # 2; # threshold of stim strength for defining hard, medium and easy trials.
        th_stim_dur = 800; # min stim duration to include a trial in timeStimOnset

    thAct = 5e-4 #will be used for soft normalization #1e-5 # neurons whose average activity during ep is less than thAct will be called non-active and will be excluded.
#    thTrsWithSpike = 1; # 3 % remove neurons that are active in <thSpTr trials.
    pnev2load = [] #[] [3] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.
    saveHTML = 0; # whether to save the html file of notebook with all figures or not.
    setNsExcluded = 1; # if 1, NsExcluded will be set even if it is already saved.    
#    neuronType = 2; # 0: excitatory, 1: inhibitory, 2: all types.        
    
#    minTrPerChoice = 15 # we need at least 15 trials (for a particular choice) to train SVM on...
#    earliestGoTone = 50 # 32 (relative to end of stim (after a single repetition)) # In X matrix ideally we don't want trials with go tone within stim (bc we dont want neural activity to be influenced by go tone); however
    # we relax the conditions a bit and allow trials in which go tone happened within the last 50ms of stimulus. (if go tone happened earlier than 50ms, trial will be excluded.)
#    trs4project = 'trained' # 'trained', 'all', 'corr', 'incorr' # trials that will be used for projections and the class accuracy trace; if 'trained', same trials that were used for SVM training will be used. "corr" and "incorr" refer to current trial's outcome, so they don't mean much if trialHistAnalysis=1. 
#    windowAvgFlg = 1 # if 0, data points during ep wont be averaged when setting X (for SVM training), instead each frame of ep will be treated as a separate datapoint. It helps with increasing number of datapoints, but will make data mor enoisy.
#    stimAligned = 1 # use stimulus-aligned traces for analyses?
#    doSVM = 1 # use SVM to decode stim category.
#    epAllStim = 1 # when doing svr on stim-aligned traces, if 1, svr will computed on X_svr (ave pop activity during the entire stim presentation); Otherwise it will be computed on X0 (ave pop activity during ep)
    # for BAGGING:
#    nRandTrSamp = 10 #nRandTrSamp = 10 #1000 # number of times to subselect trials (for the BAGGING method)
#    doData = 1


#%% Print vars

if doInhAllexcEqexc[0]==1:
    ntName = 'inh'
elif doInhAllexcEqexc[1]==1:
    ntName = 'allExc'
elif doInhAllexcEqexc[1]==2:
    ntName = 'allN'
elif doInhAllexcEqexc[2]==1:
    ntName = 'eqExc'     
elif doInhAllexcEqexc[2]==2:
    ntName = 'halfExc_halfInh'     
elif doInhAllexcEqexc[2]==3:
    ntName = 'allExc_2inhSize'     
    
if trialHistAnalysis==1:    
    if iTiFlg==0:
        itiName = 'short'
    elif iTiFlg==1:
        itiName = 'long'
    elif iTiFlg==2:
        itiName = 'all'        
    
print 'Analyzing %s' %(mousename+'_'+imagingFolder+'_'+str(mdfFileNumber)) 

if do_Ctrace:
    print 'Using C (denoised calcium) traces instead of inferred spikes'
    
if shflTrLabs:
    cbestKnown = 1 # if shflTrLabs, cbest is already saved, so we make sure it is 1.
    print 'Shuffling trial labels'

if shflTrsEachNeuron:
    print 'Breaking correlations by shuffling trials'    

print 'Analyzing %s neurons' %(ntName)

if len(doInhAllexcEqexc)==4:
    
    if doInhAllexcEqexc[3]==-1:
        testIncorr = 1
        decodeStimCateg = 0
        addNs_rand = 0
        addNs_roc = 0
        decodeOutcome = 0
        decodeChoiceEqCorrIncorr = 0
#        cbestKnown = 0
        print 'Testing decoder on incorrect trials as well'    
        
    elif doInhAllexcEqexc[3]==-2:
        decodeOutcome = 1
        decodeStimCateg = 0
        addNs_rand = 0
        addNs_roc = 0
        testIncorr = 0
        outcome2ana = 'all'
        cbestKnown = 0
        decodeChoiceEqCorrIncorr = 0
        print 'Decoding outcome (corr, incorr)'        

    elif doInhAllexcEqexc[3]==-3:
        decodeChoiceEqCorrIncorr = 1
        decodeOutcome = 0
        decodeStimCateg = 0
        addNs_rand = 0
        addNs_roc = 0
        testIncorr = 0
        outcome2ana = 'all'
        cbestKnown = 0
        print 'Decoding choice (using equal number of corr and incorr trials)'        
        
    elif doInhAllexcEqexc[3]==0:
        decodeStimCateg = 1        
        addNs_rand = 0
        addNs_roc = 0
        testIncorr = 0
        outcome2ana = 'all'
        cbestKnown = 0
        decodeOutcome = 0
        decodeChoiceEqCorrIncorr = 0
        print 'Decoding stimulus category (HR, LR)'
        
    elif doInhAllexcEqexc[3]==3:
        addNs_rand = 1        
        addNs_roc = 0
        decodeStimCateg = 0
        cbestKnown = 0 # set cbest, unless you are repeating the same analysis again for adding Ns 1 by 1 randomly...
        testIncorr = 0
        decodeOutcome = 0
        decodeChoiceEqCorrIncorr = 0
        print 'Adding neurons randomly 1 by 1 for SVM analysis'
        
    else: 
        addNs_roc = 1 # if 1 do the following analysis: add neurons 1 by 1 to the decoder based on their tuning strength to see how the decoder performance increases.        
        addNs_rand = 0
        decodeStimCateg = 0
        cbestKnown = 0 # set cbest, unless you are repeating the same analysis again for adding Ns 1 by 1 ROC...         
        testIncorr = 0
        decodeOutcome = 0
        decodeChoiceEqCorrIncorr = 0
        if doInhAllexcEqexc[3]==1:
            print 'Adding neurons 1 by 1 from high to low ROC (choice tuning) for SVM analysis' 
        elif doInhAllexcEqexc[3]==2:
            print 'Adding neurons 1 by 1 from low to high ROC (choice tuning) for SVM analysis' 
else:
    testIncorr = 0
    addNs_rand = 0
    addNs_roc = 0
    decodeStimCateg = 0
    decodeOutcome = 0
    decodeChoiceEqCorrIncorr = 0
    print 'Decoding left, right choice'

if trialHistAnalysis==1:
    testIncorr = 0

if chAl==1:
    print 'Using choice-aligned traces'
elif chAl==0:
    print 'Using stimulus-aligned traces'    
elif chAl==-1:
    print 'Using outcome-aligned traces'    

    
if trialHistAnalysis==0:
    print 'Training %s outcome trials of strength %s' %(outcome2ana, strength2ana)
else:
    print 'Analyzing previous trials of %s outcome' %(outcome2ana)
#print 'trialHistAnalysis = %i' %(trialHistAnalysis)

if trialHistAnalysis==1:
    print 'Analyzing %s ITIs' %(itiName)
#elif 'ep_ms' in locals():
#    print 'training window: [%d %d] ms' %(ep_ms[0], ep_ms[1])

#print 'windowAvgFlg = %i' %(windowAvgFlg)
print 'numSamples = %i' %(numSamples)

if cbestKnown:
    print 'Optimum c is already saved'
else:
    print 'Setting optimum c'
    
if same_HRLR_acrossDays:
    print 'Using min number of HR,LR trials across all days to train the SVM'


#%% Set vars

if chAl==1:
    al = 'chAl'
elif chAl==0:
    al = 'stAl'
elif chAl==-1:
    al = 'outAl'

if outcome2ana == 'corr': # save incorr vars too bc SVM was trained on corr, and tested on icorr.
    o2a = 'corr_'
else:
    o2a = ''
        
if doInhAllexcEqexc[0] == 1:
    ntype = 'inh'
elif doInhAllexcEqexc[1] == 1:
    ntype = 'allExc'
elif doInhAllexcEqexc[1] == 2:
    ntype = 'allN'    
elif doInhAllexcEqexc[2] == 1:
    ntype = 'eqExc'   
elif doInhAllexcEqexc[2] == 2:
    ntype = 'excInhHalf'   
elif doInhAllexcEqexc[2] == 3:
    ntype = 'allExc2inhSize'       

if shflTrsEachNeuron:
	shflname = 'shflTrsPerN_'
else:
	shflname = ''

if shflTrLabs:
    shflTrLabs_n = '_shflTrLabs'
else:
    shflTrLabs_n = ''   
   
eps = sys.float_info.epsilon #2.2204e-16
#nRandTrSamp = numSamples #10 #1000 # number of times to subselect trials (for the BAGGING method)
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(winLen/frameLength)) # 100ms # set to nan if you don't want to downsample.


    
#%% Import libraries

import scipy.io as scio
import scipy as sci
import scipy.stats as stats
#import numpy as np
import numpy.random as rng
import sys
from crossValidateModel import crossValidateModel
from linearSVM import linearSVM
from linearSVR import linearSVR
#from compiler.ast import flatten
import matplotlib 
from matplotlib import pyplot as plt
if 'ipykernel' in sys.modules and doPlots:
    get_ipython().magic(u'matplotlib inline')
    get_ipython().magic(u"config InlineBackend.figure_format = 'svg'")
matplotlib.rcParams['figure.figsize'] = (6,4) #(8,5)
#from IPython.display import display
import sklearn.svm as svm
import os
import glob

# print sys.path
sys.path.append('/home/farznaj/Documents/trial_history/imaging') # Gamal's dir needs to be added using "if" that takes the value of pwd
# print sys.path


#%% Define setImagingAnalysisNamesP


"""
Created on Wed Aug 24 15:59:12 2016

@author: farznaj

This is Farzaneh's first Python code :-) She is very happy and pleased about it :D

example call:

mousename = 'fni17'
imagingFolder = '151021'
mdfFileNumber = [1] #(1,2)

# optional inputs:
postNProvided = 1; # Default:0; If your directory does not contain pnevFile and instead it contains postFile, set this to 1 to get pnevFileName
signalCh = [2] # since gcamp is channel 2, should be 2.
pnev2load = [] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.

from setImagingAnalysisNamesP import *

imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load, postNProvided=postNProvided)

imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber)

"""

##%%
def setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, **options):

    if options.get('signalCh'):
        signalCh = options.get('signalCh');    
    else:
        signalCh = []
        
    if options.get('pnev2load'):
        pnev2load = options.get('pnev2load');    
    else:
        pnev2load = []
        
    if options.get('postNProvided'):
        postNProvided = options.get('postNProvided');    
    else:
        postNProvided = 0
        
    ##%%
    import numpy as np
    import platform
    import glob
    import os.path
        
    if len(pnev2load)==0:
        pnev2load = [0];
            
    ##%%
    dataPath = []
    if platform.system()=='Linux':
        if os.getcwd().find('grid')!=-1: # server # sonas
            dataPath = '/sonas-hs/churchland/nlsas/data/data/'
            altDataPath = '/sonas-hs/churchland/hpc/home/space_managed_data/'
        else: # office linux
#            dataPath = '/home/farznaj/Shares/Churchland/data/'
            dataPath = '/home/farznaj/Shares/Churchland_nlsas_data/data/'
            altDataPath = '/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/' # the new space-managed server (wos, to which data is migrated from grid)
    elif platform.system()=='Darwin':
        dataPath = '/Volumes/My Stu_win/ChurchlandLab'
    else:
        dataPath = '/Users/gamalamin/git_local_repository/Farzaneh/data/'

    ##%%
        
    if any([mousename in s for s in ['fni18','fni19']]): # fni18, fni19        
        tifFold = os.path.join(altDataPath+mousename, 'imaging', imagingFolder)
        dataPath = altDataPath
    else:
        tifFold = os.path.join(dataPath+mousename,'imaging',imagingFolder)
    print tifFold
#    if not os.path.exists(tifFold):
#        if 'altDataPath' in locals():            
#            tifFold = os.path.join(altDataPath+mousename, 'imaging', imagingFolder)
#            dataPath = altDataPath
#        else:
#            sys.exit('Data directory does not exist!')


    r = '%03d-'*len(mdfFileNumber)
    r = r[:-1]
    rr = r % (tuple(mdfFileNumber))
    
    date_major = imagingFolder+'_'+rr
    imfilename = os.path.join(tifFold,date_major+'.mat')
    
    ##%%
    if len(signalCh)>0:
        if postNProvided:
            pnevFileName = 'post_'+date_major+'_ch'+str(int(np.array(signalCh)))+'-Pnev*'
        else:
            pnevFileName = date_major+'_ch'+str(int(np.array(signalCh)))+'-Pnev*'            
        print pnevFileName
        # set the directory
        pnevFileName = glob.glob(os.path.join(tifFold,pnevFileName))   
        print pnevFileName
        # sort pnevFileNames by date (descending)
        pnevFileName = sorted(pnevFileName, key=os.path.getmtime)
#        print pnevFileName
        pnevFileName = pnevFileName[::-1]
#        print pnevFileName
        '''
        array = []
        for idx in range(0, len(pnevFileName)):
            array.append(os.path.getmtime(pnevFileName[idx]))

        inds = np.argsort(array)
        inds = inds[::-1]
        pnev2load = inds[pnev2load]
        '''    
        if len(pnevFileName)==0:
            c = ("No Pnev file was found"); print("%s\n" % c)
            pnevFileName = ''
        else:
            pnevFileName = pnevFileName[pnev2load[0]]
            if postNProvided:
                p = os.path.basename(pnevFileName)[5:]
                pnevFileName = os.path.join(tifFold,p)
    else:
        pnevFileName = ''
    
    ##%%
    return imfilename, pnevFileName, dataPath
    
    
##%%
#imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh, pnev2load)


#%% Define perClassError: percent difference between Y and Yhat, ie classification error

def perClassError(Y, Yhat):
    import numpy as np
    perClassEr = np.sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
    return perClassEr


#%% Deine prediction error for SVR .... 

def perClassError_sr(y,yhat):
    import numpy as np
#    err = (np.linalg.norm(yhat - y)**2)/len(y)
    err = np.linalg.norm(yhat - y)**2
    maxerr = np.linalg.norm(y+1e-10)**2 # Gamal:  try adding a small number just to make the expression stable in case y is zero    
#    maxerr = np.linalg.norm(y)**2
#    ce = err
    ce = err/ maxerr    
#    ce = np.linalg.norm(yhat - y)**2 / len(y)
    return ce

#since there is no class in svr, we need to change this mean to square error.
#eps0 = 10**-5
#def perClassError_sr(Y,Yhat,eps0=10**-5):    
#    ce = np.mean(np.logical_and(abs(Y-Yhat) > eps0 , ~np.isnan(Yhat - Y)))*100
#    return ce

    
#%% Function to predict class labels
# Lets check how predict works.... Result: for both SVM and SVR, classfier.predict equals xw+b. For SVM, xw+b gives -1 and 1, that should be changed to 0 and 1 to match svm.predict.

def predictMan(X,w,b,th=0): # set th to nan if you are predicting on svr (since for svr we dont compare with a threshold)
    yhat = np.dot(X, w.T) + b # it gives -1 and 1... change -1 to 0... so it matches svm.predict
    if np.isnan(th)==0:
        yhat[yhat<th] = 0
        yhat[yhat>th] = 1    
    return yhat


#%% Set mat-file names

pnev2load = [] #[] [3] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.
signalCh = [2] # since gcamp is channel 2, should be always 2.
postNProvided = 1; # If your directory does not contain pnevFile and instead it contains postFile, set this to 1 to get pnevFileName

# from setImagingAnalysisNamesP import *

imfilename, pnevFileName, dataPath = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load, postNProvided=postNProvided)

postName = os.path.join(os.path.dirname(pnevFileName), 'post_'+os.path.basename(pnevFileName))
moreName = os.path.join(os.path.dirname(pnevFileName), 'more_'+os.path.basename(pnevFileName))

print(imfilename)
print(pnevFileName)
print(postName)
print(moreName)




#%% Define functions to set svm file names that are already saved   (svm is already run, so now load bestc, and run it on trial-label shuffles (eg when shflTrLabs=1))

if np.logical_and(cbestKnown, doInhAllexcEqexc[1]!=2): # svm is already run on the actual data, so now load bestc, and run it on trial-label shuffles.
    
    corrTrained = 1
    ##################%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName    
    def setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc=[], regressBins=3, useEqualTrNums=1, corrTrained=0, shflTrsEachNeuron=0):
        import glob
        import os
        
        if chAl==1:
            al = 'chAl'
        elif chAl==0:
            al = 'stAl'
        elif chAl==-1:
            al = 'outAl'
        
        if corrTrained:
            o2a = 'corr_'
        else:
            o2a = ''
    
        if shflTrsEachNeuron:
            shflname = 'shflTrsPerN_'
        else:
            shflname = ''
                
        ''' 
        if len(doInhAllexcEqexc)==0: # 1st run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc at the same time, also for all days (except a few days of fni18), inhRois was used (not the new inhRois_pix)       
            if trialHistAnalysis:
                if useEqualTrNums:
                    svmn = 'excInh_SVMtrained_eachFrame_prevChoice_%s_ds%d_eqTrs_*' %(al,regressBins)
                else:
                    svmn = 'excInh_SVMtrained_eachFrame_prevChoice_%s_ds%d_*' %(al,regressBins)
            else:
                if useEqualTrNums:
                    svmn = 'excInh_SVMtrained_eachFrame_currChoice_%s_ds%d_eqTrs_*' %(al,regressBins)
                else:
                    svmn = 'excInh_SVMtrained_eachFrame_currChoice_%s_ds%d_*' %(al,regressBins)
            
        else: # 2nd run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc separately; also for all days the new vector inhRois_pix was used (not the old inhRois)       
        '''
        if doInhAllexcEqexc[0] == 1:
            ntype = 'inh'
        elif doInhAllexcEqexc[1] == 1:
            ntype = 'allExc'
        elif doInhAllexcEqexc[2] == 1:
            ntype = 'eqExc'     
        elif doInhAllexcEqexc[2]==2:
            ntype = 'excInhHalf'     
        elif doInhAllexcEqexc[2]==3:
            ntype = 'allExc2inhSize'               
            
        if trialHistAnalysis:
            if useEqualTrNums:
                svmn = 'excInh_SVMtrained_eachFrame_%s%s%s_prevChoice_%s_ds%d_eqTrs_*' %(o2a, shflname, ntype, al,regressBins)
            else:
                svmn = 'excInh_SVMtrained_eachFrame_%s%s%s_prevChoice_%s_ds%d_*' %(o2a, shflname, ntype, al,regressBins)
        else:
            if useEqualTrNums:
                svmn = 'excInh_SVMtrained_eachFrame_%s%s%s_currChoice_%s_ds%d_eqTrs_*' %(o2a, shflname, ntype, al,regressBins)
            else:
                svmn = 'excInh_SVMtrained_eachFrame_%s%s%s_currChoice_%s_ds%d_*' %(o2a, shflname, ntype, al,regressBins)
            
            
            
        svmn = svmn + os.path.basename(pnevFileName) #pnevFileName[-32:]    
        svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
        svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    
        return svmName
    

#%% SVM is already run, set its file name and load bestc

if np.logical_and(cbestKnown, doInhAllexcEqexc[1]!=2): # svm is already run on the actual data, so now load bestc, and run it on trial-label shuffles.

    svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc, regressBins, useEqualTrNums, corrTrained, shflTrsEachNeuron)[0]
    print 'loading cbest and excNsEachSamp from file ', svmName
    
    if doInhAllexcEqexc[0]==1:
        data = scio.loadmat(svmName, variable_names='cbest_inh')
        cbest_inh = data.pop('cbest_inh').flatten()
        nfrs_cbest = cbest_inh.shape[0]
        
    if doInhAllexcEqexc[1]==1:
        data = scio.loadmat(svmName, variable_names='cbest_allExc')
        cbest_allExc = data.pop('cbest_allExc').flatten()        
        nfrs_cbest = cbest_allExc.shape[0]
        
    elif doInhAllexcEqexc[2] == 1:        
        data = scio.loadmat(svmName, variable_names=['cbest_exc', 'excNsEachSamp']) #, 'excNsEachSamp_indsOutOfExcNs'])
        cbest_exc = data.pop('cbest_exc') # excSamps x nFrs
        excNsEachSamp = data.pop('excNsEachSamp') # excSamps x nExcNs
#        excNsEachSamp_indsOutOfExcNs = data.pop('excNsEachSamp_indsOutOfExcNs')
        nfrs_cbest = cbest_exc.shape[1]

else:
    cbest_inh = np.nan
    cbest_allExc = np.nan
    cbest_now = np.nan
    
    


#%% Same as the function above but set svm file name that is for allN (set in code svm_eachFrame.py)
    
if np.logical_and(cbestKnown, doInhAllexcEqexc[1]==2): # svm is already run on the actual data, so now load bestc, and run it on trial-label shuffles.    
    # allN
    corrTrained = 1
    ###%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName
    
    def setSVMname_allN_eachFrame(pnevFileName, trialHistAnalysis, chAl, regressBins=3, corrTrained=0, shflTrsEachNeuron=0):
        import glob
        import os
        
        if chAl==1:
            al = 'chAl'
        elif chAl==0:
            al = 'stAl'
        elif chAl==-1:
            al = 'outAl'
    
        if corrTrained==1: 
            o2a = '_corr'
        else:
            o2a = '' 
    
        if shflTrsEachNeuron:
        	shflname = '_shflTrsPerN'
        else:
        	shflname = ''        
        
        if trialHistAnalysis:
            svmn = 'svmPrevChoice_eachFrame_%s%s%s_ds%d_*' %(al,o2a,shflname,regressBins)
        else:
            svmn = 'svmCurrChoice_eachFrame_%s%s%s_ds%d_*' %(al,o2a,shflname,regressBins)
        
        svmn = svmn + os.path.basename(pnevFileName) #pnevFileName[-32:]    
        svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
        svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
        
    #    if len(svmName)>0:
    #        svmName = svmName[0] # get the latest file
    #    else:
    #        svmName = ''
    #    
        return svmName
        


    #####################################################################################%%    
  
    ####%% function to find best c from testing class error ran on values of cvect
    
    def findBestC(perClassErrorTest, cvect, regType, smallestC=1):
        
        import numpy as np

        numSamples = perClassErrorTest.shape[0] 
        numFrs = perClassErrorTest.shape[2]
        
        ###%% Compute average of class errors across numSamples
        
        cbestFrs = np.full((numFrs), np.nan)
           
        for ifr in range(numFrs):
            
            meanPerClassErrorTest = np.mean(perClassErrorTest[:,:,ifr], axis = 0);
            semPerClassErrorTest = np.std(perClassErrorTest[:,:,ifr], axis = 0)/np.sqrt(numSamples);
            
           
            ###%% Identify best c       
            
            # Use all range of c... it may end up a value at which all weights are 0.
            ix = np.argmin(meanPerClassErrorTest)
            if smallestC==1:
                cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
                cbest = cbest[0]; # best regularization term based on minError+SE criteria
                cbestAll = cbest
            else:
                cbestAll = cvect[ix]
            print 'best c = %.10f' %cbestAll
            
            #### Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)
            if regType == 'l1': # in l2, we don't really have 0 weights!
                sys.exit('Needs work! below wAllC has to be for 1 frame') 
                
                a = abs(wAllC)>eps # non-zero weights
                b = np.mean(a, axis=(0,2,3)) # Fraction of non-zero weights (averaged across shuffles)
                c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle
                cvectnow = cvect[c1stnon0:]
                
                meanPerClassErrorTestnow = np.mean(perClassErrorTest[:,c1stnon0:,ifr], axis = 0);
                semPerClassErrorTestnow = np.std(perClassErrorTest[:,c1stnon0:,ifr], axis = 0)/np.sqrt(numSamples);
                ix = np.argmin(meanPerClassErrorTestnow)
                if smallestC==1:
                    cbest = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
                    cbest = cbest[0]; # best regularization term based on minError+SE criteria    
                else:
                    cbest = cvectnow[ix]
        
                print 'best c (at least 1 non-0 weight) = ', cbest            
            else:
                cbest = cbestAll            
            
            
            cbestFrs[ifr] = cbest
    #        indBestC = np.in1d(cvect, cbest)      
        return cbestFrs
    
   
         
#%% allN : SVM is already run, set its file name and load bestc

if np.logical_and(cbestKnown, doInhAllexcEqexc[1]==2): # svm is already run on the actual data, so now load bestc, and run it on trial-label shuffles.    
            
    svmName = setSVMname_allN_eachFrame(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, shflTrsEachNeuron)[0]
    print 'loading cbest_allExc from file ', svmName
    
    Data = scio.loadmat(svmName, variable_names=['perClassErrorTest', 'cvect'])  
    perClassErrorTest = Data.pop('perClassErrorTest') # numSamples x len(cvect) x nFrames
    cvect = Data.pop('cvect').squeeze()
    cbest_allExc = findBestC(perClassErrorTest, cvect, regType, smallestC=0) # nFrames     	
    nfrs_cbest = cbest_allExc.shape[0]

else:
    cbest_allExc = np.nan
       
 
 
###########################################################################################################################################
###########################################################################################################################################
#%% Load ROC vars if doing the following analysis: adding neurons 1 by 1 to the decoder based on their tuning strength, and seeing how the decoder performance increases.

if addNs_roc==1:
    
    ############ Set dir for loading ROC vars
    dirn = os.path.join(dataPath+mousename,'imaging','analysis')
    
    # outcome analyzed for the ROC:
    o2aROC = '_allOutcome'; #'_corr'; # '_incorr';
    thStimSt = 0; namz = '';
    namv = 'ROC_curr_%s%s_stimstr%d%s_%s_*.mat' %(al, o2aROC, thStimSt, namz, mousename)
    a = glob.glob(os.path.join(dirn,namv)) #[0] 
    dirROC = sorted(a, key=os.path.getmtime)[::-1][0] # sort so the latest file is the 1st one. # use the latest saved file
    namtf = os.path.basename(dirROC)
    print namtf
    
    ############ Load ROC vars
    Data = scio.loadmat(dirROC, variable_names=['choicePref_all_alld_exc', 'choicePref_all_alld_inh'])  
    choicePref_all_alld_exc = Data.pop('choicePref_all_alld_exc').flatten() # nDays; each day: frames x neurons
    choicePref_all_alld_inh = Data.pop('choicePref_all_alld_inh').flatten()
    
    # Compute abs deviation of AUC from chance (.5); we want to compute |AUC-.5|, since choice pref = 2*(auc-.5), therefore |auc-.5| = 1/2 * |choice pref|
    # % now ipsi is positive bc we do minus, in the original vars (load above) contra was positive
    choicePref_exc = .5*abs(-choicePref_all_alld_exc)
    choicePref_inh = .5*abs(-choicePref_all_alld_inh)
    #plt.plot(np.sort(choicePref_exc[28][eventI_ds-1].flatten()))


    ######### Find the index in choicePref vars that belongs to day imagingFolder
    # Set days for each mouse
    if mousename=='fni16':
        # chAl & corrTrained
        days = ['150817_1', '150818_1', '150819_1', '150820_1', '150821_1-2', '150825_1-2-3', '150826_1', '150827_1', '150828_1-2', '150831_1-2', '150901_1', '150903_1', '150904_1', '150915_1', '150916_1-2', '150917_1', '150918_1-2-3-4', '150921_1', '150922_1', '150923_1', '150924_1', '150925_1-2-3', '150928_1-2', '150929_1-2', '150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2'] #'150914_1-2' : don't analyze!   
    elif mousename=='fni17':
        # chAl & corrTrained
        days = ['150814_1', '150817_1', '150824_1', '150826_1', '150827_1', '150828_1', '150831_1', '150901_1', '150902_1-2', '150903_1', '150908_1', '150909_1', '150910_1', '150914_1', '150915_1-2', '150916_1', '150917_1-2', '150918_1', '150921_1-2-3', '150922_1-2', '150923_1-2-3', '150924_1-2', '150925_1-2', '150928_1-2', '150930_1-2-3-4', '151001_1', '151002_1-2', '151005_1-2', '151006_1', '151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2']
    elif mousename=='fni18':
        # allDays
        days = ['151209_1', '151210_1', '151211_1', '151214_1-2', '151215_1-2', '151216_1', '151217_1-2']
    elif mousename=='fni19':
        # allDays & chAl
        days = ['150903_1', '150904_1', '150914_1', '150915_1', '150916_1', '150917_1', '150918_1', '150921_1', '150922_1', '150923_1', '150924_1-2', '150925_1-2', '150928_4', '150929_3', '150930_1', '151001_1', '151002_1', '151005_1-2', '151006_1', '151007_1', '151008_1-2', '151009_1-3', '151012_1-2-3', '151013_1', '151015_2', '151016_1', '151019_1', '151020_1', '151022_1-2', '151023_1', '151026_1-2-3', '151027_1', '151028_1-2', '151029_1-2-3', '151101_1']
    
    # find the index in choicePref vars that belongs to day imagingFolder
    dayInd = [x[0:6] for x in days].index(imagingFolder)
    
    choicePref_exc = choicePref_exc[dayInd] # frames x neurons
    choicePref_inh = choicePref_inh[dayInd] # frames x neurons    
    
    
###########################################################################################################################################
#%% Load matlab variables: event-aligned traces, inhibitRois, outcomes,  choice, etc
#     - traces are set in set_aligned_traces.m matlab script.

################# Load outcomes and choice (allResp_HR_LR) for the current trial
# if trialHistAnalysis==0:
Data = scio.loadmat(postName, variable_names=['outcomes', 'allResp_HR_LR'])
outcomes = (Data.pop('outcomes').astype('float'))[0,:]
# outcomes: 1: success, 0: failure, -1: early decision, -2: no decision, -3: wrong initiation, -4: no center commit, -5: no side commit
# allResp_HR_LR = (Data.pop('allResp_HR_LR').astype('float'))[0,:]
allResp_HR_LR = np.array(Data.pop('allResp_HR_LR')).flatten().astype('float')
choiceVecAll = allResp_HR_LR+0;  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
# choiceVecAll = np.transpose(allResp_HR_LR);  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
#print 'Current outcome: %d correct choices; %d incorrect choices' %(sum(outcomes==1), sum(outcomes==0))


if trialHistAnalysis:
    if outcome2ana == 'corr': # Load trialHistory structure to get choice vector of the previous trial
        Data = scio.loadmat(postName, variable_names=['trialHistory'],squeeze_me=True,struct_as_record=False)
        choiceVec0All = Data['trialHistory'].choiceVec0.astype('float')
    elif outcome2ana == 'all':
        choiceVec0All = np.concatenate(([np.nan], allResp_HR_LR)) # shift allResp_HR_LR by one, to get the choice on the previous trial.
        choiceVec0All = np.delete(choiceVec0All, -1) # remove the last element so the length of it matches allResp_HR_LR


Data = scio.loadmat(postName, variable_names=['stimrate', 'cb'])
stimrate = np.array(Data.pop('stimrate')).flatten().astype('float')
cb = np.array(Data.pop('cb')).flatten().astype('float')
    
################## Set trials strength and identify trials with stim strength of interest
if trialHistAnalysis==0:
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


#%% Load go tone, stim, choice times and look at their distribution (all trials, some might be excluded from SVM)

# Load time of some trial events    
Data = scio.loadmat(postName, variable_names=['timeCommitCL_CR_Gotone', 'timeStimOnset', 'timeStimOffset', 'time1stSideTry'])
timeCommitCL_CR_Gotone = np.array(Data.pop('timeCommitCL_CR_Gotone')).flatten().astype('float')
timeStimOnset = np.array(Data.pop('timeStimOnset')).flatten().astype('float')
timeStimOffset = np.array(Data.pop('timeStimOffset')).flatten().astype('float')
time1stSideTry = np.array(Data.pop('time1stSideTry')).flatten().astype('float')


# we need to know the time of stimOffset for a single repetition of the stim (without extra stim, etc)
# timeStimOffset includes extra stim, etc... so we set timeStimeOffset0 (which is stimOffset after 1 repetition) here
Data = scio.loadmat(postName, variable_names=['timeSingleStimOffset'])
if len(Data) > 3:
    timeStimOffset0 = np.array(Data.pop('timeSingleStimOffset')).flatten().astype('float')
else:
    import datetime
    import glob
    bFold = os.path.join(dataPath+mousename,'behavior')
    # from imagingFolder get month name (as Oct for example).
    y = 2000+int(imagingFolder[0:2])
    mname = ((datetime.datetime(y,int(imagingFolder[2:4]), int(imagingFolder[4:]))).strftime("%B"))[0:3]
    bFileName = mousename+'_'+imagingFolder[4:]+'-'+mname+'-'+str(y)+'_*.mat'
    
    bFileName = glob.glob(os.path.join(bFold,bFileName))   
    # sort pnevFileNames by date (descending)
    bFileName = sorted(bFileName, key=os.path.getmtime)
    bFileName = bFileName[::-1]
     
    stimDur_diff_all = []
    for ii in range(len(bFileName)):
        bnow = bFileName[0]       
        Data = scio.loadmat(bnow, variable_names=['all_data'],squeeze_me=True,struct_as_record=False)
        stimDur_diff_all.append(np.array([Data['all_data'][i].stimDur_diff for i in range(len(Data['all_data']))])) # remember difference indexing in matlab and python!
    sdur = np.unique(stimDur_diff_all) * 1000 # ms
    print sdur
    if len(sdur)>1:
        sys.exit('Aborting ... trials have different stimDur_diff values')
    if sdur==0: # waitDur was used to generate stimulus
        waitDuration_all = []
        for ii in range(len(bFileName)):
            sys.exit('Aborting ... needs work... u added timeSingleStimOffset to setEventTimesRelBcontrolScopeTTL.m... if not saved for a day, below needs work for days with multiple sessions!')
    #        bnow = bFileName[0]       
    #        Data = scio.loadmat(bnow, variable_names=['all_data'],squeeze_me=True,struct_as_record=False)
    #        waitDuration_all.append(np.array([Data['all_data'][i].waitDuration for i in range(len(Data['all_data']))])) # remember difference indexing in matlab and python!
    #    sdur = timeStimOnset + waitDuration_all
    timeStimOffset0 = timeStimOnset + sdur



# Print some values
minStimDur = np.floor(np.nanmin(timeStimOffset-timeStimOnset))
maxStimDur = np.floor(np.nanmax(timeStimOffset-timeStimOnset))
minChoiceTime = np.floor(np.nanmin(time1stSideTry-timeStimOnset))
minGoToneTime = np.floor(np.nanmin(timeCommitCL_CR_Gotone-timeStimOnset))
# you can use maxStimDur too but make sure maxStimDur < minChoiceTime, so in no trial you average points during choice
print 'minStimOffset= ', minStimDur
print 'maxStimOffset= ', maxStimDur
print 'minChoiceTime= ', minChoiceTime
print 'minGoToneTime= ', minGoToneTime
#ep_svr = np.arange(eventI+1, eventI + np.round(minStimDur/frameLength) + 1).astype(int) # frames on stimAl.traces that will be used for trainning SVM.   


if doPlots:
    plt.figure
    plt.subplot(2,1,1)
    plt.plot(timeCommitCL_CR_Gotone - timeStimOnset, 'b', label = 'goTone')
    plt.plot(timeStimOffset - timeStimOnset, 'r', label = 'stimOffset')
    plt.plot(timeStimOffset0 - timeStimOnset, 'k', label = 'stimOffset_1rep')
    plt.plot(time1stSideTry - timeStimOnset, 'm', label = '1stSideTry')
    if trialHistAnalysis==0:
        plt.plot([1, np.shape(timeCommitCL_CR_Gotone)[0]],[th_stim_dur, th_stim_dur], 'g:', label = 'th_stim_dur')
#    plt.plot([1, np.shape(timeCommitCL_CR_Gotone)[0]],[ep_ms[-1], ep_ms[-1]], 'k:', label = 'epoch end')
    plt.xlabel('Trial')
    plt.ylabel('Time relative to stim onset (ms)')
    plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 
    # minStimDurNoGoTone = np.nanmin(timeCommitCL_CR_Gotone - timeStimOnset); # this is the duration after stim onset during which no go tone occurred for any of the trials.
    # print 'minStimDurNoGoTone = %.2f ms' %minStimDurNoGoTone
    axes = plt.gca()
    y1,y2 = axes.get_ylim()
    plt.ylim([y1-100, y2])
    

    plt.subplot(2,1,2)
    a = timeCommitCL_CR_Gotone - timeStimOnset
    plt.hist(a[~np.isnan(a)], color='b')
    a = timeStimOffset - timeStimOnset
    plt.hist(a[~np.isnan(a)], color='r')
    a = timeStimOffset0 - timeStimOnset
    plt.hist(a[~np.isnan(a)], color='k')
    a = time1stSideTry - timeStimOnset
    plt.hist(a[~np.isnan(a)], color='m')
    plt.xlabel('Time relative to stim onset (ms)')
    plt.ylabel('# trials')
    axes = plt.gca()
    x1,x2 = axes.get_xlim()
    plt.xlim([x1-100, y2])


#%% Load inhibitRois and set traces for specific neuron types: inhibitory, excitatory or all neurons

'''
print 'Decide which one to load: inhibitRois_pix or inhibitRois!!'
Data = scio.loadmat(moreName, variable_names=['inhibitRois'])
if len(Data) == 3: # inhibitRois does not exist, only inhibitRois_pix exists!
    print 'inhibitRois does not exist. Loading inhibitRois_pix!'
    Data = scio.loadmat(moreName, variable_names=['inhibitRois_pix'])
    inhibitRois = Data.pop('inhibitRois_pix')[0,:]
else:
    print 'Loading inhibitRois (not inhibitRois_pix)!!!'
    inhibitRois = Data.pop('inhibitRois')[0,:]
'''

Data = scio.loadmat(moreName, variable_names=['inhibitRois_pix'])
inhibitRois = Data.pop('inhibitRois_pix')[0,:]    
print '%d inhibitory, %d excitatory; %d unsure class' %(np.sum(inhibitRois==1), np.sum(inhibitRois==0), np.sum(np.isnan(inhibitRois)))

'''
# Set traces for specific neuron types: inhibitory, excitatory or all neurons
if neuronType!=2:    
    nt = (inhibitRois==neuronType) # 0: excitatory, 1: inhibitory, 2: all types.
    # good_excit = inhibitRois==0;
    # good_inhibit = inhibitRois==1;        
    
    traces_al_stim = traces_al_stim[:, nt, :];
    traces_al_stimAll = traces_al_stimAll[:, nt, :];
else:
    nt = np.arange(np.shape(traces_al_stim)[1])    
'''


    
###########################################################################################################################################
#%% Set choiceVec0  (Y: the response vector)
    
if trialHistAnalysis:
    if outcome2ana=='corr':
        choiceVec0 = choiceVec0All[:,iTiFlg] # choice on the previous trial for short (or long or all) ITIs
        choiceVec0S = choiceVec0All[:,0]
        choiceVec0L = choiceVec0All[:,1]
    elif outcome2ana=='all':
        choiceVec0 = choiceVec0All
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


#%% Set Y for incorrect trials (current trials)

# set Y_incorr: vector of choices for incorrect trials
Y_incorr0 = choiceVecAll+0
Y_incorr0[outcomes!=0] = np.nan # analyze only incorrect trials.

print '\tCurrent trial: incorrect trials: %d HR; %d LR' %((Y_incorr0==1).sum(), (Y_incorr0==0).sum())


#%% print out of correct trials, how many are from each choice.... although these numbers are  not really useful, as we later set trsExcluded, so not all of these trials go into SVM analysis.

Y_corr0 = choiceVecAll+0
Y_corr0[outcomes!=1] = np.nan # only correct trials.

print '\tCurrent trial: correct trials: %d HR; %d LR' %((Y_corr0==1).sum(), (Y_corr0==0).sum())


#%% Load spikes and time traces to set X for training SVM

if chAl==1:    #%% Use choice-aligned traces 
    
    if do_Ctrace: # if 1, use C trace (denoised fluorescent trace), instead of S (inferred spikes)
        a = os.path.basename(postName)
        newPostName = os.path.join(dataPath, mousename, 'imaging', imagingFolder, 'post_Ctrace_'+a[5:])
        Data = scio.loadmat(newPostName, variable_names=['firstSideTryAl'],squeeze_me=True,struct_as_record=False)
        traces_al_1stSide = Data['firstSideTryAl'].traces.astype('float')
        time_aligned_1stSide = Data['firstSideTryAl'].time.astype('float')
        eventI = Data['firstSideTryAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
    
    else:
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
#    Y_svm = choiceVec0[~trsExcluded];
#    print 'frs x units x trials', X_svm.shape
#    print Y_svm.shape
    
    time_trace = time_aligned_1stSide

    ## incorrect trials
    trsExcluded_incorr = (np.isnan(np.sum(traces_al_1stSide, axis=(0,1))) + np.isnan(Y_incorr0)) != 0
    X_svm_incorr = traces_al_1stSide[:,:,~trsExcluded_incorr]   
    
    
    
##%%
elif chAl==0:   #%% Use stimulus-aligned traces
#    stAl = 1
    
    # Set traces_al_stim that is same as traces_al_stimAll except that in traces_al_stim some trials are set to nan, bc their stim duration is < 
    # th_stim_dur or bc their go tone happens before ep(end) or bc their choice happened before ep(end). 
    # But in traces_al_stimAll, all trials are included. 
    # You need traces_al_stim for decoding the upcoming choice bc you average responses during ep and you want to 
    # control for what happens there. But for trial-history analysis you average responses before stimOnset, so you 
    # don't care about when go tone happened or how long the stimulus was. 
    
       
    # Load stim-aligned_allTrials traces, frames, frame of event of interest
    if trialHistAnalysis==0:
        Data = scio.loadmat(postName, variable_names=['stimAl_noEarlyDec'],squeeze_me=True,struct_as_record=False)
        eventI = Data['stimAl_noEarlyDec'].eventI - 1 # remember difference indexing in matlab and python!
        traces_al_stimAll = Data['stimAl_noEarlyDec'].traces.astype('float')
        time_aligned_stim = Data['stimAl_noEarlyDec'].time.astype('float')    
        
    else:
        Data = scio.loadmat(postName, variable_names=['stimAl_allTrs'],squeeze_me=True,struct_as_record=False)
        eventI = Data['stimAl_allTrs'].eventI - 1 # remember difference indexing in matlab and python!
        traces_al_stimAll = Data['stimAl_allTrs'].traces.astype('float')
        time_aligned_stim = Data['stimAl_allTrs'].time.astype('float')
        # time_aligned_stimAll = Data['stimAl_allTrs'].time.astype('float') # same as time_aligned_stim

        
    print 'size of stimulus-aligned traces:', np.shape(traces_al_stimAll), '(frames x units x trials)'
    #DataS = Data
    traces_al_stim = traces_al_stimAll
    
    time_trace = time_aligned_stim
    
    ##%% Remove some trials from traces_al_stim (if their choice is too early or their stim duration is too short)
    ########################%% Exclude some trials from traces_al_stim ########################
    # I stopped doing below bc I don't see any point in removing such trials!!
    if 0: #trialHistAnalysis==0:            
        ep_ms = [800] # I want to make sure in none of the trials choice happened before 800ms...
        # This criteria makes sense if you want to be conservative; otherwise if ep=[1000 1300]ms, go tone will definitely be before ep end, and you cannot have the following criteria.
        # Make sure in none of the trials Go-tone happened before the end of training window (ep)
        i = (timeCommitCL_CR_Gotone - timeStimOnset) <= ep_ms[-1];
        '''
        if np.sum(i)>0:
            print 'Excluding %i trials from timeStimOnset bc their goTone is earlier than ep end' %(np.sum(i))
        #     timeStimOnset[i] = np.nan;  # by setting to nan, the aligned-traces of these trials will be computed as nan.
        else:
            print('No trials with go tone before the end of ep. Good :)')
        '''
            
        # Make sure in none of the trials choice (1st side try) happened before the end of training window (ep)
        ii = (time1stSideTry - timeStimOnset) <= ep_ms[-1];
        if np.sum(ii)>0:
            print 'Excluding %i trials from timeStimOnset bc their choice is earlier than ep end' %(np.sum(ii))
        #     timeStimOnset[i] = np.nan;  # by setting to nan, the aligned-traces of these trials will be computed as nan.
        else:
            print('No trials with choice before the end of ep. Good :)')        
    
    
        # Exclude trials whose stim duration was < th_stim_dur
        j = (timeStimOffset - timeStimOnset) < th_stim_dur;
        if np.sum(j)>0:
            print 'Excluding %i trials from timeStimOnset bc their stimDur < %dms' %(np.sum(j), th_stim_dur)
        #     timeStimOnset[j] = np.nan;
        else:
            print 'No trials with stimDur < %dms. Good :)' %th_stim_dur
    
    
    
        # Set trials to be removed from traces_al_stimAll    
        # toRmv = (i+j+ii)!=0;  
        toRmv = (j+ii)!=0; print 'Not excluding %i trials whose goTone is earlier than ep end' %sum(i)
        print 'Final: %i trials excluded in traces_al_stim' %np.sum(toRmv)
    
        
        # Set traces_al_stim for SVM classification of current choice.     
        traces_al_stim[:,:,toRmv] = np.nan
    #     traces_al_stim[:,:,outcomes==-1] = np.nan
        # print(np.shape(traces_al_stim))


    ##%%
    trsExcluded = (np.isnan(np.sum(traces_al_stim, axis=(0,1))) + np.isnan(choiceVec0)) != 0
    
    X_svm = traces_al_stim[:,:,~trsExcluded]  

    ## incorrect trials
#    if trialHistAnalysis==0:        
    trsExcluded_incorr = (np.isnan(np.sum(traces_al_stim, axis=(0,1))) + np.isnan(Y_incorr0)) != 0
    X_svm_incorr = traces_al_stim[:,:,~trsExcluded_incorr]  
#    else:
#        X_svm_incorr = 0
    

##%%
elif chAl==-1:    #%% Use outcome-aligned traces 
    # Load 1stSideTry-aligned traces, frames, frame of event of interest
    # use firstSideTryAl_COM to look at changes-of-mind (mouse made a side lick without committing it)
    Data = scio.loadmat(postName, variable_names=['rewardAl'],squeeze_me=True,struct_as_record=False)
    traces_al_rew = Data['rewardAl'].traces.astype('float')
    time_aligned_rew = Data['rewardAl'].time.astype('float')
    eventI_rew = Data['rewardAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
    #eventI_ch = Data['firstSideTryAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
    # print(np.shape(traces_al_1stSide))
     
    
    ############ incorr resp aligned
    Data = scio.loadmat(postName, variable_names=['commitIncorrAl'],squeeze_me=True,struct_as_record=False)
    traces_al_inc = Data['commitIncorrAl'].traces.astype('float')
    time_aligned_inc = Data['commitIncorrAl'].time.astype('float')
    eventI_inc = Data['commitIncorrAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
    #eventI_ch = Data['firstSideTryAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
    # print(np.shape(traces_al_1stSide))
    

    ##### align rew-al and inc-al traces    
    nbef_min = min(eventI_rew, eventI_inc) # minimum number of frames before the eventI (between rewAl and incAl traces); it will be also the new evnetI on the aligned traces that are set below
    naft_min = min((len(time_aligned_rew)-1 - eventI_rew) , (len(time_aligned_inc)-1 - eventI_inc)) # minimum number of frames after the eventI (between rewAl and incAl traces)

    # take a chunk of each trace, such that the new eventI will be nbef_min
    r_rew = np.arange(eventI_rew - nbef_min  ,  eventI_rew + naft_min+1)
    trace_rew = traces_al_rew[r_rew,:,:]
    time_aligned_outcome = time_aligned_rew[r_rew] # it is 0 at index nbef_min

    r_inc = np.arange(eventI_inc - nbef_min  ,  eventI_inc + naft_min+1)
    trace_inc = traces_al_inc[r_inc,:,:]
#    time_inc = time_aligned_inc[r_inc] # same as time_rew
    
    ##### now create a traces matrix that includes both rewAl and incAl traces...
    traces_al_outcome = np.full((trace_rew.shape), np.nan)
    traces_al_outcome[:,:,outcomes==1] = trace_rew[:,:,outcomes==1] # trace_rew is nan for outcomes other than 1
    traces_al_outcome[:,:,outcomes==0] = trace_inc[:,:,outcomes==0] # trace_inc is nan for outcomes other than 0
#    time_aligned_outcome
    eventI = nbef_min    
    
    
    ############    
    trsExcluded = (np.isnan(np.sum(traces_al_outcome, axis=(0,1))) + np.isnan(choiceVec0)) != 0
    
    X_svm = traces_al_outcome[:,:,~trsExcluded]  
#    Y_svm = choiceVec0[~trsExcluded];
#    print 'frs x units x trials', X_svm.shape
#    print Y_svm.shape
    
    time_trace = time_aligned_outcome

    ## incorrect trials
#    trsExcluded_incorr = (np.isnan(np.sum(traces_al_outcome, axis=(0,1))) + np.isnan(Y_incorr0)) != 0
#    X_svm_incorr = traces_al_1stSide[:,:,~trsExcluded_incorr]   


##%%
'''
elif chAl==2:   # Use go-tone-aligned traces
    
    goToneAl=1
    
    if noStimAft: # only analyze trials that dont have stim after go tone (more conservative)
        Data = scio.loadmat(postName, variable_names=['goToneAl_noStimAft'],squeeze_me=True,struct_as_record=False)
        traces_al_goTone = Data['goToneAl_noStimAft'].traces.astype('float')
        time_aligned_goTone = Data['goToneAl_noStimAft'].time.astype('float')
        #eventI_ch = Data['goToneAl_noStimAft'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
        # print(np.shape(traces_al_1stSide))
        
        trsExcluded = (np.isnan(np.sum(traces_al_goTone, axis=(0,1))) + np.isnan(choiceVec0)) != 0
        
        X_svm = traces_al_goTone[:,:,~trsExcluded]  
    #    Y_svm = choiceVec0[~trsExcluded];
    #    print 'frs x units x trials', X_svm.shape
    #    print Y_svm.shape
        
        time_trace = time_aligned_goTone

    else:
        print 'work on this!'
'''

print 'Number of excluded trials: %d' %(sum(trsExcluded))
print 'size of X_svm', X_svm.shape, '(frames x units x trials)'
if testIncorr:
    print 'size of X_svm_incorr', X_svm_incorr.shape, '(frames x units x trials)'


#%% Save a copy of outcomes, stimrate and allResp_HR_LR before removing trsExcluded
    
outcomes0 = outcomes+0    
stimrate0 = stimrate+0    
allResp_HR_LR0 = allResp_HR_LR+0


#%% Set Y for training SVM   

if decodeStimCateg: # Y is categorized stim rate (HR, LR)
    # Exclude trials with stimrate=cb  (We don't want to use them for training SVM though they should be fine as SVM testing trials (in fact interesting to see if variations in population activity can explain animal's choice))
    trsExcluded[stimrate==cb] = True

    Y_svm = stimrate[~trsExcluded]
    Y_svm[Y_svm<cb] = 0 # LR stimulus --> 0
    Y_svm[Y_svm>cb] = 1 # HR stimulus --> 1
#    print '%d high-rate stimuli, and %d low-rate stimuli\n' %(np.sum(Y_svm==1), np.sum(Y_svm==0))
       
elif decodeOutcome: # Y is corr, incorr
    trsExcluded = trsExcluded + ~np.logical_or(outcomes0==1, outcomes0==0) # Exclude trials with outcomes other than 0 or 1 (eg. no side commit trials)
    
    Y_svm = outcomes0[~trsExcluded]
    
else: # Y is choice
    Y_svm = choiceVec0[~trsExcluded]
    print 'Y_svm size: ', Y_svm.shape    
    
    if testIncorr: ## incorrect trials
        Y_svm_incorr = Y_incorr0[~trsExcluded_incorr]
        print 'Y_incorr size: ', Y_svm_incorr.shape
 

outcomes = outcomes0[~trsExcluded]
stimrate = stimrate0[~trsExcluded]
allResp_HR_LR = allResp_HR_LR0[~trsExcluded]


### print number of hr,lr trials ***after excluding trials*** (still before applying equal number of hr and lr trials in the decoder... so not yet the final number of trials in the decoder)

numHrLr = (Y_svm==1).sum(), (Y_svm==0).sum() # if decodeOutcome, this will be number of correct and incorrect trials.

if decodeOutcome==0:    
    
    print 'After removing trsExcluded:'
    if outcome2ana == 'corr':
        print '\tcorrect trials: %d HR; %d LR' %(numHrLr)
    elif outcome2ana == 'incorr':
        print '\tincorrect trials: %d HR; %d LR' %(numHrLr)
    else:
        print '\tall trials: %d HR; %d LR' %(numHrLr)
            
    if np.logical_and(decodeStimCateg==0, testIncorr==1):
        print '\tincorrect trials: %d HR; %d LR' %((Y_svm_incorr==1).sum(), (Y_svm_incorr==0).sum())
    
    
    if trialHistAnalysis==0:
        print '\tcorrect trials: %d HR; %d LR' %(sum(np.logical_and(outcomes==1, Y_svm==1)) , sum(np.logical_and(outcomes==1, Y_svm==0)))
        print '\tincorrect trials: %d HR; %d LR' %(sum(np.logical_and(outcomes==0, Y_svm==1)) , sum(np.logical_and(outcomes==0, Y_svm==0)))

else: # if decodeOutcome, Y_svm is outcomes
    print '\tcorrect trials: %d HR; %d LR' %(sum(np.logical_and(Y_svm==1, allResp_HR_LR==1)) , sum(np.logical_and(outcomes==1, allResp_HR_LR==0)))
    print '\tincorrect trials: %d HR; %d LR' %(sum(np.logical_and(Y_svm==0, allResp_HR_LR==1)) , sum(np.logical_and(Y_svm==0, allResp_HR_LR==0)))

# you will use hr_trs and lr_trs for a plot below... to plot the average of the trials from each class
# remember if decodeOutcome=1, then hr_trs and lr_trs are in fact corr and incorr trials.... I dont change the names bc these names are only used for a plot below and nowhere else.    
# Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
hr_trs = (Y_svm==1)
lr_trs = (Y_svm==0)
    
    
#%% We need at the very least 3 trials of each class to train SVM. So exit the analysis if this condition is not met!

if min(numHrLr) < 3:
    sys.exit('Too few trials to do SVM training! HR=%d, LR=%d' %(numHrLr))
   
    
if same_HRLR_acrossDays:
    '''
    a = os.path.join(dataPath, mousename, 'imaging', 'analysis', 'SVM_minHRLR_acrossDays_choiceDecode_chAl.mat')
    aa = scio.loadmat(a, variable_names=['mnHRLR_acrossDays'])
    mnHRLR_acrossDays = int(aa.pop('mnHRLR_acrossDays').flatten())
    '''
    mnHRLR_acrossDays = 20
    if min(numHrLr) < mnHRLR_acrossDays:
        sys.exit('Too few trials to do SVM training! HR=%d, LR=%d' %(numHrLr))

else:
    mnHRLR_acrossDays = np.nan
    
    
#%% Downsample X: average across multiple times (downsampling, not a moving average. we only average every regressBins points.)

#regressBins = 2 # number of frames to average for downsampling X
#regressBins = int(np.round(100/frameLength)) # 100ms

if np.isnan(regressBins)==0: # set to nan if you don't want to downsample.
    print 'Downsampling traces ....'    
    
    # below is problematic when it comes to aligning all sessions... they have different number of frames before eventI, so time bin 0 might be average of frames eventI-1:eventI+1 or eventI:eventI+2, etc.
    # on 10/4/17 I added the version below, where we average every 3 frames before eventI, also we average every 3 frames including eventI and after. Then we concat them.
    # this way we make sure that time bin 0, always includes eventI and 2 frames after. and time bin -1 always includes average of 3 frames before eventI.
    
    '''
    # X_svm
    T1, N1, C1 = X_svm.shape
    tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X
    X_svm = X_svm[0:regressBins*tt,:,:]
    #X_svm_d.shape
    
    X_svm = np.mean(np.reshape(X_svm, (regressBins, tt, N1, C1), order = 'F'), axis=0)
    print 'downsampled choice-aligned trace: ', X_svm.shape
        
        
    time_trace = time_trace[0:regressBins*tt]
#    print time_trace_d.shape
    time_trace = np.round(np.mean(np.reshape(time_trace, (regressBins, tt), order = 'F'), axis=0), 2)
#    print time_trace.shape

    eventI_ds = np.argwhere(np.sign(time_trace)>0)[0] # frame in downsampled trace within which event_I happened (eg time1stSideTry)    
    '''

    # new method, started on 10/4/17

    ##################################################################################
    ################################### x_svm ###################################
    ##################################################################################
    
    # set frames before frame0 (not including it)
    f = (np.arange(eventI - regressBins*np.floor(eventI/float(regressBins)) , eventI)).astype(int) # 1st frame until 1 frame before frame0 (so that the total length is a multiplicaion of regressBins)
    x = X_svm[f,:,:] # X_svmo including frames before frame0
    T1, N1, C1 = x.shape
    tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames before frame0
    xdb = np.mean(np.reshape(x, (regressBins, tt, N1, C1), order = 'F'), axis=0) # downsampled X_svmo inclusing frames before frame0
    
    
    # set frames after frame0 (including it)
    f = (np.arange(eventI , eventI + regressBins * np.floor((X_svm.shape[0] - eventI) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins    
    x = X_svm[f,:,:] # X_svmo including frames after frame0
    T1, N1, C1 = x.shape
    tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames after frame0
    xda = np.mean(np.reshape(x, (regressBins, tt, N1, C1), order = 'F'), axis=0) # downsampled X_svmo inclusing frames after frame0
    
    # set the final downsampled X_svmo: concatenate downsampled X at frames before frame0, with x at frames after (and including) frame0
    X_svm_d = np.concatenate((xdb, xda))    
#    print 'trace size--> original:',X_svm.shape, 'downsampled:', X_svm_d.shape
#    X_svm = X_svm_d
    
    ####### 
    if cbestKnown: #shflTrLabs:
        if nfrs_cbest+1 == X_svm_d.shape[0]:  # by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
            lastTimeBinMissed = 1
            print 'lastTimeBinMissed = 1! so have to set X_svm with lastTimeBinMissed to have its size matches the previously saved variable bestc!'
            # set frames after frame0 (including it)
            f = (np.arange(eventI , eventI + regressBins * np.floor((X_svm.shape[0] - (eventI+1)) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins    
            x = X_svm[f,:,:] # X_svmo including frames after frame0
            T1, N1, C1 = x.shape
            tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames after frame0
            xda = np.mean(np.reshape(x, (regressBins, tt, N1, C1), order = 'F'), axis=0) # downsampled X_svmo inclusing frames after frame0
            
            # set the final downsampled X_svmo: concatenate downsampled X at frames before frame0, with x at frames after (and including) frame0
            X_svm_d = np.concatenate((xdb, xda))    
        else:
            lastTimeBinMissed = 0
    print 'trace size--> original:',X_svm.shape, 'downsampled:', X_svm_d.shape
    X_svm = X_svm_d        




    ##################################################################################
    ################################### x_svm_incorr ###################################
    ##################################################################################
    
    if testIncorr:
        # set frames before frame0 (not including it)
        f = (np.arange(eventI - regressBins*np.floor(eventI/float(regressBins)) , eventI)).astype(int) # 1st frame until 1 frame before frame0 (so that the total length is a multiplicaion of regressBins)
        x = X_svm_incorr[f,:,:] # X_svmo including frames before frame0
        T1, N1, C1 = x.shape
        tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames before frame0
        xdb = np.mean(np.reshape(x, (regressBins, tt, N1, C1), order = 'F'), axis=0) # downsampled X_svmo inclusing frames before frame0
        
        
        # set frames after frame0 (including it)
        f = (np.arange(eventI , eventI + regressBins * np.floor((X_svm_incorr.shape[0] - eventI) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins    
        x = X_svm_incorr[f,:,:] # X_svmo including frames after frame0
        T1, N1, C1 = x.shape
        tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames after frame0
        xda = np.mean(np.reshape(x, (regressBins, tt, N1, C1), order = 'F'), axis=0) # downsampled X_svmo inclusing frames after frame0
        
        # set the final downsampled X_svmo: concatenate downsampled X at frames before frame0, with x at frames after (and including) frame0
        X_svm_d = np.concatenate((xdb, xda))    
    #    print 'trace size--> original:',X_svm.shape, 'downsampled:', X_svm_d.shape
    #    X_svm = X_svm_d
        
        ####### 
        if cbestKnown: #shflTrLabs:
            if nfrs_cbest+1 == X_svm_d.shape[0]:  # by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
                lastTimeBinMissed = 1
                print 'lastTimeBinMissed = 1! so have to set X_svm with lastTimeBinMissed to have its size matches the previously saved variable bestc!'
                # set frames after frame0 (including it)
                f = (np.arange(eventI , eventI + regressBins * np.floor((X_svm_incorr.shape[0] - (eventI+1)) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins    
                x = X_svm_incorr[f,:,:] # X_svmo including frames after frame0
                T1, N1, C1 = x.shape
                tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames after frame0
                xda = np.mean(np.reshape(x, (regressBins, tt, N1, C1), order = 'F'), axis=0) # downsampled X_svmo inclusing frames after frame0
                
                # set the final downsampled X_svmo: concatenate downsampled X at frames before frame0, with x at frames after (and including) frame0
                X_svm_d = np.concatenate((xdb, xda))    
            else:
                lastTimeBinMissed = 0
        print 'trace size--> original:',X_svm_incorr.shape, 'downsampled:', X_svm_d.shape
        X_svm_incorr = X_svm_d  


  
    
    ##################################################################################
    ################################### time_trace ###################################
    ##################################################################################
    
    # set frames before frame0 (not including it)
    f = (np.arange(eventI - regressBins*np.floor(eventI/float(regressBins)) , eventI)).astype(int) # 1st frame until 1 frame before frame0 (so that the total length is a multiplicaion of regressBins)
    x = time_trace[f] # time_trace including frames before frame0
    T1 = x.shape[0]
    tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames before frame0 # same as eventI_ds
    xdb = np.mean(np.reshape(x, (regressBins, tt), order = 'F'), axis=0) # downsampled X_svm inclusing frames before frame0
    
    
    # set frames after frame0 (including it)
    f = (np.arange(eventI , eventI + regressBins * np.floor((time_trace.shape[0] - eventI) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins    
#    f = (np.arange(eventI , eventI+regressBins * np.floor((time_trace.shape[0] - (eventI+1)) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins    
    x = time_trace[f] # X_svm including frames after frame0
    T1 = x.shape[0]
    tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames after frame0
    xda = np.mean(np.reshape(x, (regressBins, tt), order = 'F'), axis=0) # downsampled X_svm inclusing frames after frame0
    
    # set the final downsampled time_trace: concatenate downsampled X at frames before frame0, with x at frames after (including) frame0
    time_trace_d = np.concatenate((xdb, xda))   # time_traceo[eventI] will be an array if eventI is an array, but if you load it from matlab as int, it wont be an array and you have to do [time_traceo[eventI]] to make it a list so concat works below:
#    time_trace_d = np.concatenate((xdb, [time_traceo[eventI]], xda))    
#    print 'time trace size--> original:',time_trace.shape, 'downsampled:', time_trace_d.shape    
#    time_trace = time_trace_d

    if cbestKnown: #shflTrLabs:
        if lastTimeBinMissed == 1:  # by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)            
            print 'lastTimeBinMissed = 1! so have to set time_trace with lastTimeBinMissed to have its size match the previously saved variable bestc!'
            # set frames after frame0 (including it)
            f = (np.arange(eventI , eventI + regressBins * np.floor((time_trace.shape[0] - (eventI+1)) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins    
        #    f = (np.arange(eventI , eventI+regressBins * np.floor((time_trace.shape[0] - (eventI+1)) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins    
            x = time_trace[f] # X_svm including frames after frame0
            T1 = x.shape[0]
            tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames after frame0
            xda = np.mean(np.reshape(x, (regressBins, tt), order = 'F'), axis=0) # downsampled X_svm inclusing frames after frame0
            
            # set the final downsampled time_trace: concatenate downsampled X at frames before frame0, with x at frames after (including) frame0
            time_trace_d = np.concatenate((xdb, xda))   # time_traceo[eventI] will be an array if eventI is an array, but if you load it from matlab as int, it wont be an array and you have to do [time_traceo[eventI]] to make it a list so concat works below:
        #    time_trace_d = np.concatenate((xdb, [time_traceo[eventI]], xda))
        else:
            lastTimeBinMissed = 0
    print 'time trace size--> original:',time_trace.shape, 'downsampled:', time_trace_d.shape    
    time_trace = time_trace_d

    
    ######################################################################
    
    eventI_ds = np.argwhere(np.sign(time_trace_d)>0)[0]  # frame in downsampled trace within which event_I happened (eg time1stSideTry)    
    


else:
    print 'Not downsampling traces ....'


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
#if neuronType==2:
inhRois = inhibitRois[~NsExcluded]
#    print 'Number: inhibit = %d, excit = %d, unsure = %d' %(np.sum(inhRois==1), np.sum(inhRois==0), np.sum(np.isnan(inhRois)))
#    print 'Fraction: inhibit = %.2f, excit = %.2f, unsure = %.2f' %(fractInh, fractExc, fractUn)

    
#%% Print some vars

numDataPoints = X_svm.shape[2] 
print '# data points = %d' %numDataPoints
# numTrials = (~trsExcluded).sum()
# numNeurons = (~NsExcluded).sum()
numTrials, numNeurons = X_svm.shape[2], X_svm.shape[1]
print '%d trials; %d neurons' %(numTrials, numNeurons)
# print ' The data has %d frames recorded from %d neurons at %d trials' %Xt.shape    


#%% Keep a copy of X_svm before normalization

X_svm00 = X_svm + 0
if testIncorr:
    X_svm_incorr00 = X_svm_incorr + 0


#%% Center and normalize X: feature normalization and scaling: to remove effects related to scaling and bias of each neuron, we need to zscore data (i.e., make data mean 0 and variance 1 for each neuron) 

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
X_svm_N = np.full(np.shape(X_svm00), np.nan)
if testIncorr:
    X_svm_incorr_N = np.full(np.shape(X_svm_incorr00), np.nan)
meanX_fr = []
stdX_fr = []
for ifr in range(np.shape(X_svm00)[0]):
    
    if testIncorr: # use both corr and incorr traces to set mean and std for each neuron. Remember X_svm already includes incorr trials if outcome2ana is 'all'.
        xtot = np.concatenate((X_svm00, X_svm_incorr00),axis=2)
        xf = xtot[ifr,:,:]
    else:
        xf = X_svm00[ifr,:,:]    
    
    # on 12/05/2018 you noticed that you are using X_svm00 below, while it should be xf. It will make a difference for when training on correct and testing on incorrect.
#    m = np.mean(X_svm00[ifr,:,:], axis=1)
#    s = np.std(X_svm00[ifr,:,:], axis=1)   
    m = np.mean(xf, axis=1)
    s = np.std(xf, axis=1)   
    
    meanX_fr.append(m) # frs x neurons
    stdX_fr.append(s)       
    
    if softNorm==1: # soft normalziation : neurons with sd<thAct wont have too large values after normalization
        s = s+thAct     

    X_svm_N[ifr,:,:] = ((X_svm00[ifr,:,:].T - m) / s).T
    
    if testIncorr:
        X_svm_incorr_N[ifr,:,:] = ((X_svm_incorr00[ifr,:,:].T - m) / s).T  

meanX_fr = np.array(meanX_fr) # frames x neurons
stdX_fr = np.array(stdX_fr) # frames x neurons

X_svm = X_svm_N
if testIncorr:
    X_svm_incorr = X_svm_incorr_N
    


#%% Remove noise corrlations: Define function to shuffle trials in X_svm (for each neuron independently) to break correlations in neurons FRs across trials.
    # we want to see how choice decoding would be different when :
    # 1) potential correlations between neurons in a trial are broken.
    # 2) any trial-history effect on neural responses will be shuffled out.
    
def shflTrsPerN(X_svm_bp, Y_svm_bp):
    
    import numpy as np
    import numpy.random as rng
    
    ##%% Shuffle trials in X_svm (for each neuron independently) to break correlations between neurons in each trial.
    
    X_svm_hr_perm = np.full((X_svm_bp.shape[0], X_svm_bp.shape[1], sum(Y_svm_bp==1)), np.nan) # frs x neurons x hrTrs
    X_svm_lr_perm = np.full((X_svm_bp.shape[0], X_svm_bp.shape[1], sum(Y_svm_bp==0)), np.nan) # frs x neurons x lrTrs
    
    hrtrs = np.argwhere(Y_svm_bp==1).flatten() # original order of HR trials which is the same for all neurons.
    lrtrs = np.argwhere(Y_svm_bp==0).flatten()
    
    # For each neuron use a different order of trials ... do it separately for hr and lr trials, so the newly synthesized HR trial includes neural responses from only HR trials, but different HR trials for different neurons
    # shuffle order of HR trials
    hr_shfl_all = [] # neurons x numHRtrials # index of hr trials after shuffling. 
    # hr_shfl_all[:,0] shows the HR trial used for each neuron as the 1st fakeHRtr (in the actual case the 1st HR trial will be hrtrs[0] for all neurons); as if those HR trials were 1 single (fake) trial (for which all neurons were recorded simultaneously.)
    # hr_shfl_all[n,:] is the order of hr trials for neuron n, which is different from the order of hr trials for another neuron
    for ine in range(X_svm_bp.shape[1]):
    #    trs = rng.permutation(X_svm_bp.shape[2])
        hr_shfl = rng.permutation(hrtrs) # there are indeces of hr trials, but in a shuffled order
        X_svm_hr_perm[:,ine,:] = X_svm_bp[:,ine, hr_shfl] # so neuron ine in X_svm_hr_perm will have the response of this same neuron but in the order of hr_shfl
        hr_shfl_all.append(hr_shfl)
        
        
    # shuffle order of LR trials
    lr_shfl_all = [] #neurons x numLRtrials 
    for ine in range(X_svm_bp.shape[1]):
    #    trs = rng.permutation(X_svm_bp.shape[2])
        lr_shfl = rng.permutation(lrtrs)
        X_svm_lr_perm[:,ine,:] = X_svm_bp[:,ine, lr_shfl]
        lr_shfl_all.append(lr_shfl)
        
        
    # first put hr trials, then lr trials (for each neuron the order is different)
    X_svm = np.concatenate((X_svm_hr_perm, X_svm_lr_perm), axis=2)
    
    # set Y_svm_bp for the new X_svm
    Y_svm = np.concatenate((np.full(sum(Y_svm_bp==1), 1.), np.full(sum(Y_svm_bp==0), 0.)))
    
    hr_shfl_all = np.array(hr_shfl_all)
    lr_shfl_all = np.array(lr_shfl_all)
    
    return X_svm, Y_svm, hr_shfl_all, lr_shfl_all
    ##%% Again define hr and lr trials.
#    hr_trs = (Y_svm==1)
#    lr_trs = (Y_svm==0)        
#    print '%d HR trials; %d LR trials' %(sum(hr_trs), sum(lr_trs))


#%% Remove noise correlaions; if desired, shuffle trials in X_svm (for each neuron independently) to break correlations in neurons FRs across trials.

if shflTrsEachNeuron:

    ##%% Keep a copy of X_svm before shuffling trials
    X_svm_bp = X_svm+0
    Y_svm_bp = Y_svm+0    
    
    X_svm, Y_svm, hr_shfl_all, lr_shfl_all = shflTrsPerN(X_svm_bp, Y_svm_bp)
    
else:
    hr_shfl_all = [] # neurons x numHRtrials # index of hr trials after shuffling. 
    lr_shfl_all = [] # neurons x numLRtrials
    
    
###############################################################################


#%% Look at min and max of each neuron at each frame ... this is to decide for normalization
"""
mn = []
mx = []
for t in range(X_svm.shape[0]):
    x = X_svm[t,:,:].squeeze().T # trails x units
    mn.append(np.min(x,axis=0))
    mx.append(np.max(x,axis=0))

mn = np.array(mn)
mx = np.array(mx)


plt.figure(figsize=(4,4))
#fig, axes = plt.subplots(nrows=1, ncols=2)
plt.imshow(mn, cmap='jet', interpolation='nearest', aspect='auto') # frames x units
plt.colorbar()
#im = axes[0].imshow(mn, cmap='jet', aspect='auto') # frames x units
#clim=im.properties()['clim']
#clim=im.properties()['clim']
#axes[1].imshow(my_image2, clim=clim)
#plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)

plt.figure(figsize=(4,4))
plt.imshow(mx, cmap='jet', interpolation='nearest', aspect='auto') # frames x units
plt.colorbar()
"""


#%% Plot

#hr_trs = (Y_svm==1)
#lr_trs = (Y_svm==0)

if doPlots:
    plt.figure
    plt.subplot(2,2,1)
#    plt.plot(np.mean(meanX_fr,axis=0)) # average across frames for each neuron
    plt.hist(meanX_fr.reshape(-1,))
    plt.xlabel('meanX \n(mean of trials)')
    plt.ylabel('# frames x neurons')
    plt.title('min = %.6f' %(np.min(np.mean(meanX_fr,axis=0))))

    plt.subplot(2,2,3)
#    plt.plot(np.mean(stdX_fr,axis=0))
    plt.hist(stdX_fr.reshape(-1,))
    plt.xlabel('stdX \n(std of trials)')
    plt.ylabel('# frames x neurons')
    plt.title('min = %.6f' %(np.min(np.mean(stdX_fr,axis=0))))

    plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.subplots_adjust(hspace=.5)


    
# choice-aligned: classes: choices
# Plot stim-aligned averages after centering and normalization
##%% Again define hr and lr trials.

if doPlots:
#    # Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
#    hr_trs = (Y_svm==1)
#    lr_trs = (Y_svm==0)

    if decodeOutcome:
        labs = 'correct', 'incorrect'
    else:
        labs = 'high rate', 'low rate'
        
    plt.figure()
    plt.subplot(1,2,1)
    
    a1 = np.nanmean(X_svm[:, :, hr_trs],  axis=1) # frames x trials (average across neurons)
    tr1 = np.nanmean(a1,  axis = 1)
    tr1_se = np.nanstd(a1,  axis = 1) / np.sqrt(sum(hr_trs));

    a0 = np.nanmean(X_svm[:, :, lr_trs],  axis=1) # frames x trials (average across neurons)
    tr0 = np.nanmean(a0,  axis = 1)
    tr0_se = np.nanstd(a0,  axis = 1) / np.sqrt(sum(lr_trs));

#    mn = np.concatenate([tr1,tr0]).min()
#    mx = np.concatenate([tr1,tr0]).max()
#    plt.plot([win[0], win[0]], [mn, mx], 'g-.') # mark the begining and end of training window
#    plt.plot([win[-1], win[-1]], [mn, mx], 'g-.')
    plt.fill_between(time_trace, tr1-tr1_se, tr1+tr1_se, alpha=0.5, edgecolor='b', facecolor='b')
    plt.fill_between(time_trace, tr0-tr0_se, tr0+tr0_se, alpha=0.5, edgecolor='r', facecolor='r')
    plt.plot(time_trace, tr1, 'b', label = labs[0]) 
    plt.plot(time_trace, tr0, 'r', label = labs[1])
    # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, lr_trs],  axis = (1, 2)), 'r', label = 'high rate')
    # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, hr_trs],  axis = (1, 2)), 'b', label = 'low rate')
#    plt.xlabel('time aligned to stimulus onset (ms)')
    plt.title('Population average - raw')
    plt.legend()


'''
allN_hrs_fr0 = np.full((X_svm.shape[1], sum(Y_svm==1)), np.nan) # nNs x nHrTrs
for i in range(sum(Y_svm==1)):
    allN_hrs_fr0[:,i] = X_svm[0,:,np.argwhere(Y_svm==1)[i]].flatten()

allN_lrs_fr0 = np.full((X_svm.shape[1], sum(Y_svm==0)), np.nan) # nNs x nLrTrs
for i in range(sum(Y_svm==0)):
    allN_lrs_fr0[:,i] = X_svm[0,:,np.argwhere(Y_svm==0)[i]].flatten()
    

for ine in range(allN_hrs_fr0.shape[0]):
    plt.figure()
    plt.plot(allN_hrs_fr0[ine])
    plt.plot(allN_lrs_fr0[ine])
'''    


#%% Set vars for SVM

#useEqualTrNums = 1
smallestC = 0 # if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
shuffleTrs = False # set to 0 so for each iteration of numSamples, all frames are trained and tested on the same trials# If 1 shuffle trials to break any dependencies on the sequence of trails 


#%% Function to run SVM  (when X is frames x units x trials)
# Remember each numSamples will have a different set of training and testing dataset, however for each numSamples, the same set of testing/training dataset
# will be used for all frames and all values of c (unless shuffleTrs is 1, in which case different frames and c values will have different training/testing datasets.)

def setbesc_frs(X,Y,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest=np.nan,fr2an=np.nan, shflTrLabs=0, X_svm_incorr=0, Y_svm_incorr=0, mnHRLR_acrossDays=np.nan):
#    X = X_svm
#    Y = Y_svm
#    cbest=np.nan
#    fr2an=np.nan
#    X_svm_incorr=0
#    Y_svm_incorr=0
#    mnHRLR_acrossDays=np.nan
    
    # numSamples = 10; # number of iterations for finding the best c (inverse of regularization parameter)
    # if you don't want to regularize, go with a very high cbest and don't run the section below.
    # cbest = 10**6    
    # regType = 'l1'
    # kfold = 10;
            
    import numpy as np
    import numpy.random as rng
    from crossValidateModel import crossValidateModel
    from linearSVM import linearSVM
    
    def perClassError(Y, Yhat):
        perClassEr = np.sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
        return perClassEr


    if type(X_svm_incorr)==int:
        testIncorr = 0
    else:
        testIncorr = 1
        
    if testIncorr:
        len_test_incorr = len(Y_svm_incorr)
        
    # frames to do SVM analysis on
    if np.isnan(fr2an).all(): # run SVM on all frames
        frs = range(X.shape[0])
    else:
        frs = fr2an        
    
    
    # set range of c (regularization parameters) to check    
    if np.isnan(cbest).all(): # we need to set cbest
        bestcProvided = False        
        if regType == 'l1':
            print '\n-------------- Running l1 svm classification --------------\r' 
            # cvect = 10**(np.arange(-4, 6,0.2))/numTrials;
            cvect = 10**(np.arange(-4, 6,0.2))/numDataPoints
        elif regType == 'l2':
            print '\n-------------- Running l2 svm classification --------------\r' 
            cvect = 10**(np.arange(-6, 6,0.2))/numDataPoints          
        nCvals = len(cvect)
#        print 'try the following regularization values: \n', cvect
        # formattedList = ['%.2f' % member for member in cvect]
        # print 'try the following regularization values = \n', formattedList        
    else: # bestc is provided and we want to fit svm on shuffled trial labels
        bestcProvided = True           
        nCvals = 1 # cbest is already provided
        
    
#    smallestC = 0 # if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
    if smallestC==1:
        print 'bestc = smallest c whose cv error is less than 1se of min cv error'
    else:
        print 'bestc = c that gives min cv error'
    #I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.
        
        
    ##############
    hrn = (Y==1).sum()
    lrn = (Y==0).sum()

    if ~np.isnan(mnHRLR_acrossDays): # we will make sure for all days SVM is trained on similar number of trials, ie mnHRLR_acrossDays trials of HR and mnHRLR_acrossDays trials of LR.
        trsn = mnHRLR_acrossDays        
        numTrials = 2*mnHRLR_acrossDays 
        print 'using %d HR and %d LR trials for SVM training, same across all sessions' %(mnHRLR_acrossDays, mnHRLR_acrossDays)
    else:        
        if useEqualTrNums and hrn!=lrn: # if the HR and LR trials numbers are not the same, pick equal number of trials of the 2 classes!        
            trsn = min(lrn,hrn)
            if hrn > lrn:
                print 'Subselecting HR trials so both classes have the same number of trials!'
                numTrials = lrn*2
            elif lrn > hrn:
                print 'Subselecting LR trials so both classes have the same number of trials!'
                numTrials = hrn*2            
        else:
            numTrials = X.shape[2]
    
    print 'FINAL: %d trials; %d neurons' %(numTrials, X.shape[1])
    
    len_test = numTrials - int((kfold-1.)/kfold*numTrials) # length of testing trials   
            
    X0 = X + 0
    Y0 = Y + 0
    
    
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    nFrs = np.shape(X)[0]
    wAllC = np.ones((numSamples, nCvals, X.shape[1], nFrs))+np.nan;
    bAllC = np.ones((numSamples, nCvals, nFrs))+np.nan;
    
    perClassErrorTrain = np.ones((numSamples, nCvals, nFrs))+np.nan;
    perClassErrorTest = np.ones((numSamples, nCvals, nFrs))+np.nan;    
    perClassErrorTest_shfl = np.ones((numSamples, nCvals, nFrs))+np.nan;
    perClassErrorTest_chance = np.ones((numSamples, nCvals, nFrs))+np.nan
    
    testTrInds_allSamps = np.full((numSamples, len_test), np.nan)              
    Ytest_allSamps = np.full((numSamples, len_test), np.nan)              
    Ytest_hat_allSampsFrs = np.full((numSamples, nCvals, nFrs, len_test), np.nan)        
#    testTrInds_outOfY0_allSamps = np.full((numSamples, len_test), np.nan)
    trsnow_allSamps = np.full((numSamples, numTrials), np.nan)              
#    eqy = np.full((X0.shape[0], numSamples), np.nan)
    
    # incorr:
    if testIncorr:
        perClassErrorTest_incorr = np.ones((numSamples, nCvals, nFrs))+np.nan
        perClassErrorTest_shfl_incorr = np.ones((numSamples, nCvals, nFrs))+np.nan
        perClassErrorTest_chance_incorr = np.ones((numSamples, nCvals, nFrs))+np.nan        
        Ytest_hat_allSampsFrs_incorr = np.full((numSamples, nCvals, nFrs, len_test_incorr), np.nan)        
        
        
    ########################################## Train SVM numSamples times to get numSamples cross-validated datasets.    
    for s in range(numSamples): # s=0       
        print 'Iteration %d' %(s)
        
        ############ Make sure both classes have the same number of trials when training the classifier
        # set trsnow: # index of trials (out of Y0) after picking random hr (or lr) in order to make sure both classes have the same number in the final Y (on which svm was run)
        
        if ~np.isnan(mnHRLR_acrossDays): 
            randtrs_hr = np.argwhere(Y0==1)[rng.permutation(hrn)[0:trsn]].squeeze() # subselect mnHRLR_acrossDays trials from HR
            randtrs_lr = np.argwhere(Y0==0)[rng.permutation(lrn)[0:trsn]].squeeze() # subselect mnHRLR_acrossDays trials from LR
            trsnow = np.sort(np.concatenate((randtrs_hr , randtrs_lr))) # index of trials after picking random hr (or lr) in order to make sure both classes have the same number in the final Y (on which svm was run)

#            X = X0[:,:,trsnow] # trsnow : index of trials (out of X0 and Y0) that are used to set X and Y
#            Y = Y0[trsnow]
                
        else:
            if useEqualTrNums and hrn!=lrn: # if the HR and LR trials numbers are not the same, pick equal number of trials of the 2 classes!
                if hrn > lrn:
                    randtrs = np.argwhere(Y0==1)[rng.permutation(hrn)[0:trsn]].squeeze()
                    trsnow = np.sort(np.concatenate((randtrs , np.argwhere(Y0==0).squeeze()))) # index of trials after picking random hr (or lr) in order to make sure both classes have the same number in the final Y (on which svm was run)
                elif lrn > hrn:
                    randtrs = np.argwhere(Y0==0)[rng.permutation(lrn)[0:trsn]].squeeze() # random sample of the class with more trials
                    trsnow = np.sort(np.concatenate((randtrs , np.argwhere(Y0==1).squeeze()))) # all trials of the class with fewer trials + the random sample set above for the other class
    
#                X = X0[:,:,trsnow] # trsnow : index of trials (out of X0 and Y0) that are used to set X and Y
#                Y = Y0[trsnow]
    
            else: # include all trials
                trsnow = np.arange(0, len(Y0))
        
        X = X0[:,:,trsnow] # trsnow : index of trials (out of X0 and Y0) that are used to set X and Y
        Y = Y0[trsnow]
            
        trsnow_allSamps[s,:] = trsnow
#        numTrials, numNeurons = X.shape[2], X.shape[1]
#            print 'FINAL: %d trials; %d neurons' %(numTrials, numNeurons)                        
            
        ######################## Setting chance Y: same length as Y for testing data, and with equal number of classes 0 and 1.
#        no = Y.shape[0]
#        len_test = numTrials - int((kfold-1.)/kfold*numTrials)    
        permIxs = rng.permutation(len_test) # needed to set perClassErrorTest_shfl   
    
        Y_chance = np.zeros(len_test)
        if rng.rand()>.5:
            b = rng.permutation(len_test)[0:np.floor(len_test/float(2)).astype(int)]
        else:
            b = rng.permutation(len_test)[0:np.ceil(len_test/float(2)).astype(int)]
        Y_chance[b] = 1

        if testIncorr:
            permIxs_incorr = rng.permutation(len_test_incorr)
            Y_chance_incorr = np.zeros(len_test_incorr)
            if rng.rand()>.5:
                b = rng.permutation(len_test_incorr)[0:np.floor(len_test_incorr/float(2)).astype(int)]
            else:
                b = rng.permutation(len_test_incorr)[0:np.ceil(len_test_incorr/float(2)).astype(int)]
            Y_chance_incorr[b] = 1            
            
            
        ####################### Set the chance Y for training SVM on shuffled trial labels
        if shflTrLabs: # shuffle trial classes in Y
            Y = np.zeros(numTrials) # Y_chance0
            if rng.rand()>.5:
                b = rng.permutation(numTrials)[0:np.floor(numTrials/float(2)).astype(int)]
            else:
                b = rng.permutation(numTrials)[0:np.ceil(numTrials/float(2)).astype(int)]
            Y[b] = 1

        
        ######################## Shuffle trial orders, so the training and testing datasets are different for each numSamples (we only do this if shuffleTrs is 0, so crossValidateModel does not shuffle trials, so we have to do it here, otherwise all numSamples will have the same set of testing and training datasets.)
        ######################## REMEMBER : YOU ARE CHANGING THE ORDER OF TRIALS HERE!!!
        ########################
        ########################
        if shuffleTrs==0: # shuffle trials here (instead of inside crossValidateModel) to break any dependencies on the sequence of trails 
            
#            Ybefshfl = Y            
            shfl = rng.permutation(np.arange(0, numTrials)) # shfl: new order of trials ... shuffled indeces of Y... the last 1/10th indeces will be testing trials.
            
            Y = Y[shfl] 
            X = X[:,:,shfl]             
            
            # Ytest_allSamps[s,:] : Y that will be used as testing trials in this sample
            Ytest_allSamps[s,:] = Y[np.arange(numTrials-len_test, numTrials)] # the last 1/10th of Y (after applying shfl labels to it)
            testTrInds = shfl[np.arange(numTrials-len_test, numTrials)] # indeces to be applied on trsnow in order to get the trials (index out of Y0) that were used as testing trs; eg stimrate[trsnow[testTrInds]] is the stimrate of testing trials
#            testTrInds_outOfY0 = trsnow[testTrInds] # index of testing trials out of Y0 (not Y!) (that will be used in svm below)
             ######## IMPORTANT: Ybefshfl[testTrInds] is same as Y0[trsnow[testTrInds]] and same as Y[np.arange(numTrials-len_test, numTrials)] and same as summary.YTest computed below ########

            testTrInds_allSamps[s,:] = testTrInds            
#            testTrInds_outOfY0_allSamps[s,:] = testTrInds_outOfY0            
        else:
            testTrInds_allSamps = np.nan # for now, but to set it correctly: testTrInds will be set in crossValidateModel.py, you just need to output it from crossValidateModel
            Ytest_allSamps[s,:] = np.nan  

        
        ########################## Start training SVM ##########################
        for ifr in frs: # train SVM on each frame
            
            if bestcProvided:
                cvect = [cbest[ifr]]
        
#            print '\tFrame %d' %(ifr)  
            ######################## Loop over different values of regularization
            for i in range(nCvals): # train SVM using different values of regularization parameter
                if regType == 'l1':                               
                    summary,_ =  crossValidateModel(X[ifr,:,:].transpose(), Y, linearSVM, kfold = kfold, l1 = cvect[i], shflTrs = shuffleTrs)
                    
                elif regType == 'l2':
                    summary,_ =  crossValidateModel(X[ifr,:,:].transpose(), Y, linearSVM, kfold = kfold, l2 = cvect[i], shflTrs = shuffleTrs)
                        
                '''
                ###### below is the gist of the codes for svm training (done in linearSVM.py) ######
                
                import sklearn.svm as svm                        
                
                linear_svm = svm.LinearSVC(C = cbest_allExc[12], loss='squared_hinge', penalty='l2', dual=True) # get c for a frame, eg frame 12.
                linear_svm.fit(X_svm[12,:,:].transpose(), np.squeeze(Y_svm)) # x should be in trials x units
                
                linear_svm.predict(XTest)
                linear_svm.coef_
                linear_svm.intercept_
                
                ###
                # def perClassError(Y, Yhat):
                #    import numpy as np
                #    perClassEr = np.sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
                #    return perClassEr
                
                # perClassErrorTest = perClassError(YTest, linear_svm.predict(XTest)); 
                
                # summary.model = linear_svm;
                '''
                    
                wAllC[s,i,:,ifr] = np.squeeze(summary.model.coef_); # weights of all neurons for each value of c and each shuffle
                bAllC[s,i,ifr] = np.squeeze(summary.model.intercept_);
        
                # classification errors                    
                perClassErrorTrain[s,i,ifr] = summary.perClassErrorTrain
                perClassErrorTest[s,i,ifr] = summary.perClassErrorTest # perClassError(YTest, linear_svm.predict(XTest));
                
                # Testing correct shuffled data: same decoder trained on correct trials, but use shuffled trial labels to compute class error
                Ytest_hat = summary.model.predict(summary.XTest) # prediction of trial label for each trial
                perClassErrorTest_shfl[s,i,ifr] = perClassError(summary.YTest[permIxs], Ytest_hat) # fraction of incorrect predicted trial labels
                perClassErrorTest_chance[s,i,ifr] = perClassError(Y_chance, Ytest_hat)
                Ytest_hat_allSampsFrs[s,i,ifr,:] = Ytest_hat
                
                # Incorrect trials
                if testIncorr:
                    Ytest_hat_incorr = summary.model.predict(X_svm_incorr[ifr,:,:].transpose()) # prediction of trial label for each trial
                    perClassErrorTest_incorr[s,i,ifr] = perClassError(Y_svm_incorr, Ytest_hat_incorr) # fraction of incorrect predicted trial labels
                    perClassErrorTest_shfl_incorr[s,i,ifr] = perClassError(Y_svm_incorr[permIxs_incorr], Ytest_hat_incorr) # fraction of incorrect predicted trial labels
                    perClassErrorTest_chance_incorr[s,i,ifr] = perClassError(Y_chance_incorr, Ytest_hat_incorr)
                    Ytest_hat_allSampsFrs_incorr[s,i,ifr,:] = Ytest_hat_incorr
                
                
                ########## sanity check ##########
                """
                trsnow = trsnow_allSamps[s].astype(int)
                testTrInds = testTrInds_allSamps[s].astype(int)
                testTrInds_outOfY0 = trsnow[testTrInds]
                xx = X0[ifr][:,testTrInds_outOfY0]        
                yy = Y0[testTrInds_outOfY0]
                
                ww = wAllC[s,i,:,ifr]
#                normw = sci.linalg.norm(ww)   # numSamps x numFrames
#                ww = ww / normw                 
                
                bb = bAllC[s,i,ifr] 
                
                # Project population activity of each frame onto the decoder of frame ifr
                yhat = np.dot(ww, xx) + bb # testingFrs x testing trials                
                th = 0
                yhat[yhat<th] = 0 # testingFrs x testing trials
                yhat[yhat>th] = 1
                                
                d = yhat - yy  # testing Frs x nTesting Trials # difference between actual and predicted y
                c = np.mean(abs(d), axis=-1) * 100

                eqy[ifr, s] = np.equal(c, perClassErrorTest[s,i,ifr])                
                
                if eqy[ifr, s]==0:
                    print np.mean(np.equal(xx.T, summary.XTest))
                    print np.mean(np.equal(yy, summary.YTest))
                    print np.mean(np.equal(yhat, Ytest_hat))
                    print ifr, s
                    print c, perClassErrorTest[s,i,ifr]
                    sys.exit('Error!') 
                """



    ######################### Find bestc for each frame, and plot the c path 
    if bestcProvided: 
        cbestAllFrs = cbest
        cbestFrs = cbest
        
    else:
        print '--------------- Identifying best c ---------------' 
        cbestFrs = np.full((X.shape[0]), np.nan)  
        cbestAllFrs = np.full((X.shape[0]), np.nan)  
        
        for ifr in frs: #range(X.shape[0]):    
            #######%% Compute average of class errors across numSamples        
            meanPerClassErrorTrain = np.mean(perClassErrorTrain[:,:,ifr], axis = 0);
            semPerClassErrorTrain = np.std(perClassErrorTrain[:,:,ifr], axis = 0)/np.sqrt(numSamples);
            
            meanPerClassErrorTest = np.mean(perClassErrorTest[:,:,ifr], axis = 0);
            semPerClassErrorTest = np.std(perClassErrorTest[:,:,ifr], axis = 0)/np.sqrt(numSamples);
            
            meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl[:,:,ifr], axis = 0);
#            semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl[:,:,ifr], axis = 0)/np.sqrt(numSamples);            
            meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance[:,:,ifr], axis = 0);
#            semPerClassErrorTest_chance = np.std(perClassErrorTest_chance[:,:,ifr], axis = 0)/np.sqrt(numSamples);
            
            
            #######%% Identify best c                
            # Use all range of c... it may end up a value at which all weights are 0.
            ix = np.argmin(meanPerClassErrorTest)
            if smallestC==1:
                cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
                cbest = cbest[0]; # best regularization term based on minError+SE criteria
                cbestAll = cbest
            else:
                cbestAll = cvect[ix]
            print '\tFrame %d: %f' %(ifr,cbestAll)
            cbestAllFrs[ifr] = cbestAll
            
            ####### Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)
            if regType == 'l1': # in l2, we don't really have 0 weights!
                sys.exit('Needs work! below wAllC has to be for 1 frame') 
                
                a = abs(wAllC)>eps # non-zero weights
                b = np.mean(a, axis=(0,2,3)) # Fraction of non-zero weights (averaged across shuffles)
                c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle
                cvectnow = cvect[c1stnon0:]
                
                meanPerClassErrorTestnow = np.mean(perClassErrorTest[:,c1stnon0:,ifr], axis = 0);
                semPerClassErrorTestnow = np.std(perClassErrorTest[:,c1stnon0:,ifr], axis = 0)/np.sqrt(numSamples);
                ix = np.argmin(meanPerClassErrorTestnow)
                if smallestC==1:
                    cbest = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
                    cbest = cbest[0]; # best regularization term based on minError+SE criteria    
                else:
                    cbest = cvectnow[ix]
                
                print 'best c (at least 1 non-0 weight) = ', cbest
            else:
                cbest = cbestAll
                    
            cbestFrs[ifr] = cbest
            
            
            ########%% Set the decoder and class errors at best c (for data)
            """
            # you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
            # we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
            indBestC = np.in1d(cvect, cbest)
            
            w_bestc_data = wAllC[:,indBestC,:,ifr].squeeze() # numSamps x neurons
            b_bestc_data = bAllC[:,indBestC,ifr]
            
            classErr_bestC_train_data = perClassErrorTrain[:,indBestC,ifr].squeeze()
            
            classErr_bestC_test_data = perClassErrorTest[:,indBestC,ifr].squeeze()
            classErr_bestC_test_shfl = perClassErrorTest_shfl[:,indBestC,ifr].squeeze()
            classErr_bestC_test_chance = perClassErrorTest_chance[:,indBestC,ifr].squeeze()
            """
            
            
            ########### Plot C path    
            if doPlots:              
        #        print 'Best c (inverse of regularization parameter) = %.2f' %cbest
                plt.figure()
                plt.subplot(1,2,1)
                plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
                plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
            #    plt.fill_between(cvect, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
            #    plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
                
                plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
                plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation')
                plt.plot(cvect, meanPerClassErrorTest_chance, 'b', label = 'cv-chance')       
                plt.plot(cvect, meanPerClassErrorTest_shfl, 'y', label = 'cv-shfl')            
            
                plt.plot(cvect[cvect==cbest], meanPerClassErrorTest[cvect==cbest], 'bo')
                
                plt.xlim([cvect[1], cvect[-1]])
                plt.xscale('log')
                plt.xlabel('c (inverse of regularization parameter)')
                plt.ylabel('classification error (%)')
                plt.legend(loc='center left', bbox_to_anchor=(1, .7))
                
                plt.title('Frame %d' %(ifr))
                plt.tight_layout()
    
    

    ##############

    if testIncorr:
        return perClassErrorTrain, perClassErrorTest, wAllC, bAllC, cbestAllFrs, cbestFrs, cvect, perClassErrorTest_shfl, perClassErrorTest_chance, testTrInds_allSamps, Ytest_allSamps, Ytest_hat_allSampsFrs, trsnow_allSamps, perClassErrorTest_incorr, perClassErrorTest_shfl_incorr, perClassErrorTest_chance_incorr, Ytest_hat_allSampsFrs_incorr 
    else:
        return perClassErrorTrain, perClassErrorTest, wAllC, bAllC, cbestAllFrs, cbestFrs, cvect, perClassErrorTest_shfl, perClassErrorTest_chance, testTrInds_allSamps, Ytest_allSamps, Ytest_hat_allSampsFrs, trsnow_allSamps



#%% Function to run SVM to decode stimulus category using equal number of correct and incorrect trials (when X is frames x units x trials)
# Remember each numSamples will have a different set of training and testing dataset, however for each numSamples, the same set of testing/training dataset
# will be used for all frames and all values of c (unless shuffleTrs is 1, in which case different frames and c values will have different training/testing datasets.)

def setbesc_frs_stimCateg(outcomes, X,Y,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest=np.nan,fr2an=np.nan, shflTrLabs=0, nRandCorrSel=nRandCorrSel):
    # numSamples = 10; # number of iterations for finding the best c (inverse of regularization parameter)
    # if you don't want to regularize, go with a very high cbest and don't run the section below.
    # cbest = 10**6    
    # regType = 'l1'
    # kfold = 10;
    # X=X_svm; Y=Y_svm; cbest=np.nan; fr2an=np.nan; shflTrLabs=0
    import sys
    import numpy as np
    import numpy.random as rng
    from crossValidateModel import crossValidateModel
    from linearSVM import linearSVM
    
    def perClassError(Y, Yhat):
        perClassEr = np.sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
        return perClassEr
    
    ######
    X0 = X + 0
    Y0 = Y + 0
    
    
    # frames to do SVM analysis on
    if np.isnan(fr2an).all(): # run SVM on all frames
        frs = range(X.shape[0])
    else:
        frs = fr2an        
    
    
    # set range of c (regularization parameters) to check    
    if np.isnan(cbest).all(): # we need to set cbest
        bestcProvided = False        
        if regType == 'l1':
            print '\n-------------- Running l1 svm classification --------------\r' 
            # cvect = 10**(np.arange(-4, 6,0.2))/numTrials;
            cvect = 10**(np.arange(-4, 6,0.2))/numDataPoints
        elif regType == 'l2':
            print '\n-------------- Running l2 svm classification --------------\r' 
            cvect = 10**(np.arange(-6, 6,0.2))/numDataPoints          
        nCvals = len(cvect)
#        print 'try the following regularization values: \n', cvect
        # formattedList = ['%.2f' % member for member in cvect]
        # print 'try the following regularization values = \n', formattedList        
    else: # bestc is provided and we want to fit svm on shuffled trial labels
        bestcProvided = True           
        nCvals = 1 # cbest is already provided
        
    
#    smallestC = 0 # if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
    if smallestC==1:
        print 'bestc = smallest c whose cv error is less than 1se of min cv error'
    else:
        print 'bestc = c that gives min cv error'
    #I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.
        
        
    ##########################################
    nc = (outcomes==1).sum()
    ni = (outcomes==0).sum()
    
    trsCorr = np.argwhere(outcomes==1).squeeze() # indeces out of Y
    trsIncorr = np.argwhere(outcomes==0).squeeze()
    
    hr_i = (Y[trsIncorr]==1).sum()
    lr_i = (Y[trsIncorr]==0).sum()
    hr_c = (Y[trsCorr]==1).sum()
    lr_c = (Y[trsCorr]==0).sum()

    print nc, ' correct trials: ', hr_c, ' HR ', lr_c, ' LR'  
    print ni, ' incorrect trials: ', hr_i, ' HR ', lr_i, ' LR'  
       
    
    if nc!=ni:
        if nc>ni:
            if lr_c < hr_i:
                lr_tot = lr_i + lr_c
                print 'Warning: lr_c < hr_i, so we cant have equal HR and LR trials for SVM training (because we cant get hr_i num trs out of lr_c trs to combine it with lr_i)!'
                d2compens_add2_hr_c = hr_i - lr_c # bc lr_c is low, we get more of hr_c to make sure corr and incorr are equal, at the expense of more imbalance in hr and lr!
            else:       
                lr_tot = lr_i + hr_i # we get hr_i num trs out of lr_c to combine it with lr_i
                d2compens_add2_hr_c = 0
                
            if hr_c < lr_i:
                hr_tot = hr_i + hr_c
                print 'Warning: hr_c < lr_i, so we cant have equal HR and LR trials for SVM training (because we cant get lr_i num trs out of hr_c trs to combine with hr_i)!'    
                d2compens_add2_lr_c = lr_i - hr_c # bc hr_c is low, we get more of lr_c to make sure corr and incorr are equal, at the expense of more imbalance in hr and lr!
            else:
                hr_tot = hr_i + lr_i # we get lr_i num trs out of hr_c trs to combine with hr_i)
                d2compens_add2_lr_c = 0
                
                
        elif nc<ni:
            if lr_i < hr_c:
                lr_tot = lr_i + lr_c
                print 'Warning: lr_c < hr_i, so we cant have equal HR and LR trials for SVM training (because we cant get hr_i num trs out of lr_c trs to combine it with lr_i)!'
                d2compens_add2_hr_c = hr_c - lr_i # bc lr_c is low, we get more of hr_c to make sure corr and incorr are equal, at the expense of more imbalance in hr and lr!
            else:       
                lr_tot = lr_c + hr_c # we get hr_i num trs out of lr_c to combine it with lr_i
                d2compens_add2_hr_c = 0
                
            if hr_i < lr_c:
                hr_tot = hr_i + hr_c
                print 'Warning: hr_c < lr_i, so we cant have equal HR and LR trials for SVM training (because we cant get lr_i num trs out of hr_c trs to combine with hr_i)!'    
                d2compens_add2_lr_c = lr_c - hr_i # bc hr_c is low, we get more of lr_c to make sure corr and incorr are equal, at the expense of more imbalance in hr and lr!
            else:
                hr_tot = hr_c + lr_c # we get lr_i num trs out of hr_c trs to combine with hr_i)
                d2compens_add2_lr_c = 0
    
            
        lr_tot = lr_tot + d2compens_add2_lr_c
        hr_tot = hr_tot + d2compens_add2_hr_c
                
        if ~np.equal(lr_tot, hr_tot):
            print 'Warning: HR and LR tr nums are not the same:'        
        print hr_tot, 'HR', lr_tot, 'LR'
            
        
        
        if useEqualTrNums:
            if nc>ni:
                numTrials = min(lr_c, hr_i)+d2compens_add2_lr_c + lr_i    +     min(hr_c, lr_i)+d2compens_add2_hr_c + hr_i # lr_tot + hr_tot
            elif nc<ni:
                numTrials = min(lr_i, hr_c)+d2compens_add2_lr_c + lr_c    +     min(hr_i, lr_c)+d2compens_add2_hr_c + hr_c # lr_tot + hr_tot                
        else:
            numTrials = X.shape[2]


    
    #######
    if nc==ni: # number of corr and incorr trs is equal, so we don't need to subselect corr trials. Just make sure HR and LR have equal numbers.
        hrn = (Y==1).sum()
        lrn = (Y==0).sum()    
        
        if useEqualTrNums and hrn!=lrn: # if the HR and LR trials numbers are not the same, pick equal number of trials of the 2 classes!        
            trsn = min(lrn,hrn)
            if hrn > lrn:
                print 'Subselecting HR trials so both classes have the same number of trials!'
                numTrials = lrn*2
            elif lrn > hrn:
                print 'Subselecting LR trials so both classes have the same number of trials!'
                numTrials = hrn*2
            print 'FINAL: %d trials; %d neurons' %(numTrials, X.shape[1])
        else:
            numTrials = X.shape[2]

    
    len_test = numTrials - int((kfold-1.)/kfold*numTrials) # length of testing trials       
#    len_trsNowCorr = min(lr_c, hr_i) + min(hr_c, lr_i) # len(trsNowCorr)    
    len_remCorr = abs(nc - ni) #nc - len_trsNowCorr # 
    len_both = len_test+len_remCorr
    
    
    if nc > ni: 
#        nRandCorrSel = numCorrIncorrSubsamp # numSamples # number of times to randomly draw correct trials
        print '%d corr %d incorr; subselecting corr' %(ni,ni)
    elif nc==ni:
        nRandCorrSel = 1 # we don't need to subselect corr trials.
        print '%d corr %d incorr; no need to subselect corr' %(nc,ni)
    elif nc < ni:
#        nRandCorrSel = numCorrIncorrSubsamp # numSamples # number of times to randomly draw incorrect trials
        print '%d corr %d incorr; subselecting incorr' %(nc,nc)        
#        sys.exit('There are more incorrect trials, so add codes for subselecting incorrect trials (instead of subselecting correct trials)!')        
        
    print 'numRandCorrSamples = %i' %(nRandCorrSel)
    

    
    
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    nFrs = np.shape(X)[0]
    
    perClassErrorTrain = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    perClassErrorTest = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)    
    perClassErrorTest_shfl = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    perClassErrorTest_chance = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    
    perClassErrorTestRemCorr = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan) 
    perClassErrorTestRemCorr_shfl = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    perClassErrorTestRemCorr_chance = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    
    perClassErrorTestBoth = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan) 
    perClassErrorTestBoth_shfl = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    perClassErrorTestBoth_chance = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    
    bAllC = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    wAllC = np.full((numSamples, nCvals, X.shape[1], nRandCorrSel, nFrs), np.nan)
    
    Ytest_hat_allSampsFrs = np.full((numSamples, nCvals, nRandCorrSel, nFrs, len_test), np.nan)        #    testTrInds_outOfY0_allSamps = np.full((numSamples, len_test), np.nan)
    Ytest_hat_remCorr_allSampsFrs = np.full((numSamples, nCvals, nRandCorrSel, nFrs, len_remCorr), np.nan)
    Ytest_hat_both_allSampsFrs = np.full((numSamples, nCvals, nRandCorrSel, nFrs, len_both), np.nan)
    
    if nc==ni:
        trsnow_allSamps = np.full((numSamples, numTrials), np.nan)              #    eqy = np.full((X0.shape[0], numSamples), np.nan)
    else:
        trsnow_allSamps = np.full((nRandCorrSel, numTrials), np.nan) 
    trsRemCorrInds = np.full((nRandCorrSel, len_remCorr), np.nan)    
    
    Ytest_allSamps = np.full((numSamples, nRandCorrSel, len_test), np.nan)    
    testTrInds_allSamps = np.full((numSamples, nRandCorrSel, len_test), np.nan)
    trsTrainedTestedInds = np.full((numSamples, nRandCorrSel, numTrials), np.nan)
    
    
    ########################################## Randomly draw nIncorrTrNums out of correct trials ##########################################
    for sc in range(nRandCorrSel): # sc = 0
        
        if nc!=ni:
            print '===== Random correct-trial subselect %d =====' %(sc)       
            
            #######%% Subselect correct trials and combine them with all incorrect trials to set trials for svm training and testing        
            # indices are from outcomes array        
            if useEqualTrNums==0:
                trsNowCorr = trsCorr[rng.permutation(nc)[0:ni]] # n_incorr random indeces from corr trials
                trsNowAll = np.sort(np.concatenate((trsNowCorr, trsIncorr))) # set the indices for all trials to be trained
            #    print trsNowAll.shape
                print '*No* attempt to make HR and LR the same number:\n\tHR: %d, LR: %d trials' %((Y0[trsNowAll]==1).sum(), (Y0[trsNowAll]==0).sum())
                
            else:
                if nc>ni:                    
                    lrincorr = trsIncorr[Y0[trsIncorr]==0]  # these indeces are already out of Y (ie indeces to apply on Y)         
                    hrincorr = trsIncorr[Y0[trsIncorr]==1]
                
                    lrcorr = np.argwhere(Y0[trsCorr]==0).flatten()   # indeces to apply on trsCorr in order to get the indeces out of Y
                    hrcorr = np.argwhere(Y0[trsCorr]==1).flatten()                    
                    
                    # lrc_samp will be combined with lri... so if hri num is different from lri, we want to compensate for that by getting hri number of trials from lrc
                    # set lrc_samp, i.e. out of lr_corr trials, select hr_i number of trials 
                    lrc_samp = trsCorr[rng.permutation(lrcorr)[0:min(lr_c, hr_i)+d2compens_add2_lr_c]]
            
                    # hrc_samp will be combined with hri... so if hri num is different from lri, we want to compensate for that by getting lri number of trials from hrc
                    # set hrc_samp, i.e. out of hr_corr trials, select lr_i number of trials 
                    hrc_samp = trsCorr[rng.permutation(hrcorr)[0:min(hr_c, lr_i)+d2compens_add2_hr_c]]
            
                    # final: for each HR and LR, use both corr and incorr. 
                    # hr_ci and lr_ci will have the same size, because above we made sure that any difference between hri and lri gets compensated by how many we select from hrc and lrc
                    lr_ci = np.concatenate((lrc_samp, lrincorr))
                    hr_ci = np.concatenate((hrc_samp, hrincorr))                
                    
                    trsNowCorr = np.sort(np.concatenate((hrc_samp, lrc_samp))) # n_incorr indeces from corr trials (not completely random, because we chose it such that number of final hr and lr trials (corr and incorr) will be balanced)
                    trsNowAll = np.sort(np.concatenate((hr_ci, lr_ci))) # set the indices for all trials to be trained

                elif nc<ni:
                    lrcorr = trsCorr[Y0[trsCorr]==0]  # these indeces are already out of Y (ie indeces to apply on Y)         
                    hrcorr = trsCorr[Y0[trsCorr]==1]
                
                    lrincorr = np.argwhere(Y0[trsIncorr]==0).flatten()   # indeces to apply on trsCorr in order to get the indeces out of Y
                    hrincorr = np.argwhere(Y0[trsIncorr]==1).flatten()  
                    
                    # lrc_samp will be combined with lri... so if hri num is different from lri, we want to compensate for that by getting hri number of trials from lrc
                    # set lrc_samp, i.e. out of lr_corr trials, select hr_i number of trials 
                    lri_samp = trsIncorr[rng.permutation(lrincorr)[0:min(lr_i, hr_c)+d2compens_add2_lr_c]]
            
                    # hrc_samp will be combined with hri... so if hri num is different from lri, we want to compensate for that by getting lri number of trials from hrc
                    # set hrc_samp, i.e. out of hr_corr trials, select lr_i number of trials 
                    hri_samp = trsIncorr[rng.permutation(hrincorr)[0:min(hr_i, lr_c)+d2compens_add2_hr_c]]
            
                    # final: for each HR and LR, use both corr and incorr. 
                    # hr_ci and lr_ci will have the same size, because above we made sure that any difference between hri and lri gets compensated by how many we select from hrc and lrc
                    lr_ci = np.concatenate((lri_samp, lrcorr))
                    hr_ci = np.concatenate((hri_samp, hrcorr))                
                    
                    trsNowCorr = np.sort(np.concatenate((hri_samp, lri_samp))) # n_incorr indeces from corr trials (not completely random, because we chose it such that number of final hr and lr trials (corr and incorr) will be balanced)
                    trsNowAll = np.sort(np.concatenate((hr_ci, lr_ci))) # set the indices for all trials to be trained
            
            
            trsnow_allSamps[sc,:] = trsNowAll # trsnow=trsNowAll
    
                
            
            #################### Set X and Y for the random subset ####################               
            X = X0[:,:,trsNowAll]
            Y = Y0[trsNowAll]
            print 'size of X and Y: ', X.shape, Y.shape
            
            
            ####################  set X and Y for the remaining correct trials (those not in trsNowCorr). We will test on these trials the decoder trained on X and Y ####################             
            if nc>ni:
                trsRemCorr = trsCorr[~np.in1d(trsCorr, trsNowCorr)] # those correct trials (indeces in outcomes) that are not among the random pick for current trsNowCorr
            elif nc<ni:
                trsRemCorr = trsIncorr[~np.in1d(trsIncorr, trsNowCorr)] # those incorrect trials (indeces in outcomes) that are not among the random pick for current trsNowCorr
            
            trsRemCorrInds[sc,:] = trsRemCorr # (indices are from outcomes array)        
            
            X_remCorr = X0[:,:,trsRemCorr]
            Y_remCorr = Y0[trsRemCorr]
            print 'size of X_remCorr and Y_remCorr: ', X_remCorr.shape, Y_remCorr.shape
    
    
            ##### checks:
            print 'After attempting to make HR and LR the same number:\n\tHR: %d, LR: %d trials' %((Y==1).sum(), (Y==0).sum())        
            if ~np.equal((Y==1).sum(), (Y==0).sum()):
                print 'WARNING: HR and LR trial numbers are different!!'
                
            if ~np.equal((outcomes[trsNowAll]==1).sum(), (outcomes[trsNowAll]==0).sum()):
                sys.exit('Error: corr and incorr trial number are different!!')
            else:
                print 'equal corr and incorr trials:', (outcomes[trsNowAll]==1).sum(), (outcomes[trsNowAll]==0).sum()

        
        
        ########################################### Permute trials numSamples times to select various sets for training and testing SVM
        # Train SVM numSamples times to get numSamples cross-validated datasets.    
        
        for s in range(numSamples):  # s = 0      
            print 'Iteration %d' %(s)
            
            if nc==ni:
                ############ Make sure both classes have the same number of trials when training the classifier
                # set trsnow: # index of trials (out of Y0) after picking random hr (or lr) in order to make sure both classes have the same number in the final Y (on which svm was run)
                if useEqualTrNums and hrn!=lrn: # if the HR and LR trials numbers are not the same, pick equal number of trials of the 2 classes!                    
                    if hrn > lrn:
                        randtrs = np.argwhere(Y0==1)[rng.permutation(hrn)[0:trsn]].squeeze()
                        trsnow = np.sort(np.concatenate((randtrs , np.argwhere(Y0==0).squeeze()))) # index of trials after picking random hr (or lr) in order to make sure both classes have the same number in the final Y (on which svm was run)
                    elif lrn > hrn:
                        randtrs = np.argwhere(Y0==0)[rng.permutation(lrn)[0:trsn]].squeeze() # random sample of the class with more trials
                        trsnow = np.sort(np.concatenate((randtrs , np.argwhere(Y0==1).squeeze()))) # all trials of the class with fewer trials + the random sample set above for the other class
        
                    X = X0[:,:,trsnow] # trsnow : index of trials (out of X0 and Y0) that are used to set X and Y
                    Y = Y0[trsnow]
        
                else: # include all trials
                    trsnow = np.arange(0, len(Y0))
                
                trsnow_allSamps[s,:] = trsnow
                trsNowAll = trsnow       
    
    
            ######################## Set chance Y: same length as Y for testing data, and with equal number of classes 0 and 1.  
            permIxs = rng.permutation(len_test) # needed to set perClassErrorTest_shfl   
        
            Y_chance = np.zeros(len_test)
            if rng.rand()>.5:
                b = rng.permutation(len_test)[0:np.floor(len_test/float(2)).astype(int)]
            else:
                b = rng.permutation(len_test)[0:np.ceil(len_test/float(2)).astype(int)]
            Y_chance[b] = 1
    
    
            if nc!=ni:
                permIxs_remCorr = rng.permutation(X_remCorr.shape[2]);
    
                Y_chance_remCorr = np.zeros(Y_remCorr.shape[0])
                if rng.rand()>.5: # in case Y_remCorr has odd size, we make sure sometimes we round it up and sometimes down so the number of trials of the 2 classes are balanced in chance data.
                    b = rng.permutation(Y_remCorr.shape[0])[0:np.floor(Y_remCorr.shape[0]/float(2)).astype(int)]
                else:
                    b = rng.permutation(Y_remCorr.shape[0])[0:np.ceil(Y_remCorr.shape[0]/float(2)).astype(int)]
                Y_chance_remCorr[b] = 1
     


                permIxs_tot = rng.permutation(len_test+X_remCorr.shape[2])
                
                Y_chance_tot = np.zeros(permIxs_tot.shape[0])
                if rng.rand()>.5: 
                    b = rng.permutation(permIxs_tot.shape[0])[0:np.floor(permIxs_tot.shape[0]/float(2)).astype(int)]
                else:
                    b = rng.permutation(permIxs_tot.shape[0])[0:np.ceil(permIxs_tot.shape[0]/float(2)).astype(int)]
                Y_chance_tot[b] = 1    
    
    
    
            ####################### If you want to train SVM on shuffled trial labels, shuffle the Y
            if shflTrLabs: # shuffle trial classes in Y
                Y = np.zeros(numTrials) # Y_chance0
                if rng.rand()>.5:
                    b = rng.permutation(numTrials)[0:np.floor(numTrials/float(2)).astype(int)]
                else:
                    b = rng.permutation(numTrials)[0:np.ceil(numTrials/float(2)).astype(int)]
                Y[b] = 1
    
            
            ######################## Permute trial orders, so the training and testing datasets are different for each numSamples (we only do this if shuffleTrs is 0, so crossValidateModel does not shuffle trials, so we have to do it here, otherwise all numSamples will have the same set of testing and training datasets.)
            ######################## REMEMBER : YOU ARE CHANGING THE ORDER OF TRIALS HERE!!! ########################            
            if shuffleTrs==0: # if 0, shuffle trials to break any dependencies on the sequence of trails. Otherwise, do the shuffling inside crossValidateModel. I prefer to do it here to make sure all frames are trained on the same set of trials.
                
    #            Ybefshfl = Y            
                shfl = rng.permutation(np.arange(0, numTrials)) # shfl: new order of trials ... shuffled indeces of Y... the last 1/10th indeces will be testing trials.
                
                Y = Y[shfl] 
                X = X[:,:,shfl]             
                
                # Ytest_allSamps[s,:] : Y that will be used as testing trials in this sample
                Ytest_allSamps[s,sc,:] = Y[np.arange(numTrials-len_test, numTrials)] # the last 1/10th of Y (after applying shfl labels to it)
                testTrInds = shfl[np.arange(numTrials-len_test, numTrials)] # indeces to be applied on trsnow in order to get the trials (index out of Y0) that were used as testing trs; eg stimrate[trsnow[testTrInds]] is the stimrate of testing trials
    #            testTrInds_outOfY0 = trsnow[testTrInds] # index of testing trials out of Y0 (not Y!) (that will be used in svm below)
                 ######## IMPORTANT: Ybefshfl[testTrInds] is same as Y0[trsnow[testTrInds]] and same as Y[np.arange(numTrials-len_test, numTrials)] and same as summary.YTest computed below ########
    
                testTrInds_allSamps[s,sc,:] = testTrInds            
    #            testTrInds_outOfY0_allSamps[s,:] = testTrInds_outOfY0            
                # keep index of trials that were used for training, cv, and remainedCorr testing.
                trsTrainedTestedInds[s,sc,:] = trsNowAll[shfl] # first 90% training, last 10% testing (indices are from outcomes array)
            
            else:
                testTrInds_allSamps = np.nan # for now, but to set it correctly: testTrInds will be set in crossValidateModel.py, you just need to output it from crossValidateModel
                Ytest_allSamps[s,sc,:] = np.nan  
    
            
            
            #################################################################################
            ########################## Start training SVM on each frame #####################
            #################################################################################
            for ifr in frs: # ifr = 0 # train SVM on each frame
#                print '\tFrame %d' %(ifr)                  
                if bestcProvided:
                    cvect = [cbest[ifr]]
            
                ######################## Loop over different values of regularization
                for i in range(nCvals): # i = 0 # train SVM using different values of regularization parameter
                    if regType == 'l1':                               
                        summary,_ =  crossValidateModel(X[ifr,:,:].transpose(), Y, linearSVM, kfold = kfold, l1 = cvect[i], shflTrs = shuffleTrs)
                        
                    elif regType == 'l2':
                        summary,_ =  crossValidateModel(X[ifr,:,:].transpose(), Y, linearSVM, kfold = kfold, l2 = cvect[i], shflTrs = shuffleTrs)
                            
                                
                    wAllC[s,i,:,sc,ifr] = np.squeeze(summary.model.coef_); # weights of all neurons for each value of c and each shuffle
                    bAllC[s,i,sc,ifr] = np.squeeze(summary.model.intercept_);
            
                    # Classification errors                    
                    perClassErrorTrain[s,i,sc,ifr] = summary.perClassErrorTrain
                    perClassErrorTest[s,i,sc,ifr] = summary.perClassErrorTest # perClassError(YTest, linear_svm.predict(XTest));
                    
                    # Testing correct shuffled data: 
                    # same decoder trained on correct trials, but use shuffled trial labels to compute class error
                    Ytest_hat = summary.model.predict(summary.XTest) # prediction of trial label for each trial
                    perClassErrorTest_shfl[s,i,sc,ifr] = perClassError(summary.YTest[permIxs], Ytest_hat) # fraction of incorrect predicted trial labels
                    perClassErrorTest_chance[s,i,sc,ifr] = perClassError(Y_chance, Ytest_hat)
                    Ytest_hat_allSampsFrs[s,i,sc,ifr,:] = Ytest_hat
                    
                    
                    if nc!=ni:
                        # Test the decoder on remaining correct trials
                        if len(trsRemCorr)!=0:
                            Ytest_hat_remCorr = summary.model.predict(X_remCorr[ifr,:,:].transpose()) # prediction of trial label for each trial
                            perClassErrorTestRemCorr[s,i,sc,ifr] = perClassError(Y_remCorr, Ytest_hat_remCorr)
                            perClassErrorTestRemCorr_shfl[s,i,sc,ifr] = perClassError(Y_remCorr[permIxs_remCorr], Ytest_hat_remCorr)
                            perClassErrorTestRemCorr_chance[s,i,sc,ifr] = perClassError(Y_chance_remCorr, Ytest_hat_remCorr)
                            Ytest_hat_remCorr_allSampsFrs[s,i,sc,ifr,:] = Ytest_hat_remCorr
                            
                        # Pool remCorrect trials and testing trials
                        x = np.concatenate((summary.XTest, X_remCorr[ifr,:,:].transpose()), axis=0)
                        y = np.concatenate((summary.YTest, Y_remCorr))
                        Ytest_hat_both = summary.model.predict(x)
                        perClassErrorTestBoth[s,i,sc,ifr] = perClassError(y, Ytest_hat_both)
                        perClassErrorTestBoth_shfl[s,i,sc,ifr] = perClassError(y[permIxs_tot], Ytest_hat_both)
                        perClassErrorTestBoth_chance[s,i,sc,ifr] = perClassError(Y_chance_tot, Ytest_hat_both)
                        Ytest_hat_both_allSampsFrs[s,i,sc,ifr,:] = Ytest_hat_both
                    
                    
                    ########## sanity check ##########
                    """
                    trsnow = trsnow_allSamps[s].astype(int)
                    testTrInds = testTrInds_allSamps[s].astype(int)
                    testTrInds_outOfY0 = trsnow[testTrInds]
                    xx = X0[ifr][:,testTrInds_outOfY0]        
                    yy = Y0[testTrInds_outOfY0]
                    
                    ww = wAllC[s,i,:,ifr]
    #                normw = sci.linalg.norm(ww)   # numSamps x numFrames
    #                ww = ww / normw                 
                    
                    bb = bAllC[s,i,ifr] 
                    
                    # Project population activity of each frame onto the decoder of frame ifr
                    yhat = np.dot(ww, xx) + bb # testingFrs x testing trials                
                    th = 0
                    yhat[yhat<th] = 0 # testingFrs x testing trials
                    yhat[yhat>th] = 1
                                    
                    d = yhat - yy  # testing Frs x nTesting Trials # difference between actual and predicted y
                    c = np.mean(abs(d), axis=-1) * 100
    
                    eqy[ifr, s] = np.equal(c, perClassErrorTest[s,i,ifr])                
                    
                    if eqy[ifr, s]==0:
                        print np.mean(np.equal(xx.T, summary.XTest))
                        print np.mean(np.equal(yy, summary.YTest))
                        print np.mean(np.equal(yhat, Ytest_hat))
                        print ifr, s
                        print c, perClassErrorTest[s,i,ifr]
                        sys.exit('Error!') 
                    """
    

    ###########################################################################
    ############## Find bestc for each frame, and plot the c path #############
    ###########################################################################
    if bestcProvided: 
        cbestFrs = cbest
#        cbestAllFrs = cbest        
    else:
        print '--------------- Identifying best c ---------------' 
        cbestFrs = np.full((X.shape[0]), np.nan)  
#        cbestAllFrs = np.full((X.shape[0]), np.nan)  
        cbestFrs_remCorr = np.full((X.shape[0]), np.nan)  
        cbestFrs_both = np.full((X.shape[0]), np.nan)  
        
        for ifr in frs: #range(X.shape[0]):    

            ################  cbest  ################
            #######%% Compute average of class errors across numSamples and nRandCorrSel      
            meanPerClassErrorTest = np.mean(perClassErrorTest[:,:,:,ifr], axis = (0,2));
            semPerClassErrorTest = np.std(perClassErrorTest[:,:,:,ifr], axis = (0,2))/np.sqrt(numSamples);

            #######%% Identify best c                
            # Use all range of c... it may end up a value at which all weights are 0.
            ix = np.argmin(meanPerClassErrorTest)
            if smallestC==1:
                cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
                cbest = cbest[0]; # best regularization term based on minError+SE criteria
                cbestAll = cbest
            else:
                cbestAll = cvect[ix]
            print '\tFrame %d: %f' %(ifr,cbestAll)
#            cbestAllFrs[ifr] = cbestAll            
                    
            cbestFrs[ifr] = cbestAll

            
            if nc!=ni:
                ################  cbest_remCorr  ################
                #######%% Compute average of class errors across numSamples and nRandCorrSel              
                meanPerClassErrorTestRemCorr = np.mean(perClassErrorTestRemCorr[:,:,:,ifr], axis = (0,2));
                semPerClassErrorTestRemCorr = np.std(perClassErrorTestRemCorr[:,:,:,ifr], axis = (0,2))/np.sqrt(numSamples);
    
                #######%% Identify best c                
                # Use all range of c... it may end up a value at which all weights are 0.
                ix = np.argmin(meanPerClassErrorTestRemCorr)
                if smallestC==1:
                    cbest = cvect[meanPerClassErrorTestRemCorr <= (meanPerClassErrorTestRemCorr[ix]+semPerClassErrorTestRemCorr[ix])];
                    cbest = cbest[0]; # best regularization term based on minError+SE criteria
                    cbestAll = cbest
                else:
                    cbestAll = cvect[ix]
                print '\tFrame %d: %f' %(ifr,cbestAll)
    #            cbestAllFrs[ifr] = cbestAll            
    #            cbest = cbestAll
                        
                cbestFrs_remCorr[ifr] = cbestAll
    
    
                ################  cbest_both ################
                #######%% Compute average of class errors across numSamples and nRandCorrSel              
                meanPerClassErrorTestBoth = np.mean(perClassErrorTestBoth[:,:,:,ifr], axis = (0,2));
                semPerClassErrorTestBoth = np.std(perClassErrorTestBoth[:,:,:,ifr], axis = (0,2))/np.sqrt(numSamples);
    
                #######%% Identify best c                
                # Use all range of c... it may end up a value at which all weights are 0.
                ix = np.argmin(meanPerClassErrorTestBoth)
                if smallestC==1:
                    cbest = cvect[meanPerClassErrorTestBoth <= (meanPerClassErrorTestBoth[ix]+semPerClassErrorTestBoth[ix])];
                    cbest = cbest[0]; # best regularization term based on minError+SE criteria
                    cbestAll = cbest
                else:
                    cbestAll = cvect[ix]
                print '\tFrame %d: %f' %(ifr,cbestAll)
    #            cbestAllFrs[ifr] = cbestAll            
    #            cbest = cbestAll
                        
                cbestFrs_both[ifr] = cbestAll
                
                
            if nc==ni:
                cbestFrs_both = cbestFrs
                
            
            ########%% Set the decoder and class errors at best c (for data)
            """
            # you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
            # we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
            indBestC = np.in1d(cvect, cbest)
            
            w_bestc_data = wAllC[:,indBestC,:,ifr].squeeze() # numSamps x neurons
            b_bestc_data = bAllC[:,indBestC,ifr]
            
            classErr_bestC_train_data = perClassErrorTrain[:,indBestC,ifr].squeeze()
            
            classErr_bestC_test_data = perClassErrorTest[:,indBestC,ifr].squeeze()
            classErr_bestC_test_shfl = perClassErrorTest_shfl[:,indBestC,ifr].squeeze()
            classErr_bestC_test_chance = perClassErrorTest_chance[:,indBestC,ifr].squeeze()
            """
            
            ########### Plot C path    
            if doPlots:
                
                #######%% Compute average of class errors across numSamples        
                meanPerClassErrorTrain = np.mean(perClassErrorTrain[:,:,:,ifr], axis = (0,2));
                semPerClassErrorTrain = np.std(perClassErrorTrain[:,:,:,ifr], axis = (0,2))/np.sqrt(numSamples);            
                meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl[:,:,:,ifr], axis = (0,2));
#                semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl[:,:,:,ifr], axis = 0)/np.sqrt(numSamples);            
#                meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance[:,:,:,ifr], axis = (0,2));
#                semPerClassErrorTest_chance = np.std(perClassErrorTest_chance[:,:,:,ifr], axis = 0)/np.sqrt(numSamples);

#                print 'Best c (inverse of regularization parameter) = %.2f' %cbest
                plt.figure(figsize=(3,3))
#                plt.subplot(1,2,1)
                plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
                plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
                plt.fill_between(cvect, meanPerClassErrorTestRemCorr-semPerClassErrorTestRemCorr, meanPerClassErrorTestRemCorr+ semPerClassErrorTestRemCorr, alpha=0.5, edgecolor='g', facecolor='g')        
                plt.fill_between(cvect, meanPerClassErrorTestBoth-semPerClassErrorTestBoth, meanPerClassErrorTestBoth+ semPerClassErrorTestBoth, alpha=0.5, edgecolor='m', facecolor='m')        
            #    plt.fill_between(cvect, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
            #    plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
                
                plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
                plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'testing')
#                plt.plot(cvect, meanPerClassErrorTest_chance, 'b', label = 'cv-chance')       
                plt.plot(cvect, meanPerClassErrorTest_shfl, 'y', label = 'shfl')    
                plt.plot(cvect, meanPerClassErrorTestRemCorr, 'g', label = 'rem corr')        
                plt.plot(cvect, meanPerClassErrorTestBoth, 'm', label = 'test+remCorr')                   
            
                plt.plot(cvect[cvect==cbestFrs[ifr]], meanPerClassErrorTest[cvect==cbestFrs[ifr]], 'o', markerfacecolor='r', markeredgecolor='r')
                plt.plot(cvect[cvect==cbestFrs_remCorr[ifr]], meanPerClassErrorTestRemCorr[cvect==cbestFrs_remCorr[ifr]], 'o', markerfacecolor='g', markeredgecolor='g')
                plt.plot(cvect[cvect==cbestFrs_both[ifr]], meanPerClassErrorTestBoth[cvect==cbestFrs_both[ifr]], 'o', markerfacecolor='m', markeredgecolor='m')
                
                plt.xlim([cvect[1], cvect[-1]])
                plt.xscale('log')
                plt.xlabel('c (inverse of regularization parameter)')
                plt.ylabel('classification error (%)')
                plt.legend(loc='center left', bbox_to_anchor=(1, .7))                
                plt.title('Frame %d' %(ifr))
                plt.tight_layout()
    
    

    ##############
    return nRandCorrSel, wAllC, bAllC, cbestFrs, cbestFrs_remCorr, cbestFrs_both, cvect, \
    perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance, \
    perClassErrorTestRemCorr, perClassErrorTestRemCorr_shfl, perClassErrorTestRemCorr_chance, \
    perClassErrorTestBoth, perClassErrorTestBoth_shfl, perClassErrorTestBoth_chance, \
    trsnow_allSamps, testTrInds_allSamps, trsTrainedTestedInds, trsRemCorrInds, \
    Ytest_allSamps, Ytest_hat_allSampsFrs, Ytest_hat_remCorr_allSampsFrs, Ytest_hat_both_allSampsFrs



#%% Function to run SVM to decode stimulus category; For each stimulus category (HR, LR) we use equal number of correct and incorrect trials (when X is frames x units x trials)
# Remember each numSamples will have a different set of training and testing dataset, however for each numSamples, the same set of testing/training dataset
# will be used for all frames and all values of c (unless shuffleTrs is 1, in which case different frames and c values will have different training/testing datasets.)

def setbesc_frs_stimCateg_new(outcomes, X,Y,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest=np.nan,fr2an=np.nan, shflTrLabs=0, nRandCorrSel=nRandCorrSel):
    
    # Note: I commented out all parts related to remCorr in this function: we cant use remaining corr when decoding outcome, bc it includes only trials for one of the classes (corr), so remCorr_shfl will be high too... if you want to do this correctly you have use all remaining (corr and incorr) trials... but it takes a lot more memory and time to run the jobs, and having remCorr doesnt seem to add much!    

    '''
    X = X_svm
    Y = Y_svm
    cbest = np.nan
    fr2an = np.nan
    shflTrLabs = 0
    '''
    # numSamples = 10; # number of iterations for finding the best c (inverse of regularization parameter)
    # if you don't want to regularize, go with a very high cbest and don't run the section below.
    # cbest = 10**6    
    # regType = 'l1'
    # kfold = 10;
    
    import sys            
    import numpy as np
    import numpy.random as rng
    from crossValidateModel import crossValidateModel
    from linearSVM import linearSVM
    
    def perClassError(Y, Yhat):
        perClassEr = np.sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
        return perClassEr
   
#    if type(X_svm_incorr)==int:
#        testIncorr = 0
#    else:
#        testIncorr = 1
#        
#    if testIncorr:
#        len_test_incorr = len(Y_svm_incorr)
        
    # frames to do SVM analysis on
    if np.isnan(fr2an).all(): # run SVM on all frames
        frs = range(X.shape[0])
    else:
        frs = fr2an        
    
    
    # set range of c (regularization parameters) to check    
    if np.isnan(cbest).all(): # we need to set cbest
        bestcProvided = False        
        if regType == 'l1':
            print '\n-------------- Running l1 svm classification --------------\r' 
            # cvect = 10**(np.arange(-4, 6,0.2))/numTrials;
            cvect = 10**(np.arange(-4, 6,0.2))/numDataPoints
        elif regType == 'l2':
            print '\n-------------- Running l2 svm classification --------------\r' 
            cvect = 10**(np.arange(-6, 6,0.2))/numDataPoints          
        nCvals = len(cvect)
#        print 'try the following regularization values: \n', cvect
        # formattedList = ['%.2f' % member for member in cvect]
        # print 'try the following regularization values = \n', formattedList        
    else: # bestc is provided and we want to fit svm on shuffled trial labels
        bestcProvided = True           
        nCvals = 1 # cbest is already provided
        
    
#    smallestC = 0 # if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
    if smallestC==1:
        print 'bestc = smallest c whose cv error is less than 1se of min cv error'
    else:
        print 'bestc = c that gives min cv error'
    #I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.
        
        
    
    ##########################################
    # if decodeStimCateg, the vars below will show hr and lr stimulus trials (not corr and incorr)!
    nhr = (Y==1).sum() # later we change the value of nhr to the same value as numTrials_hr (ie equal corr and incorr)
    nlr = (Y==0).sum() # later we change the value of nlr to the same value as numTrials_lr (ie equal corr and incorr)
    
    nhr0 = nhr; nlr0 = nlr;
    
    # if decodeStimCateg, the vars below will show hr and lr stimulus trials (not corr and incorr)!
    trsHR = np.argwhere(Y==1).squeeze() # indeces out of Y
    trsLR = np.argwhere(Y==0).squeeze()
  
    hr_c = (outcomes[trsHR]==1).sum()
    hr_i = (outcomes[trsHR]==0).sum()    
    lr_c = (outcomes[trsLR]==1).sum()
    lr_i = (outcomes[trsLR]==0).sum()
     
    print nhr, ' HR trials: ', hr_c, ' corr ', hr_i, ' incorr'  
    print nlr, ' LR trials: ', lr_c, ' corr ', lr_i, ' incorr'
    
    
    if min([hr_c, lr_c, hr_i, lr_i]) < 3: # 3 of each will give 12 trials total, the minimum we want to train the SVM, so at least we get 1 testing trials (with 10 folds)!
        sys.exit('Too few trials to do SVM training!')
        
    
    #####################        
    # Set numTrials in the decoder so you can set len_test
    #####################
    
    # we want to have equal number of each outcome for each stimulus categ
    ##### HR
    trsn_hr = min(hr_i,hr_c)
    numTrials_hr = trsn_hr*2
        
   
    ##### LR
    trsn_lr = min(lr_i,lr_c)
    numTrials_lr = trsn_lr*2
            
    
    if numTrials_hr == numTrials_lr:
       useEqualTrNums = 0 # HR and LR are already equal! (there is a chance that the original nhr and nlr are not equal but they become equal after HR/LR subselection, we will still set useEqualTrNums to 0, so there wont be a test on remCorr trials.)
    
       
    ###### NOTE: if useEqualTrNums=1, it is either because we don't care about having equal number of corr and incorr (the 2 classes) or because the 2 classes already have equal number of trials, so we dont need to make them equal, or run analysis on remCorr trials.

       
    ####### Set numTrials
    if useEqualTrNums: # make sure both classes (HR and LR) have the same number of trials
        numTrials = 2 * min(numTrials_lr , numTrials_hr)
        len_remCorr = 0
        # Note: I am commenting below bc for decoding outcome, we cant use remaining corr, bc it includes only trials for one of the classes (corr), so remCorr_shfl will be high too... if you want to do this correctly you have use all remaining (corr and incorr) trials... but it takes a lot more memory and time to run the jobs, and having remCorr doesnt seem to add much!
        '''
        if numTrials_lr < numTrials_hr:
            len_remCorr = len(trsHR) - min(numTrials_lr , numTrials_hr) #nhr - len_trsNowCorr # 
        else:
            len_remCorr = len(trsLR) - min(numTrials_lr , numTrials_hr) #nhr - len_trsNowCorr # 
        '''
    else:        
        numTrials = numTrials_hr + numTrials_lr
        len_remCorr = 0
    
    
    len_test = numTrials - int((kfold-1.)/kfold*numTrials) # length of testing trials   
    
    len_both = len_test+len_remCorr
    
    
    #####################        
    #####################        
    
    X0 = X + 0
    Y0 = Y + 0
    
    
    print 'numRandCorrSamples = %i' %(nRandCorrSel)
        
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    nFrs = np.shape(X)[0]
    
    perClassErrorTrain = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    perClassErrorTest = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)    
    perClassErrorTest_shfl = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    perClassErrorTest_chance = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    
    # Note: I am commenting below bc for decoding outcome, we cant use remaining corr, bc it includes only trials for one of the classes (corr), so remCorr_shfl will be high too... if you want to do this correctly you have use all remaining (corr and incorr) trials... but it takes a lot more memory and time to run the jobs, and having remCorr doesnt seem to add much!
    perClassErrorTestRemCorr = np.nan #np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan) 
    perClassErrorTestRemCorr_shfl = np.nan #np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    perClassErrorTestRemCorr_chance = np.nan #np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    
    perClassErrorTestBoth = np.nan #np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan) 
    perClassErrorTestBoth_shfl = np.nan #np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    perClassErrorTestBoth_chance = np.nan #np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    
    bAllC = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    wAllC = np.full((numSamples, nCvals, X.shape[1], nRandCorrSel, nFrs), np.nan)
    
    Ytest_hat_allSampsFrs = np.full((numSamples, nCvals, nRandCorrSel, nFrs, len_test), np.nan)        #    testTrInds_outOfY0_allSamps = np.full((numSamples, len_test), np.nan)
    Ytest_hat_remCorr_allSampsFrs = np.nan #np.full((numSamples, nCvals, nRandCorrSel, nFrs, len_remCorr), np.nan)
    Ytest_hat_both_allSampsFrs = np.nan #np.full((numSamples, nCvals, nRandCorrSel, nFrs, len_both), np.nan)
    
    trsnow_allSamps = np.full((nRandCorrSel, numTrials), np.nan) 
    trsRemCorrInds = np.nan #np.full((nRandCorrSel, len_remCorr), np.nan)    
    
    Ytest_allSamps = np.full((numSamples, nRandCorrSel, len_test), np.nan)    
    testTrInds_allSamps = np.full((numSamples, nRandCorrSel, len_test), np.nan)
    trsTrainedTestedInds = np.full((numSamples, nRandCorrSel, numTrials), np.nan)
            
        
        
    
    ########################################## Make sure HR and LR have the same trial number within each class (corr and incorr), also if useEqualTrNums=1, make sure corr and incorr (ie the 2 classes in SVM) have the same trial number ##########################################
    for sc in range(nRandCorrSel): # sc = 0
        
        print '===== Random correct-trial subselect %d =====' %(sc)       
        
        ####################################
        # Make sure each stimulus category (HR, LR) has the same number of corr and incorr trials when training the classifier
        ####################################
        
        # HR stim; set trsnow_hr: index of trials on Y, HR, half corr, half incorr
        if hr_c != hr_i:
            if hr_c > hr_i: # subselect corr trials ... for HR trials, take all incorr trials and subselect corr trials.
                randtrs = np.argwhere(np.logical_and(Y0==1 , outcomes==1))[rng.permutation(hr_c)[0:trsn_hr]].squeeze() # subselect of HR correct trials.
                trsnow_hr = np.sort(np.concatenate((randtrs , np.argwhere(np.logical_and(Y0==1 , outcomes==0)).squeeze()))) # index of trials after picking all HR incorrect trials and a subselect of HR correct trials, to make sure within HR, corr and incorr have the same number in the final Y (on which svm was run)
            elif hr_c < hr_i: # subselect incorr trials ... for HR trials, take all corr trials and subselect incorr trials.
                randtrs = np.argwhere(np.logical_and(Y0==1 , outcomes==0))[rng.permutation(hr_i)[0:trsn_hr]].squeeze() # subselect of HR incorrect trials.
                trsnow_hr = np.sort(np.concatenate((randtrs , np.argwhere(np.logical_and(Y0==1 , outcomes==1)).squeeze()))) # index of trials after picking all HR correct trials and a subselect of HR incorrect trials, to make sure within HR, corr and incorr have the same number in the final Y (on which svm was run)
            nhr = len(trsnow_hr)
        else: # include all HR trials
            trsnow_hr = np.argwhere(Y0==1).squeeze() #np.arange(0, nhr)

        
        # LR stim; set trsnow_lr: index of trials on Y, LR, half corr, half incorr
        if lr_c != lr_i:
            if lr_c > lr_i: # subselect corr trials ... for LR trials, take all incorr trials and subselect corr trials.
                randtrs = np.argwhere(np.logical_and(Y0==0 , outcomes==1))[rng.permutation(lr_c)[0:trsn_lr]].squeeze() # subselect of LR correct trials.
                trsnow_lr = np.sort(np.concatenate((randtrs , np.argwhere(np.logical_and(Y0==0 , outcomes==0)).squeeze()))) # index of trials after picking all LR incorrect trials and a subselect of LR correct trials, to make sure within LR, corr and incorr have the same number in the final Y (on which svm was run)
            elif lr_c < lr_i: # subselect incorr trials ... for LR trials, take all corr trials and subselect incorr trials.
                randtrs = np.argwhere(np.logical_and(Y0==0 , outcomes==0))[rng.permutation(lr_i)[0:trsn_lr]].squeeze() # subselect of LR incorrect trials.
                trsnow_lr = np.sort(np.concatenate((randtrs , np.argwhere(np.logical_and(Y0==0 , outcomes==1)).squeeze()))) # index of trials after picking all LR correct trials and a subselect of LR incorrect trials, to make sure within LR, corr and incorr have the same number in the final Y (on which svm was run)
            nlr = len(trsnow_lr)
        else: # include all LR trials
            trsnow_lr = np.argwhere(Y0==0).squeeze() #np.arange(0, nlr)
           
        
        ####################################        
        # make sure HR and LR have the same number of trials. We have already made sure that within each stim categ, corr and incorr have the same trial number. (nhr and nlr are updated already) 
        ####################################
        
        if useEqualTrNums: # we do this so both classes (HR and LR) have the same number of trials in the training decoder, otherwise decoder performance even on shuffle trial labels will deviate from chance.
            if nhr > nlr: # subselect HR trials to the same number as LR trials                
                # we need to subselect nlr/2 trials from HR_corr and HR_incorr
                ntrnow = nlr/2
                # HR; set trsnow_hh: index of trials on Y, HR, half corr, half incorr. Each has nlr/2 trials.
                randtrs_c = np.argwhere(np.logical_and(Y0==1 , outcomes==1))[rng.permutation(hr_c)[0:ntrnow]].squeeze() # subselect of HR correct trials to nlr/2 trials
                randtrs_i = np.argwhere(np.logical_and(Y0==1 , outcomes==0))[rng.permutation(hr_i)[0:ntrnow]].squeeze() # subselect of LR correct trials to nlr/2 trials
                    
                trsnow_hh = np.sort(np.concatenate((randtrs_c , randtrs_i))) # index of trials after picking a subselect of LR correct trials (nlr/2) and a subselect of HR correct trials (nlr/2), to make sure within correct, LR and HR have the same number in the final Y (on which svm was run), also correct and incorrect trials have the same number of trials.
                    
                # set the final trsNowAll. trsnow_lr has half corr and half incorr (we already did it in the section above)
                trsNowAll = np.sort(np.concatenate((trsnow_hh, trsnow_lr))) # this includes half HR, half LR; also half corr and half incorr within each stim categ.

            elif nhr < nlr: # subselect LR trials to the same number as HR trials                
                # we need to subselect nhr/2 trials from LR_corr and LR_incorr
                ntrnow = nhr/2
                # LR; set trsnow_ll: index of trials on Y, LR, half corr, half incorr. Each has nhr/2 trials.
                randtrs_c = np.argwhere(np.logical_and(Y0==0 , outcomes==1))[rng.permutation(lr_c)[0:ntrnow]].squeeze() # subselect of HR correct trials to nlr/2 trials
                randtrs_i = np.argwhere(np.logical_and(Y0==0 , outcomes==0))[rng.permutation(lr_i)[0:ntrnow]].squeeze() # subselect of LR correct trials to nlr/2 trials
                    
                trsnow_ll = np.sort(np.concatenate((randtrs_c , randtrs_i))) # index of trials after picking a subselect of LR correct trials (nlr/2) and a subselect of HR correct trials (nlr/2), to make sure within correct, LR and HR have the same number in the final Y (on which svm was run), also correct and incorrect trials have the same number of trials.
                    
                # set the final trsNowAll. trsnow_hr has half corr and half incorr (we already did it in the section above)
                trsNowAll = np.sort(np.concatenate((trsnow_ll, trsnow_hr))) # this includes half HR, half LR; also half corr and half incorr within each stim categ.

        else: # we only make sure within each HR and LR class, there are half corr and half incorr trials. We don't care about HR and LR being equal. This is the minimum we want to have to make sure we can decode outcome correctly (ie within each stim categ, there must be equal corr and incorr trials).
            trsNowAll = np.sort(np.concatenate((trsnow_hr, trsnow_lr))) # this includes half corr and half incorr within each HR and LR subsets. but HR and LR are not necessarily the same number        
        
        trsnow_allSamps[sc,:] = trsNowAll
        
        
        
        ####################################
        # Now set X and Y only for trsNowAll trials (half corr, half incorr, half HR, half LR)
        ####################################
        
        X = X0[:,:,trsNowAll] # trsNowAll : index of trials (out of X0 and Y0) that are used to set X and Y
        Y = Y0[trsNowAll]
        
        print 'size of X and Y: ', X.shape, Y.shape
        
        outcomesn = outcomes[trsNowAll]
        
        
        ####################################
        # Set X and Y for the remaining correct trials (those not in trsNowCorr). We will test on these trials the decoder trained on X and Y ####################             
        ####################################
        
        # Note: I am commenting below bc for decoding outcome, we cant use remaining corr, bc it includes only trials for one of the classes (corr), so remCorr_shfl will be high too... if you want to do this correctly you have use all remaining (corr and incorr) trials... but it takes a lot more memory and time to run the jobs, and having remCorr doesnt seem to add much!
        if 0: #useEqualTrNums:
            if nhr > nlr:
                trsRemCorr = trsHR[~np.in1d(trsHR, trsnow_hh)] # those correct trials (indeces in outcomes) that are not among the random pick for current trsnow_hh
            elif nhr < nlr:
                trsRemCorr = trsLR[~np.in1d(trsLR, trsnow_ll)] # those incorrect trials (indeces in outcomes) that are not among the random pick for current trsnow_ll
            
            trsRemCorrInds[sc,:] = trsRemCorr # (indices are from outcomes array)        
            
            X_remCorr = X0[:,:,trsRemCorr]
            Y_remCorr = Y0[trsRemCorr]
            print 'size of X_remCorr and Y_remCorr: ', X_remCorr.shape, Y_remCorr.shape
            
    
    
        ####################################
        # Checks
        ####################################
        hrcn = sum(np.logical_and(Y==1 , outcomesn==1))
        hrin = sum(np.logical_and(Y==1 , outcomesn==0))            
        lrcn = sum(np.logical_and(Y==0 , outcomesn==1))
        lrin = sum(np.logical_and(Y==0 , outcomesn==0))

        print 'After attempting to make corr and incorr the same number:\n\tHR_corr: %d, HR_incorr: %d trials\n\tLR_corr: %d, LR_incorr: %d trials' %(hrcn, hrin, lrcn, lrin)
        
        if useEqualTrNums:
            print 'After attempting to make HR and LR the same number:\n\tHR: %d, LR: %d trials' %((Y==1).sum(), (Y==0).sum())        

        if ~np.equal(hrcn , hrin):
            sys.exit('Error: among HR, corr and incorr trial number are different!!')

        if ~np.equal(lrcn , lrin):
            sys.exit('Error: among LR, corr and incorr trial number are different!!')
            
        if useEqualTrNums:
            if ~np.equal(hrcn+hrin , lrcn+lrin):
                sys.exit('Error: HR and LR trial number are different!!')
                
                
                
                
        ########################################### Permute trials numSamples times to select various sets for training and testing SVM
        # Train SVM numSamples times to get numSamples cross-validated datasets.    
        
        for s in range(numSamples):  # s = 0      
            print 'Iteration %d' %(s)                
            
            ####################################
            # Set chance Y: same length as Y for testing data, and with equal number of classes 0 and 1.  
            ####################################
            permIxs = rng.permutation(len_test) # needed to set perClassErrorTest_shfl   
        
            Y_chance = np.zeros(len_test)
            if rng.rand()>.5:
                b = rng.permutation(len_test)[0:np.floor(len_test/float(2)).astype(int)]
            else:
                b = rng.permutation(len_test)[0:np.ceil(len_test/float(2)).astype(int)]
            Y_chance[b] = 1
    
            
            # Note: I am commenting below bc for decoding outcome, we cant use remaining corr, bc it includes only trials for one of the classes (corr), so remCorr_shfl will be high too... if you want to do this correctly you have use all remaining (corr and incorr) trials... but it takes a lot more memory and time to run the jobs, and having remCorr doesnt seem to add much!
            if 0: #nhr!=nlr: #useEqualTrNums:
                permIxs_remCorr = rng.permutation(X_remCorr.shape[2]);
    
                Y_chance_remCorr = np.zeros(Y_remCorr.shape[0])
                if rng.rand()>.5: # in case Y_remCorr has odd size, we make sure sometimes we round it up and sometimes down so the number of trials of the 2 classes are balanced in chance data.
                    b = rng.permutation(Y_remCorr.shape[0])[0:np.floor(Y_remCorr.shape[0]/float(2)).astype(int)]
                else:
                    b = rng.permutation(Y_remCorr.shape[0])[0:np.ceil(Y_remCorr.shape[0]/float(2)).astype(int)]
                Y_chance_remCorr[b] = 1
     


                permIxs_tot = rng.permutation(len_test+X_remCorr.shape[2])
                
                Y_chance_tot = np.zeros(permIxs_tot.shape[0])
                if rng.rand()>.5: 
                    b = rng.permutation(permIxs_tot.shape[0])[0:np.floor(permIxs_tot.shape[0]/float(2)).astype(int)]
                else:
                    b = rng.permutation(permIxs_tot.shape[0])[0:np.ceil(permIxs_tot.shape[0]/float(2)).astype(int)]
                Y_chance_tot[b] = 1    
    
            
            ####################################
            # If you want to train SVM on shuffled trial labels, shuffle the Y
            ####################################
            if shflTrLabs: # shuffle trial classes in Y
                Y = np.zeros(numTrials) # Y_chance0
                if rng.rand()>.5:
                    b = rng.permutation(numTrials)[0:np.floor(numTrials/float(2)).astype(int)]
                else:
                    b = rng.permutation(numTrials)[0:np.ceil(numTrials/float(2)).astype(int)]
                Y[b] = 1
    
            
            ####################################
            ######################## Permute trial orders, so the training and testing datasets are different for each numSamples (we only do this if shuffleTrs is 0, so crossValidateModel does not shuffle trials, so we have to do it here, otherwise all numSamples will have the same set of testing and training datasets.)
            ######################## REMEMBER : YOU ARE CHANGING THE ORDER OF TRIALS HERE!!! ########################
            ####################################
            if shuffleTrs==0: # if 0, shuffle trials to break any dependencies on the sequence of trails. Otherwise, do the shuffling inside crossValidateModel. I prefer to do it here to make sure all frames are trained on the same set of trials.
                
    #            Ybefshfl = Y            
                shfl = rng.permutation(np.arange(0, numTrials)) # shfl: new order of trials ... shuffled indeces of Y... the last 1/10th indeces will be testing trials.
                
                Y = Y[shfl] 
                X = X[:,:,shfl]             
                
                # Ytest_allSamps[s,:] : Y that will be used as testing trials in this sample
                Ytest_allSamps[s,sc,:] = Y[np.arange(numTrials-len_test, numTrials)] # the last 1/10th of Y (after applying shfl labels to it)
                testTrInds = shfl[np.arange(numTrials-len_test, numTrials)] # indeces to be applied on trsnow in order to get the trials (index out of Y0) that were used as testing trs; eg stimrate[trsnow[testTrInds]] is the stimrate of testing trials
    #            testTrInds_outOfY0 = trsnow[testTrInds] # index of testing trials out of Y0 (not Y!) (that will be used in svm below)
                 ######## IMPORTANT: Ybefshfl[testTrInds] is same as Y0[trsnow[testTrInds]] and same as Y[np.arange(numTrials-len_test, numTrials)] and same as summary.YTest computed below ########
    
                testTrInds_allSamps[s,sc,:] = testTrInds            
    #            testTrInds_outOfY0_allSamps[s,:] = testTrInds_outOfY0            
                # keep index of trials that were used for training, cv, and remainedCorr testing.
                trsTrainedTestedInds[s,sc,:] = trsNowAll[shfl] # first 90% training, last 10% testing (indices are from outcomes array)
            
            else:
                testTrInds_allSamps = np.nan # for now, but to set it correctly: testTrInds will be set in crossValidateModel.py, you just need to output it from crossValidateModel
                Ytest_allSamps[s,sc,:] = np.nan  
    
 
            ##################################################################################################################################################################
            ########################## Start training SVM on each frame ######################################################################################################
            ##################################################################################################################################################################
            for ifr in frs: # ifr = 0 # train SVM on each frame
#                print '\tFrame %d' %(ifr)
                if bestcProvided:
                    cvect = [cbest[ifr]]
            
                ######################## Loop over different values of regularization
                for i in range(nCvals): # i = 0 # train SVM using different values of regularization parameter
                    if regType == 'l1':                               
                        summary,_ =  crossValidateModel(X[ifr,:,:].transpose(), Y, linearSVM, kfold = kfold, l1 = cvect[i], shflTrs = shuffleTrs)
                        
                    elif regType == 'l2':
                        summary,_ =  crossValidateModel(X[ifr,:,:].transpose(), Y, linearSVM, kfold = kfold, l2 = cvect[i], shflTrs = shuffleTrs)
                            
                                
                    wAllC[s,i,:,sc,ifr] = np.squeeze(summary.model.coef_); # weights of all neurons for each value of c and each shuffle
                    bAllC[s,i,sc,ifr] = np.squeeze(summary.model.intercept_);
            
                    # Classification errors                    
                    perClassErrorTrain[s,i,sc,ifr] = summary.perClassErrorTrain
                    perClassErrorTest[s,i,sc,ifr] = summary.perClassErrorTest # perClassError(YTest, linear_svm.predict(XTest));
                    
                    # Testing correct shuffled data: 
                    # same decoder trained on correct trials, but use shuffled trial labels to compute class error
                    Ytest_hat = summary.model.predict(summary.XTest) # prediction of trial label for each trial
                    perClassErrorTest_shfl[s,i,sc,ifr] = perClassError(summary.YTest[permIxs], Ytest_hat) # fraction of incorrect predicted trial labels
                    perClassErrorTest_chance[s,i,sc,ifr] = perClassError(Y_chance, Ytest_hat)
                    Ytest_hat_allSampsFrs[s,i,sc,ifr,:] = Ytest_hat
                    
                    
                    # Note: I am commenting below bc for decoding outcome, we cant use remaining corr, bc it includes only trials for one of the classes (corr), so remCorr_shfl will be high too... if you want to do this correctly you have use all remaining (corr and incorr) trials... but it takes a lot more memory and time to run the jobs, and having remCorr doesnt seem to add much!
                    if 0: #nhr!=nlr: # useEqualTrNums: # 
                        # Test the decoder on remaining correct trials
                        if len(trsRemCorr)!=0:
                            Ytest_hat_remCorr = summary.model.predict(X_remCorr[ifr,:,:].transpose()) # prediction of trial label for each trial
                            perClassErrorTestRemCorr[s,i,sc,ifr] = perClassError(Y_remCorr, Ytest_hat_remCorr)
                            perClassErrorTestRemCorr_shfl[s,i,sc,ifr] = perClassError(Y_remCorr[permIxs_remCorr], Ytest_hat_remCorr)
                            perClassErrorTestRemCorr_chance[s,i,sc,ifr] = perClassError(Y_chance_remCorr, Ytest_hat_remCorr)
                            Ytest_hat_remCorr_allSampsFrs[s,i,sc,ifr,:] = Ytest_hat_remCorr
                            
                        # Pool remCorrect trials and testing trials
                        x = np.concatenate((summary.XTest, X_remCorr[ifr,:,:].transpose()), axis=0)
                        y = np.concatenate((summary.YTest, Y_remCorr))
                        Ytest_hat_both = summary.model.predict(x)
                        perClassErrorTestBoth[s,i,sc,ifr] = perClassError(y, Ytest_hat_both)
                        perClassErrorTestBoth_shfl[s,i,sc,ifr] = perClassError(y[permIxs_tot], Ytest_hat_both)
                        perClassErrorTestBoth_chance[s,i,sc,ifr] = perClassError(Y_chance_tot, Ytest_hat_both)
                        Ytest_hat_both_allSampsFrs[s,i,sc,ifr,:] = Ytest_hat_both
                    


    ###########################################################################
    ############## Find bestc for each frame, and plot the c path #############
    ###########################################################################
    if bestcProvided: 
        cbestFrs = cbest
#        cbestAllFrs = cbest        
    else:
        print '--------------- Identifying best c ---------------' 
        cbestFrs = np.full((X.shape[0]), np.nan)  
#        cbestAllFrs = np.full((X.shape[0]), np.nan)  
        cbestFrs_remCorr = np.full((X.shape[0]), np.nan)  
        cbestFrs_both = np.full((X.shape[0]), np.nan)  
        
        for ifr in frs: #range(X.shape[0]):    

            ################  cbest  ################
            #######%% Compute average of class errors across numSamples and nRandCorrSel      
            meanPerClassErrorTest = np.mean(perClassErrorTest[:,:,:,ifr], axis = (0,2));
            semPerClassErrorTest = np.std(perClassErrorTest[:,:,:,ifr], axis = (0,2))/np.sqrt(numSamples);

            #######%% Identify best c                
            # Use all range of c... it may end up a value at which all weights are 0.
            ix = np.argmin(meanPerClassErrorTest)
            if smallestC==1:
                cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
                cbest = cbest[0]; # best regularization term based on minError+SE criteria
                cbestAll = cbest
            else:
                cbestAll = cvect[ix]
            print '\tFrame %d: %f' %(ifr,cbestAll)
#            cbestAllFrs[ifr] = cbestAll            
                    
            cbestFrs[ifr] = cbestAll

                
            # Note: I am commenting below bc for decoding outcome, we cant use remaining corr, bc it includes only trials for one of the classes (corr), so remCorr_shfl will be high too... if you want to do this correctly you have use all remaining (corr and incorr) trials... but it takes a lot more memory and time to run the jobs, and having remCorr doesnt seem to add much!
            if 0: #nhr!=nlr:
                #######%% Compute average of class errors across numSamples and nRandCorrSel              
                meanPerClassErrorTestRemCorr = np.mean(perClassErrorTestRemCorr[:,:,:,ifr], axis = (0,2))
                semPerClassErrorTestRemCorr = np.std(perClassErrorTestRemCorr[:,:,:,ifr], axis = (0,2))/np.sqrt(numSamples)
                    
                #######%% Compute average of class errors across numSamples and nRandCorrSel              
                meanPerClassErrorTestBoth = np.mean(perClassErrorTestBoth[:,:,:,ifr], axis = (0,2))
                semPerClassErrorTestBoth = np.std(perClassErrorTestBoth[:,:,:,ifr], axis = (0,2))/np.sqrt(numSamples)
                
                ################  cbest_remCorr  ################    
                #######%% Identify best c                
                # Use all range of c... it may end up a value at which all weights are 0.
                ix = np.argmin(meanPerClassErrorTestRemCorr)
                if smallestC==1:
                    cbest = cvect[meanPerClassErrorTestRemCorr <= (meanPerClassErrorTestRemCorr[ix]+semPerClassErrorTestRemCorr[ix])];
                    cbest = cbest[0]; # best regularization term based on minError+SE criteria
                    cbestAll = cbest
                else:
                    cbestAll = cvect[ix]
                print '\tFrame %d: %f' %(ifr,cbestAll)
    #            cbestAllFrs[ifr] = cbestAll            
    #            cbest = cbestAll
                        
                cbestFrs_remCorr[ifr] = cbestAll
    
    
                ################  cbest_both ################    
                #######%% Identify best c                
                # Use all range of c... it may end up a value at which all weights are 0.
                ix = np.argmin(meanPerClassErrorTestBoth)
                if smallestC==1:
                    cbest = cvect[meanPerClassErrorTestBoth <= (meanPerClassErrorTestBoth[ix]+semPerClassErrorTestBoth[ix])];
                    cbest = cbest[0]; # best regularization term based on minError+SE criteria
                    cbestAll = cbest
                else:
                    cbestAll = cvect[ix]
                print '\tFrame %d: %f' %(ifr,cbestAll)
    #            cbestAllFrs[ifr] = cbestAll            
    #            cbest = cbestAll
                        
                cbestFrs_both[ifr] = cbestAll
                
                
            else:
                cbestFrs_remCorr = cbestFrs
                cbestFrs_both = cbestFrs
                
            
            ########%% Set the decoder and class errors at best c (for data)
            """
            # you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
            # we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
            indBestC = np.in1d(cvect, cbest)
            
            w_bestc_data = wAllC[:,indBestC,:,ifr].squeeze() # numSamps x neurons
            b_bestc_data = bAllC[:,indBestC,ifr]
            
            classErr_bestC_train_data = perClassErrorTrain[:,indBestC,ifr].squeeze()
            
            classErr_bestC_test_data = perClassErrorTest[:,indBestC,ifr].squeeze()
            classErr_bestC_test_shfl = perClassErrorTest_shfl[:,indBestC,ifr].squeeze()
            classErr_bestC_test_chance = perClassErrorTest_chance[:,indBestC,ifr].squeeze()
            """
            
            ########### Plot C path    
            if doPlots:
                
                #######%% Compute average of class errors across numSamples        
                meanPerClassErrorTrain = np.mean(perClassErrorTrain[:,:,:,ifr], axis = (0,2));
                semPerClassErrorTrain = np.std(perClassErrorTrain[:,:,:,ifr], axis = (0,2))/np.sqrt(numSamples);            
                meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl[:,:,:,ifr], axis = (0,2));
#                semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl[:,:,:,ifr], axis = 0)/np.sqrt(numSamples);            
#                meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance[:,:,:,ifr], axis = (0,2));
#                semPerClassErrorTest_chance = np.std(perClassErrorTest_chance[:,:,:,ifr], axis = 0)/np.sqrt(numSamples);

#                print 'Best c (inverse of regularization parameter) = %.2f' %cbest
                plt.figure(figsize=(3,3))
#                plt.subplot(1,2,1)
                plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
                plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
#                plt.fill_between(cvect, meanPerClassErrorTestRemCorr-semPerClassErrorTestRemCorr, meanPerClassErrorTestRemCorr+ semPerClassErrorTestRemCorr, alpha=0.5, edgecolor='g', facecolor='g')        
#                plt.fill_between(cvect, meanPerClassErrorTestBoth-semPerClassErrorTestBoth, meanPerClassErrorTestBoth+ semPerClassErrorTestBoth, alpha=0.5, edgecolor='m', facecolor='m')        
            #    plt.fill_between(cvect, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
            #    plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
                
                plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
                plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'testing')
#                plt.plot(cvect, meanPerClassErrorTest_chance, 'b', label = 'cv-chance')       
                plt.plot(cvect, meanPerClassErrorTest_shfl, 'y', label = 'shfl')    
#                plt.plot(cvect, meanPerClassErrorTestRemCorr, 'g', label = 'rem corr')        
#                plt.plot(cvect, meanPerClassErrorTestBoth, 'm', label = 'test+remCorr')                   
            
                plt.plot(cvect[cvect==cbestFrs[ifr]], meanPerClassErrorTest[cvect==cbestFrs[ifr]], 'o', markerfacecolor='r', markeredgecolor='r')
#                plt.plot(cvect[cvect==cbestFrs_remCorr[ifr]], meanPerClassErrorTestRemCorr[cvect==cbestFrs_remCorr[ifr]], 'o', markerfacecolor='g', markeredgecolor='g')
#                plt.plot(cvect[cvect==cbestFrs_both[ifr]], meanPerClassErrorTestBoth[cvect==cbestFrs_both[ifr]], 'o', markerfacecolor='m', markeredgecolor='m')
                
                plt.xlim([cvect[1], cvect[-1]])
                plt.xscale('log')
                plt.xlabel('c (inverse of regularization parameter)')
                plt.ylabel('classification error (%)')
                plt.legend(loc='center left', bbox_to_anchor=(1, .7))                
                plt.title('Frame %d' %(ifr))
                plt.tight_layout()
    
    

    ##############
    return nRandCorrSel, wAllC, bAllC, cbestFrs, cbestFrs_remCorr, cbestFrs_both, cvect, \
    perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance, \
    perClassErrorTestRemCorr, perClassErrorTestRemCorr_shfl, perClassErrorTestRemCorr_chance, \
    perClassErrorTestBoth, perClassErrorTestBoth_shfl, perClassErrorTestBoth_chance, \
    trsnow_allSamps, testTrInds_allSamps, trsTrainedTestedInds, trsRemCorrInds, \
    Ytest_allSamps, Ytest_hat_allSampsFrs, Ytest_hat_remCorr_allSampsFrs, Ytest_hat_both_allSampsFrs




#%% Function to run SVM to decode stimulus outcome (when X is frames x units x trials)
# we make sure for each outcome (corr, incorr), there are equal HR and LR trials. Also if useEqualTrNums=1, we make sure corr and incorr have the same number of trials.

# Remember each numSamples will have a different set of training and testing dataset, however for each numSamples, the same set of testing/training dataset
# will be used for all frames and all values of c (unless shuffleTrs is 1, in which case different frames and c values will have different training/testing datasets.)

def setbesc_frs_outcome(allResp_HR_LR, X,Y,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest=np.nan,fr2an=np.nan, shflTrLabs=0, nRandCorrSel=nRandCorrSel):
    
    # Note: I commented out all parts related to remCorr in this function: we cant use remaining corr when decoding outcome, bc it includes only trials for one of the classes (corr), so remCorr_shfl will be high too... if you want to do this correctly you have use all remaining (corr and incorr) trials... but it takes a lot more memory and time to run the jobs, and having remCorr doesnt seem to add much!    

    '''
    X = X_svm
    Y = Y_svm
    cbest = np.nan
    fr2an = np.nan
    shflTrLabs = 0
    '''
    # numSamples = 10; # number of iterations for finding the best c (inverse of regularization parameter)
    # if you don't want to regularize, go with a very high cbest and don't run the section below.
    # cbest = 10**6    
    # regType = 'l1'
    # kfold = 10;
            
    import numpy as np
    import numpy.random as rng
    from crossValidateModel import crossValidateModel
    from linearSVM import linearSVM
    
    def perClassError(Y, Yhat):
        perClassEr = np.sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
        return perClassEr
   
#    if type(X_svm_incorr)==int:
#        testIncorr = 0
#    else:
#        testIncorr = 1
#        
#    if testIncorr:
#        len_test_incorr = len(Y_svm_incorr)
        
    # frames to do SVM analysis on
    if np.isnan(fr2an).all(): # run SVM on all frames
        frs = range(X.shape[0])
    else:
        frs = fr2an        
    
    
    # set range of c (regularization parameters) to check    
    if np.isnan(cbest).all(): # we need to set cbest
        bestcProvided = False        
        if regType == 'l1':
            print '\n-------------- Running l1 svm classification --------------\r' 
            # cvect = 10**(np.arange(-4, 6,0.2))/numTrials;
            cvect = 10**(np.arange(-4, 6,0.2))/numDataPoints
        elif regType == 'l2':
            print '\n-------------- Running l2 svm classification --------------\r' 
            cvect = 10**(np.arange(-6, 6,0.2))/numDataPoints          
        nCvals = len(cvect)
#        print 'try the following regularization values: \n', cvect
        # formattedList = ['%.2f' % member for member in cvect]
        # print 'try the following regularization values = \n', formattedList        
    else: # bestc is provided and we want to fit svm on shuffled trial labels
        bestcProvided = True           
        nCvals = 1 # cbest is already provided
        
    
#    smallestC = 0 # if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
    if smallestC==1:
        print 'bestc = smallest c whose cv error is less than 1se of min cv error'
    else:
        print 'bestc = c that gives min cv error'
    #I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.
        
        
    
    ##########################################
    nc = (Y==1).sum() # later we change the value of nc to the same value as numTrials_c (ie equal hr_corr and lr_corr)
    ni = (Y==0).sum() # later we change the value of ni to the same value as numTrials_i (ie equal hr_incorr and lr_incorr)
    
    nc0 = nc; ni0 = ni;
    
    trsCorr = np.argwhere(Y==1).squeeze() # indeces out of Y
    trsIncorr = np.argwhere(Y==0).squeeze()

    hr_c = (allResp_HR_LR[trsCorr]==1).sum()
    lr_c = (allResp_HR_LR[trsCorr]==0).sum()    
    hr_i = (allResp_HR_LR[trsIncorr]==1).sum()
    lr_i = (allResp_HR_LR[trsIncorr]==0).sum()

    print nc, ' correct trials: ', hr_c, ' HR ', lr_c, ' LR'  
    print ni, ' incorrect trials: ', hr_i, ' HR ', lr_i, ' LR'  
    
    if min([hr_c, lr_c, hr_i, lr_i]) < 3: # 3 of each will give 12 trials total, the minimum we want to train the SVM, so at least we get 1 testing trials (with 10 folds)!
        sys.exit('Too few trials to do SVM training!')
        
    
    #####################        
    # Set numTrials in the decoder so you can set len_test
    #####################
    
    # we want to have equal number of each choice for each outcome 
    # correct
    if hr_c != lr_c:
        trsn_c = min(lr_c,hr_c)
        numTrials_c = trsn_c*2
#        if hr_c > lr_c:
#            print 'Correct trials: subselecting HR trials to make HR_corr and LR_corr equal size!'
#            numTrials_c = lr_c*2
#        else:
#            print 'Correct trials: subselecting LR trials to make HR_corr and LR_corr equal size!'
#            numTrials_c = hr_c*2
    else:
        numTrials_c = lr_c*2 # same as nc below
        
    
    # incorrect
    if hr_i != lr_i:
        trsn_i = min(lr_i,hr_i)
        numTrials_i = trsn_i*2
#        if hr_i > lr_i:
#            print 'Incorrect trials: subselecting HR trials to make HR_incorr and LR_incorr equal size!'        
#            numTrials_i = lr_i*2
#        else:
#            print 'Incorrect trials: subselecting LR trials to make HR_incorr and LR_incorr equal size!'
#            numTrials_i = hr_i*2
    else:
        numTrials_i = lr_i*2 # same as ni below
            
    
    if numTrials_c == numTrials_i:
       useEqualTrNums = 0 # corr and incorr are already equal! (there is a chance that the original nc and ni are not equal but they become equal after HR/LR subselection, we will still set useEqualTrNums to 0, so there wont be a test on remCorr trials.)
    
       
    ###### NOTE: if useEqualTrNums=1, it is either because we don't care about having equal number of corr and incorr (the 2 classes) or because the 2 classes already have equal number of trials, so we dont need to make them equal, or run analysis on remCorr trials.

       
    ####### Set numTrials
    if useEqualTrNums: # make sure both classes (corr and incorr) have the same number of trials
        numTrials = 2 * min(numTrials_i , numTrials_c)
        len_remCorr = 0
        # Note: I am commenting below bc for decoding outcome, we cant use remaining corr, bc it includes only trials for one of the classes (corr), so remCorr_shfl will be high too... if you want to do this correctly you have use all remaining (corr and incorr) trials... but it takes a lot more memory and time to run the jobs, and having remCorr doesnt seem to add much!
        '''
        if numTrials_i < numTrials_c:
            len_remCorr = len(trsCorr) - min(numTrials_i , numTrials_c) #nc - len_trsNowCorr # 
        else:
            len_remCorr = len(trsIncorr) - min(numTrials_i , numTrials_c) #nc - len_trsNowCorr # 
        '''
    else:        
        numTrials = numTrials_c + numTrials_i
        len_remCorr = 0
    
    
    len_test = numTrials - int((kfold-1.)/kfold*numTrials) # length of testing trials   
    
    len_both = len_test+len_remCorr
    
    
    #####################        
    #####################        
    
    X0 = X + 0
    Y0 = Y + 0
    
    
    print 'numRandCorrSamples = %i' %(nRandCorrSel)
        
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    nFrs = np.shape(X)[0]
    
    perClassErrorTrain = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    perClassErrorTest = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)    
    perClassErrorTest_shfl = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    perClassErrorTest_chance = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    
    # Note: I am commenting below bc for decoding outcome, we cant use remaining corr, bc it includes only trials for one of the classes (corr), so remCorr_shfl will be high too... if you want to do this correctly you have use all remaining (corr and incorr) trials... but it takes a lot more memory and time to run the jobs, and having remCorr doesnt seem to add much!
    perClassErrorTestRemCorr = np.nan #np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan) 
    perClassErrorTestRemCorr_shfl = np.nan #np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    perClassErrorTestRemCorr_chance = np.nan #np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    
    perClassErrorTestBoth = np.nan #np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan) 
    perClassErrorTestBoth_shfl = np.nan #np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    perClassErrorTestBoth_chance = np.nan #np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    
    bAllC = np.full((numSamples, nCvals, nRandCorrSel, nFrs), np.nan)
    wAllC = np.full((numSamples, nCvals, X.shape[1], nRandCorrSel, nFrs), np.nan)
    
    Ytest_hat_allSampsFrs = np.full((numSamples, nCvals, nRandCorrSel, nFrs, len_test), np.nan)        #    testTrInds_outOfY0_allSamps = np.full((numSamples, len_test), np.nan)
    Ytest_hat_remCorr_allSampsFrs = np.nan #np.full((numSamples, nCvals, nRandCorrSel, nFrs, len_remCorr), np.nan)
    Ytest_hat_both_allSampsFrs = np.nan #np.full((numSamples, nCvals, nRandCorrSel, nFrs, len_both), np.nan)
    
    trsnow_allSamps = np.full((nRandCorrSel, numTrials), np.nan) 
    trsRemCorrInds = np.nan #np.full((nRandCorrSel, len_remCorr), np.nan)    
    
    Ytest_allSamps = np.full((numSamples, nRandCorrSel, len_test), np.nan)    
    testTrInds_allSamps = np.full((numSamples, nRandCorrSel, len_test), np.nan)
    trsTrainedTestedInds = np.full((numSamples, nRandCorrSel, numTrials), np.nan)
            
        
        
    
    ########################################## Make sure HR and LR have the same trial number within each class (corr and incorr), also if useEqualTrNums=1, make sure corr and incorr (ie the 2 classes in SVM) have the same trial number ##########################################
    for sc in range(nRandCorrSel): # sc = 0
        
        print '===== Random correct-trial subselect %d =====' %(sc)       
        
        ####################################
        # Make sure each outcome (corr, incorr) has the same number of LR and HR trials when training the classifier
        ####################################
        
        # correct; set trsnow_c: index of trials on Y, correct, half HR, half LR
        if hr_c != lr_c:
            if hr_c > lr_c: # subselect HR trials ... for correct trials, take all LR trials and subselect HR trials.
                randtrs = np.argwhere(np.logical_and(Y0==1 , allResp_HR_LR==1))[rng.permutation(hr_c)[0:trsn_c]].squeeze() # subselect of HR correct trials.
                trsnow_c = np.sort(np.concatenate((randtrs , np.argwhere(np.logical_and(Y0==1 , allResp_HR_LR==0)).squeeze()))) # index of trials after picking all LR correct trials and a subselect of HR correct trials, to make sure within correct, LR and HR have the same number in the final Y (on which svm was run)
            elif hr_c < lr_c: # subselect LR trials ... for correct trials, take all HR trials and subselect LR trials.
                randtrs = np.argwhere(np.logical_and(Y0==1 , allResp_HR_LR==0))[rng.permutation(lr_c)[0:trsn_c]].squeeze() # subselect of LR correct trials.
                trsnow_c = np.sort(np.concatenate((randtrs , np.argwhere(np.logical_and(Y0==1 , allResp_HR_LR==1)).squeeze()))) # index of trials after picking all HR correct trials and a subselect of LR correct trials, to make sure within correct, LR and HR have the same number in the final Y (on which svm was run)
            nc = len(trsnow_c)
        else: # include all correct trials
            trsnow_c = np.argwhere(Y0==1).squeeze() #np.arange(0, nc)

        
        # incorrect; set trsnow_i: index of trials on Y, incorrect, half HR, half LR
        if hr_i != lr_i:
            if hr_i > lr_i: # subselect HR trials ... for incorrect trials, take all LR trials and subselect HR trials.
                randtrs = np.argwhere(np.logical_and(Y0==0 , allResp_HR_LR==1))[rng.permutation(hr_i)[0:trsn_i]].squeeze() # subselect of HR incorrect trials.
                trsnow_i = np.sort(np.concatenate((randtrs , np.argwhere(np.logical_and(Y0==0 , allResp_HR_LR==0)).squeeze()))) # index of trials after picking all LR incorrect trials and a subselect of HR incorrect trials, to make sure within incorrect, LR and HR have the same number in the final Y (on which svm was run)
            elif hr_i < lr_i: # subselect LR trials ... for incorrect trials, take all HR trials and subselect LR trials.
                randtrs = np.argwhere(np.logical_and(Y0==0 , allResp_HR_LR==0))[rng.permutation(lr_i)[0:trsn_i]].squeeze() # subselect of LR incorrect trials.
                trsnow_i = np.sort(np.concatenate((randtrs , np.argwhere(np.logical_and(Y0==0 , allResp_HR_LR==1)).squeeze()))) # index of trials after picking all HR incorrect trials and a subselect of LR incorrect trials, to make sure within incorrect, LR and HR have the same number in the final Y (on which svm was run)
            ni = len(trsnow_i)
        else: # include all incorrect trials
            trsnow_i = np.argwhere(Y0==0).squeeze() #np.arange(0, ni)

           
        
        ####################################        
        # make sure corr and incorr have the same number of trials. We have already made sure that within each outcome, HR and LR have the same trial number. (nc and ni are updated already) 
        ####################################
        
        if useEqualTrNums: # we do this so both classes (corr and incorr) have the same number of trials in the training decoder, otherwise decoder performance even on shuffle trial labels will deviate from chance.
            if nc > ni: # subselect correct trials to the same number as incorrect trials                
                # we need to subselect ni/2 trials from HR_corr and LR_corr
                ntrnow = ni/2
                # correct; set trsnow_cc: index of trials on Y, correct, half HR, half LR. Each has ni/2 trials.
                randtrs_hr = np.argwhere(np.logical_and(Y0==1 , allResp_HR_LR==1))[rng.permutation(hr_c)[0:ntrnow]].squeeze() # subselect of HR correct trials to ni/2 trials
                randtrs_lr = np.argwhere(np.logical_and(Y0==1 , allResp_HR_LR==0))[rng.permutation(lr_c)[0:ntrnow]].squeeze() # subselect of LR correct trials to ni/2 trials
                    
                trsnow_cc = np.sort(np.concatenate((randtrs_hr , randtrs_lr))) # index of trials after picking a subselect of LR correct trials (ni/2) and a subselect of HR correct trials (ni/2), to make sure within correct, LR and HR have the same number in the final Y (on which svm was run), also correct and incorrect trials have the same number of trials.
                    
                # set the final trsNowAll. trsnow_i has half HR and half LR (we already did it in the section above)
                trsNowAll = np.sort(np.concatenate((trsnow_cc, trsnow_i))) # this includes half corr, half incorr; also half HR and half LR within each corr and incorr.

            elif nc < ni: # subselect incorrect trials to the same number as correct trials                
                print '\nVery unusual: there are more incorrect trials than correct trials!!\n'
                # we need to subselect nc/2 trials from HR_incorr and LR_incorr
                ntrnow = nc/2
                # incorrect; set trsnow_ii: index of trials on Y, incorrect, half HR, half LR. Each has nc/2 trials.
                randtrs_hr = np.argwhere(np.logical_and(Y0==0 , allResp_HR_LR==1))[rng.permutation(hr_i)[0:ntrnow]].squeeze() # subselect of HR incorrect trials to nc/2 trials
                randtrs_lr = np.argwhere(np.logical_and(Y0==0 , allResp_HR_LR==0))[rng.permutation(lr_i)[0:ntrnow]].squeeze() # subselect of LR incorrect trials to nc/2 trials
                    
                trsnow_ii = np.sort(np.concatenate((randtrs_hr , randtrs_lr))) # index of trials after picking a subselect of LR correct trials (nc/2) and a subselect of HR correct trials (nc/2), to make sure within correct, LR and HR have the same number in the final Y (on which svm was run), also correct and incorrect trials have the same number of trials.
                    
                # set the final trsNowAll. trsnow_c has half HR and half LR (we already did it in the section above)
                trsNowAll = np.sort(np.concatenate((trsnow_c, trsnow_ii))) # this includes half corr, half incorr; also half HR and half LR within each corr and incorr.

        else: # we only make sure within each corr and incorr class, there are half HR and half LR trials. We don't care about corr and incorr being equal. This is the minimum we want to have to make sure we can decode outcome correctly (ie within each outcome, there must be equal HR and LR trials).
            trsNowAll = np.sort(np.concatenate((trsnow_c, trsnow_i))) # this includes half HR and half LR within each corr and incorr subsets. but corr and incorr are not necessarily the same number        
        
        trsnow_allSamps[sc,:] = trsNowAll
        
        
        
        ####################################
        # Now set X and Y only for trsNowAll trials (half corr, half incorr, half HR, half LR)
        ####################################
        
        X = X0[:,:,trsNowAll] # trsNowAll : index of trials (out of X0 and Y0) that are used to set X and Y
        Y = Y0[trsNowAll]
        
        print 'size of X and Y: ', X.shape, Y.shape
        
        allResp_HR_LRn = allResp_HR_LR[trsNowAll]
        
        
        ####################################
        # Set X and Y for the remaining correct trials (those not in trsNowCorr). We will test on these trials the decoder trained on X and Y ####################             
        ####################################
        
        # Note: I am commenting below bc for decoding outcome, we cant use remaining corr, bc it includes only trials for one of the classes (corr), so remCorr_shfl will be high too... if you want to do this correctly you have use all remaining (corr and incorr) trials... but it takes a lot more memory and time to run the jobs, and having remCorr doesnt seem to add much!
        if 0: #useEqualTrNums:
            if nc > ni:
                trsRemCorr = trsCorr[~np.in1d(trsCorr, trsnow_cc)] # those correct trials (indeces in outcomes) that are not among the random pick for current trsnow_cc
            elif nc < ni:
                trsRemCorr = trsIncorr[~np.in1d(trsIncorr, trsnow_ii)] # those incorrect trials (indeces in outcomes) that are not among the random pick for current trsnow_ii
            
            trsRemCorrInds[sc,:] = trsRemCorr # (indices are from outcomes array)        
            
            X_remCorr = X0[:,:,trsRemCorr]
            Y_remCorr = Y0[trsRemCorr]
            print 'size of X_remCorr and Y_remCorr: ', X_remCorr.shape, Y_remCorr.shape
            
    
    
        ####################################
        # Checks
        ####################################
        hrcn = sum(np.logical_and(Y==1 , allResp_HR_LRn==1))
        lrcn = sum(np.logical_and(Y==1 , allResp_HR_LRn==0))            
        hrin = sum(np.logical_and(Y==0 , allResp_HR_LRn==1))
        lrin = sum(np.logical_and(Y==0 , allResp_HR_LRn==0))

        print 'After attempting to make HR and LR the same number:\n\tHR_corr: %d, LR_corr: %d trials\n\tHR_incorr: %d, LR_incorr: %d trials' %(hrcn, lrcn, hrin, lrin)
        
        if useEqualTrNums:
            print 'After attempting to make corr and incorr the same number:\n\tCorrect: %d, Incorrect: %d trials' %((Y==1).sum(), (Y==0).sum())        

        if ~np.equal(hrcn , lrcn):
            sys.exit('Error: among corr, HR and LR trial number are different!!')

        if ~np.equal(hrin , lrin):
            sys.exit('Error: among incorr, HR and LR trial number are different!!')
            
        if useEqualTrNums:
            if ~np.equal(hrcn+lrcn , hrin+lrin):
                sys.exit('Error: corr and incorr trial number are different!!')
                
                
                
                
                
        ########################################### Permute trials numSamples times to select various sets for training and testing SVM
        # Train SVM numSamples times to get numSamples cross-validated datasets.    
        
        for s in range(numSamples):  # s = 0      
            print 'Iteration %d' %(s)                
            
            ####################################
            # Set chance Y: same length as Y for testing data, and with equal number of classes 0 and 1.  
            ####################################
            permIxs = rng.permutation(len_test) # needed to set perClassErrorTest_shfl   
        
            Y_chance = np.zeros(len_test)
            if rng.rand()>.5:
                b = rng.permutation(len_test)[0:np.floor(len_test/float(2)).astype(int)]
            else:
                b = rng.permutation(len_test)[0:np.ceil(len_test/float(2)).astype(int)]
            Y_chance[b] = 1
    
            
            # Note: I am commenting below bc for decoding outcome, we cant use remaining corr, bc it includes only trials for one of the classes (corr), so remCorr_shfl will be high too... if you want to do this correctly you have use all remaining (corr and incorr) trials... but it takes a lot more memory and time to run the jobs, and having remCorr doesnt seem to add much!
            if 0: #nc!=ni: #useEqualTrNums:
                permIxs_remCorr = rng.permutation(X_remCorr.shape[2]);
    
                Y_chance_remCorr = np.zeros(Y_remCorr.shape[0])
                if rng.rand()>.5: # in case Y_remCorr has odd size, we make sure sometimes we round it up and sometimes down so the number of trials of the 2 classes are balanced in chance data.
                    b = rng.permutation(Y_remCorr.shape[0])[0:np.floor(Y_remCorr.shape[0]/float(2)).astype(int)]
                else:
                    b = rng.permutation(Y_remCorr.shape[0])[0:np.ceil(Y_remCorr.shape[0]/float(2)).astype(int)]
                Y_chance_remCorr[b] = 1
     


                permIxs_tot = rng.permutation(len_test+X_remCorr.shape[2])
                
                Y_chance_tot = np.zeros(permIxs_tot.shape[0])
                if rng.rand()>.5: 
                    b = rng.permutation(permIxs_tot.shape[0])[0:np.floor(permIxs_tot.shape[0]/float(2)).astype(int)]
                else:
                    b = rng.permutation(permIxs_tot.shape[0])[0:np.ceil(permIxs_tot.shape[0]/float(2)).astype(int)]
                Y_chance_tot[b] = 1    
    
            
            ####################################
            # If you want to train SVM on shuffled trial labels, shuffle the Y
            ####################################
            if shflTrLabs: # shuffle trial classes in Y
                Y = np.zeros(numTrials) # Y_chance0
                if rng.rand()>.5:
                    b = rng.permutation(numTrials)[0:np.floor(numTrials/float(2)).astype(int)]
                else:
                    b = rng.permutation(numTrials)[0:np.ceil(numTrials/float(2)).astype(int)]
                Y[b] = 1
    
            
            ####################################
            ######################## Permute trial orders, so the training and testing datasets are different for each numSamples (we only do this if shuffleTrs is 0, so crossValidateModel does not shuffle trials, so we have to do it here, otherwise all numSamples will have the same set of testing and training datasets.)
            ######################## REMEMBER : YOU ARE CHANGING THE ORDER OF TRIALS HERE!!! ########################
            ####################################
            if shuffleTrs==0: # if 0, shuffle trials to break any dependencies on the sequence of trails. Otherwise, do the shuffling inside crossValidateModel. I prefer to do it here to make sure all frames are trained on the same set of trials.
                
    #            Ybefshfl = Y            
                shfl = rng.permutation(np.arange(0, numTrials)) # shfl: new order of trials ... shuffled indeces of Y... the last 1/10th indeces will be testing trials.
                
                Y = Y[shfl] 
                X = X[:,:,shfl]             
                
                # Ytest_allSamps[s,:] : Y that will be used as testing trials in this sample
                Ytest_allSamps[s,sc,:] = Y[np.arange(numTrials-len_test, numTrials)] # the last 1/10th of Y (after applying shfl labels to it)
                testTrInds = shfl[np.arange(numTrials-len_test, numTrials)] # indeces to be applied on trsnow in order to get the trials (index out of Y0) that were used as testing trs; eg stimrate[trsnow[testTrInds]] is the stimrate of testing trials
    #            testTrInds_outOfY0 = trsnow[testTrInds] # index of testing trials out of Y0 (not Y!) (that will be used in svm below)
                 ######## IMPORTANT: Ybefshfl[testTrInds] is same as Y0[trsnow[testTrInds]] and same as Y[np.arange(numTrials-len_test, numTrials)] and same as summary.YTest computed below ########
    
                testTrInds_allSamps[s,sc,:] = testTrInds            
    #            testTrInds_outOfY0_allSamps[s,:] = testTrInds_outOfY0            
                # keep index of trials that were used for training, cv, and remainedCorr testing.
                trsTrainedTestedInds[s,sc,:] = trsNowAll[shfl] # first 90% training, last 10% testing (indices are from outcomes array)
            
            else:
                testTrInds_allSamps = np.nan # for now, but to set it correctly: testTrInds will be set in crossValidateModel.py, you just need to output it from crossValidateModel
                Ytest_allSamps[s,sc,:] = np.nan  
    
 
            ##################################################################################################################################################################
            ########################## Start training SVM on each frame ######################################################################################################
            ##################################################################################################################################################################
            for ifr in frs: # ifr = 0 # train SVM on each frame
#                print '\tFrame %d' %(ifr)
                if bestcProvided:
                    cvect = [cbest[ifr]]
            
                ######################## Loop over different values of regularization
                for i in range(nCvals): # i = 0 # train SVM using different values of regularization parameter
                    if regType == 'l1':                               
                        summary,_ =  crossValidateModel(X[ifr,:,:].transpose(), Y, linearSVM, kfold = kfold, l1 = cvect[i], shflTrs = shuffleTrs)
                        
                    elif regType == 'l2':
                        summary,_ =  crossValidateModel(X[ifr,:,:].transpose(), Y, linearSVM, kfold = kfold, l2 = cvect[i], shflTrs = shuffleTrs)
                            
                                
                    wAllC[s,i,:,sc,ifr] = np.squeeze(summary.model.coef_); # weights of all neurons for each value of c and each shuffle
                    bAllC[s,i,sc,ifr] = np.squeeze(summary.model.intercept_);
            
                    # Classification errors                    
                    perClassErrorTrain[s,i,sc,ifr] = summary.perClassErrorTrain
                    perClassErrorTest[s,i,sc,ifr] = summary.perClassErrorTest # perClassError(YTest, linear_svm.predict(XTest));
                    
                    # Testing correct shuffled data: 
                    # same decoder trained on correct trials, but use shuffled trial labels to compute class error
                    Ytest_hat = summary.model.predict(summary.XTest) # prediction of trial label for each trial
                    perClassErrorTest_shfl[s,i,sc,ifr] = perClassError(summary.YTest[permIxs], Ytest_hat) # fraction of incorrect predicted trial labels
                    perClassErrorTest_chance[s,i,sc,ifr] = perClassError(Y_chance, Ytest_hat)
                    Ytest_hat_allSampsFrs[s,i,sc,ifr,:] = Ytest_hat
                    
                    
                    # Note: I am commenting below bc for decoding outcome, we cant use remaining corr, bc it includes only trials for one of the classes (corr), so remCorr_shfl will be high too... if you want to do this correctly you have use all remaining (corr and incorr) trials... but it takes a lot more memory and time to run the jobs, and having remCorr doesnt seem to add much!
                    if 0: #nc!=ni: # useEqualTrNums: # 
                        # Test the decoder on remaining correct trials
                        if len(trsRemCorr)!=0:
                            Ytest_hat_remCorr = summary.model.predict(X_remCorr[ifr,:,:].transpose()) # prediction of trial label for each trial
                            perClassErrorTestRemCorr[s,i,sc,ifr] = perClassError(Y_remCorr, Ytest_hat_remCorr)
                            perClassErrorTestRemCorr_shfl[s,i,sc,ifr] = perClassError(Y_remCorr[permIxs_remCorr], Ytest_hat_remCorr)
                            perClassErrorTestRemCorr_chance[s,i,sc,ifr] = perClassError(Y_chance_remCorr, Ytest_hat_remCorr)
                            Ytest_hat_remCorr_allSampsFrs[s,i,sc,ifr,:] = Ytest_hat_remCorr
                            
                        # Pool remCorrect trials and testing trials
                        x = np.concatenate((summary.XTest, X_remCorr[ifr,:,:].transpose()), axis=0)
                        y = np.concatenate((summary.YTest, Y_remCorr))
                        Ytest_hat_both = summary.model.predict(x)
                        perClassErrorTestBoth[s,i,sc,ifr] = perClassError(y, Ytest_hat_both)
                        perClassErrorTestBoth_shfl[s,i,sc,ifr] = perClassError(y[permIxs_tot], Ytest_hat_both)
                        perClassErrorTestBoth_chance[s,i,sc,ifr] = perClassError(Y_chance_tot, Ytest_hat_both)
                        Ytest_hat_both_allSampsFrs[s,i,sc,ifr,:] = Ytest_hat_both
                    


    ###########################################################################
    ############## Find bestc for each frame, and plot the c path #############
    ###########################################################################
    if bestcProvided: 
        cbestFrs = cbest
#        cbestAllFrs = cbest        
    else:
        print '--------------- Identifying best c ---------------' 
        cbestFrs = np.full((X.shape[0]), np.nan)  
#        cbestAllFrs = np.full((X.shape[0]), np.nan)  
        cbestFrs_remCorr = np.full((X.shape[0]), np.nan)  
        cbestFrs_both = np.full((X.shape[0]), np.nan)  
        
        for ifr in frs: #range(X.shape[0]):    

            ################  cbest  ################
            #######%% Compute average of class errors across numSamples and nRandCorrSel      
            meanPerClassErrorTest = np.mean(perClassErrorTest[:,:,:,ifr], axis = (0,2));
            semPerClassErrorTest = np.std(perClassErrorTest[:,:,:,ifr], axis = (0,2))/np.sqrt(numSamples);

            #######%% Identify best c                
            # Use all range of c... it may end up a value at which all weights are 0.
            ix = np.argmin(meanPerClassErrorTest)
            if smallestC==1:
                cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
                cbest = cbest[0]; # best regularization term based on minError+SE criteria
                cbestAll = cbest
            else:
                cbestAll = cvect[ix]
            print '\tFrame %d: %f' %(ifr,cbestAll)
#            cbestAllFrs[ifr] = cbestAll            
                    
            cbestFrs[ifr] = cbestAll

                
            # Note: I am commenting below bc for decoding outcome, we cant use remaining corr, bc it includes only trials for one of the classes (corr), so remCorr_shfl will be high too... if you want to do this correctly you have use all remaining (corr and incorr) trials... but it takes a lot more memory and time to run the jobs, and having remCorr doesnt seem to add much!
            if 0: #nc!=ni:
                #######%% Compute average of class errors across numSamples and nRandCorrSel              
                meanPerClassErrorTestRemCorr = np.mean(perClassErrorTestRemCorr[:,:,:,ifr], axis = (0,2))
                semPerClassErrorTestRemCorr = np.std(perClassErrorTestRemCorr[:,:,:,ifr], axis = (0,2))/np.sqrt(numSamples)
                    
                #######%% Compute average of class errors across numSamples and nRandCorrSel              
                meanPerClassErrorTestBoth = np.mean(perClassErrorTestBoth[:,:,:,ifr], axis = (0,2))
                semPerClassErrorTestBoth = np.std(perClassErrorTestBoth[:,:,:,ifr], axis = (0,2))/np.sqrt(numSamples)
                
                ################  cbest_remCorr  ################    
                #######%% Identify best c                
                # Use all range of c... it may end up a value at which all weights are 0.
                ix = np.argmin(meanPerClassErrorTestRemCorr)
                if smallestC==1:
                    cbest = cvect[meanPerClassErrorTestRemCorr <= (meanPerClassErrorTestRemCorr[ix]+semPerClassErrorTestRemCorr[ix])];
                    cbest = cbest[0]; # best regularization term based on minError+SE criteria
                    cbestAll = cbest
                else:
                    cbestAll = cvect[ix]
                print '\tFrame %d: %f' %(ifr,cbestAll)
    #            cbestAllFrs[ifr] = cbestAll            
    #            cbest = cbestAll
                        
                cbestFrs_remCorr[ifr] = cbestAll
    
    
                ################  cbest_both ################    
                #######%% Identify best c                
                # Use all range of c... it may end up a value at which all weights are 0.
                ix = np.argmin(meanPerClassErrorTestBoth)
                if smallestC==1:
                    cbest = cvect[meanPerClassErrorTestBoth <= (meanPerClassErrorTestBoth[ix]+semPerClassErrorTestBoth[ix])];
                    cbest = cbest[0]; # best regularization term based on minError+SE criteria
                    cbestAll = cbest
                else:
                    cbestAll = cvect[ix]
                print '\tFrame %d: %f' %(ifr,cbestAll)
    #            cbestAllFrs[ifr] = cbestAll            
    #            cbest = cbestAll
                        
                cbestFrs_both[ifr] = cbestAll
                
                
            else:
                cbestFrs_remCorr = cbestFrs
                cbestFrs_both = cbestFrs
                
            
            ########%% Set the decoder and class errors at best c (for data)
            """
            # you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
            # we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
            indBestC = np.in1d(cvect, cbest)
            
            w_bestc_data = wAllC[:,indBestC,:,ifr].squeeze() # numSamps x neurons
            b_bestc_data = bAllC[:,indBestC,ifr]
            
            classErr_bestC_train_data = perClassErrorTrain[:,indBestC,ifr].squeeze()
            
            classErr_bestC_test_data = perClassErrorTest[:,indBestC,ifr].squeeze()
            classErr_bestC_test_shfl = perClassErrorTest_shfl[:,indBestC,ifr].squeeze()
            classErr_bestC_test_chance = perClassErrorTest_chance[:,indBestC,ifr].squeeze()
            """
            
            ########### Plot C path    
            if doPlots:
                
                #######%% Compute average of class errors across numSamples        
                meanPerClassErrorTrain = np.mean(perClassErrorTrain[:,:,:,ifr], axis = (0,2));
                semPerClassErrorTrain = np.std(perClassErrorTrain[:,:,:,ifr], axis = (0,2))/np.sqrt(numSamples);            
                meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl[:,:,:,ifr], axis = (0,2));
#                semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl[:,:,:,ifr], axis = 0)/np.sqrt(numSamples);            
#                meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance[:,:,:,ifr], axis = (0,2));
#                semPerClassErrorTest_chance = np.std(perClassErrorTest_chance[:,:,:,ifr], axis = 0)/np.sqrt(numSamples);

#                print 'Best c (inverse of regularization parameter) = %.2f' %cbest
                plt.figure(figsize=(3,3))
#                plt.subplot(1,2,1)
                plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
                plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
#                plt.fill_between(cvect, meanPerClassErrorTestRemCorr-semPerClassErrorTestRemCorr, meanPerClassErrorTestRemCorr+ semPerClassErrorTestRemCorr, alpha=0.5, edgecolor='g', facecolor='g')        
#                plt.fill_between(cvect, meanPerClassErrorTestBoth-semPerClassErrorTestBoth, meanPerClassErrorTestBoth+ semPerClassErrorTestBoth, alpha=0.5, edgecolor='m', facecolor='m')        
            #    plt.fill_between(cvect, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
            #    plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
                
                plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
                plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'testing')
#                plt.plot(cvect, meanPerClassErrorTest_chance, 'b', label = 'cv-chance')       
                plt.plot(cvect, meanPerClassErrorTest_shfl, 'y', label = 'shfl')    
#                plt.plot(cvect, meanPerClassErrorTestRemCorr, 'g', label = 'rem corr')        
#                plt.plot(cvect, meanPerClassErrorTestBoth, 'm', label = 'test+remCorr')                   
            
                plt.plot(cvect[cvect==cbestFrs[ifr]], meanPerClassErrorTest[cvect==cbestFrs[ifr]], 'o', markerfacecolor='r', markeredgecolor='r')
#                plt.plot(cvect[cvect==cbestFrs_remCorr[ifr]], meanPerClassErrorTestRemCorr[cvect==cbestFrs_remCorr[ifr]], 'o', markerfacecolor='g', markeredgecolor='g')
#                plt.plot(cvect[cvect==cbestFrs_both[ifr]], meanPerClassErrorTestBoth[cvect==cbestFrs_both[ifr]], 'o', markerfacecolor='m', markeredgecolor='m')
                
                plt.xlim([cvect[1], cvect[-1]])
                plt.xscale('log')
                plt.xlabel('c (inverse of regularization parameter)')
                plt.ylabel('classification error (%)')
                plt.legend(loc='center left', bbox_to_anchor=(1, .7))                
                plt.title('Frame %d' %(ifr))
                plt.tight_layout()
    
    

    ##############
    return nRandCorrSel, wAllC, bAllC, cbestFrs, cbestFrs_remCorr, cbestFrs_both, cvect, \
    perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance, \
    perClassErrorTestRemCorr, perClassErrorTestRemCorr_shfl, perClassErrorTestRemCorr_chance, \
    perClassErrorTestBoth, perClassErrorTestBoth_shfl, perClassErrorTestBoth_chance, \
    trsnow_allSamps, testTrInds_allSamps, trsTrainedTestedInds, trsRemCorrInds, \
    Ytest_allSamps, Ytest_hat_allSampsFrs, Ytest_hat_remCorr_allSampsFrs, Ytest_hat_both_allSampsFrs



#%% Set the subsamples of exc neurons (same size as number of inh) for SVM
   
nFrs = np.shape(X_svm)[0]   
lenInh = (inhRois==1).sum()
excI = np.argwhere(inhRois==0).squeeze() # indeces of exc neurons (out of X_svm which includes all neurons)
    
## Set Xexc for n exc (pick numShufflesEx sets of n exc neurons) and 2n exc (n=lenInh)
if doInhAllexcEqexc[2]!=0: # or addNs_roc==0:    
    if np.logical_or(doInhAllexcEqexc[2] == 1 , doInhAllexcEqexc[2] == 2): ###### n exc (pick numShufflesEx sets of n exc neurons)
        XexcEq = []       
        XexcEq_incorr = []       
        excNsEachSamp_indsOutOfExcNs = [] # numShufflesExc x frs x units x trials        
        if cbestKnown: # we already have saved exc neuron indeces for each exc samp, so we use those same samps            
            for ii in range(excNsEachSamp.shape[0]):  # select n random exc (n = number of inh)         
                Xexc = X_svm[:, excNsEachSamp[ii],:]
                XexcEq.append(Xexc)            
                excNsEachSamp_indsOutOfExcNs.append(np.argwhere(np.in1d(excI, excNsEachSamp[ii])).squeeze()) # indeces of exc neurons (out of X_svm for exc neurons which includes only exc neurons) used to create XexcEq.
                if testIncorr:
                    XexcEq_incorr.append(X_svm_incorr[:, excNsEachSamp[ii],:])
        else: # set the rand exc subsamps
            excNsEachSamp = [] # numShufflesExc x frs x units x trials
            for ii in range(numShufflesExc):  # select n random exc (n = number of inh)         
                en = rng.permutation(excI)[0:lenInh].squeeze() # n randomly selected exc neurons.    
                Xexc = X_svm[:, en,:]
                XexcEq.append(Xexc)    
                excNsEachSamp.append(en) # indeces of exc neurons (out of X_svm which includes all neurons) used to create XexcEq.... you need this if you want to get svm projections for a particular exc shfl (eg w_data_exc[nExcShfl,:,:,:])
                excNsEachSamp_indsOutOfExcNs.append(np.argwhere(np.in1d(excI, en)).squeeze()) # indeces of exc neurons (out of X_svm for exc neurons which includes only exc neurons) used to create XexcEq.
                if testIncorr:
                    XexcEq_incorr.append(X_svm_incorr[:, en,:]) # incorr trials for the randomly selected subsample of exc neurons.   
                
    # set 2*lenInh population made of all exc neurons; as a control for the population made of half exc, half inh
    elif doInhAllexcEqexc[2] == 3: ###### allExc2inhSize      
        XallExc2inhSize = []
        excNsEachSamp_allExc2inhSize = []
        for ii in range(numShufflesExc):  # select n random exc (n = number of inh)         
            if len(excI) < lenInh*2: # very unlikely... exc neurons are way more than inh.
                sys.exit('excI < lenInh*2 ... you need to subsample inh!') 
            en = rng.permutation(excI)[0:lenInh*2].squeeze() # n randomly selected exc neurons.... n=2*lenInh    
            Xexc = X_svm[:, en,:]
            XallExc2inhSize.append(Xexc)    
            excNsEachSamp_allExc2inhSize.append(en) # indeces of exc neurons (our of X_svm) used for svm training in each exc shfl (below).... you need this if you want to get svm projections for a particular exc shfl (eg w_data_exc[nExcShfl,:,:,:])

    
    
    
#%% Run SVM (regular case, decoding choice, not adding Ns 1 by 1)   
####################################################################################################################################
####################################################################################################################################
# ## Compute classification error of training and testing dataset for the following cases:
#     Here we train the classifier using the best c found above (when including all neurons in the decoder) but using different sets of neurons (and 90% training, 10% testing trials.) ... I think this is problematic to use bestc of a decoder trained on a different population!
#     1) All inhibitory neurons contribute to the decoder
#     2) N excitatory neurons contribute, where n = number of inhibitory neurons. 
#     3) All excitatory neurons contribute
####################################################################################################################################
####################################################################################################################################
#### 3 populations: inh, n exc, all exc
#### For each population: find bestc ; train SVM ; do cross validation (numSamples times) ; set null distribution (by shuffling trial labels)
####################################################################################################################################

if np.all([decodeOutcome==0, decodeStimCateg==0, addNs_roc==0, addNs_rand==0, decodeChoiceEqCorrIncorr==0]):   # run svm on all neurons (ie not adding neurons one by one!) 
    
    ######################## inh ########################   
    
    if doInhAllexcEqexc[0] == 1:
        Xinh = X_svm[:, inhRois==1,:]    
        if testIncorr:
            Xinh_incorr = X_svm_incorr[:, inhRois==1,:]    
            perClassErrorTrain_inh, perClassErrorTest_inh, wAllC_inh, bAllC_inh, cbestAll_inh, cbest_inh, cvect, \
            perClassErrorTestShfl_inh, perClassErrorTestChance_inh, testTrInds_allSamps_inh, Ytest_allSamps_inh, Ytest_hat_allSampsFrs_inh0, trsnow_allSamps, \
            perClassErrorTest_incorr_inh, perClassErrorTest_shfl_incorr_inh, perClassErrorTest_chance_incorr_inh, Ytest_hat_allSampsFrs_incorr_inh \
            = setbesc_frs(Xinh,Y_svm,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest_inh,fr2an=np.nan, shflTrLabs=shflTrLabs, X_svm_incorr=Xinh_incorr, Y_svm_incorr=Y_svm_incorr, mnHRLR_acrossDays=mnHRLR_acrossDays) # outputs have size the number of shuffles in setbestc (shuffles per c value)
        else:
            perClassErrorTrain_inh, perClassErrorTest_inh, wAllC_inh, bAllC_inh, cbestAll_inh, cbest_inh, cvect, \
            perClassErrorTestShfl_inh, perClassErrorTestChance_inh, testTrInds_allSamps_inh, Ytest_allSamps_inh, Ytest_hat_allSampsFrs_inh0, trsnow_allSamps \
            = setbesc_frs(Xinh,Y_svm,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest_inh,fr2an=np.nan, shflTrLabs=shflTrLabs, X_svm_incorr=0, Y_svm_incorr=0, mnHRLR_acrossDays=mnHRLR_acrossDays) # outputs have size the number of shuffles in setbestc (shuffles per c value)
        
        # Set the parameters below at bestc (for all samples):
        perClassErrorTrain_data_inh = np.full((numSamples, nFrs), np.nan)
        perClassErrorTest_data_inh = np.full((numSamples, nFrs), np.nan)
        perClassErrorTest_shfl_inh = np.full((numSamples, nFrs), np.nan)
        perClassErrorTest_chance_inh = np.full((numSamples, nFrs), np.nan)
        w_data_inh = np.full((numSamples, lenInh, nFrs), np.nan)
        b_data_inh = np.full((numSamples, nFrs), np.nan)
        Ytest_hat_allSampsFrs_inh = np.full((numSamples, nFrs, Ytest_hat_allSampsFrs_inh0.shape[-1]), np.nan).squeeze() # squeeze helps if len_test=1                
        if testIncorr:
            perClassErrorTest_data_inh_incorr = np.full((numSamples, nFrs), np.nan)
            perClassErrorTest_shfl_inh_incorr = np.full((numSamples, nFrs), np.nan)
            perClassErrorTest_chance_inh_incorr = np.full((numSamples, nFrs), np.nan)
            Ytest_hat_allSampsFrs_inh_incorr = np.full((numSamples, nFrs, Ytest_hat_allSampsFrs_incorr_inh.shape[-1]), np.nan).squeeze() # squeeze helps if len_test=1        
        
        for ifr in range(nFrs):   
            if cbestKnown:
                indBestC = 0
            else:
                indBestC = np.in1d(cvect, cbest_inh[ifr])
            perClassErrorTrain_data_inh[:,ifr] = perClassErrorTrain_inh[:,indBestC,ifr].squeeze()
            perClassErrorTest_data_inh[:,ifr] = perClassErrorTest_inh[:,indBestC,ifr].squeeze()
            perClassErrorTest_shfl_inh[:,ifr] = perClassErrorTestShfl_inh[:,indBestC,ifr].squeeze()
            perClassErrorTest_chance_inh[:,ifr] = perClassErrorTestChance_inh[:,indBestC,ifr].squeeze()
            w_data_inh[:,:,ifr] = wAllC_inh[:,indBestC,:,ifr].squeeze()
            b_data_inh[:,ifr] = bAllC_inh[:,indBestC,ifr].squeeze()
            Ytest_hat_allSampsFrs_inh[:,ifr] = Ytest_hat_allSampsFrs_inh0[:,indBestC,ifr,:].squeeze()
            if testIncorr:
                perClassErrorTest_data_inh_incorr[:,ifr] = perClassErrorTest_incorr_inh[:,indBestC,ifr].squeeze()
                perClassErrorTest_shfl_inh_incorr[:,ifr] = perClassErrorTest_shfl_incorr_inh[:,indBestC,ifr].squeeze()
                perClassErrorTest_chance_inh_incorr[:,ifr] = perClassErrorTest_chance_incorr_inh[:,indBestC,ifr].squeeze()            
                Ytest_hat_allSampsFrs_inh_incorr[:,ifr] = Ytest_hat_allSampsFrs_incorr_inh[:,indBestC,ifr,:].squeeze()


            
    ######################## allExc (n = num all exc neurons) or allN (all exc, inh, unsure neurons) ########################
    
    elif np.logical_or(doInhAllexcEqexc[1] == 1 , doInhAllexcEqexc[1] == 2): 
        if doInhAllexcEqexc[1] == 1: # allExc
            XallExc = X_svm[:, inhRois==0,:]
            if testIncorr:
                XallExc_incorr = X_svm_incorr[:, inhRois==0,:]
        elif doInhAllexcEqexc[1] == 2: # allN
            XallExc = X_svm
            if testIncorr:
                XallExc_incorr = X_svm_incorr
            
        if testIncorr:
            perClassErrorTrain_allExc, perClassErrorTest_allExc, wAllC_allExc, bAllC_allExc, cbestAll_allExc, cbest_allExc, cvect, \
            perClassErrorTestShfl_allExc, perClassErrorTestChance_allExc, testTrInds_allSamps_allExc, Ytest_allSamps_allExc, Ytest_hat_allSampsFrs_allExc0, trsnow_allSamps, \
            perClassErrorTest_incorr_allExc, perClassErrorTest_shfl_incorr_allExc, perClassErrorTest_chance_incorr_allExc, Ytest_hat_allSampsFrs_incorr_allExc \
            = setbesc_frs(XallExc,Y_svm,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest_allExc,fr2an=np.nan, shflTrLabs=shflTrLabs, X_svm_incorr=XallExc_incorr, Y_svm_incorr=Y_svm_incorr, mnHRLR_acrossDays=mnHRLR_acrossDays) # outputs have size the number of shuffles in setbestc (shuffles per c value)
        else:
            perClassErrorTrain_allExc, perClassErrorTest_allExc, wAllC_allExc, bAllC_allExc, cbestAll_allExc, cbest_allExc, cvect, \
            perClassErrorTestShfl_allExc, perClassErrorTestChance_allExc, testTrInds_allSamps_allExc, Ytest_allSamps_allExc, Ytest_hat_allSampsFrs_allExc0, trsnow_allSamps \
            = setbesc_frs(XallExc,Y_svm,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest_allExc,fr2an=np.nan, shflTrLabs=shflTrLabs, X_svm_incorr=0, Y_svm_incorr=0, mnHRLR_acrossDays=mnHRLR_acrossDays) # outputs have size the number of shuffles in setbestc (shuffles per c value)
        
        # Take the values at bestc that you computed above (for all samples)
        perClassErrorTrain_data_allExc = np.full((numSamples, nFrs), np.nan)
        perClassErrorTest_data_allExc = np.full((numSamples, nFrs), np.nan)
        perClassErrorTest_shfl_allExc = np.full((numSamples, nFrs), np.nan)
        perClassErrorTest_chance_allExc = np.full((numSamples, nFrs), np.nan)
        w_data_allExc = np.full((numSamples, XallExc.shape[1], nFrs), np.nan)
        b_data_allExc = np.full((numSamples, nFrs), np.nan)
        Ytest_hat_allSampsFrs_allExc = np.full((numSamples, nFrs, Ytest_hat_allSampsFrs_allExc0.shape[-1]), np.nan).squeeze() # squeeze helps if len_test=1
        if testIncorr:
            perClassErrorTest_data_allExc_incorr = np.full((numSamples, nFrs), np.nan)
            perClassErrorTest_shfl_allExc_incorr = np.full((numSamples, nFrs), np.nan)
            perClassErrorTest_chance_allExc_incorr = np.full((numSamples, nFrs), np.nan)
            Ytest_hat_allSampsFrs_allExc_incorr = np.full((numSamples, nFrs, Ytest_hat_allSampsFrs_incorr_allExc.shape[-1]), np.nan).squeeze() # squeeze helps if len_test=1   
            
        for ifr in range(nFrs):    
            if cbestKnown:
                indBestC = 0
            else:        
                indBestC = np.in1d(cvect, cbest_allExc[ifr])
            perClassErrorTrain_data_allExc[:,ifr] = perClassErrorTrain_allExc[:,indBestC,ifr].squeeze()
            perClassErrorTest_data_allExc[:,ifr] = perClassErrorTest_allExc[:,indBestC,ifr].squeeze()
            perClassErrorTest_shfl_allExc[:,ifr] = perClassErrorTestShfl_allExc[:,indBestC,ifr].squeeze()
            perClassErrorTest_chance_allExc[:,ifr] = perClassErrorTestChance_allExc[:,indBestC,ifr].squeeze()
            w_data_allExc[:,:,ifr] = wAllC_allExc[:,indBestC,:,ifr].squeeze()
            b_data_allExc[:,ifr] = bAllC_allExc[:,indBestC,ifr].squeeze()
            Ytest_hat_allSampsFrs_allExc[:,ifr] = Ytest_hat_allSampsFrs_allExc0[:,indBestC,ifr,:].squeeze()
            if testIncorr:
                perClassErrorTest_data_allExc_incorr[:,ifr] = perClassErrorTest_incorr_allExc[:,indBestC,ifr].squeeze()
                perClassErrorTest_shfl_allExc_incorr[:,ifr] = perClassErrorTest_shfl_incorr_allExc[:,indBestC,ifr].squeeze()
                perClassErrorTest_chance_allExc_incorr[:,ifr] = perClassErrorTest_chance_incorr_allExc[:,indBestC,ifr].squeeze()            
                Ytest_hat_allSampsFrs_allExc_incorr[:,ifr] = Ytest_hat_allSampsFrs_incorr_allExc[:,indBestC,ifr,:].squeeze()
                
            
    
    ######################### n exc (n=lenInh) (pick random sets of n exc neurons (do it numShufflesEx times)) ########################
    
    elif doInhAllexcEqexc[2] == 1: 
        if cbestKnown:
            cbestAll_exc = cbest_exc
        else:
            cbestAll_exc = []
            cbest_exc = []
        perClassErrorTrain_data_exc = []
        perClassErrorTest_data_exc = []
        perClassErrorTest_shfl_exc = []
        perClassErrorTest_chance_exc = []
        w_data_exc = []
        b_data_exc = []
        Ytest_hat_allSampsFrs_exc = []
        testTrInds_allSamps_exc = []
        Ytest_allSamps_exc = []
        trsnow_allSamps = []
        if testIncorr:
            perClassErrorTest_data_exc_incorr = []
            perClassErrorTest_shfl_exc_incorr = []
            perClassErrorTest_chance_exc_incorr = []
            Ytest_hat_allSampsFrs_exc_incorr = []
                
        for ii in range(numShufflesExc): # loop through exc subsamples   
            print '\n\n-------- Exc subsample %d --------' %(ii)
            if cbestKnown:
                cbest_now = cbest_exc[ii]
            
            Xnow = XexcEq[ii]
            if testIncorr:
                Xnow_incorr = XexcEq_incorr[ii]
                perClassErrorTrain_exc, perClassErrorTest_exc, wAllC_exc, bAllC_exc, cbestAll_exc0, cbest_exc0, cvect, \
                perClassErrorTestShfl_exc, perClassErrorTestChance_exc, testTrInds_allSamps_exc00, Ytest_allSamps_exc00, Ytest_hat_allSampsFrs_exc00, trsnow_allSamps00, \
                perClassErrorTest_incorr_exc, perClassErrorTest_shfl_incorr_exc, perClassErrorTest_chance_incorr_exc, Ytest_hat_allSampsFrs_incorr_exc \
                = setbesc_frs(Xnow,Y_svm,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest_now,fr2an=np.nan, shflTrLabs=shflTrLabs, X_svm_incorr=Xnow_incorr, Y_svm_incorr=Y_svm_incorr, mnHRLR_acrossDays=mnHRLR_acrossDays) # outputs have size the number of shuffles in setbestc (shuffles per c value)
            else:
                perClassErrorTrain_exc, perClassErrorTest_exc, wAllC_exc, bAllC_exc, cbestAll_exc0, cbest_exc0, cvect, \
                perClassErrorTestShfl_exc, perClassErrorTestChance_exc, testTrInds_allSamps_exc00, Ytest_allSamps_exc00, Ytest_hat_allSampsFrs_exc00, trsnow_allSamps00 \
                = setbesc_frs(Xnow,Y_svm,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest_now,fr2an=np.nan, shflTrLabs=shflTrLabs, X_svm_incorr=0, Y_svm_incorr=0, mnHRLR_acrossDays=mnHRLR_acrossDays) # outputs have size the number of shuffles in setbestc (shuffles per c value)
        
            # Take the values at bestc that you computed above (for all samples)
            perClassErrorTrain_data_exc0 = np.full((numSamples, nFrs), np.nan)
            perClassErrorTest_data_exc0 = np.full((numSamples, nFrs), np.nan)
            perClassErrorTest_shfl_exc0 = np.full((numSamples, nFrs), np.nan)
            perClassErrorTest_chance_exc0 = np.full((numSamples, nFrs), np.nan)
            w_data_exc0 = np.full((numSamples, Xnow.shape[1], nFrs), np.nan)
            b_data_exc0 = np.full((numSamples, nFrs), np.nan)
            Ytest_hat_allSampsFrs_exc0 = np.full((numSamples, nFrs, Ytest_hat_allSampsFrs_exc00.shape[-1]), np.nan).squeeze() # squeeze helps if len_test=1
            if testIncorr:
                perClassErrorTest_data_exc0_incorr = np.full((numSamples, nFrs), np.nan)
                perClassErrorTest_shfl_exc0_incorr = np.full((numSamples, nFrs), np.nan)
                perClassErrorTest_chance_exc0_incorr = np.full((numSamples, nFrs), np.nan)
                Ytest_hat_allSampsFrs_exc0_incorr = np.full((numSamples, nFrs, Ytest_hat_allSampsFrs_incorr_exc.shape[-1]), np.nan).squeeze()            
                
            for ifr in range(nFrs):    
                if cbestKnown:
                    indBestC = 0
                else:
                    indBestC = np.in1d(cvect, cbest_exc0[ifr])
                perClassErrorTrain_data_exc0[:,ifr] = perClassErrorTrain_exc[:,indBestC,ifr].squeeze()
                perClassErrorTest_data_exc0[:,ifr] = perClassErrorTest_exc[:,indBestC,ifr].squeeze()
                perClassErrorTest_shfl_exc0[:,ifr] = perClassErrorTestShfl_exc[:,indBestC,ifr].squeeze()
                perClassErrorTest_chance_exc0[:,ifr] = perClassErrorTestChance_exc[:,indBestC,ifr].squeeze()
                w_data_exc0[:,:,ifr] = wAllC_exc[:,indBestC,:,ifr].squeeze()
                b_data_exc0[:,ifr] = bAllC_exc[:,indBestC,ifr].squeeze()
                Ytest_hat_allSampsFrs_exc0[:,ifr] = Ytest_hat_allSampsFrs_exc00[:,indBestC,ifr,:].squeeze()
                if testIncorr:
                    perClassErrorTest_data_exc0_incorr[:,ifr] = perClassErrorTest_incorr_exc[:,indBestC,ifr].squeeze()
                    perClassErrorTest_shfl_exc0_incorr[:,ifr] = perClassErrorTest_shfl_incorr_exc[:,indBestC,ifr].squeeze()
                    perClassErrorTest_chance_exc0_incorr[:,ifr] = perClassErrorTest_chance_incorr_exc[:,indBestC,ifr].squeeze()            
                    Ytest_hat_allSampsFrs_exc0_incorr[:,ifr] = Ytest_hat_allSampsFrs_incorr_exc[:,indBestC,ifr,:].squeeze()
                
            # collect values of all exc shuffles (at bestc)   
            if cbestKnown==False:
                cbestAll_exc.append(cbestAll_exc0)
                cbest_exc.append(cbest_exc0)
            perClassErrorTrain_data_exc.append(perClassErrorTrain_data_exc0) # numShufflesExc x numSamples x numFrames
            perClassErrorTest_data_exc.append(perClassErrorTest_data_exc0)
            perClassErrorTest_shfl_exc.append(perClassErrorTest_shfl_exc0)
            perClassErrorTest_chance_exc.append(perClassErrorTest_chance_exc0)
            w_data_exc.append(w_data_exc0) # numShufflesExc x numSamples x numNeurons x numFrames
            b_data_exc.append(b_data_exc0)
            Ytest_hat_allSampsFrs_exc.append(Ytest_hat_allSampsFrs_exc0)
            testTrInds_allSamps_exc.append(testTrInds_allSamps_exc00)
            Ytest_allSamps_exc.append(Ytest_allSamps_exc00) # numShufflesExc x numSamples x numTestTrs
            trsnow_allSamps.append(trsnow_allSamps00)
            if testIncorr:
                perClassErrorTest_data_exc_incorr.append(perClassErrorTest_data_exc0_incorr)
                perClassErrorTest_shfl_exc_incorr.append(perClassErrorTest_shfl_exc0_incorr)
                perClassErrorTest_chance_exc_incorr.append(perClassErrorTest_chance_exc0_incorr)
                Ytest_hat_allSampsFrs_exc_incorr.append(Ytest_hat_allSampsFrs_exc0_incorr)



    
    #####################################################################################
    ######################### half exc and half inh (n = lenInh) ########################
    #####################################################################################
    
    elif doInhAllexcEqexc[2] == 2: 
        if cbestKnown==False:
            cbestAll_excInhHalf = []
            cbest_excInhHalf = []
        perClassErrorTrain_data_excInhHalf = []
        perClassErrorTest_data_excInhHalf = []
        perClassErrorTest_shfl_excInhHalf = []
        perClassErrorTest_chance_excInhHalf = []
        w_data_excInhHalf = []
        b_data_excInhHalf = []
        
        for ii in range(numShufflesExc):    
            print '\n\n-------- Exc subsample %d --------' %(ii)
            
            Xnow = np.concatenate((XexcEq[ii], X_svm[:, inhRois==1,:]), axis=1)  # use half exc, and half inh in the population          
    
            # Set bestc    
        #    perClassErrorTrain_exc, perClassErrorTest_exc, wAllC_exc, bAllC_exc, cbestAll_exc0, cbest_exc0, cvect = setbesc(XexcEq[ii],Y,regType,kfold,numDataPoints,numSamples,doPlots)
            perClassErrorTrain_exc, perClassErrorTest_exc, wAllC_exc, bAllC_exc, cbestAll_exc0, cbest_exc0, cvect, perClassErrorTestShfl_exc, perClassErrorTestChance_exc,_,_,_,_ = setbesc_frs(Xnow,Y_svm,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest_now,fr2an=np.nan, shflTrLabs=shflTrLabs) # outputs have size the number of shuffles in setbestc (shuffles per c value)
        
            # Take the values at bestc that you computed above (for all samples)
            perClassErrorTrain_data_exc0 = np.full((numSamples, nFrs), np.nan)
            perClassErrorTest_data_exc0 = np.full((numSamples, nFrs), np.nan)
            perClassErrorTest_shfl_exc0 = np.full((numSamples, nFrs), np.nan)
            perClassErrorTest_chance_exc0 = np.full((numSamples, nFrs), np.nan)
            w_data_exc0 = np.full((numSamples, Xnow.shape[1], nFrs), np.nan)
            b_data_exc0 = np.full((numSamples, nFrs), np.nan)
            for ifr in range(nFrs):    
                if cbestKnown:
                    indBestC = 0
                else:
                    indBestC = np.in1d(cvect, cbest_exc0[ifr])
                perClassErrorTrain_data_exc0[:,ifr] = perClassErrorTrain_exc[:,indBestC,ifr].squeeze()
                perClassErrorTest_data_exc0[:,ifr] = perClassErrorTest_exc[:,indBestC,ifr].squeeze()
                perClassErrorTest_shfl_exc0[:,ifr] = perClassErrorTestShfl_exc[:,indBestC,ifr].squeeze()
                perClassErrorTest_chance_exc0[:,ifr] = perClassErrorTestChance_exc[:,indBestC,ifr].squeeze()
                w_data_exc0[:,:,ifr] = wAllC_exc[:,indBestC,:,ifr].squeeze()
                b_data_exc0[:,ifr] = bAllC_exc[:,indBestC,ifr].squeeze()
            
            # collect values of all exc shuffles    
            if cbestKnown==False:
                cbestAll_excInhHalf.append(cbestAll_exc0)
                cbest_excInhHalf.append(cbest_exc0)
            perClassErrorTrain_data_excInhHalf.append(perClassErrorTrain_data_exc0) # numShufflesExc x numSamples x numFrames
            perClassErrorTest_data_excInhHalf.append(perClassErrorTest_data_exc0)
            perClassErrorTest_shfl_excInhHalf.append(perClassErrorTest_shfl_exc0)
            perClassErrorTest_chance_excInhHalf.append(perClassErrorTest_chance_exc0)
            w_data_excInhHalf.append(w_data_exc0) # numShufflesExc x numSamples x numNeurons x numFrames
            b_data_excInhHalf.append(b_data_exc0)
            
            
    #####################################################################################        
    ######################### 2n exc (n = lenInh) ######################## ###### allExc2inhSize  # run svm on 2*lenInh population size made of all exc neurons; as a control for the population made of half exc, half inh        
    #####################################################################################
    
    elif doInhAllexcEqexc[2] == 3:         
        if cbestKnown==False:
            cbestAll_allExc2inhSize = []
            cbest_allExc2inhSize = []
        perClassErrorTrain_data_allExc2inhSize = []
        perClassErrorTest_data_allExc2inhSize = []
        perClassErrorTest_shfl_allExc2inhSize = []
        perClassErrorTest_chance_allExc2inhSize = []
        w_data_allExc2inhSize = []
        b_data_allExc2inhSize = []
    
        for ii in range(numShufflesExc):    
            print '\n\n-------- Exc subsample %d --------' %(ii) 
            Xnow = XallExc2inhSize[ii] 
    
            perClassErrorTrain_exc, perClassErrorTest_exc, wAllC_exc, bAllC_exc, cbestAll_exc0, cbest_exc0, cvect, perClassErrorTestShfl_exc, perClassErrorTestChance_exc,_,_,_,_ = setbesc_frs(Xnow,Y_svm,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest_now,fr2an=np.nan, shflTrLabs=shflTrLabs) # outputs have size the number of shuffles in setbestc (shuffles per c value)
        
            # Take the values at bestc that you computed above (for all samples)
            perClassErrorTrain_data_exc0 = np.full((numSamples, nFrs), np.nan)
            perClassErrorTest_data_exc0 = np.full((numSamples, nFrs), np.nan)
            perClassErrorTest_shfl_exc0 = np.full((numSamples, nFrs), np.nan)
            perClassErrorTest_chance_exc0 = np.full((numSamples, nFrs), np.nan)
            w_data_exc0 = np.full((numSamples, Xnow.shape[1], nFrs), np.nan)
            b_data_exc0 = np.full((numSamples, nFrs), np.nan)
            for ifr in range(nFrs):    
                if cbestKnown:
                    indBestC = 0
                else:
                    indBestC = np.in1d(cvect, cbest_exc0[ifr])
                perClassErrorTrain_data_exc0[:,ifr] = perClassErrorTrain_exc[:,indBestC,ifr].squeeze()
                perClassErrorTest_data_exc0[:,ifr] = perClassErrorTest_exc[:,indBestC,ifr].squeeze()
                perClassErrorTest_shfl_exc0[:,ifr] = perClassErrorTestShfl_exc[:,indBestC,ifr].squeeze()
                perClassErrorTest_chance_exc0[:,ifr] = perClassErrorTestChance_exc[:,indBestC,ifr].squeeze()
                w_data_exc0[:,:,ifr] = wAllC_exc[:,indBestC,:,ifr].squeeze()
                b_data_exc0[:,ifr] = bAllC_exc[:,indBestC,ifr].squeeze()
            
            # collect values of all exc shuffles    
            if cbestKnown==False:
                cbestAll_allExc2inhSize.append(cbestAll_exc0)
                cbest_allExc2inhSize.append(cbest_exc0)
            perClassErrorTrain_data_allExc2inhSize.append(perClassErrorTrain_data_exc0) # numShufflesExc x numSamples x numFrames
            perClassErrorTest_data_allExc2inhSize.append(perClassErrorTest_data_exc0)
            perClassErrorTest_shfl_allExc2inhSize.append(perClassErrorTest_shfl_exc0)
            perClassErrorTest_chance_allExc2inhSize.append(perClassErrorTest_chance_exc0)
            w_data_allExc2inhSize.append(w_data_exc0) # numShufflesExc x numSamples x numNeurons x numFrames
            b_data_allExc2inhSize.append(b_data_exc0)
            
        '''
        # I don't think I need this!
        # transpose so the sizes match _inh and _allExc as well as shuffle vars (below)        
        perClassErrorTrain_data_exc = np.transpose(perClassErrorTrain_data_exc) # numSamples x numShufflesExc
        perClassErrorTest_data_exc = np.transpose(perClassErrorTest_data_exc)
        w_data_exc = np.transpose(w_data_exc,(1,0,2)) # numSamples x numShufflesExc x numNeur
        b_data_exc = np.transpose(b_data_exc)
        '''    
'''
plt.plot(time_trace, 100 - np.mean(perClassErrorTrain_data_allExc, axis=0))
plt.plot(time_trace, 100 - np.mean(perClassErrorTest_data_allExc, axis=0))

plt.plot(time_trace, 100 - np.mean(perClassErrorTrain_data_allExc, axis=(0,1)))
plt.plot(time_trace, 100 - np.mean(perClassErrorTest_data_allExc, axis=(0,1)))
'''

################################################################################################################################
################################################################################################################################
#%% Same as above but run SVM for populations of increasingly bigger size (adding neurons 1 by 1 based on their tuning strength)

if addNs_roc:

    frROC = eventI_ds-1 # use choice tuning at time point fr2an to sort neurons (time window before the choice)
    # what frames to run SVM on? # set to np.nan if you want to run SVM on all frames.        
    frs = eventI_ds-1 # np.nan 
    if np.isnan(frs).all(): # run SVM on all frames
        frs = range(nFrs)
 
 
   ######################## inh ########################   
    if doInhAllexcEqexc[0] == 1:    
        
        Xinh0 = X_svm[:, inhRois==1,:]
        inh_roc_sort = np.argsort(choicePref_inh[frROC].squeeze()) # index of neurosn from low to high ROC values
        if doInhAllexcEqexc[3]==1:
            inh_roc_sort = inh_roc_sort[::-1] # index of neurosn from high to low ROC values
            
        # Loop over number of neurons in the population: from 1 to the entire population size        
        perClassErrorTrain_data_inh_numNs = []
        perClassErrorTest_data_inh_numNs = []
        perClassErrorTest_shfl_inh_numNs = []
        perClassErrorTest_chance_inh_numNs = []
        w_data_inh_numNs = []
        b_data_inh_numNs = []
        cbest_inh_numNs = []

        for numNs in np.arange(1,lenInh+1): 
            Xinh = Xinh0[:,inh_roc_sort[0:numNs],:] # neurons are added based on their tuning (abs ROC AUC): the highest tuning is added first            
            cbest_inh = np.nan  # set to nan, so in the function below we set cbest... unless you think you will run addNs_roc again... which I guess you wont!  
            # perClassErrorTrain_inh: samples x cvals x frames
            # wAllC_inh: samples x cvals x nNeurons x frames
            perClassErrorTrain_inh, perClassErrorTest_inh, wAllC_inh, bAllC_inh, cbestAll_inh, cbest_inh, cvect, perClassErrorTestShfl_inh, perClassErrorTestChance_inh,_,_,_,_ = setbesc_frs(Xinh,Y_svm,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest_inh,frs,shflTrLabs=shflTrLabs) # outputs have size the number of shuffles in setbestc (shuffles per c value)
       
            # Set the vars at bestc (for all samples):
            perClassErrorTrain_data_inh = np.full((numSamples, nFrs), np.nan) # samples x frames
            perClassErrorTest_data_inh = np.full((numSamples, nFrs), np.nan)
            perClassErrorTest_shfl_inh = np.full((numSamples, nFrs), np.nan)
            perClassErrorTest_chance_inh = np.full((numSamples, nFrs), np.nan)
            w_data_inh = np.full((numSamples, Xinh.shape[1], nFrs), np.nan)  # samples x nNs x frames
            b_data_inh = np.full((numSamples, nFrs), np.nan)
            
            for ifr in frs: #: range(nFrs):   
#                if cbestKnown:
#                    indBestC = 0
#                else:
                indBestC = np.in1d(cvect, cbest_inh[ifr])
                    
                perClassErrorTrain_data_inh[:,ifr] = perClassErrorTrain_inh[:,indBestC,ifr].squeeze()
                perClassErrorTest_data_inh[:,ifr] = perClassErrorTest_inh[:,indBestC,ifr].squeeze()
                perClassErrorTest_shfl_inh[:,ifr] = perClassErrorTestShfl_inh[:,indBestC,ifr].squeeze()
                perClassErrorTest_chance_inh[:,ifr] = perClassErrorTestChance_inh[:,indBestC,ifr].squeeze()
                w_data_inh[:,:,ifr] = wAllC_inh[:,indBestC,:,ifr]#.squeeze()
                b_data_inh[:,ifr] = bAllC_inh[:,indBestC,ifr].squeeze()
                
            # collect vars for different number of neurons in the decoder
            perClassErrorTrain_data_inh_numNs.append(perClassErrorTrain_data_inh) # number of neurons in the decoder x nSamps x nFrs ... if SVM was run on only 1 frame then values of all other frames will be nan
            perClassErrorTest_data_inh_numNs.append(perClassErrorTest_data_inh)
            perClassErrorTest_shfl_inh_numNs.append(perClassErrorTest_shfl_inh)
            perClassErrorTest_chance_inh_numNs.append(perClassErrorTest_chance_inh)
            w_data_inh_numNs.append(np.transpose(w_data_inh, (1,0,2))) # otherwise np.array and scio.savemat wont work... bc the first two dimensions will be the same for all elements but not the rest... and it will try to combine them, but it cant, so I make sure the 2nd dimension is nNs_inTheDecoder instead of nSamps
            b_data_inh_numNs.append(b_data_inh)
            cbest_inh_numNs.append(cbest_inh)
        
        perClassErrorTrain_data_inh_numNs = np.array(perClassErrorTrain_data_inh_numNs)
        perClassErrorTest_data_inh_numNs = np.array(perClassErrorTest_data_inh_numNs)
        perClassErrorTest_shfl_inh_numNs = np.array(perClassErrorTest_shfl_inh_numNs)
        perClassErrorTest_chance_inh_numNs = np.array(perClassErrorTest_chance_inh_numNs)
        w_data_inh_numNs = np.array(w_data_inh_numNs)
        b_data_inh_numNs = np.array(b_data_inh_numNs)
        cbest_inh_numNs = np.array(cbest_inh_numNs)
            
            
            
    ######################## allExc (n = num all exc neurons) ########################
    elif doInhAllexcEqexc[1] == 1: 
        
        XallExc0 = X_svm[:, inhRois==0,:]
        allExc_roc_sort = np.argsort(choicePref_exc[frROC].squeeze()) # index of neurosn from low to high ROC values
        if doInhAllexcEqexc[3]==1:
            allExc_roc_sort = allExc_roc_sort[::-1] # index of neurosn from high to low ROC values
    
        # Loop over number of neurons in the population: from 1 to the entire population size        
        perClassErrorTrain_data_allExc_numNs = []
        perClassErrorTest_data_allExc_numNs = []
        perClassErrorTest_shfl_allExc_numNs = []
        perClassErrorTest_chance_allExc_numNs = []
        w_data_allExc_numNs = []
        b_data_allExc_numNs = []
        cbest_allExc_numNs = []

        for numNs in np.arange(1,XallExc0.shape[1]+1): 
            XallExc = XallExc0[:,allExc_roc_sort[0:numNs],:] # neurons are added based on their tuning (abs ROC AUC): the highest tuning is added first            
            cbest_allExc = np.nan            
            # perClassErrorTrain_allExc: samples x cvals x frames        
            perClassErrorTrain_allExc, perClassErrorTest_allExc, wAllC_allExc, bAllC_allExc, cbestAll_allExc, cbest_allExc, cvect, perClassErrorTestShfl_allExc, perClassErrorTestChance_allExc,_,_,_,_ = setbesc_frs(XallExc,Y_svm,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest_allExc,frs,shflTrLabs=shflTrLabs) # outputs have size the number of shuffles in setbestc (shuffles per c value)
            
            # Take the values at bestc that you computed above (for all samples)
            perClassErrorTrain_data_allExc = np.full((numSamples, nFrs), np.nan)
            perClassErrorTest_data_allExc = np.full((numSamples, nFrs), np.nan)
            perClassErrorTest_shfl_allExc = np.full((numSamples, nFrs), np.nan)
            perClassErrorTest_chance_allExc = np.full((numSamples, nFrs), np.nan)
            w_data_allExc = np.full((numSamples, XallExc.shape[1], nFrs), np.nan)
            b_data_allExc = np.full((numSamples, nFrs), np.nan)
            for ifr in frs: #range(nFrs):    
#                if cbestKnown:
#                    indBestC = 0
#                else:        
                indBestC = np.in1d(cvect, cbest_allExc[ifr])
                
                perClassErrorTrain_data_allExc[:,ifr] = perClassErrorTrain_allExc[:,indBestC,ifr].squeeze()
                perClassErrorTest_data_allExc[:,ifr] = perClassErrorTest_allExc[:,indBestC,ifr].squeeze()
                perClassErrorTest_shfl_allExc[:,ifr] = perClassErrorTestShfl_allExc[:,indBestC,ifr].squeeze()
                perClassErrorTest_chance_allExc[:,ifr] = perClassErrorTestChance_allExc[:,indBestC,ifr].squeeze()
                w_data_allExc[:,:,ifr] = wAllC_allExc[:,indBestC,:,ifr]#.squeeze()
                b_data_allExc[:,ifr] = bAllC_allExc[:,indBestC,ifr].squeeze()
            
            # collect vars for different number of neurons in the decoder
            perClassErrorTrain_data_allExc_numNs.append(perClassErrorTrain_data_allExc) # number of neurons in the decoder x nSamps x nFrs ... if SVM was run on only 1 frame then values of all other frames will be nan
            perClassErrorTest_data_allExc_numNs.append(perClassErrorTest_data_allExc)
            perClassErrorTest_shfl_allExc_numNs.append(perClassErrorTest_shfl_allExc)
            perClassErrorTest_chance_allExc_numNs.append(perClassErrorTest_chance_allExc)
            w_data_allExc_numNs.append(np.transpose(w_data_allExc, (1,0,2))) # otherwise np.array and scio.savemat wont work... bc the first two dimensions will be the same for all elements but not the rest... and it will try to combine them, but it cant, so I make sure the 2nd dimension is nNs_inTheDecoder instead of nSamps
            b_data_allExc_numNs.append(b_data_allExc)
            cbest_allExc_numNs.append(cbest_allExc)
        
        perClassErrorTrain_data_allExc_numNs = np.array(perClassErrorTrain_data_allExc_numNs)
        perClassErrorTest_data_allExc_numNs = np.array(perClassErrorTest_data_allExc_numNs)
        perClassErrorTest_shfl_allExc_numNs = np.array(perClassErrorTest_shfl_allExc_numNs)
        perClassErrorTest_chance_allExc_numNs = np.array(perClassErrorTest_chance_allExc_numNs)
        w_data_allExc_numNs = np.array(w_data_allExc_numNs)
        b_data_allExc_numNs = np.array(b_data_allExc_numNs)
        cbest_allExc_numNs = np.array(cbest_allExc_numNs)
   
#    c = []
#    for ii in range(numShufflesExc):
#        print '\n\n-------- Exc subsample %d --------' %(ii)
#        
#        # Take a subsample of exc neurons
#        Xnow = XexcEq[ii]
#        # Set indeces (in Xnow) to sort it based on choice tuning from high to low
#        cp = choicePref_exc[frROC, excNsEachSamp_indsOutOfExcNs[ii]] 
#        exc_roc_sort = np.argsort(cp) # index of neurons in XexcEq[ii] from low to high ROC values
#        c.append(cp[exc_roc_sort])
            
    ######################### n exc (n=lenInh) (pick random sets of n exc neurons (do it numShufflesEx times)) ########################
    # first subsample exc neurons, then sort them based on ROC values; finally run svm.
    elif doInhAllexcEqexc[2] == 1:         
        perClassErrorTrain_data_exc_numNs = []
        perClassErrorTest_data_exc_numNs = []
        perClassErrorTest_shfl_exc_numNs = []
        perClassErrorTest_chance_exc_numNs = []
        w_data_exc_numNs = []
        b_data_exc_numNs = []
        cbest_exc_numNs = []
#        cbestAll_exc_numNs = []        
        for ii in range(numShufflesExc):    
            print '\n\n-------- Exc subsample %d --------' %(ii)
            
            # Take a subsample of exc neurons
            Xnow = XexcEq[ii]
            # Set indeces (in Xnow) to sort it based on choice tuning from high to low
            cp = choicePref_exc[frROC, excNsEachSamp_indsOutOfExcNs[ii]] # choice preference of neurons that are in Xnow
            exc_roc_sort = np.argsort(cp) # index of neurons in XexcEq[ii] from low to high ROC values
            if doInhAllexcEqexc[3]==1:
                exc_roc_sort = exc_roc_sort[::-1] # index of neurons in XexcEq[ii] from high to low ROC values                
        
            # Loop over number of neurons in the population: from 1 to the entire population size        
            perClassErrorTrain_data_exc_numNs0 = []
            perClassErrorTest_data_exc_numNs0 = []
            perClassErrorTest_shfl_exc_numNs0 = []
            perClassErrorTest_chance_exc_numNs0 = []
            w_data_exc_numNs0 = []
            b_data_exc_numNs0 = []
            cbest_exc_numNs0 = []
    
            for numNs in np.arange(1,Xnow.shape[1]+1): 
                # set X_svm for the numNs neurons with the highest choice tuning
                XExc = Xnow[:,exc_roc_sort[0:numNs],:] # neurons are added based on their tuning (abs ROC AUC): the highest tuning is added first            
                cbest_now = np.nan                    
                
                perClassErrorTrain_exc, perClassErrorTest_exc, wAllC_exc, bAllC_exc, cbestAll_exc0, cbest_exc0, cvect, perClassErrorTestShfl_exc, perClassErrorTestChance_exc,_,_,_,_ = setbesc_frs(XExc,Y_svm,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest_now, frs, shflTrLabs=shflTrLabs) # outputs have size the number of shuffles in setbestc (shuffles per c value)
        
                # Take the values at bestc that you computed above (for all samples)
                perClassErrorTrain_data_exc0 = np.full((numSamples, nFrs), np.nan)
                perClassErrorTest_data_exc0 = np.full((numSamples, nFrs), np.nan)
                perClassErrorTest_shfl_exc0 = np.full((numSamples, nFrs), np.nan)
                perClassErrorTest_chance_exc0 = np.full((numSamples, nFrs), np.nan)
                w_data_exc0 = np.full((numSamples, XExc.shape[1], nFrs), np.nan)
                b_data_exc0 = np.full((numSamples, nFrs), np.nan)
                for ifr in frs: # range(nFrs):    
#                    if cbestKnown:
#                        indBestC = 0
#                    else:
                    indBestC = np.in1d(cvect, cbest_exc0[ifr])
                        
                    perClassErrorTrain_data_exc0[:,ifr] = perClassErrorTrain_exc[:,indBestC,ifr].squeeze()
                    perClassErrorTest_data_exc0[:,ifr] = perClassErrorTest_exc[:,indBestC,ifr].squeeze()
                    perClassErrorTest_shfl_exc0[:,ifr] = perClassErrorTestShfl_exc[:,indBestC,ifr].squeeze()
                    perClassErrorTest_chance_exc0[:,ifr] = perClassErrorTestChance_exc[:,indBestC,ifr].squeeze()
                    w_data_exc0[:,:,ifr] = wAllC_exc[:,indBestC,:,ifr]#.squeeze()
                    b_data_exc0[:,ifr] = bAllC_exc[:,indBestC,ifr].squeeze()
            
                # collect vars for different number of neurons in the decoder
                perClassErrorTrain_data_exc_numNs0.append(perClassErrorTrain_data_exc0) # number of neurons in the decoder x nSamps x nFrs ... if SVM was run on only 1 frame then values of all other frames will be nan
                perClassErrorTest_data_exc_numNs0.append(perClassErrorTest_data_exc0)
                perClassErrorTest_shfl_exc_numNs0.append(perClassErrorTest_shfl_exc0)
                perClassErrorTest_chance_exc_numNs0.append(perClassErrorTest_chance_exc0)
                w_data_exc_numNs0.append(np.transpose(w_data_exc0, (1,0,2))) # otherwise np.array and scio.savemat wont work... bc the first two dimensions will be the same for all elements but not the rest... and it will try to combine them, but it cant, so I make sure the 2nd dimension is nNs_inTheDecoder instead of nSamps
                b_data_exc_numNs0.append(b_data_exc0)
                cbest_exc_numNs0.append(cbest_exc0) # number of neurons in the decoder x nFrs           
            
            
            # collect values of all exc shuffles    
            perClassErrorTrain_data_exc_numNs.append(perClassErrorTrain_data_exc_numNs0) # numShufflesExc x number of neurons in the decoder x numSamples x nFrs
            perClassErrorTest_data_exc_numNs.append(perClassErrorTest_data_exc_numNs0)
            perClassErrorTest_shfl_exc_numNs.append(perClassErrorTest_shfl_exc_numNs0)
            perClassErrorTest_chance_exc_numNs.append(perClassErrorTest_chance_exc_numNs0)
            w_data_exc_numNs.append(w_data_exc_numNs0)  # numShufflesExc x number of neurons in the decoder # each element: numNeurons x numSamples x numFrames
            b_data_exc_numNs.append(b_data_exc_numNs0)
            cbest_exc_numNs.append(cbest_exc_numNs0) # numShufflesExc x number of neurons in the decoder x nFrs 
            
       
    # Plots: how choice prediction varies by increasing the population size, compare exc vs inh
    '''
    # data
    i = np.mean(perClassErrorTest_data_inh_numNs[:,:,frs], axis=1).squeeze()
    e = np.mean(perClassErrorTest_data_allExc_numNs[:,:,frs], axis=1).squeeze()
    ee = np.mean(perClassErrorTest_data_exc_numNs, axis=(0,2))[:,frs].squeeze()    
    # shfl
    #i = np.mean(perClassErrorTest_shfl_inh_numNs[:,:,frs], axis=1).squeeze()
    #e = np.mean(perClassErrorTest_shfl_allExc_numNs[:,:,frs], axis=1).squeeze()
    # chance
    #i = np.mean(perClassErrorTest_chance_inh_numNs[:,:,frs], axis=1).squeeze()
    #e = np.mean(perClassErrorTest_chance_allExc_numNs[:,:,frs], axis=1).squeeze()
    plt.figure()
    plt.plot(100-i, color='r', label='inh')
    plt.plot(100-e, color='k', label='exc')
    plt.plot(100-ee, color='k', label='exc')
    plt.xlim([-5,len(excI)+4])    
    plt.xlabel('# Ns in the decoder')    
    plt.ylabel('Class accuracy (%)')
    plt.legend()
    '''    

    
    
    
    
    
    
    
################################################################################################################################
################################################################################################################################
#%% Same as above but run SVM for populations of increasingly bigger size (adding neurons 1 by 1 based on their tuning strength)

if addNs_rand:
    
    minShflN = 10 # at least do the selection of numNs random neurons 10 times
    # what frames to run SVM on? # set to np.nan if you want to run SVM on all frames.        
    frs = eventI_ds-1 # np.nan 
    if np.isnan(frs).all(): # run SVM on all frames
        frs = range(nFrs)
 
 
   ######################## inh ########################   
    if doInhAllexcEqexc[0] == 1:    
        
        Xinh0 = X_svm[:, inhRois==1,:]
        nnt = lenInh
        
        #### Loop over number of neurons in the population: from 1 to the entire population size 
        perClassErrorTrain_data_inh_numNs = []
        perClassErrorTest_data_inh_numNs = []
        perClassErrorTest_shfl_inh_numNs = []
        perClassErrorTest_chance_inh_numNs = []
        w_data_inh_numNs = []
        b_data_inh_numNs = []
        cbest_inh_numNs = []
       
        for numNs in np.arange(1,nnt+1): 

            # Select numNs random neurons from X, do this numShufflesN times
            numShufflesN = np.max([minShflN, np.ceil(nnt/float(numNs)).astype(int)]) # if you are selecting only 1 neuron out of 500 neurons, you will do this 500 times to get a selection of all neurons. On the other hand if you are selecting 400 neurons out of 500 neurons, you will do this only twice.
            print 'Subsampling %d neurons %d times....' %(numNs, numShufflesN)
            
            #### loop over subselecting numNs number of neurons out of all neurons.
            perClassErrorTrain_data_inh_numNsS = []
            perClassErrorTest_data_inh_numNsS = []
            perClassErrorTest_shfl_inh_numNsS = []
            perClassErrorTest_chance_inh_numNsS = []
            w_data_inh_numNsS = []
            b_data_inh_numNsS = []
            cbest_inh_numNsS = []
            
            for inN in range(numShufflesN):   # eg. for 20 iterations subselect 10 neuron out of 200 neurons
                print 'Iteration %d/%d of subsampling %d neurons' %(inN+1, numShufflesN, numNs)
            
                if numNs==1: # when only training SVM with one neuron, go through neurons from begining to end, no need to do subselection!            
                    Xinh = Xinh0[:,[inN],:]
                else: # subselect numNs number of neurons
                    inds = rng.permutation(nnt)
                    inds = inds[range(numNs)]
                    Xinh = Xinh0[:,inds,:]
                    
                cbest_inh = np.nan  # set to nan, so in the function below we set cbest... unless you think you will run addNs_roc again... which I guess you wont!  
                # perClassErrorTrain_inh: samples x cvals x frames
                # wAllC_inh: samples x cvals x nNeurons x frames
                perClassErrorTrain_inh, perClassErrorTest_inh, wAllC_inh, bAllC_inh, cbestAll_inh, cbest_inh, cvect, perClassErrorTestShfl_inh, perClassErrorTestChance_inh,_,_,_,_ = setbesc_frs(Xinh,Y_svm,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest_inh,frs,shflTrLabs=shflTrLabs) # outputs have size the number of shuffles in setbestc (shuffles per c value)
           
                # Set the vars at bestc (for all samples):
                perClassErrorTrain_data_inh = np.full((numSamples, nFrs), np.nan) # samples x frames
                perClassErrorTest_data_inh = np.full((numSamples, nFrs), np.nan)
                perClassErrorTest_shfl_inh = np.full((numSamples, nFrs), np.nan)
                perClassErrorTest_chance_inh = np.full((numSamples, nFrs), np.nan)
                w_data_inh = np.full((numSamples, Xinh.shape[1], nFrs), np.nan)  # samples x nNs x frames
                b_data_inh = np.full((numSamples, nFrs), np.nan)
                
                for ifr in frs: #: range(nFrs):   
    #                if cbestKnown:
    #                    indBestC = 0
    #                else:
                    indBestC = np.in1d(cvect, cbest_inh[ifr])
                        
                    perClassErrorTrain_data_inh[:,ifr] = perClassErrorTrain_inh[:,indBestC,ifr].squeeze()
                    perClassErrorTest_data_inh[:,ifr] = perClassErrorTest_inh[:,indBestC,ifr].squeeze()
                    perClassErrorTest_shfl_inh[:,ifr] = perClassErrorTestShfl_inh[:,indBestC,ifr].squeeze()
                    perClassErrorTest_chance_inh[:,ifr] = perClassErrorTestChance_inh[:,indBestC,ifr].squeeze()
                    w_data_inh[:,:,ifr] = wAllC_inh[:,indBestC,:,ifr]#.squeeze()
                    b_data_inh[:,ifr] = bAllC_inh[:,indBestC,ifr].squeeze()
                    
                # collect vars for all subsamples of a given population size for the decoder
                perClassErrorTrain_data_inh_numNsS.append(perClassErrorTrain_data_inh) # numShufflesN x nSamps x nFrs ... if SVM was run on only 1 frame then values of all other frames will be nan
                perClassErrorTest_data_inh_numNsS.append(perClassErrorTest_data_inh)
                perClassErrorTest_shfl_inh_numNsS.append(perClassErrorTest_shfl_inh)
                perClassErrorTest_chance_inh_numNsS.append(perClassErrorTest_chance_inh)
                w_data_inh_numNsS.append(w_data_inh) #np.transpose(w_data_inh, (1,0,2))) # otherwise np.array and scio.savemat wont work... bc the first two dimensions will be the same for all elements but not the rest... and it will try to combine them, but it cant, so I make sure the 2nd dimension is nNs_inTheDecoder instead of nSamps
                b_data_inh_numNsS.append(b_data_inh)
                cbest_inh_numNsS.append(cbest_inh)
                    
            # turn lists to arrays
            perClassErrorTrain_data_inh_numNsS = np.array(perClassErrorTrain_data_inh_numNsS)
            perClassErrorTest_data_inh_numNsS = np.array(perClassErrorTest_data_inh_numNsS)
            perClassErrorTest_shfl_inh_numNsS = np.array(perClassErrorTest_shfl_inh_numNsS)
            perClassErrorTest_chance_inh_numNsS = np.array(perClassErrorTest_chance_inh_numNsS)
            w_data_inh_numNsS = np.array(w_data_inh_numNsS)
            b_data_inh_numNsS = np.array(b_data_inh_numNsS)
            cbest_inh_numNsS = np.array(cbest_inh_numNsS)
                
            # collect vars for different number of neurons in the decoder
            perClassErrorTrain_data_inh_numNs.append(perClassErrorTrain_data_inh_numNsS) # number of neurons in the decoder # each element: numShufflesN x nSamps x nFrs ... if SVM was run on only 1 frame then values of all other frames will be nan
            perClassErrorTest_data_inh_numNs.append(perClassErrorTest_data_inh_numNsS)
            perClassErrorTest_shfl_inh_numNs.append(perClassErrorTest_shfl_inh_numNsS)
            perClassErrorTest_chance_inh_numNs.append(perClassErrorTest_chance_inh_numNsS)
            w_data_inh_numNs.append(w_data_inh_numNsS) # number of neurons in the decoder # each element: numShufflesN x nSamps x number of neurons in the decoder x nFrs ... if SVM was run on only 1 frame then values of all other frames will be nan
            b_data_inh_numNs.append(b_data_inh_numNsS) # number of neurons in the decoder # each element: numShufflesN x nSamps x nFrs 
            cbest_inh_numNs.append(cbest_inh_numNsS) # number of neurons in the decoder # each element: numShufflesN x nFrs 
        
        # turn lists to arrays
        perClassErrorTrain_data_inh_numNs = np.array(perClassErrorTrain_data_inh_numNs)
        perClassErrorTest_data_inh_numNs = np.array(perClassErrorTest_data_inh_numNs)
        perClassErrorTest_shfl_inh_numNs = np.array(perClassErrorTest_shfl_inh_numNs)
        perClassErrorTest_chance_inh_numNs = np.array(perClassErrorTest_chance_inh_numNs)
        w_data_inh_numNs = np.array(w_data_inh_numNs)
        b_data_inh_numNs = np.array(b_data_inh_numNs)
        cbest_inh_numNs = np.array(cbest_inh_numNs)
                
                
        
        
        
                
    ######################## allExc (n = num all exc neurons) ########################
    elif doInhAllexcEqexc[1] == 1: 
        
        XallExc0 = X_svm[:, inhRois==0,:]
        nnt = XallExc0.shape[1]
            
        #### Loop over number of neurons in the population: from 1 to the entire population size        
        perClassErrorTrain_data_allExc_numNs = []
        perClassErrorTest_data_allExc_numNs = []
        perClassErrorTest_shfl_allExc_numNs = []
        perClassErrorTest_chance_allExc_numNs = []
        w_data_allExc_numNs = []
        b_data_allExc_numNs = []
        cbest_allExc_numNs = []

        for numNs in np.arange(1, nnt+1): 
            
            # Select numNs random neurons from X, do this numShufflesN times            
            numShufflesN = np.max([minShflN, np.ceil(nnt/float(numNs)).astype(int)]) # if you are selecting only 1 neuron out of 500 neurons, you will do this 500 times to get a selection of all neurons. On the other hand if you are selecting 400 neurons out of 500 neurons, you will do this only twice.
            print 'Subsampling %d neurons %d times....' %(numNs, numShufflesN)
            
            #### loop over subselecting numNs number of neurons out of all neurons.
            perClassErrorTrain_data_allExc_numNsS = []
            perClassErrorTest_data_allExc_numNsS = []
            perClassErrorTest_shfl_allExc_numNsS = []
            perClassErrorTest_chance_allExc_numNsS = []
            w_data_allExc_numNsS = []
            b_data_allExc_numNsS = []
            cbest_allExc_numNsS = []
            
            for inN in range(numShufflesN):   # eg. for 20 iterations subselect 10 neuron out of 200 neurons
                print 'Iteration %d/%d of subsampling %d neurons' %(inN+1, numShufflesN, numNs)
            
                if numNs==1: # when only training SVM with one neuron, go through neurons from begining to end, no need to do subselection!            
                    XallExc = XallExc0[:,[inN],:]
                else: # subselect numNs number of neurons
                    inds = rng.permutation(nnt)
                    inds = inds[range(numNs)]
                    XallExc = XallExc0[:,inds,:]
                
                cbest_allExc = np.nan            
                # perClassErrorTrain_allExc: samples x cvals x frames        
                perClassErrorTrain_allExc, perClassErrorTest_allExc, wAllC_allExc, bAllC_allExc, cbestAll_allExc, cbest_allExc, cvect, perClassErrorTestShfl_allExc, perClassErrorTestChance_allExc,_,_,_,_ = setbesc_frs(XallExc,Y_svm,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest_allExc,frs,shflTrLabs=shflTrLabs) # outputs have size the number of shuffles in setbestc (shuffles per c value)
                
                # Take the values at bestc that you computed above (for all samples)
                perClassErrorTrain_data_allExc = np.full((numSamples, nFrs), np.nan)
                perClassErrorTest_data_allExc = np.full((numSamples, nFrs), np.nan)
                perClassErrorTest_shfl_allExc = np.full((numSamples, nFrs), np.nan)
                perClassErrorTest_chance_allExc = np.full((numSamples, nFrs), np.nan)
                w_data_allExc = np.full((numSamples, XallExc.shape[1], nFrs), np.nan)
                b_data_allExc = np.full((numSamples, nFrs), np.nan)
                for ifr in frs: #range(nFrs):    
    #                if cbestKnown:
    #                    indBestC = 0
    #                else:        
                    indBestC = np.in1d(cvect, cbest_allExc[ifr])
                    
                    perClassErrorTrain_data_allExc[:,ifr] = perClassErrorTrain_allExc[:,indBestC,ifr].squeeze()
                    perClassErrorTest_data_allExc[:,ifr] = perClassErrorTest_allExc[:,indBestC,ifr].squeeze()
                    perClassErrorTest_shfl_allExc[:,ifr] = perClassErrorTestShfl_allExc[:,indBestC,ifr].squeeze()
                    perClassErrorTest_chance_allExc[:,ifr] = perClassErrorTestChance_allExc[:,indBestC,ifr].squeeze()
                    w_data_allExc[:,:,ifr] = wAllC_allExc[:,indBestC,:,ifr]#.squeeze()
                    b_data_allExc[:,ifr] = bAllC_allExc[:,indBestC,ifr].squeeze()
             
                # collect vars for all subsamples of a given population size for the decoder
                perClassErrorTrain_data_allExc_numNsS.append(perClassErrorTrain_data_allExc) # numShufflesN x nSamps x nFrs ... if SVM was run on only 1 frame then values of all other frames will be nan
                perClassErrorTest_data_allExc_numNsS.append(perClassErrorTest_data_allExc)
                perClassErrorTest_shfl_allExc_numNsS.append(perClassErrorTest_shfl_allExc)
                perClassErrorTest_chance_allExc_numNsS.append(perClassErrorTest_chance_allExc)
                w_data_allExc_numNsS.append(w_data_allExc) #np.transpose(w_data_allExc, (1,0,2))) # otherwise np.array and scio.savemat wont work... bc the first two dimensions will be the same for all elements but not the rest... and it will try to combine them, but it cant, so I make sure the 2nd dimension is nNs_inTheDecoder instead of nSamps
                b_data_allExc_numNsS.append(b_data_allExc)
                cbest_allExc_numNsS.append(cbest_allExc)
                    
            # turn lists to arrays
            perClassErrorTrain_data_allExc_numNsS = np.array(perClassErrorTrain_data_allExc_numNsS)
            perClassErrorTest_data_allExc_numNsS = np.array(perClassErrorTest_data_allExc_numNsS)
            perClassErrorTest_shfl_allExc_numNsS = np.array(perClassErrorTest_shfl_allExc_numNsS)
            perClassErrorTest_chance_allExc_numNsS = np.array(perClassErrorTest_chance_allExc_numNsS)
            w_data_allExc_numNsS = np.array(w_data_allExc_numNsS)
            b_data_allExc_numNsS = np.array(b_data_allExc_numNsS)
            cbest_allExc_numNsS = np.array(cbest_allExc_numNsS)
                
            # collect vars for different number of neurons in the decoder
            perClassErrorTrain_data_allExc_numNs.append(perClassErrorTrain_data_allExc_numNsS) # number of neurons in the decoder # each element: numShufflesN x nSamps x nFrs ... if SVM was run on only 1 frame then values of all other frames will be nan
            perClassErrorTest_data_allExc_numNs.append(perClassErrorTest_data_allExc_numNsS)
            perClassErrorTest_shfl_allExc_numNs.append(perClassErrorTest_shfl_allExc_numNsS)
            perClassErrorTest_chance_allExc_numNs.append(perClassErrorTest_chance_allExc_numNsS)
            w_data_allExc_numNs.append(w_data_allExc_numNsS) # number of neurons in the decoder # each element: numShufflesN x nSamps x number of neurons in the decoder x nFrs ... if SVM was run on only 1 frame then values of all other frames will be nan
            b_data_allExc_numNs.append(b_data_allExc_numNsS) # number of neurons in the decoder # each element: numShufflesN x nSamps x nFrs 
            cbest_allExc_numNs.append(cbest_allExc_numNsS) # number of neurons in the decoder # each element: numShufflesN x nFrs 
        
        # turn lists to arrays
        perClassErrorTrain_data_allExc_numNs = np.array(perClassErrorTrain_data_allExc_numNs)
        perClassErrorTest_data_allExc_numNs = np.array(perClassErrorTest_data_allExc_numNs)
        perClassErrorTest_shfl_allExc_numNs = np.array(perClassErrorTest_shfl_allExc_numNs)
        perClassErrorTest_chance_allExc_numNs = np.array(perClassErrorTest_chance_allExc_numNs)
        w_data_allExc_numNs = np.array(w_data_allExc_numNs)
        b_data_allExc_numNs = np.array(b_data_allExc_numNs)
        cbest_allExc_numNs = np.array(cbest_allExc_numNs)
       



    ######################### n exc (n=lenInh) (pick random sets of n exc neurons (do it numShufflesEx times)) ########################
    # first subsample exc neurons, then sort them based on ROC values; finally run svm.
    elif doInhAllexcEqexc[2] == 1:         
        
        #### Loop over random subsamples of exc neurons
        perClassErrorTrain_data_exc_numNs = []
        perClassErrorTest_data_exc_numNs = []
        perClassErrorTest_shfl_exc_numNs = []
        perClassErrorTest_chance_exc_numNs = []
        w_data_exc_numNs = []
        b_data_exc_numNs = []
        cbest_exc_numNs = []
#        cbestAll_exc_numNs = []        
        
        for ii in range(numShufflesExc):    
            print '\n\n-------- Exc subsample %d --------' %(ii)
            
            # Take a subsample of exc neurons
            XExc0 = XexcEq[ii]
            nnt = XExc0.shape[1]           
       
            #### Loop over number of neurons in the population: from 1 to the entire population size        
            perClassErrorTrain_data_exc_numNs0 = []
            perClassErrorTest_data_exc_numNs0 = []
            perClassErrorTest_shfl_exc_numNs0 = []
            perClassErrorTest_chance_exc_numNs0 = []
            w_data_exc_numNs0 = []
            b_data_exc_numNs0 = []
            cbest_exc_numNs0 = []
    
            for numNs in np.arange(1, nnt+1): 
                
                # Select numNs random neurons from X, do this numShufflesN times                
                numShufflesN = np.max([minShflN, np.ceil(nnt/float(numNs)).astype(int)]) # if you are selecting only 1 neuron out of 500 neurons, you will do this 500 times to get a selection of all neurons. On the other hand if you are selecting 400 neurons out of 500 neurons, you will do this only twice.
                print 'Subsampling %d neurons %d times....' %(numNs, numShufflesN)
                
                perClassErrorTrain_data_exc_numNsS = []
                perClassErrorTest_data_exc_numNsS = []
                perClassErrorTest_shfl_exc_numNsS = []
                perClassErrorTest_chance_exc_numNsS = []
                w_data_exc_numNsS = []
                b_data_exc_numNsS = []
                cbest_exc_numNsS = []
                
                #### loop over subselecting numNs number of neurons out of all neurons.
                for inN in range(numShufflesN):   # eg. for 20 iterations subselect 10 neuron out of 200 neurons
                    print 'Iteration %d/%d of subsampling %d neurons' %(inN+1, numShufflesN, numNs)
                
                    if numNs==1: # when only training SVM with one neuron, go through neurons from begining to end, no need to do subselection!            
                        XExc = XExc0[:,[inN],:]
                    else: # subselect numNs number of neurons
                        inds = rng.permutation(nnt)
                        inds = inds[range(numNs)]
                        XExc = XExc0[:,inds,:]
                        
                    cbest_now = np.nan                    
                    
                    perClassErrorTrain_exc, perClassErrorTest_exc, wAllC_exc, bAllC_exc, cbestAll_exc0, cbest_exc0, cvect, perClassErrorTestShfl_exc, perClassErrorTestChance_exc,_,_,_,_ = setbesc_frs(XExc,Y_svm,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest_now, frs, shflTrLabs=shflTrLabs) # outputs have size the number of shuffles in setbestc (shuffles per c value)
            
                    # Take the values at bestc that you computed above (for all samples)
                    perClassErrorTrain_data_exc0 = np.full((numSamples, nFrs), np.nan)
                    perClassErrorTest_data_exc0 = np.full((numSamples, nFrs), np.nan)
                    perClassErrorTest_shfl_exc0 = np.full((numSamples, nFrs), np.nan)
                    perClassErrorTest_chance_exc0 = np.full((numSamples, nFrs), np.nan)
                    w_data_exc0 = np.full((numSamples, XExc.shape[1], nFrs), np.nan)
                    b_data_exc0 = np.full((numSamples, nFrs), np.nan)
                    for ifr in frs: # range(nFrs):    
    #                    if cbestKnown:
    #                        indBestC = 0
    #                    else:
                        indBestC = np.in1d(cvect, cbest_exc0[ifr])
                            
                        perClassErrorTrain_data_exc0[:,ifr] = perClassErrorTrain_exc[:,indBestC,ifr].squeeze()
                        perClassErrorTest_data_exc0[:,ifr] = perClassErrorTest_exc[:,indBestC,ifr].squeeze()
                        perClassErrorTest_shfl_exc0[:,ifr] = perClassErrorTestShfl_exc[:,indBestC,ifr].squeeze()
                        perClassErrorTest_chance_exc0[:,ifr] = perClassErrorTestChance_exc[:,indBestC,ifr].squeeze()
                        w_data_exc0[:,:,ifr] = wAllC_exc[:,indBestC,:,ifr]#.squeeze()
                        b_data_exc0[:,ifr] = bAllC_exc[:,indBestC,ifr].squeeze()
                
                
                    # collect vars for all subsamples of a given population size for the decoder
                    perClassErrorTrain_data_exc_numNsS.append(perClassErrorTrain_data_exc0) # numShufflesN x nSamps x nFrs ... if SVM was run on only 1 frame then values of all other frames will be nan
                    perClassErrorTest_data_exc_numNsS.append(perClassErrorTest_data_exc0)
                    perClassErrorTest_shfl_exc_numNsS.append(perClassErrorTest_shfl_exc0)
                    perClassErrorTest_chance_exc_numNsS.append(perClassErrorTest_chance_exc0)
                    w_data_exc_numNsS.append(w_data_exc0) # numShufflesN x nSamps x number of neurons in the decoder x nFrs  #np.transpose(w_data_exc, (1,0,2))) # otherwise np.array and scio.savemat wont work... bc the first two dimensions will be the same for all elements but not the rest... and it will try to combine them, but it cant, so I make sure the 2nd dimension is nNs_inTheDecoder instead of nSamps
                    b_data_exc_numNsS.append(b_data_exc0)
                    cbest_exc_numNsS.append(cbest_exc0) # numShufflesN x nFrs
                        
                # turn lists to arrays
                perClassErrorTrain_data_exc_numNsS = np.array(perClassErrorTrain_data_exc_numNsS)
                perClassErrorTest_data_exc_numNsS = np.array(perClassErrorTest_data_exc_numNsS)
                perClassErrorTest_shfl_exc_numNsS = np.array(perClassErrorTest_shfl_exc_numNsS)
                perClassErrorTest_chance_exc_numNsS = np.array(perClassErrorTest_chance_exc_numNsS)
                w_data_exc_numNsS = np.array(w_data_exc_numNsS)
                b_data_exc_numNsS = np.array(b_data_exc_numNsS)
                cbest_exc_numNsS = np.array(cbest_exc_numNsS)                
                
                # collect vars for different number of neurons in the decoder
                perClassErrorTrain_data_exc_numNs0.append(perClassErrorTrain_data_exc_numNsS) # number of neurons in the decoder # each element: numShufflesN x nSamps x nFrs ... if SVM was run on only 1 frame then values of all other frames will be nan
                perClassErrorTest_data_exc_numNs0.append(perClassErrorTest_data_exc_numNsS)
                perClassErrorTest_shfl_exc_numNs0.append(perClassErrorTest_shfl_exc_numNsS)
                perClassErrorTest_chance_exc_numNs0.append(perClassErrorTest_chance_exc_numNsS)
                w_data_exc_numNs0.append(w_data_exc_numNsS) # number of neurons in the decoder # each element: numShufflesN x nSamps x number of neurons in the decoder x nFrs
                b_data_exc_numNs0.append(b_data_exc_numNsS)
                cbest_exc_numNs0.append(cbest_exc_numNsS) # number of neurons in the decoder # each element: numShufflesN x nFrs 
            
            # turn lists to arrays
            perClassErrorTrain_data_exc_numNs0 = np.array(perClassErrorTrain_data_exc_numNs0) # number of neurons in the decoder # each element: numShufflesN x nSamps x nFrs ... 
            perClassErrorTest_data_exc_numNs0 = np.array(perClassErrorTest_data_exc_numNs0)
            perClassErrorTest_shfl_exc_numNs0 = np.array(perClassErrorTest_shfl_exc_numNs0)
            perClassErrorTest_chance_exc_numNs0 = np.array(perClassErrorTest_chance_exc_numNs0)
            w_data_exc_numNs0 = np.array(w_data_exc_numNs0)  # number of neurons in the decoder # each element: numShufflesN x nSamps x number of neurons in the decoder x nFrs
            b_data_exc_numNs0 = np.array(b_data_exc_numNs0)
            cbest_exc_numNs0 = np.array(cbest_exc_numNs0) # number of neurons in the decoder # each element: numShufflesN x nFrs
            
            # collect values of all exc shuffles                
            perClassErrorTrain_data_exc_numNs.append(perClassErrorTrain_data_exc_numNs0) # numShufflesExc x number of neurons in the decoder # each element: numShufflesN x nSamps x nFrs ...
            perClassErrorTest_data_exc_numNs.append(perClassErrorTest_data_exc_numNs0)
            perClassErrorTest_shfl_exc_numNs.append(perClassErrorTest_shfl_exc_numNs0)
            perClassErrorTest_chance_exc_numNs.append(perClassErrorTest_chance_exc_numNs0)
            w_data_exc_numNs.append(w_data_exc_numNs0)  # numShufflesExc x number of neurons in the decoder # each element: numShufflesN x nSamps x number of neurons in the decoder x nFrs
            b_data_exc_numNs.append(b_data_exc_numNs0)
            cbest_exc_numNs.append(cbest_exc_numNs0) # numShufflesExc x number of neurons in the decoder # each element: numShufflesN x nFrs
            
        # turn lists to arrays
        perClassErrorTrain_data_exc_numNs = np.array(perClassErrorTrain_data_exc_numNs)
        perClassErrorTest_data_exc_numNs = np.array(perClassErrorTest_data_exc_numNs)
        perClassErrorTest_shfl_exc_numNs = np.array(perClassErrorTest_shfl_exc_numNs)
        perClassErrorTest_chance_exc_numNs = np.array(perClassErrorTest_chance_exc_numNs)
        w_data_exc_numNs = np.array(w_data_exc_numNs)
        b_data_exc_numNs = np.array(b_data_exc_numNs)
        cbest_exc_numNs = np.array(cbest_exc_numNs)
        
        
       
    # Plots: how choice prediction varies by increasing the population size, compare exc vs inh
    '''
    # data
    i = np.mean(perClassErrorTest_data_inh_numNs[:,:,frs], axis=1).squeeze()
    e = np.mean(perClassErrorTest_data_allExc_numNs[:,:,frs], axis=1).squeeze()
    ee = np.mean(perClassErrorTest_data_exc_numNs, axis=(0,2))[:,frs].squeeze()    
    # shfl
    #i = np.mean(perClassErrorTest_shfl_inh_numNs[:,:,frs], axis=1).squeeze()
    #e = np.mean(perClassErrorTest_shfl_allExc_numNs[:,:,frs], axis=1).squeeze()
    # chance
    #i = np.mean(perClassErrorTest_chance_inh_numNs[:,:,frs], axis=1).squeeze()
    #e = np.mean(perClassErrorTest_chance_allExc_numNs[:,:,frs], axis=1).squeeze()
    plt.figure()
    plt.plot(100-i, color='r', label='inh')
    plt.plot(100-e, color='k', label='exc')
    plt.xlim([-5,len(excI)+4])    
    plt.xlabel('# Ns in the decoder')    
    plt.ylabel('Class accuracy (%)')
    plt.legend()
    '''    


    
#%% Run SVM to decode stimulus category : new version (within each stimulus category (HR and LR) we make sure there are equal number of correct and incorrect trials)
####################################################################################################################################
####################################################################################################################################

if decodeStimCateg or decodeChoiceEqCorrIncorr:   # run svm on all neurons (ie not adding neurons one by one!) 
    
    cbestBoth = 1
    cbestCV = 0
    cbestRemCorr = 0    
    
    ######################## inh ########################   
    if doInhAllexcEqexc[0] == 1:
        Xinh = X_svm[:, inhRois==1,:]            
        
        nRandCorrSel, wAllC_inh, bAllC_inh, cbestFrs_inh, cbestFrs_remCorr_inh, cbestFrs_both_inh, cvect, \
        perClassErrorTrain_inh0, perClassErrorTest_inh0, perClassErrorTest_shfl_inh0, perClassErrorTest_chance_inh0, \
        perClassErrorTestRemCorr_inh0, perClassErrorTestRemCorr_shfl_inh0, perClassErrorTestRemCorr_chance_inh0, \
        perClassErrorTestBoth_inh0, perClassErrorTestBoth_shfl_inh0, perClassErrorTestBoth_chance_inh0, \
        trsnow_allSamps_inh, testTrInds_allSamps_inh, trsTrainedTestedInds_inh, trsRemCorrInds_inh, \
        Ytest_allSamps_inh, Ytest_hat_allSampsFrs_inh0, Ytest_hat_remCorr_allSampsFrs_inh0, Ytest_hat_both_allSampsFrs_inh0, \
         = setbesc_frs_stimCateg_new(outcomes, Xinh, Y_svm, regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,\
                                 cbest_inh,fr2an=np.nan,shflTrLabs=shflTrLabs)
    
                
        # Set the parameters below at bestc (for all samples):
        w_data_inh = np.full((numSamples, Xinh.shape[1], nRandCorrSel, nFrs), np.nan)
        b_data_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTrain_data_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTest_data_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTest_shfl_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTest_chance_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestRemCorr_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestRemCorr_shfl_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestRemCorr_chance_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestBoth_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestBoth_shfl_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestBoth_chance_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)     
        if nRandCorrSel!=1:
            Ytest_hat_allSampsFrs_inh = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_allSampsFrs_inh0.shape[-1]), np.nan).squeeze() # squeeze helps if len_test=1        
            Ytest_hat_remCorr_allSampsFrs_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_remCorr_allSampsFrs_inh0.shape[-1]), np.nan).squeeze()
            Ytest_hat_both_allSampsFrs_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_both_allSampsFrs_inh0.shape[-1]), np.nan).squeeze()
        else:
            Ytest_hat_allSampsFrs_inh = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_allSampsFrs_inh0.shape[-1]), np.nan)
            Ytest_hat_remCorr_allSampsFrs_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_remCorr_allSampsFrs_inh0.shape[-1]), np.nan)
            Ytest_hat_both_allSampsFrs_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_both_allSampsFrs_inh0.shape[-1]), np.nan)
        
        for ifr in range(nFrs):   
            if cbestKnown:
                indBestC = 0
            else:                
                if cbestCV: # find bestc on the same choice testing dataset               
                    indBestC = np.in1d(cvect, cbestFrs_inh[ifr])
                    
                elif cbestRemCorr: # find bestc on other choice testing dataset                               
                    indBestC = np.in1d(cvect, cbestFrs_remCorr_inh[ifr])
                    
                elif cbestBoth: # Allow bestc to take different values for each testing dataset                             
                    indBestC = np.in1d(cvect, cbestFrs_both_inh[ifr])
            
            if nRandCorrSel!=1:
                perClassErrorTrain_data_inh[:,:,ifr] = perClassErrorTrain_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_data_inh[:,:,ifr] = perClassErrorTest_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_shfl_inh[:,:,ifr] = perClassErrorTest_shfl_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_chance_inh[:,:,ifr] = perClassErrorTest_chance_inh0[:,indBestC,:,ifr].squeeze()            
#                perClassErrorTestRemCorr_inh[:,:,ifr] = perClassErrorTestRemCorr_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestRemCorr_shfl_inh[:,:,ifr] = perClassErrorTestRemCorr_shfl_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestRemCorr_chance_inh[:,:,ifr] = perClassErrorTestRemCorr_chance_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_inh[:,:,ifr] = perClassErrorTestBoth_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_shfl_inh[:,:,ifr] = perClassErrorTestBoth_shfl_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_chance_inh[:,:,ifr] = perClassErrorTestBoth_chance_inh0[:,indBestC,:,ifr].squeeze()            
                b_data_inh[:,:,ifr] = bAllC_inh[:,indBestC,:,ifr].squeeze()                      
                w_data_inh[:,:,:,ifr] = wAllC_inh[:,indBestC,:,:,ifr].squeeze()
                Ytest_hat_allSampsFrs_inh[:,:,ifr,:] = Ytest_hat_allSampsFrs_inh0[:,indBestC,:,ifr,:].squeeze()
#                Ytest_hat_remCorr_allSampsFrs_inh[:,:,ifr,:] = Ytest_hat_remCorr_allSampsFrs_inh0[:,indBestC,:,ifr,:].squeeze()
#                Ytest_hat_both_allSampsFrs_inh[:,:,ifr,:] = Ytest_hat_both_allSampsFrs_inh0[:,indBestC,:,ifr,:].squeeze()    
            else:
                perClassErrorTrain_data_inh[:,0,ifr] = perClassErrorTrain_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_data_inh[:,0,ifr] = perClassErrorTest_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_shfl_inh[:,0,ifr] = perClassErrorTest_shfl_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_chance_inh[:,0,ifr] = perClassErrorTest_chance_inh0[:,indBestC,:,ifr].squeeze()            
#                perClassErrorTestRemCorr_inh[:,0,ifr] = perClassErrorTestRemCorr_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestRemCorr_shfl_inh[:,0,ifr] = perClassErrorTestRemCorr_shfl_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestRemCorr_chance_inh[:,0,ifr] = perClassErrorTestRemCorr_chance_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_inh[:,0,ifr] = perClassErrorTestBoth_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_shfl_inh[:,0,ifr] = perClassErrorTestBoth_shfl_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_chance_inh[:,0,ifr] = perClassErrorTestBoth_chance_inh0[:,indBestC,:,ifr].squeeze()            
                b_data_inh[:,0,ifr] = bAllC_inh[:,indBestC,:,ifr].squeeze()                      
                w_data_inh[:,:,0,ifr] = wAllC_inh[:,indBestC,:,:,ifr].squeeze()
                Ytest_hat_allSampsFrs_inh[:,0,ifr,:] = Ytest_hat_allSampsFrs_inh0[:,indBestC,:,ifr,:].squeeze()
#                Ytest_hat_remCorr_allSampsFrs_inh[:,0,ifr,:] = Ytest_hat_remCorr_allSampsFrs_inh0[:,indBestC,:,ifr,:].squeeze()
#                Ytest_hat_both_allSampsFrs_inh[:,0,ifr,:] = Ytest_hat_both_allSampsFrs_inh0[:,indBestC,:,ifr,:].squeeze()    
                
    
    
    ######################## allExc (n = num all exc neurons) or allN (all exc, inh, unsure neurons) ########################
    elif np.logical_or(doInhAllexcEqexc[1] == 1 , doInhAllexcEqexc[1] == 2): 
        if doInhAllexcEqexc[1] == 1: # allExc
            XallExc = X_svm[:, inhRois==0,:]
        elif doInhAllexcEqexc[1] == 2: # allN
            XallExc = X_svm
        
        nRandCorrSel, wAllC_allExc, bAllC_allExc, cbestFrs_allExc, cbestFrs_remCorr_allExc, cbestFrs_both_allExc, cvect, \
        perClassErrorTrain_allExc0, perClassErrorTest_allExc0, perClassErrorTest_shfl_allExc0, perClassErrorTest_chance_allExc0, \
        perClassErrorTestRemCorr_allExc0, perClassErrorTestRemCorr_shfl_allExc0, perClassErrorTestRemCorr_chance_allExc0, \
        perClassErrorTestBoth_allExc0, perClassErrorTestBoth_shfl_allExc0, perClassErrorTestBoth_chance_allExc0, \
        trsnow_allSamps_allExc, testTrInds_allSamps_allExc, trsTrainedTestedInds_allExc, trsRemCorrInds_allExc, \
        Ytest_allSamps_allExc, Ytest_hat_allSampsFrs_allExc0, Ytest_hat_remCorr_allSampsFrs_allExc0, Ytest_hat_both_allSampsFrs_allExc0, \
         = setbesc_frs_stimCateg_new(outcomes, XallExc, Y_svm, regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,\
                                 cbest_allExc,fr2an=np.nan,shflTrLabs=shflTrLabs)
    
                
        # Set the parameters below at bestc (for all samples):
        w_data_allExc = np.full((numSamples, XallExc.shape[1], nRandCorrSel, nFrs), np.nan)
        b_data_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTrain_data_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTest_data_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTest_shfl_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTest_chance_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestRemCorr_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestRemCorr_shfl_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestRemCorr_chance_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestBoth_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestBoth_shfl_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestBoth_chance_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)        
        if nRandCorrSel!=1:
            Ytest_hat_allSampsFrs_allExc = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_allSampsFrs_allExc0.shape[-1]), np.nan).squeeze() # squeeze helps if len_test=1        
            Ytest_hat_remCorr_allSampsFrs_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_remCorr_allSampsFrs_allExc0.shape[-1]), np.nan).squeeze()
            Ytest_hat_both_allSampsFrs_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_both_allSampsFrs_allExc0.shape[-1]), np.nan).squeeze()
        else:
            Ytest_hat_allSampsFrs_allExc = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_allSampsFrs_allExc0.shape[-1]), np.nan)
            Ytest_hat_remCorr_allSampsFrs_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_remCorr_allSampsFrs_allExc0.shape[-1]), np.nan)
            Ytest_hat_both_allSampsFrs_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_both_allSampsFrs_allExc0.shape[-1]), np.nan)

        
        for ifr in range(nFrs):   
            if cbestKnown:
                indBestC = 0
            else:                
                if cbestCV: # find bestc on the same choice testing dataset               
                    indBestC = np.in1d(cvect, cbestFrs_allExc[ifr])
                    
                elif cbestRemCorr: # find bestc on other choice testing dataset                               
                    indBestC = np.in1d(cvect, cbestFrs_remCorr_allExc[ifr])
                    
                elif cbestBoth: # Allow bestc to take different values for each testing dataset                             
                    indBestC = np.in1d(cvect, cbestFrs_both_allExc[ifr])
            
            if nRandCorrSel!=1:
                perClassErrorTrain_data_allExc[:,:,ifr] = perClassErrorTrain_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_data_allExc[:,:,ifr] = perClassErrorTest_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_shfl_allExc[:,:,ifr] = perClassErrorTest_shfl_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_chance_allExc[:,:,ifr] = perClassErrorTest_chance_allExc0[:,indBestC,:,ifr].squeeze()            
#                perClassErrorTestRemCorr_allExc[:,:,ifr] = perClassErrorTestRemCorr_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestRemCorr_shfl_allExc[:,:,ifr] = perClassErrorTestRemCorr_shfl_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestRemCorr_chance_allExc[:,:,ifr] = perClassErrorTestRemCorr_chance_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_allExc[:,:,ifr] = perClassErrorTestBoth_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_shfl_allExc[:,:,ifr] = perClassErrorTestBoth_shfl_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_chance_allExc[:,:,ifr] = perClassErrorTestBoth_chance_allExc0[:,indBestC,:,ifr].squeeze()            
                b_data_allExc[:,:,ifr] = bAllC_allExc[:,indBestC,:,ifr].squeeze()                      
                w_data_allExc[:,:,:,ifr] = wAllC_allExc[:,indBestC,:,:,ifr].squeeze()
                Ytest_hat_allSampsFrs_allExc[:,:,ifr,:] = Ytest_hat_allSampsFrs_allExc0[:,indBestC,:,ifr,:].squeeze()
#                Ytest_hat_remCorr_allSampsFrs_allExc[:,:,ifr,:] = Ytest_hat_remCorr_allSampsFrs_allExc0[:,indBestC,:,ifr,:].squeeze()
#                Ytest_hat_both_allSampsFrs_allExc[:,:,ifr,:] = Ytest_hat_both_allSampsFrs_allExc0[:,indBestC,:,ifr,:].squeeze()    
            else:
                perClassErrorTrain_data_allExc[:,0,ifr] = perClassErrorTrain_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_data_allExc[:,0,ifr] = perClassErrorTest_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_shfl_allExc[:,0,ifr] = perClassErrorTest_shfl_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_chance_allExc[:,0,ifr] = perClassErrorTest_chance_allExc0[:,indBestC,:,ifr].squeeze()            
#                perClassErrorTestRemCorr_allExc[:,0,ifr] = perClassErrorTestRemCorr_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestRemCorr_shfl_allExc[:,0,ifr] = perClassErrorTestRemCorr_shfl_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestRemCorr_chance_allExc[:,0,ifr] = perClassErrorTestRemCorr_chance_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_allExc[:,0,ifr] = perClassErrorTestBoth_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_shfl_allExc[:,0,ifr] = perClassErrorTestBoth_shfl_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_chance_allExc[:,0,ifr] = perClassErrorTestBoth_chance_allExc0[:,indBestC,:,ifr].squeeze()            
                b_data_allExc[:,0,ifr] = bAllC_allExc[:,indBestC,:,ifr].squeeze()                      
                w_data_allExc[:,:,0,ifr] = wAllC_allExc[:,indBestC,:,:,ifr].squeeze()
                Ytest_hat_allSampsFrs_allExc[:,0,ifr,:] = Ytest_hat_allSampsFrs_allExc0[:,indBestC,:,ifr,:].squeeze()
#                Ytest_hat_remCorr_allSampsFrs_allExc[:,0,ifr,:] = Ytest_hat_remCorr_allSampsFrs_allExc0[:,indBestC,:,ifr,:].squeeze()
#                Ytest_hat_both_allSampsFrs_allExc[:,0,ifr,:] = Ytest_hat_both_allSampsFrs_allExc0[:,indBestC,:,ifr,:].squeeze()    

            
    
    
    ######################### n exc (n=lenInh) (pick random sets of n exc neurons (do it numShufflesEx times)) ########################
    elif doInhAllexcEqexc[2] == 1: 
        if cbestKnown==False:
            cbestFrs_exc = []
            cbestFrs_remCorr_exc = []
            cbestFrs_both_exc = []
        perClassErrorTrain_data_exc = []
        perClassErrorTest_data_exc = []
        perClassErrorTest_shfl_exc = []
        perClassErrorTest_chance_exc = []
        perClassErrorTestRemCorr_exc = []
        perClassErrorTestRemCorr_shfl_exc = []
        perClassErrorTestRemCorr_chance_exc = []
        perClassErrorTestBoth_exc = []
        perClassErrorTestBoth_shfl_exc = []
        perClassErrorTestBoth_chance_exc = []
        b_data_exc = []
        w_data_exc = []        
        Ytest_hat_allSampsFrs_exc = []        
        Ytest_hat_remCorr_allSampsFrs_exc = []
        Ytest_hat_both_allSampsFrs_exc = []
        Ytest_allSamps_exc = []
        trsnow_allSamps_exc = []
        testTrInds_allSamps_exc = []        
        trsTrainedTestedInds_exc = []
        trsRemCorrInds_exc = []
        
        
        for ii in range(numShufflesExc): # ii=0 # loop through exc subsamples   
            print '\n\n-------- Exc subsample %d --------' %(ii)
            if cbestKnown:
                cbest_now = cbest_exc[ii]
            
            Xnow = XexcEq[ii]
            
            nRandCorrSel, wAllC_exc, bAllC_exc, cbestFrs_exc00, cbestFrs_remCorr_exc00, cbestFrs_both_exc00, cvect, \
            perClassErrorTrain_exc00, perClassErrorTest_exc00, perClassErrorTest_shfl_exc00, perClassErrorTest_chance_exc00, \
            perClassErrorTestRemCorr_exc00, perClassErrorTestRemCorr_shfl_exc00, perClassErrorTestRemCorr_chance_exc00, \
            perClassErrorTestBoth_exc00, perClassErrorTestBoth_shfl_exc00, perClassErrorTestBoth_chance_exc00, \
            trsnow_allSamps_exc00, testTrInds_allSamps_exc00, trsTrainedTestedInds_exc00, trsRemCorrInds_exc00, \
            Ytest_allSamps_exc00, Ytest_hat_allSampsFrs_exc00, Ytest_hat_remCorr_allSampsFrs_exc00, Ytest_hat_both_allSampsFrs_exc00, \
             = setbesc_frs_stimCateg_new(outcomes, Xnow, Y_svm, regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,\
                                     cbest_now,fr2an=np.nan,shflTrLabs=shflTrLabs)
        
                    
            # Set the parameters below at bestc (for all samples):
            w_data_exc0 = np.full((numSamples, Xnow.shape[1], nRandCorrSel, nFrs), np.nan)
            b_data_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTrain_data_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTest_data_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTest_shfl_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTest_chance_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTestRemCorr_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTestRemCorr_shfl_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTestRemCorr_chance_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTestBoth_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTestBoth_shfl_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTestBoth_chance_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)    
            if nRandCorrSel!=1:
                Ytest_hat_allSampsFrs_exc0 = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_allSampsFrs_exc00.shape[-1]), np.nan).squeeze() # squeeze helps if len_test=1        
                Ytest_hat_remCorr_allSampsFrs_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_remCorr_allSampsFrs_exc00.shape[-1]), np.nan).squeeze()
                Ytest_hat_both_allSampsFrs_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_both_allSampsFrs_exc00.shape[-1]), np.nan).squeeze()
            else:
                Ytest_hat_allSampsFrs_exc0 = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_allSampsFrs_exc00.shape[-1]), np.nan)
                Ytest_hat_remCorr_allSampsFrs_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_remCorr_allSampsFrs_exc00.shape[-1]), np.nan)
                Ytest_hat_both_allSampsFrs_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_both_allSampsFrs_exc00.shape[-1]), np.nan)

            
            for ifr in range(nFrs):   
                if cbestKnown:
                    indBestC = 0
                else:                
                    if cbestCV: # find bestc on the same choice testing dataset               
                        indBestC = np.in1d(cvect, cbestFrs_exc00[ifr])
                        
                    elif cbestRemCorr: # find bestc on other choice testing dataset                               
                        indBestC = np.in1d(cvect, cbestFrs_remCorr_exc00[ifr])
                        
                    elif cbestBoth: # Allow bestc to take different values for each testing dataset                             
                        indBestC = np.in1d(cvect, cbestFrs_both_exc00[ifr])
              
                if nRandCorrSel!=1:
                    perClassErrorTrain_data_exc0[:,:,ifr] = perClassErrorTrain_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTest_data_exc0[:,:,ifr] = perClassErrorTest_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTest_shfl_exc0[:,:,ifr] = perClassErrorTest_shfl_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTest_chance_exc0[:,:,ifr] = perClassErrorTest_chance_exc00[:,indBestC,:,ifr].squeeze()            
#                    perClassErrorTestRemCorr_exc0[:,:,ifr] = perClassErrorTestRemCorr_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestRemCorr_shfl_exc0[:,:,ifr] = perClassErrorTestRemCorr_shfl_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestRemCorr_chance_exc0[:,:,ifr] = perClassErrorTestRemCorr_chance_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestBoth_exc0[:,:,ifr] = perClassErrorTestBoth_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestBoth_shfl_exc0[:,:,ifr] = perClassErrorTestBoth_shfl_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestBoth_chance_exc0[:,:,ifr] = perClassErrorTestBoth_chance_exc00[:,indBestC,:,ifr].squeeze()            
                    b_data_exc0[:,:,ifr] = bAllC_exc[:,indBestC,:,ifr].squeeze()                      
                    w_data_exc0[:,:,:,ifr] = wAllC_exc[:,indBestC,:,:,ifr].squeeze()
                    Ytest_hat_allSampsFrs_exc0[:,:,ifr,:] = Ytest_hat_allSampsFrs_exc00[:,indBestC,:,ifr,:].squeeze()
#                    Ytest_hat_remCorr_allSampsFrs_exc0[:,:,ifr,:] = Ytest_hat_remCorr_allSampsFrs_exc00[:,indBestC,:,ifr,:].squeeze()
#                    Ytest_hat_both_allSampsFrs_exc0[:,:,ifr,:] = Ytest_hat_both_allSampsFrs_exc00[:,indBestC,:,ifr,:].squeeze()               
                else:
                    perClassErrorTrain_data_exc0[:,0,ifr] = perClassErrorTrain_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTest_data_exc0[:,0,ifr] = perClassErrorTest_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTest_shfl_exc0[:,0,ifr] = perClassErrorTest_shfl_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTest_chance_exc0[:,0,ifr] = perClassErrorTest_chance_exc00[:,indBestC,:,ifr].squeeze()            
#                    perClassErrorTestRemCorr_exc0[:,0,ifr] = perClassErrorTestRemCorr_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestRemCorr_shfl_exc0[:,0,ifr] = perClassErrorTestRemCorr_shfl_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestRemCorr_chance_exc0[:,0,ifr] = perClassErrorTestRemCorr_chance_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestBoth_exc0[:,0,ifr] = perClassErrorTestBoth_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestBoth_shfl_exc0[:,0,ifr] = perClassErrorTestBoth_shfl_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestBoth_chance_exc0[:,0,ifr] = perClassErrorTestBoth_chance_exc00[:,indBestC,:,ifr].squeeze()            
                    b_data_exc0[:,0,ifr] = bAllC_exc[:,indBestC,:,ifr].squeeze()                      
                    w_data_exc0[:,:,0,ifr] = wAllC_exc[:,indBestC,:,:,ifr].squeeze()
                    Ytest_hat_allSampsFrs_exc0[:,0,ifr,:] = Ytest_hat_allSampsFrs_exc00[:,indBestC,:,ifr,:].squeeze()
#                    Ytest_hat_remCorr_allSampsFrs_exc0[:,0,ifr,:] = Ytest_hat_remCorr_allSampsFrs_exc00[:,indBestC,:,ifr,:].squeeze()
#                    Ytest_hat_both_allSampsFrs_exc0[:,0,ifr,:] = Ytest_hat_both_allSampsFrs_exc00[:,indBestC,:,ifr,:].squeeze()               

            

            
            # Collect values of all exc shuffles (at bestc)   
            if cbestKnown==False:
                cbestFrs_exc.append(cbestFrs_exc00)
                cbestFrs_remCorr_exc.append(cbestFrs_remCorr_exc00)
                cbestFrs_both_exc.append(cbestFrs_both_exc00)                
            perClassErrorTrain_data_exc.append(perClassErrorTrain_data_exc0) # numShufflesExc x numSamples x nRandCorrSel x numFrames
            perClassErrorTest_data_exc.append(perClassErrorTest_data_exc0)
            perClassErrorTest_shfl_exc.append(perClassErrorTest_shfl_exc0)
            perClassErrorTest_chance_exc.append(perClassErrorTest_chance_exc0)
            perClassErrorTestRemCorr_exc.append(perClassErrorTestRemCorr_exc0)
            perClassErrorTestRemCorr_shfl_exc.append(perClassErrorTestRemCorr_shfl_exc0)
            perClassErrorTestRemCorr_chance_exc.append(perClassErrorTestRemCorr_chance_exc0)
            perClassErrorTestBoth_exc.append(perClassErrorTestBoth_exc0)
            perClassErrorTestBoth_shfl_exc.append(perClassErrorTestBoth_shfl_exc0)
            perClassErrorTestBoth_chance_exc.append(perClassErrorTestBoth_chance_exc0)            
            b_data_exc.append(b_data_exc0)
            w_data_exc.append(w_data_exc0) # numShufflesExc x numSamples x numNeurons x nRandCorrSel x numFrames                        
            Ytest_hat_allSampsFrs_exc.append(Ytest_hat_allSampsFrs_exc0)            
            Ytest_hat_remCorr_allSampsFrs_exc.append(Ytest_hat_remCorr_allSampsFrs_exc0)
            Ytest_hat_both_allSampsFrs_exc.append(Ytest_hat_both_allSampsFrs_exc0)            
            Ytest_allSamps_exc.append(Ytest_allSamps_exc00) # numShufflesExc x numSamples x nRandCorrSel x numTestTrs            
            testTrInds_allSamps_exc.append(testTrInds_allSamps_exc00)
            trsTrainedTestedInds_exc.append(trsTrainedTestedInds_exc00)
            trsnow_allSamps_exc.append(trsnow_allSamps_exc00) # nRandCorrSel x numTrs (test+train)
            trsRemCorrInds_exc.append(trsRemCorrInds_exc00) # nRandCorrSel x numRemCorrTrs     
    
    
#%% Run SVM to decode stimulus category : old version: it uses equal corr and incorr trials, but the correct way is to use equal corr and incorr trials WITHIN each stim category (HR, and LR)
####################################################################################################################################
####################################################################################################################################
"""
if decodeStimCateg:   # run svm on all neurons (ie not adding neurons one by one!) 
    
    cbestBoth = 1
    cbestCV = 0
    cbestRemCorr = 0    
    
    ######################## inh ########################   
    if doInhAllexcEqexc[0] == 1:
        Xinh = X_svm[:, inhRois==1,:]            
        
        nRandCorrSel, wAllC_inh, bAllC_inh, cbestFrs_inh, cbestFrs_remCorr_inh, cbestFrs_both_inh, cvect, \
        perClassErrorTrain_inh0, perClassErrorTest_inh0, perClassErrorTest_shfl_inh0, perClassErrorTest_chance_inh0, \
        perClassErrorTestRemCorr_inh0, perClassErrorTestRemCorr_shfl_inh0, perClassErrorTestRemCorr_chance_inh0, \
        perClassErrorTestBoth_inh0, perClassErrorTestBoth_shfl_inh0, perClassErrorTestBoth_chance_inh0, \
        trsnow_allSamps_inh, testTrInds_allSamps_inh, trsTrainedTestedInds_inh, trsRemCorrInds_inh, \
        Ytest_allSamps_inh, Ytest_hat_allSampsFrs_inh0, Ytest_hat_remCorr_allSampsFrs_inh0, Ytest_hat_both_allSampsFrs_inh0, \
         = setbesc_frs_stimCateg(outcomes, Xinh, Y_svm, regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,\
                                 cbest_inh,fr2an=np.nan,shflTrLabs=shflTrLabs)
    
                
        # Set the parameters below at bestc (for all samples):
        w_data_inh = np.full((numSamples, Xinh.shape[1], nRandCorrSel, nFrs), np.nan)
        b_data_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTrain_data_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTest_data_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTest_shfl_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTest_chance_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestRemCorr_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestRemCorr_shfl_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestRemCorr_chance_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestBoth_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestBoth_shfl_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestBoth_chance_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)     
        if nRandCorrSel!=1:
            Ytest_hat_allSampsFrs_inh = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_allSampsFrs_inh0.shape[-1]), np.nan).squeeze() # squeeze helps if len_test=1        
            Ytest_hat_remCorr_allSampsFrs_inh = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_remCorr_allSampsFrs_inh0.shape[-1]), np.nan).squeeze()
            Ytest_hat_both_allSampsFrs_inh = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_both_allSampsFrs_inh0.shape[-1]), np.nan).squeeze()
        else:
            Ytest_hat_allSampsFrs_inh = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_allSampsFrs_inh0.shape[-1]), np.nan)
            Ytest_hat_remCorr_allSampsFrs_inh = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_remCorr_allSampsFrs_inh0.shape[-1]), np.nan)
            Ytest_hat_both_allSampsFrs_inh = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_both_allSampsFrs_inh0.shape[-1]), np.nan)
        
        for ifr in range(nFrs):   
            if cbestKnown:
                indBestC = 0
            else:                
                if cbestCV: # find bestc on the same choice testing dataset               
                    indBestC = np.in1d(cvect, cbestFrs_inh[ifr])
                    
                elif cbestRemCorr: # find bestc on other choice testing dataset                               
                    indBestC = np.in1d(cvect, cbestFrs_remCorr_inh[ifr])
                    
                elif cbestBoth: # Allow bestc to take different values for each testing dataset                             
                    indBestC = np.in1d(cvect, cbestFrs_both_inh[ifr])
            
            if nRandCorrSel!=1:
                perClassErrorTrain_data_inh[:,:,ifr] = perClassErrorTrain_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_data_inh[:,:,ifr] = perClassErrorTest_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_shfl_inh[:,:,ifr] = perClassErrorTest_shfl_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_chance_inh[:,:,ifr] = perClassErrorTest_chance_inh0[:,indBestC,:,ifr].squeeze()            
                perClassErrorTestRemCorr_inh[:,:,ifr] = perClassErrorTestRemCorr_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestRemCorr_shfl_inh[:,:,ifr] = perClassErrorTestRemCorr_shfl_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestRemCorr_chance_inh[:,:,ifr] = perClassErrorTestRemCorr_chance_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestBoth_inh[:,:,ifr] = perClassErrorTestBoth_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestBoth_shfl_inh[:,:,ifr] = perClassErrorTestBoth_shfl_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestBoth_chance_inh[:,:,ifr] = perClassErrorTestBoth_chance_inh0[:,indBestC,:,ifr].squeeze()            
                b_data_inh[:,:,ifr] = bAllC_inh[:,indBestC,:,ifr].squeeze()                      
                w_data_inh[:,:,:,ifr] = wAllC_inh[:,indBestC,:,:,ifr].squeeze()
                Ytest_hat_allSampsFrs_inh[:,:,ifr,:] = Ytest_hat_allSampsFrs_inh0[:,indBestC,:,ifr,:].squeeze()
                Ytest_hat_remCorr_allSampsFrs_inh[:,:,ifr,:] = Ytest_hat_remCorr_allSampsFrs_inh0[:,indBestC,:,ifr,:].squeeze()
                Ytest_hat_both_allSampsFrs_inh[:,:,ifr,:] = Ytest_hat_both_allSampsFrs_inh0[:,indBestC,:,ifr,:].squeeze()    
            else:
                perClassErrorTrain_data_inh[:,0,ifr] = perClassErrorTrain_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_data_inh[:,0,ifr] = perClassErrorTest_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_shfl_inh[:,0,ifr] = perClassErrorTest_shfl_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_chance_inh[:,0,ifr] = perClassErrorTest_chance_inh0[:,indBestC,:,ifr].squeeze()            
                perClassErrorTestRemCorr_inh[:,0,ifr] = perClassErrorTestRemCorr_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestRemCorr_shfl_inh[:,0,ifr] = perClassErrorTestRemCorr_shfl_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestRemCorr_chance_inh[:,0,ifr] = perClassErrorTestRemCorr_chance_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestBoth_inh[:,0,ifr] = perClassErrorTestBoth_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestBoth_shfl_inh[:,0,ifr] = perClassErrorTestBoth_shfl_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestBoth_chance_inh[:,0,ifr] = perClassErrorTestBoth_chance_inh0[:,indBestC,:,ifr].squeeze()            
                b_data_inh[:,0,ifr] = bAllC_inh[:,indBestC,:,ifr].squeeze()                      
                w_data_inh[:,:,0,ifr] = wAllC_inh[:,indBestC,:,:,ifr].squeeze()
                Ytest_hat_allSampsFrs_inh[:,0,ifr,:] = Ytest_hat_allSampsFrs_inh0[:,indBestC,:,ifr,:].squeeze()
#                Ytest_hat_remCorr_allSampsFrs_inh[:,0,ifr,:] = Ytest_hat_remCorr_allSampsFrs_inh0[:,indBestC,:,ifr,:].squeeze()
                Ytest_hat_both_allSampsFrs_inh[:,0,ifr,:] = Ytest_hat_both_allSampsFrs_inh0[:,indBestC,:,ifr,:].squeeze()    
                
    
    
    ######################## allExc (n = num all exc neurons) or allN (all exc, inh, unsure neurons) ########################
    elif np.logical_or(doInhAllexcEqexc[1] == 1 , doInhAllexcEqexc[1] == 2): 
        if doInhAllexcEqexc[1] == 1: # allExc
            XallExc = X_svm[:, inhRois==0,:]
        elif doInhAllexcEqexc[1] == 2: # allN
            XallExc = X_svm
        
        nRandCorrSel, wAllC_allExc, bAllC_allExc, cbestFrs_allExc, cbestFrs_remCorr_allExc, cbestFrs_both_allExc, cvect, \
        perClassErrorTrain_allExc0, perClassErrorTest_allExc0, perClassErrorTest_shfl_allExc0, perClassErrorTest_chance_allExc0, \
        perClassErrorTestRemCorr_allExc0, perClassErrorTestRemCorr_shfl_allExc0, perClassErrorTestRemCorr_chance_allExc0, \
        perClassErrorTestBoth_allExc0, perClassErrorTestBoth_shfl_allExc0, perClassErrorTestBoth_chance_allExc0, \
        trsnow_allSamps_allExc, testTrInds_allSamps_allExc, trsTrainedTestedInds_allExc, trsRemCorrInds_allExc, \
        Ytest_allSamps_allExc, Ytest_hat_allSampsFrs_allExc0, Ytest_hat_remCorr_allSampsFrs_allExc0, Ytest_hat_both_allSampsFrs_allExc0, \
         = setbesc_frs_stimCateg(outcomes, XallExc, Y_svm, regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,\
                                 cbest_allExc,fr2an=np.nan,shflTrLabs=shflTrLabs)
    
                
        # Set the parameters below at bestc (for all samples):
        w_data_allExc = np.full((numSamples, XallExc.shape[1], nRandCorrSel, nFrs), np.nan)
        b_data_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTrain_data_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTest_data_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTest_shfl_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTest_chance_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestRemCorr_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestRemCorr_shfl_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestRemCorr_chance_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestBoth_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestBoth_shfl_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestBoth_chance_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)        
        if nRandCorrSel!=1:
            Ytest_hat_allSampsFrs_allExc = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_allSampsFrs_allExc0.shape[-1]), np.nan).squeeze() # squeeze helps if len_test=1        
            Ytest_hat_remCorr_allSampsFrs_allExc = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_remCorr_allSampsFrs_allExc0.shape[-1]), np.nan).squeeze()
            Ytest_hat_both_allSampsFrs_allExc = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_both_allSampsFrs_allExc0.shape[-1]), np.nan).squeeze()
        else:
            Ytest_hat_allSampsFrs_allExc = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_allSampsFrs_allExc0.shape[-1]), np.nan)
            Ytest_hat_remCorr_allSampsFrs_allExc = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_remCorr_allSampsFrs_allExc0.shape[-1]), np.nan)
            Ytest_hat_both_allSampsFrs_allExc = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_both_allSampsFrs_allExc0.shape[-1]), np.nan)

        
        for ifr in range(nFrs):   
            if cbestKnown:
                indBestC = 0
            else:                
                if cbestCV: # find bestc on the same choice testing dataset               
                    indBestC = np.in1d(cvect, cbestFrs_allExc[ifr])
                    
                elif cbestRemCorr: # find bestc on other choice testing dataset                               
                    indBestC = np.in1d(cvect, cbestFrs_remCorr_allExc[ifr])
                    
                elif cbestBoth: # Allow bestc to take different values for each testing dataset                             
                    indBestC = np.in1d(cvect, cbestFrs_both_allExc[ifr])
            
            if nRandCorrSel!=1:
                perClassErrorTrain_data_allExc[:,:,ifr] = perClassErrorTrain_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_data_allExc[:,:,ifr] = perClassErrorTest_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_shfl_allExc[:,:,ifr] = perClassErrorTest_shfl_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_chance_allExc[:,:,ifr] = perClassErrorTest_chance_allExc0[:,indBestC,:,ifr].squeeze()            
                perClassErrorTestRemCorr_allExc[:,:,ifr] = perClassErrorTestRemCorr_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestRemCorr_shfl_allExc[:,:,ifr] = perClassErrorTestRemCorr_shfl_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestRemCorr_chance_allExc[:,:,ifr] = perClassErrorTestRemCorr_chance_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestBoth_allExc[:,:,ifr] = perClassErrorTestBoth_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestBoth_shfl_allExc[:,:,ifr] = perClassErrorTestBoth_shfl_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestBoth_chance_allExc[:,:,ifr] = perClassErrorTestBoth_chance_allExc0[:,indBestC,:,ifr].squeeze()            
                b_data_allExc[:,:,ifr] = bAllC_allExc[:,indBestC,:,ifr].squeeze()                      
                w_data_allExc[:,:,:,ifr] = wAllC_allExc[:,indBestC,:,:,ifr].squeeze()
                Ytest_hat_allSampsFrs_allExc[:,:,ifr,:] = Ytest_hat_allSampsFrs_allExc0[:,indBestC,:,ifr,:].squeeze()
                Ytest_hat_remCorr_allSampsFrs_allExc[:,:,ifr,:] = Ytest_hat_remCorr_allSampsFrs_allExc0[:,indBestC,:,ifr,:].squeeze()
                Ytest_hat_both_allSampsFrs_allExc[:,:,ifr,:] = Ytest_hat_both_allSampsFrs_allExc0[:,indBestC,:,ifr,:].squeeze()    
            else:
                perClassErrorTrain_data_allExc[:,0,ifr] = perClassErrorTrain_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_data_allExc[:,0,ifr] = perClassErrorTest_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_shfl_allExc[:,0,ifr] = perClassErrorTest_shfl_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_chance_allExc[:,0,ifr] = perClassErrorTest_chance_allExc0[:,indBestC,:,ifr].squeeze()            
                perClassErrorTestRemCorr_allExc[:,0,ifr] = perClassErrorTestRemCorr_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestRemCorr_shfl_allExc[:,0,ifr] = perClassErrorTestRemCorr_shfl_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestRemCorr_chance_allExc[:,0,ifr] = perClassErrorTestRemCorr_chance_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestBoth_allExc[:,0,ifr] = perClassErrorTestBoth_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestBoth_shfl_allExc[:,0,ifr] = perClassErrorTestBoth_shfl_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTestBoth_chance_allExc[:,0,ifr] = perClassErrorTestBoth_chance_allExc0[:,indBestC,:,ifr].squeeze()            
                b_data_allExc[:,0,ifr] = bAllC_allExc[:,indBestC,:,ifr].squeeze()                      
                w_data_allExc[:,:,0,ifr] = wAllC_allExc[:,indBestC,:,:,ifr].squeeze()
                Ytest_hat_allSampsFrs_allExc[:,0,ifr,:] = Ytest_hat_allSampsFrs_allExc0[:,indBestC,:,ifr,:].squeeze()
                Ytest_hat_remCorr_allSampsFrs_allExc[:,0,ifr,:] = Ytest_hat_remCorr_allSampsFrs_allExc0[:,indBestC,:,ifr,:].squeeze()
                Ytest_hat_both_allSampsFrs_allExc[:,0,ifr,:] = Ytest_hat_both_allSampsFrs_allExc0[:,indBestC,:,ifr,:].squeeze()    

            
    
    
    ######################### n exc (n=lenInh) (pick random sets of n exc neurons (do it numShufflesEx times)) ########################
    elif doInhAllexcEqexc[2] == 1: 
        if cbestKnown==False:
            cbestFrs_exc = []
            cbestFrs_remCorr_exc = []
            cbestFrs_both_exc = []
        perClassErrorTrain_data_exc = []
        perClassErrorTest_data_exc = []
        perClassErrorTest_shfl_exc = []
        perClassErrorTest_chance_exc = []
        perClassErrorTestRemCorr_exc = []
        perClassErrorTestRemCorr_shfl_exc = []
        perClassErrorTestRemCorr_chance_exc = []
        perClassErrorTestBoth_exc = []
        perClassErrorTestBoth_shfl_exc = []
        perClassErrorTestBoth_chance_exc = []
        b_data_exc = []
        w_data_exc = []        
        Ytest_hat_allSampsFrs_exc = []        
        Ytest_hat_remCorr_allSampsFrs_exc = []
        Ytest_hat_both_allSampsFrs_exc = []
        Ytest_allSamps_exc = []
        trsnow_allSamps_exc = []
        testTrInds_allSamps_exc = []        
        trsTrainedTestedInds_exc = []
        trsRemCorrInds_exc = []
        
        
        for ii in range(numShufflesExc): # ii=0 # loop through exc subsamples   
            print '\n\n-------- Exc subsample %d --------' %(ii)
            if cbestKnown:
                cbest_now = cbest_exc[ii]
            
            Xnow = XexcEq[ii]
            
            nRandCorrSel, wAllC_exc, bAllC_exc, cbestFrs_exc00, cbestFrs_remCorr_exc00, cbestFrs_both_exc00, cvect, \
            perClassErrorTrain_exc00, perClassErrorTest_exc00, perClassErrorTest_shfl_exc00, perClassErrorTest_chance_exc00, \
            perClassErrorTestRemCorr_exc00, perClassErrorTestRemCorr_shfl_exc00, perClassErrorTestRemCorr_chance_exc00, \
            perClassErrorTestBoth_exc00, perClassErrorTestBoth_shfl_exc00, perClassErrorTestBoth_chance_exc00, \
            trsnow_allSamps_exc00, testTrInds_allSamps_exc00, trsTrainedTestedInds_exc00, trsRemCorrInds_exc00, \
            Ytest_allSamps_exc00, Ytest_hat_allSampsFrs_exc00, Ytest_hat_remCorr_allSampsFrs_exc00, Ytest_hat_both_allSampsFrs_exc00, \
             = setbesc_frs_stimCateg(outcomes, Xnow, Y_svm, regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,\
                                     cbest_now,fr2an=np.nan,shflTrLabs=shflTrLabs)
        
                    
            # Set the parameters below at bestc (for all samples):
            w_data_exc0 = np.full((numSamples, Xnow.shape[1], nRandCorrSel, nFrs), np.nan)
            b_data_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTrain_data_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTest_data_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTest_shfl_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTest_chance_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTestRemCorr_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTestRemCorr_shfl_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTestRemCorr_chance_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTestBoth_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTestBoth_shfl_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTestBoth_chance_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)    
            if nRandCorrSel!=1:
                Ytest_hat_allSampsFrs_exc0 = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_allSampsFrs_exc00.shape[-1]), np.nan).squeeze() # squeeze helps if len_test=1        
                Ytest_hat_remCorr_allSampsFrs_exc0 = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_remCorr_allSampsFrs_exc00.shape[-1]), np.nan).squeeze()
                Ytest_hat_both_allSampsFrs_exc0 = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_both_allSampsFrs_exc00.shape[-1]), np.nan).squeeze()
            else:
                Ytest_hat_allSampsFrs_exc0 = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_allSampsFrs_exc00.shape[-1]), np.nan)
                Ytest_hat_remCorr_allSampsFrs_exc0 = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_remCorr_allSampsFrs_exc00.shape[-1]), np.nan)
                Ytest_hat_both_allSampsFrs_exc0 = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_both_allSampsFrs_exc00.shape[-1]), np.nan)

            
            for ifr in range(nFrs):   
                if cbestKnown:
                    indBestC = 0
                else:                
                    if cbestCV: # find bestc on the same choice testing dataset               
                        indBestC = np.in1d(cvect, cbestFrs_exc00[ifr])
                        
                    elif cbestRemCorr: # find bestc on other choice testing dataset                               
                        indBestC = np.in1d(cvect, cbestFrs_remCorr_exc00[ifr])
                        
                    elif cbestBoth: # Allow bestc to take different values for each testing dataset                             
                        indBestC = np.in1d(cvect, cbestFrs_both_exc00[ifr])
              
                if nRandCorrSel!=1:
                    perClassErrorTrain_data_exc0[:,:,ifr] = perClassErrorTrain_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTest_data_exc0[:,:,ifr] = perClassErrorTest_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTest_shfl_exc0[:,:,ifr] = perClassErrorTest_shfl_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTest_chance_exc0[:,:,ifr] = perClassErrorTest_chance_exc00[:,indBestC,:,ifr].squeeze()            
                    perClassErrorTestRemCorr_exc0[:,:,ifr] = perClassErrorTestRemCorr_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTestRemCorr_shfl_exc0[:,:,ifr] = perClassErrorTestRemCorr_shfl_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTestRemCorr_chance_exc0[:,:,ifr] = perClassErrorTestRemCorr_chance_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTestBoth_exc0[:,:,ifr] = perClassErrorTestBoth_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTestBoth_shfl_exc0[:,:,ifr] = perClassErrorTestBoth_shfl_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTestBoth_chance_exc0[:,:,ifr] = perClassErrorTestBoth_chance_exc00[:,indBestC,:,ifr].squeeze()            
                    b_data_exc0[:,:,ifr] = bAllC_exc[:,indBestC,:,ifr].squeeze()                      
                    w_data_exc0[:,:,:,ifr] = wAllC_exc[:,indBestC,:,:,ifr].squeeze()
                    Ytest_hat_allSampsFrs_exc0[:,:,ifr,:] = Ytest_hat_allSampsFrs_exc00[:,indBestC,:,ifr,:].squeeze()
                    Ytest_hat_remCorr_allSampsFrs_exc0[:,:,ifr,:] = Ytest_hat_remCorr_allSampsFrs_exc00[:,indBestC,:,ifr,:].squeeze()
                    Ytest_hat_both_allSampsFrs_exc0[:,:,ifr,:] = Ytest_hat_both_allSampsFrs_exc00[:,indBestC,:,ifr,:].squeeze()               
                else:
                    perClassErrorTrain_data_exc0[:,0,ifr] = perClassErrorTrain_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTest_data_exc0[:,0,ifr] = perClassErrorTest_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTest_shfl_exc0[:,0,ifr] = perClassErrorTest_shfl_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTest_chance_exc0[:,0,ifr] = perClassErrorTest_chance_exc00[:,indBestC,:,ifr].squeeze()            
                    perClassErrorTestRemCorr_exc0[:,0,ifr] = perClassErrorTestRemCorr_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTestRemCorr_shfl_exc0[:,0,ifr] = perClassErrorTestRemCorr_shfl_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTestRemCorr_chance_exc0[:,0,ifr] = perClassErrorTestRemCorr_chance_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTestBoth_exc0[:,0,ifr] = perClassErrorTestBoth_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTestBoth_shfl_exc0[:,0,ifr] = perClassErrorTestBoth_shfl_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTestBoth_chance_exc0[:,0,ifr] = perClassErrorTestBoth_chance_exc00[:,indBestC,:,ifr].squeeze()            
                    b_data_exc0[:,0,ifr] = bAllC_exc[:,indBestC,:,ifr].squeeze()                      
                    w_data_exc0[:,:,0,ifr] = wAllC_exc[:,indBestC,:,:,ifr].squeeze()
                    Ytest_hat_allSampsFrs_exc0[:,0,ifr,:] = Ytest_hat_allSampsFrs_exc00[:,indBestC,:,ifr,:].squeeze()
                    Ytest_hat_remCorr_allSampsFrs_exc0[:,0,ifr,:] = Ytest_hat_remCorr_allSampsFrs_exc00[:,indBestC,:,ifr,:].squeeze()
                    Ytest_hat_both_allSampsFrs_exc0[:,0,ifr,:] = Ytest_hat_both_allSampsFrs_exc00[:,indBestC,:,ifr,:].squeeze()               

            

            
            # Collect values of all exc shuffles (at bestc)   
            if cbestKnown==False:
                cbestFrs_exc.append(cbestFrs_exc00)
                cbestFrs_remCorr_exc.append(cbestFrs_remCorr_exc00)
                cbestFrs_both_exc.append(cbestFrs_both_exc00)                
            perClassErrorTrain_data_exc.append(perClassErrorTrain_data_exc0) # numShufflesExc x numSamples x nRandCorrSel x numFrames
            perClassErrorTest_data_exc.append(perClassErrorTest_data_exc0)
            perClassErrorTest_shfl_exc.append(perClassErrorTest_shfl_exc0)
            perClassErrorTest_chance_exc.append(perClassErrorTest_chance_exc0)
            perClassErrorTestRemCorr_exc.append(perClassErrorTestRemCorr_exc0)
            perClassErrorTestRemCorr_shfl_exc.append(perClassErrorTestRemCorr_shfl_exc0)
            perClassErrorTestRemCorr_chance_exc.append(perClassErrorTestRemCorr_chance_exc0)
            perClassErrorTestBoth_exc.append(perClassErrorTestBoth_exc0)
            perClassErrorTestBoth_shfl_exc.append(perClassErrorTestBoth_shfl_exc0)
            perClassErrorTestBoth_chance_exc.append(perClassErrorTestBoth_chance_exc0)            
            b_data_exc.append(b_data_exc0)
            w_data_exc.append(w_data_exc0) # numShufflesExc x numSamples x numNeurons x nRandCorrSel x numFrames                        
            Ytest_hat_allSampsFrs_exc.append(Ytest_hat_allSampsFrs_exc0)            
            Ytest_hat_remCorr_allSampsFrs_exc.append(Ytest_hat_remCorr_allSampsFrs_exc0)
            Ytest_hat_both_allSampsFrs_exc.append(Ytest_hat_both_allSampsFrs_exc0)            
            Ytest_allSamps_exc.append(Ytest_allSamps_exc00) # numShufflesExc x numSamples x nRandCorrSel x numTestTrs            
            testTrInds_allSamps_exc.append(testTrInds_allSamps_exc00)
            trsTrainedTestedInds_exc.append(trsTrainedTestedInds_exc00)
            trsnow_allSamps_exc.append(trsnow_allSamps_exc00) # nRandCorrSel x numTrs (test+train)
            trsRemCorrInds_exc.append(trsRemCorrInds_exc00) # nRandCorrSel x numRemCorrTrs     
"""    


#%% Run SVM to decode trial outcome (corr, incorr)
####################################################################################################################################
####################################################################################################################################

if decodeOutcome:   # run svm on all neurons (ie not adding neurons one by one!) 
    
    cbestBoth = 1
    cbestCV = 0
    cbestRemCorr = 0    
    
    ######################## inh ########################   
    if doInhAllexcEqexc[0] == 1:
        Xinh = X_svm[:, inhRois==1,:]            
        
        nRandCorrSel, wAllC_inh, bAllC_inh, cbestFrs_inh, cbestFrs_remCorr_inh, cbestFrs_both_inh, cvect, \
        perClassErrorTrain_inh0, perClassErrorTest_inh0, perClassErrorTest_shfl_inh0, perClassErrorTest_chance_inh0, \
        perClassErrorTestRemCorr_inh0, perClassErrorTestRemCorr_shfl_inh0, perClassErrorTestRemCorr_chance_inh0, \
        perClassErrorTestBoth_inh0, perClassErrorTestBoth_shfl_inh0, perClassErrorTestBoth_chance_inh0, \
        trsnow_allSamps_inh, testTrInds_allSamps_inh, trsTrainedTestedInds_inh, trsRemCorrInds_inh, \
        Ytest_allSamps_inh, Ytest_hat_allSampsFrs_inh0, Ytest_hat_remCorr_allSampsFrs_inh0, Ytest_hat_both_allSampsFrs_inh0, \
         = setbesc_frs_outcome(allResp_HR_LR, Xinh, Y_svm, regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,\
                                 cbest_inh,fr2an=np.nan,shflTrLabs=shflTrLabs)
    
                
        # Set the parameters below at bestc (for all samples):
        w_data_inh = np.full((numSamples, Xinh.shape[1], nRandCorrSel, nFrs), np.nan)
        b_data_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTrain_data_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTest_data_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTest_shfl_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTest_chance_inh = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestRemCorr_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestRemCorr_shfl_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestRemCorr_chance_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestBoth_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestBoth_shfl_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestBoth_chance_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)     
        if nRandCorrSel!=1:
            Ytest_hat_allSampsFrs_inh = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_allSampsFrs_inh0.shape[-1]), np.nan).squeeze() # squeeze helps if len_test=1        
            Ytest_hat_remCorr_allSampsFrs_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_remCorr_allSampsFrs_inh0.shape[-1]), np.nan).squeeze()
            Ytest_hat_both_allSampsFrs_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_both_allSampsFrs_inh0.shape[-1]), np.nan).squeeze()
        else:
            Ytest_hat_allSampsFrs_inh = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_allSampsFrs_inh0.shape[-1]), np.nan)
            Ytest_hat_remCorr_allSampsFrs_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_remCorr_allSampsFrs_inh0.shape[-1]), np.nan)
            Ytest_hat_both_allSampsFrs_inh = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_both_allSampsFrs_inh0.shape[-1]), np.nan)
        
        for ifr in range(nFrs):   
            if cbestKnown:
                indBestC = 0
            else:                
                if cbestCV: # find bestc on the same choice testing dataset               
                    indBestC = np.in1d(cvect, cbestFrs_inh[ifr])
                    
                elif cbestRemCorr: # find bestc on other choice testing dataset                               
                    indBestC = np.in1d(cvect, cbestFrs_remCorr_inh[ifr])
                    
                elif cbestBoth: # Allow bestc to take different values for each testing dataset                             
                    indBestC = np.in1d(cvect, cbestFrs_both_inh[ifr])
            
            if nRandCorrSel!=1:
                perClassErrorTrain_data_inh[:,:,ifr] = perClassErrorTrain_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_data_inh[:,:,ifr] = perClassErrorTest_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_shfl_inh[:,:,ifr] = perClassErrorTest_shfl_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_chance_inh[:,:,ifr] = perClassErrorTest_chance_inh0[:,indBestC,:,ifr].squeeze()            
#                perClassErrorTestRemCorr_inh[:,:,ifr] = perClassErrorTestRemCorr_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestRemCorr_shfl_inh[:,:,ifr] = perClassErrorTestRemCorr_shfl_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestRemCorr_chance_inh[:,:,ifr] = perClassErrorTestRemCorr_chance_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_inh[:,:,ifr] = perClassErrorTestBoth_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_shfl_inh[:,:,ifr] = perClassErrorTestBoth_shfl_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_chance_inh[:,:,ifr] = perClassErrorTestBoth_chance_inh0[:,indBestC,:,ifr].squeeze()            
                b_data_inh[:,:,ifr] = bAllC_inh[:,indBestC,:,ifr].squeeze()                      
                w_data_inh[:,:,:,ifr] = wAllC_inh[:,indBestC,:,:,ifr].squeeze()
                Ytest_hat_allSampsFrs_inh[:,:,ifr,:] = Ytest_hat_allSampsFrs_inh0[:,indBestC,:,ifr,:].squeeze()
#                Ytest_hat_remCorr_allSampsFrs_inh[:,:,ifr,:] = Ytest_hat_remCorr_allSampsFrs_inh0[:,indBestC,:,ifr,:].squeeze()
#                Ytest_hat_both_allSampsFrs_inh[:,:,ifr,:] = Ytest_hat_both_allSampsFrs_inh0[:,indBestC,:,ifr,:].squeeze()    
            else:
                perClassErrorTrain_data_inh[:,0,ifr] = perClassErrorTrain_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_data_inh[:,0,ifr] = perClassErrorTest_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_shfl_inh[:,0,ifr] = perClassErrorTest_shfl_inh0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_chance_inh[:,0,ifr] = perClassErrorTest_chance_inh0[:,indBestC,:,ifr].squeeze()            
#                perClassErrorTestRemCorr_inh[:,0,ifr] = perClassErrorTestRemCorr_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestRemCorr_shfl_inh[:,0,ifr] = perClassErrorTestRemCorr_shfl_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestRemCorr_chance_inh[:,0,ifr] = perClassErrorTestRemCorr_chance_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_inh[:,0,ifr] = perClassErrorTestBoth_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_shfl_inh[:,0,ifr] = perClassErrorTestBoth_shfl_inh0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_chance_inh[:,0,ifr] = perClassErrorTestBoth_chance_inh0[:,indBestC,:,ifr].squeeze()            
                b_data_inh[:,0,ifr] = bAllC_inh[:,indBestC,:,ifr].squeeze()                      
                w_data_inh[:,:,0,ifr] = wAllC_inh[:,indBestC,:,:,ifr].squeeze()
                Ytest_hat_allSampsFrs_inh[:,0,ifr,:] = Ytest_hat_allSampsFrs_inh0[:,indBestC,:,ifr,:].squeeze()
#                Ytest_hat_remCorr_allSampsFrs_inh[:,0,ifr,:] = Ytest_hat_remCorr_allSampsFrs_inh0[:,indBestC,:,ifr,:].squeeze()
#                Ytest_hat_both_allSampsFrs_inh[:,0,ifr,:] = Ytest_hat_both_allSampsFrs_inh0[:,indBestC,:,ifr,:].squeeze()    
                
    
    
    ######################## allExc (n = num all exc neurons) or allN (all exc, inh, unsure neurons) ########################
    elif np.logical_or(doInhAllexcEqexc[1] == 1 , doInhAllexcEqexc[1] == 2): 
        if doInhAllexcEqexc[1] == 1: # allExc
            XallExc = X_svm[:, inhRois==0,:]
        elif doInhAllexcEqexc[1] == 2: # allN
            XallExc = X_svm
        
        nRandCorrSel, wAllC_allExc, bAllC_allExc, cbestFrs_allExc, cbestFrs_remCorr_allExc, cbestFrs_both_allExc, cvect, \
        perClassErrorTrain_allExc0, perClassErrorTest_allExc0, perClassErrorTest_shfl_allExc0, perClassErrorTest_chance_allExc0, \
        perClassErrorTestRemCorr_allExc0, perClassErrorTestRemCorr_shfl_allExc0, perClassErrorTestRemCorr_chance_allExc0, \
        perClassErrorTestBoth_allExc0, perClassErrorTestBoth_shfl_allExc0, perClassErrorTestBoth_chance_allExc0, \
        trsnow_allSamps_allExc, testTrInds_allSamps_allExc, trsTrainedTestedInds_allExc, trsRemCorrInds_allExc, \
        Ytest_allSamps_allExc, Ytest_hat_allSampsFrs_allExc0, Ytest_hat_remCorr_allSampsFrs_allExc0, Ytest_hat_both_allSampsFrs_allExc0, \
         = setbesc_frs_outcome(allResp_HR_LR, XallExc, Y_svm, regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,\
                                 cbest_allExc,fr2an=np.nan,shflTrLabs=shflTrLabs)
    
                
        # Set the parameters below at bestc (for all samples):
        w_data_allExc = np.full((numSamples, XallExc.shape[1], nRandCorrSel, nFrs), np.nan)
        b_data_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTrain_data_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTest_data_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTest_shfl_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTest_chance_allExc = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestRemCorr_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestRemCorr_shfl_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestRemCorr_chance_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestBoth_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestBoth_shfl_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
        perClassErrorTestBoth_chance_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)        
        if nRandCorrSel!=1:
            Ytest_hat_allSampsFrs_allExc = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_allSampsFrs_allExc0.shape[-1]), np.nan).squeeze() # squeeze helps if len_test=1        
            Ytest_hat_remCorr_allSampsFrs_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_remCorr_allSampsFrs_allExc0.shape[-1]), np.nan).squeeze()
            Ytest_hat_both_allSampsFrs_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_both_allSampsFrs_allExc0.shape[-1]), np.nan).squeeze()
        else:
            Ytest_hat_allSampsFrs_allExc = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_allSampsFrs_allExc0.shape[-1]), np.nan)
            Ytest_hat_remCorr_allSampsFrs_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_remCorr_allSampsFrs_allExc0.shape[-1]), np.nan)
            Ytest_hat_both_allSampsFrs_allExc = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_both_allSampsFrs_allExc0.shape[-1]), np.nan)

        
        for ifr in range(nFrs):   
            if cbestKnown:
                indBestC = 0
            else:                
                if cbestCV: # find bestc on the same choice testing dataset               
                    indBestC = np.in1d(cvect, cbestFrs_allExc[ifr])
                    
                elif cbestRemCorr: # find bestc on other choice testing dataset                               
                    indBestC = np.in1d(cvect, cbestFrs_remCorr_allExc[ifr])
                    
                elif cbestBoth: # Allow bestc to take different values for each testing dataset                             
                    indBestC = np.in1d(cvect, cbestFrs_both_allExc[ifr])
            
            if nRandCorrSel!=1:
                perClassErrorTrain_data_allExc[:,:,ifr] = perClassErrorTrain_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_data_allExc[:,:,ifr] = perClassErrorTest_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_shfl_allExc[:,:,ifr] = perClassErrorTest_shfl_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_chance_allExc[:,:,ifr] = perClassErrorTest_chance_allExc0[:,indBestC,:,ifr].squeeze()            
#                perClassErrorTestRemCorr_allExc[:,:,ifr] = perClassErrorTestRemCorr_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestRemCorr_shfl_allExc[:,:,ifr] = perClassErrorTestRemCorr_shfl_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestRemCorr_chance_allExc[:,:,ifr] = perClassErrorTestRemCorr_chance_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_allExc[:,:,ifr] = perClassErrorTestBoth_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_shfl_allExc[:,:,ifr] = perClassErrorTestBoth_shfl_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_chance_allExc[:,:,ifr] = perClassErrorTestBoth_chance_allExc0[:,indBestC,:,ifr].squeeze()            
                b_data_allExc[:,:,ifr] = bAllC_allExc[:,indBestC,:,ifr].squeeze()                      
                w_data_allExc[:,:,:,ifr] = wAllC_allExc[:,indBestC,:,:,ifr].squeeze()
                Ytest_hat_allSampsFrs_allExc[:,:,ifr,:] = Ytest_hat_allSampsFrs_allExc0[:,indBestC,:,ifr,:].squeeze()
#                Ytest_hat_remCorr_allSampsFrs_allExc[:,:,ifr,:] = Ytest_hat_remCorr_allSampsFrs_allExc0[:,indBestC,:,ifr,:].squeeze()
#                Ytest_hat_both_allSampsFrs_allExc[:,:,ifr,:] = Ytest_hat_both_allSampsFrs_allExc0[:,indBestC,:,ifr,:].squeeze()    
            else:
                perClassErrorTrain_data_allExc[:,0,ifr] = perClassErrorTrain_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_data_allExc[:,0,ifr] = perClassErrorTest_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_shfl_allExc[:,0,ifr] = perClassErrorTest_shfl_allExc0[:,indBestC,:,ifr].squeeze()
                perClassErrorTest_chance_allExc[:,0,ifr] = perClassErrorTest_chance_allExc0[:,indBestC,:,ifr].squeeze()            
#                perClassErrorTestRemCorr_allExc[:,0,ifr] = perClassErrorTestRemCorr_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestRemCorr_shfl_allExc[:,0,ifr] = perClassErrorTestRemCorr_shfl_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestRemCorr_chance_allExc[:,0,ifr] = perClassErrorTestRemCorr_chance_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_allExc[:,0,ifr] = perClassErrorTestBoth_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_shfl_allExc[:,0,ifr] = perClassErrorTestBoth_shfl_allExc0[:,indBestC,:,ifr].squeeze()
#                perClassErrorTestBoth_chance_allExc[:,0,ifr] = perClassErrorTestBoth_chance_allExc0[:,indBestC,:,ifr].squeeze()            
                b_data_allExc[:,0,ifr] = bAllC_allExc[:,indBestC,:,ifr].squeeze()                      
                w_data_allExc[:,:,0,ifr] = wAllC_allExc[:,indBestC,:,:,ifr].squeeze()
                Ytest_hat_allSampsFrs_allExc[:,0,ifr,:] = Ytest_hat_allSampsFrs_allExc0[:,indBestC,:,ifr,:].squeeze()
#                Ytest_hat_remCorr_allSampsFrs_allExc[:,0,ifr,:] = Ytest_hat_remCorr_allSampsFrs_allExc0[:,indBestC,:,ifr,:].squeeze()
#                Ytest_hat_both_allSampsFrs_allExc[:,0,ifr,:] = Ytest_hat_both_allSampsFrs_allExc0[:,indBestC,:,ifr,:].squeeze()    

            
    
    
    ######################### n exc (n=lenInh) (pick random sets of n exc neurons (do it numShufflesEx times)) ########################
    elif doInhAllexcEqexc[2] == 1: 
        if cbestKnown==False:
            cbestFrs_exc = []
            cbestFrs_remCorr_exc = []
            cbestFrs_both_exc = []
        perClassErrorTrain_data_exc = []
        perClassErrorTest_data_exc = []
        perClassErrorTest_shfl_exc = []
        perClassErrorTest_chance_exc = []
        perClassErrorTestRemCorr_exc = []
        perClassErrorTestRemCorr_shfl_exc = []
        perClassErrorTestRemCorr_chance_exc = []
        perClassErrorTestBoth_exc = []
        perClassErrorTestBoth_shfl_exc = []
        perClassErrorTestBoth_chance_exc = []
        b_data_exc = []
        w_data_exc = []        
        Ytest_hat_allSampsFrs_exc = []        
        Ytest_hat_remCorr_allSampsFrs_exc = []
        Ytest_hat_both_allSampsFrs_exc = []
        Ytest_allSamps_exc = []
        trsnow_allSamps_exc = []
        testTrInds_allSamps_exc = []        
        trsTrainedTestedInds_exc = []
        trsRemCorrInds_exc = []
        
        
        for ii in range(numShufflesExc): # ii=0 # loop through exc subsamples   
            print '\n\n-------- Exc subsample %d --------' %(ii)
            if cbestKnown:
                cbest_now = cbest_exc[ii]
            
            Xnow = XexcEq[ii]
            
            nRandCorrSel, wAllC_exc, bAllC_exc, cbestFrs_exc00, cbestFrs_remCorr_exc00, cbestFrs_both_exc00, cvect, \
            perClassErrorTrain_exc00, perClassErrorTest_exc00, perClassErrorTest_shfl_exc00, perClassErrorTest_chance_exc00, \
            perClassErrorTestRemCorr_exc00, perClassErrorTestRemCorr_shfl_exc00, perClassErrorTestRemCorr_chance_exc00, \
            perClassErrorTestBoth_exc00, perClassErrorTestBoth_shfl_exc00, perClassErrorTestBoth_chance_exc00, \
            trsnow_allSamps_exc00, testTrInds_allSamps_exc00, trsTrainedTestedInds_exc00, trsRemCorrInds_exc00, \
            Ytest_allSamps_exc00, Ytest_hat_allSampsFrs_exc00, Ytest_hat_remCorr_allSampsFrs_exc00, Ytest_hat_both_allSampsFrs_exc00, \
             = setbesc_frs_outcome(allResp_HR_LR, Xnow, Y_svm, regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,\
                                     cbest_now,fr2an=np.nan,shflTrLabs=shflTrLabs)
        
                    
            # Set the parameters below at bestc (for all samples):
            w_data_exc0 = np.full((numSamples, Xnow.shape[1], nRandCorrSel, nFrs), np.nan)
            b_data_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTrain_data_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTest_data_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTest_shfl_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTest_chance_exc0 = np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTestRemCorr_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTestRemCorr_shfl_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTestRemCorr_chance_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTestBoth_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTestBoth_shfl_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)
            perClassErrorTestBoth_chance_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs), np.nan)    
            if nRandCorrSel!=1:
                Ytest_hat_allSampsFrs_exc0 = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_allSampsFrs_exc00.shape[-1]), np.nan).squeeze() # squeeze helps if len_test=1        
                Ytest_hat_remCorr_allSampsFrs_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_remCorr_allSampsFrs_exc00.shape[-1]), np.nan).squeeze()
                Ytest_hat_both_allSampsFrs_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_both_allSampsFrs_exc00.shape[-1]), np.nan).squeeze()
            else:
                Ytest_hat_allSampsFrs_exc0 = np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_allSampsFrs_exc00.shape[-1]), np.nan)
                Ytest_hat_remCorr_allSampsFrs_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_remCorr_allSampsFrs_exc00.shape[-1]), np.nan)
                Ytest_hat_both_allSampsFrs_exc0 = np.nan #np.full((numSamples, nRandCorrSel, nFrs, Ytest_hat_both_allSampsFrs_exc00.shape[-1]), np.nan)

            
            for ifr in range(nFrs):   
                if cbestKnown:
                    indBestC = 0
                else:                
                    if cbestCV: # find bestc on the same choice testing dataset               
                        indBestC = np.in1d(cvect, cbestFrs_exc00[ifr])
                        
                    elif cbestRemCorr: # find bestc on other choice testing dataset                               
                        indBestC = np.in1d(cvect, cbestFrs_remCorr_exc00[ifr])
                        
                    elif cbestBoth: # Allow bestc to take different values for each testing dataset                             
                        indBestC = np.in1d(cvect, cbestFrs_both_exc00[ifr])
              
                if nRandCorrSel!=1:
                    perClassErrorTrain_data_exc0[:,:,ifr] = perClassErrorTrain_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTest_data_exc0[:,:,ifr] = perClassErrorTest_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTest_shfl_exc0[:,:,ifr] = perClassErrorTest_shfl_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTest_chance_exc0[:,:,ifr] = perClassErrorTest_chance_exc00[:,indBestC,:,ifr].squeeze()            
#                    perClassErrorTestRemCorr_exc0[:,:,ifr] = perClassErrorTestRemCorr_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestRemCorr_shfl_exc0[:,:,ifr] = perClassErrorTestRemCorr_shfl_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestRemCorr_chance_exc0[:,:,ifr] = perClassErrorTestRemCorr_chance_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestBoth_exc0[:,:,ifr] = perClassErrorTestBoth_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestBoth_shfl_exc0[:,:,ifr] = perClassErrorTestBoth_shfl_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestBoth_chance_exc0[:,:,ifr] = perClassErrorTestBoth_chance_exc00[:,indBestC,:,ifr].squeeze()            
                    b_data_exc0[:,:,ifr] = bAllC_exc[:,indBestC,:,ifr].squeeze()                      
                    w_data_exc0[:,:,:,ifr] = wAllC_exc[:,indBestC,:,:,ifr].squeeze()
                    Ytest_hat_allSampsFrs_exc0[:,:,ifr,:] = Ytest_hat_allSampsFrs_exc00[:,indBestC,:,ifr,:].squeeze()
#                    Ytest_hat_remCorr_allSampsFrs_exc0[:,:,ifr,:] = Ytest_hat_remCorr_allSampsFrs_exc00[:,indBestC,:,ifr,:].squeeze()
#                    Ytest_hat_both_allSampsFrs_exc0[:,:,ifr,:] = Ytest_hat_both_allSampsFrs_exc00[:,indBestC,:,ifr,:].squeeze()               
                else:
                    perClassErrorTrain_data_exc0[:,0,ifr] = perClassErrorTrain_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTest_data_exc0[:,0,ifr] = perClassErrorTest_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTest_shfl_exc0[:,0,ifr] = perClassErrorTest_shfl_exc00[:,indBestC,:,ifr].squeeze()
                    perClassErrorTest_chance_exc0[:,0,ifr] = perClassErrorTest_chance_exc00[:,indBestC,:,ifr].squeeze()            
#                    perClassErrorTestRemCorr_exc0[:,0,ifr] = perClassErrorTestRemCorr_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestRemCorr_shfl_exc0[:,0,ifr] = perClassErrorTestRemCorr_shfl_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestRemCorr_chance_exc0[:,0,ifr] = perClassErrorTestRemCorr_chance_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestBoth_exc0[:,0,ifr] = perClassErrorTestBoth_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestBoth_shfl_exc0[:,0,ifr] = perClassErrorTestBoth_shfl_exc00[:,indBestC,:,ifr].squeeze()
#                    perClassErrorTestBoth_chance_exc0[:,0,ifr] = perClassErrorTestBoth_chance_exc00[:,indBestC,:,ifr].squeeze()            
                    b_data_exc0[:,0,ifr] = bAllC_exc[:,indBestC,:,ifr].squeeze()                      
                    w_data_exc0[:,:,0,ifr] = wAllC_exc[:,indBestC,:,:,ifr].squeeze()
                    Ytest_hat_allSampsFrs_exc0[:,0,ifr,:] = Ytest_hat_allSampsFrs_exc00[:,indBestC,:,ifr,:].squeeze()
#                    Ytest_hat_remCorr_allSampsFrs_exc0[:,0,ifr,:] = Ytest_hat_remCorr_allSampsFrs_exc00[:,indBestC,:,ifr,:].squeeze()
#                    Ytest_hat_both_allSampsFrs_exc0[:,0,ifr,:] = Ytest_hat_both_allSampsFrs_exc00[:,indBestC,:,ifr,:].squeeze()               

            

            
            # Collect values of all exc shuffles (at bestc)   
            if cbestKnown==False:
                cbestFrs_exc.append(cbestFrs_exc00)
                cbestFrs_remCorr_exc.append(cbestFrs_remCorr_exc00)
                cbestFrs_both_exc.append(cbestFrs_both_exc00)                
            perClassErrorTrain_data_exc.append(perClassErrorTrain_data_exc0) # numShufflesExc x numSamples x nRandCorrSel x numFrames
            perClassErrorTest_data_exc.append(perClassErrorTest_data_exc0)
            perClassErrorTest_shfl_exc.append(perClassErrorTest_shfl_exc0)
            perClassErrorTest_chance_exc.append(perClassErrorTest_chance_exc0)
            perClassErrorTestRemCorr_exc.append(perClassErrorTestRemCorr_exc0)
            perClassErrorTestRemCorr_shfl_exc.append(perClassErrorTestRemCorr_shfl_exc0)
            perClassErrorTestRemCorr_chance_exc.append(perClassErrorTestRemCorr_chance_exc0)
            perClassErrorTestBoth_exc.append(perClassErrorTestBoth_exc0)
            perClassErrorTestBoth_shfl_exc.append(perClassErrorTestBoth_shfl_exc0)
            perClassErrorTestBoth_chance_exc.append(perClassErrorTestBoth_chance_exc0)            
            b_data_exc.append(b_data_exc0)
            w_data_exc.append(w_data_exc0) # numShufflesExc x numSamples x numNeurons x nRandCorrSel x numFrames                        
            Ytest_hat_allSampsFrs_exc.append(Ytest_hat_allSampsFrs_exc0)            
            Ytest_hat_remCorr_allSampsFrs_exc.append(Ytest_hat_remCorr_allSampsFrs_exc0)
            Ytest_hat_both_allSampsFrs_exc.append(Ytest_hat_both_allSampsFrs_exc0)            
            Ytest_allSamps_exc.append(Ytest_allSamps_exc00) # numShufflesExc x numSamples x nRandCorrSel x numTestTrs            
            testTrInds_allSamps_exc.append(testTrInds_allSamps_exc00)
            trsTrainedTestedInds_exc.append(trsTrainedTestedInds_exc00)
            trsnow_allSamps_exc.append(trsnow_allSamps_exc00) # nRandCorrSel x numTrs (test+train)
            trsRemCorrInds_exc.append(trsRemCorrInds_exc00) # nRandCorrSel x numRemCorrTrs     
    
    
            
#%% Quick plots
'''
# corr vs incorr
aa = np.mean(perClassErrorTest_data_inh,axis=0)
a = np.mean(perClassErrorTest_data_inh_incorr,axis=0)

aa = np.mean(perClassErrorTest_data_allExc,axis=0)
a = np.mean(perClassErrorTest_data_allExc_incorr,axis=0)

aa = np.mean(perClassErrorTest_data_exc,axis=(0,1))
a = np.mean(perClassErrorTest_data_exc_incorr,axis=(0,1))

iexc = 1
aa = np.mean(perClassErrorTest_data_exc[iexc],axis=0)
a = np.mean(perClassErrorTest_data_exc_incorr[iexc],axis=0)

aas = np.mean(perClassErrorTest_shfl_inh,axis=0)
aas = np.mean(perClassErrorTest_shfl_allExc,axis=0)
aas = np.mean(perClassErrorTest_shfl_exc[iexc],axis=0)

# decodeOutcome
aa = np.mean(perClassErrorTest_data_exc,axis=(0,1,2))
aas = np.mean(perClassErrorTest_shfl_exc,axis=(0,1,2))

aa = np.mean(perClassErrorTest_data_allExc,axis=(0,1))
aas = np.mean(perClassErrorTest_shfl_allExc,axis=(0,1))

plt.figure()
plt.plot(time_trace, 100-aa, color='k') # corr
plt.plot(time_trace, 100-aas, color='g') # shfl
plt.plot(time_trace, 100-a, color='r') # incorr

'''


            
#%%
####################################################################################################################################
############################################ Save results as .mat files in a folder named svm ####################################################################################################################################
####################################################################################################################################

#%% Save SVM results

# saveResults = 1
if addNs_roc:
    diffn = 'diffNumNsROC_'
    if doInhAllexcEqexc[3]==1:
        h2l = 'hi2loROC_'
    else:
        h2l = 'lo2hiROC_'        
elif addNs_rand:        
    diffn = 'diffNumNsRand_'
    h2l = ''
elif decodeStimCateg:
    diffn = 'decodeStimCateg_'
    h2l = ''
elif decodeOutcome:
    diffn = 'decodeOutcome_'
    h2l = ''
elif same_HRLR_acrossDays:
    diffn = 'sameTrNumAllDays_'
    h2l = ''
elif do_Ctrace:
    diffn = 'Ctrace_'
    h2l = ''
elif decodeChoiceEqCorrIncorr:
    diffn = 'decodeChoiceEqCorrIncorr_'
    h2l = ''
else:
    diffn = ''
    h2l = ''
    
if testIncorr:
    inc = 'testIncorr_'
else:
    inc = ''


if trialHistAnalysis:
    if useEqualTrNums:
        svmn = '%s%sexcInh_SVMtrained_eachFrame_%s%s%s%s%s_prevChoice_%s_ds%d_eqTrs_%s_' %(diffn, h2l, o2a, inc, shflname, ntype, shflTrLabs_n, al,regressBins,nowStr)
    else:
        svmn = '%s%sexcInh_SVMtrained_eachFrame_%s%s%s%s%s_prevChoice_%s_ds%d_%s_' %(diffn, h2l, o2a, inc, shflname, ntype, shflTrLabs_n, al,regressBins,nowStr)
else:
    if useEqualTrNums:
        svmn = '%s%sexcInh_SVMtrained_eachFrame_%s%s%s%s%s_currChoice_%s_ds%d_eqTrs_%s_' %(diffn, h2l, o2a, inc, shflname, ntype, shflTrLabs_n, al,regressBins,nowStr)
    else:
        svmn = '%s%sexcInh_SVMtrained_eachFrame_%s%s%s%s%s_currChoice_%s_ds%d_%s_' %(diffn, h2l, o2a, inc, shflname, ntype, shflTrLabs_n, al,regressBins,nowStr)
print '\n', svmn[:-1]



if saveResults:
    print 'Saving .mat file'
    d = os.path.join(os.path.dirname(pnevFileName), 'svm')
    if not os.path.exists(d):
        print 'creating svm folder'
        os.makedirs(d)

    svmName = os.path.join(d, svmn+os.path.basename(pnevFileName))
    print(svmName)

    ################# 
    if np.all([decodeOutcome==0, decodeStimCateg==0, addNs_roc==0, addNs_rand==0, decodeChoiceEqCorrIncorr==0]):

        if doInhAllexcEqexc[0] == 1: # inh
            if testIncorr:
                scio.savemat(svmName, {'thAct':thAct, 'numShufflesExc':numShufflesExc, 'numSamples':numSamples, 'softNorm':softNorm, 'meanX_fr':meanX_fr, 'stdX_fr':stdX_fr,
                                       'winLen':winLen, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'regType':regType, 'cvect':cvect, 'eventI_ds':eventI_ds, #'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,
                                       'cbest_inh':cbest_inh, 'w_data_inh':w_data_inh, 'b_data_inh':b_data_inh, 'hr_shfl_all':hr_shfl_all, 'lr_shfl_all':lr_shfl_all,
                                       'perClassErrorTrain_data_inh':perClassErrorTrain_data_inh,
                                       'perClassErrorTest_data_inh':perClassErrorTest_data_inh,
                                       'perClassErrorTest_shfl_inh':perClassErrorTest_shfl_inh,
                                       'perClassErrorTest_chance_inh':perClassErrorTest_chance_inh,
                                       'testTrInds_allSamps_inh':testTrInds_allSamps_inh, 
                                       'Ytest_allSamps_inh':Ytest_allSamps_inh,
                                       'Ytest_hat_allSampsFrs_inh':Ytest_hat_allSampsFrs_inh,
                                       'trsnow_allSamps':trsnow_allSamps,
                                       'perClassErrorTest_data_inh_incorr':perClassErrorTest_data_inh_incorr,
                                       'perClassErrorTest_shfl_inh_incorr':perClassErrorTest_shfl_inh_incorr,
                                       'perClassErrorTest_chance_inh_incorr':perClassErrorTest_chance_inh_incorr,
                                       'Ytest_hat_allSampsFrs_inh_incorr':Ytest_hat_allSampsFrs_inh_incorr})
            else:                
                scio.savemat(svmName, {'thAct':thAct, 'numShufflesExc':numShufflesExc, 'numSamples':numSamples, 'softNorm':softNorm, 'meanX_fr':meanX_fr, 'stdX_fr':stdX_fr,
                                       'winLen':winLen, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'regType':regType, 'cvect':cvect, 'eventI_ds':eventI_ds, #'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,
                                       'cbest_inh':cbest_inh, 'w_data_inh':w_data_inh, 'b_data_inh':b_data_inh, 'hr_shfl_all':hr_shfl_all, 'lr_shfl_all':lr_shfl_all,
                                       'perClassErrorTrain_data_inh':perClassErrorTrain_data_inh,
                                       'perClassErrorTest_data_inh':perClassErrorTest_data_inh,
                                       'perClassErrorTest_shfl_inh':perClassErrorTest_shfl_inh,
                                       'perClassErrorTest_chance_inh':perClassErrorTest_chance_inh,
                                       'testTrInds_allSamps_inh':testTrInds_allSamps_inh, 
                                       'Ytest_allSamps_inh':Ytest_allSamps_inh,
                                       'Ytest_hat_allSampsFrs_inh':Ytest_hat_allSampsFrs_inh,
                                       'trsnow_allSamps':trsnow_allSamps})
     
                              
        elif np.logical_or(doInhAllexcEqexc[1] == 1, doInhAllexcEqexc[1] == 2): # allExc or allN
            if testIncorr:
                scio.savemat(svmName, {'thAct':thAct, 'numShufflesExc':numShufflesExc, 'numSamples':numSamples, 'softNorm':softNorm, 'meanX_fr':meanX_fr, 'stdX_fr':stdX_fr,
                                       'winLen':winLen, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'regType':regType, 'cvect':cvect, 'eventI_ds':eventI_ds, #'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,
                                       'cbest_allExc':cbest_allExc, 'w_data_allExc':w_data_allExc, 'b_data_allExc':b_data_allExc, 'hr_shfl_all':hr_shfl_all, 'lr_shfl_all':lr_shfl_all,
                                       'perClassErrorTrain_data_allExc':perClassErrorTrain_data_allExc,
                                       'perClassErrorTest_data_allExc':perClassErrorTest_data_allExc,
                                       'perClassErrorTest_shfl_allExc':perClassErrorTest_shfl_allExc,
                                       'perClassErrorTest_chance_allExc':perClassErrorTest_chance_allExc,
                                       'testTrInds_allSamps_allExc':testTrInds_allSamps_allExc, 
                                       'Ytest_allSamps_allExc':Ytest_allSamps_allExc,
                                       'Ytest_hat_allSampsFrs_allExc':Ytest_hat_allSampsFrs_allExc,
                                       'trsnow_allSamps':trsnow_allSamps,
                                       'perClassErrorTest_data_allExc_incorr':perClassErrorTest_data_allExc_incorr,
                                       'perClassErrorTest_shfl_allExc_incorr':perClassErrorTest_shfl_allExc_incorr,
                                       'perClassErrorTest_chance_allExc_incorr':perClassErrorTest_chance_allExc_incorr,
                                       'Ytest_hat_allSampsFrs_allExc_incorr':Ytest_hat_allSampsFrs_allExc_incorr})
            else:
                scio.savemat(svmName, {'thAct':thAct, 'numShufflesExc':numShufflesExc, 'numSamples':numSamples, 'softNorm':softNorm, 'meanX_fr':meanX_fr, 'stdX_fr':stdX_fr,
                                       'winLen':winLen, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'regType':regType, 'cvect':cvect, 'eventI_ds':eventI_ds, #'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,
                                       'cbest_allExc':cbest_allExc, 'w_data_allExc':w_data_allExc, 'b_data_allExc':b_data_allExc, 'hr_shfl_all':hr_shfl_all, 'lr_shfl_all':lr_shfl_all,
                                       'perClassErrorTrain_data_allExc':perClassErrorTrain_data_allExc,
                                       'perClassErrorTest_data_allExc':perClassErrorTest_data_allExc,
                                       'perClassErrorTest_shfl_allExc':perClassErrorTest_shfl_allExc,
                                       'perClassErrorTest_chance_allExc':perClassErrorTest_chance_allExc,
                                       'testTrInds_allSamps_allExc':testTrInds_allSamps_allExc, 
                                       'Ytest_allSamps_allExc':Ytest_allSamps_allExc,
                                       'Ytest_hat_allSampsFrs_allExc':Ytest_hat_allSampsFrs_allExc,
                                       'trsnow_allSamps':trsnow_allSamps})
        
           
        elif doInhAllexcEqexc[2] == 1: # eqEx      
            if testIncorr:
                scio.savemat(svmName, {'thAct':thAct, 'numShufflesExc':numShufflesExc, 'numSamples':numSamples, 'softNorm':softNorm, 'meanX_fr':meanX_fr, 'stdX_fr':stdX_fr,
                                       'winLen':winLen, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'regType':regType, 'cvect':cvect, 'eventI_ds':eventI_ds, #'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,
                                       'cbest_exc':cbest_exc, 'w_data_exc':w_data_exc, 'b_data_exc':b_data_exc, 'excNsEachSamp':excNsEachSamp, 'hr_shfl_all':hr_shfl_all, 'lr_shfl_all':lr_shfl_all,
                                       'perClassErrorTrain_data_exc':perClassErrorTrain_data_exc,
                                       'perClassErrorTest_data_exc':perClassErrorTest_data_exc,
                                       'perClassErrorTest_shfl_exc':perClassErrorTest_shfl_exc,
                                       'perClassErrorTest_chance_exc':perClassErrorTest_chance_exc,
                                       'testTrInds_allSamps_exc':testTrInds_allSamps_exc, 
                                       'Ytest_allSamps_exc':Ytest_allSamps_exc,
                                       'Ytest_hat_allSampsFrs_exc':Ytest_hat_allSampsFrs_exc,
                                       'trsnow_allSamps':trsnow_allSamps,
                                       'perClassErrorTest_data_exc_incorr':perClassErrorTest_data_exc_incorr,
                                       'perClassErrorTest_shfl_exc_incorr':perClassErrorTest_shfl_exc_incorr,
                                       'perClassErrorTest_chance_exc_incorr':perClassErrorTest_chance_exc_incorr,
                                       'Ytest_hat_allSampsFrs_exc_incorr':Ytest_hat_allSampsFrs_exc_incorr})
            else:
                scio.savemat(svmName, {'thAct':thAct, 'numShufflesExc':numShufflesExc, 'numSamples':numSamples, 'softNorm':softNorm, 'meanX_fr':meanX_fr, 'stdX_fr':stdX_fr,
                                       'winLen':winLen, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'regType':regType, 'cvect':cvect, 'eventI_ds':eventI_ds, #'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,
                                       'cbest_exc':cbest_exc, 'w_data_exc':w_data_exc, 'b_data_exc':b_data_exc, 'excNsEachSamp':excNsEachSamp, 'hr_shfl_all':hr_shfl_all, 'lr_shfl_all':lr_shfl_all,
                                       'perClassErrorTrain_data_exc':perClassErrorTrain_data_exc,
                                       'perClassErrorTest_data_exc':perClassErrorTest_data_exc,
                                       'perClassErrorTest_shfl_exc':perClassErrorTest_shfl_exc,
                                       'perClassErrorTest_chance_exc':perClassErrorTest_chance_exc,
                                       'testTrInds_allSamps_exc':testTrInds_allSamps_exc, 
                                       'Ytest_allSamps_exc':Ytest_allSamps_exc,
                                       'Ytest_hat_allSampsFrs_exc':Ytest_hat_allSampsFrs_exc,
                                       'trsnow_allSamps':trsnow_allSamps})
                
        
        ##################################################
        elif doInhAllexcEqexc[2] == 2: # halfInh_halfExc      
            scio.savemat(svmName, {'thAct':thAct, 'numShufflesExc':numShufflesExc, 'numSamples':numSamples, 'softNorm':softNorm, 'meanX_fr':meanX_fr, 'stdX_fr':stdX_fr,
                                   'winLen':winLen, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'regType':regType, 'cvect':cvect, 'eventI_ds':eventI_ds, #'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,
                                   'cbest_excInhHalf':cbest_excInhHalf, 'w_data_excInhHalf':w_data_excInhHalf, 'b_data_excInhHalf':b_data_excInhHalf, 'excNsEachSamp':excNsEachSamp, 'hr_shfl_all':hr_shfl_all, 'lr_shfl_all':lr_shfl_all,
                                   'perClassErrorTrain_data_excInhHalf':perClassErrorTrain_data_excInhHalf,
                                   'perClassErrorTest_data_excInhHalf':perClassErrorTest_data_excInhHalf,
                                   'perClassErrorTest_shfl_excInhHalf':perClassErrorTest_shfl_excInhHalf,
                                   'perClassErrorTest_chance_excInhHalf':perClassErrorTest_chance_excInhHalf})
    
    
        elif doInhAllexcEqexc[2] == 3: # allExc (size = 2*lenInh)
            scio.savemat(svmName, {'thAct':thAct, 'numShufflesExc':numShufflesExc, 'numSamples':numSamples, 'softNorm':softNorm, 'meanX_fr':meanX_fr, 'stdX_fr':stdX_fr,
                                   'winLen':winLen, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'regType':regType, 'cvect':cvect, 'eventI_ds':eventI_ds, #'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,
                                   'cbest_allExc2inhSize':cbest_allExc2inhSize, 'w_data_allExc2inhSize':w_data_allExc2inhSize, 'b_data_allExc2inhSize':b_data_allExc2inhSize, 'excNsEachSamp_allExc2inhSize':excNsEachSamp_allExc2inhSize, 'hr_shfl_all':hr_shfl_all, 'lr_shfl_all':lr_shfl_all,
                                   'perClassErrorTrain_data_allExc2inhSize':perClassErrorTrain_data_allExc2inhSize,
                                   'perClassErrorTest_data_allExc2inhSize':perClassErrorTest_data_allExc2inhSize,
                                   'perClassErrorTest_shfl_allExc2inhSize':perClassErrorTest_shfl_allExc2inhSize,
                                   'perClassErrorTest_chance_allExc2inhSize':perClassErrorTest_chance_allExc2inhSize})
    
    
    
    ##############################################################################
    ################# decodes stimulus category or trial outcome #################
    ##############################################################################
    elif np.any([decodeOutcome, decodeStimCateg, decodeChoiceEqCorrIncorr]):
        if doInhAllexcEqexc[0] == 1: # inh
            scio.savemat(svmName, {'thAct':thAct, 'numShufflesExc':numShufflesExc, 'numSamples':numSamples, 'softNorm':softNorm, 'meanX_fr':meanX_fr, 'stdX_fr':stdX_fr,
                                   'winLen':winLen, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'regType':regType, 'cvect':cvect, 'eventI_ds':eventI_ds, 'hr_shfl_all':hr_shfl_all, 'lr_shfl_all':lr_shfl_all,
                                   'cbestFrs_inh':cbestFrs_inh, 'cbestFrs_remCorr_inh':cbestFrs_remCorr_inh, 'cbestFrs_both_inh':cbestFrs_both_inh,
                                   'w_data_inh':w_data_inh, 'b_data_inh':b_data_inh, 
                                   'perClassErrorTrain_data_inh':perClassErrorTrain_data_inh,
                                   'perClassErrorTest_data_inh':perClassErrorTest_data_inh,
                                   'perClassErrorTest_shfl_inh':perClassErrorTest_shfl_inh,
                                   'perClassErrorTest_chance_inh':perClassErrorTest_chance_inh,
                                   'perClassErrorTestRemCorr_inh':perClassErrorTestRemCorr_inh, 
                                   'perClassErrorTestRemCorr_shfl_inh':perClassErrorTestRemCorr_shfl_inh, 
                                   'perClassErrorTestRemCorr_chance_inh':perClassErrorTestRemCorr_chance_inh,
                                   'perClassErrorTestBoth_inh':perClassErrorTestBoth_inh, 
                                   'perClassErrorTestBoth_shfl_inh':perClassErrorTestBoth_shfl_inh, 
                                   'perClassErrorTestBoth_chance_inh':perClassErrorTestBoth_chance_inh,                            
                                   'trsnow_allSamps_inh':trsnow_allSamps_inh,
                                   'testTrInds_allSamps_inh':testTrInds_allSamps_inh, 
                                   'trsTrainedTestedInds_inh':trsTrainedTestedInds_inh, 
                                   'trsRemCorrInds_inh':trsRemCorrInds_inh,                                   
                                   'Ytest_allSamps_inh':Ytest_allSamps_inh,
                                   'Ytest_hat_allSampsFrs_inh':Ytest_hat_allSampsFrs_inh,
                                   'Ytest_hat_remCorr_allSampsFrs_inh':Ytest_hat_remCorr_allSampsFrs_inh, 
                                   'Ytest_hat_both_allSampsFrs_inh':Ytest_hat_both_allSampsFrs_inh})
           
        elif np.logical_or(doInhAllexcEqexc[1] == 1, doInhAllexcEqexc[1] == 2): # allExc or allN
            scio.savemat(svmName, {'thAct':thAct, 'numShufflesExc':numShufflesExc, 'numSamples':numSamples, 'softNorm':softNorm, 'meanX_fr':meanX_fr, 'stdX_fr':stdX_fr,
                                   'winLen':winLen, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'regType':regType, 'cvect':cvect, 'eventI_ds':eventI_ds, 'hr_shfl_all':hr_shfl_all, 'lr_shfl_all':lr_shfl_all,
                                   'cbestFrs_allExc':cbestFrs_allExc, 'cbestFrs_remCorr_allExc':cbestFrs_remCorr_allExc, 'cbestFrs_both_allExc':cbestFrs_both_allExc,
                                   'w_data_allExc':w_data_allExc, 'b_data_allExc':b_data_allExc, 
                                   'perClassErrorTrain_data_allExc':perClassErrorTrain_data_allExc,
                                   'perClassErrorTest_data_allExc':perClassErrorTest_data_allExc,
                                   'perClassErrorTest_shfl_allExc':perClassErrorTest_shfl_allExc,
                                   'perClassErrorTest_chance_allExc':perClassErrorTest_chance_allExc,
                                   'perClassErrorTestRemCorr_allExc':perClassErrorTestRemCorr_allExc, 
                                   'perClassErrorTestRemCorr_shfl_allExc':perClassErrorTestRemCorr_shfl_allExc, 
                                   'perClassErrorTestRemCorr_chance_allExc':perClassErrorTestRemCorr_chance_allExc,
                                   'perClassErrorTestBoth_allExc':perClassErrorTestBoth_allExc, 
                                   'perClassErrorTestBoth_shfl_allExc':perClassErrorTestBoth_shfl_allExc, 
                                   'perClassErrorTestBoth_chance_allExc':perClassErrorTestBoth_chance_allExc,                            
                                   'trsnow_allSamps_allExc':trsnow_allSamps_allExc,
                                   'testTrInds_allSamps_allExc':testTrInds_allSamps_allExc, 
                                   'trsTrainedTestedInds_allExc':trsTrainedTestedInds_allExc, 
                                   'trsRemCorrInds_allExc':trsRemCorrInds_allExc,                                   
                                   'Ytest_allSamps_allExc':Ytest_allSamps_allExc,
                                   'Ytest_hat_allSampsFrs_allExc':Ytest_hat_allSampsFrs_allExc,
                                   'Ytest_hat_remCorr_allSampsFrs_allExc':Ytest_hat_remCorr_allSampsFrs_allExc, 
                                   'Ytest_hat_both_allSampsFrs_allExc':Ytest_hat_both_allSampsFrs_allExc})

        elif doInhAllexcEqexc[2] == 1: # eqEx      
            scio.savemat(svmName, {'thAct':thAct, 'numShufflesExc':numShufflesExc, 'numSamples':numSamples, 'softNorm':softNorm, 'meanX_fr':meanX_fr, 'stdX_fr':stdX_fr,
                                   'winLen':winLen, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'regType':regType, 'cvect':cvect, 'eventI_ds':eventI_ds, 'hr_shfl_all':hr_shfl_all, 'lr_shfl_all':lr_shfl_all,
                                   'cbestFrs_exc':cbestFrs_exc, 'cbestFrs_remCorr_exc':cbestFrs_remCorr_exc, 'cbestFrs_both_exc':cbestFrs_both_exc,
                                   'w_data_exc':w_data_exc, 'b_data_exc':b_data_exc, 
                                   'perClassErrorTrain_data_exc':perClassErrorTrain_data_exc,
                                   'perClassErrorTest_data_exc':perClassErrorTest_data_exc,
                                   'perClassErrorTest_shfl_exc':perClassErrorTest_shfl_exc,
                                   'perClassErrorTest_chance_exc':perClassErrorTest_chance_exc,
                                   'perClassErrorTestRemCorr_exc':perClassErrorTestRemCorr_exc, 
                                   'perClassErrorTestRemCorr_shfl_exc':perClassErrorTestRemCorr_shfl_exc, 
                                   'perClassErrorTestRemCorr_chance_exc':perClassErrorTestRemCorr_chance_exc,
                                   'perClassErrorTestBoth_exc':perClassErrorTestBoth_exc, 
                                   'perClassErrorTestBoth_shfl_exc':perClassErrorTestBoth_shfl_exc, 
                                   'perClassErrorTestBoth_chance_exc':perClassErrorTestBoth_chance_exc,                            
                                   'trsnow_allSamps_exc':trsnow_allSamps_exc,
                                   'testTrInds_allSamps_exc':testTrInds_allSamps_exc, 
                                   'trsTrainedTestedInds_exc':trsTrainedTestedInds_exc, 
                                   'trsRemCorrInds_exc':trsRemCorrInds_exc,                                   
                                   'Ytest_allSamps_exc':Ytest_allSamps_exc,
                                   'Ytest_hat_allSampsFrs_exc':Ytest_hat_allSampsFrs_exc,
                                   'Ytest_hat_remCorr_allSampsFrs_exc':Ytest_hat_remCorr_allSampsFrs_exc, 
                                   'Ytest_hat_both_allSampsFrs_exc':Ytest_hat_both_allSampsFrs_exc})       
        
        
        
    #############################################################    
    ############ add Ns 1 by 1 (based on ROC or randomly) #######
    #############################################################
    elif np.logical_or(addNs_roc==1, addNs_rand==1):
        
        if doInhAllexcEqexc[0] == 1: # inh
            scio.savemat(svmName, {'thAct':thAct, 'numShufflesExc':numShufflesExc, 'numSamples':numSamples, 'softNorm':softNorm, 'meanX_fr':meanX_fr, 'stdX_fr':stdX_fr,
                                   'winLen':winLen, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'regType':regType, 'cvect':cvect, 'eventI_ds':eventI_ds, #'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,
                                   'cbest_inh_numNs':cbest_inh_numNs, 'w_data_inh_numNs':w_data_inh_numNs, 'b_data_inh_numNs':b_data_inh_numNs, 'hr_shfl_all':hr_shfl_all, 'lr_shfl_all':lr_shfl_all,
                                   'perClassErrorTrain_data_inh_numNs':perClassErrorTrain_data_inh_numNs,
                                   'perClassErrorTest_data_inh_numNs':perClassErrorTest_data_inh_numNs,
                                   'perClassErrorTest_shfl_inh_numNs':perClassErrorTest_shfl_inh_numNs,
                                   'perClassErrorTest_chance_inh_numNs':perClassErrorTest_chance_inh_numNs})
                                   
        elif doInhAllexcEqexc[1] == 1: # allExc
            scio.savemat(svmName, {'thAct':thAct, 'numShufflesExc':numShufflesExc, 'numSamples':numSamples, 'softNorm':softNorm, 'meanX_fr':meanX_fr, 'stdX_fr':stdX_fr,
                                   'winLen':winLen, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'regType':regType, 'cvect':cvect, 'eventI_ds':eventI_ds, #'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,
                                   'cbest_allExc_numNs':cbest_allExc_numNs, 'w_data_allExc_numNs':w_data_allExc_numNs, 'b_data_allExc_numNs':b_data_allExc_numNs, 'hr_shfl_all':hr_shfl_all, 'lr_shfl_all':lr_shfl_all,
                                   'perClassErrorTrain_data_allExc_numNs':perClassErrorTrain_data_allExc_numNs,
                                   'perClassErrorTest_data_allExc_numNs':perClassErrorTest_data_allExc_numNs,
                                   'perClassErrorTest_shfl_allExc_numNs':perClassErrorTest_shfl_allExc_numNs,
                                   'perClassErrorTest_chance_allExc_numNs':perClassErrorTest_chance_allExc_numNs})
                                   
        elif doInhAllexcEqexc[2] == 1: # eqExc
            scio.savemat(svmName, {'thAct':thAct, 'numShufflesExc':numShufflesExc, 'numSamples':numSamples, 'softNorm':softNorm, 'meanX_fr':meanX_fr, 'stdX_fr':stdX_fr,
                                   'winLen':winLen, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'regType':regType, 'cvect':cvect, 'eventI_ds':eventI_ds, #'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,
                                   'cbest_exc_numNs':cbest_exc_numNs, 'w_data_exc_numNs':w_data_exc_numNs, 'b_data_exc_numNs':b_data_exc_numNs, 'hr_shfl_all':hr_shfl_all, 'lr_shfl_all':lr_shfl_all,
                                   'perClassErrorTrain_data_exc_numNs':perClassErrorTrain_data_exc_numNs,
                                   'perClassErrorTest_data_exc_numNs':perClassErrorTest_data_exc_numNs,
                                   'perClassErrorTest_shfl_exc_numNs':perClassErrorTest_shfl_exc_numNs,
                                   'perClassErrorTest_chance_exc_numNs':perClassErrorTest_chance_exc_numNs})    
else:
    print 'Not saving .mat file'



    
    
    








#%%
"""
'''
########################################################################################################################################################       
#################### common eventI ##############################################################################################    
########################################################################################################################################################    
'''

#%% Loop over days

trace_len = np.full([numDays,1], np.nan).flatten()
eventI_allDays = np.full([numDays,1], np.nan).flatten().astype('int')

for iday in range(len(days)):
    
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
   
   
    #%% Load eventI (we need it for the final alignment of corrClass traces of all days)
    
    # Load stim-aligned_allTrials traces, frames, frame of event of interest
    Data = scio.loadmat(postName, variable_names=['firstSideTryAl'],squeeze_me=True,struct_as_record=False)
#    traces_al_1stSide = Data['firstSideTryAl'].traces.astype('float')
    time_aligned_1stSide = Data['firstSideTryAl'].time.astype('float')
    eventI_ch = Data['firstSideTryAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
    # print(np.shape(traces_al_1stSide))    
        
    eventI_allDays[iday] = eventI_ch
    trace_len[iday] = time_aligned_1stSide.shape[0]
        



#%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
# By common eventI, we  mean the index on which all traces will be aligned.
        
nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
for iday in range(numDays):
    nPost[iday] = (trace_len[iday] - eventI_allDays[iday] - 1)

nPreMin = min(eventI_allDays) # number of frames before the common eventI, also the index of common eventI. 
nPostMin = min(nPost)
print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)


#%% Set the time array for the across-day aligned traces

#a = -(np.asarray(frameLength) * range(nPreMin+1)[::-1])
#b = (np.asarray(frameLength) * range(1, nPostMin+1))
#time_aligned = np.concatenate((a,b))


#%% In downsampled traces

nPreMinD = nPreMin/regressBins
nPostMinD = nPostMin/regressBins
"""

#%%
'''
#%%##### Function for identifying the best regularization parameter
#     Perform 10-fold cross validation to obtain the best regularization parameter
#         More specifically: "crossValidateModel" divides data into training and test datasets. It calls linearSVM.py, which does linear SVM using XTrain, and returns percent class loss for XTrain and XTest.
#     This procedure gets repeated for numSamples (100 times) for each value of regulariazation parameter. 
#     An average across all 100 samples is computed to find the minimum test class loss.
#     Best regularization parameter is defined as the smallest regularization parameter whose test-dataset class loss is within mean+sem of minimum test class loss.

# X is trials x units
def setbesc(X,Y,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs):
    import numpy as np
    from crossValidateModel import crossValidateModel
    from linearSVM import linearSVM
    # numSamples = 10; # number of iterations for finding the best c (inverse of regularization parameter)
    # if you don't want to regularize, go with a very high cbest and don't run the section below.
    # cbest = 10**6
    
    # regType = 'l1'
    # kfold = 10;
    if regType == 'l1':
        print '\nRunning l1 svm classification\r' 
        # cvect = 10**(np.arange(-4, 6,0.2))/numTrials;
        cvect = 10**(np.arange(-4, 6,0.2))/numDataPoints;
    elif regType == 'l2':
        print '\nRunning l2 svm classification\r' 
        cvect = 10**(np.arange(-6, 6,0.2))/numDataPoints;
    
    print 'try the following regularization values: \n', cvect
    # formattedList = ['%.2f' % member for member in cvect]
    # print 'try the following regularization values = \n', formattedList
    
    wAllC = np.ones((numSamples, len(cvect), X.shape[1]))+np.nan;
    bAllC = np.ones((numSamples, len(cvect)))+np.nan;
    
    perClassErrorTrain = np.ones((numSamples, len(cvect)))+np.nan;
    perClassErrorTest = np.ones((numSamples, len(cvect)))+np.nan;
    
    perClassErrorTest_shfl = np.ones((numSamples, len(cvect)))+np.nan;
    perClassErrorTest_chance = np.ones((numSamples, len(cvect)))+np.nan
    
    
    ##############
    hrn = (Y==1).sum()
    lrn = (Y==0).sum()    
    
    if useEqualTrNums and hrn!=lrn: # if the HR and LR trials numbers are not the same, pick equal number of trials of the 2 classes!        
        trsn = min(lrn,hrn)
        if hrn > lrn:
            print 'Subselecting HR trials so both classes have the same number of trials!'
        elif lrn > hrn:
            print 'Subselecting LR trials so both classes have the same number of trials!'
                
    X0 = X + 0
    Y0 = Y + 0


                
    ############## Train SVM numSamples times to get numSamples cross-validated datasets.
    for s in range(numSamples):        
        print 'Iteration %d' %(s)
        
        ############ Make sure both classes have the same number of trials when training the classifier
        if useEqualTrNums and hrn!=lrn: # if the HR and LR trials numbers are not the same, pick equal number of trials of the 2 classes!                    
            if hrn > lrn:
                randtrs = np.argwhere(Y0==1)[rng.permutation(hrn)[0:trsn]].squeeze()
                trsnow = np.sort(np.concatenate((randtrs , np.argwhere(Y0==0).squeeze())))
            elif lrn > hrn:
                randtrs = np.argwhere(Y0==0)[rng.permutation(lrn)[0:trsn]].squeeze() # random sample of the class with more trials
                trsnow = np.sort(np.concatenate((randtrs , np.argwhere(Y0==1).squeeze()))) # all trials of the class with fewer trials + the random sample set above for the other class
                            
            X = X0[trsnow,:]
            Y = Y0[trsnow]
      
            numTrials, numNeurons = X.shape[0], X.shape[1]
            print 'FINAL: %d trials; %d neurons' %(numTrials, numNeurons)                
        
        ######################## Setting chance Y        
        no = Y.shape[0]
        len_test = no - int((kfold-1.)/kfold*no)    
        permIxs = rng.permutation(len_test)    
    
        Y_chance = np.zeros(len_test)
        if rng.rand()>.5:
            b = rng.permutation(len_test)[0:np.floor(len_test/float(2)).astype(int)]
        else:
            b = rng.permutation(len_test)[0:np.ceil(len_test/float(2)).astype(int)]
        Y_chance[b] = 1



        ######################## Loop over different values of regularization
        for i in range(len(cvect)): # train SVM using different values of regularization parameter
            
            if regType == 'l1':                       
                summary, shfl =  crossValidateModel(X, Y, linearSVM, kfold = kfold, l1 = cvect[i], shflTrs = shuffleTrs)
            elif regType == 'l2':
                summary, shfl =  crossValidateModel(X, Y, linearSVM, kfold = kfold, l2 = cvect[i], shflTrs = shuffleTrs)
                        
            wAllC[s,i,:] = np.squeeze(summary.model.coef_); # weights of all neurons for each value of c and each shuffle
            bAllC[s,i] = np.squeeze(summary.model.intercept_);
    
            # classification errors                    
            perClassErrorTrain[s,i] = summary.perClassErrorTrain;
            perClassErrorTest[s,i] = summary.perClassErrorTest;                
            
            # Testing correct shuffled data: 
            # same decoder trained on correct trials, make predictions on correct with shuffled labels.
            perClassErrorTest_shfl[s,i] = perClassError(summary.YTest[permIxs], summary.model.predict(summary.XTest));
            perClassErrorTest_chance[s,i] = perClassError(Y_chance, summary.model.predict(summary.XTest));
            
    
    

    ######################### Find bestc for each frame, and plot the c path    
        
    #######%% Compute average of class errors across numSamples        
    meanPerClassErrorTrain = np.mean(perClassErrorTrain, axis = 0);
    semPerClassErrorTrain = np.std(perClassErrorTrain, axis = 0)/np.sqrt(numSamples);
    
    meanPerClassErrorTest = np.mean(perClassErrorTest, axis = 0);
    semPerClassErrorTest = np.std(perClassErrorTest, axis = 0)/np.sqrt(numSamples);
    
    meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl, axis = 0);
    semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl, axis = 0)/np.sqrt(numSamples);
    
    meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance, axis = 0);
    semPerClassErrorTest_chance = np.std(perClassErrorTest_chance, axis = 0)/np.sqrt(numSamples);
    
    
    #######%% Identify best c        
#        smallestC = 0 # if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
    if smallestC==1:
        print 'bestc = smallest c whose cv error is less than 1se of min cv error'
    else:
        print 'bestc = c that gives min cv error'
    #I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.
    
    
    # Use all range of c... it may end up a value at which all weights are 0.
    ix = np.argmin(meanPerClassErrorTest)
    if smallestC==1:
        cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
        cbest = cbest[0]; # best regularization term based on minError+SE criteria
        cbestAll = cbest
    else:
        cbestAll = cvect[ix]
    print '\t = ', cbestAll
    
    
    ####### Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)
    if regType == 'l1': # in l2, we don't really have 0 weights!
        sys.exit('Needs work! below wAllC has to be for 1 frame') 
        
        a = abs(wAllC)>eps # non-zero weights
        b = np.mean(a, axis=(0,2,3)) # Fraction of non-zero weights (averaged across shuffles)
        c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle
        cvectnow = cvect[c1stnon0:]
        
        meanPerClassErrorTestnow = np.mean(perClassErrorTest[:,c1stnon0:], axis = 0);
        semPerClassErrorTestnow = np.std(perClassErrorTest[:,c1stnon0:], axis = 0)/np.sqrt(numSamples);
        ix = np.argmin(meanPerClassErrorTestnow)
        if smallestC==1:
            cbest = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
            cbest = cbest[0]; # best regularization term based on minError+SE criteria    
        else:
            cbest = cvectnow[ix]
        
        print 'best c (at least 1 non-0 weight) = ', cbest
    else:
        cbest = cbestAll
            
    
    ########%% Set the decoder and class errors at best c (for data)
    """
    # you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
    # we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
    indBestC = np.in1d(cvect, cbest)
    
    w_bestc_data = wAllC[:,indBestC,:,ifr].squeeze() # numSamps x neurons
    b_bestc_data = bAllC[:,indBestC,ifr]
    
    classErr_bestC_train_data = perClassErrorTrain[:,indBestC,ifr].squeeze()
    
    classErr_bestC_test_data = perClassErrorTest[:,indBestC,ifr].squeeze()
    classErr_bestC_test_shfl = perClassErrorTest_shfl[:,indBestC,ifr].squeeze()
    classErr_bestC_test_chance = perClassErrorTest_chance[:,indBestC,ifr].squeeze()
    """
    
    
    ########### Plot C path    
    if doPlots:              
#        print 'Best c (inverse of regularization parameter) = %.2f' %cbest
        plt.figure()
        plt.subplot(1,2,1)
        plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
        plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
    #    plt.fill_between(cvect, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
    #    plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
        
        plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
        plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation')
        plt.plot(cvect, meanPerClassErrorTest_chance, 'b', label = 'cv-chance')       
        plt.plot(cvect, meanPerClassErrorTest_shfl, 'y', label = 'cv-shfl')            
    
        plt.plot(cvect[cvect==cbest], meanPerClassErrorTest[cvect==cbest], 'bo')
        
        plt.xlim([cvect[1], cvect[-1]])
        plt.xscale('log')
        plt.xlabel('c (inverse of regularization parameter)')
        plt.ylabel('classification error (%)')
        plt.legend(loc='center left', bbox_to_anchor=(1, .7))
        
#        plt.title('Frame %d' %(ifr))
        plt.tight_layout()
    
    

    ##############
    return perClassErrorTrain, perClassErrorTest, wAllC, bAllC, cbestAll, cbest, cvect, perClassErrorTest_shfl, perClassErrorTest_chance


'''