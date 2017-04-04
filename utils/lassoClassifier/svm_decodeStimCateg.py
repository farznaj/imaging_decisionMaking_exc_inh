# -*- coding: utf-8 -*-
# NOTE: add for saving meanX, stdX and _chAl mean and std.


"""
Train SVM on equal number of correct and incorrect trials to decode the stimulus category (ie evidence)


IMPORTANT NOTE FOR ALL YOUR STIM-AL (ave over entire stim) analyses:
# since u r doing single-trial analysis, u have to average all trials over the same window... it doesn't make sense to average each trial over a different window.
# I think u should find trsExcluded below, then find min (or max) goTone time (or stimoffset0 time)...   



Created on Fri Feb 10 20:20:08 2017
@author: farznaj
"""


#%% For Jupiter: Add the option to toggle on/off the raw code. Copied from http://stackoverflow.com/questions/27934885/how-to-hide-code-from-cells-in-ipython-notebook-visualized-with-nbviewer

import sys
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


#%% Set vars

import sys
import os
import numpy as np
from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')

# Only run the following section if you are running the code in jupyter or in spyder, but not on the cluster!
if ('ipykernel' in sys.modules) or any('SPYDER' in name for name in os.environ):
    
    # Set these variables:
    mousename = 'fni17' #'fni17' #'fni16' #
    imagingFolder = '151014' #'151023' #'151001' #
    mdfFileNumber = [1] #[1] 

    numSamples = 10 #100; # number of iterations for finding the best c (inverse of regularization parameter)
    nRandCorrSel = 10
    
    minTrPerChoice = 15 # we need at least 15 trials (for a particular choice) to train SVM on...
    earliestGoTone = 50 # 32 (relative to end of stim (after a single repetition)) # In X matrix ideally we don't want trials with go tone within stim (bc we dont want neural activity to be influenced by go tone); however
    # we relax the conditions a bit and allow trials in which go tone happened within the last 50ms of stimulus. (if go tone happened earlier than 50ms, trial will be excluded.)
    
    outcome2ana = 'all' # '', corr', 'incorr' # trials to use for SVM training (all, correct or incorrect trials) # outcome2ana will be used if trialHistAnalysis is 0. When it is 1, by default we are analyzing past correct trials. If you want to change that, set it in the matlab code.        
#    stimAligned = 1 # use stimulus-aligned traces for analyses?
#    doSVM = 1 # use SVM to decode stim category.
#    epAllStim = 1 # when doing svr on stim-aligned traces, if 1, svr will computed on X_svr (ave pop activity during the entire stim presentation); Otherwise it will be computed on X0 (ave pop activity during ep)

    trialHistAnalysis = 0;    
#    roundi = 1; # For the same dataset we run the code multiple times, each time we select a random subset of neurons (of size n, n=.95*numTrials)

    iTiFlg = 1; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.
    setNsExcluded = 1; # if 1, NsExcluded will be set even if it is already saved.    
    neuronType = 2; # 0: excitatory, 1: inhibitory, 2: all types.    
    saveResults = 0; # save results in mat file.

    # for BAGGING:
#    nRandTrSamp = 10 #nRandTrSamp = 10 #1000 # number of times to subselect trials (for the BAGGING method)
#    doData = 1
    
#    doNsRand = 0; # if 1, a random set of neurons will be selected to make sure we have fewer neurons than trials. 
    regType = 'l2' # 'l2' : regularization type
    kfold = 10;
#    compExcInh = 0 # if 1, analyses will be run to compare exc inh neurons.
    
    # The following vars don't usually need to be changed    
    doPlots = 1; # Whether to make plots or not.
    saveHTML = 0; # whether to save the html file of notebook with all figures or not.

#    if trialHistAnalysis==1: # more parameters are specified in popClassifier_trialHistory.m
    #        iTiFlg = 1; # 0: short ITI, 1: long ITI, 2: all ITIs.
#        epEnd_rel2stimon_fr = 0 # 3; # -2 # epEnd = eventI + epEnd_rel2stimon_fr
    if trialHistAnalysis==0:
        # not needed to set ep_ms here, later you define it as [choiceTime-300 choiceTime]ms # we also go 30ms back to make sure we are not right on the choice time!
#        ep_ms = [809, 1109] #[425, 725] # optional, it will be set according to min choice time if not provided.# training epoch relative to stimOnset % we want to decode animal's upcoming choice by traninig SVM for neural average responses during ep ms after stimulus onset. [1000, 1300]; #[700, 900]; # [500, 700]; 
        strength2ana = 'all' # 'all', easy', 'medium', 'hard' % What stim strength to use for training?
        thStimStrength = 3; # 2; # threshold of stim strength for defining hard, medium and easy trials.
        th_stim_dur = 800; # min stim duration to include a trial in timeStimOnset

    trs4project = 'trained' # 'trained', 'all', 'corr', 'incorr' # trials that will be used for projections and the class accuracy trace; if 'trained', same trials that were used for SVM training will be used. "corr" and "incorr" refer to current trial's outcome, so they don't mean much if trialHistAnalysis=1. 
#    windowAvgFlg = 1 # if 0, data points during ep wont be averaged when setting X (for SVM training), instead each frame of ep will be treated as a separate datapoint. It helps with increasing number of datapoints, but will make data mor enoisy.

    thAct = 5e-4 #5e-4; # 1e-5 # neurons whose average activity during ep is less than thAct will be called non-active and will be excluded.
    thTrsWithSpike = 1; # 3 % remove neurons that are active in <thSpTr trials.

    pnev2load = [] #[] [3] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.



eps = sys.float_info.epsilon #2.2204e-16
#nRandTrSamp = numSamples #10 #1000 # number of times to subselect trials (for the BAGGING method)


#%% Print vars

if neuronType==0:
    ntName = 'excit'
elif neuronType==1:
    ntName = 'inhibit'
elif neuronType==2:
    ntName = 'all'     

if trialHistAnalysis==1:    
    if iTiFlg==0:
        itiName = 'short'
    elif iTiFlg==1:
        itiName = 'long'
    elif iTiFlg==2:
        itiName = 'all'        
    
print 'Analyzing %s' %(mousename+'_'+imagingFolder+'_'+str(mdfFileNumber)) 
if trialHistAnalysis==0:
    print 'Training %s trials of strength %s. Making projections for %s trials' %(outcome2ana, strength2ana, trs4project)
print 'trialHistAnalysis = %i' %(trialHistAnalysis)
print 'Analyzing %s neurons' %(ntName)
if trialHistAnalysis==1:
    print 'Analyzing %s ITIs' %(itiName)
#elif 'ep_ms' in locals():
#    print 'training window: [%d %d] ms' %(ep_ms[0], ep_ms[1])
#print 'windowAvgFlg = %i' %(windowAvgFlg)
print 'numSamples = %i' %(numSamples)
print 'earliestGoTone = %d ms before stim end (after a single repetition)' %(earliestGoTone)
print 'minTrPerChoice = %d' %(minTrPerChoice)

# ## Import Libraries and Modules

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


#%% Define ttest2

# Extend the built in two tailed ttest function to one-tailed
def ttest2(a, b, **tailOption):
    import scipy.stats as stats
    import numpy as np
    h, p = stats.ttest_ind(a, b)
    d = np.mean(a)-np.mean(b)
    if tailOption.get('tail'):
        tail = tailOption.get('tail').lower()
        if tail == 'right':
            p = p/2.*(d>0) + (1-p/2.)*(d<0)
        elif tail == 'left':
            p = (1-p/2.)*(d>0) + p/2.*(d<0)
    if d==0:
        p = 1;
    return p


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
#    import numpy as np
    import platform
    import glob
    import os.path
        
    if len(pnev2load)==0:
        pnev2load = [0];
            
    ##%%
    dataPath = []
    if platform.system()=='Linux':
        if os.getcwd().find('sonas')==1: # server
            dataPath = '/sonas-hs/churchland/nlsas/data/data/'
        else: # office linux
            dataPath = '/home/farznaj/Shares/Churchland/data/'
    else:
        dataPath = '/Users/gamalamin/git_local_repository/Farzaneh/data/'
        
    ##%%        
    tifFold = os.path.join(dataPath+mousename,'imaging',imagingFolder)
    r = '%03d-'*len(mdfFileNumber)
    r = r[:-1]
    rr = r % (tuple(mdfFileNumber))
    
    date_major = imagingFolder+'_'+rr
    imfilename = os.path.join(tifFold,date_major+'.mat')
    
    ##%%
    if len(signalCh)>0:
        if postNProvided:
            pnevFileName = 'post_'+date_major+'_ch'+str(signalCh)+'-Pnev*'
        else:
            pnevFileName = date_major+'_ch'+str(signalCh)+'-Pnev*'
            
        pnevFileName = glob.glob(os.path.join(tifFold,pnevFileName))   
        # sort pnevFileNames by date (descending)
        pnevFileName = sorted(pnevFileName, key=os.path.getmtime)
        pnevFileName = pnevFileName[::-1]
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
    if len(y)!=len(yhat):
        sys.exit('y and yhat mush have the same length')
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



###########################################################################################################################################
#%% Load matlab variables: event-aligned traces, inhibitRois, outcomes,  choice, etc
#     - traces are set in set_aligned_traces.m matlab script.

# Set traces_al_stim that is same as traces_al_stimAll except that in traces_al_stim some trials are set to nan, bc their stim duration is < 
# th_stim_dur or bc their go tone happens before ep(end) or bc their choice happened before ep(end). 
# But in traces_al_stimAll, all trials are included. 
# You need traces_al_stim for decoding the upcoming choice bc you average responses during ep and you want to 
# control for what happens there. But for trial-history analysis you average responses before stimOnset, so you 
# don't care about when go tone happened or how long the stimulus was. 

frameLength = 1000/30.9; # sec.
    
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
DataS = Data
traces_al_stim = traces_al_stimAll


# Load outcomes and choice (allResp_HR_LR) for the current trial
# if trialHistAnalysis==0:
Data = scio.loadmat(postName, variable_names=['outcomes', 'allResp_HR_LR'])
outcomes = (Data.pop('outcomes').astype('float'))[0,:]
outcomes0 = outcomes+0
# allResp_HR_LR = (Data.pop('allResp_HR_LR').astype('float'))[0,:]
allResp_HR_LR = np.array(Data.pop('allResp_HR_LR')).flatten().astype('float')
choiceVecAll = allResp_HR_LR+0;  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
# choiceVecAll = np.transpose(allResp_HR_LR);  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
print 'Current outcome: %d correct choices; %d incorrect choices' %(sum(outcomes==1), sum(outcomes==0))


if trialHistAnalysis:
    # Load trialHistory structure to get choice vector of the previous trial
    Data = scio.loadmat(postName, variable_names=['trialHistory'],squeeze_me=True,struct_as_record=False)
    choiceVec0All = Data['trialHistory'].choiceVec0.astype('float')


    
# Set trials strength and identify trials with stim strength of interest
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


###########################################################################################################################################
#%% Set the time window for training SVM (ep) also remove some trials from traces_al_stim (if their choice is too early or their stim duration is too short)

# Load time of some trial events    
Data = scio.loadmat(postName, variable_names=['timeCommitCL_CR_Gotone', 'timeStimOnset', 'timeStimOffset', 'time1stSideTry'])
timeCommitCL_CR_Gotone = np.array(Data.pop('timeCommitCL_CR_Gotone')).flatten().astype('float')
timeStimOnset = np.array(Data.pop('timeStimOnset')).flatten().astype('float')
timeStimOffset = np.array(Data.pop('timeStimOffset')).flatten().astype('float')
time1stSideTry = np.array(Data.pop('time1stSideTry')).flatten().astype('float')


if doPlots:
    plt.figure
    plt.subplot(1,2,1)
    plt.plot(timeCommitCL_CR_Gotone - timeStimOnset, label = 'goTone')
    plt.plot(timeStimOffset - timeStimOnset, 'r', label = 'stimOffset')
    plt.plot(time1stSideTry - timeStimOnset, 'm', label = '1stSideTry')
    plt.plot([1, np.shape(timeCommitCL_CR_Gotone)[0]],[th_stim_dur, th_stim_dur], 'g:', label = 'th_stim_dur')
#    plt.plot([1, np.shape(timeCommitCL_CR_Gotone)[0]],[ep_ms[-1], ep_ms[-1]], 'k:', label = 'epoch end')
    plt.xlabel('Trial')
    plt.ylabel('Time relative to stim onset (ms)')
    plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 
    # minStimDurNoGoTone = np.nanmin(timeCommitCL_CR_Gotone - timeStimOnset); # this is the duration after stim onset during which no go tone occurred for any of the trials.
    # print 'minStimDurNoGoTone = %.2f ms' %minStimDurNoGoTone
        

###########################################################################################################################################
#%% Load inhibitRois and set traces for specific neuron types: inhibitory, excitatory or all neurons

Data = scio.loadmat(moreName, variable_names=['inhibitRois'])
inhibitRois = Data.pop('inhibitRois')[0,:]
# print '%d inhibitory, %d excitatory; %d unsure class' %(np.sum(inhibitRois==1), np.sum(inhibitRois==0), np.sum(np.isnan(inhibitRois)))

    
# Set traces for specific neuron types: inhibitory, excitatory or all neurons
if neuronType!=2:    
    nt = (inhibitRois==neuronType) # 0: excitatory, 1: inhibitory, 2: all types.
    # good_excit = inhibitRois==0;
    # good_inhibit = inhibitRois==1;        
    
    traces_al_stim = traces_al_stim[:, nt, :];
    traces_al_stimAll = traces_al_stimAll[:, nt, :];
else:
    nt = np.arange(np.shape(traces_al_stim)[1])    


      
###########################################################################################################################################
#%% Set X (trials x neurons) and Y (trials x 1) for training the SVM classifier.
#     X matrix (size trials x neurons); steps include:
#        - average neural responses during ep for each trial.
#        - remove nan trials (trsExcluded)
#        - exclude non-active neurons (NsExcluded)
#        - [if doing doRands, get X only for NsRand]
#        - center and normalize X (by meanX and stdX)

#     Y choice of high rate (modeled as 1) and low rate (modeled as 0)


##%% Set choiceVec0  (Y: the response vector)
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


'''
# Set spikeAveEp0  (X: the predictor matrix (trials x neurons) that shows average of spikes for a particular epoch for each trial and neuron.)
if trialHistAnalysis:
    # either of the two cases below should be fine (init-aligned traces or stim-aligned traces.)
    spikeAveEp0 = np.transpose(np.nanmean(traces_al_stimAll[ep,:,:], axis=0)) # trials x neurons    
#    spikeAveEp0_svr = np.transpose(np.nanmean(traces_al_stimAll[ep,:,:], axis=0)) # trials x neurons    
else:    
    spikeAveEp0 = np.transpose(np.nanmean(traces_al_stim[ep,:,:], axis=0)) # trials x neurons    

# X = spikeAveEp0;
print 'Size of spikeAveEp0 (trs x neurons): ', spikeAveEp0.shape
'''

#%% Set stimulus-aligned traces (X, Y and Ysr)
'''
# set trsExcluded and exclude them to set X and Y; trsExcluded are trials that are nan either in traces or in choice vector.
# Identify nan trials
trsExcluded = (np.sum(np.isnan(spikeAveEp0), axis = 1) + np.isnan(choiceVec0)) != 0 # NaN trials # trsExcluded
# print sum(trsExcluded), 'NaN trials'

# Exclude nan trials
X = spikeAveEp0[~trsExcluded,:]; # trials x neurons
Y = choiceVec0[~trsExcluded];
print '%d high-rate choices, and %d low-rate choices\n' %(np.sum(Y==1), np.sum(Y==0))


#trsExcludedSr = (np.isnan(np.sum(spikeAveEp0, axis=(0,1)))) != 0 # for the stimulus decoder we dont care if trial was correct or incorrect
Ysr = stimrate[~trsExcluded] #exclude incorr trials, etc # we dont need to exclude trials that animal did not make a choice for the stimulus decoder analysis... unless you want to use the same set of trials for the choice and stimulus decoder...
'''

#%% Set X_svr, ie average popoulation activity from stim onset to go tone... we will use it for stim decoding analysis (svr).
# Set ep for decoding stimulus (we want to average population activity during the entire stimulus presentation)
# Set spikeAveEp0_svr: average population activity during ep_svr
# We do this only for stimulus-aligned traces ... we dont know on choice-aligned traces when is the stimulus presented.
# This is to use our best guess for when popoulation activity represents the stimulus.

# use a fixed ep_svr for all trials
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

# we cannot do below, bc traces_al_stim lasts until go tone, so we have to define ep_svr as stimOnset until goTone
# bc in some sessions stim continues after the choice, I don't use below, instead ep_svr = stim onset to choice, ie ave pop activity from stim onset until choice is made.
# instead of averaging over a fixed window for all trials, average each trial during its stim presentation... this helps when stim was on for a different dur for each trial.
spikeAveEp0 = np.full((traces_al_stim.shape[2], traces_al_stim.shape[1]), np.nan)
for tr in range(traces_al_stim.shape[2]):
#    stimDurNow = np.floor(timeStimOffset[tr]-timeStimOnset[tr])
#    stimDurNow = np.floor(time1stSideTry[tr]-timeStimOnset[tr])
    stimDurNow = np.floor(timeCommitCL_CR_Gotone[tr]-timeStimOnset[tr])
    epnow = np.arange(eventI+1, eventI + np.round(stimDurNow/frameLength) + 1).astype(int)
    spikeAveEp0[tr,:] = np.mean(traces_al_stim[epnow,:,tr], axis=0) # neurons # average of frames during epnow for each neuron in trial tr


###### IMPORTANT NOTE: ######
# since u r doing single-trial analysis, u have to average all trials over the same window... it doesn't make sense to average each trial over a different window.
# I think u should find trsExcluded below, then find min (or max) goTone time (or stimoffset0 time)...   


#%% Set nan trials

# to find trials in which go tone happened within the stim, we need to know the time of stimOffset for a single repetition of the stim (without extra stim, etc)
# but timeStimOffset includes extra stim, etc... so we set timeStimeOffset0 (which is stimOffset after 1 repetition) here
import datetime
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
        sys.exit('Aborting ... needs work... u started saving timeSingleStimOffset... if not saved, below needs work for days with multiple sessions!')
#        bnow = bFileName[0]       
#        Data = scio.loadmat(bnow, variable_names=['all_data'],squeeze_me=True,struct_as_record=False)
#        waitDuration_all.append(np.array([Data['all_data'][i].waitDuration for i in range(len(Data['all_data']))])) # remember difference indexing in matlab and python!
#    sdur = timeStimOnset + waitDuration_all
timeStimOffset0 = timeStimOnset + sdur


#X_svr = spikeAveEp0_svr[~trsExcluded,:]; # trials x neurons

# Trials to excluce: go tone happened within stimulus (we are permissive if go tone happened within the last 50ms of the stim)
#earliestGoTone = 50 # 32 # In X matrix ideally we don't want trials with go tone within stim (bc we dont want neural activity to be influenced by go tone); however
# we relax the conditions a bit and allow trials in which go tone happened within the last 50ms of stimulus. (if go tone happened earlier than 50ms, trial will be excluded.)
trsExcluded = np.logical_or(timeCommitCL_CR_Gotone < timeStimOffset0-earliestGoTone, np.isnan(timeCommitCL_CR_Gotone - timeStimOffset0))
#trsExcluded_svr = np.logical_or(timeCommitCL_CR_Gotone < timeStimOffset-earliestGoTone, np.isnan(timeCommitCL_CR_Gotone - timeStimOffset))
print (~trsExcluded).sum(), 'out of', trsExcluded.shape[0]
# also exclude any nan trials in choiceVec and spikeAveEp0_svr ... I think trsExcluded_svr computed above already excludes all nans in spikeAveEp0_svr... otherwise I dont know how spikeAve can have nan and not be in trsExcluded_svr
trsExcluded = np.logical_or(trsExcluded, np.isnan(np.sum(spikeAveEp0, axis = 1)))
print (~trsExcluded).sum()
## remove nan trials in choiceVec
##trsExcluded = np.logical_or(trsExcluded, np.isnan(choiceVec0))
#print (~trsExcluded).sum()
# remove cb trials (bc we cannot categorize them as HR or LR)
trsExcluded = np.logical_or(trsExcluded, stimrate==cb)
print (~trsExcluded).sum()

#%% Exclude nan trials and set final X and Y

X = spikeAveEp0[~trsExcluded,:]; # trials x neurons
#Y = choiceVec0[~trsExcluded]; # if you want to decoder choice, use this
# Decode stimulus category
Y = stimrate[~trsExcluded]
Y[Y<cb] = 0 # LR stimulus --> 0
Y[Y>cb] = 1 # HR stimulus --> 1
print '%d high-rate stimuli, and %d low-rate stimuli\n' %(np.sum(Y==1), np.sum(Y==0))
print X.shape, Y.shape

outcomes = outcomes0[~trsExcluded]


#%% Set NsExcluded : Identify neurons that did not fire in any of the trials (during ep) and then exclude them. Otherwise they cause problem for feature normalization.
# thAct and thTrsWithSpike are parameters that you can play with.

# If it is already saved, load it (the idea is to use the same NsExcluded for all the analyses of a session). Otherwise set it.
if trialHistAnalysis==0:
    svmnowname = 'svmCurrChoice_allN' + '_*-' + pnevFileName[-32:]
else:
    svmnowname = 'svmPrevChoice_allN_allITIs' + '_*-' + pnevFileName[-32:]
svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmnowname))
svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    

if setNsExcluded==0 and np.shape(svmName)[0]!=0: # NsExcluded is already set and saved # 0: #    
    svmName = svmName[0]  # get the latest file
    print 'loading NsExcluded from file', svmName    
    Data = scio.loadmat(svmName, variable_names=['NsExcluded'])
    NsExcluded = Data.pop('NsExcluded')[0,:].astype('bool')
    NsExcluded = NsExcluded[nt]      

    stdX = np.std(X, axis = 0); # define stdX for all neurons; later we reset it only including active neurons
    if min(stdX[~NsExcluded]) < thAct: # make sure the loaded NsExcluded makes sense; ie stdX of ~NsExcluded is above thAct
        sys.exit(('min of stdX= %.8f; not supposed to be <%d (thAct)!'  %(min(stdX), thAct)))
else:
    
    print 'NsExcluded not saved, so setting it here'
    
    if trialHistAnalysis and iTiFlg!=2:
        # NEEDS WORK
        # set X for short-ITI and long-ITI cases (XS,XL).
        trsExcludedS = (np.sum(np.isnan(spikeAveEp0), axis = 1) + np.isnan(choiceVec0S)) != 0 
        XS = spikeAveEp0[~trsExcludedS,:]; # trials x neurons
        trsExcludedL = (np.sum(np.isnan(spikeAveEp0), axis = 1) + np.isnan(choiceVec0L)) != 0 
        XL = spikeAveEp0[~trsExcludedL,:]; # trials x neurons

        # Define NsExcluded as neurons with low stdX for either short ITI or long ITI trials. 
        # This is to make sure short and long ITI cases will include the same set of neurons.
        stdXS = np.std(XS, axis = 0);
        stdXL = np.std(XL, axis = 0);

        NsExcluded = np.sum([stdXS < thAct, stdXL < thAct], axis=0)!=0 # if a neurons is non active for either short ITI or long ITI trials, exclude it.
    
    else:

        # NOT SURE: bc we are going to test on incorr trials too, I think we should define
        # NsExcluded based on the activity in all trials not just corr trials.
    
        # NsExcluded defined based on corr trials... so some of these neurons may not be active for the average of incorr trials.
    
    
        # NsExcluded defined based on both corr and incorr trials
    
        # Define NsExcluded as neurons with low stdX
        stdX = np.std(X, axis = 0);
        NsExcluded = stdX < thAct

print '%d = Final # non-active neurons' %(sum(NsExcluded))
# a = size(spikeAveEp0,2) - sum(NsExcluded);
print 'Using %d out of %d neurons; Fraction excluded = %.2f\n' %(np.shape(spikeAveEp0)[1]-sum(NsExcluded), np.shape(spikeAveEp0)[1], sum(NsExcluded)/float(np.shape(spikeAveEp0)[1]))


print '%i, %i, %i: #original inh, excit, unsure' %(np.sum(inhibitRois==1), np.sum(inhibitRois==0), np.sum(np.isnan(inhibitRois)))
# Check what fraction of inhibitRois are excluded, compare with excitatory neurons.
if neuronType==2:    
    print '%i, %i, %i: #excluded inh, excit, unsure' %(np.sum(inhibitRois[NsExcluded]==1), np.sum(inhibitRois[NsExcluded]==0), np.sum(np.isnan(inhibitRois[NsExcluded])))
    print '%.2f, %.2f, %.2f: fraction excluded inh, excit, unsure\n' %(np.sum(inhibitRois[NsExcluded]==1)/float(np.sum(inhibitRois==1)), np.sum(inhibitRois[NsExcluded]==0)/float(np.sum(inhibitRois==0)), np.sum(np.isnan(inhibitRois[NsExcluded]))/float(np.sum(np.isnan(inhibitRois))))



#%% Exclude non-active neurons from X and set inhRois (ie neurons that don't fire in any of the trials during ep)

##### stimulus-aligned
X = X[:,~NsExcluded]
print 'stim-aligned traces (trs x units): ', np.shape(X)
    
# Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)
if neuronType==2:
    inhRois = inhibitRois[~NsExcluded]
    # print 'Number: inhibit = %d, excit = %d, unsure = %d' %(np.sum(inhRois==1), np.sum(inhRois==0), np.sum(np.isnan(inhRois)))
    # print 'Fraction: inhibit = %.2f, excit = %.2f, unsure = %.2f' %(fractInh, fractExc, fractUn)

    
# Handle imbalance in the number of trials:
# unlike matlab, it doesn't seem to be a problem here... so we don't make trial numbers of HR and LR the same.
    
    
# Print some numbers
numDataPoints = X.shape[0] 
print '# data points = %d' %numDataPoints
# numTrials = (~trsExcluded).sum()
# numNeurons = (~NsExcluded).sum()
numTrials, numNeurons = X.shape
print 'stim-aligned traces: %d trials; %d neurons' %(numTrials, numNeurons)
# print ' The data has %d frames recorded from %d neurons at %d trials' %Xt.shape    


#%% Center and normalize X: feature normalization and scaling: to remove effects related to scaling and bias of each neuron, we need to zscore data (i.e., make data mean 0 and variance 1 for each neuron) 

#### stimulus-aligned
meanX = np.mean(X, axis = 0);
stdX = np.std(X, axis = 0);

# normalize X
X = (X-meanX)/stdX;


#%% save a copy of X and Y (DO NOT run this after you run the parts below that redefine X and Y!)

X0 = X+0
Y0 = Y+0


#%% Plot mean and std of X

if doPlots:
    # correct trials
    
    # stim-al
    plt.figure
    plt.subplot(2,2,1)
    plt.plot(meanX)
    plt.ylabel('meanX \n(ep-ave FR, mean of trials)')
    plt.title('min = %.6f' %(np.min(meanX)))

    plt.subplot(2,2,3)
    plt.plot(stdX)
    plt.ylabel('stdX \n(ep-ave FR, std of trials)')
    plt.xlabel('neurons')
    plt.title('min = %.6f' %(np.min(stdX)))
    plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.subplots_adjust(hspace=.5)


#%% Plot traces averaged for each choice, for both correct and incorrect trials; aligned on both stimulus and choice.

if doPlots:

    ######################################## Plot stim-aligned averages after centering and normalization
    plt.figure()
    #################### correct trials
    # Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
    hr_trs = (Y==1)
    lr_trs = (Y==0)


    # Set traces
    Xt = traces_al_stim[:, :, ~trsExcluded];
    # Exclude non-active neurons (ie neurons that don't fire in any of the trials during ep)
    Xt = Xt[:,~NsExcluded,:]
    ## Feature normalization and scaling    
    # normalize stim-aligned traces
    T, N, C = Xt.shape
    Xt_N = np.reshape(Xt.transpose(0 ,2 ,1), (T*C, N), order = 'F')
    Xt_N = (Xt_N-meanX)/stdX
    Xt = np.reshape(Xt_N, (T, C, N), order = 'F').transpose(0 ,2 ,1)


    # Plot
    plt.subplot(1,2,1) # I think you should use Xtsa here to make it compatible with the plot above.
    a1 = np.nanmean(Xt[:, :, hr_trs],  axis=1) # frames x trials (average across neurons)
    tr1 = np.nanmean(a1,  axis = 1)
    tr1_se = np.nanstd(a1,  axis = 1) / np.sqrt(numTrials);
    a0 = np.nanmean(Xt[:, :, lr_trs],  axis=1) # frames x trials (average across neurons)
    tr0 = np.nanmean(a0,  axis = 1)
    tr0_se = np.nanstd(a0,  axis = 1) / np.sqrt(numTrials);
    plt.fill_between(time_aligned_stim, tr1-tr1_se, tr1+tr1_se, alpha=0.5, edgecolor='b', facecolor='b')
    plt.fill_between(time_aligned_stim, tr0-tr0_se, tr0+tr0_se, alpha=0.5, edgecolor='r', facecolor='r')
    plt.plot(time_aligned_stim, tr1, 'b', label = 'high rate')
    plt.plot(time_aligned_stim, tr0, 'r', label = 'low rate')
    # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, lr_trs],  axis = (1, 2)), 'r', label = 'high rate')
    # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, hr_trs],  axis = (1, 2)), 'b', label = 'low rate')
    plt.xlabel('time aligned to stimulus onset (ms)')
    plt.title('Correct trials')
    
    


#%% SVM - stimulus-aligned - epAllStim: use stim-aligned traces that are averaged during stim presentation
#################################################################################
#################################################################################
#################################################################################
##############################  SVM, L1/L2 penalty; epAllStim; decode stim category ('evidence')  ##################################
###############################################################################   
    
#%% Find the optimal regularization parameter:
# Rrain SVM on a range of c, using training and testing dataset ... 
#    make class predictions for testing dataset and incorrect trials. 
#    Also shuffle labels for correct and incorrect trials and make predictions (to set null distributions)
# Repeat the above procedure for numSamples times.


# numSamples = 10; # number of iterations for finding the best c (inverse of regularization parameter)
# if you don't want to regularize, go with a very high cbest and don't run the section below.
# cbest = 10**6


nc = (outcomes==1).sum()
ni = (outcomes==0).sum()
print nc, ' correct trials; ', ni, ' incorrect trials'
trsCorr = np.argwhere(outcomes==1).squeeze()
trsIncorr = np.argwhere(outcomes==0).squeeze()



if regType == 'l1':
    print '\nRunning l1 svm classification\r' 
    # cvect = 10**(np.arange(-4, 6,0.2))/numTrials;
    cvect = 10**(np.arange(-4, 6,0.2))/len(trsIncorr)*2 #X.shape[0];
#    cvect = 10**(np.arange(-4, 3.11e2,10.5))/X.shape[0];
elif regType == 'l2':
    print '\nRunning l2 svm classification\r' 
    cvect = 10**(np.arange(-6, 6,0.2))/len(trsIncorr)*2 #X.shape[0]
#    cvect = 10**(np.arange(-6, 6,0.2));

print 'try the following regularization values: \n', cvect
# formattedList = ['%.2f' % member for member in cvect]
# print 'try the following regularization values = \n', formattedList


#%% Pick n random corr trials, where n = num incorr trials


wAllC = np.ones((numSamples, len(cvect), X.shape[1], nRandCorrSel))+np.nan;
bAllC = np.ones((numSamples, len(cvect), nRandCorrSel))+np.nan;

perClassErrorTrain = np.ones((numSamples, len(cvect), nRandCorrSel))+np.nan;
perClassErrorTest = np.ones((numSamples, len(cvect), nRandCorrSel))+np.nan;
perClassErrorTest_shfl = np.ones((numSamples, len(cvect), nRandCorrSel))+np.nan;
perClassErrorTest_chance = np.ones((numSamples, len(cvect), nRandCorrSel))+np.nan

perClassErrorTestRemCorr = np.ones((numSamples, len(cvect), nRandCorrSel))+np.nan 
perClassErrorTestRemCorr_shfl = np.ones((numSamples, len(cvect), nRandCorrSel))+np.nan
perClassErrorTestRemCorr_chance = np.ones((numSamples, len(cvect), nRandCorrSel))+np.nan

perClassErrorTestBoth = np.ones((numSamples, len(cvect), nRandCorrSel))+np.nan 
perClassErrorTestBoth_shfl = np.ones((numSamples, len(cvect), nRandCorrSel))+np.nan
perClassErrorTestBoth_chance = np.ones((numSamples, len(cvect), nRandCorrSel))+np.nan

trsTrainedTestedInds = np.ones((len(trsIncorr)*2, numSamples, len(cvect), nRandCorrSel))+np.nan
trsRemCorrInds = np.ones((len(trsCorr)-len(trsIncorr), numSamples, len(cvect), nRandCorrSel))+np.nan


#nRandCorrSel = numSamples # number of times to randomly draw correc trials.
for sc in range(nRandCorrSel):

    print 'Rand corr subselect %d' %(sc)
#    trsTrainedTestedInds[s,i,sc] = np.ones((len(trsIncorr)*2))+np.nan # training and testing dataset in X
#    trsRemCorrInds[s,i,sc] =  np.ones((len(trsCorr)-len(trsIncorr)))+np.nan # correct trials did not use in X
    
    
    #%% indices are from outcomes array
    
    trsNowCorr = trsCorr[rng.permutation(nc)[0:ni]] # n_incorr random indeces from corr trials
    trsNowAll = np.sort(np.concatenate((trsNowCorr, trsIncorr))) # set the indices for all trials to be trained
#    print trsNowAll.shape
    
     
    #%% Set X and Y for the random subset
           
    X = X0[trsNowAll, :]
    Y = Y0[trsNowAll]
    print X.shape, Y.shape
    
    
    # set X and Y for the remaining correct trials (those not in trsNowCorr). We will test on these trials the decoder trained on X and Y
    trsRemCorr = trsCorr[~np.in1d(trsCorr, trsNowCorr)] # those correct trials (indeces in outcomes) that are not among the random pick for current trsNowCorr
    X_remCorr = X0[trsRemCorr, :]
    Y_remCorr = Y0[trsRemCorr]
    print X_remCorr.shape, Y_remCorr.shape
    
    
    #%%        
    
    no = X.shape[0]
    len_test = no - int((kfold-1.)/kfold*no)
    
    for s in range(numSamples):
        
        print '\titeration %d' %(s)
        permIxs_remCorr = rng.permutation(X_remCorr.shape[0]);
    #    permIxs = rng.permutation(X.shape[0]);
        permIxs = rng.permutation(len_test)
        permIxs_tot = rng.permutation(len_test+X_remCorr.shape[0])
    
        a_corr = np.zeros(len_test)
        if rng.rand()>.5:
            b = rng.permutation(len_test)[0:np.floor(len_test/float(2)).astype(int)]
        else:
            b = rng.permutation(len_test)[0:np.ceil(len_test/float(2)).astype(int)]
        a_corr[b] = 1
        
        a_remCorr = np.zeros(Y_remCorr.shape[0])
        if rng.rand()>.5: # in case Y_remCorr has odd size, we make sure sometimes we round it up and sometimes down so the number of trials of the 2 classes are balanced in chance data.
            b = rng.permutation(Y_remCorr.shape[0])[0:np.floor(Y_remCorr.shape[0]/float(2)).astype(int)]
        else:
            b = rng.permutation(Y_remCorr.shape[0])[0:np.ceil(Y_remCorr.shape[0]/float(2)).astype(int)]
        a_remCorr[b] = 1
 
        a_tot = np.zeros(permIxs_tot.shape[0])
        if rng.rand()>.5: 
            b = rng.permutation(permIxs_tot.shape[0])[0:np.floor(permIxs_tot.shape[0]/float(2)).astype(int)]
        else:
            b = rng.permutation(permIxs_tot.shape[0])[0:np.ceil(permIxs_tot.shape[0]/float(2)).astype(int)]
        a_tot[b] = 1
        
        for i in range(len(cvect)):
            
            if regType == 'l1':
                summary, shfl =  crossValidateModel(X, Y, linearSVM, kfold = kfold, l1 = cvect[i])
            elif regType == 'l2':
                summary, shfl =  crossValidateModel(X, Y, linearSVM, kfold = kfold, l2 = cvect[i])
                        
            wAllC[s,i,:,sc] = np.squeeze(summary.model.coef_); # weights of all neurons for each value of c and each shuffle
            bAllC[s,i,sc] = np.squeeze(summary.model.intercept_);
    
            # classification errors                    
            perClassErrorTrain[s, i,sc] = summary.perClassErrorTrain;
            perClassErrorTest[s, i,sc] = summary.perClassErrorTest;
                    
            # Testing correct shuffled data: 
            # same decoder trained on correct trials, make predictions on correct with shuffled labels.
            perClassErrorTest_shfl[s, i,sc] = perClassError(summary.YTest[permIxs], summary.model.predict(summary.XTest));
            perClassErrorTest_chance[s, i,sc] = perClassError(a_corr, summary.model.predict(summary.XTest));
            
            # also test the decoder on remaining correct trials
            if len(trsRemCorr)!=0:
                perClassErrorTestRemCorr[s, i,sc] = perClassError(Y_remCorr, summary.model.predict(X_remCorr))
                perClassErrorTestRemCorr_shfl[s, i,sc] = perClassError(Y_remCorr[permIxs_remCorr], summary.model.predict(X_remCorr))
                perClassErrorTestRemCorr_chance[s, i,sc] = perClassError(a_remCorr, summary.model.predict(X_remCorr))
    
            # Pool remainaing correct and cv...    
            x = np.concatenate((summary.XTest, X_remCorr), axis=0)
            y = np.concatenate((summary.YTest, Y_remCorr))
            perClassErrorTestBoth[s, i,sc] = perClassError(y, summary.model.predict(x))
            perClassErrorTestBoth_shfl[s, i,sc] = perClassError(y[permIxs_tot], summary.model.predict(x))
            perClassErrorTestBoth_chance[s, i,sc] = perClassError(a_tot, summary.model.predict(x))
            
    
            # keep index of trials that were used for training, cv, and remainedCorr testing.
            trsTrainedTestedInds[:,s,i,sc] = trsNowAll[shfl] # first 90% training, last 10% testing (indices are from outcomes array)
            trsRemCorrInds[:,s,i,sc] = trsRemCorr # (indices are from outcomes array)


#%% Compute average of class errors across numSamples and nRandCorrSel

meanPerClassErrorTrain = np.mean(perClassErrorTrain, axis = (0,2));
semPerClassErrorTrain = np.std(perClassErrorTrain, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);

meanPerClassErrorTest = np.mean(perClassErrorTest, axis = (0,2));
semPerClassErrorTest = np.std(perClassErrorTest, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);

meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl, axis = (0,2));
semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);

meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance, axis = (0,2));
semPerClassErrorTest_chance = np.std(perClassErrorTest_chance, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);

meanPerClassErrorTestRemCorr = np.mean(perClassErrorTestRemCorr, axis = (0,2));
semPerClassErrorTestRemCorr = np.std(perClassErrorTestRemCorr, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);

meanPerClassErrorTestRemCorr_shfl = np.mean(perClassErrorTestRemCorr_shfl, axis = (0,2));
semPerClassErrorTestRemCorr_shfl = np.std(perClassErrorTestRemCorr_shfl, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);

meanPerClassErrorTestRemCorr_chance = np.mean(perClassErrorTestRemCorr_chance, axis = (0,2));
semPerClassErrorTestRemCorr_chance = np.std(perClassErrorTestRemCorr_chance, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);

meanPerClassErrorTestBoth = np.mean(perClassErrorTestBoth, axis = (0,2));
semPerClassErrorTestBoth = np.std(perClassErrorTestBoth, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);

meanPerClassErrorTestBoth_shfl = np.mean(perClassErrorTestBoth_shfl, axis = (0,2));
semPerClassErrorTestBoth_shfl = np.std(perClassErrorTestBoth_shfl, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);

meanPerClassErrorTestBoth_chance = np.mean(perClassErrorTestBoth_chance, axis = (0,2));
semPerClassErrorTestBoth_chance = np.std(perClassErrorTestBoth_chance, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);


#%% Identify best c : 

smallestC = 0 # if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
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
print 'best c = ', cbestAll



# Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)
a = abs(wAllC)>eps # non-zero weights
b = np.mean(a, axis=(0,2,3)) # Fraction of non-zero weights (averaged across shuffles)
c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle
cvectnow = cvect[c1stnon0:]

meanPerClassErrorTestnow = np.mean(perClassErrorTest[:,c1stnon0:], axis = (0,2));
semPerClassErrorTestnow = np.std(perClassErrorTest[:,c1stnon0:], axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);
ix = np.argmin(meanPerClassErrorTestnow)
if smallestC==1:
    cbest = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
    cbest = cbest[0]; # best regularization term based on minError+SE criteria    
else:
    cbest = cvectnow[ix]

print 'best c (at least 1 non-0 weight) = ', cbest


# find bestc based on remCorr trial class error
meanPerClassErrorTestnow = np.mean(perClassErrorTestRemCorr[:,c1stnon0:], axis = (0,2));
semPerClassErrorTestnow = np.std(perClassErrorTestRemCorr[:,c1stnon0:], axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);
ix = np.argmin(meanPerClassErrorTestnow)
if smallestC==1:    
    cbest_remCorr = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
    cbest_remCorr = cbest_remCorr[0]; # best regularization term based on minError+SE criteria
else:   
    cbest_remCorr = cvectnow[ix]
    
print 'remCorr: best c (at least 1 non-0 weight) = ', cbest_remCorr


# find bestc based on both (pooled cv and remCorr) trial class error
meanPerClassErrorTestnow = np.mean(perClassErrorTestBoth[:,c1stnon0:], axis = (0,2));
semPerClassErrorTestnow = np.std(perClassErrorTestBoth[:,c1stnon0:], axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);
ix = np.argmin(meanPerClassErrorTestnow)
if smallestC==1:    
    cbest_both = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
    cbest_both = cbest_both[0]; # best regularization term based on minError+SE criteria
else:   
    cbest_both = cvectnow[ix]
    
print 'both: best c (at least 1 non-0 weight) = ', cbest_both



#%% Set the decoder and class errors at best c (for data)

# you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
# we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
indBestC = np.in1d(cvect, cbest)

w_bestc_data = wAllC[:,indBestC,:,:].squeeze() # numSamps x neurons
b_bestc_data = bAllC[:,indBestC,:]

classErr_bestC_train_data = perClassErrorTrain[:,indBestC,:].squeeze()

classErr_bestC_test_data = perClassErrorTest[:,indBestC,:].squeeze()
classErr_bestC_test_shfl = perClassErrorTest_shfl[:,indBestC,:].squeeze()
classErr_bestC_test_chance = perClassErrorTest_chance[:,indBestC,:].squeeze()

classErr_bestC_remCorr_data = perClassErrorTestRemCorr[:,indBestC,:].squeeze()
classErr_bestC_remCorr_shfl = perClassErrorTestRemCorr_shfl[:,indBestC,:].squeeze()
classErr_bestC_remCorr_chance = perClassErrorTestRemCorr_chance[:,indBestC,:].squeeze()



#%% plot C path
       
if doPlots:
    print 'Best c (inverse of regularization parameter) = %.2f' %cbest
    plt.figure('cross validation')
    plt.subplot(1,2,1)
    plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
    plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
    plt.fill_between(cvect, meanPerClassErrorTestRemCorr-semPerClassErrorTestRemCorr, meanPerClassErrorTestRemCorr+ semPerClassErrorTestRemCorr, alpha=0.5, edgecolor='g', facecolor='g')        
    plt.fill_between(cvect, meanPerClassErrorTestBoth-semPerClassErrorTestBoth, meanPerClassErrorTestBoth+ semPerClassErrorTestBoth, alpha=0.5, edgecolor='m', facecolor='m')        

#    plt.fill_between(cvect, meanPerClassErrorTestRemCorr_chance-semPerClassErrorTestRemCorr_chance, meanPerClassErrorTestRemCorr_chance+ semPerClassErrorTestRemCorr_chance, alpha=0.5, edgecolor='m', facecolor='m')        
#    plt.fill_between(cvect, meanPerClassErrorTestRemCorr_shfl-semPerClassErrorTestRemCorr_shfl, meanPerClassErrorTestRemCorr_shfl+ semPerClassErrorTestRemCorr_shfl, alpha=0.5, edgecolor='c', facecolor='c')        
#    plt.fill_between(cvect, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
#    plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
    
    plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
    plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation')
    plt.plot(cvect, meanPerClassErrorTestRemCorr, 'g', label = 'rem corr')        
    plt.plot(cvect, meanPerClassErrorTestBoth, 'm', label = 'cv+remCorr')        

#    plt.plot(cvect, meanPerClassErrorTestRemCorr_chance, 'm', label = 'remCorr-chance')            
#    plt.plot(cvect, meanPerClassErrorTestRemCorr_shfl, 'c', label = 'incorr-shfl')            
#    plt.plot(cvect, meanPerClassErrorTest_chance, 'b', label = 'cv-chance')       
#    plt.plot(cvect, meanPerClassErrorTest_shfl, 'y', label = 'corr-shfl')            
    plt.plot(cvect, meanPerClassErrorTestBoth_chance, 'b', label = 'cv+remCorr-chance')       

    plt.plot(cvect[cvect==cbest], meanPerClassErrorTest[cvect==cbest], 'bo')
    plt.plot(cvect[cvect==cbest_remCorr], meanPerClassErrorTestRemCorr[cvect==cbest_remCorr], 'bo')
    plt.plot(cvect[cvect==cbest_both], meanPerClassErrorTestBoth[cvect==cbest_both], 'bo')
 
    plt.xlim([cvect[1], cvect[-1]])
    plt.xscale('log')
    plt.xlabel('c (inverse of regularization parameter)')
    plt.ylabel('classification error (%)')
    plt.legend(loc='center left', bbox_to_anchor=(1, .7))
    plt.tight_layout()


   
#%% Plot class error for data and shuffled (for corr trained, corr cv, incorr cv)

"""
pvalueTest_incorr_corr = ttest2(classErr_bestC_incorr_data, classErr_bestC_test_shfl, tail = 'left');

#print 'Training error: Mean actual: %.2f%%, Mean shuffled: %.2f%%, p-value = %.2f' %(np.mean(perClassErrorTrain_data), np.mean(perClassErrorTrain_shfl), pvalueTrain)
print 'Testing error: Mean actual: %.2f%%, Mean shuffled: %.2f%%, p-value = %.2f' %(np.mean(classErr_bestC_test_data), np.mean(classErr_bestC_test_shfl), pvalueTest_corr)
print 'Incorr trials: error: Mean actual: %.2f%%, Mean chance: %.2f%%, p-value = %.2f' %(np.mean(classErr_bestC_incorr_data), np.mean(classErr_bestC_incorr_chance), pvalueTest_incorr)


# Plot the histograms
if doPlots:
    binEvery = 3; # bin width
    
    plt.figure()
#    plt.subplot(1,2,1)
#    plt.hist(perClassErrorTrain_data, np.arange(0,100,binEvery), color = 'k', label = 'data');
#    plt.hist(perClassErrorTrain_shfl, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
#    plt.xlabel('Training classification error (%)')
#    plt.ylabel('count')
#    plt.title('Mean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(perClassErrorTrain_data), np.mean(perClassErrorTrain_shfl), pvalueTrain), fontsize = 10)
#    plt.legend()

    plt.subplot(1,2,1)
    plt.hist(classErr_bestC_test_data, np.arange(0,100,binEvery), color = 'k', label = 'data');
    plt.hist(classErr_bestC_test_shfl, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
    plt.legend()
    plt.xlabel('Testing classification error (%)')
    plt.title('Testing correct\nMean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(classErr_bestC_test_data), np.mean(classErr_bestC_test_shfl), pvalueTest_corr), fontsize = 10)
    plt.ylabel('count')
    plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)

#    plt.figure()
    plt.subplot(1,2,2)
    plt.hist(classErr_bestC_incorr_data, np.arange(0,100,binEvery), color = 'k', label = 'data');
    plt.hist(classErr_bestC_incorr_chance, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
    plt.legend()
    plt.xlabel('Incorrect classification error (%)')
    plt.title('Incorrect\nMean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(classErr_bestC_incorr_data), np.mean(classErr_bestC_incorr_chance), pvalueTest_incorr), fontsize = 10)
    plt.ylabel('count')
    plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)


    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(classErr_bestC_test_data, np.arange(0,100,binEvery), color = 'k', label = 'testing corr');
    plt.hist(classErr_bestC_incorr_data, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'incorr');
    plt.legend()
    plt.xlabel('Classification error (%)')
    plt.title('Testing correct vs incorr\nMean corr: %.2f %%, Mean incorr: %.2f %%\n p-value = %.2f' %(np.mean(classErr_bestC_test_data), np.mean(classErr_bestC_incorr_data), pvalueTest_incorr_corr), fontsize = 10)
    plt.ylabel('count')
    plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
"""
#    pvalueTest_corr = ttest2(classErr_bestC_test_data, classErr_bestC_test_chance, tail = 'left');
#    pvalueTest_remCorr = ttest2(classErr_bestC_remCorr_data, classErr_bestC_remCorr_chance, tail = 'left');
#    print pvalueTest_corr, pvalueTest_remCorr




#%% Compare angle between decoders at different values of c
# just curious how decoders change during c path
"""
if doPlots:
    # Normalize wAllC
    nw = np.linalg.norm(wAllC, axis=2) # numSamps x len(cvect); 2-norm of weights 
    wAllC_n = np.transpose(np.transpose(wAllC,(2,0,1))/nw, (1,2,0)) # numSamps x len(cvect) x neurons
    
    # Take average decoder across shuffles and normalize it
    wAv = np.mean(wAllC_n,axis=0) # len(cvect) x neurons
    nw = np.linalg.norm(wAv, axis=1) # len(cvect); 2-norm of weights 
    wAv_n = np.transpose(np.transpose(wAv)/nw) # len(cvect) x neurons
    
    angc = np.full((len(cvect),len(cvect)), np.nan)
    angcAv = np.full((len(cvect),len(cvect)), np.nan)
    s = rng.permutation(numSamples)[0]
    for i in range(len(cvect)):
        for j in range(len(cvect)):
            angc[i,j] = np.arccos(abs(np.dot(wAllC_n[s,i,:], wAllC_n[s,j,:].transpose())))*180/np.pi
            angcAv[i,j] = np.arccos(abs(np.dot(wAv_n[i,:], wAv_n[j,:])))*180/np.pi
    
    # decoders at smaller values of c are all 0 weight, so their angles are nan
    '''
    # decoders of one of the shuffles
    plt.figure()
    #plt.subplot(121)
    plt.imshow(angc, cmap='jet_r')
    plt.colorbar()
    '''
    # average decoder across shuffles
    plt.figure()
    #plt.subplot(121)
    plt.imshow(angcAv, cmap='jet_r')
    plt.colorbar()
    
    plt.figure()
    #plt.subplot(122)
    plt.plot(meanPerClassErrorTrain, 'k', label = 'training')
    plt.plot(meanPerClassErrorTest, 'r', label = 'validation')
    plt.plot(meanPerClassErrorTest_shfl, 'y', label = 'corr-shfl')   
    plt.plot(meanPerClassErrorTestRemCorr, 'g', label = 'incorr')        
    plt.plot(meanPerClassErrorTestRemCorr_shfl, 'c', label = 'incorr-shfl')            
    
    #plt.axis('equal')
    #plt.gca().set_aspect('equal', adjustable='box')

"""

    
    
#%%
####################################################################################################################################
############################################ Save results as .mat files in a folder named svm ####################################################################################################################################
####################################################################################################################################

#%% Save SVM results

if trialHistAnalysis:
    svmn = 'svmPrevChoice_stimCateg_stAl_epAllStim_%s_' %(nowStr)
else:
    svmn = 'svmCurrChoice_stimCateg_stAl_epAllStim_%s_' %(nowStr)
print '\n', svmn[:-1]

if saveResults:
    print 'Saving .mat file'
    d = os.path.join(os.path.dirname(pnevFileName), 'svm')
    if not os.path.exists(d):
        print 'creating svm folder'
        os.makedirs(d)

    svmName = os.path.join(d, svmn+os.path.basename(pnevFileName))
    print(svmName)
    
    scio.savemat(svmName, {'thAct':thAct, 'thTrsWithSpike':thTrsWithSpike, 'minTrPerChoice': minTrPerChoice, 'earliestGoTone': earliestGoTone,
                           'th_stim_dur':th_stim_dur, 'numSamples':numSamples, 'nRandCorrSel':nRandCorrSel,
                           'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded,
                           'regType':regType, 'cvect':cvect,
                           'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,'cbest_remCorr':cbest_remCorr, 'cbest_both':cbest_both,
                           'wAllC':wAllC, 'bAllC':bAllC,
                           'perClassErrorTrain':perClassErrorTrain,
                           'perClassErrorTest':perClassErrorTest,
                           'perClassErrorTest_shfl':perClassErrorTest_shfl,
                           'perClassErrorTest_chance':perClassErrorTest_chance,
                           'perClassErrorTestRemCorr':perClassErrorTestRemCorr,
                           'perClassErrorTestRemCorr_shfl':perClassErrorTestRemCorr_shfl,
                           'perClassErrorTestRemCorr_chance':perClassErrorTestRemCorr_chance,
                           'trsTrainedTestedInds':trsTrainedTestedInds, 'trsRemCorrInds':trsRemCorrInds,
                           'perClassErrorTestBoth':perClassErrorTestBoth,
                           'perClassErrorTestBoth_shfl':perClassErrorTestBoth_shfl,
                           'perClassErrorTestBoth_chance':perClassErrorTestBoth_chance}) 
    
else:
    print 'Not saving .mat file'


