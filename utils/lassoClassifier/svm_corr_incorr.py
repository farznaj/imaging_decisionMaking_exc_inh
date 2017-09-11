# -*- coding: utf-8 -*-
# NOTE: add for saving meanX, stdX and _chAl mean and std.

"""
Train SVM and SVR on correct trials and test its classification accuracy on incorrect trials.
[Also as usual we test it on correct trials (both training and testing datasets) 
and we do trial-label shuffles to set null distributions for the classification 
accuracy of the classifiers.]


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
    mousename = 'fni19' #'fni16'
    imagingFolder = '150901' #'151001'
    mdfFileNumber = [1] #[1] 

    choiceAligned = 1 # use choice-aligned traces for analyses?
    stimAligned = 0 # use stimulus-aligned traces for analyses?
    doSVM = 1 # decode choice?
    doSVR = 0 # decode stimulus?
    epAllStim = 1 # when doing svr on stim-aligned traces, if 1, svr will computed on X_svr (ave pop activity during the entire stim presentation); Otherwise it will be computed on X0 (ave pop activity during ep)
    doBagging = 0 # BAGGING will be performed (in addition to l2) for svm/svr

    trialHistAnalysis = 0;    
#    roundi = 1; # For the same dataset we run the code multiple times, each time we select a random subset of neurons (of size n, n=.95*numTrials)

    iTiFlg = 1; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.
    setNsExcluded = 1; # if 1, NsExcluded will be set even if it is already saved.
    numSamples = 10 #100; # number of iterations for finding the best c (inverse of regularization parameter)
    neuronType = 2; # 0: excitatory, 1: inhibitory, 2: all types.    
    saveResults = 0; # save results in mat file.

    # for BAGGING:
#    nRandTrSamp = 10 #nRandTrSamp = 10 #1000 # number of times to subselect trials (for the BAGGING method)
    doData = 1
    
    doNsRand = 0; # if 1, a random set of neurons will be selected to make sure we have fewer neurons than trials. 
    regType = 'l2' # 'l2' : regularization type
    kfold = 10;
    compExcInh = 0 # if 1, analyses will be run to compare exc inh neurons.
    
    # The following vars don't usually need to be changed    
    doPlots = 1; # Whether to make plots or not.
    saveHTML = 0; # whether to save the html file of notebook with all figures or not.

    if trialHistAnalysis==1: # more parameters are specified in popClassifier_trialHistory.m
    #        iTiFlg = 1; # 0: short ITI, 1: long ITI, 2: all ITIs.
        epEnd_rel2stimon_fr = 0 # 3; # -2 # epEnd = eventI + epEnd_rel2stimon_fr
    else:
        # not needed to set ep_ms here, later you define it as [choiceTime-300 choiceTime]ms # we also go 30ms back to make sure we are not right on the choice time!
        ep_ms = [809, 1109] #[425, 725] # optional, it will be set according to min choice time if not provided.# training epoch relative to stimOnset % we want to decode animal's upcoming choice by traninig SVM for neural average responses during ep ms after stimulus onset. [1000, 1300]; #[700, 900]; # [500, 700]; 
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



eps = sys.float_info.epsilon #2.2204e-16
nRandTrSamp = numSamples #10 #1000 # number of times to subselect trials (for the BAGGING method)


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
elif 'ep_ms' in locals():
    print 'training window: [%d %d] ms' %(ep_ms[0], ep_ms[1])
print 'windowAvgFlg = %i' %(windowAvgFlg)
print 'numSamples = %i' %(numSamples)


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
    return imfilename, pnevFileName
    
    
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

imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load, postNProvided=postNProvided)

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


if trialHistAnalysis==1:    
    # either of the two below (stimulus-aligned and initTone-aligned) would be fine
    # eventI = DataI['initToneAl'].eventI - 1
    eventI = DataS['stimAl_allTrs'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
    epEnd = eventI + epEnd_rel2stimon_fr #- 2 # to be safe for decoder training for trial-history analysis we go upto the frame before the stim onset
    # epEnd = DataI['initToneAl'].eventI - 2 # to be safe for decoder training for trial-history analysis we go upto the frame before the initTone onset
    ep = np.arange(epEnd+1)
    print 'training epoch is {} ms'.format(np.round((ep-eventI)*frameLength))
    
    ep_ms = list(np.round((ep[[0,-1]]-eventI)*frameLength).astype(int)) # so it is the same format as ep_ms when trialHistAnalysis is 0
    
else:
    # Set ep_ms if it is not provided: [choiceTime-300 choiceTime]ms # we also go 30ms back to make sure we are not right on the choice time!
    # by doing this you wont need to set ii below.
    
    '''
    # We first set to nan timeStimOnset of trials that anyway wont matter bc their outcome is  not of interest. we do this to make sure these trials dont affect our estimate of ep_ms
    if outcome2ana == 'corr':
        timeStimOnset[outcomes!=1] = np.nan; # analyze only correct trials.
    elif outcome2ana == 'incorr':
        timeStimOnset[outcomes!=0] = np.nan; # analyze only incorrect trials.   
    '''    
    if not 'ep_ms' in locals(): 
        ep_ms = [np.floor(np.nanmin(time1stSideTry-timeStimOnset))-30-300, np.floor(np.nanmin(time1stSideTry-timeStimOnset))-30]
        print 'Training window: [%d %d] ms' %(ep_ms[0], ep_ms[1])
    
    epStartRel2Event = np.ceil(ep_ms[0]/frameLength); # the start point of the epoch relative to alignedEvent for training SVM. (500ms)
    epEndRel2Event = np.ceil(ep_ms[1]/frameLength); # the end point of the epoch relative to alignedEvent for training SVM. (700ms)
    ep = np.arange(eventI+epStartRel2Event, eventI+epEndRel2Event+1).astype(int); # frames on stimAl.traces that will be used for trainning SVM.   
    print 'Training epoch relative to stimOnset is {} ms'.format(np.round((ep-eventI)*frameLength - frameLength/2)) # print center of frames in ms
            


########################%% Exclude some trials from traces_al_stim ########################
if trialHistAnalysis==0:            
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


    # Make sure trials that you use for SVM (decoding upcoming choice from
    # neural responses during stimulus) have a certain stimulus duration. Of
    # course stimulus at least needs to continue until the end of ep. 
    # go with either 900 or 800ms. Since the preference is to have at least
    # ~100ms after ep which contains stimulus and without any go tones, go with 800ms
    # bc in many sessions go tone happened early... so you will loose lots of
    # trials if you go with 900ms.
    # th_stim_dur = 800; # min stim duration to include a trial in timeStimOnset

    if doPlots:
        plt.figure
        plt.subplot(1,2,1)
        plt.plot(timeCommitCL_CR_Gotone - timeStimOnset, label = 'goTone')
        plt.plot(timeStimOffset - timeStimOnset, 'r', label = 'stimOffset')
        plt.plot(time1stSideTry - timeStimOnset, 'm', label = '1stSideTry')
        plt.plot([1, np.shape(timeCommitCL_CR_Gotone)[0]],[th_stim_dur, th_stim_dur], 'g:', label = 'th_stim_dur')
        plt.plot([1, np.shape(timeCommitCL_CR_Gotone)[0]],[ep_ms[-1], ep_ms[-1]], 'k:', label = 'epoch end')
        plt.xlabel('Trial')
        plt.ylabel('Time relative to stim onset (ms)')
        plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 
        # minStimDurNoGoTone = np.nanmin(timeCommitCL_CR_Gotone - timeStimOnset); # this is the duration after stim onset during which no go tone occurred for any of the trials.
        # print 'minStimDurNoGoTone = %.2f ms' %minStimDurNoGoTone


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



# Set spikeAveEp0  (X: the predictor matrix (trials x neurons) that shows average of spikes for a particular epoch for each trial and neuron.)
if trialHistAnalysis:
    # either of the two cases below should be fine (init-aligned traces or stim-aligned traces.)
    spikeAveEp0 = np.transpose(np.nanmean(traces_al_stimAll[ep,:,:], axis=0)) # trials x neurons    
#    spikeAveEp0_svr = np.transpose(np.nanmean(traces_al_stimAll[ep,:,:], axis=0)) # trials x neurons    
else:    
    spikeAveEp0 = np.transpose(np.nanmean(traces_al_stim[ep,:,:], axis=0)) # trials x neurons    

# X = spikeAveEp0;
print 'Size of spikeAveEp0 (trs x neurons): ', spikeAveEp0.shape


#%% Set stimulus-aligned traces (X, Y and Ysr)

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
spikeAveEp0_svr = np.full((traces_al_stim.shape[2], traces_al_stim.shape[1]), np.nan)
for tr in range(traces_al_stim.shape[2]):
#    stimDurNow = np.floor(timeStimOffset[tr]-timeStimOnset[tr])
#    stimDurNow = np.floor(time1stSideTry[tr]-timeStimOnset[tr])
    stimDurNow = np.floor(timeCommitCL_CR_Gotone[tr]-timeStimOnset[tr])
    epnow = np.arange(eventI+1, eventI + np.round(stimDurNow/frameLength) + 1).astype(int)
    spikeAveEp0_svr[tr,:] = np.mean(traces_al_stim[epnow,:,tr], axis=0) # neurons # average of frames during epnow for each neuron in trial tr


# Exclude nan trials
X_svr = spikeAveEp0_svr[~trsExcluded,:]; # trials x neurons



#%% Set choice-aligned traces (X and Y)

# Load 1stSideTry-aligned traces, frames, frame of event of interest
# use firstSideTryAl_COM to look at changes-of-mind (mouse made a side lick without committing it)
Data = scio.loadmat(postName, variable_names=['firstSideTryAl'],squeeze_me=True,struct_as_record=False)
traces_al_1stSide = Data['firstSideTryAl'].traces.astype('float')
time_aligned_1stSide = Data['firstSideTryAl'].time.astype('float')
eventI_ch = Data['firstSideTryAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
# print(np.shape(traces_al_1stSide))

# training epoch: 300ms before the choice is made.
epSt = eventI_ch - np.round(300/frameLength) # the start point of the epoch relative to alignedEvent for training SVM. (500ms)
epEn = eventI_ch-1 # the end point of the epoch relative to alignedEvent for training SVM. (700ms)
ep_ch = np.arange(epSt, epEn+1).astype(int) # frames on stimAl.traces that will be used for trainning SVM.  

# average frames during ep_ch (ie 300ms before choice onset)
X_choice0 = np.transpose(np.nanmean(traces_al_1stSide[ep_ch,:,:], axis=0)) # trials x neurons 


# determine trsExcluded for traces_al_1stSide
# remember traces_al_1stSide can have some nan trials that are change-of-mind trials... and they wont be nan in choiceVec0
trsExcluded_chAl = (np.sum(np.isnan(X_choice0), axis = 1) + np.isnan(choiceVec0)) != 0 # NaN trials # trsExcluded


# exclude trsExcluded_chAl
X_chAl = X_choice0[~trsExcluded_chAl,:]; # trials x neurons
Y_chAl = choiceVec0[~trsExcluded_chAl];

Ysr_chAl = stimrate[~trsExcluded_chAl] # we dont need to exclude trials that animal did not make a choice for the stimulus decoder analysis... unless you want to use the same set of trials for the choice and stimulus decoder...


#%% Set X and Y for incorrect trials (both stimulus-aligned and choice-aligned)

# set Y_incorr: vector of choices for incorrect trials
Y_incorr0 = choiceVecAll+0
Y_incorr0[outcomes!=0] = np.nan; # analyze only incorrect trials.


######## Stim aligned
# Exclude nan trials
trsExcluded_incorr = (np.sum(np.isnan(spikeAveEp0), axis = 1) + np.isnan(Y_incorr0)) != 0 # NaN trials # trsExcluded

X_incorr = spikeAveEp0[~trsExcluded_incorr,:]
Y_incorr = Y_incorr0[~trsExcluded_incorr]
#X_incorr.shape
#Y_incorr

Ysr_incorr = stimrate[~trsExcluded_incorr] #exclude incorr trials, etc # we dont need to exclude trials that animal did not make a choice for the stimulus decoder analysis... unless you want to use the same set of trials for the choice and stimulus decoder...

# X_svr for incorr trials (average pop activity during stim for incorr trs)
X_svr_incorr = spikeAveEp0_svr[~trsExcluded_incorr,:]


######### Choice aligned
# Exclude nan trials
trsExcluded_incorr_chAl = (np.sum(np.isnan(X_choice0), axis = 1) + np.isnan(Y_incorr0)) != 0 # NaN trials # trsExcluded

X_chAl_incorr = X_choice0[~trsExcluded_incorr_chAl,:]
Y_chAl_incorr = Y_incorr0[~trsExcluded_incorr_chAl]
#X_incorr.shape
#Y_incorr

Ysr_chAl_incorr = stimrate[~trsExcluded_incorr_chAl] #exclude incorr trials, etc # we dont need to exclude trials that animal did not make a choice for the stimulus decoder analysis... unless you want to use the same set of trials for the choice and stimulus decoder...


#%% Normalize Y (subtract midrange and divide by range)

yci = np.concatenate((Ysr, Ysr_incorr))
Ysr = (Ysr - (yci.min() + np.ptp(yci)/2)) / np.ptp(yci) # when you did not do this, angle_stim_sh (ie stimulus shuffled) gave you weird results... during baseline (~15 frames at the beginning and almost ~15 frames at the end) you got alignment between population responses in the shuffled data!
#Ysr = (Ysr - Ysr.mean()) / Ysr.std()
#Ysr = (Ysr-cb) / np.ptp(stimrate) # not sure if we need this, also stimrate in some days did not span the entire range (4-28hz)

Ysr_incorr = (Ysr_incorr - (yci.min() + np.ptp(yci)/2)) / np.ptp(yci) # when you did not do this, angle_stim_sh (ie stimulus shuffled) gave you weird results... during baseline (~15 frames at the beginning and almost ~15 frames at the end) you got alignment between population responses in the shuffled data!


# chAl
yci = np.concatenate((Ysr_chAl, Ysr_chAl_incorr))
Ysr_chAl = (Ysr_chAl - (yci.min() + np.ptp(yci)/2)) / np.ptp(yci) # when you did not do this, angle_stim_sh (ie stimulus shuffled) gave you weird results... during baseline (~15 frames at the beginning and almost ~15 frames at the end) you got alignment between population responses in the shuffled data!

Ysr_chAl_incorr = (Ysr_chAl_incorr - (yci.min() + np.ptp(yci)/2)) / np.ptp(yci) # when you did not do this, angle_stim_sh (ie stimulus shuffled) gave you weird results... during baseline (~15 frames at the beginning and almost ~15 frames at the end) you got alignment between population responses in the shuffled data!


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
#        stdX = np.std(np.concatenate((X,X_incorr)), axis = 0);
        stdX = np.std(X, axis = 0);
        NsExcluded = stdX < thAct
#        np.argwhere(NsExcluded)
        stdX = np.std(X_incorr, axis = 0);
        NsExcluded_incorr = stdX < thAct        
#        np.argwhere(NsExcluded_incorr)        
        NsExcluded = np.logical_or(NsExcluded, NsExcluded_incorr)

        # X_chAl
#        stdX_chAl = np.std(np.concatenate((X_chAl,X_chAl_incorr)), axis = 0);
        stdX = np.std(X_chAl, axis = 0);
        NsExcluded_chAl = stdX < thAct        
#        np.argwhere(NsExcluded_chAl)
        stdX = np.std(X_chAl_incorr, axis = 0);
        NsExcluded_chAl_incorr = stdX < thAct       
#        np.argwhere(NsExcluded_chAl_incorr)
        NsExcluded_chAl = np.logical_or(NsExcluded_chAl, NsExcluded_chAl_incorr)        
        
        
        # X_svr
        stdX = np.std(X_svr, axis = 0);
        NsExcluded_svr = stdX < thAct
#        np.argwhere(NsExcluded)
        stdX = np.std(X_svr_incorr, axis = 0);
        NsExcluded_svr_incorr = stdX < thAct        
#        np.argwhere(NsExcluded_incorr)        
        NsExcluded_svr = np.logical_or(NsExcluded_svr, NsExcluded_svr_incorr)

        
        # you are not training on incorrect trials ... only testing on incorrect trials... so we dont need the following
        '''        
        # X_incorr
        stdX_incorr = np.std(X_incorr, axis = 0);
        NsExcluded_incorr = stdX_incorr < thAct

        # X_chAl_incorr
        stdX_choice_incorr = np.std(X_chAl_incorr, axis = 0);
        NsExcluded_choice_incorr = stdX_choice_incorr < thAct        
        '''
        
        
        '''
        # Set nonActiveNs, ie neurons whose average activity during ep is less than thAct.
    #     spikeAveEpAveTrs = np.nanmean(spikeAveEp0, axis=0); # 1 x units % response of each neuron averaged across epoch ep and trials.
        spikeAveEpAveTrs = np.nanmean(X, axis=0); # 1 x units % response of each neuron averaged across epoch ep and trials.
        # thAct = 5e-4; # 1e-5 #quantile(spikeAveEpAveTrs, .1);
        nonActiveNs = spikeAveEpAveTrs < thAct;
        print '\t%d neurons with ave activity in ep < %.5f' %(np.sum(nonActiveNs), thAct)
        np.sum(nonActiveNs)

        # Set NsFewTrActiv, ie neurons that are active in very few trials (by active I mean average activity during epoch ep)
        # thTrsWithSpike = 1; # 3; # ceil(thMinFractTrs * size(spikeAveEp0,1)); % 30  % remove neurons with activity in <thSpTr trials.
        nTrsWithSpike = np.sum(X > thAct, axis=0) # 0 # shows for each neuron, in how many trials the activity was above 0.
        NsFewTrActiv = (nTrsWithSpike < thTrsWithSpike) # identify neurons that were active fewer than thTrsWithSpike.
        print '\t%d neurons are active in < %i trials' %(np.sum(NsFewTrActiv), thTrsWithSpike)

        # Now set the final NxExcluded: (neurons to exclude)
        NsExcluded = (NsFewTrActiv + nonActiveNs)!=0
        '''

print '%d = Final # non-active neurons' %(sum(NsExcluded))
# a = size(spikeAveEp0,2) - sum(NsExcluded);
print 'Using %d out of %d neurons; Fraction excluded = %.2f\n' %(np.shape(spikeAveEp0)[1]-sum(NsExcluded), np.shape(spikeAveEp0)[1], sum(NsExcluded)/float(np.shape(spikeAveEp0)[1]))


print '%i, %i, %i: #original inh, excit, unsure' %(np.sum(inhibitRois==1), np.sum(inhibitRois==0), np.sum(np.isnan(inhibitRois)))
# Check what fraction of inhibitRois are excluded, compare with excitatory neurons.
if neuronType==2:    
    print '%i, %i, %i: #excluded inh, excit, unsure' %(np.sum(inhibitRois[NsExcluded]==1), np.sum(inhibitRois[NsExcluded]==0), np.sum(np.isnan(inhibitRois[NsExcluded])))
    print '%.2f, %.2f, %.2f: fraction excluded inh, excit, unsure\n' %(np.sum(inhibitRois[NsExcluded]==1)/float(np.sum(inhibitRois==1)), np.sum(inhibitRois[NsExcluded]==0)/float(np.sum(inhibitRois==0)), np.sum(np.isnan(inhibitRois[NsExcluded]))/float(np.sum(np.isnan(inhibitRois))))


print 'choice-al: %d = Final # non-active neurons' %(sum(NsExcluded_chAl))


#%% Exclude non-active neurons from X and set inhRois (ie neurons that don't fire in any of the trials during ep)

##### stimulus-aligned
X = X[:,~NsExcluded]
print 'stim-aligned traces (trs x units): ', np.shape(X)
    
# Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)
if neuronType==2:
    inhRois = inhibitRois[~NsExcluded]
    # print 'Number: inhibit = %d, excit = %d, unsure = %d' %(np.sum(inhRois==1), np.sum(inhRois==0), np.sum(np.isnan(inhRois)))
    # print 'Fraction: inhibit = %.2f, excit = %.2f, unsure = %.2f' %(fractInh, fractExc, fractUn)

# X_incorr
X_incorr = X_incorr[:,~NsExcluded]
print 'stim-aligned traces, incorrect trials (trs x units): ', np.shape(X_incorr)


## X_svr
X_svr = X_svr[:,~NsExcluded_svr]
print 'stim-aligned traces averaged during entire stim (trs x units): ', np.shape(X_svr)

# X_svr_incorr
X_svr_incorr = X_svr_incorr[:,~NsExcluded_svr]
print 'stim-aligned traces, incorrect trials, ave entire stim (trs x units): ', np.shape(X_svr_incorr)



##### choice-aligned    
# X_chAl    
X_chAl = X_chAl[:,~NsExcluded_chAl]
print 'choice-aligned traces (trs x units): ', np.shape(X_chAl)

# X_chAl_incorr
X_chAl_incorr = X_chAl_incorr[:,~NsExcluded_chAl]
print 'choice-aligned traces, incorrect trials (trs x units): ', np.shape(X_chAl_incorr)


#%% Not doing this anymore:
##  If number of neurons is more than 95% of trial numbers, identify n random neurons, where n= 0.95 * number of trials. This is to make sure we have more observations (trials) than features (neurons)

nTrs = np.shape(X)[0]
nNeuronsOrig = np.shape(X)[1]
nNeuronsNow = np.int(np.floor(nTrs * .95))

if doNsRand==0:
    print 'Chose not to do random selection of neurons' 
    NsRand = np.ones(np.shape(X)[1]).astype('bool')

elif nNeuronsNow < nNeuronsOrig:
    if neuronType==2:
        fractInh = np.sum(inhRois==1) / float(nNeuronsOrig)
        fractExc = np.sum(inhRois==0) / float(nNeuronsOrig)
        fractUn = np.sum(np.isnan(inhRois)) / float(nNeuronsOrig)
        print 'Number: inhibit = %d, excit = %d, unsure = %d' %(np.sum(inhRois==1), np.sum(inhRois==0), np.sum(np.isnan(inhRois)))
        print 'Fraction: inhibit = %.2f, excit = %.2f, unsure = %.2f' %(fractInh, fractExc, fractUn)
    elif neuronType==0: # exc
        fractInh = 0;
        fractExc = 1;
        fractUn = 0;
    elif neuronType==1: # inh
        fractInh = 1;
        fractExc = 0;
        fractUn = 0;    

    # Define how many neurons you need to pick from each pool of inh, exc, unsure.
    nInh = int(np.ceil(fractInh*nNeuronsNow))
    nExc = int(np.ceil(fractExc*nNeuronsNow))
    nUn = nNeuronsNow - (nInh + nExc) # fractUn*nNeuronsNow

    print '\nThere are', nTrs, 'trials; So selecting', nNeuronsNow, 'neurons out of', nNeuronsOrig
    print '%i, %i, %i: number of selected inh, excit, unsure' %(nInh, nExc, nUn)

    # Select nInh random indeces out of the inhibibory pool
    inhI = np.argwhere(inhRois==1)
    inhNow = rng.permutation(inhI)[0:nInh].flatten() # random indeces

    # Select nExc random indeces out of the excitatory pool
    excI = np.argwhere(inhRois==0)
    excNow = rng.permutation(excI)[0:nExc].flatten()

    # Select nUn random indeces out of the unsure pool
    unI = np.argwhere(np.isnan(inhRois))
    unNow = rng.permutation(unI)[0:nUn].flatten()

    # Put all the 3 groups together 
    neuronsNow = np.sort(np.concatenate([inhNow,excNow,unNow]), axis=None)
    np.shape(neuronsNow)
    # neuronsNow
    # np.max(neuronsNow)

    # Define a logical array with 1s for randomly selected neurons (length = number of neurons in X (after excluding NsExcluded))
    NsRand = np.arange(np.shape(X)[1])
    NsRand = np.in1d(NsRand, neuronsNow)
    # np.shape(NsRand)
    # NsRand    
elif nNeuronsNow >= nNeuronsOrig: # if number of neurons is already <= .95*numTrials, include all neurons.
    print 'Not doing random selection of neurons (nNeurons=%d already fewer than .95*nTrials=%d)' %(np.shape(X)[1], nTrs)
    NsRand = np.ones(np.shape(X)[1]).astype('bool')
    


#%% Set X and inhRois only for the randomly selected set of neurons

X = X[:,NsRand]
if neuronType==2:
    inhRois = inhRois[NsRand]

if windowAvgFlg==0:
    a = np.transpose(traces_al_stim[ep,:,:][:,~NsExcluded,:][:,:,~trsExcluded], (0,2,1))  # ep_frames x trials x units
    a = a[:,:,neuronsNow]
    X = np.reshape(a, (ep.shape[0]*(~trsExcluded).sum(), (~NsExcluded).sum())) # (ep_frames x trials) x units

    Y = np.tile(np.reshape(choiceVec0[~trsExcluded], (1,-1)), (ep.shape[0], 1)).flatten()    

    
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

print 'choice-aligned traces: %d trials; %d neurons' %(X_chAl.shape[0], X_chAl.shape[1])


#%% Center and normalize X: feature normalization and scaling: to remove effects related to scaling and bias of each neuron, we need to zscore data (i.e., make data mean 0 and variance 1 for each neuron) 

##### stimulus-aligned
# use corr
#meanX = np.mean(X, axis = 0);
#stdX = np.std(X, axis = 0);
# use both corr and incorr
xtot = np.concatenate((X,X_incorr),axis=0)
meanX = np.mean(xtot, axis = 0);
stdX = np.std(xtot, axis = 0);

# normalize X
X = (X-meanX)/stdX;
# X_incorr
X_incorr = (X_incorr - meanX) / stdX
'''
meanX_incorr = np.mean(X_incorr, axis = 0);
stdX_incorr = np.std(X_incorr, axis = 0);
# normalize X
X_incorr = (X_incorr - meanX_incorr) / stdX_incorr;
'''

# X_svr
# use both corr and incorr
xtot = np.concatenate((X_svr,X_svr_incorr),axis=0)
meanX_svr = np.mean(xtot, axis = 0);
stdX_svr = np.std(xtot, axis = 0);

# normalize X
X_svr = (X_svr-meanX_svr)/stdX_svr
# X_incorr
X_svr_incorr = (X_svr_incorr - meanX_svr) / stdX_svr




##### choice-aligned
# use corr
#meanX_chAl = np.mean(X_chAl, axis = 0);
#stdX_chAl = np.std(X_chAl, axis = 0);
# use both corr and incorr
xtot = np.concatenate((X_chAl,X_chAl_incorr),axis=0)
meanX_chAl = np.mean(xtot, axis = 0);
stdX_chAl = np.std(xtot, axis = 0);

# normalize X
X_chAl = (X_chAl - meanX_chAl) / stdX_chAl;
# X_chAl_incorr  
X_chAl_incorr = (X_chAl_incorr - meanX_chAl) / stdX_chAl;
'''
meanX_choice_incorr = np.mean(X_chAl_incorr, axis = 0);
stdX_choice_incorr = np.std(X_chAl_incorr, axis = 0);
# normalize X
X_chAl_incorr = (X_chAl_incorr - meanX_choice_incorr) / stdX_choice_incorr;
'''


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

    # choice-al
    plt.figure
    plt.subplot(2,2,1)
    plt.plot(meanX_chAl)
    plt.ylabel('meanX \n(ep-ave FR, mean of trials)')
    plt.title('min = %.6f' %(np.min(meanX_chAl)))

    plt.subplot(2,2,3)
    plt.plot(stdX_chAl)
    plt.ylabel('stdX \n(ep-ave FR, std of trials)')
    plt.xlabel('neurons')
    plt.title('min = %.6f' %(np.min(stdX_chAl)))
    plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.subplots_adjust(hspace=.5)


    # stim-al, ave stim presentaiton
    plt.figure
    plt.subplot(2,2,1)
    plt.plot(meanX_svr)
    plt.ylabel('meanX \n(ep-ave FR, mean of trials)')
    plt.title('min = %.6f' %(np.min(meanX_svr)))

    plt.subplot(2,2,3)
    plt.plot(stdX_svr)
    plt.ylabel('stdX \n(ep-ave FR, std of trials)')
    plt.xlabel('neurons')
    plt.title('min = %.6f' %(np.min(stdX_svr)))

    plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.subplots_adjust(hspace=.5)
    
    
    # incorrect trials
    '''
    # stim-al
    plt.figure
    plt.subplot(2,2,1)
    plt.plot(meanX_incorr)
    plt.ylabel('meanX \n(ep-ave FR, mean of trials)')
    plt.title('min = %.6f' %(np.min(meanX_incorr)))

    plt.subplot(2,2,3)
    plt.plot(stdX_incorr)
    plt.ylabel('stdX \n(ep-ave FR, std of trials)')
    plt.xlabel('neurons')
    plt.title('min = %.6f' %(np.min(stdX_incorr)))

    plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.subplots_adjust(hspace=.5)

    # choice-al
    plt.figure
    plt.subplot(2,2,1)
    plt.plot(meanX_choice_incorr)
    plt.ylabel('meanX \n(ep-ave FR, mean of trials)')
    plt.title('min = %.6f' %(np.min(meanX_choice_incorr)))

    plt.subplot(2,2,3)
    plt.plot(stdX_choice_incorr)
    plt.ylabel('stdX \n(ep-ave FR, std of trials)')
    plt.xlabel('neurons')
    plt.title('min = %.6f' %(np.min(stdX_choice_incorr)))

    plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.subplots_adjust(hspace=.5)
    '''



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
    
    
    #################### incorrect trials
    # Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
    hr_trs = (Y_incorr==1)
    lr_trs = (Y_incorr==0)
    
    # Set traces
    Xt = traces_al_stim[:, :, ~trsExcluded_incorr];
    # Exclude non-active neurons (ie neurons that don't fire in any of the trials during ep)
    Xt = Xt[:,~NsExcluded,:] # NsExcluded defined based on corr trials... so some of these neurons may not be active for the average of incorr trials.
    ## Feature normalization and scaling    
    # normalize stim-aligned traces
    T, N, C = Xt.shape
    Xt_N = np.reshape(Xt.transpose(0 ,2 ,1), (T*C, N), order = 'F')
    Xt_N = (Xt_N-meanX)/stdX # again we are normalizing based on meanX and stdX found on correct trials.
    Xt = np.reshape(Xt_N, (T, C, N), order = 'F').transpose(0 ,2 ,1)


    # Plot
    plt.subplot(1,2,2) # I think you should use Xtsa here to make it compatible with the plot above.
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
    plt.title('Incorrect trials')
    
    
    
    ######################################## Plot choice-aligned averages after centering and normalization
    numTrials_chAl = X_chAl.shape[0]
    
    plt.figure()
    #################### correct trials
    # Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
    hr_trs = (Y_chAl==1)
    lr_trs = (Y_chAl==0)


    # Set traces
    Xt = traces_al_1stSide[:, :, ~trsExcluded_chAl];
    # Exclude non-active neurons (ie neurons that don't fire in any of the trials during ep)
    Xt = Xt[:,~NsExcluded_chAl,:]
    ## Feature normalization and scaling    
    # normalize stim-aligned traces
    T, N, C = Xt.shape
    Xt_N = np.reshape(Xt.transpose(0 ,2 ,1), (T*C, N), order = 'F')
    Xt_N = (Xt_N-meanX_chAl)/stdX_chAl
    Xt = np.reshape(Xt_N, (T, C, N), order = 'F').transpose(0 ,2 ,1)


    # Plot
    plt.subplot(1,2,1) # I think you should use Xtsa here to make it compatible with the plot above.
    a1 = np.nanmean(Xt[:, :, hr_trs],  axis=1) # frames x trials (average across neurons)
    tr1 = np.nanmean(a1,  axis = 1)
    tr1_se = np.nanstd(a1,  axis = 1) / np.sqrt(numTrials_chAl);
    a0 = np.nanmean(Xt[:, :, lr_trs],  axis=1) # frames x trials (average across neurons)
    tr0 = np.nanmean(a0,  axis = 1)
    tr0_se = np.nanstd(a0,  axis = 1) / np.sqrt(numTrials_chAl);    
    plt.fill_between(time_aligned_1stSide, tr1-tr1_se, tr1+tr1_se, alpha=0.5, edgecolor='b', facecolor='b')
    plt.fill_between(time_aligned_1stSide, tr0-tr0_se, tr0+tr0_se, alpha=0.5, edgecolor='r', facecolor='r')
    plt.plot(time_aligned_1stSide, tr1, 'b', label = 'high rate')
    plt.plot(time_aligned_1stSide, tr0, 'r', label = 'low rate')
    # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, lr_trs],  axis = (1, 2)), 'r', label = 'high rate')
    # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, hr_trs],  axis = (1, 2)), 'b', label = 'low rate')
    plt.xlabel('time aligned to choice onset (ms)')
    plt.title('Correct trials')
    
    
    #################### incorrect trials
    # Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
    hr_trs = (Y_chAl_incorr==1)
    lr_trs = (Y_chAl_incorr==0)
    
    # Set traces
    Xt = traces_al_1stSide[:, :, ~trsExcluded_incorr_chAl];
    # Exclude non-active neurons (ie neurons that don't fire in any of the trials during ep)
    Xt = Xt[:,~NsExcluded_chAl,:] # NsExcluded defined based on corr trials... so some of these neurons may not be active for the average of incorr trials.
    ## Feature normalization and scaling    
    # normalize stim-aligned traces
    T, N, C = Xt.shape
    Xt_N = np.reshape(Xt.transpose(0 ,2 ,1), (T*C, N), order = 'F')
    Xt_N = (Xt_N-meanX_chAl)/stdX_chAl # again we are normalizing based on meanX and stdX found on correct trials.
    Xt = np.reshape(Xt_N, (T, C, N), order = 'F').transpose(0 ,2 ,1)


    # Plot
    plt.subplot(1,2,2) # I think you should use Xtsa here to make it compatible with the plot above.
    a1 = np.nanmean(Xt[:, :, hr_trs],  axis=1) # frames x trials (average across neurons)
    tr1 = np.nanmean(a1,  axis = 1)
    tr1_se = np.nanstd(a1,  axis = 1) / np.sqrt(numTrials_chAl);
    a0 = np.nanmean(Xt[:, :, lr_trs],  axis=1) # frames x trials (average across neurons)
    tr0 = np.nanmean(a0,  axis = 1)
    tr0_se = np.nanstd(a0,  axis = 1) / np.sqrt(numTrials_chAl);
    plt.fill_between(time_aligned_1stSide, tr1-tr1_se, tr1+tr1_se, alpha=0.5, edgecolor='b', facecolor='b')
    plt.fill_between(time_aligned_1stSide, tr0-tr0_se, tr0+tr0_se, alpha=0.5, edgecolor='r', facecolor='r')
    plt.plot(time_aligned_1stSide, tr1, 'b', label = 'high rate')
    plt.plot(time_aligned_1stSide, tr0, 'r', label = 'low rate')
    # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, lr_trs],  axis = (1, 2)), 'r', label = 'high rate')
    # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, hr_trs],  axis = (1, 2)), 'b', label = 'low rate')
    plt.xlabel('time aligned to choice onset (ms)')
    plt.title('Incorrect trials')
    
    
'''
# classes: stimulus rates    
if doPlots:    
    
    cmap = matplotlib.cm.get_cmap('seismic', 11)    
    # stim-aligned projections and raw average
    plt.figure()
#    plt.subplot(1,2,1)
    cnt=0
    for i in np.unique(Ysr):
        cnt = cnt+1
#        tr1 = np.nanmean(X[:, Ysr==i],  axis = 1)
#        tr1_se = np.nanstd(X[:, Ysr==i],  axis = 1) / np.sqrt(numTrials);
        a1 = np.nanmean(X[:, :, Ysr==i],  axis=1) # frames x trials (average across neurons)
        tr1 = np.nanmean(a1,  axis = 1)
        tr1_se = np.nanstd(a1,  axis = 1) / np.sqrt(numTrials);
        plt.fill_between(time_aligned_stim, tr1-tr1_se, tr1+tr1_se, alpha=0.5, edgecolor=cmap(cnt)[:3], facecolor=cmap(cnt)[:3])
    #    plt.fill_between(time_aligned_stim, tr0-tr0_se, tr0+tr0_se, alpha=0.5, edgecolor='r', facecolor='r')
        plt.plot(time_aligned_stim, tr1, color=cmap(cnt)[:3], label = 'high rate')
#    plt.legend()
'''    
    
    


########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
#%%
####################################################################################
########## Training SVM to decode choice on choice-aligned traces ##########
####################################################################################

if choiceAligned:    
    #%%     
    ##############################  SVM, L1/L2 penalty  ##################################
    ###############################################################################   
    
    if doSVM==1:
        #%% Find the optimal regularization parameter:
        # Rrain SVM on a range of c, using training and testing dataset ... 
        #    make class predictions for testing dataset and incorrect trials. 
        #    Also shuffle labels for correct and incorrect trials and make predictions (to set null distributions)
        # Repeat the above procedure for numSamples times.
        
        
        # numSamples = 10; # number of iterations for finding the best c (inverse of regularization parameter)
        # if you don't want to regularize, go with a very high cbest and don't run the section below.
        # cbest = 10**6
        
        
        # important vars below:
        #perClassErrorTest_chAl : classErr on testing correct
        #perClassErrorTest_chAl_shfl : classErr on shuffled testing correct
        #perClassErrorTest_incorr_chAl : classErr on testing incorrect
        #perClassErrorTest_incorr_chAl_chance : classErr on shuffled testing incorrect
        
        # regType = 'l1'
        # kfold = 10;
    
        
        if regType == 'l1':
            print '\nRunning l1 svm classification\r' 
            # cvect = 10**(np.arange(-4, 6,0.2))/numTrials;
            cvect = 10**(np.arange(-4, 6,0.2))/X_chAl.shape[0];
        #    cvect = 10**(np.arange(-4, 3.11e2,10.5))/X_chAl.shape[0];
        elif regType == 'l2':
            print '\nRunning l2 svm classification\r' 
            cvect = 10**(np.arange(-6, 6,0.2))/X_chAl.shape[0]
        #    cvect = 10**(np.arange(-6, 6,0.2));
        
        print 'try the following regularization values: \n', cvect
        # formattedList = ['%.2f' % member for member in cvect]
        # print 'try the following regularization values = \n', formattedList
        
        wAllC_chAl = np.ones((numSamples, len(cvect), X_chAl.shape[1]))+np.nan;
        bAllC_chAl = np.ones((numSamples, len(cvect)))+np.nan;
        
        perClassErrorTrain_chAl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_chAl = np.ones((numSamples, len(cvect)))+np.nan;
        
        perClassErrorTest_incorr_chAl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_incorr_chAl_shfl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_incorr_chAl_chance = np.ones((numSamples, len(cvect)))+np.nan;
        #perClassErrorTrain_shfl_chAl = np.ones((numSamples, len(cvect)))+np.nan;
        #perClassErrorTest_shfl_chAl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_chAl_shfl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_chAl_chance = np.ones((numSamples, len(cvect)))+np.nan
        
        #perClassErrorTest_chAl_all0 = np.ones((numSamples, len(cvect)))+np.nan
        
        no = X_chAl.shape[0]
        len_test = no - int((kfold-1.)/kfold*no)
        
        for s in range(numSamples):
            
            print 'iteration %d' %(s)
            permIxs_incorr = rng.permutation(X_chAl_incorr.shape[0]);
        #    permIxs = rng.permutation(X_chAl.shape[0]);
            permIxs = rng.permutation(len_test)
        
        
            a_corr = np.zeros(len_test)
            b = rng.permutation(len_test)[0:len_test/2]
            a_corr[b] = 1
            
            a_incorr = np.zeros(Y_chAl_incorr.shape[0])
            b = rng.permutation(Y_chAl_incorr.shape[0])[0:Y_chAl_incorr.shape[0]/2]
            a_incorr[b] = 1
            
            for i in range(len(cvect)):
                
                if regType == 'l1':                       
                    summary =  crossValidateModel(X_chAl, Y_chAl, linearSVM, kfold = kfold, l1 = cvect[i])
                elif regType == 'l2':
                    summary =  crossValidateModel(X_chAl, Y_chAl, linearSVM, kfold = kfold, l2 = cvect[i])
                            
                wAllC_chAl[s,i,:] = np.squeeze(summary.model.coef_); # weights of all neurons for each value of c and each shuffle
                bAllC_chAl[s,i] = np.squeeze(summary.model.intercept_);
        
                # classification errors                    
                perClassErrorTrain_chAl[s, i] = summary.perClassErrorTrain;
                perClassErrorTest_chAl[s, i] = summary.perClassErrorTest;                
                
                # Testing correct shuffled data: 
                # same decoder trained on correct trials, make predictions on correct with shuffled labels.
                perClassErrorTest_chAl_shfl[s, i] = perClassError(summary.YTest[permIxs], summary.model.predict(summary.XTest));
                perClassErrorTest_chAl_chance[s, i] = perClassError(a_corr, summary.model.predict(summary.XTest));
        #        perClassErrorTest_chAl_shfl[s, i] = perClassError(Y_chAl[permIxs], summary.model.predict(X_chAl));
        
                # this time refit the classifier on correct trials using shuffled labels (and doing testing, training datasets)
                '''
        #        summary_shfl = crossValidateModel(X_chAl_incorr, Y_chAl_incorr[permIxs], linearSVM, kfold = kfold, l1 = cvect[i])
                summary_shfl = crossValidateModel(X_chAl, Y_chAl[permIxs], linearSVM, kfold = kfold, l1 = cvect[i])
                perClassErrorTrain_shfl_chAl[s, i] = summary_shfl.perClassErrorTrain;
                perClassErrorTest_shfl_chAl[s, i] = summary_shfl.perClassErrorTest;
                '''
        
                # Incorrect trials (using the decoder trained on correct trials)
                #linear_svm.fit(XTrain, np.squeeze(YTrain))    # # linear_svm is trained using Xtrain and Ytrain... #train using Xtrain_corr... test on Xtest_corr and X_incorr.
                #summary.model = linear_svm;
                perClassErrorTest_incorr_chAl[s, i] = perClassError(Y_chAl_incorr, summary.model.predict(X_chAl_incorr));
                
                # Null distribution for class error on incorr trials: 
                # We want to use the same decoder (trained on correct trials) to predict the choice on incorrect trials with shuffled class labels.             
                perClassErrorTest_incorr_chAl_shfl[s, i] = perClassError(Y_chAl_incorr[permIxs_incorr], summary.model.predict(X_chAl_incorr));
                perClassErrorTest_incorr_chAl_chance[s, i] = perClassError(a_incorr, summary.model.predict(X_chAl_incorr)); # half of the trials assigned randomly to one choice and the other half to the other choice
        
        
        #        # control: use an all-0 decoder and see what classErr u get
        #        yhat0 = predictMan(summary.XTest, np.zeros((1,summary.model.coef_.shape[1])).squeeze(), bAllC_chAl[s,i])
        #        #perClassError(YTest, yhat0)
        #        perClassErrorTest_chAl_all0[s, i] = abs(yhat0 - summary.YTest).mean() * 100
            
        #        yhat0 = predictMan(summary.XTest, wAllC_chAl[s,i], bAllC_chAl[s,i])
            
            
        #%% Compute average of class errors across numSamples
        
        meanPerClassErrorTest_corr_shfl = np.mean(perClassErrorTest_chAl_shfl, axis = 0);
        semPerClassErrorTest_corr_shfl = np.std(perClassErrorTest_chAl_shfl, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_corr_chance = np.mean(perClassErrorTest_chAl_chance, axis = 0);
        semPerClassErrorTest_corr_chance = np.std(perClassErrorTest_chAl_chance, axis = 0)/np.sqrt(numSamples);
        
        #meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl_chAl, axis = 0);
        #semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl_chAl, axis = 0)/np.sqrt(numSamples);
        
        #meanPerClassErrorTrain_shfl = np.mean(perClassErrorTrain_shfl_chAl, axis = 0);
        #semPerClassErrorTrain_shfl = np.std(perClassErrorTrain_shfl_chAl, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_incorr = np.mean(perClassErrorTest_incorr_chAl, axis = 0);
        semPerClassErrorTest_incorr = np.std(perClassErrorTest_incorr_chAl, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_incorr_shfl = np.mean(perClassErrorTest_incorr_chAl_shfl, axis = 0);
        semPerClassErrorTest_incorr_shfl = np.std(perClassErrorTest_incorr_chAl_shfl, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_incorr_chance = np.mean(perClassErrorTest_incorr_chAl_chance, axis = 0);
        semPerClassErrorTest_incorr_chance = np.std(perClassErrorTest_incorr_chAl_chance, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTrain = np.mean(perClassErrorTrain_chAl, axis = 0);
        semPerClassErrorTrain = np.std(perClassErrorTrain_chAl, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest = np.mean(perClassErrorTest_chAl, axis = 0);
        semPerClassErrorTest = np.std(perClassErrorTest_chAl, axis = 0)/np.sqrt(numSamples);
        
        
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
            cbest_chAl = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
            cbest_chAl = cbest_chAl[0]; # best regularization term based on minError+SE criteria
            cbestAll_chAl = cbest_chAl
        else:
            cbestAll_chAl = cvect[ix]
        print 'best c = ', cbestAll_chAl
        
        
        
        # Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)
        a = (wAllC_chAl!=0) # non-zero weights
        b = np.mean(a, axis=(0,2)) # Fraction of non-zero weights (averaged across shuffles)
        c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle
        cvectnow = cvect[c1stnon0:]
        
        meanPerClassErrorTestnow = np.mean(perClassErrorTest_chAl[:,c1stnon0:], axis = 0);
        semPerClassErrorTestnow = np.std(perClassErrorTest_chAl[:,c1stnon0:], axis = 0)/np.sqrt(numSamples);
        ix = np.argmin(meanPerClassErrorTestnow)
        if smallestC==1:
            cbest_chAl = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
            cbest_chAl = cbest_chAl[0]; # best regularization term based on minError+SE criteria    
        else:
            cbest_chAl = cvectnow[ix]
        
        print 'best c (at least 1 non-0 weight) = ', cbest_chAl
        
        
        # find bestc based on incorr trial class error
        meanPerClassErrorTestnow = np.mean(perClassErrorTest_incorr_chAl[:,c1stnon0:], axis = 0);
        semPerClassErrorTestnow = np.std(perClassErrorTest_incorr_chAl[:,c1stnon0:], axis = 0)/np.sqrt(numSamples);
        ix = np.argmin(meanPerClassErrorTestnow)
        if smallestC==1:    
            cbest_chAl_incorr = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
            cbest_chAl_incorr = cbest_chAl_incorr[0]; # best regularization term based on minError+SE criteria
        else:   
            cbest_chAl_incorr = cvectnow[ix]
            
        print 'incorr: best c (at least 1 non-0 weight) = ', cbest_chAl_incorr
        
        
        
        #%% Set the decoder and class errors at best c (for data)
        
        # you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
        # we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
        indBestC = np.in1d(cvect, cbest_chAl)
        
        w_bestc_data_chAl = wAllC_chAl[:,indBestC,:].squeeze() # numSamps x neurons
        b_bestc_data_chAl = bAllC_chAl[:,indBestC]
        
        classErr_bestC_test_data_chAl = perClassErrorTest_chAl[:,indBestC].squeeze()
        classErr_bestC_test_shfl_chAl = perClassErrorTest_chAl_shfl[:,indBestC].squeeze()
        classErr_bestC_incorr_data_chAl = perClassErrorTest_incorr_chAl[:,indBestC].squeeze()
        classErr_bestC_incorr_chance_chAl = perClassErrorTest_incorr_chAl_chance[:,indBestC].squeeze()
        classErr_bestC_train_data_chAl = perClassErrorTrain_chAl[:,indBestC].squeeze()
        #perClassErrorTest_shfl_incorr_chAl = perClassErrorTest_incorr_chAl_shfl[:,indBestC].squeeze()
        
        
        #%% plot C path
               
        if doPlots:
            print 'Best c (inverse of regularization parameter) = %.2f' %cbest_chAl
            plt.figure('cross validation')
            plt.subplot(1,2,1)
            plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
            plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
            plt.fill_between(cvect, meanPerClassErrorTest_incorr-semPerClassErrorTest_incorr, meanPerClassErrorTest_incorr+ semPerClassErrorTest_incorr, alpha=0.5, edgecolor='g', facecolor='g')        
            plt.fill_between(cvect, meanPerClassErrorTest_incorr_chance-semPerClassErrorTest_incorr_chance, meanPerClassErrorTest_incorr_chance+ semPerClassErrorTest_incorr_chance, alpha=0.5, edgecolor='m', facecolor='m')        
            plt.fill_between(cvect, meanPerClassErrorTest_incorr_shfl-semPerClassErrorTest_incorr_shfl, meanPerClassErrorTest_incorr_shfl+ semPerClassErrorTest_incorr_shfl, alpha=0.5, edgecolor='c', facecolor='c')        
            plt.fill_between(cvect, meanPerClassErrorTest_corr_shfl-semPerClassErrorTest_corr_shfl, meanPerClassErrorTest_corr_shfl+ semPerClassErrorTest_corr_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
            plt.fill_between(cvect, meanPerClassErrorTest_corr_chance-semPerClassErrorTest_corr_chance, meanPerClassErrorTest_corr_chance+ semPerClassErrorTest_corr_chance, alpha=0.5, edgecolor='b', facecolor='b')        
            plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
            plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation')
            plt.plot(cvect, meanPerClassErrorTest_incorr, 'g', label = 'incorr')        
            plt.plot(cvect, meanPerClassErrorTest_incorr_chance, 'm', label = 'incorr-chance')            
            plt.plot(cvect, meanPerClassErrorTest_incorr_shfl, 'c', label = 'incorr-shfl')            
            plt.plot(cvect, meanPerClassErrorTest_corr_shfl, 'y', label = 'corr-shfl')            
            plt.plot(cvect, meanPerClassErrorTest_corr_chance, 'b', label = 'corr-chance')       
            plt.plot(cvect[cvect==cbest_chAl], meanPerClassErrorTest[cvect==cbest_chAl], 'bo')
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('classification error (%)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            plt.tight_layout()
        
        '''
        if doPlots:
            print 'Best c (inverse of regularization parameter) = %.2f' %cbest_chAl
            plt.figure('cross validation')
            plt.subplot(1,2,1)
        #    plt.fill_between(cvect, meanPerClassErrorTrain_shfl-semPerClassErrorTrain_shfl, meanPerClassErrorTrain_shfl+ semPerClassErrorTrain_shfl, alpha=0.5, edgecolor='k', facecolor='k')
        #    plt.fill_between(cvect, meanPerClassErrorTest_corr_shfl-semPerClassErrorTest_corr_shfl, meanPerClassErrorTest_corr_shfl+ semPerClassErrorTest_corr_shfl, alpha=0.5, edgecolor='r', facecolor='r')
            plt.fill_between(cvect, meanPerClassErrorTest_incorr_shfl-semPerClassErrorTest_incorr_shfl, meanPerClassErrorTest_incorr_shfl+ semPerClassErrorTest_incorr_shfl, alpha=0.5, edgecolor='c', facecolor='c')        
        
        #    plt.plot(cvect, meanPerClassErrorTrain_shfl, 'k', label = 'training')
        #    plt.plot(cvect, meanPerClassErrorTest_corr_shfl, 'r', label = 'validation')
            plt.plot(cvect, meanPerClassErrorTest_incorr_shfl, 'c', label = 'incorr-shfl')            
        
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('classification error (%)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            plt.tight_layout()
        '''
        '''
        if doPlots:
            print 'Best c (inverse of regularization parameter) = %.2f' %cbest_chAl
            plt.figure('cross validation')
            plt.subplot(1,2,1)
            plt.fill_between(cvect, meanPerClassErrorTrain_shfl-semPerClassErrorTrain_shfl, meanPerClassErrorTrain_shfl+ semPerClassErrorTrain_shfl, alpha=0.5, edgecolor='k', facecolor='k')
            plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='r', facecolor='r')
            
            plt.plot(cvect, meanPerClassErrorTrain_shfl, 'k', label = 'training')
            plt.plot(cvect, meanPerClassErrorTest_shfl, 'r', label = 'validation')
        
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('classification error (%)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            plt.tight_layout()
        '''
        
        
           
        #%% Plot class error for data and shuffled (for corr trained, corr cv, incorr cv)
        
        pvalueTest_corr = ttest2(classErr_bestC_test_shfl_chAl, classErr_bestC_test_data_chAl, tail = 'left');
        pvalueTest_incorr = ttest2(classErr_bestC_incorr_data_chAl, classErr_bestC_incorr_chance_chAl, tail = 'left');
        pvalueTest_incorr_corr = ttest2(classErr_bestC_incorr_data_chAl, classErr_bestC_test_shfl_chAl, tail = 'left');
        
        #print 'Training error: Mean actual: %.2f%%, Mean shuffled: %.2f%%, p-value = %.2f' %(np.mean(perClassErrorTrain_data_chAl), np.mean(perClassErrorTrain_shfl_chAl), pvalueTrain)
        print 'Testing error: Mean actual: %.2f%%, Mean shuffled: %.2f%%, p-value = %.2f' %(np.mean(classErr_bestC_test_data_chAl), np.mean(classErr_bestC_test_shfl_chAl), pvalueTest_corr)
        print 'Incorr trials: error: Mean actual: %.2f%%, Mean chance: %.2f%%, p-value = %.2f' %(np.mean(classErr_bestC_incorr_data_chAl), np.mean(classErr_bestC_incorr_chance_chAl), pvalueTest_incorr)
        
        
        # Plot the histograms
        if doPlots:
            binEvery = 3; # bin width
            
            plt.figure()
        #    plt.subplot(1,2,1)
        #    plt.hist(perClassErrorTrain_data_chAl, np.arange(0,100,binEvery), color = 'k', label = 'data');
        #    plt.hist(perClassErrorTrain_shfl_chAl, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
        #    plt.xlabel('Training classification error (%)')
        #    plt.ylabel('count')
        #    plt.title('Mean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(perClassErrorTrain_data_chAl), np.mean(perClassErrorTrain_shfl_chAl), pvalueTrain), fontsize = 10)
        #    plt.legend()
        
            plt.subplot(1,2,1)
            plt.hist(classErr_bestC_test_data_chAl, np.arange(0,100,binEvery), color = 'k', label = 'data');
            plt.hist(classErr_bestC_test_shfl_chAl, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
            plt.legend()
            plt.xlabel('Testing classification error (%)')
            plt.title('Testing correct\nMean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(classErr_bestC_test_data_chAl), np.mean(classErr_bestC_test_shfl_chAl), pvalueTest_corr), fontsize = 10)
            plt.ylabel('count')
            plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
        
        #    plt.figure()
            plt.subplot(1,2,2)
            plt.hist(classErr_bestC_incorr_data_chAl, np.arange(0,100,binEvery), color = 'k', label = 'data');
            plt.hist(classErr_bestC_incorr_chance_chAl, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
            plt.legend()
            plt.xlabel('Incorrect classification error (%)')
            plt.title('Incorrect\nMean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(classErr_bestC_incorr_data_chAl), np.mean(classErr_bestC_incorr_chance_chAl), pvalueTest_incorr), fontsize = 10)
            plt.ylabel('count')
            plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
        
        
            plt.figure()
            plt.subplot(1,2,1)
            plt.hist(classErr_bestC_test_data_chAl, np.arange(0,100,binEvery), color = 'k', label = 'testing corr');
            plt.hist(classErr_bestC_incorr_data_chAl, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'incorr');
            plt.legend()
            plt.xlabel('Classification error (%)')
            plt.title('Testing correct vs incorr\nMean corr: %.2f %%, Mean incorr: %.2f %%\n p-value = %.2f' %(np.mean(classErr_bestC_test_data_chAl), np.mean(classErr_bestC_incorr_data_chAl), pvalueTest_incorr_corr), fontsize = 10)
            plt.ylabel('count')
            plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
        
        
        
        
        #%% Compare angle between decoders at different values of c
        # just curious how decoders change during c path
        
        if doPlots:
            # Normalize wAllC_chAl
            nw = np.linalg.norm(wAllC_chAl, axis=2) # numSamps x len(cvect); 2-norm of weights 
            wAllC_chAl_n = np.transpose(np.transpose(wAllC_chAl,(2,0,1))/nw, (1,2,0)) # numSamps x len(cvect) x neurons
            
            # Take average decoder across shuffles and normalize it
            wAv = np.mean(wAllC_chAl_n,axis=0) # len(cvect) x neurons
            nw = np.linalg.norm(wAv, axis=1) # len(cvect); 2-norm of weights 
            wAv_n = np.transpose(np.transpose(wAv)/nw) # len(cvect) x neurons
            
            angc = np.full((len(cvect),len(cvect)), np.nan)
            angcAv = np.full((len(cvect),len(cvect)), np.nan)
            s = rng.permutation(numSamples)[0]
            for i in range(len(cvect)):
                for j in range(len(cvect)):
                    angc[i,j] = np.arccos(abs(np.dot(wAllC_chAl_n[s,i,:], wAllC_chAl_n[s,j,:].transpose())))*180/np.pi
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
            plt.plot(meanPerClassErrorTest_incorr, 'g', label = 'incorr')        
            plt.plot(meanPerClassErrorTest_incorr_shfl, 'c', label = 'incorr-shfl')            
            plt.plot(meanPerClassErrorTest_corr_shfl, 'y', label = 'corr-shfl')   
            #plt.axis('equal')
            #plt.gca().set_aspect('equal', adjustable='box')
        
        
        
        
        #%%     
        ###############################################################################
        ###############################################################################
        ##############################  SVM, BAGGING    ####################################
        ###############################################################################    
        ###############################################################################
        
        if doBagging:
            
            cbeg = cbest_chAl+0 # what c to use for bagging #10
            
            #nRandTrSamp = 10 #nRandTrSamp = 10 #1000 # number of times to subselect trials (for the BAGGING method)
            #doData = 1
            
            w_chAl_all = []
            b_chAl_all = []
            XTest_all = []
            XTrain_all = []
            YTest_all = []
            YTrain_all = []
            
            for i in range(numSamples): # multiple iterations to choose training/testing datasets, so we have numSamples BAGGED decoders to test, so we have numSamps of testing class error.
                print 'Subselect XTrain and XTest, iteration %d' %(i)
                
                ##%% Set aside a group of trials as testing dataset and never use them for training the decoders (which will be later used for BAGGING)    
                #"shuffle trials to break any dependencies on the sequence of trails"....
                
                numObservations, numFeatures = X_chAl.shape
                ## %%%%%% shuffle trials to break any dependencies on the sequence of trails 
                shfl = rng.permutation(np.arange(0, numObservations));
                Ys = Y_chAl[shfl];
                Xs = X_chAl[shfl, :]; 
                ## %%%%% divide data to training and testin sets
                YTrain = Ys[np.arange(0, int((kfold-1.)/kfold*numObservations))]; # Take the first 90% of trials as training set
                YTest = Ys[np.arange(int((kfold-1.)/kfold*numObservations), numObservations)]; # Take the last 10% of trials as testing set
                
                XTrain = Xs[np.arange(0, int((kfold-1.)/kfold*numObservations)), :];
                XTest = Xs[np.arange(int((kfold-1.)/kfold*numObservations), numObservations), :];
                
                XTest_all.append(XTest)
                XTrain_all.append(XTrain)
                YTest_all.append(YTest)
                YTrain_all.append(YTrain)
                   
                   
                ##%% Choose n random trials (n = 80% of min of HR, LR in YTrain)
                   
                nrtr = int(np.ceil(.8 * min((YTrain==0).sum(), (YTrain==1).sum())))
                print 'selecting %d random trials of each category' %(nrtr)
                
                
                ##%% Train SVM on XTrain using BAGGING (subselection of trials on XTrain)
                
                if doData:
                    # data
                    w_chAl = np.full((XTrain.shape[1], nRandTrSamp), np.nan) # neurons x nrandtr
                    b_chAl =  np.full((nRandTrSamp), np.nan) # nrandtr
                    trainE_chAl =  np.full((nRandTrSamp), np.nan) # nrandtr
                        
                    for ir in range(nRandTrSamp): # subselect a group of trials
                        print '\tBAGGING iteration %d' %(ir) # Subselect out of XTrain
                        
                        # subselect nrtr random HR and nrtr random LR trials.
                        lrTrs = np.argwhere(YTrain==0)
                        randLrTrs = rng.permutation(lrTrs)[0:nrtr].flatten() # random LR trials
                    
                        hrTrs = np.argwhere(YTrain==1)
                        randHrTrs = rng.permutation(hrTrs)[0:nrtr].flatten() # random HR trials
                        
                        randTrs = np.sort(np.concatenate((randLrTrs, randHrTrs)))
                        
                        
                        X = XTrain[randTrs,:]
                        Y = YTrain[randTrs]
                    
                    
                        ##%% train SVM on each frame using the sebselect of trials        
                        linear_svm = svm.LinearSVC(C=cbeg) # default c, also l2    # linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l1', dual=False)   
                        
                        # data
                        linear_svm.fit(X, Y)
                            
                        w_chAl[:, ir] = np.squeeze(linear_svm.coef_);
                        b_chAl[ir] = linear_svm.intercept_;        
                        trainE_chAl[ir] = abs(linear_svm.predict(X)-Y.astype('float')).sum()/len(Y)*100;
                
                    w_chAl_all.append(w_chAl)
                    b_chAl_all.append(b_chAl)
                    
                
            w_chAl_all = np.array(w_chAl_all) #numSamps x neurons x nRandTrSamp
            b_chAl_all = np.array(b_chAl_all)
            XTest_all = np.array(XTest_all) #numSamps x testingTrs x neurons
            XTrain_all = np.array(XTrain_all) #numSamps x trainingTrs x neurons
            YTest_all = np.array(YTest_all) #numSamps x testingTrs
            YTrain_all = np.array(YTrain_all) #numSamps x trainingTrs
            
            
            
            #%% Now find the final BAGGED decoder for each round of training/testing trial subselection you performed above. And compute class error on different datasets (testing, training, incorrect, shuffled incorrect)
            
            classErr_corrTrain_all = np.full((numSamples), np.nan)
            classErr_corrTrainTest_all = np.full((numSamples), np.nan)
            classErr_corrTest_all = np.full((numSamples), np.nan)
            classErr_incorrTest_all = np.full((numSamples), np.nan)
            classErr_incorrTest_sh_all = [] # numSamps x numShuffles to get chance/shuffled data: classErr_corrTest_chance_all[i,:] shows chance data for bagged decoder i.
            classErr_incorrTest_chance_all = []
            classErr_corrTest_sh_all = []
            classErr_corrTest_chance_all = []
            
            for i in range(numSamples):
                print '--------- iter ', i, ' ---------'
                w_chAl = w_chAl_all[i,:,:]
                b_chAl = b_chAl_all[i,:]
                
                XTest = XTest_all[i,:,:]
                XTrain = XTrain_all[i,:,:]
                YTest = YTest_all[i,:]
                YTrain = YTrain_all[i,:]
                    
                ##%% Normalize weights (ie the w vector at each frame must have length 1)
                
                normw = np.linalg.norm(w_chAl, axis=0) # nrandtr; 2-norm of weights 
                w_chAl_normed = w_chAl/normw # neurons x nrandtr; normalize weights so weights (of each frame) have length 1
                if sum(normw<=eps).sum()!=0:
                    print 'take care of this; you need to reshape w_normed first'
                                #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
                
                
                ##%% Set the final decoder by aggregating decoders across trial subselects (ie average ws across all trial subselects), then again normalize them so they have length 1 (Bagging)
                
                w_chAl_nb = np.mean(w_chAl_normed, axis=(1)) # neurons 
                nw = np.linalg.norm(w_chAl_nb, axis=0) # scalar; 2-norm of weights 
                w_bag = w_chAl_nb/nw # neurons
                # not sure what to do with the intercept term in the BAGGED method... lets go with the average of b across trials subselects for now:
                b_bag = b_chAl.mean()
                
                # matt's method to find bagged intercept ... classErrs you get using this b_bag are very close to the b_bag (averaged b) you computed above...
                '''
                x = XTrain#XTest #X_chAl
                y = YTrain#YTest #Y_chAl
                projs = np.dot(x, w_bag) # not sure if you should use X_chAll or XTest
                hrp = projs[y==1]    
                lrp = projs[y==0]    
                mu0 = hrp.mean()
                mu1 = lrp.mean()
                sd0 = hrp.std()
                sd1 = lrp.std()
                thresh = (sd0 * mu1 + sd1 * mu0) / (sd1 + sd0)
                b_bag = thresh
                '''
            
                ##%% Now make predictions on different datasets using the BAGGED decodcer
                
                # training data
                yhat_corrTrain = predictMan(XTrain, w_bag, b_bag)
                a = abs(yhat_corrTrain - YTrain).mean() * 100
                print a, '= % class error on training dataset (correct trials)'
                classErr_corrTrain_all[i] = a
                
                
                # all correct data
                yhat_corrAll = predictMan(X_chAl, w_bag, b_bag)
                a = abs(yhat_corrAll - Y_chAl).mean() * 100
                print np.round(a, 2), '= % class error on all correct trials'
                classErr_corrTrainTest_all[i] = a
                
                
                # testing correct
                yhat0 = predictMan(XTest, w_bag, b_bag)
                #perClassError(YTest, yhat0)
                classErr_corrTest = abs(yhat0 - YTest).mean() * 100
                print np.round(classErr_corrTest, 2), '= % class error on correct trials (testing dataset)'
                classErr_corrTest_all[i] = classErr_corrTest
                
            
                # shuffled correct: compare yhat0 with shuffled labels
                classErr_corrTest_sh = []
                classErr_corrTest_chance = []
                for ii in range(numSamples):
                    # generate chance trial labels, half for each choice
                    Y_corr_chance = np.zeros(YTest.shape[0])
                    j = rng.permutation(YTest.shape[0])[0:YTest.shape[0]/2]
                    Y_corr_chance[j] = 1
                    classErr_corrTest_chance.append(abs(yhat0 - Y_corr_chance).mean() * 100)
                    
                    # just shuffle Y_incorr
                    permIx = rng.permutation(YTest.shape[0])
                    classErr_corrTest_sh.append(abs(yhat0 - YTest[permIx]).mean() * 100)    
                    #print np.round(classErr_incorrTest_chance, 2), '= % class error on incorrect trials'
                
                classErr_corrTest_sh = np.array(classErr_corrTest_sh)
                classErr_corrTest_chance = np.array(classErr_corrTest_chance)
                
                classErr_corrTest_sh_all.append(classErr_corrTest_sh) 
                classErr_corrTest_chance_all.append(classErr_corrTest_chance)
                
                print np.round(classErr_corrTest_chance.mean(), 2), '= % class error on correct trials with random labels (equal number for the 2 choices)'
                print np.round(classErr_corrTest_sh.mean(), 2), '= % class error on correct trials with shuffled labels'
                if (YTest==0).sum() > (YTest==1).sum():
                    print np.round(50 - (classErr_corrTest * (1 - .5/(YTest==0).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
                elif (Y_chAl_incorr==0).sum() < (Y_chAl_incorr==1).sum():
                    print np.round(50 - (classErr_corrTest * (1 - .5/(YTest==1).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
            
            
            
                # incorrect trials
                yhat_incorr = predictMan(X_chAl_incorr, w_bag, b_bag)
                classErr_incorrTest = abs(yhat_incorr - Y_chAl_incorr).mean() * 100
                print np.round(classErr_incorrTest, 2), '= % class error on incorrect trials'
                classErr_incorrTest_all[i] = classErr_incorrTest
                
                
                # shuffled incorrect: compare yhat_incorr with shuffled labels
                classErr_incorrTest_sh = []
                classErr_incorrTest_chance = []
                for iii in range(numSamples):
                    # generate chance trial labels, half for each choice
                    Y_incorr_chance = np.zeros(Y_chAl_incorr.shape[0])
                    j = rng.permutation(Y_chAl_incorr.shape[0])[0:Y_chAl_incorr.shape[0]/2]
                    Y_incorr_chance[j] = 1
                    classErr_incorrTest_chance.append(abs(yhat_incorr - Y_incorr_chance).mean() * 100)
                    
                    # just shuffle Y_incorr
                    permIx = rng.permutation(Y_chAl_incorr.shape[0])
                    classErr_incorrTest_sh.append(abs(yhat_incorr - Y_chAl_incorr[permIx]).mean() * 100)    
                    #print np.round(classErr_incorrTest_chance, 2), '= % class error on incorrect trials'
                
                classErr_incorrTest_sh = np.array(classErr_incorrTest_sh)
                classErr_incorrTest_chance = np.array(classErr_incorrTest_chance)
                
                classErr_incorrTest_sh_all.append(classErr_incorrTest_sh)
                classErr_incorrTest_chance_all.append(classErr_incorrTest_chance)
                
                print np.round(classErr_incorrTest_chance.mean(), 2), '= % class error on incorrect trials with random labels (equal number for the 2 choices)'
                print np.round(classErr_incorrTest_sh.mean(), 2), '= % class error on incorrect trials with shuffled labels'
                if (Y_chAl_incorr==0).sum() > (Y_chAl_incorr==1).sum():
                    print np.round(50 - (classErr_incorrTest * (1 - .5/(Y_chAl_incorr==0).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
                elif (Y_chAl_incorr==0).sum() < (Y_chAl_incorr==1).sum():
                    print np.round(50 - (classErr_incorrTest * (1 - .5/(Y_chAl_incorr==1).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
            
            
            classErr_corrTest_sh_all = np.array(classErr_corrTest_sh_all)
            classErr_corrTest_chance_all = np.array(classErr_corrTest_chance_all)
            classErr_incorrTest_sh_all = np.array(classErr_incorrTest_sh_all)
            classErr_incorrTest_chance_all = np.array(classErr_incorrTest_chance_all)
            
            print '---------------Final Results---------------'
            print '---------------BAGGING SVM---------------'
            print 'Testing correct: ', classErr_corrTest_all.mean()
            print 'Incorrect: ', classErr_incorrTest_all.mean()
            print 'Training: ', classErr_corrTrain_all.mean()
            print 'Train and test: ', classErr_corrTrainTest_all.mean()
            print 'Testing correct, chance: ', classErr_corrTest_chance_all.mean()
            print 'Testing correct, shfl: ', classErr_corrTest_sh_all.mean()
            print 'Incorrect, chance: ', classErr_incorrTest_chance_all.mean()
            print 'Incorrect, shfl: ', classErr_incorrTest_sh_all.mean()
            
            
            
            #%% Below is why I am using a faked shuffled data (classErr_incorrTest_chance) which has equal number of trials from each class to set the null distribution:
            '''
            # my own understanding for why and how much classErr_incorrTest_sh may deviate from 50:
            # if there is an imbalance is hr vs lr, then y_shfl will ratain same labels as y
            # for a fraction of trials. Therefore classErr_shfl will be biased to classErr by a degree
            # that is determined by the fraction of trials which retain the same label.
            # We can quanity this in the following way:
            50 - (classErr_incorrTest * (1 - .5/(Y_chAl_incorr==0).mean()))
            # lets break it down:
            a = (1 - .5/(Y_chAl_incorr==0).mean()) # a is the degree by which the original labels are retained in the shuffled data... if fully shuffled it will be 0.
            b = classErr_incorrTest * a # b is the degree by which classErr will bias the chance performance
            c = 50 - b # c is the final chance performance
            '''
            
            
            #%% Lets check how svm.predict and svr.predict work.... Result: for both SVM and SVR, classfier.predict equals xw+b. For SVM, xw+b gives -1 and 1, that should be changed to 0 and 1 to match svm.predict.
            '''
            # check SVR prediction
            # from Gamal:
            #X features matrix (ie matrix of neural data of size trials x neurons);
            #y observations array (ie array of stimulus rate of size trials)
            from sklearn.svm import LinearSVR
            svr = LinearSVR(C=10., epsilon=0.0, dual = True, tol = 1e-9, fit_intercept = True)
            svr.fit(X,Y)
            w = svr.coef_; # weights vector
            b = svr.intercept_; # bias term
            print abs(np.dot(X, w.T)+b - svr.predict(X)).sum() # this should be 0
            
            
            # check SVM prediction
            #from sklearn.svm import LinearSVR
            linear_svm = svm.LinearSVC(C=10.)
            linear_svm.fit(X,Y)
            w = linear_svm.coef_.squeeze() # weights vector
            b = linear_svm.intercept_.squeeze() # bias term
            yhat0 = np.dot(X, w.T)+b # it gives -1 and 1... we need to compare it to threshold (0):
            yhat0[yhat0<0] = 0
            yhat0[yhat0>0] = 1
            print abs(yhat0 - linear_svm.predict(X)).sum() # this should be 0
            '''
            
            
            
            #%%
            ################ Compare BAGGED decoder and L1/L2 decoder ##############
            
            #%% Compute angle between optimal L1 decoders (of all numSamps); and between BAGGED decoder and L1 decoder... we want to see how similar they are
            
            # Normalize w_bestc_data_chAl
            nw = np.linalg.norm(w_bestc_data_chAl, axis=1) # numSamps; 2-norm of weights 
            w_data_chAl_n = (w_bestc_data_chAl.T/nw).T # numSamps x neurons
            
            # average (across numSamps) decoders at best c
            wNow = np.mean(w_data_chAl_n, axis=0) # neurons
            # again normalize it
            nw = np.linalg.norm(wNow) # numSamps; 2-norm of weights 
            wNow_n = wNow/nw # neurons
            
            
            # train svm on all data using best c
            linear_svm = svm.LinearSVC(C = cbest_chAl, loss='squared_hinge', penalty=regType, dual=False)
            linear_svm.fit(X_chAl, Y_chAl)   
            w = np.squeeze(linear_svm.coef_);
            b = linear_svm.intercept_;
            #trainE = abs(linear_svm.predict(X)-Y.astype('float')).sum()/len(Y)*100;
            
            # normalize w
            nw = np.linalg.norm(w) # numSamps; 2-norm of weights 
            w_n = w/nw # neurons
            
            
            # compute angle between the averaged across samples decoder and the one trained using all data (without testing and training)
            print np.arccos(abs(np.dot(wNow_n, w_n.transpose())))*180/np.pi
            
            #s = 20
            #np.arccos(abs(np.dot(w_data_chAl_n[s,:].squeeze(), w_n.transpose())))*180/np.pi
            
            
            # angle between bestc decoders of different shuffles
            ang_samps = np.full((numSamples,numSamples), np.nan)
            for s in range(numSamples):
                for ss in range(numSamples):
                    ang_samps[s,ss] = np.arccos(abs(np.dot(w_data_chAl_n[s,:].squeeze(), w_data_chAl_n[ss,:].squeeze())))*180/np.pi
            
            # average angle between decoders of different samples
            # first set diag elements to nan (because we know their angle is 0)
            np.fill_diagonal(ang_samps, np.nan)
            
            plt.figure()
            plt.imshow(ang_samps, cmap='jet_r')
            plt.colorbar()
            
            print 'Average angle btwn optimal decoders of all shuffles = ', np.nanmean(ang_samps)
            
            
            ########## now compare angle between bagged decoder and l1 optimal decoder
            print np.arccos(abs(np.dot(w_n, w_bag)))*180/np.pi
            
            # with average l1 decoder
            print np.arccos(abs(np.dot(wNow_n, w_bag)))*180/np.pi
            
            # with each l1 decoder
            angs = []
            for s in range(numSamples):
                angs.append(np.arccos(abs(np.dot(w_data_chAl_n[s,:].squeeze(), w_bag)))*180/np.pi)
            
            angs = np.array(angs)
            plt.figure()
            plt.plot(angs)
            print 'Angle between BAGGED decoder and optimal L1 decoder = ', angs.mean()
            
            
            #%% predictions on incorr and testing corr, using l1 decoder vs bagged decoder
            
            # bagged decoder
            print 'BAGGED'
            print 'train: ', np.round(classErr_corrTrain_all.mean(), 2)
            print 'corr: ', np.round(classErr_corrTest_all.mean(), 2)
            print 'incorr: ', np.round(classErr_incorrTest_all.mean(), 2), '\n'
            
            
            # L1 decocer
            print regType,' decocer'
            print 'train: ', np.round(classErr_bestC_train_data_chAl.mean(), 2)
            print 'corr: ', np.round(classErr_bestC_test_data_chAl.mean(), 2)
            print 'incorr: ', np.round(classErr_bestC_incorr_data_chAl.mean(), 2), '\n'
            '''
            # decoder found using all data (without doing testing and training)
            yhat0 = predictMan(X_chAl_incorr, w_n, b)
            print perClassError(yhat0, Y_chAl_incorr)
            
            # averaged decoder across samples
            yhat0 = predictMan(X_chAl_incorr, wNow_n, b_bestc_data_chAl.mean())
            print perClassError(yhat0, Y_chAl_incorr)
            '''
            print regType,' decocer', '(bestc found separately for incorr and testing corr)'
            # min err on testing corr for Bagged decoders (with any c valur larger than bestc)
            print 'corr: ', np.round(np.mean(perClassErrorTest_chAl[:,np.argwhere(indBestC).squeeze():], axis=0).min(), 2)
            # min err on incorr for Bagged decoders (with any c valur larger than bestc)
            print 'incorr: ', np.round(np.mean(perClassErrorTest_incorr_chAl[:,np.argwhere(np.in1d(cvect, cbest_chAl_incorr)).squeeze():], axis=0).min(), 2), '\n'
        
        
        
        #%%
        ####################################################################################################################################
        ############################################ Save results as .mat files in a folder named svm ####################################################################################################################################
        ####################################################################################################################################
        
        #%% Save SVM results
        
        if trialHistAnalysis:
            svmn = 'svmPrevChoice_corrIncorr_chAl_%s_' %(nowStr)
        else:
            svmn = 'svmCurrChoice_corrIncorr_chAl_%s_' %(nowStr)
        print '\n', svmn[:-1]
        
        if saveResults:
            print 'Saving .mat file'
            d = os.path.join(os.path.dirname(pnevFileName), 'svm')
            if not os.path.exists(d):
                print 'creating svm folder'
                os.makedirs(d)
        
            svmName = os.path.join(d, svmn+os.path.basename(pnevFileName))
            print(svmName)
            
            if doBagging==0:
                scio.savemat(svmName, {'thAct':thAct, 'thTrsWithSpike':thTrsWithSpike, 'ep_ms':ep_ms, 'ep_ch':ep_ch,
                                       'th_stim_dur':th_stim_dur, 'numSamples':numSamples, 'nRandTrSamp':nRandTrSamp,
                                       'trsExcluded':trsExcluded, 'trsExcluded_chAl':trsExcluded_chAl, 'trsExcluded_incorr':trsExcluded_incorr, 'trsExcluded_incorr_chAl':trsExcluded_incorr_chAl,
                                       'NsExcluded':NsExcluded, 'NsExcluded_chAl':NsExcluded_chAl,
                                       'regType':regType, 'cvect':cvect,
                                       'smallestC':smallestC, 'cbestAll_chAl':cbestAll_chAl, 'cbest_chAl':cbest_chAl,'cbest_chAl_incorr':cbest_chAl_incorr,
                                       'wAllC_chAl':wAllC_chAl,
                                       'bAllC_chAl':bAllC_chAl,
                                       'perClassErrorTrain_chAl':perClassErrorTrain_chAl,
                                       'perClassErrorTest_chAl':perClassErrorTest_chAl,
                                       'perClassErrorTest_chAl_shfl':perClassErrorTest_chAl_shfl,
                                       'perClassErrorTest_chAl_chance':perClassErrorTest_chAl_chance,
                                       'perClassErrorTest_incorr_chAl':perClassErrorTest_incorr_chAl,
                                       'perClassErrorTest_incorr_chAl_shfl':perClassErrorTest_incorr_chAl_shfl,
                                       'perClassErrorTest_incorr_chAl_chance':perClassErrorTest_incorr_chAl_chance}) 
            else:
                scio.savemat(svmName, {'thAct':thAct, 'thTrsWithSpike':thTrsWithSpike, 'ep_ms':ep_ms, 'ep_ch':ep_ch,
                                       'th_stim_dur':th_stim_dur, 'numSamples':numSamples, 'nRandTrSamp':nRandTrSamp,
                                       'trsExcluded':trsExcluded, 'trsExcluded_chAl':trsExcluded_chAl, 'trsExcluded_incorr':trsExcluded_incorr, 'trsExcluded_incorr_chAl':trsExcluded_incorr_chAl,
                                       'NsExcluded':NsExcluded, 'NsExcluded_chAl':NsExcluded_chAl,
                                       'regType':regType, 'cvect':cvect,
                                       'smallestC':smallestC, 'cbestAll_chAl':cbestAll_chAl, 'cbest_chAl':cbest_chAl,'cbest_chAl_incorr':cbest_chAl_incorr,
                                       'wAllC_chAl':wAllC_chAl,
                                       'bAllC_chAl':bAllC_chAl,
                                       'perClassErrorTrain_chAl':perClassErrorTrain_chAl,
                                       'perClassErrorTest_chAl':perClassErrorTest_chAl,
                                       'perClassErrorTest_chAl_shfl':perClassErrorTest_chAl_shfl,
                                       'perClassErrorTest_chAl_chance':perClassErrorTest_chAl_chance,
                                       'perClassErrorTest_incorr_chAl':perClassErrorTest_incorr_chAl,
                                       'perClassErrorTest_incorr_chAl_shfl':perClassErrorTest_incorr_chAl_shfl,
                                       'perClassErrorTest_incorr_chAl_chance':perClassErrorTest_incorr_chAl_chance,
                                        'cbeg':cbeg, 'w_chAl_all':w_chAl_all, 'b_chAl_all':b_chAl_all, # starting from here are BAGGING vars.
                                        'classErr_corrTest_all':classErr_corrTest_all,
                                        'classErr_incorrTest_all':classErr_incorrTest_all,
                                        'classErr_corrTrain_all':classErr_corrTrain_all,
                                        'classErr_corrTrainTest_all':classErr_corrTrainTest_all,
                                        'classErr_corrTest_chance_all':classErr_corrTest_chance_all,
                                        'classErr_corrTest_sh_all':classErr_corrTest_sh_all,
                                        'classErr_incorrTest_chance_all':classErr_incorrTest_chance_all,
                                        'classErr_incorrTest_sh_all':classErr_incorrTest_sh_all})         
                
            
        else:
            print 'Not saving .mat file'
        
    
    
    
    
    #%% SVR
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    #%%
    ####################################################################################
    ########## Training SVR to decode stimulus on choice-aligned traces ##########
    ####################################################################################
    
    #%%     
    ##############################   SVR; L1/L2 penalty  ##################################
    ###############################################################################   
    
    if doSVR==1:    
        
        from sklearn.svm import LinearSVR
        
        #%% Find the optimal regularization parameter:
        # Rrain SVM on a range of c, using training and testing dataset ... 
        #    make class predictions for testing dataset and incorrect trials. 
        #    Also shuffle labels for correct and incorrect trials and make predictions (to set null distributions)
        # Repeat the above procedure for numSamples times.
        
        
        # numSamples = 10; # number of iterations for finding the best c (inverse of regularization parameter)
        # if you don't want to regularize, go with a very high cbest and don't run the section below.
        # cbest = 10**6
        
        
        # important vars below:
        #perClassErrorTest_chAl : classErr on testing correct
        #perClassErrorTest_chAl_shfl : classErr on shuffled testing correct
        #perClassErrorTest_incorr_chAl : classErr on testing incorrect
        #perClassErrorTest_incorr_chAl_chance : classErr on shuffled testing incorrect
        
        
        
        #cvect = [eps,10**-6,10**-1,10**3]
        cvect = 10**(np.arange(-6, 6,0.2))/X_chAl.shape[0]
        print 'try the following regularization values: \n', cvect
        # formattedList = ['%.2f' % member for member in cvect]
        # print 'try the following regularization values = \n', formattedList
        
        wAllC_chAl = np.ones((numSamples, len(cvect), X_chAl.shape[1]))+np.nan;
        bAllC_chAl = np.ones((numSamples, len(cvect)))+np.nan;
        
        perClassErrorTrain_chAl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_chAl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_incorr_chAl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_incorr_chAl_shfl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_incorr_chAl_chance = np.ones((numSamples, len(cvect)))+np.nan;
        #perClassErrorTrain_shfl_chAl = np.ones((numSamples, len(cvect)))+np.nan;
        #perClassErrorTest_shfl_chAl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_chAl_shfl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_chAl_chance = np.ones((numSamples, len(cvect)))+np.nan;
        
        #perClassErrorTest_chAl_all0 = np.ones((numSamples, len(cvect)))+np.nan
        
        no = X_chAl.shape[0]
        len_test = no - int((kfold-1.)/kfold*no)
        
        for s in range(numSamples):
            print 'iteration %d' %(s)
            permIxs_incorr = rng.permutation(X_chAl_incorr.shape[0]);
        #    permIxs = rng.permutation(X_chAl.shape[0]);
            permIxs = rng.permutation(len_test)
        
            # set y for chance_corr and chance_incorr ... this is random numbers spanning the same range as y ... we prefer this over shuffling y for the null dist
            # bc in case y doesn't have much variability then shuffling it wont give us a very different null dist than original dist.
            aa = Ysr_chAl.min() 
            bb = Ysr_chAl.max()
            a_incorr = (bb-aa) * np.random.random(Y_chAl_incorr.shape[0]) + aa
        #    a_corr = (bb-aa) * np.random.random(Y_chAl.shape[0]) + aa
            a_corr = (bb-aa) * np.random.random(len_test) + aa
            
            
            for i in range(len(cvect)):
                    
        #        summary =  linearSVR(X_chAl, Ysr_chAl, X_chAl, Ysr_chAl, l=10)
                summary =  crossValidateModel(X_chAl, Ysr_chAl, linearSVR, kfold = kfold, l = cvect[i])
        #        summary =  crossValidateModel(X_chAl_incorr, Ysr_chAl_incorr, linearSVR, kfold = kfold, l = cvect[i]) # train on incorr trials.... I cannot trust it bc there are few trials.
                
                wAllC_chAl[s,i,:] = np.squeeze(summary.model.coef_); # weights of all neurons for each value of c and each shuffle
                bAllC_chAl[s,i] = np.squeeze(summary.model.intercept_);
        
                # classification errors                    
                perClassErrorTrain_chAl[s, i] = summary.perClassErrorTrain;
                perClassErrorTest_chAl[s, i] = summary.perClassErrorTest;        
                
                # Testing correct shuffled data: 
                # same decoder trained on correct trials, make predictions on correct with shuffled labels.
                perClassErrorTest_chAl_shfl[s, i] = perClassError_sr(summary.YTest[permIxs], summary.model.predict(summary.XTest));
                perClassErrorTest_chAl_chance[s, i] = perClassError_sr(a_corr, summary.model.predict(summary.XTest));        
                # all correct shuffled
        #        perClassErrorTest_chAl_shfl[s, i] = perClassError_sr(Ysr_chAl[permIxs], summary.model.predict(X_chAl));
        #        perClassErrorTest_chAl_chance[s, i] = perClassError_sr(a_corr, summary.model.predict(X_chAl));        
        
                # this time refit the classifier on correct trials using shuffled labels (and doing testing, training datasets)
                '''
        #        summary_shfl = crossValidateModel(X_chAl_incorr, Y_chAl_incorr[permIxs], linearSVM, kfold = kfold, l1 = cvect[i])
                summary_shfl = crossValidateModel(X_chAl, Y_chAl[permIxs], linearSVM, kfold = kfold, l1 = cvect[i])
                perClassErrorTrain_shfl_chAl[s, i] = summary_shfl.perClassErrorTrain;
                perClassErrorTest_shfl_chAl[s, i] = summary_shfl.perClassErrorTest;
                '''
        
                # Incorrect trials (using the decoder trained on correct trials)
                #linear_svm.fit(XTrain, np.squeeze(YTrain))    # # linear_svm is trained using Xtrain and Ytrain... #train using Xtrain_corr... test on Xtest_corr and X_incorr.
                #summary.model = linear_svm;
                perClassErrorTest_incorr_chAl[s, i] = perClassError_sr(Ysr_chAl_incorr, summary.model.predict(X_chAl_incorr));
                
                # Null distribution for class error on incorr trials: 
                # We want to use the same decoder (trained on correct trials) to predict the choice on incorrect trials with shuffled class labels.             
                perClassErrorTest_incorr_chAl_shfl[s, i] = perClassError_sr(Ysr_chAl_incorr[permIxs_incorr], summary.model.predict(X_chAl_incorr));
                perClassErrorTest_incorr_chAl_chance[s, i] = perClassError_sr(a_incorr, summary.model.predict(X_chAl_incorr)); # half of the trials assigned randomly to one choice and the other half to the other choice
        
        
                # control: use an all-0 decoder and see what classErr u get
        #        yhat0 = predictMan(summary.XTest, np.zeros((1,summary.model.coef_.shape[1])).squeeze(), bAllC_chAl[s,i])
        ##        yhat0 = predictMan(summary.XTest, 10**-5*np.ones((1,summary.model.coef_.shape[1])).squeeze(), bAllC_chAl[s,i])
        #        perClassError(YTest, yhat0)
        #        perClassErrorTest_chAl_all0[s, i] = perClassError_sr(yhat0, summary.YTest)
        
        
        #plt.plot(np.mean(perClassErrorTest_chAl_all0, axis=0))
        #plt.plot(np.mean(perClassErrorTest_chAl, axis=0))
        
        
        #%% Compute average of class errors across numSamples
        
        meanPerClassErrorTest_corr_shfl = np.mean(perClassErrorTest_chAl_shfl, axis = 0);
        semPerClassErrorTest_corr_shfl = np.std(perClassErrorTest_chAl_shfl, axis = 0)/np.sqrt(numSamples);
        
        #meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl_chAl, axis = 0);
        #semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl_chAl, axis = 0)/np.sqrt(numSamples);
        
        #meanPerClassErrorTrain_shfl = np.mean(perClassErrorTrain_shfl_chAl, axis = 0);
        #semPerClassErrorTrain_shfl = np.std(perClassErrorTrain_shfl_chAl, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_incorr = np.mean(perClassErrorTest_incorr_chAl, axis = 0);
        semPerClassErrorTest_incorr = np.std(perClassErrorTest_incorr_chAl, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_incorr_shfl = np.mean(perClassErrorTest_incorr_chAl_shfl, axis = 0);
        semPerClassErrorTest_incorr_shfl = np.std(perClassErrorTest_incorr_chAl_shfl, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_incorr_chance = np.mean(perClassErrorTest_incorr_chAl_chance, axis = 0);
        semPerClassErrorTest_incorr_chance = np.std(perClassErrorTest_incorr_chAl_chance, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTrain = np.mean(perClassErrorTrain_chAl, axis = 0);
        semPerClassErrorTrain = np.std(perClassErrorTrain_chAl, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest = np.mean(perClassErrorTest_chAl, axis = 0);
        semPerClassErrorTest = np.std(perClassErrorTest_chAl, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chAl_chance, axis = 0);
        semPerClassErrorTest_chance = np.std(perClassErrorTest_chAl_chance, axis = 0)/np.sqrt(numSamples);
        
        
        
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
            cbest_chAl = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
            cbest_chAl = cbest_chAl[0]; # best regularization term based on minError+SE criteria
            cbestAll_chAl = cbest_chAl
        else:
            cbestAll_chAl = cvect[ix]
        print 'best c = ', cbestAll_chAl
        
        
        eps = 10**-10 # tiny number below which weight is considered 0
        
        # Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)
        a = abs(wAllC_chAl)>eps # non-zero weights
        #a = (wAllC_chAl!=0) # non-zero weights
        b = np.mean(a, axis=(0,2)) # Fraction of non-zero weights (averaged across shuffles)
        c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle
        cvectnow = cvect[c1stnon0:]
        
        meanPerClassErrorTestnow = np.mean(perClassErrorTest_chAl[:,c1stnon0:], axis = 0);
        semPerClassErrorTestnow = np.std(perClassErrorTest_chAl[:,c1stnon0:], axis = 0)/np.sqrt(numSamples);
        ix = np.argmin(meanPerClassErrorTestnow)
        if smallestC==1:
            cbest_chAl = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
            cbest_chAl = cbest_chAl[0]; # best regularization term based on minError+SE criteria    
        else:
            cbest_chAl = cvectnow[ix]
        
        print 'best c (at least 1 non-0 weight) = ', cbest_chAl
        
        
        # find bestc based on incorr trial class error
        meanPerClassErrorTestnow = np.mean(perClassErrorTest_incorr_chAl[:,c1stnon0:], axis = 0);
        semPerClassErrorTestnow = np.std(perClassErrorTest_incorr_chAl[:,c1stnon0:], axis = 0)/np.sqrt(numSamples);
        ix = np.argmin(meanPerClassErrorTestnow)
        if smallestC==1:    
            cbest_chAl_incorr = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
            cbest_chAl_incorr = cbest_chAl_incorr[0]; # best regularization term based on minError+SE criteria
        else:   
            cbest_chAl_incorr = cvectnow[ix]
            
        print 'incorr: best c (at least 1 non-0 weight) = ', cbest_chAl_incorr
        
        
        
        #%% Set the decoder and class errors at best c (for data)
        
        # you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
        # we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
        indBestC = np.in1d(cvect, cbest_chAl)
        
        w_bestc_data_chAl = wAllC_chAl[:,indBestC,:].squeeze() # numSamps x neurons
        b_bestc_data_chAl = bAllC_chAl[:,indBestC]
        
        classErr_bestC_test_data_chAl = perClassErrorTest_chAl[:,indBestC].squeeze()
        classErr_bestC_test_shfl_chAl = perClassErrorTest_chAl_shfl[:,indBestC].squeeze()
        classErr_bestC_incorr_data_chAl = perClassErrorTest_incorr_chAl[:,indBestC].squeeze()
        classErr_bestC_incorr_chance_chAl = perClassErrorTest_incorr_chAl_chance[:,indBestC].squeeze()
        classErr_bestC_chance_chAl = perClassErrorTest_chAl_chance[:,indBestC].squeeze()
        classErr_bestC_train_data_chAl = perClassErrorTrain_chAl[:,indBestC].squeeze()
        #perClassErrorTest_shfl_incorr_chAl = perClassErrorTest_incorr_chAl_shfl[:,indBestC].squeeze()
        
        
        #%% plot C path
               
        if doPlots:
            print 'Best c (inverse of regularization parameter) = %.2f' %cbest_chAl
            plt.figure('cross validation')
            plt.subplot(1,2,1)
            plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
            plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
            plt.fill_between(cvect, meanPerClassErrorTest_incorr-semPerClassErrorTest_incorr, meanPerClassErrorTest_incorr+ semPerClassErrorTest_incorr, alpha=0.5, edgecolor='g', facecolor='g')        
            plt.fill_between(cvect, meanPerClassErrorTest_incorr_chance-semPerClassErrorTest_incorr_chance, meanPerClassErrorTest_incorr_chance+ semPerClassErrorTest_incorr_chance, alpha=0.5, edgecolor='m', facecolor='m')        
            plt.fill_between(cvect, meanPerClassErrorTest_incorr_shfl-semPerClassErrorTest_incorr_shfl, meanPerClassErrorTest_incorr_shfl+ semPerClassErrorTest_incorr_shfl, alpha=0.5, edgecolor='c', facecolor='c')        
            plt.fill_between(cvect, meanPerClassErrorTest_corr_shfl-semPerClassErrorTest_corr_shfl, meanPerClassErrorTest_corr_shfl+ semPerClassErrorTest_corr_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
            plt.fill_between(cvect, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
            plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
            plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation')
            plt.plot(cvect, meanPerClassErrorTest_incorr, 'g', label = 'incorr')        
            plt.plot(cvect, meanPerClassErrorTest_incorr_chance, 'm', label = 'incorr-chance')            
            plt.plot(cvect, meanPerClassErrorTest_incorr_shfl, 'c', label = 'incorr-shfl')            
            plt.plot(cvect, meanPerClassErrorTest_corr_shfl, 'y', label = 'corr-shfl')            
            plt.plot(cvect, meanPerClassErrorTest_chance, 'b', label = 'corr-chance')        
            plt.plot(cvect[cvect==cbest_chAl], meanPerClassErrorTest[cvect==cbest_chAl], 'bo')
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('prediction error (normalized sum of square error)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            plt.tight_layout()
        #    plt.ylim([-10,110])
        
        '''
        if doPlots:
            print 'Best c (inverse of regularization parameter) = %.2f' %cbest_chAl
            plt.figure('cross validation')
            plt.subplot(1,2,1)
        #    plt.fill_between(cvect, meanPerClassErrorTrain_shfl-semPerClassErrorTrain_shfl, meanPerClassErrorTrain_shfl+ semPerClassErrorTrain_shfl, alpha=0.5, edgecolor='k', facecolor='k')
        #    plt.fill_between(cvect, meanPerClassErrorTest_corr_shfl-semPerClassErrorTest_corr_shfl, meanPerClassErrorTest_corr_shfl+ semPerClassErrorTest_corr_shfl, alpha=0.5, edgecolor='r', facecolor='r')
            plt.fill_between(cvect, meanPerClassErrorTest_incorr_shfl-semPerClassErrorTest_incorr_shfl, meanPerClassErrorTest_incorr_shfl+ semPerClassErrorTest_incorr_shfl, alpha=0.5, edgecolor='c', facecolor='c')        
        
        #    plt.plot(cvect, meanPerClassErrorTrain_shfl, 'k', label = 'training')
        #    plt.plot(cvect, meanPerClassErrorTest_corr_shfl, 'r', label = 'validation')
            plt.plot(cvect, meanPerClassErrorTest_incorr_shfl, 'c', label = 'incorr-shfl')            
        
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('classification error (%)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            plt.tight_layout()
        '''
        '''
        if doPlots:
            print 'Best c (inverse of regularization parameter) = %.2f' %cbest_chAl
            plt.figure('cross validation')
            plt.subplot(1,2,1)
            plt.fill_between(cvect, meanPerClassErrorTrain_shfl-semPerClassErrorTrain_shfl, meanPerClassErrorTrain_shfl+ semPerClassErrorTrain_shfl, alpha=0.5, edgecolor='k', facecolor='k')
            plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='r', facecolor='r')
            
            plt.plot(cvect, meanPerClassErrorTrain_shfl, 'k', label = 'training')
            plt.plot(cvect, meanPerClassErrorTest_shfl, 'r', label = 'validation')
        
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('classification error (%)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            plt.tight_layout()
        '''
        
        
        #%% Plot class error for data and shuffled (for corr trained, corr cv, incorr cv)
        
        pvalueTest_corr = ttest2(classErr_bestC_test_shfl_chAl, classErr_bestC_test_data_chAl, tail = 'left');
        pvalueTest_incorr = ttest2(classErr_bestC_incorr_data_chAl, classErr_bestC_incorr_chance_chAl, tail = 'left');
        pvalueTest_incorr_corr = ttest2(classErr_bestC_incorr_data_chAl, classErr_bestC_test_shfl_chAl, tail = 'left');
        
        a = np.concatenate((classErr_bestC_test_shfl_chAl, classErr_bestC_test_data_chAl, classErr_bestC_incorr_data_chAl, classErr_bestC_incorr_chance_chAl, classErr_bestC_incorr_data_chAl, classErr_bestC_test_shfl_chAl))
        mn = a.min()
        mx = a.max()
        
        #print 'Training error: Mean actual: %.2f%%, Mean shuffled: %.2f%%, p-value = %.2f' %(np.mean(perClassErrorTrain_data_chAl), np.mean(perClassErrorTrain_shfl_chAl), pvalueTrain)
        print 'Testing error: Mean actual: %.2f, Mean shuffled: %.2f, p-value = %.2f' %(np.mean(classErr_bestC_test_data_chAl), np.mean(classErr_bestC_test_shfl_chAl), pvalueTest_corr)
        print 'Incorr trials: error: Mean actual: %.2f, Mean chance: %.2f, p-value = %.2f' %(np.mean(classErr_bestC_incorr_data_chAl), np.mean(classErr_bestC_incorr_chance_chAl), pvalueTest_incorr)
        
        
        # Plot the histograms
        if doPlots:
            binEvery = (mx-mn)/50# 3; # bin width
            
            plt.figure()
        #    plt.subplot(1,2,1)
        #    plt.hist(perClassErrorTrain_data_chAl, np.arange(0,100,binEvery), color = 'k', label = 'data');
        #    plt.hist(perClassErrorTrain_shfl_chAl, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
        #    plt.xlabel('Training classification error (%)')
        #    plt.ylabel('count')
        #    plt.title('Mean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(perClassErrorTrain_data_chAl), np.mean(perClassErrorTrain_shfl_chAl), pvalueTrain), fontsize = 10)
        #    plt.legend()
        
            plt.subplot(1,2,1)
            plt.hist(classErr_bestC_test_data_chAl, np.arange(mn,mx,binEvery), color = 'k', label = 'data');
            plt.hist(classErr_bestC_test_shfl_chAl, np.arange(mn,mx,binEvery), color = 'k', alpha=.5, label = 'shuffled');
            plt.legend()
            plt.xlabel('Testing prediction error (sum (y-yhat)^2 / sum y^2)')
            plt.title('Testing correct\nMean data: %.2f, Mean shuffled: %.2f\n p-value = %.2f' %(np.mean(classErr_bestC_test_data_chAl), np.mean(classErr_bestC_test_shfl_chAl), pvalueTest_corr), fontsize = 10)
            plt.ylabel('count')
            plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
        
        #    plt.figure()
            plt.subplot(1,2,2)
            plt.hist(classErr_bestC_incorr_data_chAl, np.arange(mn,mx,binEvery), color = 'k', label = 'data');
            plt.hist(classErr_bestC_incorr_chance_chAl, np.arange(mn,mx,binEvery), color = 'k', alpha=.5, label = 'shuffled');
            plt.legend()
            plt.xlabel('Incorrect prediction error (sum (y-yhat)^2 / sum y^2)')
            plt.title('Incorrect\nMean data: %.2f, Mean shuffled: %.2f\n p-value = %.2f' %(np.mean(classErr_bestC_incorr_data_chAl), np.mean(classErr_bestC_incorr_chance_chAl), pvalueTest_incorr), fontsize = 10)
            plt.ylabel('count')
            plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
        
        
            plt.figure()
            plt.subplot(1,2,1)
            plt.hist(classErr_bestC_test_data_chAl, np.arange(mn,mx,binEvery), color = 'k', label = 'testing corr');
            plt.hist(classErr_bestC_incorr_data_chAl, np.arange(mn,mx,binEvery), color = 'k', alpha=.5, label = 'incorr');
            plt.legend()
            plt.xlabel('prediction error (sum (y-yhat)^2 / sum y^2)')
            plt.title('Testing correct vs incorr\nMean corr: %.2f, Mean incorr: %.2f\n p-value = %.2f' %(np.mean(classErr_bestC_test_data_chAl), np.mean(classErr_bestC_incorr_data_chAl), pvalueTest_incorr_corr), fontsize = 10)
            plt.ylabel('count')
            plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
        
        
        
        
        
        #%% Compare angle between decoders at different values of c
        # just curious how decoders change during c path
        
        if doPlots:
            # Normalize wAllC_chAl
            nw = np.linalg.norm(wAllC_chAl, axis=2) # numSamps x len(cvect); 2-norm of weights 
            wAllC_chAl_n = np.transpose(np.transpose(wAllC_chAl,(2,0,1))/nw, (1,2,0)) # numSamps x len(cvect) x neurons
            
            # Take average decoder across shuffles and normalize it
            wAv = np.mean(wAllC_chAl_n,axis=0) # len(cvect) x neurons
            nw = np.linalg.norm(wAv, axis=1) # len(cvect); 2-norm of weights 
            wAv_n = np.transpose(np.transpose(wAv)/nw) # len(cvect) x neurons
            
            angc = np.full((len(cvect),len(cvect)), np.nan)
            angcAv = np.full((len(cvect),len(cvect)), np.nan)
            s = rng.permutation(numSamples)[0]
            for i in range(len(cvect)):
                for j in range(len(cvect)):
                    angc[i,j] = np.arccos(abs(np.dot(wAllC_chAl_n[s,i,:], wAllC_chAl_n[s,j,:].transpose())))*180/np.pi
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
            plt.plot(meanPerClassErrorTest_incorr, 'g', label = 'incorr')        
            plt.plot(meanPerClassErrorTest_incorr_shfl, 'c', label = 'incorr-shfl')            
            plt.plot(meanPerClassErrorTest_corr_shfl, 'y', label = 'corr-shfl')   
            #plt.axis('equal')
            #plt.gca().set_aspect('equal', adjustable='box')
            
        
        
        
        
        
        #%%     
        ###############################################################################    
        ##############################  SVR; BAGGING    ###############################
        ###############################################################################    
        
        if doBagging==1:
            cbeg = cbest_chAl+0 # 10
            
            #nRandTrSamp = 10 #nRandTrSamp = 10 #1000 # number of times to subselect trials (for the BAGGING method)
            #doData = 1
            
            w_chAl_all = []
            b_chAl_all = []
            XTest_sr_all = []
            XTrain_sr_all = []
            YTest_sr_all = []
            YTrain_sr_all = []
            
            for i in range(numSamples): # multiple iterations to choose training/testing datasets, so we have numSamples BAGGED decoders to test, so we have numSamps of testing class error.
                print 'Subselect XTrain and XTest, iteration %d' %(i)
                
                ##%% Set aside a group of trials as testing dataset and never use them for training the decoders (which will be later used for BAGGING)    
                #"shuffle trials to break any dependencies on the sequence of trails"....
                
                numObservations, numFeatures = X_chAl.shape
                ## %%%%%% shuffle trials to break any dependencies on the sequence of trails 
                shfl = rng.permutation(np.arange(0, numObservations));
                Ys = Ysr_chAl[shfl]; # stim rates
                Xs = X_chAl[shfl, :]; 
                ## %%%%% divide data to training and testin sets
                YTrain = Ys[np.arange(0, int((kfold-1.)/kfold*numObservations))]; # Take the first 90% of trials as training set
                YTest = Ys[np.arange(int((kfold-1.)/kfold*numObservations), numObservations)]; # Take the last 10% of trials as testing set
                
                XTrain = Xs[np.arange(0, int((kfold-1.)/kfold*numObservations)), :];
                XTest = Xs[np.arange(int((kfold-1.)/kfold*numObservations), numObservations), :];
                
                XTest_sr_all.append(XTest)
                XTrain_sr_all.append(XTrain)
                YTest_sr_all.append(YTest)
                YTrain_sr_all.append(YTrain)
                   
                   
                ##%% Choose n random trials (n = 80% of min of HR, LR in YTrain)
                   
            #    nrtr = int(np.ceil(.8 * min((YTrain==0).sum(), (YTrain==1).sum())))
                nrtr = int(np.ceil(.8*len(YTrain))) # we will subselect 80% of trials
                print 'selecting %d random trials of each category' %(nrtr)
                
                
                ##%% Train SVM on XTrain using BAGGING (subselection of trials on XTrain)
                
                if doData:
                    # data
                    w_chAl = np.full((XTrain.shape[1], nRandTrSamp), np.nan) # neurons x nrandtr
                    b_chAl =  np.full((nRandTrSamp), np.nan) # nrandtr
            #        trainE_chAl =  np.full((nRandTrSamp), np.nan) # nrandtr
                        
                    for ir in range(nRandTrSamp): # subselect a group of trials
                        print '\tBAGGING iteration %d' %(ir) # Subselect out of XTrain
                        
                        # subselect nrtr random trials 
                        randTrs = rng.permutation(len(YTrain))[0:nrtr].flatten() # random trials
                        
                        
                        X = XTrain[randTrs,:]
                        Y = YTrain[randTrs]
                    
                    
                        ##%% train SVM on each frame using the sebselect of trials        
            #            linear_svm = svm.LinearSVC(C=10.) # default c, also l2    # linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l1', dual=False)   
                        linear_svr = LinearSVR(C=cbeg, epsilon=0.0, dual = True, tol = 1e-9, fit_intercept = True)
                        
                        # data
                        linear_svr.fit(X, Y)
                            
                        w_chAl[:, ir] = np.squeeze(linear_svr.coef_);
                        b_chAl[ir] = linear_svr.intercept_;        
            #            trainE_chAl[ir] = abs(linear_svr.predict(X)-Y.astype('float')).sum()/len(Y)*100;
                
                    w_chAl_all.append(w_chAl)
                    b_chAl_all.append(b_chAl)
                    
                
            w_chAl_all = np.array(w_chAl_all) #numSamps x neurons x nRandTrSamp
            b_chAl_all = np.array(b_chAl_all) #numSamps x nRandTrSamp
            XTest_sr_all = np.array(XTest_sr_all) #numSamps x testingTrs x neurons
            XTrain_sr_all = np.array(XTrain_sr_all) #numSamps x trainingTrs x neurons
            YTest_sr_all = np.array(YTest_sr_all) #numSamps x testingTrs
            YTrain_sr_all = np.array(YTrain_sr_all) #numSamps x trainingTrs
            
            #yhat_corrTrain = predictMan(X, w_chAl[:,ir], b_chAl[ir], np.nan)
            #a = perClassError_sr(yhat_corrTrain, Y)
            
            #yhat_incorr = predictMan(X_chAl_incorr, w_chAl[:,ir], b_chAl[ir], np.nan)
            #classErr_incorrTest = abs(yhat_incorr - Ysr_chAl_incorr).mean() * 100
            
            #yhat0 = predictMan(XTest, w_chAl[:,ir], b_chAl[ir], np.nan)
            #classErr_corrTest = abs(yhat0 - YTest).mean() * 100
                
            #%% Now find the final BAGGED decoder for each round of training/testing trial subselection you performed above. And compute class error on different datasets (testing, training, incorrect, shuffled incorrect)
            
            classErr_corrTrain_all = np.full((numSamples), np.nan)
            classErr_corrTrainTest_all = np.full((numSamples), np.nan)
            classErr_corrTest_all = np.full((numSamples), np.nan)
            classErr_incorrTest_all = np.full((numSamples), np.nan)
            classErr_incorrTest_sh_all = []
            classErr_incorrTest_chance_all = []
            classErr_corrTest_sh_all = []
            classErr_corrTest_chance_all = []
            
            aa = Ysr_chAl.min() # -.5  
            bb = Ysr_chAl.max() # .5   
             
               
            for i in range(numSamples):
                print '--------- iter ', i, ' ---------'
                w_chAl = w_chAl_all[i,:,:]
                b_chAl = b_chAl_all[i,:]
                
                XTest = XTest_sr_all[i,:,:]
                XTrain = XTrain_sr_all[i,:,:]
                YTest = YTest_sr_all[i,:]
                YTrain = YTrain_sr_all[i,:]
                    
                ##%% Normalize weights (ie the w vector at each frame must have length 1)
                
                normw = np.linalg.norm(w_chAl, axis=0) # nrandtr; 2-norm of weights 
                w_chAl_normed = w_chAl/normw # neurons x nrandtr; normalize weights so weights (of each frame) have length 1
                if sum(normw<=eps).sum()!=0:
                    print 'take care of this; you need to reshape w_normed first'
                                #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
                
                
                ##%% Set the final decoder by aggregating decoders across trial subselects (ie average ws across all trial subselects), then again normalize them so they have length 1 (Bagging)
                
                w_chAl_nb = np.mean(w_chAl_normed, axis=(1)) # neurons 
                nw = np.linalg.norm(w_chAl_nb, axis=0) # scalar; 2-norm of weights 
                w_bag = w_chAl_nb/nw # neurons
                # not sure what to do with the intercept term in the BAGGED method... lets go with the average of b across trials subselects for now:
                b_bag = b_chAl.mean()
                
                # the following method wont work for svr, bc we dont have only two gaussians for 2 choices....
                # matt's method to find bagged intercept ... classErrs you get using this b_bag are very close to the b_bag (averaged b) you computed above...
                '''
                x = XTrain#XTest #X_chAl
                y = YTrain#YTest #Y_chAl
                projs = np.dot(x, w_bag) # not sure if you should use X_chAll or XTest
                hrp = projs[y==1]    
                lrp = projs[y==0]    
                mu0 = hrp.mean()
                mu1 = lrp.mean()
                sd0 = hrp.std()
                sd1 = lrp.std()
                thresh = (sd0 * mu1 + sd1 * mu0) / (sd1 + sd0)
                b_bag = thresh
                '''
            
                ##%% Now make predictions on different datasets using the BAGGED decodcer
              
                
                # training data
                yhat_corrTrain = predictMan(XTrain, w_bag, b_bag, np.nan)
                a = perClassError_sr(yhat_corrTrain, YTrain)
                print a, '= predict error on training dataset (correct trials)'
                classErr_corrTrain_all[i] = a
                
                
                # all correct data
                yhat_corrAll = predictMan(X_chAl, w_bag, b_bag, np.nan)
                a = perClassError_sr(yhat_corrAll, Ysr_chAl)
                print np.round(a, 2), '= predict error on all correct trials'
                classErr_corrTrainTest_all[i] = a
                
                
                # testing correct
                yhat0 = predictMan(XTest, w_bag, b_bag, np.nan)
                #perClassError(YTest, yhat0)
                classErr_corrTest = perClassError_sr(yhat0, YTest)
                print np.round(classErr_corrTest, 2), '= predict error on correct trials (testing dataset)'
                classErr_corrTest_all[i] = classErr_corrTest
                
            
                # shuffled correct: compare yhat0 with shuffled labels
                classErr_corrTest_sh = []
                classErr_corrTest_chance = []
                for ii in range(numSamples):
                    # set y for chance_corr and chance_incorr ... this is random numbers spanning the same range as y ... we prefer this over shuffling y for the null dist
                    # bc in case y doesn't have much variability then shuffling it wont give us a very different null dist than original dist.        
                    a_corr = (bb-aa) * np.random.random(YTest.shape[0]) + aa # Y_chAl
                    # chance trial labels
                    classErr_corrTest_chance.append(perClassError_sr(yhat0, a_corr))
                    
                    # just shuffle Y_incorr
                    permIx = rng.permutation(YTest.shape[0])
                    classErr_corrTest_sh.append(perClassError_sr(yhat0, YTest[permIx]))    
                    #print np.round(classErr_incorrTest_chance, 2), '= predict error on incorrect trials'
                
                classErr_corrTest_sh = np.array(classErr_corrTest_sh)
                classErr_corrTest_chance = np.array(classErr_corrTest_chance)    
                
                classErr_corrTest_sh_all.append(classErr_corrTest_sh)
                classErr_corrTest_chance_all.append(classErr_corrTest_chance)
                
                print np.round(classErr_corrTest_chance.mean(), 2), '= predict error on correct trials with random labels (equal number for the 2 choices)'
                print np.round(classErr_corrTest_sh.mean(), 2), '= predict error on correct trials with shuffled labels'
            #    if (YTest==0).sum() > (YTest==1).sum():
            #        print np.round(50 - (classErr_corrTest * (1 - .5/(YTest==0).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
            #    elif (Y_chAl_incorr==0).sum() < (Y_chAl_incorr==1).sum():
            #        print np.round(50 - (classErr_corrTest * (1 - .5/(YTest==1).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
            
            
            
                # incorrect trials
                yhat_incorr = predictMan(X_chAl_incorr, w_bag, b_bag, np.nan)
                classErr_incorrTest = perClassError_sr(yhat_incorr, Ysr_chAl_incorr)
                print np.round(classErr_incorrTest, 2), '= predict error on incorrect trials'
                classErr_incorrTest_all[i] = classErr_incorrTest
                
                
                # shuffled incorrect: compare yhat_incorr with shuffled labels
                classErr_incorrTest_sh = []
                classErr_incorrTest_chance = []
                for iii in range(numSamples):
                    # set y for chance_corr and chance_incorr ... this is random numbers spanning the same range as y ... we prefer this over shuffling y for the null dist
                    # bc in case y doesn't have much variability then shuffling it wont give us a very different null dist than original dist.
                    a_incorr = (bb-aa) * np.random.random(Y_chAl_incorr.shape[0]) + aa
                    
                    # chance trial labels
                    classErr_incorrTest_chance.append(perClassError_sr(yhat_incorr, a_incorr))
                    
                    # just shuffle Y_incorr
                    permIx = rng.permutation(Y_chAl_incorr.shape[0])
                    classErr_incorrTest_sh.append(perClassError_sr(yhat_incorr, Ysr_chAl_incorr[permIx]))
                    #print np.round(classErr_incorrTest_chance, 2), '= predict error on incorrect trials'
                
                classErr_incorrTest_sh = np.array(classErr_incorrTest_sh)
                classErr_incorrTest_chance = np.array(classErr_incorrTest_chance)    
                
                classErr_incorrTest_sh_all.append(classErr_incorrTest_sh)
                classErr_incorrTest_chance_all.append(classErr_incorrTest_chance)
                
                print np.round(classErr_incorrTest_chance.mean(), 2), '= predict error on incorrect trials with random labels (equal number for the 2 choices)'
                print np.round(classErr_incorrTest_sh.mean(), 2), '= predict error on incorrect trials with shuffled labels'
            #    if (Y_chAl_incorr==0).sum() > (Y_chAl_incorr==1).sum():
            #        print np.round(50 - (classErr_incorrTest * (1 - .5/(Y_chAl_incorr==0).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
            #    elif (Y_chAl_incorr==0).sum() < (Y_chAl_incorr==1).sum():
            #        print np.round(50 - (classErr_incorrTest * (1 - .5/(Y_chAl_incorr==1).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
            
            
            classErr_corrTest_sh_all = np.array(classErr_corrTest_sh_all)
            classErr_corrTest_chance_all = np.array(classErr_corrTest_chance_all)
            classErr_incorrTest_sh_all = np.array(classErr_incorrTest_sh_all)
            classErr_incorrTest_chance_all = np.array(classErr_incorrTest_chance_all)
            
            print '---------------Final Results---------------'
            print '---------------BAGGING SVR---------------'
            print 'Testing correct: ', classErr_corrTest_all.mean()
            print 'Incorrect: ', classErr_incorrTest_all.mean()
            print 'Training: ', classErr_corrTrain_all.mean()
            print 'Train and test: ', classErr_corrTrainTest_all.mean()
            print 'Testing correct, chance: ', classErr_corrTest_chance_all.mean()
            print 'Testing correct, shfl: ', classErr_corrTest_sh_all.mean()
            print 'Incorrect, chance: ', classErr_incorrTest_chance_all.mean()
            print 'Incorrect, shfl: ', classErr_incorrTest_sh_all.mean()
            
            
            
            #%% Below is why I am using a faked shuffled data (classErr_incorrTest_chance) which has equal number of trials from each class to set the null distribution:
            '''
            # my own understanding for why and how much classErr_incorrTest_sh may deviate from 50:
            # if there is an imbalance is hr vs lr, then y_shfl will ratain same labels as y
            # for a fraction of trials. Therefore classErr_shfl will be biased to classErr by a degree
            # that is determined by the fraction of trials which retain the same label.
            # We can quanity this in the following way:
            50 - (classErr_incorrTest * (1 - .5/(Y_chAl_incorr==0).mean()))
            # lets break it down:
            a = (1 - .5/(Y_chAl_incorr==0).mean()) # a is the degree by which the original labels are retained in the shuffled data... if fully shuffled it will be 0.
            b = classErr_incorrTest * a # b is the degree by which classErr will bias the chance performance
            c = 50 - b # c is the final chance performance
            '''
            
            
            #%% Lets check how svm.predict and svr.predict work.... Result: for both SVM and SVR, classfier.predict equals xw+b. For SVM, xw+b gives -1 and 1, that should be changed to 0 and 1 to match svm.predict.
            '''
            # check SVR prediction
            # from Gamal:
            #X features matrix (ie matrix of neural data of size trials x neurons);
            #y observations array (ie array of stimulus rate of size trials)
            from sklearn.svm import LinearSVR
            svr = LinearSVR(C=10., epsilon=0.0, dual = True, tol = 1e-9, fit_intercept = True)
            svr.fit(X,Y)
            w = svr.coef_; # weights vector
            b = svr.intercept_; # bias term
            print abs(np.dot(X, w.T)+b - svr.predict(X)).sum() # this should be 0
            
            
            # check SVM prediction
            #from sklearn.svm import LinearSVR
            linear_svm = svm.LinearSVC(C=10.)
            linear_svm.fit(X,Y)
            w = linear_svm.coef_.squeeze() # weights vector
            b = linear_svm.intercept_.squeeze() # bias term
            yhat0 = np.dot(X, w.T)+b # it gives -1 and 1... we need to compare it to threshold (0):
            yhat0[yhat0<0] = 0
            yhat0[yhat0>0] = 1
            print abs(yhat0 - linear_svm.predict(X)).sum() # this should be 0
            '''
            
            #%%
            ################ Compare BAGGED decoder and L1/L2 decoder ##############
            
            #%% Compute angle between optimal L1 decoders (of all numSamps); and between BAGGED decoder and L1 decoder... we want to see how similar they are
            
            # Normalize w_bestc_data_chAl
            nw = np.linalg.norm(w_bestc_data_chAl, axis=1) # numSamps; 2-norm of weights 
            w_data_chAl_n = (w_bestc_data_chAl.T/nw).T # numSamps x neurons
            
            # average (across numSamps) decoders at best c
            wNow = np.mean(w_data_chAl_n, axis=0) # neurons
            # again normalize it
            nw = np.linalg.norm(wNow) # numSamps; 2-norm of weights 
            wNow_n = wNow/nw # neurons
            
            
            # train svm on all data using best c
            linear_svm = svm.LinearSVC(C = cbest_chAl, loss='squared_hinge', penalty=regType, dual=False)
            linear_svm.fit(X_chAl, Y_chAl)   
            w = np.squeeze(linear_svm.coef_);
            b = linear_svm.intercept_;
            #trainE = abs(linear_svm.predict(X)-Y.astype('float')).sum()/len(Y)*100;
            
            # normalize w
            nw = np.linalg.norm(w) # numSamps; 2-norm of weights 
            w_n = w/nw # neurons
            
            
            # compute angle between the averaged across samples decoder and the one trained using all data (without testing and training)
            print np.arccos(abs(np.dot(wNow_n, w_n.transpose())))*180/np.pi
            
            #s = 20
            #np.arccos(abs(np.dot(w_data_chAl_n[s,:].squeeze(), w_n.transpose())))*180/np.pi
            
            
            # angle between bestc decoders of different shuffles
            ang_samps = np.full((numSamples,numSamples), np.nan)
            for s in range(numSamples):
                for ss in range(numSamples):
                    ang_samps[s,ss] = np.arccos(abs(np.dot(w_data_chAl_n[s,:].squeeze(), w_data_chAl_n[ss,:].squeeze())))*180/np.pi
            
            # average angle between decoders of different samples
            # first set diag elements to nan (because we know their angle is 0)
            np.fill_diagonal(ang_samps, np.nan)
            
            plt.figure()
            plt.imshow(ang_samps, cmap='jet_r')
            plt.colorbar()
            
            print 'Average angle btwn optimal decoders of all shuffles = ', np.nanmean(ang_samps)
            
            
            ########## now compare angle between bagged decoder and l1 optimal decoder
            print np.arccos(abs(np.dot(w_n, w_bag)))*180/np.pi
            
            # with average l1 decoder
            print np.arccos(abs(np.dot(wNow_n, w_bag)))*180/np.pi
            
            # with each l1 decoder
            angs = []
            for s in range(numSamples):
                angs.append(np.arccos(abs(np.dot(w_data_chAl_n[s,:].squeeze(), w_bag)))*180/np.pi)
            
            angs = np.array(angs)
            plt.figure()
            plt.plot(angs)
            print 'Angle between BAGGED decoder and optimal L1 decoder = ', angs.mean()
            
            
            #%% predictions on incorr and testing corr, using l1 decoder vs bagged decoder
            
            # bagged decoder
            print 'BAGGED'
            # testing corr
            print 'train: ', np.round(classErr_corrTrain_all.mean(), 2)
            print 'corr: ', np.round(classErr_corrTest_all.mean(), 2)
            print 'incorr: ', np.round(classErr_incorrTest_all.mean(), 2), '\n'
            
            
            # L1/L2 decocer
            print regType,' decocer'
            print 'train: ', np.round(classErr_bestC_train_data_chAl.mean(), 2)
            print 'corr: ', np.round(classErr_bestC_test_data_chAl.mean(), 2)
            print 'incorr: ', np.round(classErr_bestC_incorr_data_chAl.mean(), 2), '\n'
            '''
            # decoder found using all data (without doing testing and training)
            yhat0 = predictMan(X_chAl_incorr, w_n, b)
            print perClassError(yhat0, Y_chAl_incorr)
            
            # averaged decoder across samples
            yhat0 = predictMan(X_chAl_incorr, wNow_n, b_bestc_data_chAl.mean())
            print perClassError(yhat0, Y_chAl_incorr)
            '''
            print regType,' decocer', '(bestc found separately for incorr and testing corr)'
            # min err on testing corr for Bagged decoders (with any c valur larger than bestc)
            print 'corr: ', np.round(np.mean(perClassErrorTest_chAl[:,np.argwhere(indBestC).squeeze():], axis=0).min(), 2)
            # min err on incorr for Bagged decoders (with any c valur larger than bestc)
            print 'incorr: ', np.round(np.mean(perClassErrorTest_incorr_chAl[:,np.argwhere(np.in1d(cvect, cbest_chAl_incorr)).squeeze():], axis=0).min(), 2), '\n'
        
        
        
        #%%
        ####################################################################################################################################
        ############################################ Save results as .mat files in a folder named svm ####################################################################################################################################
        ####################################################################################################################################
        
        #%% Save SVR results
        
        if trialHistAnalysis:
            svmn = 'svrPrevChoice_corrIncorr_chAl_%s_' %(nowStr)
        else:
            svmn = 'svrCurrChoice_corrIncorr_chAl_%s_' %(nowStr)
        print '\n', svmn[:-1]
        
        if saveResults:
            print 'Saving .mat file'
            d = os.path.join(os.path.dirname(pnevFileName), 'svm')
            if not os.path.exists(d):
                print 'creating svm folder'
                os.makedirs(d)
        
            svmName = os.path.join(d, svmn+os.path.basename(pnevFileName))
            print(svmName)
            
            if doBagging==0:
                scio.savemat(svmName, {'thAct':thAct, 'thTrsWithSpike':thTrsWithSpike, 'ep_ms':ep_ms, 'ep_ch':ep_ch,
                                       'th_stim_dur':th_stim_dur, 'numSamples':numSamples, 'nRandTrSamp':nRandTrSamp,
                                       'trsExcluded':trsExcluded, 'trsExcluded_chAl':trsExcluded_chAl, 'trsExcluded_incorr':trsExcluded_incorr, 'trsExcluded_incorr_chAl':trsExcluded_incorr_chAl,
                                       'NsExcluded':NsExcluded, 'NsExcluded_chAl':NsExcluded_chAl,
                                       'regType':regType, 'cvect':cvect,
                                       'smallestC':smallestC, 'cbestAll_chAl':cbestAll_chAl, 'cbest_chAl':cbest_chAl,'cbest_chAl_incorr':cbest_chAl_incorr,
                                       'wAllC_chAl':wAllC_chAl,
                                       'bAllC_chAl':bAllC_chAl,
                                       'perClassErrorTrain_chAl':perClassErrorTrain_chAl,
                                       'perClassErrorTest_chAl':perClassErrorTest_chAl,
                                       'perClassErrorTest_chAl_shfl':perClassErrorTest_chAl_shfl,
                                       'perClassErrorTest_chAl_chance':perClassErrorTest_chAl_chance,
                                       'perClassErrorTest_incorr_chAl':perClassErrorTest_incorr_chAl,
                                       'perClassErrorTest_incorr_chAl_shfl':perClassErrorTest_incorr_chAl_shfl,
                                       'perClassErrorTest_incorr_chAl_chance':perClassErrorTest_incorr_chAl_chance}) 
            else:
                scio.savemat(svmName, {'thAct':thAct, 'thTrsWithSpike':thTrsWithSpike, 'ep_ms':ep_ms, 'ep_ch':ep_ch,
                                       'th_stim_dur':th_stim_dur, 'numSamples':numSamples, 'nRandTrSamp':nRandTrSamp,
                                       'trsExcluded':trsExcluded, 'trsExcluded_chAl':trsExcluded_chAl, 'trsExcluded_incorr':trsExcluded_incorr, 'trsExcluded_incorr_chAl':trsExcluded_incorr_chAl,
                                       'NsExcluded':NsExcluded, 'NsExcluded_chAl':NsExcluded_chAl,
                                       'regType':regType, 'cvect':cvect,
                                       'smallestC':smallestC, 'cbestAll_chAl':cbestAll_chAl, 'cbest_chAl':cbest_chAl,'cbest_chAl_incorr':cbest_chAl_incorr,
                                       'wAllC_chAl':wAllC_chAl,
                                       'bAllC_chAl':bAllC_chAl,
                                       'perClassErrorTrain_chAl':perClassErrorTrain_chAl,
                                       'perClassErrorTest_chAl':perClassErrorTest_chAl,
                                       'perClassErrorTest_chAl_shfl':perClassErrorTest_chAl_shfl,
                                       'perClassErrorTest_chAl_chance':perClassErrorTest_chAl_chance,
                                       'perClassErrorTest_incorr_chAl':perClassErrorTest_incorr_chAl,
                                       'perClassErrorTest_incorr_chAl_shfl':perClassErrorTest_incorr_chAl_shfl,
                                       'perClassErrorTest_incorr_chAl_chance':perClassErrorTest_incorr_chAl_chance,
                                        'cbeg':cbeg, 'w_chAl_all':w_chAl_all, 'b_chAl_all':b_chAl_all, # starting from here are BAGGING vars.
                                        'classErr_corrTest_all':classErr_corrTest_all,
                                        'classErr_incorrTest_all':classErr_incorrTest_all,
                                        'classErr_corrTrain_all':classErr_corrTrain_all,
                                        'classErr_corrTrainTest_all':classErr_corrTrainTest_all,
                                        'classErr_corrTest_chance_all':classErr_corrTest_chance_all,
                                        'classErr_corrTest_sh_all':classErr_corrTest_sh_all,
                                        'classErr_incorrTest_chance_all':classErr_incorrTest_chance_all,
                                        'classErr_incorrTest_sh_all':classErr_incorrTest_sh_all})         
                
            
        else:
            print 'Not saving .mat file'
        





#%% Same analyses as above but for stimulus-aligned traces.
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
#%%
####################################################################################
########## Training SVM to decode choice on stimulus-aligned traces ##########
####################################################################################

if stimAligned:    
    #%%     
    ##############################  SVM, L1/L2 penalty  ##################################
    ###############################################################################   
    
    if doSVM==1:
        #%% Find the optimal regularization parameter:
        # Rrain SVM on a range of c, using training and testing dataset ... 
        #    make class predictions for testing dataset and incorrect trials. 
        #    Also shuffle labels for correct and incorrect trials and make predictions (to set null distributions)
        # Repeat the above procedure for numSamples times.
        
        
        # numSamples = 10; # number of iterations for finding the best c (inverse of regularization parameter)
        # if you don't want to regularize, go with a very high cbest and don't run the section below.
        # cbest = 10**6
        
        
        # important vars below:
        #perClassErrorTest_chAl : classErr on testing correct
        #perClassErrorTest_chAl_shfl : classErr on shuffled testing correct
        #perClassErrorTest_incorr_chAl : classErr on testing incorrect
        #perClassErrorTest_incorr_chAl_chance : classErr on shuffled testing incorrect
        
        # regType = 'l1'
        # kfold = 10;
    
        
        if regType == 'l1':
            print '\nRunning l1 svm classification\r' 
            # cvect = 10**(np.arange(-4, 6,0.2))/numTrials;
            cvect = 10**(np.arange(-4, 6,0.2))/X0.shape[0];
        #    cvect = 10**(np.arange(-4, 3.11e2,10.5))/X0.shape[0];
        elif regType == 'l2':
            print '\nRunning l2 svm classification\r' 
            cvect = 10**(np.arange(-6, 6,0.2))/X0.shape[0]
        #    cvect = 10**(np.arange(-6, 6,0.2));
        
        print 'try the following regularization values: \n', cvect
        # formattedList = ['%.2f' % member for member in cvect]
        # print 'try the following regularization values = \n', formattedList
        
        wAllC = np.ones((numSamples, len(cvect), X0.shape[1]))+np.nan;
        bAllC = np.ones((numSamples, len(cvect)))+np.nan;
        
        perClassErrorTrain = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest = np.ones((numSamples, len(cvect)))+np.nan;
        
        perClassErrorTest_incorr = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_incorr_shfl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_incorr_chance = np.ones((numSamples, len(cvect)))+np.nan;
        #perClassErrorTrain_shfl = np.ones((numSamples, len(cvect)))+np.nan;
        #perClassErrorTest_shfl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_shfl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_chance = np.ones((numSamples, len(cvect)))+np.nan
        
        #perClassErrorTest_all0 = np.ones((numSamples, len(cvect)))+np.nan
        
        no = X0.shape[0]
        len_test = no - int((kfold-1.)/kfold*no)
        
        for s in range(numSamples):
            
            print 'iteration %d' %(s)
            permIxs_incorr = rng.permutation(X_incorr.shape[0]);
        #    permIxs = rng.permutation(X0.shape[0]);
            permIxs = rng.permutation(len_test)
        
        
            a_corr = np.zeros(len_test)
            b = rng.permutation(len_test)[0:len_test/2]
            a_corr[b] = 1
            
            a_incorr = np.zeros(Y_incorr.shape[0])
            b = rng.permutation(Y_incorr.shape[0])[0:Y_incorr.shape[0]/2]
            a_incorr[b] = 1
            
            for i in range(len(cvect)):
                
                if regType == 'l1':                       
                    summary =  crossValidateModel(X0, Y0, linearSVM, kfold = kfold, l1 = cvect[i])
                elif regType == 'l2':
                    summary =  crossValidateModel(X0, Y0, linearSVM, kfold = kfold, l2 = cvect[i])
                            
                wAllC[s,i,:] = np.squeeze(summary.model.coef_); # weights of all neurons for each value of c and each shuffle
                bAllC[s,i] = np.squeeze(summary.model.intercept_);
        
                # classification errors                    
                perClassErrorTrain[s, i] = summary.perClassErrorTrain;
                perClassErrorTest[s, i] = summary.perClassErrorTest;                
                
                # Testing correct shuffled data: 
                # same decoder trained on correct trials, make predictions on correct with shuffled labels.
                perClassErrorTest_shfl[s, i] = perClassError(summary.YTest[permIxs], summary.model.predict(summary.XTest));
                perClassErrorTest_chance[s, i] = perClassError(a_corr, summary.model.predict(summary.XTest));
        #        perClassErrorTest_shfl[s, i] = perClassError(Y0[permIxs], summary.model.predict(X0));
        
                # this time refit the classifier on correct trials using shuffled labels (and doing testing, training datasets)
                '''
        #        summary_shfl = crossValidateModel(X0_incorr, Y_incorr[permIxs], linearSVM, kfold = kfold, l1 = cvect[i])
                summary_shfl = crossValidateModel(X0, Y0[permIxs], linearSVM, kfold = kfold, l1 = cvect[i])
                perClassErrorTrain_shfl[s, i] = summary_shfl.perClassErrorTrain;
                perClassErrorTest_shfl[s, i] = summary_shfl.perClassErrorTest;
                '''
        
                # Incorrect trials (using the decoder trained on correct trials)
                #linear_svm.fit(XTrain, np.squeeze(YTrain))    # # linear_svm is trained using Xtrain and Ytrain... #train using Xtrain_corr... test on Xtest_corr and X_incorr.
                #summary.model = linear_svm;
                perClassErrorTest_incorr[s, i] = perClassError(Y_incorr, summary.model.predict(X_incorr));
                
                # Null distribution for class error on incorr trials: 
                # We want to use the same decoder (trained on correct trials) to predict the choice on incorrect trials with shuffled class labels.             
                perClassErrorTest_incorr_shfl[s, i] = perClassError(Y_incorr[permIxs_incorr], summary.model.predict(X_incorr));
                perClassErrorTest_incorr_chance[s, i] = perClassError(a_incorr, summary.model.predict(X_incorr)); # half of the trials assigned randomly to one choice and the other half to the other choice
        
        
        #        # control: use an all-0 decoder and see what classErr u get
        #        yhat0 = predictMan(summary.XTest, np.zeros((1,summary.model.coef_.shape[1])).squeeze(), bAllC[s,i])
        #        #perClassError(YTest, yhat0)
        #        perClassErrorTest_all0[s, i] = abs(yhat0 - summary.YTest).mean() * 100
            
        #        yhat0 = predictMan(summary.XTest, wAllC[s,i], bAllC[s,i])
            
            
        #%% Compute average of class errors across numSamples
        
        meanPerClassErrorTest_corr_shfl = np.mean(perClassErrorTest_shfl, axis = 0);
        semPerClassErrorTest_corr_shfl = np.std(perClassErrorTest_shfl, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_corr_chance = np.mean(perClassErrorTest_chance, axis = 0);
        semPerClassErrorTest_corr_chance = np.std(perClassErrorTest_chance, axis = 0)/np.sqrt(numSamples);
        
        #meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl, axis = 0);
        #semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl, axis = 0)/np.sqrt(numSamples);
        
        #meanPerClassErrorTrain_shfl = np.mean(perClassErrorTrain_shfl, axis = 0);
        #semPerClassErrorTrain_shfl = np.std(perClassErrorTrain_shfl, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_incorr = np.mean(perClassErrorTest_incorr, axis = 0);
        semPerClassErrorTest_incorr = np.std(perClassErrorTest_incorr, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_incorr_shfl = np.mean(perClassErrorTest_incorr_shfl, axis = 0);
        semPerClassErrorTest_incorr_shfl = np.std(perClassErrorTest_incorr_shfl, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_incorr_chance = np.mean(perClassErrorTest_incorr_chance, axis = 0);
        semPerClassErrorTest_incorr_chance = np.std(perClassErrorTest_incorr_chance, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTrain = np.mean(perClassErrorTrain, axis = 0);
        semPerClassErrorTrain = np.std(perClassErrorTrain, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest = np.mean(perClassErrorTest, axis = 0);
        semPerClassErrorTest = np.std(perClassErrorTest, axis = 0)/np.sqrt(numSamples);
        
        
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
        a = (wAllC!=0) # non-zero weights
        b = np.mean(a, axis=(0,2)) # Fraction of non-zero weights (averaged across shuffles)
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
        
        
        # find bestc based on incorr trial class error
        meanPerClassErrorTestnow = np.mean(perClassErrorTest_incorr[:,c1stnon0:], axis = 0);
        semPerClassErrorTestnow = np.std(perClassErrorTest_incorr[:,c1stnon0:], axis = 0)/np.sqrt(numSamples);
        ix = np.argmin(meanPerClassErrorTestnow)
        if smallestC==1:    
            cbest_incorr = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
            cbest_incorr = cbest_incorr[0]; # best regularization term based on minError+SE criteria
        else:   
            cbest_incorr = cvectnow[ix]
            
        print 'incorr: best c (at least 1 non-0 weight) = ', cbest_incorr
        
        
        
        #%% Set the decoder and class errors at best c (for data)
        
        # you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
        # we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
        indBestC = np.in1d(cvect, cbest)
        
        w_bestc_data = wAllC[:,indBestC,:].squeeze() # numSamps x neurons
        b_bestc_data = bAllC[:,indBestC]
        
        classErr_bestC_test_data = perClassErrorTest[:,indBestC].squeeze()
        classErr_bestC_test_shfl = perClassErrorTest_shfl[:,indBestC].squeeze()
        classErr_bestC_incorr_data = perClassErrorTest_incorr[:,indBestC].squeeze()
        classErr_bestC_incorr_chance = perClassErrorTest_incorr_chance[:,indBestC].squeeze()
        classErr_bestC_train_data = perClassErrorTrain[:,indBestC].squeeze()
        #perClassErrorTest_shfl_incorr = perClassErrorTest_incorr_shfl[:,indBestC].squeeze()
        
        
        #%% plot C path
               
        if doPlots:
            print 'Best c (inverse of regularization parameter) = %.2f' %cbest
            plt.figure('cross validation')
            plt.subplot(1,2,1)
            plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
            plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
            plt.fill_between(cvect, meanPerClassErrorTest_incorr-semPerClassErrorTest_incorr, meanPerClassErrorTest_incorr+ semPerClassErrorTest_incorr, alpha=0.5, edgecolor='g', facecolor='g')        
            plt.fill_between(cvect, meanPerClassErrorTest_incorr_chance-semPerClassErrorTest_incorr_chance, meanPerClassErrorTest_incorr_chance+ semPerClassErrorTest_incorr_chance, alpha=0.5, edgecolor='m', facecolor='m')        
            plt.fill_between(cvect, meanPerClassErrorTest_incorr_shfl-semPerClassErrorTest_incorr_shfl, meanPerClassErrorTest_incorr_shfl+ semPerClassErrorTest_incorr_shfl, alpha=0.5, edgecolor='c', facecolor='c')        
            plt.fill_between(cvect, meanPerClassErrorTest_corr_shfl-semPerClassErrorTest_corr_shfl, meanPerClassErrorTest_corr_shfl+ semPerClassErrorTest_corr_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
            plt.fill_between(cvect, meanPerClassErrorTest_corr_chance-semPerClassErrorTest_corr_chance, meanPerClassErrorTest_corr_chance+ semPerClassErrorTest_corr_chance, alpha=0.5, edgecolor='b', facecolor='b')        
            plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
            plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation')
            plt.plot(cvect, meanPerClassErrorTest_incorr, 'g', label = 'incorr')        
            plt.plot(cvect, meanPerClassErrorTest_incorr_chance, 'm', label = 'incorr-chance')            
            plt.plot(cvect, meanPerClassErrorTest_incorr_shfl, 'c', label = 'incorr-shfl')            
            plt.plot(cvect, meanPerClassErrorTest_corr_shfl, 'y', label = 'corr-shfl')            
            plt.plot(cvect, meanPerClassErrorTest_corr_chance, 'b', label = 'corr-chance')       
            plt.plot(cvect[cvect==cbest], meanPerClassErrorTest[cvect==cbest], 'bo')
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('classification error (%)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            plt.tight_layout()
        
        '''
        if doPlots:
            print 'Best c (inverse of regularization parameter) = %.2f' %cbest
            plt.figure('cross validation')
            plt.subplot(1,2,1)
        #    plt.fill_between(cvect, meanPerClassErrorTrain_shfl-semPerClassErrorTrain_shfl, meanPerClassErrorTrain_shfl+ semPerClassErrorTrain_shfl, alpha=0.5, edgecolor='k', facecolor='k')
        #    plt.fill_between(cvect, meanPerClassErrorTest_corr_shfl-semPerClassErrorTest_corr_shfl, meanPerClassErrorTest_corr_shfl+ semPerClassErrorTest_corr_shfl, alpha=0.5, edgecolor='r', facecolor='r')
            plt.fill_between(cvect, meanPerClassErrorTest_incorr_shfl-semPerClassErrorTest_incorr_shfl, meanPerClassErrorTest_incorr_shfl+ semPerClassErrorTest_incorr_shfl, alpha=0.5, edgecolor='c', facecolor='c')        
        
        #    plt.plot(cvect, meanPerClassErrorTrain_shfl, 'k', label = 'training')
        #    plt.plot(cvect, meanPerClassErrorTest_corr_shfl, 'r', label = 'validation')
            plt.plot(cvect, meanPerClassErrorTest_incorr_shfl, 'c', label = 'incorr-shfl')            
        
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('classification error (%)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            plt.tight_layout()
        '''
        '''
        if doPlots:
            print 'Best c (inverse of regularization parameter) = %.2f' %cbest
            plt.figure('cross validation')
            plt.subplot(1,2,1)
            plt.fill_between(cvect, meanPerClassErrorTrain_shfl-semPerClassErrorTrain_shfl, meanPerClassErrorTrain_shfl+ semPerClassErrorTrain_shfl, alpha=0.5, edgecolor='k', facecolor='k')
            plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='r', facecolor='r')
            
            plt.plot(cvect, meanPerClassErrorTrain_shfl, 'k', label = 'training')
            plt.plot(cvect, meanPerClassErrorTest_shfl, 'r', label = 'validation')
        
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('classification error (%)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            plt.tight_layout()
        '''
        
        
           
        #%% Plot class error for data and shuffled (for corr trained, corr cv, incorr cv)
        
        pvalueTest_corr = ttest2(classErr_bestC_test_shfl, classErr_bestC_test_data, tail = 'left');
        pvalueTest_incorr = ttest2(classErr_bestC_incorr_data, classErr_bestC_incorr_chance, tail = 'left');
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
        
        
        
        
        #%% Compare angle between decoders at different values of c
        # just curious how decoders change during c path
        
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
            plt.plot(meanPerClassErrorTest_incorr, 'g', label = 'incorr')        
            plt.plot(meanPerClassErrorTest_incorr_shfl, 'c', label = 'incorr-shfl')            
            plt.plot(meanPerClassErrorTest_corr_shfl, 'y', label = 'corr-shfl')   
            #plt.axis('equal')
            #plt.gca().set_aspect('equal', adjustable='box')
        
        
        
        
        #%%     
        ###############################################################################
        ###############################################################################
        ##############################  SVM, BAGGING    ####################################
        ###############################################################################    
        ###############################################################################
        
        if doBagging:
            
            cbeg = cbest+0 # what c to use for bagging #10
            
            #nRandTrSamp = 10 #nRandTrSamp = 10 #1000 # number of times to subselect trials (for the BAGGING method)
            #doData = 1
            
            w_all = []
            b_all = []
            XTest_all = []
            XTrain_all = []
            YTest_all = []
            YTrain_all = []
            
            for i in range(numSamples): # multiple iterations to choose training/testing datasets, so we have numSamples BAGGED decoders to test, so we have numSamps of testing class error.
                print 'Subselect XTrain and XTest, iteration %d' %(i)
                
                ##%% Set aside a group of trials as testing dataset and never use them for training the decoders (which will be later used for BAGGING)    
                #"shuffle trials to break any dependencies on the sequence of trails"....
                
                numObservations, numFeatures = X0.shape
                ## %%%%%% shuffle trials to break any dependencies on the sequence of trails 
                shfl = rng.permutation(np.arange(0, numObservations));
                Ys = Y0[shfl];
                Xs = X0[shfl, :]; 
                ## %%%%% divide data to training and testin sets
                YTrain = Ys[np.arange(0, int((kfold-1.)/kfold*numObservations))]; # Take the first 90% of trials as training set
                YTest = Ys[np.arange(int((kfold-1.)/kfold*numObservations), numObservations)]; # Take the last 10% of trials as testing set
                
                XTrain = Xs[np.arange(0, int((kfold-1.)/kfold*numObservations)), :];
                XTest = Xs[np.arange(int((kfold-1.)/kfold*numObservations), numObservations), :];
                
                XTest_all.append(XTest)
                XTrain_all.append(XTrain)
                YTest_all.append(YTest)
                YTrain_all.append(YTrain)
                   
                   
                ##%% Choose n random trials (n = 80% of min of HR, LR in YTrain)
                   
                nrtr = int(np.ceil(.8 * min((YTrain==0).sum(), (YTrain==1).sum())))
                print 'selecting %d random trials of each category' %(nrtr)
                
                
                ##%% Train SVM on XTrain using BAGGING (subselection of trials on XTrain)
                
                if doData:
                    # data
                    w = np.full((XTrain.shape[1], nRandTrSamp), np.nan) # neurons x nrandtr
                    b =  np.full((nRandTrSamp), np.nan) # nrandtr
                    trainE =  np.full((nRandTrSamp), np.nan) # nrandtr
                        
                    for ir in range(nRandTrSamp): # subselect a group of trials
                        print '\tBAGGING iteration %d' %(ir) # Subselect out of XTrain
                        
                        # subselect nrtr random HR and nrtr random LR trials.
                        lrTrs = np.argwhere(YTrain==0)
                        randLrTrs = rng.permutation(lrTrs)[0:nrtr].flatten() # random LR trials
                    
                        hrTrs = np.argwhere(YTrain==1)
                        randHrTrs = rng.permutation(hrTrs)[0:nrtr].flatten() # random HR trials
                        
                        randTrs = np.sort(np.concatenate((randLrTrs, randHrTrs)))
                        
                        
                        X = XTrain[randTrs,:]
                        Y = YTrain[randTrs]
                    
                    
                        ##%% train SVM on each frame using the sebselect of trials        
                        linear_svm = svm.LinearSVC(C=cbeg) # default c, also l2    # linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l1', dual=False)   
                        
                        # data
                        linear_svm.fit(X, Y)
                            
                        w[:, ir] = np.squeeze(linear_svm.coef_);
                        b[ir] = linear_svm.intercept_;        
                        trainE[ir] = abs(linear_svm.predict(X)-Y.astype('float')).sum()/len(Y)*100;
                
                    w_all.append(w)
                    b_all.append(b)
                    
                
            w_all = np.array(w_all) #numSamps x neurons x nRandTrSamp
            b_all = np.array(b_all)
            XTest_all = np.array(XTest_all) #numSamps x testingTrs x neurons
            XTrain_all = np.array(XTrain_all) #numSamps x trainingTrs x neurons
            YTest_all = np.array(YTest_all) #numSamps x testingTrs
            YTrain_all = np.array(YTrain_all) #numSamps x trainingTrs
            
            
            
            #%% Now find the final BAGGED decoder for each round of training/testing trial subselection you performed above. And compute class error on different datasets (testing, training, incorrect, shuffled incorrect)
            
            classErr_corrTrain_all = np.full((numSamples), np.nan)
            classErr_corrTrainTest_all = np.full((numSamples), np.nan)
            classErr_corrTest_all = np.full((numSamples), np.nan)
            classErr_incorrTest_all = np.full((numSamples), np.nan)
            classErr_incorrTest_sh_all = [] # numSamps x numShuffles to get chance/shuffled data: classErr_corrTest_chance_all[i,:] shows chance data for bagged decoder i.
            classErr_incorrTest_chance_all = []
            classErr_corrTest_sh_all = []
            classErr_corrTest_chance_all = []
            
            for i in range(numSamples):
                print '--------- iter ', i, ' ---------'
                w = w_all[i,:,:]
                b = b_all[i,:]
                
                XTest = XTest_all[i,:,:]
                XTrain = XTrain_all[i,:,:]
                YTest = YTest_all[i,:]
                YTrain = YTrain_all[i,:]
                    
                ##%% Normalize weights (ie the w vector at each frame must have length 1)
                
                normw = np.linalg.norm(w, axis=0) # nrandtr; 2-norm of weights 
                w_normed = w/normw # neurons x nrandtr; normalize weights so weights (of each frame) have length 1
                if sum(normw<=eps).sum()!=0:
                    print 'take care of this; you need to reshape w_normed first'
                                #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
                
                
                ##%% Set the final decoder by aggregating decoders across trial subselects (ie average ws across all trial subselects), then again normalize them so they have length 1 (Bagging)
                
                w_nb = np.mean(w_normed, axis=(1)) # neurons 
                nw = np.linalg.norm(w_nb, axis=0) # scalar; 2-norm of weights 
                w_bag = w_nb/nw # neurons
                # not sure what to do with the intercept term in the BAGGED method... lets go with the average of b across trials subselects for now:
                b_bag = b.mean()
                
                # matt's method to find bagged intercept ... classErrs you get using this b_bag are very close to the b_bag (averaged b) you computed above...
                '''
                x = XTrain#XTest #X0
                y = YTrain#YTest #Y0
                projs = np.dot(x, w_bag) # not sure if you should use Xl or XTest
                hrp = projs[y==1]    
                lrp = projs[y==0]    
                mu0 = hrp.mean()
                mu1 = lrp.mean()
                sd0 = hrp.std()
                sd1 = lrp.std()
                thresh = (sd0 * mu1 + sd1 * mu0) / (sd1 + sd0)
                b_bag = thresh
                '''
            
                ##%% Now make predictions on different datasets using the BAGGED decodcer
                
                # training data
                yhat_corrTrain = predictMan(XTrain, w_bag, b_bag)
                a = abs(yhat_corrTrain - YTrain).mean() * 100
                print a, '= % class error on training dataset (correct trials)'
                classErr_corrTrain_all[i] = a
                
                
                # all correct data
                yhat_corrAll = predictMan(X0, w_bag, b_bag)
                a = abs(yhat_corrAll - Y0).mean() * 100
                print np.round(a, 2), '= % class error on all correct trials'
                classErr_corrTrainTest_all[i] = a
                
                
                # testing correct
                yhat0 = predictMan(XTest, w_bag, b_bag)
                #perClassError(YTest, yhat0)
                classErr_corrTest = abs(yhat0 - YTest).mean() * 100
                print np.round(classErr_corrTest, 2), '= % class error on correct trials (testing dataset)'
                classErr_corrTest_all[i] = classErr_corrTest
                
            
                # shuffled correct: compare yhat0 with shuffled labels
                classErr_corrTest_sh = []
                classErr_corrTest_chance = []
                for ii in range(numSamples):
                    # generate chance trial labels, half for each choice
                    Y_corr_chance = np.zeros(YTest.shape[0])
                    j = rng.permutation(YTest.shape[0])[0:YTest.shape[0]/2]
                    Y_corr_chance[j] = 1
                    classErr_corrTest_chance.append(abs(yhat0 - Y_corr_chance).mean() * 100)
                    
                    # just shuffle Y_incorr
                    permIx = rng.permutation(YTest.shape[0])
                    classErr_corrTest_sh.append(abs(yhat0 - YTest[permIx]).mean() * 100)    
                    #print np.round(classErr_incorrTest_chance, 2), '= % class error on incorrect trials'
                
                classErr_corrTest_sh = np.array(classErr_corrTest_sh)
                classErr_corrTest_chance = np.array(classErr_corrTest_chance)
                
                classErr_corrTest_sh_all.append(classErr_corrTest_sh) 
                classErr_corrTest_chance_all.append(classErr_corrTest_chance)
                
                print np.round(classErr_corrTest_chance.mean(), 2), '= % class error on correct trials with random labels (equal number for the 2 choices)'
                print np.round(classErr_corrTest_sh.mean(), 2), '= % class error on correct trials with shuffled labels'
                if (YTest==0).sum() > (YTest==1).sum():
                    print np.round(50 - (classErr_corrTest * (1 - .5/(YTest==0).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
                elif (Y_incorr==0).sum() < (Y_incorr==1).sum():
                    print np.round(50 - (classErr_corrTest * (1 - .5/(YTest==1).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
            
            
            
                # incorrect trials
                yhat_incorr = predictMan(X_incorr, w_bag, b_bag)
                classErr_incorrTest = abs(yhat_incorr - Y_incorr).mean() * 100
                print np.round(classErr_incorrTest, 2), '= % class error on incorrect trials'
                classErr_incorrTest_all[i] = classErr_incorrTest
                
                
                # shuffled incorrect: compare yhat_incorr with shuffled labels
                classErr_incorrTest_sh = []
                classErr_incorrTest_chance = []
                for iii in range(numSamples):
                    # generate chance trial labels, half for each choice
                    Y_incorr_chance = np.zeros(Y_incorr.shape[0])
                    j = rng.permutation(Y_incorr.shape[0])[0:Y_incorr.shape[0]/2]
                    Y_incorr_chance[j] = 1
                    classErr_incorrTest_chance.append(abs(yhat_incorr - Y_incorr_chance).mean() * 100)
                    
                    # just shuffle Y_incorr
                    permIx = rng.permutation(Y_incorr.shape[0])
                    classErr_incorrTest_sh.append(abs(yhat_incorr - Y_incorr[permIx]).mean() * 100)    
                    #print np.round(classErr_incorrTest_chance, 2), '= % class error on incorrect trials'
                
                classErr_incorrTest_sh = np.array(classErr_incorrTest_sh)
                classErr_incorrTest_chance = np.array(classErr_incorrTest_chance)
                
                classErr_incorrTest_sh_all.append(classErr_incorrTest_sh)
                classErr_incorrTest_chance_all.append(classErr_incorrTest_chance)
                
                print np.round(classErr_incorrTest_chance.mean(), 2), '= % class error on incorrect trials with random labels (equal number for the 2 choices)'
                print np.round(classErr_incorrTest_sh.mean(), 2), '= % class error on incorrect trials with shuffled labels'
                if (Y_incorr==0).sum() > (Y_incorr==1).sum():
                    print np.round(50 - (classErr_incorrTest * (1 - .5/(Y_incorr==0).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
                elif (Y_incorr==0).sum() < (Y_incorr==1).sum():
                    print np.round(50 - (classErr_incorrTest * (1 - .5/(Y_incorr==1).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
            
            
            classErr_corrTest_sh_all = np.array(classErr_corrTest_sh_all)
            classErr_corrTest_chance_all = np.array(classErr_corrTest_chance_all)
            classErr_incorrTest_sh_all = np.array(classErr_incorrTest_sh_all)
            classErr_incorrTest_chance_all = np.array(classErr_incorrTest_chance_all)
            
            print '---------------Final Results---------------'
            print '---------------BAGGING SVM---------------'
            print 'Testing correct: ', classErr_corrTest_all.mean()
            print 'Incorrect: ', classErr_incorrTest_all.mean()
            print 'Training: ', classErr_corrTrain_all.mean()
            print 'Train and test: ', classErr_corrTrainTest_all.mean()
            print 'Testing correct, chance: ', classErr_corrTest_chance_all.mean()
            print 'Testing correct, shfl: ', classErr_corrTest_sh_all.mean()
            print 'Incorrect, chance: ', classErr_incorrTest_chance_all.mean()
            print 'Incorrect, shfl: ', classErr_incorrTest_sh_all.mean()
            
            
            
            #%% Below is why I am using a faked shuffled data (classErr_incorrTest_chance) which has equal number of trials from each class to set the null distribution:
            '''
            # my own understanding for why and how much classErr_incorrTest_sh may deviate from 50:
            # if there is an imbalance is hr vs lr, then y_shfl will ratain same labels as y
            # for a fraction of trials. Therefore classErr_shfl will be biased to classErr by a degree
            # that is determined by the fraction of trials which retain the same label.
            # We can quanity this in the following way:
            50 - (classErr_incorrTest * (1 - .5/(Y_incorr==0).mean()))
            # lets break it down:
            a = (1 - .5/(Y_incorr==0).mean()) # a is the degree by which the original labels are retained in the shuffled data... if fully shuffled it will be 0.
            b = classErr_incorrTest * a # b is the degree by which classErr will bias the chance performance
            c = 50 - b # c is the final chance performance
            '''
            
            
            #%% Lets check how svm.predict and svr.predict work.... Result: for both SVM and SVR, classfier.predict equals xw+b. For SVM, xw+b gives -1 and 1, that should be changed to 0 and 1 to match svm.predict.
            '''
            # check SVR prediction
            # from Gamal:
            #X features matrix (ie matrix of neural data of size trials x neurons);
            #y observations array (ie array of stimulus rate of size trials)
            from sklearn.svm import LinearSVR
            svr = LinearSVR(C=10., epsilon=0.0, dual = True, tol = 1e-9, fit_intercept = True)
            svr.fit(X,Y)
            w = svr.coef_; # weights vector
            b = svr.intercept_; # bias term
            print abs(np.dot(X, w.T)+b - svr.predict(X)).sum() # this should be 0
            
            
            # check SVM prediction
            #from sklearn.svm import LinearSVR
            linear_svm = svm.LinearSVC(C=10.)
            linear_svm.fit(X,Y)
            w = linear_svm.coef_.squeeze() # weights vector
            b = linear_svm.intercept_.squeeze() # bias term
            yhat0 = np.dot(X, w.T)+b # it gives -1 and 1... we need to compare it to threshold (0):
            yhat0[yhat0<0] = 0
            yhat0[yhat0>0] = 1
            print abs(yhat0 - linear_svm.predict(X)).sum() # this should be 0
            '''
            
            
            
            #%%
            ################ Compare BAGGED decoder and L1/L2 decoder ##############
            
            #%% Compute angle between optimal L1 decoders (of all numSamps); and between BAGGED decoder and L1 decoder... we want to see how similar they are
            
            # Normalize w_bestc_data
            nw = np.linalg.norm(w_bestc_data, axis=1) # numSamps; 2-norm of weights 
            w_data_n = (w_bestc_data.T/nw).T # numSamps x neurons
            
            # average (across numSamps) decoders at best c
            wNow = np.mean(w_data_n, axis=0) # neurons
            # again normalize it
            nw = np.linalg.norm(wNow) # numSamps; 2-norm of weights 
            wNow_n = wNow/nw # neurons
            
            
            # train svm on all data using best c
            linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty=regType, dual=False)
            linear_svm.fit(X0, Y0)   
            w = np.squeeze(linear_svm.coef_);
            b = linear_svm.intercept_;
            #trainE = abs(linear_svm.predict(X)-Y.astype('float')).sum()/len(Y)*100;
            
            # normalize w
            nw = np.linalg.norm(w) # numSamps; 2-norm of weights 
            w_n = w/nw # neurons
            
            
            # compute angle between the averaged across samples decoder and the one trained using all data (without testing and training)
            print np.arccos(abs(np.dot(wNow_n, w_n.transpose())))*180/np.pi
            
            #s = 20
            #np.arccos(abs(np.dot(w_data_n[s,:].squeeze(), w_n.transpose())))*180/np.pi
            
            
            # angle between bestc decoders of different shuffles
            ang_samps = np.full((numSamples,numSamples), np.nan)
            for s in range(numSamples):
                for ss in range(numSamples):
                    ang_samps[s,ss] = np.arccos(abs(np.dot(w_data_n[s,:].squeeze(), w_data_n[ss,:].squeeze())))*180/np.pi
            
            # average angle between decoders of different samples
            # first set diag elements to nan (because we know their angle is 0)
            np.fill_diagonal(ang_samps, np.nan)
            
            plt.figure()
            plt.imshow(ang_samps, cmap='jet_r')
            plt.colorbar()
            
            print 'Average angle btwn optimal decoders of all shuffles = ', np.nanmean(ang_samps)
            
            
            ########## now compare angle between bagged decoder and l1 optimal decoder
            print np.arccos(abs(np.dot(w_n, w_bag)))*180/np.pi
            
            # with average l1 decoder
            print np.arccos(abs(np.dot(wNow_n, w_bag)))*180/np.pi
            
            # with each l1 decoder
            angs = []
            for s in range(numSamples):
                angs.append(np.arccos(abs(np.dot(w_data_n[s,:].squeeze(), w_bag)))*180/np.pi)
            
            angs = np.array(angs)
            plt.figure()
            plt.plot(angs)
            print 'Angle between BAGGED decoder and optimal L1 decoder = ', angs.mean()
            
            
            #%% predictions on incorr and testing corr, using l1 decoder vs bagged decoder
            
            # bagged decoder
            print 'BAGGED'
            print 'train: ', np.round(classErr_corrTrain_all.mean(), 2)
            print 'corr: ', np.round(classErr_corrTest_all.mean(), 2)
            print 'incorr: ', np.round(classErr_incorrTest_all.mean(), 2), '\n'
            
            
            # L1 decocer
            print regType,' decocer'
            print 'train: ', np.round(classErr_bestC_train_data.mean(), 2)
            print 'corr: ', np.round(classErr_bestC_test_data.mean(), 2)
            print 'incorr: ', np.round(classErr_bestC_incorr_data.mean(), 2), '\n'
            '''
            # decoder found using all data (without doing testing and training)
            yhat0 = predictMan(X_incorr, w_n, b)
            print perClassError(yhat0, Y_incorr)
            
            # averaged decoder across samples
            yhat0 = predictMan(X_incorr, wNow_n, b_bestc_data.mean())
            print perClassError(yhat0, Y_incorr)
            '''
            print regType,' decocer', '(bestc found separately for incorr and testing corr)'
            # min err on testing corr for Bagged decoders (with any c valur larger than bestc)
            print 'corr: ', np.round(np.mean(perClassErrorTest[:,np.argwhere(indBestC).squeeze():], axis=0).min(), 2)
            # min err on incorr for Bagged decoders (with any c valur larger than bestc)
            print 'incorr: ', np.round(np.mean(perClassErrorTest_incorr[:,np.argwhere(np.in1d(cvect, cbest_incorr)).squeeze():], axis=0).min(), 2), '\n'
        
        
        
        #%%
        ####################################################################################################################################
        ############################################ Save results as .mat files in a folder named svm ####################################################################################################################################
        ####################################################################################################################################
        
        #%% Save SVM results
        
        if trialHistAnalysis:
            svmn = 'svmPrevChoice_corrIncorr_stAl_%s_' %(nowStr)
        else:
            svmn = 'svmCurrChoice_corrIncorr_stAl_%s_' %(nowStr)
        print '\n', svmn[:-1]
        
        if saveResults:
            print 'Saving .mat file'
            d = os.path.join(os.path.dirname(pnevFileName), 'svm')
            if not os.path.exists(d):
                print 'creating svm folder'
                os.makedirs(d)
        
            svmName = os.path.join(d, svmn+os.path.basename(pnevFileName))
            print(svmName)
            
            if doBagging==0:
                scio.savemat(svmName, {'thAct':thAct, 'thTrsWithSpike':thTrsWithSpike, 'ep_ms':ep_ms, 'ep_ch':ep_ch,
                                       'th_stim_dur':th_stim_dur, 'numSamples':numSamples, 'nRandTrSamp':nRandTrSamp,
                                       'trsExcluded':trsExcluded, 'trsExcluded_chAl':trsExcluded_chAl, 'trsExcluded_incorr':trsExcluded_incorr, 'trsExcluded_incorr_chAl':trsExcluded_incorr_chAl,
                                       'NsExcluded':NsExcluded, 'NsExcluded_chAl':NsExcluded_chAl,
                                       'regType':regType, 'cvect':cvect,
                                       'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,'cbest_incorr':cbest_incorr,
                                       'wAllC':wAllC,
                                       'bAllC':bAllC,
                                       'perClassErrorTrain':perClassErrorTrain,
                                       'perClassErrorTest':perClassErrorTest,
                                       'perClassErrorTest_shfl':perClassErrorTest_shfl,
                                       'perClassErrorTest_chance':perClassErrorTest_chance,
                                       'perClassErrorTest_incorr':perClassErrorTest_incorr,
                                       'perClassErrorTest_incorr_shfl':perClassErrorTest_incorr_shfl,
                                       'perClassErrorTest_incorr_chance':perClassErrorTest_incorr_chance}) 
            else:
                scio.savemat(svmName, {'thAct':thAct, 'thTrsWithSpike':thTrsWithSpike, 'ep_ms':ep_ms, 'ep_ch':ep_ch,
                                       'th_stim_dur':th_stim_dur, 'numSamples':numSamples, 'nRandTrSamp':nRandTrSamp,
                                       'trsExcluded':trsExcluded, 'trsExcluded_chAl':trsExcluded_chAl, 'trsExcluded_incorr':trsExcluded_incorr, 'trsExcluded_incorr_chAl':trsExcluded_incorr_chAl,
                                       'NsExcluded':NsExcluded, 'NsExcluded_chAl':NsExcluded_chAl,
                                       'regType':regType, 'cvect':cvect,
                                       'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,'cbest_incorr':cbest_incorr,
                                       'wAllC':wAllC,
                                       'bAllC':bAllC,
                                       'perClassErrorTrain':perClassErrorTrain,
                                       'perClassErrorTest':perClassErrorTest,
                                       'perClassErrorTest_shfl':perClassErrorTest_shfl,
                                       'perClassErrorTest_chance':perClassErrorTest_chance,
                                       'perClassErrorTest_incorr':perClassErrorTest_incorr,
                                       'perClassErrorTest_incorr_shfl':perClassErrorTest_incorr_shfl,
                                       'perClassErrorTest_incorr_chance':perClassErrorTest_incorr_chance,
                                        'cbeg':cbeg, 'w_all':w_all, 'b_all':b_all, # starting from here are BAGGING vars.
                                        'classErr_corrTest_all':classErr_corrTest_all,
                                        'classErr_incorrTest_all':classErr_incorrTest_all,
                                        'classErr_corrTrain_all':classErr_corrTrain_all,
                                        'classErr_corrTrainTest_all':classErr_corrTrainTest_all,
                                        'classErr_corrTest_chance_all':classErr_corrTest_chance_all,
                                        'classErr_corrTest_sh_all':classErr_corrTest_sh_all,
                                        'classErr_incorrTest_chance_all':classErr_incorrTest_chance_all,
                                        'classErr_incorrTest_sh_all':classErr_incorrTest_sh_all})         
                
            
        else:
            print 'Not saving .mat file'
        
    
    
    
    
    #%% SVR - stimulus-aligned
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    #%%
    ####################################################################################
    ########## Training SVR to decode stimulus on stimulus-aligned traces ##########
    ####################################################################################
    
    #%%     
    ##############################   SVR; L1/L2 penalty  ##################################
    ###############################################################################   
    
    if doSVR==1 and epAllStim==0:    
        
        from sklearn.svm import LinearSVR
        
        #%% Find the optimal regularization parameter:
        # Rrain SVM on a range of c, using training and testing dataset ... 
        #    make class predictions for testing dataset and incorrect trials. 
        #    Also shuffle labels for correct and incorrect trials and make predictions (to set null distributions)
        # Repeat the above procedure for numSamples times.
        
        
        # numSamples = 10; # number of iterations for finding the best c (inverse of regularization parameter)
        # if you don't want to regularize, go with a very high cbest and don't run the section below.
        # cbest = 10**6
        
        
        # important vars below:
        #perClassErrorTest : classErr on testing correct
        #perClassErrorTest_shfl : classErr on shuffled testing correct
        #perClassErrorTest_incorr : classErr on testing incorrect
        #perClassErrorTest_incorr_chance : classErr on shuffled testing incorrect
        
        
        
        #cvect = [eps,10**-6,10**-1,10**3]
        cvect = 10**(np.arange(-6, 6,0.2))/X0.shape[0]
        print 'try the following regularization values: \n', cvect
        # formattedList = ['%.2f' % member for member in cvect]
        # print 'try the following regularization values = \n', formattedList
        
        wAllC = np.ones((numSamples, len(cvect), X0.shape[1]))+np.nan;
        bAllC = np.ones((numSamples, len(cvect)))+np.nan;
        
        perClassErrorTrain = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_incorr = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_incorr_shfl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_incorr_chance = np.ones((numSamples, len(cvect)))+np.nan;
        #perClassErrorTrain_shfl = np.ones((numSamples, len(cvect)))+np.nan;
        #perClassErrorTest_shfl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_shfl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_chance = np.ones((numSamples, len(cvect)))+np.nan;
        
        #perClassErrorTest_all0 = np.ones((numSamples, len(cvect)))+np.nan
        
        no = X0.shape[0]
        len_test = no - int((kfold-1.)/kfold*no)
        
        for s in range(numSamples):
            print 'iteration %d' %(s)
            permIxs_incorr = rng.permutation(X_incorr.shape[0]);
        #    permIxs = rng.permutation(X0.shape[0]);
            permIxs = rng.permutation(len_test)
        
            # set y for chance_corr and chance_incorr ... this is random numbers spanning the same range as y ... we prefer this over shuffling y for the null dist
            # bc in case y doesn't have much variability then shuffling it wont give us a very different null dist than original dist.
            aa = Ysr.min() 
            bb = Ysr.max()
            a_incorr = (bb-aa) * np.random.random(Y_incorr.shape[0]) + aa
        #    a_corr = (bb-aa) * np.random.random(Y0.shape[0]) + aa
            a_corr = (bb-aa) * np.random.random(len_test) + aa
            
            
            for i in range(len(cvect)):
                    
        #        summary =  linearSVR(X0, Ysr, X0, Ysr, l=10)
                summary =  crossValidateModel(X0, Ysr, linearSVR, kfold = kfold, l = cvect[i])
        #        summary =  crossValidateModel(X_incorr, Ysr_incorr, linearSVR, kfold = kfold, l = cvect[i]) # train on incorr trials.... I cannot trust it bc there are few trials.
                
                wAllC[s,i,:] = np.squeeze(summary.model.coef_); # weights of all neurons for each value of c and each shuffle
                bAllC[s,i] = np.squeeze(summary.model.intercept_);
        
                # classification errors                    
                perClassErrorTrain[s, i] = summary.perClassErrorTrain;
                perClassErrorTest[s, i] = summary.perClassErrorTest;        
                
                # Testing correct shuffled data: 
                # same decoder trained on correct trials, make predictions on correct with shuffled labels.
                perClassErrorTest_shfl[s, i] = perClassError_sr(summary.YTest[permIxs], summary.model.predict(summary.XTest));
                perClassErrorTest_chance[s, i] = perClassError_sr(a_corr, summary.model.predict(summary.XTest));        
                # all correct shuffled
        #        perClassErrorTest_shfl[s, i] = perClassError_sr(Ysr[permIxs], summary.model.predict(X0));
        #        perClassErrorTest_chance[s, i] = perClassError_sr(a_corr, summary.model.predict(X0));        
        
                # this time refit the classifier on correct trials using shuffled labels (and doing testing, training datasets)
                '''
        #        summary_shfl = crossValidateModel(X_incorr, Y_incorr[permIxs], linearSVM, kfold = kfold, l1 = cvect[i])
                summary_shfl = crossValidateModel(X0, Y0[permIxs], linearSVM, kfold = kfold, l1 = cvect[i])
                perClassErrorTrain_shfl[s, i] = summary_shfl.perClassErrorTrain;
                perClassErrorTest_shfl[s, i] = summary_shfl.perClassErrorTest;
                '''
        
                # Incorrect trials (using the decoder trained on correct trials)
                #linear_svm.fit(XTrain, np.squeeze(YTrain))    # # linear_svm is trained using Xtrain and Ytrain... #train using Xtrain_corr... test on Xtest_corr and X_incorr.
                #summary.model = linear_svm;
                perClassErrorTest_incorr[s, i] = perClassError_sr(Ysr_incorr, summary.model.predict(X_incorr));
                
                # Null distribution for class error on incorr trials: 
                # We want to use the same decoder (trained on correct trials) to predict the choice on incorrect trials with shuffled class labels.             
                perClassErrorTest_incorr_shfl[s, i] = perClassError_sr(Ysr_incorr[permIxs_incorr], summary.model.predict(X_incorr));
                perClassErrorTest_incorr_chance[s, i] = perClassError_sr(a_incorr, summary.model.predict(X_incorr)); # half of the trials assigned randomly to one choice and the other half to the other choice
        
        
                # control: use an all-0 decoder and see what classErr u get
        #        yhat0 = predictMan(summary.XTest, np.zeros((1,summary.model.coef_.shape[1])).squeeze(), bAllC[s,i])
        ##        yhat0 = predictMan(summary.XTest, 10**-5*np.ones((1,summary.model.coef_.shape[1])).squeeze(), bAllC[s,i])
        #        perClassError(YTest, yhat0)
        #        perClassErrorTest_all0[s, i] = perClassError_sr(yhat0, summary.YTest)
        
        
        #plt.plot(np.mean(perClassErrorTest_all0, axis=0))
        #plt.plot(np.mean(perClassErrorTest, axis=0))
        
        
        #%% Compute average of class errors across numSamples
        
        meanPerClassErrorTest_corr_shfl = np.mean(perClassErrorTest_shfl, axis = 0);
        semPerClassErrorTest_corr_shfl = np.std(perClassErrorTest_shfl, axis = 0)/np.sqrt(numSamples);
        
        #meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl, axis = 0);
        #semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl, axis = 0)/np.sqrt(numSamples);
        
        #meanPerClassErrorTrain_shfl = np.mean(perClassErrorTrain_shfl, axis = 0);
        #semPerClassErrorTrain_shfl = np.std(perClassErrorTrain_shfl, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_incorr = np.mean(perClassErrorTest_incorr, axis = 0);
        semPerClassErrorTest_incorr = np.std(perClassErrorTest_incorr, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_incorr_shfl = np.mean(perClassErrorTest_incorr_shfl, axis = 0);
        semPerClassErrorTest_incorr_shfl = np.std(perClassErrorTest_incorr_shfl, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_incorr_chance = np.mean(perClassErrorTest_incorr_chance, axis = 0);
        semPerClassErrorTest_incorr_chance = np.std(perClassErrorTest_incorr_chance, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTrain = np.mean(perClassErrorTrain, axis = 0);
        semPerClassErrorTrain = np.std(perClassErrorTrain, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest = np.mean(perClassErrorTest, axis = 0);
        semPerClassErrorTest = np.std(perClassErrorTest, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance, axis = 0);
        semPerClassErrorTest_chance = np.std(perClassErrorTest_chance, axis = 0)/np.sqrt(numSamples);
        
        
        
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
        
        
        eps = 10**-10 # tiny number below which weight is considered 0
        
        # Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)
        a = abs(wAllC)>eps # non-zero weights
        #a = (wAllC!=0) # non-zero weights
        b = np.mean(a, axis=(0,2)) # Fraction of non-zero weights (averaged across shuffles)
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
        
        
        # find bestc based on incorr trial class error
        meanPerClassErrorTestnow = np.mean(perClassErrorTest_incorr[:,c1stnon0:], axis = 0);
        semPerClassErrorTestnow = np.std(perClassErrorTest_incorr[:,c1stnon0:], axis = 0)/np.sqrt(numSamples);
        ix = np.argmin(meanPerClassErrorTestnow)
        if smallestC==1:    
            cbest_incorr = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
            cbest_incorr = cbest_incorr[0]; # best regularization term based on minError+SE criteria
        else:   
            cbest_incorr = cvectnow[ix]
            
        print 'incorr: best c (at least 1 non-0 weight) = ', cbest_incorr
        
        
        
        #%% Set the decoder and class errors at best c (for data)
        
        # you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
        # we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
        indBestC = np.in1d(cvect, cbest)
        
        w_bestc_data = wAllC[:,indBestC,:].squeeze() # numSamps x neurons
        b_bestc_data = bAllC[:,indBestC]
        
        classErr_bestC_test_data = perClassErrorTest[:,indBestC].squeeze()
        classErr_bestC_test_shfl = perClassErrorTest_shfl[:,indBestC].squeeze()
        classErr_bestC_incorr_data = perClassErrorTest_incorr[:,indBestC].squeeze()
        classErr_bestC_incorr_chance = perClassErrorTest_incorr_chance[:,indBestC].squeeze()
        classErr_bestC_chance = perClassErrorTest_chance[:,indBestC].squeeze()
        classErr_bestC_train_data = perClassErrorTrain[:,indBestC].squeeze()
        #perClassErrorTest_shfl_incorr = perClassErrorTest_incorr_shfl[:,indBestC].squeeze()
        
        
        #%% plot C path
               
        if doPlots:
            print 'Best c (inverse of regularization parameter) = %.2f' %cbest
            plt.figure('cross validation')
            plt.subplot(1,2,1)
            plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
            plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
            plt.fill_between(cvect, meanPerClassErrorTest_incorr-semPerClassErrorTest_incorr, meanPerClassErrorTest_incorr+ semPerClassErrorTest_incorr, alpha=0.5, edgecolor='g', facecolor='g')        
            plt.fill_between(cvect, meanPerClassErrorTest_incorr_chance-semPerClassErrorTest_incorr_chance, meanPerClassErrorTest_incorr_chance+ semPerClassErrorTest_incorr_chance, alpha=0.5, edgecolor='m', facecolor='m')        
            plt.fill_between(cvect, meanPerClassErrorTest_incorr_shfl-semPerClassErrorTest_incorr_shfl, meanPerClassErrorTest_incorr_shfl+ semPerClassErrorTest_incorr_shfl, alpha=0.5, edgecolor='c', facecolor='c')        
            plt.fill_between(cvect, meanPerClassErrorTest_corr_shfl-semPerClassErrorTest_corr_shfl, meanPerClassErrorTest_corr_shfl+ semPerClassErrorTest_corr_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
            plt.fill_between(cvect, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
            plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
            plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation')
            plt.plot(cvect, meanPerClassErrorTest_incorr, 'g', label = 'incorr')        
            plt.plot(cvect, meanPerClassErrorTest_incorr_chance, 'm', label = 'incorr-chance')            
            plt.plot(cvect, meanPerClassErrorTest_incorr_shfl, 'c', label = 'incorr-shfl')            
            plt.plot(cvect, meanPerClassErrorTest_corr_shfl, 'y', label = 'corr-shfl')            
            plt.plot(cvect, meanPerClassErrorTest_chance, 'b', label = 'corr-chance')        
            plt.plot(cvect[cvect==cbest], meanPerClassErrorTest[cvect==cbest], 'bo')
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('prediction error (normalized sum of square error)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            plt.tight_layout()
        #    plt.ylim([-10,110])
        
        '''
        if doPlots:
            print 'Best c (inverse of regularization parameter) = %.2f' %cbest
            plt.figure('cross validation')
            plt.subplot(1,2,1)
        #    plt.fill_between(cvect, meanPerClassErrorTrain_shfl-semPerClassErrorTrain_shfl, meanPerClassErrorTrain_shfl+ semPerClassErrorTrain_shfl, alpha=0.5, edgecolor='k', facecolor='k')
        #    plt.fill_between(cvect, meanPerClassErrorTest_corr_shfl-semPerClassErrorTest_corr_shfl, meanPerClassErrorTest_corr_shfl+ semPerClassErrorTest_corr_shfl, alpha=0.5, edgecolor='r', facecolor='r')
            plt.fill_between(cvect, meanPerClassErrorTest_incorr_shfl-semPerClassErrorTest_incorr_shfl, meanPerClassErrorTest_incorr_shfl+ semPerClassErrorTest_incorr_shfl, alpha=0.5, edgecolor='c', facecolor='c')        
        
        #    plt.plot(cvect, meanPerClassErrorTrain_shfl, 'k', label = 'training')
        #    plt.plot(cvect, meanPerClassErrorTest_corr_shfl, 'r', label = 'validation')
            plt.plot(cvect, meanPerClassErrorTest_incorr_shfl, 'c', label = 'incorr-shfl')            
        
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('classification error (%)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            plt.tight_layout()
        '''
        '''
        if doPlots:
            print 'Best c (inverse of regularization parameter) = %.2f' %cbest
            plt.figure('cross validation')
            plt.subplot(1,2,1)
            plt.fill_between(cvect, meanPerClassErrorTrain_shfl-semPerClassErrorTrain_shfl, meanPerClassErrorTrain_shfl+ semPerClassErrorTrain_shfl, alpha=0.5, edgecolor='k', facecolor='k')
            plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='r', facecolor='r')
            
            plt.plot(cvect, meanPerClassErrorTrain_shfl, 'k', label = 'training')
            plt.plot(cvect, meanPerClassErrorTest_shfl, 'r', label = 'validation')
        
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('classification error (%)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            plt.tight_layout()
        '''
        
        
        #%% Plot class error for data and shuffled (for corr trained, corr cv, incorr cv)
        
        pvalueTest_corr = ttest2(classErr_bestC_test_shfl, classErr_bestC_test_data, tail = 'left');
        pvalueTest_incorr = ttest2(classErr_bestC_incorr_data, classErr_bestC_incorr_chance, tail = 'left');
        pvalueTest_incorr_corr = ttest2(classErr_bestC_incorr_data, classErr_bestC_test_shfl, tail = 'left');
        
        a = np.concatenate((classErr_bestC_test_shfl, classErr_bestC_test_data, classErr_bestC_incorr_data, classErr_bestC_incorr_chance, classErr_bestC_incorr_data, classErr_bestC_test_shfl))
        mn = a.min()
        mx = a.max()
        
        #print 'Training error: Mean actual: %.2f%%, Mean shuffled: %.2f%%, p-value = %.2f' %(np.mean(perClassErrorTrain_data), np.mean(perClassErrorTrain_shfl), pvalueTrain)
        print 'Testing error: Mean actual: %.2f, Mean shuffled: %.2f, p-value = %.2f' %(np.mean(classErr_bestC_test_data), np.mean(classErr_bestC_test_shfl), pvalueTest_corr)
        print 'Incorr trials: error: Mean actual: %.2f, Mean chance: %.2f, p-value = %.2f' %(np.mean(classErr_bestC_incorr_data), np.mean(classErr_bestC_incorr_chance), pvalueTest_incorr)
        
        
        # Plot the histograms
        if doPlots:
            binEvery = (mx-mn)/50# 3; # bin width
            
            plt.figure()
        #    plt.subplot(1,2,1)
        #    plt.hist(perClassErrorTrain_data, np.arange(0,100,binEvery), color = 'k', label = 'data');
        #    plt.hist(perClassErrorTrain_shfl, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
        #    plt.xlabel('Training classification error (%)')
        #    plt.ylabel('count')
        #    plt.title('Mean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(perClassErrorTrain_data), np.mean(perClassErrorTrain_shfl), pvalueTrain), fontsize = 10)
        #    plt.legend()
        
            plt.subplot(1,2,1)
            plt.hist(classErr_bestC_test_data, np.arange(mn,mx,binEvery), color = 'k', label = 'data');
            plt.hist(classErr_bestC_test_shfl, np.arange(mn,mx,binEvery), color = 'k', alpha=.5, label = 'shuffled');
            plt.legend()
            plt.xlabel('Testing prediction error (sum (y-yhat)^2 / sum y^2)')
            plt.title('Testing correct\nMean data: %.2f, Mean shuffled: %.2f\n p-value = %.2f' %(np.mean(classErr_bestC_test_data), np.mean(classErr_bestC_test_shfl), pvalueTest_corr), fontsize = 10)
            plt.ylabel('count')
            plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
        
        #    plt.figure()
            plt.subplot(1,2,2)
            plt.hist(classErr_bestC_incorr_data, np.arange(mn,mx,binEvery), color = 'k', label = 'data');
            plt.hist(classErr_bestC_incorr_chance, np.arange(mn,mx,binEvery), color = 'k', alpha=.5, label = 'shuffled');
            plt.legend()
            plt.xlabel('Incorrect prediction error (sum (y-yhat)^2 / sum y^2)')
            plt.title('Incorrect\nMean data: %.2f, Mean shuffled: %.2f\n p-value = %.2f' %(np.mean(classErr_bestC_incorr_data), np.mean(classErr_bestC_incorr_chance), pvalueTest_incorr), fontsize = 10)
            plt.ylabel('count')
            plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
        
        
            plt.figure()
            plt.subplot(1,2,1)
            plt.hist(classErr_bestC_test_data, np.arange(mn,mx,binEvery), color = 'k', label = 'testing corr');
            plt.hist(classErr_bestC_incorr_data, np.arange(mn,mx,binEvery), color = 'k', alpha=.5, label = 'incorr');
            plt.legend()
            plt.xlabel('prediction error (sum (y-yhat)^2 / sum y^2)')
            plt.title('Testing correct vs incorr\nMean corr: %.2f, Mean incorr: %.2f\n p-value = %.2f' %(np.mean(classErr_bestC_test_data), np.mean(classErr_bestC_incorr_data), pvalueTest_incorr_corr), fontsize = 10)
            plt.ylabel('count')
            plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
        
        
        
        
        
        #%% Compare angle between decoders at different values of c
        # just curious how decoders change during c path
        
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
            plt.plot(meanPerClassErrorTest_incorr, 'g', label = 'incorr')        
            plt.plot(meanPerClassErrorTest_incorr_shfl, 'c', label = 'incorr-shfl')            
            plt.plot(meanPerClassErrorTest_corr_shfl, 'y', label = 'corr-shfl')   
            #plt.axis('equal')
            #plt.gca().set_aspect('equal', adjustable='box')
        
        
        
        
        
        
        #%%     
        ###############################################################################    
        ##############################  SVR; BAGGING    ###############################
        ###############################################################################    
        
        if doBagging==1:
            cbeg = cbest+0 # 10
            
            #nRandTrSamp = 10 #nRandTrSamp = 10 #1000 # number of times to subselect trials (for the BAGGING method)
            #doData = 1
            
            w_all = []
            b_all = []
            XTest_sr_all = []
            XTrain_sr_all = []
            YTest_sr_all = []
            YTrain_sr_all = []
            
            for i in range(numSamples): # multiple iterations to choose training/testing datasets, so we have numSamples BAGGED decoders to test, so we have numSamps of testing class error.
                print 'Subselect XTrain and XTest, iteration %d' %(i)
                
                ##%% Set aside a group of trials as testing dataset and never use them for training the decoders (which will be later used for BAGGING)    
                #"shuffle trials to break any dependencies on the sequence of trails"....
                
                numObservations, numFeatures = X0.shape
                ## %%%%%% shuffle trials to break any dependencies on the sequence of trails 
                shfl = rng.permutation(np.arange(0, numObservations));
                Ys = Ysr[shfl]; # stim rates
                Xs = X0[shfl, :]; 
                ## %%%%% divide data to training and testin sets
                YTrain = Ys[np.arange(0, int((kfold-1.)/kfold*numObservations))]; # Take the first 90% of trials as training set
                YTest = Ys[np.arange(int((kfold-1.)/kfold*numObservations), numObservations)]; # Take the last 10% of trials as testing set
                
                XTrain = Xs[np.arange(0, int((kfold-1.)/kfold*numObservations)), :];
                XTest = Xs[np.arange(int((kfold-1.)/kfold*numObservations), numObservations), :];
                
                XTest_sr_all.append(XTest)
                XTrain_sr_all.append(XTrain)
                YTest_sr_all.append(YTest)
                YTrain_sr_all.append(YTrain)
                   
                   
                ##%% Choose n random trials (n = 80% of min of HR, LR in YTrain)
                   
            #    nrtr = int(np.ceil(.8 * min((YTrain==0).sum(), (YTrain==1).sum())))
                nrtr = int(np.ceil(.8*len(YTrain))) # we will subselect 80% of trials
                print 'selecting %d random trials of each category' %(nrtr)
                
                
                ##%% Train SVM on XTrain using BAGGING (subselection of trials on XTrain)
                
                if doData:
                    # data
                    w = np.full((XTrain.shape[1], nRandTrSamp), np.nan) # neurons x nrandtr
                    b =  np.full((nRandTrSamp), np.nan) # nrandtr
            #        trainE =  np.full((nRandTrSamp), np.nan) # nrandtr
                        
                    for ir in range(nRandTrSamp): # subselect a group of trials
                        print '\tBAGGING iteration %d' %(ir) # Subselect out of XTrain
                        
                        # subselect nrtr random trials 
                        randTrs = rng.permutation(len(YTrain))[0:nrtr].flatten() # random trials
                        
                        
                        X = XTrain[randTrs,:]
                        Y = YTrain[randTrs]
                    
                    
                        ##%% train SVM on each frame using the sebselect of trials        
            #            linear_svm = svm.LinearSVC(C=10.) # default c, also l2    # linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l1', dual=False)   
                        linear_svr = LinearSVR(C=cbeg, epsilon=0.0, dual = True, tol = 1e-9, fit_intercept = True)
                        
                        # data
                        linear_svr.fit(X, Y)
                            
                        w[:, ir] = np.squeeze(linear_svr.coef_);
                        b[ir] = linear_svr.intercept_;        
            #            trainE[ir] = abs(linear_svr.predict(X)-Y.astype('float')).sum()/len(Y)*100;
                
                    w_all.append(w)
                    b_all.append(b)
                    
                
            w_all = np.array(w_all) #numSamps x neurons x nRandTrSamp
            b_all = np.array(b_all) #numSamps x nRandTrSamp
            XTest_sr_all = np.array(XTest_sr_all) #numSamps x testingTrs x neurons
            XTrain_sr_all = np.array(XTrain_sr_all) #numSamps x trainingTrs x neurons
            YTest_sr_all = np.array(YTest_sr_all) #numSamps x testingTrs
            YTrain_sr_all = np.array(YTrain_sr_all) #numSamps x trainingTrs
            
            #yhat_corrTrain = predictMan(X, w[:,ir], b[ir], np.nan)
            #a = perClassError_sr(yhat_corrTrain, Y)
            
            #yhat_incorr = predictMan(X_incorr, w[:,ir], b[ir], np.nan)
            #classErr_incorrTest = abs(yhat_incorr - Ysr_incorr).mean() * 100
            
            #yhat0 = predictMan(XTest, w[:,ir], b[ir], np.nan)
            #classErr_corrTest = abs(yhat0 - YTest).mean() * 100
                
            #%% Now find the final BAGGED decoder for each round of training/testing trial subselection you performed above. And compute class error on different datasets (testing, training, incorrect, shuffled incorrect)
            
            classErr_corrTrain_all = np.full((numSamples), np.nan)
            classErr_corrTrainTest_all = np.full((numSamples), np.nan)
            classErr_corrTest_all = np.full((numSamples), np.nan)
            classErr_incorrTest_all = np.full((numSamples), np.nan)
            classErr_incorrTest_sh_all = []
            classErr_incorrTest_chance_all = []
            classErr_corrTest_sh_all = []
            classErr_corrTest_chance_all = []
            
            aa = Ysr.min() # -.5  
            bb = Ysr.max() # .5   
             
               
            for i in range(numSamples):
                print '--------- iter ', i, ' ---------'
                w = w_all[i,:,:]
                b = b_all[i,:]
                
                XTest = XTest_sr_all[i,:,:]
                XTrain = XTrain_sr_all[i,:,:]
                YTest = YTest_sr_all[i,:]
                YTrain = YTrain_sr_all[i,:]
                    
                ##%% Normalize weights (ie the w vector at each frame must have length 1)
                
                normw = np.linalg.norm(w, axis=0) # nrandtr; 2-norm of weights 
                w_normed = w/normw # neurons x nrandtr; normalize weights so weights (of each frame) have length 1
                if sum(normw<=eps).sum()!=0:
                    print 'take care of this; you need to reshape w_normed first'
                                #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
                
                
                ##%% Set the final decoder by aggregating decoders across trial subselects (ie average ws across all trial subselects), then again normalize them so they have length 1 (Bagging)
                
                w_nb = np.mean(w_normed, axis=(1)) # neurons 
                nw = np.linalg.norm(w_nb, axis=0) # scalar; 2-norm of weights 
                w_bag = w_nb/nw # neurons
                # not sure what to do with the intercept term in the BAGGED method... lets go with the average of b across trials subselects for now:
                b_bag = b.mean()
                
                # the following method wont work for svr, bc we dont have only two gaussians for 2 choices....
                # matt's method to find bagged intercept ... classErrs you get using this b_bag are very close to the b_bag (averaged b) you computed above...
                '''
                x = XTrain#XTest #X0
                y = YTrain#YTest #Y0
                projs = np.dot(x, w_bag) # not sure if you should use Xl or XTest
                hrp = projs[y==1]    
                lrp = projs[y==0]    
                mu0 = hrp.mean()
                mu1 = lrp.mean()
                sd0 = hrp.std()
                sd1 = lrp.std()
                thresh = (sd0 * mu1 + sd1 * mu0) / (sd1 + sd0)
                b_bag = thresh
                '''
            
                ##%% Now make predictions on different datasets using the BAGGED decodcer
              
                
                # training data
                yhat_corrTrain = predictMan(XTrain, w_bag, b_bag, np.nan)
                a = perClassError_sr(yhat_corrTrain, YTrain)
                print a, '= predict error on training dataset (correct trials)'
                classErr_corrTrain_all[i] = a
                
                
                # all correct data
                yhat_corrAll = predictMan(X0, w_bag, b_bag, np.nan)
                a = perClassError_sr(yhat_corrAll, Ysr)
                print np.round(a, 2), '= predict error on all correct trials'
                classErr_corrTrainTest_all[i] = a
                
                
                # testing correct
                yhat0 = predictMan(XTest, w_bag, b_bag, np.nan)
                #perClassError(YTest, yhat0)
                classErr_corrTest = perClassError_sr(yhat0, YTest)
                print np.round(classErr_corrTest, 2), '= predict error on correct trials (testing dataset)'
                classErr_corrTest_all[i] = classErr_corrTest
                
            
                # shuffled correct: compare yhat0 with shuffled labels
                classErr_corrTest_sh = []
                classErr_corrTest_chance = []
                for ii in range(numSamples):
                    # set y for chance_corr and chance_incorr ... this is random numbers spanning the same range as y ... we prefer this over shuffling y for the null dist
                    # bc in case y doesn't have much variability then shuffling it wont give us a very different null dist than original dist.        
                    a_corr = (bb-aa) * np.random.random(YTest.shape[0]) + aa # Y0
                    # chance trial labels
                    classErr_corrTest_chance.append(perClassError_sr(yhat0, a_corr))
                    
                    # just shuffle Y_incorr
                    permIx = rng.permutation(YTest.shape[0])
                    classErr_corrTest_sh.append(perClassError_sr(yhat0, YTest[permIx]))    
                    #print np.round(classErr_incorrTest_chance, 2), '= predict error on incorrect trials'
                
                classErr_corrTest_sh = np.array(classErr_corrTest_sh)
                classErr_corrTest_chance = np.array(classErr_corrTest_chance)    
                
                classErr_corrTest_sh_all.append(classErr_corrTest_sh)
                classErr_corrTest_chance_all.append(classErr_corrTest_chance)
                
                print np.round(classErr_corrTest_chance.mean(), 2), '= predict error on correct trials with random labels (equal number for the 2 choices)'
                print np.round(classErr_corrTest_sh.mean(), 2), '= predict error on correct trials with shuffled labels'
            #    if (YTest==0).sum() > (YTest==1).sum():
            #        print np.round(50 - (classErr_corrTest * (1 - .5/(YTest==0).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
            #    elif (Y_incorr==0).sum() < (Y_incorr==1).sum():
            #        print np.round(50 - (classErr_corrTest * (1 - .5/(YTest==1).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
            
            
            
                # incorrect trials
                yhat_incorr = predictMan(X_incorr, w_bag, b_bag, np.nan)
                classErr_incorrTest = perClassError_sr(yhat_incorr, Ysr_incorr)
                print np.round(classErr_incorrTest, 2), '= predict error on incorrect trials'
                classErr_incorrTest_all[i] = classErr_incorrTest
                
                
                # shuffled incorrect: compare yhat_incorr with shuffled labels
                classErr_incorrTest_sh = []
                classErr_incorrTest_chance = []
                for iii in range(numSamples):
                    # set y for chance_corr and chance_incorr ... this is random numbers spanning the same range as y ... we prefer this over shuffling y for the null dist
                    # bc in case y doesn't have much variability then shuffling it wont give us a very different null dist than original dist.
                    a_incorr = (bb-aa) * np.random.random(Y_incorr.shape[0]) + aa
                    
                    # chance trial labels
                    classErr_incorrTest_chance.append(perClassError_sr(yhat_incorr, a_incorr))
                    
                    # just shuffle Y_incorr
                    permIx = rng.permutation(Y_incorr.shape[0])
                    classErr_incorrTest_sh.append(perClassError_sr(yhat_incorr, Ysr_incorr[permIx]))
                    #print np.round(classErr_incorrTest_chance, 2), '= predict error on incorrect trials'
                
                classErr_incorrTest_sh = np.array(classErr_incorrTest_sh)
                classErr_incorrTest_chance = np.array(classErr_incorrTest_chance)    
                
                classErr_incorrTest_sh_all.append(classErr_incorrTest_sh)
                classErr_incorrTest_chance_all.append(classErr_incorrTest_chance)
                
                print np.round(classErr_incorrTest_chance.mean(), 2), '= predict error on incorrect trials with random labels (equal number for the 2 choices)'
                print np.round(classErr_incorrTest_sh.mean(), 2), '= predict error on incorrect trials with shuffled labels'
            #    if (Y_incorr==0).sum() > (Y_incorr==1).sum():
            #        print np.round(50 - (classErr_incorrTest * (1 - .5/(Y_incorr==0).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
            #    elif (Y_incorr==0).sum() < (Y_incorr==1).sum():
            #        print np.round(50 - (classErr_incorrTest * (1 - .5/(Y_incorr==1).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
            
            
            classErr_corrTest_sh_all = np.array(classErr_corrTest_sh_all)
            classErr_corrTest_chance_all = np.array(classErr_corrTest_chance_all)
            classErr_incorrTest_sh_all = np.array(classErr_incorrTest_sh_all)
            classErr_incorrTest_chance_all = np.array(classErr_incorrTest_chance_all)
            
            print '---------------Final Results---------------'
            print '---------------BAGGING SVR---------------'
            print 'Testing correct: ', classErr_corrTest_all.mean()
            print 'Incorrect: ', classErr_incorrTest_all.mean()
            print 'Training: ', classErr_corrTrain_all.mean()
            print 'Train and test: ', classErr_corrTrainTest_all.mean()
            print 'Testing correct, chance: ', classErr_corrTest_chance_all.mean()
            print 'Testing correct, shfl: ', classErr_corrTest_sh_all.mean()
            print 'Incorrect, chance: ', classErr_incorrTest_chance_all.mean()
            print 'Incorrect, shfl: ', classErr_incorrTest_sh_all.mean()
            
            
            
            #%% Below is why I am using a faked shuffled data (classErr_incorrTest_chance) which has equal number of trials from each class to set the null distribution:
            '''
            # my own understanding for why and how much classErr_incorrTest_sh may deviate from 50:
            # if there is an imbalance is hr vs lr, then y_shfl will ratain same labels as y
            # for a fraction of trials. Therefore classErr_shfl will be biased to classErr by a degree
            # that is determined by the fraction of trials which retain the same label.
            # We can quanity this in the following way:
            50 - (classErr_incorrTest * (1 - .5/(Y_incorr==0).mean()))
            # lets break it down:
            a = (1 - .5/(Y_incorr==0).mean()) # a is the degree by which the original labels are retained in the shuffled data... if fully shuffled it will be 0.
            b = classErr_incorrTest * a # b is the degree by which classErr will bias the chance performance
            c = 50 - b # c is the final chance performance
            '''
            
            
            #%% Lets check how svm.predict and svr.predict work.... Result: for both SVM and SVR, classfier.predict equals xw+b. For SVM, xw+b gives -1 and 1, that should be changed to 0 and 1 to match svm.predict.
            '''
            # check SVR prediction
            # from Gamal:
            #X features matrix (ie matrix of neural data of size trials x neurons);
            #y observations array (ie array of stimulus rate of size trials)
            from sklearn.svm import LinearSVR
            svr = LinearSVR(C=10., epsilon=0.0, dual = True, tol = 1e-9, fit_intercept = True)
            svr.fit(X,Y)
            w = svr.coef_; # weights vector
            b = svr.intercept_; # bias term
            print abs(np.dot(X, w.T)+b - svr.predict(X)).sum() # this should be 0
            
            
            # check SVM prediction
            #from sklearn.svm import LinearSVR
            linear_svm = svm.LinearSVC(C=10.)
            linear_svm.fit(X,Y)
            w = linear_svm.coef_.squeeze() # weights vector
            b = linear_svm.intercept_.squeeze() # bias term
            yhat0 = np.dot(X, w.T)+b # it gives -1 and 1... we need to compare it to threshold (0):
            yhat0[yhat0<0] = 0
            yhat0[yhat0>0] = 1
            print abs(yhat0 - linear_svm.predict(X)).sum() # this should be 0
            '''
            
            #%%
            ################ Compare BAGGED decoder and L1/L2 decoder ##############
            
            #%% Compute angle between optimal L1 decoders (of all numSamps); and between BAGGED decoder and L1 decoder... we want to see how similar they are
            
            # Normalize w_bestc_data
            nw = np.linalg.norm(w_bestc_data, axis=1) # numSamps; 2-norm of weights 
            w_data_n = (w_bestc_data.T/nw).T # numSamps x neurons
            
            # average (across numSamps) decoders at best c
            wNow = np.mean(w_data_n, axis=0) # neurons
            # again normalize it
            nw = np.linalg.norm(wNow) # numSamps; 2-norm of weights 
            wNow_n = wNow/nw # neurons
            
            
            # train svm on all data using best c
            linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty=regType, dual=False)
            linear_svm.fit(X0, Y0)   
            w = np.squeeze(linear_svm.coef_);
            b = linear_svm.intercept_;
            #trainE = abs(linear_svm.predict(X)-Y.astype('float')).sum()/len(Y)*100;
            
            # normalize w
            nw = np.linalg.norm(w) # numSamps; 2-norm of weights 
            w_n = w/nw # neurons
            
            
            # compute angle between the averaged across samples decoder and the one trained using all data (without testing and training)
            print np.arccos(abs(np.dot(wNow_n, w_n.transpose())))*180/np.pi
            
            #s = 20
            #np.arccos(abs(np.dot(w_data_n[s,:].squeeze(), w_n.transpose())))*180/np.pi
            
            
            # angle between bestc decoders of different shuffles
            ang_samps = np.full((numSamples,numSamples), np.nan)
            for s in range(numSamples):
                for ss in range(numSamples):
                    ang_samps[s,ss] = np.arccos(abs(np.dot(w_data_n[s,:].squeeze(), w_data_n[ss,:].squeeze())))*180/np.pi
            
            # average angle between decoders of different samples
            # first set diag elements to nan (because we know their angle is 0)
            np.fill_diagonal(ang_samps, np.nan)
            
            plt.figure()
            plt.imshow(ang_samps, cmap='jet_r')
            plt.colorbar()
            
            print 'Average angle btwn optimal decoders of all shuffles = ', np.nanmean(ang_samps)
            
            
            ########## now compare angle between bagged decoder and l1 optimal decoder
            print np.arccos(abs(np.dot(w_n, w_bag)))*180/np.pi
            
            # with average l1 decoder
            print np.arccos(abs(np.dot(wNow_n, w_bag)))*180/np.pi
            
            # with each l1 decoder
            angs = []
            for s in range(numSamples):
                angs.append(np.arccos(abs(np.dot(w_data_n[s,:].squeeze(), w_bag)))*180/np.pi)
            
            angs = np.array(angs)
            plt.figure()
            plt.plot(angs)
            print 'Angle between BAGGED decoder and optimal L1 decoder = ', angs.mean()
            
            
            #%% predictions on incorr and testing corr, using l1 decoder vs bagged decoder
            
            # bagged decoder
            print 'BAGGED'
            # testing corr
            print 'train: ', np.round(classErr_corrTrain_all.mean(), 2)
            print 'corr: ', np.round(classErr_corrTest_all.mean(), 2)
            print 'incorr: ', np.round(classErr_incorrTest_all.mean(), 2), '\n'
            
            
            # L1/L2 decocer
            print regType,' decocer'
            print 'train: ', np.round(classErr_bestC_train_data.mean(), 2)
            print 'corr: ', np.round(classErr_bestC_test_data.mean(), 2)
            print 'incorr: ', np.round(classErr_bestC_incorr_data.mean(), 2), '\n'
            '''
            # decoder found using all data (without doing testing and training)
            yhat0 = predictMan(X_incorr, w_n, b)
            print perClassError(yhat0, Y_incorr)
            
            # averaged decoder across samples
            yhat0 = predictMan(X_incorr, wNow_n, b_bestc_data.mean())
            print perClassError(yhat0, Y_incorr)
            '''
            print regType,' decocer', '(bestc found separately for incorr and testing corr)'
            # min err on testing corr for Bagged decoders (with any c valur larger than bestc)
            print 'corr: ', np.round(np.mean(perClassErrorTest[:,np.argwhere(indBestC).squeeze():], axis=0).min(), 2)
            # min err on incorr for Bagged decoders (with any c valur larger than bestc)
            print 'incorr: ', np.round(np.mean(perClassErrorTest_incorr[:,np.argwhere(np.in1d(cvect, cbest_incorr)).squeeze():], axis=0).min(), 2), '\n'
        
        
        
        #%%
        ####################################################################################################################################
        ############################################ Save results as .mat files in a folder named svm ####################################################################################################################################
        ####################################################################################################################################
        
        #%% Save SVR results
        
        if trialHistAnalysis:
            svmn = 'svrPrevChoice_corrIncorr_stAl_%s_' %(nowStr)
        else:
            svmn = 'svrCurrChoice_corrIncorr_stAl_%s_' %(nowStr)
        print '\n', svmn[:-1]
        
        if saveResults:
            print 'Saving .mat file'
            d = os.path.join(os.path.dirname(pnevFileName), 'svm')
            if not os.path.exists(d):
                print 'creating svm folder'
                os.makedirs(d)
        
            svmName = os.path.join(d, svmn+os.path.basename(pnevFileName))
            print(svmName)
            
            if doBagging==0:
                scio.savemat(svmName, {'thAct':thAct, 'thTrsWithSpike':thTrsWithSpike, 'ep_ms':ep_ms, 'ep_ch':ep_ch,
                                       'th_stim_dur':th_stim_dur, 'numSamples':numSamples, 'nRandTrSamp':nRandTrSamp,
                                       'trsExcluded':trsExcluded, 'trsExcluded_chAl':trsExcluded_chAl, 'trsExcluded_incorr':trsExcluded_incorr, 'trsExcluded_incorr_chAl':trsExcluded_incorr_chAl,
                                       'NsExcluded':NsExcluded, 'NsExcluded_chAl':NsExcluded_chAl,
                                       'regType':regType, 'cvect':cvect,
                                       'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,'cbest_incorr':cbest_incorr,
                                       'wAllC':wAllC,
                                       'bAllC':bAllC,
                                       'perClassErrorTrain':perClassErrorTrain,
                                       'perClassErrorTest':perClassErrorTest,
                                       'perClassErrorTest_shfl':perClassErrorTest_shfl,
                                       'perClassErrorTest_chance':perClassErrorTest_chance,
                                       'perClassErrorTest_incorr':perClassErrorTest_incorr,
                                       'perClassErrorTest_incorr_shfl':perClassErrorTest_incorr_shfl,
                                       'perClassErrorTest_incorr_chance':perClassErrorTest_incorr_chance}) 
            else:
                scio.savemat(svmName, {'thAct':thAct, 'thTrsWithSpike':thTrsWithSpike, 'ep_ms':ep_ms, 'ep_ch':ep_ch,
                                       'th_stim_dur':th_stim_dur, 'numSamples':numSamples, 'nRandTrSamp':nRandTrSamp,
                                       'trsExcluded':trsExcluded, 'trsExcluded_chAl':trsExcluded_chAl, 'trsExcluded_incorr':trsExcluded_incorr, 'trsExcluded_incorr_chAl':trsExcluded_incorr_chAl,
                                       'NsExcluded':NsExcluded, 'NsExcluded_chAl':NsExcluded_chAl,
                                       'regType':regType, 'cvect':cvect,
                                       'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,'cbest_incorr':cbest_incorr,
                                       'wAllC':wAllC,
                                       'bAllC':bAllC,
                                       'perClassErrorTrain':perClassErrorTrain,
                                       'perClassErrorTest':perClassErrorTest,
                                       'perClassErrorTest_shfl':perClassErrorTest_shfl,
                                       'perClassErrorTest_chance':perClassErrorTest_chance,
                                       'perClassErrorTest_incorr':perClassErrorTest_incorr,
                                       'perClassErrorTest_incorr_shfl':perClassErrorTest_incorr_shfl,
                                       'perClassErrorTest_incorr_chance':perClassErrorTest_incorr_chance,
                                        'cbeg':cbeg, 'w_all':w_all, 'b_all':b_all, # starting from here are BAGGING vars.
                                        'classErr_corrTest_all':classErr_corrTest_all,
                                        'classErr_incorrTest_all':classErr_incorrTest_all,
                                        'classErr_corrTrain_all':classErr_corrTrain_all,
                                        'classErr_corrTrainTest_all':classErr_corrTrainTest_all,
                                        'classErr_corrTest_chance_all':classErr_corrTest_chance_all,
                                        'classErr_corrTest_sh_all':classErr_corrTest_sh_all,
                                        'classErr_incorrTest_chance_all':classErr_incorrTest_chance_all,
                                        'classErr_incorrTest_sh_all':classErr_incorrTest_sh_all})         
                
            
        else:
            print 'Not saving .mat file'
        










    #%% SVR - stimulus-aligned - epAllStim: use stim-aligned traces that are averaged during stim presentation
    #################################################################################
    #################################################################################
    #################################################################################
    #################################################################################        
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    #%%
    ####################################################################################
    ########## Training SVR to decode stimulus on stimulus-aligned traces ##########
    ####################################################################################
    
    #%%     
    ##############################   SVR; L1/L2 penalty; epAllStim  ##################################
    ###############################################################################   
if stimAligned:
    
    if doSVR==1 and epAllStim==1:    
        
        from sklearn.svm import LinearSVR
        
        #%% Find the optimal regularization parameter:
        # Rrain SVM on a range of c, using training and testing dataset ... 
        #    make class predictions for testing dataset and incorrect trials. 
        #    Also shuffle labels for correct and incorrect trials and make predictions (to set null distributions)
        # Repeat the above procedure for numSamples times.
        
        
        # numSamples = 10; # number of iterations for finding the best c (inverse of regularization parameter)
        # if you don't want to regularize, go with a very high cbest and don't run the section below.
        # cbest = 10**6
        
        
        # important vars below:
        #perClassErrorTest : classErr on testing correct
        #perClassErrorTest_shfl : classErr on shuffled testing correct
        #perClassErrorTest_incorr : classErr on testing incorrect
        #perClassErrorTest_incorr_chance : classErr on shuffled testing incorrect
        
        
        
        #cvect = [eps,10**-6,10**-1,10**3]
        cvect = 10**(np.arange(-6, 6,0.2))/X_svr.shape[0]
        print 'try the following regularization values: \n', cvect
        # formattedList = ['%.2f' % member for member in cvect]
        # print 'try the following regularization values = \n', formattedList
        
        wAllC = np.ones((numSamples, len(cvect), X_svr.shape[1]))+np.nan;
        bAllC = np.ones((numSamples, len(cvect)))+np.nan;
        
        perClassErrorTrain = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_incorr = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_incorr_shfl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_incorr_chance = np.ones((numSamples, len(cvect)))+np.nan;
        #perClassErrorTrain_shfl = np.ones((numSamples, len(cvect)))+np.nan;
        #perClassErrorTest_shfl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_shfl = np.ones((numSamples, len(cvect)))+np.nan;
        perClassErrorTest_chance = np.ones((numSamples, len(cvect)))+np.nan;
        
        #perClassErrorTest_all0 = np.ones((numSamples, len(cvect)))+np.nan
        
        no = X_svr.shape[0]
        len_test = no - int((kfold-1.)/kfold*no)
        
        for s in range(numSamples):
            print 'iteration %d' %(s)
            permIxs_incorr = rng.permutation(X_svr_incorr.shape[0]);
        #    permIxs = rng.permutation(X_svr.shape[0]);
            permIxs = rng.permutation(len_test)
        
            # set y for chance_corr and chance_incorr ... this is random numbers spanning the same range as y ... we prefer this over shuffling y for the null dist
            # bc in case y doesn't have much variability then shuffling it wont give us a very different null dist than original dist.
            aa = Ysr.min() 
            bb = Ysr.max()
            a_incorr = (bb-aa) * np.random.random(Y_incorr.shape[0]) + aa
        #    a_corr = (bb-aa) * np.random.random(Y0.shape[0]) + aa
            a_corr = (bb-aa) * np.random.random(len_test) + aa
            
            
            for i in range(len(cvect)):
                    
        #        summary =  linearSVR(X0, Ysr, X_svr, Ysr, l=10)
                summary =  crossValidateModel(X_svr, Ysr, linearSVR, kfold = kfold, l = cvect[i])
        #        summary =  crossValidateModel(X_svr_incorr, Ysr_incorr, linearSVR, kfold = kfold, l = cvect[i]) # train on incorr trials.... I cannot trust it bc there are few trials.
                
                wAllC[s,i,:] = np.squeeze(summary.model.coef_); # weights of all neurons for each value of c and each shuffle
                bAllC[s,i] = np.squeeze(summary.model.intercept_);
        
                # classification errors                    
                perClassErrorTrain[s, i] = summary.perClassErrorTrain;
                perClassErrorTest[s, i] = summary.perClassErrorTest;        
                
                # Testing correct shuffled data: 
                # same decoder trained on correct trials, make predictions on correct with shuffled labels.
                perClassErrorTest_shfl[s, i] = perClassError_sr(summary.YTest[permIxs], summary.model.predict(summary.XTest));
                perClassErrorTest_chance[s, i] = perClassError_sr(a_corr, summary.model.predict(summary.XTest));        
                # all correct shuffled
        #        perClassErrorTest_shfl[s, i] = perClassError_sr(Ysr[permIxs], summary.model.predict(X_svr));
        #        perClassErrorTest_chance[s, i] = perClassError_sr(a_corr, summary.model.predict(X_svr));        
        
                # this time refit the classifier on correct trials using shuffled labels (and doing testing, training datasets)
                '''
        #        summary_shfl = crossValidateModel(X_svr_incorr, Y_incorr[permIxs], linearSVM, kfold = kfold, l1 = cvect[i])
                summary_shfl = crossValidateModel(X_svr, Y0[permIxs], linearSVM, kfold = kfold, l1 = cvect[i])
                perClassErrorTrain_shfl[s, i] = summary_shfl.perClassErrorTrain;
                perClassErrorTest_shfl[s, i] = summary_shfl.perClassErrorTest;
                '''
        
                # Incorrect trials (using the decoder trained on correct trials)
                #linear_svm.fit(XTrain, np.squeeze(YTrain))    # # linear_svm is trained using Xtrain and Ytrain... #train using Xtrain_corr... test on Xtest_corr and X_svr_incorr.
                #summary.model = linear_svm;
                perClassErrorTest_incorr[s, i] = perClassError_sr(Ysr_incorr, summary.model.predict(X_svr_incorr));
                
                # Null distribution for class error on incorr trials: 
                # We want to use the same decoder (trained on correct trials) to predict the choice on incorrect trials with shuffled class labels.             
                perClassErrorTest_incorr_shfl[s, i] = perClassError_sr(Ysr_incorr[permIxs_incorr], summary.model.predict(X_svr_incorr));
                perClassErrorTest_incorr_chance[s, i] = perClassError_sr(a_incorr, summary.model.predict(X_svr_incorr)); # half of the trials assigned randomly to one choice and the other half to the other choice
        
        
                # control: use an all-0 decoder and see what classErr u get
        #        yhat0 = predictMan(summary.XTest, np.zeros((1,summary.model.coef_.shape[1])).squeeze(), bAllC[s,i])
        ##        yhat0 = predictMan(summary.XTest, 10**-5*np.ones((1,summary.model.coef_.shape[1])).squeeze(), bAllC[s,i])
        #        perClassError(YTest, yhat0)
        #        perClassErrorTest_all0[s, i] = perClassError_sr(yhat0, summary.YTest)
        
        
        #plt.plot(np.mean(perClassErrorTest_all0, axis=0))
        #plt.plot(np.mean(perClassErrorTest, axis=0))
        
        
        #%% Compute average of class errors across numSamples
        
        meanPerClassErrorTest_corr_shfl = np.mean(perClassErrorTest_shfl, axis = 0);
        semPerClassErrorTest_corr_shfl = np.std(perClassErrorTest_shfl, axis = 0)/np.sqrt(numSamples);
        
        #meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl, axis = 0);
        #semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl, axis = 0)/np.sqrt(numSamples);
        
        #meanPerClassErrorTrain_shfl = np.mean(perClassErrorTrain_shfl, axis = 0);
        #semPerClassErrorTrain_shfl = np.std(perClassErrorTrain_shfl, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_incorr = np.mean(perClassErrorTest_incorr, axis = 0);
        semPerClassErrorTest_incorr = np.std(perClassErrorTest_incorr, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_incorr_shfl = np.mean(perClassErrorTest_incorr_shfl, axis = 0);
        semPerClassErrorTest_incorr_shfl = np.std(perClassErrorTest_incorr_shfl, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_incorr_chance = np.mean(perClassErrorTest_incorr_chance, axis = 0);
        semPerClassErrorTest_incorr_chance = np.std(perClassErrorTest_incorr_chance, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTrain = np.mean(perClassErrorTrain, axis = 0);
        semPerClassErrorTrain = np.std(perClassErrorTrain, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest = np.mean(perClassErrorTest, axis = 0);
        semPerClassErrorTest = np.std(perClassErrorTest, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance, axis = 0);
        semPerClassErrorTest_chance = np.std(perClassErrorTest_chance, axis = 0)/np.sqrt(numSamples);
        
        
        
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
        
        
        eps = 10**-10 # tiny number below which weight is considered 0
        
        # Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)
        a = abs(wAllC)>eps # non-zero weights
        #a = (wAllC!=0) # non-zero weights
        b = np.mean(a, axis=(0,2)) # Fraction of non-zero weights (averaged across shuffles)
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
        
        
        # find bestc based on incorr trial class error
        meanPerClassErrorTestnow = np.mean(perClassErrorTest_incorr[:,c1stnon0:], axis = 0);
        semPerClassErrorTestnow = np.std(perClassErrorTest_incorr[:,c1stnon0:], axis = 0)/np.sqrt(numSamples);
        ix = np.argmin(meanPerClassErrorTestnow)
        if smallestC==1:    
            cbest_incorr = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
            cbest_incorr = cbest_incorr[0]; # best regularization term based on minError+SE criteria
        else:   
            cbest_incorr = cvectnow[ix]
            
        print 'incorr: best c (at least 1 non-0 weight) = ', cbest_incorr
        
        
        
        #%% Set the decoder and class errors at best c (for data)
        
        # you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
        # we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
        indBestC = np.in1d(cvect, cbest)
        
        w_bestc_data = wAllC[:,indBestC,:].squeeze() # numSamps x neurons
        b_bestc_data = bAllC[:,indBestC]
        
        classErr_bestC_test_data = perClassErrorTest[:,indBestC].squeeze()
        classErr_bestC_test_shfl = perClassErrorTest_shfl[:,indBestC].squeeze()
        classErr_bestC_incorr_data = perClassErrorTest_incorr[:,indBestC].squeeze()
        classErr_bestC_incorr_chance = perClassErrorTest_incorr_chance[:,indBestC].squeeze()
        classErr_bestC_chance = perClassErrorTest_chance[:,indBestC].squeeze()
        classErr_bestC_train_data = perClassErrorTrain[:,indBestC].squeeze()
        #perClassErrorTest_shfl_incorr = perClassErrorTest_incorr_shfl[:,indBestC].squeeze()
        
        
        #%% plot C path
               
        if doPlots:
            print 'Best c (inverse of regularization parameter) = %.2f' %cbest
            plt.figure('cross validation')
            plt.subplot(1,2,1)
            plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
            plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
            plt.fill_between(cvect, meanPerClassErrorTest_incorr-semPerClassErrorTest_incorr, meanPerClassErrorTest_incorr+ semPerClassErrorTest_incorr, alpha=0.5, edgecolor='g', facecolor='g')        
            plt.fill_between(cvect, meanPerClassErrorTest_incorr_chance-semPerClassErrorTest_incorr_chance, meanPerClassErrorTest_incorr_chance+ semPerClassErrorTest_incorr_chance, alpha=0.5, edgecolor='m', facecolor='m')        
            plt.fill_between(cvect, meanPerClassErrorTest_incorr_shfl-semPerClassErrorTest_incorr_shfl, meanPerClassErrorTest_incorr_shfl+ semPerClassErrorTest_incorr_shfl, alpha=0.5, edgecolor='c', facecolor='c')        
            plt.fill_between(cvect, meanPerClassErrorTest_corr_shfl-semPerClassErrorTest_corr_shfl, meanPerClassErrorTest_corr_shfl+ semPerClassErrorTest_corr_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
            plt.fill_between(cvect, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
            plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
            plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation')
            plt.plot(cvect, meanPerClassErrorTest_incorr, 'g', label = 'incorr')        
            plt.plot(cvect, meanPerClassErrorTest_incorr_chance, 'm', label = 'incorr-chance')            
            plt.plot(cvect, meanPerClassErrorTest_incorr_shfl, 'c', label = 'incorr-shfl')            
            plt.plot(cvect, meanPerClassErrorTest_corr_shfl, 'y', label = 'corr-shfl')            
            plt.plot(cvect, meanPerClassErrorTest_chance, 'b', label = 'corr-chance')        
            plt.plot(cvect[cvect==cbest], meanPerClassErrorTest[cvect==cbest], 'bo')
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('prediction error (normalized sum of square error)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            plt.tight_layout()
        #    plt.ylim([-10,110])
        
        '''
        if doPlots:
            print 'Best c (inverse of regularization parameter) = %.2f' %cbest
            plt.figure('cross validation')
            plt.subplot(1,2,1)
        #    plt.fill_between(cvect, meanPerClassErrorTrain_shfl-semPerClassErrorTrain_shfl, meanPerClassErrorTrain_shfl+ semPerClassErrorTrain_shfl, alpha=0.5, edgecolor='k', facecolor='k')
        #    plt.fill_between(cvect, meanPerClassErrorTest_corr_shfl-semPerClassErrorTest_corr_shfl, meanPerClassErrorTest_corr_shfl+ semPerClassErrorTest_corr_shfl, alpha=0.5, edgecolor='r', facecolor='r')
            plt.fill_between(cvect, meanPerClassErrorTest_incorr_shfl-semPerClassErrorTest_incorr_shfl, meanPerClassErrorTest_incorr_shfl+ semPerClassErrorTest_incorr_shfl, alpha=0.5, edgecolor='c', facecolor='c')        
        
        #    plt.plot(cvect, meanPerClassErrorTrain_shfl, 'k', label = 'training')
        #    plt.plot(cvect, meanPerClassErrorTest_corr_shfl, 'r', label = 'validation')
            plt.plot(cvect, meanPerClassErrorTest_incorr_shfl, 'c', label = 'incorr-shfl')            
        
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('classification error (%)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            plt.tight_layout()
        '''
        '''
        if doPlots:
            print 'Best c (inverse of regularization parameter) = %.2f' %cbest
            plt.figure('cross validation')
            plt.subplot(1,2,1)
            plt.fill_between(cvect, meanPerClassErrorTrain_shfl-semPerClassErrorTrain_shfl, meanPerClassErrorTrain_shfl+ semPerClassErrorTrain_shfl, alpha=0.5, edgecolor='k', facecolor='k')
            plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='r', facecolor='r')
            
            plt.plot(cvect, meanPerClassErrorTrain_shfl, 'k', label = 'training')
            plt.plot(cvect, meanPerClassErrorTest_shfl, 'r', label = 'validation')
        
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('classification error (%)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            plt.tight_layout()
        '''
        
        
        #%% Plot class error for data and shuffled (for corr trained, corr cv, incorr cv)
        
        pvalueTest_corr = ttest2(classErr_bestC_test_shfl, classErr_bestC_test_data, tail = 'left');
        pvalueTest_incorr = ttest2(classErr_bestC_incorr_data, classErr_bestC_incorr_chance, tail = 'left');
        pvalueTest_incorr_corr = ttest2(classErr_bestC_incorr_data, classErr_bestC_test_shfl, tail = 'left');
        
        a = np.concatenate((classErr_bestC_test_shfl, classErr_bestC_test_data, classErr_bestC_incorr_data, classErr_bestC_incorr_chance, classErr_bestC_incorr_data, classErr_bestC_test_shfl))
        mn = a.min()
        mx = a.max()
        
        #print 'Training error: Mean actual: %.2f%%, Mean shuffled: %.2f%%, p-value = %.2f' %(np.mean(perClassErrorTrain_data), np.mean(perClassErrorTrain_shfl), pvalueTrain)
        print 'Testing error: Mean actual: %.2f, Mean shuffled: %.2f, p-value = %.2f' %(np.mean(classErr_bestC_test_data), np.mean(classErr_bestC_test_shfl), pvalueTest_corr)
        print 'Incorr trials: error: Mean actual: %.2f, Mean chance: %.2f, p-value = %.2f' %(np.mean(classErr_bestC_incorr_data), np.mean(classErr_bestC_incorr_chance), pvalueTest_incorr)
        
        
        # Plot the histograms
        if doPlots:
            binEvery = (mx-mn)/50# 3; # bin width
            
            plt.figure()
        #    plt.subplot(1,2,1)
        #    plt.hist(perClassErrorTrain_data, np.arange(0,100,binEvery), color = 'k', label = 'data');
        #    plt.hist(perClassErrorTrain_shfl, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
        #    plt.xlabel('Training classification error (%)')
        #    plt.ylabel('count')
        #    plt.title('Mean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(perClassErrorTrain_data), np.mean(perClassErrorTrain_shfl), pvalueTrain), fontsize = 10)
        #    plt.legend()
        
            plt.subplot(1,2,1)
            plt.hist(classErr_bestC_test_data, np.arange(mn,mx,binEvery), color = 'k', label = 'data');
            plt.hist(classErr_bestC_test_shfl, np.arange(mn,mx,binEvery), color = 'k', alpha=.5, label = 'shuffled');
            plt.legend()
            plt.xlabel('Testing prediction error (sum (y-yhat)^2 / sum y^2)')
            plt.title('Testing correct\nMean data: %.2f, Mean shuffled: %.2f\n p-value = %.2f' %(np.mean(classErr_bestC_test_data), np.mean(classErr_bestC_test_shfl), pvalueTest_corr), fontsize = 10)
            plt.ylabel('count')
            plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
        
        #    plt.figure()
            plt.subplot(1,2,2)
            plt.hist(classErr_bestC_incorr_data, np.arange(mn,mx,binEvery), color = 'k', label = 'data');
            plt.hist(classErr_bestC_incorr_chance, np.arange(mn,mx,binEvery), color = 'k', alpha=.5, label = 'shuffled');
            plt.legend()
            plt.xlabel('Incorrect prediction error (sum (y-yhat)^2 / sum y^2)')
            plt.title('Incorrect\nMean data: %.2f, Mean shuffled: %.2f\n p-value = %.2f' %(np.mean(classErr_bestC_incorr_data), np.mean(classErr_bestC_incorr_chance), pvalueTest_incorr), fontsize = 10)
            plt.ylabel('count')
            plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
        
        
            plt.figure()
            plt.subplot(1,2,1)
            plt.hist(classErr_bestC_test_data, np.arange(mn,mx,binEvery), color = 'k', label = 'testing corr');
            plt.hist(classErr_bestC_incorr_data, np.arange(mn,mx,binEvery), color = 'k', alpha=.5, label = 'incorr');
            plt.legend()
            plt.xlabel('prediction error (sum (y-yhat)^2 / sum y^2)')
            plt.title('Testing correct vs incorr\nMean corr: %.2f, Mean incorr: %.2f\n p-value = %.2f' %(np.mean(classErr_bestC_test_data), np.mean(classErr_bestC_incorr_data), pvalueTest_incorr_corr), fontsize = 10)
            plt.ylabel('count')
            plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
        
        
        
        
        
        #%% Compare angle between decoders at different values of c
        # just curious how decoders change during c path
        
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
            plt.plot(meanPerClassErrorTest_incorr, 'g', label = 'incorr')        
            plt.plot(meanPerClassErrorTest_incorr_shfl, 'c', label = 'incorr-shfl')            
            plt.plot(meanPerClassErrorTest_corr_shfl, 'y', label = 'corr-shfl')   
            #plt.axis('equal')
            #plt.gca().set_aspect('equal', adjustable='box')
        
        
        
        
        
        
        #%%     
        ###############################################################################    
        ##############################  SVR; BAGGING; epAllStim    ###############################
        ###############################################################################    
        
        if doBagging==1:
            cbeg = cbest+0 # 10
            
            #nRandTrSamp = 10 #nRandTrSamp = 10 #1000 # number of times to subselect trials (for the BAGGING method)
            #doData = 1
            
            w_all = []
            b_all = []
            XTest_sr_all = []
            XTrain_sr_all = []
            YTest_sr_all = []
            YTrain_sr_all = []
            
            for i in range(numSamples): # multiple iterations to choose training/testing datasets, so we have numSamples BAGGED decoders to test, so we have numSamps of testing class error.
                print 'Subselect XTrain and XTest, iteration %d' %(i)
                
                ##%% Set aside a group of trials as testing dataset and never use them for training the decoders (which will be later used for BAGGING)    
                #"shuffle trials to break any dependencies on the sequence of trails"....
                
                numObservations, numFeatures = X_svr.shape
                ## %%%%%% shuffle trials to break any dependencies on the sequence of trails 
                shfl = rng.permutation(np.arange(0, numObservations));
                Ys = Ysr[shfl]; # stim rates
                Xs = X_svr[shfl, :]; 
                ## %%%%% divide data to training and testin sets
                YTrain = Ys[np.arange(0, int((kfold-1.)/kfold*numObservations))]; # Take the first 90% of trials as training set
                YTest = Ys[np.arange(int((kfold-1.)/kfold*numObservations), numObservations)]; # Take the last 10% of trials as testing set
                
                XTrain = Xs[np.arange(0, int((kfold-1.)/kfold*numObservations)), :];
                XTest = Xs[np.arange(int((kfold-1.)/kfold*numObservations), numObservations), :];
                
                XTest_sr_all.append(XTest)
                XTrain_sr_all.append(XTrain)
                YTest_sr_all.append(YTest)
                YTrain_sr_all.append(YTrain)
                   
                   
                ##%% Choose n random trials (n = 80% of min of HR, LR in YTrain)
                   
            #    nrtr = int(np.ceil(.8 * min((YTrain==0).sum(), (YTrain==1).sum())))
                nrtr = int(np.ceil(.8*len(YTrain))) # we will subselect 80% of trials
                print 'selecting %d random trials of each category' %(nrtr)
                
                
                ##%% Train SVM on XTrain using BAGGING (subselection of trials on XTrain)
                
                if doData:
                    # data
                    w = np.full((XTrain.shape[1], nRandTrSamp), np.nan) # neurons x nrandtr
                    b =  np.full((nRandTrSamp), np.nan) # nrandtr
            #        trainE =  np.full((nRandTrSamp), np.nan) # nrandtr
                        
                    for ir in range(nRandTrSamp): # subselect a group of trials
                        print '\tBAGGING iteration %d' %(ir) # Subselect out of XTrain
                        
                        # subselect nrtr random trials 
                        randTrs = rng.permutation(len(YTrain))[0:nrtr].flatten() # random trials
                        
                        
                        X = XTrain[randTrs,:]
                        Y = YTrain[randTrs]
                    
                    
                        ##%% train SVM on each frame using the sebselect of trials        
            #            linear_svm = svm.LinearSVC(C=10.) # default c, also l2    # linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l1', dual=False)   
                        linear_svr = LinearSVR(C=cbeg, epsilon=0.0, dual = True, tol = 1e-9, fit_intercept = True)
                        
                        # data
                        linear_svr.fit(X, Y)
                            
                        w[:, ir] = np.squeeze(linear_svr.coef_);
                        b[ir] = linear_svr.intercept_;        
            #            trainE[ir] = abs(linear_svr.predict(X)-Y.astype('float')).sum()/len(Y)*100;
                
                    w_all.append(w)
                    b_all.append(b)
                    
                
            w_all = np.array(w_all) #numSamps x neurons x nRandTrSamp
            b_all = np.array(b_all) #numSamps x nRandTrSamp
            XTest_sr_all = np.array(XTest_sr_all) #numSamps x testingTrs x neurons
            XTrain_sr_all = np.array(XTrain_sr_all) #numSamps x trainingTrs x neurons
            YTest_sr_all = np.array(YTest_sr_all) #numSamps x testingTrs
            YTrain_sr_all = np.array(YTrain_sr_all) #numSamps x trainingTrs
            
            #yhat_corrTrain = predictMan(X, w[:,ir], b[ir], np.nan)
            #a = perClassError_sr(yhat_corrTrain, Y)
            
            #yhat_incorr = predictMan(X_svr_incorr, w[:,ir], b[ir], np.nan)
            #classErr_incorrTest = abs(yhat_incorr - Ysr_incorr).mean() * 100
            
            #yhat0 = predictMan(XTest, w[:,ir], b[ir], np.nan)
            #classErr_corrTest = abs(yhat0 - YTest).mean() * 100
                
            #%% Now find the final BAGGED decoder for each round of training/testing trial subselection you performed above. And compute class error on different datasets (testing, training, incorrect, shuffled incorrect)
            
            classErr_corrTrain_all = np.full((numSamples), np.nan)
            classErr_corrTrainTest_all = np.full((numSamples), np.nan)
            classErr_corrTest_all = np.full((numSamples), np.nan)
            classErr_incorrTest_all = np.full((numSamples), np.nan)
            classErr_incorrTest_sh_all = []
            classErr_incorrTest_chance_all = []
            classErr_corrTest_sh_all = []
            classErr_corrTest_chance_all = []
            
            aa = Ysr.min() # -.5  
            bb = Ysr.max() # .5   
             
               
            for i in range(numSamples):
                print '--------- iter ', i, ' ---------'
                w = w_all[i,:,:]
                b = b_all[i,:]
                
                XTest = XTest_sr_all[i,:,:]
                XTrain = XTrain_sr_all[i,:,:]
                YTest = YTest_sr_all[i,:]
                YTrain = YTrain_sr_all[i,:]
                    
                ##%% Normalize weights (ie the w vector at each frame must have length 1)
                
                normw = np.linalg.norm(w, axis=0) # nrandtr; 2-norm of weights 
                w_normed = w/normw # neurons x nrandtr; normalize weights so weights (of each frame) have length 1
                if sum(normw<=eps).sum()!=0:
                    print 'take care of this; you need to reshape w_normed first'
                                #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
                
                
                ##%% Set the final decoder by aggregating decoders across trial subselects (ie average ws across all trial subselects), then again normalize them so they have length 1 (Bagging)
                
                w_nb = np.mean(w_normed, axis=(1)) # neurons 
                nw = np.linalg.norm(w_nb, axis=0) # scalar; 2-norm of weights 
                w_bag = w_nb/nw # neurons
                # not sure what to do with the intercept term in the BAGGED method... lets go with the average of b across trials subselects for now:
                b_bag = b.mean()
                
                # the following method wont work for svr, bc we dont have only two gaussians for 2 choices....
                # matt's method to find bagged intercept ... classErrs you get using this b_bag are very close to the b_bag (averaged b) you computed above...
                '''
                x = XTrain#XTest #X_svr
                y = YTrain#YTest #Y0
                projs = np.dot(x, w_bag) # not sure if you should use Xl or XTest
                hrp = projs[y==1]    
                lrp = projs[y==0]    
                mu0 = hrp.mean()
                mu1 = lrp.mean()
                sd0 = hrp.std()
                sd1 = lrp.std()
                thresh = (sd0 * mu1 + sd1 * mu0) / (sd1 + sd0)
                b_bag = thresh
                '''
            
                ##%% Now make predictions on different datasets using the BAGGED decodcer
              
                
                # training data
                yhat_corrTrain = predictMan(XTrain, w_bag, b_bag, np.nan)
                a = perClassError_sr(yhat_corrTrain, YTrain)
                print a, '= predict error on training dataset (correct trials)'
                classErr_corrTrain_all[i] = a
                
                
                # all correct data
                yhat_corrAll = predictMan(X_svr, w_bag, b_bag, np.nan)
                a = perClassError_sr(yhat_corrAll, Ysr)
                print np.round(a, 2), '= predict error on all correct trials'
                classErr_corrTrainTest_all[i] = a
                
                
                # testing correct
                yhat0 = predictMan(XTest, w_bag, b_bag, np.nan)
                #perClassError(YTest, yhat0)
                classErr_corrTest = perClassError_sr(yhat0, YTest)
                print np.round(classErr_corrTest, 2), '= predict error on correct trials (testing dataset)'
                classErr_corrTest_all[i] = classErr_corrTest
                
            
                # shuffled correct: compare yhat0 with shuffled labels
                classErr_corrTest_sh = []
                classErr_corrTest_chance = []
                for ii in range(numSamples):
                    # set y for chance_corr and chance_incorr ... this is random numbers spanning the same range as y ... we prefer this over shuffling y for the null dist
                    # bc in case y doesn't have much variability then shuffling it wont give us a very different null dist than original dist.        
                    a_corr = (bb-aa) * np.random.random(YTest.shape[0]) + aa # Y0
                    # chance trial labels
                    classErr_corrTest_chance.append(perClassError_sr(yhat0, a_corr))
                    
                    # just shuffle Y_incorr
                    permIx = rng.permutation(YTest.shape[0])
                    classErr_corrTest_sh.append(perClassError_sr(yhat0, YTest[permIx]))    
                    #print np.round(classErr_incorrTest_chance, 2), '= predict error on incorrect trials'
                
                classErr_corrTest_sh = np.array(classErr_corrTest_sh)
                classErr_corrTest_chance = np.array(classErr_corrTest_chance)    
                
                classErr_corrTest_sh_all.append(classErr_corrTest_sh)
                classErr_corrTest_chance_all.append(classErr_corrTest_chance)
                
                print np.round(classErr_corrTest_chance.mean(), 2), '= predict error on correct trials with random labels (equal number for the 2 choices)'
                print np.round(classErr_corrTest_sh.mean(), 2), '= predict error on correct trials with shuffled labels'
            #    if (YTest==0).sum() > (YTest==1).sum():
            #        print np.round(50 - (classErr_corrTest * (1 - .5/(YTest==0).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
            #    elif (Y_incorr==0).sum() < (Y_incorr==1).sum():
            #        print np.round(50 - (classErr_corrTest * (1 - .5/(YTest==1).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
            
            
            
                # incorrect trials
                yhat_incorr = predictMan(X_svr_incorr, w_bag, b_bag, np.nan)
                classErr_incorrTest = perClassError_sr(yhat_incorr, Ysr_incorr)
                print np.round(classErr_incorrTest, 2), '= predict error on incorrect trials'
                classErr_incorrTest_all[i] = classErr_incorrTest
                
                
                # shuffled incorrect: compare yhat_incorr with shuffled labels
                classErr_incorrTest_sh = []
                classErr_incorrTest_chance = []
                for iii in range(numSamples):
                    # set y for chance_corr and chance_incorr ... this is random numbers spanning the same range as y ... we prefer this over shuffling y for the null dist
                    # bc in case y doesn't have much variability then shuffling it wont give us a very different null dist than original dist.
                    a_incorr = (bb-aa) * np.random.random(Y_incorr.shape[0]) + aa
                    
                    # chance trial labels
                    classErr_incorrTest_chance.append(perClassError_sr(yhat_incorr, a_incorr))
                    
                    # just shuffle Y_incorr
                    permIx = rng.permutation(Y_incorr.shape[0])
                    classErr_incorrTest_sh.append(perClassError_sr(yhat_incorr, Ysr_incorr[permIx]))
                    #print np.round(classErr_incorrTest_chance, 2), '= predict error on incorrect trials'
                
                classErr_incorrTest_sh = np.array(classErr_incorrTest_sh)
                classErr_incorrTest_chance = np.array(classErr_incorrTest_chance)    
                
                classErr_incorrTest_sh_all.append(classErr_incorrTest_sh)
                classErr_incorrTest_chance_all.append(classErr_incorrTest_chance)
                
                print np.round(classErr_incorrTest_chance.mean(), 2), '= predict error on incorrect trials with random labels (equal number for the 2 choices)'
                print np.round(classErr_incorrTest_sh.mean(), 2), '= predict error on incorrect trials with shuffled labels'
            #    if (Y_incorr==0).sum() > (Y_incorr==1).sum():
            #        print np.round(50 - (classErr_incorrTest * (1 - .5/(Y_incorr==0).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
            #    elif (Y_incorr==0).sum() < (Y_incorr==1).sum():
            #        print np.round(50 - (classErr_incorrTest * (1 - .5/(Y_incorr==1).mean())), 2), ' = my measure of class error on shuffled data, should be close to value above'
            
            
            classErr_corrTest_sh_all = np.array(classErr_corrTest_sh_all)
            classErr_corrTest_chance_all = np.array(classErr_corrTest_chance_all)
            classErr_incorrTest_sh_all = np.array(classErr_incorrTest_sh_all)
            classErr_incorrTest_chance_all = np.array(classErr_incorrTest_chance_all)
            
            print '---------------Final Results---------------'
            print '---------------BAGGING SVR---------------'
            print 'Testing correct: ', classErr_corrTest_all.mean()
            print 'Incorrect: ', classErr_incorrTest_all.mean()
            print 'Training: ', classErr_corrTrain_all.mean()
            print 'Train and test: ', classErr_corrTrainTest_all.mean()
            print 'Testing correct, chance: ', classErr_corrTest_chance_all.mean()
            print 'Testing correct, shfl: ', classErr_corrTest_sh_all.mean()
            print 'Incorrect, chance: ', classErr_incorrTest_chance_all.mean()
            print 'Incorrect, shfl: ', classErr_incorrTest_sh_all.mean()
            
            
            
            #%% Below is why I am using a faked shuffled data (classErr_incorrTest_chance) which has equal number of trials from each class to set the null distribution:
            '''
            # my own understanding for why and how much classErr_incorrTest_sh may deviate from 50:
            # if there is an imbalance is hr vs lr, then y_shfl will ratain same labels as y
            # for a fraction of trials. Therefore classErr_shfl will be biased to classErr by a degree
            # that is determined by the fraction of trials which retain the same label.
            # We can quanity this in the following way:
            50 - (classErr_incorrTest * (1 - .5/(Y_incorr==0).mean()))
            # lets break it down:
            a = (1 - .5/(Y_incorr==0).mean()) # a is the degree by which the original labels are retained in the shuffled data... if fully shuffled it will be 0.
            b = classErr_incorrTest * a # b is the degree by which classErr will bias the chance performance
            c = 50 - b # c is the final chance performance
            '''
            
            
            #%% Lets check how svm.predict and svr.predict work.... Result: for both SVM and SVR, classfier.predict equals xw+b. For SVM, xw+b gives -1 and 1, that should be changed to 0 and 1 to match svm.predict.
            '''
            # check SVR prediction
            # from Gamal:
            #X features matrix (ie matrix of neural data of size trials x neurons);
            #y observations array (ie array of stimulus rate of size trials)
            from sklearn.svm import LinearSVR
            svr = LinearSVR(C=10., epsilon=0.0, dual = True, tol = 1e-9, fit_intercept = True)
            svr.fit(X,Y)
            w = svr.coef_; # weights vector
            b = svr.intercept_; # bias term
            print abs(np.dot(X, w.T)+b - svr.predict(X)).sum() # this should be 0
            
            
            # check SVM prediction
            #from sklearn.svm import LinearSVR
            linear_svm = svm.LinearSVC(C=10.)
            linear_svm.fit(X,Y)
            w = linear_svm.coef_.squeeze() # weights vector
            b = linear_svm.intercept_.squeeze() # bias term
            yhat0 = np.dot(X, w.T)+b # it gives -1 and 1... we need to compare it to threshold (0):
            yhat0[yhat0<0] = 0
            yhat0[yhat0>0] = 1
            print abs(yhat0 - linear_svm.predict(X)).sum() # this should be 0
            '''
            
            #%%
            ################ Compare BAGGED decoder and L1/L2 decoder ##############
            
            #%% Compute angle between optimal L1 decoders (of all numSamps); and between BAGGED decoder and L1 decoder... we want to see how similar they are
            
            # Normalize w_bestc_data
            nw = np.linalg.norm(w_bestc_data, axis=1) # numSamps; 2-norm of weights 
            w_data_n = (w_bestc_data.T/nw).T # numSamps x neurons
            
            # average (across numSamps) decoders at best c
            wNow = np.mean(w_data_n, axis=0) # neurons
            # again normalize it
            nw = np.linalg.norm(wNow) # numSamps; 2-norm of weights 
            wNow_n = wNow/nw # neurons
            
            
            # train svm on all data using best c
            linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty=regType, dual=False)
            linear_svm.fit(X_svr, Y0)   
            w = np.squeeze(linear_svm.coef_);
            b = linear_svm.intercept_;
            #trainE = abs(linear_svm.predict(X)-Y.astype('float')).sum()/len(Y)*100;
            
            # normalize w
            nw = np.linalg.norm(w) # numSamps; 2-norm of weights 
            w_n = w/nw # neurons
            
            
            # compute angle between the averaged across samples decoder and the one trained using all data (without testing and training)
            print np.arccos(abs(np.dot(wNow_n, w_n.transpose())))*180/np.pi
            
            #s = 20
            #np.arccos(abs(np.dot(w_data_n[s,:].squeeze(), w_n.transpose())))*180/np.pi
            
            
            # angle between bestc decoders of different shuffles
            ang_samps = np.full((numSamples,numSamples), np.nan)
            for s in range(numSamples):
                for ss in range(numSamples):
                    ang_samps[s,ss] = np.arccos(abs(np.dot(w_data_n[s,:].squeeze(), w_data_n[ss,:].squeeze())))*180/np.pi
            
            # average angle between decoders of different samples
            # first set diag elements to nan (because we know their angle is 0)
            np.fill_diagonal(ang_samps, np.nan)
            
            plt.figure()
            plt.imshow(ang_samps, cmap='jet_r')
            plt.colorbar()
            
            print 'Average angle btwn optimal decoders of all shuffles = ', np.nanmean(ang_samps)
            
            
            ########## now compare angle between bagged decoder and l1 optimal decoder
            print np.arccos(abs(np.dot(w_n, w_bag)))*180/np.pi
            
            # with average l1 decoder
            print np.arccos(abs(np.dot(wNow_n, w_bag)))*180/np.pi
            
            # with each l1 decoder
            angs = []
            for s in range(numSamples):
                angs.append(np.arccos(abs(np.dot(w_data_n[s,:].squeeze(), w_bag)))*180/np.pi)
            
            angs = np.array(angs)
            plt.figure()
            plt.plot(angs)
            print 'Angle between BAGGED decoder and optimal L1 decoder = ', angs.mean()
            
            
            #%% predictions on incorr and testing corr, using l1 decoder vs bagged decoder
            
            # bagged decoder
            print 'BAGGED'
            # testing corr
            print 'train: ', np.round(classErr_corrTrain_all.mean(), 2)
            print 'corr: ', np.round(classErr_corrTest_all.mean(), 2)
            print 'incorr: ', np.round(classErr_incorrTest_all.mean(), 2), '\n'
            
            
            # L1/L2 decocer
            print regType,' decocer'
            print 'train: ', np.round(classErr_bestC_train_data.mean(), 2)
            print 'corr: ', np.round(classErr_bestC_test_data.mean(), 2)
            print 'incorr: ', np.round(classErr_bestC_incorr_data.mean(), 2), '\n'
            '''
            # decoder found using all data (without doing testing and training)
            yhat0 = predictMan(X_svr_incorr, w_n, b)
            print perClassError(yhat0, Y_incorr)
            
            # averaged decoder across samples
            yhat0 = predictMan(X_svr_incorr, wNow_n, b_bestc_data.mean())
            print perClassError(yhat0, Y_incorr)
            '''
            print regType,' decocer', '(bestc found separately for incorr and testing corr)'
            # min err on testing corr for Bagged decoders (with any c valur larger than bestc)
            print 'corr: ', np.round(np.mean(perClassErrorTest[:,np.argwhere(indBestC).squeeze():], axis=0).min(), 2)
            # min err on incorr for Bagged decoders (with any c valur larger than bestc)
            print 'incorr: ', np.round(np.mean(perClassErrorTest_incorr[:,np.argwhere(np.in1d(cvect, cbest_incorr)).squeeze():], axis=0).min(), 2), '\n'
        
        
        
        #%%
        ####################################################################################################################################
        ############################################ Save results as .mat files in a folder named svm ####################################################################################################################################
        ####################################################################################################################################
        
        #%% Save SVR results
        
        if trialHistAnalysis:
            svmn = 'svrPrevChoice_corrIncorr_stAl_epAllStim_%s_' %(nowStr)
        else:
            svmn = 'svrCurrChoice_corrIncorr_stAl_epAllStim_%s_' %(nowStr)
        print '\n', svmn[:-1]
        
        if saveResults:
            print 'Saving .mat file'
            d = os.path.join(os.path.dirname(pnevFileName), 'svm')
            if not os.path.exists(d):
                print 'creating svm folder'
                os.makedirs(d)
        
            svmName = os.path.join(d, svmn+os.path.basename(pnevFileName))
            print(svmName)
            
            if doBagging==0:
                scio.savemat(svmName, {'thAct':thAct, 'thTrsWithSpike':thTrsWithSpike, 'ep_ms':ep_ms, 'ep_ch':ep_ch,
                                       'th_stim_dur':th_stim_dur, 'numSamples':numSamples, 'nRandTrSamp':nRandTrSamp,
                                       'trsExcluded':trsExcluded, 'trsExcluded_chAl':trsExcluded_chAl, 'trsExcluded_incorr':trsExcluded_incorr, 'trsExcluded_incorr_chAl':trsExcluded_incorr_chAl,
                                       'NsExcluded':NsExcluded, 'NsExcluded_chAl':NsExcluded_chAl,
                                       'regType':regType, 'cvect':cvect,
                                       'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,'cbest_incorr':cbest_incorr,
                                       'wAllC':wAllC,
                                       'bAllC':bAllC,
                                       'perClassErrorTrain':perClassErrorTrain,
                                       'perClassErrorTest':perClassErrorTest,
                                       'perClassErrorTest_shfl':perClassErrorTest_shfl,
                                       'perClassErrorTest_chance':perClassErrorTest_chance,
                                       'perClassErrorTest_incorr':perClassErrorTest_incorr,
                                       'perClassErrorTest_incorr_shfl':perClassErrorTest_incorr_shfl,
                                       'perClassErrorTest_incorr_chance':perClassErrorTest_incorr_chance}) 
            else:
                scio.savemat(svmName, {'thAct':thAct, 'thTrsWithSpike':thTrsWithSpike, 'ep_ms':ep_ms, 'ep_ch':ep_ch,
                                       'th_stim_dur':th_stim_dur, 'numSamples':numSamples, 'nRandTrSamp':nRandTrSamp,
                                       'trsExcluded':trsExcluded, 'trsExcluded_chAl':trsExcluded_chAl, 'trsExcluded_incorr':trsExcluded_incorr, 'trsExcluded_incorr_chAl':trsExcluded_incorr_chAl,
                                       'NsExcluded':NsExcluded, 'NsExcluded_chAl':NsExcluded_chAl,
                                       'regType':regType, 'cvect':cvect,
                                       'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,'cbest_incorr':cbest_incorr,
                                       'wAllC':wAllC,
                                       'bAllC':bAllC,
                                       'perClassErrorTrain':perClassErrorTrain,
                                       'perClassErrorTest':perClassErrorTest,
                                       'perClassErrorTest_shfl':perClassErrorTest_shfl,
                                       'perClassErrorTest_chance':perClassErrorTest_chance,
                                       'perClassErrorTest_incorr':perClassErrorTest_incorr,
                                       'perClassErrorTest_incorr_shfl':perClassErrorTest_incorr_shfl,
                                       'perClassErrorTest_incorr_chance':perClassErrorTest_incorr_chance,
                                       'cbeg':cbeg, 'w_all':w_all, 'b_all':b_all, # starting from here are BAGGING vars.
                                       'classErr_corrTest_all':classErr_corrTest_all,
                                       'classErr_incorrTest_all':classErr_incorrTest_all,
                                       'classErr_corrTrain_all':classErr_corrTrain_all,
                                       'classErr_corrTrainTest_all':classErr_corrTrainTest_all,
                                       'classErr_corrTest_chance_all':classErr_corrTest_chance_all,
                                       'classErr_corrTest_sh_all':classErr_corrTest_sh_all,
                                       'classErr_incorrTest_chance_all':classErr_incorrTest_chance_all,
                                       'classErr_incorrTest_sh_all':classErr_incorrTest_sh_all})         
                
            
        else:
            print 'Not saving .mat file'
        
