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

##%%
####### NOTE ########
#print 'Decide which one to load: inhibitRois_pix or inhibitRois!!'


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
    mousename = 'fni19' #'fni17' #'fni16' #
    imagingFolder = '150930'#'150923' #'151023' #'151023' #'151001' #
    mdfFileNumber = [1] #[1] 

    shflTrsEachNeuron = 1  # Set to 0 for normal SVM training. # Shuffle trials in X_svm (for each neuron independently) to break correlations between neurons in each trial.
    outcome2ana = 'corr' #'all' # '', 'corr', 'incorr' # trials to use for SVM training (all, correct or incorrect trials) # outcome2ana will be used if trialHistAnalysis is 0. When it is 1, by default we are analyzing past correct trials. If you want to change that, set it in the matlab code.        
    doInhAllexcEqexc = [0,1,0]
    #    1st element: analyze inhibitory neurons (train SVM for numSamples for each value of C)
    #    2nd element: analyze all excitatory neurons (train SVM for numSamples for each value of C)   
    #    3rd element: analyze excitatory neurons, equal number to inhibitory neurons (train SVM for numSamples for each value of C, repeat this numShufflesExc times (each time subselecting n exc neurons))


    chAl = 1 # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM.     
    numSamples = 3 #100; # number of iterations for finding the best c (inverse of regularization parameter)
    numShufflesExc = 2 # we choose 10 sets of n random excitatory neurons.    

    softNorm = 1 # if 1, no neurons will be excluded, bc we do soft normalization of FRs, so non-active neurons wont be problematic. if softNorm = 0, NsExcluded will be found
    useEqualTrNums = 1 # Make sure both classes have the same number of trials when training the classifier
    winLen = 100 # ms, length of window for downsampling; svm will be trained in non-overlapping windows of size winLen ms.
    

    regType = 'l2' # 'l2' : regularization type
    kfold = 10

    doPlots = 1 # Whether to make plots or not.
    saveResults = 0 # save results in mat file.

    trialHistAnalysis = 0
    iTiFlg = 1 # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.

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


eps = sys.float_info.epsilon #2.2204e-16
#nRandTrSamp = numSamples #10 #1000 # number of times to subselect trials (for the BAGGING method)
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(winLen/frameLength)) # 100ms # set to nan if you don't want to downsample.


#%% Print vars

if doInhAllexcEqexc[0]==1:
    ntName = 'inh'
elif doInhAllexcEqexc[1]==1:
    ntName = 'allExc'
elif doInhAllexcEqexc[2]==1:
    ntName = 'eqExc'     

if trialHistAnalysis==1:    
    if iTiFlg==0:
        itiName = 'short'
    elif iTiFlg==1:
        itiName = 'long'
    elif iTiFlg==2:
        itiName = 'all'        
    
print 'Analyzing %s' %(mousename+'_'+imagingFolder+'_'+str(mdfFileNumber)) 
if chAl==1:
    print 'Using choice-aligned traces'
else:
    print 'Using stimulus-aligned traces'    
if trialHistAnalysis==0:
    print 'Training %s trials of strength %s' %(outcome2ana, strength2ana)
print 'trialHistAnalysis = %i' %(trialHistAnalysis)
print 'Analyzing %s neurons' %(ntName)
if trialHistAnalysis==1:
    print 'Analyzing %s ITIs' %(itiName)
elif 'ep_ms' in locals():
    print 'training window: [%d %d] ms' %(ep_ms[0], ep_ms[1])
#print 'windowAvgFlg = %i' %(windowAvgFlg)
print 'numSamples = %i' %(numSamples)



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
'''
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
'''

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
        if os.getcwd().find('grid')!=-1: # server # sonas
            dataPath = '/sonas-hs/churchland/nlsas/data/data/'
            altDataPath = '/sonas-hs/churchland/hpc/home/space_managed_data/'
        else: # office linux
            dataPath = '/home/farznaj/Shares/Churchland/data/'
            altDataPath = '/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/' # the new space-managed server (wos, to which data is migrated from grid)
    elif platform.system()=='Darwin':
        dataPath = '/Volumes/My Stu_win/ChurchlandLab'
    else:
        dataPath = '/Users/gamalamin/git_local_repository/Farzaneh/data/'

    ##%%
    tifFold = os.path.join(dataPath+mousename,'imaging',imagingFolder)
        
    if not os.path.exists(tifFold):
        if 'altDataPath' in locals():
            tifFold = os.path.join(altDataPath+mousename, 'imaging', imagingFolder)
        else:
            sys.exit('Data directory does not exist!')


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

"""    
# Set traces for specific neuron types: inhibitory, excitatory or all neurons
if neuronType!=2:    
    nt = (inhibitRois==neuronType) # 0: excitatory, 1: inhibitory, 2: all types.
    # good_excit = inhibitRois==0;
    # good_inhibit = inhibitRois==1;        
    
    traces_al_stim = traces_al_stim[:, nt, :];
    traces_al_stimAll = traces_al_stimAll[:, nt, :];
else:
    nt = np.arange(np.shape(traces_al_stim)[1])    
"""


    
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
#    Y_svm = choiceVec0[~trsExcluded];
#    print 'frs x units x trials', X_svm.shape
#    print Y_svm.shape
    
    time_trace = time_aligned_1stSide


##%%
elif goToneAl==1:   # Use go-tone-aligned traces
    
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

    
    
##%%
elif stAl==1:   #%% Use stimulus-aligned traces
    
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
    
    ep_ms = [800] # I want to make sure in none of the trials choice happened before 800ms...
    
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

print 'frs x units x trials', X_svm.shape    



#%% Set Y for training SVM   

Y_svm = choiceVec0[~trsExcluded]
#len(Y_svm)
# Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
hr_trs = (Y_svm==1)
lr_trs = (Y_svm==0)
    
print 'Y size: ', Y_svm.shape

## print number of hr,lr trials after excluding trials
if outcome2ana == 'corr':
    print '\tcorrect trials: %d HR; %d LR' %((Y_svm==1).sum(), (Y_svm==0).sum())
elif outcome2ana == 'incorr':
    print '\tincorrect trials: %d HR; %d LR' %((Y_svm==1).sum(), (Y_svm==0).sum())
else:
    print '\tall trials: %d HR; %d LR' %((Y_svm==1).sum(), (Y_svm==0).sum())
    
    

#%% I think we need at the very least 3 trials of each class to train SVM. So exit the analysis if this condition is not met!

if min((Y_svm==1).sum(), (Y_svm==0).sum()) < 3:
    sys.exit('Too few trials to do SVM training! HR=%d, LR=%d' %((Y_svm==1).sum(), (Y_svm==0).sum()))
    

    
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

    ##### x_svm
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
    print 'trace size--> original:',X_svm.shape, 'downsampled:', X_svm_d.shape
    X_svm = X_svm_d
    
    
    
    ##### time_trace
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
    print 'time trace size--> original:',time_trace.shape, 'downsampled:', time_trace_d.shape    
    time_trace = time_trace_d

    
    
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

    
#%% 

numDataPoints = X_svm.shape[2] 
print '# data points = %d' %numDataPoints
# numTrials = (~trsExcluded).sum()
# numNeurons = (~NsExcluded).sum()
numTrials, numNeurons = X_svm.shape[2], X_svm.shape[1]
print '%d trials; %d neurons' %(numTrials, numNeurons)
# print ' The data has %d frames recorded from %d neurons at %d trials' %Xt.shape    


#%% Keep a copy of X_svm before normalization

X_svm00 = X_svm + 0


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
meanX_fr = []
stdX_fr = []
for ifr in range(np.shape(X_svm00)[0]):
    m = np.mean(X_svm00[ifr,:,:], axis=1)
    s = np.std(X_svm00[ifr,:,:], axis=1)   
    meanX_fr.append(m) # frs x neurons
    stdX_fr.append(s)       
    
    if softNorm==1: # soft normalziation : neurons with sd<thAct wont have too large values after normalization
        s = s+thAct     

    X_svm_N[ifr,:,:] = ((X_svm00[ifr,:,:].T - m) / s).T

meanX_fr = np.array(meanX_fr) # frames x neurons
stdX_fr = np.array(stdX_fr) # frames x neurons

X_svm = X_svm_N
    


#%% ###############################################################################
##%% Shuffle trials in X_svm (for each neuron independently) to break correlations between neurons in each trial.
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
    
    # For each neuron use a different order of trials ... do it separately for hr and lr trials, so a constituted HR trial includes neural responses from only HR trials, but shuffled
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


#%%
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

if doPlots:
    plt.figure
    plt.subplot(2,2,1)
#    plt.plot(np.mean(meanX_fr,axis=0)) # average across frames for each neuron
    plt.hist(meanX_fr.reshape(-1,))
    plt.xlabel('meanX \n(mean of trials)')
    plt.ylabel('frames x neurons')
    plt.title('min = %.6f' %(np.min(np.mean(meanX_fr,axis=0))))

    plt.subplot(2,2,3)
#    plt.plot(np.mean(stdX_fr,axis=0))
    plt.hist(stdX_fr.reshape(-1,))
    plt.xlabel('stdX \n(std of trials)')
    plt.ylabel('frames x neurons')
    plt.title('min = %.6f' %(np.min(np.mean(stdX_fr,axis=0))))

    plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.subplots_adjust(hspace=.5)


    
# choice-aligned: classes: choices
# Plot stim-aligned averages after centering and normalization
##%% Again define hr and lr trials.
hr_trs = (Y_svm==1)
lr_trs = (Y_svm==0)
if doPlots:
#    # Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
#    hr_trs = (Y_svm==1)
#    lr_trs = (Y_svm==0)

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
    plt.plot(time_trace, tr1, 'b', label = 'high rate')
    plt.plot(time_trace, tr0, 'r', label = 'low rate')
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


#%% Same as above but when X is frames x units x trials
# Remember each numSamples will have a different set of training and testing dataset, however for each numSamples, the same set of testing/training dataset
# will be used for all frames and all values of c (unless shuffleTrs is 1, in which case different frames and c values will have different training/testing datasets.)

def setbesc_frs(X,Y,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs):
    
    import numpy as np
    import numpy.random as rng
    from crossValidateModel import crossValidateModel
    from linearSVM import linearSVM
    
    def perClassError(Y, Yhat):
#        import numpy as np
        perClassEr = np.sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
        return perClassEr
    # numSamples = 10; # number of iterations for finding the best c (inverse of regularization parameter)
    # if you don't want to regularize, go with a very high cbest and don't run the section below.
    # cbest = 10**6
    
    # regType = 'l1'
    # kfold = 10;
        
    if regType == 'l1':
        print '\n-------------- Running l1 svm classification --------------\r' 
        # cvect = 10**(np.arange(-4, 6,0.2))/numTrials;
        cvect = 10**(np.arange(-4, 6,0.2))/numDataPoints;
    elif regType == 'l2':
        print '\n-------------- Running l2 svm classification --------------\r' 
        cvect = 10**(np.arange(-6, 6,0.2))/numDataPoints;    
#    print 'try the following regularization values: \n', cvect
    # formattedList = ['%.2f' % member for member in cvect]
    # print 'try the following regularization values = \n', formattedList
    
#    smallestC = 0 # if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
    if smallestC==1:
        print 'bestc = smallest c whose cv error is less than 1se of min cv error'
    else:
        print 'bestc = c that gives min cv error'
    #I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.
        
        
        
    wAllC = np.ones((numSamples, len(cvect), X.shape[1], np.shape(X)[0]))+np.nan;
    bAllC = np.ones((numSamples, len(cvect), np.shape(X)[0]))+np.nan;
    
    perClassErrorTrain = np.ones((numSamples, len(cvect), np.shape(X)[0]))+np.nan;
    perClassErrorTest = np.ones((numSamples, len(cvect), np.shape(X)[0]))+np.nan;
    
    perClassErrorTest_shfl = np.ones((numSamples, len(cvect), np.shape(X)[0]))+np.nan;
    perClassErrorTest_chance = np.ones((numSamples, len(cvect), np.shape(X)[0]))+np.nan
    
    
    ##############
    hrn = (Y==1).sum()
    lrn = (Y==0).sum()    
    
    if useEqualTrNums and hrn!=lrn: # if the HR and LR trials numbers are not the same, pick equal number of trials of the 2 classes!        
        trsn = min(lrn,hrn)
        if hrn > lrn:
            print 'Subselecting HR trials so both classes have the same number of trials!'
            n = lrn*2
        elif lrn > hrn:
            print 'Subselecting LR trials so both classes have the same number of trials!'
            n = hrn*2
        print 'FINAL: %d trials; %d neurons' %(n, X.shape[1])
            
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
                            
            X = X0[:,:,trsnow]
            Y = Y0[trsnow]
      
        numTrials, numNeurons = X.shape[2], X.shape[1]
#            print 'FINAL: %d trials; %d neurons' %(numTrials, numNeurons)                
        
        
        ######################## Shuffle trial orders, so the training and testing datasets are different for each numSamples (we only do this if shuffleTrs is 0, so crossValidateModel does not shuffle trials, so we have to do it here, otherwise all numSamples will have the same set of testing and training datasets.)
        if shuffleTrs==0: # shuffle trials to break any dependencies on the sequence of trails 
            shfl = rng.permutation(np.arange(0, numTrials));
            Y = Y[shfl];
            X = X[:,:,shfl]; 

            
        ######################## Setting chance Y: same length as Y for testing data, and with equal number of classes 0 and 1.
        no = Y.shape[0]
        len_test = no - int((kfold-1.)/kfold*no)    
        permIxs = rng.permutation(len_test)    
    
        Y_chance = np.zeros(len_test)
        if rng.rand()>.5:
            b = rng.permutation(len_test)[0:np.floor(len_test/float(2)).astype(int)]
        else:
            b = rng.permutation(len_test)[0:np.ceil(len_test/float(2)).astype(int)]
        Y_chance[b] = 1



        ############# Start training SVM
        for ifr in range(X.shape[0]): # train SVM on each frame
#            print '\tFrame %d' %(ifr)  
            ######################## Loop over different values of regularization
            for i in range(len(cvect)): # train SVM using different values of regularization parameter
                
                if regType == 'l1':                       
                    summary,_ =  crossValidateModel(X[ifr,:,:].transpose(), Y, linearSVM, kfold = kfold, l1 = cvect[i], shflTrs = shuffleTrs)
                elif regType == 'l2':
                    summary,_ =  crossValidateModel(X[ifr,:,:].transpose(), Y, linearSVM, kfold = kfold, l2 = cvect[i], shflTrs = shuffleTrs)
                            
                wAllC[s,i,:,ifr] = np.squeeze(summary.model.coef_); # weights of all neurons for each value of c and each shuffle
                bAllC[s,i,ifr] = np.squeeze(summary.model.intercept_);
        
                # classification errors                    
                perClassErrorTrain[s,i,ifr] = summary.perClassErrorTrain;
                perClassErrorTest[s,i,ifr] = summary.perClassErrorTest;                
                
                # Testing correct shuffled data: 
                # same decoder trained on correct trials, make predictions on correct with shuffled labels.
                perClassErrorTest_shfl[s,i] = perClassError(summary.YTest[permIxs], summary.model.predict(summary.XTest));
                perClassErrorTest_chance[s,i] = perClassError(Y_chance, summary.model.predict(summary.XTest));
            
    
    

    ######################### Find bestc for each frame, and plot the c path 
    print '--------------- Identifying best c ---------------' 
    cbestFrs = np.full((X.shape[0]), np.nan)  
    cbestAllFrs = np.full((X.shape[0]), np.nan)  
    for ifr in range(X.shape[0]):    
        #######%% Compute average of class errors across numSamples        
        meanPerClassErrorTrain = np.mean(perClassErrorTrain[:,:,ifr], axis = 0);
        semPerClassErrorTrain = np.std(perClassErrorTrain[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest = np.mean(perClassErrorTest[:,:,ifr], axis = 0);
        semPerClassErrorTest = np.std(perClassErrorTest[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl[:,:,ifr], axis = 0);
        semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance[:,:,ifr], axis = 0);
        semPerClassErrorTest_chance = np.std(perClassErrorTest_chance[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        
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
    return perClassErrorTrain, perClassErrorTest, wAllC, bAllC, cbestAllFrs, cbestFrs, cvect, perClassErrorTest_shfl, perClassErrorTest_chance



#%%
useEqualTrNums = 1
smallestC = 0 # if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
shuffleTrs = False; # set to 0 so for each iteration of numSamples, all frames are trained and tested on the same trials# If 1 shuffle trials to break any dependencies on the sequence of trails 


#%%
####################################################################################################################################
# ## Compute classification error of training and testing dataset for the following cases:
#     Here we train the classifier using the best c found above (when including all neurons in the decoder) but using different sets of neurons (and 90% training, 10% testing trials.) ... I think this is problematic to use bestc of a decoder trained on a different population!
#     
#     1) All inhibitory neurons contribute to the decoder
#     2) N excitatory neurons contribute, where n = number of inhibitory neurons. 
#     3) All excitatory neurons contribute
####################################################################################################################################
####################################################################################################################################
#### 3 populations: inh, n exc, all exc
#### For each population: find bestc ; train SVM ; do cross validation (numSamples times) ; set null distribution (by shuffling trial labels)
####################################################################################################################################

#%%
   
nFrs = np.shape(X_svm)[0]   
lenInh = (inhRois==1).sum()
excI = np.argwhere(inhRois==0)    
    
################## set bestc and then class error,w,b for data ############
    
if doInhAllexcEqexc[0] == 1: ###### inh
    Xinh = X_svm[:, inhRois==1,:]
    perClassErrorTrain_inh, perClassErrorTest_inh, wAllC_inh, bAllC_inh, cbestAll_inh, cbest_inh, cvect, perClassErrorTestShfl_inh, perClassErrorTestChance_inh = setbesc_frs(Xinh,Y_svm,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs) # outputs have size the number of shuffles in setbestc (shuffles per c value)
    
    # Set the parameters below at bestc (for all samples):
    perClassErrorTrain_data_inh = np.full((numSamples, nFrs), np.nan)
    perClassErrorTest_data_inh = np.full((numSamples, nFrs), np.nan)
    perClassErrorTest_shfl_inh = np.full((numSamples, nFrs), np.nan)
    perClassErrorTest_chance_inh = np.full((numSamples, nFrs), np.nan)
    w_data_inh = np.full((numSamples, lenInh, nFrs), np.nan)
    b_data_inh = np.full((numSamples, nFrs), np.nan)
    
    for ifr in range(nFrs):    
        indBestC = np.in1d(cvect, cbest_inh[ifr])
        perClassErrorTrain_data_inh[:,ifr] = perClassErrorTrain_inh[:,indBestC,ifr].squeeze()
        perClassErrorTest_data_inh[:,ifr] = perClassErrorTest_inh[:,indBestC,ifr].squeeze()
        perClassErrorTest_shfl_inh[:,ifr] = perClassErrorTestShfl_inh[:,indBestC,ifr].squeeze()
        perClassErrorTest_chance_inh[:,ifr] = perClassErrorTestChance_inh[:,indBestC,ifr].squeeze()
        w_data_inh[:,:,ifr] = wAllC_inh[:,indBestC,:,ifr].squeeze()
        b_data_inh[:,ifr] = bAllC_inh[:,indBestC,ifr].squeeze()


elif doInhAllexcEqexc[1] == 1: ###### allExc
    XallExc = X_svm[:, inhRois==0,:]
    perClassErrorTrain_allExc, perClassErrorTest_allExc, wAllC_allExc, bAllC_allExc, cbestAll_allExc, cbest_allExc, cvect, perClassErrorTestShfl_allExc, perClassErrorTestChance_allExc = setbesc_frs(XallExc,Y_svm,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs) # outputs have size the number of shuffles in setbestc (shuffles per c value)
    
    # Take the values at bestc that you computed above (for all samples)
    perClassErrorTrain_data_allExc = np.full((numSamples, nFrs), np.nan)
    perClassErrorTest_data_allExc = np.full((numSamples, nFrs), np.nan)
    perClassErrorTest_shfl_allExc = np.full((numSamples, nFrs), np.nan)
    perClassErrorTest_chance_allExc = np.full((numSamples, nFrs), np.nan)
    w_data_allExc = np.full((numSamples, len(excI), nFrs), np.nan)
    b_data_allExc = np.full((numSamples, nFrs), np.nan)
    for ifr in range(nFrs):    
        indBestC = np.in1d(cvect, cbest_allExc[ifr])
        perClassErrorTrain_data_allExc[:,ifr] = perClassErrorTrain_allExc[:,indBestC,ifr].squeeze()
        perClassErrorTest_data_allExc[:,ifr] = perClassErrorTest_allExc[:,indBestC,ifr].squeeze()
        perClassErrorTest_shfl_allExc[:,ifr] = perClassErrorTestShfl_allExc[:,indBestC,ifr].squeeze()
        perClassErrorTest_chance_allExc[:,ifr] = perClassErrorTestChance_allExc[:,indBestC,ifr].squeeze()
        w_data_allExc[:,:,ifr] = wAllC_allExc[:,indBestC,:,ifr].squeeze()
        b_data_allExc[:,ifr] = bAllC_allExc[:,indBestC,ifr].squeeze()
    

elif doInhAllexcEqexc[2] == 1: ###### n exc (pick numShufflesEx sets of n exc neurons)
    XexcEq = []
    #tempExcAll = []    
    excNsEachSamp = []
    for ii in range(numShufflesExc):  # select n random exc (n = number of inh)         
        en = rng.permutation(excI)[0:lenInh].squeeze() # n randomly selected exc neurons.    
        Xexc = X_svm[:, en,:]
        XexcEq.append(Xexc)    
        excNsEachSamp.append(en) # indeces of exc neurons (our of X_svm) used for svm training in each exc shfl (below).... you need this if you want to get svm projections for a particular exc shfl (eg w_data_exc[nExcShfl,:,:,:])
        
        # you need the traces for setting correct classification traces in the next section
    #    tempExc = Xt[:,en,:] # stimulus-aligned traces (I think you should use Xtsa for consistency... u get projections from Xtsa)
    #    tempExcAll.append(tempExc)
    
    cbestAll_exc = []
    cbest_exc = []
    perClassErrorTrain_data_exc = []
    perClassErrorTest_data_exc = []
    perClassErrorTest_shfl_exc = []
    perClassErrorTest_chance_exc = []
    w_data_exc = []
    b_data_exc = []    
    for ii in range(numShufflesExc):    
        print '\n\n-------- Exc shuffle %d --------' %(ii)
        # Set bestc    
    #    perClassErrorTrain_exc, perClassErrorTest_exc, wAllC_exc, bAllC_exc, cbestAll_exc0, cbest_exc0, cvect = setbesc(XexcEq[ii],Y,regType,kfold,numDataPoints,numSamples,doPlots)
        perClassErrorTrain_exc, perClassErrorTest_exc, wAllC_exc, bAllC_exc, cbestAll_exc0, cbest_exc0, cvect, perClassErrorTestShfl_exc, perClassErrorTestChance_exc = setbesc_frs(XexcEq[ii],Y_svm,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs) # outputs have size the number of shuffles in setbestc (shuffles per c value)
    
        # Take the values at bestc that you computed above (for all samples)
        perClassErrorTrain_data_exc0 = np.full((numSamples, nFrs), np.nan)
        perClassErrorTest_data_exc0 = np.full((numSamples, nFrs), np.nan)
        perClassErrorTest_shfl_exc0 = np.full((numSamples, nFrs), np.nan)
        perClassErrorTest_chance_exc0 = np.full((numSamples, nFrs), np.nan)
        w_data_exc0 = np.full((numSamples, lenInh, nFrs), np.nan)
        b_data_exc0 = np.full((numSamples, nFrs), np.nan)
        for ifr in range(nFrs):    
            indBestC = np.in1d(cvect, cbest_exc0[ifr])
            perClassErrorTrain_data_exc0[:,ifr] = perClassErrorTrain_exc[:,indBestC,ifr].squeeze()
            perClassErrorTest_data_exc0[:,ifr] = perClassErrorTest_exc[:,indBestC,ifr].squeeze()
            perClassErrorTest_shfl_exc0[:,ifr] = perClassErrorTestShfl_exc[:,indBestC,ifr].squeeze()
            perClassErrorTest_chance_exc0[:,ifr] = perClassErrorTestChance_exc[:,indBestC,ifr].squeeze()
            w_data_exc0[:,:,ifr] = wAllC_exc[:,indBestC,:,ifr].squeeze()
            b_data_exc0[:,ifr] = bAllC_exc[:,indBestC,ifr].squeeze()
        
        # collect values of all exc shuffles    
        cbestAll_exc.append(cbestAll_exc0)
        cbest_exc.append(cbest_exc0)
        perClassErrorTrain_data_exc.append(perClassErrorTrain_data_exc0) # numShufflesExc x numSamples x numFrames
        perClassErrorTest_data_exc.append(perClassErrorTest_data_exc0)
        perClassErrorTest_shfl_exc.append(perClassErrorTest_shfl_exc0)
        perClassErrorTest_chance_exc.append(perClassErrorTest_chance_exc0)
        w_data_exc.append(w_data_exc0) # numShufflesExc x numSamples x numNeurons x numFrames
        b_data_exc.append(b_data_exc0)
        
    '''
    # I don't think I need this!
    # transpose so the sizes match _inh and _allExc as well as shuffle vars (below)        
    perClassErrorTrain_data_exc = np.transpose(perClassErrorTrain_data_exc) # numSamples x numShufflesExc
    perClassErrorTest_data_exc = np.transpose(perClassErrorTest_data_exc)
    w_data_exc = np.transpose(w_data_exc,(1,0,2)) # numSamples x numShufflesExc x numNeur
    b_data_exc = np.transpose(b_data_exc)
    '''    

      
        
    
#%%
####################################################################################################################################
############################################ Save results as .mat files in a folder named svm ####################################################################################################################################
####################################################################################################################################

#%% Save SVM results

if chAl==1:
    al = 'chAl'
else:
    al = 'stAl'

if outcome2ana == 'corr': # save incorr vars too bc SVM was trained on corr, and tested on icorr.
    o2a = 'corr_'
else:
    o2a = ''
        
if doInhAllexcEqexc[0] == 1:
    ntype = 'inh'
elif doInhAllexcEqexc[1] == 1:
    ntype = 'allExc'
elif doInhAllexcEqexc[2] == 1:
    ntype = 'eqExc'   

if shflTrsEachNeuron:
	shflname = 'shflTrsPerN_'
else:
	shflname = ''

    
if trialHistAnalysis:
    if useEqualTrNums:
        svmn = 'excInh_SVMtrained_eachFrame_%s%s%s_prevChoice_%s_ds%d_eqTrs_%s_' %(o2a, shflname, ntype, al,regressBins,nowStr)
    else:
        svmn = 'excInh_SVMtrained_eachFrame_%s%s%s_prevChoice_%s_ds%d_%s_' %(o2a, shflname, ntype, al,regressBins,nowStr)
else:
    if useEqualTrNums:
        svmn = 'excInh_SVMtrained_eachFrame_%s%s%s_currChoice_%s_ds%d_eqTrs_%s_' %(o2a, shflname, ntype, al,regressBins,nowStr)
    else:
        svmn = 'excInh_SVMtrained_eachFrame_%s%s%s_currChoice_%s_ds%d_%s_' %(o2a, shflname, ntype, al,regressBins,nowStr)
print '\n', svmn[:-1]



if saveResults:
    print 'Saving .mat file'
    d = os.path.join(os.path.dirname(pnevFileName), 'svm')
    if not os.path.exists(d):
        print 'creating svm folder'
        os.makedirs(d)

    svmName = os.path.join(d, svmn+os.path.basename(pnevFileName))
    print(svmName)
    
    if doInhAllexcEqexc[0] == 1:
        scio.savemat(svmName, {'thAct':thAct, 'numShufflesExc':numShufflesExc, 'numSamples':numSamples, 'softNorm':softNorm, 'meanX_fr':meanX_fr, 'stdX_fr':stdX_fr,
                               'winLen':winLen, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'regType':regType, 'cvect':cvect, 'eventI_ds':eventI_ds, #'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,
                               'cbest_inh':cbest_inh, 'w_data_inh':w_data_inh, 'b_data_inh':b_data_inh, 'hr_shfl_all':hr_shfl_all, 'lr_shfl_all':lr_shfl_all,
                               'perClassErrorTrain_data_inh':perClassErrorTrain_data_inh,
                               'perClassErrorTest_data_inh':perClassErrorTest_data_inh,
                               'perClassErrorTest_shfl_inh':perClassErrorTest_shfl_inh,
                               'perClassErrorTest_chance_inh':perClassErrorTest_chance_inh})
                               
    elif doInhAllexcEqexc[1] == 1:
        scio.savemat(svmName, {'thAct':thAct, 'numShufflesExc':numShufflesExc, 'numSamples':numSamples, 'softNorm':softNorm, 'meanX_fr':meanX_fr, 'stdX_fr':stdX_fr,
                               'winLen':winLen, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'regType':regType, 'cvect':cvect, 'eventI_ds':eventI_ds, #'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,
                               'cbest_allExc':cbest_allExc, 'w_data_allExc':w_data_allExc, 'b_data_allExc':b_data_allExc, 'hr_shfl_all':hr_shfl_all, 'lr_shfl_all':lr_shfl_all,
                               'perClassErrorTrain_data_allExc':perClassErrorTrain_data_allExc,
                               'perClassErrorTest_data_allExc':perClassErrorTest_data_allExc,
                               'perClassErrorTest_shfl_allExc':perClassErrorTest_shfl_allExc,
                               'perClassErrorTest_chance_allExc':perClassErrorTest_chance_allExc})       
       
    elif doInhAllexcEqexc[2] == 1:       
        scio.savemat(svmName, {'thAct':thAct, 'numShufflesExc':numShufflesExc, 'numSamples':numSamples, 'softNorm':softNorm, 'meanX_fr':meanX_fr, 'stdX_fr':stdX_fr,
                               'winLen':winLen, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'regType':regType, 'cvect':cvect, 'eventI_ds':eventI_ds, #'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,
                               'cbest_exc':cbest_exc, 'w_data_exc':w_data_exc, 'b_data_exc':b_data_exc, 'excNsEachSamp':excNsEachSamp, 'hr_shfl_all':hr_shfl_all, 'lr_shfl_all':lr_shfl_all,
                               'perClassErrorTrain_data_exc':perClassErrorTrain_data_exc,
                               'perClassErrorTest_data_exc':perClassErrorTest_data_exc,
                               'perClassErrorTest_shfl_exc':perClassErrorTest_shfl_exc,
                               'perClassErrorTest_chance_exc':perClassErrorTest_chance_exc})

       
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