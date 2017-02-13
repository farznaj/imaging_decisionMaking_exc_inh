# -*- coding: utf-8 -*-
"""
Train SVM on correct trials and test its classification accuracy on incorrect trials.
[Also as usual we test it on correct trials (both training and testing datasets) 
and we do trila-label shuffles to set null distributions for the classification 
accuracy of the classifiers.]


Created on Fri Feb 10 20:20:08 2017
@author: farznaj
"""


# coding: utf-8

# ## Specify variables for the analysis: 
#     - Data (mouse, day, sessions)
#     - Neuron type: excit, inhibit, or all
#     - Current-choice or previous-choice SVM training
#         if current-choice, specify epoch of training, the outcome (corr, incorr, all) and strength (easy, medium, hard, all) of trials for training SVM.
#         if previous-choice, specify ITI flag
#     - Trials that will be used for projections and class accuracy traces (corr, incorr, all, trained).

# In[1]:

# Add the option to toggle on/off the raw code. Copied from http://stackoverflow.com/questions/27934885/how-to-hide-code-from-cells-in-ipython-notebook-visualized-with-nbviewer
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


# In[3]:

import sys
import os
import numpy as np
from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')

# Only run the following section if you are running the code in jupyter or in spyder, but not on the cluster!
if ('ipykernel' in sys.modules) or any('SPYDER' in name for name in os.environ):
    
    # Set these variables:
    mousename = 'fni16'
    imagingFolder = '151001'
    mdfFileNumber = [1] 

    trialHistAnalysis = 0;    
    roundi = 1; # For the same dataset we run the code multiple times, each time we select a random subset of neurons (of size n, n=.95*numTrials)

    iTiFlg = 1; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.
    setNsExcluded = 1; # if 1, NsExcluded will be set even if it is already saved.
    numSamples = 10 #100; # number of iterations for finding the best c (inverse of regularization parameter)
    neuronType = 2; # 0: excitatory, 1: inhibitory, 2: all types.    
    saveResults = 0; # save results in mat file.

    
    doNsRand = 0; # if 1, a random set of neurons will be selected to make sure we have fewer neurons than trials. 
    regType = 'l1' # 'l2' : regularization type
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


# In[4]:

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

# In[5]:

import scipy.io as scio
import scipy as sci
import scipy.stats as stats
#import numpy as np
import numpy.random as rng
import sys
from crossValidateModel import crossValidateModel
from linearSVM import linearSVM
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


# In[7]:

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


# In[8]:


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

#%%
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
    
    
#%%
#imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh, pnev2load)



# ## Set mat-file names

# In[9]:

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
# ## Load matlab variables: event-aligned traces, inhibitRois, outcomes,  choice, etc
#     - traces are set in set_aligned_traces.m matlab script.

# In[10]:

# Set traces_al_stim that is same as traces_al_stimAll except that in traces_al_stim some trials are set to nan, bc their stim duration is < 
# th_stim_dur or bc their go tone happens before ep(end) or bc their choice happened before ep(end). 
# But in traces_al_stimAll, all trials are included. 
# You need traces_al_stim for decoding the upcoming choice bc you average responses during ep and you want to 
# control for what happens there. But for trial-history analysis you average responses before stimOnset, so you 
# don't care about when go tone happened or how long the stimulus was. 

frameLength = 1000/30.9; # sec.
    
# Load time of some trial events    
Data = scio.loadmat(postName, variable_names=['timeCommitCL_CR_Gotone', 'timeStimOnset', 'timeStimOffset', 'time1stSideTry'])
timeCommitCL_CR_Gotone = np.array(Data.pop('timeCommitCL_CR_Gotone')).flatten().astype('float')
timeStimOnset = np.array(Data.pop('timeStimOnset')).flatten().astype('float')
timeStimOffset = np.array(Data.pop('timeStimOffset')).flatten().astype('float')
time1stSideTry = np.array(Data.pop('time1stSideTry')).flatten().astype('float')


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
# ## Set the time window for training SVM (ep) and traces_al_stim

# In[11]:

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
    
    # We first set to nan timeStimOnset of trials that anyway wont matter bc their outcome is  not of interest. we do this to make sure these trials dont affect our estimate of ep_ms
    if outcome2ana == 'corr':
        timeStimOnset[outcomes!=1] = np.nan; # analyze only correct trials.
    elif outcome2ana == 'incorr':
        timeStimOnset[outcomes!=0] = np.nan; # analyze only incorrect trials.   
        
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
# ## Set X (trials x neurons) and Y (trials x 1) for training the SVM classifier.
#     X matrix (size trials x neurons); steps include:
#        - average neural responses during ep for each trial.
#        - remove nan trials (trsExcluded)
#        - exclude non-active neurons (NsExcluded)
#        - [if doing doRands, get X only for NsRand]
#        - center and normalize X (by meanX and stdX)

#     Y choice of high rate (modeled as 1) and low rate (modeled as 0)

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

# Set spikeAveEp0  (X: the predictor matrix (trials x neurons) that shows average of spikes for a particular epoch for each trial and neuron.)

if trialHistAnalysis:
    # either of the two cases below should be fine (init-aligned traces or stim-aligned traces.)
    spikeAveEp0 = np.transpose(np.nanmean(traces_al_stimAll[ep,:,:], axis=0)) # trials x neurons    
    # spikeAveEp0 = np.transpose(np.nanmean(traces_al_init[ep,:,:], axis=0)) # trials x neurons    
else:    
    spikeAveEp0 = np.transpose(np.nanmean(traces_al_stim[ep,:,:], axis=0)) # trials x neurons    

# X = spikeAveEp0;
print 'Size of spikeAveEp0 (trs x neurons): ', spikeAveEp0.shape


#%% Set stimulus-aligned traces (X and Y)

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
Ysr = (Ysr - (Ysr.min()+np.ptp(Ysr)/2)) / np.ptp(Ysr) # when you did not do this, angle_stim_sh (ie stimulus shuffled) gave you weird results... during baseline (~15 frames at the beginning and almost ~15 frames at the end) you got alignment between population responses in the shuffled data!
#Ysr = (Ysr - Ysr.mean()) / Ysr.std()
#Ysr = (Ysr-cb) / np.ptp(stimrate) # not sure if we need this, also stimrate in some days did not span the entire range (4-28hz)


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
Ysr_chAl = (Ysr_chAl - (Ysr_chAl.min()+np.ptp(Ysr_chAl)/2)) / np.ptp(Ysr_chAl) # when you did not do this, angle_stim_sh (ie stimulus shuffled) gave you weird results... during baseline (~15 frames at the beginning and almost ~15 frames at the end) you got alignment between population responses in the shuffled data!


#%% Set X and Y for incorrect trials (both stimulus-aligned and choice-aligned)

# set Y_incorr: vector of choices for incorrect trials
Y_incorr0 = choiceVecAll+0
Y_incorr0[outcomes!=0] = np.nan; # analyze only incorrect trials.


## Stim aligned
# Exclude nan trials
trsExcluded_incorr = (np.sum(np.isnan(spikeAveEp0), axis = 1) + np.isnan(Y_incorr0)) != 0 # NaN trials # trsExcluded

X_incorr = spikeAveEp0[~trsExcluded_incorr,:]
Y_incorr = Y_incorr0[~trsExcluded_incorr]
#X_incorr.shape
#Y_incorr

Ysr_incorr = stimrate[~trsExcluded_incorr] #exclude incorr trials, etc # we dont need to exclude trials that animal did not make a choice for the stimulus decoder analysis... unless you want to use the same set of trials for the choice and stimulus decoder...
Ysr_incorr = (Ysr_incorr - (Ysr_incorr.min()+np.ptp(Ysr_incorr)/2)) / np.ptp(Ysr_incorr) # when you did not do this, angle_stim_sh (ie stimulus shuffled) gave you weird results... during baseline (~15 frames at the beginning and almost ~15 frames at the end) you got alignment between population responses in the shuffled data!



## Choice aligned
# Exclude nan trials
trsExcluded_incorr_chAl = (np.sum(np.isnan(X_choice0), axis = 1) + np.isnan(Y_incorr0)) != 0 # NaN trials # trsExcluded

X_chAl_incorr = X_choice0[~trsExcluded_incorr_chAl,:]
Y_chAl_incorr = Y_incorr0[~trsExcluded_incorr_chAl]
#X_incorr.shape
#Y_incorr

Ysr_chAl_incorr = stimrate[~trsExcluded_incorr_chAl] #exclude incorr trials, etc # we dont need to exclude trials that animal did not make a choice for the stimulus decoder analysis... unless you want to use the same set of trials for the choice and stimulus decoder...
Ysr_chAl_incorr = (Ysr_chAl_incorr - (Ysr_chAl_incorr.min()+np.ptp(Ysr_chAl_incorr)/2)) / np.ptp(Ysr_chAl_incorr) # when you did not do this, angle_stim_sh (ie stimulus shuffled) gave you weird results... during baseline (~15 frames at the beginning and almost ~15 frames at the end) you got alignment between population responses in the shuffled data!



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
    
        # Define NsExcluded as neurons with low stdX
#        stdX = np.std(np.concatenate((X,X_incorr)), axis = 0);
        stdX = np.std(X, axis = 0);
        NsExcluded = stdX < thAct
        # np.sum(stdX < thAct)

        # X_chAl
#        stdX_chAl = np.std(np.concatenate((X_chAl,X_chAl_incorr)), axis = 0);
        stdX_chAl = np.std(X_chAl, axis = 0);
        NsExcluded_chAl = stdX_chAl < thAct        


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

# stimulus-aligned
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


# choice-aligned    
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

# I am going to use the mean and std of correct trials for normalization (of both correct and incorrect trials), 
# bc SVM will be trained on correct trials.


# stimulus-aligned
meanX = np.mean(X, axis = 0);
stdX = np.std(X, axis = 0);
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


# choice-aligned
# X_chAl  
meanX_chAl = np.mean(X_chAl, axis = 0);
stdX_chAl = np.std(X_chAl, axis = 0);
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


# In[23]:

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



#%% Plot traces averaged for each choice

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
    plt.title('Population average')
    
    
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
    plt.title('Population average')
    
    
    
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
    plt.title('Population average')
    
    
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
    plt.title('Population average')
    

    
    
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
    
    



#%% SVM starts here:
########################################################################################################################################
########################################################################################################################################
# ## Identify the best regularization parameter
#     Perform 10-fold cross validation to obtain the best regularization parameter
#         More specifically: "crossValidateModel" divides data into training and test datasets. It calls linearSVM.py, which does linear SVM using XTrain, and returns percent class loss for XTrain and XTest.
#     This procedure gets repeated for numSamples (100 times) for each value of regulariazation parameter. 
#     An average across all 100 samples is computed to find the minimum test class loss.
#     Best regularization parameter is defined as the smallest regularization parameter whose test-dataset class loss is within mean+sem of minimum test class loss.
########################################################################################################################################
########################################################################################################################################

def perClassError(Y, Yhat):
    import numpy as np
    perClassEr = np.sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
    return perClassEr
    
    
########## Training SVM to decode choice on stim-aligned traces ##########
    
# In[25]:

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
    cvect = 10**(np.arange(-4, 6,0.2));

print 'try the following regularization values: \n', cvect
# formattedList = ['%.2f' % member for member in cvect]
# print 'try the following regularization values = \n', formattedList

perClassErrorTrain = np.ones((numSamples, len(cvect)))+np.nan;
perClassErrorTest = np.ones((numSamples, len(cvect)))+np.nan;
wAllC = np.ones((numSamples, len(cvect), X.shape[1]))+np.nan;
bAllC = np.ones((numSamples, len(cvect)))+np.nan;
perClassErrorTest_incorr = np.ones((numSamples, len(cvect)))+np.nan;

for s in range(numSamples):
    for i in range(len(cvect)):
        if regType == 'l1':                       
            summary =  crossValidateModel(X, Y, linearSVM, kfold = kfold, l1 = cvect[i])
        elif regType == 'l2':
            summary =  crossValidateModel(X, Y, linearSVM, kfold = kfold, l2 = cvect[i])

        perClassErrorTrain[s, i] = summary.perClassErrorTrain;
        perClassErrorTest[s, i] = summary.perClassErrorTest;        
        wAllC[s,i,:] = np.squeeze(summary.model.coef_); # weights of all neurons for each value of c and each shuffle
        bAllC[s,i] = np.squeeze(summary.model.intercept_);

        # compute classError for incorrect trials (using the decoder trained on correct trials)
        #linear_svm.fit(XTrain, np.squeeze(YTrain))    # # linear_svm is trained using Xtrain and Ytrain... #train using Xtrain_corr... test on Xtest_corr and X_incorr.
        #summary.model = linear_svm;
        perClassErrorTest_incorr[s, i] = perClassError(Y_incorr, summary.model.predict(X_incorr));


meanPerClassErrorTest_incorr = np.mean(perClassErrorTest_incorr, axis = 0);
semPerClassErrorTest_incorr = np.std(perClassErrorTest_incorr, axis = 0)/np.sqrt(numSamples);


# compute averages across shuffles
meanPerClassErrorTrain = np.mean(perClassErrorTrain, axis = 0);
semPerClassErrorTrain = np.std(perClassErrorTrain, axis = 0)/np.sqrt(numSamples);

# Pick bestc from all range of c... it may end up be a value at which all weights are 0.
meanPerClassErrorTest = np.mean(perClassErrorTest, axis = 0);
semPerClassErrorTest = np.std(perClassErrorTest, axis = 0)/np.sqrt(numSamples);
ix = np.argmin(meanPerClassErrorTest)
cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
cbest = cbest[0]; # best regularization term based on minError+SE criteria
cbestAll = cbest

# Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)
a = (wAllC!=0) # non-zero weights
b = np.mean(a, axis=(0,2)) # Fraction of non-zero weights (averaged across shuffles)
c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle

cvectnow = cvect[c1stnon0:]
meanPerClassErrorTestnow = np.mean(perClassErrorTest[:,c1stnon0:], axis = 0);
semPerClassErrorTestnow = np.std(perClassErrorTest[:,c1stnon0:], axis = 0)/np.sqrt(numSamples);
ix = np.argmin(meanPerClassErrorTestnow)
cbest = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
cbest = cbest[0]; # best regularization term based on minError+SE criteria


#%% plot C path

if doPlots:
    print 'Best c (inverse of regularization parameter) = %.2f' %cbest
    plt.figure('cross validation')
    plt.subplot(1,2,1)
    plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
    plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
    plt.fill_between(cvect, meanPerClassErrorTest_incorr-semPerClassErrorTest_incorr, meanPerClassErrorTest_incorr+ semPerClassErrorTest_incorr, alpha=0.5, edgecolor='g', facecolor='g')
    plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
    plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation')
    plt.plot(cvect, meanPerClassErrorTest_incorr, 'g', label = 'incorr')        
    plt.plot(cvect[cvect==cbest], meanPerClassErrorTest[cvect==cbest], 'bo')
    plt.xlim([cvect[1], cvect[-1]])
    plt.xscale('log')
    plt.xlabel('c (inverse of regularization parameter)')
    plt.ylabel('classification error (%)')
    plt.legend(loc='center left', bbox_to_anchor=(1, .7))
    plt.tight_layout()


# ## Train SVM model using the best regularization parameter
#     All data in X are used for training.
#     linear_svm is the trained SVM model that includes weights (w) and intercept (b).

#%% I don't think you need the following
'''
if regType == 'l1':
    linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l1', dual=False)
elif regType == 'l2':
    linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l2', dual=True)

linear_svm.fit(X, Y)
    
w = np.squeeze(linear_svm.coef_);
b = linear_svm.intercept_;

trainE = abs(linear_svm.predict(X)-Y.astype('float')).sum()/len(Y)*100;

# keep a copy of linear_svm
import copy
linear_svm_0 = copy.deepcopy(linear_svm) 


#%% Plot weights

if doPlots:
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(w[np.argsort(abs(w))[::-1]], 'k.', label = 'weights')
    plt.plot(np.ones(len(w))*b, 'k--', label = 'bias')
    plt.xlabel('sorted neurons')
    plt.legend()
    plt.title(('Training error: %.2f %%' %(trainE)))

    plt.subplot(1,2,2)
    plt.hist(w, 20,orientation='horizontal', color = 'k')
    plt.ylabel('weights')
    plt.xlabel('count')
    plt.tight_layout()

    # print abs(((np.dot(X,w)+b)>0).astype('float')-Y.astype('float')).sum()/len(Y)*100 # this is the prediction formula
    print 'Fraction of non-zero weight neurons = %.2f' %(np.mean(w!=0))
    print 'Training error = %.2f%%' %trainE
'''


#%%
####################################################################################################################################
# ## Do cross-validation and set null distributions for classification error by shuffling trial labels
#     Compute distritbutions of class loss for train and test datasets by fitting SVM for 100 times.
#     Do this for both actual data and shuffled data (ie data in which Y is shuffled but X is not to serve as null distribution.)
# 
# Below we call function crossValidateModel.py; It does the following:
#1. shuffles trials to break any dependencies on the sequence of trails 
#2. divides data to training and testin sets
#3. calls linearSVM.py which does the following:
#3.1. Create a SVM classifier using cbest as the regularization paramter.
#3.2. Computes classficiation error on the training and testing datasets.
####################################################################################################################################


#%% Set class errors for data (at bestc)

# you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
# we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
indBestC = np.in1d(cvect, cbest)
perClassErrorTrain_data = perClassErrorTrain[:,indBestC].squeeze()
perClassErrorTest_data = perClassErrorTest[:,indBestC].squeeze()
w_data = wAllC[:,indBestC,:].squeeze()
b_data = bAllC[:,indBestC]
perClassErrorTest_data_incorr = perClassErrorTest_incorr[:,indBestC].squeeze()


#%% Set shuffled (null) distributions of class error by shuffling trial label of correct trials to train.

numShuffles = numSamples #100
#summary_data = [];
#perClassErrorTrain_data = [];
#perClassErrorTest_data = []
#w_data = []
#b_data = []
summary_shfl = [];
perClassErrorTrain_shfl = [];
perClassErrorTest_shfl = [];
w_shfl = []
b_shfl = []
permIxsList = [];
for i in range(numShuffles):
    # permIxs = rng.permutation(numTrials);
    permIxs = rng.permutation(numDataPoints);
    permIxsList.append(permIxs);
    if regType == 'l1':
#        summary_data.append(crossValidateModel(X, Y, linearSVM, kfold = kfold, l1 = cbest))
        summary_shfl.append(crossValidateModel(X, Y[permIxs], linearSVM, kfold = kfold, l1 = cbest))
    elif regType == 'l2':
#        summary_data.append(crossValidateModel(X, Y, linearSVM, kfold = kfold, l2 = cbest))
        summary_shfl.append(crossValidateModel(X, Y[permIxs], linearSVM, kfold = kfold, l2 = cbest))
        
#    perClassErrorTrain_data.append(summary_data[i].perClassErrorTrain);
#    perClassErrorTest_data.append(summary_data[i].perClassErrorTest);
#    w_data.append(np.squeeze(summary_data[i].model.coef_));
#    b_data.append(summary_data[i].model.intercept_);
        
    perClassErrorTrain_shfl.append(summary_shfl[i].perClassErrorTrain);
    perClassErrorTest_shfl.append(summary_shfl[i].perClassErrorTest);
    w_shfl.append(np.squeeze(summary_shfl[i].model.coef_));
    b_shfl.append(summary_shfl[i].model.intercept_);

    
#%% Plot class error for data and shuffled (for corr trained, corr cv, incorr cv)

pvalueTrain = ttest2(perClassErrorTrain_data, perClassErrorTrain_shfl, tail = 'left');
pvalueTest = ttest2(perClassErrorTest_data, perClassErrorTest_shfl, tail = 'left');
pvalueTest_incorr = ttest2(perClassErrorTest_data_incorr, perClassErrorTest_shfl, tail = 'left');

print 'Training error: Mean actual: %.2f%%, Mean shuffled: %.2f%%, p-value = %.2f' %(np.mean(perClassErrorTrain_data), np.mean(perClassErrorTrain_shfl), pvalueTrain)
print 'Testing error: Mean actual: %.2f%%, Mean shuffled: %.2f%%, p-value = %.2f' %(np.mean(perClassErrorTest_data), np.mean(perClassErrorTest_shfl), pvalueTest)
print 'Incorr trials: error: Mean actual: %.2f%%, Mean shuffled (corr): %.2f%%, p-value = %.2f' %(np.mean(perClassErrorTest_data_incorr), np.mean(perClassErrorTest_shfl), pvalueTest)


# Plot the histograms
if doPlots:
    binEvery = 3; # bin width
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(perClassErrorTrain_data, np.arange(0,100,binEvery), color = 'k', label = 'data');
    plt.hist(perClassErrorTrain_shfl, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
    plt.xlabel('Training classification error (%)')
    plt.ylabel('count')
    plt.title('Mean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(perClassErrorTrain_data), np.mean(perClassErrorTrain_shfl), pvalueTrain), fontsize = 10)
    plt.legend()

    plt.subplot(1,2,2)
    plt.hist(perClassErrorTest_data, np.arange(0,100,binEvery), color = 'k', label = 'data');
    plt.hist(perClassErrorTest_shfl, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
    plt.legend()
    plt.xlabel('Testing classification error (%)')
    plt.title('Mean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(perClassErrorTest_data), np.mean(perClassErrorTest_shfl), pvalueTest), fontsize = 10)
    plt.ylabel('count')
    plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)

    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(perClassErrorTest_data_incorr, np.arange(0,100,binEvery), color = 'k', label = 'data');
    plt.hist(perClassErrorTest_shfl, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
    plt.legend()
    plt.xlabel('Testing classification error (%)')
    plt.title('Mean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(perClassErrorTest_data_incorr), np.mean(perClassErrorTest_shfl), pvalueTest), fontsize = 10)
    plt.ylabel('count')
    plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)



#%%
####################################################################################
########## Training SVM to decode choice on choice-aligned traces ##########
####################################################################################
    
#%% lets try bagging too
nRandTrSamp = 10
    
    

#%% start subselecting trials (for BAGGING)
# choose n random trials (n = 80% of min of HR, LR)
nrtr = int(np.ceil(.8 * min((Y_chAl==0).sum(), (Y_chAl==1).sum())))
print 'selecting %d random trials of each category' %(nrtr)


#%% Train SVM on each frame using BAGGING (subselection of trials)

#nRandTrSamp = 10 #1000 # number of times to subselect trials (for the BAGGING method)

if doData:
    # data
    w_chAl = np.full((X_chAl.shape[0], X_chAl.shape[1], nRandTrSamp), np.nan) # frames x neurons x nrandtr
    b_chAl =  np.full((X_chAl.shape[0], nRandTrSamp), np.nan) # frames x nrandtr
    trainE_chAl =  np.full((X_chAl.shape[0], nRandTrSamp), np.nan) # frames x nrandtr
        
    for ir in range(nRandTrSamp): # subselect a group of trials
        print 'BAGGING iteration %d' %(ir)
        
        # subselect nrtr random HR and nrtr random LR trials.
        lrTrs = np.argwhere(Y_chAl==0)
        randLrTrs = rng.permutation(lrTrs)[0:nrtr].flatten() # random LR trials
    
        hrTrs = np.argwhere(Y_chAl==1)
        randHrTrs = rng.permutation(hrTrs)[0:nrtr].flatten() # random HR trials
        
        randTrs = np.sort(np.concatenate((randLrTrs, randHrTrs)))
        
        
        X = X_chAl[:,:,randTrs]
        Y = Y_chAl[randTrs]
    
    
        ##%% train SVM on each frame using the sebselect of trials
        '''
        if regType == 'l1':
            linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l1', dual=False)
        elif regType == 'l2':
            linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l2', dual=True)
        '''
        
        linear_svm = svm.LinearSVC(C=10.) # default c, also l2
        
        # train SVM on each frame
        for ifr in range(np.shape(X)[0]):
            
            # data
            linear_svm.fit(X[ifr,:,:].transpose(), Y)
                
            w_chAl[ifr,:, ir] = np.squeeze(linear_svm.coef_);
            b_chAl[ifr, ir] = linear_svm.intercept_;        
            trainE_chAl[ifr, ir] = abs(linear_svm.predict(X[ifr,:,:].transpose())-Y.astype('float')).sum()/len(Y)*100;


#%%

# numSamples = 10; # number of iterations for finding the best c (inverse of regularization parameter)
# if you don't want to regularize, go with a very high cbest and don't run the section below.
# cbest = 10**6

# regType = 'l1'
# kfold = 10;
if regType == 'l1':
    print '\nRunning l1 svm classification\r' 
    # cvect = 10**(np.arange(-4, 6,0.2))/numTrials;
#    cvect = 10**(np.arange(-4, 6,0.2))/X_chAl.shape[0];
    cvect = 10**(np.arange(-4, 3.11e2,10.5))/X_chAl.shape[0];
elif regType == 'l2':
    print '\nRunning l2 svm classification\r' 
    cvect = 10**(np.arange(-4, 6,0.2));

print 'try the following regularization values: \n', cvect
# formattedList = ['%.2f' % member for member in cvect]
# print 'try the following regularization values = \n', formattedList

perClassErrorTrain_chAl = np.ones((numSamples, len(cvect)))+np.nan;
perClassErrorTest_chAl = np.ones((numSamples, len(cvect)))+np.nan;
wAllC_chAl = np.ones((numSamples, len(cvect), X_chAl.shape[1]))+np.nan;
bAllC_chAl = np.ones((numSamples, len(cvect)))+np.nan;
perClassErrorTest_incorr_chAl = np.ones((numSamples, len(cvect)))+np.nan;
perClassErrorTest_incorr_chAl_shfl = np.ones((numSamples, len(cvect)))+np.nan;

perClassErrorTrain_shfl_chAl = np.ones((numSamples, len(cvect)))+np.nan;
perClassErrorTest_shfl_chAl = np.ones((numSamples, len(cvect)))+np.nan;

perClassErrorTest_corr_shfl_chAl = np.ones((numSamples, len(cvect)))+np.nan;


for s in range(numSamples):
    print 'iteration %d' %(s)
    permIxs_incorr = rng.permutation(X_chAl_incorr.shape[0]);
    permIxs = rng.permutation(X_chAl.shape[0]);

#    a = np.zeros(Y_chAl_incorr.shape[0])
#    b = rng.permutation(Y_chAl_incorr.shape[0])[0:Y_chAl_incorr.shape[0]/2]
#    a[b] = 1
    
    for i in range(len(cvect)):
        
        if regType == 'l1':                       
            summary =  crossValidateModel(X_chAl, Y_chAl, linearSVM, kfold = kfold, l1 = cvect[i])
        elif regType == 'l2':
            summary =  crossValidateModel(X_chAl, Y_chAl, linearSVM, kfold = kfold, l2 = cvect[i])

#        perClassErrorTest_corr_shfl_chAl[s, i] = perClassError(Y_chAl[permIxs], summary.model.predict(X_chAl));
        
        
        perClassErrorTrain_chAl[s, i] = summary.perClassErrorTrain;
        perClassErrorTest_chAl[s, i] = summary.perClassErrorTest;        
        wAllC_chAl[s,i,:] = np.squeeze(summary.model.coef_); # weights of all neurons for each value of c and each shuffle
        bAllC_chAl[s,i] = np.squeeze(summary.model.intercept_);

        # compute classError for incorrect trials (using the decoder trained on correct trials)
        #linear_svm.fit(XTrain, np.squeeze(YTrain))    # # linear_svm is trained using Xtrain and Ytrain... #train using Xtrain_corr... test on Xtest_corr and X_incorr.
        #summary.model = linear_svm;
        perClassErrorTest_incorr_chAl[s, i] = perClassError(Y_chAl_incorr, summary.model.predict(X_chAl_incorr));
        
        # Null distribution for class error on incorr trials: 
        # We want to use the same decoder (trained on correct trials) to predict the choice on incorrect trials with shuffled class labels.             
        perClassErrorTest_incorr_chAl_shfl[s, i] = perClassError(Y_chAl_incorr[permIxs_incorr], summary.model.predict(X_chAl_incorr));
#        perClassErrorTest_incorr_chAl_shfl[s, i] = perClassError(a, summary.model.predict(X_chAl_incorr));
        


#        summary_shfl = crossValidateModel(X_chAl_incorr, Y_chAl_incorr[permIxs], linearSVM, kfold = kfold, l1 = cvect[i])
        summary_shfl = crossValidateModel(X_chAl, Y_chAl[permIxs], linearSVM, kfold = kfold, l1 = cvect[i])
        perClassErrorTrain_shfl_chAl[s, i] = summary_shfl.perClassErrorTrain;
        perClassErrorTest_shfl_chAl[s, i] = summary_shfl.perClassErrorTest;
        

meanPerClassErrorTest_corr_shfl = np.mean(perClassErrorTest_corr_shfl_chAl, axis = 0);
semPerClassErrorTest_corr_shfl = np.std(perClassErrorTest_corr_shfl_chAl, axis = 0)/np.sqrt(numSamples);


meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl_chAl, axis = 0);
semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl_chAl, axis = 0)/np.sqrt(numSamples);

meanPerClassErrorTrain_shfl = np.mean(perClassErrorTrain_shfl_chAl, axis = 0);
semPerClassErrorTrain_shfl = np.std(perClassErrorTrain_shfl_chAl, axis = 0)/np.sqrt(numSamples);


# compute averages across shuffles        
meanPerClassErrorTest_incorr = np.mean(perClassErrorTest_incorr_chAl, axis = 0);
semPerClassErrorTest_incorr = np.std(perClassErrorTest_incorr_chAl, axis = 0)/np.sqrt(numSamples);

meanPerClassErrorTest_incorr_shfl = np.mean(perClassErrorTest_incorr_chAl_shfl, axis = 0);
semPerClassErrorTest_incorr_shfl = np.std(perClassErrorTest_incorr_chAl_shfl, axis = 0)/np.sqrt(numSamples);


meanPerClassErrorTrain = np.mean(perClassErrorTrain_chAl, axis = 0);
semPerClassErrorTrain = np.std(perClassErrorTrain_chAl, axis = 0)/np.sqrt(numSamples);

meanPerClassErrorTest = np.mean(perClassErrorTest_chAl, axis = 0);
semPerClassErrorTest = np.std(perClassErrorTest_chAl, axis = 0)/np.sqrt(numSamples);

# Identify best c : smallest c whose CV error falls below 1 se of min CV error
# Use all range of c... it may end up a value at which all weights are 0.
ix = np.argmin(meanPerClassErrorTest)
cbest_chAl = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
cbest_chAl = cbest_chAl[0]; # best regularization term based on minError+SE criteria
cbestAll_chAl = cbest_chAl

# Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)
a = (wAllC_chAl!=0) # non-zero weights
b = np.mean(a, axis=(0,2)) # Fraction of non-zero weights (averaged across shuffles)
c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle

cvectnow = cvect[c1stnon0:]
meanPerClassErrorTestnow = np.mean(perClassErrorTest_chAl[:,c1stnon0:], axis = 0);
semPerClassErrorTestnow = np.std(perClassErrorTest_chAl[:,c1stnon0:], axis = 0)/np.sqrt(numSamples);
ix = np.argmin(meanPerClassErrorTestnow)
cbest_chAl = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
cbest_chAl = cbest_chAl[0]; # best regularization term based on minError+SE criteria


#%% plot C path
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
    
    
if doPlots:
    print 'Best c (inverse of regularization parameter) = %.2f' %cbest_chAl
    plt.figure('cross validation')
    plt.subplot(1,2,1)
    plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
    plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
    plt.fill_between(cvect, meanPerClassErrorTest_incorr-semPerClassErrorTest_incorr, meanPerClassErrorTest_incorr+ semPerClassErrorTest_incorr, alpha=0.5, edgecolor='g', facecolor='g')        
    plt.fill_between(cvect, meanPerClassErrorTest_incorr_shfl-semPerClassErrorTest_incorr_shfl, meanPerClassErrorTest_incorr_shfl+ semPerClassErrorTest_incorr_shfl, alpha=0.5, edgecolor='c', facecolor='c')        
    plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
    plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation')
    plt.plot(cvect, meanPerClassErrorTest_incorr, 'g', label = 'incorr')        
    plt.plot(cvect, meanPerClassErrorTest_incorr_shfl, 'c', label = 'incorr-shfl')            
    plt.plot(cvect[cvect==cbest_chAl], meanPerClassErrorTest[cvect==cbest_chAl], 'bo')
    plt.xlim([cvect[1], cvect[-1]])
    plt.xscale('log')
    plt.xlabel('c (inverse of regularization parameter)')
    plt.ylabel('classification error (%)')
    plt.legend(loc='center left', bbox_to_anchor=(1, .7))
    plt.tight_layout()


# ## Train SVM model using the best regularization parameter
#     All data in X are used for training.
#     linear_svm is the trained SVM model that includes weights (w) and intercept (b).


#%% I don't think you need the following
'''
if regType == 'l1':
    linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l1', dual=False)
elif regType == 'l2':
    linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l2', dual=True)

linear_svm.fit(X, Y)
    
w = np.squeeze(linear_svm.coef_);
b = linear_svm.intercept_;

trainE = abs(linear_svm.predict(X)-Y.astype('float')).sum()/len(Y)*100;

# keep a copy of linear_svm
import copy
linear_svm_0 = copy.deepcopy(linear_svm) 


#%% Plot weights

if doPlots:
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(w[np.argsort(abs(w))[::-1]], 'k.', label = 'weights')
    plt.plot(np.ones(len(w))*b, 'k--', label = 'bias')
    plt.xlabel('sorted neurons')
    plt.legend()
    plt.title(('Training error: %.2f %%' %(trainE)))

    plt.subplot(1,2,2)
    plt.hist(w, 20,orientation='horizontal', color = 'k')
    plt.ylabel('weights')
    plt.xlabel('count')
    plt.tight_layout()

    # print abs(((np.dot(X,w)+b)>0).astype('float')-Y.astype('float')).sum()/len(Y)*100 # this is the prediction formula
    print 'Fraction of non-zero weight neurons = %.2f' %(np.mean(w!=0))
    print 'Training error = %.2f%%' %trainE
'''


#%%
####################################################################################################################################
# ## Do cross-validation and set null distributions for classification error by shuffling trial labels
#     Compute distritbutions of class loss for train and test datasets by fitting SVM for 100 times.
#     Do this for both actual data and shuffled data (ie data in which Y is shuffled but X is not to serve as null distribution.)
# 
# Below we call function crossValidateModel.py; It does the following:
#1. shuffles trials to break any dependencies on the sequence of trails 
#2. divides data to training and testin sets
#3. calls linearSVM.py which does the following:
#3.1. Create a SVM classifier using cbest as the regularization paramter.
#3.2. Computes classficiation error on the training and testing datasets.

####################################################################################################################################


#%% Set class errors for data (at bestc)

# you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
# we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
indBestC = np.in1d(cvect, cbest_chAl)
perClassErrorTrain_data_chAl = perClassErrorTrain_chAl[:,indBestC].squeeze()
perClassErrorTest_data_chAl = perClassErrorTest_chAl[:,indBestC].squeeze()
w_data_chAl = wAllC_chAl[:,indBestC,:].squeeze()
b_data_chAl = bAllC_chAl[:,indBestC]
perClassErrorTest_data_incorr_chAl = perClassErrorTest_incorr_chAl[:,indBestC].squeeze()
perClassErrorTest_shfl_incorr_chAl = perClassErrorTest_incorr_chAl_shfl[:,indBestC].squeeze()

pvalueTest_incorr = ttest2(perClassErrorTest_data_incorr_chAl, perClassErrorTest_shfl_incorr_chAl, tail = 'left');


#%% Set shuffled (null) distributions of class error by shuffling trial label of correct trials to train.

# We want to use the same decoder (trained on correct trials) to predict the choice on incorrect trials with shuffled class labels.

# Gamal:
# Our question is: given a known decoder (the decoder we obtain from the correct trials), does this decoder 
#perform well on incorrect trials? My definition of performing well is relative to some random data 
#that have the same firing rates pattern as incorrect data but with arbitrary relation to the class labels. 
#According to that question, we should compare to shuffled incorrect trials without refitting the decoder as our question is specific to the decoder trained on the correct data.

numShuffles = numSamples #100
#summary_data = [];
#perClassErrorTrain_data = [];
#perClassErrorTest_data = []
#w_data = []
#b_data = []
summary_shfl = [];
perClassErrorTrain_shfl_chAl = [];
perClassErrorTest_shfl_chAl = [];
w_shfl_chAl = []
b_shfl_chAl = []
permIxsList = [];
for i in range(numShuffles):
    # permIxs = rng.permutation(numTrials);
    permIxs = rng.permutation(X_chAl.shape[0]);
    permIxsList.append(permIxs);
    if regType == 'l1':
#        summary_data.append(crossValidateModel(X, Y, linearSVM, kfold = kfold, l1 = cbest))
        summary_shfl.append(crossValidateModel(X_chAl, Y_chAl[permIxs], linearSVM, kfold = kfold, l1 = cbest))
    elif regType == 'l2':
#        summary_data.append(crossValidateModel(X, Y, linearSVM, kfold = kfold, l2 = cbest))
        summary_shfl.append(crossValidateModel(X_chAl, Y_chAl[permIxs], linearSVM, kfold = kfold, l2 = cbest))
        
#    perClassErrorTrain_data.append(summary_data[i].perClassErrorTrain);
#    perClassErrorTest_data.append(summary_data[i].perClassErrorTest);
#    w_data.append(np.squeeze(summary_data[i].model.coef_));
#    b_data.append(summary_data[i].model.intercept_);
        
    perClassErrorTrain_shfl_chAl.append(summary_shfl[i].perClassErrorTrain);
    perClassErrorTest_shfl_chAl.append(summary_shfl[i].perClassErrorTest);
    w_shfl_chAl.append(np.squeeze(summary_shfl[i].model.coef_));
    b_shfl_chAl.append(summary_shfl[i].model.intercept_);

    
#%% Plot class error for data and shuffled (for corr trained, corr cv, incorr cv)

pvalueTrain = ttest2(perClassErrorTrain_data_chAl, perClassErrorTrain_shfl_chAl, tail = 'left');
pvalueTest = ttest2(perClassErrorTest_data_chAl, perClassErrorTest_shfl_chAl, tail = 'left');
pvalueTest_incorr = ttest2(perClassErrorTest_data_incorr_chAl, perClassErrorTest_shfl_chAl, tail = 'left');

print 'Training error: Mean actual: %.2f%%, Mean shuffled: %.2f%%, p-value = %.2f' %(np.mean(perClassErrorTrain_data_chAl), np.mean(perClassErrorTrain_shfl_chAl), pvalueTrain)
print 'Testing error: Mean actual: %.2f%%, Mean shuffled: %.2f%%, p-value = %.2f' %(np.mean(perClassErrorTest_data_chAl), np.mean(perClassErrorTest_shfl_chAl), pvalueTest)
print 'Incorr trials: error: Mean actual: %.2f%%, Mean shuffled (corr): %.2f%%, p-value = %.2f' %(np.mean(perClassErrorTest_data_incorr_chAl), np.mean(perClassErrorTest_shfl_chAl), pvalueTest)


# Plot the histograms
if doPlots:
    binEvery = 3; # bin width
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(perClassErrorTrain_data_chAl, np.arange(0,100,binEvery), color = 'k', label = 'data');
    plt.hist(perClassErrorTrain_shfl_chAl, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
    plt.xlabel('Training classification error (%)')
    plt.ylabel('count')
    plt.title('Mean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(perClassErrorTrain_data_chAl), np.mean(perClassErrorTrain_shfl_chAl), pvalueTrain), fontsize = 10)
    plt.legend()

    plt.subplot(1,2,2)
    plt.hist(perClassErrorTest_data_chAl, np.arange(0,100,binEvery), color = 'k', label = 'data');
    plt.hist(perClassErrorTest_shfl_chAl, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
    plt.legend()
    plt.xlabel('Testing classification error (%)')
    plt.title('Mean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(perClassErrorTest_data_chAl), np.mean(perClassErrorTest_shfl_chAl), pvalueTest), fontsize = 10)
    plt.ylabel('count')
    plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)

    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(perClassErrorTest_data_incorr_chAl, np.arange(0,100,binEvery), color = 'k', label = 'data');
    plt.hist(perClassErrorTest_shfl_chAl, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
    plt.legend()
    plt.xlabel('Testing classification error (%)')
    plt.title('Mean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(perClassErrorTest_data_incorr_chAl), np.mean(perClassErrorTest_shfl_chAl), pvalueTest), fontsize = 10)

    plt.ylabel('count')
    plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)



#%%
####################################################################################################################################
############################################ Save results as .mat files in a folder named svm ####################################################################################################################################
####################################################################################################################################


#%% Save results

if trialHistAnalysis:
#     ep_ms = np.round((ep-eventI)*frameLength)
    th_stim_dur = []
    svmn = 'svmPrevChoice_%sN_%sITIs_ep%d-%dms_%s' %(ntName, itiName, ep_ms[0], ep_ms[-1], rt)
else:
    svmn = 'svmCurrChoice_%sN_ep%d-%dms_%s' %(ntName, ep_ms[0], ep_ms[-1], rt)   
print '\n', svmn[:-1]

if saveResults:
    print 'Saving .mat file'
    d = os.path.join(os.path.dirname(pnevFileName), 'svm')
    if not os.path.exists(d):
        print 'creating svm folder'
        os.makedirs(d)

    svmName = os.path.join(d, svmn+os.path.basename(pnevFileName))
    print(svmName)
    
    scio.savemat(svmName, {'thAct':thAct, 'thTrsWithSpike':thTrsWithSpike, 'ep_ms':ep_ms, 
                           'th_stim_dur':th_stim_dur, 'numSamples':numSamples, 
                           'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 
                           'NsRand':NsRand, 'meanX':meanX, 'stdX':stdX, 
                           'regType':regType, 'cvect':cvect, 'cbest':cbest, 'cbestAll':cbestAll,
                           'perClassErrorTest':perClassErrorTest, 'perClassErrorTrain':perClassErrorTrain, 'wAllC':wAllC, 'bAllC':bAllC,
                           'w':w, 'b':b, 'trainE':trainE, 
                           'corrClass':corrClass, 'corrClassShfl':corrClassShfl,
                           'perClassErrorTrain_data':perClassErrorTrain_data, 
                           'perClassErrorTrain_shfl':perClassErrorTrain_shfl, 
                           'perClassErrorTest_data':perClassErrorTest_data, 
                           'perClassErrorTest_shfl':perClassErrorTest_shfl,  
                           'w_data':w_data,'b_data':b_data,'w_shfl':w_shfl,'b_shfl':b_shfl,
                           'perActive_inh':perActive_inh, 'perActive_exc':perActive_exc, # vars related to exc, inh start from this line.
                           'wei_all':wei_all, 'perClassEr':perClassEr, 
                           'perActive_inh_allExc':perActive_inh_allExc, 'perActive_exc_allExc':perActive_exc_allExc, 
                           'perClassEr_allExc':perClassEr_allExc, 'wei_all_allExc':wei_all_allExc, 'cvect_':cvect_,
                           'train_err_exc0':train_err_exc0, 'train_err_inh0':train_err_inh0, 
                           'corrClass_exc0':corrClass_exc0, 'corrClass_inh0':corrClass_inh0, 
                           'train_err_allExc0':train_err_allExc0, 'corrClass_allExc0':corrClass_allExc0, 
                           'perClassErrorTrain_data_inh':perClassErrorTrain_data_inh, 
                           'perClassErrorTest_data_inh':perClassErrorTest_data_inh, 
                           'perClassErrorTrain_shfl_inh':perClassErrorTrain_shfl_inh, 
                           'perClassErrorTest_shfl_inh':perClassErrorTest_shfl_inh, 
                           'perClassErrorTrain_data_allExc':perClassErrorTrain_data_allExc, 
                           'perClassErrorTest_data_allExc':perClassErrorTest_data_allExc, 
                           'perClassErrorTrain_shfl_allExc':perClassErrorTrain_shfl_allExc, 
                           'perClassErrorTest_shfl_allExc':perClassErrorTest_shfl_allExc, 
                           'perClassErrorTrain_data_exc':perClassErrorTrain_data_exc, 
                           'perClassErrorTest_data_exc':perClassErrorTest_data_exc, 
                           'perClassErrorTrain_shfl_exc':perClassErrorTrain_shfl_exc, 
                           'perClassErrorTest_shfl_exc':perClassErrorTest_shfl_exc, 
                           'corrClass_allExc':corrClass_allExc, 'corrClass_exc':corrClass_exc, 'corrClass_inh':corrClass_inh,
                           'w_data_inh':w_data_inh,'b_data_inh':b_data_inh,'w_shfl_inh':w_shfl_inh,'b_shfl_inh':b_shfl_inh,
                           'w_data_allExc':w_data_allExc,'b_data_allExc':b_data_allExc,'w_shfl_allExc':w_shfl_allExc,'b_shfl_allExc':b_shfl_allExc,
                           'w_data_exc':w_data_exc,'b_data_exc':b_data_exc,'w_shfl_exc':w_shfl_exc,'b_shfl_exc':b_shfl_exc}) 


    # save normalized traces as well                       
    # scio.savemat(svmName, {w':w, 'b':b, 'cbest':cbest, 'corrClass':corrClass, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'meanX':meanX, 'stdX':stdX, 'X':X, 'Y':Y, 'Xt':Xt, 'Xtg':Xtg, 'Xtc':Xtc, 'Xtr':Xtr, 'Xtp':Xtp})
    # 'linear_svm':linear_svm, 

    # append : doesn't quite work
    # if os.path.isfile(svmName): 
    #     with open(svmName,'ab') as f:
    #         sci.io.savemat(f, {'perClassErrorTrain_data':perClassErrorTrain_data, 'perClassErrorTrain_shfl':perClassErrorTrain_shfl, 'perClassErrorTest_data':perClassErrorTest_data, 'perClassErrorTest_shfl':perClassErrorTest_shfl}) # append
    # else:
else:
    print 'Not saving .mat file'
    
    
# If you want to save the linear_svm objects as mat file, you need to take care of 
# None values and set to them []:

# import inspect
# inspect.getmembers(summary_shfl_exc[0])

# for i in range(np.shape(summary_shfl_exc)[0]):
#     summary_shfl_exc[i].model.random_state = []
#     summary_shfl_exc[i].model.class_weight = []
# [summary_shfl_exc[i].model.random_state for i in range(np.shape(summary_shfl_exc)[0])]
# [summary_shfl_exc[i].model.class_weight for i in range(np.shape(summary_shfl_exc)[0])]

# scio.savemat(svmName, {'summary_shfl_exc':summary_shfl_exc})

# Data = scio.loadmat(svmName, variable_names=['summary_shfl_exc'] ,squeeze_me=True,struct_as_record=False)
# summary_shfl_exc = Data.pop('summary_shfl_exc')
# summary_shfl_exc[0].model.intercept_
    


# ## Move the autosaved .html file to a folder named "figs"
#     Notebook html file will be autosaved in the notebook directory if jupyter_notebook_config.py exists in ~/.jupyter and includes the function script_post_save. Below we move the html to a directory named figs inside the root directory which contains moreFile, etc.

# In[45]:

# make sure autosave is done so you move the most recent html file to figs directory.
if 'ipykernel' in sys.modules and saveHTML:
    get_ipython().magic(u'autosave 1')
    # import time    
    # time.sleep(2) 
    
    d = os.path.join(os.path.dirname(pnevFileName),'figs')
    if not os.path.exists(d):
        print 'creating figs folder'
        os.makedirs(d)

    htmlName = os.path.join(d, svmn[:-1]+'.html')
    print htmlName
    import shutil
    shutil.move(os.path.join(os.getcwd(), 'mainSVM_notebook.html'), htmlName)

    # go back to default autosave interval
    get_ipython().magic(u'autosave 120')

