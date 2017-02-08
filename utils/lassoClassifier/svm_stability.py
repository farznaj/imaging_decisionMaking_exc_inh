# This is the main SVM script that includes all analyses (training SVM, comparing exc, inh etc.... )
# For comparing exc, inh you have svm_excInh_cPath which is more extensive ... use that.
# This script should be run on the cluster.


# coding: utf-8

# ## Specify variables for the analysis: 
#     - Data (mouse, day, sessions)
#     - Neuron type: excit, inhibit, or all
#     - Current-choice or previous-choice SVM training
#         if current-choice, specify epoch of training, the outcome (corr, incorr, all) and strength (easy, medium, hard, all) of trials for training SVM.
#         if previous-choice, specify ITI flag
#     - Trials that will be used for projections and class accuracy traces (corr, incorr, all, trained).


#%%
import numpy as np
frameLength = 1000/30.9; # sec.

regressBins = int(np.round(100/frameLength)) # 100ms # set to nan if you don't want to downsample.
#regressBins = 2 # number of frames to average for downsampling X

doData = 1 # whether to run the analysis on data (ie not shuffling trial labels)
doShuffles = 0 # whether to run the analysis on trial-label shuffles to get null distributions


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
    numSamples = 2 #100; # number of iterations for finding the best c (inverse of regularization parameter)
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

#frameLength = 1000/30.9; # sec.
    
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
# Note: for this analysis (ie we train SVM on every frame), we dont need to set ep ... but below you also 
# exclude trials that have short stim duration (or that have choice before the end of ep)... so I leave it there.

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


    '''
    # Set ep
    if len(ep_ms)==0: # load ep from matlab
        # Load stimulus-aligned traces, frames, frame of event of interest, and epoch over which we will average the responses to do SVM analysis
        Data = scio.loadmat(postName, variable_names=['stimAl'],squeeze_me=True,struct_as_record=False)
        # eventI = Data['stimAl'].eventI - 1 # remember difference indexing in matlab and python!
        # traces_al_stim = Data['stimAl'].traces.astype('float') # traces_al_stim
        # time_aligned_stim = Data['stimAl'].time.astype('float')

        ep = Data['stimAl'].ep - 1
        ep_ms = np.round((ep-eventI)*frameLength).astype(int)
        
    else: # set ep here:
    '''        




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





#%%
###########################################################################################################################################
###########################################################################################################################################
# ## Set X (trials x neurons) and Y (trials x 1) for training the SVM classifier.
#     X matrix (size trials x neurons); steps include:
#        - average neural responses during ep for each trial.
#        - remove nan trials (trsExcluded)
#        - exclude non-active neurons (NsExcluded)
#        - [if doing doRands, get X only for NsRand]
#        - center and normalize X (by meanX and stdX)

#     Y choice of high rate (modeled as 1) and low rate (modeled as 0)


# In[13]:

# Set choiceVec0  (Y: the response vector)

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


# Set spikeAveEp0  (X: the predictor matrix (trials x neurons) that shows average of spikes for a particular epoch for each trial and neuron.)

if trialHistAnalysis:
    spikeAveEp0 = traces_al_stimAll
    # either of the two cases below should be fine (init-aligned traces or stim-aligned traces.)
#    spikeAveEp0 = np.transpose(np.nanmean(traces_al_stimAll[ep,:,:], axis=0)) # trials x neurons    
    # spikeAveEp0 = np.transpose(np.nanmean(traces_al_init[ep,:,:], axis=0)) # trials x neurons    
else:
    spikeAveEp0 = traces_al_stim    
#    spikeAveEp0 = np.transpose(np.nanmean(traces_al_stim[ep,:,:], axis=0)) # trials x neurons    

# X = spikeAveEp0;
#print 'Size of spikeAveEp0 (trs x neurons): ', spikeAveEp0.shape


# In[14]:

# set trsExcluded and exclude them to set X and Y; trsExcluded are trials that are nan either in traces or in choice vector.

'''
#dirName = 'SVM_151102_001-002_ch2-PnevPanResults-160624-113108';
dirName = 'SVM_151029_003_ch2-PnevPanResults-160426-191859';
#dirName = '/home/farznaj/Shares/Churchland/data/fni17/imaging/151022/XY_fni17_151022 XY_lassoSVM.mat';
Data = scio.loadmat(dirName, variable_names=['X', 'Y', 'time_aligned_stim', 'non_filtered', 'traces_al_1stSideTry', 'time_aligned_stim_1stSideTry']);
X = Data.pop('X').astype('float')
Y = np.squeeze(Data.pop('Y')).astype('int')
time_aligned_stim = np.squeeze(Data.pop('time_aligned_stim')).astype('float')
Xt = Data.pop('non_filtered').astype('float')
Xt_choiceAl = Data.pop('traces_al_1stSideTry').astype('float')
time_aligned_1stSide = np.squeeze(Data.pop('time_aligned_stim_1stSideTry')).astype('float')
'''

# Identify nan trials
#trsExcluded = (np.sum(np.isnan(spikeAveEp0), axis = 1) + np.isnan(choiceVec0)) != 0 # NaN trials # trsExcluded
trsExcluded = (np.isnan(np.sum(spikeAveEp0, axis=(0,1))) + np.isnan(choiceVec0)) != 0
# print sum(trsExcluded), 'NaN trials'

# Exclude nan trials
X = spikeAveEp0[:,:,~trsExcluded] #spikeAveEp0[~trsExcluded,:]; # trials x neurons
Y = choiceVec0[~trsExcluded];
print '%d high-rate choices, and %d low-rate choices\n' %(np.sum(Y==1), np.sum(Y==0))


#trsExcludedSr = (np.isnan(np.sum(spikeAveEp0, axis=(0,1)))) != 0 # for the stimulus decoder we dont care if trial was correct or incorrect
Ysr = stimrate[~trsExcluded] # we dont need to exclude trials that animal did not make a choice for the stimulus decoder analysis... unless you want to use the same set of trials for the choice and stimulus decoder...
Ysr = (Ysr - Ysr.mean()) / np.ptp(stimrate) # when you did not do this, angle_stim_sh (ie stimulus shuffled) gave you weird results... during baseline (~15 frames at the beginning and almost ~15 frames at the end) you got alignment between population responses in the shuffled data!
#Ysr = (Ysr - Ysr.mean()) / Ysr.std()
#Ysr = (Ysr-cb) / np.ptp(stimrate) # not sure if we need this, also stimrate in some days did not span the entire range (4-28hz)


#%% Set choice-aligned traces

# Load 1stSideTry-aligned traces, frames, frame of event of interest
# use firstSideTryAl_COM to look at changes-of-mind (mouse made a side lick without committing it)
Data = scio.loadmat(postName, variable_names=['firstSideTryAl'],squeeze_me=True,struct_as_record=False)
traces_al_1stSide = Data['firstSideTryAl'].traces.astype('float')
time_aligned_1stSide = Data['firstSideTryAl'].time.astype('float')
# print(np.shape(traces_al_1stSide))

# determine trsExcluded for traces_al_1stSide
# remember traces_al_1stSide can have some nan trials that are change-of-mind trials... and they wont be nan in choiceVec0
trsExcluded_choice = (np.isnan(np.sum(traces_al_1stSide, axis=(0,1))) + np.isnan(choiceVec0)) != 0

X_choice = traces_al_1stSide[:,:,~trsExcluded_choice]  
Y_choice = choiceVec0[~trsExcluded_choice];


#%% Downsample X: average across multiple times (downsampling, not a moving average. we only average every regressBins points.)

#regressBins = 2 # number of frames to average for downsampling X
#regressBins = int(np.round(100/frameLength)) # 100ms

if np.isnan(regressBins)==0: # set to nan if you don't want to downsample.
    print 'Downsampling traces ....'
    # X
    T1, N1, C1 = X.shape
    tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X
    X = X[0:regressBins*tt,:,:]
    #X_d.shape
    
    X = np.mean(np.reshape(X, (regressBins, tt, N1, C1), order = 'F'), axis=0)
    print 'downsampled stim-aligned trace: ', X.shape
        
    time_aligned_stim = time_aligned_stim[0:regressBins*tt]
    #time_aligned_stim.shape
    time_aligned_stim = np.round(np.mean(np.reshape(time_aligned_stim, (regressBins, tt), order = 'F'), axis=0), 2)
    #time_aligned_stim_d.shape
    
    
    # X_choice
    T1, N1, C1 = X_choice.shape
    tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X
    X_choice = X_choice[0:regressBins*tt,:,:]
    #X_choice_d.shape
    
    X_choice = np.mean(np.reshape(X_choice, (regressBins, tt, N1, C1), order = 'F'), axis=0)
    print 'downsampled choice-aligned trace: ', X_choice.shape
        
    time_aligned_1stSide = time_aligned_1stSide[0:regressBins*tt]
    #time_aligned_1stSide_d.shape
    time_aligned_1stSide = np.round(np.mean(np.reshape(time_aligned_1stSide, (regressBins, tt), order = 'F'), axis=0), 2)
    #time_aligned_1stSide_d.shape
else:
    print 'Now downsampling traces ....'


#%% Set NsExcluded : Identify neurons that did not fire in any of the trials (during ep) and then exclude them. Otherwise they cause problem for feature normalization.
# thAct and thTrsWithSpike are parameters that you can play with.

'''
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
'''    
if trialHistAnalysis and iTiFlg!=2:
    # set X for short-ITI and long-ITI cases (XS,XL).
#        trsExcludedS = (np.sum(np.isnan(spikeAveEp0), axis = 1) + np.isnan(choiceVec0S)) != 0
    trsExcludedS = (np.isnan(np.sum(spikeAveEp0, axis=(0,1))) + np.isnan(choiceVec0S)) != 0
    XS = spikeAveEp0[:,:,~trsExcludedS]; # trials x neurons
#        trsExcludedL = (np.sum(np.isnan(spikeAveEp0), axis = 1) + np.isnan(choiceVec0L)) != 0 
    trsExcludedL = (np.isnan(np.sum(spikeAveEp0, axis=(0,1))) + np.isnan(choiceVec0L)) != 0
    XL = spikeAveEp0[:,:,~trsExcludedL]; # trials x neurons

    T, N, C = XS.shape
    XS_N = np.reshape(XS.transpose(0 ,2 ,1), (T*C, N), order = 'F') # (frames x trials) x units
    
    T, N, C = XL.shape
    XL_N = np.reshape(XL.transpose(0 ,2 ,1), (T*C, N), order = 'F') # (frames x trials) x units
    
    # Define NsExcluded as neurons with low stdX for either short ITI or long ITI trials. 
    # This is to make sure short and long ITI cases will include the same set of neurons.
    stdXS = np.std(XS, axis = 0);
    stdXL = np.std(XL, axis = 0);

    NsExcluded = np.sum([stdXS < thAct, stdXL < thAct], axis=0)!=0 # if a neurons is non active for either short ITI or long ITI trials, exclude it.

else:
    
    T, N, C = X.shape
    X_N = np.reshape(X.transpose(0 ,2 ,1), (T*C, N), order = 'F') # (frames x trials) x units
    
    # Define NsExcluded as neurons with low stdX
    stdX = np.std(X_N, axis = 0);
    NsExcluded = stdX < thAct
    # np.sum(stdX < thAct)
    
    
    # X_choice
    T1, N1, C1 = X_choice.shape
    X_choice_N = np.reshape(X_choice.transpose(0 ,2 ,1), (T1*C1, N1), order = 'F') # (frames x trials) x units
    
    # Define NsExcluded as neurons with low stdX
    stdX_choice = np.std(X_choice_N, axis = 0);
    NsExcluded_choice = stdX_choice < thAct
    # np.sum(stdX < thAct)
    

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



#%% Exclude non-active neurons from X and set inhRois (ie neurons that don't fire in any of the trials during ep)

X = X[:,~NsExcluded,:]
print 'stim-aligned traces: ', np.shape(X)
    
# Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)
if neuronType==2:
    inhRois = inhibitRois[~NsExcluded]
    # print 'Number: inhibit = %d, excit = %d, unsure = %d' %(np.sum(inhRois==1), np.sum(inhRois==0), np.sum(np.isnan(inhRois)))
    # print 'Fraction: inhibit = %.2f, excit = %.2f, unsure = %.2f' %(fractInh, fractExc, fractUn)
    

X_choice = X_choice[:,~NsExcluded_choice,:]
print 'choice-aligned traces: ', np.shape(X_choice)


#%% Not doing the following anymore:
##  If number of neurons is more than 95% of trial numbers, identify n random neurons, where n= 0.95 * number of trials. This is to make sure we have more observations (trials) than features (neurons)

nTrs = np.shape(X)[2]
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

X = X[:,NsRand,:]
if neuronType==2:
    inhRois = inhRois[NsRand]

# below needs checking
'''
if windowAvgFlg==0:
    a = np.transpose(traces_al_stim[ep,:,:][:,~NsExcluded,:][:,:,~trsExcluded], (0,2,1))  # ep_frames x trials x units
    a = a[:,:,neuronsNow]
    X = np.reshape(a, (ep.shape[0]*(~trsExcluded).sum(), (~NsExcluded).sum())) # (ep_frames x trials) x units

    Y = np.tile(np.reshape(choiceVec0[~trsExcluded], (1,-1)), (ep.shape[0], 1)).flatten()    
'''
    
# Handle imbalance in the number of trials:
# unlike matlab, it doesn't seem to be a problem here... so we don't make trial numbers of HR and LR the same.
    
    
# Print some numbers
numDataPoints = X.shape[2] 
print '# data points = %d' %numDataPoints
# numTrials = (~trsExcluded).sum()
# numNeurons = (~NsExcluded).sum()
numTrials, numNeurons = X.shape[2], X.shape[1]
print '%d trials; %d neurons' %(numTrials, numNeurons)
# print ' The data has %d frames recorded from %d neurons at %d trials' %Xt.shape    


#%% Center and normalize X: feature normalization and scaling: to remove effects related to scaling and bias of each neuron, we need to zscore data (i.e., make data mean 0 and variance 1 for each neuron) 

T, N, C = X.shape
X_N = np.reshape(X.transpose(0 ,2 ,1), (T*C, N), order = 'F') # (frames x trials) x units
    
meanX = np.mean(X_N, axis = 0);
stdX = np.std(X_N, axis = 0);
# normalize X
X_N = (X_N-meanX)/stdX;
X = np.reshape(X_N, (T, C, N), order = 'F').transpose(0 ,2 ,1)


# X_choice
T1, N1, C1 = X_choice.shape
X_choice_N = np.reshape(X_choice.transpose(0 ,2 ,1), (T1*C1, N1), order = 'F') # (frames x trials) x units
    
meanX_choice = np.mean(X_choice_N, axis = 0);
stdX_choice = np.std(X_choice_N, axis = 0);
# normalize X
X_choice_N = (X_choice_N - meanX_choice) / stdX_choice;
X_choice = np.reshape(X_choice_N, (T1, C1, N1), order = 'F').transpose(0 ,2 ,1)



# In[23]:

if doPlots:
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



# classes: choices
# Plot stim-aligned averages after centering and normalization
if doPlots:
    # Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
    hr_trs = (Y==1)
    lr_trs = (Y==0)

    plt.figure()
    plt.subplot(1,2,1)
    a1 = np.nanmean(X[:, :, hr_trs],  axis=1) # frames x trials (average across neurons)
    tr1 = np.nanmean(a1,  axis = 1)
    tr1_se = np.nanstd(a1,  axis = 1) / np.sqrt(numTrials);
    a0 = np.nanmean(X[:, :, lr_trs],  axis=1) # frames x trials (average across neurons)
    tr0 = np.nanmean(a0,  axis = 1)
    tr0_se = np.nanstd(a0,  axis = 1) / np.sqrt(numTrials);
    mn = np.concatenate([tr1,tr0]).min()
    mx = np.concatenate([tr1,tr0]).max()
#    plt.plot([win[0], win[0]], [mn, mx], 'g-.') # mark the begining and end of training window
#    plt.plot([win[-1], win[-1]], [mn, mx], 'g-.')
    plt.fill_between(time_aligned_stim, tr1-tr1_se, tr1+tr1_se, alpha=0.5, edgecolor='b', facecolor='b')
    plt.fill_between(time_aligned_stim, tr0-tr0_se, tr0+tr0_se, alpha=0.5, edgecolor='r', facecolor='r')
    plt.plot(time_aligned_stim, tr1, 'b', label = 'high rate')
    plt.plot(time_aligned_stim, tr0, 'r', label = 'low rate')
    # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, lr_trs],  axis = (1, 2)), 'r', label = 'high rate')
    # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, hr_trs],  axis = (1, 2)), 'b', label = 'low rate')
    plt.xlabel('time aligned to stimulus onset (ms)')
    plt.title('Population average - raw')
    plt.legend()
    
    
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
    
    
    
# choice-aligned: classes: choices
# Plot stim-aligned averages after centering and normalization
if doPlots:
    # Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
    hr_trs = (Y_choice==1)
    lr_trs = (Y_choice==0)

    plt.figure()
    plt.subplot(1,2,1)
    a1 = np.nanmean(X_choice[:, :, hr_trs],  axis=1) # frames x trials (average across neurons)
    tr1 = np.nanmean(a1,  axis = 1)
    tr1_se = np.nanstd(a1,  axis = 1) / np.sqrt(numTrials);
    a0 = np.nanmean(X_choice[:, :, lr_trs],  axis=1) # frames x trials (average across neurons)
    tr0 = np.nanmean(a0,  axis = 1)
    tr0_se = np.nanstd(a0,  axis = 1) / np.sqrt(numTrials);
    mn = np.concatenate([tr1,tr0]).min()
    mx = np.concatenate([tr1,tr0]).max()
#    plt.plot([win[0], win[0]], [mn, mx], 'g-.') # mark the begining and end of training window
#    plt.plot([win[-1], win[-1]], [mn, mx], 'g-.')
    plt.fill_between(time_aligned_1stSide, tr1-tr1_se, tr1+tr1_se, alpha=0.5, edgecolor='b', facecolor='b')
    plt.fill_between(time_aligned_1stSide, tr0-tr0_se, tr0+tr0_se, alpha=0.5, edgecolor='r', facecolor='r')
    plt.plot(time_aligned_1stSide, tr1, 'b', label = 'high rate')
    plt.plot(time_aligned_1stSide, tr0, 'r', label = 'low rate')
    # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, lr_trs],  axis = (1, 2)), 'r', label = 'high rate')
    # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, hr_trs],  axis = (1, 2)), 'b', label = 'low rate')
    plt.xlabel('time aligned to stimulus onset (ms)')
    plt.title('Population average - raw')
    plt.legend()


#%% save a copy of X and Y (DO NOT run this after you run the parts below that redefine X and Y!)

X0 = X+0
Y0 = Y+0


#%%
##################################################################################################################
##################################################################################################################
########################################### SVM training starts here #############################################
##################################################################################################################
##################################################################################################################

nRandTrSamp = numSamples #10 #1000 # number of times to subselect trials (for the BAGGING method)
nShflSamp = numSamples #10 # number of times for shuffling trial labels to get null distribution of angles between decoders at different time points


########################## stimulus-aligned traces ############################

#%% start subselecting trials (for BAGGING)
# choose n random trials (n = 80% of min of HR, LR)
nrtr = int(np.ceil(.8 * min((Y0==0).sum(), (Y0==1).sum())))
print 'selecting %d random trials of each category' %(nrtr)


#%% Train SVM on each frame using BAGGING (subselection of trials)

#nRandTrSamp = 10 #1000 # number of times to subselect trials (for the BAGGING method)

if doData:
    # data
    w = np.full((np.shape(X)[0], X.shape[1], nRandTrSamp), np.nan) # frames x neurons x nrandtr
    b =  np.full((1, X.shape[0], nRandTrSamp), np.nan).squeeze() # frames x nrandtr
    trainE =  np.full((1, X.shape[0], nRandTrSamp), np.nan).squeeze() # frames x nrandtr
        
    for ir in range(nRandTrSamp): # subselect a group of trials
        print 'BAGGING iteration %d' %(ir)
        
        # subselect nrtr random HR and nrtr random LR trials.
        lrTrs = np.argwhere(Y0==0)
        randLrTrs = rng.permutation(lrTrs)[0:nrtr].flatten() # random LR trials
    
        hrTrs = np.argwhere(Y0==1)
        randHrTrs = rng.permutation(hrTrs)[0:nrtr].flatten() # random HR trials
        
        randTrs = np.sort(np.concatenate((randLrTrs, randHrTrs)))
        
        
        X = X0[:,:,randTrs]
        Y = Y0[randTrs]
    
    
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
                
            w[ifr,:, ir] = np.squeeze(linear_svm.coef_);
            b[ifr, ir] = linear_svm.intercept_;        
            trainE[ifr, ir] = abs(linear_svm.predict(X[ifr,:,:].transpose())-Y.astype('float')).sum()/len(Y)*100;
    
        # keep a copy of linear_svm
        '''
        import copy
        linear_svm_0 = copy.deepcopy(linear_svm) 
        '''


#%% shuffle

if doShuffles:
    # shuffle so we have a null distribution of angles between decoders (weights) at different times.
    # eg there will be nShflSamp decoders for each frame
    # so there will be nShflSamp angles between the decoder of frame f1 and the decoder of frame f2, 
    # this will give us a null distribution (of size nShflSamp) for the angle between decoders of frame f1 and f2.
    # we will later compare the angle between the decoders of these 2 frames in the actual data with this null distribution.
    
    #nShflSamp = 10 # number of times for shuffling trial labels to get null distribution of angles between decoders at different time points
    
    # shuffle
    wsh = np.full((np.shape(X)[0], X.shape[1], nRandTrSamp, nShflSamp), np.nan) # frames x neurons x nrandtr x nshfl
    bsh =  np.full((1, X.shape[0], nRandTrSamp, nShflSamp), np.nan).squeeze() # frames x nrandtr x nshfl
    trainEsh =  np.full((1, X.shape[0], nRandTrSamp, nShflSamp), np.nan).squeeze() # frames x nrandtr x nshfl
    
    for ish in range(nShflSamp):
        print 'Shuffle iteration %d' %(ish)
    
        # for each trial-label shuffle we find decoders at different time points (frames) (also we do subselection of trials so we can do Bagging later)    
        permIxs = rng.permutation(nrtr*2);     #index of trials for shuffling Y.
        
        for ir in range(nRandTrSamp):
            print '\tBAGGING iteration %d' %(ir)
    #        randTrs = rng.permutation(len(Y0))[0:nrtr].flatten() # random trials
            # subselect nrtr random HR and nrtr random LR trials.
            lrTrs = np.argwhere(Y0==0)
            randLrTrs = rng.permutation(lrTrs)[0:nrtr].flatten() # random LR trials
        
            hrTrs = np.argwhere(Y0==1)
            randHrTrs = rng.permutation(hrTrs)[0:nrtr].flatten() # random HR trials
            
            randTrs = np.sort(np.concatenate((randLrTrs, randHrTrs)))
            
            
            X = X0[:,:,randTrs]
            Y = Y0[randTrs]
            
                    
            ## 
            '''
            if regType == 'l1':
                linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l1', dual=False)
            elif regType == 'l2':
                linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l2', dual=True)
            '''        
            linear_svm = svm.LinearSVC(C=10.) # default c, also l2
            
            # train SVM on each frame
            for ifr in range(np.shape(X)[0]):
                    
                linear_svm.fit(X[ifr,:,:].transpose(), Y[permIxs])
                    
                wsh[ifr,:, ir, ish] = np.squeeze(linear_svm.coef_);
                bsh[ifr, ir, ish] = linear_svm.intercept_;        
                trainEsh[ifr, ir, ish] = abs(linear_svm.predict(X[ifr,:,:].transpose())-Y[permIxs].astype('float')).sum()/len(Y[permIxs])*100;
                
                 
###############################################################################
###############################################################################
"""         
#%% normalize weights (ie the w vector at each frame must have length 1)

eps = sys.float_info.epsilon #2.2204e-16
# data
normw = np.linalg.norm(w, axis=1) # frames x nrandtr; 2-norm of weights 
w_normed = np.transpose(np.transpose(w,(1,0,2))/normw, (1,0,2)) # frames x neurons x nrandtr; normalize weights so weights (of each frame) have length 1
if sum(normw<=eps).sum()!=0:
    print 'take care of this; you need to reshape w_normed first'
#    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero

# shuffle
normwsh = np.linalg.norm(wsh, axis=1) # frames x nrandtr x nshfl; 2-norm of weights 
wsh_normed = np.transpose(np.transpose(wsh,(1,0,2,3))/normwsh, (1,0,2,3)) # frames x neurons x nrandtr x nshfl; normalize weights so weights (of each frame) have length 1
if sum(normwsh<=eps).sum()!=0:
    print 'take care of this'
    wsh_normed[normwsh<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero


#%% average ws across all trial subselects, then again normalize them so they have length 1 (Bagging)

# data
w_nb = np.mean(w_normed, axis=(2)) # frames x neurons 
nw = np.linalg.norm(w_nb, axis=1) # frames; 2-norm of weights 
w_n2b = np.transpose(np.transpose(w_nb,(1,0))/nw, (1,0)) # frames x neurons

# shuffle
wsh_nb = np.mean(wsh_normed, axis=(2)) # frames x neurons x nshfl
nwsh = np.linalg.norm(wsh_nb, axis=1) # frames x nshfl; 2-norm of weights 
wsh_n2b = np.transpose(np.transpose(wsh_nb,(1,0,2))/nwsh, (1,0,2)) # frames x neurons x nshfl


#%% compute angle between ws at different times (remember angles close to 0 indicate more aligned decoders at 2 different time points.)

# some of the angles are nan because the dot product is slightly above 1, so its arccos is nan!

# we restrict angles to [0 90], so we don't care about direction of vectors, ie tunning reversal.
angle_choice = np.arccos(abs(np.dot(w_n2b, w_n2b.transpose())))*180/np.pi # frames x frames; angle between ws at different times

angle_choice_sh = np.full((w.shape[0],w.shape[0],nShflSamp), np.nan) # frames x frames x nshfl
for ish in range(nShflSamp):
    angle_choice_sh[:,:,ish] = np.arccos(abs(np.dot(wsh_n2b[:,:,ish], wsh_n2b[:,:,ish].transpose())))*180/np.pi # frames x frames x nshfl

'''
angle_choice_sh = np.arccos(abs(np.dot(wsh_n2b, wsh_n2b.transpose())))*180/np.pi # frames x frames; angle between ws at different times
a = np.reshape(np.transpose(wsh_n2b, (0,2,1)),(np.shape(wsh_n2b)[0]*np.shape(wsh_n2b)[2], np.shape(wsh_n2b)[1]), order = 'F') # (frames x nshfl) x neurons
b = np.arccos(abs(np.dot(a, a.transpose())))*180/np.pi # (frames x nshfl) x (frames x nshfl) 
np.reshape(b, (np.shape(wsh_n2b)[0], np.shape(wsh_n2b)[2], np.shape(wsh_n2b)[0], np.shape(wsh_n2b)[2]))
'''

# plot angle between decoders at different time points
'''
# data
plt.figure()
plt.imshow(angle_choice, cmap='jet_r')
plt.colorbar()

# shuffle (average of angles across shuffles)
plt.figure()
plt.imshow(np.nanmean(angle_choice_sh, axis=2), cmap='jet_r') # average across shuffles
plt.colorbar()
'''
"""
###############################################################################
###############################################################################


#%% SVR ; find stimulus decoder
############################################################################################################
############################################################################################################
############################################################################################################

#%% SVR, stimulus decoder 

# from Gamal:
#X features matrix (ie matrix of neural data of size trials x neurons);
#y observations array (ie array of stimulus rate of size trials)
from sklearn.svm import LinearSVR
'''
svr = LinearSVR(C=10., epsilon=0.0, dual = True, tol = 1e-9, fit_intercept = True)
svr.fit(X0[ifr,:,:].transpose(), Ysr)
w = svr.coef_;   # weights vector
b = svr.intercept_; # bias term
yhat = svr.predict(X0[ifr,:,:].transpose()) # prediction of rate equivalent to np.dot(X, w.T)+b
'''

nrtr = int(np.ceil(.8*len(Y0))) # we will subselect 80% of trials


if doData:
    # data
    w_sr = np.full((np.shape(X)[0], X.shape[1], nRandTrSamp), np.nan) # frames x neurons x nrandtr
    b_sr =  np.full((1, X.shape[0], nRandTrSamp), np.nan).squeeze() # frames x nrandtr
    trainE_sr =  np.full((1, X.shape[0], nRandTrSamp), np.nan).squeeze() # frames x nrandtr
        
    for ir in range(nRandTrSamp): # subselect a group of trials
        print 'BAGGING iteration %d' %(ir)
        
        # subselect nrtr random trials 
        randTrs = rng.permutation(len(Y0))[0:nrtr].flatten() # random trials
        
        X = X0[:,:,randTrs]
        Y = Ysr[randTrs] # now Y is the vector of stimulus rates
    
    
        ##%% train SVM on each frame using the sebselect of trials
        '''
        if regType == 'l1':
            linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l1', dual=False)
        elif regType == 'l2':
            linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l2', dual=True)
        '''    
    #    linear_svm = svm.LinearSVC() # default c, also l2
        svr = LinearSVR(C=10., epsilon=0.0, dual = True, tol = 1e-9, fit_intercept = True)
        
        # train SVR on each frame
        for ifr in range(np.shape(X)[0]):
            
            # data
            svr.fit(X[ifr,:,:].transpose(), Y)
                
            w_sr[ifr,:, ir] = np.squeeze(svr.coef_);
            b_sr[ifr, ir] = svr.intercept_;        
            trainE_sr[ifr, ir] = abs(svr.predict(X[ifr,:,:].transpose())-Y.astype('float')).sum()/len(Y)*100;



#%% shuffle (SVR, stim decoder)

if doShuffles:
    # shuffle so we have a null distribution of angles between decoders (weights) at different times.
    # eg there will be nShflSamp decoders for each frame
    # so there will be nShflSamp angles between the decoder of frame f1 and the decoder of frame f2, 
    # this will give us a null distribution (of size nShflSamp) for the angle between decoders of frame f1 and f2.
    # we will later compare the angle between the decoders of these 2 frames in the actual data with this null distribution.
    
    
    # shuffle
    wsh_sr = np.full((np.shape(X)[0], X.shape[1], nRandTrSamp, nShflSamp), np.nan) # frames x neurons x nrandtr x nshfl
    bsh_sr =  np.full((1, X.shape[0], nRandTrSamp, nShflSamp), np.nan).squeeze() # frames x nrandtr x nshfl
    trainEsh_sr =  np.full((1, X.shape[0], nRandTrSamp, nShflSamp), np.nan).squeeze() # frames x nrandtr x nshfl
    
    for ish in range(nShflSamp):
        print 'Shuffle iteration %d' %(ish)
    
        # for each trial-label shuffle we find decoders at different time points (frames) (also we do subselection of trials so we can do Bagging later)    
        permIxs = rng.permutation(nrtr);     #index of trials for shuffling Y.
        
        for ir in range(nRandTrSamp):
            print '\tBAGGING iteration %d' %(ir)
            
            # subselect nrtr random trials 
            randTrs = rng.permutation(len(Y0))[0:nrtr].flatten() # random trials
            
            X = X0[:,:,randTrs]
            Y = Ysr[randTrs] # now Y is the vector of stimulus rates
            
                    
            ##
            '''
            if regType == 'l1':
                linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l1', dual=False)
            elif regType == 'l2':
                linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l2', dual=True)
            '''
    #                linear_svm = svm.LinearSVC() # default c, also l2
            svr = LinearSVR(C=10., epsilon=0.0, dual = True, tol = 1e-9, fit_intercept = True)
            
            # train SVM on each frame
            for ifr in range(np.shape(X)[0]):
                    
                svr.fit(X[ifr,:,:].transpose(), Y[permIxs])
                    
                wsh_sr[ifr,:, ir, ish] = np.squeeze(svr.coef_);
                bsh_sr[ifr, ir, ish] = svr.intercept_;        
                trainEsh_sr[ifr, ir, ish] = abs(svr.predict(X[ifr,:,:].transpose())-Y[permIxs].astype('float')).sum()/len(Y[permIxs])*100;
                
                

#%%
###############################################################################
###############################################################################                
"""
#%% normalize weights (ie the w vector at each frame must have length 1)

#eps = sys.float_info.epsilon #2.2204e-16
# data
normw = np.linalg.norm(w_sr, axis=1) # frames x nrandtr; 2-norm of weights 
w_sr_normed = np.transpose(np.transpose(w_sr,(1,0,2))/normw, (1,0,2)) # frames x neurons x nrandtr; normalize weights so weights (of each frame) have length 1
if sum(normw<=eps).sum()!=0:
    print 'take care of this; you need to reshape w_normed first'
#    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero

# shuffle
normwsh = np.linalg.norm(wsh_sr, axis=1) # frames x nrandtr x nshfl; 2-norm of weights 
wsh_sr_normed = np.transpose(np.transpose(wsh_sr,(1,0,2,3))/normwsh, (1,0,2,3)) # frames x neurons x nrandtr x nshfl; normalize weights so weights (of each frame) have length 1
if sum(normwsh<=eps).sum()!=0:
    print 'take care of this'
    wsh_normed[normwsh<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero


#%% average ws across all trial subselects, then again normalize them so they have length 1 (Bagging)

# data
w_sr_nb = np.mean(w_sr_normed, axis=(2)) # frames x neurons 
nw = np.linalg.norm(w_sr_nb, axis=1) # frames; 2-norm of weights 
w_sr_n2b = np.transpose(np.transpose(w_sr_nb,(1,0))/nw, (1,0)) # frames x neurons

# shuffle
wsh_sr_nb = np.mean(wsh_sr_normed, axis=(2)) # frames x neurons x nshfl
nwsh = np.linalg.norm(wsh_sr_nb, axis=1) # frames x nshfl; 2-norm of weights 
wsh_sr_n2b = np.transpose(np.transpose(wsh_sr_nb,(1,0,2))/nwsh, (1,0,2)) # frames x neurons x nshfl


#%% compute angle between ws at different times (remember angles close to 0 indicate more aligned decoders at 2 different time points.)

# some of the angles are nan because the dot product is slightly above 1, so its arccos is nan!

# we restrict angles to [0 90], so we don't care about direction of vectors, ie tunning reversal.
angle_stim = np.arccos(abs(np.dot(w_sr_n2b, w_sr_n2b.transpose())))*180/np.pi # frames x frames; angle between ws at different times

angle_stim_sh = np.full((w_sr.shape[0],w_sr.shape[0],nShflSamp), np.nan) # frames x frames x nshfl
for ish in range(nShflSamp):
    angle_stim_sh[:,:,ish] = np.arccos(abs(np.dot(wsh_sr_n2b[:,:,ish], wsh_sr_n2b[:,:,ish].transpose())))*180/np.pi # frames x frames x nshfl


# plot angle between decoders at different time points
'''
# data
plt.figure()
plt.imshow(angle_stim, cmap='jet_r')
plt.colorbar()

# shuffle (average of angles across shuffles)
plt.figure()
plt.imshow(np.nanmean(angle_stim_sh, axis=2), cmap='jet_r') # average across shuffles
plt.colorbar()
'''

#avw = []
#for ifr in range(np.shape(X)[0]):
#    avw.append(wsh_sr_n2b[ifr,:,ish].mean())
    
#%%
"""
###############################################################################
###############################################################################                
#%% Same thing as above but now decoding the stimulus using svm
# bc some stim rates have very few trials... u ended up not doing this
"""
'''
for i in range(29): 
    print i, sum(Ysr==i)
'''    
    
linear_svm = svm.LinearSVC() # default c, also l2

numClass = len(np.unique(Ysr)) # number of classes (ie number of unique stimulus rates)

'''
for ifr in range(np.shape(X)[0]):
    
    linear_svm.fit(X0[ifr,:,:].transpose(), Ysr)
        
    w_sr[ifr,:,:] = np.squeeze(linear_svm.coef_) # numClass x neurons
    b_sr[ifr,:] = linear_svm.intercept_ # numClass
    
    trainE_sr[ifr] = abs(linear_svm.predict(X0[ifr,:,:].transpose())-Ysr.astype('float')).sum()/len(Ysr)*100
'''

# the following needs work... you need to subselect trials based on stim rates (now it is based on choice: 0,1)... if in different subselects u end up having differnt number of classes, then make sure in w and b you save the correct class for the correct dimension (not like how you are doing now)

# data
w_sr = np.full((np.shape(X)[0], numClass, X.shape[1], nRandTrSamp), np.nan) # frames x numClass x neurons x nrandtr
b_sr =  np.full((X.shape[0], numClass, nRandTrSamp), np.nan) # frames x numClass x nrandtr
trainE_sr =  np.full((X.shape[0], nRandTrSamp), np.nan) # frames x nrandtr
    
for ir in range(nRandTrSamp): # subselect a group of trials
    print 'BAGGING iteration %d' %(ir)
    
    # subselect nrtr random HR and nrtr random LR trials.
    lrTrs = np.argwhere(Y0==0)
    randLrTrs = rng.permutation(lrTrs)[0:nrtr].flatten() # random LR trials

    hrTrs = np.argwhere(Y0==1)
    randHrTrs = rng.permutation(hrTrs)[0:nrtr].flatten() # random HR trials
    
    randTrs = np.sort(np.concatenate((randLrTrs, randHrTrs)))
    
    
    X = X0[:,:,randTrs]
    Y = Y0[randTrs]


    ##%% train SVM on each frame using the sebselect of trials    
    
    linear_svm = svm.LinearSVC() # default c, also l2
    
    # train SVM on each frame
    for ifr in range(np.shape(X)[0]):
        
        # data
        linear_svm.fit(X[ifr,:,:].transpose(), Ysr)
            
        w_sr[ifr,:,:, ir] = np.squeeze(linear_svm.coef_);
        b_sr[ifr,:, ir] = linear_svm.intercept_;        
        trainE_sr[ifr, ir] = abs(linear_svm.predict(X[ifr,:,:].transpose())-Ysr.astype('float')).sum()/len(Ysr)*100;
"""



#%%
###############################################################################
###############################################################################

#%% SVM, choice decoder, but on choice-aligned traces (does movement cause stability in the population activity)

########################## choice-aligned traces ############################

#%% start subselecting trials (for BAGGING)
# choose n random trials (n = 80% of min of HR, LR)
nrtr = int(np.ceil(.8 * min((Y_choice==0).sum(), (Y_choice==1).sum())))
print 'selecting %d random trials of each category' %(nrtr)


#%% Train SVM on each frame using BAGGING (subselection of trials)

#nRandTrSamp = 10 #1000 # number of times to subselect trials (for the BAGGING method)

if doData:
    # data
    w_chAl = np.full((X_choice.shape[0], X_choice.shape[1], nRandTrSamp), np.nan) # frames x neurons x nrandtr
    b_chAl =  np.full((X_choice.shape[0], nRandTrSamp), np.nan) # frames x nrandtr
    trainE_chAl =  np.full((X_choice.shape[0], nRandTrSamp), np.nan) # frames x nrandtr
        
    for ir in range(nRandTrSamp): # subselect a group of trials
        print 'BAGGING iteration %d' %(ir)
        
        # subselect nrtr random HR and nrtr random LR trials.
        lrTrs = np.argwhere(Y_choice==0)
        randLrTrs = rng.permutation(lrTrs)[0:nrtr].flatten() # random LR trials
    
        hrTrs = np.argwhere(Y_choice==1)
        randHrTrs = rng.permutation(hrTrs)[0:nrtr].flatten() # random HR trials
        
        randTrs = np.sort(np.concatenate((randLrTrs, randHrTrs)))
        
        
        X = X_choice[:,:,randTrs]
        Y = Y_choice[randTrs]
    
    
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
    
        # keep a copy of linear_svm
        '''
        import copy
        linear_svm_0 = copy.deepcopy(linear_svm) 
        '''


#%% shuffle

if doShuffles:
    # shuffle so we have a null distribution of angles between decoders (weights) at different times.
    # eg there will be nShflSamp decoders for each frame
    # so there will be nShflSamp angles between the decoder of frame f1 and the decoder of frame f2, 
    # this will give us a null distribution (of size nShflSamp) for the angle between decoders of frame f1 and f2.
    # we will later compare the angle between the decoders of these 2 frames in the actual data with this null distribution.
    
    #nShflSamp = 10 # number of times for shuffling trial labels to get null distribution of angles between decoders at different time points
    
    # shuffle
    wsh_chAl = np.full((X_choice.shape[0], X_choice.shape[1], nRandTrSamp, nShflSamp), np.nan) # frames x neurons x nrandtr x nshfl
    bsh_chAl =  np.full((X_choice.shape[0], nRandTrSamp, nShflSamp), np.nan) # frames x nrandtr x nshfl
    trainEsh_chAl =  np.full((X_choice.shape[0], nRandTrSamp, nShflSamp), np.nan) # frames x nrandtr x nshfl
    
    for ish in range(nShflSamp):
        print 'Shuffle iteration %d' %(ish)
    
        # for each trial-label shuffle we find decoders at different time points (frames) (also we do subselection of trials so we can do Bagging later)    
        permIxs = rng.permutation(nrtr*2);     #index of trials for shuffling Y.
        
        for ir in range(nRandTrSamp):
            print '\tBAGGING iteration %d' %(ir)
    #        randTrs = rng.permutation(len(Y0))[0:nrtr].flatten() # random trials
            # subselect nrtr random HR and nrtr random LR trials.
            lrTrs = np.argwhere(Y_choice==0)
            randLrTrs = rng.permutation(lrTrs)[0:nrtr].flatten() # random LR trials
        
            hrTrs = np.argwhere(Y_choice==1)
            randHrTrs = rng.permutation(hrTrs)[0:nrtr].flatten() # random HR trials
            
            randTrs = np.sort(np.concatenate((randLrTrs, randHrTrs)))
            
            
            X = X_choice[:,:,randTrs]
            Y = Y_choice[randTrs]
            
                    
            ## 
            '''
            if regType == 'l1':
                linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l1', dual=False)
            elif regType == 'l2':
                linear_svm = svm.LinearSVC(C = cbest, loss='squared_hinge', penalty='l2', dual=True)
            '''        
            linear_svm = svm.LinearSVC(C=10.) # default c, also l2
            
            # train SVM on each frame
            for ifr in range(np.shape(X)[0]):
                    
                linear_svm.fit(X[ifr,:,:].transpose(), Y[permIxs])
                    
                wsh_chAl[ifr,:, ir, ish] = np.squeeze(linear_svm.coef_);
                bsh_chAl[ifr, ir, ish] = linear_svm.intercept_;        
                trainEsh_chAl[ifr, ir, ish] = abs(linear_svm.predict(X[ifr,:,:].transpose())-Y[permIxs].astype('float')).sum()/len(Y[permIxs])*100;
                
                 
###############################################################################
###############################################################################
    
#%% normalize weights (ie the w vector at each frame must have length 1)
"""
eps = sys.float_info.epsilon #2.2204e-16
# data
normw = np.linalg.norm(w_chAl, axis=1) # frames x nrandtr; 2-norm of weights 
w_normed_chAl = np.transpose(np.transpose(w_chAl,(1,0,2))/normw, (1,0,2)) # frames x neurons x nrandtr; normalize weights so weights (of each frame) have length 1
if sum(normw<=eps).sum()!=0:
    print 'take care of this; you need to reshape w_normed first'
#    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero

# shuffle
normwsh = np.linalg.norm(wsh_chAl, axis=1) # frames x nrandtr x nshfl; 2-norm of weights 
wsh_normed_chAl = np.transpose(np.transpose(wsh_chAl,(1,0,2,3))/normwsh, (1,0,2,3)) # frames x neurons x nrandtr x nshfl; normalize weights so weights (of each frame) have length 1
if sum(normwsh<=eps).sum()!=0:
    print 'take care of this'
    wsh_normed[normwsh<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero


#%% average ws across all trial subselects, then again normalize them so they have length 1 (Bagging)

# data
w_nb = np.mean(w_normed_chAl, axis=(2)) # frames x neurons 
nw = np.linalg.norm(w_nb, axis=1) # frames; 2-norm of weights 
w_n2b_chAl = np.transpose(np.transpose(w_nb,(1,0))/nw, (1,0)) # frames x neurons

# shuffle
wsh_nb = np.mean(wsh_normed_chAl, axis=(2)) # frames x neurons x nshfl
nwsh = np.linalg.norm(wsh_nb, axis=1) # frames x nshfl; 2-norm of weights 
wsh_n2b_chAl = np.transpose(np.transpose(wsh_nb,(1,0,2))/nwsh, (1,0,2)) # frames x neurons x nshfl


#%% compute angle between ws at different times (remember angles close to 0 indicate more aligned decoders at 2 different time points.)

# some of the angles are nan because the dot product is slightly above 1, so its arccos is nan!

# we restrict angles to [0 90], so we don't care about direction of vectors, ie tunning reversal.
angle_choice_chAl = np.arccos(abs(np.dot(w_n2b_chAl, w_n2b_chAl.transpose())))*180/np.pi # frames x frames; angle between ws at different times

angle_choice_chAl_sh = np.full((w_chAl.shape[0],w_chAl.shape[0],nShflSamp), np.nan) # frames x frames x nshfl
for ish in range(nShflSamp):
    angle_choice_chAl_sh[:,:,ish] = np.arccos(abs(np.dot(wsh_n2b_chAl[:,:,ish], wsh_n2b_chAl[:,:,ish].transpose())))*180/np.pi # frames x frames x nshfl

'''
angle_choice_sh = np.arccos(abs(np.dot(wsh_n2b, wsh_n2b.transpose())))*180/np.pi # frames x frames; angle between ws at different times
a = np.reshape(np.transpose(wsh_n2b, (0,2,1)),(np.shape(wsh_n2b)[0]*np.shape(wsh_n2b)[2], np.shape(wsh_n2b)[1]), order = 'F') # (frames x nshfl) x neurons
b = np.arccos(abs(np.dot(a, a.transpose())))*180/np.pi # (frames x nshfl) x (frames x nshfl) 
np.reshape(b, (np.shape(wsh_n2b)[0], np.shape(wsh_n2b)[2], np.shape(wsh_n2b)[0], np.shape(wsh_n2b)[2]))
'''

# plot angle between decoders at different time points
'''
# data
plt.figure()
plt.imshow(angle_choice_chAl, cmap='jet_r')
plt.colorbar()

# shuffle (average of angles across shuffles)
plt.figure()
plt.imshow(np.nanmean(angle_choice_chAl_sh, axis=2), cmap='jet_r') # average across shuffles
plt.colorbar()
'''
"""
###############################################################################
###############################################################################



# In[39]:
"""
# Plot weights
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
"""

#%%
####################################################################################################################################
####################################################################################################################################
# ## Project traces onto SVM weights
#     Stimulus-aligned, choice-aligned, etc traces projected onto SVM fitted weights.
#     More specifically, project traces of all trials onto normalized w (ie SVM weights computed from fitting model using X and Y of all trials).
####################################################################################################################################
####################################################################################################################################
'''
if doPlots:
    # pick the decoder of one of the frames ... u r exercizing for now... this is not final
    w_normalized = w_sr_n2b[35,:] #w/sci.linalg.norm(w);
#    w_normalized = w_sr[1,:,0]
    
    # stim-aligned traces
    XtN_w = np.dot(X_N, w_normalized);
    Xt_w = np.reshape(XtN_w, (T,C), order='F');
    
    
    cmap = plt.cm.get_cmap('seismic', 11)    
    # stim-aligned projections and raw average
    plt.figure()
#    plt.subplot(1,2,1)
    cnt=0
    for i in np.unique(Ysr):
        cnt=cnt+1
        tr1 = np.nanmean(Xt_w[:, Ysr==i],  axis = 1)
        tr1_se = np.nanstd(Xt_w[:, Ysr==i],  axis = 1) / np.sqrt(numTrials);
    #    tr0 = np.nanmean(Xt_w[:, Ysr==26],  axis = 1)
    #    tr0_se = np.nanstd(Xt_w[:, Ysr==26],  axis = 1) / np.sqrt(numTrials);
        plt.fill_between(time_aligned_stim, tr1-tr1_se, tr1+tr1_se, alpha=0.5, edgecolor=cmap(cnt)[:3], facecolor=cmap(cnt)[:3])
    #    plt.fill_between(time_aligned_stim, tr0-tr0_se, tr0+tr0_se, alpha=0.5, edgecolor='r', facecolor='r')
        plt.plot(time_aligned_stim, tr1, color=cmap(cnt)[:3], label = 'high rate')
#    plt.legend()
'''    
    
    






#%%
####################################################################################################################################
############################################ Save results as .mat files in a folder named svm ####################################################################################################################################
####################################################################################################################################


# In[526]:

from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')

if trialHistAnalysis:
    svmn = 'svmPrevChoice_allN_allFrsTrained_%s_' %(nowStr)
else:
    if doData and doShuffles==0:
        svmn = 'svmCurrChoice_allN_allFrsTrained_data_%s_' %(nowStr)
    elif doData==0 and doShuffles:
        svmn = 'svmCurrChoice_allN_allFrsTrained_shfl_%s_' %(nowStr)
    else:
        svmn = 'svmCurrChoice_allN_allFrsTrained_%s_' %(nowStr)
        
print '\n', svmn[:-1]


if saveResults:
    print 'Saving .mat file'
    d = os.path.join(os.path.dirname(pnevFileName), 'svm')
    if not os.path.exists(d):
        print 'creating svm folder'
        os.makedirs(d)

    svmName = os.path.join(d, svmn+os.path.basename(pnevFileName))
    print(svmName)

    if doData and doShuffles:
        scio.savemat(svmName, {'w':w, 'b':b, 'trainE':trainE,
                               'wsh':wsh, 'bsh':bsh, 'trainEsh':trainEsh,
                               'w_sr':w_sr, 'b_sr':b_sr, 'trainE_sr':trainE_sr,
                               'wsh_sr':wsh_sr, 'bsh_sr':bsh_sr, 'trainEsh_sr':trainEsh_sr,
    #                           'NsExcluded':NsExcluded, 'trsExcluded':trsExcluded, 'meanX': meanX, 'stdX':stdX, # not sure if you need any of these
                               'w_chAl': w_chAl, 'b_chAl': b_chAl, 'trainE_chAl': trainE_chAl,
                               'wsh_chAl': wsh_chAl, 'bsh_chAl': bsh_chAl, 'trainEsh_chAl': trainEsh_chAl})
    elif doData:
        scio.savemat(svmName, {'w':w, 'b':b, 'trainE':trainE,
                               'w_sr':w_sr, 'b_sr':b_sr, 'trainE_sr':trainE_sr,
                               'w_chAl': w_chAl, 'b_chAl': b_chAl, 'trainE_chAl': trainE_chAl})


    elif doShuffles:
        scio.savemat(svmName, {'wsh':wsh, 'bsh':bsh, 'trainEsh':trainEsh,                               
                               'wsh_sr':wsh_sr, 'bsh_sr':bsh_sr, 'trainEsh_sr':trainEsh_sr,
                               'wsh_chAl': wsh_chAl, 'bsh_chAl': bsh_chAl, 'trainEsh_chAl': trainEsh_chAl})
                               
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

