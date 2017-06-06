# -*- coding: utf-8 -*-
"""
# Trains decoder using only inh or only exc neurons after finding bestc for each decoder separately.


Created on Fri Dec  9 15:04:41 2016

@author: farznaj
"""

'''
#globals().update(locals())
from svm_excInh_setVars import *
# Run mainSVM_notebook
execfile("svm_excInh_setVars.py")
'''    
#import sys
#import os
#if ('ipykernel' in sys.modules) or any('SPYDER' in name for name in os.environ):
#    from svm_excInh_setVars import *
#    # Run mainSVM_notebook
#    execfile("svm_excInh_setVars.py")

numShufflesExc = 10 # we choose 10 sets of n random excitatory neurons.


#%% The following is basically svm_excInh_setVars .... apparently you need to turn that into a function so you can use its vars here when running it on the cluster.
#%%
# -*- coding: utf-8 -*-
"""
# Initialize vars for set_excInh_cPath.py and svm_excInh_trainDecoder.py. They both call this script.


Created on Fri Dec  9 15:25:41 2016

@author: farznaj
"""

# ## Specify variables for the analysis: 
#     - Data (mouse, day, sessions)
#     - Neuron type: excit, inhibit, or all
#     - Current-choice or previous-choice SVM training
#         if current-choice, specify epoch of training, the outcome (corr, incorr, all) and strength (easy, medium, hard, all) of trials for training SVM.
#         if previous-choice, specify ITI flag
#     - Trials that will be used for projections and class accuracy traces (corr, incorr, all, trained).

# In[1]:


#def svm_excInh_setVars():
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

# Only run the following section if you are running the code in jupyter, not if on the cluster or in spyder!
if ('ipykernel' in sys.modules) or any('SPYDER' in name for name in os.environ):
    
    # Set these variables:
    mousename = 'fni17'
    imagingFolder = '151102'
    mdfFileNumber = [1,2] 

    trialHistAnalysis = 0;    
#    roundi = 1; # For the same dataset we run the code multiple times, each time we select a random subset of neurons (of size n, n=.95*numTrials)
    iTiFlg = 1; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.

    softNorm = 1 # if 1, no neurons will be excluded, bc we do soft normalization of FRs, so non-active neurons wont be problematic. if softNorm = 0, NsExcluded will be found

    setNsExcluded = 1; # if 1, NsExcluded will be set even if it is already saved.
    numSamples = 2 #100; # number of iterations for finding the best c (inverse of regularization parameter)
    neuronType = 2; # 0: excitatory, 1: inhibitory, 2: all types.    
    saveResults = 0; # save results in mat file.

    
    doNsRand = 0; # if 1, a random set of neurons will be selected to make sure we have fewer neurons than trials. 
    regType = 'l2' # 'l1' : regularization type
    kfold = 10;
    compExcInh = 1 # if 1, analyses will be run to compare exc inh neurons.
    
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
import numpy as np
import numpy.random as rng
import sys
from crossValidateModel import crossValidateModel
from linearSVM import linearSVM
from compiler.ast import flatten # gives deprecation warning... try: from funcy import flatten, isa
import matplotlib 
from matplotlib import pyplot as plt
if 'ipykernel' in sys.modules and doPlots:
    get_ipython().magic(u'matplotlib inline')
    get_ipython().magic(u"config InlineBackend.figure_format = 'svg'")
matplotlib.rcParams['figure.figsize'] = (6,4) #(8,5)
from IPython.display import display
import sklearn.svm as svm
import os
import glob

'''
# print sys.path
if os.getcwd().find('farznaj')>0:
    sys.path.append('/home/farznaj/Documents/trial_history/imaging') # Gamal's dir needs to be added using "if" that takes the value of pwd
elif os.getcwd().find('churchland')>0:
#    sys.path.append('/sonas-hs/churchland/hpc/home/fnajafi/lassoClassifier' # you don't have trial_history/imaging on the server!
# print sys.path
from setImagingAnalysisNamesP import *
'''

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
        if os.getcwd().find('grid')!=-1: # server # sonas
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





# ## Set mat-file names

# In[9]:

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


# ## Set the time window for training SVM (ep) and traces_al_stim

# In[11]:

traces_al_stim = traces_al_stimAll

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
            
        
    #%% Exclude some trials from traces_al_stim
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


# In[12]:

'''
# Load 1stSideTry-aligned traces, frames, frame of event of interest
# use firstSideTryAl_COM to look at changes-of-mind (mouse made a side lick without committing it)
Data = scio.loadmat(postName, variable_names=['firstSideTryAl'],squeeze_me=True,struct_as_record=False)
traces_al_1stSide = Data['firstSideTryAl'].traces.astype('float')
time_aligned_1stSide = Data['firstSideTryAl'].time.astype('float')
# print(np.shape(traces_al_1stSide))


# Load goTone-aligned traces, frames, frame of event of interest
# use goToneAl_noStimAft to make sure there was no stim after go tone.
Data = scio.loadmat(postName, variable_names=['goToneAl'],squeeze_me=True,struct_as_record=False)
traces_al_go = Data['goToneAl'].traces.astype('float')
time_aligned_go = Data['goToneAl'].time.astype('float')
# print(np.shape(traces_al_go))


# Load reward-aligned traces, frames, frame of event of interest
Data = scio.loadmat(postName, variable_names=['rewardAl'],squeeze_me=True,struct_as_record=False)
traces_al_rew = Data['rewardAl'].traces.astype('float')
time_aligned_rew = Data['rewardAl'].time.astype('float')
# print(np.shape(traces_al_rew))


# Load commitIncorrect-aligned traces, frames, frame of event of interest
Data = scio.loadmat(postName, variable_names=['commitIncorrAl'],squeeze_me=True,struct_as_record=False)
traces_al_incorrResp = Data['commitIncorrAl'].traces.astype('float')
time_aligned_incorrResp = Data['commitIncorrAl'].time.astype('float')
# print(np.shape(traces_al_incorrResp))


# Load initiationTone-aligned traces, frames, frame of event of interest
Data = scio.loadmat(postName, variable_names=['initToneAl'],squeeze_me=True,struct_as_record=False)
traces_al_init = Data['initToneAl'].traces.astype('float')
time_aligned_init = Data['initToneAl'].time.astype('float')
# print(np.shape(traces_al_init))
# DataI = Data
'''

'''
if trialHistAnalysis:
    # either of the two below (stimulus-aligned and initTone-aligned) would be fine
    # eventI = DataI['initToneAl'].eventI
    eventI = DataS['stimAl_allTrs'].eventI    
    epEnd = eventI + epEnd_rel2stimon_fr #- 2 # to be safe for decoder training for trial-history analysis we go upto the frame before the stim onset
    # epEnd = DataI['initToneAl'].eventI - 2 # to be safe for decoder training for trial-history analysis we go upto the frame before the initTone onset
    ep = np.arange(epEnd+1)
    print 'training epoch is {} ms'.format(np.round((ep-eventI)*frameLength))
'''
    

# Load inhibitRois
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
    '''
    traces_al_1stSide = traces_al_1stSide[:, nt, :];
    traces_al_go = traces_al_go[:, nt, :];
    traces_al_rew = traces_al_rew[:, nt, :];
    traces_al_incorrResp = traces_al_incorrResp[:, nt, :];
    traces_al_init = traces_al_init[:, nt, :];    
    '''
else:
    nt = np.arange(np.shape(traces_al_stim)[1])    


# ## Set X (trials x neurons) and Y (trials x 1) for training the SVM classifier.
#     X matrix (size trials x neurons) that contains neural responses at different trials.
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
    # either of the two cases below should be fine (init-aligned traces or stim-aligned traces.)
    spikeAveEp0 = np.transpose(np.nanmean(traces_al_stimAll[ep,:,:], axis=0)) # trials x neurons    
    # spikeAveEp0 = np.transpose(np.nanmean(traces_al_init[ep,:,:], axis=0)) # trials x neurons    
else:    
    spikeAveEp0 = np.transpose(np.nanmean(traces_al_stim[ep,:,:], axis=0)) # trials x neurons    

# X = spikeAveEp0;
print 'Size of spikeAveEp0 (trs x neurons): ', spikeAveEp0.shape


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
trsExcluded = (np.sum(np.isnan(spikeAveEp0), axis = 1) + np.isnan(choiceVec0)) != 0 # NaN trials # trsExcluded
# print sum(trsExcluded), 'NaN trials'

# Exclude nan trials
X = spikeAveEp0[~trsExcluded,:]; # trials x neurons
Y = choiceVec0[~trsExcluded];
print '%d high-rate choices, and %d low-rate choices\n' %(np.sum(Y==1), np.sum(Y==0))


# In[15]:

# Set NsExcluded : Identify neurons that did not fire in any of the trials (during ep) and then exclude them. Otherwise they cause problem for feature normalization.
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
    
        # Define NsExcluded as neurons with low stdX
        stdX = np.std(X, axis = 0);
        NsExcluded = stdX < thAct
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



# In[16]:

# Exclude non-active neurons from X and set inhRois (ie neurons that don't fire in any of the trials during ep)

X = X[:,~NsExcluded]
print np.shape(X)
    
# Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)
if neuronType==2:
    inhRois = inhibitRois[~NsExcluded]
    # print 'Number: inhibit = %d, excit = %d, unsure = %d' %(np.sum(inhRois==1), np.sum(inhRois==0), np.sum(np.isnan(inhRois)))
    # print 'Fraction: inhibit = %.2f, excit = %.2f, unsure = %.2f' %(fractInh, fractExc, fractUn)
    


# In[15]:

NsRand = np.ones(np.shape(X)[1]).astype('bool')

'''
##  If number of neurons is more than 95% of trial numbers, identify n random neurons, where n= 0.95 * number of trials. This is to make sure we have more observations (trials) than features (neurons)

nTrs = np.shape(X)[0]
nNeuronsOrig = np.shape(X)[1]
nNeuronsNow = np.int(np.floor(nTrs * .95))

if nNeuronsNow < nNeuronsOrig:
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
else: # if number of neurons is already <= .95*numTrials, include all neurons.
    print 'Not doing random selection of neurons (nNeurons=%d already fewer than .95*nTrials=%d)' %(np.shape(X)[1], nTrs)
    NsRand = np.ones(np.shape(X)[1]).astype('bool')
'''    


# In[21]:

# Set X and inhRois only for the randomly selected set of neurons

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
print '%d trials; %d neurons' %(numTrials, numNeurons)
# print ' The data has %d frames recorded from %d neurons at %d trials' %Xt.shape    


# In[22]:

# Center and normalize X: feature normalization and scaling: to remove effects related to scaling and bias of each neuron, we need to zscore data (i.e., make data mean 0 and variance 1 for each neuron) 

meanX = np.mean(X, axis = 0);
stdX = np.std(X, axis = 0);
# normalize X
X = (X-meanX)/stdX;


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


# ## Set the traces that will be used for projections and plotting 
#     Traces are of size (frames x neurons x trials)
#     Choose trials that will be used for projections (trs4project = 'trained', 'all', 'corr', 'incorr')
#     Remove non-active neurons
#     Do feature normalization and scaling for the traces (using mean and sd of X)

# In[24]:

# trs4project = 'incorr' # 'trained', 'all', 'corr', 'incorr'

# Data = scio.loadmat(postName, variable_names=['outcomes', 'allResp_HR_LR'])
# choiceVecAll = (Data.pop('allResp_HR_LR').astype('float'))[0,:]

# Set trials that will be used for projection traces

if trs4project=='all': 
    Xt = traces_al_stim
    '''
    Xt_choiceAl = traces_al_1stSide
    Xt_goAl = traces_al_go
    Xt_rewAl = traces_al_rew
    Xt_incorrRespAl = traces_al_incorrResp
    Xt_initAl = traces_al_init
    '''
    Xt_stimAl_all = traces_al_stimAll
    choiceVecNow = choiceVecAll
elif trs4project=='trained':
    Xt = traces_al_stim[:, :, ~trsExcluded];
    '''
    Xt_choiceAl = traces_al_1stSide[:, :, ~trsExcluded];
    Xt_goAl = traces_al_go[:, :, ~trsExcluded];
    Xt_rewAl = traces_al_rew[:, :, ~trsExcluded];
    Xt_incorrRespAl = traces_al_incorrResp[:, :, ~trsExcluded];
    Xt_initAl = traces_al_init[:, :, ~trsExcluded];
    '''
    Xt_stimAl_all = traces_al_stimAll[:, :, ~trsExcluded];
    choiceVecNow = Y    
elif trs4project=='corr':
    Xt = traces_al_stim[:, :, outcomes==1];
    '''
    Xt_choiceAl = traces_al_1stSide[:, :, outcomes==1];
    Xt_goAl = traces_al_go[:, :, outcomes==1];
    Xt_rewAl = traces_al_rew[:, :, outcomes==1];
    Xt_incorrRespAl = traces_al_incorrResp[:, :, outcomes==1];
    Xt_initAl = traces_al_init[:, :, outcomes==1];
    '''
    Xt_stimAl_all = traces_al_stimAll[:, :, outcomes==1];
    choiceVecNow = choiceVecAll[outcomes==1]
    
elif trs4project=='incorr':
    Xt = traces_al_stim[:, :, outcomes==0];
    '''
    Xt_choiceAl = traces_al_1stSide[:, :, outcomes==0];
    Xt_goAl = traces_al_go[:, :, outcomes==0];
    Xt_rewAl = traces_al_rew[:, :, outcomes==0];
    Xt_incorrRespAl = traces_al_incorrResp[:, :, outcomes==0];
    Xt_initAl = traces_al_init[:, :, outcomes==0];
    '''
    Xt_stimAl_all = traces_al_stimAll[:, :, outcomes==0];
    choiceVecNow = choiceVecAll[outcomes==0]
    
## Xt = traces_al_stim[:, :, np.sum(np.sum(np.isnan(traces_al_stim), axis =0), axis =0)==0];
## Xt_choiceAl = traces_al_1stSide[:, :, np.sum(np.sum(np.isnan(traces_al_1stSide), axis =0), axis =0)==0];



# Exclude non-active neurons (ie neurons that don't fire in any of the trials during ep)
Xt = Xt[:,~NsExcluded,:]
'''
Xt_choiceAl = Xt_choiceAl[:,~NsExcluded,:]
Xt_goAl = Xt_goAl[:,~NsExcluded,:]
Xt_rewAl = Xt_rewAl[:,~NsExcluded,:]
Xt_incorrRespAl = Xt_incorrRespAl[:,~NsExcluded,:]
Xt_initAl = Xt_initAl[:,~NsExcluded,:]
'''
Xt_stimAl_all = Xt_stimAl_all[:,~NsExcluded,:]
    
# Only include the randomly selected set of neurons
Xt = Xt[:,NsRand,:]
'''
Xt_choiceAl = Xt_choiceAl[:,NsRand,:]
Xt_goAl = Xt_goAl[:,NsRand,:]
Xt_rewAl = Xt_rewAl[:,NsRand,:]
Xt_incorrRespAl = Xt_incorrRespAl[:,NsRand,:]
Xt_initAl = Xt_initAl[:,NsRand,:]
'''
Xt_stimAl_all = Xt_stimAl_all[:,NsRand,:]



# Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
hr_trs = (choiceVecNow==1)
lr_trs = (choiceVecNow==0)
# print 'Projection traces have %d high-rate trials, and %d low-rate trials' %(np.sum(hr_trs), np.sum(lr_trs))
    
    

# window of training (ep)
win = (ep-eventI)*frameLength

# Plot stim-aligned averages after centering and normalization
if doPlots:
    plt.figure()
    plt.subplot(1,2,1)
    a1 = np.nanmean(Xt[:, :, hr_trs],  axis=1) # frames x trials (average across neurons)
    tr1 = np.nanmean(a1,  axis = 1)
    tr1_se = np.nanstd(a1,  axis = 1) / np.sqrt(numTrials);
    a0 = np.nanmean(Xt[:, :, lr_trs],  axis=1) # frames x trials (average across neurons)
    tr0 = np.nanmean(a0,  axis = 1)
    tr0_se = np.nanstd(a0,  axis = 1) / np.sqrt(numTrials);
    mn = np.concatenate([tr1,tr0]).min()
    mx = np.concatenate([tr1,tr0]).max()
    plt.plot([win[0], win[0]], [mn, mx], 'g-.') # mark the begining and end of training window
    plt.plot([win[-1], win[-1]], [mn, mx], 'g-.')
    plt.fill_between(time_aligned_stim, tr1-tr1_se, tr1+tr1_se, alpha=0.5, edgecolor='b', facecolor='b')
    plt.fill_between(time_aligned_stim, tr0-tr0_se, tr0+tr0_se, alpha=0.5, edgecolor='r', facecolor='r')
    plt.plot(time_aligned_stim, tr1, 'b', label = 'high rate')
    plt.plot(time_aligned_stim, tr0, 'r', label = 'low rate')
    # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, lr_trs],  axis = (1, 2)), 'r', label = 'high rate')
    # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, hr_trs],  axis = (1, 2)), 'b', label = 'low rate')
    plt.xlabel('time aligned to stimulus onset (ms)')
    plt.title('Population average - raw')
    plt.legend()
    
    
    
    
    
    
## Feature normalization and scaling

# normalize stim-aligned traces
T, N, C = Xt.shape
Xt_N = np.reshape(Xt.transpose(0 ,2 ,1), (T*C, N), order = 'F')
Xt_N = (Xt_N-meanX)/stdX
Xt = np.reshape(Xt_N, (T, C, N), order = 'F').transpose(0 ,2 ,1)

# normalize stimAll-aligned traces
Tsa, Nsa, Csa = Xt_stimAl_all.shape
Xtsa_N = np.reshape(Xt_stimAl_all.transpose(0 ,2 ,1), (Tsa*Csa, Nsa), order = 'F')
Xtsa_N = (Xtsa_N-meanX)/stdX
Xtsa = np.reshape(Xtsa_N, (Tsa, Csa, Nsa), order = 'F').transpose(0 ,2 ,1)

'''
# normalize goTome-aligned traces
Tg, Ng, Cg = Xt_goAl.shape
Xtg_N = np.reshape(Xt_goAl.transpose(0 ,2 ,1), (Tg*Cg, Ng), order = 'F')
Xtg_N = (Xtg_N-meanX)/stdX
Xtg = np.reshape(Xtg_N, (Tg, Cg, Ng), order = 'F').transpose(0 ,2 ,1)

# normalize choice-aligned traces
Tc, Nc, Cc = Xt_choiceAl.shape
Xtc_N = np.reshape(Xt_choiceAl.transpose(0 ,2 ,1), (Tc*Cc, Nc), order = 'F')
Xtc_N = (Xtc_N-meanX)/stdX
Xtc = np.reshape(Xtc_N, (Tc, Cc, Nc), order = 'F').transpose(0 ,2 ,1)

# normalize reward-aligned traces
Tr, Nr, Cr = Xt_rewAl.shape
Xtr_N = np.reshape(Xt_rewAl.transpose(0 ,2 ,1), (Tr*Cr, Nr), order = 'F')
Xtr_N = (Xtr_N-meanX)/stdX
Xtr = np.reshape(Xtr_N, (Tr, Cr, Nr), order = 'F').transpose(0 ,2 ,1)

# normalize commitIncorrect-aligned traces
Tp, Np, Cp = Xt_incorrRespAl.shape
Xtp_N = np.reshape(Xt_incorrRespAl.transpose(0 ,2 ,1), (Tp*Cp, Np), order = 'F')
Xtp_N = (Xtp_N-meanX)/stdX
Xtp = np.reshape(Xtp_N, (Tp, Cp, Np), order = 'F').transpose(0 ,2 ,1)

# normalize init-aligned traces
Ti, Ni, Ci = Xt_initAl.shape
Xti_N = np.reshape(Xt_initAl.transpose(0 ,2 ,1), (Ti*Ci, Ni), order = 'F')
Xti_N = (Xti_N-meanX)/stdX
Xti = np.reshape(Xti_N, (Ti, Ci, Ni), order = 'F').transpose(0 ,2 ,1)    
'''
np.shape(Xt)


# window of training (ep)
# win = (ep-eventI)*frameLength

# Plot stim-aligned averages after centering and normalization
if doPlots:
#     plt.figure()
    plt.subplot(1,2,2)
    a1 = np.nanmean(Xt[:, :, hr_trs],  axis=1) # frames x trials (average across neurons)
    tr1 = np.nanmean(a1,  axis = 1)
    tr1_se = np.nanstd(a1,  axis = 1) / np.sqrt(numTrials);
    a0 = np.nanmean(Xt[:, :, lr_trs],  axis=1) # frames x trials (average across neurons)
    tr0 = np.nanmean(a0,  axis = 1)
    tr0_se = np.nanstd(a0,  axis = 1) / np.sqrt(numTrials);
    mn = np.concatenate([tr1,tr0]).min()
    mx = np.concatenate([tr1,tr0]).max()
    plt.plot([win[0], win[0]], [mn, mx], 'g-.') # mark the begining and end of training window
    plt.plot([win[-1], win[-1]], [mn, mx], 'g-.')
    plt.fill_between(time_aligned_stim, tr1-tr1_se, tr1+tr1_se, alpha=0.5, edgecolor='b', facecolor='b')
    plt.fill_between(time_aligned_stim, tr0-tr0_se, tr0+tr0_se, alpha=0.5, edgecolor='r', facecolor='r')
    plt.plot(time_aligned_stim, tr1, 'b', label = 'high rate')
    plt.plot(time_aligned_stim, tr0, 'r', label = 'low rate')
    # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, lr_trs],  axis = (1, 2)), 'r', label = 'high rate')
    # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, hr_trs],  axis = (1, 2)), 'b', label = 'low rate')
    plt.xlabel('time aligned to stimulus onset (ms)')
    plt.title('Population average - normalized')
    plt.legend()











































# In[25]:
##### Function for identifying the best regularization parameter
#     Perform 10-fold cross validation to obtain the best regularization parameter
#         More specifically: "crossValidateModel" divides data into training and test datasets. It calls linearSVM.py, which does linear SVM using XTrain, and returns percent class loss for XTrain and XTest.
#     This procedure gets repeated for numSamples (100 times) for each value of regulariazation parameter. 
#     An average across all 100 samples is computed to find the minimum test class loss.
#     Best regularization parameter is defined as the smallest regularization parameter whose test-dataset class loss is within mean+sem of minimum test class loss.

def setbesc(X,Y,regType,kfold,numDataPoints,numSamples,doPlots):
    import numpy as np
    from crossValidateModel import *
    from linearSVM import *
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
    

    ##%%%%%% plot C path 
    if doPlots:
        print 'Best c (inverse of regularization parameter) = %.2f' %cbest
        plt.figure('cross validation')
        plt.subplot(1,2,1)
        plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
        plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
        plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
        plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation')
        plt.plot(cvect[cvect==cbest], meanPerClassErrorTest[cvect==cbest], 'bo')
        plt.xlim([cvect[1], cvect[-1]])
        plt.xscale('log')
        plt.xlabel('c (inverse of regularization parameter)')
        plt.ylabel('classification error (%)')
        plt.legend(loc='center left', bbox_to_anchor=(1, .7))
        plt.tight_layout()


    return perClassErrorTrain, perClassErrorTest, wAllC, bAllC, cbestAll, cbest, cvect






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



# In[33]:
   
Xinh = X[:, inhRois==1]
XallExc = X[:, inhRois==0]

lenInh = (inhRois==1).sum()
excI = np.argwhere(inhRois==0)  

XexcAll = []
tempExcAll = []    
for ii in range(numShufflesExc):  # select n random exc (n = number of inh)         
    en = rng.permutation(excI)[0:lenInh].squeeze() # n randomly selected exc neurons.
    
    Xexc = X[:, en]
    XexcAll.append(Xexc)
    
    # you need the traces for setting correct classification traces in the next section
    tempExc = Xt[:,en,:] # stimulus-aligned traces (I think you should use Xtsa for consistency... u get projections from Xtsa)
    tempExcAll.append(tempExc)
    
    
################## set bestc and then class error,w,b for data ############
###### inh
perClassErrorTrain_inh, perClassErrorTest_inh, wAllC_inh, bAllC_inh, cbestAll_inh, cbest_inh, cvect = setbesc(Xinh,Y,regType,kfold,numDataPoints,numSamples,doPlots) # outputs have size the number of shuffles in setbestc (shuffles per c value)
# data: take the values at bestc that you computed above
indBestC = np.in1d(cvect, cbest_inh)
perClassErrorTrain_data_inh = perClassErrorTrain_inh[:,indBestC].squeeze()
perClassErrorTest_data_inh = perClassErrorTest_inh[:,indBestC].squeeze()
w_data_inh = wAllC_inh[:,indBestC,:].squeeze()
b_data_inh = bAllC_inh[:,indBestC]

###### allExc
perClassErrorTrain_allExc, perClassErrorTest_allExc, wAllC_allExc, bAllC_allExc, cbestAll_allExc, cbest_allExc, cvect = setbesc(XallExc,Y,regType,kfold,numDataPoints,numSamples,doPlots)
# data: take the values at bestc that you computed above
indBestC = np.in1d(cvect, cbest_allExc)
perClassErrorTrain_data_allExc = perClassErrorTrain_allExc[:,indBestC].squeeze()
perClassErrorTest_data_allExc = perClassErrorTest_allExc[:,indBestC].squeeze()
w_data_allExc = wAllC_allExc[:,indBestC,:].squeeze()
b_data_allExc = bAllC_allExc[:,indBestC]            

###### n exc (pick numShufflesEx sets of n exc neurons)
cbestAll_exc = []
cbest_exc = []
perClassErrorTrain_data_exc = []
perClassErrorTest_data_exc = []
w_data_exc = []
b_data_exc = []    
for ii in range(numShufflesExc):    
    # Set bestc    
    perClassErrorTrain_exc, perClassErrorTest_exc, wAllC_exc, bAllC_exc, cbestAll_exc0, cbest_exc0, cvect = setbesc(XexcAll[ii],Y,regType,kfold,numDataPoints,numSamples,doPlots)
    # data: take the values at bestc that you computed above
    cbestAll_exc.append(cbestAll_exc0)
    cbest_exc.append(cbest_exc0)
    indBestC = np.in1d(cvect, cbest_exc0)
    perClassErrorTrain_data_exc.append(perClassErrorTrain_exc[:,indBestC].squeeze()) # length of each is the number of samples in setbesc
    perClassErrorTest_data_exc.append(perClassErrorTest_exc[:,indBestC].squeeze())
    w_data_exc.append(wAllC_exc[:,indBestC,:].squeeze()) # numShufflesExc x numSamples x numNeur
    b_data_exc.append(bAllC_exc[:,indBestC].squeeze())     

# transpose so the sizes match _inh and _allExc as well as shuffle vars (below)        
perClassErrorTrain_data_exc = np.transpose(perClassErrorTrain_data_exc) # numSamples x numShufflesExc
perClassErrorTest_data_exc = np.transpose(perClassErrorTest_data_exc)
w_data_exc = np.transpose(w_data_exc,(1,0,2)) # numSamples x numShufflesExc x numNeur
b_data_exc = np.transpose(b_data_exc)
    
    
    
    
    
################## set class error,w,b for trial-label shuffles (null distribution) ############   
# shuffles
numShuffles = numSamples #100 # shuffles for cross validation
#numShufflesExc = 10 # we choose n random excitatory neurons, how many times to do this?

#    summary_data_inh = [];
summary_shfl_inh = [];
#    summary_data_allExc = [];
summary_shfl_allExc = [];
#    summary_data_exc = [];
summary_shfl_exc = [];
#    perClassErrorTrain_data_inh = [];
#    perClassErrorTest_data_inh = []
perClassErrorTrain_shfl_inh = [];
perClassErrorTest_shfl_inh = [];
#    w_data_inh = []
#    b_data_inh = []
w_shfl_inh = []
b_shfl_inh = []
#    perClassErrorTrain_data_allExc = [];
#    perClassErrorTest_data_allExc = []
perClassErrorTrain_shfl_allExc = [];
perClassErrorTest_shfl_allExc = [];
#    w_data_allExc = []
#    b_data_allExc = []
w_shfl_allExc = []
b_shfl_allExc = []
#    perClassErrorTrain_data_exc = [];
#    perClassErrorTest_data_exc = []
perClassErrorTrain_shfl_exc = [];
perClassErrorTest_shfl_exc = [];
#    w_data_exc = []
#    b_data_exc = []
w_shfl_exc = []
b_shfl_exc = []

permIxsList = [];
counter = 0    
   
   
for i in range(numShuffles): # 100 shuffles to set null distributions (by shuffling trial labels); (we also divide trials into testing and training). 
    # permIxs = rng.permutation(numTrials);
    permIxs = rng.permutation(numDataPoints);
    permIxsList.append(permIxs);    

    if regType == 'l1':
        # all inh            
        # Train decoder, do cross validation
#            summary_data_inh.append(crossValidateModel(Xinh, Y, linearSVM, kfold = kfold, l1 = c0best_inh))
        summary_shfl_inh.append(crossValidateModel(Xinh, Y[permIxs], linearSVM, kfold = kfold, l1 = cbest_inh))
       

        # all exc  
        # Train decoder, do cross validation            
#            summary_data_allExc.append(crossValidateModel(XallExc, Y, linearSVM, kfold = kfold, l1 = c0best_allExc))
        summary_shfl_allExc.append(crossValidateModel(XallExc, Y[permIxs], linearSVM, kfold = kfold, l1 = cbest_allExc))
        

        # n exc (n = number of inh)                    
        for ii in range(numShufflesExc):  # select n random exc (n = number of inh)         
#                en = rng.permutation(excI)[0:lenInh].squeeze() # n randomly selected exc neurons.
#                Xexc = X[:, en]
        
            # Train decoder, do cross validation                            
#                summary_data_exc.append(crossValidateModel(Xexc, Y, linearSVM, kfold = kfold, l1 = c00best_exc)) # summary_data_exc is of size numShuffles x numShufflesExc
#                summary_shfl_exc.append(crossValidateModel(Xexc, Y[permIxs], linearSVM, kfold = kfold, l1 = cbest_exc0))
            summary_shfl_exc.append(crossValidateModel(XexcAll[ii], Y[permIxs], linearSVM, kfold = kfold, l1 = cbest_exc[ii]))
                               
            counter = counter+1
#             print en
            
        # Set inds once done with all numShufflesExc
#             print counter
        inds = np.arange(counter-numShufflesExc,counter)
#             print inds

    ''' L2 needs work
    elif regType == 'l2':
        # all inh
        summary_data_inh.append(crossValidateModel(Xinh, Y, linearSVM, kfold = kfold, l2 = cbest))
        summary_shfl_inh.append(crossValidateModel(Xinh, Y[permIxs], linearSVM, kfold = kfold, l2 = cbest))


        # all exc
        summary_data_allExc.append(crossValidateModel(XallExc, Y, linearSVM, kfold = kfold, l2 = cbest))
        summary_shfl_allExc.append(crossValidateModel(XallExc, Y[permIxs], linearSVM, kfold = kfold, l2 = cbest))


        # n exc (n = number of inh)     
        for ii in range(numShufflesExc):
            Xexc = X[:, en]
            summary_data_exc.append(crossValidateModel(XexcAll[ii], Y, linearSVM, kfold = kfold, l2 = cbest_exc[ii]))
            summary_shfl_exc.append(crossValidateModel(XexcAll[ii], Y[permIxs], linearSVM, kfold = kfold, l2 = cbest_exc[ii]))

            counter = counter+1
#             print counter
        inds = np.arange(counter-numShufflesExc,counter)
#             print inds
    '''    

    
    # Collect values for each shuffle of trial labels (i)
    # inh        
#        perClassErrorTrain_data_inh.append(summary_data_inh[i].perClassErrorTrain);
#        perClassErrorTest_data_inh.append(summary_data_inh[i].perClassErrorTest);
#        w_data_inh.append(np.squeeze(summary_data_inh[i].model.coef_));
#        b_data_inh.append(summary_data_inh[i].model.intercept_);

    perClassErrorTrain_shfl_inh.append(summary_shfl_inh[i].perClassErrorTrain);
    perClassErrorTest_shfl_inh.append(summary_shfl_inh[i].perClassErrorTest);
    w_shfl_inh.append(np.squeeze(summary_shfl_inh[i].model.coef_));
    b_shfl_inh.append(summary_shfl_inh[i].model.intercept_);

    # all exc
#        perClassErrorTrain_data_allExc.append(summary_data_allExc[i].perClassErrorTrain);
#        perClassErrorTest_data_allExc.append(summary_data_allExc[i].perClassErrorTest);
#        w_data_allExc.append(np.squeeze(summary_data_allExc[i].model.coef_));
#        b_data_allExc.append(summary_data_allExc[i].model.intercept_);

    perClassErrorTrain_shfl_allExc.append(summary_shfl_allExc[i].perClassErrorTrain);
    perClassErrorTest_shfl_allExc.append(summary_shfl_allExc[i].perClassErrorTest);
    w_shfl_allExc.append(np.squeeze(summary_shfl_allExc[i].model.coef_));
    b_shfl_allExc.append(summary_shfl_allExc[i].model.intercept_);

    # n exc
#        perClassErrorTrain_data_exc.append(np.mean([summary_data_exc[inds[j]].perClassErrorTrain for j in range(len(inds))], axis=0)); # append the average of shuffles for selecting n exc neurons);
#        perClassErrorTest_data_exc.append(np.mean([summary_data_exc[inds[j]].perClassErrorTest for j in range(len(inds))], axis=0));
#        w_data_exc.append(np.mean([(summary_data_exc[inds[j]].model.coef_).squeeze() for j in range(len(inds))], axis=0));
#        b_data_exc.append(np.mean([(summary_data_exc[inds[j]].model.intercept_).squeeze() for j in range(len(inds))], axis=0));

#        perClassErrorTrain_shfl_exc.append(np.mean([summary_shfl_exc[inds[j]].perClassErrorTrain for j in range(len(inds))], axis=0)) # average across shuffles for selecting n exc neurons);
#        perClassErrorTest_shfl_exc.append(np.mean([summary_shfl_exc[inds[j]].perClassErrorTest for j in range(len(inds))], axis=0));
#        w_shfl_exc.append(np.mean([(summary_shfl_exc[inds[j]].model.coef_).squeeze() for j in range(len(inds))], axis=0));
#        b_shfl_exc.append(np.mean([(summary_shfl_exc[inds[j]].model.intercept_).squeeze() for j in range(len(inds))], axis=0));
  
    perClassErrorTrain_shfl_exc.append([summary_shfl_exc[inds[j]].perClassErrorTrain for j in range(len(inds))])  # numSamples x numShufflesExc # keep values of all shuffls.
    perClassErrorTest_shfl_exc.append([summary_shfl_exc[inds[j]].perClassErrorTest for j in range(len(inds))]);  # numSamples x numShufflesExc
    w_shfl_exc.append(np.squeeze([(summary_shfl_exc[inds[j]].model.coef_) for j in range(len(inds))])); # numSamples x numShufflesExc x numNeur
    b_shfl_exc.append([(summary_shfl_exc[inds[j]].model.intercept_).squeeze() for j in range(len(inds))]);  # numSamples x numShufflesExc


#     # n exc
#     perClassErrorTrain_data_exc.append(summary_data_exc[i].perClassErrorTrain);
#     perClassErrorTest_data_exc.append(summary_data_exc[i].perClassErrorTest);
#     w_data_exc.append(np.squeeze(summary_data_exc[i].model.coef_));
#     b_data_exc.append(summary_data_exc[i].model.intercept_);

#     perClassErrorTrain_shfl_exc.append(summary_shfl_exc[i].perClassErrorTrain);
#     perClassErrorTest_shfl_exc.append(summary_shfl_exc[i].perClassErrorTest);
#     w_shfl_exc.append(np.squeeze(summary_shfl_exc[i].model.coef_));
#     b_shfl_exc.append(summary_shfl_exc[i].model.intercept_);


# Simply get all values of all shuffles instead of appending like above    
# perClassErrorTrain_data_exc = [summary_data_exc[i].perClassErrorTrain for i in range(np.shape(summary_data_exc)[0])]
# perClassErrorTest_data_exc = [summary_data_exc[i].perClassErrorTest for i in range(np.shape(summary_data_exc)[0])]



#%%
####################################################################################################################################
##### PLOTS
####################################################################################################################################

# In[34]:

# Plot histograms of class error for training and testing datasets
   
# inh
print 'All inhibitory neurons'
pd = perClassErrorTrain_data_inh
ps = perClassErrorTrain_shfl_inh
pd0 = perClassErrorTest_data_inh
ps0 = perClassErrorTest_shfl_inh
pvalueTrain = ttest2(pd, ps, tail = 'left');
pvalueTest = ttest2(pd0, ps0, tail = 'left');

print '\tTraining error: Mean actual: %.2f%%, Mean shuffled: %.2f%%, p-value = %.2f' %(np.mean(pd), np.mean(ps), pvalueTrain)
print '\tTesting error: Mean actual: %.2f%%, Mean shuffled: %.2f%%, p-value = %.2f' %(np.mean(pd0), np.mean(ps0), pvalueTest)


# Plot the histograms
if doPlots:
    binEvery = 3; # bin width

    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(pd, np.arange(0,100,binEvery), color = 'k', label = 'data');
    plt.hist(ps, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
    plt.xlabel('Training classification error (%)')
    plt.ylabel('count')
    plt.title('Inhibitory neurons\nMean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(pd), np.mean(ps), pvalueTrain), fontsize = 10)
    plt.legend()

    plt.subplot(1,2,2)
    plt.hist(pd0, np.arange(0,100,binEvery), color = 'k', label = 'data');
    plt.hist(ps0, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
    plt.legend()
    plt.xlabel('Testing classification error (%)')
    plt.title('Mean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(pd0), np.mean(ps0), pvalueTest), fontsize = 10)
    plt.ylabel('count')
    plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)




# n exc    
print 'Only n of the excitory neurons'
#    pd = perClassErrorTrain_data_exc
#    ps = perClassErrorTrain_shfl_exc
#    pd0 = perClassErrorTest_data_exc
#    ps0 = perClassErrorTest_shfl_exc
pd = np.reshape(perClassErrorTrain_data_exc,(-1,)) # pool all neuron and trial shuffles
ps = np.reshape(perClassErrorTrain_shfl_exc,(-1,))
pd0 = np.reshape(perClassErrorTest_data_exc,(-1,))
ps0 = np.reshape(perClassErrorTest_shfl_exc,(-1,))    
pvalueTrain = ttest2(pd, ps, tail = 'left');
pvalueTest = ttest2(pd0, ps0, tail = 'left');

print '\tTraining error: Mean actual: %.2f%%, Mean shuffled: %.2f%%, p-value = %.2f' %(np.mean(pd), np.mean(ps), pvalueTrain)
print '\tTesting error: Mean actual: %.2f%%, Mean shuffled: %.2f%%, p-value = %.2f' %(np.mean(pd0), np.mean(ps0), pvalueTest)


# Plot the histograms
if doPlots:
    binEvery = 3; # bin width

    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(pd, np.arange(0,100,binEvery), color = 'k', label = 'data');
    plt.hist(ps, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
    plt.xlabel('Training classification error (%)')
    plt.ylabel('count')
    plt.title('n excitatory neurons\nMean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(pd), np.mean(ps), pvalueTrain), fontsize = 10)
    plt.legend()

    plt.subplot(1,2,2)
    plt.hist(pd0, np.arange(0,100,binEvery), color = 'k', label = 'data');
    plt.hist(ps0, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
    plt.legend()
    plt.xlabel('Testing classification error (%)')
    plt.title('Mean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(pd0), np.mean(ps0), pvalueTest), fontsize = 10)
    plt.ylabel('count')
    plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)





# all exc    
print 'All excitatory neurons'
pd = perClassErrorTrain_data_allExc
ps = perClassErrorTrain_shfl_allExc
pd0 = perClassErrorTest_data_allExc
ps0 = perClassErrorTest_shfl_allExc
pvalueTrain = ttest2(pd, ps, tail = 'left');
pvalueTest = ttest2(pd0, ps0, tail = 'left');

print '\tTraining error: Mean actual: %.2f%%, Mean shuffled: %.2f%%, p-value = %.2f' %(np.mean(pd), np.mean(ps), pvalueTrain)
print '\tTesting error: Mean actual: %.2f%%, Mean shuffled: %.2f%%, p-value = %.2f' %(np.mean(pd0), np.mean(ps0), pvalueTest)


# Plot the histograms
if doPlots:
    binEvery = 3; # bin width

    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(pd, np.arange(0,100,binEvery), color = 'k', label = 'data');
    plt.hist(ps, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
    plt.xlabel('Training classification error (%)')
    plt.ylabel('count')
    plt.title('All excitatory neurons\nMean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(pd), np.mean(ps), pvalueTrain), fontsize = 10)
    plt.legend()

    plt.subplot(1,2,2)
    plt.hist(pd0, np.arange(0,100,binEvery), color = 'k', label = 'data');
    plt.hist(ps0, np.arange(0,100,binEvery), color = 'k', alpha=.5, label = 'shuffled');
    plt.legend()
    plt.xlabel('Testing classification error (%)')
    plt.title('Mean data: %.2f %%, Mean shuffled: %.2f %%\n p-value = %.2f' %(np.mean(pd0), np.mean(ps0), pvalueTest), fontsize = 10)
    plt.ylabel('count')
    plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)    






#%%
####################################################################################################################################
####################################################################################################################################
# ## Set correct classification traces for the following cases:
#     
#     1) All inhibitory neurons contribute to the decoder
#     2) N excitatory neurons contribute, where n = number of inhibitory neurons. 
#     3) All excitatory neurons contribute
#     
#     Using best c we fit the classifier to Xinh, Xexc, XallExc using all trials.
####################################################################################################################################
####################################################################################################################################

## First train SVM using bestc for each population that was found above. Use all trials. 
# Then use the SVM classifier to predict choices at each time point.
# Also shuffle trial labels at each time point to set chance correct classification traces.


#%% inh
#    linear_svm_inh = copy.deepcopy(linear_svm_0)
if regType == 'l1':
    linear_svm_inh = svm.LinearSVC(C = cbest_inh, loss='squared_hinge', penalty='l1', dual=False)
elif regType == 'l2':
    linear_svm_inh = svm.LinearSVC(C = cbest_inh, loss='squared_hinge', penalty='l2', dual=True)

linear_svm_inh.fit(Xinh, Y)
# w = np.squeeze(linear_svm.coef_);
# b = linear_svm.intercept_;
# trainE_inh = abs(linear_svm_inh.predict(Xinh)-Y.astype('float')).sum()/len(Y)*100;
# trainE_inh

tempInh = Xt[:,inhRois==1,:] # stimulus-aligned traces (I think you should use Xtsa for consistency... u get projections from Xtsa) # set in svm_excInh_setVars.py
temp = tempInh
linear_svm_now = linear_svm_inh

nnf, nnu, nnt = temp.shape
corrClass_inh = np.ones((T, nnt)) + np.nan # frames x trials
corrClassShfl_inh = np.ones((T, nnt, numSamples)) + np.nan # frames x trials x numSamples

for t in range(T):
    trac = np.squeeze(temp[t, :, :]).T # trials x neurons  # stimulus-aligned trace at time t
    corrFract = 1 - abs(linear_svm_now.predict(trac)-choiceVecNow); # trials x 1 # fraction of correct choice classification using stimulus-aligned neural responses at time t and the trained SVM model linear_svm.
    # corrFract = 1 - abs(linear_svm.predict(trac)-Y); # trials x 1 # fraction of correct choice classification using stimulus-aligned neural responses at time t and the trained SVM model linear_svm.
    corrClass_inh[t,:] = corrFract # frames x trials % fraction correct classification for all trials
    # perClassCorr_t.append(corrFract.mean()*100) # average correct classification across trials for time point t. Same as np.mean(corrClass, axis=1)    

    # shuffled data
    for i in range(numSamples):
        permIxs = rng.permutation(numDataPoints);
        corrFractSh = 1 - abs(linear_svm_now.predict(trac)-choiceVecNow[permIxs]);
        corrClassShfl_inh[t,:,i] = corrFractSh # frames x trials x numSamples
#    corrFractShfl = np.mean(corrFractSh,axis=None) # average of trials across shuffles    
#    corrClassShfl[numSamples,t,:] = corrFractShfl # frames x trials


#%% all exc
#    linear_svm_allExc = copy.deepcopy(linear_svm_0)
if regType == 'l1':
    linear_svm_allExc = svm.LinearSVC(C = cbest_allExc, loss='squared_hinge', penalty='l1', dual=False)
elif regType == 'l2':
    linear_svm_allExc = svm.LinearSVC(C = cbest_allExc, loss='squared_hinge', penalty='l2', dual=True)
    
linear_svm_allExc.fit(XallExc, Y)
# w = np.squeeze(linear_svm.coef_);
# b = linear_svm.intercept_;
# trainE_allExc = abs(linear_svm_allExc.predict(XallExc)-Y.astype('float')).sum()/len(Y)*100;
# trainE_allExc

tempAllExc = Xt[:,inhRois==0,:] # stimulus-aligned traces (I think you should use Xtsa for consistency... u get projections from Xtsa)
temp = tempAllExc
linear_svm_now = linear_svm_allExc

nnf, nnu, nnt = temp.shape
corrClass_allExc = np.ones((T, nnt)) + np.nan # frames x trials
corrClassShfl_allExc = np.ones((T, nnt, numSamples)) + np.nan # frames x trials x numSamples

for t in range(T):
    trac = np.squeeze(temp[t, :, :]).T # trials x neurons  # stimulus-aligned trace at time t
    corrFract = 1 - abs(linear_svm_now.predict(trac)-choiceVecNow); # trials x 1 # fraction of correct choice classification using stimulus-aligned neural responses at time t and the trained SVM model linear_svm.
    # corrFract = 1 - abs(linear_svm.predict(trac)-Y); # trials x 1 # fraction of correct choice classification using stimulus-aligned neural responses at time t and the trained SVM model linear_svm.
    corrClass_allExc[t,:] = corrFract # frames x trials % fraction correct classification for all trials
    # perClassCorr_t.append(corrFract.mean()*100) # average correct classification across trials for time point t. Same as np.mean(corrClass, axis=1)    

    # shuffled data
    for i in range(numSamples):
        permIxs = rng.permutation(numDataPoints);
        corrFractSh = 1 - abs(linear_svm_now.predict(trac)-choiceVecNow[permIxs]);
        corrClassShfl_allExc[t,:,i] = corrFractSh # # frames x trials x numSamples



#%% n exc
#    numShufflesExc = 10 # we choose n random excitator neurons, how many times to do this?
corrClass_exc0 = []
corrClassShfl_exc0 = []
for ii in range(numShufflesExc):            
#        en = rng.permutation(excI)[0:lenInh].squeeze() # n randomly selected exc neurons.
#        Xexc = X[:, en]

#        linear_svm_exc = copy.deepcopy(linear_svm_0)
    if regType == 'l1':
        linear_svm_exc = svm.LinearSVC(C = cbest_exc[ii], loss='squared_hinge', penalty='l1', dual=False)
    elif regType == 'l2':
        linear_svm_exc = svm.LinearSVC(C = cbest_exc[ii], loss='squared_hinge', penalty='l2', dual=True)
        
    linear_svm_exc.fit(XexcAll[ii], Y)
    # w = np.squeeze(linear_svm.coef_);
    # b = linear_svm.intercept_;
    # trainE_exc = abs(linear_svm_exc.predict(Xexc)-Y.astype('float')).sum()/len(Y)*100;
    # trainE_exc

#        tempExc = Xt[:,en,:] # stimulus-aligned traces (I think you should use Xtsa for consistency... u get projections from Xtsa)
    tempExc = tempExcAll[ii]
    temp = tempExc
    linear_svm_now = linear_svm_exc

    nnf, nnu, nnt = temp.shape
    corrClass2 = np.ones((T, nnt)) + np.nan # frames x trials
    corrClassShfl2 = np.ones((T, nnt, numSamples)) + np.nan # frames x trials x numSamples
    
    for t in range(T):
        trac = np.squeeze(temp[t, :, :]).T # trials x neurons  # stimulus-aligned trace at time t
        corrFract = 1 - abs(linear_svm_now.predict(trac)-choiceVecNow); # trials x 1 # fraction of correct choice classification using stimulus-aligned neural responses at time t and the trained SVM model linear_svm.
        # corrFract = 1 - abs(linear_svm.predict(trac)-Y); # trials x 1 # fraction of correct choice classification using stimulus-aligned neural responses at time t and the trained SVM model linear_svm.
        corrClass2[t,:] = corrFract # frames x trials % fraction correct classification for all trials
        # perClassCorr_t.append(corrFract.mean()*100) # average correct classification across trials for time point t. Same as np.mean(corrClass, axis=1)    

        # shuffled data
        for i in range(numSamples):
            permIxs = rng.permutation(numDataPoints);
            corrFractSh = 1 - abs(linear_svm_now.predict(trac)-choiceVecNow[permIxs]);
            corrClassShfl2[t,:,i] = corrFractSh # frames x trials
        
        
    corrClass_exc0.append(corrClass2) # numShuffExc x frames x trials. Average correct class across trials for each shuffle.           
    corrClassShfl_exc0.append(corrClassShfl2) # numShuffExc x frames x trials x numSamples. Average correct class across trials for each shuffle.           
    

s1, s2, s3 = np.shape(corrClass_exc0)
corrClass_exc = np.reshape(np.transpose(corrClass_exc0, (1,0,2)), (s2,s1*s3), order = 'F') # frames x (numShuffExc x trials)

s1, s2, s3, s4 = np.shape(corrClassShfl_exc0)
corrClassShfl_exc = np.reshape(np.transpose(corrClassShfl_exc0, (1,0,2,3)), (s2,s1*s3,s4), order = 'F') # frames x (numShuffExc x trials) x numSamples

#         corrClass_exc.append(np.mean(corrClass2, axis = 1)) # numShuffExc x frames. Average correct class across trials for each shuffle.    
#     corrClass_exc = np.transpose(corrClass_exc) # frames x numShuffExc



#%%
####################################################################################################################################
##### PLOTS
####################################################################################################################################

# In[36]:

# Plot correct classification traces for the following cases:
#     1) All inhibitory neurons contribute to the decoder
#     2) N excitatory neurons contribute, where n = number of inhibitory neurons. 
#     3) All excitatory neurons contribute
    
if doPlots:
    
    plt.figure()   
#     plt.subplot(1,2,1)
    
    
    # Plot corrClass for all inhibitory neurons
    a1 = np.mean(corrClass_inh, axis=1)*100
    s1 = np.std(corrClass_inh, axis=1)*100 /np.sqrt(corrClass_inh.shape[1])    

    a2 = np.mean(corrClassShfl_inh, axis=(1,2))*100 # average across all shuffles and trials
    s2 = np.std(corrClassShfl_inh, axis=(1,2))*100 /np.sqrt(numTrials*numSamples);
    
    plt.fill_between(time_aligned_stim, a1-s1, a1+s1, alpha=0.3, edgecolor='b', facecolor='b')
    plt.plot(time_aligned_stim, a1, 'b', label = 'all inh')
    plt.fill_between(time_aligned_stim, a2-s2, a2+s2, alpha=0.5, edgecolor='k', facecolor='k')
    plt.plot(time_aligned_stim, a2, 'k')        
    
    plt.xlabel('Time since stimulus onset (ms)')
#     plt.ylabel('Classification accuracy (%)')    
    plt.ylabel('Classification accuracy (%)')    
    plt.xlim([time_aligned_stim[0], time_aligned_stim[-1]])
    
    plt.legend(loc='center left', bbox_to_anchor=(1, .7))



    
    # Plot corrClass for all excitatory neurons
    a1 = np.mean(corrClass_allExc, axis=1)*100
    s1 = np.std(corrClass_allExc, axis=1)*100 /np.sqrt(corrClass_allExc.shape[1])    

    a2 = np.mean(corrClassShfl_allExc, axis=(1,2))*100 # average across all shuffles and trials
    s2 = np.std(corrClassShfl_allExc, axis=(1,2))*100 /np.sqrt(numTrials*numSamples);
    '''
    # first average across shuffles, then across shuffle-averaged trials ... the std of this will be like the average std of s2 above across all times.
    a0 = np.mean(corrClassShfl, axis=2) # average across all shuffles
    a2 = np.mean(a0, axis=1)*100 # average across shuffle-averaged trials
    s2 = np.std(a0, axis=1)*100 /np.sqrt(numTrials);
    '''
    
    plt.fill_between(time_aligned_stim, a1-s1, a1+s1, alpha=0.3, edgecolor='m', facecolor='m')
    plt.plot(time_aligned_stim, a1, 'm', label = 'all exc')
    plt.fill_between(time_aligned_stim, a2-s2, a2+s2, alpha=0.5, edgecolor='k', facecolor='k')
    plt.plot(time_aligned_stim, a2, 'k')    

    plt.xlabel('Time since stimulus onset (ms)')
#     plt.ylabel('Classification accuracy (%)')    
    plt.ylabel('Classification accuracy (%)')    
    plt.xlim([time_aligned_stim[0], time_aligned_stim[-1]])
    
    
    
    
    # Plot corrClass for n excitatory neurons
    a1 = np.mean(corrClass_exc, axis=1)*100
    s1 = np.std(corrClass_exc, axis=1)*100 /np.sqrt(corrClass_exc.shape[1])    

    a2 = np.mean(corrClassShfl_exc, axis=(1,2))*100 # average across all shuffles and trials
    s2 = np.std(corrClassShfl_exc, axis=(1,2))*100 /np.sqrt(np.prod(np.array(np.shape(corrClassShfl_exc))[[1,2]]));
    
    plt.fill_between(time_aligned_stim, a1-s1, a1+s1, alpha=0.3, edgecolor='r', facecolor='r')
    plt.plot(time_aligned_stim, a1, 'r', label = 'n exc')
    plt.fill_between(time_aligned_stim, a2-s2, a2+s2, alpha=0.5, edgecolor='k', facecolor='k')
    plt.plot(time_aligned_stim, a2, 'k')    
    
    plt.xlabel('Time since stimulus onset (ms)')
#     plt.ylabel('Classification accuracy (%)')    
    plt.ylabel('Classification accuracy (%)')    
    plt.xlim([time_aligned_stim[0], time_aligned_stim[-1]])
    
    
    








#%%
####################################################################################################################################
############################################ Save results as .mat files in a folder named svm ####################################################################################################################################
####################################################################################################################################


# In[526]:

if trialHistAnalysis:
#     ep_ms = np.round((ep-eventI)*frameLength)
    th_stim_dur = []
    svmn = 'excInh_SVMtrained_prevChoice_%sN_%sITIs_ep%d-%dms_%s_' %(ntName, itiName, ep_ms[0], ep_ms[-1], nowStr)
else:
    svmn = 'excInh_SVMtrained_currChoice_%sN_ep%d-%dms_%s_' %(ntName, ep_ms[0], ep_ms[-1], nowStr)   
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
                           'regType':regType, 'cvect':cvect, 
                           'cbestAll_inh':cbestAll_inh,
                           'cbest_inh':cbest_inh,
                           'perClassErrorTrain_data_inh':perClassErrorTrain_data_inh,
                           'perClassErrorTest_data_inh':perClassErrorTest_data_inh,
                           'w_data_inh':w_data_inh,
                           'b_data_inh':b_data_inh,
                           'cbestAll_allExc':cbestAll_allExc,
                           'cbest_allExc':cbest_allExc,
                           'perClassErrorTrain_data_allExc':perClassErrorTrain_data_allExc,
                           'perClassErrorTest_data_allExc':perClassErrorTest_data_allExc,
                           'w_data_allExc':w_data_allExc,
                           'b_data_allExc':b_data_allExc,
                           'cbestAll_exc':cbestAll_exc,
                           'cbest_exc':cbest_exc,
                           'perClassErrorTrain_data_exc':perClassErrorTrain_data_exc,
                           'perClassErrorTest_data_exc':perClassErrorTest_data_exc,
                           'w_data_exc':w_data_exc,
                           'b_data_exc':b_data_exc,
                           'perClassErrorTrain_shfl_inh':perClassErrorTrain_shfl_inh,
                           'perClassErrorTest_shfl_inh':perClassErrorTest_shfl_inh,
                           'w_shfl_inh':w_shfl_inh,
                           'b_shfl_inh':b_shfl_inh,
                           'perClassErrorTrain_shfl_allExc':perClassErrorTrain_shfl_allExc,
                           'perClassErrorTest_shfl_allExc':perClassErrorTest_shfl_allExc,
                           'w_shfl_allExc':w_shfl_allExc,
                           'b_shfl_allExc':b_shfl_allExc,
                           'perClassErrorTrain_shfl_exc':perClassErrorTrain_shfl_exc,
                           'perClassErrorTest_shfl_exc':perClassErrorTest_shfl_exc,
                           'w_shfl_exc':w_shfl_exc,
                           'b_shfl_exc':b_shfl_exc,
                           'corrClass_inh':corrClass_inh,
                           'corrClassShfl_inh':corrClassShfl_inh,
                           'corrClass_exc':corrClass_exc,
                           'corrClassShfl_exc':corrClassShfl_exc,
                           'corrClass_allExc':corrClass_allExc,
                           'corrClassShfl_allExc':corrClassShfl_allExc}) 

else:
    print 'Not saving .mat file'
    
    
    
    
    