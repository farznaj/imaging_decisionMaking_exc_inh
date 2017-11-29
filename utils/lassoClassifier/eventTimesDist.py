# -*- coding: utf-8 -*-
"""
Get distributions of all event times (stimulus offset, go tone, choice, etc), pooled across all days, from stim onset and well as choice onset.
Also get reaction times (go tone to choice time)

Plots will be saved in folder: classAccurTraces_eachFrame_timeDists

If you want to exclude trials, you need to set the svmName file which contains trsExcluded

Created on Thu Apr  6 16:46:51 2017
@author: farznaj
"""


#mousename = 'fni19' #'fni17'
#savefigs = 1
doPlots = 0 #1 # plot event times for each day
# I think it maks sense to set excludeTrs to 0 bc those trials that are nan will have nan event Times...
excludeTrs = 0 # If 1, some trials will be excluded from the dists; You need to set the svmName file which contains trsExcluded
trialHistAnalysis = 0

execfile("defFuns.py")
#execfile("svm_plots_setVars_n.py")  
days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays)


#%% Set the following vars:
'''
mousename = 'fni17' #'fni17'

trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
execfile("svm_plots_setVars.py")  

savefigs = 1
doPlots = 0 #1 # plot dist of each day 

chAl = 0 # Related to setting svmName; If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
'''

#%% 
import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.

dnow = '/classAccurTraces_eachFrame_timeDists/'+mousename+'/'


#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName

def setSVMname(pnevFileName, trialHistAnalysis, chAl, regressBins=3):
    import glob

    if chAl==1:
        al = 'chAl'
    else:
        al = 'stAl'

    if trialHistAnalysis:
        svmn = 'svmPrevChoice_eachFrame_%s_ds%d_*' %(al,regressBins)
    else:
        svmn = 'svmCurrChoice_eachFrame_%s_ds%d_*' %(al,regressBins)
    
    svmn = svmn + os.path.basename(pnevFileName) #pnevFileName[-32:]    
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
    svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    
#    if len(svmName)>0:
#        svmName = svmName[0] # get the latest file
#    else:
#        svmName = ''
#    
    return svmName
    

#%%
timeStimOnset_all = []
timeStimOffset0_all = []
timeStimOffset_all = []
timeCommitCL_CR_Gotone_all = []
time1stSideTry_all = []
timeInitTone_1st_all = []
timeInitTone_last_all = []
timeReward_all = []
timeCommitIncorrResp_all = []
time1stCorrectTry_all = []
time1stIncorrectTry_all = []
   
   
for iday in range(len(days)):    
    #%% Plot distributions of go tone, stim, choice times
    
    imagingFolder = days[iday][0:6]; #'151013'
    mdfFileNumber = map(int, (days[iday][7:]).split("-")); #[1,2] 
        
    ##%% Set .mat file names
    pnev2load = [] #[] [3] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.
    signalCh = [2] # since gcamp is channel 2, should be always 2.
    postNProvided = 1; # If your directory does not contain pnevFile and instead it contains postFile, set this to 1 to get pnevFileName
        
    imfilename, pnevFileName, dataPath = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load, postNProvided=postNProvided, nOuts=3)   
    postName = os.path.join(os.path.dirname(pnevFileName), 'post_'+os.path.basename(pnevFileName))
    
    # no need for the following bc we load trsExcluded later
#    Data = scio.loadmat(postName, variable_names=['outcomes'])
#    outcomes = (Data.pop('outcomes').astype('float'))[0,:]
    
    
    ######### Load time of some trial events    
    Data = scio.loadmat(postName, variable_names=['timeCommitCL_CR_Gotone', 'timeStimOnset', 'timeStimOffset', 'time1stSideTry', 'timeReward', 'timeCommitIncorrResp', 'time1stCorrectTry', 'time1stIncorrectTry'])
    timeCommitCL_CR_Gotone = np.array(Data.pop('timeCommitCL_CR_Gotone')).flatten().astype('float')
    timeStimOnset = np.array(Data.pop('timeStimOnset')).flatten().astype('float')
    timeStimOffset = np.array(Data.pop('timeStimOffset')).flatten().astype('float')
    time1stSideTry = np.array(Data.pop('time1stSideTry')).flatten().astype('float')    
    timeReward = np.array(Data.pop('timeReward')).flatten().astype('float')
    timeCommitIncorrResp = np.array(Data.pop('timeCommitIncorrResp')).flatten().astype('float')
    time1stCorrectTry = np.array(Data.pop('time1stCorrectTry')).flatten().astype('float')
    time1stIncorrectTry = np.array(Data.pop('time1stIncorrectTry')).flatten().astype('float')
    
    Data = scio.loadmat(postName, variable_names=['timeInitTone']) #,squeeze_me=True,struct_as_record=False)
    timeInitTone = np.array(Data.pop('timeInitTone')).flatten() #.astype('float') # it is a cell in matlab, which will be like a list in python
    timeInitTone_1st = np.array([timeInitTone[i][0] for i in range(len(timeInitTone))]).squeeze().astype('float') # first initTone of each trial
    timeInitTone_last = np.array([timeInitTone[i][-1] for i in range(len(timeInitTone))]).squeeze().astype('float') # last initTone of each trial
    
    # Set the time of stimOffset for a single repetition of the stim (without extra stim, etc)
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

    

    ######### load trsExcluded if you want to check the dist of only those trials that went into SVM     
    if excludeTrs==1:
        svmName = setSVMname(pnevFileName, trialHistAnalysis, chAl, regressBins) # for chAl: the latest file is with soft norm; earlier file is 
        svmName = svmName[0]
    #    print os.path.basename(svmName)    
    
        Data = scio.loadmat(svmName, variable_names=['trsExcluded'])    
        trsExcluded = Data.pop('trsExcluded').astype('bool').squeeze()                    
    
    
        ######### Exclude trsExcluded from event times
        timeStimOnset[trsExcluded] = np.nan
        timeStimOffset0[trsExcluded] = np.nan    
        timeStimOffset[trsExcluded] = np.nan
        timeCommitCL_CR_Gotone[trsExcluded] = np.nan    
        time1stSideTry[trsExcluded] = np.nan    
        timeInitTone_1st[trsExcluded] = np.nan    
        timeInitTone_last[trsExcluded] = np.nan    
        timeReward[trsExcluded] = np.nan    
        timeCommitIncorrResp[trsExcluded] = np.nan    
        time1stCorrectTry[trsExcluded] = np.nan    
        time1stIncorrectTry[trsExcluded] = np.nan    
    
    
    
    ######### Keep vars for all days
    timeStimOnset_all.append(timeStimOnset)
    timeStimOffset0_all.append(timeStimOffset0)
    timeStimOffset_all.append(timeStimOffset)
    timeCommitCL_CR_Gotone_all.append(timeCommitCL_CR_Gotone)    
    time1stSideTry_all.append(time1stSideTry)    
    timeInitTone_1st_all.append(timeInitTone_1st)    
    timeInitTone_last_all.append(timeInitTone_last)    
    timeReward_all.append(timeReward)
    timeCommitIncorrResp_all.append(timeCommitIncorrResp)
    time1stCorrectTry_all.append(time1stCorrectTry)
    time1stIncorrectTry_all.append(time1stIncorrectTry)
    
    
    ######### Print some values
    print '______________'
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
    
    
    ######### Plots    
    if doPlots:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(timeCommitCL_CR_Gotone - timeStimOnset, 'b', label = 'goTone')
        plt.plot(timeStimOffset - timeStimOnset, 'r', label = 'stimOffset')
        plt.plot(timeStimOffset0 - timeStimOnset, 'k', label = 'stimOffset_1rep')
        plt.plot(time1stSideTry - timeStimOnset, 'm', label = '1stSideTry')
    #        plt.plot([1, np.shape(timeCommitCL_CR_Gotone)[0]],[th_stim_dur, th_stim_dur], 'g:', label = 'th_stim_dur')
    #    plt.plot([1, np.shape(timeCommitCL_CR_Gotone)[0]],[ep_ms[-1], ep_ms[-1]], 'k:', label = 'epoch end')
        plt.xlabel('Trial')
        plt.ylabel('Time relative to stim onset (ms)')
        plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 
        ax = plt.gca(); a = ax.get_ylim()
        plt.ylim([np.nanmin(timeCommitCL_CR_Gotone-timeStimOnset)-100, a[1]])
        # minStimDurNoGoTone = np.nanmin(timeCommitCL_CR_Gotone - timeStimOnset); # this is the duration after stim onset during which no go tone occurred for any of the trials.
        # print 'minStimDurNoGoTone = %.2f ms' %minStimDurNoGoTone
    
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
        


#############################################################################################
        
#%% For each session subtract all event times from stimOnset times

timeStimOffset0_all_relOn = np.array([timeStimOffset0_all[iday] - timeStimOnset_all[iday] for iday in range(len(days))])
timeStimOffset_all_relOn = np.array([timeStimOffset_all[iday] - timeStimOnset_all[iday] for iday in range(len(days))])
timeCommitCL_CR_Gotone_all_relOn = np.array([timeCommitCL_CR_Gotone_all[iday] - timeStimOnset_all[iday] for iday in range(len(days))])
time1stSideTry_all_relOn = np.array([time1stSideTry_all[iday] - timeStimOnset_all[iday] for iday in range(len(days))])
timeInitTone_1st_all_relOn = np.array([timeInitTone_1st_all[iday] - timeStimOnset_all[iday] for iday in range(len(days))])
timeInitTone_last_all_relOn = np.array([timeInitTone_last_all[iday] - timeStimOnset_all[iday] for iday in range(len(days))])
timeReward_all_relOn = np.array([timeReward_all[iday] - timeStimOnset_all[iday] for iday in range(len(days))])
timeCommitIncorrResp_all_relOn = np.array([timeCommitIncorrResp_all[iday] - timeStimOnset_all[iday] for iday in range(len(days))])
time1stCorrectTry_all_relOn = np.array([time1stCorrectTry_all[iday] - timeStimOnset_all[iday] for iday in range(len(days))])
time1stIncorrectTry_all_relOn = np.array([time1stIncorrectTry_all[iday] - timeStimOnset_all[iday] for iday in range(len(days))])

time1stSideTry_all_relGo = np.array([time1stSideTry_all[iday] - timeCommitCL_CR_Gotone_all[iday] for iday in range(len(days))]) # reaction time... from go tone to choice


#%% Pool all trials of all days
 
initTone_1st_pooled = np.concatenate((timeInitTone_1st_all_relOn))
initTone_1st_pooled = initTone_1st_pooled[~np.isnan(initTone_1st_pooled)]
print initTone_1st_pooled.shape

initTone_last_pooled = np.concatenate((timeInitTone_last_all_relOn))
initTone_last_pooled = initTone_last_pooled[~np.isnan(initTone_last_pooled)]
print initTone_last_pooled.shape
 
stimOff0_pooled = np.concatenate((timeStimOffset0_all_relOn))
#print stimOff0_pooled.shape
stimOff0_pooled = stimOff0_pooled[~np.isnan(stimOff0_pooled)]
print stimOff0_pooled.shape

stimOff_pooled = np.concatenate((timeStimOffset_all_relOn))
#print stimOff_pooled.shape
stimOff_pooled = stimOff_pooled[~np.isnan(stimOff_pooled)]
print stimOff_pooled.shape

goTone_pooled = np.concatenate((timeCommitCL_CR_Gotone_all_relOn))
#print goTone_pooled.shape
goTone_pooled = goTone_pooled[~np.isnan(goTone_pooled)]
print goTone_pooled.shape

choice_pooled = np.concatenate((time1stSideTry_all_relOn))
#print choice_pooled.shape
choice_pooled = choice_pooled[~np.isnan(choice_pooled)]
print choice_pooled.shape

choiceCorr_pooled = np.concatenate((time1stCorrectTry_all_relOn))
choiceCorr_pooled = choiceCorr_pooled[~np.isnan(choiceCorr_pooled)]
print choiceCorr_pooled.shape

choiceIncorr_pooled = np.concatenate((time1stIncorrectTry_all_relOn))
choiceIncorr_pooled = choiceIncorr_pooled[~np.isnan(choiceIncorr_pooled)]
print choiceIncorr_pooled.shape

rew_pooled = np.concatenate((timeReward_all_relOn))
rew_pooled = rew_pooled[~np.isnan(rew_pooled)]
print rew_pooled.shape

punish_pooled = np.concatenate((timeCommitIncorrResp_all_relOn))
punish_pooled = punish_pooled[~np.isnan(punish_pooled)]
print punish_pooled.shape


rt_pooled = np.concatenate((time1stSideTry_all_relGo))
#print rt_pooled.shape
rt_pooled = rt_pooled[~np.isnan(rt_pooled)]
print rt_pooled.shape


#%% Plot dist of init tone

plt.figure()

plt.subplot(211)
plt.hist(initTone_1st_pooled, color='m', alpha=.5, label = 'initTone')

plt.subplot(212)
plt.hist(initTone_last_pooled, color='r', alpha=.5, label = 'initTone')


#%% Plot dist of 1stTry and commit resp for corr and incorr trials

plt.figure()

plt.subplot(211)
myarray = choiceCorr_pooled
weights = np.ones_like(myarray)/len(myarray)
plt.hist(choiceCorr_pooled, alpha=.5, label = '1stCorr', weights = weights)

myarray = choiceIncorr_pooled
weights = np.ones_like(myarray)/len(myarray)
plt.hist(choiceIncorr_pooled, alpha=.5, label = '1stIncorr', weights = weights)

plt.xlabel('Time relative to stim onset (ms)')
plt.ylabel('Number of trials')
plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 

_,p = stats.ttest_ind(choiceCorr_pooled, choiceIncorr_pooled, nan_policy = 'omit')
print p
print choiceCorr_pooled.mean(), choiceIncorr_pooled.mean()

###############
plt.subplot(212)
myarray = rew_pooled
weights = np.ones_like(myarray)/len(myarray)
plt.hist(rew_pooled, alpha=.5, label = 'reward', weights = weights)

myarray = punish_pooled
weights = np.ones_like(myarray)/len(myarray)
plt.hist(punish_pooled, alpha=.5, label = 'punish', weights = weights)

_,p = stats.ttest_ind(rew_pooled, punish_pooled, nan_policy = 'omit')
print p
print rew_pooled.mean(), punish_pooled.mean()


plt.xlabel('Time relative to stim onset (ms)')
plt.ylabel('Number of trials')
plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 



############################### Relative to stimulus onset ###############################

#%% Plot dist of stimOffset, goTone, choice, reaction time

binwidth = 100

plt.figure()

plt.subplot(311)
"""
plt.hist(stimOff0_pooled, color='k', alpha=.5, label = 'stimOffset_1rep', bins=range(min(stimOff0_pooled), max(stimOff0_pooled) + binwidth, binwidth))
plt.hist(stimOff_pooled, color='r', alpha=.5, label = 'stimOffset')       
plt.hist(goTone_pooled, color='b', alpha=.5, label = 'goTone')
plt.hist(choice_pooled, color='m', alpha=.5, label = '1stSideTry')
"""
colors = 'k','r','b','m'
labs = 'stimOffset_1rep', 'stimOffset', 'goTone', '1stSideTry'
a = stimOff0_pooled, stimOff_pooled, goTone_pooled, choice_pooled
binEvery = 100

bn = np.arange(np.min(np.concatenate((a))), np.max(np.concatenate((a))), binEvery)
bn[-1] = np.max(np.concatenate(a)) # unlike digitize, histogram doesn't count the right most value

# set hists
for i in range(len(a)):
    hist, bin_edges = np.histogram(a[i], bins=bn)
#    hist = hist/float(np.sum(hist))     # use this if you want to get fraction of trials instead of number of trials
    plt.bar(bin_edges[0:-1], hist, binEvery, alpha=.4, color=colors[i], label=labs[i]) 

plt.xlabel('Time relative to stim onset (ms)')
plt.ylabel('Number of trials')
plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 

ax = plt.gca();
makeNicePlots(ax)
yl = ax.get_ylim()
plt.ylim([-yl[1]/10, yl[1]])
#np.mean(goTone_pooled<999) # fraction of trials with go tone before stim offset



############## reaction time ##############

plt.subplot(312)
h = plt.hist(rt_pooled, color='k', alpha=.5, label = 'goTone to choice', bins = len(bn))
ax = plt.gca(); ymx = ax.get_ylim()
yAtMean = h[0][np.argwhere(rt_pooled.mean()<h[1])[0]]; y = np.mean([ymx[1], yAtMean])
plt.plot(rt_pooled.mean(), y ,'*r')

plt.xlabel('Reaction time (ms)')
plt.ylabel('Number of trials')
plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 

makeNicePlots(ax)

plt.subplots_adjust(hspace=0.65)





#%%
############################### Relative to choice onset ###############################

#%%
timeStimOnset_all_relCh = np.array([timeStimOnset_all[iday] - time1stSideTry_all[iday] for iday in range(len(days))])
timeStimOffset0_all_relCh = np.array([timeStimOffset0_all[iday] - time1stSideTry_all[iday] for iday in range(len(days))])
timeStimOffset_all_relCh = np.array([timeStimOffset_all[iday] - time1stSideTry_all[iday] for iday in range(len(days))])
timeCommitCL_CR_Gotone_all_relCh = np.array([timeCommitCL_CR_Gotone_all[iday] - time1stSideTry_all[iday] for iday in range(len(days))])
#time1stSideTry_all_relCh = np.array([time1stSideTry_all[iday] - time1stSideTry_all[iday] for iday in range(len(days))])
timeInitTone_1st_all_relCh = np.array([timeInitTone_1st_all[iday] - time1stSideTry_all[iday] for iday in range(len(days))])
timeInitTone_last_all_relCh = np.array([timeInitTone_last_all[iday] - time1stSideTry_all[iday] for iday in range(len(days))])
timeReward_all_relCh = np.array([timeReward_all[iday] - time1stSideTry_all[iday] for iday in range(len(days))])
timeCommitIncorrResp_all_relCh = np.array([timeCommitIncorrResp_all[iday] - time1stSideTry_all[iday] for iday in range(len(days))])
time1stCorrectTry_all_relCh = np.array([time1stCorrectTry_all[iday] - time1stSideTry_all[iday] for iday in range(len(days))])
time1stIncorrectTry_all_relCh = np.array([time1stIncorrectTry_all[iday] - time1stSideTry_all[iday] for iday in range(len(days))])


#%% Pool all trials of all days
 
initTone_1st_pooled = np.concatenate((timeInitTone_1st_all_relCh))
initTone_1st_pooled = initTone_1st_pooled[~np.isnan(initTone_1st_pooled)]
print initTone_1st_pooled.shape

initTone_last_pooled = np.concatenate((timeInitTone_last_all_relCh))
initTone_last_pooled = initTone_last_pooled[~np.isnan(initTone_last_pooled)]
print initTone_last_pooled.shape

stimOn_pooled = np.concatenate((timeStimOnset_all_relCh))
stimOn_pooled = stimOn_pooled[~np.isnan(stimOn_pooled)]
print stimOn_pooled.shape

stimOff0_pooled = np.concatenate((timeStimOffset0_all_relCh))
#print stimOff0_pooled.shape
stimOff0_pooled = stimOff0_pooled[~np.isnan(stimOff0_pooled)]
print stimOff0_pooled.shape

stimOff_pooled = np.concatenate((timeStimOffset_all_relCh))
#print stimOff_pooled.shape
stimOff_pooled = stimOff_pooled[~np.isnan(stimOff_pooled)]
print stimOff_pooled.shape

goTone_pooled = np.concatenate((timeCommitCL_CR_Gotone_all_relCh))
#print goTone_pooled.shape
goTone_pooled = goTone_pooled[~np.isnan(goTone_pooled)]
print goTone_pooled.shape

'''
choice_pooled = np.concatenate((time1stSideTry_all_relCh))
#print choice_pooled.shape
choice_pooled = choice_pooled[~np.isnan(choice_pooled)]
print choice_pooled.shape
'''

choiceCorr_pooled = np.concatenate((time1stCorrectTry_all_relCh))
choiceCorr_pooled = choiceCorr_pooled[~np.isnan(choiceCorr_pooled)]
print choiceCorr_pooled.shape

choiceIncorr_pooled = np.concatenate((time1stIncorrectTry_all_relCh))
choiceIncorr_pooled = choiceIncorr_pooled[~np.isnan(choiceIncorr_pooled)]
print choiceIncorr_pooled.shape

rew_pooled = np.concatenate((timeReward_all_relCh))
rew_pooled = rew_pooled[~np.isnan(rew_pooled)]
print rew_pooled.shape

punish_pooled = np.concatenate((timeCommitIncorrResp_all_relCh))
punish_pooled = punish_pooled[~np.isnan(punish_pooled)]
print punish_pooled.shape


#%%
plt.subplot(313)
"""
plt.hist(stimOff0_pooled, color='k', alpha=.5, label = 'stimOffset_1rep', bins=range(min(stimOff0_pooled), max(stimOff0_pooled) + binwidth, binwidth))
plt.hist(stimOff_pooled, color='r', alpha=.5, label = 'stimOffset')       
plt.hist(goTone_pooled, color='b', alpha=.5, label = 'goTone')
plt.hist(choice_pooled, color='m', alpha=.5, label = '1stSideTry')
"""
colors = 'k','r','b','m'
labs = 'stimOnset', 'stimOffset_1rep', 'stimOffset', 'goTone'
a = stimOn_pooled, stimOff0_pooled, stimOff_pooled, goTone_pooled
binEvery = 100

bn = np.arange(np.min(np.concatenate((a))), np.max(np.concatenate((a))), binEvery)
bn[-1] = np.max(np.concatenate(a)) # unlike digitize, histogram doesn't count the right most value

# set hists
for i in range(len(a)):
    hist, bin_edges = np.histogram(a[i], bins=bn)
#    hist = hist/float(np.sum(hist))     # use this if you want to get fraction of trials instead of number of trials
    plt.bar(bin_edges[0:-1], hist, binEvery, alpha=.4, color=colors[i], label=labs[i]) 

plt.xlabel('Time relative to choice onset (ms)')
plt.ylabel('Number of trials')
plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 

ax = plt.gca();
makeNicePlots(ax)
yl = ax.get_ylim()
plt.ylim([-100, yl[1]])
#np.mean(goTone_pooled<999) # fraction of trials with go tone before stim offset



##%% Save the figure    
if savefigs:
    dd = 'timeDists_allPooled'
    '''
    if chAl==1:
        dd = 'chAl_timeDists'
    else:
        dd = 'stAl_timeDists'
    '''
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)


