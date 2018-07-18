# -*- coding: utf-8 -*-
"""
Get distributions of all event times (stimulus offset, go tone, choice, etc), pooled across all days, from stim onset and well as choice onset.
Also get reaction times (go tone to choice time)

Plots will be saved in folder: classAccurTraces_eachFrame_timeDists

If you want to exclude trials, you need to set the svmName file which contains trsExcluded

Created on Thu Apr  6 16:46:51 2017
@author: farznaj
"""

mousename = 'fni19' #'fni19' #


excludeTrs = 1 # If 1, some trials will be excluded from the dists; You need to set the svmName file which contains trsExcluded # I think it maks sense to set excludeTrs to 0 bc those trials that are nan will have nan event Times...

# below matters if excludeTrs=1
testIncorr = 1 # set to 1 for the latest SVM file (it also includes testing incorr trs)
decodeStimCateg = 0
noExtraStimDayAnalysis = 0 # if 1, exclude days with extraStim from analysis

savefigs = 1


corrTrained = 1 # whether svm was trained using only correct trials (set to 1) or all trials (set to 0).
thTrained = 10 # number of trials of each class used for svm training, min acceptable value to include a day in analysis
doPlots = 0 #1 # plot event times for each day
shflTrsEachNeuron = 0
useEqualTrNums = 1
import numpy as np

doInhAllexcEqexc = [1,0,0,0]
if testIncorr:
    doInhAllexcEqexc[3]=-1
elif decodeStimCateg:
    doInhAllexcEqexc[3]=0
else:
    doInhAllexcEqexc = np.delete(doInhAllexcEqexc, -1)
        
ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone.
if mousename == 'fni18': #set one of the following to 1:
    allDays = 1# all 7 days will be used (last 3 days have z motion!)
    noZmotionDays = 0 # 4 days that dont have z motion will be used.
    noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!
    noExtraStimDays = 0
elif mousename == 'fni19':    
    allDays = 1
    noExtraStimDays = 0   
    noZmotionDays = 0 # 4 days that dont have z motion will be used.
    noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!    
else:    
    allDays = np.nan
    noZmotionDays = np.nan
    noZmotionDays_strict = np.nan
    noExtraStimDays = np.nan

    
trialHistAnalysis = 0
iTiFlg = 2 # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]

execfile("defFuns.py")

if decodeStimCateg:
    corrTrained=0
    
if corrTrained==0:
    corrn = 'allOutcomeTrained_'
else:
    corrn = 'corrTrained_'
    
if decodeStimCateg:
    corrn = 'decodeStimCateg_' + corrn

if noExtraStimDayAnalysis:
    corrn = corrn + 'noExtraStim_'
    
if excludeTrs:
    corrn = corrn + 'onlySVMtrs_'   
else:
    corrn = corrn + 'allTrs_'
    
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.

dnow = '/timeDists/'+mousename+'/'
#dnow = '/classAccurTraces_eachFrame_timeDists/'+mousename+'/'
    
from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')

    
#%% Set days

#execfile("svm_plots_setVars_n.py")  
#days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays)
days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays, loadYtest=0, decodeStimCateg=decodeStimCateg)


#%%
timeStimOnset_all = []
timeSingleStimOffset_all = []
timeStimOffset_all = []
timeCommitCL_CR_Gotone_all = []
time1stSideTry_all = []
timeInitTone_1st_all = []
timeInitTone_last_all = []
timeReward_all = []
timeCommitIncorrResp_all = []
time1stCorrectTry_all = []
time1stIncorrectTry_all = []
corr_hr_lr = np.full((len(days),2), np.nan) # number of hr, lr correct trials for each day
fractTrs_50msExtraStim = np.full(len(days), np.nan)    
   
# iday = 35 # fni17: 151015_1   
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
    

    #%% Set number of hr, lr trials that were used for svm training
    
    svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, [1,0,0], regressBins, useEqualTrNums, corrTrained, shflTrsEachNeuron)[0]   
    
    corr_hr, corr_lr = set_corr_hr_lr(postName, svmName)    
    corr_hr_lr[iday,:] = [corr_hr, corr_lr]        

    
    #%% Load time of some trial events    
    
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
        timeSingleStimOffset = np.array(Data.pop('timeSingleStimOffset')).flatten().astype('float')
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
        timeSingleStimOffset = timeStimOnset + sdur    

    
    #%%
    fractTrs_50msExtraStim[iday] = np.mean((timeStimOffset - timeSingleStimOffset)>50) 

            
    #%% Load trsExcluded if you want to check the dist of only those trials that went into SVM     
    
    if excludeTrs==1:
#        svmName = setSVMname(pnevFileName, trialHistAnalysis, chAl, regressBins) # for chAl: the latest file is with soft norm; earlier file is 
        svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc, regressBins=3, useEqualTrNums=1, corrTrained=corrTrained, shflTrsEachNeuron=shflTrsEachNeuron, shflTrLabs=0)
        svmName = svmName[0]
        print os.path.basename(svmName)    
    
        Data = scio.loadmat(svmName, variable_names=['trsExcluded'])    
        trsExcluded = Data.pop('trsExcluded').astype('bool').squeeze()                    
    
    
        ######### Exclude trsExcluded from event times
        timeStimOnset[trsExcluded] = np.nan
        timeSingleStimOffset[trsExcluded] = np.nan    
        timeStimOffset[trsExcluded] = np.nan
        timeCommitCL_CR_Gotone[trsExcluded] = np.nan    
        time1stSideTry[trsExcluded] = np.nan    
        timeInitTone_1st[trsExcluded] = np.nan    
        timeInitTone_last[trsExcluded] = np.nan    
        timeReward[trsExcluded] = np.nan    
        timeCommitIncorrResp[trsExcluded] = np.nan    
        time1stCorrectTry[trsExcluded] = np.nan    
        time1stIncorrectTry[trsExcluded] = np.nan    
    
    
    
    #%% Keep vars for all days
    
    timeStimOnset_all.append(timeStimOnset)
    timeSingleStimOffset_all.append(timeSingleStimOffset)
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
    
    
    #%% Plots   
    
    if doPlots:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(timeCommitCL_CR_Gotone - timeStimOnset, 'b', label = 'goTone')
        plt.plot(timeStimOffset - timeStimOnset, 'r', label = 'stimOffset')
        plt.plot(timeSingleStimOffset - timeStimOnset, 'k', label = 'stimOffset_1rep')
        plt.plot(time1stSideTry - timeStimOnset, 'm', label = '1stSideTry')
        plt.plot(timeReward - timeStimOnset, 'g', label = 'reward')
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
        a = timeSingleStimOffset - timeStimOnset
        plt.hist(a[~np.isnan(a)], color='k')
        a = time1stSideTry - timeStimOnset
        plt.hist(a[~np.isnan(a)], color='m')
        a = timeReward - timeStimOnset
        plt.hist(a[~np.isnan(a)], color='g')
        plt.xlabel('Time relative to stim onset (ms)')
        plt.ylabel('# trials')
        


#%%
mn_corr = np.min(corr_hr_lr,axis=1) # number of trials of each class. 90% of this was used for training, and 10% for testing.

print 'num days to be excluded with few svm-trained trs:', sum(mn_corr < thTrained)    
print np.array(days)[mn_corr < thTrained]


####### Set days without extraStim
#    days=np.array(days); days[fractTrs_50msExtraStim<.1]
if noExtraStimDayAnalysis: # dont analyze days with extra stim
    days2an_noExtraStim = np.logical_and(fractTrs_50msExtraStim < .1 , mn_corr >= thTrained)
else:
    days2an_noExtraStim = mn_corr >= thTrained #fractTrs_50msExtraStim < 2 # include all days
    
days2an_final = days2an_noExtraStim
    
    
#############################################################################################
#############################################################################################
        
#%% For each session subtract all event times from stimOnset times

lenD = len(timeSingleStimOffset_all)

############################### Relative to stimulus onset ###############################
timeSingleStimOffset_all_relOn = np.array([timeSingleStimOffset_all[iday] - timeStimOnset_all[iday] for iday in range(lenD)])
timeStimOffset_all_relOn = np.array([timeStimOffset_all[iday] - timeStimOnset_all[iday] for iday in range(lenD)])
timeCommitCL_CR_Gotone_all_relOn = np.array([timeCommitCL_CR_Gotone_all[iday] - timeStimOnset_all[iday] for iday in range(lenD)])
time1stSideTry_all_relOn = np.array([time1stSideTry_all[iday] - timeStimOnset_all[iday] for iday in range(lenD)])
timeInitTone_1st_all_relOn = np.array([timeInitTone_1st_all[iday] - timeStimOnset_all[iday] for iday in range(lenD)])
timeInitTone_last_all_relOn = np.array([timeInitTone_last_all[iday] - timeStimOnset_all[iday] for iday in range(lenD)])
timeReward_all_relOn = np.array([timeReward_all[iday] - timeStimOnset_all[iday] for iday in range(lenD)])
timeCommitIncorrResp_all_relOn = np.array([timeCommitIncorrResp_all[iday] - timeStimOnset_all[iday] for iday in range(lenD)])
time1stCorrectTry_all_relOn = np.array([time1stCorrectTry_all[iday] - timeStimOnset_all[iday] for iday in range(lenD)])
time1stIncorrectTry_all_relOn = np.array([time1stIncorrectTry_all[iday] - timeStimOnset_all[iday] for iday in range(lenD)])

time1stSideTry_all_relGo = np.array([time1stSideTry_all[iday] - timeCommitCL_CR_Gotone_all[iday] for iday in range(lenD)]) # reaction time... from go tone to choice


#%% Pool all trials of all days... only use the good days
 
initTone_1st_pooled = np.concatenate((timeInitTone_1st_all_relOn[days2an_final]))
initTone_1st_pooled = initTone_1st_pooled[~np.isnan(initTone_1st_pooled)]
print initTone_1st_pooled.shape

initTone_last_pooled = np.concatenate((timeInitTone_last_all_relOn[days2an_final]))
initTone_last_pooled = initTone_last_pooled[~np.isnan(initTone_last_pooled)]
print initTone_last_pooled.shape
 
stimOff0_pooled = np.concatenate((timeSingleStimOffset_all_relOn[days2an_final]))
#print stimOff0_pooled.shape
stimOff0_pooled = stimOff0_pooled[~np.isnan(stimOff0_pooled)]
print stimOff0_pooled.shape

stimOff_pooled = np.concatenate((timeStimOffset_all_relOn[days2an_final]))
#print stimOff_pooled.shape
stimOff_pooled = stimOff_pooled[~np.isnan(stimOff_pooled)]
print stimOff_pooled.shape

goTone_pooled = np.concatenate((timeCommitCL_CR_Gotone_all_relOn[days2an_final]))
#print goTone_pooled.shape
goTone_pooled = goTone_pooled[~np.isnan(goTone_pooled)]
print goTone_pooled.shape

choice_pooled = np.concatenate((time1stSideTry_all_relOn[days2an_final]))
#print choice_pooled.shape
choice_pooled = choice_pooled[~np.isnan(choice_pooled)]
print choice_pooled.shape

choiceCorr_pooled = np.concatenate((time1stCorrectTry_all_relOn[days2an_final]))
choiceCorr_pooled = choiceCorr_pooled[~np.isnan(choiceCorr_pooled)]
print choiceCorr_pooled.shape

choiceIncorr_pooled = np.concatenate((time1stIncorrectTry_all_relOn[days2an_final]))
choiceIncorr_pooled = choiceIncorr_pooled[~np.isnan(choiceIncorr_pooled)]
print choiceIncorr_pooled.shape

rew_pooled = np.concatenate((timeReward_all_relOn[days2an_final]))
rew_pooled = rew_pooled[~np.isnan(rew_pooled)]
print rew_pooled.shape

punish_pooled = np.concatenate((timeCommitIncorrResp_all_relOn[days2an_final]))
punish_pooled = punish_pooled[~np.isnan(punish_pooled)]
print punish_pooled.shape


rt_pooled = np.concatenate((time1stSideTry_all_relGo[days2an_final]))
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

xl = plt.gca().get_xlim()
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

xll = plt.gca().get_xlim()
mnxl = min([xl[0], xll[0]])
mxxl = max([xl[1], xll[1]])

plt.subplot(211)
plt.xlim([mnxl, mxxl])

plt.subplot(212)
plt.xlim([mnxl, mxxl])



#%%
############################### Relative to choice onset ###############################

#%%
timeStimOnset_all_relCh = np.array([timeStimOnset_all[iday] - time1stSideTry_all[iday] for iday in range(lenD)])
timeSingleStimOffset_all_relCh = np.array([timeSingleStimOffset_all[iday] - time1stSideTry_all[iday] for iday in range(lenD)])
timeStimOffset_all_relCh = np.array([timeStimOffset_all[iday] - time1stSideTry_all[iday] for iday in range(lenD)])
timeCommitCL_CR_Gotone_all_relCh = np.array([timeCommitCL_CR_Gotone_all[iday] - time1stSideTry_all[iday] for iday in range(lenD)])
#time1stSideTry_all_relCh = np.array([time1stSideTry_all[iday] - time1stSideTry_all[iday] for iday in range(lenD)])
timeInitTone_1st_all_relCh = np.array([timeInitTone_1st_all[iday] - time1stSideTry_all[iday] for iday in range(lenD)])
timeInitTone_last_all_relCh = np.array([timeInitTone_last_all[iday] - time1stSideTry_all[iday] for iday in range(lenD)])
timeReward_all_relCh = np.array([timeReward_all[iday] - time1stSideTry_all[iday] for iday in range(lenD)])
timeCommitIncorrResp_all_relCh = np.array([timeCommitIncorrResp_all[iday] - time1stSideTry_all[iday] for iday in range(lenD)])
time1stCorrectTry_all_relCh = np.array([time1stCorrectTry_all[iday] - time1stSideTry_all[iday] for iday in range(lenD)])
time1stIncorrectTry_all_relCh = np.array([time1stIncorrectTry_all[iday] - time1stSideTry_all[iday] for iday in range(lenD)])


#%% Pool all trials of all days... only good days
 
initTone_1st_pooled_relCh = np.concatenate((timeInitTone_1st_all_relCh[days2an_final]))
initTone_1st_pooled_relCh = initTone_1st_pooled_relCh[~np.isnan(initTone_1st_pooled_relCh)]
print initTone_1st_pooled_relCh.shape

initTone_last_pooled_relCh = np.concatenate((timeInitTone_last_all_relCh[days2an_final]))
initTone_last_pooled_relCh = initTone_last_pooled_relCh[~np.isnan(initTone_last_pooled_relCh)]
print initTone_last_pooled_relCh.shape

stimOn_pooled_relCh = np.concatenate((timeStimOnset_all_relCh[days2an_final]))
stimOn_pooled_relCh = stimOn_pooled_relCh[~np.isnan(stimOn_pooled_relCh)]
print stimOn_pooled_relCh.shape

stimOff0_pooled_relCh = np.concatenate((timeSingleStimOffset_all_relCh[days2an_final]))
#print stimOff0_pooled_relCh.shape
stimOff0_pooled_relCh = stimOff0_pooled_relCh[~np.isnan(stimOff0_pooled_relCh)]
print stimOff0_pooled_relCh.shape

stimOff_pooled_relCh = np.concatenate((timeStimOffset_all_relCh[days2an_final]))
#print stimOff_pooled_relCh.shape
stimOff_pooled_relCh = stimOff_pooled_relCh[~np.isnan(stimOff_pooled_relCh)]
print stimOff_pooled_relCh.shape

goTone_pooled_relCh = np.concatenate((timeCommitCL_CR_Gotone_all_relCh[days2an_final]))
#print goTone_pooled_relCh.shape
goTone_pooled_relCh = goTone_pooled_relCh[~np.isnan(goTone_pooled_relCh)]
print goTone_pooled_relCh.shape

'''
choice_pooled_relCh = np.concatenate((time1stSideTry_all_relCh[days2an_final]))
#print choice_pooled_relCh.shape
choice_pooled_relCh = choice_pooled_relCh[~np.isnan(choice_pooled_relCh)]
print choice_pooled_relCh.shape
'''

choiceCorr_pooled_relCh = np.concatenate((time1stCorrectTry_all_relCh[days2an_final]))
choiceCorr_pooled_relCh = choiceCorr_pooled_relCh[~np.isnan(choiceCorr_pooled_relCh)]
print choiceCorr_pooled_relCh.shape

choiceIncorr_pooled_relCh = np.concatenate((time1stIncorrectTry_all_relCh[days2an_final]))
choiceIncorr_pooled_relCh = choiceIncorr_pooled_relCh[~np.isnan(choiceIncorr_pooled_relCh)]
print choiceIncorr_pooled_relCh.shape

rew_pooled_relCh = np.concatenate((timeReward_all_relCh[days2an_final]))
rew_pooled_relCh = rew_pooled_relCh[~np.isnan(rew_pooled_relCh)]
print rew_pooled_relCh.shape

punish_pooled_relCh = np.concatenate((timeCommitIncorrResp_all_relCh[days2an_final]))
punish_pooled_relCh = punish_pooled_relCh[~np.isnan(punish_pooled_relCh)]
print punish_pooled_relCh.shape





#%% Plot dist of stimOffset, goTone, choice, reaction time

binwidth = 100

#plt.figure(figsize=(4.5,6))
plt.figure(figsize=(3,8))

############################### Relative to stimulus onset ###############################
plt.subplot(311)
"""
plt.hist(stimOff0_pooled, color='k', alpha=.5, label = 'stimOffset_1rep', bins=range(min(stimOff0_pooled), max(stimOff0_pooled) + binwidth, binwidth))
plt.hist(stimOff_pooled, color='r', alpha=.5, label = 'stimOffset')       
plt.hist(goTone_pooled, color='b', alpha=.5, label = 'goTone')
plt.hist(choice_pooled, color='m', alpha=.5, label = '1stSideTry')
"""
colors = 'k','r','b','m','g'
labs = 'stimOffset_1rep', 'stimOffset', 'goTone', '1stSideTry', 'reward'
a = stimOff0_pooled, stimOff_pooled, goTone_pooled, choice_pooled, rew_pooled
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
makeNicePlots(ax)#,0,1)
yl = ax.get_ylim()
plt.ylim([yl[0]-np.diff(yl)/20, yl[1]+np.diff(yl)/20])
xl = ax.get_xlim()
xlmn = np.min([np.percentile(a[i],1) for i in range(len(a))])
xlmx = np.max([np.percentile(a[i],99) for i in range(len(a))])
plt.xlim([xlmn, xlmx])
#plt.ylim([-yl[1]/20, yl[1]+yl[1]/20])
#plt.ylim([-yl[1]/20, yl[1]])
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

makeNicePlots(ax,0,1)
yl = ax.get_ylim()
plt.ylim([yl[0]-np.diff(yl)/20, yl[1]+np.diff(yl)/20])
#plt.ylim([-yl[1]/20, yl[1]+yl[1]/20])



############################### Relative to choice onset ###############################
plt.subplot(313)
"""
plt.hist(stimOff0_pooled_relCh, color='k', alpha=.5, label = 'stimOffset_1rep', bins=range(min(stimOff0_pooled_relCh), max(stimOff0_pooled_relCh) + binwidth, binwidth))
plt.hist(stimOff_pooled_relCh, color='r', alpha=.5, label = 'stimOffset')       
plt.hist(goTone_pooled_relCh, color='b', alpha=.5, label = 'goTone')
plt.hist(choice_pooled_relCh, color='m', alpha=.5, label = '1stSideTry')
"""

#colors = 'm','k','r','b','g' # 'k','r','b','m','g'
#labs = 'stimOnset', 'stimOffset_1rep', 'stimOffset', 'goTone', 'reward'
#a = stimOn_pooled_relCh, stimOff0_pooled_relCh, stimOff_pooled_relCh, goTone_pooled_relCh, rew_pooled_relCh

#### no stimOffset1rep
colors = 'm','r','b','g'
labs = 'stimOnset', 'stimOffset', 'goTone', 'reward'
a = stimOn_pooled_relCh, stimOff_pooled_relCh, goTone_pooled_relCh, rew_pooled_relCh

#### only go tone and reward
#colors = 'b','g'
#labs = 'goTone', 'reward'
#a = goTone_pooled_relCh, rew_pooled_relCh


binEvery = 100
bn = np.arange(np.min(np.concatenate((a))), np.max(np.concatenate((a))), binEvery)
bn[-1] = np.max(np.concatenate(a)) # unlike digitize, histogram doesn't count the right most value

# set hists
for i in range(len(a)):
    hist, bin_edges = np.histogram(a[i], bins=bn)
#    hist = hist/float(np.sum(hist))     # use this if you want to get fraction of trials instead of number of trials
    plt.bar(bin_edges[0:-1], hist, binEvery, alpha=.4, facecolor=colors[i], label=labs[i], edgecolor=colors[i]) 

plt.xlabel('Time relative to choice onset (ms)')
plt.ylabel('Number of trials')
plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 

ax = plt.gca()
yl = ax.get_ylim()
plt.ylim([yl[0]-np.diff(yl)/20, yl[1]+np.diff(yl)/20])
xl = ax.get_xlim()
xlmn = np.min([np.percentile(a[i],1) for i in range(len(a))])
xlmx = np.max([np.percentile(a[i],99) for i in range(len(a))])
plt.xlim([xlmn, xlmx])
#plt.xlim([-1000, 800]) # partOfX
makeNicePlots(ax,1,0)


plt.subplots_adjust(hspace=0.65)


##%%
##%% Save the figure    
if savefigs:
    dd = 'timeDists_allDaysPooled_' + corrn +  days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
#    dd = 'timeDists_allDaysPooled_' + corrn +  days[35][0:6] + '_' + nowStr
#    dd = 'timeDists_allDaysPooled_' + corrn +  days[35][0:6] + '_partOfX_' + nowStr
#    dd = 'timeDists_allDaysPooled_' + corrn + days[0][0:6] + '-to-' + days[-1][0:6] + '_partOfX' + '_' + nowStr
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, dd+'.'+fmt[0])
    
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)


                
        
        
        

