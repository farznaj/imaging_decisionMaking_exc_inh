# -*- coding: utf-8 -*-
"""
Plots class accuracy for svm trained on non-overlapping time windows  (outputs of file svm_eachFrame.py)
 ... svm trained to decode choice on choice-aligned or stimulus-aligned traces.
It will also call eventTimeDist.py to plot dist of event times and compare it with class acuur traces.
 
 
Remember for fni18 there are 2 svm_eachFrame mat files, the earlier file is using all trials (unequal HR, LR, like how you've done all your analysis). 
The later mat file is with equal number of hr and lr trials (subselecting trials)... this helped with 151209 class accur trace which was weird in the earlier mat file.
 
Created on Sun Mar 12 15:12:29 2017
@author: farznaj
"""

#%%
###########################################################################
######################### NOTE #########################
###########################################################################
# make sure to correct how you compute eventI_ds in this script if you run the new version of svm_eachFrame which downsamples the traces correctly.
##################################################
###########################################################################
###########################################################################

#%% Change the following vars:

mousename = 'fni18' #'fni17'

savefigs = 0

thTrained = 30#10 # number of trials of each class used for svm training, min acceptable value to include a day in analysis
thIncorr = 4#10 # number of trials of the fewer class in incorr trials, min acceptable value to include a day in analysis of incorr trials.
lastTimeBinMissed = 1 # if 0, things were run fine; if 1: by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
corrTrained = 1 # whether svm was trained using only correct trials (set to 1) or all trials (set to 0).
doIncorr = 1 # plot incorr class accur too
loadWeights = 0


ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone.
if mousename == 'fni18': #set one of the following to 1:
    allDays = 1# all 7 days will be used (last 3 days have z motion!)
    noZmotionDays = 0 # 4 days that dont have z motion will be used.
    noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!
elif mousename == 'fni19':    
    allDays = 1
    noExtraStimDays = 0   
else:
    import numpy as np
    allDays = np.nan
    noZmotionDays = np.nan
    noZmotionDays_strict = np.nan
    noExtraStimDays = np.nan

    
trialHistAnalysis = 0
iTiFlg = 2 # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
#execfile("svm_plots_setVars_n.py")  
days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays)

#chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]
doPlots = 0 #1 # plot c path of each day 

#eps = 10**-10 # tiny number below which weight is considered 0
#thNon0Ws = 2 # For samples with <2 non0 weights, we manually set their class error to 50 ... the idea is that bc of difference in number of HR and LR trials, in these samples class error is not accurately computed!
#thSamps = 10  # Days that have <thSamps samples that satisfy >=thNon0W non0 weights will be manually set to 50 (class error of all their samples) ... bc we think <5 samples will not give us an accurate measure of class error of a day.
#setTo50 = 1 # if 1, the above two jobs will be done.


import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.

dnow = '/classAccurTraces_eachFrame_timeDists/'+mousename+'/'
if corrTrained:
    dnow = dnow+'corrTrained/'
        
smallestC = 0 # Identify best c: if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
if smallestC==1:
    print 'bestc = smallest c whose cv error is less than 1se of min cv error'
else:
    print 'bestc = c that gives min cv error'
#I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.

from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')

execfile("defFuns.py")

   
#%% 
'''
#####################################################################################################################################################   
#####################################################################################################################################################
'''
            
#%% Loop over days    

classErr_bestC_train_data_all = [] # np.full((numSamp*nRandCorrSel, len(days)), np.nan)
classErr_bestC_test_data_all = []
classErr_bestC_test_shfl_all = []
classErr_bestC_test_chance_all = []
cbestFrs_all = []
classErr_bestC_test_incorr_all = []
classErr_bestC_test_incorr_shfl_all = []
classErr_bestC_test_incorr_chance_all = []
eventI_allDays = np.full((len(days)), np.nan)
eventI_ds_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
corr_hr_lr = np.full((len(days),2), np.nan) # number of hr, lr correct trials for each day
incorr_hr_lr = np.full((len(days),2), np.nan) # number of hr, lr incorrect trials for each day

for iday in range(len(days)): 

    #%%            
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


    #%%
    svmName = setSVMname_allN_eachFrame(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained) # for chAl: the latest file is with soft norm; earlier file is 
    svmName = svmName[0]
    print os.path.basename(svmName)    


    #%% Set eventI_ds (downsampled eventI)

    eventI, eventI_ds = setEventIds(postName, chAl, regressBins=3, trialHistAnalysis=0)
    
    eventI_allDays[iday] = eventI
    eventI_ds_allDays[iday] = eventI_ds    

    
    #%% Get number of hr, lr trials that were used for svm training
        
    corr_hr, corr_lr, incorr_hr, incorr_lr = set_corr_hr_lr(postName, svmName, doIncorr)
    corr_hr_lr[iday,:] = [corr_hr, corr_lr]        
    incorr_hr_lr[iday,:] = [incorr_hr, incorr_lr]    
    
    
    #%% Load SVM vars

    if doIncorr==1:
        classErr_bestC_train_data, classErr_bestC_test_data, classErr_bestC_test_shfl, classErr_bestC_test_chance, cbestFrs, w_bestc_data, b_bestc_data, classErr_bestC_test_incorr, classErr_bestC_test_incorr_shfl, classErr_bestC_test_incorr_chance = loadSVM_allN(svmName, doPlots, doIncorr, loadWeights)
    else:
        classErr_bestC_train_data, classErr_bestC_test_data, classErr_bestC_test_shfl, classErr_bestC_test_chance, cbestFrs, w_bestc_data, b_bestc_data = loadSVM_allN(svmName, doPlots, doIncorr, loadWeights)
    
        
            
    #%% Once done with all frames, save vars for all days
           
    # Delete vars before starting the next day           
#    del eventI, time_trace, perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance, perClassErrorTest_incorr, perClassErrorTest_incorr_shfl, perClassErrorTest_incorr_chance
#    if loadWeights==1:
#        del wAllC
        
    classErr_bestC_train_data_all.append(classErr_bestC_train_data) # each day: samps x numFrs
    classErr_bestC_test_data_all.append(classErr_bestC_test_data)
    classErr_bestC_test_shfl_all.append(classErr_bestC_test_shfl)
    classErr_bestC_test_chance_all.append(classErr_bestC_test_chance)
    cbestFrs_all.append(cbestFrs)            
    classErr_bestC_test_incorr_all.append(classErr_bestC_test_incorr)
    classErr_bestC_test_incorr_shfl_all.append(classErr_bestC_test_incorr_shfl)
    classErr_bestC_test_incorr_chance_all.append(classErr_bestC_test_incorr_chance)            

cbestFrs_all = np.array(cbestFrs_all) 
eventI_allDays = eventI_allDays.astype(int)   
eventI_ds_allDays = eventI_ds_allDays.astype(int)
#classErr_bestC_train_data_all = np.array(classErr_bestC_train_data_all)
#classErr_bestC_test_data_all = np.array(classErr_bestC_test_data_all)
#classErr_bestC_test_shfl_all = np.array(classErr_bestC_test_shfl_all)
#classErr_bestC_test_chance_all = np.array(classErr_bestC_test_chance_all)
numSamples = classErr_bestC_train_data[0].shape[0] 
numFrs = classErr_bestC_train_data[0].shape[1]


#%%
############################%%%%%%%%%%%%%%%%%%############################%%%%%%%%%%%%%%%%%%############################%%%%%%%%%%%%%%%%%%############################%%%%%%%%%%%%%%%%%%
############################%%%%%%%%%%%%%%%%%%############################%%%%%%%%%%%%%%%%%%############################%%%%%%%%%%%%%%%%%%############################%%%%%%%%%%%%%%%%%%
############################%%%%%%%%%%%%%%%%%%############################%%%%%%%%%%%%%%%%%%############################%%%%%%%%%%%%%%%%%%############################%%%%%%%%%%%%%%%%%%
############################%%%%%%%%%%%%%%%%%%############################%%%%%%%%%%%%%%%%%%############################%%%%%%%%%%%%%%%%%%############################%%%%%%%%%%%%%%%%%%

#%%
#import matplotlib as mpl
#mpl.style.use('default')
plt.style.use('default')


#%% Decide what days to analyze: exclude days with too few trials used for training SVM, also exclude incorr from days with too few incorr trials.

# th for min number of trs of each class
'''
thTrained = 30 #25; # 1/10 of this will be the testing tr num! and 9/10 was used for training
thIncorr = 4 #5
'''
mn_corr = np.min(corr_hr_lr,axis=1) # number of trials of each class. 90% of this was used for training, and 10% for testing.
mn_incorr = np.min(incorr_hr_lr,axis=1) # number of incorr trials, min of the 2 class

print 'num days to be excluded:\n\tlow trained trs:', sum(mn_corr < thTrained), '\n\tlow incorr trs', sum(mn_incorr < thIncorr)

print np.array(days)[mn_corr < thTrained]
print np.array(days)[mn_incorr < thIncorr]
#mn_incorr[mn_incorr < thIncorr]


#%% Plot number of trials used for training and incorr testing (remember testing data is 10% of mn_corr, and training is 90% of it!)

plt.figure(figsize=(4,3))
plt.plot(mn_corr, label='min(corr,hr lr)', color='b')
plt.plot(mn_incorr, label='min(incorr,hr lr)', color='g')
plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)
xl = plt.gca().get_xlim()
plt.hlines(thTrained, xl[0], xl[1], color='b', linestyle='--')
plt.hlines(thIncorr, xl[0], xl[1], color='g', linestyle='--')
yl = plt.gca().get_ylim()
plt.ylim([-10, yl[1]])
plt.xlabel('Days')
plt.ylabel('Number of trials')

##%% Save the figure of each day   
if savefigs:#% Save the figure
    if chAl==1:
        dd = 'chAl_nTrsSVMtrained'
    else:
        dd = 'stAl_nTrsSVMtrained'
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)            
    
'''
# compare cbest between frames
plt.figure(figsize=(3,2))    
plt.plot(cbestFrs); plt.xlabel('frames'); plt.ylabel('best C'); 
r = max(cbestFrs)-min(cbestFrs)
plt.ylim([-r/10, max(cbestFrs)+r/10])
plt.vlines(eventI_ds, -r/10, max(cbestFrs)+r/10, color='r') # mark eventI_ds
'''    


#%% Set vars for aligning class accur traces of all days on common eventI
 
##%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
# By common eventI, we  mean the index on which all traces will be aligned.
        
nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
for iday in range(numDays):
    nPost[iday] = (classErr_bestC_test_data_all[iday].shape[1] - eventI_ds_allDays[iday] - 1)

nPreMin = int(min(eventI_ds_allDays)) # number of frames before the common eventI, also the index of common eventI. 
nPostMin = int(min(nPost))
print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)


#%% Set the time array for the across-day aligned traces
'''
if corrTrained==0: # remember below has issues...
    a = -(np.asarray(frameLength*regressBins) * range(nPreMin+1)[::-1])
    b = (np.asarray(frameLength*regressBins) * range(1, nPostMin+1))
    time_aligned = np.concatenate((a,b))
else:
'''
totLen = nPreMin + nPostMin +1
time_aligned = set_time_al(totLen, min(eventI_allDays), lastTimeBinMissed)

print time_aligned




#%%
##################################################################################################################################################################
##################################################################################################################################################################

for ave in [0,1]: # whether to compute average or median across samples
    
    if ave==1: 
        avmd0 = 'ave'
        avmd1 = 'sd'
    else:
        avmd0 = 'med'
        avmd1 = 'iqr'
    
    
    #%% Set data - median(shfl) for all samples of each day
    
    # each data - med(shfl) for each day...so a dist of data-shfl for each day
    classAcc_bestC_test_dms_all = classErr_bestC_test_data_all+[0]
    for iday in range(len(days)):    # we do class error shfl - class error data, which is same as class accur data - shfl 
        # CA_dms = 100-CE_data  -  100-CE_shfl  =  CE_shfl - CE_data
        if ave==1:
            classAcc_bestC_test_dms_all[iday] = -classErr_bestC_test_data_all[iday] + np.mean(classErr_bestC_test_shfl_all[iday],axis=0)
        else:    
            classAcc_bestC_test_dms_all[iday] = -classErr_bestC_test_data_all[iday] + np.median(classErr_bestC_test_shfl_all[iday],axis=0)
    
    if doIncorr==1:    
        classAcc_bestC_test_incorr_dms_all = classErr_bestC_test_data_all+[0]
        for iday in range(len(days)):    
            if ave==1:
                classAcc_bestC_test_incorr_dms_all[iday] = -classErr_bestC_test_incorr_all[iday] + np.mean(classErr_bestC_test_incorr_shfl_all[iday],axis=0)
            else:
                classAcc_bestC_test_incorr_dms_all[iday] = -classErr_bestC_test_incorr_all[iday] + np.median(classErr_bestC_test_incorr_shfl_all[iday],axis=0)    
        
        
    
    #%% Align class accur traces that include all samps on the common eventI (you just aligned the ave traces above)
    ### excluding low-trial days
    
    #### align all samp traces of each day (you only aligned the ave samp traces above)
    # samps x frs(aligned) x days
    classAcc_train_d_aligned = np.ones((numSamples, nPreMin + nPostMin + 1, numDays)) + np.nan # samps x frames x days, aligned on common eventI (equals nPreMin)
    classAcc_test_d_aligned = np.ones((numSamples, nPreMin + nPostMin + 1, numDays)) + np.nan # samps x frames x days, aligned on common eventI (equals nPreMin)
    classAcc_test_s_aligned = np.ones((numSamples, nPreMin + nPostMin + 1, numDays)) + np.nan
    classAcc_test_c_aligned = np.ones((numSamples, nPreMin + nPostMin + 1, numDays)) + np.nan
    classAcc_test_dms_aligned = np.ones((numSamples, nPreMin + nPostMin + 1, numDays)) + np.nan # samps x frames x days, aligned on common eventI (equals nPreMin)
    if doIncorr==1:    
        classAcc_test_incorr_d_aligned = np.ones((numSamples, nPreMin + nPostMin + 1, numDays)) + np.nan # samps x frames x days, aligned on common eventI (equals nPreMin)
        classAcc_test_incorr_s_aligned = np.ones((numSamples, nPreMin + nPostMin + 1, numDays)) + np.nan
        classAcc_test_incorr_c_aligned = np.ones((numSamples, nPreMin + nPostMin + 1, numDays)) + np.nan
        classAcc_test_incorr_dms_aligned = np.ones((numSamples, nPreMin + nPostMin + 1, numDays)) + np.nan # samps x frames x days, aligned on common eventI (equals nPreMin)
        
    for iday in range(numDays):
        if mn_corr[iday] >= thTrained:
            classAcc_train_d_aligned[:,:, iday] = 100-classErr_bestC_train_data_all[iday][: , eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            classAcc_test_d_aligned[:,:, iday] = 100-classErr_bestC_test_data_all[iday][: , eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            classAcc_test_s_aligned[:,:, iday] = 100-classErr_bestC_test_shfl_all[iday][: , eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            classAcc_test_c_aligned[:,:, iday] = 100-classErr_bestC_test_chance_all[iday][: , eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]        
            classAcc_test_dms_aligned[:,:, iday] = classAcc_bestC_test_dms_all[iday][: , eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
        if doIncorr==1 and mn_incorr[iday] >= thIncorr:
            classAcc_test_incorr_d_aligned[:,:, iday] = 100-classErr_bestC_test_incorr_all[iday][: , eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            classAcc_test_incorr_s_aligned[:,:, iday] = 100-classErr_bestC_test_incorr_shfl_all[iday][: , eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            classAcc_test_incorr_c_aligned[:,:, iday] = 100-classErr_bestC_test_incorr_chance_all[iday][: , eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]        
            classAcc_test_incorr_dms_aligned[:,:, iday] = classAcc_bestC_test_incorr_dms_all[iday][: , eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            
    
    
        
    #%% Look at dists of CA: all samples pooled  (of all days and all frames) 
    
    def plotflastdist(l, col, lab='', p=np.nan):
        if isinstance(l, list):
            flat_list = [item for sublist in l for item in sublist.flatten()]
        else:
            flat_list = l.flatten()
        a = plt.hist(np.array(flat_list)[~np.isnan(flat_list)], alpha=.5, color=col, label=lab)
        # mark mean and median of dists
        plt.plot(np.nanmean(flat_list), max(a[0]), marker='*', color='k', markersize=6) # average of class accuracies
        plt.plot(np.nanmedian(flat_list), max(a[0]), marker='s', color='k', markersize=4) # median
        if (~np.isnan(p)).all(): # show p
            plt.title('p_wmt=%.3f %.3f %.3f'%(p[0], p[1], p[2]))
        plt.legend(loc=0)
    
        
    # samples are not averaged, pool all samples and all frames of all days
    plt.figure()
    plt.subplot(221)
    a = classAcc_test_d_aligned
    b = classAcc_test_incorr_d_aligned
    p = pwm(a,b)
    plotflastdist(a, 'b', lab='corr')  # samps x alFrs x days     #l = 100-classErr_bestC_test_data_all
    plotflastdist(b, 'r', lab='incorr', p=p)
    ax = plt.gca()
    makeNicePlots(ax)
    plt.xlabel('CA (data)')
    plt.ylabel('# (pooled samples,frames,days)')
        
    plt.subplot(222)
    a = classAcc_test_s_aligned
    b = classAcc_test_incorr_s_aligned
    p = pwm(a,b)
    plotflastdist(a, 'b')  # samps x alFrs x days     #l = 100-classErr_bestC_test_data_all
    plotflastdist(b, 'r', p=p)
    ax = plt.gca()
    makeNicePlots(ax)
    plt.xlabel('CA (shfl)')
    
    plt.subplot(223)
    a = classAcc_test_dms_aligned
    b = classAcc_test_incorr_dms_aligned
    p = pwm(a,b)
    plotflastdist(a, 'b')  # samps x alFrs x days     #l = 100-classErr_bestC_test_data_all
    plotflastdist(b, 'r', p=p)
    ax = plt.gca()
    makeNicePlots(ax)
    plt.xlabel('CA (data-shfl)')
    
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    
    # 
    ##%% Save the figure of each day   
    if savefigs:#% Save the figure
        if chAl==1:
            dd = 'chAl_distCA_allSampsPooled_'+ avmd0+'dms' # it means data-shfl was computed as data-mean(shfl) if ave=1, and as data -med(shfl) if ave=0
        else:
            dd = 'stAl_distCA_allSampsPooled_'+ avmd0+'dms'
            
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)            
    
    #for sublist in l:
    #    for item in sublist:
    #        flat_list.append(item)
    # do for each frame    
    
    
    
    #%% for each day, compute whether each frame (acorss all samples) is diff btwn test and shfl (or test and incorr)
     
    pwmt_testIncorr_eachDay = []
    for iday in range(len(days)):
        #a = classAcc_test_d_aligned[:,:, iday] # classErr_bestC_test_data_all[iday]
        #b = classAcc_test_s_aligned[:,:, iday] #  classErr_bestC_test_shfl_all[iday]
        #pwmt_frs(a,b)
        a = classAcc_test_dms_aligned[:,:, iday] # classErr_bestC_test_data_all[iday]
        b = classAcc_test_incorr_dms_aligned[:,:, iday] # classErr_bestC_test_incorr_all[iday]
        pwmt_testIncorr_eachDay.append(pwmt_frs(a,b))
    
    pwmt_testIncorr_eachDay = np.array(pwmt_testIncorr_eachDay) # numDays x 3 x numFrs # 3 is p val of wilcoxon, mann whitney, and ttest
    
    
    ####### Plot fraction of days with sig p value between test (data-shfl) and incorr (data-shfl)
    plt.figure(figsize=(4,3))
    a = np.mean(pwmt_testIncorr_eachDay[:,0,:] < .05, axis=0) # fraction of days with sig p value between test (data-shfl) and incorr (data-shfl)
    plt.plot(time_aligned, a, label='Wilc')
    a = np.mean(pwmt_testIncorr_eachDay[:,1,:] < .05, axis=0)
    plt.plot(time_aligned, a, label='MW')
    a = np.mean(pwmt_testIncorr_eachDay[:,2,:] < .05, axis=0)
    plt.plot(time_aligned, a, label='ttest')
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False) 
    
    
    #%%
    ###########################################################################
    ############### Do average (or median) across samples ###############
    ###########################################################################
    
    ##%% Average and std of class accuracies across samples (all CV samples) ... for each day
    # NOTE: maybe it makes more sense to get median across samples, bc dists of real data are skewed towards 0 class accur...
    
    def funcentdisp(x, ave, ax=0, sd=0): 
        if ax==0:
            nsamps = x.shape[0]
        else:
            nsamps = x.shape[1]
        if ave==1: # mean, sd
            a = np.nanmean(x, axis=ax)            
            b = np.nanstd(x, axis=ax) # standard deviation
            if sd==0: # standard error
                b = b / np.sqrt(nsamps) # se
            y = np.array([a, a-b, a+b]) # mean, mean-sd, mean+sd        
        else: # median, 25,75 percentile
    #        y = np.nanmedian(x, axis=ax)
            a = np.nanpercentile(x, [25,50,75], axis=ax) # 3xframes
            if ax==1:
                a = a.T
            y = np.array([a[1,:], a[0,:], a[2,:]]) # mean, mean-sd, mean+sd        
        return y
        
       
    ##%% 
    av_l2_train_d = np.array([funcentdisp(100-classErr_bestC_train_data_all[iday], ave)[0] for iday in range(len(days))]) # numDays; each day has size numFrs
    sd_l2_train_d = [funcentdisp(100-classErr_bestC_train_data_all[iday], ave)[[1,2]].T for iday in range(len(days))] # for each day it has size frsx2  # lower error bar and upper error bar (it is mean-sd and mean+sd or 25th and 75th perc)
    
    av_l2_test_d = np.array([funcentdisp(100-classErr_bestC_test_data_all[iday], ave)[0] for iday in range(len(days))]) # numDays
    sd_l2_test_d = [funcentdisp(100-classErr_bestC_test_data_all[iday], ave)[[1,2]].T for iday in range(len(days))]
    
    av_l2_test_s = np.array([funcentdisp(100-classErr_bestC_test_shfl_all[iday], ave)[0] for iday in range(len(days))]) # numDays
    sd_l2_test_s = [funcentdisp(100-classErr_bestC_test_shfl_all[iday], ave)[[1,2]].T for iday in range(len(days))]
    
    av_l2_test_c = np.array([funcentdisp(100-classErr_bestC_test_chance_all[iday], ave)[0] for iday in range(len(days))]) # numDays
    sd_l2_test_c = [funcentdisp(100-classErr_bestC_test_chance_all[iday], ave)[[1,2]].T for iday in range(len(days))]
    
    av_l2_test_dms = np.array([funcentdisp(classAcc_bestC_test_dms_all[iday], ave)[0] for iday in range(len(days))]) # numDays
    sd_l2_test_dms = [funcentdisp(classAcc_bestC_test_dms_all[iday], ave)[[1,2]].T for iday in range(len(days))]
    
    if doIncorr==1:
        av_l2_test_incorr_d = np.array([funcentdisp(100-classErr_bestC_test_incorr_all[iday], ave)[0] for iday in range(len(days))]) # numDays
        sd_l2_test_incorr_d = [funcentdisp(100-classErr_bestC_test_incorr_all[iday], ave)[[1,2]].T for iday in range(len(days))]
        
        av_l2_test_incorr_s = np.array([funcentdisp(100-classErr_bestC_test_incorr_shfl_all[iday], ave)[0] for iday in range(len(days))]) # numDays
        sd_l2_test_incorr_s = [funcentdisp(100-classErr_bestC_test_incorr_shfl_all[iday], ave)[[1,2]].T for iday in range(len(days))]
    
        av_l2_test_incorr_c = np.array([funcentdisp(100-classErr_bestC_test_incorr_chance_all[iday], ave)[0] for iday in range(len(days))]) # numDays
        sd_l2_test_incorr_c = [funcentdisp(100-classErr_bestC_test_incorr_chance_all[iday], ave)[[1,2]].T for iday in range(len(days))]
    
        av_l2_test_incorr_dms = np.array([funcentdisp(classAcc_bestC_test_incorr_dms_all[iday], ave)[0] for iday in range(len(days))]) # numDays
        sd_l2_test_incorr_dms = [funcentdisp(classAcc_bestC_test_incorr_dms_all[iday], ave)[[1,2]].T for iday in range(len(days))]
    
    #_,p = stats.ttest_ind(l1_err_test_data, l1_err_test_shfl, nan_policy = 'omit')
    
        
    #%% Align the above computed traces (CA averaged across CV samples) of each day on the common eventI
    # Also exclude days with too few trials used for training SVM, also exclude incorr from days with too few incorr trials.
    '''
    thTrained=25; # 1/10 of this will be the testing tr num!
    thIncorr=5
    '''
    #print 'num days to be excluded:\n\tlow trained trs:', sum(mn_corr<thTrained), '\n\tlow incorr trs', sum(mn_incorr<thIncorr)
    
    def alCAavSamp(av_l2_train_d, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained):
        av_l2_train_d_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)

        for iday in range(numDays):
            if mn_corr[iday] >= thTrained:
                av_l2_train_d_aligned[:, iday] = av_l2_train_d[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
                
        return av_l2_train_d_aligned

    
    av_l2_train_d_aligned = alCAavSamp(av_l2_train_d, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained)
    av_l2_test_d_aligned = alCAavSamp(av_l2_test_d, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained)
    av_l2_test_s_aligned = alCAavSamp(av_l2_test_s, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained)
    av_l2_test_c_aligned = alCAavSamp(av_l2_test_c, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained)
    av_l2_test_dms_aligned = alCAavSamp(av_l2_test_dms, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained)
    
    sd_l2_train_d_aligned = alCAavSamp(sd_l2_train_d, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained)
    sd_l2_test_d_aligned = alCAavSamp(sd_l2_test_d, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained)
    sd_l2_test_s_aligned = alCAavSamp(sd_l2_test_s, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained)
    sd_l2_test_c_aligned = alCAavSamp(sd_l2_test_c, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained)
    sd_l2_test_dms_aligned = alCAavSamp(sd_l2_test_dms, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained)

    if doIncorr==1:
        av_l2_test_incorr_d_aligned = alCAavSamp(av_l2_test_incorr_d, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained)
        av_l2_test_incorr_s_aligned = alCAavSamp(av_l2_test_incorr_s, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained)
        av_l2_test_incorr_c_aligned = alCAavSamp(av_l2_test_incorr_c, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained)
        av_l2_test_incorr_dms_aligned = alCAavSamp(av_l2_test_incorr_dms, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained)
    
        sd_l2_test_incorr_d_aligned = alCAavSamp(sd_l2_test_incorr_d, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained)
        sd_l2_test_incorr_s_aligned = alCAavSamp(sd_l2_test_incorr_s, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained)
        sd_l2_test_incorr_c_aligned = alCAavSamp(sd_l2_test_incorr_c, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained)
        sd_l2_test_incorr_dms_aligned = alCAavSamp(sd_l2_test_incorr_dms, eventI_ds_allDays, nPreMin, nPostMin, numDays, mn_corr, thTrained)
       
    # delete below once the function above is confirmed to work       
    '''    
    av_l2_train_d_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
    av_l2_test_d_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
    av_l2_test_s_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan
    av_l2_test_c_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan
    av_l2_test_dms_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # data-shfl
    
    sd_l2_train_d_aligned = np.ones((nPreMin + nPostMin + 1, 2, numDays)) + np.nan # frames x 2 x days, aligned on common eventI (equals nPreMin)
    sd_l2_test_d_aligned = np.ones((nPreMin + nPostMin + 1, 2, numDays)) + np.nan 
    sd_l2_test_s_aligned = np.ones((nPreMin + nPostMin + 1, 2, numDays)) + np.nan
    sd_l2_test_c_aligned = np.ones((nPreMin + nPostMin + 1, 2, numDays)) + np.nan
    sd_l2_test_dms_aligned = np.ones((nPreMin + nPostMin + 1, 2, numDays)) + np.nan # data-shfl
    
    if doIncorr==1:
        av_l2_test_incorr_d_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
        av_l2_test_incorr_s_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan
        av_l2_test_incorr_c_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan
        av_l2_test_incorr_dms_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # data-shfl (for each day, samp-avraged data - samp-averaged shfl)
    
        sd_l2_test_incorr_d_aligned = np.ones((nPreMin + nPostMin + 1, 2, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
        sd_l2_test_incorr_s_aligned = np.ones((nPreMin + nPostMin + 1, 2, numDays)) + np.nan
        sd_l2_test_incorr_c_aligned = np.ones((nPreMin + nPostMin + 1, 2, numDays)) + np.nan
        sd_l2_test_incorr_dms_aligned = np.ones((nPreMin + nPostMin + 1, 2, numDays)) + np.nan # data-shfl (for each day, samp-avraged data - samp-averaged shfl)
    
    for iday in range(numDays):
        if mn_corr[iday] >= thTrained:
            av_l2_train_d_aligned[:, iday] = av_l2_train_d[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            av_l2_test_d_aligned[:, iday] = av_l2_test_d[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            av_l2_test_s_aligned[:, iday] = av_l2_test_s[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            av_l2_test_c_aligned[:, iday] = av_l2_test_c[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]        
            av_l2_test_dms_aligned[:, iday] = av_l2_test_dms[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1] #av_l2_test_dms_aligned[:, iday] = av_l2_test_d_aligned[:, iday] - av_l2_test_s_aligned[:, iday]
            
            sd_l2_train_d_aligned[:,:, iday] = sd_l2_train_d[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            sd_l2_test_d_aligned[:,:, iday] = sd_l2_test_d[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            sd_l2_test_s_aligned[:,:, iday] = sd_l2_test_s[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            sd_l2_test_c_aligned[:,:, iday] = sd_l2_test_c[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            sd_l2_test_dms_aligned[:,:, iday] = sd_l2_test_dms[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            
        if doIncorr==1 and mn_incorr[iday] >= thIncorr:    
            av_l2_test_incorr_d_aligned[:, iday] = av_l2_test_incorr_d[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            av_l2_test_incorr_s_aligned[:, iday] = av_l2_test_incorr_s[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            av_l2_test_incorr_c_aligned[:, iday] = av_l2_test_incorr_c[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]    
            av_l2_test_incorr_dms_aligned[:, iday] = av_l2_test_incorr_dms[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1] #av_l2_test_incorr_dms_aligned[:, iday] = av_l2_test_incorr_d_aligned[:, iday] - av_l2_test_incorr_s_aligned[:, iday]
    
            sd_l2_test_incorr_d_aligned[:,:, iday] = sd_l2_test_incorr_d[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            sd_l2_test_incorr_s_aligned[:,:, iday] = sd_l2_test_incorr_s[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            sd_l2_test_incorr_c_aligned[:,:, iday] = sd_l2_test_incorr_c[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]    
            sd_l2_test_incorr_dms_aligned[:,:, iday] = sd_l2_test_incorr_dms[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1] #av_l2_test_incorr_dms_aligned[:, iday] = av_l2_test_incorr_d_aligned[:, iday] - av_l2_test_incorr_s_aligned[:, iday]
            
    '''
    
    ####### compute p values for each frame across days whether samp-averaged traces of testing data are diff from shfl, also whether testing is diff from incorr: 
    
    # use t test
    # test vs shfl
    _,pcorrtrace = stats.ttest_ind(av_l2_test_d_aligned.transpose(), av_l2_test_s_aligned.transpose(), nan_policy='omit') 
    # incorr vs test
    if doIncorr==1:
        _,pincorrtrace = stats.ttest_ind(av_l2_test_d_aligned.transpose(), av_l2_test_incorr_d_aligned.transpose(), nan_policy='omit')         
        _,pincorrtracedms = stats.ttest_ind(av_l2_test_dms_aligned.transpose(), av_l2_test_incorr_dms_aligned.transpose(), nan_policy='omit')             
    #_,pcorrtrace0 = stats.ttest_1samp(av_l2_test_d_aligned.transpose(), 50) # p value of class accuracy being different from 50
        
    
    # use Wilcoxon test:
    pcorrtracew = np.full((len(time_aligned)), np.nan)
    pincorrtracedmsw = np.full((len(time_aligned)), np.nan)
    for f in range(len(time_aligned)):
        _,p  = sci.stats.ranksums(av_l2_test_d_aligned[f,:], av_l2_test_s_aligned[f,:])
        _,pp  = sci.stats.ranksums(av_l2_test_dms_aligned[f,:], av_l2_test_incorr_dms_aligned[f,:])
        pcorrtracew[f] = p
        pincorrtracedmsw[f] = pp
    
    # use Mann-Whitney test:
    pcorrtracem = np.full((len(time_aligned)), np.nan)
    pincorrtracedmsm = np.full((len(time_aligned)), np.nan)
    for f in range(len(time_aligned)):
        _,p  = sci.stats.mannwhitneyu(av_l2_test_d_aligned[f,:], av_l2_test_s_aligned[f,:], alternative='two-sided')
        _,pp  = sci.stats.mannwhitneyu(av_l2_test_dms_aligned[f,:], av_l2_test_incorr_dms_aligned[f,:], alternative='two-sided')
        pcorrtracem[f] = p    
        pincorrtracedmsm[f] = pp
    
    
    
        
    #%%
    ######################### Take ave (or med) across days #########################
    ##%% Average and SD of samp-averaged CA across days (each day includes the average class accuracy across samples.)
    # or median and iqr depending on value of ave
        
    av_l2_train_d_aligned_ave = funcentdisp(av_l2_train_d_aligned, ave, ax=1, sd=1)[0] #a
    av_l2_train_d_aligned_std = funcentdisp(av_l2_train_d_aligned, ave, ax=1, sd=1)[[1,2]].T #b  # sd = b.T-a
        
    av_l2_test_d_aligned_ave = funcentdisp(av_l2_test_d_aligned, ave, ax=1, sd=1)[0] #a
    av_l2_test_d_aligned_std = funcentdisp(av_l2_test_d_aligned, ave, ax=1, sd=1)[[1,2]].T #b  # sd = b.T-a
    
    av_l2_test_s_aligned_ave = funcentdisp(av_l2_test_s_aligned, ave, ax=1, sd=1)[0] #a
    av_l2_test_s_aligned_std = funcentdisp(av_l2_test_s_aligned, ave, ax=1, sd=1)[[1,2]].T #b  # sd = b.T-a
    
    av_l2_test_c_aligned_ave = funcentdisp(av_l2_test_c_aligned, ave, ax=1, sd=1)[0] #a
    av_l2_test_c_aligned_std = funcentdisp(av_l2_test_c_aligned, ave, ax=1, sd=1)[[1,2]].T #b  # sd = b.T-a
    
    av_l2_test_dms_aligned_ave = funcentdisp(av_l2_test_dms_aligned, ave, ax=1, sd=1)[0] #a
    av_l2_test_dms_aligned_std = funcentdisp(av_l2_test_dms_aligned, ave, ax=1, sd=1)[[1,2]].T #b  # sd = b.T-a
       
    if doIncorr==1:
        av_l2_test_incorr_d_aligned_ave = funcentdisp(av_l2_test_incorr_d_aligned, ave, ax=1, sd=1)[0] #a
        av_l2_test_incorr_d_aligned_std = funcentdisp(av_l2_test_incorr_d_aligned, ave, ax=1, sd=1)[[1,2]].T #b  # sd = b.T-a
    
        av_l2_test_incorr_s_aligned_ave = funcentdisp(av_l2_test_incorr_s_aligned, ave, ax=1, sd=1)[0] #a
        av_l2_test_incorr_s_aligned_std = funcentdisp(av_l2_test_incorr_s_aligned, ave, ax=1, sd=1)[[1,2]].T #b  # sd = b.T-a
    
        av_l2_test_incorr_c_aligned_ave = funcentdisp(av_l2_test_incorr_c_aligned, ave, ax=1, sd=1)[0] #a
        av_l2_test_incorr_c_aligned_std = funcentdisp(av_l2_test_incorr_c_aligned, ave, ax=1, sd=1)[[1,2]].T #b  # sd = b.T-a
    
        av_l2_test_incorr_dms_aligned_ave = funcentdisp(av_l2_test_incorr_dms_aligned, ave, ax=1, sd=1)[0] #a
        av_l2_test_incorr_dms_aligned_std = funcentdisp(av_l2_test_incorr_dms_aligned, ave, ax=1, sd=1)[[1,2]].T #b  # sd = b.T-a
    
    
    
        
    #%% Compare sd of samples for each day (for data and shfl) (well we are comparing upper bound - lower bound: ie 2*sd if ave=1, and 75th percentile - 25th if ave=0)
    
    # average of dispersion across days
    def pltDisp(x,ti):
        a = np.nanmean(np.diff(x, axis=1), axis=2).flatten()
        b = np.nanstd(np.diff(x, axis=1), axis=2).flatten() / np.sqrt(numDays)
        plt.fill_between(time_aligned, a-b, a+b, alpha=0.5, edgecolor='k', facecolor='k')
        plt.plot(time_aligned, a, color='k', linewidth=2)
        yl = plt.gca().get_ylim()
        plt.vlines(0, yl[0], yl[1])
        makeNicePlots(plt.gca(),1,1)
        plt.title(ti)
        #for iday in range(len(days)):
        #    plt.plot(np.diff(sd_l2_test_d_aligned[:,:,iday]), color=[.7,.7,.7])
    
    
    plt.figure(figsize=(6,7))    
    plt.subplot(321)
    pltDisp(sd_l2_test_dms_aligned, 'test_data-shfl') # same as data (bc data-shfl is in fact data-mean(shfl))
    plt.ylabel('CA dispersion across samples\n(mean+/-se across days)')
    plt.xlabel('Time since choice onset')
    plt.subplot(323)
    pltDisp(sd_l2_test_d_aligned, 'test_data')
    plt.subplot(325)
    pltDisp(sd_l2_test_s_aligned, 'test_shfl')
    
    plt.subplot(322)
    pltDisp(sd_l2_test_incorr_dms_aligned, 'incorr_data-shfl')
    plt.subplot(324)
    pltDisp(sd_l2_test_incorr_d_aligned, 'incorr_data')
    plt.subplot(326)
    pltDisp(sd_l2_test_incorr_s_aligned, 'incorr_shfl')
    
    plt.subplots_adjust(wspace=0.3, hspace=0.6)
    
    
    ##%% Save the figure of each day   
    if savefigs:#% Save the figure
        if chAl==1:
            dd = 'chAl_dispersionCA_'+ avmd1 + '_' + nowStr
        else:
            dd = 'stAl_dispersionCA_'+ avmd1 + '_' + nowStr
            
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)            
        
        
    # plot each day
    '''
    for iday in range(len(days)):
            
        plt.figure(figsize=(3,2))
        if mn_corr[iday] >= thTrained:
            plt.plot(np.diff(sd_l2_test_d[iday]), label='test_data')
            plt.plot(np.diff(sd_l2_test_s[iday]), label='test_shfl')
        if corrTrained==1 and mn_incorr[iday] >= thIncorr:       
            plt.plot(np.diff(sd_l2_test_incorr_d[iday]), label='incorr_data')
            plt.plot(np.diff(sd_l2_test_incorr_s[iday]), label='incorr_shfl')
    
        yl = plt.gca().get_ylim()
        plt.vlines(eventI_ds_allDays[iday], yl[0], yl[1])
     
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)    
        plt.title(days[iday][0:6])
    '''    
        
    
    
    #%% Plot dists of samp-avraged CA pooled across days and frames 
    
    # samples are not averaged, pool all samples and all frames of all days
    plt.figure()
    plt.subplot(221)
    a = av_l2_test_d_aligned
    b = av_l2_test_incorr_d_aligned
    p = pwm(a,b)
    plotflastdist(a, 'b', lab='corr')  # samps x alFrs x days     #l = 100-classErr_bestC_test_data_all
    plotflastdist(b, 'r', lab='incorr', p=p)
    ax = plt.gca()
    makeNicePlots(ax)
    plt.xlabel('CA (data)')
    plt.ylabel('# (pooled samples,frames,days)')
        
    plt.subplot(222)
    a = av_l2_test_s_aligned
    b = av_l2_test_incorr_s_aligned
    p = pwm(a,b)
    plotflastdist(a, 'b')  # samps x alFrs x days     #l = 100-classErr_bestC_test_data_all
    plotflastdist(b, 'r', p=p)
    ax = plt.gca()
    makeNicePlots(ax)
    plt.xlabel('CA (shfl)')
    
    plt.subplot(223)
    a = av_l2_test_dms_aligned
    b = av_l2_test_incorr_dms_aligned
    p = pwm(a,b)
    plotflastdist(a, 'b')  # samps x alFrs x days     #l = 100-classErr_bestC_test_data_all
    plotflastdist(b, 'r', p=p)
    ax = plt.gca()
    makeNicePlots(ax)
    plt.xlabel('CA (data-shfl)')
    
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    
    
    ##%% Save the figure of each day   
    if savefigs:#% Save the figure
        if chAl==1:
            dd = 'chAl_distCA_%sSamps' %(avmd0) + '_' + nowStr
        else:
            dd = 'stAl_distCA_%sSamps' %(avmd0) + '_' + nowStr
            
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)            
    
    
    '''
    for f in range(av_l2_test_d_aligned.shape[0]):
        plt.figure(figsize=(9,6))
        plt.subplot(221)
        plt.hist(av_l2_test_d_aligned[f,:][~np.isnan(av_l2_test_d_aligned[f,:])])
        plt.title(days[iday])
        plt.subplot(222)
        plt.hist(av_l2_test_s_aligned[f,:][~np.isnan(av_l2_test_s_aligned[f,:])])
        plt.subplot(223)
        plt.hist(av_l2_test_incorr_d_aligned[f,:][~np.isnan(av_l2_test_incorr_d_aligned[f,:])])
        plt.subplot(224)
        plt.hist(av_l2_test_incorr_s_aligned[f,:][~np.isnan(av_l2_test_incorr_s_aligned[f,:])])
    '''    
    '''
    # samples are already averaged for each day # pool all frames and all days
    plt.figure(figsize=(9,6))
    plt.subplot(221)
    plt.hist(av_l2_test_d_aligned.flatten()[~np.isnan(av_l2_test_d_aligned.flatten())], label='test')
    plt.plot(np.nanmean(av_l2_test_d_aligned.flatten()), 50, marker='*', color='r', markersize=7) # average of class accuracies
    plt.plot(np.nanmedian(av_l2_test_d_aligned.flatten()), 50, marker='*', color='k', markersize=7) # median       
    plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
    
    plt.subplot(222)
    plt.hist(av_l2_test_s_aligned.flatten()[~np.isnan(av_l2_test_s_aligned.flatten())], label='test_shfl')
    plt.plot(np.nanmean(av_l2_test_s_aligned.flatten()), 50, marker='*', color='r', markersize=7) # average of class accuracies
    plt.plot(np.nanmedian(av_l2_test_s_aligned.flatten()), 50, marker='*', color='k', markersize=7) # median       
    plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
    
    plt.subplot(223)
    plt.hist(av_l2_test_incorr_d_aligned.flatten()[~np.isnan(av_l2_test_incorr_d_aligned.flatten())], label='incorr')
    plt.plot(np.nanmean(av_l2_test_incorr_d_aligned.flatten()), 50, marker='*', color='r', markersize=7) # average of class accuracies
    plt.plot(np.nanmedian(av_l2_test_incorr_d_aligned.flatten()), 50, marker='*', color='k', markersize=7) # median   
    plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
    
    plt.subplot(224)
    plt.hist(av_l2_test_incorr_s_aligned.flatten()[~np.isnan(av_l2_test_incorr_s_aligned.flatten())], label='incorr_shfl')
    plt.plot(np.nanmean(av_l2_test_incorr_s_aligned.flatten()), 50, marker='*', color='r', markersize=7) # average of class accuracies
    plt.plot(np.nanmedian(av_l2_test_incorr_s_aligned.flatten()), 50, marker='*', color='k', markersize=7) # median   
    plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
    '''
    
    
    #%% Plot sample-averaged class accur trace for each day: testing, training, incorr data, also data-shfl
    
    daysSup = 0 # whether to superimpose all days, or make a separate figure for each day
    # if 1, plot separate subplots for testing data, shuffle, chance and training data
    
    #####################################
    def plotClassAll(av1, sd1, av2, sd2, av3, sd3, av4, sd4, av5, sd5, av6, sd6, tits):
        
    #    print np.shape((av3, sd3, av1, sd1, av2, sd2, av4, sd4))
        if daysSup:
            plt.figure()
            
        for iday in range(len(days)):
    #        print iday
            if daysSup==0: # a figure for each day
                plt.figure(figsize=(4,6))
            totLen = len(av1[iday])
            if corrTrained==0: # remember below has problems...but if downsampling in svm_eachFrame was done in the old way, you have to use below.
                nPre = eventI_ds_allDays[iday] # number of frames before the common eventI, also the index of common eventI. 
                nPost = (len(av1[iday]) - eventI_ds_allDays[iday] - 1)    
                a = -(np.asarray(frameLength*regressBins) * range(nPre+1)[::-1])
                b = (np.asarray(frameLength*regressBins) * range(1, nPost+1))
                time_al = np.concatenate((a,b))
            else:
                time_al = set_time_al(totLen, eventI_allDays[iday], lastTimeBinMissed)
                    
    #        print time_al.shape
    #        print av1[iday].shape
    #        print sd1[iday].shape
            if daysSup:                
                plt.subplot(221)
            else:
                plt.subplot(311)
            plt.errorbar(time_al, av1[iday], yerr = abs(sd1[iday].T - av1[iday]), label=tits[0])    
        #    plt.plot([0,0], [50,100], color='k', linestyle=':')
            #plt.errorbar(range(len(av1[iday])), av1[iday], yerr = sd1[iday])
        #    plt.plot(time_al, av1[iday]-av2[iday], color=colors[iday])
            
            if daysSup:
                plt.subplot(222)
            plt.errorbar(time_al, av2[iday], yerr = abs(sd2[iday].T - av2[iday]), label=tits[1])
    
            if daysSup:    
                plt.subplot(223)
            plt.errorbar(time_al, av3[iday], yerr = abs(sd3[iday].T - av3[iday]), label=tits[2])
    
            if daysSup:    
                plt.subplot(224)
            plt.errorbar(time_al, av4[iday], yerr = abs(sd4[iday].T - av4[iday]), label=tits[3])    
        #    plt.show() # Uncomment this if you a separate plot for each day.
            
            if daysSup==0:
                t = '%s_%s' %(days[iday][0:6], str(tin[iday,:].astype(int)))
                plt.title(t) # Uncomment this if you a separate plot for each day.
                plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False) 
    
                if chAl==1:
                    plt.xlabel('Time since choice onset (ms)', fontsize=13)
                else:
                    plt.xlabel('Time since stim onset (ms)', fontsize=13)
                plt.ylabel('Class accuracy(%)', fontsize=13)
                ax = plt.gca()
                xl = ax.get_xlim()
                plt.hlines(50, xl[0], xl[1], linestyle=':')
                yl = ax.get_ylim()        
                plt.vlines(0, yl[0],yl[1], linestyle=':')
                makeNicePlots(ax,0,1)  
    #            plt.show()
                
                # find high cbest points
                thcbest = 1            
                a = np.full(cbestFrs_all[iday].shape, np.nan)
                
                
                # data-shfl for testing and incorr
                plt.subplot(312) 
                # figure out what would be the variance of two dists subtracted from each other, so you can add error bars
    #            plt.plot(time_al, av1[iday]-av2[iday], label=tits[0]+'-'+tits[1], color='b')   
                plt.errorbar(time_al, av5[iday], yerr = abs(sd5[iday].T - av5[iday]), label=tits[0]+'-'+tits[1], color='b')    
                if doIncorr==1:
    #                plt.plot(time_al, av3[iday]-av4[iday], label=tits[2]+'-'+tits[3], color='r')   # av_l2_test_incorr_d[iday]-av_l2_test_incorr_s[iday]
                    plt.errorbar(time_al, av6[iday], yerr = abs(sd6[iday].T - av6[iday]), label=tits[2]+'-'+tits[3], color='r')
                plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False) 
                ax = plt.gca()
                xl = ax.get_xlim()
                plt.hlines(0, xl[0], xl[1], linestyle=':')
                yl = ax.get_ylim()        
                plt.vlines(0, yl[0],yl[1], linestyle=':')
                # mark high cbest points
                a[cbestFrs_all[iday]>thcbest] = yl[1]-(yl[1]-yl[0])/10
                plt.plot(time_al, a, marker='*', color='r')            
                makeNicePlots(ax,0,1)  
                plt.ylabel('data-shfl(%)', fontsize=13)
                makeNicePlots(ax,0,1)  
                if chAl==1:
                    plt.xlabel('Time since choice onset (ms)', fontsize=13)
                else:
                    plt.xlabel('Time since stim onset (ms)', fontsize=13)            
                    
                
                # cbest for each frame
                plt.subplot(313) 
                plt.plot(time_al, cbestFrs_all[iday]); 
                plt.ylabel('best C'); 
                ax = plt.gca()
                xl = ax.get_xlim()
                plt.hlines(0, xl[0], xl[1], linestyle=':')
                yl = ax.get_ylim()        
                plt.vlines(0, yl[0],yl[1], linestyle=':') 
                # mark high cbest points
                a[cbestFrs_all[iday]>thcbest] = yl[1]-(yl[1]-yl[0])/10            
                plt.plot(time_al, a, marker='*', color='r')
                makeNicePlots(ax,0,1)  
                if chAl==1:
                    plt.xlabel('Time since choice onset (ms)', fontsize=13)
                else:
                    plt.xlabel('Time since stim onset (ms)', fontsize=13)            
    
                plt.subplots_adjust(hspace=0.65)
                
                ##%% Save the figure of each day   
                if savefigs:#% Save the figure
                    if chAl==1:
                        dd = 'chAl_testTrain_'+avmd0 + '_' + days[iday][0:6]
                    else:
                        dd = 'stAl_testTrain_'+avmd0 + '_' + days[iday][0:6]
                        
                    d = os.path.join(svmdir+dnow+'eachDay')
                    if not os.path.exists(d):
                        print 'creating folder'
                        os.makedirs(d)
                            
                    fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])
                    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)            
        
        
        ##%%
        if daysSup:    
            plt.subplot(221)
            plt.title(tits[1])
            if chAl==1:
                plt.xlabel('Time since choice onset (ms)', fontsize=13)
            else:
                plt.xlabel('Time since stim onset (ms)', fontsize=13)
            plt.ylabel('Classification accuracy (%)', fontsize=13)
            ax = plt.gca()
            xl = ax.get_xlim()
            plt.hlines(50, xl[0], xl[1])
            yl = ax.get_ylim()        
            plt.vlines(0, yl[0],yl[1], linestyle=':')        
            makeNicePlots(ax,1)
            
            plt.subplot(222)
            plt.title(tits[2])
            ax = plt.gca()
            xl = ax.get_xlim()
            plt.hlines(50, xl[0], xl[1])
            yl = ax.get_ylim()        
            plt.vlines(0, yl[0],yl[1], linestyle=':')        
            makeNicePlots(ax,1)
            
            plt.subplot(223)        
            plt.title(tits[0])
            ax = plt.gca()
            xl = ax.get_xlim()
            plt.hlines(50, xl[0], xl[1])
            yl = ax.get_ylim()        
            plt.vlines(0, yl[0],yl[1], linestyle=':')        
            makeNicePlots(ax,1)
            
            plt.subplot(224)
            plt.title(tits[3])
            ax = plt.gca()
            xl = ax.get_xlim()
            plt.hlines(50, xl[0], xl[1])
            yl = ax.get_ylim()        
            plt.vlines(0, yl[0],yl[1], linestyle=':')        
            makeNicePlots(ax,1)
            
            plt.subplots_adjust(hspace=0.65)
        
        
            ##%% Save the figure    
            if savefigs:#% Save the figure
                if chAl==1:
                    dd = 'chAl_testTrain_'+avmd0 + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
                else:
                    dd = 'stAl_testTrain_'+avmd0 + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
                    
                d = os.path.join(svmdir+dnow)
                if not os.path.exists(d):
                    print 'creating folder'
                    os.makedirs(d)
                        
                fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])
                plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        
        
    '''
    # plot test minus test_shuffle
    plt.figure()
    for iday in range(len(days)):    
    #    plt.figure()
        nPre = eventI_ds_allDays[iday] # number of frames before the common eventI, also the index of common eventI. 
        nPost = (len(av_l2_test_d[iday]) - eventI_ds_allDays[iday] - 1)    
        a = -(np.asarray(frameLength*regressBins) * range(nPre+1)[::-1])
        b = (np.asarray(frameLength*regressBins) * range(1, nPost+1))
        time_al = np.concatenate((a,b))
        
    #    plt.subplot(221)
    #    plt.errorbar(time_al, av_l2_test_d[iday], yerr = sd_l2_test_d[iday], label=' ')    
        plt.title(days[iday])
    #    plt.plot([0,0], [50,100], color='k', linestyle=':')
        #plt.errorbar(range(len(av_l2_test_d[iday])), av_l2_test_d[iday], yerr = sd_l2_test_d[iday])
        plt.plot(time_al, av_l2_test_d[iday]-av_l2_test_s[iday], color=colors[iday])
    #    plt.plot(time_al, av_l2_test_d[iday]-av_l2_test_s[iday], color=colors[iday])
    '''    
    
    
    #%% Plot class accur trace for all days. Separate subplots for testing data, shuffle, chance and training data
    
    #tits = 'Train', 'Test', 'Test_shfl', 'Test_chance' # labels of traces in order given to plotClassAll
    #plotClassAll(av_l2_train_d, sd_l2_train_d, av_l2_test_d, sd_l2_test_d, av_l2_test_s, sd_l2_test_s, av_l2_test_c, sd_l2_test_c, tits)
    if doIncorr==1:
    #    tits = 'Test', 'Incorr', 'Incorr_shfl', 'Incorr_chance'
    #    plotClassAll(av_l2_test_d, sd_l2_test_d, av_l2_test_incorr_d, sd_l2_test_incorr_d, av_l2_test_incorr_s, sd_l2_test_incorr_s, av_l2_test_incorr_c, sd_l2_test_incorr_c, tits)
        tin = np.concatenate((corr_hr_lr,incorr_hr_lr),axis=1)
        tits = 'Test', 'Test_shfl', 'Incorr', 'Incorr_shfl'
        plotClassAll(av_l2_test_d, sd_l2_test_d, av_l2_test_s, sd_l2_test_s, av_l2_test_incorr_d, sd_l2_test_incorr_d, av_l2_test_incorr_s, sd_l2_test_incorr_s, av_l2_test_dms, sd_l2_test_dms, av_l2_test_incorr_dms, sd_l2_test_incorr_dms, tits)
        
    
    #%%
    #cbestIFrs_all = np.full(cbestFrs_all.shape, np.nan)
    #for iday in range(numDays):
    #    cbestIFrs_all[iday] = np.array([np.argwhere(np.in1d(cvect, cbestFrs_all[iday][i])) for i in range(len(cbestFrs_all[iday]))]).flatten()
    
    
    #%%
    ################## Plots of ave days ################## 
    
    ##%% Plot the average of aligned traces (data, shfl, data-shfl) across all days
    
    plt.figure(figsize=(4.5,5))
    
    ######## data & shfl ########
    plt.subplot(211)
    # data
    #plt.fill_between(time_aligned, av_l2_test_d_aligned_ave - av_l2_test_d_aligned_std, av_l2_test_d_aligned_ave + av_l2_test_d_aligned_std, alpha=0.5, edgecolor='b', facecolor='b')
    plt.fill_between(time_aligned, av_l2_test_d_aligned_std[:,0], av_l2_test_d_aligned_std[:,1], alpha=0.5, edgecolor='b', facecolor='b')
    plt.plot(time_aligned, av_l2_test_d_aligned_ave, 'b', label='testing data')
    
    # incorr
    if doIncorr==1:    
    #    plt.fill_between(time_aligned, av_l2_test_incorr_d_aligned_ave - av_l2_test_incorr_d_aligned_std, av_l2_test_incorr_d_aligned_ave + av_l2_test_incorr_d_aligned_std, alpha=0.5, edgecolor='r', facecolor='r')
        plt.fill_between(time_aligned, av_l2_test_incorr_d_aligned_std[:,0], av_l2_test_incorr_d_aligned_std[:,1], alpha=0.5, edgecolor='r', facecolor='r')
        plt.plot(time_aligned, av_l2_test_incorr_d_aligned_ave, 'r', label='incorr')
    
    # test_chance
    #plt.fill_between(time_aligned, av_l2_test_c_aligned_ave - av_l2_test_c_aligned_std, av_l2_test_c_aligned_ave + av_l2_test_c_aligned_std, alpha=0.5, edgecolor='k', facecolor='k')
    plt.fill_between(time_aligned, av_l2_test_c_aligned_std[:,0], av_l2_test_c_aligned_std[:,1], alpha=0.5, edgecolor='k', facecolor='k')
    plt.plot(time_aligned, av_l2_test_c_aligned_ave, 'k', label='chance')
    
    # test_shfl
    #plt.fill_between(time_aligned, av_l2_test_s_aligned_ave - av_l2_test_s_aligned_std, av_l2_test_s_aligned_ave + av_l2_test_s_aligned_std, alpha=0.5, edgecolor='g', facecolor='g')
    plt.fill_between(time_aligned, av_l2_test_s_aligned_std[:,0], av_l2_test_s_aligned_std[:,1], alpha=0.5, edgecolor='g', facecolor='g')
    plt.plot(time_aligned, av_l2_test_s_aligned_ave, 'g', label='shfl')
    
    # incorr
    if doIncorr==1:    
        # test_incorr_chance
    #    plt.fill_between(time_aligned, av_l2_test_incorr_c_aligned_ave - av_l2_test_incorr_c_aligned_std, av_l2_test_incorr_c_aligned_ave + av_l2_test_incorr_c_aligned_std, alpha=0.5, edgecolor='k', facecolor='k')
        plt.fill_between(time_aligned, av_l2_test_incorr_c_aligned_std[:,0], av_l2_test_incorr_c_aligned_std[:,1], alpha=0.5, edgecolor='k', facecolor='k')
        plt.plot(time_aligned, av_l2_test_incorr_c_aligned_ave, 'k', label='chance')
        
        # test_incorr_shfl
    #    plt.fill_between(time_aligned, av_l2_test_incorr_s_aligned_ave - av_l2_test_incorr_s_aligned_std, av_l2_test_incorr_s_aligned_ave + av_l2_test_incorr_s_aligned_std, alpha=0.5, edgecolor='g', facecolor='g')
        plt.fill_between(time_aligned, av_l2_test_incorr_s_aligned_std[:,0], av_l2_test_incorr_s_aligned_std[:,1], alpha=0.5, edgecolor='g', facecolor='g')
        plt.plot(time_aligned, av_l2_test_incorr_s_aligned_ave, 'g', label='shfl')
    
    plt.ylabel('Class accuracy(%)', fontsize=13)
    plt.title('SVM trained on non-overlapping %.2f ms windows' %(regressBins*frameLength), fontsize=13)
    ax = plt.gca()
    # mark significant time points
    ymin, ymax = ax.get_ylim()
    pp = pcorrtrace+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
    plt.plot(time_aligned, pp, color='k')
    if doIncorr==1:
        pp = pincorrtrace+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax-3
        plt.plot(time_aligned, pp, color='g')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #if trialHistAnalysis:
    #    plt.xticks(np.arange(-400,600,200))
    #else:    
    #    plt.xticks(np.arange(0,1400,400))
    makeNicePlots(ax,0,1)
    
    
    ######################## data -shfl
    plt.subplot(212)
    # data
    #plt.fill_between(time_aligned, av_l2_test_dms_aligned_ave - av_l2_test_dms_aligned_std, av_l2_test_dms_aligned_ave + av_l2_test_dms_aligned_std, alpha=0.5, edgecolor='b', facecolor='b')
    plt.fill_between(time_aligned,av_l2_test_dms_aligned_std[:,0], av_l2_test_dms_aligned_std[:,1], alpha=0.5, edgecolor='b', facecolor='b')
    plt.plot(time_aligned, av_l2_test_dms_aligned_ave, 'b', label='testing data-shfl')
    
    # incorr
    if doIncorr==1:    
    #    plt.fill_between(time_aligned, av_l2_test_incorr_dms_aligned_ave - av_l2_test_incorr_dms_aligned_std, av_l2_test_incorr_dms_aligned_ave + av_l2_test_incorr_dms_aligned_std, alpha=0.5, edgecolor='r', facecolor='r')
        plt.fill_between(time_aligned, av_l2_test_incorr_dms_aligned_std[:,0], av_l2_test_incorr_dms_aligned_std[:,1], alpha=0.5, edgecolor='r', facecolor='r')
        plt.plot(time_aligned, av_l2_test_incorr_dms_aligned_ave, 'r', label='incorr data-shfl')
    
    if chAl==1:
        plt.xlabel('Time since choice onset (ms)', fontsize=13)
    else:
        plt.xlabel('Time since stim onset (ms)', fontsize=13)
    plt.ylabel('data-shfl', fontsize=13)
    ax = plt.gca()
    # mark significant time points
    ymin, ymax = ax.get_ylim()
    if doIncorr==1:
        pp = pincorrtracedms+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax-3
        plt.plot(time_aligned, pp, color='g')
    makeNicePlots(ax,0,1)
    
    
    
    ##%% Save the figure    
    if savefigs:#% Save the figure
        if chAl==1:
            dd = 'chAl_'+avmd0+ 'Days_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_'+avmd0+ 'Days_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
    
    
    
    
    
    
no
#%%
############### Compare CA of each time bin across days ###############

#%% For each time bin, plot CA across day number. Show mean and dispersion across samples for each day.

# each frame, all days, mean and dispersion across samples
# for each day set perc for each frame across all samps
def plotCAeachtimeAlldays(CA_corr, CA_incorr, doMed):
    perc = np.full((3, CA_corr.shape[1], CA_corr.shape[2]), np.nan)
    percincorr = np.full((3, CA_corr.shape[1], CA_corr.shape[2]), np.nan)
    for iday in range(len(days)):
        if doMed:
            # do median, 25th and 75th percentiles
            perc[:,:,iday] = np.nanpercentile(CA_corr[:,:,iday], [25,50,75], axis=0) # compute percentiles across samples for each frame
            percincorr[:,:,iday] = np.nanpercentile(CA_incorr[:,:,iday], [25,50,75], axis=0)
        else:
            # do mean +/- sd        
            perc[1,:,iday] = np.nanmean(CA_corr[:,:,iday], axis=0) # mean across samples for each frame
            perc[0,:,iday] = perc[1,:,iday] - np.nanstd(CA_corr[:,:,iday], axis=0) # mean-std across samples for each frame
            perc[2,:,iday] = perc[1,:,iday] + np.nanstd(CA_corr[:,:,iday], axis=0) # mean+std across samples for each frame
            # incorr        
            percincorr[1,:,iday] = np.nanmean(CA_incorr[:,:,iday], axis=0) # mean across samples for each frame
            percincorr[0,:,iday] = percincorr[1,:,iday] - np.nanstd(CA_incorr[:,:,iday], axis=0) # mean-std across samples for each frame
            percincorr[2,:,iday] = percincorr[1,:,iday] + np.nanstd(CA_incorr[:,:,iday], axis=0) # mean+std across samples for each frame
        
    for f in range(len(time_aligned)): # loop over all times bins
        plt.figure(figsize=(4,3))
        plt.fill_between(range(len(days)), perc[0,f,:], perc[2,f,:], alpha=0.5, edgecolor='b', facecolor='b')
        plt.plot(range(len(days)), perc[1,f,:], 'b', label='testing data')
        
        plt.fill_between(range(len(days)), percincorr[0,f,:], percincorr[2,f,:], alpha=0.5, edgecolor='r', facecolor='r')
        plt.plot(range(len(days)), percincorr[1,f,:], 'r', label='testing data')
        plt.title(np.round(time_aligned[f]))


#%%

doMed = 1 # whether to plot median +/- percentiles, or to plot mean +/- sd
plotCAeachtimeAlldays(classAcc_test_d_aligned, classAcc_test_incorr_d_aligned, doMed)
plotCAeachtimeAlldays(classAcc_test_s_aligned, classAcc_test_incorr_s_aligned)
plotCAeachtimeAlldays(classAcc_test_dms_aligned, classAcc_test_incorr_dms_aligned)




#%% Compare data-shfl class accur in each timebin across days

time2an = -1; # 1 bin before eventI

for t2an in range(av_l2_test_s_aligned.shape[0]): 
    if chAl==1: # chAl: frame before choice
        a = av_l2_test_d_aligned[t2an,:] - av_l2_test_s_aligned[t2an,:] # same as av_l2_test_dms_aligned[t2an,:]
        yl = 'class accuracy (testing data - shfl) %dms before choice' %(round(regressBins*frameLength))
        if doIncorr==1:
            ain = av_l2_test_incorr_d_aligned[t2an,:] - av_l2_test_incorr_s_aligned[t2an,:]
    else: # stAl: last frame # needs work
        a = av_l2_test_d_aligned[-1,:] - av_l2_test_s_aligned[-1,:]
        yl = 'class accuracy (testing data - shfl) (last %d bin)' %(round(regressBins*frameLength))
        
    plt.figure(figsize=(3,1.5))
    
    plt.plot(range(len(days)), a, marker='o', color='b', markersize=5)
    if doIncorr==1:
        plt.plot(range(len(days)), ain, marker='o', color='r', markersize=5)
#    plt.xticks(range(len(days)), [days[i][0:6] for i in range(len(days))], rotation='vertical')
#    plt.ylabel(yl)    
    plt.title(int(np.round(time_aligned[t2an],0)))   
    makeNicePlots(plt.gca())


##%% Save the figure    
if savefigs:#% Save the figure
    if chAl==1:
        dd = 'chAl_time' + str(time2an) + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
    else:
        dd = 'stAl_time' + str(time2an) + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)







#%% Change color order to jet 

colorOrder(nlines=len(days))


#%% Plot all days testing real data - shuffle

plt.figure(figsize=(5.5,10))

plt.subplot(311)
lineObjects = plt.plot(time_aligned, av_l2_test_d_aligned - av_l2_test_s_aligned, label=days)
plt.plot([0,0], [0,50], color='k', linestyle=':')
plt.legend(iter(lineObjects), (days), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('Class accuracy (testing data - shfl)', fontsize=13)
if chAl==1:
    plt.xlabel('Time relative to choice onset (ms)', fontsize=13)
else:
    plt.xlabel('Time relative stim onset (ms)', fontsize=13)
makeNicePlots(plt.gca())


##%%
#plt.figure(figsize=(5.5,6))
plt.subplot(312)
lineObjects = plt.plot(time_aligned, av_l2_test_d_aligned, label=days)
plt.plot([0,0], [50,100], color='k', linestyle=':')
plt.ylabel('Class accuracy (data)', fontsize=13)
#plt.legend(iter(lineObjects), (days), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.ylabel('class accuracy (testing data - shfl)', fontsize=13)
#if chAl==1:
#    plt.xlabel('Time relative to choice onset (ms)', fontsize=13)
#else:
#    plt.xlabel('Time relative stim onset (ms)', fontsize=13)
if chAl==1:
    plt.xlabel('Time relative to choice onset (ms)', fontsize=13)
else:
    plt.xlabel('Time relative stim onset (ms)', fontsize=13)
makeNicePlots(plt.gca())


plt.subplot(313)
lineObjects = plt.plot(time_aligned, av_l2_test_s_aligned, label=days)
plt.plot([0,0], [50,100], color='k', linestyle=':')
plt.ylabel('Class accuracy (shfl)', fontsize=13)
if chAl==1:
    plt.xlabel('Time relative to choice onset (ms)', fontsize=13)
else:
    plt.xlabel('Time relative stim onset (ms)', fontsize=13)
makeNicePlots(plt.gca())

plt.subplots_adjust(wspace=1, hspace=.5)


if doIncorr==1:
    plt.figure(figsize=(5.5,3))
#    plt.subplot(311)
    lineObjects = plt.plot(time_aligned, av_l2_test_incorr_d_aligned - av_l2_test_incorr_s_aligned, label=days)
    plt.plot([0,0], [0,50], color='k', linestyle=':')
    plt.legend(iter(lineObjects), (days), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('Class accuracy (testing data - shfl)', fontsize=13)
    if chAl==1:
        plt.xlabel('Time relative to choice onset (ms)', fontsize=13)
    else:
        plt.xlabel('Time relative stim onset (ms)', fontsize=13)
    makeNicePlots(plt.gca())


##%% Save the figure    
if savefigs:#% Save the figure
    if chAl==1:
        dd = 'chAl_testMshfl_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
    else:
        dd = 'stAl_testMshfl_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
    
    







#%%
#%% Keep vars for chAl and stAl

if chAl==1:    
    eventI_ds_allDays_ch = eventI_ds_allDays + 0    
    av_l2_train_d_ch = av_l2_train_d + 0
    sd_l2_train_d_ch = sd_l2_train_d + 0    
    av_l2_test_d_ch = av_l2_test_d + 0
    sd_l2_test_d_ch = sd_l2_test_d + 0    
    av_l2_test_s_ch = av_l2_test_s + 0
    sd_l2_test_s_ch = sd_l2_test_s + 0    
    av_l2_test_c_ch = av_l2_test_c + 0
    sd_l2_test_c_ch = sd_l2_test_c + 0
    if doIncorr==1:
        av_l2_test_incorr_d_ch = av_l2_test_incorr_d + 0
        sd_l2_test_incorr_d_ch = sd_l2_test_incorr_d + 0        
        av_l2_test_incorr_s_ch = av_l2_test_incorr_s + 0
        sd_l2_test_incorr_s_ch = sd_l2_test_incorr_s + 0        
        av_l2_test_incorr_c_ch = av_l2_test_incorr_c + 0
        sd_l2_test_incorr_c_ch = sd_l2_test_incorr_c + 0        
else:        
    eventI_ds_allDays_st = eventI_ds_allDays + 0    
    av_l2_train_d_st = av_l2_train_d + 0
    sd_l2_train_d_st = sd_l2_train_d + 0    
    av_l2_test_d_st = av_l2_test_d + 0
    sd_l2_test_d_st = sd_l2_test_d + 0    
    av_l2_test_s_st = av_l2_test_s + 0
    sd_l2_test_s_st = sd_l2_test_s + 0    
    av_l2_test_c_st = av_l2_test_c + 0
    sd_l2_test_c_st = sd_l2_test_c + 0
    

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% # For each day plot dist of event times as well as class accur traces. Do it for both stim-aligned and choice-aligned data 
# For this section you need to run eventTimesDist.py; also run the codes above once with chAl=0 and once with chAl=1.
no
# check if it should be eventI_allDays or eventI_ds_allDays
if 'eventI_allDays_ch' in locals() and 'eventI_allDays_st' in locals():
    
    #%% Set time vars needed below and Plot time dist of trial events for all days (pooled trials)
    
    execfile("eventTimesDist.py")
    
    
    #%% Plot class accur and event time dists for each day
    
    for iday in range(len(days)):    
    
        #%% Set event time vars
    
        # relative to stim onset
        timeStimOffset0_all_relOn = np.nanmean(timeStimOffset0_all[iday] - timeStimOnset_all[iday])
        timeStimOffset_all_relOn = np.nanmean(timeStimOffset_all[iday] - timeStimOnset_all[iday])
        timeCommitCL_CR_Gotone_all_relOn = np.nanmean(timeCommitCL_CR_Gotone_all[iday] - timeStimOnset_all[iday])
        time1stSideTry_all_relOn = np.nanmean(time1stSideTry_all[iday] - timeStimOnset_all[iday])
    
        # relative to choice onset
        timeStimOnset_all_relCh = np.nanmean(timeStimOnset_all[iday] - time1stSideTry_all[iday])    
        timeStimOffset0_all_relCh = np.nanmean(timeStimOffset0_all[iday] - time1stSideTry_all[iday])
        timeStimOffset_all_relCh = np.nanmean(timeStimOffset_all[iday] - time1stSideTry_all[iday])
        timeCommitCL_CR_Gotone_all_relCh = np.nanmean(timeCommitCL_CR_Gotone_all[iday] - time1stSideTry_all[iday])
        
        
        #%% stAl; Plot class accur trace and timeDists for each day
        
        plt.figure()
        
        ############### class accuracy
        colors = 'k','r','b','m'
        nPre = eventI_allDays_st[iday] # number of frames before the common eventI, also the index of common eventI. 
        nPost = (len(av_l2_test_d_st[iday]) - eventI_allDays_st[iday] - 1)
        
        a = -(np.asarray(frameLength*regressBins) * range(nPre+1)[::-1])
        b = (np.asarray(frameLength*regressBins) * range(1, nPost+1))
        time_al = np.concatenate((a,b))
        
        plt.subplot(223)
        plt.errorbar(time_al, av_l2_test_d_st[iday], yerr = sd_l2_test_d_st[iday])    
    
        # mark event times relative to stim onset
        plt.plot([0, 0], [50, 100], color='g')
        plt.plot([timeStimOffset0_all_relOn, timeStimOffset0_all_relOn], [50, 100], color=colors[0])
        plt.plot([timeStimOffset_all_relOn, timeStimOffset_all_relOn], [50, 100], color=colors[1])
        plt.plot([timeCommitCL_CR_Gotone_all_relOn, timeCommitCL_CR_Gotone_all_relOn], [50, 100], color=colors[2])
        plt.plot([time1stSideTry_all_relOn, time1stSideTry_all_relOn], [50, 100], color=colors[3])
        plt.xlabel('Time relative to stim onset (ms)')
        plt.ylabel('Classification accuracy (%)') #, fontsize=13)
        
        ax = plt.gca(); xl = ax.get_xlim(); makeNicePlots(ax,1)
        
        
        
        
        ################# Plot hist of event times
        stimOffset0_st = timeStimOffset0_all[iday] - timeStimOnset_all[iday]
        stimOffset_st = timeStimOffset_all[iday] - timeStimOnset_all[iday]
        goTone_st = timeCommitCL_CR_Gotone_all[iday] - timeStimOnset_all[iday]
        sideTry_st = time1stSideTry_all[iday] - timeStimOnset_all[iday]
    
        stimOffset0_st = stimOffset0_st[~np.isnan(stimOffset0_st)]
        stimOffset_st = stimOffset_st[~np.isnan(stimOffset_st)]
        goTone_st = goTone_st[~np.isnan(goTone_st)]
        sideTry_st = sideTry_st[~np.isnan(sideTry_st)]
        
        labs = 'stimOffset_1rep', 'stimOffset', 'goTone', '1stSideTry'
        a = stimOffset0_st, stimOffset_st, goTone_st, sideTry_st
        binEvery = 100
        
        bn = np.arange(np.min(np.concatenate((a))), np.max(np.concatenate((a))), binEvery)
        bn[-1] = np.max(np.concatenate(a)) # unlike digitize, histogram doesn't count the right most value
        
        plt.subplot(221)
        # set hists
        for i in range(len(a)):
            hist, bin_edges = np.histogram(a[i], bins=bn)
        #    hist = hist/float(np.sum(hist))     # use this if you want to get fraction of trials instead of number of trials
            hb = plt.bar(bin_edges[0:-1], hist, binEvery, alpha=.4, color=colors[i], label=labs[i]) 
        
    #    plt.xlabel('Time relative to stim onset (ms)')
        plt.ylabel('Number of trials')
        plt.legend(handles=[hb], loc='center left', bbox_to_anchor=(.7, .7)) 
        
        plt.title(days[iday])
        plt.xlim(xl)    
        ax = plt.gca(); makeNicePlots(ax,1)
    
    
    
        #%% chAl; plot class accuracy and timeDists
    
        ############### class accuracy
        colors = 'g','k','r','b'
        nPre = eventI_allDays_ch[iday] # number of frames before the common eventI, also the index of common eventI. 
        nPost = (len(av_l2_test_d_ch[iday]) - eventI_allDays_ch[iday] - 1)
        
        a = -(np.asarray(frameLength*regressBins) * range(nPre+1)[::-1])
        b = (np.asarray(frameLength*regressBins) * range(1, nPost+1))
        time_al = np.concatenate((a,b))
        
        plt.subplot(224)
        plt.errorbar(time_al, av_l2_test_d_ch[iday], yerr = sd_l2_test_d_ch[iday])  
        
        # mark event times relative to choice onset
        plt.plot([timeStimOnset_all_relCh, timeStimOnset_all_relCh], [50, 100], color=colors[0])    
        plt.plot([timeStimOffset0_all_relCh, timeStimOffset0_all_relCh], [50, 100], color=colors[1])
        plt.plot([timeStimOffset_all_relCh, timeStimOffset_all_relCh], [50, 100], color=colors[2])
        plt.plot([timeCommitCL_CR_Gotone_all_relCh, timeCommitCL_CR_Gotone_all_relCh], [50, 100], color=colors[3])
        plt.plot([0, 0], [50, 100], color='m')
        plt.xlabel('Time relative to choice onset (ms)')
        ax = plt.gca(); xl = ax.get_xlim(); makeNicePlots(ax,1)
        
        
        
        ################## Plot hist of event times
        stimOnset_ch = timeStimOnset_all[iday] - time1stSideTry_all[iday]
        stimOffset0_ch = timeStimOffset0_all[iday] - time1stSideTry_all[iday]
        stimOffset_ch = timeStimOffset_all[iday] - time1stSideTry_all[iday]
        goTone_ch = timeCommitCL_CR_Gotone_all[iday] - time1stSideTry_all[iday]
        
        stimOnset_ch = stimOnset_ch[~np.isnan(stimOnset_ch)]
        stimOffset0_ch = stimOffset0_ch[~np.isnan(stimOffset0_ch)]
        stimOffset_ch = stimOffset_ch[~np.isnan(stimOffset_ch)]
        goTone_ch = goTone_ch[~np.isnan(goTone_ch)]    
        
        labs = 'stimOnset','stimOffset_1rep', 'stimOffset', 'goTone'
        a = stimOnset_ch, stimOffset0_ch, stimOffset_ch, goTone_ch
        binEvery = 100
        
        bn = np.arange(np.min(np.concatenate((a))), np.max(np.concatenate((a))), binEvery)
        bn[-1] = np.max(np.concatenate(a)) # unlike digitize, histogram doesn't count the right most value
        
        plt.subplot(222)
        # set hists
        for i in range(len(a)):
            hist, bin_edges = np.histogram(a[i], bins=bn)
        #    hist = hist/float(np.sum(hist))     # use this if you want to get fraction of trials instead of number of trials
            plt.bar(bin_edges[0:-1], hist, binEvery, alpha=.4, color=colors[i], label=labs[i]) 
        
    #    plt.xlabel('Time relative to choice onset (ms)')
    #    plt.ylabel('Number of trials')
        plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 
        
        plt.xlim(xl)    
        ax = plt.gca(); makeNicePlots(ax,1)
        
        plt.subplots_adjust(wspace=1, hspace=.5)
        


        ##############%% Save the figure for each day 
        if savefigs:
            dd = 'timeDists_' + days[iday]
                
            d = os.path.join(svmdir+dnow)
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)
                    
            fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        
        
        """
        plt.subplot(223)
        a = timeStimOffset0_all[iday] - timeStimOnset_all[iday]
        plt.hist(a[~np.isnan(a)], color='b')
        
        a = timeStimOffset_all[iday] - timeStimOnset_all[iday]
        plt.hist(a[~np.isnan(a)], color='r')
    
        a = timeCommitCL_CR_Gotone_all[iday] - timeStimOnset_all[iday]
        plt.hist(a[~np.isnan(a)], color='k')
    
        a = time1stSideTry_all[iday] - timeStimOnset_all[iday]
        plt.hist(a[~np.isnan(a)], color='m')
        
        plt.xlabel('Time relative to stim onset (ms)')
        plt.ylabel('# trials')   
        """
    
        
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
