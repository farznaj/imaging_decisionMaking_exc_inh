# -*- coding: utf-8 -*-
"""
Summary of all mice: 
Plots svm class accuracy in last the time bin before the choice
svm trained using all neurons. 
 
Remember for fni18 there are 2 svm_eachFrame mat files, the earlier file is using all trials (unequal HR, LR, like how you've done all your analysis). 
The later mat file is with equal number of hr and lr trials (subselecting trials)... this helped with 151209 class accur trace which was weird in the earlier mat file.
 
Created on Sun Mar 12 15:12:29 2017
@author: farznaj
"""     


#%%
mice = 'fni16', 'fni17', 'fni18', 'fni19'

savefigs = 1
corrTrained = 1 # whether svm was trained using only correct trials (set to 1) or all trials (set to 0).
doIncorr = 0 # plot incorr CA 
time2an = -1 # relative to eventI, look at classErr in what time stamp.

thTrained = 10 # number of trials of each class used for svm training, min acceptable value to include a day in analysis
superimpose = 1 # the averaged aligned traces of testing and shuffled will be plotted on the same figure
loadWeights = 0

ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone.
doPlots = 0 #1 # plot c path of each day 
trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  

#eps = 10**-10 # tiny number below which weight is considered 0
#thNon0Ws = 2 # For samples with <2 non0 weights, we manually set their class error to 50 ... the idea is that bc of difference in number of HR and LR trials, in these samples class error is not accurately computed!
#thSamps = 10  # Days that have <thSamps samples that satisfy >=thNon0W non0 weights will be manually set to 50 (class error of all their samples) ... bc we think <5 samples will not give us an accurate measure of class error of a day.
#setTo50 = 1 # if 1, the above two jobs will be done.

import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.
    
smallestC = 0 # Identify best c: if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
if smallestC==1:
    print 'bestc = smallest c whose cv error is less than 1se of min cv error'
else:
    print 'bestc = c that gives min cv error'
#I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.

chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]
    
dnow0 = '/classAccurTraces_eachFrame_timeDists/'
#dnow0 = '/classAccurTraces_eachFrame/frame'+str(time2an)+'/'
       
from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')

execfile("defFuns.py") # Define common funcitons


#%%
av_train_data_allMice = []
sd_train_data_allMice = []
av_test_data_allMice = []
sd_test_data_allMice = []
av_test_shfl_allMice = []
sd_test_shfl_allMice = []
av_test_chance_allMice = []
sd_test_chance_allMice = []
corr_hr_lr_allMice = []

for im in range(len(mice)):
        
    #%%                
    mousename = mice[im] # mousename = 'fni16' #'fni17'
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
 
    
#    execfile("svm_plots_setVars_n.py")      #    execfile("svm_plots_setVars.py")      
    days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays)
    
    
    #%%     
    dnow = '/classAccurTraces_eachFrame_timeDists/'+mousename+'/'
#    dnow = '/classAccurTraces_eachFrame/frame'+str(time2an)+'/'+mousename+'/'
    if corrTrained:
        dnow = dnow+'corrTrained/'              
                
                
    #%% Loop over days    
    
    eventI_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
    eventI_ds_allDays = np.full((len(days)), np.nan)
    classErr_bestC_train_data_all = [] # np.full((numSamp*nRandCorrSel, len(days)), np.nan)
    classErr_bestC_test_data_all = []
    classErr_bestC_test_shfl_all = []
    classErr_bestC_test_chance_all = []
    cbestFrs_all = []        
    corr_hr_lr = np.full((len(days),2), np.nan) # number of hr, lr correct trials for each day
#    if loadWeights==1:            
#        numInh = np.full((len(days)), np.nan)
#        numAllexc = np.full((len(days)), np.nan)    
       
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
    
    
        #%% Load matlab vars to set eventI_ds (downsampled eventI)

        eventI, eventI_ds = setEventIds(postName, chAl, regressBins=3, trialHistAnalysis=0)
        
        eventI_allDays[iday] = eventI
        eventI_ds_allDays[iday] = eventI_ds            

                
        #%%
        svmName = setSVMname_allN_eachFrame(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained) # for chAl: the latest file is with soft norm; earlier file is 
        
        svmName = svmName[0]
        print os.path.basename(svmName)


        #%% Get number of hr, lr trials that were used for svm training
            
        corr_hr, corr_lr = set_corr_hr_lr(postName, svmName)
        corr_hr_lr[iday,:] = [corr_hr, corr_lr]        


        #%% Load SVM vars
    
        if doIncorr==1:
            classErr_bestC_train_data, classErr_bestC_test_data, classErr_bestC_test_shfl, classErr_bestC_test_chance, cbestFrs, w_bestc_data, b_bestc_data, classErr_bestC_test_incorr, classErr_bestC_test_incorr_shfl, classErr_bestC_test_incorr_chance = loadSVM_allN(svmName, doPlots, doIncorr, loadWeights)
        else:
            classErr_bestC_train_data, classErr_bestC_test_data, classErr_bestC_test_shfl, classErr_bestC_test_chance, cbestFrs, w_bestc_data, b_bestc_data = loadSVM_allN(svmName, doPlots, doIncorr, loadWeights)


        #%% Get class error values for the time bin before choice
        
        classErr_bestC_train_data = classErr_bestC_train_data[:,eventI_ds+time2an].squeeze() # numSamps
        classErr_bestC_test_data = classErr_bestC_test_data[:,eventI_ds+time2an].squeeze() # numSamps
        classErr_bestC_test_shfl = classErr_bestC_test_shfl[:,eventI_ds+time2an].squeeze() # numSamps
        classErr_bestC_test_chance = classErr_bestC_test_chance[:,eventI_ds+time2an].squeeze() # numSamps
    
    
        #%% Keep vars for all days
               
        # Delete vars before starting the next day           
#        del perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance            
        classErr_bestC_train_data_all.append(classErr_bestC_train_data) 
        classErr_bestC_test_data_all.append(classErr_bestC_test_data)
        classErr_bestC_test_shfl_all.append(classErr_bestC_test_shfl)
        classErr_bestC_test_chance_all.append(classErr_bestC_test_chance)
        cbestFrs_all.append(cbestFrs)    
    

    
    ######################################################################################################################################################        
    ######################################################################################################################################################        
    #%% Done with all days
    
    eventI_allDays = eventI_allDays.astype('int')
    eventI_ds_allDays = eventI_ds_allDays.astype('int')
    
    classErr_bestC_train_data_all = np.array(classErr_bestC_train_data_all) # days x samps
    classErr_bestC_test_data_all = np.array(classErr_bestC_test_data_all)
    classErr_bestC_test_shfl_all = np.array(classErr_bestC_test_shfl_all)
    classErr_bestC_test_chance_all = np.array(classErr_bestC_test_chance_all)
    

    #%% Keep original days before removing low tr days.
    
    numDays0 = numDays
    days0 = days
    
    
    #%% Decide what days to analyze: exclude days with too few trials used for training SVM, also exclude incorr from days with too few incorr trials.
    
    # th for min number of trs of each class
    '''
    thTrained = 30 #25; # 1/10 of this will be the testing tr num! and 9/10 was used for training
    thIncorr = 4 #5
    '''
    mn_corr = np.min(corr_hr_lr,axis=1) # number of trials of each class. 90% of this was used for training, and 10% for testing.
    
    print 'num days to be excluded with few svm-trained trs:', sum(mn_corr < thTrained)    
    print np.array(days)[mn_corr < thTrained]
    
    numDays = sum(mn_corr>=thTrained)
    days = np.array(days)[mn_corr>=thTrained]

    
    #%% Average and std of class accuracies across CV samples ... for each day
    #### ONLY include days with enough trs used for svm training!
    
    av_train_data = 100-np.mean(classErr_bestC_train_data_all[mn_corr >= thTrained,:], axis=1) # average across cv samples
    sd_train_data = np.std(classErr_bestC_train_data_all[mn_corr >= thTrained,:], axis=1) / np.sqrt(numSamples)

    av_test_data = 100-np.mean(classErr_bestC_test_data_all[mn_corr >= thTrained,:], axis=1)
    sd_test_data = np.std(classErr_bestC_test_data_all[mn_corr >= thTrained,:], axis=1) / np.sqrt(numSamples)
        
    av_test_shfl = 100-np.mean(classErr_bestC_test_shfl_all[mn_corr >= thTrained,:], axis=1)
    sd_test_shfl = np.std(classErr_bestC_test_shfl_all[mn_corr >= thTrained,:], axis=1) / np.sqrt(numSamples)

    av_test_chance = 100-np.mean(classErr_bestC_test_chance_all[mn_corr >= thTrained,:], axis=1)
    sd_test_chance = np.std(classErr_bestC_test_chance_all[mn_corr >= thTrained,:], axis=1) / np.sqrt(numSamples)
    
    
    
    #%% Keep values of all mice (cvSample-averaged)
    
    corr_hr_lr_allMice.append(corr_hr_lr)
    
    av_train_data_allMice.append(av_train_data)
    sd_train_data_allMice.append(sd_train_data)
    
    av_test_data_allMice.append(av_test_data)
    sd_test_data_allMice.append(sd_test_data)

    av_test_shfl_allMice.append(av_test_shfl)
    sd_test_shfl_allMice.append(sd_test_shfl)

    av_test_chance_allMice.append(av_test_chance)
    sd_test_chance_allMice.append(sd_test_chance)

    
    _,pcorrtrace = stats.ttest_ind(av_test_data, av_test_shfl) # p value of class accuracy being different from 50
    print 'p value =', pcorrtrace     
    
    
    #%%
    ######################## PLOTS ########################
    
    #%% Plot class accuracy in the frame before the choice onset for each session
    
    plt.figure(figsize=(4.5,3))
    
    plt.errorbar(range(numDays), av_train_data, sd_train_data, label='train', color='r')
    plt.errorbar(range(numDays), av_test_data, sd_test_data, label='test', color='b')
    plt.errorbar(range(numDays), av_test_shfl, sd_test_shfl, label='shfl', color='k')
    plt.errorbar(range(numDays), av_test_chance, sd_test_chance, label='chance', color=[.6,.6,.6])
    
    plt.legend(loc='center left', bbox_to_anchor=(1, .7)) 
    plt.xlabel('Days')
    plt.ylabel('Classification accuracy (%)')
    makeNicePlots(plt.gca())
    plt.xlim([-.2,len(days)-1+.2])
    
    if savefigs:#% Save the figure
        if chAl==1:
            dd = 'chAl_eachDay_time' + str(time2an) + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
        else:
            dd = 'stAl_eachDay_time' + str(time2an) + '_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr
            
        d = os.path.join(svmdir+dnow)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
                
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    


######################################################################################################################################################        
######################################################################################################################################################        
#%%
av_train_data_allMice = np.array(av_train_data_allMice) # for each mouse, size: numDays
sd_train_data_allMice = np.array(sd_train_data_allMice)

av_test_data_allMice = np.array(av_test_data_allMice)
sd_test_data_allMice = np.array(sd_test_data_allMice)

av_test_shfl_allMice = np.array(av_test_shfl_allMice)
sd_test_shfl_allMice = np.array(sd_test_shfl_allMice)

av_test_chance_allMice = np.array(av_test_chance_allMice)
sd_test_chance_allMice = np.array(sd_test_chance_allMice)
    

#%% Average and se of classErr across sessions for each mouse

numMice = len(mice)

av_av_train_data_allMice = np.array([np.nanmean(av_train_data_allMice[im], axis=0) for im in range(numMice)])
av_av_test_data_allMice = np.array([np.nanmean(av_test_data_allMice[im], axis=0) for im in range(numMice)])
av_av_test_shfl_allMice = np.array([np.nanmean(av_test_shfl_allMice[im], axis=0) for im in range(numMice)])
av_av_test_chance_allMice = np.array([np.nanmean(av_test_chance_allMice[im], axis=0) for im in range(numMice)])

sd_av_train_data_allMice = np.array([np.nanstd(av_train_data_allMice[im], axis=0)/np.sqrt(numMice) for im in range(numMice)])
sd_av_test_data_allMice = np.array([np.nanstd(av_test_data_allMice[im], axis=0)/np.sqrt(numMice) for im in range(numMice)])
sd_av_test_shfl_allMice = np.array([np.nanstd(av_test_shfl_allMice[im], axis=0)/np.sqrt(numMice) for im in range(numMice)])
sd_av_test_chance_allMice = np.array([np.nanstd(av_test_chance_allMice[im], axis=0)/np.sqrt(numMice) for im in range(numMice)])


#%% Plot session-averaged and se of classErr for each mouse

plt.figure(figsize=(2,3))

plt.errorbar(range(numMice), av_av_train_data_allMice, sd_av_train_data_allMice, fmt='o', label='train', color='r')
plt.errorbar(range(numMice), av_av_test_data_allMice, sd_av_test_data_allMice, fmt='o', label='test', color='b')
plt.errorbar(range(numMice), av_av_test_shfl_allMice, sd_av_test_shfl_allMice, fmt='o', label='shfl', color='k')
plt.errorbar(range(numMice), av_av_test_chance_allMice, sd_av_test_chance_allMice, fmt='o', label='chance', color=[.6,.6,.6])

plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1) 
plt.xlabel('Mice', fontsize=11)
plt.ylabel('Classification accuracy (%)\n [-97 0] ms rel. choice', fontsize=11)
plt.xlim([-.2,numMice-1+.2])
plt.xticks(range(numMice),mice)
ax = plt.gca()
makeNicePlots(ax)


if savefigs:#% Save the figure

    if corrTrained:
        ct = 'corrTrained_'
    else:
        ct = ''
        
    if chAl==1:
        dd = 'chAl_eachDay_time' + str(time2an) + '_' + ct+'_'.join(mice) + '_' + nowStr # + days[0][0:6] + '-to-' + days[-1][0:6]
    else:
        dd = 'stAl_eachDay_time' + str(time2an) + '_' + ct+'_'.join(mice) + '_' + nowStr # + days[0][0:6] + '-to-' + days[-1][0:6]
        
    d = os.path.join(svmdir+dnow0)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow0, suffn[0:5]+dd+'.'+fmt[0])

    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)



