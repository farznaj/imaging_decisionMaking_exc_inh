# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:26:41 2017

@author: farznaj
"""

# -*- coding: utf-8 -*-
"""
Plots histogram of weights for exc, inh (svm trained on non-overlapping time windows)
 ... svm trained to decode choice on choice-aligned or stimulus-aligned traces.
 
 
Remember for fni18 there are 2 svm_eachFrame mat files, the earlier file is using all trials (unequal HR, LR, like how you've done all your analysis). 
The later mat file is with equal number of hr and lr trials (subselecting trials)... this helped with 151209 class accur trace which was weird in the earlier mat file.
 
Created on Sun Mar 12 15:12:29 2017
@author: farznaj
"""

#%%
mice = 'fni16', 'fni17', 'fni18', 'fni19'

savefigs = 0
doPlots = 1 #1 # plot hists of w per mouse
time2an = -1 # relative to eventI, look at classErr in what time stamp.
poolDaysOfMice = 0 # for plotting w traces; if 0, average and sd across mice (each mouse has session-averaged weights); if 1, average and sd across all sessions of all mice
thTrained = 10#10 # number of trials of each class used for svm training, min acceptable value to include a day in analysis
corrTrained = 1
doIncorr = 0
lastTimeBinMissed = 1 # if 0, things were run fine; if 1: by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)

trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]

import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.
    
smallestC = 0 # Identify best c: if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
if smallestC==1:
    print 'bestc = smallest c whose cv error is less than 1se of min cv error'
else:
    print 'bestc = c that gives min cv error'
#I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.
    
#dnow0 = '/excInh_trainDecoder_eachFrame_weights/frame'+str(time2an)+'/'
dnow0 = '/excInh_trainDecoder_eachFrame_weights/'
       
execfile("defFuns.py")  

from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')
dp = 'time' + str(time2an) + '_' + nowStr    
    
#%% 
'''
#####################################################################################################################################################   
############################ stimulus-aligned, SVR (trained on trials of 1 choice to avoid variations in neural response due to animal's choice... we only want stimulus rate to be different) ###################################################################################################     
#####################################################################################################################################################
'''

#%%

winhAve_all_allMice = []
wexcAve_all_allMice = []

winhAve_all2_allMice = []
wexcAve_all2_allMice = []

eventI_allDays_allMice = []
eventI_ds_allDays_allMice = []
corr_hr_lr_allMice = []
numDaysAll = np.full(len(mice), np.nan)


#%%
for im in [1,2,3]:#range(len(mice)):
        
    #%%            
    mousename = mice[im] # mousename = 'fni16' #'fni17'
    if mousename == 'fni18': #set one of the following to 1:
        allDays = 1# all 7 days will be used (last 3 days have z motion!)
        noZmotionDays = 0 # 4 days that dont have z motion will be used.
        noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!
    if mousename == 'fni19':    
        allDays = 1
        noExtraStimDays = 0   
        
    execfile("svm_plots_setVars_n.py")      
#    execfile("svm_plots_setVars.py")      
    numDaysAll[im] = len(days)

    
    #%%     
    dnow = '/excInh_trainDecoder_eachFrame_weights/'+mousename+'/'

            
    #%% Loop over days    
    
    eventI_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
    cbestFrs_all = []
    winh_all = []
    wexc_all = []
    winhAve_all = []
    wexcAve_all = []
    b_bestc_data_all = []        
    eventI_allDays = np.full((len(days)), np.nan)
    eventI_ds_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
    corr_hr_lr = np.full((len(days),2), np.nan) # number of hr, lr correct trials for each day
#    incorr_hr_lr = np.full((len(days),2), np.nan) # number of hr, lr incorrect trials for each day
    
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
        
        
        #%% Load inhibitRois
        
        Data = scio.loadmat(moreName, variable_names=['inhibitRois_pix'])
        inhRois = Data.pop('inhibitRois_pix')[0,:]    
#        print '%d inhibitory, %d excitatory; %d unsure class' %(np.sum(inhibitRois==1), np.sum(inhibitRois==0), np.sum(np.isnan(inhibitRois)))
        
        
        #%% Set eventI_ds (downsampled eventI)
    
        eventI, eventI_ds = setEventIds(postName, chAl, regressBins=3, trialHistAnalysis=0)
        
        eventI_allDays[iday] = eventI
        eventI_ds_allDays[iday] = eventI_ds    

        
        #%% Set number of hr, lr trials that were used for svm training
        
        corr_hr, corr_lr = set_corr_hr_lr(postName, svmName)    
        corr_hr_lr[iday,:] = [corr_hr, corr_lr]        

        
        #%% Load SVM vars
            
        Data = scio.loadmat(svmName, variable_names=['regType','cvect','perClassErrorTest']) #,'perClassErrorTrain','perClassErrorTest','perClassErrorTest_chance','perClassErrorTest_shfl'])
        
        regType = Data.pop('regType').astype('str')
        cvect = Data.pop('cvect').squeeze()
        perClassErrorTest = Data.pop('perClassErrorTest')
        Data = scio.loadmat(svmName, variable_names=['wAllC','bAllC'])
        wAllC = Data.pop('wAllC') # numSamples x len(cvect) x nNeurons x nFrames
        bAllC = Data.pop('bAllC') # numSamples x len(cvect) x nFrames        
           
            
        ####%% Find bestc for each frame (funcion is defined in defFuns.py)            
        cbestFrs = findBestC(perClassErrorTest, cvect, regType, smallestC) # nFrames        
    	
    	
        #####%% Set class error values at best C (if desired plot c path) (funcion is defined in defFuns.py)    
        doPlotscpath = 0;
        _, _, _, _, w_bestc_data, b_bestc_data = setClassErrAtBestc(cbestFrs, cvect, doPlotscpath, np.nan, np.nan, np.nan, np.nan, wAllC, bAllC)
        del perClassErrorTest, wAllC, bAllC    
    
    
        #%% Set w for inh and exc
               
        winh0 = w_bestc_data[:,inhRois==1,:] # numSamples x nNeurons x nFrames
        wexc0 = w_bestc_data[:,inhRois==0,:]

        # Take average across samples (this is like getting a single decoder across all trial subselects... so I guess bagging)... (not ave of abs... the idea is that ave of shuffles represents the neurons weights for the decoder given the entire dataset (all trial subselects)... if a w switches signs across samples then it means this neuron is not very consistently contributing to the choice... so i think we should average its w and not its abs(w))
        winhAve = np.mean(winh0, axis=0) # neurons x frames
        wexcAve = np.mean(wexc0, axis=0)
        
        
        #%% Once done with all frames, keep vars for all days
               
        # Delete vars before starting the next day           
    #    del perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance, wAllC, bAllC
            
        winhAve_all.append(winhAve) # aveSamps len: numdays; each day: (nNeurons x nFrames)
        wexcAve_all.append(wexcAve)

        # use weights of each samp... I think it is completely find to work the ave-samp weights... this ave decoder is the one that best represents each neurons weight                
        '''
        winh_all.append(winh0) # each samp (numSamples x nNeurons x nFrames)
        wexc_all.append(wexc0)     
        '''        
        b_bestc_data_all.append(b_bestc_data) 
    #    winh.append(np.mean(abs(winh0), axis=0))
    #    wexc.append(np.mean(abs(wexc0), axis=0))
        cbestFrs_all.append(cbestFrs)
        
    
    #%% Once done with all days keep vars for all days
    
    eventI_allDays = eventI_allDays.astype('int')
    eventI_ds_allDays = eventI_ds_allDays.astype('int')
    numDaysAll = numDaysAll.astype(int)
    cbestFrs_all = np.array(cbestFrs_all)    


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
    
#    numDays = sum(mn_corr>=thTrained)
#    days = np.array(days)[mn_corr>=thTrained]
    
    
    #%% Set weights for frame -1 (at the end of this page you compute inha and exca, which are a general version of _all2 vars.... ie they include w values for all frames instead of just frame -1.)
    # only include days with enough svm-trained trials    

    winhAve_all2 = np.array([winhAve_all[iday][:, eventI_ds_allDays[iday]+time2an] for iday in range(len(days)) if mn_corr[iday]>=thTrained]) #numGoodDays; each day:nNeurons
    wexcAve_all2 = np.array([wexcAve_all[iday][:, eventI_ds_allDays[iday]+time2an] for iday in range(len(days)) if mn_corr[iday]>=thTrained])
    

    #%% Keep values of all mice
    
    # aveSamps, all time bins
    winhAve_all_allMice.append(winhAve_all) # winhAve_all_allMice[im][iday] has size neurons x frames
    wexcAve_all_allMice.append(wexcAve_all)    

    # aveSamps, time bin -1
    # only include days with enough svm-trained trials    
    winhAve_all2_allMice.append(winhAve_all2)  # each day element: num neurons
    wexcAve_all2_allMice.append(wexcAve_all2)    
    
    eventI_allDays_allMice.append(eventI_allDays)
    eventI_ds_allDays_allMice.append(eventI_ds_allDays)
    corr_hr_lr_allMice.append(corr_hr_lr)
    
    


    
    
    #%% 
    ##############################################################################################################################################################
    #################################### PLOTS ###################################################################################################################
    ##############################################################################################################################################################
    ###### Histogram of w at frame -1 for each mouse #####################################################################################
    ##############################################################################################################################################################
    ##############################################################################################################################################################
    
    #%% only include days with enough svm-trained trials    
    
    if doPlots:
        lab1 = 'exc'
        lab2 = 'inh'
#        colors = ['k','r']
        colors = ['b','r']        
        
        
        ########################################
        ###############%% w  ###############
        ########################################
        ### hist and P val for all days pooled (exc vs inh)
        lab = 'w'
        binEvery = .001 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
        
        a = np.concatenate((wexcAve_all2)) #np.reshape(wexc,(-1,)) 
        b = np.concatenate((winhAve_all2)) #np.reshape(winh,(-1,))
        
        print np.sum(abs(a)<=eps), ' exc ws < eps'
        print np.sum(abs(b)<=eps), ' inh ws < eps'
        print [len(a), len(b)]
        _, p = stats.ttest_ind(a, b, nan_policy='omit')
        
        plt.figure(figsize=(5,5))    
        gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
        h1 = gs[0,0:2]
        h2 = gs[0,2:3]
        ax1,_ = histerrbar(a,b,binEvery,p,colors)
        #plt.xlabel(lab)
        ax1.set_xlim([-.05, .05])
        
        
        ##%% show individual days
        a = wexcAve_all2
        b = winhAve_all2
        # if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
        p = []
        for i in range(len(a)):
            _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
            p.append(p0)
        p = np.array(p)    
        print p
        #ax = plt.subplot(gs[1,0:3])
        errbarAllDays(a,b,p,colors)
        #plt.ylabel(lab)
        
        
        if savefigs:#% Save the figure
            d = os.path.join(svmdir+dnow)
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)

            if chAl==1:
                dd = 'chAl_excInh_w_days_' + days[0][0:6] + '-to-' + days[-1][0:6]
            else:
                dd = 'stAl_excInh_w_days_' + days[0][0:6] + '-to-' + days[-1][0:6]
         
            fign = os.path.join(d, suffn[0:5]+dd+'_'+dp+'.'+fmt[0])         
#            fign = os.path.join(d, suffn[0:5]+'excVSinh_allDays_w_'+dp+'.'+fmt[0])
            plt.savefig(fign, bbox_inches='tight')
            
            
        
        #%%
        ########################################
        ###############%% abs w  ###############
        ########################################
        ### hist and P val for all days pooled (exc vs inh)
        lab = 'abs w'
        binEvery = .002 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
        
        a = np.concatenate(abs(wexcAve_all2)) #np.reshape(wexc,(-1,)) 
        b = np.concatenate(abs(winhAve_all2)) #np.reshape(winh,(-1,))
        
        print np.sum(abs(a)<=eps), ' exc ws < eps'
        print np.sum(abs(b)<=eps), ' inh ws < eps'
        print [len(a), len(b)]
        _, p = stats.ttest_ind(a, b, nan_policy='omit')
        
        plt.figure(figsize=(5,5))    
        gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
        h1 = gs[0,0:2]
        h2 = gs[0,2:3]
        ax1,_ = histerrbar(a,b,binEvery,p,colors)
        #plt.xlabel(lab)
        ax1.set_xlim([0, .07])
        
        
        
        ### show individual days
        a = abs(wexcAve_all2)
        b = abs(winhAve_all2)
        # if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
        p = []
        for i in range(len(a)):
            _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
            p.append(p0)
        p = np.array(p)    
        print p
        #ax = plt.subplot(gs[1,0:3])
        errbarAllDays(a,b,p,colors)
        #plt.ylabel(lab)
        
        
        if savefigs:#% Save the figure
            d = os.path.join(svmdir+dnow)
            if not os.path.exists(d):
                print 'creating folder'
                os.makedirs(d)
                
            if chAl==1:
                dd = 'chAl_excInh_absw_days_' + days[0][0:6] + '-to-' + days[-1][0:6]
            else:
                dd = 'stAl_excInh_absw_days_' + days[0][0:6] + '-to-' + days[-1][0:6]
         
            fign = os.path.join(d, suffn[0:5]+dd+'_'+dp+'.'+fmt[0])
#            fign = os.path.join(d, suffn[0:5]+'excVSinh_allDays_absw_'+dp+'.'+fmt[0])            
            plt.savefig(fign, bbox_inches='tight')
    
    

        

    
no   
#%% PLOTS OF ALL MICE
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
#######################################################################################################################################################
##############################################################################################################################################################

dp = 'time' + str(time2an) #+ '_' + nowStr    

#%% Hist of w at frame -1 --> at the end of this page you compute inha and exca, which are a general version of _all2 vars.... ie they include w values for all frames instead of just frame -1.

if len(mice) > 1:
    dpp = '_'.join(mice) + '_' + nowStr   
    ##%%
    lab1 = 'exc'
    lab2 = 'inh'
#    colors = ['k','r']
#    colors = ['b','r']
    
    ########################################
    ###############%% w  ###############
    ########################################
    ### hist and P val for all days pooled (exc vs inh)
    lab = 'w'
    binEvery = .001 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
    
    aa = np.concatenate((wexcAve_all2_allMice)) #np.reshape(wexc,(-1,)) 
    bb = np.concatenate((winhAve_all2_allMice)) #np.reshape(winh,(-1,))
    a = np.concatenate((aa))
    b = np.concatenate((bb))
    
    print np.sum(abs(a)<=eps), ' exc ws < eps'
    print np.sum(abs(b)<=eps), ' inh ws < eps'
    print [len(a), len(b)]
    _, p = stats.ttest_ind(a, b, nan_policy='omit')
    
    fig = plt.figure(figsize=(5,5))    
    gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
    h1 = gs[0,0:2]
    h2 = gs[0,2:3]
    ax1,_ = histerrbar(a,b,binEvery,p,colors)
    #plt.xlabel(lab)
    ax1.set_xlim([-.05, .05])
    ax1.set_title('num exc = %d; inh = %d' %(len(a), len(b)))
    #fig.savefig('test.jpg')
    
    
    
    
    ############%% show individual mouse
    wexc_aveM = np.array([np.mean(np.concatenate((wexcAve_all2_allMice[im]))) for im in range(len(mice))])        
    winh_aveM = np.array([np.mean(np.concatenate((winhAve_all2_allMice[im]))) for im in range(len(mice))])
    
    wexc_sdM = np.array([np.std(np.concatenate((wexcAve_all2_allMice[im])))/np.sqrt(np.concatenate((wexcAve_all2_allMice[im])).shape[0]) for im in range(len(mice))])        
    winh_sdM = np.array([np.std(np.concatenate((winhAve_all2_allMice[im])))/np.sqrt(np.concatenate((winhAve_all2_allMice[im])).shape[0]) for im in range(len(mice))])
    
    
    #plt.figure(figsize=(2,3))
    ax = plt.subplot(gs[1,0:2])
    plt.errorbar(range(len(mice)), wexc_aveM, wexc_sdM, fmt='o', label='exc', color=colors[0])
    plt.errorbar(range(len(mice)), winh_aveM, winh_sdM, fmt='o', label='inh', color=colors[1])
    
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1) 
    plt.xlabel('Mice', fontsize=11)
    plt.ylabel('Classifier weights', fontsize=11)
    plt.xlim([-.2,len(mice)-1+.2])
    plt.xticks(range(len(mice)),mice)
    ax = plt.gca()
    makeNicePlots(ax)
    
    
    
    if savefigs:#% Save the figure
        d = os.path.join(svmdir+dnow0)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
     
        fign = os.path.join(d, suffn[0:5]+dp+'_w_allDays_excVSinh_'+dpp+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight')
        
        
    
    #%%
    ########################################
    ###############%% abs w  ###############
    ########################################
    ### hist and P val for all days pooled (exc vs inh)
    lab = 'abs w'
    binEvery = .002 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
    
    a = np.concatenate((wexcAve_all2_allMice)) #np.reshape(wexc,(-1,)) 
    b = np.concatenate((winhAve_all2_allMice)) #np.reshape(winh,(-1,))
    a = np.concatenate(abs(a))
    b = np.concatenate(abs(b))
    
    
    print np.sum(abs(a)<=eps), ' exc ws < eps'
    print np.sum(abs(b)<=eps), ' inh ws < eps'
    print [len(a), len(b)]
    _, p = stats.ttest_ind(a, b, nan_policy='omit')
    
    plt.figure(figsize=(5,5))    
    gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
    h1 = gs[0,0:2]
    h2 = gs[0,2:3]
    ax1,_ = histerrbar(a,b,binEvery,p,colors)
    #plt.xlabel(lab)
    ax1.set_xlim([0, .07])
    ax1.set_title('num exc = %d; inh = %d' %(len(a), len(b)))
    
    
    ##%% show individual mouse
    wexc_aveM = np.array([np.mean(np.concatenate(abs(wexcAve_all2_allMice[im]))) for im in range(len(mice))])        
    winh_aveM = np.array([np.mean(np.concatenate(abs(winhAve_all2_allMice[im]))) for im in range(len(mice))])
    
    wexc_sdM = np.array([np.std(np.concatenate(abs(wexcAve_all2_allMice[im])))/np.sqrt(np.concatenate((wexcAve_all2_allMice[im])).shape[0]) for im in range(len(mice))])        
    winh_sdM = np.array([np.std(np.concatenate(abs(winhAve_all2_allMice[im])))/np.sqrt(np.concatenate((winhAve_all2_allMice[im])).shape[0]) for im in range(len(mice))])
    
    
    #plt.figure(figsize=(2,3))
    ax = plt.subplot(gs[1,0:2])
    plt.errorbar(range(len(mice)), wexc_aveM, wexc_sdM, fmt='o', label='exc', color=colors[0])
    plt.errorbar(range(len(mice)), winh_aveM, winh_sdM, fmt='o', label='inh', color=colors[1])
    
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1) 
    plt.xlabel('Mice', fontsize=11)
    plt.ylabel('Abs (classifier weights)', fontsize=11)
    plt.xlim([-.2,len(mice)-1+.2])
    plt.xticks(range(len(mice)),mice)
    ax = plt.gca()
    makeNicePlots(ax)
    
    
    if savefigs:#% Save the figure
        d = os.path.join(svmdir+dnow0)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
     
        fign = os.path.join(d, suffn[0:5]+dp+'_absw_allDays_excVSinh_'+dpp+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight')
    
        
        
        
    
        
    
no
    #%%    
    ####################################################################################    
    ####################################################################################
    ####################################################################################
        
    ##%% Plot traces of w magnitude (ie w magnitude at each frame):
                
    ##%% Plot ave+-sd of w at each frame
    
    ##################################################################################################
    ############## Align weights at all frames of all days to make a final average trace ##############
    ################################################################################################## 
     ##### nPreMin : index of eventI on aligned traces ####
     
    # use eventI_allDays_allMice to align the above for all sessions (of perhaps all mice)    
    
    #%%  Compute average weights across neurons for each frame
       
    # average weights across neurons for each frame ... for each session of each mouse
    winhAve_all_allMice_eachFr = []
    wexcAve_all_allMice_eachFr = []
    winhAbsAve_all_allMice_eachFr = []
    wexcAbsAve_all_allMice_eachFr = []
    for im in range(len(mice)):
        # average across neurons for each frame
        winhAve_all_allMice_eachFr.append(np.array([np.mean(winhAve_all_allMice[im][iday], axis=0) for iday in range(numDaysAll[im])]))
        wexcAve_all_allMice_eachFr.append(np.array([np.mean(wexcAve_all_allMice[im][iday], axis=0) for iday in range(numDaysAll[im])]))    
         # ABS w: average across neurons for each frame
        winhAbsAve_all_allMice_eachFr.append(np.array([np.mean(abs(winhAve_all_allMice[im][iday]), axis=0) for iday in range(numDaysAll[im])]))
        wexcAbsAve_all_allMice_eachFr.append(np.array([np.mean(abs(wexcAve_all_allMice[im][iday]), axis=0) for iday in range(numDaysAll[im])]))    

    winhAve_all_allMice_eachFr = np.array(winhAve_all_allMice_eachFr)
    wexcAve_all_allMice_eachFr = np.array(wexcAve_all_allMice_eachFr)
    winhAbsAve_all_allMice_eachFr = np.array(winhAbsAve_all_allMice_eachFr)
    wexcAbsAve_all_allMice_eachFr = np.array(wexcAbsAve_all_allMice_eachFr)
    
    
    
    
    #%% 
    ##%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
    # By common eventI, we  mean the index on which all traces will be aligned.
    
    nPostAll = []
    for im in range(len(mice)):
        numDays = numDaysAll[im]
        nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
        for iday in range(numDays):
            nPost[iday] = (len(winhAve_all_allMice_eachFr[im][iday]) - eventI_ds_allDays_allMice[im][iday] - 1)
        nPostAll.append(nPost)
        
    nPreMin = np.min(np.concatenate((eventI_ds_allDays_allMice))) #min(eventI_allDays) # number of frames before the common eventI, also the index of common eventI. 
    nPostMin = min(np.concatenate((nPostAll)))
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
    time_aligned = set_time_al(totLen, np.min(np.concatenate((eventI_allDays_allMice))), lastTimeBinMissed)
    
    print time_aligned
    
    
    #%% Align neuron-averaged weight traces of all days on the common eventI
    # only include days with enough svm-trained trials    
    
    ##### average-neuron weights
    winhAve_all_allMice_eachFr_aligned_all = [] # each mouse is frames x sessions
    wexcAve_all_allMice_eachFr_aligned_all = []
    winhAbsAve_all_allMice_eachFr_aligned_all = [] # each mouse is frames x sessions
    wexcAbsAve_all_allMice_eachFr_aligned_all = []    
    for im in range(len(mice)):
        
        corr_hr_lr = corr_hr_lr_allMice[im]
        mn_corr = np.min(corr_hr_lr,axis=1)
        
        numDays = numDaysAll[im]
        
        winhAve_all_allMice_eachFr_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
        wexcAve_all_allMice_eachFr_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)    
        winhAbsAve_all_allMice_eachFr_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
        wexcAbsAve_all_allMice_eachFr_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)    
        
        for iday in range(numDays):
            if mn_corr[iday] >= thTrained:
                winhAve_all_allMice_eachFr_aligned[:, iday] = winhAve_all_allMice_eachFr[im][iday][eventI_ds_allDays_allMice[im][iday] - nPreMin  :  eventI_ds_allDays_allMice[im][iday] + nPostMin + 1]
                wexcAve_all_allMice_eachFr_aligned[:, iday] = wexcAve_all_allMice_eachFr[im][iday][eventI_ds_allDays_allMice[im][iday] - nPreMin  :  eventI_ds_allDays_allMice[im][iday] + nPostMin + 1]
                winhAbsAve_all_allMice_eachFr_aligned[:, iday] = winhAbsAve_all_allMice_eachFr[im][iday][eventI_ds_allDays_allMice[im][iday] - nPreMin  :  eventI_ds_allDays_allMice[im][iday] + nPostMin + 1]
                wexcAbsAve_all_allMice_eachFr_aligned[:, iday] = wexcAbsAve_all_allMice_eachFr[im][iday][eventI_ds_allDays_allMice[im][iday] - nPreMin  :  eventI_ds_allDays_allMice[im][iday] + nPostMin + 1]
    
        winhAve_all_allMice_eachFr_aligned_all.append(winhAve_all_allMice_eachFr_aligned)
        wexcAve_all_allMice_eachFr_aligned_all.append(wexcAve_all_allMice_eachFr_aligned)    
        winhAbsAve_all_allMice_eachFr_aligned_all.append(winhAbsAve_all_allMice_eachFr_aligned)
        wexcAbsAve_all_allMice_eachFr_aligned_all.append(wexcAbsAve_all_allMice_eachFr_aligned)    
    
    
        
    #%% Average neuron-averaged weights across sessions for each mouse ... these vars will be used if poolDaysOfMice = 0 
    
    winhAveDay_aligned = np.array([np.nanmean(winhAve_all_allMice_eachFr_aligned_all[im], axis=1) for im in range(len(mice))]) # numMouse x numFrames
    wexcAveDay_aligned = np.array([np.nanmean(wexcAve_all_allMice_eachFr_aligned_all[im], axis=1) for im in range(len(mice))]) # numMouse x numFrames
    
    winhAbsAveDay_aligned = np.array([np.nanmean(winhAbsAve_all_allMice_eachFr_aligned_all[im], axis=1) for im in range(len(mice))]) # numMouse x numFrames
    wexcAbsAveDay_aligned = np.array([np.nanmean(wexcAbsAve_all_allMice_eachFr_aligned_all[im], axis=1) for im in range(len(mice))]) # numMouse x numFrames
    
    
    #%% Pool neuron-average weights across sessions and mice ... these vars will be used if poolDaysOfMice = 1
    
    winhAllDaysMouse = [] # (numFrs x numMice) ... first all frames of mouse 0, then all frames of mouse 1, then all frames of mouse 2
    wexcAllDaysMouse = []
    winhAbsAllDaysMouse = [] # (numFrs x numMice) ... first all frames of mouse 0, then all frames of mouse 1, then all frames of mouse 2
    wexcAbsAllDaysMouse = []
    for im in range(len(mice)):
        for ifr in range(len(time_aligned)):
            winhAllDaysMouse.append(winhAve_all_allMice_eachFr_aligned_all[im][ifr]) # weights for all sessions of mouse im, and frame ifr
            wexcAllDaysMouse.append(wexcAve_all_allMice_eachFr_aligned_all[im][ifr])
            winhAbsAllDaysMouse.append(winhAbsAve_all_allMice_eachFr_aligned_all[im][ifr]) # weights for all sessions of mouse im, and frame ifr
            wexcAbsAllDaysMouse.append(wexcAbsAve_all_allMice_eachFr_aligned_all[im][ifr])    
    #if len(mice)==1:
    #    winhAllDaysMouse[0] = winhAllDaysMouse
    #    wexcAllDaysMouse[0] = wexcAllDaysMouse
            
    a = np.reshape(winhAllDaysMouse, (len(time_aligned), len(mice)), order = 'F') # frames x mice, each contains all sessions of a mouse
    winhAllDaysMice_aligned = np.array([np.concatenate(a[ifr]) for ifr in range(len(time_aligned))]).transpose() # (sessions of all mice) x frames
    a = np.reshape(wexcAllDaysMouse, (len(time_aligned), len(mice)), order = 'F') # frames x mice, each contains all sessions of a mouse
    wexcAllDaysMice_aligned = np.array([np.concatenate(a[ifr]) for ifr in range(len(time_aligned))]).transpose() # (sessions of all mice) x frames
    
    a = np.reshape(winhAbsAllDaysMouse, (len(time_aligned), len(mice)), order = 'F') # frames x mice, each contains all sessions of a mouse
    winhAbsAllDaysMice_aligned = np.array([np.concatenate(a[ifr]) for ifr in range(len(time_aligned))]).transpose() # (sessions of all mice) x frames
    a = np.reshape(wexcAbsAllDaysMouse, (len(time_aligned), len(mice)), order = 'F') # frames x mice, each contains all sessions of a mouse
    wexcAbsAllDaysMice_aligned = np.array([np.concatenate(a[ifr]) for ifr in range(len(time_aligned))]).transpose() # (sessions of all mice) x frames
    
    ### Abs
#    winhAbsAllDaysMice_aligned = abs(winhAllDaysMice_aligned)
#    wexcAbsAllDaysMice_aligned = abs(wexcAllDaysMice_aligned)
    
    
    #%% Take the final average and sd
    
#    poolDaysOfMice = 0 # for plotting w traces; if 0, average and sd across mice (each mouse has session-averaged weights); if 1, average and sd across all sessions of all mice
    
    if poolDaysOfMice==0:
        # average and sd across mice (each mouse has session-averaged weights)
        a = winhAveDay_aligned
        b = wexcAveDay_aligned
        aa = winhAbsAveDay_aligned
        bb = wexcAbsAveDay_aligned
    
    else:
        # average and sd across all sessions of all mice
        a = winhAllDaysMice_aligned
        b = wexcAllDaysMice_aligned
        aa = winhAbsAllDaysMice_aligned
        bb = wexcAbsAllDaysMice_aligned
    
    
    #a = winh_fractnon0
    #b = wexc_fractnon0
        
    av_winh = np.nanmean(a, axis=0)
    sd_winh = np.nanstd(a, axis=0)/ np.shape(a)[0]
    
    av_wexc = np.nanmean(b, axis=0)
    sd_wexc = np.nanstd(b, axis=0)/ np.shape(b)[0]
    
    #### Abs
    av_winh_abs = np.nanmean(aa, axis=0)
    sd_winh_abs = np.nanstd(aa, axis=0)/ np.shape(aa)[0]
    
    av_wexc_abs = np.nanmean(bb, axis=0)
    sd_wexc_abs = np.nanstd(bb, axis=0)/ np.shape(bb)[0]
    
    
    _,pcorrtrace = stats.ttest_ind(a, b) # p value of class accuracy being different from 50
    _,pAbscorrtrace = stats.ttest_ind(aa, bb)
    
    
           
    #%% Plot the average of aligned traces across all days
    
    fig = plt.figure(figsize=(3,4))
    gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
    h1 = gs[0,0:3]
    h2 = gs[1,0:3]
    
    
    plt.subplot(h1)
    plt.fill_between(time_aligned, av_wexc - sd_wexc, av_wexc + sd_wexc, alpha=0.5, edgecolor=colors[0], facecolor=colors[0])
    plt.plot(time_aligned, av_wexc, colors[0], label='exc')
    
    plt.fill_between(time_aligned, av_winh - sd_winh, av_winh + sd_winh, alpha=0.5, edgecolor=colors[1], facecolor=colors[1])
    plt.plot(time_aligned, av_winh, colors[1], label='inh')
    
    
    if chAl==1:
        plt.xlabel('Time since choice onset (ms)', fontsize=13)
    else:
        plt.xlabel('Time since stim onset (ms)', fontsize=13)
    plt.ylabel('Classifier weights', fontsize=11)
    
    #plt.title('SVM trained on non-overlapping %.2f ms windows' %(regressBins*frameLength), fontsize=13)
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1) 
    
    ax = plt.gca()
    #if trialHistAnalysis:
    #    plt.xticks(np.arange(-400,600,200))
    #else:    
    #    plt.xticks(np.arange(0,1400,400))
    makeNicePlots(ax,1,1)
    
    # Plot a dot for significant time points
    ymin, ymax = ax.get_ylim()
    
    pp = pcorrtrace+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
    plt.plot(time_aligned, pp, color='k')
    
    
    
    
    
    ######## Abs Weights
    plt.subplot(h2)
    plt.fill_between(time_aligned, av_wexc_abs - sd_wexc_abs, av_wexc_abs + sd_wexc_abs, alpha=0.5, edgecolor=colors[0], facecolor=colors[0])
    plt.plot(time_aligned, av_wexc_abs, colors[0], label='exc')
    
    plt.fill_between(time_aligned, av_winh_abs - sd_winh_abs, av_winh_abs + sd_winh_abs, alpha=0.5, edgecolor=colors[1], facecolor=colors[1])
    plt.plot(time_aligned, av_winh_abs, colors[1], label='inh')
    
    
    if chAl==1:
        plt.xlabel('Time since choice onset (ms)', fontsize=13)
    else:
        plt.xlabel('Time since stim onset (ms)', fontsize=13)
    plt.ylabel('Abs (classifier weights)', fontsize=11)
    
    #plt.title('SVM trained on non-overlapping %.2f ms windows' %(regressBins*frameLength), fontsize=13)
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1) 
    
    ax = plt.gca()
    #if trialHistAnalysis:
    #    plt.xticks(np.arange(-400,600,200))
    #else:    
    #    plt.xticks(np.arange(0,1400,400))
    makeNicePlots(ax,1,1)
    
    # Plot a dot for significant time points
    ymin, ymax = ax.get_ylim()
    
    pp = pAbscorrtrace+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
    plt.plot(time_aligned, pp, color='k')
    
    
    plt.subplots_adjust(wspace=1, hspace=.8)
    
    
    
    ##%% Save the figure    
    if savefigs:#% Save the figure
        d = os.path.join(svmdir+dnow0)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
     
        if poolDaysOfMice==0:
            fign = os.path.join(d, suffn[0:5]+'timeCourse_w_absw_aveMice_aveDays_aveNs_excVSinh'+nowStr+'.'+fmt[0])
        else:
            fign = os.path.join(d, suffn[0:5]+'timeCourse_w_absw_aveMiceDays_aveNs_excVSinh'+nowStr+'.'+fmt[0])    
        fig.savefig(fign, bbox_inches='tight')
    
        

        
    
    #%% Plot hist of w for each frame 
    ####################################################################################    
    ####################################################################################
    ####################################################################################    
    
    #%% plot hist of w for each frame  
    
    # Above you took average of weights across neurons and sessions and mice; 
    # Here we take each neuron's weight (without any averaging) and look into the histograms of weights.   
    
    ####################################################################################    
    ####################################################################################
    ####################################################################################
    
    
    #%% Get values of w (for all neurons) at different time points from the aligned traces; (comapre w _all2 to make sure eveything is fine)
    
    # winhAve_all_allMice_eachFr_aligned_allN is same as winhAve_all_allMice (it includes all neurons' weights at all frames for all days and mice)
    # except it is aligned on the common eventI.
    
    ##### all neurons' weights    --->    Align traces of all days on the common eventI
    winhAve_all_allMice_eachFr_aligned_allN = [] # each mouse is frames x sessions
    wexcAve_all_allMice_eachFr_aligned_allN = []
    
    for im in range(len(mice)):
        numDays = numDaysAll[im]
        
        corr_hr_lr = corr_hr_lr_allMice[im]
        mn_corr = np.min(corr_hr_lr,axis=1)
        
        winhAve_all_allMice_eachFr_aligned = [] # each day is neurons x alignedFrames, aligned on common eventI (equals nPreMin)
        wexcAve_all_allMice_eachFr_aligned = [] 
        
        for iday in range(numDays):
            if mn_corr[iday] >= thTrained:
                winhAve_all_allMice_eachFr_aligned.append(winhAve_all_allMice[im][iday][:, eventI_ds_allDays_allMice[im][iday] - nPreMin  :  eventI_ds_allDays_allMice[im][iday] + nPostMin + 1])
                wexcAve_all_allMice_eachFr_aligned.append(wexcAve_all_allMice[im][iday][:, eventI_ds_allDays_allMice[im][iday] - nPreMin  :  eventI_ds_allDays_allMice[im][iday] + nPostMin + 1])
    
        winhAve_all_allMice_eachFr_aligned_allN.append(winhAve_all_allMice_eachFr_aligned)
        wexcAve_all_allMice_eachFr_aligned_allN.append(wexcAve_all_allMice_eachFr_aligned)    
    
        
    
        
    # Get values of winhAve_all_allMice_eachFr_aligned_allN for each frame    
    
    # np.shape(winhAve_allFrs_allMice): mice x alignedFrames
    # np.shape(winhAve_allFrs_allMice[0]): frames x sessions
    # np.shape(winhAve_allFrs_allMice[0][0]): sessions
    # np.shape(winhAve_allFrs_allMice[0][0][0]): neurons
    winhAve_allFrs_allMice = [] 
    wexcAve_allFrs_allMice = [] 
    for im in range(len(mice)): 
        corr_hr_lr = corr_hr_lr_allMice[im]
        mn_corr = np.min(corr_hr_lr,axis=1)
        
        winhAve_allFrs = []
        wexcAve_allFrs = []
        for fr in range(len(time_aligned)):
    #        nPreMin+fr2an
            winhAve_allFrs.append(np.array([winhAve_all_allMice_eachFr_aligned_allN[im][iday][:,fr] for iday in range(sum(mn_corr>=thTrained))]))
            wexcAve_allFrs.append(np.array([wexcAve_all_allMice_eachFr_aligned_allN[im][iday][:,fr] for iday in range(sum(mn_corr>=thTrained))]))
        
        winhAve_allFrs_allMice.append(winhAve_allFrs)
        wexcAve_allFrs_allMice.append(wexcAve_allFrs)    
    
        
        
    # get values at time point fr2an    
    # inha is exactly like wexcAve_all2_allMice when fr2an = time2an
    fr2an = 0
    inha = []
    exca = []
    for im in range(len(mice)):    
        corr_hr_lr = corr_hr_lr_allMice[im]
        mn_corr = np.min(corr_hr_lr,axis=1)
#        numDays = numDaysAll[im]
        inha.append(np.array([winhAve_allFrs_allMice[im][nPreMin+fr2an][iday] for iday in range(sum(mn_corr>=thTrained))]))
        exca.append(np.array([wexcAve_allFrs_allMice[im][nPreMin+fr2an][iday] for iday in range(sum(mn_corr>=thTrained))]))
        
    
    
    ############################################################################################
    ############################################################################################
        
    #%% Plot fraction non-zero weights
    
    eps = 1e-10
    
    winh_fractnon0 = [] # daysOfAllMice x alignedFrames
    wexc_fractnon0 = []
    for im in range(len(mice)):
        corr_hr_lr = corr_hr_lr_allMice[im]
        mn_corr = np.min(corr_hr_lr,axis=1)
    #    inh0 = []
    #    exc0 = []
        for iday in range(sum(mn_corr>=thTrained)):
            if mn_corr[iday] >= thTrained:
        #        inh0.append(np.mean(abs(winhAve_all_allMice_eachFr_aligned_allN[im][iday]) > eps, axis=0)) # nFrames # fraction non0 weights for each frame
        #        exc0.append(np.mean(abs(wexcAve_all_allMice_eachFr_aligned_allN[im][iday]) > eps, axis=0)) 
                winh_fractnon0.append(np.mean(abs(winhAve_all_allMice_eachFr_aligned_allN[im][iday]) > eps, axis=0)) # nFrames # fraction non0 weights for each frame
                wexc_fractnon0.append(np.mean(abs(wexcAve_all_allMice_eachFr_aligned_allN[im][iday]) > eps, axis=0)) 
        #    winh_fractnon0.append(inh0)
        #    wexc_fractnon0.append(exc0)    
        
    
    ##### PLOT
    a = winh_fractnon0
    b = wexc_fractnon0
        
    av_winh = np.mean(a, axis=0)
    sd_winh = np.std(a, axis=0)/ np.shape(a)[0]
    av_wexc = np.mean(b, axis=0)
    sd_wexc = np.std(b, axis=0)/ np.shape(b)[0]
    
    _,pcorrtrace = stats.ttest_ind(a, b) # p value of class accuracy being different from 50
    
           
    ##%% Plot the average of aligned traces across all days
    
    fig = plt.figure(figsize=(3,4))
    gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
    h1 = gs[0,0:3]
    h2 = gs[1,0:3]
    
    
    plt.subplot(h1)
    plt.fill_between(time_aligned, av_wexc - sd_wexc, av_wexc + sd_wexc, alpha=0.5, edgecolor=colors[0], facecolor=colors[0])
    plt.plot(time_aligned, av_wexc, colors[0], label='exc')
    
    plt.fill_between(time_aligned, av_winh - sd_winh, av_winh + sd_winh, alpha=0.5, edgecolor=colors[1], facecolor=colors[1])
    plt.plot(time_aligned, av_winh, colors[1], label='inh')
    
    
    if chAl==1:
        plt.xlabel('Time since choice onset (ms)', fontsize=13)
    else:
        plt.xlabel('Time since stim onset (ms)', fontsize=13)
    plt.ylabel('Classifier weights', fontsize=11)
    
    #plt.title('SVM trained on non-overlapping %.2f ms windows' %(regressBins*frameLength), fontsize=13)
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1) 
    
    ax = plt.gca()
    #if trialHistAnalysis:
    #    plt.xticks(np.arange(-400,600,200))
    #else:    
    #    plt.xticks(np.arange(0,1400,400))
    makeNicePlots(ax,1,1)
    
    # Plot a dot for significant time points
    ymin, ymax = ax.get_ylim()
    
    pp = pcorrtrace+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
    plt.plot(time_aligned, pp, color='k')
    
    
    ##%% Save the figure    
    if savefigs:#% Save the figure
        d = os.path.join(svmdir+dnow0)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
     
        fign = os.path.join(d, suffn[0:5]+'fractNon0w_timeCourse_aveMiceDays_aveNs_excVSinh'+nowStr+'.'+fmt[0])    
        fig.savefig(fign, bbox_inches='tight')
    


#######################################################        
#######################################################
    #%% Plot hist of w and abs(w) for exc, inh for each time point relative to eventI (on the aligned trace)
    # (pooled across all sessions and all mice and all neurons)
    
    for fr in range(len(time_aligned)):
        inha = []
        exca = []
        for im in range(len(mice)):    
            corr_hr_lr = corr_hr_lr_allMice[im]
            mn_corr = np.min(corr_hr_lr,axis=1)            
#            numDays = numDaysAll[im]
            inha.append(np.array([winhAve_allFrs_allMice[im][fr][iday] for iday in range(sum(mn_corr>=thTrained))]))
            exca.append(np.array([wexcAve_allFrs_allMice[im][fr][iday] for iday in range(sum(mn_corr>=thTrained))]))
            
            
        ### PLOTS    
        ########################################
        ###############%% w  ###############
        ########################################
        ### hist and P val for all days pooled (exc vs inh)
        lab = 'w'
        binEvery = .001 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
        
        aa = np.concatenate((exca)) #np.reshape(wexc,(-1,)) 
        bb = np.concatenate((inha)) #np.reshape(winh,(-1,))
        a = np.concatenate((aa))
        b = np.concatenate((bb))
        
        print np.sum(abs(a)<=eps), ' exc ws < eps'
        print np.sum(abs(b)<=eps), ' inh ws < eps'
        print [len(a), len(b)]
        _, p = stats.ttest_ind(a, b, nan_policy='omit')
        
        fig = plt.figure(figsize=(5,5))    
        gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
        h1 = gs[0,0:2]
        h2 = gs[0,2:3]
        ax1,_ = histerrbar(a,b,binEvery,p,colors)
        #plt.xlabel(lab)
        ax1.set_xlim([-.05, .05])
        #ax1.set_title('num exc = %d; inh = %d' %(len(a), len(b)))
        ax1.set_title(str(fr-nPreMin))
        #fig.savefig('test.jpg')
        
        
        
        
        ##%%%
        lab = 'abs w'
        binEvery = .002 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
        
        a = np.concatenate((exca)) #np.reshape(wexc,(-1,)) 
        b = np.concatenate((inha)) #np.reshape(winh,(-1,))
        a = np.concatenate(abs(a))
        b = np.concatenate(abs(b))
        
        
        print np.sum(abs(a)<=eps), ' exc ws < eps'
        print np.sum(abs(b)<=eps), ' inh ws < eps'
        print [len(a), len(b)]
        _, p = stats.ttest_ind(a, b, nan_policy='omit')
        
        #plt.figure(figsize=(5,5))    
        gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
        h1 = gs[1,0:2]
        h2 = gs[1,2:3]
        ax1,_ = histerrbar(a,b,binEvery,p,colors)
        #plt.xlabel(lab)
        ax1.set_xlim([0, .07])
        ax1.set_title('num exc = %d; inh = %d' %(len(a), len(b)))
    
    
        
    
    
        
        

    