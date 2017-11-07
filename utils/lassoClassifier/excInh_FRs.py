# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:52:31 2017

@author: farznaj
"""

#%% Change the following vars:

mice = 'fni16', 'fni17', 'fni18', 'fni19'
normX = 1 # after downsampling set max peak to 1. Makes sense to do, since traces before downsampling are normalized to max peak... so we should do it again after downsampling too.
zscoreX = 0 # z-score X (just how SVM was performed) before computing FRs
outcome2ana = 'corr' # '', corr', 'incorr' # trials to use for SVM training (all, correct or incorrect trials) # outcome2ana will be used if trialHistAnalysis is 0. When it is 1, by default we are analyzing past correct trials. If you want to change that, set it in the matlab code.        
thTrained = 10 # if days have fewer than this number of hr and lr trials, they would be excluded from plots

savefigs = 0
doPlots = 0 # plot for each mouse hists of FR at timebin -1 

if normX:
    nmd = 'norm2max_'
else:
    nmd = ''
    
# related to sum plot of all mice (noZmotionDays will be rewritten)
noZmotionDays_allMPlots = 1 # remove the last 4 days of fni18 (ie keep only the 1st 3 days)
# I prefer to exclude the zMotion days from fni18 bc inh is very high for them and it makes the se for this mouse crazy!
# fni18: days[4] is the one with hugly high inh FR which makes se across days bad for this mouse!

from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')

if noZmotionDays_allMPlots:
    fni18n = '(noZmotionDays)'
    im = mice.index('fni18')
    # below is the reall noZoMotion days ... 
    #days2keep = [0,1,2,3] # days to keep
    # 151215_1-2', (days[4]) is the one with hugly high inh FR which makes se across days bad for this mouse!
#    days2keep = [0,1,2,3,5,6]     
    dpm0 = nmd+'_'.join(mice[0:im+1])+fni18n+'_'+'_'.join(mice[im+1:])+'_'+nowStr
else:
    dpm0 = nmd+'_'.join(mice)+'_'+nowStr


        

lastTimeBinMissed = 0 #I think it should be 0 bc the exc,inh svm data was run that way. # if 0, things were ran fine; if 1: by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
strength2ana = 'all' # 'all', easy', 'medium', 'hard' % What stim strength to use for training?

trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 

execfile("defFuns.py")

chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]
softNorm = 1 # if 1, no neurons will be excluded, bc we do soft normalization of FRs, so non-active neurons wont be problematic. if softNorm = 0, NsExcluded will be found
#neuronType = 2
thAct = 5e-4 #5e-4; # 1e-5 # neurons whose average activity during ep is less than thAct will be called non-active and will be excluded.

import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.

dnow0 = '/excInh_FRs'

if zscoreX==1:
    dp = '_normed_'
else:
    dp = '_'    

    
#%%
Y_svm_allDays_allMice = []
Xinh_allDays_allMice = []
XallExc_allDays_allMice = []
eventI_ds_allDays_allMice = []
eventI_allDays_allMice = []
dpmAllm = []
numDaysAll = np.full(len(mice), np.nan).astype(int)     
    
#%%
for im in range(len(mice)):
        
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
    numDaysAll[im] = len(days)
        
    dnow = os.path.join(dnow0,mousename)   
    dpm = nmd+'days_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr   
    dpmAllm.append(dpm)

    
    #%% Loop over days
    
    Y_svm_allDays = []
    Xinh_allDays = []
    XallExc_allDays = []   
    eventI_ds_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
    eventI_allDays = np.full((len(days)), np.nan)
#    corr_hr_lr = np.full((len(days),2), np.nan) # number of hr, lr correct trials for each day
    
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
            
        
        #%% Load inhibitRois
        
        Data = scio.loadmat(moreName, variable_names=['inhibitRois_pix'])
        inhibitRois = Data.pop('inhibitRois_pix')[0,:]    
        print '%d inhibitory, %d excitatory; %d unsure class' %(np.sum(inhibitRois==1), np.sum(inhibitRois==0), np.sum(np.isnan(inhibitRois)))
        
            
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
            
            time_trace = time_aligned_1stSide    
        
        print 'frs x units x trials', X_svm.shape    
        
        ##%%
#        corr_hr = sum(np.logical_and(allResp_HR_LR==1 , ~trsExcluded)).astype(int)
#        corr_lr = sum(np.logical_and(allResp_HR_LR==0 , ~trsExcluded)).astype(int)           
        
        
        #%% Set Y for training SVM   
        
        Y_svm = choiceVec0[~trsExcluded]
        #len(Y_svm)
        # Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
        hr_trs = (Y_svm==1)
        lr_trs = (Y_svm==0)
            
        print '%d HR trials; %d LR trials' %(sum(hr_trs), sum(lr_trs))
        
        print 'Y size: ', Y_svm.shape
        
        ## print number of hr,lr trials after excluding trials
        if outcome2ana == 'corr':
            print '\tcorrect trials: %d HR; %d LR' %((Y_svm==1).sum(), (Y_svm==0).sum())
        elif outcome2ana == 'incorr':
            print '\tincorrect trials: %d HR; %d LR' %((Y_svm==1).sum(), (Y_svm==0).sum())
        else:
            print '\tall trials: %d HR; %d LR' %((Y_svm==1).sum(), (Y_svm==0).sum())
    
    
        #%% I think we need at the very least 3 trials of each class to train SVM. So exit the analysis if this condition is not met!
        
    #    if min((Y_svm==1).sum(), (Y_svm==0).sum()) < 3:
    #        sys.exit('Too few trials to do SVM training! HR=%d, LR=%d' %((Y_svm==1).sum(), (Y_svm==0).sum()))
        
           
        #%% Downsample X: average across multiple times (downsampling, not a moving average. we only average every regressBins points.)
        
        X_svm_o = X_svm
        time_trace_o = time_trace
        
        
        #%%
        X_svm, time_trace, eventI_ds = downsampXsvmTime(X_svm_o, time_trace_o, eventI, regressBins, lastTimeBinMissed)
        
        
        #%% After downsampling normalize X_svm so each neuron's max is at 1 (you do this in matlab for S traces before downsampling... so it makes sense to again normalize the traces After downsampling so max peak is at 1)                
        
        if normX:
            m = np.max(X_svm,axis=(0,2))   # find the max of each neurons across all trials and frames # max(X_svm.flatten())            
            X_svm = np.transpose(np.transpose(X_svm,(0,2,1)) / m, (0,2,1))
        
        
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
        inhRois = inhibitRois[~NsExcluded]
        print 'Number: inhibit = %d, excit = %d, unsure = %d' %(np.sum(inhRois==1), np.sum(inhRois==0), np.sum(np.isnan(inhRois)))
        #    print 'Fraction: inhibit = %.2f, excit = %.2f, unsure = %.2f' %(fractInh, fractExc, fractUn)
        
            
        #%%    
    #    numDataPoints = X_svm.shape[2] 
    #    print '# data points = %d' %numDataPoints
        # numTrials = (~trsExcluded).sum()
        # numNeurons = (~NsExcluded).sum()
        numTrials, numNeurons = X_svm.shape[2], X_svm.shape[1]
        print '%d trials; %d neurons' %(numTrials, numNeurons)
        # print ' The data has %d frames recorded from %d neurons at %d trials' %Xt.shape            
           
            
        #%% Center and normalize X: feature normalization and scaling: to remove effects related to scaling and bias of each neuron, we need to zscore data (i.e., make data mean 0 and variance 1 for each neuron) 
        
        if zscoreX:
            #% Keep a copy of X_svm before normalization
            X_svm00 = X_svm + 0        
            
            ##%%
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
            X_svm_N = np.full(np.shape(X_svm), np.nan)
            meanX_fr = []
            stdX_fr = []
            for ifr in range(np.shape(X_svm)[0]):
                m = np.mean(X_svm[ifr,:,:], axis=1)
                s = np.std(X_svm[ifr,:,:], axis=1)   
                meanX_fr.append(m) # frs x neurons
                stdX_fr.append(s)       
                
                if softNorm==1: # soft normalziation : neurons with sd<thAct wont have too large values after normalization
                    s = s+thAct     
            
                X_svm_N[ifr,:,:] = ((X_svm[ifr,:,:].T - m) / s).T
            
            meanX_fr = np.array(meanX_fr) # frames x neurons
            stdX_fr = np.array(stdX_fr) # frames x neurons
            
            
            X_svm = X_svm_N
        
        
        #%%
        Xinh = X_svm[:, inhRois==1,:]              
        XallExc = X_svm[:, inhRois==0,:]            
    #    Xinh_lr = Xinh[:,:,lr_trs]
    #    Xinh_hr = Xinh[:,:,hr_trs]    
    #    XallExc_lr = XallExc[:,:,lr_trs]
    #    XallExc_hr = XallExc[:,:,hr_trs]    
        # all frames pooled
    #    a = XallExc.flatten()
    #    b = Xinh.flatten()    
        #plt.plot(np.mean(XallExc,axis=(1,2))); plt.plot(np.mean(Xinh,axis=(1,2)))    
    
        ###############%%    
        #%% Keep vars of all days
    
        Y_svm_allDays.append(Y_svm)
        Xinh_allDays.append(Xinh)
        XallExc_allDays.append(XallExc)    
        eventI_ds_allDays[iday] = eventI_ds.astype(int)
        eventI_allDays[iday] = eventI #.astype(int)
        
        
        #%%
        #############################################
        if doPlots: 
            def plotfr(a,b,hi,tlab):
                h1 = gs[hi,0:2]
                h2 = gs[hi,2:3]
            
                mx = np.max(np.concatenate((a,b))) 
                mn = np.min(np.concatenate((a,b)))
                r = mx - mn
                binEvery = r/float(100)
            
                _, p = stats.ttest_ind(a, b, nan_policy='omit')    
                
                ax1, ax2 = histerrbar(h1,h2,a,b,binEvery,p,lab='FR (a.u.)',colors = ['b','r'],ylab='Fraction',lab1='exc',lab2='inh', plotCumsum=1)        
                ax1.set_title('%d exc; %d inh' %(len(a),len(b)))
                
               
            #%% hists at a particular time point
        
            fr = eventI_ds - 1
            
            hr_trs = (Y_svm==1)
            lr_trs = (Y_svm==0)
            
            trss = hr_trs+lr_trs, hr_trs, lr_trs
            lls = 'AllTr','HR','LR'
            tlab = '%dms,AllTr,HR,LR' %(time_trace[fr])    
           
            
            fig = plt.figure(figsize=(5,6))    
            gs = gridspec.GridSpec(3, 4)#, width_ratios=[2, 1]) 
        
            for hi in [0,1,2]: # all trials, hr, lr
        
        #        tlab = '%dms, %s' %(time_trace[fr], lls[hi])
                a = XallExc[fr,:,trss[hi]].flatten()
                b = Xinh[fr,:,trss[hi]].flatten()
                
                plotfr(a,b,hi,tlab)
        
            
            plt.subplots_adjust(wspace=2, hspace=1)
            fig.suptitle(tlab, fontsize=11)
        
        
    #######################################################################################################################################        
    #######################################################################################################################################                
    #%% Keep vars of all mice

    Y_svm_allDays = np.array(Y_svm_allDays)
    Xinh_allDays = np.array(Xinh_allDays)
    XallExc_allDays = np.array(XallExc_allDays)

    Y_svm_allDays_allMice.append(Y_svm_allDays)
    Xinh_allDays_allMice.append(Xinh_allDays)
    XallExc_allDays_allMice.append(XallExc_allDays) 
    eventI_ds_allDays_allMice.append(eventI_ds_allDays)    
    eventI_allDays_allMice.append(eventI_allDays)    

        
Y_svm_allDays_allMice = np.array(Y_svm_allDays_allMice)
Xinh_allDays_allMice = np.array(Xinh_allDays_allMice)
XallExc_allDays_allMice = np.array(XallExc_allDays_allMice)
eventI_ds_allDays_allMice = np.array(eventI_ds_allDays_allMice)
eventI_allDays_allMice = np.array(eventI_allDays_allMice)
eventI_ds_allDays_allMice = np.array([[int(i) for i in eventI_ds_allDays_allMice[im]] for im in range(len(mice))]) # convert eventIs to int
#numDaysAll = numDaysAll.astype(int)


#%% Set number of trials for each day

corr_hr_lr_allMice = []
for im in range(len(mice)):
    
    corr_hr_lr = np.full((len(Y_svm_allDays_allMice[im]),2), np.nan)
    for iday in range(len(Y_svm_allDays_allMice[im])):
        corr_hr_lr[iday,:] = np.array([sum(Y_svm_allDays_allMice[im][iday]==1), sum(Y_svm_allDays_allMice[im][iday]==0)])
    corr_hr_lr_allMice.append(corr_hr_lr)
corr_hr_lr_allMice = np.array(corr_hr_lr_allMice)


#%% Decide what days to analyze: exclude days with too few trials used for training SVM, also exclude incorr from days with too few incorr trials.
# th for min number of trs of each class    

mn_corr_allMice = []

for im in range(len(mice)):
    mn_corr_allMice.append(np.min(corr_hr_lr_allMice[im], axis=1)) # number of trials of each class. 90% of this was used for training, and 10% for testing.  
    
mn_corr_allMice = np.array(mn_corr_allMice)    

if noZmotionDays_allMPlots: # remove the last 3 days of fni18 (ie keep only the 1st 4 days)    
    im = mice.index('fni18') # remove its 4th day
    mn_corr_allMice[im][4] = thTrained-1 # day 151215

numDaysAllGood = np.array([sum(mn_corr_allMice[im]>=thTrained) for im in range(len(mice))])
print numDaysAllGood


#%% set Y_svm only for good days

Y_svm_goodDays_allMice = np.array([Y_svm_allDays_allMice[im][mn_corr_allMice[im] >= thTrained] for im in range(len(mice))])


#%% Align Xinh and XallExc traces of all days for each mouse
# exclude low trial days

Xinh_al_allMice = [] # nMice; #nGoodDays; #nAlighnedFrames x units x trials
XallExc_al_allMice = []
time_al_allMice = []
nPreMin_allMice = []

for im in range(len(mice)):

    ##%% Decide what days to analyze: exclude days with too few trials used for training SVM, also exclude incorr from days with too few incorr trials.
    # th for min number of trs of each class    
    mn_corr = mn_corr_allMice[im] #np.min(corr_hr_lr_allMice[im], axis=1) # number of trials of each class. 90% of this was used for training, and 10% for testing.  
#    print 'num days to be excluded with few svm-trained trs:', sum(mn_corr < thTrained)
#    print np.array(days)[mn_corr < thTrained]

    time_al, nPreMin, nPostMin = set_nprepost(Xinh_allDays_allMice[im], eventI_ds_allDays_allMice[im], mn_corr, thTrained=10, regressBins=3)    
    time_al_allMice.append(time_al)
    nPreMin_allMice.append(nPreMin)

    # Align traces of all days   
    Xinh_al_allMice.append(alTrace(Xinh_allDays_allMice[im], eventI_ds_allDays_allMice[im], nPreMin, nPostMin, mn_corr, thTrained))
    XallExc_al_allMice.append(alTrace(XallExc_allDays_allMice[im], eventI_ds_allDays_allMice[im], nPreMin, nPostMin, mn_corr, thTrained))
    
    
Xinh_al_allMice = np.array(Xinh_al_allMice)
XallExc_al_allMice = np.array(XallExc_al_allMice)
time_al_allMice = np.array(time_al_allMice)
nPreMin_allMice = np.array(nPreMin_allMice)


#%% pool traces for all neurons and all trials for each day, each frame (each mouse)

def poolAllNTr(XallExc_al_allMice, Xinh_al_allMice, trtype=0, avtrs=0):
    
    exc_allFr_allMice = [] # nMice; each mouse: alignedFrames x nGoodDays; each element: pooled neurons and trials
    inh_allFr_allMice = []
    
    for im in range(len(mice)):
        
        ndays = len(XallExc_al_allMice[im])    
        nFrs = XallExc_al_allMice[im][0].shape[0]    
            
        exc_allFr = [] # frames x days; each element: pooled neurons and trials
        inh_allFr = []
        for fr in range(nFrs):
            # For each day pool all neurons and trials at time point fr
            a = []; b = []    
            for iday in range(ndays):       
    
                Y_svm = Y_svm_goodDays_allMice[im][iday]       
                hr_trs = (Y_svm==1)
                lr_trs = (Y_svm==0)
                trss = hr_trs+lr_trs, hr_trs, lr_trs
                
                XallExc = XallExc_al_allMice[im][iday]
                Xinh = Xinh_al_allMice[im][iday]
                
                a0 = XallExc[fr,:,trss[trtype]]#.flatten() #trs x units
                b0 = Xinh[fr,:,trss[trtype]]#.flatten() #trs x units       
                
                if avtrs: # for each neuron average across trials.
                    a0 = np.mean(a0,axis=0)
                    b0 = np.mean(b0,axis=0)
                a0 = a0.flatten() # pool all neurons and trials                   
                b0 = b0.flatten()
                
                a.append(a0)
                b.append(b0)
            
            exc_allFr.append(a)
            inh_allFr.append(b)
            
        exc_allFr = np.array(exc_allFr)
        inh_allFr = np.array(inh_allFr)
        
        exc_allFr_allMice.append(exc_allFr)
        inh_allFr_allMice.append(inh_allFr)
    
    exc_allFr_allMice = np.array(exc_allFr_allMice)
    inh_allFr_allMice = np.array(inh_allFr_allMice)

    return exc_allFr_allMice, inh_allFr_allMice


#%% For each frame average FR across all neurons and trials

def aveNsTrs(exc_allFr): 

    nF = np.shape(exc_allFr)[0] #nFrs
    nD = np.shape(exc_allFr)[1] #nDays
    
    # For each frame average across all neurons and trials
    avFR = np.full((nF,nD), np.nan) # frames x days
    sdFR = np.full((nF,nD), np.nan)
    seFR = np.full((nF,nD), np.nan)
    for fr in range(nF):
        avFR[fr,:] = np.array([np.mean(exc_allFr[fr,iday]) for iday in range(nD)])
        sdFR[fr,:] = np.array([np.std(exc_allFr[fr,iday]) for iday in range(nD)])
        seFR[fr,:] = np.array([np.std(exc_allFr[fr,iday]) / np.sqrt(len(exc_allFr[fr,iday])) for iday in range(nD)])       
   
    return avFR #, sdFR, seFR
       

#%% plot timecourse of ave exc vs ave inh FR (ave across days; each day was averaged across all neurons and trials)

def plotTimecourse(avFR, time_al=[], col='b', lab='', linstyl='', alph=''): 
    # average and se across days         
    av = np.mean(avFR,axis=1) 
    sd = np.std(avFR,axis=1)/np.sqrt(avFR.shape[1]) 
    
    plt.errorbar(time_al, av, sd, color=col, label=lab, linestyle=linstyl, alpha=alph)
        

#%%##%% Pool traces for all neurons and all trials for each day, each frame (each mouse)
#######################################################################################################################################                
#######################################################################################################################################                

trtypeFinal = 0 # trial types to plot: all, HR, LR
avtrs = 1 # for each neuron average across trials, otherwise pool neural activities across all trials

exc_allFr_allMice, inh_allFr_allMice = poolAllNTr(XallExc_al_allMice, Xinh_al_allMice, trtypeFinal, avtrs) # nNice; each element: dayAlignedFrames x nGoodDays; each element: pooled neurons and trials

#if noZmotionDays_allMPlots: # remove the last 3 days of fni18 (ie keep only the 1st 4 days)    
#    im = mice.index('fni18')
#    exc_allFr_allMice[im] = exc_allFr_allMice[im][:, days2keep]    
#    inh_allFr_allMice[im] = inh_allFr_allMice[im][:, days2keep]    

    
#%% Do for each mouse: For each frame average FR across all neurons and trials; 

avFRexc_allMice = []
avFRinh_allMice = []
for im in range(len(mice)):

    exc_allFr = exc_allFr_allMice[im] # dayAlignedFrames x nGoodDays; each element: pooled neurons and trials
    inh_allFr = inh_allFr_allMice[im]    
    time_al = time_al_allMice[im]
        
    avFRexc_allMice.append(aveNsTrs(exc_allFr)) # avFRexc: alignedFrames x days (each element was averaged across all neurons and trials))
    avFRinh_allMice.append(aveNsTrs(inh_allFr))    
#    p = np.full((avFRexc.shape[0]), np.nan)
#    for fr in range(avFRexc.shape[0]):
#        _,p[fr] = stats.ttest_ind(avFRexc[fr], avFRinh[fr], nan_policy='omit')    
    
avFRexc_allMice = np.array(avFRexc_allMice)  # nMice; each element: # alignedFrames x nGoodDays  
avFRinh_allMice = np.array(avFRinh_allMice)    


############################################################################
#%% Align neuron&trial-averaged traces across mice

time_al_final, nPreMin_final, nPostMin_final = set_nprepost(avFRexc_allMice, nPreMin_allMice)

# input: avFRexc_allMice: nMice; each mouse: alignedFrames (across days for that mouse) x nGoodDays
# output: avFRexc_allMice_al nMice; each mouse: alignedFrames (across mice) x nGoodDays
avFRexc_allMice_al = alTrace(avFRexc_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)
avFRinh_allMice_al = alTrace(avFRinh_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)

#if noZmotionDays_allMPlots: # remove the last 3 days of fni18 (ie keep only the 1st 4 days)    
#    im = mice.index('fni18')
#    avFRexc_allMice_al[im] = avFRexc_allMice_al[im][:, days2keep]
#    avFRinh_allMice_al[im] = avFRinh_allMice_al[im][:, days2keep]
    
    
#%% Align individual neural and trial traces across mice    

exc_allFr_allMice_al = alTrace(exc_allFr_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)  # nNice; each element: mouseAlignedFrames x nGoodDays; each element: pooled neurons and trials
inh_allFr_allMice_al = alTrace(inh_allFr_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)



#%% 
#########       Plots of individual mice    ########################################################################################        
#######################################################################################################################################        
#######################################################################################################################################        
#######################################################################################################################################        

#%% Plot for each mouse timecourse of session-averaged FR for exc and inh (ave across days; each day was averaged across all neurons and trials)

avtrs = 1 # for each neuron average across trials, otherwise pool neural activities across all trials
trtyps2an = [trtypeFinal] #[0,1,2] #[1,2] #[0,1,2] : trial types to plot: all, HR, LR # NOTE:
# if you want to try trTypes other than trtypeFinal, you must run :
#exc_allFr_allMice, inh_allFr_allMice = poolAllNTr(XallExc_al_allMice, Xinh_al_allMice, trtype, avtrs)
#again, also after this code ... if for the rest of the codes below you want to plot trtypeFinal

linstyls = '-','-','-'
alphas = 1,.5,.2  # all, HR, LR

for im in range(len(mice)):

    mousename = mice[im]
#    execfile("svm_plots_setVars_n.py")      
    dnow = os.path.join(dnow0, mousename)       

    plt.figure(figsize=(3,2))        

    for trtype in trtyps2an: # trial types to plot: 0:all trs; 1: hr trs; 2: lr trs
        '''
        exc_allFr_allMice, inh_allFr_allMice = poolAllNTr(XallExc_al_allMice, Xinh_al_allMice, trtype, avtrs)
    
        exc_allFr = exc_allFr_allMice[im] # alignedFrames x days; each element: pooled neurons and trials
        inh_allFr = inh_allFr_allMice[im]    
        time_al = time_al_allMice[im]
        '''    
        plotTimecourse(avFRexc_allMice[im], time_al_allMice[im], col='b', lab='exc', linstyl=linstyls[trtype], alph=alphas[trtype]) # avFRexc: alignedFrames x days (each element was averaged across all neurons and trials))
        plotTimecourse(avFRinh_allMice[im], time_al_allMice[im], col='r', lab='inh', linstyl=linstyls[trtype], alph=alphas[trtype])
        
        p = np.full((avFRexc_allMice[im].shape[0]), np.nan)
        for fr in range(avFRexc_allMice[im].shape[0]):
            _,p[fr] = stats.ttest_ind(avFRexc_allMice[im][fr,:], avFRinh_allMice[im][fr,:], nan_policy='omit')    
        
        yl = plt.gca().get_ylim()
        pp = p
        pp[p>.05] = np.nan
        pp[p<=.05] = yl[1]
        plt.plot(time_al_allMice[im], pp, marker='*',color='r', markeredgecolor='r', linestyle='', markersize=3)
        
    plt.ylabel('FR (a.u.)')        
    plt.xlabel('Time relative to choice onset')
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)    
    makeNicePlots(plt.gca(),1,0)        
    
    #% Save the figure           
    if savefigs:
        if chAl:
            cha = 'chAl_'
            
        d = os.path.join(svmdir+dnow)        
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
        
        fign = os.path.join(d, suffn[0:5]+cha+'FR_timeCourse_aveDays_allTrsNeursPooled'+dp+dpmAllm[im]+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight') 
        
        
#%% Plot time course for each day ... do it for each mouse (only good days)

for im in range(len(mice)):
    mousename = mice[im]
    dnow = os.path.join(dnow0, mousename)        
    execfile("svm_plots_setVars_n.py")          
    daysGood = np.array(days)[mn_corr_allMice[im]>=thTrained]
    print 'Number of good days: ', len(daysGood)

    # only good days
    nDays = len(daysGood) #np.shape(exc_allFr_allMice[im][0])[0] # len(days)
    
    plt.figure(figsize=(2.5,2*nDays))   
        
    for iday in range(nDays):     
        # exc   
        a = np.array(exc_allFr_allMice[im][:,iday]) 
        ave = [a[fr].mean() for fr in range(len(a))] # same as avFRexc_allMice[im][:,iday]
        see = [a[fr].std()/np.sqrt(a[fr].shape[0]) for fr in range(len(a))]
        
        # inh
        a = np.array(inh_allFr_allMice[im][:,iday])
        avi = [a[fr].mean() for fr in range(len(a))]
        sei = [a[fr].std()/np.sqrt(a[fr].shape[0]) for fr in range(len(a))]
        
        plt.subplot(nDays,1,iday+1)
        plt.errorbar(time_al_allMice[im], ave, see, color='b', label='exc')
        plt.errorbar(time_al_allMice[im], avi, sei, color='r', label='inh')
        plt.title(daysGood[iday][0:6])
        if iday==0:
            plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)   
        ax = plt.gca()
        makeNicePlots(ax,1,1)
    
    plt.subplots_adjust(wspace=2, hspace=1)
    
    #% Save the figure           
    if savefigs:
        if chAl:
            cha = 'chAl_'
            
        d = os.path.join(svmdir+dnow)        
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
        
        fign = os.path.join(d, suffn[0:5]+cha+'FR_timeCourse_eachDay_allTrsNeursPooled'+dp+dpmAllm[im]+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight') 
    
    
#%% Plot ave+/-se FRs for each day (x axis: days); do it for each time bin separately

colors = ['b','r']; lab1='exc'; lab2='inh'; lab='FR (a.u.)'
time2an = -1

for im in range(len(mice)):

    mousename = mice[im]
#    execfile("svm_plots_setVars_n.py")      
    dnow = os.path.join(dnow0, mousename)       

    exc_allFr = exc_allFr_allMice[im] # alignedFrames x nGoodDays; each element: pooled neurons and trials
    inh_allFr = inh_allFr_allMice[im]    
    
    nFrs = exc_allFr.shape[0]
    frBef = [nPreMin_allMice[im] + time2an]
    frs2p = range(nFrs)    # plot all frames
#    frs2p = frBef       # only plot the time bin before the choice
    
    plt.figure(figsize=(7,5*len(frs2p)))    
    gs = gridspec.GridSpec(2*len(frs2p), 4)#, width_ratios=[2, 1])         

    cnt = -1
    for fr in frs2p:
        cnt=cnt+1
        
        a = exc_allFr[fr]
        b = inh_allFr[fr]    
        
        p = np.full((len(a)), np.nan)
        for iday in range(len(p)):        
            _, p[iday] = stats.ttest_ind(a[iday], b[iday], nan_policy='omit')    
        
        
        ####%% Plot average and se of FRs (across pooled neurons and trials) for each day       
    #    plt.figure(figsize=(7,5))    
    #    gs = gridspec.GridSpec(2, 4)#, width_ratios=[2, 1])         
        h1 = gs[cnt,0:2]
        h2 = gs[cnt,2:3]
        
        ax1,ax2 = errbarAllDays(a,b,p,gs, colors, lab1, lab2, lab, h1, h2)
        ax1.set_title(('%.0f ms' %time_al_allMice[im][fr]))
        
    plt.subplots_adjust(wspace=1, hspace=.7)    
    
    
    #% Save the figure           
    if savefigs:
        if chAl:
            cha = 'chAl_'
            
        d = os.path.join(svmdir+dnow)        
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
        
        fign = os.path.join(d, suffn[0:5]+cha+'FR_allFrs_eachDay_allTrsNeursPooled'+dp+dpmAllm[im]+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight')        


#######################################################################################################################################               
#######################################################################################################################################                
#%% Summary plots of all mice    
#######################################################################################################################################        
#######################################################################################################################################        
             
#%% Plot time course of FRs: take average across days for each mouse; then plot ave+/-se across mice

def avMice(avFRexc_allMice_al):
    # average across days for each mouse
    avD_exc = np.array([np.mean(avFRexc_allMice_al[im],axis=1) for im in range(len(mice))]) # nMice x alignedFrames (across mice)
    # average and se across mice
    av_exc = np.mean(avD_exc,axis=0)
    se_exc = np.std(avD_exc,axis=0)/np.sqrt(avD_exc.shape[0]) 
    
    return avD_exc, av_exc, se_exc


#####################
def pltSumMiceAvMice(avFRexc_allMice_al, avFRinh_allMice_al, lab1, lab2, alph=1, dpp='_allTrs'):
    avD_exc, av_exc, se_exc = avMice(avFRexc_allMice_al)
    avD_inh, av_inh, se_inh = avMice(avFRinh_allMice_al)
    
    _,p = stats.ttest_ind(avD_exc, avD_inh, nan_policy='omit')    
    
#    plt.figure(figsize=(3,2))   
    
    plt.errorbar(time_al_final, av_exc, se_exc, color='b', label=lab1, alpha=alph)
    plt.errorbar(time_al_final, av_inh, se_inh, color='r', label=lab2, alpha=alph)
    
    yl = plt.gca().get_ylim()
    pp = p
    pp[p>.05] = np.nan
    pp[p<=.05] = yl[1]
    plt.plot(time_al_final, pp, marker='*',color='r', markeredgecolor='r', linestyle='', markersize=3)
    
    plt.ylabel('FR (a.u.)')        
    plt.xlabel('Time relative to choice onset')
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)    
    makeNicePlots(plt.gca(),1,0)        
    
    #% Save the figure           
    if savefigs:
        if chAl:
            cha = 'chAl_'
            
        d = os.path.join(svmdir+dnow0)        
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
        
        fign = os.path.join(d, suffn[0:5]+cha+'FR_timeCourse_aveMice_aveDays_allTrsNeursPooled'+dpp+dp+dpm0+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight') 

       
#%% Plot time course of FRs: Pool all days of all mice, then make an average
              
def pltSumMiceAvAllDays(avFRexc_allMice_al, avFRinh_allMice_al, lab1, lab2, alph=1, dpp='_allTrs'):              
    av_allD_exc = np.hstack(avFRexc_allMice_al) # nAlignedFrames (across mice) x nAlldays(all mice)
    av_allD_inh = np.hstack(avFRinh_allMice_al)
    
    _,p = stats.ttest_ind(av_allD_exc.T, av_allD_inh.T, nan_policy='omit')    
        
    # average across pooled days    
    ave = np.mean(av_allD_exc, axis=1)
    see = np.std(av_allD_exc, axis=1)/np.sqrt(av_allD_exc.shape[1])
    avi = np.mean(av_allD_inh, axis=1)
    sei = np.std(av_allD_inh, axis=1)/np.sqrt(av_allD_inh.shape[1])    
        
#    plt.figure(figsize=(3,2))   
    
    plt.errorbar(time_al_final, ave, see, color='b', label=lab1, alpha=alph)
    plt.errorbar(time_al_final, avi, sei, color='r', label=lab2, alpha=alph)
    
    
    yl = plt.gca().get_ylim()
    pp = p
    pp[p>.05] = np.nan
    pp[p<=.05] = yl[1]
    plt.plot(time_al_final, pp, marker='*',color='r', markeredgecolor='r', linestyle='', markersize=3)
    
    plt.ylabel('FR (a.u.)')        
    plt.xlabel('Time relative to choice onset')
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)    
    makeNicePlots(plt.gca(),1,0)        
    
    #% Save the figure           
    if savefigs:
        if chAl:
            cha = 'chAl_'
            
        d = os.path.join(svmdir+dnow0)        
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
        
        fign = os.path.join(d, suffn[0:5]+cha+'FR_timeCourse_aveMiceDaysPooled_allTrsNeursPooled'+dpp+dp+dpm0+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight') 
    


#%% Plot time course of FRs; summary of mice

# Take average across days for each mouse; then plot ave+/-se across mice
plt.figure(figsize=(3,2))
pltSumMiceAvMice(avFRexc_allMice_al, avFRinh_allMice_al, lab1, lab2, alph=1, dpp='_allTrs')

# Pool all days of all mice, then make an average
plt.figure(figsize=(3,2)) 
pltSumMiceAvAllDays(avFRexc_allMice_al, avFRinh_allMice_al, lab1, lab2, alph=1, dpp='_allTrs')


    
#%% Plot FR of exc and inh at time -1 for each mouse

def avEachMouse(avFRexc_allMice_al, fr):
    # take Neuron&Trial-averaged FRs for each day at time point -1 (avFRexc_allMice_al: FRs are averaged across all neurons and trials)
    timeM1_exc = np.array([avFRexc_allMice_al[im][fr] for im in range(len(mice))]) # nMice; each mouse: nGoodDays
    # compute ave and se across days for each mouse        
    ave_e = np.array([np.mean(timeM1_exc[im]) for im in range(len(mice))])
    se_e = np.array([np.std(timeM1_exc[im])/np.sqrt(len(timeM1_exc[im])) for im in range(len(mice))])
    
    return ave_e, se_e, timeM1_exc
     

#%% Plot FR of exc and inh at time -1 for each mouse

fr = nPreMin_final-1     

ave_e, se_e, timeM1_exc = avEachMouse(avFRexc_allMice_al, fr)     
ave_i, se_i, timeM1_inh = avEachMouse(avFRinh_allMice_al, fr)     


plt.figure(figsize=(2,3))

plt.errorbar(range(len(mice)), ave_e, se_e, color='b', fmt='o')
plt.errorbar(range(len(mice)), ave_i, se_i, color='r', fmt='o')

plt.legend(loc='center left', bbox_to_anchor=(1, .7), numpoints=1)#, frameon=False) 
plt.xlabel('Mice', fontsize=11)
plt.ylabel('FR (a.u.)', fontsize=11)
plt.xlim([-.2,len(mice)-1+.2])
plt.xticks(range(len(mice)),mice)
ax = plt.gca()
makeNicePlots(ax)
     
#% Save the figure           
if savefigs:
    if chAl:
        cha = 'chAl_'
        
    d = os.path.join(svmdir+dnow0)        
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
    
    fign = os.path.join(d, suffn[0:5]+cha+'FR_time-1_eachMouse_aveDays_allTrsNeursPooled'+dp+dpm0+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight') 
     


#%% Set vars to plot hist of FRs in frame -1, all neurons of all trials of all days and all mice pooled

fr = nPreMin_final-1     
                
#exc_allFr_allMice_al[im][fr,:][iday] # Fr of all neurons in time -1 in all trials of day iday
#a = np.concatenate((exc_allFr_allMice_al[im][fr,:]))  # Fr of all neurons in time -1 in all trials of all days

a0 = np.array([np.concatenate((exc_allFr_allMice_al[im][fr,:])) for im in range(len(mice))]) #nMice; each mouse: FR of all neurons in time -1 in all trials of all days
b0 = np.array([np.concatenate((inh_allFr_allMice_al[im][fr,:])) for im in range(len(mice))]) #nMice; each mouse: FR of all neurons in time -1 in all trials of all days

# pool all mice
a = np.concatenate((a0)) # pool all mice (allNeurons, allTrs, allDays)
b = np.concatenate((b0)) # pool all mice (allNeurons, allTrs, allDays)


#%% Plot hist of FRs in frame -1, all neurons of all days and all mice pooled

def plthistfr(a,b,dnow0,dpm0):
    
    mx = np.max(np.concatenate((a,b))) 
    mn = np.min(np.concatenate((a,b)))
    r = mx - mn
    binEvery = r/float(30)    
    
    plt.figure(figsize=(7,4))    #5,3
    gs = gridspec.GridSpec(2, 4)#, width_ratios=[2, 1]) 
    h1 = gs[0,0:2]
    h2 = gs[0,2:3]
    yl0 = 'Fraction'
    plotcumsum = 0
    
    ax1, ax2 = histerrbar(h1,h2,a,b,binEvery,p,lab='FR (a.u.)',colors = ['b','r'], ylab=yl0, lab1='exc',lab2='inh', plotCumsum=plotcumsum)        
    ax1.set_title('%d exc; %d inh' %(len(a),len(b)))
    #ax1.set_xlim([-.0005,.015])            
    plt.subplots_adjust(wspace=1.5)
    makeNicePlots(ax1,1,0)
    
    #% Save the figure           
    if savefigs:
        if chAl:
            cha = 'chAl_'
            
        d = os.path.join(svmdir+dnow0)        
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
        
        fign = os.path.join(d, suffn[0:5]+cha+'FR_time-1_dist_allNsTrsDaysPooled'+dp+dpm0+'.'+fmt[0]) # FR_time-1_dist_allNsTrsDaysMicePooled
        print fign
        plt.savefig(fign, bbox_inches='tight') 
    
    

#%% Plot hist of all mice pooled

_, p = stats.ttest_ind(a, b, nan_policy='omit')    

plthistfr(a,b,dnow0,dpm0)

    
#%% Plot hist for each mouse

for im in range(len(mice)):
    a = a0[im]
    b = b0[im]
    
    _, p = stats.ttest_ind(a, b, nan_policy='omit')    
    
    mousename = mice[im]
    dnow = os.path.join(dnow0, mousename)       
    
    plthistfr(a,b,dnow,dpmAllm[im])
    
    

####################################################################################
#%% Plot ave FR for HR and LR trials separately

def plothrlrtc():
    
    trlab = '_HR','_LR'
    alphas = [1,.8,.4]

    for trtypeFinal in [1,2]: # trial types to plot: all, HR, LR
        
        lab1 = 'exc'+trlab[trtypeFinal-1] 
        lab2 = 'inh'+trlab[trtypeFinal-1]
        
        #%% pool traces for all neurons and all trials for each day, each frame (each mouse)
        
        exc_allFr_allMice, inh_allFr_allMice = poolAllNTr(XallExc_al_allMice, Xinh_al_allMice, trtypeFinal)
        
        
        #%% For each frame average FR across all neurons and trials; 
        
        avFRexc_allMice = []
        avFRinh_allMice = []
        for im in range(len(mice)):
        
            exc_allFr = exc_allFr_allMice[im] # alignedFrames x nGoodDays; each element: pooled neurons and trials
            inh_allFr = inh_allFr_allMice[im]    
#            time_al = time_al_allMice[im]
                
            avFRexc_allMice.append(aveNsTrs(exc_allFr)) # avFRexc: alignedFrames x days (each element was averaged across all neurons and trials))
            avFRinh_allMice.append(aveNsTrs(inh_allFr))    
        #    p = np.full((avFRexc.shape[0]), np.nan)
        #    for fr in range(avFRexc.shape[0]):
        #        _,p[fr] = stats.ttest_ind(avFRexc[fr], avFRinh[fr], nan_policy='omit')    
            
        avFRexc_allMice = np.array(avFRexc_allMice)  # nMice; each element: # alignedFrames x nGoodDays  
        avFRinh_allMice = np.array(avFRinh_allMice)    
        
        
        #%% Align traces among mice
        
        time_al_final, nPreMin_final, nPostMin_final = set_nprepost(avFRexc_allMice, nPreMin_allMice)
        
        # input: avFRexc_allMice: nMice; each mouse: alignedFrames (across days for that mouse) x nGoodDays
        # output: avFRexc_allMice_al nMice; each mouse: alignedFrames (across mice) x nGoodDays
        avFRexc_allMice_al = alTrace(avFRexc_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)
        avFRinh_allMice_al = alTrace(avFRinh_allMice, nPreMin_allMice, nPreMin_final, nPostMin_final)
        
    #    if noZmotionDays_allMPlots: # remove the last 3 days of fni18 (ie keep only the 1st 4 days)    
    #        im = mice.index('fni18')
    #        avFRexc_allMice_al[im] = avFRexc_allMice_al[im][:, days2keep]
    #        avFRinh_allMice_al[im] = avFRinh_allMice_al[im][:, days2keep]
        
        
        #%%
    #    pltSumMiceAvMice(avFRexc_allMice_al, avFRinh_allMice_al, alph=alphas[trtypeFinal])    
        pltSumMiceAvAllDays(avFRexc_allMice_al, avFRinh_allMice_al, lab1, lab2, alph=alphas[trtypeFinal], dpp='_HrLr')


plt.figure(figsize=(3,2))     
plothrlrtc()

    
#%% Plot histograms of exc,inh FRs
    # you can use aligned traces below (now you are using the original traces of each day)
'''
fr = eventI_ds-1
lls = 'AllTr','HR','LR'
tl = ',AllTr,HR,LR'
tlab = '%dms' %(time_trace[fr])+tl    
yl0 = 'Fraction'

for im in range(len(mice)):
    
    mousename = mice[im]
    execfile("svm_plots_setVars_n.py")      
    dnow = os.path.join(dnow0, mousename)   
    ndays = numDaysAll[im]
    
    mn_corr = np.min(corr_hr_lr_allMice[im], axis=1) # number of trials of each class. 90% of this was used for training, and 10% for testing.  
    
    fig = plt.figure(figsize=(6*3, 2.1*ndays))    
    gs = gridspec.GridSpec(ndays, 4*3)#, width_ratios=[2, 1]) 
            
            
    for iday in range(ndays):
        
        if mn_corr[iday] >= thTrained: # exclude days with very low trials
        
            eventI_ds = eventI_ds_allDays_allMice[im][iday]        
            fr = eventI_ds-1
            
            XallExc = XallExc_allDays_allMice[im][iday]
            Xinh = Xinh_allDays_allMice[im][iday]
            
            Y_svm = Y_svm_allDays_allMice[im][iday]       
            hr_trs = (Y_svm==1)
            lr_trs = (Y_svm==0)
            
            trss = hr_trs+lr_trs, hr_trs, lr_trs
      
          
            for hi in [1,2,3]: # all trials, hr, lr

            #    plotfr(a,b,hi,tlab)
                h1 = gs[iday, (hi-1)*3 : hi*3-1] #[hi,0:2]
                h2 = gs[iday, hi*3-1 : hi*3] #[hi,2:3]
                
                a = XallExc[fr,:,trss[hi-1]].flatten()
                b = Xinh[fr,:,trss[hi-1]].flatten()
                            
                mx = np.max(np.concatenate((a,b))) 
                mn = np.min(np.concatenate((a,b)))
                r = mx - mn
                binEvery = r/float(100)
            
                _, p = stats.ttest_ind(a, b, nan_policy='omit')    
                
                if hi==1:
                    yl = '%s\n%s' %(days[iday][0:6],yl0)
                else:
                    yl = yl0

                ax1, ax2 = histerrbar(h1,h2,a,b,binEvery,p,lab='FR (a.u.)',colors = ['b','r'],ylab=yl,lab1='exc',lab2='inh', plotCumsum=1)        
                ax1.set_title('%d exc; %d inh' %(len(a),len(b)))
            
            plt.subplots_adjust(wspace=2, hspace=1)
    
    fig.suptitle(tlab, fontsize=11)
    plt.subplots_adjust(wspace=2, hspace=1)        
'''                
 
    