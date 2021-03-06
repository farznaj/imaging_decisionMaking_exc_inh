# -*- coding: utf-8 -*-
"""
compute angles between weights found in script excInh_SVMtrained_eachFrame, ie
weights computed by training the decoder using only only inh or only exc neurons.

NOTE:
the parts of this script that set behavioral vars and class accuracy vars (but not the angle vars) are saved as script "svm_stability_plots_excInhEachFrame_setVars_behCA.py". 
This is bc in the first run of the script below behavioral and class accuracy vars were not saved properly... so I am again saving them here for each mouse.


Plots class accuracy for svm trained on non-overlapping time windows  (outputs of file svm_eachFrame.py)
 ... svm trained to decode choice on choice-aligned or stimulus-aligned traces.
 
 
Remember for fni18 there are 2 svm_eachFrame mat files, the earlier file is using all trials (unequal HR, LR, like how you've done all your analysis). 
The later mat file is with equal number of hr and lr trials (subselecting trials)... this helped with 151209 class accur trace which was weird in the earlier mat file.
 
Created on Sun Mar 12 15:12:29 2017
@author: farznaj
"""     

#%% Set the following vars:

mice = 'fni16', 'fni17', 'fni18', 'fni19' # 'fni17',

shflTrLabs = 1 # svm is already run on the actual data, so now load bestc, and run it on trial-label shuffles.
saveVars = 1 # if 1, a mat file will be saved including angle variables

sigPerc = 1 #5 # percentile to determine significancy (if data angle is lower than 5th percentile of shuffle angles, we call it siginificantly lower!)
excludeLowTrDays = 1 # remove days with too few trials
#doPlotsEachMouse = 1 # make plots for each mouse
#doExcSamps = 1 # if 1, for exc use vars defined by averaging trial subselects first for each exc samp and then averaging across exc samps. This is better than pooling all trial samps and exc samps ...    
#savefigs = 1

nsh = 1000 # number of times to shuffle the decoders to get null distribution of angles
loadWeights = 1

# quantile of stim strength - cb to use         
# 0: hard (min <= sr < 25th percentile of stimrate-cb); 
# 1: medium hard (25th<= sr < 50th); 
# 2: medium easy (50th<= sr < 75th); 
# 3: easy (75th<= sr <= max);
thQStimStrength = 3 # 0 to 3 : hard to easy # # set to nan if you want to include all strengths in computing behaioral performance

doAllN = 1 # plot allN, instead of allExc
thTrained = 10#10 # number of trials of each class used for svm training, min acceptable value to include a day in analysis
corrTrained = 1
doIncorr = 0

ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. #chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces.  #chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 


useEqualTrNums = 1
chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]
if doAllN==1:
    labAll = 'allN'
else:
    labAll = 'allExc'    
if doAllN==1:
    smallestC = 0 # Identify best c: if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
    if smallestC==1:
        print 'bestc = smallest c whose cv error is less than 1se of min cv error'
    else:
        print 'bestc = c that gives min cv error'

from datetime import datetime
nowStr = datetime.now().strftime('%y%m%d-%H%M%S')
#time2an = -1; # relative to eventI, look at classErr in what time stamp.
shflTrsEachNeuron = 0

trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.
#dnow0 = '/stability/'


#execfile("defFuns.py")
from defFuns import *


           
#%% 
#####################################################################################################################################################   
###############################################################################################################################     
#####################################################################################################################################################

numDaysAll = np.full(len(mice), np.nan, dtype=int)
days_allMice = [] 
eventI_ds_allMice = [] # you need this var to align traces of all mice


#%% im = 0     
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
    days_allMice.append(days)
    numDaysAll[im] = len(days)
   
    dnow = '/stability/'+mousename+'/'

            
    #%% Loop over days for each mouse   
    
    perClassErrorTest_data_inh_all = []
    perClassErrorTest_shfl_inh_all = []
    perClassErrorTest_chance_inh_all = []
    perClassErrorTest_data_allExc_all = []
    perClassErrorTest_shfl_allExc_all = []
    perClassErrorTest_chance_allExc_all = []
    perClassErrorTest_data_exc_all = []
    perClassErrorTest_shfl_exc_all = []
    perClassErrorTest_chance_exc_all = []
    numInh = np.full((len(days)), np.nan)
    numAllexc = np.full((len(days)), np.nan)
    angleInh_all = []
    angleExc_all = []
    angleAllExc_all = []
    angleInhS_all = []
    angleExcS_all = []
    angleAllExcS_all = []
    angleExc_excsh_all = []
    angleExc_excshS_all = []    
    behCorr_all = []
    behCorrHR_all = []
    behCorrLR_all = []    
    eventI_ds_allDays = np.full((len(days)), np.nan)    
    eventI_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
    corr_hr_lr = np.full((len(days),2), np.nan) # number of hr, lr correct trials for each day    
    
    
    #%%
#    iday = 0
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
    

        #%% Set number of hr, lr trials that were used for svm training
        
        svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, [1,0,0], regressBins, useEqualTrNums, corrTrained, shflTrsEachNeuron)[0]   
        
        corr_hr, corr_lr = set_corr_hr_lr(postName, svmName)    
        corr_hr_lr[iday,:] = [corr_hr, corr_lr]    

    
        #%% Set behavioral perforamnce: fraction of easy trials that are correct 
        
        Data = scio.loadmat(postName, variable_names=['outcomes', 'allResp_HR_LR', 'stimrate', 'cb'])
        outcomes = (Data.pop('outcomes').astype('float'))[0,:]         # allResp_HR_LR = (Data.pop('allResp_HR_LR').astype('float'))[0,:]
        allResp_HR_LR = np.array(Data.pop('allResp_HR_LR')).flatten().astype('float')
        stimrate = np.array(Data.pop('stimrate')).flatten().astype('float')
        cb = np.array(Data.pop('cb')).flatten().astype('float')
            
        # change-of-mind trials are excluded from svm (trsExcluded), but not from above ... I think I should exclude them from here too 
        '''
        outcomes[trsExcluded] = np.nan
        allResp_HR_LR[trsExcluded] = np.nan
        '''
        
        if ~np.isnan(thQStimStrength): # set to nan if you want to include all strengths in computing behaioral performance
            '''
            # we divide stimuli to hard, medium hard, medium easy, easy, so 25th,50th and 75th percentile of stimrate-cb will be the borders
            thHR = np.round(np.percentile(np.unique(stimrate[stimrate>=cb]), [25,50,75]))
            thLR = np.round(np.percentile(np.unique(stimrate[stimrate<=cb]), [25,50,75]))#[::-1]
           
            # if strength is hard, trials with stimrate=cb, appear in both snh and snr.
            # HR:
            inds = np.digitize(stimrate, thHR)
            snh = np.logical_and(stimrate>=cb , inds == thQStimStrength) # index of hr trials whose strength is easy
            # LR        
            inds = 3 - np.digitize(stimrate, thLR, right=True) # we subtract it from 3 so indeces match HR
            snl = np.logical_and(stimrate<=cb , inds == thQStimStrength) # index of lr trials whose strength is easy
            '''
            #############
            maxDiff = np.floor(min([max(stimrate)-cb, cb-min(stimrate)])) # assuming each session always included easy trials.
            wid = maxDiff/4 
            thHR = np.arange(cb, max(stimrate), wid)[1:]
            thHR[-1] = max(stimrate)+1
            thLR = np.arange(cb, min(stimrate), -wid)[1:]
            thLR[-1] = min(stimrate)-1
            thLR = thLR[::-1]
            
            # HR:
            inds = np.digitize(stimrate, thHR)
            snh = (inds == thQStimStrength) # index of easy hr trials
            # LR:
            inds = 4 - np.digitize(stimrate, thLR)
            snl = (inds == thQStimStrength) # index of easy hr trials

            str2ana = snh+snl # index of trials with stim strength of interest    

        else:
            str2ana = np.full(np.shape(outcomes), True).astype('bool')
            
        print 'Number of trials with stim strength of interest = %i' %(str2ana.sum())
        print 'Stim rates for training = {}'.format(np.unique(stimrate[str2ana]))

        # now you have easy trials... see what percentage of them are correct         # remember only the corr and incorr trials were used for svm training ... doesnt really matter here...
        v = outcomes[str2ana]
        v = v[v>=0] # only use corr or incorr trials
        if len(v) < thTrained: #10:
            sys.exit(('Too few easy trials; only %d. Is behavioral performance meaningful?!'  %(len(v))))
        behCorr = (v==1).mean()         #behCorr = (outcomes[str2ana]==1).mean()        
#        (outcomes[str2ana]==0).mean()        
        # what fraction of HR trials were correct? 
        v = outcomes[np.logical_and(stimrate>cb , str2ana)]
        v = v[v>=0] # only use corr or incorr trials
        behCorrHR = (v==1).mean()   #(outcomes[np.logical_and(stimrate>cb , str2ana)]==1).mean()
        # what fraction of LR trials were correct? 
        v = outcomes[np.logical_and(stimrate<cb , str2ana)]
        v = v[v>=0] # only use corr or incorr trials        
        behCorrLR = (v==1).mean()   #(outcomes[np.logical_and(stimrate<cb , str2ana)]==1).mean()
        
        behCorr_all.append(behCorr)
        behCorrHR_all.append(behCorrHR)
        behCorrLR_all.append(behCorrLR)    


        #%% Set eventI_ds (downsampled eventI)

        eventI, eventI_ds = setEventIds(postName, chAl, regressBins=3, trialHistAnalysis=0)
        
        eventI_allDays[iday] = eventI
        eventI_ds_allDays[iday] = eventI_ds


        #%% Load SVM vars : loadSVM_excInh
    
        perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN = \
            loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron, shflTrLabs)
        
        ##%% Get number of inh and exc        
        if loadWeights!=0: # weights were loaded
            numInh[iday] = w_data_inh.shape[1]
            numAllexc[iday] = w_data_allExc.shape[1]        
        
        ####%% Take care of lastTimeBinMissed = 0 #1# if 0, things were ran fine; if 1: by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
        ### allN data was run with lastTimeBinMissed = 1 
#        if perClassErrorTest_chance_inh.shape[1] != perClassErrorTest_chance_allExc.shape[1]:
#            if perClassErrorTest_chance_inh.shape[1] - perClassErrorTest_chance_allExc.shape[1] == 1:
        if w_data_inh.shape[2] != w_data_allExc.shape[2]:
            if w_data_inh.shape[2] - w_data_allExc.shape[2] == 1:
                print '================== lastTimeBinMissed=1 for allN =================='
                print '======== removing last element from inh/exc to match the size with allN ========'
                
                if loadWeights!=2: # class accuracies were load
                    perClassErrorTest_data_inh = np.delete(perClassErrorTest_data_inh, -1, axis=-1)
                    perClassErrorTest_shfl_inh = np.delete(perClassErrorTest_shfl_inh, -1, axis=-1)
                    perClassErrorTest_chance_inh = np.delete(perClassErrorTest_chance_inh, -1, axis=-1)
                    perClassErrorTest_data_exc = np.delete(perClassErrorTest_data_exc, -1, axis=-1)
                    perClassErrorTest_shfl_exc = np.delete(perClassErrorTest_shfl_exc, -1, axis=-1)
                    perClassErrorTest_chance_exc = np.delete(perClassErrorTest_chance_exc, -1, axis=-1)
                w_data_inh = np.delete(w_data_inh, -1, axis=-1)
                w_data_exc = np.delete(w_data_exc, -1, axis=-1)
                b_data_inh = np.delete(b_data_inh, -1, axis=-1)
                b_data_exc = np.delete(b_data_exc, -1, axis=-1)
                
            else:
                sys.exit('something wrong')
                

        numExcSamples = w_data_inh.shape[0]

        
        #%% Keep class accuracy vars for all days
        
        if loadWeights!=2: # class accuracies were load
            perClassErrorTest_data_inh_all.append(perClassErrorTest_data_inh) # each day: samps x numFrs    
            perClassErrorTest_shfl_inh_all.append(perClassErrorTest_shfl_inh)
            perClassErrorTest_chance_inh_all.append(perClassErrorTest_chance_inh)
            
            perClassErrorTest_data_allExc_all.append(perClassErrorTest_data_allExc) # each day: samps x numFrs    
            perClassErrorTest_shfl_allExc_all.append(perClassErrorTest_shfl_allExc)
            perClassErrorTest_chance_allExc_all.append(perClassErrorTest_chance_allExc) 
    
            perClassErrorTest_data_exc_all.append(perClassErrorTest_data_exc) # each day: numShufflesExc x numSamples x numFrames    
            perClassErrorTest_shfl_exc_all.append(perClassErrorTest_shfl_exc)
            perClassErrorTest_chance_exc_all.append(perClassErrorTest_chance_exc)
    
    
        #%% Normalize weights (ie the w vector at each frame must have length 1)
        
        # inh
        w = w_data_inh + 0 # numSamples x nNeurons x nFrames
        normw = np.linalg.norm(w, axis=1) # numSamples x frames ; 2-norm of weights 
        wInh_normed = np.transpose(np.transpose(w,(1,0,2))/normw, (1,0,2)) # numSamples x nNeurons x nFrames; normalize weights so weights (of each frame) have length 1
        if sum(normw<=eps).sum()!=0:
            print 'take care of this; you need to reshape w_normed first'
        #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
        
        # exc
        w = w_data_exc + 0 # numShufflesExc x numSamples x numNeurons x numFrames
        normw = np.linalg.norm(w, axis=2) # numShufflesExc x numSamples x frames ; 2-norm of weights 
        wExc_normed = np.transpose(np.transpose(w,(2,0,1,3))/normw, (1,2,0,3)) # numShufflesExc x numSamples x numNeurons x numFrames; normalize weights so weights (of each frame) have length 1
        if sum(normw<=eps).sum()!=0:
            print 'take care of this; you need to reshape w_normed first'
        #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
         
        # allExc            
        w = w_data_allExc + 0 # numSamples x nNeurons x nFrames
        normw = np.linalg.norm(w, axis=1) # numSamples x frames ; 2-norm of weights 
        wAllExc_normed = np.transpose(np.transpose(w,(1,0,2))/normw, (1,0,2)) # numSamples x nNeurons x nFrames; normalize weights so weights (of each frame) have length 1
        if sum(normw<=eps).sum()!=0:
            print 'take care of this; you need to reshape w_normed first'
        #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero

    
        numSamples = wInh_normed.shape[0]
        nfrs = wInh_normed.shape[-1]
        
        
        
        
        
        
        
                
        
        #%% Compute angle between decoders of different trial subsamples
        '''
        a = wInh_normed
        a = wAllExc_normed
        
        # same frame, 50 samples... get the angles between these 50 samples (50 x 50)
     
        angleInh_samps = np.full((numSamples,numSamples,nfrs), np.nan)
        
        for ifr in range(nfrs):
            v1 = a[:,:,ifr].T # neurons x samples
            angleInh_samps[:,:,ifr] = np.arccos(abs(np.dot(v1.transpose(), v1)))*180/np.pi # samps x samps; angle between decoders of different trial subsamples
            # set the diagonal (angle of each samp by itself) to nan
            np.fill_diagonal(angleInh_samps[:,:,ifr], np.nan)           
            
            # only take the lower triangle since the matrix is symmetric    
            angleInh_samps[:,:,ifr][np.triu_indices(numSamples)] = np.nan      
            
        angleInh_avSamps = np.nanmean(angleInh_samps, axis=(0,1)) # nfrs
#        angleInh_avSamps = np.nanmedian(angleInh_samps, axis=(0,1)) # nfrs
        plt.plot(angleInh_avSamps)
#        plt.imshow(angleInh_samps); plt.colorbar()               
#        np.nanmean(angleInh_samps) , np.nanmedian(angleInh_samps)
        
        # look at dist of angles between samples for each frame
        plt.figure(figsize=(2,nfrs*2))
        for ifr in range(nfrs):
            plt.subplot(nfrs,1,ifr+1)
            plt.hist(angleInh_samps[:,:,ifr][~np.isnan(angleInh_samps[:,:,ifr])]);
        

        
        # same frame, 50 samples... get the angles between these 50 samples (50 x 50)
        angleInh_samps = np.full((numSamples,numSamples,numExcSamples, nfrs), np.nan)
        
        for iexc in range(numExcSamples):
            a = wExc_normed[iexc]
            for ifr in range(nfrs):
                v1 = a[:,:,ifr].T # neurons x samples
                angleInh_samps[:,:,iexc,ifr] = np.arccos(abs(np.dot(v1.transpose(), v1)))*180/np.pi # samps x samps; angle between decoders of different trial subsamples
                # set the diagonal (angle of each samp by itself) to nan
                np.fill_diagonal(angleInh_samps[:,:,iexc,ifr], np.nan)
            
        # set the diagonal (angle of each samp by itself) to nan
        angleInh_avSamps = np.nanmean(angleInh_samps, axis=(0,1,2)) # nfrs
#        angleInh_avSamps = np.nanmedian(angleInh_samps, axis=(0,1)) # nfrs
        plt.plot(angleInh_avSamps)




#        a = wInh_normed
        angleInhS_samps = np.full((numSamples,numSamples,nfrs, nsh), np.nan)

        for ish in range(nsh):
            # remember bc of the way u shuffle neurons of each decoder, angle between shuffled decoder of frame 1 and shuffled decoder of frame 2 wont be the same  as that of frame 2 and frame 1 (bc neurons are shuffled once in nord and again nord1... also angle between decoders of the same frame are not really 0 bc the order of neurons in frame 1 (nord) is not the same as the order in frame 1 (nord1))        
            nord = rng.permutation(a.shape[1]) # shuffle neurons 
            nord1 = rng.permutation(a.shape[1])            
            for ifr in range(nfrs):
                v1 = a[:,:,ifr].T # neurons x samples
                angleInhS_samps[:,:,ifr,ish] = np.arccos(abs(np.dot(v1[nord,:].transpose(), v1[nord1,:])))*180/np.pi # samps x samps; angle between decoders of different trial subsamples
                # set the diagonal (angle of each samp by itself) to nan
                np.fill_diagonal(angleInhS_samps[:,:,ifr,ish], np.nan)
            
        angleInhS_avSamps = np.nanmean(angleInhS_samps, axis=(0,1,3)) # nfrs
#        angleInh_avSamps = np.nanmedian(angleInh_samps, axis=(0,1)) # nfrs
        plt.plot(angleInhS_avSamps)            
        

        plt.figure()
        plt.plot(angleInhS_avSamps - angleInh_avSamps)
        

        
        #%% instead of taking angle between samples (which seems to be noisy for some frames)... do bootstrapping (random sampling with replacemenet) and then average the decoders and then compute angle like below
        
#        a = wInh_normed
        a = wAllExc_normed
        # inh                
        wInh_n2b_bootstrap = [] # nsh x neurons x frames
        for ish in range(nsh):
            randsamps = np.random.choice(numSamples, numSamples) # take random samples with replacement            
            w_nb = np.mean(a[randsamps,:,:], axis=(0)) # neurons x frames 
            nw = np.linalg.norm(w_nb, axis=0) # frames; 2-norm of weights 
            wInh_n2b = w_nb/nw # neurons x frames
            wInh_n2b_bootstrap.append(wInh_n2b)
        wInh_n2b_bootstrap = np.array(wInh_n2b_bootstrap)


        # now compute the angle between samples in frame ifr
        a = wInh_n2b_bootstrap

        angleInh_samps = np.full((nsh,nsh,nfrs), np.nan)
        angleInhS_samps = np.full((nsh,nsh,nfrs), np.nan)
        for ifr in range(nfrs):
            v1 = a[:,:,ifr].T # neurons x samples
            # real data
            angleInh_samps[:,:,ifr] = np.arccos(abs(np.dot(v1.transpose(), v1)))*180/np.pi # samps x samps; angle between decoders of different trial subsamples
            
            # remember bc of the way u shuffle neurons of each decoder, angle between shuffled decoder of frame 1 and shuffled decoder of frame 2 wont be the same  as that of frame 2 and frame 1 (bc neurons are shuffled once in nord and again nord1... also angle between decoders of the same frame are not really 0 bc the order of neurons in frame 1 (nord) is not the same as the order in frame 1 (nord1))        
            nord = rng.permutation(v1.shape[0]) # shuffle neurons 
            nord1 = rng.permutation(v1.shape[0])
            angleInhS_samps[:,:,ifr] = np.arccos(abs(np.dot(v1[nord,:].transpose(), v1[nord1,:])))*180/np.pi 
            # set the diagonal (angle of each samp by itself) to 0
            np.fill_diagonal(angleInh_samps[:,:,ifr], 0)            
            np.fill_diagonal(angleInhS_samps[:,:,ifr], 0)            
            
            # set the diagonal (angle of each samp by itself) to nan
            np.fill_diagonal(angleInh_samps[:,:,ifr], np.nan)            
            # only take the lower triangle since the matrix is symmetric    
            angleInh_samps[:,:,ifr][np.triu_indices(numSamples)] = np.nan      
            
        angleInh_avSamps = np.nanmean(angleInh_samps, axis=(0,1)) # nfrs
        angleInhS_avSamps = np.nanmean(angleInhS_samps, axis=(0,1)) # nfrs
#        angleInh_avSamps = np.nanmedian(angleInh_samps, axis=(0,1)) # nfrs
        plt.figure(); plt.plot(angleInh_avSamps)
        plt.figure(); plt.plot(angleInhS_avSamps)
        plt.figure(); plt.plot(angleInhS_avSamps - angleInh_avSamps)
#        plt.imshow(angleInh_samps); plt.colorbar()               
#        np.nanmean(angleInh_samps) , np.nanmedian(angleInh_samps)
        
        # look at dist of angles between samples for each frame
        plt.figure(figsize=(2,nfrs*2))
        for ifr in range(nfrs):
            plt.subplot(nfrs,1,ifr+1)
            plt.hist(angleInh_samps[:,:,ifr][~np.isnan(angleInh_samps[:,:,ifr])]);
        
        '''
        
        

            
        
        #%% Set the final decoders by aggregating decoders across trial subselects (ie average ws across all trial subselects), then again normalize them so they have length 1 (Bagging)
        
        # inh                    
        w_nb = np.mean(wInh_normed, axis=(0)) # neurons x frames 
        nw = np.linalg.norm(w_nb, axis=0) # frames; 2-norm of weights 
        wInh_n2b = w_nb/nw # neurons x frames
    
        # exc (average across trial subselects as well as exc neurons subselects)
        #### NOTE: we are pooling decoders from trial and exc samples here and then taking an average across them.
        ### I do not think this is right to do! we should get decoders for each exc sample, compute their angles. Do the same for all exc samples and then perhaps compute the average of angles across exc samps.
        # this is what you do for wExc_n2b_excsh
        w_nb = np.mean(wExc_normed, axis=(0,1)) # neurons x frames 
        nw = np.linalg.norm(w_nb, axis=0) # frames; 2-norm of weights 
        wExc_n2b = w_nb/nw # neurons x frames
        
        # exc, compute ave of tr subselects for each exc shuffle separately.
        wExc_n2b_excsh = np.full((wExc_normed.shape[0], wExc_n2b.shape[0], wExc_n2b.shape[1]), np.nan)
        for ish in range(wExc_normed.shape[0]):
            w_nb = np.mean(wExc_normed[ish], axis=(0)) # neurons x frames ...> average each exc sample across all trial subselects
            nw = np.linalg.norm(w_nb, axis=0) # frames; 2-norm of weights 
            wExc_n2b_excsh[ish,:,:] = w_nb/nw # excShfl x neurons x frames
        
        # allN
        w_nb = np.mean(wAllExc_normed, axis=(0)) # neurons x frames 
        nw = np.linalg.norm(w_nb, axis=0) # frames; 2-norm of weights 
        wAllExc_n2b = w_nb/nw # neurons x frames
    
        
        #%% Compute angle between decoders (weights) at different times (remember angles close to 0 indicate more aligned decoders at 2 different time points.)
        
        # some of the angles are nan because the dot product is slightly above 1, so its arccos is nan!    
        # we restrict angles to [0 90], so we don't care about direction of vectors, ie tunning reversal.
        
        angleInh = np.arccos(abs(np.dot(wInh_n2b.transpose(), wInh_n2b)))*180/np.pi # frames x frames; angle between ws at different times        
        angleExc = np.arccos(abs(np.dot(wExc_n2b.transpose(), wExc_n2b)))*180/np.pi # frames x frames; angle between ws at different times
        angleAllExc = np.arccos(abs(np.dot(wAllExc_n2b.transpose(), wAllExc_n2b)))*180/np.pi # frames x frames; angle between ws at different times
        # each exc shfl
        angleExc_excsh = np.full((angleExc.shape[0],angleExc.shape[0],wExc_normed.shape[0]), np.nan) # frs x frs x excShfl
        for ish in range(wExc_normed.shape[0]):        
            angleExc_excsh[:,:,ish] = np.arccos(abs(np.dot(wExc_n2b_excsh[ish].transpose(), wExc_n2b_excsh[ish])))*180/np.pi
     
     
        # shuffles: angle between the real decoders and the shuffled decoders (same as real except the indexing of weights is shuffled)     
#        nsh = 1000 # number of times to shuffle the decoders to get null distribution of angles
        frs = wInh_n2b.shape[1] # number of frames
        angleInhS = np.full((frs, frs, nsh), np.nan)
        angleExcS = np.full((frs, frs, nsh), np.nan)
        angleAllExcS = np.full((frs, frs, nsh), np.nan)
        angleExc_excshS = np.full((frs, frs, nsh, wExc_normed.shape[0]), np.nan) # frs x frs x angShfl x excShfl

        for ish in range(nsh):
            # remember bc of the way u shuffle neurons of each decoder, angle between shuffled decoder of frame 1 and shuffled decoder of frame 2 wont be the same  as that of frame 2 and frame 1 (bc neurons are shuffled once in nord and again nord1... also angle between decoders of the same frame are not really 0 bc the order of neurons in frame 1 (nord) is not the same as the order in frame 1 (nord1))        
            nord = rng.permutation(wInh_n2b.shape[0]) # shuffle neurons 
            nord1 = rng.permutation(wInh_n2b.shape[0])
            angleInhS[:,:,ish] = np.arccos(abs(np.dot(wInh_n2b[nord,:].transpose(), wInh_n2b[nord1,:])))*180/np.pi 

            nord = rng.permutation(wExc_n2b.shape[0])            
            nord1 = rng.permutation(wExc_n2b.shape[0])
            angleExcS[:,:,ish] = np.arccos(abs(np.dot(wExc_n2b[nord,:].transpose(), wExc_n2b[nord1,:])))*180/np.pi 

            nord = rng.permutation(wAllExc_n2b.shape[0])                        
            nord1 = rng.permutation(wAllExc_n2b.shape[0])
            angleAllExcS[:,:,ish] = np.arccos(abs(np.dot(wAllExc_n2b[nord,:].transpose(), wAllExc_n2b[nord1,:])))*180/np.pi
            
            # each exc shfl
            angleExc_excsh0 = np.full((wExc_normed.shape[0],angleExc.shape[0],angleExc.shape[0]), np.nan) # excShfl x frs x frs
            for ish2 in range(wExc_normed.shape[0]):        
                nord = rng.permutation(wInh_n2b.shape[0]) # same as rng.permutation(wExc_n2b_excsh[ish2].shape[0])
                nord1 = rng.permutation(wInh_n2b.shape[0])
                angleExc_excsh0[ish2] = np.arccos(abs(np.dot(wExc_n2b_excsh[ish2][nord,:].transpose(), wExc_n2b_excsh[ish2][nord1,:])))*180/np.pi        
            angleExc_excshS[:,:,ish,:] = np.transpose(angleExc_excsh0, (1,2,0))
        
        
        
        
        
        
        
        
        
        
        
        #%% compute angle between decoders at each time point and their shuffled version
        '''
        # angle between wInh_n2b and its shuffled version 
#        a = wInh_n2b  # neurons x frames        
        a = wAllExc_n2b
        
        ash = a[rng.permutation(a.shape[0]),:] # neurons order is shuffled  # neurons x frames
        angleInh_S = np.arccos(abs(np.dot(a.transpose(), ash)))*180/np.pi # frames x frames; angle between ws at different times        
        # all we care about is the diagonal of the above matrix... bc that shows angle btwn a decoder and its shuffled version for each time point (we dont care about angle between decoder(t) and shuffled_decoder(t'))
        angleInh_S_d = np.diagonal(angleInh_S)
        plt.plot(angleInh_S_d)
        '''
        
        
        
        
        
        #%% Keep vars from all days
         
        angleInh_all.append(angleInh)
        angleExc_all.append(angleExc)
        angleAllExc_all.append(angleAllExc)    
        angleExc_excsh_all.append(angleExc_excsh)
         
        angleInhS_all.append(angleInhS)
        angleExcS_all.append(angleExcS)
        angleAllExcS_all.append(angleAllExcS)    
        angleExc_excshS_all.append(angleExc_excshS)
         

        #%% Delete vars before starting the next day    
         
        del w_data_inh, w_data_allExc, w_data_exc, perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc
    
    
    #%% Done with all days, keep values of all mice
    
    eventI_ds_allDays = eventI_ds_allDays.astype('int')
    eventI_allDays = eventI_allDays.astype('int')
    angleInh_all = np.array(angleInh_all)
    angleExc_all = np.array(angleExc_all)
    angleAllExc_all = np.array(angleAllExc_all)
    angleExc_excsh_all = np.array(angleExc_excsh_all)
    angleInhS_all = np.array(angleInhS_all)
    angleExcS_all = np.array(angleExcS_all)
    angleAllExcS_all = np.array(angleAllExcS_all)
    angleExc_excshS_all = np.array(angleExc_excshS_all)
    
    ####################################################################################
    ####################################################################################
    ####################################################################################

    #%% Decide what days to analyze: exclude days with too few trials used for training SVM, also exclude incorr from days with too few incorr trials.
    
    # th for min number of trs of each class
    '''
    thTrained = 30 #25; # 1/10 of this will be the testing tr num! and 9/10 was used for training
    thIncorr = 4 #5
    '''
    mn_corr = np.min(corr_hr_lr,axis=1) # number of trials of each class. 90% of this was used for training, and 10% for testing.
    
    print 'num days to be excluded with few svm-trained trs:', sum(mn_corr < thTrained)    
    print np.array(days)[mn_corr < thTrained]
    
    numGoodDays = sum(mn_corr>=thTrained)    
    numOrigDays = numDaysAll[im].astype(int)
    
    dayinds = np.arange(numOrigDays)
    dayinds = np.delete(dayinds, np.argwhere(mn_corr < thTrained))
   
    
    #%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
    # By common eventI, we  mean the index on which all traces will be aligned.
    
    time_aligned, nPreMin, nPostMin = set_nprepost(angleInh_all, eventI_ds_allDays, mn_corr, thTrained, regressBins) # av_test_data_inh

    eventI_ds_allMice.append(nPreMin) # you need this var to align traces of all mice
    
    
    
    #%%    
    #################################################################
    #################### Take care of class accuracies ########################
    #################################################################
      
    if loadWeights!=2: # class accuracies were load  
        ##%% Average and st error of class accuracies across CV samples ... for each day
        
        numSamples, numExcSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, av_test_data_exc, sd_test_data_exc, av_test_shfl_exc, sd_test_shfl_exc, av_test_chance_exc, sd_test_chance_exc, av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc \
            = av_se_CA_trsamps(numOrigDays, perClassErrorTest_data_inh_all, perClassErrorTest_shfl_inh_all, perClassErrorTest_chance_inh_all, perClassErrorTest_data_exc_all, perClassErrorTest_shfl_exc_all, perClassErrorTest_chance_exc_all, perClassErrorTest_data_allExc_all, perClassErrorTest_shfl_allExc_all, perClassErrorTest_chance_allExc_all)
    
         
        #%% Align traces of all days on the common eventI
        
        #same as alTrace in defFuns, except here the output is an array of size frs x days, there the output is a list of size days x frs
        def alTrace(trace, eventI_ds_allDays, nPreMin, nPostMin, mn_corr, thTrained=10):    
            numDays = len(trace) 
            trace_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)     
            for iday in range(trace.shape[0]):         
                if mn_corr[iday] >= thTrained: # dont include days with too few
                    trace_aligned[:, iday] = trace[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]    
            return trace_aligned
            
        
        av_test_data_inh_aligned = alTrace(av_test_data_inh, eventI_ds_allDays, nPreMin, nPostMin, mn_corr)
        av_test_shfl_inh_aligned = alTrace(av_test_shfl_inh, eventI_ds_allDays, nPreMin, nPostMin, mn_corr)
        av_test_chance_inh_aligned = alTrace(av_test_chance_inh, eventI_ds_allDays, nPreMin, nPostMin, mn_corr)
        
        av_test_data_exc_aligned = alTrace(av_test_data_exc, eventI_ds_allDays, nPreMin, nPostMin, mn_corr)
        av_test_shfl_exc_aligned = alTrace(av_test_shfl_exc, eventI_ds_allDays, nPreMin, nPostMin, mn_corr)
        av_test_chance_exc_aligned = alTrace(av_test_chance_exc, eventI_ds_allDays, nPreMin, nPostMin, mn_corr)
        
        av_test_data_allExc_aligned = alTrace(av_test_data_allExc, eventI_ds_allDays, nPreMin, nPostMin, mn_corr)
        av_test_shfl_allExc_aligned = alTrace(av_test_shfl_allExc, eventI_ds_allDays, nPreMin, nPostMin, mn_corr)
        av_test_chance_allExc_aligned = alTrace(av_test_chance_allExc, eventI_ds_allDays, nPreMin, nPostMin, mn_corr)
        
        
        #%% Compute data - shuffle for the aligned class accuracy traces
        #### only good days
        
        classAccurTMS_inh = np.array([(av_test_data_inh_aligned[:, iday] - av_test_shfl_inh_aligned[:, iday]) for iday in dayinds])
        classAccurTMS_exc = np.array([(av_test_data_exc_aligned[:, iday] - av_test_shfl_exc_aligned[:, iday]) for iday in dayinds])
        classAccurTMS_allExc = np.array([(av_test_data_allExc_aligned[:, iday] - av_test_shfl_allExc_aligned[:, iday]) for iday in dayinds])
    
    
    
    #%%
    #################################################################
    #################### Take care of angles ########################
    #################################################################
    
    #%% Align angles of all days on the common eventI
    
    totLen = len(time_aligned) # nPreMin + nPostMin + 1
    
    angleInh_aligned = np.ones((totLen,totLen, numOrigDays)) + np.nan
    angleExc_aligned = np.ones((totLen,totLen, numOrigDays)) + np.nan
    angleAllExc_aligned = np.ones((totLen,totLen, numOrigDays)) + np.nan  # frames x frames x days, aligned on common eventI (equals nPreMin)
    angleExc_excsh_aligned = np.ones((totLen,totLen, numExcSamples, numOrigDays)) + np.nan
    
    angleInhS_aligned = np.ones((totLen,totLen,nsh, numOrigDays)) + np.nan
    angleExcS_aligned = np.ones((totLen,totLen,nsh, numOrigDays)) + np.nan
    angleAllExcS_aligned = np.ones((totLen,totLen,nsh, numOrigDays)) + np.nan  # frames x frames x days, aligned on common eventI (equals nPreMin)
    angleExc_excshS_aligned = np.ones((totLen,totLen,nsh, numExcSamples, numOrigDays)) + np.nan
    
    for iday in range(numOrigDays):
    #    angleInh_aligned[:, iday] = angleInh_all[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1  ,  eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
        inds = np.arange(eventI_ds_allDays[iday] - nPreMin,  eventI_ds_allDays[iday] + nPostMin + 1)
        
        angleInh_aligned[:,:,iday] = angleInh_all[iday][inds][:,inds]
        angleExc_aligned[:,:,iday] = angleExc_all[iday][inds][:,inds]
        angleAllExc_aligned[:,:,iday] = angleAllExc_all[iday][inds][:,inds]
        angleExc_excsh_aligned[:,:,:,iday] = angleExc_excsh_all[iday][inds][:,inds] # frames x frames x nExcShfls
        # shuffles
        angleInhS_aligned[:,:,:, iday] = angleInhS_all[iday][inds][:,inds] # frames x frames x nShfls
        angleExcS_aligned[:,:,:, iday] = angleExcS_all[iday][inds][:,inds] # frames x frames x nShfls
        angleAllExcS_aligned[:,:,:, iday] = angleAllExcS_all[iday][inds][:,inds] # frames x frames x nShfls
        angleExc_excshS_aligned[:,:,:,:,iday] = angleExc_excshS_all[iday][inds][:,inds] # frames x frames x nShfls x nExcShfls            
            

    #%%            
    ################################### starting from here we only work with good days ###################################
            
    #%% Remove low trial-number days from average angles and average class accuracies (data-shfl)
    
    if excludeLowTrDays==1:
        ####################### REDEFINING days so it includes only days with enough number of trials ########################
        days = np.array(days)[dayinds]
        
        angleInh_aligned = angleInh_aligned[:,:,dayinds]
        angleExc_aligned = angleExc_aligned[:,:,dayinds]
        angleAllExc_aligned = angleAllExc_aligned[:,:,dayinds]
        angleExc_excsh_aligned = angleExc_excsh_aligned[:,:,:,dayinds]
        
        angleInhS_aligned = angleInhS_aligned[:,:,:,dayinds]
        angleExcS_aligned = angleExcS_aligned[:,:,:,dayinds]
        angleAllExcS_aligned = angleAllExcS_aligned[:,:,:,dayinds]
        angleExc_excshS_aligned = angleExc_excshS_aligned[:,:,:,:,dayinds]
        
        
    #%% Find time points at which decoders angles are significantly different from the null distribution
        
#    sigPerc = 5 # percentile to determine significancy (if data angle is lower than 5th percentile of shuffle angles, we calle it siginificantly lower!)
    sigAngInh = np.full((np.shape(angleInh_aligned)), False, dtype=bool) # frames x frames x days # matrix of 0s and 1s; 0 mean the decoders of those time points are mis-aligned (ie their angle is similar to null dist); 1 means the 2 decoders are aligned.
    sigAngExc = np.full((np.shape(angleExc_aligned)), False, dtype=bool)
    sigAngAllExc = np.full((np.shape(angleAllExc_aligned)), False, dtype=bool)
    sigAngExc_excsh = np.full((np.shape(angleExc_excsh_aligned)), False, dtype=bool) # frs x frs x excSamps x days
    
    pvals = np.arange(0,99,1) # percentile values # I will use 90 degrees as the 100th percentile, that's why I dont do np.arange(0,100,1)
    percAngInh = np.full((np.shape(angleExc_aligned)), np.nan)
    percAngExc = np.full((np.shape(angleExc_aligned)), np.nan)
    percAngAllExc = np.full((np.shape(angleExc_aligned)), np.nan)
    percAngExc_excsh = np.full((np.shape(angleExc_excsh_aligned)), np.nan) # frs x frs x excSamps x days
    
    for iday in range(numGoodDays):
        for f1 in range(totLen):
            for f2 in np.delete(range(totLen),f1): #range(totLen):
                ##### whether real angle is smaller than 5th percentile of shuffle angles.
                h = np.percentile(angleInhS_aligned[f1,f2,:, iday], sigPerc)
                sigAngInh[f1,f2,iday] = angleInh_aligned[f1,f2, iday] < h # if the real angle is < 5th percentile of the shuffled angles, then it is significantly differnet from null dist.  
                
                h = np.percentile(angleExcS_aligned[f1,f2,:, iday], sigPerc)
                sigAngExc[f1,f2,iday] = angleExc_aligned[f1,f2, iday] < h            

                h = np.percentile(angleAllExcS_aligned[f1,f2,:, iday], sigPerc)
                sigAngAllExc[f1,f2,iday] = angleAllExc_aligned[f1,f2, iday] < h

                h = np.percentile(angleExc_excshS_aligned[f1,f2,:,:, iday], sigPerc, axis=0) # for each exc samp, compute the percentile over the 1000 shfl angles
                sigAngExc_excsh[f1,f2,:,iday] = angleExc_excsh_aligned[f1,f2, :, iday] < h  # exc samps            

                
                ##### compute percentile of significancy ... did not turn out very useful
                # percentile of shuffled angles that are bigger than the real angle (ie more mis-aligned)
                # larger percAngInh means more alignment relative to shuffled dist. (big percentage of shuffled dist has larger angles than the real angle)
                percAngInh[f1,f2,iday] = 100 - np.argwhere(angleInh_aligned[f1,f2, iday] - np.concatenate((np.percentile(angleInhS_aligned[f1,f2,:, iday], pvals), [90])) < 0)[0]
                percAngExc[f1,f2,iday] = 100 - np.argwhere(angleExc_aligned[f1,f2, iday] - np.concatenate((np.percentile(angleExcS_aligned[f1,f2,:, iday], pvals), [90])) < 0)[0]
                percAngAllExc[f1,f2,iday] = 100 - np.argwhere(angleAllExc_aligned[f1,f2, iday] - np.concatenate((np.percentile(angleAllExcS_aligned[f1,f2,:, iday], pvals), [90])) < 0)[0]
                
                aa = np.percentile(angleExc_excshS_aligned[f1,f2,:,:, iday], pvals, axis=0)  # 99 x excSamps       
                a = np.array([np.concatenate((aa[:,ies] , [90])) for ies in range(numExcSamples)]) # excSamps x 100
                b = (angleExc_excsh_aligned[f1,f2, :,iday] - a.T).T  # excSamps x 100
                percAngExc_excsh[f1,f2,:,iday] = 100 - np.array([np.argwhere(b[ies] < 0)[0] for ies in range(numExcSamples)]).flatten()  # excSamps

    #np.mean(sigAngInh, axis=-1)) # each element shows fraction of days in which angle between decoders was significant.
    # eg I will call an angle siginificant if in >=.5 of days it was significant.
    #plt.imshow(np.mean(sigAngInh, axis=-1)); plt.colorbar() # fraction of days in which angle between decoders was significant.
    #plt.imshow(np.mean(angleInh_aligned, axis=-1)); plt.colorbar()        
    

    #%% Set diagonal elements (belonging to angles between the same decoders (ie decoders of the same frames) to nan
               
    [np.fill_diagonal(angleInh_aligned[:,:,iday], np.nan) for iday in range(numGoodDays)];
    [np.fill_diagonal(angleExc_aligned[:,:,iday], np.nan) for iday in range(numGoodDays)];
    [np.fill_diagonal(angleAllExc_aligned[:,:,iday], np.nan) for iday in range(numGoodDays)];    
    [np.fill_diagonal(angleExc_excsh_aligned[:,:,iesh,iday], np.nan) for iday in range(numGoodDays) for iesh in range(numExcSamples)];

    # shuffled
    [np.fill_diagonal(angleInhS_aligned[:,:,ish,iday], np.nan) for iday in range(numGoodDays) for ish in range(nsh)];
    [np.fill_diagonal(angleExcS_aligned[:,:,ish,iday], np.nan) for iday in range(numGoodDays) for ish in range(nsh)];
    [np.fill_diagonal(angleAllExcS_aligned[:,:,ish,iday], np.nan) for iday in range(numGoodDays) for ish in range(nsh)];    
    [np.fill_diagonal(angleExc_excshS_aligned[:,:,ish,iesh,iday], np.nan) for iday in range(numGoodDays) for ish in range(nsh) for iesh in range(numExcSamples)];
    
    # sig ... NOTE: since this is boolean not sure if turning to nan works! check!
    [np.fill_diagonal(sigAngInh[:,:,iday], np.nan) for iday in range(numGoodDays)];
    [np.fill_diagonal(sigAngExc[:,:,iday], np.nan) for iday in range(numGoodDays)];
    [np.fill_diagonal(sigAngAllExc[:,:,iday], np.nan) for iday in range(numGoodDays)];
    [np.fill_diagonal(sigAngExc_excsh[:,:,iesh,iday], np.nan) for iday in range(numGoodDays) for iesh in range(numExcSamples)];



    #%% Average shfl across the 1000 shfls

    angleInhS_aligned_avSh = np.nanmean(angleInhS_aligned,axis=2) # frs x frs x days
    angleExcS_aligned_avSh = np.nanmean(angleExcS_aligned,axis=2)
    angleAllExcS_aligned_avSh = np.nanmean(angleAllExcS_aligned,axis=2)
    angleExc_excshS_aligned_avSh = np.nanmean(angleExc_excshS_aligned,axis=2) # frs x frs x excSamps x days
        
        
    #%% Compute a measure for stability of decoders at each time point
    ################################################################################################    
    ######################## IMPORTANT: In the measures below you are averaging each column... since stability is symmetric around the diagonal, it means naturally you will get lower values at the begining and at the end of the trial
    ################################################################################################
    
    # I think meas=0 the best one. 
    
#    dostabplots = 0 # set to 1 if you want to compare the measures below

    ######## data : for each timebin average angle of the decoder with other decoders (of all other timepoints)
    stabScoreInh0_data = np.array([np.nanmean(angleInh_aligned[:,:,iday] , axis=0) for iday in range(numGoodDays)]) # days x frs
    stabScoreExc0_data = np.array([np.nanmean(angleExc_aligned[:,:,iday] , axis=0) for iday in range(numGoodDays)]) # days x frs
    stabScoreAllExc0_data = np.array([np.nanmean(angleAllExc_aligned[:,:,iday] , axis=0) for iday in range(numGoodDays)])    # days x frs     
    stabScoreExc_excsh0_data = np.full((numGoodDays, numExcSamples, angleExc_excsh_aligned.shape[0]), np.nan) # days x excSamps x frs
    for iday in range(numGoodDays):
        stabScoreExc_excsh0_data[iday,:,:] = np.array([np.nanmean(angleExc_excsh_aligned[:,:,iesh,iday] , axis=0) for iesh in range(numExcSamples)])  # excSamps x frs
    stabLab0_data = 'Instability (average angle with other decoders)'    


    ######## shfl : for each timebin average angle of the decoder with other decoders (of all other timepoints) (decoders are already averaged across 1000 shfls)
    stabScoreInh0_shfl = np.array([np.nanmean(angleInhS_aligned_avSh[:,:,iday] , axis=0) for iday in range(numGoodDays)]) # days x frs
    stabScoreExc0_shfl = np.array([np.nanmean(angleExcS_aligned_avSh[:,:,iday] , axis=0) for iday in range(numGoodDays)]) # days x frs
    stabScoreAllExc0_shfl = np.array([np.nanmean(angleAllExcS_aligned_avSh[:,:,iday] , axis=0) for iday in range(numGoodDays)])  # days x frs       
    stabScoreExc_excsh0_shfl = np.full((numGoodDays, numExcSamples, angleExc_excshS_aligned.shape[0]), np.nan) # days x excSamps x frs
    for iday in range(numGoodDays):
        stabScoreExc_excsh0_shfl[iday,:,:] = np.array([np.nanmean(angleExc_excshS_aligned_avSh[:,:,iesh,iday] , axis=0) for iesh in range(numExcSamples)])  # excSamps x frs
    stabLab0_shfl = 'Instability (shfl; average angle with other decoders)'    
    
    
    ######## degrees away from full mis-alignment :  for each time bin average the following quantity with all other decoders quantity : ave(shfl)-data
    stabScoreInh0 = np.array([np.nanmean((angleInhS_aligned_avSh[:,:,iday] - angleInh_aligned[:,:,iday]) , axis=0) for iday in range(numGoodDays)]) # days x frs
    stabScoreExc0 = np.array([np.nanmean((angleExcS_aligned_avSh[:,:,iday] - angleExc_aligned[:,:,iday]) , axis=0) for iday in range(numGoodDays)]) # days x frs
    stabScoreAllExc0 = np.array([np.nanmean((angleAllExcS_aligned_avSh[:,:,iday] - angleAllExc_aligned[:,:,iday]) , axis=0) for iday in range(numGoodDays)]) # days x frs        
    stabScoreExc_excsh0 = np.full((numGoodDays, numExcSamples, angleExc_excshS_aligned.shape[0]), np.nan) # days x excSamps x frs
    for iday in range(numGoodDays):
        stabScoreExc_excsh0[iday,:,:] = np.array([np.nanmean((angleExc_excshS_aligned_avSh[:,:,iesh,iday] - angleExc_excsh_aligned[:,:,iesh,iday]) , axis=0) for iesh in range(numExcSamples)])  # excSamps x frs    
    stabLab0 = 'Stability (degrees away from full mis-alignment)'    
        
        
    ######## number of decoders that are "aligned" with each decoder (at each timepoint) (we get this by summing sigAng across rows gives us the)
    # aligned decoder is defined as a decoder whose angle with the decoder of interest is <5th percentile of dist of shuffled angles.
    # sigAng: for each day shows whether angle betweeen the decoders at specific timepoints is significantly different than null dist (ie angle is smaller than 5th percentile of shuffle angles).
    stabScoreInh1 = np.array([np.nansum(sigAngInh[:,:,iday], axis=0) for iday in range(numGoodDays)]) # days x frs
    stabScoreExc1 = np.array([np.nansum(sigAngExc[:,:,iday], axis=0) for iday in range(numGoodDays)]) 
    stabScoreAllExc1 = np.array([np.nansum(sigAngAllExc[:,:,iday], axis=0) for iday in range(numGoodDays)]) 
    stabScoreExc_excsh1 = np.transpose(np.array([np.nansum(sigAngExc_excsh[:,:,:,iday], axis=0) for iday in range(numGoodDays)]), (0,2,1)) # days x excSamps x frs
    stabLab1 = 'Stability (number of aligned decoders)' # with the decoder of interest (ie the decoder at the specific timepont we are looking at)            
    
    
    ########################## since you dont seem to use the following 2 measures, u didnt add codes for each excSamp to them
    '''
    ######## how many decoders are at least 10 degrees away from the null decoder. # how stable is each decoder at other time points measure as how many decoders are at 70 degrees or less with each of the decoders (trained on a particular frame).. or ... .      
    th = 10  
    #np.mean(angle_aligned[:,:,iday],axis=0) # average stability ? (angle with all decoders at other time points)
    #stabAng = angle_aligned[:,:,iday] < th # how stable is each decoder at other time points measure as how many decoders are at 70 degrees or less with each of the decoders (trained on a particular frame).. or ... .
#    stabScoreInh = np.array([np.sum(angleInh_aligned[:,:,iday]<th, axis=0) for iday in range(numGoodDays)]) 
#    stabScoreExc = np.array([np.sum(angleExc_aligned[:,:,iday]<th, axis=0) for iday in range(numGoodDays)]) 
    stabScoreInh2 = np.array([np.sum(((np.mean(angleInhS_aligned,axis=2))[:,:,iday] - angleInh_aligned[:,:,iday]) >= th, axis=0) for iday in range(numGoodDays)]) 
    stabScoreExc2 = np.array([np.sum(((np.mean(angleExcS_aligned,axis=2))[:,:,iday] - angleExc_aligned[:,:,iday]) >= th, axis=0) for iday in range(numGoodDays)])     
    stabScoreAllExc2 = np.array([np.sum(((np.mean(angleAllExcS_aligned,axis=2))[:,:,iday] - angleAllExc_aligned[:,:,iday]) >= th, axis=0) for iday in range(numGoodDays)]) 
    stabLab2 = 'Stability (number of decoders > 10 degrees away from full mis-alignment )'
        
    ######## percentile of shuffled angles that are bigger than the real angle (ie more mis-aligned)
    # not a good measure bc does not properly distinguish eg .99 from .94 (.99 is significant, but .94 is not)
    stabScoreInh3 = np.array([np.nanmean(percAngInh[:,:,iday], axis=0) for iday in range(numGoodDays)]) 
    stabScoreExc3 = np.array([np.nanmean(percAngExc[:,:,iday], axis=0) for iday in range(numGoodDays)]) 
    stabScoreAllExc3 = np.array([np.nanmean(percAngAllExc[:,:,iday], axis=0) for iday in range(numGoodDays)]) 
    stabLab3 = 'Stability (number of decoders aligned with)'
    '''

#    ssi = stabScoreInh0, stabScoreInh1
#    sse = stabScoreExc0, stabScoreExc1
##    sse_excsh = stabScoreExc_excsh0, stabScoreExc_excsh1
#    sse_excsh = stabScoreExc_excsh0_avExcsh, stabScoreExc_excsh1_avExcsh
#    ssa = stabScoreAllExc0, stabScoreAllExc1        
#    ssl = stabLab0, stabLab1
    

    
    #%% Set average of angles across days    
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
    
    angInh_av = np.nanmean(angleInh_aligned, axis=-1) # frames x frames
    angExc_av = np.nanmean(angleExc_aligned, axis=-1)
    angAllExc_av = np.nanmean(angleAllExc_aligned, axis=-1)
    # average and std across excSamps for each day
    angleExc_excsh_aligned_avExcsh = np.nanmean(angleExc_excsh_aligned, axis=2) # frs x frs x days
    angleExc_excsh_aligned_sdExcsh = np.nanstd(angleExc_excsh_aligned, axis=2) # frs x frs x days
    # now average of excSamp-averages across days
    angExc_excsh_av = np.nanmean(angleExc_excsh_aligned_avExcsh, axis=-1) # frs x frs 
    
    
    ######## shuffles 
    # average across days (already averaged across shuffles) # frames x frames
    angInhS_av = np.nanmean(angleInhS_aligned_avSh, axis=-1) # frames x frames
    angExcS_av = np.nanmean(angleExcS_aligned_avSh, axis=-1)
    angAllExcS_av = np.nanmean(angleAllExcS_aligned_avSh, axis=-1)
    # average and std across excSamps for each day
    angleExc_excshS_aligned_avExcsh = np.nanmean(angleExc_excshS_aligned_avSh, axis=2) # frs x frs x days
    angleExc_excshS_aligned_sdExcsh = np.nanstd(angleExc_excshS_aligned_avSh, axis=2) # frs x frs x days
    # now average of excSamp-averages across days
    angExc_excshS_av = np.nanmean(angleExc_excshS_aligned_avExcsh, axis=-1) # frs x frs     
#    angInhS_av = np.nanmean(angleInhS_aligned, axis=(2,-1)) # average across days and shuffles # frames x frames
#    angExcS_av = np.nanmean(angleExcS_aligned, axis=(2,-1))
#    angAllExcS_av = np.nanmean(angleAllExcS_aligned, axis=(2,-1))
    # ave shfls for each excSamp (for each day)
    
    
    ######## significance values
    #np.mean(sigAngInh, axis=-1)) # each element shows fraction of days in which angle between decoders was significant.
    # eg I will call an angle significant if in more than .5 of days it was significant.
    
    sigAngInh_av = np.nanmean(sigAngInh, axis=-1) # frames x frames
    sigAngExc_av = np.nanmean(sigAngExc, axis=-1)
    sigAngAllExc_av = np.nanmean(sigAngAllExc, axis=-1)
    # average and std across excSamps for each day
    sigAngExc_excsh_avExcsh = np.nanmean(sigAngExc_excsh, axis=2) # frs x frs x days
    sigAngExc_excsh_sdExcsh = np.nanstd(sigAngExc_excsh, axis=2) # frs x frs x days
    # now average of excSamp-averages across days
    sigAngExc_excsh_av = np.nanmean(sigAngExc_excsh_avExcsh, axis=-1) # frs x frs         
#    angAllExcSig_excsh_av = np.nanmean(sigAngExc_excsh, axis=-1) # frames x frames x excSamps

    
    # instab_data
    stabScoreInh0_data_av = np.nanmean(stabScoreInh0_data, axis=0)  # frs 
    stabScoreExc0_data_av = np.nanmean(stabScoreExc0_data, axis=0)
    stabScoreAllExc0_data_av = np.nanmean(stabScoreAllExc0_data, axis=0)
     # average and std across excSamps for each day     
    stabScoreExc_excsh0_data_avExcsh = np.nanmean(stabScoreExc_excsh0_data, axis=1)  # days x frs    
    stabScoreExc_excsh0_data_sdExcsh = np.nanstd(stabScoreExc_excsh0_data, axis=1)  # days x frs    
     # now average of excSamp-averages across days
    stabScoreExc_excsh0_data_av = np.nanmean(stabScoreExc_excsh0_data_avExcsh, axis=0) # frs
    
    
    # instab_shfl
    stabScoreInh0_shfl_av = np.nanmean(stabScoreInh0_shfl, axis=0)  # frs 
    stabScoreExc0_shfl_av = np.nanmean(stabScoreExc0_shfl, axis=0)
    stabScoreAllExc0_shfl_av = np.nanmean(stabScoreAllExc0_shfl, axis=0)
     # average and std across excSamps for each day     
    stabScoreExc_excsh0_shfl_avExcsh = np.nanmean(stabScoreExc_excsh0_shfl, axis=1)  # days x frs    
    stabScoreExc_excsh0_shfl_sdExcsh = np.nanstd(stabScoreExc_excsh0_shfl, axis=1)  # days x frs    
     # now average of excSamp-averages across days
    stabScoreExc_excsh0_shfl_av = np.nanmean(stabScoreExc_excsh0_shfl_avExcsh, axis=0) # frs
    
    
    
    #stab (shfl-data)
    stabScoreInh0_av = np.nanmean(stabScoreInh0, axis=0)  # frs 
    stabScoreExc0_av = np.nanmean(stabScoreExc0, axis=0)
    stabScoreAllExc0_av = np.nanmean(stabScoreAllExc0, axis=0)
     # average and std across excSamps for each day     
    stabScoreExc_excsh0_avExcsh = np.nanmean(stabScoreExc_excsh0, axis=1)  # days x frs    
    stabScoreExc_excsh0_sdExcsh = np.nanstd(stabScoreExc_excsh0, axis=1)  # days x frs    
     # now average of excSamp-averages across days
    stabScoreExc_excsh0_av = np.nanmean(stabScoreExc_excsh0_avExcsh, axis=0) # frs    
    
    
    # num aligned decoder
    stabScoreInh1_av = np.nanmean(stabScoreInh1, axis=0)  # frs 
    stabScoreExc1_av = np.nanmean(stabScoreExc1, axis=0)
    stabScoreAllExc1_av = np.nanmean(stabScoreAllExc1, axis=0)
     # average and std across excSamps for each day     
    stabScoreExc_excsh1_avExcsh = np.nanmean(stabScoreExc_excsh1, axis=1)  # days x frs    
    stabScoreExc_excsh1_sdExcsh = np.nanstd(stabScoreExc_excsh1, axis=1)  # days x frs    
     # now average of excSamp-averages across days
    stabScoreExc_excsh1_av = np.nanmean(stabScoreExc_excsh1_avExcsh, axis=0) # frs    
 
     
    '''
    ######### average of significancy percentiles (I believe diag elements are already nan, so we dont need to do the step of fill_diag)
    percAngInh_av = np.nanmean(percAngInh, axis=-1)
    percAngExc_av = np.nanmean(percAngExc, axis=-1)
    percAngAllExc_av = np.nanmean(percAngAllExc, axis=-1)   
    '''
    
    
    #%% Save vars for each mouse
    
    if saveVars:
        
        if shflTrLabs:
            shflTrLabs_n = 'shflTrLabs_'
        else:
            shflTrLabs_n = ''
            
        fname = os.path.join(os.path.dirname(os.path.dirname(imfilename)), 'analysis')    
        if not os.path.exists(fname):
            print 'creating folder'
            os.makedirs(fname)    
            
        finame = os.path.join(fname, ('svm_stability_%s%s.mat' %(shflTrLabs_n,nowStr)))
    
        scio.savemat(finame,{'angleInh_aligned':angleInh_aligned,
                        'angleExc_aligned':angleExc_aligned,
                        'angleAllExc_aligned':angleAllExc_aligned,
                        'angleExc_excsh_aligned':angleExc_excsh_aligned,
                        'angleExc_excsh_aligned_avExcsh':angleExc_excsh_aligned_avExcsh,
                        'angleExc_excsh_aligned_sdExcsh':angleExc_excsh_aligned_sdExcsh,
                        'angleInhS_aligned_avSh':angleInhS_aligned_avSh,
                        'angleExcS_aligned_avSh':angleExcS_aligned_avSh,
                        'angleAllExcS_aligned_avSh':angleAllExcS_aligned_avSh,
                        'angleExc_excshS_aligned_avSh':angleExc_excshS_aligned_avSh,
                        'angleExc_excshS_aligned_avExcsh':angleExc_excshS_aligned_avExcsh,
                        'angleExc_excshS_aligned_sdExcsh':angleExc_excshS_aligned_sdExcsh,
                        'sigAngInh':sigAngInh,
                        'sigAngExc':sigAngExc,
                        'sigAngAllExc':sigAngAllExc,
                        'sigAngExc_excsh':sigAngExc_excsh,
                        'sigAngExc_excsh_avExcsh':sigAngExc_excsh_avExcsh,
                        'sigAngExc_excsh_sdExcsh':sigAngExc_excsh_sdExcsh,
                        'stabScoreInh0_data':stabScoreInh0_data,
                        'stabScoreExc0_data':stabScoreExc0_data,
                        'stabScoreAllExc0_data':stabScoreAllExc0_data,
                        'stabScoreExc_excsh0_data':stabScoreExc_excsh0_data,
                        'stabScoreExc_excsh0_data_avExcsh':stabScoreExc_excsh0_data_avExcsh,
                        'stabScoreExc_excsh0_data_sdExcsh':stabScoreExc_excsh0_data_sdExcsh,
                        'stabScoreInh0_shfl':stabScoreInh0_shfl,
                        'stabScoreExc0_shfl':stabScoreExc0_shfl,
                        'stabScoreAllExc0_shfl':stabScoreAllExc0_shfl,
                        'stabScoreExc_excsh0_shfl':stabScoreExc_excsh0_shfl,
                        'stabScoreExc_excsh0_shfl_avExcsh':stabScoreExc_excsh0_shfl_avExcsh,
                        'stabScoreExc_excsh0_shfl_sdExcsh':stabScoreExc_excsh0_shfl_sdExcsh,
                        'stabScoreInh0':stabScoreInh0,
                        'stabScoreExc0':stabScoreExc0,
                        'stabScoreAllExc0':stabScoreAllExc0,
                        'stabScoreExc_excsh0':stabScoreExc_excsh0,
                        'stabScoreExc_excsh0_avExcsh':stabScoreExc_excsh0_avExcsh,
                        'stabScoreExc_excsh0_sdExcsh':stabScoreExc_excsh0_sdExcsh,
                        'stabScoreInh1':stabScoreInh1,
                        'stabScoreExc1':stabScoreExc1,
                        'stabScoreAllExc1':stabScoreAllExc1,
                        'stabScoreExc_excsh1':stabScoreExc_excsh1,
                        'stabScoreExc_excsh1_avExcsh':stabScoreExc_excsh1_avExcsh,
                        'stabScoreExc_excsh1_sdExcsh':stabScoreExc_excsh1_sdExcsh,
                        'angInh_av':angInh_av,
                        'angExc_av':angExc_av,
                        'angAllExc_av':angAllExc_av,
                        'angExc_excsh_av':angExc_excsh_av,
                        'angInhS_av':angInhS_av,
                        'angExcS_av':angExcS_av,
                        'angAllExcS_av':angAllExcS_av,
                        'angExc_excshS_av':angExc_excshS_av,
                        'sigAngInh_av':sigAngInh_av,
                        'sigAngExc_av':sigAngExc_av,
                        'sigAngAllExc_av':sigAngAllExc_av,
                        'sigAngExc_excsh_av':sigAngExc_excsh_av,
                        'stabScoreInh0_data_av':stabScoreInh0_data_av,
                        'stabScoreExc0_data_av':stabScoreExc0_data_av,
                        'stabScoreAllExc0_data_av':stabScoreAllExc0_data_av,
                        'stabScoreExc_excsh0_data_av':stabScoreExc_excsh0_data_av,
                        'stabScoreInh0_shfl_av':stabScoreInh0_shfl_av,
                        'stabScoreExc0_shfl_av':stabScoreExc0_shfl_av,
                        'stabScoreAllExc0_shfl_av':stabScoreAllExc0_shfl_av,
                        'stabScoreExc_excsh0_shfl_av':stabScoreExc_excsh0_shfl_av,
                        'stabScoreInh0_av':stabScoreInh0_av,
                        'stabScoreExc0_av':stabScoreExc0_av,
                        'stabScoreAllExc0_av':stabScoreAllExc0_av,
                        'stabScoreExc_excsh0_av':stabScoreExc_excsh0_av,
                        'stabScoreInh1_av':stabScoreInh1_av,
                        'stabScoreExc1_av':stabScoreExc1_av,
                        'stabScoreAllExc1_av':stabScoreAllExc1_av,
                        'stabScoreExc_excsh1_av':stabScoreExc_excsh1_av,
                        'corr_hr_lr':corr_hr_lr,
                        'eventI_allDays':eventI_allDays,
                        'eventI_ds_allDays':eventI_ds_allDays,
                        'behCorr_all':behCorr_all,
                        'behCorrHR_all':behCorrHR_all,
                        'behCorrLR_all':behCorrLR_all,
                        'classAccurTMS_inh':classAccurTMS_inh,
                        'classAccurTMS_exc':classAccurTMS_exc,
                        'classAccurTMS_allExc':classAccurTMS_allExc})       
        


        