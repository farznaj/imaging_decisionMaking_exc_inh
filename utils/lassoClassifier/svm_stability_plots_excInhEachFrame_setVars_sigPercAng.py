# -*- coding: utf-8 -*-
# This script is like svm_stability_plots_excInhEachFrame_setVars.py EXCEPT:
# it only sets sigAng variables (like sigAngInh) (in case you want to use a different sigPerc than what was used and saved in the script above)... remember you are not saving 
# shuffled angle vars (like angleInhS_aligned) bc they are large matrices!... so you will have to run this script if you want to set sigAngs using a different percentile for significancy!

# This script does not set any var other sigAng vars


"""
Created on Tue Nov  7 14:26:57 2017

@author: farznaj
"""

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

loadWeights = 2
sigPerc = 1 # 5 # percentile to determine significancy (if data angle is lower than 5th percentile of shuffle angles, we calle it siginificantly lower!)

saveVars = 1 # if 1, a mat file will be saved including angle variables

nsh = 1000 # number of times to shuffle the decoders to get null distribution of angles

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
excludeLowTrDays = 1 # remove days with too few trials

shflTrsEachNeuron = 0
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
trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  

import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.
#dnow0 = '/stability/'


execfile("defFuns.py")


           
#%% 
#####################################################################################################################################################   
###############################################################################################################################     
#####################################################################################################################################################

numDaysAll = np.full(len(mice), np.nan, dtype=int)
days_allMice = [] 
eventI_ds_allMice = [] # you need this var to align traces of all mice

# im = 0     
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
#    execfile("svm_plots_setVars.py")      
    days_allMice.append(days)
    numDaysAll[im] = len(days)
   
    dnow = '/stability/'+mousename+'/'

            
    angleInh_all = []
    angleExc_all = []
    angleAllExc_all = []
    angleInhS_all = []
    angleExcS_all = []
    angleAllExcS_all = []
    angleExc_excsh_all = []
    angleExc_excshS_all = []    
    corr_hr_lr = np.full((len(days),2), np.nan) # number of hr, lr correct trials for each day
    eventI_ds_allDays = np.full((len(days)), np.nan)    
    eventI_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
    
    
    #%%
#    iday = 15
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



        #%% Set eventI_ds (downsampled eventI)

        eventI, eventI_ds = setEventIds(postName, chAl, regressBins=3, trialHistAnalysis=0)
        
        eventI_allDays[iday] = eventI
        eventI_ds_allDays[iday] = eventI_ds


        #%% Load SVM vars : loadSVM_excInh
    
        perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN = \
            loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron)
        
        
        ####%% Take care of lastTimeBinMissed = 0 #1# if 0, things were ran fine; if 1: by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
        ### allN data was run with lastTimeBinMissed = 1 
        if w_data_inh.shape[2] != w_data_allExc.shape[2]:
            if w_data_inh.shape[2] - w_data_allExc.shape[2] == 1:
                print '================== lastTimeBinMissed=1 for allN =================='
                print '======== removing last element from inh/exc to match the size with allN ========'
                

                w_data_inh = np.delete(w_data_inh, -1, axis=-1)
                w_data_exc = np.delete(w_data_exc, -1, axis=-1)
                b_data_inh = np.delete(b_data_inh, -1, axis=-1)
                b_data_exc = np.delete(b_data_exc, -1, axis=-1)
                
            else:
                sys.exit('something wrong')
                

        numExcSamples = w_data_inh.shape[0]
        

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
        
        # allExc
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
         
        del w_data_inh, w_data_allExc, w_data_exc #, perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc
    
    
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
   
   

    #%%
    ##################################################################################################
    ############## Align class accur traces of all days to make a final average trace ##############
    ################################################################################################## 
      
    ##%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
    # By common eventI, we  mean the index on which all traces will be aligned.
    
    time_aligned, nPreMin, nPostMin = set_nprepost(angleInh_all, eventI_ds_allDays, mn_corr, thTrained, regressBins)  # av_test_data_inh

    eventI_ds_allMice.append(nPreMin) # you need this var to align traces of all mice
    
    
    
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
    
#    pvals = np.arange(0,99,1) # percentile values # I will use 90 degrees as the 100th percentile, that's why I dont do np.arange(0,100,1)
#    percAngInh = np.full((np.shape(angleExc_aligned)), np.nan)
#    percAngExc = np.full((np.shape(angleExc_aligned)), np.nan)
#    percAngAllExc = np.full((np.shape(angleExc_aligned)), np.nan)
#    percAngExc_excsh = np.full((np.shape(angleExc_excsh_aligned)), np.nan) # frs x frs x excSamps x days
    
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
#                percAngInh[f1,f2,iday] = 100 - np.argwhere(angleInh_aligned[f1,f2, iday] - np.concatenate((np.percentile(angleInhS_aligned[f1,f2,:, iday], pvals), [90])) < 0)[0]
#                percAngExc[f1,f2,iday] = 100 - np.argwhere(angleExc_aligned[f1,f2, iday] - np.concatenate((np.percentile(angleExcS_aligned[f1,f2,:, iday], pvals), [90])) < 0)[0]
#                percAngAllExc[f1,f2,iday] = 100 - np.argwhere(angleAllExc_aligned[f1,f2, iday] - np.concatenate((np.percentile(angleAllExcS_aligned[f1,f2,:, iday], pvals), [90])) < 0)[0]
#                
#                aa = np.percentile(angleExc_excshS_aligned[f1,f2,:,:, iday], pvals, axis=0)  # 99 x excSamps       
#                a = np.array([np.concatenate((aa[:,ies] , [90])) for ies in range(numExcSamples)]) # excSamps x 100
#                b = (angleExc_excsh_aligned[f1,f2, :,iday] - a.T).T  # excSamps x 100
#                percAngExc_excsh[f1,f2,:,iday] = 100 - np.array([np.argwhere(b[ies] < 0)[0] for ies in range(numExcSamples)]).flatten()  # excSamps

    #np.mean(sigAngInh, axis=-1)) # each element shows fraction of days in which angle between decoders was significant.
    # eg I will call an angle siginificant if in >=.5 of days it was significant.
    #plt.imshow(np.mean(sigAngInh, axis=-1)); plt.colorbar() # fraction of days in which angle between decoders was significant.
    #plt.imshow(np.mean(angleInh_aligned, axis=-1)); plt.colorbar()        
    

    
    #%% Set average of angles across days    
   
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



    #%% Set number of decoders that are "aligned" with each decoder (at each timepoint) (we get this by summing sigAng across rows gives us the)
    # aligned decoder is defined as a decoder whose angle with the decoder of interest is <5th percentile of dist of shuffled angles.

    # sigAng: for each day shows whether angle betweeen the decoders at specific timepoints is significantly different than null dist (ie angle is smaller than 5th percentile of shuffle angles).
    stabScoreInh1 = np.array([np.nansum(sigAngInh[:,:,iday], axis=0) for iday in range(numGoodDays)]) # days x frs
    stabScoreExc1 = np.array([np.nansum(sigAngExc[:,:,iday], axis=0) for iday in range(numGoodDays)]) 
    stabScoreAllExc1 = np.array([np.nansum(sigAngAllExc[:,:,iday], axis=0) for iday in range(numGoodDays)]) 
    stabScoreExc_excsh1 = np.transpose(np.array([np.nansum(sigAngExc_excsh[:,:,:,iday], axis=0) for iday in range(numGoodDays)]), (0,2,1)) # days x excSamps x frs
#    stabLab1 = 'Stability (number of aligned decoders)' # with the decoder of interest (ie the decoder at the specific timepont we are looking at)            

    #### Averages across days
    # num aligned decoder
    stabScoreInh1_av = np.nanmean(stabScoreInh1, axis=0)  # frs 
    stabScoreExc1_av = np.nanmean(stabScoreExc1, axis=0)
    stabScoreAllExc1_av = np.nanmean(stabScoreAllExc1, axis=0)
     # average and std across excSamps for each day     
    stabScoreExc_excsh1_avExcsh = np.nanmean(stabScoreExc_excsh1, axis=1)  # days x frs    
    stabScoreExc_excsh1_sdExcsh = np.nanstd(stabScoreExc_excsh1, axis=1)  # days x frs    
     # now average of excSamp-averages across days
    stabScoreExc_excsh1_av = np.nanmean(stabScoreExc_excsh1_avExcsh, axis=0) # frs   

           
    #%% Save vars for each mouse
    
    if saveVars:
        fname = os.path.join(os.path.dirname(os.path.dirname(imfilename)), 'analysis')    
        if not os.path.exists(fname):
            print 'creating folder'
            os.makedirs(fname)    
            
        finame = os.path.join(fname, ('svm_stability_sigAngPerc%d_%s.mat' %(sigPerc, nowStr)))
    
        scio.savemat(finame,{'sigAngInh':sigAngInh,
                        'sigAngExc':sigAngExc,
                        'sigAngAllExc':sigAngAllExc,
                        'sigAngExc_excsh':sigAngExc_excsh,
                        'sigAngExc_excsh_avExcsh':sigAngExc_excsh_avExcsh,
                        'sigAngExc_excsh_sdExcsh':sigAngExc_excsh_sdExcsh,
                        'sigAngInh_av':sigAngInh_av,
                        'sigAngExc_av':sigAngExc_av,
                        'sigAngAllExc_av':sigAngAllExc_av,
                        'sigAngExc_excsh_av':sigAngExc_excsh_av,
                        'stabScoreInh1':stabScoreInh1,
                        'stabScoreExc1':stabScoreExc1,
                        'stabScoreAllExc1':stabScoreAllExc1,
                        'stabScoreExc_excsh1':stabScoreExc_excsh1,
                        'stabScoreExc_excsh1_avExcsh':stabScoreExc_excsh1_avExcsh,
                        'stabScoreExc_excsh1_sdExcsh':stabScoreExc_excsh1_sdExcsh,
                        'stabScoreInh1_av':stabScoreInh1_av,
                        'stabScoreExc1_av':stabScoreExc1_av,
                        'stabScoreAllExc1_av':stabScoreAllExc1_av,
                        'stabScoreExc_excsh1_av':stabScoreExc_excsh1_av})       
                        
        