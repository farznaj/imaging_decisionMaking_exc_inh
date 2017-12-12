# -*- coding: utf-8 -*-
"""
the parts of script "svm_stability_plots_excInhEachFrame_setVars.py" that set behavioral vars and class accuracy vars are copied here (but not the angle vars). This is bc
in the first run of the the script mentioned above, the behavioral and class accuracy vars were not saved properly... so I am again saving them here for each mouse.

Created on Thu Nov  2 12:02:46 2017

@author: farznaj
"""

# -*- coding: utf-8 -*-
"""
compute angles between weights found in script excInh_SVMtrained_eachFrame, ie
weights computed by training the decoder using only only inh or only exc neurons.


Plots class accuracy for svm trained on non-overlapping time windows  (outputs of file svm_eachFrame.py)
 ... svm trained to decode choice on choice-aligned or stimulus-aligned traces.
 
 
Remember for fni18 there are 2 svm_eachFrame mat files, the earlier file is using all trials (unequal HR, LR, like how you've done all your analysis). 
The later mat file is with equal number of hr and lr trials (subselecting trials)... this helped with 151209 class accur trace which was weird in the earlier mat file.
 
Created on Sun Mar 12 15:12:29 2017
@author: farznaj
"""     

#%% Set the following vars:

mice = 'fni16', 'fni17', 'fni18', 'fni19' # 'fni17',

saveVars = 0 # if 1, a mat file will be saved including angle variables
doCA = 0 # only compute beh vars or also do svm class accur vars
loadWeights = 0

thQStimStrength = [3]#3 # 0 to 3 : hard to easy # # set to nan if you want to include all strengths in computing behaioral performance
# quantile of stim strength - cb to use         
# 0: hard (min <= sr < 25th percentile of stimrate-cb); 
# 1: medium hard (25th<= sr < 50th); 
# 2: medium easy (50th<= sr < 75th); 
# 3: easy (75th<= sr <= max);


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
dnow0 = '/stability/'
excludeLowTrDays = 1 # remove days with too few trials
doPlotsEachMouse = 1 # make plots for each mouse
doExcSamps = 1 # if 1, for exc use vars defined by averaging trial subselects first for each exc samp and then averaging across exc samps. This is better than pooling all trial samps and exc samps ...    
savefigs = 0

nsh = 1000 # number of times to shuffle the decoders to get null distribution of angles


execfile("defFuns.py")


           
#%% 
#####################################################################################################################################################   
###############################################################################################################################     
#####################################################################################################################################################

behCorr_allMice = [] 
behCorrHR_allMice = [] 
behCorrLR_allMice = [] 

eventI_ds_allDays_allMice = []
eventI_allDays_allMice = []

numDaysAll = np.full(len(mice), np.nan, dtype=int)
days_allMice = [] 
corr_hr_lr_allMice = []
     
#%%
# im = 0     
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
   
        
#    execfile("svm_plots_setVars_n.py")     #    execfile("svm_plots_setVars.py")      
    days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays)
    days_allMice.append(days)
    numDaysAll[im] = len(days)
   
    dnow = '/stability/'+mousename+'/'

            
    #%% Loop over days for each mouse   
    
    if doCA:
        perClassErrorTest_data_inh_all = []
        perClassErrorTest_shfl_inh_all = []
        perClassErrorTest_chance_inh_all = []
        perClassErrorTest_data_allExc_all = []
        perClassErrorTest_shfl_allExc_all = []
        perClassErrorTest_chance_allExc_all = []
        perClassErrorTest_data_exc_all = []
        perClassErrorTest_shfl_exc_all = []
        perClassErrorTest_chance_exc_all = []
        corr_hr_lr = np.full((len(days),2), np.nan) # number of hr, lr correct trials for each day
        eventI_ds_allDays = np.full((len(days)), np.nan)    
        eventI_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
    
    behCorr_all = []
    behCorrHR_all = []
    behCorrLR_all = []    
    
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
        if doCA:
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
        
        if ~np.isnan(thQStimStrength).all(): # set to nan if you want to include all strengths in computing behaioral performance
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
            snh = (np.in1d(inds, thQStimStrength)) # index of easy hr trials
            # LR:
            inds = 4 - np.digitize(stimrate, thLR)
            snl = (np.in1d(inds, thQStimStrength)) # index of easy hr trials

            str2ana = snh+snl # index of trials with stim strength of interest    

        else:
            str2ana = np.full(np.shape(outcomes), True).astype('bool')
            
        print 'Number of trials with stim strength of interest = %i' %(str2ana.sum())
        print 'Stim rates for training = {}'.format(np.unique(stimrate[str2ana]))

        # now you have easy trials... see what percentage of them are correct         # remember only the corr and incorr trials were used for svm training ... doesnt really matter here...
        v = outcomes[str2ana]
        v = v[v>=0] # only use corr or incorr trials
#        if len(v) < thTrained: #10:
#            sys.exit(('Too few easy trials; only %d. Is behavioral performance meaningful?!'  %(len(v))))
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

        
        if doCA:
            #%% Set eventI_ds (downsampled eventI)
    
            eventI, eventI_ds = setEventIds(postName, chAl, regressBins=3, trialHistAnalysis=0)
            
            eventI_allDays[iday] = eventI
            eventI_ds_allDays[iday] = eventI_ds
    
    
            #%% Load SVM vars : loadSVM_excInh
        
            perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN = \
                loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, 0, doIncorr, 0, doAllN, useEqualTrNums, shflTrsEachNeuron)
                    
            ####%% Take care of lastTimeBinMissed = 0 #1# if 0, things were ran fine; if 1: by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
            ### allN data was run with lastTimeBinMissed = 1 
            if perClassErrorTest_chance_inh.shape[1] != perClassErrorTest_chance_allExc.shape[1]:
                if perClassErrorTest_chance_inh.shape[1] - perClassErrorTest_chance_allExc.shape[1] == 1:
                    print '================== lastTimeBinMissed=1 for allN =================='
                    print '======== removing last element from inh/exc to match the size with allN ========'
                    
                    perClassErrorTest_data_inh = np.delete(perClassErrorTest_data_inh, -1, axis=-1)
                    perClassErrorTest_shfl_inh = np.delete(perClassErrorTest_shfl_inh, -1, axis=-1)
                    perClassErrorTest_chance_inh = np.delete(perClassErrorTest_chance_inh, -1, axis=-1)
                    perClassErrorTest_data_exc = np.delete(perClassErrorTest_data_exc, -1, axis=-1)
                    perClassErrorTest_shfl_exc = np.delete(perClassErrorTest_shfl_exc, -1, axis=-1)
                    perClassErrorTest_chance_exc = np.delete(perClassErrorTest_chance_exc, -1, axis=-1)
    
                    
                else:
                    sys.exit('something wrong')
                    
    
            
            #%% Keep vars for all days
            
            perClassErrorTest_data_inh_all.append(perClassErrorTest_data_inh) # each day: samps x numFrs    
            perClassErrorTest_shfl_inh_all.append(perClassErrorTest_shfl_inh)
            perClassErrorTest_chance_inh_all.append(perClassErrorTest_chance_inh)
            
            perClassErrorTest_data_allExc_all.append(perClassErrorTest_data_allExc) # each day: samps x numFrs    
            perClassErrorTest_shfl_allExc_all.append(perClassErrorTest_shfl_allExc)
            perClassErrorTest_chance_allExc_all.append(perClassErrorTest_chance_allExc) 
    
            perClassErrorTest_data_exc_all.append(perClassErrorTest_data_exc) # each day: numShufflesExc x numSamples x numFrames    
            perClassErrorTest_shfl_exc_all.append(perClassErrorTest_shfl_exc)
            perClassErrorTest_chance_exc_all.append(perClassErrorTest_chance_exc)
        
        
        
            #%% Done with all days, keep values of all mice
            
            eventI_ds_allDays = eventI_ds_allDays.astype('int')
            eventI_allDays = eventI_allDays.astype('int')
        
            
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
            ######################################################################################################################################################    
            ######################################################################################################################################################          
            ##%% Average and st error of class accuracies across CV samples ... for each day
            
            numSamples, numExcSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, av_test_data_exc, sd_test_data_exc, av_test_shfl_exc, sd_test_shfl_exc, av_test_chance_exc, sd_test_chance_exc, av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc \
                = av_se_CA_trsamps(numOrigDays, perClassErrorTest_data_inh_all, perClassErrorTest_shfl_inh_all, perClassErrorTest_chance_inh_all, perClassErrorTest_data_exc_all, perClassErrorTest_shfl_exc_all, perClassErrorTest_chance_exc_all, perClassErrorTest_data_allExc_all, perClassErrorTest_shfl_allExc_all, perClassErrorTest_chance_allExc_all)
        
            
            #%%
            ##################################################################################################
            ############## Align class accur traces of all days to make a final average trace ##############
            ################################################################################################## 
              
            ##%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
            # By common eventI, we  mean the index on which all traces will be aligned.
            
            time_aligned, nPreMin, nPostMin = set_nprepost(av_test_data_inh, eventI_ds_allDays, mn_corr, thTrained, regressBins)
        
            
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
        
        
    #%% Save vars for each mouse
    
    if saveVars:
        fname = os.path.join(os.path.dirname(os.path.dirname(imfilename)), 'analysis')    
        if not os.path.exists(fname):
            print 'creating folder'
            os.makedirs(fname)    
            
        finame = os.path.join(fname, ('svm_stabilityBehCA_%s.mat' %nowStr))
    
        scio.savemat(finame,{'behCorr_all':behCorr_all,
                        'behCorrHR_all':behCorrHR_all,
                        'behCorrLR_all':behCorrLR_all,
                        'classAccurTMS_inh':classAccurTMS_inh,
                        'classAccurTMS_exc':classAccurTMS_exc,
                        'classAccurTMS_allExc':classAccurTMS_allExc})       
    


    #%%
    behCorr_allMice.append(np.array(behCorr_all))
    behCorrHR_allMice.append(np.array(behCorrHR_all))
    behCorrLR_allMice.append(np.array(behCorrLR_all))           
    
    
#%%
behCorr_allMice = np.array(behCorr_allMice)
behCorrHR_allMice = np.array(behCorrHR_allMice)
behCorrLR_allMice = np.array(behCorrLR_allMice)

    