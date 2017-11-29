# -*- coding: utf-8 -*-
"""
compute angle between decoders at different time points; 
It uses the weights computed in script svmCurrChoice_eachFrame 
for inh stability, it only gets inh weights
for exc stability, it only gets exc weights
not sure if this is legitimate to do; in svm_stability_plots_n2 we compute angles between weights found in script excInh_SVMtrained_eachFrame, ie
weights computed by training the decoder using only only inh or only exc neurons.


 
Remember for fni18 there are 2 svm_eachFrame mat files, the earlier file is using all trials (unequal HR, LR, like how you've done all your analysis). 
The later mat file is with equal number of hr and lr trials (subselecting trials)... this helped with 151209 class accur trace which was weird in the earlier mat file.
 
Created on Sun Mar 12 15:12:29 2017
@author: farznaj
"""

#%%
mice = 'fni17', #'fni16', 'fni17', 'fni18', 'fni19'

ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
savefigs = 0
#superimpose = 1 # the averaged aligned traces of testing and shuffled will be plotted on the same figure
#loadWeights = 1
doPlots = 1 #1 # plot angles for each day
# following will be needed for 'fni18': #set one of the following to 1:
allDays = 0# all 7 days will be used (last 3 days have z motion!)
noZmotionDays = 1 # 4 days that dont have z motion will be used.
noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!
noExtraStimDays = 1 # for fni19; the 1st 7 days wont be used (0922 to 0930); if u want all days set allDays=1.

time2an = -1 # relative to eventI, look at classErr in what time stamp.
poolDaysOfMice = 0 # for plotting w traces; if 0, average and sd across mice (each mouse has session-averaged weights); if 1, average and sd across all sessions of all mice

trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
#eps = 10**-10 # tiny number below which weight is considered 0
#thNon0Ws = 2 # For samples with <2 non0 weights, we manually set their class error to 50 ... the idea is that bc of difference in number of HR and LR trials, in these samples class error is not accurately computed!
#thSamps = 10  # Days that have <thSamps samples that satisfy >=thNon0W non0 weights will be manually set to 50 (class error of all their samples) ... bc we think <5 samples will not give us an accurate measure of class error of a day.
#setTo50 = 1 # if 1, the above two jobs will be done.
svmfold='svm'
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
    
#dnow = '/stability'       

execfile("defFuns.py")  



#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName

def setSVMname(pnevFileName, trialHistAnalysis, chAl, regressBins=3, svmfold='svm'):
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
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), svmfold, svmn))
    svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    
#    if len(svmName)>0:
#        svmName = svmName[0] # get the latest file
#    else:
#        svmName = ''
#    
    return svmName
    

    
    
#%% 
'''
#####################################################################################################################################################   
###############################################################################################################################     
#####################################################################################################################################################
'''

#%%

eventI_allDays_allMice = []
numDaysAll = np.full(len(mice), np.nan)
time_trace_all_allMice = []
angle_all_allMice = []
angleInh_all_allMice = []
angleExc_all_allMice = []
classErr_bestC_test_data_all = []
classErr_bestC_test_shfl_all = []
classErr_bestC_test_chance_all = []
 
       
#%%
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
    numDaysAll[im] = len(days)

    
    #%%     
    dnow = '/stability/'+mousename+'/'

            
    #%% Loop over days for each mouse   
    
    eventI_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
    angle_all = []
    angleInh_all = []
    angleExc_all = []    
    cbestFrs_all = []
    time_trace_all = []
    
    for iday in range(len(days)): 
    
        #%%            
        print '___________________'
        imagingFolder = days[iday][0:6]; #'151013'
        mdfFileNumber = map(int, (days[iday][7:]).split("-")); #[1,2] 
            
        ##%% Set .mat file names
        pnev2load = [] #[] [3] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.
        signalCh = 2 #[2] # since gcamp is channel 2, should be always 2.
        postNProvided = 1; # If your directory does not contain pnevFile and instead it contains postFile, set this to 1 to get pnevFileName
        
        # from setImagingAnalysisNamesP import *
        
        imfilename, pnevFileName, dataPath = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load, postNProvided=postNProvided, nOuts=3)
        
        postName = os.path.join(os.path.dirname(pnevFileName), 'post_'+os.path.basename(pnevFileName))
        moreName = os.path.join(os.path.dirname(pnevFileName), 'more_'+os.path.basename(pnevFileName))
        
        print(os.path.basename(imfilename))
    
    
        #%%
        svmName = setSVMname(pnevFileName, trialHistAnalysis, chAl, regressBins, svmfold) # for chAl: the latest file is with soft norm; earlier file is 
    
        svmName = svmName[0]
        print os.path.basename(svmName)    
    
    
        #%% Set eventI
        
        if chAl==1:    #%% Use choice-aligned traces 
            # Load 1stSideTry-aligned traces, frames, frame of event of interest
            # use firstSideTryAl_COM to look at changes-of-mind (mouse made a side lick without committing it)
            Data = scio.loadmat(postName, variable_names=['firstSideTryAl'],squeeze_me=True,struct_as_record=False)
        #    traces_al_1stSide = Data['firstSideTryAl'].traces.astype('float')
            time_aligned_1stSide = Data['firstSideTryAl'].time.astype('float')
            time_trace = time_aligned_1stSide
            
        else:   #%% Use stimulus-aligned traces           
            # Load stim-aligned_allTrials traces, frames, frame of event of interest
            if trialHistAnalysis==0:
                Data = scio.loadmat(postName, variable_names=['stimAl_noEarlyDec'],squeeze_me=True,struct_as_record=False)
    #            eventI = Data['stimAl_noEarlyDec'].eventI - 1 # remember difference indexing in matlab and python!
    #            traces_al_stimAll = Data['stimAl_noEarlyDec'].traces.astype('float')
                time_aligned_stim = Data['stimAl_noEarlyDec'].time.astype('float')        
            else:
                Data = scio.loadmat(postName, variable_names=['stimAl_allTrs'],squeeze_me=True,struct_as_record=False)
    #            eventI = Data['stimAl_allTrs'].eventI - 1 # remember difference indexing in matlab and python!
    #            traces_al_stimAll = Data['stimAl_allTrs'].traces.astype('float')
                time_aligned_stim = Data['stimAl_allTrs'].time.astype('float')
                # time_aligned_stimAll = Data['stimAl_allTrs'].time.astype('float') # same as time_aligned_stim        
            
            time_trace = time_aligned_stim
            
        print(np.shape(time_trace))
          
        
        ##%% Downsample traces: average across multiple times (downsampling, not a moving average. we only average every regressBins points.)
        
        if np.isnan(regressBins)==0: # set to nan if you don't want to downsample.
            print 'Downsampling traces ....'    
                
            T1 = time_trace.shape[0]
            tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X            
    
            time_trace = time_trace[0:regressBins*tt]
            time_trace = np.round(np.mean(np.reshape(time_trace, (regressBins, tt), order = 'F'), axis=0), 2)
            print time_trace.shape
        
            eventI_ds = np.argwhere(np.sign(time_trace)>0)[0] # frame in downsampled trace within which event_I happened (eg time1stSideTry)    
        
        else:
            print 'Not downsampling traces ....'        
    #        eventI_ch = Data['firstSideTryAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
    #        eventI_ds = eventI_ch
        
        eventI_allDays[iday] = eventI_ds
        time_trace_all.append(time_trace)
        

        #%%
        '''
        print 'Decide which one to load: inhibitRois_pix or inhibitRois!!'
        Data = scio.loadmat(moreName, variable_names=['inhibitRois'])
        if len(Data) == 3: # inhibitRois does not exist, only inhibitRois_pix exists!
            print 'inhibitRois does not exist. Loading inhibitRois_pix!'
            Data = scio.loadmat(moreName, variable_names=['inhibitRois_pix'])
            inhibitRois = Data.pop('inhibitRois_pix')[0,:]
        else:
            print 'Loading inhibitRois (not inhibitRois_pix)!!!'
            inhibitRois = Data.pop('inhibitRois')[0,:]
        '''
        
        Data = scio.loadmat(moreName, variable_names=['inhibitRois_pix'])
        inhibitRois = Data.pop('inhibitRois_pix')[0,:]    
#        Data = scio.loadmat(moreName, variable_names=['inhibitRois'])
#        inhibitRois = Data.pop('inhibitRois')[0,:]
    #    Data = scio.loadmat(svmName, variable_names=['NsExcluded_chAl'])        
    #    NsExcluded_chAl = Data.pop('NsExcluded_chAl')[0,:].astype('bool')
        # Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)        
        inhRois = inhibitRois #[~NsExcluded_chAl]   
        
                
        #%% Load SVM vars
            
        Data = scio.loadmat(svmName, variable_names=['regType','cvect','perClassErrorTest','perClassErrorTest_chance','perClassErrorTest_shfl','perClassErrorTrain'])
        
        regType = Data.pop('regType').astype('str')
        cvect = Data.pop('cvect').squeeze()
        perClassErrorTrain = Data.pop('perClassErrorTrain')
        perClassErrorTest = Data.pop('perClassErrorTest')
        perClassErrorTest_chance = Data.pop('perClassErrorTest_chance')
        perClassErrorTest_shfl = Data.pop('perClassErrorTest_shfl')
        
        numSamples = perClassErrorTest.shape[0] 
        numFrs = perClassErrorTest.shape[2]        
        
        Data = scio.loadmat(svmName, variable_names=['wAllC','bAllC'])
        wAllC = Data.pop('wAllC') # numSamples x len(cvect) x nNeurons x nFrames
        bAllC = Data.pop('bAllC') # numSamples x len(cvect) x nFrames    
          
            
        #%% Find bestc for each frame (funcion is defined in defFuns.py)
            
        cbestFrs = findBestC(perClassErrorTest, cvect, regType, smallestC) # nFrames        
    	
    	
        #%% Set class error values at best C (if desired plot c path) (funcion is defined in defFuns.py)
    
        doPlotscpath = 0;
#        classErr_bestC_train_data, classErr_bestC_test_data, classErr_bestC_test_shfl, classErr_bestC_test_chance, w_bestc_data, b_bestc_data = setClassErrAtBestc(cbestFrs, cvect, doPlotscpath, np.nan, np.nan, np.nan, np.nan, wAllC, bAllC)
        classErr_bestC_train_data, classErr_bestC_test_data, classErr_bestC_test_shfl, classErr_bestC_test_chance, w_bestc_data, b_bestc_data = setClassErrAtBestc(cbestFrs, cvect, doPlotscpath, perClassErrorTrain, perClassErrorTest,  perClassErrorTest_shfl,  perClassErrorTest_chance, wAllC, bAllC)
        
    
        #%% Set w including only inh or exc neurons
        
        winh0 = w_bestc_data[:,inhRois==1] # numSamples x nNeurons x nFrames
        wexc0 = w_bestc_data[:,inhRois==0]
       
        # Take average across samples (this is like getting a single decoder across all trial subselects... so I guess bagging)... (not ave of abs... the idea is that ave of shuffles represents the neurons weights for the decoder given the entire dataset (all trial subselects)... if a w switches signs across samples then it means this neuron is not very consistently contributing to the choice... so i think we should average its w and not its abs(w))
        winhAve = np.mean(winh0, axis=0) # neurons x frames
        wexcAve = np.mean(wexc0, axis=0)
        
        
        
        #%% Normalize weights (ie the w vector at each frame must have length 1)
        
        # all neurons
        w = w_bestc_data + 0 # numSamples x nNeurons x nFrames
        normw = np.linalg.norm(w, axis=1) # numSamples x frames ; 2-norm of weights 
        w_normed = np.transpose(np.transpose(w,(1,0,2))/normw, (1,0,2)) # numSamples x nNeurons x nFrames; normalize weights so weights (of each frame) have length 1
        if sum(normw<=eps).sum()!=0:
            print 'take care of this; you need to reshape w_normed first'
        #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero

        # inh
        w = winh0 + 0 # numSamples x nNeurons x nFrames
        normw = np.linalg.norm(w, axis=1) # numSamples x frames ; 2-norm of weights 
        wInh_normed = np.transpose(np.transpose(w,(1,0,2))/normw, (1,0,2)) # numSamples x nNeurons x nFrames; normalize weights so weights (of each frame) have length 1
        if sum(normw<=eps).sum()!=0:
            print 'take care of this; you need to reshape w_normed first'
        #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
        
        # exc            
        w = wexc0 + 0 # numSamples x nNeurons x nFrames
        normw = np.linalg.norm(w, axis=1) # numSamples x frames ; 2-norm of weights 
        wExc_normed = np.transpose(np.transpose(w,(1,0,2))/normw, (1,0,2)) # numSamples x nNeurons x nFrames; normalize weights so weights (of each frame) have length 1
        if sum(normw<=eps).sum()!=0:
            print 'take care of this; you need to reshape w_normed first'
        #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero

        
        #%% Set the final decoders by aggregating decoders across trial subselects (ie average ws across all trial subselects), then again normalize them so they have length 1 (Bagging)
        
        # all neurons
        w_nb = np.mean(w_normed, axis=(0)) # neurons x frames 
        nw = np.linalg.norm(w_nb, axis=0) # frames; 2-norm of weights 
        w_n2b = w_nb/nw # neurons x frames
                    
        # inh                    
        w_nb = np.mean(wInh_normed, axis=(0)) # neurons x frames 
        nw = np.linalg.norm(w_nb, axis=0) # frames; 2-norm of weights 
        wInh_n2b = w_nb/nw # neurons x frames

        # exc
        w_nb = np.mean(wExc_normed, axis=(0)) # neurons x frames 
        nw = np.linalg.norm(w_nb, axis=0) # frames; 2-norm of weights 
        wExc_n2b = w_nb/nw # neurons x frames

        
        #%% Compute angle between decoders (weights) at different times (remember angles close to 0 indicate more aligned decoders at 2 different time points.)
        
        # some of the angles are nan because the dot product is slightly above 1, so its arccos is nan!    
        # we restrict angles to [0 90], so we don't care about direction of vectors, ie tunning reversal.
        
        angle = np.arccos(abs(np.dot(w_n2b.transpose(), w_n2b)))*180/np.pi # frames x frames; angle between ws at different times
        angleInh = np.arccos(abs(np.dot(wInh_n2b.transpose(), wInh_n2b)))*180/np.pi # frames x frames; angle between ws at different times
        angleExc = np.arccos(abs(np.dot(wExc_n2b.transpose(), wExc_n2b)))*180/np.pi # frames x frames; angle between ws at different times
     
         
        #%% Plot angles for each day
         
        if doPlots:
            plt.figure()
            plt.subplot(221)
            plt.imshow(angle, cmap='jet_r')
            plt.colorbar()
            plt.title('all neurons')
            plt.plot([eventI_ds, eventI_ds], [0, len(angle)], color='r')
            plt.plot([0, len(angle)], [eventI_ds, eventI_ds], color='r')
            plt.xlim([0, len(angle)])
            plt.ylim([0, len(angle)][::-1])
            
            plt.subplot(222)
            plt.imshow(angleInh, cmap='jet_r')
            plt.colorbar()
            plt.title('inh')
            plt.plot([eventI_ds, eventI_ds], [0, len(angle)], color='r')
            plt.plot([0, len(angle)], [eventI_ds, eventI_ds], color='r')
            plt.xlim([0, len(angle)])
            plt.ylim([0, len(angle)][::-1])

            plt.subplot(224)
            plt.imshow(angleExc, cmap='jet_r')
            plt.colorbar()
            plt.title('exc')
            plt.plot([eventI_ds, eventI_ds], [0, len(angle)], color='r')
            plt.plot([0, len(angle)], [eventI_ds, eventI_ds], color='r')
            plt.xlim([0, len(angle)])
            plt.ylim([0, len(angle)][::-1])
            
            makeNicePlots(plt.gca())
            plt.subplots_adjust(hspace=.3)
            
        
        #%% Done with each day, keep vars of all days
        
        angle_all.append(angle)    
        angleInh_all.append(angleInh)
        angleExc_all.append(angleExc)
#        classErr_bestC_train_data_all.append(classErr_bestC_train_data) # each day: samps x numFrs
        classErr_bestC_test_data_all.append(classErr_bestC_test_data)
        classErr_bestC_test_shfl_all.append(classErr_bestC_test_shfl)
        classErr_bestC_test_chance_all.append(classErr_bestC_test_chance)
        cbestFrs_all.append(cbestFrs)     
           
        # Delete vars before starting the next day           
        del wAllC, perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance
            
    
    #%% Done with all days, keep values of all mice

    angle_all = np.array(angle_all)
    angleInh_all = np.array(angleInh_all)
    angleExc_all = np.array(angleExc_all)
    cbestFrs_all = np.array(cbestFrs_all)    
    numDaysAll = numDaysAll.astype(int)    
    eventI_allDays = eventI_allDays.astype('int')    
    time_trace_all = np.array(time_trace_all)
    
    
    angle_all_allMice.append(angle_all)
    angleInh_all_allMice.append(angleInh_all)
    angleExc_all_allMice.append(angleExc_all)
    eventI_allDays_allMice.append(eventI_allDays)        
    time_trace_all_allMice.append(time_trace_all)
    
   
    #%% Average and std of class accuracies across samples (all CV samples) ... for each day
    
    #    av_l2_train_d = np.array([100-np.nanmean(classErr_bestC_train_data_all[iday], axis=0) for iday in range(len(days))]) # numDays; each day has size numFrs
    #    sd_l2_train_d = np.array([np.nanstd(classErr_bestC_train_data_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(len(days))])  
    
    av_l2_test_d = np.array([100-np.nanmean(classErr_bestC_test_data_all[iday], axis=0) for iday in range(len(days))]) # numDays
    sd_l2_test_d = np.array([np.nanstd(classErr_bestC_test_data_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(len(days))])  
    
    av_l2_test_s = np.array([100-np.nanmean(classErr_bestC_test_shfl_all[iday], axis=0) for iday in range(len(days))]) # numDays
    sd_l2_test_s = np.array([np.nanstd(classErr_bestC_test_shfl_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(len(days))])  
    
    av_l2_test_c = np.array([100-np.nanmean(classErr_bestC_test_chance_all[iday], axis=0) for iday in range(len(days))]) # numDays
    sd_l2_test_c = np.array([np.nanstd(classErr_bestC_test_chance_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(len(days))])  

  

#%%    
eventI_allDays_allMice = np.array(eventI_allDays_allMice)
angle_all_allMice = np.array(angle_all_allMice)
angleInh_all_allMice = np.array(angleInh_all_allMice)
angleExc_all_allMice = np.array(angleExc_all_allMice)
    



    
#%%
##################################################################################################
############## Align angle traces of all days to make a final average trace ##############
################################################################################################## 
 
 
##%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
# By common eventI, we  mean the index on which all traces will be aligned.
        
nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
for iday in range(numDays):
    nPost[iday] = (len(angle_all[iday]) - eventI_allDays[iday] - 1)

nPreMin = min(eventI_allDays) # number of frames before the common eventI, also the index of common eventI. 
nPostMin = min(nPost)
print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)


#%% Set the time array for the across-day aligned traces

# this is wrong. ... look at evernote for an explanation of it!
'''
a = -(np.asarray(frameLength*regressBins) * range(nPreMin+1)[::-1])
b = (np.asarray(frameLength*regressBins) * range(1, nPostMin+1))
time_aligned = np.concatenate((a,b))
'''

##%% Align traces of all days on the common eventI
time_trace_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)

for iday in range(numDays):
    time_trace_aligned[:, iday] = time_trace_all[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    
time_aligned = np.mean(time_trace_aligned,1)


#%% Align angles of all days on the common eventI

r = nPreMin + nPostMin + 1

angle_aligned = np.ones((r,r, numDays)) + np.nan # frames x frames x days, aligned on common eventI (equals nPreMin)
angleInh_aligned = np.ones((r,r, numDays)) + np.nan
angleExc_aligned = np.ones((r,r, numDays)) + np.nan

for iday in range(numDays):
#    angleInh_aligned[:, iday] = angleInh_all[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1  ,  eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    inds = np.arange(eventI_allDays[iday] - nPreMin,  eventI_allDays[iday] + nPostMin + 1)
    
    angle_aligned[:,:,iday] = angle_all[iday][inds][:,inds]
    angleInh_aligned[:,:,iday] = angleInh_all[iday][inds][:,inds]
    angleExc_aligned[:,:,iday] = angleExc_all[iday][inds][:,inds]

        
#%% Average across days

ang_av = np.nanmean(angle_aligned, axis=-1)
angInh_av = np.nanmean(angleInh_aligned, axis=-1)
angExc_av = np.nanmean(angleExc_aligned, axis=-1)

### set diagonal elements (whose angles are 0) to nan, so heatmaps span a smaller range and can be better seen.
np.fill_diagonal(ang_av, np.nan)
np.fill_diagonal(angInh_av, np.nan)
np.fill_diagonal(angExc_av, np.nan)

cmin = np.floor(np.min([np.nanmin(ang_av), np.nanmin(angInh_av), np.nanmin(angExc_av)]))

#np.diagonal(ang_choice_av)
#for i in range(ang_choice_av.shape[0]):
#    ang_choice_av[i,i] = np.nan


#%% Plot averages across days (plot angle between decoders at different time points)

step = 4
x = (np.unique(np.concatenate((np.arange(np.argwhere(time_aligned>=0)[0], -.5, -step), 
           np.arange(np.argwhere(time_aligned>=0)[0], r, step))))).astype(int)
#x = np.arange(0,r,step)
if chAl==1:
    xl = 'Time since choice onset (ms)'
else:
    xl = 'Time since stimulus onset (ms)'

           
# choice
plt.figure(figsize=(8,8))

# all neurons
plt.subplot(221)
plt.imshow(ang_av, cmap='jet_r') #, interpolation='nearest') #, extent=time_aligned)
plt.colorbar()
plt.xticks(x, np.round(time_aligned[x]))
plt.yticks(x, np.round(time_aligned[x]))
makeNicePlots(plt.gca())
#ax.set_xticklabels(np.round(time_aligned[np.arange(0,60,10)]))
plt.title('all neurons')
plt.xlabel(xl)
plt.clim(cmin, 90)
plt.plot([nPreMin, nPreMin], [0, len(angle)], color='r')
plt.plot([0, len(time_aligned)], [nPreMin, nPreMin], color='r')
plt.xlim([0, len(time_aligned)])
plt.ylim([0, len(time_aligned)][::-1])


# inh    
plt.subplot(222)
plt.imshow(angInh_av, cmap='jet_r') #, interpolation='nearest') #, extent=time_aligned)
plt.colorbar()
plt.xticks(x, np.round(time_aligned[x]))
plt.yticks(x, np.round(time_aligned[x]))
makeNicePlots(plt.gca())
#ax.set_xticklabels(np.round(time_aligned[np.arange(0,60,10)]))
plt.title('inh')
plt.xlabel(xl)
plt.clim(cmin, 90)
plt.plot([nPreMin, nPreMin], [0, len(time_aligned)], color='r')
plt.plot([0, len(time_aligned)], [nPreMin, nPreMin], color='r')
plt.xlim([0, len(time_aligned)])
plt.ylim([0, len(time_aligned)][::-1])

         
# exc    
plt.subplot(224)
plt.imshow(angExc_av, cmap='jet_r') #, interpolation='nearest') #, extent=time_aligned)
plt.colorbar()
plt.xticks(x, np.round(time_aligned[x]))
plt.yticks(x, np.round(time_aligned[x]))
makeNicePlots(plt.gca())
#ax.set_xticklabels(np.round(time_aligned[np.arange(0,60,10)]))
plt.title('exc')
plt.xlabel(xl)
plt.clim(cmin, 90)
plt.plot([nPreMin, nPreMin], [0, len(time_aligned)], color='r')
plt.plot([0, len(time_aligned)], [nPreMin, nPreMin], color='r')
plt.xlim([0, len(time_aligned)])
plt.ylim([0, len(time_aligned)][::-1])
    
       
plt.subplots_adjust(hspace=.3)



if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnow) #,mousename)
       
    if chAl==1:
        dd = 'chAl_anglesAveDays_' + days[0][0:6] + '-to-' + days[-1][0:6]
    else:
        dd = 'stAl_anglesAveDays_' + days[0][0:6] + '-to-' + days[-1][0:6]
        
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
#    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    fign = os.path.join(d, suffn[0:5]+dd+'.'+fmt[0])

    plt.savefig(fign, bbox_inches='tight')

    




#%% Align traces of all days on the common eventI

#av_l2_train_d_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
av_l2_test_d_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
av_l2_test_s_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan
av_l2_test_c_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan

for iday in range(numDays):
#    av_l2_train_d_aligned[:, iday] = av_l2_train_d[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    av_l2_test_d_aligned[:, iday] = av_l2_test_d[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    av_l2_test_s_aligned[:, iday] = av_l2_test_s[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    av_l2_test_c_aligned[:, iday] = av_l2_test_c[iday][eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1]
    
        
           
#%% Set diagonal elements (whose angles are 0) to nan, so heatmaps span a smaller range and can be better seen.
           
[np.fill_diagonal(angle_aligned[:,:,iday], np.nan) for iday in range(len(days))];
[np.fill_diagonal(angleInh_aligned[:,:,iday], np.nan) for iday in range(len(days))];
[np.fill_diagonal(angleExc_aligned[:,:,iday], np.nan) for iday in range(len(days))];


#%%
th = 70  # how stable is each decoder at other time points measure as how many decoders are at 70 degrees or less with each of the decoders (trained on a particular frame).. or ... .
        
#np.mean(angle_aligned[:,:,iday],axis=0) # average stability ? (angle with all decoders at other time points)
#stabAng = angle_aligned[:,:,iday] < th # how stable is each decoder at other time points measure as how many decoders are at 70 degrees or less with each of the decoders (trained on a particular frame).. or ... .
stabScore = np.array([np.sum(angle_aligned[:,:,iday]<th, axis=0) for iday in range(len(days))]) 

classAccurTMS = np.array([(av_l2_test_d_aligned[:, iday] - av_l2_test_s_aligned[:, iday]) for iday in range(len(days))])


for iday in range(len(days)):
    plt.plot(stabScore[iday,:], classAccurTMS[iday,:])           
    plt.show()           
           

#%%
from matplotlib import cm
from numpy import linspace
start = 0.0
stop = 1.0
number_of_lines = len(days)
cm_subsection = linspace(start, stop, number_of_lines) 
colors = [ cm.jet(x) for x in cm_subsection ]
           
# change color order to jet 
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)


#%%
#plt.plot(stabScore.T);
#plt.legend(days, loc='center left', bbox_to_anchor=(1, .7))

step = 4
x = (np.unique(np.concatenate((np.arange(np.argwhere(time_aligned>=0)[0], -.5, -step), 
           np.arange(np.argwhere(time_aligned>=0)[0], r, step))))).astype(int)

plt.figure(figsize=(10,5))

plt.subplot(121)
plt.imshow(stabScore)
plt.plot([nPreMin, nPreMin], [0, len(days)], color='r')
plt.xlim([0, len(time_aligned)])
plt.ylim([0, len(days)][::-1])
plt.colorbar(label='Stability')
plt.xticks(x, np.round(time_aligned[x]).astype(int))
plt.ylabel('Days')
plt.xlabel(xl)
makeNicePlots(plt.gca())

plt.subplot(122)
plt.imshow(classAccurTMS)
plt.plot([nPreMin, nPreMin], [0, len(days)], color='r')
plt.xlim([0, len(time_aligned)])
plt.ylim([0, len(days)][::-1])
plt.colorbar(label='Class accuracy (data-shfl)')
plt.xticks(x, np.round(time_aligned[x]).astype(int))
plt.ylabel('Days')
plt.xlabel(xl)
makeNicePlots(plt.gca())



#%%
plt.plot(np.mean(stabScore, axis=0))
           
           
           
           