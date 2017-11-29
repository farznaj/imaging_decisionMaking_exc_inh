# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:36:50 2017

@author: farznaj
"""

mousename = 'fni17' #'fni17'
#ch_st_goAl = [0,1,0] # whether do analysis on traces aligned on choice, stim or go tone.
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

    
trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
execfile("defFuns.py")
#execfile("svm_plots_setVars_n.py")  
days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays)

#chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
#chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
#stAl = ch_st_goAl[1]
#goToneAl = ch_st_goAl[2]
#doPlots = 0 #1 # plot c path of each day 
#savefigs = 1

# Go to svm_plots_setVars and define vars!
#execfile("svm_plots_setVars.py")    
# if you want to remove some days:
#del(days[0])
#numDays = len(days)

savefigs = 0

eps = sys.float_info.epsilon #2.2204e-16
plotEachDay = 0 # if 1, when loading each day a plot of angles between decoders will be made too.
#doData = 0 # if 1, data file will be loaded; if 0, shuffle file will be loaded. Once run with doData=1 and then with doData=0, to get all data and shuffle vars ... then make plots.
#doShuffles = 1
#do_srChAl = 1 # vars for stimulus decoder on choice-aligned traces are saved in a separate mat file





#%%
dnow = '/stability'

if trialHistAnalysis:
    dp = '/previousChoice'
else:
    dp = '/currentChoice'


#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName

def setSVMname(pnevFileName, trialHistAnalysis, ntName, doData):
    import glob
    
    if trialHistAnalysis:
    #     ep_ms = np.round((ep-eventI)*frameLength)
        th_stim_dur = []
        svmn = 'svmPrevChoice_allN_allFrsTrained_' 
    else:
#        svmn = 'svmCurrChoice_allN_allFrsTrained_' 
        if doData:# and doShuffles==0:
            if do_srChAl==0:
                svmn = 'svmCurrChoice_allN_allFrsTrained_data_*' 
            else:
                svmn = 'svmCurrChoice_allN_allFrsTrained_srChAl_data_*' 
        elif doData==0:# and doShuffles:
            if do_srChAl==0:
                svmn = 'svmCurrChoice_allN_allFrsTrained_shfl_*' 
            else:
                svmn = 'svmCurrChoice_allN_allFrsTrained_srChAl_shfl_*'                         
    
    svmn = svmn + os.path.basename(pnevFileName) #pnevFileName[-32:]    
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
    svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    svmName = svmName[0] # get the latest file
    
    return svmName
    


###############################################################################
###############################################################################
###############################################################################

#%% ANALYSIS STARTS HERE (once run with doData=1, then with doData=0 to get both data and shuffled vars)

for do_srChAl in [0,1]:
    for doData in [0,1]:
        if doData:
            if do_srChAl==0:
                angle_choice_all = []
                angle_stim_all = []
                angle_choice_chAl_all = []
            else:
                angle_stim_chAl_all = []
            
        elif doData==0:
            if do_srChAl==0:        
                angle_choice_sh_all = []
                angle_stim_sh_all = []
                angle_choice_chAl_sh_all = []
            else:
                angle_stim_chAl_sh_all = []        
    
        
        for iday in range(len(days)):
        
            print '___________________'
            imagingFolder = days[iday][0:6]; #'151013'
            mdfFileNumber = map(int, (days[iday][7:]).split("-")); #[1,2] 
        
                
            #%% Set .mat file names
                
            pnev2load = [] #[] [3] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.
            signalCh = [2] # since gcamp is channel 2, should be always 2.
            postNProvided = 1; # If your directory does not contain pnevFile and instead it contains postFile, set this to 1 to get pnevFileName
            
            # from setImagingAnalysisNamesP import *
            
            imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load, postNProvided=postNProvided)
            
            postName = os.path.join(os.path.dirname(pnevFileName), 'post_'+os.path.basename(pnevFileName))
            moreName = os.path.join(os.path.dirname(pnevFileName), 'more_'+os.path.basename(pnevFileName))
            
            print(os.path.basename(imfilename))
            
            # Load inhibitRois
        #    Data = scio.loadmat(moreName, variable_names=['inhibitRois'])
        #    inhibitRois = Data.pop('inhibitRois')[0,:]
        
            
            #%% Load vars
            
            svmName = setSVMname(pnevFileName, trialHistAnalysis, ntName, doData)    
            print os.path.basename(svmName)
                
        #    Data = scio.loadmat(svmName, variable_names=['w', 'wsh', 'w_sr', 'wsh_sr'])
            if doData:
                if do_srChAl==0:
                    Data = scio.loadmat(svmName, variable_names=['w', 'w_sr', 'w_chAl'])
                    w = Data.pop('w') # frames x neurons x nrandtr
                    w_sr = Data.pop('w_sr') # frames x neurons x nrandtr
                    w_chAl = Data.pop('w_chAl') # frames x neurons x nrandtr
                else:
                    Data = scio.loadmat(svmName, variable_names=['w_sr_chAl'])
                    w_sr_chAl = Data.pop('w_sr_chAl') # frames x neurons x nrandtr                
            
            elif doData==0:
                if do_srChAl==0:            
                    Data = scio.loadmat(svmName, variable_names=['wsh', 'wsh_sr', 'wsh_chAl'])                
                    wsh = Data.pop('wsh') # frames x neurons x nrandtr x nshfl    
                    wsh_sr = Data.pop('wsh_sr') # frames x neurons x nrandtr x nshfl
                    wsh_chAl = Data.pop('wsh_chAl') # frames x neurons x nrandtr x nshfl
                else:
                    Data = scio.loadmat(svmName, variable_names=['wsh_sr_chAl'])
                    wsh_sr_chAl = Data.pop('wsh_sr_chAl') # frames x neurons x nrandtr x nshfl
                
            ##%% Load vars (w, etc)
        #    Data = scio.loadmat(svmName, variable_names=['NsExcluded'])
        #    NsExcluded = Data.pop('NsExcluded')[0,:].astype('bool')
        
            # Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)        
        #    inhRois = inhibitRois[~NsExcluded]        
            
            
        ###############################################################################      
            #%% Normalize weights (ie the w vector at each frame must have length 1)
            
            # data
            if doData:
                if do_srChAl==0:
                    normw = np.linalg.norm(w, axis=1) # frames x nrandtr; 2-norm of weights 
                    w_normed = np.transpose(np.transpose(w,(1,0,2))/normw, (1,0,2)) # frames x neurons x nrandtr; normalize weights so weights (of each frame) have length 1
                    if sum(normw<=eps).sum()!=0:
                        print 'take care of this; you need to reshape w_normed first'
                    #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
            
                    normw = np.linalg.norm(w_sr, axis=1) # frames x nrandtr; 2-norm of weights 
                    w_sr_normed = np.transpose(np.transpose(w_sr,(1,0,2))/normw, (1,0,2)) # frames x neurons x nrandtr; normalize weights so weights (of each frame) have length 1
                    if sum(normw<=eps).sum()!=0:
                        print 'take care of this; you need to reshape w_normed first'
                    #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
            
                    normw = np.linalg.norm(w_chAl, axis=1) # frames x nrandtr; 2-norm of weights 
                    w_chAl_normed = np.transpose(np.transpose(w_chAl,(1,0,2))/normw, (1,0,2)) # frames x neurons x nrandtr; normalize weights so weights (of each frame) have length 1
                    if sum(normw<=eps).sum()!=0:
                        print 'take care of this; you need to reshape w_normed first'
                    #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
                else:
                    normw = np.linalg.norm(w_sr_chAl, axis=1) # frames x nrandtr; 2-norm of weights 
                    w_sr_chAl_normed = np.transpose(np.transpose(w_sr_chAl,(1,0,2))/normw, (1,0,2)) # frames x neurons x nrandtr; normalize weights so weights (of each frame) have length 1
                    if sum(normw<=eps).sum()!=0:
                        print 'take care of this; you need to reshape w_normed first'
                    #    w_normed[normw<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
                        
                    
            # shuffle
            elif doData==0:
                if do_srChAl==0:
                    normwsh = np.linalg.norm(wsh, axis=1) # frames x nrandtr x nshfl; 2-norm of weights 
                    wsh_normed = np.transpose(np.transpose(wsh,(1,0,2,3))/normwsh, (1,0,2,3)) # frames x neurons x nrandtr x nshfl; normalize weights so weights (of each frame) have length 1
                    if sum(normwsh<=eps).sum()!=0:
                        print 'take care of this'
                        wsh_normed[normwsh<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
                
                    normwsh = np.linalg.norm(wsh_sr, axis=1) # frames x nrandtr x nshfl; 2-norm of weights 
                    wsh_sr_normed = np.transpose(np.transpose(wsh_sr,(1,0,2,3))/normwsh, (1,0,2,3)) # frames x neurons x nrandtr x nshfl; normalize weights so weights (of each frame) have length 1
                    if sum(normwsh<=eps).sum()!=0:
                        print 'take care of this'
                        wsh_normed[normwsh<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
            
                    normwsh = np.linalg.norm(wsh_chAl, axis=1) # frames x nrandtr x nshfl; 2-norm of weights 
                    wsh_chAl_normed = np.transpose(np.transpose(wsh_chAl,(1,0,2,3))/normwsh, (1,0,2,3)) # frames x neurons x nrandtr x nshfl; normalize weights so weights (of each frame) have length 1
                    if sum(normwsh<=eps).sum()!=0:
                        print 'take care of this'
                        wsh_normed[normwsh<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
        
                else:
                    normwsh = np.linalg.norm(wsh_sr_chAl, axis=1) # frames x nrandtr x nshfl; 2-norm of weights 
                    wsh_sr_chAl_normed = np.transpose(np.transpose(wsh_sr_chAl,(1,0,2,3))/normwsh, (1,0,2,3)) # frames x neurons x nrandtr x nshfl; normalize weights so weights (of each frame) have length 1
                    if sum(normwsh<=eps).sum()!=0:
                        print 'take care of this'
                        wsh_normed[normwsh<=eps, :] = 0 # set the direction to zero if the magnitude of the vector is zero
                    
            
            #%% Set the final decoders by aggregating decoders across trial subselects (ie average ws across all trial subselects), then again normalize them so they have length 1 (Bagging)
            
            # data
            if doData:
                if do_srChAl==0:
                    w_nb = np.mean(w_normed, axis=(2)) # frames x neurons 
                    nw = np.linalg.norm(w_nb, axis=1) # frames; 2-norm of weights 
                    w_n2b = np.transpose(np.transpose(w_nb,(1,0))/nw, (1,0)) # frames x neurons
            
                    w_sr_nb = np.mean(w_sr_normed, axis=(2)) # frames x neurons 
                    nw = np.linalg.norm(w_sr_nb, axis=1) # frames; 2-norm of weights 
                    w_sr_n2b = np.transpose(np.transpose(w_sr_nb,(1,0))/nw, (1,0)) # frames x neurons
                    
                    w_chAl_nb = np.mean(w_chAl_normed, axis=(2)) # frames x neurons 
                    nw = np.linalg.norm(w_chAl_nb, axis=1) # frames; 2-norm of weights 
                    w_chAl_n2b = np.transpose(np.transpose(w_chAl_nb,(1,0))/nw, (1,0)) # frames x neurons
                else:    
                    w_sr_chAl_nb = np.mean(w_sr_chAl_normed, axis=(2)) # frames x neurons 
                    nw = np.linalg.norm(w_sr_chAl_nb, axis=1) # frames; 2-norm of weights 
                    w_sr_chAl_n2b = np.transpose(np.transpose(w_sr_chAl_nb,(1,0))/nw, (1,0)) # frames x neurons
                    
                
            # shuffle
            elif doData==0:
                if do_srChAl==0:
                    wsh_nb = np.mean(wsh_normed, axis=(2)) # frames x neurons x nshfl
                    nwsh = np.linalg.norm(wsh_nb, axis=1) # frames x nshfl; 2-norm of weights 
                    wsh_n2b = np.transpose(np.transpose(wsh_nb,(1,0,2))/nwsh, (1,0,2)) # frames x neurons x nshfl
                
                    wsh_sr_nb = np.mean(wsh_sr_normed, axis=(2)) # frames x neurons x nshfl
                    nwsh = np.linalg.norm(wsh_sr_nb, axis=1) # frames x nshfl; 2-norm of weights 
                    wsh_sr_n2b = np.transpose(np.transpose(wsh_sr_nb,(1,0,2))/nwsh, (1,0,2)) # frames x neurons x nshfl
            
                    wsh_chAl_nb = np.mean(wsh_chAl_normed, axis=(2)) # frames x neurons x nshfl
                    nwsh = np.linalg.norm(wsh_chAl_nb, axis=1) # frames x nshfl; 2-norm of weights 
                    wsh_chAl_n2b = np.transpose(np.transpose(wsh_chAl_nb,(1,0,2))/nwsh, (1,0,2)) # frames x neurons x nshfl
                else:
                    wsh_sr_chAl_nb = np.mean(wsh_sr_chAl_normed, axis=(2)) # frames x neurons x nshfl
                    nwsh = np.linalg.norm(wsh_sr_chAl_nb, axis=1) # frames x nshfl; 2-norm of weights 
                    wsh_sr_chAl_n2b = np.transpose(np.transpose(wsh_sr_chAl_nb,(1,0,2))/nwsh, (1,0,2)) # frames x neurons x nshfl
    
                    
            
            #%% Compute angle between decoders (weights) at different times (remember angles close to 0 indicate more aligned decoders at 2 different time points.)
            
            # some of the angles are nan because the dot product is slightly above 1, so its arccos is nan!    
            # we restrict angles to [0 90], so we don't care about direction of vectors, ie tunning reversal.
            
            # data
            if doData:
                if do_srChAl==0:
                    angle_choice = np.arccos(abs(np.dot(w_n2b, w_n2b.transpose())))*180/np.pi # frames x frames; angle between ws at different times
                    angle_stim = np.arccos(abs(np.dot(w_sr_n2b, w_sr_n2b.transpose())))*180/np.pi # frames x frames; angle between ws at different times
                    angle_choice_chAl = np.arccos(abs(np.dot(w_chAl_n2b, w_chAl_n2b.transpose())))*180/np.pi # frames x frames; angle between ws at different times
                else:
                    angle_stim_chAl = np.arccos(abs(np.dot(w_sr_chAl_n2b, w_sr_chAl_n2b.transpose())))*180/np.pi # frames x frames; angle between ws at different times
                    
                    
            # shuffle
            elif doData==0:
                if do_srChAl==0:
                    nShflSamp = wsh.shape[-1]
                    
                    angle_choice_sh = np.full((np.shape(wsh)[0],np.shape(wsh)[0],nShflSamp), np.nan) # frames x frames x nshfl    
                    for ish in range(nShflSamp):
                        angle_choice_sh[:,:,ish] = np.arccos(abs(np.dot(wsh_n2b[:,:,ish], wsh_n2b[:,:,ish].transpose())))*180/np.pi # frames x frames x nshfl
                
                    angle_stim_sh = np.full((np.shape(wsh)[0],np.shape(wsh)[0],nShflSamp), np.nan) # frames x frames x nshfl
                    for ish in range(nShflSamp):
                        angle_stim_sh[:,:,ish] = np.arccos(abs(np.dot(wsh_sr_n2b[:,:,ish], wsh_sr_n2b[:,:,ish].transpose())))*180/np.pi # frames x frames x nshfl
            
                    angle_choice_chAl_sh = np.full((np.shape(wsh_chAl)[0],np.shape(wsh_chAl)[0],nShflSamp), np.nan) # frames x frames x nshfl    
                    for ish in range(nShflSamp):
                        angle_choice_chAl_sh[:,:,ish] = np.arccos(abs(np.dot(wsh_chAl_n2b[:,:,ish], wsh_chAl_n2b[:,:,ish].transpose())))*180/np.pi # frames x frames x nshfl
    
                else:
                    nShflSamp = wsh_sr_chAl.shape[-1]
    
                    angle_stim_chAl_sh = np.full((np.shape(wsh_sr_chAl)[0],np.shape(wsh_sr_chAl)[0],nShflSamp), np.nan) # frames x frames x nshfl    
                    for ish in range(nShflSamp):
                        angle_stim_chAl_sh[:,:,ish] = np.arccos(abs(np.dot(wsh_sr_chAl_n2b[:,:,ish], wsh_sr_chAl_n2b[:,:,ish].transpose())))*180/np.pi # frames x frames x nshfl
                        
                        
            # plot angle between decoders at different time points    
        #    print '____________________________________'
        #    print(os.path.basename(imfilename))
            if plotEachDay:
                # data
                if doData:
                    if do_srChAl==0:
                        plt.figure()
                        plt.imshow(angle_choice, cmap='jet_r')
                        plt.colorbar()
                
                        plt.figure()
                        plt.imshow(angle_stim, cmap='jet_r')
                        plt.colorbar()
                
                        plt.figure()
                        plt.imshow(angle_choice_chAl, cmap='jet_r')
                        plt.colorbar()
                    else:
                        plt.figure()
                        plt.imshow(angle_stim_chAl, cmap='jet_r')
                        plt.colorbar()
                        
                    
                # shuffle (average of angles across shuffles)
                elif doData==0:
                    if do_srChAl==0:
                        plt.figure()
                        plt.imshow(np.nanmean(angle_choice_sh, axis=2), cmap='jet_r') # average across shuffles
                        plt.colorbar()    
                
                        plt.figure()
                        plt.imshow(np.nanmean(angle_stim_sh, axis=2), cmap='jet_r') # average across shuffles
                        plt.colorbar()
                 
                        plt.figure()
                        plt.imshow(np.nanmean(angle_choice_chAl_sh, axis=2), cmap='jet_r') # average across shuffles
                        plt.colorbar()    
                    else:
                        plt.figure()
                        plt.imshow(np.nanmean(angle_stim_chAl_sh, axis=2), cmap='jet_r') # average across shuffles
                        plt.colorbar()                    
            #    print '____________________________________'    
        ###############################################################################
        
        
            #%% Keep vars of all days
        
            if doData:
                if do_srChAl==0:
                    angle_choice_all.append(angle_choice)    
                    angle_stim_all.append(angle_stim)
                    angle_choice_chAl_all.append(angle_choice_chAl)    
                else:
                    angle_stim_chAl_all.append(angle_stim_chAl)    
                                    
            elif doData==0:    
                if do_srChAl==0:
                    angle_choice_sh_all.append(angle_choice_sh)
                    angle_stim_sh_all.append(angle_stim_sh)
                    angle_choice_chAl_sh_all.append(angle_choice_chAl_sh)
                else:
                    angle_stim_chAl_sh_all.append(angle_stim_chAl_sh)    
    


#%%
#if doData:
angle_choice_all = np.array(angle_choice_all)
angle_stim_all = np.array(angle_stim_all)
angle_choice_chAl_all = np.array(angle_choice_chAl_all)
angle_stim_chAl_all = np.array(angle_stim_chAl_all)

#elif doData==0:    
angle_choice_sh_all = np.array(angle_choice_sh_all)
angle_stim_sh_all = np.array(angle_stim_sh_all)
angle_choice_chAl_sh_all = np.array(angle_choice_chAl_sh_all)
angle_stim_chAl_sh_all = np.array(angle_stim_chAl_sh_all)


###############################################################################
###############################################################################
###############################################################################

#%%
########################### Get eventI for all days ####################################################

#%%
a = svmName.find('_ds')
if a!=-1:
    regressBins = int(svmName[a+3])
else:
    regressBins = np.nan
    print 'No downsampling was performed!'
    
    
#%% Loop over days

eventI_allDays = np.full([numDays,1], np.nan).flatten().astype('int')
eventI_1stSide_allDays = np.full([numDays,1], np.nan).flatten().astype('int')

for iday in range(len(days)):
    
    imagingFolder = days[iday][0:6]; #'151013'
    mdfFileNumber = map(int, (days[iday][7:]).split("-")); #[1,2] 
        
    #%% Set .mat file names
    pnev2load = [] #[] [3] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.
    signalCh = [2] # since gcamp is channel 2, should be always 2.
    postNProvided = 1; # If your directory does not contain pnevFile and instead it contains postFile, set this to 1 to get pnevFileName
    
    # from setImagingAnalysisNamesP import *
    
    imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load, postNProvided=postNProvided)
    
    postName = os.path.join(os.path.dirname(pnevFileName), 'post_'+os.path.basename(pnevFileName))
    moreName = os.path.join(os.path.dirname(pnevFileName), 'more_'+os.path.basename(pnevFileName))
    
    print(imfilename)        
    
    #%% Load eventI (we need it for the final alignment of corrClass traces of all days)

    # Load aligned traces, times, frame of event of interest
    if trialHistAnalysis==0:
        Data = scio.loadmat(postName, variable_names=['stimAl_noEarlyDec'],squeeze_me=True,struct_as_record=False)
        if np.isnan(regressBins)==0:        
            time_aligned_stim = Data['stimAl_noEarlyDec'].time.astype('float')
        else:
            eventI = Data['stimAl_noEarlyDec'].eventI - 1 # remember difference indexing in matlab and python!        
    else:
        Data = scio.loadmat(postName, variable_names=['stimAl_allTrs'],squeeze_me=True,struct_as_record=False)
        if np.isnan(regressBins)==0:        
            time_aligned_stim = Data['stimAl_allTrs'].time.astype('float')
        else:
            eventI = Data['stimAl_allTrs'].eventI - 1 # remember difference indexing in matlab and python!


    ##%% choice-aligned
    Data = scio.loadmat(postName, variable_names=['firstSideTryAl'],squeeze_me=True,struct_as_record=False)
    if np.isnan(regressBins)==0:
        time_aligned_1stSide = Data['firstSideTryAl'].time.astype('float')
    else:
        eventI_1stSide = Data['firstSideTryAl'].eventI - 1 # remember difference indexing in matlab and python!
    
    
    #%% If downsampling was performed get eventI for downsampled time_aligned traces.
    
    if np.isnan(regressBins)==0: # regressBin is nan if no downsampling was performed.            
    
        # stim-aligned
        tt = np.shape(angle_choice_all[iday])[0] # number of time points in the downsampled X
        time_aligned_stim = time_aligned_stim[0:regressBins*tt]
        #time_aligned_stim.shape
        time_aligned_stim = np.round(np.mean(np.reshape(time_aligned_stim, (regressBins, tt), order = 'F'), axis=0), 2)
        #time_aligned_stim.shape
        if sum(np.sign(time_aligned_stim)==0)!=0:
            eventI = np.argwhere(np.sign(time_aligned_stim)==0)[0]
        else:
            eventI = np.argwhere(np.sign(time_aligned_stim)==1)[0]

        
        # choice-aligned            
        tt = np.shape(angle_choice_chAl_all[iday])[0] # number of time points in the downsampled X
        time_aligned_1stSide = time_aligned_1stSide[0:regressBins*tt]
        #time_aligned_1stSide.shape
        time_aligned_1stSide = np.round(np.mean(np.reshape(time_aligned_1stSide, (regressBins, tt), order = 'F'), axis=0), 2)
        #time_aligned_1stSide.shape
        if sum(np.sign(time_aligned_1stSide)==0)!=0:
            eventI_1stSide = np.argwhere(np.sign(time_aligned_1stSide)==0)[0]
        else:
            eventI_1stSide = np.argwhere(np.sign(time_aligned_1stSide)==1)[0]
        
    
    eventI_allDays[iday] = eventI  
    eventI_1stSide_allDays[iday] = eventI_1stSide  
    
       
##%%        
print 'stim-aligned eventI of all days:', eventI_allDays
print 'stim-aligned eventI of all days:', eventI_1stSide_allDays



#%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
# By common eventI, we  mean the index on which all traces will be aligned.

################ stimulus-aligned traces ################  

if np.isnan(regressBins)==0:        
    mf = regressBins * frameLength
else:
    mf = frameLength
        
# stim-aligned traces        
nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
for iday in range(numDays):
    nPost[iday] = (np.shape(angle_choice_all[iday])[0] - eventI_allDays[iday] - 1)

nPreMin = min(eventI_allDays) # number of frames before the common eventI, also the index of common eventI. 
nPostMin = min(nPost)
print 'stim-aligned: Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)

##%% Set the time array for the across-day aligned traces
a = -(np.asarray(mf) * range(nPreMin+1)[::-1])
b = (np.asarray(mf) * range(1, nPostMin+1))
time_aligned = np.concatenate((a,b))


################ choice-aligned traces ################  
     
nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
for iday in range(numDays):
    nPost[iday] = (np.shape(angle_choice_chAl_all[iday])[0] - eventI_1stSide_allDays[iday] - 1)

nPreMin_chAl = min(eventI_1stSide_allDays) # number of frames before the common eventI, also the index of common eventI. 
nPostMin_chAl = min(nPost)
print 'choice-aligned: Number of frames before = %d, and after = %d the common eventI' %(nPreMin_chAl, nPostMin_chAl)

##%% Set the time array for the across-day aligned traces
a = -(np.asarray(mf) * range(nPreMin_chAl+1)[::-1])
b = (np.asarray(mf) * range(1, nPostMin_chAl+1))
time_aligned_chAl = np.concatenate((a,b))



####################################################################################################
####################################################################################################
#%% Align angles of all days on the common eventI

r = nPreMin + nPostMin + 1
r_chAl = nPreMin_chAl + nPostMin_chAl + 1

#if doData:
angle_choice_aligned = np.ones((r,r, numDays)) + np.nan # frames x frames x days, aligned on common eventI (equals nPreMin)
angle_stim_aligned = np.ones((r,r, numDays)) + np.nan # frames x frames x days, aligned on common eventI (equals nPreMin)
angle_choice_chAl_aligned = np.ones((r_chAl,r_chAl, numDays)) + np.nan # frames x frames x days, aligned on common eventI (equals nPreMin)
angle_stim_chAl_aligned = np.ones((r_chAl,r_chAl, numDays)) + np.nan # frames x frames x days, aligned on common eventI (equals nPreMin)

#elif doData==0:
angle_choice_sh_aligned = np.ones((r,r,nShflSamp, numDays)) + np.nan # frames x frames x nShfl x days, aligned on common eventI (equals nPreMin)
angle_stim_sh_aligned = np.ones((r,r,nShflSamp, numDays)) + np.nan # frames x frames x nShfl x days, aligned on common eventI (equals nPreMin)
angle_choice_chAl_sh_aligned = np.ones((r_chAl,r_chAl,nShflSamp, numDays)) + np.nan # frames x frames x nShfl x days, aligned on common eventI (equals nPreMin)
angle_stim_chAl_sh_aligned = np.ones((r_chAl,r_chAl,nShflSamp, numDays)) + np.nan # frames x frames x nShfl x days, aligned on common eventI (equals nPreMin)


for iday in range(numDays):
#    inds = eventI_allDays[iday] - nPreMin  :  eventI_allDays[iday] + nPostMin + 1
    inds = np.arange(eventI_allDays[iday] - nPreMin,  eventI_allDays[iday] + nPostMin + 1)
    inds_chAl = np.arange(eventI_1stSide_allDays[iday] - nPreMin_chAl,  eventI_1stSide_allDays[iday] + nPostMin_chAl + 1)
    
#    if doData:    
    angle_choice_aligned[:,:,iday] = angle_choice_all[iday][inds][:,inds]
    angle_stim_aligned[:,:,iday] = angle_stim_all[iday][inds][:,inds]
    angle_choice_chAl_aligned[:,:,iday] = angle_choice_chAl_all[iday][inds_chAl][:,inds_chAl]
    angle_stim_chAl_aligned[:,:,iday] = angle_stim_chAl_all[iday][inds_chAl][:,inds_chAl]
    
#    elif doData==0:
    angle_choice_sh_aligned[:,:,:,iday] = angle_choice_sh_all[iday][inds][:,inds,:]
    angle_stim_sh_aligned[:,:,:,iday] = angle_stim_sh_all[iday][inds][:,inds,:]
    angle_choice_chAl_sh_aligned[:,:,:,iday] = angle_choice_chAl_sh_all[iday][inds_chAl][:,inds_chAl]
    angle_stim_chAl_sh_aligned[:,:,:,iday] = angle_stim_chAl_sh_all[iday][inds_chAl][:,inds_chAl]


#%% Average across days

#if doData:
ang_choice_av = np.nanmean(angle_choice_aligned, axis=-1)
ang_stim_av = np.nanmean(angle_stim_aligned, axis=-1)
ang_choice_chAl_av = np.nanmean(angle_choice_chAl_aligned, axis=-1)
ang_stim_chAl_av = np.nanmean(angle_stim_chAl_aligned, axis=-1)

#elif doData==0:    
ang_choice_sh_av = np.nanmean(angle_choice_sh_aligned, axis=-1)
ang_stim_sh_av = np.nanmean(angle_stim_sh_aligned, axis=-1)
ang_choice_chAl_sh_av = np.nanmean(angle_choice_chAl_sh_aligned, axis=-1)
ang_stim_chAl_sh_av = np.nanmean(angle_stim_chAl_sh_aligned, axis=-1)


### set diagonal elements (whose angles are 0) to nan, so heatmaps span a smaller range and are more visible.
np.fill_diagonal(ang_choice_av, np.nan)
np.fill_diagonal(ang_stim_av, np.nan)
np.fill_diagonal(ang_choice_chAl_av, np.nan)
np.fill_diagonal(ang_stim_chAl_av, np.nan)

[np.fill_diagonal(ang_choice_sh_av[:,:,i], np.nan) for i in range(nShflSamp)]
[np.fill_diagonal(ang_stim_sh_av[:,:,i], np.nan) for i in range(nShflSamp)]
[np.fill_diagonal(ang_choice_chAl_sh_av[:,:,i], np.nan) for i in range(nShflSamp)]
[np.fill_diagonal(ang_stim_chAl_sh_av[:,:,i], np.nan) for i in range(nShflSamp)]

cmin = np.floor(np.min([np.nanmin(ang_choice_av), np.nanmin(ang_stim_av), np.nanmin(ang_choice_chAl_av), np.nanmin(ang_stim_chAl_av)]))

#np.diagonal(ang_choice_av)
#for i in range(ang_choice_av.shape[0]):
#    ang_choice_av[i,i] = np.nan


#%% Plot averages across days (plot angle between decoders at different time points)

################################# stimulus-aligned #################################
step = 4
x = (np.unique(np.concatenate((np.arange(np.argwhere(time_aligned==0), -.5, -step), 
           np.arange(np.argwhere(time_aligned==0), r, step))))).astype(int)
#x = np.arange(0,r,step)
           
# choice
# data
#if doData:
plt.figure(figsize=(8,16))
plt.subplot(421)
plt.imshow(ang_choice_av, cmap='jet_r') #, interpolation='nearest') #, extent=time_aligned)
plt.colorbar()
plt.xticks(x, np.round(time_aligned[x]))
plt.yticks(x, np.round(time_aligned[x]))
makeNicePlots(plt.gca())
#ax.set_xticklabels(np.round(time_aligned[np.arange(0,60,10)]))
plt.title('choice')
plt.xlabel('Time since stimulus onset (ms)')
plt.clim(cmin, 90)

# shuffle (average of angles across shuffles)
#elif doData==0:    
#plt.figure()
plt.subplot(422)
plt.imshow(np.nanmean(ang_choice_sh_av, axis=2), cmap='jet_r') # average across shuffles
plt.colorbar()
plt.xticks(x, np.round(time_aligned[x]))
plt.yticks(x, np.round(time_aligned[x]))
makeNicePlots(plt.gca())
plt.title('choice-shuffle')
plt.xlabel('Time since stimulus onset (ms)')
plt.clim(cmin, 90)


# stimulus
# data
#if doData:
#plt.figure()
plt.subplot(423)
plt.imshow(ang_stim_av, cmap='jet_r')
plt.colorbar()
plt.xticks(x, np.round(time_aligned[x]))
plt.yticks(x, np.round(time_aligned[x]))
makeNicePlots(plt.gca())
plt.title('stimulus')
plt.xlabel('Time since stimulus onset (ms)')
plt.clim(cmin, 90)

# shuffle (average of angles across shuffles)
#elif doData==0:    
#plt.figure()
plt.subplot(424)
plt.imshow(np.nanmean(ang_stim_sh_av, axis=2), cmap='jet_r') # average across shuffles
plt.colorbar()
plt.xticks(x, np.round(time_aligned[x]))
plt.yticks(x, np.round(time_aligned[x]))
makeNicePlots(plt.gca())
plt.title('stimulus-shuffle')
plt.xlabel('Time since stimulus onset (ms)')
plt.clim(cmin, 90)


################################# choice-aligned #################################
step = 5
x = (np.unique(np.concatenate((np.arange(np.argwhere(time_aligned_chAl==0), -.5, -step), 
           np.arange(np.argwhere(time_aligned_chAl==0), r_chAl, step))))).astype(int)
#x = np.arange(0,r_chAl,step)           
           
# choice
# data
#if doData:
#plt.figure(figsize=(8,8))
plt.subplot(425)
plt.imshow(ang_choice_chAl_av, cmap='jet_r') #, interpolation='nearest') #, extent=time_aligned)
plt.colorbar()
plt.xticks(x, np.round(time_aligned_chAl[x]))
plt.yticks(x, np.round(time_aligned_chAl[x]))
makeNicePlots(plt.gca())
#ax.set_xticklabels(np.round(time_aligned[np.arange(0,60,10)]))
plt.title('choice')
plt.xlabel('Time since choice onset (ms)')
plt.clim(cmin, 90)

# shuffle (average of angles across shuffles)
#elif doData==0:    
#plt.figure()
plt.subplot(426)
plt.imshow(np.nanmean(ang_choice_chAl_sh_av, axis=2), cmap='jet_r') # average across shuffles
plt.colorbar()
plt.xticks(x, np.round(time_aligned_chAl[x]))
plt.yticks(x, np.round(time_aligned_chAl[x]))
makeNicePlots(plt.gca())
plt.title('choice-shuffle')
plt.xlabel('Time since choice onset (ms)')
plt.clim(cmin, 90)
    

           
# stimulus
# data
#if doData:
#plt.figure(figsize=(8,8))
plt.subplot(427)
plt.imshow(ang_stim_chAl_av, cmap='jet_r') #, interpolation='nearest') #, extent=time_aligned)
plt.colorbar()
plt.xticks(x, np.round(time_aligned_chAl[x]))
plt.yticks(x, np.round(time_aligned_chAl[x]))
makeNicePlots(plt.gca())
#ax.set_xticklabels(np.round(time_aligned[np.arange(0,60,10)]))
plt.title('stimulus')
plt.xlabel('Time since choice onset (ms)')
plt.clim(cmin, 90)

# shuffle (average of angles across shuffles)
#elif doData==0:    
#plt.figure()
plt.subplot(428)
plt.imshow(np.nanmean(ang_stim_chAl_sh_av, axis=2), cmap='jet_r') # average across shuffles
plt.colorbar()
plt.xticks(x, np.round(time_aligned_chAl[x]))
plt.yticks(x, np.round(time_aligned_chAl[x]))
makeNicePlots(plt.gca())
plt.title('stimulus-shuffle')
plt.xlabel('Time since choice onset (ms)')
plt.clim(cmin, 90)
    

plt.subplots_adjust(hspace=.3)



if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnow,mousename+dp)
#    d = os.path.join(svmdir+dnow,mousename+'/bestC'+dp) #, dgc, dta)
#    d = os.path.join(svmdir+dnow,mousename+'/bestC'+dp)
#    d = os.path.join(svmdir+dnow+'/bestC',mousename+dp)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
    
    fign = os.path.join(d, suffn[0:5]+'angles.'+fmt[0])
#    fign = os.path.join(d, suffn[0:5]+'classErrVsC_'+days[iday][0:6]+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')


    
    
#%% Plot each day    
'''    
# plot angle between decoders at different time points

for iday in range(numDays):
    
    # choice
    # data
    plt.figure(figsize=(8,8))
    plt.subplot(221)
    plt.imshow(angle_choice_aligned[:,:,iday], cmap='jet_r') #, interpolation='nearest') #, extent=time_aligned)
    plt.colorbar()
    plt.xticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
    plt.yticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
    makeNicePlots(plt.gca())
    #ax.set_xticklabels(np.round(time_aligned[np.arange(0,60,10)]))
    plt.title('choice')
    
    # shuffle (average of angles across shuffles)
    #plt.figure()
    plt.subplot(222)
    plt.imshow(np.nanmean(angle_choice_sh_aligned[:,:,iday], axis=2), cmap='jet_r') # average across shuffles
    plt.colorbar()
    plt.xticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
    plt.yticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
    makeNicePlots(plt.gca())
    plt.title('choice-shuffle')
    
    # stimulus
    # data
    #plt.figure()
    plt.subplot(223)
    plt.imshow(angle_stim_aligned[:,:,iday], cmap='jet_r')
    plt.colorbar()
    plt.xticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
    plt.yticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
    makeNicePlots(plt.gca())
    plt.title('stimulus')
    
    # shuffle (average of angles across shuffles)
    #plt.figure()
    plt.subplot(224)
    plt.imshow(np.nanmean(angle_stim_sh_aligned[:,:,iday], axis=2), cmap='jet_r') # average across shuffles
    plt.colorbar()
    plt.xticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
    plt.yticks(np.arange(0,r,nPreMin), np.round(time_aligned[np.arange(0,r,nPreMin)]))
    makeNicePlots(plt.gca())
    plt.title('stimulus-shuffle')
    
    plt.subplots_adjust(hspace=.3)
'''
    