# -*- coding: utf-8 -*-
"""
Plots class accuracy for svm trained on non-overlapping time windows  (outputs of file svm_eachFrame.py)
 ... svm trained to decode choice on choice-aligned or stimulus-aligned traces.
 
 
Remember for fni18 there are 2 svm_eachFrame mat files, the earlier file is using all trials (unequal HR, LR, like how you've done all your analysis). 
The later mat file is with equal number of hr and lr trials (subselecting trials)... this helped with 151209 class accur trace which was weird in the earlier mat file.
 
Created on Sun Mar 12 15:12:29 2017
@author: farznaj
"""


#%% Change the following vars:

mousename = 'fni18' #'fni17'
if mousename == 'fni18':
    allDays = 0# all 7 days will be used (last 3 days have z motion!)
    noZmotionDays = 1 # 4 days that dont have z motion will be used.
    noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!
    
trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
execfile("svm_plots_setVars_n.py")  
execfile("defFuns.py")  

chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
doPlots = 0 #1 # plot c path of each day 
savefigs = 0

#eps = 10**-10 # tiny number below which weight is considered 0
#thNon0Ws = 2 # For samples with <2 non0 weights, we manually set their class error to 50 ... the idea is that bc of difference in number of HR and LR trials, in these samples class error is not accurately computed!
#thSamps = 10  # Days that have <thSamps samples that satisfy >=thNon0W non0 weights will be manually set to 50 (class error of all their samples) ... bc we think <5 samples will not give us an accurate measure of class error of a day.
#setTo50 = 1 # if 1, the above two jobs will be done.


#%% 
import numpy as np
frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.

dnow = '/classAccurTraces_eachFrame/'+mousename+'/'

smallestC = 0 # Identify best c: if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
if smallestC==1:
    print 'bestc = smallest c whose cv error is less than 1se of min cv error'
else:
    print 'bestc = c that gives min cv error'
#I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.


#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName

def setSVMname(pnevFileName, trialHistAnalysis, chAl, regressBins=3):
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
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
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
############################ stimulus-aligned, SVR (trained on trials of 1 choice to avoid variations in neural response due to animal's choice... we only want stimulus rate to be different) ###################################################################################################     
#####################################################################################################################################################
'''
            
#%% Loop over days    

eventI_allDays = np.full((len(days)), np.nan) # frame at which choice happened (if traces were downsampled in svm_eachFrame, it will be the downsampled frame number)
cbestFrs_all = []
winh_all = []
wexc_all = []
winhAve_all = []
wexcAve_all = []
b_bestc_data_all        

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
    
    imfilename, pnevFileName,_ = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load, postNProvided=postNProvided)
    
    postName = os.path.join(os.path.dirname(pnevFileName), 'post_'+os.path.basename(pnevFileName))
    moreName = os.path.join(os.path.dirname(pnevFileName), 'more_'+os.path.basename(pnevFileName))
    
    print(os.path.basename(imfilename))


    #%%
    svmName = setSVMname(pnevFileName, trialHistAnalysis, chAl, regressBins) # for chAl: the latest file is with soft norm; earlier file is 

    svmName = svmName[0]
    print os.path.basename(svmName)    


    #%%
    Data = scio.loadmat(moreName, variable_names=['inhibitRois'])
    inhibitRois = Data.pop('inhibitRois')[0,:]
#    Data = scio.loadmat(svmName, variable_names=['NsExcluded_chAl'])        
#    NsExcluded_chAl = Data.pop('NsExcluded_chAl')[0,:].astype('bool')
    # Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)        
    inhRois = inhibitRois #[~NsExcluded_chAl]   
    
    
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
    
    
    #%% Load SVM vars
        
    Data = scio.loadmat(svmName, variable_names=['regType','cvect','perClassErrorTest']) #,'perClassErrorTrain','perClassErrorTest','perClassErrorTest_chance','perClassErrorTest_shfl'])
    
    regType = Data.pop('regType').astype('str')
    cvect = Data.pop('cvect').squeeze()
    perClassErrorTest = Data.pop('perClassErrorTest')
    """
#    trsExcluded_svr = Data.pop('trsExcluded_svr').astype('bool').squeeze()                    
    perClassErrorTrain = Data.pop('perClassErrorTrain') # numSamples x len(cvect) x nFrames    
    perClassErrorTest_chance = Data.pop('perClassErrorTest_chance')
    perClassErrorTest_shfl = Data.pop('perClassErrorTest_shfl')
    """
    Data = scio.loadmat(svmName, variable_names=['wAllC','bAllC'])
    wAllC = Data.pop('wAllC') # numSamples x len(cvect) x nNeurons x nFrames
    bAllC = Data.pop('bAllC') # numSamples x len(cvect) x nFrames    

    #'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'trsTrainedTestedInds':trsTrainedTestedInds, 'trsRemCorrInds':trsRemCorrInds,
#    Data = scio.loadmat(svmName, variable_names=['trsTrainedTestedInds','trsRemCorrInds'])
#    trsTrainedTestedInds = Data.pop('trsTrainedTestedInds')
#    trsRemCorrInds = Data.pop('trsRemCorrInds')                                 
#    numSamples = perClassErrorTest.shape[0] 
#    numFrs = perClassErrorTest.shape[2]
        
        
        
    #%% Find bestc for each frame (funcion is defined in defFuns.py)
        
    cbestFrs = findBestC(perClassErrorTest, cvect, regType, smallestC) # nFrames        
	
	
    #%% Set class error values at best C (if desired plot c path) (funcion is defined in defFuns.py)

    classErr_bestC_train_data, classErr_bestC_test_data, classErr_bestC_test_shfl, classErr_bestC_test_chance, w_bestc_data, b_bestc_data = setClassErrAtBestc(cbestFrs, cvect, doPlots, np.nan, np.nan, np.nan, np.nan, wAllC, bAllC)
    

    #%% Set w for inh vs exc
    
    winh0 = w_bestc_data[:,inhRois==1] # numSamples x nNeurons x nFrames
    wexc0 = w_bestc_data[:,inhRois==0]
   
    # Take average across samples (this is like getting a single decoder across all trial subselects... so I guess bagging)... (not ave of abs... the idea is that ave of shuffles represents the neurons weights for the decoder given the entire dataset (all trial subselects)... if a w switches signs across samples then it means this neuron is not very consistently contributing to the choice... so i think we should average its w and not its abs(w))
    winhAve = np.mean(winh0, axis=0) # neurons x frames
    wexcAve = np.mean(wexc0, axis=0)
    
    
    #%% Once done with all frames, save vars for all days
           
    # Delete vars before starting the next day           
#    del perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance, wAllC, bAllC
    
    cbestFrs_all.append(cbestFrs)

    winh_all.append(winh0) 
    wexc_all.append(wexc0)     

    winhAve_all.append(winhAve)
    wexcAve_all.append(wexcAve)
    
    winhAve_all.append(winhAve)
    wexcAve_all.append(wexcAve)    

    b_bestc_data_all.append(b_bestc_data) 
#    winh.append(np.mean(abs(winh0), axis=0))
#    wexc.append(np.mean(abs(wexc0), axis=0))

eventI_allDays = eventI_allDays.astype('int')
cbestFrs_all = np.array(cbestFrs_all)    



#%%
winhAve_all2 = np.array([winhAve_all[iday][:, eventI_allDays[iday]] for iday in range(len(days))])
wexcAve_all2 = np.array([wexcAve_all[iday][:, eventI_allDays[iday]] for iday in range(len(days))])


#%%
###############################################################################
#################################### PLOTS ####################################
###############################################################################
#%%
lab1 = 'exc'
lab2 = 'inh'
colors = ['k','r']

########################################
###############%% w  ###############
########################################
### hist and P val for all days pooled (exc vs inh)
lab = 'w'
binEvery = .001 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)

a = np.concatenate((wexcAve_all2)) #np.reshape(wexc,(-1,)) 
b = np.concatenate((winhAve_all2)) #np.reshape(winh,(-1,))

print np.sum(abs(a)<=eps), ' inh ws < eps'
print np.sum(abs(b)<=eps), ' exc ws < eps'
print [len(a), len(b)]
_, p = stats.ttest_ind(a, b, nan_policy='omit')

plt.figure(figsize=(5,5))    
gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[0,0:2]
h2 = gs[0,2:3]
ax1,_ = histerrbar(a,b,binEvery,p,colors)
#plt.xlabel(lab)
ax1.set_xlim([-.05, .05])



### show individual days
a = wexc
b = winh
# if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
p = []
for i in range(len(a)):
    _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
    p.append(p0)
p = np.array(p)    
print p
#ax = plt.subplot(gs[1,0:3])
errbarAllDays(a,b,p)
#plt.ylabel(lab)


if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnow,mousename+dp)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
 
    fign = os.path.join(d, suffn[0:5]+'excVSinh_allDays_w'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')
    
    

########################################
###############%% abs w  ###############
########################################
### hist and P val for all days pooled (exc vs inh)
lab = 'abs w'
binEvery = .002 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)

a = np.concatenate(abs(wexc)) #np.reshape(wexc,(-1,)) 
b = np.concatenate(abs(winh)) #np.reshape(winh,(-1,))

print np.sum(abs(a)<=eps), ' inh ws < eps'
print np.sum(abs(b)<=eps), ' exc ws < eps'
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
a = abs(wexc)
b = abs(winh)
# if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
p = []
for i in range(len(a)):
    _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
    p.append(p0)
p = np.array(p)    
print p
#ax = plt.subplot(gs[1,0:3])
errbarAllDays(a,b,p)
#plt.ylabel(lab)


if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnow,mousename+dp)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
 
    fign = os.path.join(d, suffn[0:5]+'excVSinh_allDays_absw'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')
    
    

