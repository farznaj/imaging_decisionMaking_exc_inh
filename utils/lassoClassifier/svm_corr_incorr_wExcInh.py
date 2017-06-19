# -*- coding: utf-8 -*-
"""
For svm decoder (L2, trained on corr, tested on corr and incorr) compute magnitude of weights between exc and inh neurons.

(Data was driven from svm_corr_incorr.py)

Created on Sun Feb 19 03:13:36 2017
@author: farznaj
"""

# -*- coding: utf-8 -*-

#%% Set vars
doSVM = 1 # analyze svm: 1; analyze svr: 0 ; (on both choice-aligned and stim-aligned)

# Go to svm_plots_setVars and define vars!
from svm_plots_setVars import *
execfile("svm_plots_setVars.py")  
#savefigs = 1


#%%
if doSVM==0:
    epall = [0,1] #epAllStim = 1
else:
    epall = [0]


numSamp = 100 # 500 perClassErrorTest_shfl.shape[0]  # number of shuffles for doing cross validation (ie number of random sets of test/train trials.... You changed the following: in mainSVM_notebook.py this is set to 100, so unless you change the value inside the code it should be always 100.)
dnow = '/excInh_l2'
if trialHistAnalysis:
    dp = '/previousChoice'
else:
    dp = '/currentChoice'
    

#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName

def setSVMname(pnevFileName, trialHistAnalysis, doSVM):
    import glob

    if trialHistAnalysis:
        if doSVM:
            svmn = 'svmPrevChoice_corrIncorr_chAl_*'
        else:
            svmn = 'svrPrevChoice_corrIncorr_chAl_*'
#        svmn = 'svrPrevChoice_corrIncorr_stAl_*'
    else:
        if doSVM:
            svmn = 'svmCurrChoice_corrIncorr_chAl_*'
        else:
            svmn = 'svrCurrChoice_corrIncorr_chAl_*'
#        svmn = 'svrCurrChoice_corrIncorr_stAl_*'
    
    svmn = svmn + os.path.basename(pnevFileName) #pnevFileName[-32:]    
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
    svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    svmName = svmName[0] # get the latest file
    
    return svmName
    
    
#%% Function to predict class labels
# Lets check how predict works.... Result: for both SVM and SVR, classfier.predict equals xw+b. For SVM, xw+b gives -1 and 1, that should be changed to 0 and 1 to match svm.predict.

def predictMan(X,w,b,th=0): # set th to nan if you are predicting on svr (since for svr we dont compare with a threshold)
    yhat = np.dot(X, w.T) + b # it gives -1 and 1... change -1 to 0... so it matches svm.predict
    if np.isnan(th)==0:
        yhat[yhat<th] = 0
        yhat[yhat>th] = 1    
    return yhat

    
#%% PLOTS; define functions

def histerrbar(a,b,binEvery,p,colors = ['g','k']):
#    import matplotlib.gridspec as gridspec
    
#    r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
#    binEvery = r/float(10)

#    _, p = stats.ttest_ind(a, b, nan_policy='omit')

#    plt.figure(figsize=(5,3))    
#    gs = gridspec.GridSpec(2, 4)#, width_ratios=[2, 1]) 
#    h1 = gs[0,0:2]
#    h2 = gs[0,2:3]

#    lab1 = 'exc'
#    lab2 = 'inh'

#    colors = ['g','k']
    
    # set bins
    bn = np.arange(np.min(np.concatenate((a,b))), np.max(np.concatenate((a,b))), binEvery)
    bn[-1] = np.max([np.max(a),np.max(b)]) # unlike digitize, histogram doesn't count the right most value
    # set hist of a
    hist, bin_edges = np.histogram(a, bins=bn)
    hist = hist/float(np.sum(hist))    
    # Plot hist of a
    ax1 = plt.subplot(h1) #(gs[0,0:2])
    plt.bar(bin_edges[0:-1], hist, binEvery, color=colors[0], alpha=.4, label=lab1)
    
    # set his of b
    hist, bin_edges = np.histogram(b, bins=bn)
    hist = hist/float(np.sum(hist));     #d = stats.mode(np.diff(bin_edges))[0]/float(2)
    # Plot hist of b
    plt.bar(bin_edges[0:-1], hist, binEvery, color=colors[1], alpha=.4, label=lab2)
    
    plt.legend(loc=0, frameon=False)
    plt.ylabel('Prob (all days & N shuffs at bestc)')
#    plt.title('mean diff= %.3f, p=%.3f' %(np.mean(a)-np.mean(b), p))
    plt.title('mean diff= %.3f' %(np.mean(a)-np.mean(b)))
    #plt.xlim([-.5,.5])
    plt.xlabel(lab)
    makeNicePlots(ax1,0,1)

    
    # errorbar: mean and st error
    ax2 = plt.subplot(h2) #(gs[0,2:3])
    plt.errorbar([0,1], [a.mean(),b.mean()], [a.std()/np.sqrt(len(a)), b.std()/np.sqrt(len(b))], marker='o',color='k', fmt='.')
    plt.xlim([-1,2])
#    plt.title('%.3f, %.3f' %(a.mean(), b.mean()))
    plt.xticks([0,1], (lab1, lab2), rotation='vertical')
    plt.ylabel(lab)
    plt.title('p=%.3f' %(p))
    makeNicePlots(ax2,0,1)
#    plt.tick_params
    
    plt.subplots_adjust(wspace=1, hspace=.5)
    return ax1,ax2



def errbarAllDays(a,b,p):
    eav = [np.mean(a[i]) for i in range(len(a))] #np.nanmean(a, axis=1) # average across shuffles
    iav = [np.mean(b[i]) for i in range(len(b))] #np.nanmean(b, axis=1)
    ele = [len(a[i]) for i in range(len(a))] #np.shape(a)[1] - np.sum(np.isnan(a),axis=1) # number of non-nan shuffles of each day
    ile = [len(b[i]) for i in range(len(b))] #np.shape(b)[1] - np.sum(np.isnan(b),axis=1) # number of non-nan shuffles of each day
    esd = np.divide([np.std(a[i]) for i in range(len(a))], np.sqrt(ele))
    isd = np.divide([np.std(b[i]) for i in range(len(b))], np.sqrt(ile))
    
    pp = p
    pp[p>.05] = np.nan
    pp[p<=.05] = np.max((eav,iav))
    x = np.arange(np.shape(eav)[0])
    
    ax = plt.subplot(gs[1,0:2])
    plt.errorbar(x, eav, esd, color='k')
    plt.errorbar(x, iav, isd, color='r')
    plt.plot(x, pp, marker='*',color='r', linestyle='')
    plt.xlim([-1, x[-1]+1])
    plt.xlabel('Days')
    plt.ylabel(lab)
    makeNicePlots(ax,0,1)

    ax = plt.subplot(gs[1,2:3])
    plt.errorbar(0, np.nanmean(eav), np.nanstd(eav)/np.sqrt(len(eav)), marker='o', color='k')
    plt.errorbar(1, np.nanmean(iav), np.nanstd(iav)/np.sqrt(len(eav)), marker='o', color='k')
    plt.xticks([0,1], (lab1, lab2), rotation='vertical')
    plt.xlim([-1,2])
    makeNicePlots(ax,0,1)

    _, p = stats.ttest_ind(eav, iav, nan_policy='omit')
    plt.title('p=%.3f' %(p))

    plt.subplots_adjust(wspace=1, hspace=.5)

    
    
    
#%%
'''
#####################################################################################################################################################   
############################ choice-aligned, SVM (L2,corr,incorr) ###################################################################################################     
#####################################################################################################################################################
'''

#%% Loop over days    
# test in the names below means cv (cross validated or testing data!)    

winh = []
wexc = []
        
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
    
    svmName = setSVMname(pnevFileName, trialHistAnalysis, doSVM) # latest is l1, then l2. we use [] to get both files. # after that you again ran analysis with cross validation shuffles (l1).

    print os.path.basename(svmName)    
        
        
    #%% Load vars
        
    Data = scio.loadmat(moreName, variable_names=['inhibitRois'])
    inhibitRois = Data.pop('inhibitRois')[0,:]
    Data = scio.loadmat(svmName, variable_names=['NsExcluded_chAl'])        
    NsExcluded_chAl = Data.pop('NsExcluded_chAl')[0,:].astype('bool')
    # Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)        
    inhRois = inhibitRois[~NsExcluded_chAl]        

        
    Data = scio.loadmat(svmName, variable_names=['cvect','cbest_chAl','wAllC_chAl','bAllC_chAl'])
#    Data = scio.loadmat(svmName, variable_names=['cvect','cbest_chAl','perClassErrorTest_chAl','perClassErrorTest_chAl_chance','perClassErrorTest_incorr_chAl','perClassErrorTest_incorr_chAl_chance'])
    cvect = Data.pop('cvect')
    cbest_chAl = Data.pop('cbest_chAl')
    wAllC_chAl = Data.pop('wAllC_chAl')
    bAllC_chAl = Data.pop('bAllC_chAl')
     
    
    #%%
    indBestC = np.in1d(cvect, cbest_chAl)
    
    w_bestc_data_chAl = wAllC_chAl[:,indBestC,:].squeeze() # numSamps x neurons
    b_bestc_data_chAl = bAllC_chAl[:,indBestC]
    
    # bc l2 was used, we dont need to worry about all weights being 0       
#    a = (abs(w_bestc_data_chAl)>eps) # non-zero weights
#    b = np.mean(a, axis=(1)) # Fraction of non-zero weights (averaged across shuffles)


    #%% Set w for inh vs exc
    
    winh0 = w_bestc_data_chAl[:,inhRois==1]
    wexc0 = w_bestc_data_chAl[:,inhRois==0]

#    for i in range(10):    
#        plt.plot(winh[:,i])
#        plt.imshow

    # Take average across samples (this is like getting a single decoder across all trial subselects... so I guess bagging)... (not ave of abs... the idea is that ave of shuffles represents the neurons weights for the decoder given the entire dataset (all trial subselects)... if a w switches signs across samples then it means this neuron is not very consistently contributing to the choice... so i think we should average its w and not its abs(w))
    winh.append(np.mean(winh0, axis=0))
    wexc.append(np.mean(wexc0, axis=0))
#    winh.append(np.mean(abs(winh0), axis=0))
#    wexc.append(np.mean(abs(wexc0), axis=0))


winh = np.array(winh)
wexc = np.array(wexc)


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

a = np.concatenate((wexc)) #np.reshape(wexc,(-1,)) 
b = np.concatenate((winh)) #np.reshape(winh,(-1,))

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
    
    








#%% Get class accuracy traces. (And Projections)

numSamples = 100 # number of times to shuffle trial labels for getting shuffled class accur trace
frameLength = 1000/30.9; # sec.
outcome2ana = 'corr'
eventI_allDays = (np.ones((numDays,1))+np.nan).flatten().astype('int')
eventI_allDays_chAl = (np.ones((numDays,1))+np.nan).flatten().astype('int')
corrClassAll = []
corrClassShflAll = []
bAll = []


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
    
    svmName = setSVMname(pnevFileName, trialHistAnalysis, doSVM) # latest is l1, then l2. we use [] to get both files. # after that you again ran analysis with cross validation shuffles (l1).

    print os.path.basename(svmName)    
        
        
    #%% Load vars
        
    Data = scio.loadmat(moreName, variable_names=['inhibitRois'])
    inhibitRois = Data.pop('inhibitRois')[0,:]
    Data = scio.loadmat(svmName, variable_names=['NsExcluded_chAl'])        
    NsExcluded_chAl = Data.pop('NsExcluded_chAl')[0,:].astype('bool')
    # Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)        
    inhRois = inhibitRois[~NsExcluded_chAl]        


    Data = scio.loadmat(svmName, variable_names=['trsExcluded_chAl', 'trsExcluded_incorr_chAl'])        
    trsExcluded_chAl = Data.pop('trsExcluded_chAl')[0,:].astype('bool')
    trsExcluded_incorr_chAl = Data.pop('trsExcluded_incorr_chAl')[0,:].astype('bool')
    
    
    Data = scio.loadmat(svmName, variable_names=['cvect','cbest_chAl','wAllC_chAl','bAllC_chAl'])
#    Data = scio.loadmat(svmName, variable_names=['cvect','cbest_chAl','perClassErrorTest_chAl','perClassErrorTest_chAl_chance','perClassErrorTest_incorr_chAl','perClassErrorTest_incorr_chAl_chance'])
    cvect = Data.pop('cvect')
    cbest_chAl = Data.pop('cbest_chAl')
    wAllC_chAl = Data.pop('wAllC_chAl')
    bAllC_chAl = Data.pop('bAllC_chAl')
     
    
    #%%
    indBestC = np.in1d(cvect, cbest_chAl)
    
    w_bestc_data_chAl = wAllC_chAl[:,indBestC,:].squeeze() # numSamps x neurons
    b_bestc_data_chAl = bAllC_chAl[:,indBestC]
    
    # bc l2 was used, we dont need to worry about all weights being 0       
#    a = (abs(w_bestc_data_chAl)>eps) # non-zero weights
#    b = np.mean(a, axis=(1)) # Fraction of non-zero weights (averaged across shuffles)

    # average decoders across samples to get a single decoder:
    w = np.mean(w_bestc_data_chAl, axis = 0)
    # b is very negative (~.3) for fni16,151029... and i see that averaged b screws up predictions! so I believe the following is wrong!
    # I would also average b across decoder to get a single b... not sure if this is right to do ... but I've tried it and it is fine.
    b = np.mean(b_bestc_data_chAl, axis = 0)
    bAll.append(b)
    
    
    #%% Load traces
    
    ############ stim-aligned traces #############
    
    # Load stim-aligned_allTrials traces, frames, frame of event of interest
    if trialHistAnalysis==0:
        Data = scio.loadmat(postName, variable_names=['stimAl_noEarlyDec'],squeeze_me=True,struct_as_record=False)
        eventI = Data['stimAl_noEarlyDec'].eventI - 1 # remember difference indexing in matlab and python!
        traces_al_stimAll = Data['stimAl_noEarlyDec'].traces.astype('float')
    #    time_aligned_stim = Data['stimAl_noEarlyDec'].time.astype('float')
    
    else:
        Data = scio.loadmat(postName, variable_names=['stimAl_allTrs'],squeeze_me=True,struct_as_record=False)
        eventI = Data['stimAl_allTrs'].eventI - 1 # remember difference indexing in matlab and python!
        traces_al_stimAll = Data['stimAl_allTrs'].traces.astype('float')
    #    time_aligned_stim = Data['stimAl_allTrs'].time.astype('float')        
    
    #print 'size of stimulus-aligned traces:', np.shape(traces_al_stimAll), '(frames x units x trials)'
    #DataS = Data
    traces_al_stim = traces_al_stimAll
    eventI_allDays[iday] = eventI
    
    
    ################################## Outcomes and choiceVec ##################################
    # Load outcomes and choice (allResp_HR_LR) for the current trial    
    Data = scio.loadmat(postName, variable_names=['outcomes', 'allResp_HR_LR'])
    outcomes = (Data.pop('outcomes').astype('float'))[0,:]
    allResp_HR_LR = np.array(Data.pop('allResp_HR_LR')).flatten().astype('float')
    choiceVecAll = allResp_HR_LR+0;  # trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.        
#    print 'Current outcome: %d correct choices; %d incorrect choices' %(sum(outcomes==1), sum(outcomes==0))
    
    
    if trialHistAnalysis:
        # Load trialHistory structure to get choice vector of the previous trial
        Data = scio.loadmat(postName, variable_names=['trialHistory'],squeeze_me=True,struct_as_record=False)
        choiceVec0All = Data['trialHistory'].choiceVec0.astype('float')


    ##%% Set choiceVec0  (Y: the response vector)
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
        
#        choiceVec0[~str2ana] = np.nan   
        # Y = choiceVec0
        # print(choiceVec0.shape)
    print '%d correct trials; %d incorrect trials' %((outcomes==1).sum(), (outcomes==0).sum())
    
    Y_chAl = choiceVec0[~trsExcluded_chAl]
    

    #################### choice-aligned traces #####################
    # Load 1stSideTry-aligned traces, frames, frame of event of interest
    # use firstSideTryAl_COM to look at changes-of-mind (mouse made a side lick without committing it)
    Data = scio.loadmat(postName, variable_names=['firstSideTryAl'],squeeze_me=True,struct_as_record=False)
    traces_al_1stSide = Data['firstSideTryAl'].traces.astype('float')
    #time_aligned_1stSide = Data['firstSideTryAl'].time.astype('float')
    eventI_ch = Data['firstSideTryAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
    # print(np.shape(traces_al_1stSide))
    eventI_allDays_chAl[iday] = eventI_ch
        
    
    #%%    
    ######################### we need all the following to set meanX and stdX for normalization    
    # training epoch: 300ms before the choice is made.
    epSt = eventI_ch - np.round(300/frameLength) # the start point of the epoch relative to alignedEvent for training SVM. (500ms)
    epEn = eventI_ch-1 # the end point of the epoch relative to alignedEvent for training SVM. (700ms)
    ep_ch = np.arange(epSt, epEn+1).astype(int) # frames on stimAl.traces that will be used for trainning SVM.  
    
    # average frames during ep_ch (ie 300ms before choice onset)
    X_choice0 = np.transpose(np.nanmean(traces_al_1stSide[ep_ch,:,:], axis=0)) # trials x neurons 
    
    
    # exclude trsExcluded_chAl
    X_chAl = X_choice0[~trsExcluded_chAl,:]; # trials x neurons
    X_chAl_incorr = X_choice0[~trsExcluded_incorr_chAl,:]
    
    #Y_chAl = choiceVec0[~trsExcluded_chAl]    
    #Ysr_chAl = stimrate[~trsExcluded_chAl] # we dont need to exclude trials that animal did not make a choice for the stimulus decoder analysis... unless you want to use the same set of trials for the choice and stimulus decoder...
    

    #%% Exclude neurons from traces
    
    traces_al_stim = traces_al_stim[:,~NsExcluded_chAl,:]
    traces_al_1stSide = traces_al_1stSide[:,~NsExcluded_chAl,:]

    X_chAl = X_chAl[:,~NsExcluded_chAl]
    X_chAl_incorr = X_chAl_incorr[:,~NsExcluded_chAl]
    

    #%% Set normalization params (meanX and stdX)

    xtot = np.concatenate((X_chAl,X_chAl_incorr),axis=0)
    meanX_chAl = np.mean(xtot, axis = 0);
    stdX_chAl = np.std(xtot, axis = 0);


    #%% Feature normalization and scaling    
    
    # normalize both stim-al and choice-al to meanX_chAl and stdX_chAl
    
    # stimulus-aligned
    Xt = traces_al_stim
    T, N, C = Xt.shape
    Xt_N = np.reshape(Xt.transpose(0 ,2 ,1), (T*C, N), order = 'F')
    Xt_N = (Xt_N-meanX_chAl)/stdX_chAl
    Xt_st = np.reshape(Xt_N, (T, C, N), order = 'F').transpose(0 ,2 ,1)

    # choice-aligned
    Xt_chAl = traces_al_1stSide
    T, N, C = Xt_chAl.shape
    Xt_N = np.reshape(Xt_chAl.transpose(0 ,2 ,1), (T*C, N), order = 'F')
    Xt_N = (Xt_N-meanX_chAl)/stdX_chAl
    Xt_ch = np.reshape(Xt_N, (T, C, N), order = 'F').transpose(0 ,2 ,1)


    #%% Project onto the decoder
    
    
    
    
    #%% Classification accuracy at each time point
#     Use the same SVM model trained during ep to predict animal's choice from the population responses at all time points.
#     Same trials that went into projection traces will be used here.

    # what if you just use intercept of 0    
    b = 0
    
    # perClassCorr_t = [];
    # corrClass = np.ones((T, numTrials)) + np.nan # frames x trials
    
    # temp = Xti # for trial-history you may want to use this:
    temp = Xt_chAl[:,:,~trsExcluded_chAl] # stimulus-aligned traces (I think you should use Xtsa for consistency... u get projections from Xtsa)
    nnf, nnu, nnt = temp.shape
    corrClass = np.ones((T, nnt)) + np.nan # frames x trials
    corrClassShfl = np.ones((T, nnt, numSamples)) + np.nan # frames x trials x numSamples
    # if trs4project!='trained':  # onlyTrainedTrs==0: 
    #     temp = temp[:,:,~trsExcluded] # make sure it has same size as Y, you need this for svm.predict below. 
    for t in range(T):
        trac = np.squeeze(temp[t, :, :]).T # trials x neurons  # stimulus-aligned trace at time t
        # instead of below we manually predict the choie
        py = predictMan(trac, w, b) # linear_svm.predict(trac)
        corrFract = 1 - abs(py-Y_chAl) # trials x 1 # fraction of correct choice classification using stimulus-aligned neural responses at time t and the trained SVM model linear_svm.
        
        # corrFract = 1 - abs(linear_svm.predict(trac)-Y); # trials x 1 # fraction of correct choice classification using stimulus-aligned neural responses at time t and the trained SVM model linear_svm.
        corrClass[t,:] = corrFract # frames x trials % fraction correct classification for all trials
        # perClassCorr_t.append(corrFract.mean()*100) # average correct classification across trials for time point t. Same as np.mean(corrClass, axis=1)    
        
        # shuffled data
        for i in range(numSamples):
            permIxs = rng.permutation(len(Y_chAl));
            corrFractSh = 1 - abs(py-Y_chAl[permIxs]);
            corrClassShfl[t,:,i] = corrFractSh # frames x trials
    #    corrFractShfl = np.mean(corrFractSh,axis=None) # average of trials across shuffles    
    #    corrClassShfl[numSamples,t,:] = corrFractShfl # frames x trials
        
    
    #%%        
    corrClassAll.append(corrClass)
    corrClassShflAll.append(corrClassShfl)


#%%
corrClassAll = np.array(corrClassAll)
corrClassShflAll = np.array(corrClassShflAll)

for iday in range(len(days)):
    plt.plot(np.mean(corrClassAll[iday],axis=1))
    

'''
plt.plot(np.mean(temp[:,:,Y_chAl==0],axis=(1,2)))
plt.plot(np.mean(temp[:,:,Y_chAl==1],axis=(1,2)))

#Tsa, Nsa, Csa = Xt_stimAl_all.shape
#Xtsa_N = np.reshape(Xt_stimAl_all.transpose(0 ,2 ,1), (Tsa*Csa, Nsa), order = 'F')
#Xtsa_N = (Xtsa_N-meanX)/stdX
#Xtsa = np.reshape(Xtsa_N, (Tsa, Csa, Nsa), order = 'F').transpose(0 ,2 ,1)
#w_normalized=w
#XtN_w = np.dot(Xtsa_N, w_normalized);
#Xt_w_i = np.reshape(XtN_w, (Tsa,Csa), order='F');

plt.plot(np.mean(Xt_w_i[:,Y_chAl==0],axis=(1)))
plt.plot(np.mean(Xt_w_i[:,Y_chAl==1],axis=(1)))

a0 = np.argwhere(Y_chAl==0)
a1 = np.argwhere(Y_chAl==1)

for i in range(len(a0)):
    plt.figure()
    plt.plot(np.mean(Xt_w_i[:,a0[i]],axis=(1)),color=[.5,.5,.5])
    plt.plot(np.mean(Xt_w_i[:,a0],axis=(1)),color='r')
    
for i in range(len(a1)):    
    plt.plot(np.mean(Xt_w_i[:,a1[i]],axis=(1)),color=[0,.1,0])
    plt.plot(np.mean(Xt_w_i[:,a1],axis=(1)),color='g')

plt.plot(np.mean(Xt_w_i[:,a0],axis=(1)),color='r')
plt.plot(np.mean(Xt_w_i[:,a1],axis=(1)),color='g')
    
    
    yhat = np.dot(X, w.T) + b # it gives -1 and 1... change -1 to 0... so it matches svm.predict
    if np.isnan(th)==0:
        yhat[yhat<th] = 0
        yhat[yhat>th] = 1    
'''
    
#%% Compute average across trials (also shuffles for corrClassShfl)

corrClass_ave = []
corrClassShfl_ave = []

for iday in range(numDays):
    corrClass_ave.append(np.mean(corrClassAll[iday], axis=1)) # numFrames x 1
    corrClassShfl_ave.append(np.mean(corrClassShflAll[iday], axis=(1,2))) # numFrames x 1
    
    
#%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
# By common eventI, we  mean the index on which all traces will be aligned.
        
nPost = (np.ones((numDays,1))+np.nan).flatten().astype('int')
for iday in range(numDays):
    nPost[iday] = (np.shape(corrClassAll[iday])[0] - eventI_allDays_chAl[iday] - 1)

nPreMin = min(eventI_allDays_chAl) # number of frames before the common eventI, also the index of common eventI. 
nPostMin = min(nPost)
print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)


#%% Set the time array for the across-day aligned traces

a = -(np.asarray(frameLength) * range(nPreMin+1)[::-1])
b = (np.asarray(frameLength) * range(1, nPostMin+1))
time_aligned = np.concatenate((a,b))


#%% Align traces of all days on the common eventI

corrClass_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)
corrClassShfl_aligned = np.ones((nPreMin + nPostMin + 1, numDays)) + np.nan # frames x days, aligned on common eventI (equals nPreMin)

for iday in range(numDays):
    corrClass_aligned[:, iday] = corrClass_ave[iday][eventI_allDays_chAl[iday] - nPreMin  :  eventI_allDays_chAl[iday] + nPostMin + 1]
    corrClassShfl_aligned[:, iday] = corrClassShfl_ave[iday][eventI_allDays_chAl[iday] - nPreMin  :  eventI_allDays_chAl[iday] + nPostMin + 1]


#%% keep short and long ITI results to plot them against each other later.
if trialHistAnalysis:
    if iTiFlg==0:
        corrClass_aligned0 = corrClass_aligned

    elif iTiFlg==1:
        corrClass_aligned1 = corrClass_aligned
        
        
#%% Average across days

corrClass_aligned_ave = np.mean(corrClass_aligned, axis=1) * 100
corrClass_aligned_std = np.std(corrClass_aligned, axis=1) * 100

corrClassShfl_aligned_ave = np.mean(corrClassShfl_aligned, axis=1) * 100
corrClassShfl_aligned_std = np.std(corrClassShfl_aligned, axis=1) * 100

_,pcorrtrace0 = stats.ttest_1samp(corrClass_aligned.transpose(), 50) # p value of class accuracy being different from 50

_,pcorrtrace = stats.ttest_ind(corrClass_aligned.transpose(), corrClassShfl_aligned.transpose()) # p value of class accuracy being different from 50
        
       
#%% Plot the average traces across all days
#ep_ms_allDays
plt.figure(figsize=(4.5,3))

plt.fill_between(time_aligned, corrClassShfl_aligned_ave - corrClassShfl_aligned_std, corrClassShfl_aligned_ave + corrClassShfl_aligned_std, alpha=0.5, edgecolor='k', facecolor='k')
plt.plot(time_aligned, corrClassShfl_aligned_ave, 'k')

plt.fill_between(time_aligned, corrClass_aligned_ave - corrClass_aligned_std, corrClass_aligned_ave + corrClass_aligned_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, corrClass_aligned_ave, 'r')

plt.xlabel('Time since choice onset (ms)', fontsize=13)
plt.ylabel('Classification accuracy (%)', fontsize=13)
plt.legend()
ax = plt.gca()
if trialHistAnalysis:
    plt.xticks(np.arange(-400,600,200))
else:    
    plt.xticks(np.sort(np.concatenate((np.arange(0,-1400,-400), np.arange(400,1000,400))))) #np.arange(-1000,600,400)

makeNicePlots(ax)

# Plot a dot for significant time points
ymin, ymax = ax.get_ylim()

pp = pcorrtrace+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
plt.plot(time_aligned, pp, color='k')

#plt.xticks(np.arange(-400,1400,200))
#win=[time_aligned[0], 0]
#plt.plot([win[0], win[0]], [ymin, ymax], '-.', color=[.7, .7, .7])
#plt.plot([win[1], win[1]], [ymin, ymax], '-.', color=[.7, .7, .7])
    
# Plot lines for the training epoch
if 'ep_ms_allDays' in locals():
    win = np.mean(ep_ms_allDays, axis=0)
    plt.plot([win[0], win[0]], [ymin, ymax], '-.', color=[.7, .7, .7])
    plt.plot([win[1], win[1]], [ymin, ymax], '-.', color=[.7, .7, .7])

#plt.savefig('classaccur_chAl_fni17.pdf', bbox_inches='tight')
##%% Save the figure    
if savefigs:
    for i in [0]:#range(np.shape(fmt)[0]): # save in all format of fmt         
        fign_ = suffn+'corrClassTrace'
        fign = os.path.join(svmdir+dnow, fign_+'.'+fmt[i])
        
        plt.savefig(fign, bbox_inches='tight')#, bbox_extra_artists=(lgd,))



  

















#%% I think we can project both stim-al and choice-al traces onto the decoder (found on choice-al) ... I don't think bc decoder was found on choice-al we have to project choice-al traces on it... I am actually curious how projs of stim-al on it looks like.

    ######################################## Plot choice-aligned averages after centering and normalization
    numTrials_chAl = X_chAl.shape[0]
    
    plt.figure()
    #################### correct trials
    # Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
    hr_trs = (Y_chAl==1)
    lr_trs = (Y_chAl==0)


    # Set traces
    Xt = traces_al_1stSide[:, :, ~trsExcluded_chAl];
    # Exclude non-active neurons (ie neurons that don't fire in any of the trials during ep)
    Xt = Xt[:,~NsExcluded_chAl,:]
    ## Feature normalization and scaling    
    # normalize stim-aligned traces
    T, N, C = Xt.shape
    Xt_N = np.reshape(Xt.transpose(0 ,2 ,1), (T*C, N), order = 'F')
    Xt_N = (Xt_N-meanX_chAl)/stdX_chAl
    Xt = np.reshape(Xt_N, (T, C, N), order = 'F').transpose(0 ,2 ,1)


    # Plot
    plt.subplot(1,2,1) # I think you should use Xtsa here to make it compatible with the plot above.
    a1 = np.nanmean(Xt[:, :, hr_trs],  axis=1) # frames x trials (average across neurons)
    tr1 = np.nanmean(a1,  axis = 1)
    tr1_se = np.nanstd(a1,  axis = 1) / np.sqrt(numTrials_chAl);
    a0 = np.nanmean(Xt[:, :, lr_trs],  axis=1) # frames x trials (average across neurons)
    tr0 = np.nanmean(a0,  axis = 1)
    tr0_se = np.nanstd(a0,  axis = 1) / np.sqrt(numTrials_chAl);    
    plt.fill_between(time_aligned_1stSide, tr1-tr1_se, tr1+tr1_se, alpha=0.5, edgecolor='b', facecolor='b')
    plt.fill_between(time_aligned_1stSide, tr0-tr0_se, tr0+tr0_se, alpha=0.5, edgecolor='r', facecolor='r')
    plt.plot(time_aligned_1stSide, tr1, 'b', label = 'high rate')
    plt.plot(time_aligned_1stSide, tr0, 'r', label = 'low rate')
    # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, lr_trs],  axis = (1, 2)), 'r', label = 'high rate')
    # plt.plot(time_aligned_stim, np.nanmean(Xt[:, :, hr_trs],  axis = (1, 2)), 'b', label = 'low rate')
    plt.xlabel('time aligned to choice onset (ms)')
    plt.title('Correct trials')
    
    
    

