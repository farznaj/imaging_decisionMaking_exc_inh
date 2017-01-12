# -*- coding: utf-8 -*-


# IMPORTANT NOTE: I think you should exclude days with chance testing-data decoding performance... bc it doesn't really make sense to compare their weight or fract-non0 of exc and inh neurons.
# For this you will need to get shuffles (shuffle labels of each trial)


"""
 You don't need to run any other scripts before this one. It works on its own.
 
 This script includes analyses to compare exc vs inh decoding [when same  number of exc and inh are concatenated to train SVM]
 
 fract non0 and weights are compared between exc and inh (at all c and at bestc).
 
 To get vars for this script, you need to run mainSVM_excInh_cPath.py... normally on the cluster!
 
 The saved mat files which will be loaded here are named excInhC_svmCurrChoice_ and excInhC_svmPrevChoice (for fni17, names are 'C2', bc mat files named excInhC do not have cross validation) 
 
 


Pool SVM results of all days and plot summary figures

Created on Sun Oct 30 14:41:01 2016
@author: farznaj
"""

# Go to svm_plots_setVars and define vars!
execfile("svm_plots_setVars.py")   

dnow = '/excInh_cPath'

if trialHistAnalysis:
    dp = '/previousChoice'
else:
    dp = '/currentChoice'
    
    
#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName
def setSVMname(pnevFileName, trialHistAnalysis, ntName, roundi, itiName='all'):
    import glob
    
    if trialHistAnalysis:
        if pnevFileName.find('fni17')!=-1: # fni17: C2 has test/trial shuffles as well. C doesn't have it.
            svmn = 'excInhC2_svmPrevChoice_%sN_%sITIs_ep*-*ms_*' %(ntName, itiName) 
        else:
            svmn = 'excInhC_svmPrevChoice_%sN_%sITIs_ep*-*ms_*' %(ntName, itiName) 
    else:
        if pnevFileName.find('fni17')!=-1: # fni17: C2 has test/trial shuffles as well. C doesn't have it.
            svmn = 'excInhC2_svmCurrChoice_%sN_ep%d-%dms_*' %(ntName, ep_ms[0], ep_ms[-1])
        else:
            svmn = 'excInhC_svmCurrChoice_%sN_ep*-*ms_*' %(ntName)   
                           
    svmn = svmn + pnevFileName[-32:]    
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
    svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    svmName = svmName[0] # get the latest file
    
    return svmName

 
    
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
    eav = np.nanmean(a, axis=1) # average across shuffles
    iav = np.nanmean(b, axis=1)
    ele = np.shape(a)[1] - np.sum(np.isnan(a),axis=1) # number of non-nan shuffles of each day
    ile = np.shape(b)[1] - np.sum(np.isnan(b),axis=1) # number of non-nan shuffles of each day
    esd = np.divide(np.nanstd(a, axis=1), np.sqrt(ele))
    isd = np.divide(np.nanstd(b, axis=1), np.sqrt(ile))
    
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
################# Excitatory vs inhibitory neurons:  c path #########################################################################################
######## Fraction of non-zero weights vs. different values of c #####################################################################################
#####################################################################################################################################################   
'''    
# Analysis done in svm_excInh_cPath.py:
# 1. For numSamples times a different X is created, which includes different sets of exc neurons but same set of inh neurons.
# 2. For each X, classifier is trained numShuffles_ei times (using different sets of training and testing datasets).
# 3. The above procedure is done for each value of c.

# Here:
# We take average across cvTrial shuffles. For weights, we also take average weight of all neurons
# Then (for plots) we take averages across neuron shuffles.
# As a result for each value of c we get average w, b, %non-0 w, and classError across all shuffles of cv trials and neurons.


wall_exc = []
wall_inh = []
wall_exc_abs = []
wall_inh_abs = []
wall_exc_abs_non0 = []
wall_inh_abs_non0 = []
wall_exc_non0 = []
wall_inh_non0 = []
perActiveAll_exc = []
perActiveAll_inh = []
cvect_all = []
perClassErAll = []
ball = []
perClassErTestAll = []    

# not averaged across trial shuffles
perClassErAll2 = [] # not averaged across trial shuffles
perClassErTestAll2 = [] # not averaged across trial shuffles
perActiveAll2_exc = [] # not averaged across trial shuffles
perActiveAll2_inh = [] # not averaged across trial shuffles
ball2 = []
wall2_exc = [] # average of neurons (but not averaged across trial or neuron shuffles)
wall2_inh = []
wall2_exc_abs = []
wall2_inh_abs = []
wall2_exc_abs_non0 = []
wall2_inh_abs_non0 = []

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
    Data = scio.loadmat(moreName, variable_names=['inhibitRois'])
    inhibitRois = Data.pop('inhibitRois')[0,:]

    
    #%%     
    svmName = setSVMname(pnevFileName, trialHistAnalysis, ntName, np.nan, itiName)    
    print os.path.basename(svmName)
        
    ##%% Load vars
    Data = scio.loadmat(svmName, variable_names=['perActive_exc', 'perActive_inh', 'wei_all', 'perClassEr', 'cvect_', 'bei_all', 'perClassErTest'])
    perActive_exc = Data.pop('perActive_exc') # numSamples x numTrialShuff x length(cvect_)  # numSamples x length(cvect_)  # samples: shuffles of neurons
    perActive_inh = Data.pop('perActive_inh') # numSamples x numTrialShuff x length(cvect_)  # numSamples x length(cvect_)        
    wei_all = Data.pop('wei_all') # numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers) # numSamples x length(cvect_) x numNeurons(inh+exc equal numbers)  
    perClassEr = Data.pop('perClassEr') # numSamples x numTrialShuff x length(cvect_)
    cvect_ = Data.pop('cvect_').flatten() # length(cvect_)
    bei_all = Data.pop('bei_all') # numSamples x numTrialShuff x length(cvect_)  # numTrialShuff is number of times you shuffled trials to do cross validation. (it is variable numShuffles_ei in function inh_exc_classContribution)
    perClassErTest = Data.pop('perClassErTest') # numSamples x numTrialShuff x length(cvect_) 
    
    if abs(wei_all.sum()) < eps:
        print '\tWeights of all c and all shuffles are 0' # ... not analyzing' # I dont think you should do this!!
        
#    else:            
    ##%% Load vars (w, etc)
    Data = scio.loadmat(svmName, variable_names=['NsExcluded', 'NsRand'])        
    NsExcluded = Data.pop('NsExcluded')[0,:].astype('bool')
    NsRand = Data.pop('NsRand')[0,:].astype('bool')

    # Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)        
    inhRois = inhibitRois[~NsExcluded]        
    inhRois = inhRois[NsRand]
        
    n = sum(inhRois==1)                
    inhRois_ei = np.zeros((2*n));
    inhRois_ei[n:2*n] = 1;


    # keep %non-0 w and classError for all shuffles of cvTrials and neurons (for all c values for all days)
    perActiveAll2_exc.append(perActive_exc) # numDays x numSamples x numTrialShuff x length(cvect_)        
    perActiveAll2_inh.append(perActive_inh)  # numDays x numSamples x numTrialShuff x length(cvect_)               
    perClassErAll2.append(perClassEr) # numDays x numSamples x numTrialShuff x length(cvect_)    
    perClassErTestAll2.append(perClassErTest) # numDays x numSamples x numTrialShuff x length(cvect_)        
    ball2.append(bei_all) # numDays x numSamples x numTrialShuff x length(cvect_) 
         
    wall2_exc.append(np.mean(wei_all[:,:,:,inhRois_ei==0], axis=-1)) #average across exc (inh) neurons # numDays (each day: numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers))     
    wall2_inh.append(np.mean(wei_all[:,:,:,inhRois_ei==1], axis=-1))                
    wall2_exc_abs.append(np.mean(abs(wei_all[:,:,:,inhRois_ei==0]), axis=-1)) #average across exc (inh) neurons # numDays (each day: numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers))     
    wall2_inh_abs.append(np.mean(abs(wei_all[:,:,:,inhRois_ei==1]), axis=-1)) 
    # non0
    a = wei_all[:,:,:,inhRois_ei==0] + 0
    a[a==0] = np.nan
    wall2_exc_abs_non0.append(np.nanmean(abs(a), axis=-1)) # average across neurons   
    
    b = wei_all[:,:,:,inhRois_ei==1] + 0
    b[b==0] = np.nan
    wall2_inh_abs_non0.append(np.nanmean(abs(b), axis=-1))    
       
       
    ######### Take averages across trialShuffles        
    wall_exc.append(np.mean(wei_all[:,:,:,inhRois_ei==0], axis=1)) #average w across trialShfls # numDays (each day: numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers))     
    wall_inh.append(np.mean(wei_all[:,:,:,inhRois_ei==1], axis=1))      
    # I dont think this argument makes sense: for abs, I am not sure if you need to do the following... If you don't do it, the idea is that w of each neuron is represented by the average of its w across all trial shuffles... so we dont need to do abs!          
    wall_exc_abs.append(np.mean(abs(wei_all[:,:,:,inhRois_ei==0]), axis=1)) #average abs w across trialShfls # numDays (each day: numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers))     
    wall_inh_abs.append(np.mean(abs(wei_all[:,:,:,inhRois_ei==1]), axis=1))    
    # take average of abs of non0 ws across tr shuffles.
    wall_exc_abs_non0.append(np.nanmean(abs(a), axis=1))    
    wall_inh_abs_non0.append(np.nanmean(abs(b), axis=1))        
    # take average of non0 ws across tr shuffles.
    wall_exc_non0.append(np.nanmean(a, axis=1))    
    wall_inh_non0.append(np.nanmean(b, axis=1))            
#    wall_exc.append(wei_all[:,:,inhRois_ei==0]) # numDays (each day: numSamples x length(cvect_) x numNeurons(inh+exc equal numbers))     
#    wall_inh.append(wei_all[:,:,inhRois_ei==1])        
    
    ball.append(np.mean(bei_all, axis=1)) # numDays x numSamples x length(cvect_)   
    
    perActiveAll_exc.append(np.mean(perActive_exc, axis=1)) # numDays x numSamples x length(cvect_)        
    perActiveAll_inh.append(np.mean(perActive_inh, axis=1))        

    cvect_all.append(cvect_) # numDays x length(cvect_)
    perClassErAll.append(np.mean(perClassEr, axis=1)) # numDays x numSamples x length(cvect_)     
    perClassErTestAll.append(np.mean(perClassErTest, axis=1)) # numDays x numSamples x length(cvect_)     




#%%    
perActiveAll_exc = np.array(perActiveAll_exc)  # days x numSamples x length(cvect_) # percent non-zero weights
perActiveAll_inh = np.array(perActiveAll_inh)  # days x numSamples x length(cvect_) # percent non-zero weights
perClassErAll = np.array(perClassErAll)  # days x numSamples x length(cvect_) # classification error
cvect_all = np.array(cvect_all) # days x length(cvect_)  # c values
perClassErTestAll = np.array(perClassErTestAll)
ball = np.array(ball)

perClassErAll2 = np.array(perClassErAll2) # numDays x numSamples x numTrialShuff x length(cvect_) 
perClassErTestAll2 = np.array(perClassErTestAll2) # numDays x numSamples x numTrialShuff x length(cvect_) 
perActiveAll2_exc = np.array(perActiveAll2_exc)
perActiveAll2_inh = np.array(perActiveAll2_inh)
ball2 = np.array(ball2)
wall2_exc = np.array(wall2_exc)
wall2_inh = np.array(wall2_inh)
wall2_exc_abs = np.array(wall2_exc_abs)
wall2_inh_abs = np.array(wall2_inh_abs)
wall2_exc_abs_non0 = np.array(wall2_exc_abs_non0)
wall2_inh_abs_non0 = np.array(wall2_inh_abs_non0)


#%% For each day average weights across neurons
nax = 2 # dimension of neurons in wall_exc (since u're averaging across trialShuffles, it will be 2.) (if cross validation is performed, it will be 3, ie in excinhC2 mat files. Otherwise it will be 2.)
wall_exc_aveN = []
wall_inh_aveN = []
wall_abs_exc_aveN = [] # average of abs values of weights
wall_abs_inh_aveN = []
wall_non0_exc_aveN = [] # average of abs values of weights
wall_non0_inh_aveN = []
wall_non0_abs_exc_aveN = [] # average of abs values of weights
wall_non0_abs_inh_aveN = []

for iday in range(len(days)):

    wall_exc_aveN.append(np.mean(wall_exc[iday], axis=nax)) # days x numSamples x length(cvect_)  # ave w across neurons
    wall_inh_aveN.append(np.mean(wall_inh[iday], axis=nax)) # days x numSamples x length(cvect_)  # ave w across neurons


    # non0 w, exc
    a = wall_exc_non0[iday] + 0
    a[wall_exc_non0[iday]==0] = np.nan # only take non-0 weights
    wall_non0_exc_aveN.append(np.nanmean(a, axis=nax)) # days x numSamples x length(cvect_)  # ave w across neurons
    # non0 w, inh
    a = wall_inh_non0[iday] + 0 
    a[wall_inh_non0[iday]==0] = np.nan
    wall_non0_inh_aveN.append(np.nanmean(a, axis=nax)) # days x numSamples x length(cvect_)  # ave w across neurons    

    
    # You fixed the following by using wall_exc_abs instead of wall_exc below (although if u think of average w of tr shufs as the representative of the neuron w you may not want use abs... i dont know...im confused); NOTE: wall_exc is average w of trial shuffles, not average of abs w... so I think all the abs values that you are computing below are wrong!    
    # abs w
    wall_abs_exc_aveN.append(np.mean(abs(wall_exc_abs[iday]), axis=nax)) # days x numSamples x length(cvect_)  # ave w across neurons
    wall_abs_inh_aveN.append(np.mean(abs(wall_inh_abs[iday]), axis=nax)) # days x numSamples x length(cvect_)  # ave w across neurons
    
    
    # non0 abs w, exc
    a = wall_exc_abs_non0[iday] + 0
    a[wall_exc_abs_non0[iday]==0] = np.nan # only take non-0 weights
    wall_non0_abs_exc_aveN.append(np.nanmean(abs(a), axis=nax)) # days x numSamples x length(cvect_)  # ave w across neurons   
    # non0 abs w, inh
    a = wall_inh_abs_non0[iday] + 0 
    a[wall_inh_abs_non0[iday]==0] = np.nan
    wall_non0_abs_inh_aveN.append(np.nanmean(abs(a), axis=nax)) # days x numSamples x length(cvect_)  # ave w across neurons

    
wall_exc_aveN = np.array(wall_exc_aveN)
wall_inh_aveN = np.array(wall_inh_aveN)
wall_abs_exc_aveN = np.array(wall_abs_exc_aveN)
wall_abs_inh_aveN = np.array(wall_abs_inh_aveN)
wall_non0_exc_aveN = np.array(wall_non0_exc_aveN)
wall_non0_inh_aveN = np.array(wall_non0_inh_aveN)
wall_non0_abs_exc_aveN = np.array(wall_non0_abs_exc_aveN)
wall_non0_abs_inh_aveN = np.array(wall_non0_abs_inh_aveN)


#%% Average weights,etc across neuron shuffles for each day : numDays x length(cvect_) (by shuffles we mean neuron shuffles ... not trial shuffles... you alreade averaged that when loading vars above)
# NOTE: you may want to exclude shuffles with all0 weights from averaging

#avax = (0,1) # (0) # average across axis; use (0,1) if shuffles of trials (for cross validation) as well as neurons are available. Otherwise use (0) if no crossvalidation was performed

wall_exc_aveNS = np.array([np.mean(wall_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # average w across shuffles for each day
wall_inh_aveNS = np.array([np.mean(wall_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
wall_abs_exc_aveNS = np.array([np.mean(wall_abs_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # average w across shuffles for each day
wall_abs_inh_aveNS = np.array([np.mean(wall_abs_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
wall_non0_exc_aveNS = np.array([np.nanmean(wall_non0_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # average w across shuffles for each day
wall_non0_inh_aveNS = np.array([np.nanmean(wall_non0_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
wall_non0_abs_exc_aveNS = np.array([np.nanmean(wall_non0_abs_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # average w across shuffles for each day
wall_non0_abs_inh_aveNS = np.array([np.nanmean(wall_non0_abs_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
perClassErAll_aveS = np.array([np.mean(perClassErAll[iday,:,:], axis=0) for iday in range(len(days))])
perClassErTestAll_aveS = np.array([np.mean(perClassErTestAll[iday,:,:], axis=0) for iday in range(len(days))])
perActiveAll_exc_aveS = np.array([np.mean(perActiveAll_exc[iday,:,:], axis=0) for iday in range(len(days))])
perActiveAll_inh_aveS = np.array([np.mean(perActiveAll_inh[iday,:,:], axis=0) for iday in range(len(days))])

# std
wall_exc_stdNS = np.array([np.std(wall_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # std of w across shuffles for each day
wall_inh_stdNS = np.array([np.std(wall_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
wall_abs_exc_stdNS = np.array([np.std(wall_abs_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # std of w across shuffles for each day
wall_abs_inh_stdNS = np.array([np.std(wall_abs_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
wall_non0_exc_stdNS = np.array([np.nanstd(wall_non0_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # average w across shuffles for each day
wall_non0_inh_stdNS = np.array([np.nanstd(wall_non0_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
wall_non0_abs_exc_stdNS = np.array([np.nanstd(wall_non0_abs_exc_aveN[iday,:,:], axis=0) for iday in range(len(days))]) # days x length(cvect_) # average w across shuffles for each day
wall_non0_abs_inh_stdNS = np.array([np.nanstd(wall_non0_abs_inh_aveN[iday,:,:], axis=0) for iday in range(len(days))])
perClassErAll_stdS = np.array([np.std(perClassErAll[iday,:,:], axis=0) for iday in range(len(days))])
perClassErTestAll_stdS = np.array([np.std(perClassErTestAll[iday,:,:], axis=0) for iday in range(len(days))])
perActiveAll_exc_stdS = np.array([np.std(perActiveAll_exc[iday,:,:], axis=0) for iday in range(len(days))])
perActiveAll_inh_stdS = np.array([np.std(perActiveAll_inh[iday,:,:], axis=0) for iday in range(len(days))])



#%%
# BEST C

##########################################################################################################################################################################
################################## compare weights and fract non-0 of exc and inh at bestc of each day ##################################
##########################################################################################################################################################################

#%% Compute bestc for each day (for the exc_inh population decdoder) using average of classError across neuron shuffles and trial shuffles.

# before anything take a look at cv class error vs c to make sure it has a legitimate shape! if it doesnt, finding bestc is meaningless!
'''
for iday in range(len(days)):
    plt.plot(np.mean(perClassErTestAll2[iday,:,:,:], axis = (0,1))) # average across neuron and trial shuffles
    plt.show()
'''    
    
#######%%%%%%        
numSamples = np.shape(perClassErTestAll)[1]

'''
for iday in range(len(days)):   
    plt.figure()
    plt.title(days[iday])
    for ineurshfl in range(numSamples):
        a = np.mean(perClassErTestAll2[iday,ineurshfl,:,:], axis = 0)
        plt.plot(cvect_all[iday,:], a, color=[.6,.6,.6]) # average across neuron and trial shuffles        
        plt.xscale('log')
        
    a = np.mean(perClassErTestAll2[iday,:,:,:], axis = (0,1))
    plt.plot(cvect_all[iday,:], a, color='r')
#    plt.plot(cbestAll[iday], a[cvect_all[iday,:]==cbestAll[iday]], 'ro')
    plt.xlabel('c')
    plt.ylabel('% CV Class error')
#        plt.show()
'''    
    
cbestAll = np.full((len(days)), np.nan)
for iday in range(len(days)):

    cvect = cvect_all[iday,:]

    # Pick bestc from all range of c... it may end up a value at which all weights are 0. In the analysis below though you exclude shuffles with all0 weights.
    meanPerClassErrorTest = np.mean(perClassErTestAll[iday,:,:], axis = 0) # perClassErTestAll_aveS[iday,:]
    semPerClassErrorTest = np.std(perClassErTestAll[iday,:,:], axis = 0)/np.sqrt(numSamples) # perClassErTestAll_stdS[iday,:]/np.sqrt(np.shape(perClassErTestAll)[1])
    ix = np.nanargmin(meanPerClassErrorTest)
    cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
    cbest = cbest[0]; # best regularization term based on minError+SE criteria
    cbestAll[iday] = cbest
    
    
    # Another method: Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)        
    # compute percent non-zero w at each c
    a = perActiveAll_exc[iday,:,:].squeeze()
    b = perActiveAll_inh[iday,:,:].squeeze()    
    c = np.mean((a,b), axis=0) # percent non-0 w for each shuffle at each c (including both exc and inh neurons. Remember SVM was trained on both so there is a single decoder including both exc and inh)
    
#    a = (wAllC!=0) # non-zero weights
#    b = np.mean(a, axis=(0,2)) # Fraction of non-zero weights (averaged across shuffles)
    b = np.mean(c, axis=(0)) # Fraction of non-zero weights (averaged across shuffles)
    c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle
    
    cvectnow = cvect[c1stnon0:]
    meanPerClassErrorTestnow = np.mean(perClassErTestAll[iday,:,c1stnon0:], axis = 0);
    semPerClassErrorTestnow = np.std(perClassErTestAll[iday,:,c1stnon0:], axis = 0)/np.sqrt(numSamples);
    ix = np.argmin(meanPerClassErrorTestnow)
    cbest = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
    cbest = cbest[0]; # best regularization term based on minError+SE criteria
    cbestAll[iday] = cbest

print cbestAll


# show cbestAll
'''
for iday in range(len(days)):    
    a = np.mean(perClassErTestAll2[iday,:,:,:], axis = (0,1))
    plt.plot(cvect_all[iday,:], a) # average across neuron and trial shuffles    
    plt.plot(cbestAll[iday], a[cvect_all[iday,:]==cbestAll[iday]], 'ro')
    plt.xscale('log')
    
    plt.plot(cvect_all[iday,:], np.mean(perActiveAll2_exc[iday,:,:,:],axis=(0,1)), color='k')
    plt.plot(cvect_all[iday,:], np.mean(perActiveAll2_inh[iday,:,:,:],axis=(0,1)), color='r')       
#    a = perActiveAll2_exc[iday,:,:,cvect_all[iday,:]==cbestAll[iday]].squeeze()
#    np.mean(a,axis=(0,1))   
    plt.show()
'''    

plt.figure(figsize=(6,17))    
for iday in range(len(days)):   
#    plt.figure()
    plt.subplot(np.ceil(len(days)/float(3)),3,iday+1)
    plt.title(days[iday])
    for ineurshfl in range(numSamples):
        a = np.mean(perClassErTestAll2[iday,ineurshfl,:,:], axis = 0)
        plt.plot(cvect_all[iday,:], a, color=[.6,.6,.6]) # average across neuron and trial shuffles        
        plt.xscale('log')
        
    a = np.mean(perClassErTestAll2[iday,:,:,:], axis = (0,1))
    plt.plot(cvect_all[iday,:], a, color='r')
    plt.plot(cbestAll[iday], a[cvect_all[iday,:]==cbestAll[iday]], 'ro')
    ax = plt.gca()
    makeNicePlots(ax)
    
    if iday!=len(days)-1:
        ax.xaxis.set_ticklabels([])
    
plt.xlabel('c')
plt.subplot(np.ceil(len(days)/float(3)),3,iday+1)
plt.ylabel('% CV Class error')
plt.subplots_adjust(wspace=.8, hspace=.4)


if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnow,mousename+dp+'/bestC')
#    d = os.path.join(svmdir+dnow,mousename+'/bestC'+dp) #, dgc, dta)
#    d = os.path.join(svmdir+dnow,mousename+'/bestC'+dp)
#    d = os.path.join(svmdir+dnow+'/bestC',mousename+dp)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
    
    fign = os.path.join(d, suffn[0:5]+'classErrVsC_eachDay.'+fmt[0])
#    fign = os.path.join(d, suffn[0:5]+'classErrVsC_'+days[iday][0:6]+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')
        
        



########################################################################    
# Another method: For each neuron shuffle compute bestc (not using the class error that is average neuron shuffles)
    # you figured it is not a good method because bestc estimation is much better when u do it on neuron shuffle averages... some of the single neuron shuffles have bad c plots, which will result in bad estimated of bestc... 
    # in brief, it is much better to find bestc on a nice and smooth c plot!
numCVtrShfls = np.shape(perClassErTestAll)[1]
"""
numSamples = np.shape(perClassErTestAll)[1]
cbestAll = np.full((len(days),numSamples), np.nan)

for iday in range(len(days)):

    cvect = cvect_all[iday,:]

    for ineurshfl in range(numSamples):
        # Pick bestc from all range of c... it may end up a value at which all weights are 0. In the analysis below though you exclude shuffles with all0 weights.
        # set (for each neuron shuffle) average and sd of classError across trialShuffles.
        meanPerClassErrorTest = np.mean(perClassErTestAll2[iday,ineurshfl,:,:], axis = 0) # perClassErTestAll_aveS[iday,:]
        semPerClassErrorTest = np.std(perClassErTestAll2[iday,ineurshfl,:,:], axis = 0)/np.sqrt(numCVtrShfls) # perClassErTestAll_stdS[iday,:]/np.sqrt(np.shape(perClassErTestAll)[1])
        ix = np.nanargmin(meanPerClassErrorTest)
        cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
        cbest = cbest[0]; # best regularization term based on minError+SE criteria
        cbestAll[iday,ineurshfl] = cbest
        
        
        # Another method: Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)        
        # compute percent non-zero w at each c
        a = perActiveAll2_exc[iday,ineurshfl,:,:].squeeze() # fraction of non0 exc neurons for all trial shuffles and c values of a particular day and neuron shuffle
        b = perActiveAll2_inh[iday,ineurshfl,:,:].squeeze()    
        c = np.mean((a,b), axis=0) #numTrialShuff x length(cvect_) # percent non-0 w for each shuffle at each c (including both exc and inh neurons. Remember SVM was trained on both so there is a single decoder including both exc and inh)
        
    #    a = (wAllC!=0) # non-zero weights
    #    b = np.mean(a, axis=(0,2)) # Fraction of non-zero weights (averaged across shuffles)
        b = np.mean(c, axis=(0)) # Fraction of non-zero weights (averaged across trial shuffles)
        c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle
        
        cvectnow = cvect[c1stnon0:]
        meanPerClassErrorTestnow = np.mean(perClassErTestAll2[iday,ineurshfl,:,c1stnon0:], axis = 0);
        semPerClassErrorTestnow = np.std(perClassErTestAll2[iday,ineurshfl,:,c1stnon0:], axis = 0)/np.sqrt(numCVtrShfls);
        ix = np.argmin(meanPerClassErrorTestnow)
        cbest = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
        cbest = cbest[0]; # best regularization term based on minError+SE criteria
        cbestAll[iday,ineurshfl] = cbest
        
print cbestAll
'''
# show cbestAll
for iday in range(len(days)):   
    for ineurshfl in range(numSamples):
        a = np.mean(perClassErTestAll2[iday,ineurshfl,:,:], axis = 0)
        plt.plot(cvect_all[iday,:], a) # average across neuron and trial shuffles
        plt.plot(cbestAll[iday,ineurshfl], a[cvect_all[iday,:]==cbestAll[iday,ineurshfl]], 'ro')
        plt.xscale('log')
        plt.show()
'''        

"""


########################################################################        
########################################################################        
#%% given the figure above decide what days you want to exclude (days with bad shaped c plots!).. for now you are doing this manually... later u can use: days w shuffles for which %non0 at best c is 0 are always among bad days, but not all of the bad days... at best classErr must be less than classErr at the very low c... if you get the diff of c plot it must decrease at some point...

if mousename=='fni16':
    if trialHistAnalysis:
        days2excl = ['1012','1027','1029', '1028','1023','1021'] # fni16, prev #1016
    else:
        days2excl = ['1016','1026','1027','1028','1029', '1012','1020'] # fni16, curr 
elif mousename=='fni17':
    if trialHistAnalysis:
        days2excl = ['1007','1010','1013','1014','1019','1022','1023','1027','1028','1029','1101','1102'] # fni16, prev #1016
    else:
        days2excl = ['1102'] # fni16, curr         


# identify the index of days to be excluded!
#[x for x in days if '1012' in x]
xall = []
for i in range(len(days2excl)):
    xall.append([x for x in range(len(days)) if days[x].find(days2excl[i])!=-1])
xall = np.array(xall).squeeze()    

print np.array(days)[xall]


#%% Set the two vars below: 1) whether you want to exclude days with bad shape for c plot (identified above). 2) whether you want to pool or average trial shuffles.

onlygoodc = 1 # if 1 only good c days will be plotted (for now u r manually choosing days to be excluded based on their cv class error vs c plot)
trShflAve = 0 # if 1, %non0 and w will be computed on trial-shuffle averaged variales... if 0, they will be computed for all trial shuffles (and neuron shuffles) and plots in the next section will be made on the pooled shuffles (instead of tr shfl averages) 

days2ana = np.array(range(len(days))) 


if onlygoodc:    
    dgc = 'onlyGoodcDays'
else:
    dgc = 'allDays'


if trShflAve:
    dta = 'trShfl_averaged'
else:
    dta = 'trShfl_notAveraged'
print dta

#####
if onlygoodc:
    days2ana = np.delete(days2ana,xall)

print days2ana
print len(days2ana), 'days for analysis'
    



#%% Set %non0 and weights for exc and inh at bestc (across all shuffls) 
# you also set shuffles with all0 w (for both exc and inh) to nan.
# QUITE IMPORTANT NOTE: I think you should exclude days with poor c plot (or chance testing-data decoding performance)... bc it doesn't really make sense to compare their weight or fract-non0 of exc and inh neurons.
# For this you will need to get shuffles (shuffle labels of each trial)

#trShflAve = 1 # if 1, %non0 and w will be computed on trial-shuffle averaged variales... if 0, they will be computed for all trial shuffles (and neuron shuffles) and plots in the next section will be made on the pooled shuffles (instead of tr shfl averages) 

if trShflAve==1: #
    perActiveExc = []
    perActiveInh = []
    wNon0AbsExc = []
    wNon0AbsInh = []    
    wAbsExc = []
    wAbsInh = []
    wExc = []
    wInh = []
    perClassErTest_bestc = np.full(len(days), np.nan)
    perClassErTest_bestc_sd = np.full(len(days), np.nan)
    
    for iday in days2ana: #range(len(days)): # days2ana (if you want to exclude days2excl)
        
        ibc = cvect_all[iday,:]==cbestAll[iday]
        
        # class error for cv data at bestc 
        perClassErTest_bestc[iday] = perClassErTestAll_aveS[iday,ibc]
        perClassErTest_bestc_sd[iday] = perClassErTestAll_stdS[iday,ibc]
        
        
        # percent non-zero w
        a = perActiveAll_exc[iday,:,ibc].squeeze() # perActiveAll_exc includes average across trial shuffles.
        b = perActiveAll_inh[iday,:,ibc].squeeze()
        c = np.mean((a,b), axis=0) #nShuffs x 1 # percent non-0 w for each shuffle (including both exc and inh neurons. Remember SVM was trained on both so there is a single decoder including both exc and inh)
        i0 = (c==0) # shuffles at which all w (of both exc and inh) are 0, ie no decoder was identified for these shuffles.
        a[i0] = np.nan # ignore shuffles with all-0 weights
        b[i0] = np.nan # ignore shuffles with all-0 weights    
        #    a[a==0.] = np.nan # ignore shuffles with all-0 weights ... this is problematic bc u will exclude a shuffle if inh w are all 0 even if exc w are not.
        #    b[b==0.] = np.nan # ignore shuffles with all-0 weights
        perActiveExc.append(a)
        perActiveInh.append(b)
        
        # abs non-zero w
        a = wall_non0_abs_exc_aveN[iday,:,ibc].squeeze() # for some shuffles this will be nan bc all weights were 0 ... so automatically you are ignoring shuffles w all-0 weights
        b = wall_non0_abs_inh_aveN[iday,:,ibc].squeeze()
        #    print np.argwhere(np.isnan(a))
        wNon0AbsExc.append(a)
        wNon0AbsInh.append(b)
        
        # abs w
        a = wall_abs_exc_aveN[iday,:,ibc].squeeze() 
        b = wall_abs_inh_aveN[iday,:,ibc].squeeze()
        a[i0] = np.nan # ignore shuffles with all-0 weights
        b[i0] = np.nan # ignore shuffles with all-0 weights    
        #    a[a==0.] = np.nan # ignore shuffles with all-0 weights
        #    b[b==0.] = np.nan # ignore shuffles with all-0 weights    
        wAbsExc.append(a)
        wAbsInh.append(b)
        
        
        # w
        a = wall_non0_exc_aveN[iday,:,ibc].squeeze() 
        b = wall_non0_inh_aveN[iday,:,ibc].squeeze()
        a[i0] = np.nan # ignore shuffles with all-0 weights
        b[i0] = np.nan # ignore shuffles with all-0 weights    
        #    a[a==0.] = np.nan # ignore shuffles with all-0 weights
        #    b[b==0.] = np.nan # ignore shuffles with all-0 weights    
        wExc.append(a)
        wInh.append(b)
    
        
    perActiveExc = np.array(perActiveExc)
    perActiveInh = np.array(perActiveInh)
    wNon0AbsExc = np.array(wNon0AbsExc)
    wNon0AbsInh = np.array(wNon0AbsInh)    
    wAbsExc = np.array(wAbsExc)
    wAbsInh = np.array(wAbsInh) 
    wExc = np.array(wExc)
    wInh = np.array(wInh) 


else:
    ######## not averaged across trial shuffles
    # we are using the bestc found on the average of neuron shuffles but we take % non-0 and abs w for each individual trial and neuron shuffle (unlike above which is done on the average of trial shuffles)
    #np.shape(perActiveAll2_inh[iday,ineurshfl,:,])
    
    perActiveExc = np.full((len(days),numSamples,numCVtrShfls), np.nan)
    perActiveInh = np.full((len(days),numSamples,numCVtrShfls), np.nan)
    wNon0AbsExc = np.full((len(days),numSamples,numCVtrShfls), np.nan)
    wNon0AbsInh = np.full((len(days),numSamples,numCVtrShfls), np.nan)    
    wAbsExc = np.full((len(days),numSamples,numCVtrShfls), np.nan)
    wAbsInh = np.full((len(days),numSamples,numCVtrShfls), np.nan)
    perClassErTest_bestc = np.full((len(days),numSamples), np.nan)
    perClassErTest_bestc_sd = np.full((len(days),numSamples), np.nan)
    wExc = np.full((len(days),numSamples,numCVtrShfls), np.nan)
    wInh = np.full((len(days),numSamples,numCVtrShfls), np.nan)
    
    for iday in days2ana: #range(len(days)):
        
        for ineurshfl in range(numSamples):
    #        ibc = cvect_all[iday,:]==cbestAll[iday,ineurshfl]
            ibc = cvect_all[iday,:]==cbestAll[iday]
        
            # class error of cv data for each neuron shuffle at bestc (averaged across trial shuffles)
            perClassErTest_bestc[iday,ineurshfl] = np.mean(perClassErAll2[iday,ineurshfl,:,ibc]) #perClassErTestAll_aveS[iday,ibc]
            perClassErTest_bestc_sd[iday,ineurshfl] = np.std(perClassErAll2[iday,ineurshfl,:,ibc]) #
            
            
            # percent non-zero w for all trial shuffles (of each particular neuron shuffle) at bestc
            a = perActiveAll2_exc[iday,ineurshfl,:,ibc].squeeze()
            b = perActiveAll2_inh[iday,ineurshfl,:,ibc].squeeze()
            # To find bestc u made sure in the average of trial shuffles at least 1 weight is non-0.... here u exclude each trial shuffle that has all 0 weights.... so it is not same as what you did for bestc.
            c = np.mean((a,b), axis=0) #trialShfl x 1 # percent non-0 w for each trial shuffle (including both exc and inh neurons. Remember SVM was trained on both so there is a single decoder including both exc and inh)
            if sum(c==0)>0: # these are bad days and u can exclude them! (automatic way of finding bad c plot days)
                print iday,ineurshfl,sum(c==0)
            i0 = (c==0) # trial shuffles at which all w (of both exc and inh) are 0, ie no decoder was identified for these shuffles.
            a[i0] = np.nan # ignore trial shuffles with all-0 weights
            b[i0] = np.nan # ignore trial shuffles with all-0 weights    
        #    a[a==0.] = np.nan # ignore shuffles with all-0 weights ... this is problematic bc u will exclude a shuffle if inh w are all 0 even if exc w are not.
        #    b[b==0.] = np.nan # ignore shuffles with all-0 weights
            perActiveExc[iday,ineurshfl,:] = a # days x neuron shuffles x trial shuffles
            perActiveInh[iday,ineurshfl,:] = b # days x neuron shuffles x trial shuffles
    
    
            # abs non-zero w for all trial shuffles (of each particular neuron shuffle) at bestc
            a = wall2_exc_abs_non0[iday,ineurshfl,:,ibc].squeeze() + 0
            a[a==0] = np.nan # only take non-0 weights
            wNon0AbsExc[iday,ineurshfl,:] = abs(a) # abs average w of exc neurons for all trial shuffles of a particular day and neuron shuffle at bestc # for some shuffles this will be nan bc all weights were 0 ... so automatically you are ignoring shuffles w all-0 weights
            
            b = wall2_inh_abs_non0[iday,ineurshfl,:,ibc].squeeze() + 0 # abs average w of inh neurons for all trial shuffles of a particular day and neuron shuffle at bestc
            b[b==0] = np.nan # only take non-0 weights
            wNon0AbsInh[iday,ineurshfl,:] = abs(b)
            
            
            
            # abs w
            a = abs(wall2_exc_abs[iday,ineurshfl,:,ibc]).squeeze() # abs average w of exc neurons for all trial shuffles of a particular day and neuron shuffle at bestc
            b = abs(wall2_inh_abs[iday,ineurshfl,:,ibc]).squeeze()
            a[i0] = np.nan # ignore trial shuffles with all-0 weights
            b[i0] = np.nan # ignore trial shuffles with all-0 weights    
        #    a[a==0.] = np.nan # ignore shuffles with all-0 weights
        #    b[b==0.] = np.nan # ignore shuffles with all-0 weights    
            wAbsExc[iday,ineurshfl,:] = a
            wAbsInh[iday,ineurshfl,:] = b
        
    
    
            # w
            a = wall2_exc[iday,ineurshfl,:,ibc].squeeze() # abs average w of exc neurons for all trial shuffles of a particular day and neuron shuffle at bestc
            b = wall2_inh[iday,ineurshfl,:,ibc].squeeze()
            a[i0] = np.nan # ignore trial shuffles with all-0 weights
            b[i0] = np.nan # ignore trial shuffles with all-0 weights    
        #    a[a==0.] = np.nan # ignore shuffles with all-0 weights
        #    b[b==0.] = np.nan # ignore shuffles with all-0 weights    
            wExc[iday,ineurshfl,:] = a
            wInh[iday,ineurshfl,:] = b
            
            
    # average across trial shuffles
    '''
    perActiveExc = np.nanmean(perActiveExc, axis=-1)
    perActiveInh = np.nanmean(perActiveInh, axis=-1)
    
    wNon0AbsExc = np.nanmean(wNon0AbsExc, axis=-1)
    wNon0AbsInh = np.nanmean(wNon0AbsInh, axis=-1)
    
    wAbsExc = np.nanmean(wAbsExc, axis=-1)
    wAbsInh = np.nanmean(wAbsInh, axis=-1)
    '''

#%%
###################################### PLOTS ###################################### 

#%%
###############%% classification accuracy of cv data at best c for each day ###############
# for each day plot averages across shuffles (neuron shuffles)
if trShflAve:
    plt.figure(figsize=(5,2.5))    
    gs = gridspec.GridSpec(1, 10)#, width_ratios=[2, 1]) 
    h1 = gs[0,0:7]
    ax = plt.subplot(h1)
    plt.errorbar(np.arange(len(days)), 100-perClassErTest_bestc, perClassErTest_bestc_sd, color='k')
    plt.xlim([-1,len(days)+1])
    plt.xlabel('Days')
    plt.ylabel('% Class accuracy')
    ymin, ymax = ax.get_ylim()
    makeNicePlots(plt.gca())
    
    # ave and std across days
    h2 = gs[0,7:8]
    ax = plt.subplot(h2)
    plt.errorbar(0, np.nanmean(100-perClassErTest_bestc), np.nanstd(100-perClassErTest_bestc), marker='o', color='k')
    plt.ylim([ymin,ymax])
    #plt.xlim([-.01,.01])
    plt.axis('off')
    plt.subplots_adjust(wspace=0)
    #makeNicePlots(ax)
    
    if savefigs:#% Save the figure
        d = os.path.join(svmdir+dnow,mousename+dp+'/bestC', dgc, dta)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
        fign = os.path.join(d, suffn[0:5]+'classAccur'+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight')



#%%  This is the only figure that will be affected by trShflAve (ie you have both trShfl averaged and not averaged variables for it)
######################################  All days pooled ###################################### 
### 1st row: pool all days and all neuron shuffles to get the histograms. Next to histograms plot the mean and st error of all shuffles of all days
### On the 2nd row plot ave and st error across shuffles for each day. Next to it plot the ave and st error across days (averages of shuffles)

lab1 = 'exc'
lab2 = 'inh'
colors = ['k','r']


###############%% percent non-zero w ###############    
### hist and P val for all days pooled (exc vs inh)
lab = '% non-0 w'
binEvery = 2#5# .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
a = np.reshape(perActiveExc,(-1,))# perActiveExc includes average across neuron shuffles for each day.... 
b = np.reshape(perActiveInh,(-1,))  
a = a[~(np.isnan(a) + np.isinf(a))]
b = b[~(np.isnan(b) + np.isinf(b))]
print [len(a), len(b)]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
'''
print 'exc vs inh (bestc):p value= %.3f' %(p)
print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
'''
plt.figure(figsize=(5,5))    
gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[0,0:2]
h2 = gs[0,2:3]
histerrbar(a,b,binEvery,p,colors)
#plt.xlabel(lab)

### show ave and std across shuffles for each day
# if averaged across trial shuffles
a = perActiveExc # numDays x numSamples x numTrialShuff
b = perActiveInh
# if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
if trShflAve==0:
    a = np.reshape(perActiveExc, (len(days), numCVtrShfls*numSamples), order = 'F')
    b = np.reshape(perActiveInh, (len(days), numCVtrShfls*numSamples), order = 'F')
_,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
print p
#plt.figure(); plot.subplot(221)
#ax = plt.subplot(gs[1,0:3])
errbarAllDays(a,b,p)
#plt.ylabel(lab)


#plt.savefig('cpath_all_absnon0w.svg', format='svg', dpi=300)
if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnow,mousename+dp+'/bestC', dgc, dta)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
        
    fign = os.path.join(d, suffn[0:5]+'excVSinh_allDays_percNon0W'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')



###############%% abs non-0 w ###############
### hist and P val for all days pooled (exc vs inh)
lab = 'abs non-0 w'
binEvery = .005# .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
a = np.reshape(wNon0AbsExc,(-1,)) 
b = np.reshape(wNon0AbsInh,(-1,))

a = a[~(np.isnan(a) + np.isinf(a))]
b = b[~(np.isnan(b) + np.isinf(b))]
print [len(a), len(b)]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
'''
print 'exc vs inh (bestc):p value= %.3f' %(p)
print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
'''
plt.figure(figsize=(5,5))    
gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[0,0:2]
h2 = gs[0,2:3]
ax1,_ = histerrbar(a,b,binEvery,p,colors)
#plt.xlabel(lab)
#ax1.set_xlim([0, .06])
#ax1.set_xlim([0, .1])
ax1.set_xlim([0, .5])

### show individual days
a = wNon0AbsExc
b = wNon0AbsInh
# if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
if trShflAve==0:
    a = np.reshape(wNon0AbsExc, (len(days), numCVtrShfls*numSamples), order = 'F')
    b = np.reshape(wNon0AbsInh, (len(days), numCVtrShfls*numSamples), order = 'F')
_,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
print p
#ax = plt.subplot(gs[1,0:3])
errbarAllDays(a,b,p)
#plt.ylabel(lab)


if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnow,mousename+dp+'/bestC', dgc, dta)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
 
    fign = os.path.join(d, suffn[0:5]+'excVSinh_allDays_absNon0W'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')
    
    
    
###############%% abs w  ###############
### hist and P val for all days pooled (exc vs inh)
lab = 'abs w'
binEvery = .002 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
a = np.reshape(wAbsExc,(-1,)) 
b = np.reshape(wAbsInh,(-1,))

a = a[~(np.isnan(a) + np.isinf(a))]
b = b[~(np.isnan(b) + np.isinf(b))]
print [len(a), len(b)]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
'''
print 'exc vs inh (bestc):p value= %.3f' %(p)
print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
'''
plt.figure(figsize=(5,5))    
gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[0,0:2]
h2 = gs[0,2:3]
ax1,_ = histerrbar(a,b,binEvery,p,colors)
#plt.xlabel(lab)
ax1.set_xlim([0, .15])

### show individual days
a = wAbsExc
b = wAbsInh
# if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
if trShflAve==0:
    a = np.reshape(wAbsExc, (len(days), numCVtrShfls*numSamples), order = 'F')
    b = np.reshape(wAbsInh, (len(days), numCVtrShfls*numSamples), order = 'F')
_,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
print p
#ax = plt.subplot(gs[1,0:3])
errbarAllDays(a,b,p)
#plt.ylabel(lab)


if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnow,mousename+dp+'/bestC', dgc, dta)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
 
    fign = os.path.join(d, suffn[0:5]+'excVSinh_allDays_absW'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')




    
###############%% w  ###############
### hist and P val for all days pooled (exc vs inh)
lab = 'w'
binEvery = .001 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
a = np.reshape(wExc,(-1,)) 
b = np.reshape(wInh,(-1,))

a = a[~(np.isnan(a) + np.isinf(a))]
b = b[~(np.isnan(b) + np.isinf(b))]
print [len(a), len(b)]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
'''
print 'exc vs inh (bestc):p value= %.3f' %(p)
print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
'''
plt.figure(figsize=(5,5))    
gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[0,0:2]
h2 = gs[0,2:3]
ax1,_ = histerrbar(a,b,binEvery,p,colors)
#plt.xlabel(lab)
ax1.set_xlim([-.05, .03])

### show individual days
a = wExc
b = wInh
# if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
if trShflAve==0:
    a = np.reshape(wExc, (len(days), numCVtrShfls*numSamples), order = 'F')
    b = np.reshape(wInh, (len(days), numCVtrShfls*numSamples), order = 'F')
_,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
print p
#ax = plt.subplot(gs[1,0:3])
errbarAllDays(a,b,p)
#plt.ylabel(lab)


if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnow,mousename+dp+'/bestC', dgc, dta)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
 
    fign = os.path.join(d, suffn[0:5]+'excVSinh_allDays_w'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')
    
    


#%%%% For all the analyses below you need to run the codes above with trShflAve = 1.
    
#%%
###################################### Plot each day separately ######################################
for iday in range(len(days)):
    
    print '\n_________________'+days[iday]+'_________________'    
    
    ###################
    lab = '% non-0 w'
#    binEvery = 10# .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
    a = np.reshape(perActiveExc[iday,:],(-1,)) 
    b = np.reshape(perActiveInh[iday,:],(-1,))
    
    a = a[~(np.isnan(a) + np.isinf(a))]
    b = b[~(np.isnan(b) + np.isinf(b))]
    r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
    binEvery = r/float(10)    

    _, p = stats.ttest_ind(a, b, nan_policy='omit')
    '''
    print 'exc vs inh (bestc):p value= %.3f' %(p)
    print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
    '''
    plt.figure(figsize=(6,8.5))    
    gs = gridspec.GridSpec(3, 3)#, width_ratios=[2, 1]) 
    h1 = gs[0,0:2]
    h2 = gs[0,2:3]
    histerrbar(a,b,binEvery,p,colors)    
    plt.title('cvClassEr= %.2f' %(perClassErTest_bestc[iday]))

    ###################
    lab = 'abs non-0 w'
    a = np.reshape(wNon0AbsExc[iday,:],(-1,)) 
    b = np.reshape(wNon0AbsInh[iday,:],(-1,))
    
    a = a[~(np.isnan(a) + np.isinf(a))]
    b = b[~(np.isnan(b) + np.isinf(b))]
    r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
    binEvery = r/float(10)    

    _, p = stats.ttest_ind(a, b, nan_policy='omit')
    '''
    print 'exc vs inh (bestc):p value= %.3f' %(p)
    print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
    '''
    h1 = gs[1,0:2]
    h2 = gs[1,2:3]
    if np.logical_and(b.shape[0], a.shape[0])!=0:
        histerrbar(a,b,binEvery,p,colors)


    ###################
    lab = 'abs w'
    a = np.reshape(wAbsExc[iday,:],(-1,)) 
    b = np.reshape(wAbsInh[iday,:],(-1,))
    
    a = a[~(np.isnan(a) + np.isinf(a))]
    b = b[~(np.isnan(b) + np.isinf(b))]
    r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
    binEvery = r/float(10)    

    _, p = stats.ttest_ind(a, b, nan_policy='omit')
    '''
    print 'exc vs inh (bestc):p value= %.3f' %(p)
    print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
    '''
    h1 = gs[2,0:2]
    h2 = gs[2,2:3]
    histerrbar(a,b,binEvery,p,colors) 
    
    
    if savefigs:#% Save the figure
        d = os.path.join(svmdir+dnow,mousename+dp+'/bestC', dgc, dta)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
     
        fign = os.path.join(d, suffn[0:5]+'excVSinh_'+days[iday][0:6]+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight')


    plt.show()
    



#%% Run the following script to get vars related to exc - inh
execfile("svm_excInh_cPath_excMinusInh.py")


#%% exc minus inh at best c (average of shuffles) and null dist (subtraction of rand shuffles) at bestc... this is already done in computing b0100nan vars: you should perhaps exclude shuffles with all-0 weights bc they may mask the effects. (?)

perActiveEmI_bestc = np.full((len(days)), np.nan)
r_perActive_bestc = np.full((len(days),2*l), np.nan)
wabsEmI_bestc = np.full((len(days)), np.nan)
r_wabs_bestc = np.full((len(days),2*l), np.nan)
wabsEmI_non0_bestc = np.full((len(days)), np.nan)
r_non0_wabs_bestc = np.full((len(days),2*l), np.nan)
wEmI_bestc = np.full((len(days)), np.nan)
r_w_bestc = np.full((len(days),2*l), np.nan)
wEmI_non0_bestc = np.full((len(days)), np.nan)
r_non0_w_bestc = np.full((len(days),2*l), np.nan)

p_perActive = np.full((len(days)), np.nan)
p_wAbsNon0 = np.full((len(days)), np.nan)

for iday in days2ana: #range(len(days)):
    
    ibc = cvect_all[iday,:]==cbestAll[iday] # index of cbest
#    print np.argwhere(ibc)
    
    perActiveEmI_bestc[iday] = perActiveEmI_b0100nan[iday,ibc] # perActiveAll_exc_aveS[iday, ibc] - perActiveAll_inh_aveS[iday, ibc]
    r_perActive_bestc[iday,:] = randAll_perActiveAll_ei_aveS_b0100nan[iday,:,ibc]
    
    wEmI_bestc[iday] = wEmI_b0100nan[iday,ibc]
    r_w_bestc[iday,:] = randAll_w_ei_aveS_b0100nan[iday,:,ibc]
    
    wEmI_non0_bestc[iday] = wEmI_non0_b0100nan[iday,ibc]
    r_non0_w_bestc[iday,:] = randAll_non0_w_ei_aveS_b0100nan[iday,:,ibc]
    
    wabsEmI_bestc[iday] = wabsEmI_b0100nan[iday,ibc] 
    r_wabs_bestc[iday,:] = randAll_wabs_ei_aveS_b0100nan[iday,:,ibc]   
    
    wabsEmI_non0_bestc[iday] = wabsEmI_non0_b0100nan[iday,ibc] 
    r_non0_wabs_bestc[iday,:] = randAll_non0_wabs_ei_aveS_b0100nan[iday,:,ibc]    
#    _,p = stats.ttest_ind(perActiveAll_exc[iday,:,ibc].flatten(), perActiveAll_inh[iday,:,ibc].flatten()) #, nan_policy='omit')
#plt.hist(r_perActive_bestc[iday,:])
    _,p = stats.ttest_1samp(r_perActive_bestc[iday,:], perActiveEmI_bestc[iday], nan_policy='omit') 
    print p, '---', perActiveEmI_bestc[iday]
    p_perActive[iday] = p
    _,p = stats.ttest_1samp(r_non0_wabs_bestc[iday,:], wabsEmI_non0_bestc[iday], nan_policy='omit') 
    print p, '---', wabsEmI_non0_bestc[iday]
    p_wAbsNon0[iday] = p
    print '____'

'''
perActiveEmI_bestc[p_perActive<=.05]
wabsEmI_non0_bestc[p_wAbsNon0<=.05]


plt.plot(p_perActive); plt.plot(perActiveEmI_bestc)
plt.plot(p_wAbsNon0); plt.plot(wabsEmI_non0_bestc)

np.mean(p_perActive<=.05)
np.mean(p_wAbsNon0<=.05)
'''

########################################### PLOTS ###########################################
lab1 = 'exc - inh'
lab2 = 'rand'
bindiv = 10 #10

###############%% percent non-zero w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(perActiveEmI_bestc,(-1,)) # pool across days x length(cvect_)  
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(r_perActive_bestc,(-1,)) # pool across days x numSamples x length(cvect_)
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
# against 0
_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)


### hist and P val for all days pooled (exc vs inh)
lab = '% non-0 w'
#binEvery = 3 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
binEvery = r/float(bindiv)
plt.figure(figsize=(5,15))    
gs = gridspec.GridSpec(5, 3)#, width_ratios=[2, 1]) 
h1 = gs[0,0:2]
h2 = gs[0,2:3]
histerrbar(a,b,binEvery,p)
#plt.xlabel(lab)


###############%% abs non-0 w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(wabsEmI_non0_bestc,(-1,)) 
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(r_non0_wabs_bestc,(-1,))
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
# against 0
_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)


lab = 'abs non-0 w'
#binEvery = .006# .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
binEvery = r/float(bindiv)
#plt.figure(figsize=(5,5))    
#gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[1,0:2]
h2 = gs[1,2:3]
histerrbar(a,b,binEvery,p)
#plt.xlabel(lab)


###############%% abs w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(wabsEmI_bestc,(-1,)) 
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(r_wabs_bestc,(-1,))
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
# against 0
_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)


lab = 'abs w'
#binEvery = .002# .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
binEvery = r/float(bindiv)
#plt.figure(figsize=(5,5))    
#gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[2,0:2]
h2 = gs[2,2:3]
histerrbar(a,b,binEvery,p)
#plt.xlabel(lab)


###############%% w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(wEmI_bestc,(-1,)) 
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(r_w_bestc,(-1,))
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
# against 0
_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)


lab = 'w'
#binEvery = .003 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
binEvery = r/float(bindiv)
#plt.figure(figsize=(5,5))    
#gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[3,0:2]
h2 = gs[3,2:3]
histerrbar(a,b,binEvery,p)
#plt.xlabel(lab)


###############%% non0 w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(wEmI_non0_bestc,(-1,)) 
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(r_non0_w_bestc,(-1,))
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
# against 0
_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)


lab = 'non-0 w'
#binEvery = .006# .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
binEvery = r/float(bindiv)
#plt.figure(figsize=(5,5))    
#gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[4,0:2]
h2 = gs[4,2:3]
histerrbar(a,b,binEvery,p)
#plt.xlabel(lab)


plt.subplots_adjust(hspace=.8)

#plt.savefig('cpath_all_absnon0w.svg', format='svg', dpi=300)
if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnow,mousename+dp+'/bestC', dgc, dta)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
 
    fign = os.path.join(d, suffn[0:5]+'excMinusInh_allDays'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')








########################################################################################################################################################################
# ALL C

#%%
################################################################################################################################################################################################
########################## Pool all days and all c values, make histograms of exc - inh, and compare with null dists  ################################################
################################################################################################################################################################################################

###############%% percent non-zero w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(perActiveEmI_b0100nan,(-1,)) # pool across days x length(cvect_)  
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(randAll_perActiveAll_ei_aveS_b0100nan,(-1,)) # pool across days x numSamples x length(cvect_)
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
#print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
#print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
## against 0
#_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
#print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)



binEvery = .5
#mv = 2
#bn = np.arange(0.05,mv,binEvery)
bn = np.arange(np.min(np.concatenate((a,b))), np.max(np.concatenate((a,b))), binEvery)
#bn = np.arange(-4,4, binEvery)
bn[-1] = np.max([np.max(a),np.max(b)]) # unlike digitize, histogram doesn't count the right most value
#bn[0] = np.min([np.min(a),np.min(b)])

hist, bin_edges = np.histogram(a, bins=bn)
hist = hist/float(np.sum(hist))
#d = stats.mode(np.diff(bin_edges))[0]/float(2)

plt.figure(figsize=(3,20))    
plt.subplot(511)
plt.bar(bin_edges[0:-1], hist, binEvery, color='g', alpha=.5, label='exc-inh')

hist, bin_edges = np.histogram(b, bins=bn)
hist = hist/float(np.sum(hist))
#d = stats.mode(np.diff(bin_edges))[0]/float(2)
plt.bar(bin_edges[0:-1], hist, binEvery, color='k', alpha=.3, label='exc-exc, inh-inh')
plt.legend()
#plt.ylabel('Prob (data: all days, all c pooled; null: all shuff, all days, all c pooled)')
plt.title('mean = %.3f, p=%.3f' %(np.mean(a), p))
plt.xlim([-20,20])
plt.xlabel('%non-zero w')
#plt.savefig('cpath_all_percnon0.svg', format='svg', dpi=300)


###############%% w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(wEmI_b0100nan,(-1,)) 
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(randAll_w_ei_aveS_b0100nan,(-1,))
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
#print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
#print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
## against 0
#_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
#print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)


binEvery = .01
mv = 2
bn = np.arange(0.05,mv,binEvery)
bn = np.arange(np.min(np.concatenate((a,b))), np.max(np.concatenate((a,b))), binEvery)
bn[-1] = np.max([np.max(a),np.max(b)]) # unlike digitize, histogram doesn't count the right most value

hist, bin_edges = np.histogram(a, bins=bn)
hist = hist/float(np.sum(hist))

#d = stats.mode(np.diff(bin_edges))[0]/float(2)
#plt.figure()    
plt.subplot(512)
plt.bar(bin_edges[0:-1], hist, binEvery, color='g', alpha=.5, label='exc-inh')

hist, bin_edges = np.histogram(b, bins=bn)
hist = hist/float(np.sum(hist))
#d = stats.mode(np.diff(bin_edges))[0]/float(2)
plt.bar(bin_edges[0:-1], hist, binEvery, color='k', alpha=.3, label='exc-exc, inh-inh')
plt.legend()
#plt.ylabel('Prob (data: all days, all c pooled; null: all shuff, all days, all c pooled)')
plt.title('mean = %.3f, p=%.3f' %(np.mean(a), p))
plt.xlim([-.5,.5])
plt.xlabel('w')
#plt.savefig('cpath_all_absw.svg', format='svg', dpi=300)


###############%% non0 w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(wEmI_non0_b0100nan,(-1,)) 
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(randAll_non0_w_ei_aveS_b0100nan,(-1,))
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
#print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
#print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
## against 0
#_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
#print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)


binEvery = .01
mv = 2
bn = np.arange(0.05,mv,binEvery)
bn = np.arange(np.min(np.concatenate((a,b))), np.max(np.concatenate((a,b))), binEvery)
bn[-1] = np.max([np.max(a),np.max(b)]) # unlike digitize, histogram doesn't count the right most value

hist, bin_edges = np.histogram(a, bins=bn)
hist = hist/float(np.sum(hist))

#d = stats.mode(np.diff(bin_edges))[0]/float(2)
#plt.figure()   
plt.subplot(513) 
plt.bar(bin_edges[0:-1], hist, binEvery, color='g', alpha=.5, label='exc-inh')

hist, bin_edges = np.histogram(b, bins=bn)
hist = hist/float(np.sum(hist))
#d = stats.mode(np.diff(bin_edges))[0]/float(2)
plt.bar(bin_edges[0:-1], hist, binEvery, color='k', alpha=.3, label='exc-exc, inh-inh')
plt.legend()
plt.ylabel('Prob (data: all days, all c pooled; null: all shuff, all days, all c pooled)')
plt.title('mean = %.3f, p=%.3f' %(np.mean(a), p))
plt.xlim([-.5,.5])
plt.xlabel('non-0 w')
#plt.savefig('cpath_all_absw.svg', format='svg', dpi=300)



###############%% abs w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(wabsEmI_b0100nan,(-1,)) 
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(randAll_wabs_ei_aveS_b0100nan,(-1,))
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
#print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
#print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
## against 0
#_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
#print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)


binEvery = .01
mv = 2
bn = np.arange(0.05,mv,binEvery)
bn = np.arange(np.min(np.concatenate((a,b))), np.max(np.concatenate((a,b))), binEvery)
bn[-1] = np.max([np.max(a),np.max(b)]) # unlike digitize, histogram doesn't count the right most value

hist, bin_edges = np.histogram(a, bins=bn)
hist = hist/float(np.sum(hist))

#d = stats.mode(np.diff(bin_edges))[0]/float(2)
#plt.figure()  
plt.subplot(514)  
plt.bar(bin_edges[0:-1], hist, binEvery, color='g', alpha=.5, label='exc-inh')

hist, bin_edges = np.histogram(b, bins=bn)
hist = hist/float(np.sum(hist))
#d = stats.mode(np.diff(bin_edges))[0]/float(2)
plt.bar(bin_edges[0:-1], hist, binEvery, color='k', alpha=.3, label='exc-exc, inh-inh')
plt.legend()
#plt.ylabel('Prob (data: all days, all c pooled; null: all shuff, all days, all c pooled)')
plt.title('mean = %.3f, p=%.3f' %(np.mean(a), p))
plt.xlim([-.5,.5])
plt.xlabel('abs w')
#plt.savefig('cpath_all_absw.svg', format='svg', dpi=300)


###############%% abs non-0 w: hist and P val for all days pooled (exc - inh)  ###############
a = np.reshape(wabsEmI_non0_b0100nan,(-1,)) 
a = a[~(np.isnan(a) + np.isinf(a))]
# against random null dist
b = np.reshape(randAll_non0_wabs_ei_aveS_b0100nan,(-1,))
b = b[~(np.isnan(b) + np.isinf(b))]
_, p = stats.ttest_ind(a, b, nan_policy='omit')
#print 'rand null dist: p value:diff c path (all c) = %.3f' %(p)
#print '\tmean: data: %.3f; null: %.3f' %(np.mean(a), np.mean(b))
## against 0
#_, pdiff = stats.ttest_1samp(a, 0, nan_policy='omit')
#print 'p value vs 0:diff c path (all c) = %.3f' %(pdiff)


binEvery = .01
mv = 2
bn = np.arange(0.05,mv,binEvery)
bn = np.arange(np.min(np.concatenate((a,b))), np.max(np.concatenate((a,b))), binEvery)
bn[-1] = np.max([np.max(a),np.max(b)]) # unlike digitize, histogram doesn't count the right most value

hist, bin_edges = np.histogram(a, bins=bn)
hist = hist/float(np.sum(hist))

#d = stats.mode(np.diff(bin_edges))[0]/float(2)
#plt.figure()    
plt.subplot(515)
plt.bar(bin_edges[0:-1], hist, binEvery, color='g', alpha=.5, label='exc-inh')

hist, bin_edges = np.histogram(b, bins=bn)
hist = hist/float(np.sum(hist))
#d = stats.mode(np.diff(bin_edges))[0]/float(2)
plt.bar(bin_edges[0:-1], hist, binEvery, color='k', alpha=.3, label='exc-exc, inh-inh')
plt.legend(loc=0)
#plt.ylabel('Prob (data: all days, all c pooled; null: all shuff, all days, all c pooled)')
plt.title('mean = %.3f, p=%.3f' %(np.mean(a), p))
plt.xlim([-.5,.5])
plt.xlabel('abs non-0 w')
#plt.savefig('cpath_all_absnon0w.svg', format='svg', dpi=300)

plt.subplots_adjust(hspace=.3)


if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnow,mousename+dp+'/allC', dgc, dta)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
 
    fign = os.path.join(d, suffn[0:5]+'excMinusInh_AllDays'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')



#%%
############################# Plot c-path of exc and inh for each day one by one #################################
####################################################################################
##%% Look at days one by one (averages across shuffles)

for iday in range(len(days)):
    
    print '____________________', days[iday], '____________________'
#    plt.figure()    
#    plt.errorbar(cvect_all[iday,:], perClassErTestAll_aveS[iday,:], perClassErTestAll_stdS[iday,:], color = 'g')
#    plt.xscale('log')
    
    plt.figure(figsize=(12,2))
    '''    
    plt.subplot(121)
    plt.errorbar(perClassErAll_aveS[iday,:], wall_exc_aveNS[iday,:], wall_exc_stdNS[iday,:], color = 'b', label = 'excit')
    plt.errorbar(perClassErAll_aveS[iday,:], wall_inh_aveNS[iday,:], wall_inh_stdNS[iday,:], color = 'r', label = 'inhibit')
    plt.xscale('log')    
    plt.xlabel('Training error %')
    plt.ylabel('Mean+std of weights')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1,1))
    '''
    ax = plt.subplot(151)
    plt.errorbar(cvect_all[iday,:], perActiveAll_exc_aveS[iday,:], perActiveAll_exc_stdS[iday,:], color = 'b', label = 'excit')
    plt.errorbar(cvect_all[iday,:], perActiveAll_inh_aveS[iday,:], perActiveAll_inh_stdS[iday,:], color = 'r', label = 'inhibit')
    plt.errorbar(cvect_all[iday,:], perClassErAll_aveS[iday,:], perClassErAll_stdS[iday,:], color = 'k')
    plt.errorbar(cvect_all[iday,:], perClassErTestAll_aveS[iday,:], perClassErTestAll_stdS[iday,:], color = 'g')
    plt.xscale('log')
#    plt.xlabel('c (inverse of regularization parameter)')
#    plt.ylabel('% non-0 weights')   
    aa = perActiveEmI_b0100nan[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
    bb = randAll_perActiveAll_ei_aveS_b0100nan[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]    
    _, p_ei = stats.ttest_ind(aa, bb)
    plt.title('%%non-0\np=%.3f;av=%.3f' %(p_ei, np.mean(aa)))
    ymin, ymax = ax.get_ylim()
    plt.plot([cbestAll[iday],cbestAll[iday]],[ymin,ymax])
    
    
    ax = plt.subplot(152)
    plt.errorbar(cvect_all[iday,:], wall_exc_aveNS[iday,:], wall_exc_stdNS[iday,:], color = 'b', label = 'excit')
    plt.errorbar(cvect_all[iday,:], wall_inh_aveNS[iday,:], wall_inh_stdNS[iday,:], color = 'r', label = 'inhibit')
    plt.errorbar(cvect_all[iday,:], perClassErAll_aveS[iday,:]/100, perClassErAll_stdS[iday,:]/100, color = 'k')
    plt.errorbar(cvect_all[iday,:], perClassErTestAll_aveS[iday,:]/100, perClassErTestAll_stdS[iday,:]/100, color = 'g')    
    plt.xscale('log')
#    plt.xlabel('c (inverse of regularization parameter)')
#    plt.ylabel('Weights')
    aa = wEmI_b0100nan[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
    bb = randAll_w_ei_aveS_b0100nan[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb)    
    plt.title('W\np=%.3f;av=%.3f' %(p_ei, np.mean(aa)))
    ymin, ymax = ax.get_ylim()
    plt.plot([cbestAll[iday],cbestAll[iday]],[ymin,ymax])
    

    ax = plt.subplot(153)
    plt.errorbar(cvect_all[iday,:], wall_non0_exc_aveNS[iday,:], wall_non0_exc_stdNS[iday,:], color = 'b', label = 'excit')
    plt.errorbar(cvect_all[iday,:], wall_non0_inh_aveNS[iday,:], wall_non0_inh_stdNS[iday,:], color = 'r', label = 'inhibit')
    plt.errorbar(cvect_all[iday,:], perClassErAll_aveS[iday,:]/100, perClassErAll_stdS[iday,:]/100, color = 'k')
    plt.errorbar(cvect_all[iday,:], perClassErTestAll_aveS[iday,:]/100, perClassErTestAll_stdS[iday,:]/100, color = 'g')    
    plt.xscale('log')
    plt.xlabel('c (inverse of regularization parameter)')
#    plt.ylabel('Non-0 weights')
    aa = wEmI_non0_b0100nan[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
    bb = randAll_non0_w_ei_aveS_b0100nan[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb)    
    plt.title('Non0 w\np=%.3f;av=%.3f' %(p_ei, np.mean(aa)))
    ymin, ymax = ax.get_ylim()
    plt.plot([cbestAll[iday],cbestAll[iday]],[ymin,ymax])
    
    
    ax = plt.subplot(154)
    plt.errorbar(cvect_all[iday,:], wall_abs_exc_aveNS[iday,:], wall_abs_exc_stdNS[iday,:], color = 'b', label = 'excit')
    plt.errorbar(cvect_all[iday,:], wall_abs_inh_aveNS[iday,:], wall_abs_inh_stdNS[iday,:], color = 'r', label = 'inhibit')
    plt.errorbar(cvect_all[iday,:], perClassErAll_aveS[iday,:]/100, perClassErAll_stdS[iday,:]/100, color = 'k')
    plt.errorbar(cvect_all[iday,:], perClassErTestAll_aveS[iday,:]/100, perClassErTestAll_stdS[iday,:]/100, color = 'g')    
    plt.xscale('log')
#    plt.xlabel('c (inverse of regularization parameter)')
#    plt.ylabel('Abs weights')
    aa = wabsEmI_b0100nan[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
    bb = randAll_wabs_ei_aveS_b0100nan[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb)
    plt.title('Abs w\np=%.3f;av=%.3f' %(p_ei, np.mean(aa)))
    ymin, ymax = ax.get_ylim()
    plt.plot([cbestAll[iday],cbestAll[iday]],[ymin,ymax])
    

    ax = plt.subplot(155)
    plt.errorbar(cvect_all[iday,:], wall_non0_abs_exc_aveNS[iday,:], wall_non0_abs_exc_stdNS[iday,:], color = 'b', label = 'excit')
    plt.errorbar(cvect_all[iday,:], wall_non0_abs_inh_aveNS[iday,:], wall_non0_abs_inh_stdNS[iday,:], color = 'r', label = 'inhibit')
    plt.errorbar(cvect_all[iday,:], perClassErAll_aveS[iday,:]/100, perClassErAll_stdS[iday,:]/100, color = 'k')
    plt.errorbar(cvect_all[iday,:], perClassErTestAll_aveS[iday,:]/100, perClassErTestAll_stdS[iday,:]/100, color = 'g')    
    plt.xscale('log')
#    plt.xlabel('c (inverse of regularization parameter)')
#    plt.ylabel('Abs non-0 weights')   
    aa = wabsEmI_non0_b0100nan[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
    bb = randAll_non0_wabs_ei_aveS_b0100nan[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb)
    plt.title('Abs non0 w\np=%.3f;av=%.3f' %(p_ei, np.mean(aa)))
    ymin, ymax = ax.get_ylim()
    plt.plot([cbestAll[iday],cbestAll[iday]],[ymin,ymax])    
    
    plt.subplots_adjust(hspace=.5, wspace=.35)      
#    raw_input()
#    plt.savefig('cpath'+days[iday]+'.svg', format='svg', dpi=300)    
      
    if savefigs:#% Save the figure
        d = os.path.join(svmdir+dnow,mousename+dp+'/allC', dgc, dta)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
    
        fign = os.path.join(d, suffn[0:5]+'excInh_'+days[iday][0:6]+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight')

   
    plt.show()       
   
    ######################################################
    ###### p values: compare dists w each other ######
    # percent non-zero
    aa = perActiveAll_exc_aveS[iday,:]+0
    bb = perActiveAll_inh_aveS[iday,:]+0
    aa = aa[~np.isnan(aa)] # + np.isinf(a))]
    bb = bb[~np.isnan(bb)]
    h, p_two = stats.ttest_ind(aa, bb)
    p_tl = ttest2(aa, bb, tail='left')
    p_tr = ttest2(aa, bb, tail='right')
    print '\n%%non-zero p value (pooled for all values of c):\nexc ~= inh : %.2f\texc < inh : %.2f\texc > inh : %.2f' %(p_two, p_tl, p_tr)   
    '''
    # plot hists
    hist, bin_edges = np.histogram(aa, bins=30)
    hist = hist/float(np.sum(hist))
    fig = plt.figure(figsize=(4,2))
    plt.bar(bin_edges[0:-1], hist, .7, color='b')
    
    hist, bin_edges = np.histogram(bb, bins=30)
    hist = hist/float(np.sum(hist))
    plt.bar(bin_edges[0:-1], hist, .7, color='r')
    '''

    # weight
    aa = wall_exc_aveNS[iday,:]+0
    bb = wall_inh_aveNS[iday,:]+0
    aa = aa[~np.isnan(aa)] # + np.isinf(a))]
    bb = bb[~np.isnan(bb)]
    h, p_two = stats.ttest_ind(aa, bb)
    p_tl = ttest2(aa, bb, tail='left')
    p_tr = ttest2(aa, bb, tail='right')
    print '\nw: p value (pooled for all values of c):\nexc ~= inh : %.2f\texc < inh : %.2f\texc > inh : %.2f' %(p_two, p_tl, p_tr)


    # non-0 weight
    aa = wall_non0_exc_aveNS[iday,:]+0
    bb = wall_non0_inh_aveNS[iday,:]+0
    aa = aa[~np.isnan(aa)] # + np.isinf(a))]
    bb = bb[~np.isnan(bb)]
    h, p_two = stats.ttest_ind(aa, bb)
    p_tl = ttest2(aa, bb, tail='left')
    p_tr = ttest2(aa, bb, tail='right')
    print '\nnon0 w: p value (pooled for all values of c):\nexc ~= inh : %.2f\texc < inh : %.2f\texc > inh : %.2f' %(p_two, p_tl, p_tr)

    
    # abs weight
    aa = wall_abs_exc_aveNS[iday,:]+0
    bb = wall_abs_inh_aveNS[iday,:]+0
    aa = aa[~np.isnan(aa)] # + np.isinf(a))]
    bb = bb[~np.isnan(bb)]
    h, p_two = stats.ttest_ind(aa, bb)
    p_tl = ttest2(aa, bb, tail='left')
    p_tr = ttest2(aa, bb, tail='right')
    print '\nabs w: p value (pooled for all values of c):\nexc ~= inh : %.2f\texc < inh : %.2f\texc > inh : %.2f' %(p_two, p_tl, p_tr)   


    # abs non0 weight
    aa = wall_non0_abs_exc_aveNS[iday,:]+0
    bb = wall_non0_abs_inh_aveNS[iday,:]+0
    aa = aa[~np.isnan(aa)] # + np.isinf(a))]
    bb = bb[~np.isnan(bb)]
    h, p_two = stats.ttest_ind(aa, bb)
    p_tl = ttest2(aa, bb, tail='left')
    p_tr = ttest2(aa, bb, tail='right')
    print '\nabs non0 w: p value (pooled for all values of c):\nexc ~= inh : %.2f\texc < inh : %.2f\texc > inh : %.2f' %(p_two, p_tl, p_tr)   
    
    
    '''
    # plot hists
    hist, bin_edges = np.histogram(aa, bins=30)
    hist = hist/float(np.sum(hist))
    fig = plt.figure(figsize=(4,2))
    plt.bar(bin_edges[0:-1], hist, .01, color='b')
    
    hist, bin_edges = np.histogram(bb, bins=30)
    hist = hist/float(np.sum(hist))
    plt.bar(bin_edges[0:-1], hist, .01, color='r')
    '''

 

    '''
    ########################################################################
    ###### p values of difference of dists (diff (exc - inh)) relative to null dists you created above
    # percent non-0 w 
    aa = perActiveEmI[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    bb = randAll_perActiveAll_ei_aveS[iday,:,:].flatten()
    _, p_ei = stats.ttest_ind(aa, bb)
    print '\n%%non-0: rand null dist p val = %.3f' %(p_ei)
    _, pdiff = stats.ttest_1samp(aa, 0)
    print '\t1 samp p val = %.3f' %(pdiff)
    print '\tmean of diff of c path= %.2f; null= %.2f' %(np.mean(aa), np.mean(bb))
    

    # w
    aa = wEmI[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
    bb = randAll_w_ei_aveS[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb)
    print '\nw: rand null dist p val = %.3f' %(p_ei)
    _, pdiff = stats.ttest_1samp(aa, 0)
    print '\t1 samp p val = %.3f' %(pdiff)
    print '\tmean of diff of c path= %.2f; null= %.2f' %(np.mean(aa), np.mean(bb))


    # non-0 w
    aa = wEmI_non0[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
    bb = randAll_non0_w_ei_aveS[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb)
    print '\nnon0 w: rand null dist p val = %.3f' %(p_ei)
    _, pdiff = stats.ttest_1samp(aa, 0)
    print '\t1 samp p val = %.3f' %(pdiff)
    print '\tmean of diff of c path= %.2f; null= %.2f' %(np.mean(aa), np.mean(bb))
    

    # abs w
    aa = wabsEmI[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
    bb = randAll_wabs_ei_aveS[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb)
    print '\nabs w: rand null dist p val = %.3f' %(p_ei)
    _, pdiff = stats.ttest_1samp(aa, 0)
    print '\t1 samp p val = %.3f' %(pdiff)
    print '\tmean of diff of c path= %.2f; null= %.2f' %(np.mean(aa), np.mean(bb))


    # abs non-0 w
    aa = wabsEmI_non0[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
    bb = randAll_non0_wabs_ei_aveS[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb)
    print '\nabs non-0 w: rand null dist p val = %.3f' %(p_ei)
    _, pdiff = stats.ttest_1samp(aa, 0)
    print '\t1 samp p val = %.3f' %(pdiff)
    print '\tmean of diff of c path= %.2f; null= %.2f' %(np.mean(aa), np.mean(bb))
    '''


#%% For each day compare p value for exc - inh vs 0 with p value for exc - inh vs random exc-exc (inh-inh)    
"""
for iday in range(len(days)):    
    
    print '____________________', days[iday], '____________________'
    
    aa = perActiveEmI[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
#    _, pdiff = stats.ttest_1samp(aa, 1, nan_policy='omit')
    _, pdiff = stats.ttest_1samp(aa, 0, nan_policy='omit')
    
    bb = randAll_perActiveAll_ei_aveS[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb, nan_policy='omit')
    print '%.3f, %.3f' %(pdiff, p_ei)
    '''    
    bb = randAll_perActiveAll_exc_aveS[iday,:,:].flatten() # exc - exc for rand shuffles (not averaged across shuffles)    
    _, p_e = stats.ttest_ind(aa, bb)
    bb = randAll_perActiveAll_inh_aveS[iday,:,:].flatten()
    _, p_i = stats.ttest_ind(aa, bb)    
    print '%.3f, %.3f, %.3f, %.3f' %(pdiff, p_ei, p_e, p_i)
    '''
    

    aa = wabsEmI[iday,:].flatten() # length(cvect_) # exc minus inh for shuffle averaged
    aa = aa[~(np.isnan(aa) + np.isinf(aa))] # remove nan and inf values
#    _, pdiff = stats.ttest_1samp(aa, 1, nan_policy='omit')
    _, pdiff = stats.ttest_1samp(aa, 0, nan_policy='omit')
    
    bb = randAll_wabs_ei_aveS[iday,:,:].flatten()
    bb = bb[~(np.isnan(bb) + np.isinf(bb))]
    _, p_ei = stats.ttest_ind(aa, bb, nan_policy='omit')
    print '%.3f, %.3f' %(pdiff, p_ei)
    
#    print '_____'
"""


#%% Average across days ... not sure if this is valid bc the relatin of weights vs c (or training error) is not the same for different days and by averaging we will cancel out any difference between exc and inh populations.

def plotav(a,b):
    ave = np.mean(a, axis=0) # length(cvect_)
    sde = np.std(a, axis=0)
    avi = np.mean(b, axis=0)
    sdi = np.std(b, axis=0)
    
    plt.errorbar(cvect_all[0,:], ave, sde, fmt='b.-')
    plt.errorbar(cvect_all[0,:], avi, sdi, fmt='r.-')
    plt.xscale('log')
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    plt.plot([np.mean(cbestAll), np.mean(cbestAll)],[ymin,ymax])    
    
    
    
##%% Average across all days (perc class error for train and test)
aa = perClassErAll_aveS
aat = perClassErTestAll_aveS

avr = np.mean(aa, axis=0) # length(cvect_)
sdr = np.std(aa, axis=0)
avt = np.mean(aat, axis=0) # length(cvect_)
sdt = np.std(aat, axis=0)

plt.figure(figsize=(3,15))
ax = plt.subplot(611)    
plt.errorbar(cvect_all[0,:], avr, sdr, fmt='k.-', label='train')
plt.errorbar(cvect_all[0,:], avt, sdt, fmt='g.-', label='test')
plt.xscale('log')
plt.xlabel('c')
plt.ylabel('class error')
plt.legend(loc=0, frameon=False)
ymin, ymax = ax.get_ylim()
plt.plot([np.mean(cbestAll), np.mean(cbestAll)],[ymin,ymax])    

    
plt.subplot(612)    
a = perActiveAll_exc_aveS
b = perActiveAll_inh_aveS
plotav(a,b)
plt.xlabel('c')
plt.ylabel('% non0 w')
    
plt.subplot(613)        
a = wall_exc_aveNS # wall_non0_abs_exc_aveNS
b = wall_inh_aveNS
plotav(a,b)
plt.xlabel('c')
plt.ylabel('w')

plt.subplot(614)    
a = wall_non0_exc_aveNS
b = wall_non0_inh_aveNS
plotav(a,b)
plt.xlabel('c')
plt.ylabel('non0 w')

plt.subplot(615)    
a = wall_abs_exc_aveNS
b = wall_abs_inh_aveNS
plotav(a,b)
plt.xlabel('c')
plt.ylabel('abs w')

plt.subplot(616)    
a = wall_non0_abs_exc_aveNS
b = wall_non0_abs_inh_aveNS
plotav(a,b)
plt.xlabel('c')
plt.ylabel('abs non0 w')


if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnow,mousename+dp+'/allC', 'allDays', dta)
#    d = os.path.join(svmdir+dnow,mousename+dp+'/allC', dgc, dta)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)

    fign = os.path.join(d, suffn[0:5]+'excInh_aveAllDays'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')



#%% Plot average of fractnon0 and weights against class error (instad of c)

# is averaging right... remember u had to get diff, or perhaps do ratio...
# look at abs of weights and ave of non-zeros w


# u're going to run the code w the same cvect so u can easily plot all days against cvect... but u should be fine to plot it against classError
# work on this: plot ave and avi against classError

#np.min(perClassErAll_aveS)
step = 5
b = np.arange(0, np.ceil(np.nanmax(perClassErAll_aveS)).astype(int)+step, step)
b = np.arange(0, np.ceil(np.nanmax(perClassErTestAll_aveS)).astype(int)+step, step)

perActiveAll_exc_cerr = np.full((len(days), len(b)), np.nan)
perActiveAll_inh_cerr = np.full((len(days), len(b)), np.nan)
wall_abs_exc_cerr = np.full((len(days), len(b)), np.nan)
wall_abs_inh_cerr = np.full((len(days), len(b)), np.nan)
wall_non0_abs_exc_cerr = np.full((len(days), len(b)), np.nan)
wall_non0_abs_inh_cerr = np.full((len(days), len(b)), np.nan)
wall_exc_cerr = np.full((len(days), len(b)), np.nan)
wall_inh_cerr = np.full((len(days), len(b)), np.nan)
wall_non0_exc_cerr = np.full((len(days), len(b)), np.nan)
wall_non0_inh_cerr = np.full((len(days), len(b)), np.nan)

for iday in days2ana: #range(len(days)):
    
    inds = np.digitize(perClassErAll_aveS[iday,:], bins=b)
    inds = np.digitize(perClassErTestAll_aveS[iday,:], bins=b)

    for i in range(len(b)): #b[-1]

       perActiveAll_exc_cerr[iday,i] = np.mean(perActiveAll_exc_aveS[iday, inds==i+1])
       perActiveAll_inh_cerr[iday,i] = np.mean(perActiveAll_inh_aveS[iday, inds==i+1])
       
       wall_abs_exc_cerr[iday,i] = np.mean(wall_abs_exc_aveNS[iday, inds==i+1]) # average weights that correspond to class errors in bin i+1
       wall_abs_inh_cerr[iday,i] = np.mean(wall_abs_inh_aveNS[iday, inds==i+1])

       wall_non0_abs_exc_cerr[iday,i] = np.mean(wall_non0_abs_exc_aveNS[iday, inds==i+1]) # average weights that correspond to class errors in bin i+1
       wall_non0_abs_inh_cerr[iday,i] = np.mean(wall_non0_abs_inh_aveNS[iday, inds==i+1])

       wall_exc_cerr[iday,i] = np.mean(wall_exc_aveNS[iday, inds==i+1]) # average weights that correspond to class errors in bin i+1
       wall_inh_cerr[iday,i] = np.mean(wall_inh_aveNS[iday, inds==i+1])

       wall_non0_exc_cerr[iday,i] = np.mean(wall_non0_exc_aveNS[iday, inds==i+1]) # average weights that correspond to class errors in bin i+1
       wall_non0_inh_cerr[iday,i] = np.mean(wall_non0_inh_aveNS[iday, inds==i+1])


#%%       
#plt.plot(b, wall_exc_cerr[iday,:],'o')
plt.figure(figsize=(3,10))

plt.subplot(511)
plt.errorbar(b, np.nanmean(perActiveAll_exc_cerr, axis=0), yerr=np.nanstd(perActiveAll_exc_cerr, axis=0), fmt='b.-')
plt.errorbar(b, np.nanmean(perActiveAll_inh_cerr, axis=0), yerr=np.nanstd(perActiveAll_inh_cerr, axis=0), fmt='r.-')
plt.xlabel('class error')
plt.ylabel('% non 0')

plt.subplot(512)
plt.errorbar(b, np.nanmean(wall_non0_abs_exc_cerr, axis=0), yerr=np.nanstd(wall_non0_abs_exc_cerr, axis=0), fmt='b.-')
plt.errorbar(b, np.nanmean(wall_non0_abs_inh_cerr, axis=0), yerr=np.nanstd(wall_non0_abs_inh_cerr, axis=0), fmt='r.-')
plt.xlabel('class error')
plt.ylabel('non0 abs w')

plt.subplot(513)
plt.errorbar(b, np.nanmean(wall_abs_exc_cerr, axis=0), yerr=np.nanstd(wall_abs_exc_cerr, axis=0), fmt='b.-')
plt.errorbar(b, np.nanmean(wall_abs_inh_cerr, axis=0), yerr=np.nanstd(wall_abs_inh_cerr, axis=0), fmt='r.-')
plt.xlabel('class error')
plt.ylabel('abs w')

plt.subplot(514)
plt.errorbar(b, np.nanmean(wall_exc_cerr, axis=0), yerr=np.nanstd(wall_exc_cerr, axis=0), fmt='b.-')
plt.errorbar(b, np.nanmean(wall_inh_cerr, axis=0), yerr=np.nanstd(wall_inh_cerr, axis=0), fmt='r.-')
plt.xlabel('class error')
plt.ylabel('w')

plt.subplot(515)
plt.errorbar(b, np.nanmean(wall_non0_exc_cerr, axis=0), yerr=np.nanstd(wall_non0_exc_cerr, axis=0), fmt='b.-')
plt.errorbar(b, np.nanmean(wall_non0_inh_cerr, axis=0), yerr=np.nanstd(wall_non0_inh_cerr, axis=0), fmt='r.-')
plt.xlabel('class error')
plt.ylabel('non0 w')


if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnow,mousename+dp+'/allC', dgc, dta)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)

    fign = os.path.join(d, suffn[0:5]+'excInh_vsClassErr_aveAllDays'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')

