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

allExc = 1

singleDecoder = 1 # if 1, decoders of all tr subselects will be averaged to get a single decoder on which fract active weights, w magnitude, etc will be computed.
# if 0, these params will be computed for each decoder separately, then an average across decoders will be computed.


#%%
# Go to svm_plots_setVars and define vars!
execfile("svm_plots_setVars.py")   

dnow = '/excInh_cPath'

if trialHistAnalysis:
    dp = '/previousChoice'
else:
    dp = '/currentChoice'
    
    
#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName
def setSVMname(pnevFileName, trialHistAnalysis, allExc, itiName='all'):
    import glob
 
    if allExc:
        r = 'allExc'
    else:
        r = 'eqExcInh'
        
    if trialHistAnalysis:
        svmn = 'excInhC_chAl_%s_svmPrevChoice_%sITIs_*' %(r, itiName)
    else:
        svmn = 'excInhC_chAl_%s_svmCurrChoice_*' %(r)   
#    print '\n', svmn[:-1]
                           
    svmn = svmn + pnevFileName[-32:]    
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
    svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    svmName = svmName[0] # get the latest file
    
    return svmName

    
#%% PLOTS; define functions

def histerrbar(a,b,binEvery,p,colors = ['g','k'],dosd=0):
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
    if dosd:
        l1 = 1
        l2 = 1
    else:
        l1 = len(a)
        l2 = len(b)
    ax2 = plt.subplot(h2) #(gs[0,2:3])
    plt.errorbar([0,1], [a.mean(),b.mean()], [a.std()/np.sqrt(l1), b.std()/np.sqrt(l2)], marker='o',color='k', fmt='.')
    plt.xlim([-1,2])
#    plt.title('%.3f, %.3f' %(a.mean(), b.mean()))
    plt.xticks([0,1], (lab1, lab2), rotation='vertical')
    plt.ylabel(lab)
    plt.title('p=%.3f' %(p))
    makeNicePlots(ax2,0,1)
#    plt.tick_params
    
    plt.subplots_adjust(wspace=1, hspace=.5)
    return ax1,ax2



def errbarAllDays(a,b,p,dosd=0):
    '''
    eav = np.nanmean(a, axis=1) # average across shuffles
    iav = np.nanmean(b, axis=1)
    ele = np.shape(a)[1] - np.sum(np.isnan(a),axis=1) # number of non-nan shuffles of each day
    ile = np.shape(b)[1] - np.sum(np.isnan(b),axis=1) # number of non-nan shuffles of each day
    esd = np.divide(np.nanstd(a, axis=1), np.sqrt(ele))
    isd = np.divide(np.nanstd(b, axis=1), np.sqrt(ile))
    '''    
    eav = [np.nanmean(a[i]) for i in range(len(a))] #np.nanmean(a, axis=1) # average across shuffles
    iav = [np.nanmean(b[i]) for i in range(len(b))] #np.nanmean(b, axis=1)
    ele = [len(a[i]) for i in range(len(a))] #np.shape(a)[1] - np.sum(np.isnan(a),axis=1) # number of non-nan shuffles of each day
    ile = [len(b[i]) for i in range(len(b))] #np.shape(b)[1] - np.sum(np.isnan(b),axis=1) # number of non-nan shuffles of each day
    if dosd:# if u want sd and not se
        ele = 1
        ile = 1
    esd = np.divide([np.nanstd(a[i]) for i in range(len(a))], np.sqrt(ele))
    isd = np.divide([np.nanstd(b[i]) for i in range(len(b))], np.sqrt(ile))    
    
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

    
    


#%% Analysis starts here
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

perClassErTestChanceAll2 = []
perClassErTestShflAll2 = []

for iday in range(len(days)):    
       
    #%% Set .mat file names
       
    print '___________________'
    imagingFolder = days[iday][0:6]; #'151013'
    mdfFileNumber = map(int, (days[iday][7:]).split("-")); #[1,2] 
       
    pnev2load = [] #[] [3] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.
    signalCh = [2] # since gcamp is channel 2, should be always 2.
    postNProvided = 1; # If your directory does not contain pnevFile and instead it contains postFile, set this to 1 to get pnevFileName
    
    # from setImagingAnalysisNamesP import *
    
    imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load, postNProvided=postNProvided)
    
    postName = os.path.join(os.path.dirname(pnevFileName), 'post_'+os.path.basename(pnevFileName))
    moreName = os.path.join(os.path.dirname(pnevFileName), 'more_'+os.path.basename(pnevFileName))
    
    print(os.path.basename(imfilename))

    svmName = setSVMname(pnevFileName, trialHistAnalysis, allExc, itiName)    
    print os.path.basename(svmName)


    #%%     
    # if allExc=1, the dimension numSamples wont exist in the arrays below (this is the dimension where we shuffled exc neurons to make population of equal number exc and inh)
    # numTrialShuff is number of times you shuffled trials to do cross validation. (it is the variable numShuffles_ei in function inh_exc_classContribution)        

    if allExc:
        Data = scio.loadmat(svmName, variable_names=['perActive_exc_allExc', 'perActive_inh_allExc', 'wei_all_allExc', 'perClassEr_allExc', 'cvect_', 'bei_all_allExc', 'perClassErTest_allExc', 'perClassErTest_chance_allExc', 'perClassErTest_shfl_allExc'])
        perActive_exc = Data.pop('perActive_exc_allExc') # numSamples x numTrialShuff x length(cvect_)  # numSamples x length(cvect_)  # samples: shuffles of neurons
        perActive_inh = Data.pop('perActive_inh_allExc') # numSamples x numTrialShuff x length(cvect_)  # numSamples x length(cvect_)        
        wei_all = Data.pop('wei_all_allExc') # numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers) # numSamples x length(cvect_) x numNeurons(inh+exc equal numbers)  
        perClassEr = Data.pop('perClassEr_allExc') # numSamples x numTrialShuff x length(cvect_)
        cvect_ = Data.pop('cvect_').flatten() # length(cvect_)
        bei_all = Data.pop('bei_all_allExc') # numSamples x numTrialShuff x length(cvect_)  # numTrialShuff is number of times you shuffled trials to do cross validation. (it is variable numShuffles_ei in function inh_exc_classContribution)
        perClassErTest = Data.pop('perClassErTest_allExc') # numSamples x numTrialShuff x length(cvect_) 
        perClassErTest_chance = Data.pop('perClassErTest_chance_allExc')
        perClassErTest_shfl = Data.pop('perClassErTest_shfl_allExc')
    else:
        Data = scio.loadmat(svmName, variable_names=['perActive_exc', 'perActive_inh', 'wei_all', 'perClassEr', 'cvect_', 'bei_all', 'perClassErTest', 'perClassErTestChance', 'perClassErTestShfl'])
        perActive_exc = Data.pop('perActive_exc') # numSamples x numTrialShuff x length(cvect_)  # numSamples x length(cvect_)  # samples: shuffles of neurons
        perActive_inh = Data.pop('perActive_inh') # numSamples x numTrialShuff x length(cvect_)  # numSamples x length(cvect_)        
        wei_all = Data.pop('wei_all') # numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers) # numSamples x length(cvect_) x numNeurons(inh+exc equal numbers)  
        perClassEr = Data.pop('perClassEr') # numSamples x numTrialShuff x length(cvect_)
        cvect_ = Data.pop('cvect_').flatten() # length(cvect_)
        bei_all = Data.pop('bei_all') # numSamples x numTrialShuff x length(cvect_)  # numTrialShuff is number of times you shuffled trials to do cross validation. (it is variable numShuffles_ei in function inh_exc_classContribution)
        perClassErTest = Data.pop('perClassErTest') # numSamples x numTrialShuff x length(cvect_) 
        perClassErTest_chance = Data.pop('perClassErTestChance')
        perClassErTest_shfl = Data.pop('perClassErTestShfl')
    
   
    if abs(wei_all.sum()) < eps:
        print '\tWeights of all c and all shuffles are 0' # ... not analyzing' # I dont think you should do this!!
        
    Data = scio.loadmat(moreName, variable_names=['inhibitRois'])
    inhibitRois = Data.pop('inhibitRois')[0,:]

    Data = scio.loadmat(svmName, variable_names=['NsExcluded'])        
    NsExcluded = Data.pop('NsExcluded')[0,:].astype('bool')

    # Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)            
    inhibitRois = inhibitRois[~NsExcluded]            
    inhRois = inhibitRois[~np.isnan(inhibitRois)]
        
    if allExc==0:
        n = sum(inhRois==1)                
        inhRois_ei = np.zeros((2*n)); # first half is exc; 2nd half is inh
        inhRois_ei[n:2*n] = 1;
    else:
        inhRois_ei = inhRois

    # keep %non-0 w and classError for all shuffles of cvTrials and neurons (for all c values for all days)
    perActiveAll2_exc.append(perActive_exc) # numDays x numSamples x numTrialShuff x length(cvect_)        
    perActiveAll2_inh.append(perActive_inh)  # numDays x numSamples x numTrialShuff x length(cvect_)               
    perClassErAll2.append(perClassEr) # numDays x numSamples x numTrialShuff x length(cvect_)    
    perClassErTestAll2.append(perClassErTest) # numDays x numSamples x numTrialShuff x length(cvect_)        
    perClassErTestChanceAll2.append(perClassErTest_chance)
    perClassErTestShflAll2.append(perClassErTest_shfl)
    ball2.append(bei_all) # numDays x numSamples x numTrialShuff x length(cvect_) 
    cvect_all.append(cvect_) # numDays x length(cvect_)
    
    
    # I think it makes more sense to get average of each neuron's weigh across trial shuffles, and then compare dist of all neurons' ws btwn exc and inh
    # so I don't like below where you average ws across all neurons
    ########################### average ws across exc (inh) neurons # numDays (each day: numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers))      
    if allExc:
        wall2_exc.append(np.mean(wei_all[:,:,inhRois_ei==0], axis=-1)) 
        wall2_inh.append(np.mean(wei_all[:,:,inhRois_ei==1], axis=-1))                
    else:
        wall2_exc.append(np.mean(wei_all[:,:,:,inhRois_ei==0], axis=-1)) 
        wall2_inh.append(np.mean(wei_all[:,:,:,inhRois_ei==1], axis=-1))                
        
    # non0 ws... again I think it makes more sense to call a w 0 if it is 0 across all tr shuffles (bc ave across tr shfls is the best representative of the neuron's w)
    # exc
    if allExc:        
        a0 = wei_all[:,:,inhRois_ei==0] + 0
    else:
        a0 = wei_all[:,:,:,inhRois_ei==0] + 0
    a0[a0==0] = np.nan
    # inh
    if allExc:
        b0 = wei_all[:,:,inhRois_ei==1] + 0        
    else:
        b0 = wei_all[:,:,:,inhRois_ei==1] + 0
    b0[b0==0] = np.nan
 
 
   # for abs I think it makes more sense to average ws across tr shuffles, so we get a better representation of a neurons w, and then run abs on that   
    if allExc:         
        wall2_exc_abs.append(np.mean(abs(wei_all[:,:,inhRois_ei==0]), axis=-1)) #average across exc (inh) neurons # numDays (each day: numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers))     
        wall2_inh_abs.append(np.mean(abs(wei_all[:,:,inhRois_ei==1]), axis=-1)) 
    else:
        wall2_exc_abs.append(np.mean(abs(wei_all[:,:,:,inhRois_ei==0]), axis=-1)) #average across exc (inh) neurons # numDays (each day: numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers))     
        wall2_inh_abs.append(np.mean(abs(wei_all[:,:,:,inhRois_ei==1]), axis=-1)) 
        
    # non0
    wall2_exc_abs_non0.append(np.nanmean(abs(a0), axis=-1)) # average across neurons   
    wall2_inh_abs_non0.append(np.nanmean(abs(b0), axis=-1))    
    

    ########################### Take averages across trialShuffles    (# numTrialShuff is number of times you shuffled trials to do cross validation. (it is the variable numShuffles_ei in function inh_exc_classContribution)        )    
    if allExc:         
        axd = 0 # axes of tr shuffles
    else:
        axd = 1 # axes of tr shuffles
        
    ball.append(np.mean(bei_all, axis=0)) # numDays x numSamples x length(cvect_)   
    
    # ave ws across tr shuffles   ... this is like getting a single final decoder by averaging all decoders of all trial subselects.
    if allExc:         
        wall_exc.append(np.mean(wei_all[:,:,inhRois_ei==0], axis = axd)) # wall_exc[iday].shape: numSamples x length(cvect_) x numNeurons(inh+exc equal numbers)  #average w across trialShfls # numDays (each day: numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers))     
        wall_inh.append(np.mean(wei_all[:,:,inhRois_ei==1], axis = axd))      
    else:
        wall_exc.append(np.mean(wei_all[:,:,:,inhRois_ei==0], axis = axd)) # wall_exc[iday].shape: numSamples x length(cvect_) x numNeurons(inh+exc equal numbers)  #average w across trialShfls # numDays (each day: numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers))     
        wall_inh.append(np.mean(wei_all[:,:,:,inhRois_ei==1], axis = axd))      

    # abs of ave ws across tr shuffles. for abs I think it makes more sense to average ws across tr shuffles, so we get a better representation of a neurons w, and then run abs on that   
    if singleDecoder:
        wall_exc_abs.append(abs(wall_exc[iday])) #average abs w across trialShfls # numDays (each day: numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers))     
        wall_inh_abs.append(abs(wall_inh[iday]))        
    else: # for each decoder we get the abs of ws, then we average across decoders
#         I dont think this argument makes sense: for abs, I am not sure if you need to do the following... If you don't do it, the idea is that w of each neuron is represented by the average of its w across all trial shuffles... so we dont need to do abs!          
        if allExc:         
            wall_exc_abs.append(np.mean(abs(wei_all[:,:,inhRois_ei==0]), axis = axd)) #average abs w across trialShfls # numDays (each day: numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers))     
            wall_inh_abs.append(np.mean(abs(wei_all[:,:,inhRois_ei==1]), axis = axd))    
        else:
            wall_exc_abs.append(np.mean(abs(wei_all[:,:,:,inhRois_ei==0]), axis = axd)) #average abs w across trialShfls # numDays (each day: numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers))     
            wall_inh_abs.append(np.mean(abs(wei_all[:,:,:,inhRois_ei==1]), axis = axd))    


    # non0 ... again I think it makes more sense to call a w 0 if it is 0 across all tr shuffles (bc ave across tr shfls is the best representative of the neuron's w)
    # this is like bagging ... bc we are getting a single decoder by averaging weights across trials subselects. And then on this single decoder we compute abs w, etc..
    # in this way fract active weights should be also computed on this averaged final decoder.
    # exc
    a = wall_exc[iday] + 0
    a[a==0] = np.nan
    # inh
    b = wall_inh[iday] + 0
    b[b==0] = np.nan
    
    # take average of non0 ws across tr shuffles.
    if singleDecoder:
        wall_exc_non0.append(a)    
        wall_inh_non0.append(b)            
    else:
        # if you do nanmean, only tr shuffles w non0 ws will contribute to the average; if you do mean, even if in 1 tr shfl a weight is 0, it will turn its average ws across all tr shuffles also 0.
        wall_exc_non0.append(np.nanmean(a0, axis = axd)) # numDays (each day: numSamples x length(cvect_) x numNeurons(inh+exc equal numbers))     
        wall_inh_non0.append(np.nanmean(b0, axis = axd))        

    
    # abs of ave of non0 ws across tr shuffles
    if singleDecoder:
        wall_exc_abs_non0.append(abs(a))    
        wall_inh_abs_non0.append(abs(b))    
    else: # take average of abs of non0 ws across tr shuffles.    
        wall_exc_abs_non0.append(np.nanmean(abs(a0), axis = axd))        
        wall_inh_abs_non0.append(np.nanmean(abs(b0), axis = axd))        
    
    
    # fract active ws... computed on the averaged final decoder
    if singleDecoder:
        if allExc:
            perActiveAll_exc.append(100*np.mean(~np.isnan(a),axis=1))
            perActiveAll_inh.append(100*np.mean(~np.isnan(b),axis=1))  
        else:
            perActiveAll_exc.append(100*np.mean(~np.isnan(a),axis=2))
            perActiveAll_inh.append(100*np.mean(~np.isnan(b),axis=2))  
            
    else:
    # below is averaged across decoder ... not computed on the averaged final decoder    
        perActiveAll_exc.append(np.mean(perActive_exc, axis = axd)) # numDays x numSamples x length(cvect_)        
        perActiveAll_inh.append(np.mean(perActive_inh, axis = axd))        


    # below is averaged across decoder ... not computed on the averaged final decoder    
    perClassErAll.append(np.mean(perClassEr, axis = axd)) # numDays x numSamples x length(cvect_)     
    perClassErTestAll.append(np.mean(perClassErTest, axis = axd)) # numDays x numSamples x length(cvect_)     




#%%    
cvect_all = np.array(cvect_all) # days x length(cvect_)  # c values
ball = np.array(ball)

perActiveAll_exc = np.array(perActiveAll_exc)  # days x numSamples x length(cvect_) # percent non-zero weights
perActiveAll_inh = np.array(perActiveAll_inh)  # days x numSamples x length(cvect_) # percent non-zero weights
perClassErAll = np.array(perClassErAll)  # days x numSamples x length(cvect_) # classification error
perClassErTestAll = np.array(perClassErTestAll)
#wall_exc = np.array(wall_exc)
#wall_inh = np.array(wall_inh)
#wall_exc_non0 = np.array(wall_exc_non0)
#wall_inh_non0 = np.array(wall_inh_non0)
#wall_exc_abs = np.array(wall_exc_abs)
#wall_inh_abs = np.array(wall_inh_abs)
#wall_exc_abs_non0 = np.array(wall_exc_abs_non0)
#wall_inh_abs_non0 = np.array(wall_inh_abs_non0)

perClassErAll2 = np.array(perClassErAll2) # numDays x numSamples x numTrialShuff x length(cvect_) 
perClassErTestAll2 = np.array(perClassErTestAll2) # numDays x numSamples x numTrialShuff x length(cvect_) 
perClassErTestChanceAll2 = np.array(perClassErTestChanceAll2)
perClassErTestShflAll2 = np.array(perClassErTestShflAll2)
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

'''
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
'''


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
if allExc:    
    #######%%%%%%        
    if allExc==0:    
        numSamples = np.shape(perClassErTestAll)[1]
    nn = np.shape(perClassErTestAll2)[1]
    
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
        meanPerClassErrorTest = np.mean(perClassErTestAll2[iday,:,:], axis = 0) # perClassErTestAll_aveS[iday,:]
        semPerClassErrorTest = np.std(perClassErTestAll2[iday,:,:], axis = 0)/np.sqrt(nn) # perClassErTestAll_stdS[iday,:]/np.sqrt(np.shape(perClassErTestAll)[1])
        ix = np.nanargmin(meanPerClassErrorTest)
        cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
        cbest = cbest[0]; # best regularization term based on minError+SE criteria
        cbestAll[iday] = cbest
        
        
        # Another method: Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)        
        # compute percent non-zero w at each c
        a = perActiveAll2_exc[iday,:,:].squeeze()
        b = perActiveAll2_inh[iday,:,:].squeeze()    
        c = np.mean((a,b), axis=0) # percent non-0 w for each shuffle at each c (including both exc and inh neurons. Remember SVM was trained on both so there is a single decoder including both exc and inh)
        
    #    a = (wAllC!=0) # non-zero weights
    #    b = np.mean(a, axis=(0,2)) # Fraction of non-zero weights (averaged across shuffles)
        b = np.mean(c, axis=(0)) # Fraction of non-zero weights (averaged across shuffles)
        c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle
        
        cvectnow = cvect[c1stnon0:]
        meanPerClassErrorTestnow = np.mean(perClassErTestAll2[iday,:,c1stnon0:], axis = 0);
        semPerClassErrorTestnow = np.std(perClassErTestAll2[iday,:,c1stnon0:], axis = 0)/np.sqrt(nn);
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
        
    #    for ineurshfl in range(numSamples):
    #    a = np.mean(perClassErTestAll2[iday,:,:], axis = 0)
    #    plt.plot(cvect_all[iday,:], a, color=[.6,.6,.6]) # average across neuron and trial shuffles        
        '''
        plt.xscale('log')            
        a = np.mean(perClassErTestAll2[iday,:,:], axis = (0))
        plt.plot(cvect_all[iday,:], a, color='r')
        plt.plot(cbestAll[iday], a[cvect_all[iday,:]==cbestAll[iday]], 'ro')
        '''
        a = np.mean(perClassErTestAll2[iday,:,:], axis = (0))
        plt.plot(a, color='r')
        plt.plot(perActiveAll_exc[iday,:])
        plt.plot(perActiveAll_inh[iday,:])
        plt.plot(np.argwhere(cvect_all[iday,:]==cbestAll[iday]), a[cvect_all[iday,:]==cbestAll[iday]], 'ro')
        
#        for iday in range(len(days)):
#            plt.hist(80-activeInThisManyCvalsExc[iday][-1,:])            
        ax = plt.gca()
        makeNicePlots(ax)
        
        if iday!=len(days)-1:
            ax.xaxis.set_ticklabels([])
        
    plt.xlabel('c')
    plt.subplot(np.ceil(len(days)/float(3)),3,iday+1)
    plt.ylabel('% CV Class error')
    plt.subplots_adjust(wspace=.8, hspace=.4)
    
#    fign = '/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/SVM/excInh_cPath/fni17/currentChoice/rank/choiceAligned/curr_cPath.pdf'
#    plt.savefig(fign, bbox_inches='tight')
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
        
###########
else: # equal number of exc and inh
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
            
            
        


#%% part below commented bc on choice aligned traces we get very good performance and all c plots look good.

days2ana = np.array(range(len(days))) 

########################################################################        
########################################################################        
#%% given the figure above decide what days you want to exclude (days with bad shaped c plots!).. for now you are doing this manually... later u can use: days w shuffles for which %non0 at best c is 0 are always among bad days, but not all of the bad days... at best classErr must be less than classErr at the very low c... if you get the diff of c plot it must decrease at some point...
'''
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
trShflAve = 1 # if 1, %non0 and w will be computed on trial-shuffle averaged variales... if 0, they will be computed for all trial shuffles (and neuron shuffles) and plots in the next section will be made on the pooled shuffles (instead of tr shfl averages) 

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
    
'''


#%% Set %non0 and weights for exc and inh at bestc (across all shuffls) 
# you also set shuffles with all0 w (for both exc and inh) to nan.
# QUITE IMPORTANT NOTE: I think you should exclude days with poor c plot (or chance testing-data decoding performance)... bc it doesn't really make sense to compare their weight or fract-non0 of exc and inh neurons.
# For this you will need to get shuffles (shuffle labels of each trial)

#trShflAve = 1 # if 1, %non0 and w will be computed on trial-shuffle averaged variales... if 0, they will be computed for all trial shuffles (and neuron shuffles) and plots in the next section will be made on the pooled shuffles (instead of tr shfl averages) 

perActiveExc = []
perActiveInh = []
wNon0AbsExc = []
wNon0AbsInh = []    
wAbsExc = []
wAbsInh = []
wExc = []
wInh = []
wNon0Exc = []
wNon0Inh = []
    
if allExc:    
    perClassErTest_bestc = np.full(len(days), np.nan)
    perClassErTest_bestc_sd = np.full(len(days), np.nan)

    for iday in days2ana: #range(len(days)): # days2ana (if you want to exclude days2excl)
        
        ibc = cvect_all[iday,:]==cbestAll[iday]
        
        # class error for cv data at bestc 
        perClassErTest_bestc[iday] = perClassErTestAll[iday,ibc]
        perClassErTest_bestc_sd[iday] = np.std(perClassErTestAll2[iday,:,:], axis=0)[ibc] #perClassErTestAll_stdS[iday,ibc]
        
        
        # All the vars below are set using params computed on the final averaged decoder across trial subselects
        # percent non-zero w
        a = perActiveAll_exc[iday,ibc].squeeze() # perActiveAll_exc includes average across trial shuffles.
        b = perActiveAll_inh[iday,ibc].squeeze()
        '''
        c = np.mean((a,b), axis=0) #nShuffs x 1 # percent non-0 w for each shuffle (including both exc and inh neurons. Remember SVM was trained on both so there is a single decoder including both exc and inh)
        i0 = (c==0) # shuffles at which all w (of both exc and inh) are 0, ie no decoder was identified for these shuffles.
        a[i0] = np.nan # ignore shuffles with all-0 weights
        b[i0] = np.nan # ignore shuffles with all-0 weights    
        #    a[a==0.] = np.nan # ignore shuffles with all-0 weights ... this is problematic bc u will exclude a shuffle if inh w are all 0 even if exc w are not.
        #    b[b==0.] = np.nan # ignore shuffles with all-0 weights
        '''
        perActiveExc.append(a)
        perActiveInh.append(b)
        
        
        # abs non-zero w
        a = wall_exc_abs_non0[iday][ibc,:].squeeze()
        b = wall_inh_abs_non0[iday][ibc,:].squeeze()
    #        a = wall_non0_abs_exc_aveN[iday,:,ibc].squeeze() # for some shuffles this will be nan bc all weights were 0 ... so automatically you are ignoring shuffles w all-0 weights
    #        b = wall_non0_abs_inh_aveN[iday,:,ibc].squeeze()
        #    print np.argwhere(np.isnan(a))
        wNon0AbsExc.append(a)
        wNon0AbsInh.append(b)
        
        
        # abs w
        a = wall_exc_abs[iday][ibc,:].squeeze()
        b = wall_inh_abs[iday][ibc,:].squeeze()        
        '''
    #        a = wall_abs_exc_aveN[iday,:,ibc].squeeze() 
    #        b = wall_abs_inh_aveN[iday,:,ibc].squeeze()
        a[i0] = np.nan # ignore shuffles with all-0 weights
        b[i0] = np.nan # ignore shuffles with all-0 weights    
        #    a[a==0.] = np.nan # ignore shuffles with all-0 weights
        #    b[b==0.] = np.nan # ignore shuffles with all-0 weights    
        '''
        wAbsExc.append(a)
        wAbsInh.append(b)
        
        
        # w
        a = wall_exc[iday][ibc,:].squeeze()
        b = wall_inh[iday][ibc,:].squeeze()                
        '''
        a = wall_non0_exc_aveN[iday,:,ibc].squeeze() 
        b = wall_non0_inh_aveN[iday,:,ibc].squeeze()
        a[i0] = np.nan # ignore shuffles with all-0 weights
        b[i0] = np.nan # ignore shuffles with all-0 weights    
        #    a[a==0.] = np.nan # ignore shuffles with all-0 weights
        #    b[b==0.] = np.nan # ignore shuffles with all-0 weights    
        '''
        wExc.append(a)
        wInh.append(b)
    
    
        # non0 w
        a = wall_exc_non0[iday][ibc,:].squeeze()
        b = wall_inh_non0[iday][ibc,:].squeeze()               
        '''
        a = wall_non0_exc_aveN[iday,:,ibc].squeeze() 
        b = wall_non0_inh_aveN[iday,:,ibc].squeeze()
        a[i0] = np.nan # ignore shuffles with all-0 weights
        b[i0] = np.nan # ignore shuffles with all-0 weights    
        #    a[a==0.] = np.nan # ignore shuffles with all-0 weights
        #    b[b==0.] = np.nan # ignore shuffles with all-0 weights    
        '''
        wNon0Exc.append(a)
        wNon0Inh.append(b)

    perActiveExc = np.array(perActiveExc)
    perActiveInh = np.array(perActiveInh)
    wNon0AbsExc = np.array(wNon0AbsExc)
    wNon0AbsInh = np.array(wNon0AbsInh)    
    wAbsExc = np.array(wAbsExc)
    wAbsInh = np.array(wAbsInh) 
    wExc = np.array(wExc)
    wInh = np.array(wInh)
    wNon0Exc = np.array(wNon0Exc)
    wNon0Inh = np.array(wNon0Inh)

##################################################    
else: # equal number of exc and inh
    perClassErTest_bestc = np.full((len(days),numSamples), np.nan)
    perClassErTest_bestc_sd = np.full((len(days),numSamples), np.nan)    

    for iday in days2ana: #range(len(days)): # days2ana (if you want to exclude days2excl)
        
        ibc = cvect_all[iday,:]==cbestAll[iday]
        
        # class error for cv data at bestc 
        perClassErTest_bestc[iday,:] = perClassErTestAll[iday,:,ibc].squeeze()
        perClassErTest_bestc_sd[iday,:] = np.std(perClassErTestAll2[iday,:,:,ibc], axis=1).squeeze() #perClassErTestAll_stdS[iday,ibc]
        
        
        # All the vars below are set using params computed on the final averaged decoder across trial subselects
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
        a = wall_exc_abs_non0[iday][:,ibc,:].squeeze()
        b = wall_inh_abs_non0[iday][:,ibc,:].squeeze()
    #        a = wall_non0_abs_exc_aveN[iday,:,ibc].squeeze() # for some shuffles this will be nan bc all weights were 0 ... so automatically you are ignoring shuffles w all-0 weights
    #        b = wall_non0_abs_inh_aveN[iday,:,ibc].squeeze()
        #    print np.argwhere(np.isnan(a))
        wNon0AbsExc.append(a)
        wNon0AbsInh.append(b)
        
        
        # abs w
        a = wall_exc_abs[iday][:,ibc,:].squeeze()
        b = wall_inh_abs[iday][:,ibc,:].squeeze()        

    #        a = wall_abs_exc_aveN[iday,:,ibc].squeeze() 
    #        b = wall_abs_inh_aveN[iday,:,ibc].squeeze()
        a[i0] = np.nan # ignore shuffles with all-0 weights
        b[i0] = np.nan # ignore shuffles with all-0 weights    
        #    a[a==0.] = np.nan # ignore shuffles with all-0 weights
        #    b[b==0.] = np.nan # ignore shuffles with all-0 weights    
        
        wAbsExc.append(a)
        wAbsInh.append(b)
        
        
        # w
        a = wall_exc[iday][:,ibc,:].squeeze()
        b = wall_inh[iday][:,ibc,:].squeeze()                
        
#        a = wall_non0_exc_aveN[iday,:,ibc].squeeze() 
#        b = wall_non0_inh_aveN[iday,:,ibc].squeeze()
        a[i0] = np.nan # ignore shuffles with all-0 weights
        b[i0] = np.nan # ignore shuffles with all-0 weights    
        #    a[a==0.] = np.nan # ignore shuffles with all-0 weights
        #    b[b==0.] = np.nan # ignore shuffles with all-0 weights    
        
        wExc.append(a)
        wInh.append(b)
    
    
        # non0 w
        a = wall_exc_non0[iday][:,ibc,:].squeeze()
        b = wall_inh_non0[iday][:,ibc,:].squeeze()               
        
#        a = wall_non0_exc_aveN[iday,:,ibc].squeeze() 
#        b = wall_non0_inh_aveN[iday,:,ibc].squeeze()
        a[i0] = np.nan # ignore shuffles with all-0 weights
        b[i0] = np.nan # ignore shuffles with all-0 weights    
        #    a[a==0.] = np.nan # ignore shuffles with all-0 weights
        #    b[b==0.] = np.nan # ignore shuffles with all-0 weights    
        
        wNon0Exc.append(a)
        wNon0Inh.append(b)
    
    
    perActiveExc = np.array(perActiveExc)
    perActiveInh = np.array(perActiveInh)
    
    

#%%
###################################### PLOTS ###################################### 

#%%
if allExc:
    ###############%% classification accuracy of cv data at best c for each day ###############
    # for each day plot averages across shuffles (neuron shuffles)
    
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
    
    
    # below wont work if you computed fract active weights on the averaged decoder across trial subselects.
    ### show ave and std across shuffles for each day
    '''
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
    '''
    
    
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
    binEvery = .001# .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
    #a = np.reshape(wNon0AbsExc,(-1,)) 
    #b = np.reshape(wNon0AbsInh,(-1,))
    a = np.concatenate(wNon0AbsExc) 
    b = np.concatenate(wNon0AbsInh)
    
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
    ax1.set_xlim([0, .1])
    #ax1.set_xlim([0, .5])
    
    
    ### show individual days
    a = wNon0AbsExc
    b = wNon0AbsInh
    # if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
    p = []
    for i in range(len(a)):
        _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
        p.append(p0)
    p = np.array(p)    
#    print p
    '''
    if trShflAve==0:
        a = np.reshape(wNon0AbsExc, (len(days), numCVtrShfls*numSamples), order = 'F')
        b = np.reshape(wNon0AbsInh, (len(days), numCVtrShfls*numSamples), order = 'F')
    _,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
    print p
    '''
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
    binEvery = .001 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
    #a = np.reshape(wAbsExc,(-1,)) 
    #b = np.reshape(wAbsInh,(-1,))
    a = np.concatenate(wAbsExc) 
    b = np.concatenate(wAbsInh)
    
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
    ax1.set_xlim([0, .08])
    
    ### show individual days
    a = wAbsExc
    b = wAbsInh
    # if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
    p = []
    for i in range(len(a)):
        _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
        p.append(p0)
    p = np.array(p)    
#    print p
    #if trShflAve==0:
    #    a = np.reshape(wAbsExc, (len(days), numCVtrShfls*numSamples), order = 'F')
    #    b = np.reshape(wAbsInh, (len(days), numCVtrShfls*numSamples), order = 'F')
    #_,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
    #print p
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
    #a = np.reshape(wExc,(-1,)) 
    #b = np.reshape(wInh,(-1,))
    a = np.concatenate(wExc) 
    b = np.concatenate(wInh)
    
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
    ax1.set_xlim([-.03, .03])
    
    
    ### show individual days
    a = wExc
    b = wInh
    # if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
    p = []
    for i in range(len(a)):
        _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
        p.append(p0)
    p = np.array(p)    
#    print p
    #if trShflAve==0:
    #    a = np.reshape(wExc, (len(days), numCVtrShfls*numSamples), order = 'F')
    #    b = np.reshape(wInh, (len(days), numCVtrShfls*numSamples), order = 'F')
    #_,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
    #print p
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
        
        
    
    
    
    ###############%% non0 w  ###############
    ### hist and P val for all days pooled (exc vs inh)
    lab = 'non0 w'
    binEvery = .001 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
    #a = np.reshape(wExc,(-1,)) 
    #b = np.reshape(wInh,(-1,))
    a = np.concatenate(wNon0Exc) 
    b = np.concatenate(wNon0Inh)
    
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
    ax1.set_xlim([-.08, .08])
#    ax1.set_xlim([-.04, .04])    
#    makeNicePlots(ax1,1,1)
#    plt.savefig('non0w_bestc_excInh_fni16.pdf', bbox_inches='tight')
    
    ### show individual days
    a = wNon0Exc
    b = wNon0Inh
    # if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
    p = []
    for i in range(len(a)):
        _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
        p.append(p0)
    p = np.array(p)    
#    print p
    #if trShflAve==0:
    #    a = np.reshape(wExc, (len(days), numCVtrShfls*numSamples), order = 'F')
    #    b = np.reshape(wInh, (len(days), numCVtrShfls*numSamples), order = 'F')
    #_,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
    #print p
    #ax = plt.subplot(gs[1,0:3])
    errbarAllDays(a,b,p)
    #plt.ylabel(lab)
    
    
    if savefigs:#% Save the figure
        d = os.path.join(svmdir+dnow,mousename+dp+'/bestC', dgc, dta)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
     
        fign = os.path.join(d, suffn[0:5]+'excVSinh_allDays_non0W'+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight')
    
    

#%% equal number of exc and inh ... remember you are pooling across neural shuffles... so you are comparing a lot more weights here than allExc=0 ... many of the weights supposedly correspond to the same neuron, but used in different decoders... for this reason I am not sure if sig test is valid here.
else:

    #%%
    ###############%% classification accuracy of cv data at best c for each day ###############
    # for each day plot averages across shuffles (neuron shuffles)
    
    plt.figure(figsize=(5,2.5))    
    gs = gridspec.GridSpec(1, 10)#, width_ratios=[2, 1]) 
    h1 = gs[0,0:7]
    ax = plt.subplot(h1)
    plt.errorbar(np.arange(len(days)), 100-np.mean(perClassErTest_bestc,axis=1), np.std(perClassErTest_bestc,axis=1), color='k')
    plt.xlim([-1,len(days)+1])
    plt.xlabel('Days')
    plt.ylabel('% Class accuracy')
    ymin, ymax = ax.get_ylim()
    makeNicePlots(plt.gca())
    
    # ave and std across days
    h2 = gs[0,7:8]
    ax = plt.subplot(h2)
    plt.errorbar(0, np.nanmean(100-perClassErTest_bestc.reshape(-1,)), np.nanstd(100-perClassErTest_bestc.reshape(-1,)), marker='o', color='k')
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
    
    
    # below wont work if you computed fract active weights on the averaged decoder across trial subselects.
    ### show ave and std across shuffles for each day
    
    # if averaged across trial shuffles
    a = perActiveExc # numDays x numSamples x numTrialShuff
    b = perActiveInh
    p = []
    for i in range(len(a)):
        _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
        p.append(p0)
    p = np.array(p)    
    print p
    # if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
    #if trShflAve==0:
    #    a = np.reshape(perActiveExc, (len(days), numCVtrShfls*numSamples), order = 'F')
    #    b = np.reshape(perActiveInh, (len(days), numCVtrShfls*numSamples), order = 'F')
    #_,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
#    print p
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
    binEvery = .001# .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
    #a = np.reshape(wNon0AbsExc,(-1,)) 
    #b = np.reshape(wNon0AbsInh,(-1,))
    # pool across all neuron shuffles
    aa = np.array([np.reshape(wNon0AbsExc[iday],(-1,)) for iday in range(len(days))])
    bb = np.array([np.reshape(wNon0AbsInh[iday],(-1,)) for iday in range(len(days))])
    #np.sum([wNon0AbsInh[iday].shape[1] for iday in range(len(days))]) # total number of inh(exc) neurons for all days (in one of the neuron shuffles)
    a = np.concatenate(aa) 
    b = np.concatenate(bb)
    
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
    ax1.set_xlim([0, .1])
    #ax1.set_xlim([0, .5])
    
    
    ### show individual days
    a = aa#wNon0AbsExc
    b = bb#wNon0AbsInh
    # if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
    p = []
    for i in range(len(a)):
        _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
        p.append(p0)
    p = np.array(p)    
    print p
    '''
    if trShflAve==0:
        a = np.reshape(wNon0AbsExc, (len(days), numCVtrShfls*numSamples), order = 'F')
        b = np.reshape(wNon0AbsInh, (len(days), numCVtrShfls*numSamples), order = 'F')
    _,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
    print p
    '''
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
    binEvery = .001 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
    #a = np.reshape(wAbsExc,(-1,)) 
    #b = np.reshape(wAbsInh,(-1,))
    # pool across all neuron shuffles
    aa = np.array([np.reshape(wAbsExc[iday],(-1,)) for iday in range(len(days))])
    bb = np.array([np.reshape(wAbsInh[iday],(-1,)) for iday in range(len(days))])
    a = np.concatenate(aa) 
    b = np.concatenate(bb)
    
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
    ax1.set_xlim([0, .08])
    
    ### show individual days
    a = aa#wAbsExc
    b = bb#wAbsInh
    # if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
    p = []
    for i in range(len(a)):
        _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
        p.append(p0)
    p = np.array(p)    
    print p
    #if trShflAve==0:
    #    a = np.reshape(wAbsExc, (len(days), numCVtrShfls*numSamples), order = 'F')
    #    b = np.reshape(wAbsInh, (len(days), numCVtrShfls*numSamples), order = 'F')
    #_,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
    #print p
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
    #a = np.reshape(wExc,(-1,)) 
    #b = np.reshape(wInh,(-1,))
    # pool across all neuron shuffles
    aa = np.array([np.reshape(wExc[iday],(-1,)) for iday in range(len(days))])
    bb = np.array([np.reshape(wInh[iday],(-1,)) for iday in range(len(days))])
    a = np.concatenate(aa) 
    b = np.concatenate(bb)
    
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
    ax1.set_xlim([-.03, .03])
    
    
    ### show individual days
    a = aa#wExc
    b = bb#wInh
    # if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
    p = []
    for i in range(len(a)):
        _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
        p.append(p0)
    p = np.array(p)    
    print p
    #if trShflAve==0:
    #    a = np.reshape(wExc, (len(days), numCVtrShfls*numSamples), order = 'F')
    #    b = np.reshape(wInh, (len(days), numCVtrShfls*numSamples), order = 'F')
    #_,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
    #print p
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
        
        
    
    
    
    ###############%% non0 w  ###############
    ### hist and P val for all days pooled (exc vs inh)
    lab = 'non0 w'
    binEvery = .001 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
    #a = np.reshape(wExc,(-1,)) 
    #b = np.reshape(wInh,(-1,))
    # pool across all neuron shuffles
    aa = np.array([np.reshape(wNon0Exc[iday],(-1,)) for iday in range(len(days))])
    bb = np.array([np.reshape(wNon0Inh[iday],(-1,)) for iday in range(len(days))])
    a = np.concatenate(aa) 
    b = np.concatenate(bb)
    
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
    ax1.set_xlim([-.08, .08])
    
    
    ### show individual days
    a = aa#wNon0Exc
    b = bb#wNon0Inh
    # if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
    p = []
    for i in range(len(a)):
        _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
        p.append(p0)
    p = np.array(p)    
    print p
    #if trShflAve==0:
    #    a = np.reshape(wExc, (len(days), numCVtrShfls*numSamples), order = 'F')
    #    b = np.reshape(wInh, (len(days), numCVtrShfls*numSamples), order = 'F')
    #_,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
    #print p
    #ax = plt.subplot(gs[1,0:3])
    errbarAllDays(a,b,p)
    #plt.ylabel(lab)
    
    
    if savefigs:#% Save the figure
        d = os.path.join(svmdir+dnow,mousename+dp+'/bestC', dgc, dta)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
     
        fign = os.path.join(d, suffn[0:5]+'excVSinh_allDays_non0W'+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight')
        
       
       
       
       
       



#%% all C (same as above, but pulling all values of c instead of only selecting bestc values)

if allExc:
    perActiveExc = []
    perActiveInh = []
    wNon0AbsExc = []
    wNon0AbsInh = []    
    wAbsExc = []
    wAbsInh = []
    wExc = []
    wInh = []
    wNon0Exc = []
    wNon0Inh = []
    perClassErTest_bestc = np.full(len(days), np.nan)
    perClassErTest_bestc_sd = np.full(len(days), np.nan)
    
    for iday in days2ana: #range(len(days)): # days2ana (if you want to exclude days2excl)
        
        ibc = cvect_all[iday,:]==cbestAll[iday]
        
        # class error for cv data at bestc 
        perClassErTest_bestc[iday] = perClassErTestAll[iday,ibc]
        perClassErTest_bestc_sd[iday] = np.std(perClassErTestAll2[iday,:,:], axis=0)[ibc] #perClassErTestAll_stdS[iday,ibc]
        
        
        # All the vars below are set using params computed on the final averaged decoder across trial subselects
        # percent non-zero w
        a = perActiveAll_exc[iday,:].squeeze() # perActiveAll_exc includes average across trial shuffles.
        b = perActiveAll_inh[iday,:].squeeze()
        
        c = np.mean((a,b), axis=0) #nShuffs x 1 # percent non-0 w for each shuffle (including both exc and inh neurons. Remember SVM was trained on both so there is a single decoder including both exc and inh)
        # c values at which all weights are 0 or all neurons are contributing.
        i0 = (c==0)
        i0 = (np.logical_or(c==0,c==100)) # shuffles at which all w (of both exc and inh) are 0, ie no decoder was identified for these shuffles.
        a[i0] = np.nan # ignore shuffles with all-0 weights
        b[i0] = np.nan # ignore shuffles with all-0 weights    
        #    a[a==0.] = np.nan # ignore shuffles with all-0 weights ... this is problematic bc u will exclude a shuffle if inh w are all 0 even if exc w are not.
        #    b[b==0.] = np.nan # ignore shuffles with all-0 weights
        
        perActiveExc.append(a) 
        perActiveInh.append(b)
        
        
        
    
        # abs non-zero w (it is already nan for 0 ws... so you dont need to set i0 to nan)
        a = wall_exc_abs_non0[iday].squeeze()+0
        b = wall_inh_abs_non0[iday].squeeze()+0
    #        a = wall_non0_abs_exc_aveN[iday,:,ibc].squeeze() # for some shuffles this will be nan bc all weights were 0 ... so automatically you are ignoring shuffles w all-0 weights
    #        b = wall_non0_abs_inh_aveN[iday,:,ibc].squeeze()
        #    print np.argwhere(np.isnan(a))
        wNon0AbsExc.append(a.reshape((-1,))) # all weights at all c values
        wNon0AbsInh.append(b.reshape((-1,)))
        
        
        
        
        # abs w
        a = wall_exc_abs[iday].squeeze()+0
        b = wall_inh_abs[iday].squeeze()+0        
    
    #        a = wall_abs_exc_aveN[iday,:,ibc].squeeze() 
    #        b = wall_abs_inh_aveN[iday,:,ibc].squeeze()
#        a[i0,:] = np.nan # ignore shuffles with all-0 weights
#        b[i0,:] = np.nan # ignore shuffles with all-0 weights    
        #    a[a==0.] = np.nan # ignore shuffles with all-0 weights
        #    b[b==0.] = np.nan # ignore shuffles with all-0 weights    
        
        wAbsExc.append(a.reshape((-1,)))
        wAbsInh.append(b.reshape((-1,)))
        
        
        
        
        # w
        a = wall_exc[iday].squeeze()+0
        b = wall_inh[iday].squeeze()+0                
        
    #        a = wall_non0_exc_aveN[iday,:,ibc].squeeze() 
    #        b = wall_non0_inh_aveN[iday,:,ibc].squeeze()
#        a[i0,:] = np.nan # ignore shuffles with all-0 weights
#        b[i0,:] = np.nan # ignore shuffles with all-0 weights    
        #    a[a==0.] = np.nan # ignore shuffles with all-0 weights
        #    b[b==0.] = np.nan # ignore shuffles with all-0 weights    
        
        wExc.append(a.reshape((-1,)))
        wInh.append(b.reshape((-1,)))
    
    
    
    
    
        # non0 w
        a = wall_exc_non0[iday].squeeze()+0
        b = wall_inh_non0[iday].squeeze()+0               
        
    #        a = wall_non0_exc_aveN[iday,:,ibc].squeeze() 
    #        b = wall_non0_inh_aveN[iday,:,ibc].squeeze()
#        a[i0,:] = np.nan # ignore shuffles with all-0 weights
#        b[i0,:] = np.nan # ignore shuffles with all-0 weights    
        #    a[a==0.] = np.nan # ignore shuffles with all-0 weights
        #    b[b==0.] = np.nan # ignore shuffles with all-0 weights    
        
        wNon0Exc.append(a.reshape((-1,)))
        wNon0Inh.append(b.reshape((-1,)))
        
        
    perActiveExc = np.array(perActiveExc)
    perActiveInh = np.array(perActiveInh)
    wNon0AbsExc = np.array(wNon0AbsExc)
    wNon0AbsInh = np.array(wNon0AbsInh)    
    wAbsExc = np.array(wAbsExc)
    wAbsInh = np.array(wAbsInh) 
    wExc = np.array(wExc)
    wInh = np.array(wInh)
    wNon0Exc = np.array(wNon0Exc)
    wNon0Inh = np.array(wNon0Inh)
        
        
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
    
    
    # below wont work if you computed fract active weights on the averaged decoder across trial subselects.
    ### show ave and std across shuffles for each day
    '''
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
    '''
    
    
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
    binEvery = .001# .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
    #a = np.reshape(wNon0AbsExc,(-1,)) 
    #b = np.reshape(wNon0AbsInh,(-1,))
    a = np.concatenate(wNon0AbsExc) 
    b = np.concatenate(wNon0AbsInh)
    
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
    ax1.set_xlim([0, .1])
    #ax1.set_xlim([0, .5])
    
    
    ### show individual days
    a = wNon0AbsExc
    b = wNon0AbsInh
    # if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
    p = []
    for i in range(len(a)):
        _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
        p.append(p0)
    p = np.array(p)    
#    print p
    '''
    if trShflAve==0:
        a = np.reshape(wNon0AbsExc, (len(days), numCVtrShfls*numSamples), order = 'F')
        b = np.reshape(wNon0AbsInh, (len(days), numCVtrShfls*numSamples), order = 'F')
    _,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
    print p
    '''
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
    binEvery = .001 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
    #a = np.reshape(wAbsExc,(-1,)) 
    #b = np.reshape(wAbsInh,(-1,))
    a = np.concatenate(wAbsExc) 
    b = np.concatenate(wAbsInh)
    
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
    ax1.set_xlim([0, .08])
    
    ### show individual days
    a = wAbsExc
    b = wAbsInh
    # if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
    p = []
    for i in range(len(a)):
        _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
        p.append(p0)
    p = np.array(p)    
#    print p
    #if trShflAve==0:
    #    a = np.reshape(wAbsExc, (len(days), numCVtrShfls*numSamples), order = 'F')
    #    b = np.reshape(wAbsInh, (len(days), numCVtrShfls*numSamples), order = 'F')
    #_,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
    #print p
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
    #a = np.reshape(wExc,(-1,)) 
    #b = np.reshape(wInh,(-1,))
    a = np.concatenate(wExc) 
    b = np.concatenate(wInh)
    
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
    ax1.set_xlim([-.03, .03])
    
    
    ### show individual days
    a = wExc
    b = wInh
    # if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
    p = []
    for i in range(len(a)):
        _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
        p.append(p0)
    p = np.array(p)    
#    print p
    #if trShflAve==0:
    #    a = np.reshape(wExc, (len(days), numCVtrShfls*numSamples), order = 'F')
    #    b = np.reshape(wInh, (len(days), numCVtrShfls*numSamples), order = 'F')
    #_,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
    #print p
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
        
        
    
    
    
    ###############%% non0 w  ###############
    ### hist and P val for all days pooled (exc vs inh)
    lab = 'non0 w'
    binEvery = .001 # .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)
    #a = np.reshape(wExc,(-1,)) 
    #b = np.reshape(wInh,(-1,))
    a = np.concatenate(wNon0Exc) 
    b = np.concatenate(wNon0Inh)
    
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
    ax1.set_xlim([-.08, .08])
    
    
    ### show individual days
    a = wNon0Exc
    b = wNon0Inh
    # if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
    p = []
    for i in range(len(a)):
        _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
        p.append(p0)
    p = np.array(p)    
#    print p
    #if trShflAve==0:
    #    a = np.reshape(wExc, (len(days), numCVtrShfls*numSamples), order = 'F')
    #    b = np.reshape(wInh, (len(days), numCVtrShfls*numSamples), order = 'F')
    #_,p = stats.ttest_ind(a.transpose(), b.transpose(), nan_policy='omit')    
    #print p
    #ax = plt.subplot(gs[1,0:3])
    errbarAllDays(a,b,p)
    #plt.ylabel(lab)
    
    
    if savefigs:#% Save the figure
        d = os.path.join(svmdir+dnow,mousename+dp+'/bestC', dgc, dta)
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)
     
        fign = os.path.join(d, suffn[0:5]+'excVSinh_allDays_non0W'+'.'+fmt[0])
        plt.savefig(fign, bbox_inches='tight')
        
        
        
        
        
    