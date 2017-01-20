# -*- coding: utf-8 -*-
"""
Compare in the top 10 most appearing ws (at all values of c), what fraction are exc and what fraction are inh


Created on Tue Jan 17 18:50:59 2017
@author: farznaj
"""

#%% Below is copied from the beginning of svm_excInh_cPath_plots; except it only computes wall_exc and wall_inh.
# If you have run svm_excInh_cPath_plots, you don't need to run the parts below.

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
    Data = scio.loadmat(svmName, variable_names=['wei_all'])
    wei_all = Data.pop('wei_all') # numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers) # numSamples x length(cvect_) x numNeurons(inh+exc equal numbers)  
    
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
    inhRois_ei = np.zeros((2*n)); # first half is exc; 2nd half is inh
    inhRois_ei[n:2*n] = 1;
     
       
    ######### Take averages across trialShuffles        
    wall_exc.append(np.mean(wei_all[:,:,:,inhRois_ei==0], axis=1)) # wall_exc[iday].shape: numSamples x length(cvect_) x numNeurons(inh+exc equal numbers)  #average w across trialShfls # numDays (each day: numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers))     
    wall_inh.append(np.mean(wei_all[:,:,:,inhRois_ei==1], axis=1))      



#%%
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

# below will be always 1 for the rank analysis
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
    














#%%
'''
#####################################################################################################################################################    
################# Compare probability of top-rank neurons (ie neurons that appear at many values of c) #########################################################################################
######## to be excitatory vs inhibitory #####################################################################################
#####################################################################################################################################################   
'''    
#%% Count number of times that each w appears while we increase c (regularization term)

#rankw = np.full((len(days),1), np.nan).squeeze()

numSamples = np.shape(wall_exc[0])[0]
lenC = np.shape(wall_exc[0])[1]

activeInThisManyCvals = []

for iday in range(len(days)):
    print '______________',iday,'______________'

#    a = np.full((numSamples, len(cvect_), wall_exc[iday].shape[2]*2), np.nan)
    a = np.zeros((numSamples, lenC, wall_exc[iday].shape[2]*2)) #numSamples x length(cvect_) x numNeurons(inh+exc equal numbers)

    activeInThisManyCvals.append(a)
    
    print np.shape(activeInThisManyCvals[iday])
    
    for ineurshfl in range(numSamples):
        print ineurshfl
        
#        rk = 1
        for icval in np.arange(1,lenC): #range(len(cvect_)):
#			wall_exc[iday][ineurshfl, ibc, :].squeeze() #ws of exc neurons at bestc for a particular neuron shuffle averaged across trial shuffles #numNeurons #wall_exc[iday]: numSamples x length(cvect_) x numNeurons(inh+exc equal numbers)
            a1 = wall_exc[iday][ineurshfl, icval, :] #num exc Neurons
            a2 = wall_inh[iday][ineurshfl, icval, :]
            a = np.concatenate((a1,a2)) # ws of exc first; then inh
            
            non0ws = (a!=0)
#            print icval, sum(non0ws)>0
               
            if sum(non0ws)>0: # at least 1 non0 w exists at icval-th c value
#                print 'non0 found'
#                print activeInThisManyCvals[iday][ineurshfl][icval-1][non0ws]
                activeInThisManyCvals[iday][ineurshfl][icval][non0ws] = activeInThisManyCvals[iday][ineurshfl][icval-1][non0ws] + 1 # add 1 if neuron is non0 in this c val # assign rank to non0 ws that appear at this c val
#                rk = rk+1
#                print activeInThisManyCvals[iday][ineurshfl][icval][non0ws]

# activeInThisManyCvals[iday][ineurshfl][-1]: shows in how many c vals each neuron appeared. 


#%% compute in the top 10 most appearing ws, what fraction are exc and what fraction are inh

highAppsExc = np.full((len(days), numSamples), np.nan)
highAppsInh = np.full((len(days), numSamples), np.nan)

for iday in days2ana: #range(len(days)):
#    print '______________',iday,'______________'
    for ineurshfl in range(numSamples):
#        print ineurshfl    
        a = np.argsort((activeInThisManyCvals[iday][ineurshfl][-1]))
        u = np.unique(activeInThisManyCvals[iday][ineurshfl][-1][a[-10:]]) 
        highApps = np.in1d(activeInThisManyCvals[iday][ineurshfl][-1], u) # indeces of the top 10 most appearing ws (could be more than 10 if ws had equal number of appeareances)
        #[x for x in activeInThisManyCvals[iday][ineurshfl][-1] if x in u]        
        
        # now form the exc inh array
        n = len(activeInThisManyCvals[iday][ineurshfl][icval])/2 # number of inh + exc neurons
        inhRois_ei = np.zeros((2*n));
        inhRois_ei[0:n] = 0;
        inhRois_ei[n:2*n] = 1;
        
        # now compute the fraction of high appearance ws that are exc and that are inh
        highAppsExc[iday][ineurshfl] = (highApps[inhRois_ei==0]).sum() / float(highApps.sum()) # out of the highApps, how many were exc
        highAppsInh[iday][ineurshfl] = (highApps[inhRois_ei==1]).sum() / float(highApps.sum()) # out of the highApps, how many were inh



np.nanmean(highAppsExc, axis=(0,1)), np.nanmean(highAppsInh, axis=(0,1))


#%% plot dists
lab1 = 'exc'
lab2 = 'inh'
colors = ['k','r']

lab = 'P(top-rank w being exc or inh)'
#binEvery = 2#5# .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)

a = np.reshape(highAppsExc,(-1,))# perActiveExc includes average across neuron shuffles for each day.... 
b = np.reshape(highAppsInh,(-1,))  
a = a[~(np.isnan(a) + np.isinf(a))]
b = b[~(np.isnan(b) + np.isinf(b))]

r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
binEvery = r/float(10)  

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
plt.subplot(h1)
plt.ylabel('Prob (all days & N shuffs)')


if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnow,mousename+dp+'/rank', dgc, dta)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
    fign = os.path.join(d, suffn[0:5]+'probTopRank'+'.'+fmt[0])
    plt.savefig(fign, bbox_inches='tight')
        
        