# -*- coding: utf-8 -*-
"""
Compare FRs of exc vs inh neurons for choice-aligned X (traces averaged 300ms before choice) that went into SVM.
Both single-trial FRs and trial-averaged FRs are plotted.

Created on Tue Mar 14 18:27:36 2017
@author: farznaj
"""

mousename = 'fni17' #'fni17'

trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
execfile("svm_plots_setVars.py")   

savefigs = 1

normX = 0 # normalize X (just how SVM was performed) before computing FRs


#%%
allExc = 1
dnow = '/excInh_FRs'

if normX==1:
    dp = '_norm.'
else:
    dp = '.'

    
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

       

####################################################################################################
####################################################################################################
#%%

X_chAl_exc_all = []
X_chAl_inh_all = []
meanX_chAl_exc_all = []
meanX_chAl_inh_all = []
    
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

   
    #%% Load vars
   
    Data = scio.loadmat(moreName, variable_names=['inhibitRois'])
    inhibitRois = Data.pop('inhibitRois')[0,:]

    Data = scio.loadmat(svmName, variable_names=['NsExcluded'])        
    NsExcluded = Data.pop('NsExcluded')[0,:].astype('bool')
    
    Data = scio.loadmat(svmName, variable_names=['trsExcluded'])        
    trsExcluded_chAl = Data.pop('trsExcluded')[0,:].astype('bool')

    inhibitRois = inhibitRois[~NsExcluded] # Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)                       
  
  
    #%%
    # Load 1stSideTry-aligned traces, frames, frame of event of interest
    # use firstSideTryAl_COM to look at changes-of-mind (mouse made a side lick without committing it)
    Data = scio.loadmat(postName, variable_names=['firstSideTryAl'],squeeze_me=True,struct_as_record=False)
    traces_al_1stSide = Data['firstSideTryAl'].traces.astype('float')
    time_aligned_1stSide = Data['firstSideTryAl'].time.astype('float')
    eventI_ch = Data['firstSideTryAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
    # print(np.shape(traces_al_1stSide))
    
    # training epoch: 300ms before the choice is made.
    epSt = eventI_ch - np.round(300/frameLength) # the start point of the epoch relative to alignedEvent for training SVM. (500ms)
    epEn = eventI_ch-1 # the end point of the epoch relative to alignedEvent for training SVM. (700ms)
    ep_ch = np.arange(epSt, epEn+1).astype(int) # frames on stimAl.traces that will be used for trainning SVM.  
    
    # average frames during ep_ch (ie 300ms before choice onset)
    X_choice0 = np.transpose(np.nanmean(traces_al_1stSide[ep_ch,:,:], axis=0)) # trials x neurons 
    
    
    # determine trsExcluded for traces_al_1stSide
    # remember traces_al_1stSide can have some nan trials that are change-of-mind trials... and they wont be nan in choiceVec0
#    trsExcluded_chAl = (np.sum(np.isnan(X_choice0), axis = 1) + np.isnan(choiceVec0)) != 0 # NaN trials # trsExcluded
    
    
    # exclude trsExcluded_chAl
    X_chAl = X_choice0[~trsExcluded_chAl,:]; # trials x neurons
    print 'choice-aligned traces (trs x units): ', np.shape(X_chAl)
    
    # exclude NsExcluded
    X_chAl = X_chAl[:,~NsExcluded]
    print 'choice-aligned traces (trs x units): ', np.shape(X_chAl)

    # exclude unsure neurons (non exc, non inh)
    X_chAl = X_chAl[:, ~np.isnan(inhibitRois)]
    print 'choice-aligned traces (trs x units): ', np.shape(X_chAl)


    #%% Set inhRois_ei

    inhRois = inhibitRois[~np.isnan(inhibitRois)]            
        
    if allExc==0:
        n = sum(inhRois==1)                
        inhRois_ei = np.zeros((2*n)); # first half is exc; 2nd half is inh
        inhRois_ei[n:2*n] = 1;
    else:
        inhRois_ei = inhRois

            
    #%% Average across trials
    
    meanX_chAl = np.mean(X_chAl, axis = 0)
    
    if normX:
        stdX_chAl = np.std(X_chAl, axis = 0)    
        X_chAl = (X_chAl - meanX_chAl) / stdX_chAl
        meanX_chAl = np.mean(X_chAl, axis = 0)
        
    
    #%% Single trial-FR averaged in ep for exc and inh neurons, for each trial

    X_chAl_exc = X_chAl[:,inhRois_ei==0]    # exc 
    X_chAl_inh = X_chAl[:,inhRois_ei==1]    # inh
    print X_chAl_exc.shape
    print X_chAl_inh.shape
        
    
    ##%% Keep values of all days (pool all trials and all neurons)
    
    X_chAl_exc_all.append(X_chAl_exc.reshape((-1,)))
    X_chAl_inh_all.append(X_chAl_inh.reshape((-1,)))
    
    
    #%% Trial-averaged FR in X_chAl (ie in window ep) for exc and inh neurons
    
    meanX_chAl_exc = meanX_chAl[inhRois_ei==0]    # exc 
    print meanX_chAl_exc.shape
    
    meanX_chAl_inh = meanX_chAl[inhRois_ei==1]    # inh 
    print meanX_chAl_inh.shape

    
    ##%% Keep values of all days
    
    meanX_chAl_exc_all.append(meanX_chAl_exc)
    meanX_chAl_inh_all.append(meanX_chAl_inh)
    
    

####################################################################################################
####################################################################################################
#%% Plot histograms of FR (for exc,inh, single trial and trial averaged) for each day and average across days

lab1 = 'exc'
lab2 = 'inh'
colors = ['k','r']


#############################################################
################### Single-trial FR #######################
#############################################################
### hist and P val for all days pooled (exc vs inh)
lab = 'FR'

a = np.concatenate(X_chAl_exc_all) 
b = np.concatenate(X_chAl_inh_all)
#a = a[~(np.isnan(a) + np.isinf(a))]
#b = b[~(np.isnan(b) + np.isinf(b))]
print [len(a), len(b)]

_, p = stats.ttest_ind(a, b, nan_policy='omit')

mx = max(a.max(),b.max())
mn = min(a.min(),b.min())
#mx = .005
binEvery = (mx - mn)/100 #.001# .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)


plt.figure(figsize=(5,5))    
gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[0,0:2]
h2 = gs[0,2:3]
ax1,_ = histerrbar(a,b,binEvery,p,colors)
#plt.xlabel(lab)
if normX:
    ax1.set_xlim([-1, 2])
else:
    ax1.set_xlim([-.001, .05])
ymx = ax1.get_ylim()
if normX:
    ax1.set_ylim([-.01, ymx[1]])
else:
    ax1.set_ylim([-.05, ymx[1]])


############ show individual days
a = X_chAl_exc_all
b = X_chAl_inh_all
# if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
p = []
for i in range(len(a)):
    _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
    p.append(p0)
p = np.array(p)    
#ax = plt.subplot(gs[1,0:3])
errbarAllDays(a,b,p)
#plt.ylabel(lab)


if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnow,mousename)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
 
    fign = os.path.join(d, suffn[0:5]+'XchAl_singleTrFR'+dp+fmt[0])
    plt.savefig(fign, bbox_inches='tight')
    



#############################################################
################### Trial-averaged FR #######################
#############################################################
### hist and P val for all days pooled (exc vs inh)
lab = 'FR'

a = np.concatenate(meanX_chAl_exc_all) 
b = np.concatenate(meanX_chAl_inh_all)
#a = a[~(np.isnan(a) + np.isinf(a))]
#b = b[~(np.isnan(b) + np.isinf(b))]
print [len(a), len(b)]

_, p = stats.ttest_ind(a, b, nan_policy='omit')

mx = max(a.max(),b.max())
mn = min(a.min(),b.min())
#mx = .005
binEvery = (mx - mn)/100 #.001# .01 #mv = 2; bn = np.arange(0.05,mv,binEvery)


plt.figure(figsize=(5,5))    
gs = gridspec.GridSpec(2, 3)#, width_ratios=[2, 1]) 
h1 = gs[0,0:2]
h2 = gs[0,2:3]
ax1,_ = histerrbar(a,b,binEvery,p,colors)
#plt.xlabel(lab)
#ax1.set_xlim([mn, .023])


############ show individual days
a = meanX_chAl_exc_all
b = meanX_chAl_inh_all
# if no averaging across neuron or trial shuffles, pool neuron and trial shuffles
p = []
for i in range(len(a)):
    _,p0 = stats.ttest_ind(a[i], b[i], nan_policy='omit')
    p.append(p0)
p = np.array(p)    
#ax = plt.subplot(gs[1,0:3])
errbarAllDays(a,b,p)
#plt.ylabel(lab)


if savefigs:#% Save the figure
    d = os.path.join(svmdir+dnow,mousename)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
 
    fign = os.path.join(d, suffn[0:5]+'XchAl_aveTrFR'+dp+fmt[0])
    plt.savefig(fign, bbox_inches='tight')
    