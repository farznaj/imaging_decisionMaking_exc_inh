# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 03:13:36 2017

@author: farznaj
"""

# -*- coding: utf-8 -*-
"""
This script computes classification accuracy when no subselection of neurons was performed. (ie there are no "rounds" for each session.)
To compare results with subselection, run the class accuracy section of svm_notebook_plots.py to get class accur.
"""

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
dnow = '/classAccur/'+mousename+'/chAl_stAl/'


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
    
    
    
#%%
'''
#####################################################################################################################################################   
############################ choice-aligned, SVM (L2,corr,incorr) ###################################################################################################     
#####################################################################################################################################################
'''

#%% Loop over days    
# test in the names below means cv (cross validated or testing data!)    

classErr_bestC_test_data_all = np.full((numSamp, len(days)), np.nan)
classErr_bestC_test_chance_all = np.full((numSamp, len(days)), np.nan)
classErr_bestC_incorr_data_all = np.full((numSamp, len(days)), np.nan)
classErr_bestC_incorr_chance_all = np.full((numSamp, len(days)), np.nan)
#l2_wnon0_all = np.full((1, len(days)), np.nan).flatten()
#l2_b_all = np.full((1, len(days)), np.nan).flatten()

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
        
#    Data = scio.loadmat(svmName, variable_names=['cvect','cbest_chAl','perClassErrorTest_chAl','perClassErrorTest_chAl_chance','perClassErrorTest_incorr_chAl','perClassErrorTest_incorr_chAl_chance','wAllC_chAl','bAllC_chAl'])
    Data = scio.loadmat(svmName, variable_names=['cvect','cbest_chAl','perClassErrorTest_chAl','perClassErrorTest_chAl_chance','perClassErrorTest_incorr_chAl','perClassErrorTest_incorr_chAl_chance'])
    cvect = Data.pop('cvect')
    cbest_chAl = Data.pop('cbest_chAl')
    perClassErrorTest_chAl = Data.pop('perClassErrorTest_chAl')
    perClassErrorTest_chAl_chance = Data.pop('perClassErrorTest_chAl_chance')
    perClassErrorTest_incorr_chAl = Data.pop('perClassErrorTest_incorr_chAl')
    perClassErrorTest_incorr_chAl_chance = Data.pop('perClassErrorTest_incorr_chAl_chance')
#    wAllC_chAl = Data.pop('wAllC_chAl')
#    bAllC_chAl = Data.pop('bAllC_chAl')
        

    #%%
    indBestC = np.in1d(cvect, cbest_chAl)
    
#    w_bestc_data_chAl = wAllC_chAl[:,indBestC,:].squeeze() # numSamps x neurons
#    b_bestc_data_chAl = bAllC_chAl[:,indBestC]
    
    classErr_bestC_test_data_chAl = perClassErrorTest_chAl[:,indBestC].squeeze() #numSamps
    classErr_bestC_test_chance_chAl = perClassErrorTest_chAl_chance[:,indBestC].squeeze()
    classErr_bestC_incorr_data_chAl = perClassErrorTest_incorr_chAl[:,indBestC].squeeze()
    classErr_bestC_incorr_chance_chAl = perClassErrorTest_incorr_chAl_chance[:,indBestC].squeeze()
 
    # bc l2 was used, we dont need to worry about all weights being 0       
#    a = (abs(w_bestc_data_chAl)>eps) # non-zero weights
#    b = np.mean(a, axis=(1)) # Fraction of non-zero weights (averaged across shuffles)

    classErr_bestC_test_data_all[:, iday] = classErr_bestC_test_data_chAl
    classErr_bestC_test_chance_all[:, iday] = classErr_bestC_test_chance_chAl
    classErr_bestC_incorr_data_all[:, iday] = classErr_bestC_incorr_data_chAl
    classErr_bestC_incorr_chance_all[:, iday] = classErr_bestC_incorr_chance_chAl



#%% Average and std across shuffles

if doSVM:
    av_l2_test_d = 100-np.nanmean(classErr_bestC_test_data_all, axis=0) # numDays
    av_l2_test_s = 100-np.nanmean(classErr_bestC_test_chance_all, axis=0) 
    
    av_l2_incorr_d = 100-np.nanmean(classErr_bestC_incorr_data_all, axis=0) # numDays
    av_l2_incorr_s = 100-np.nanmean(classErr_bestC_incorr_chance_all, axis=0) 
else:
    av_l2_test_d = np.nanmean(classErr_bestC_test_data_all, axis=0) # numDays
    av_l2_test_s = np.nanmean(classErr_bestC_test_chance_all, axis=0) 
    
    av_l2_incorr_d = np.nanmean(classErr_bestC_incorr_data_all, axis=0) # numDays
    av_l2_incorr_s = np.nanmean(classErr_bestC_incorr_chance_all, axis=0) 


sd_l2_test_d = np.nanstd(classErr_bestC_test_data_all, axis=0) / np.sqrt(numSamp)  
sd_l2_test_s = np.nanstd(classErr_bestC_test_chance_all, axis=0) / np.sqrt(numSamp)  

sd_l2_incorr_d = np.nanstd(classErr_bestC_incorr_data_all, axis=0) / np.sqrt(numSamp)  
sd_l2_incorr_s = np.nanstd(classErr_bestC_incorr_chance_all, axis=0) / np.sqrt(numSamp)

#_,p = stats.ttest_ind(l1_err_test_data, l1_err_test_shfl, nan_policy = 'omit')
#p


######################## PLOTS ########################

#################################################################
#################### Testing data: corr, inorr #########################
#################################################################
#%% Plot average across samples for each day

# L2;
plt.figure(figsize=(6,2.5))
gs = gridspec.GridSpec(1,10)#, width_ratios=[2, 1]) 

ax = plt.subplot(gs[0:-5])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), av_l2_test_d, yerr = sd_l2_test_d, color='g', label='Correct')
plt.errorbar(range(numDays), av_l2_incorr_d, yerr = sd_l2_incorr_d, color='c', label='Incorrect')
plt.errorbar(range(numDays), av_l2_test_s, yerr = sd_l2_test_s, color='k', label='Shuffled')
plt.errorbar(range(numDays), av_l2_incorr_s, yerr = sd_l2_incorr_s, color=[.5,.5,.5])#, label='Shuffled')
plt.xlabel('Days', fontsize=13, labelpad=10)
if doSVM:
    plt.ylabel('Classification accuracy (%)\n(cross-validated)', fontsize=13, labelpad=10)
else:
    plt.ylabel('Prediction error \n(cross-validated)', fontsize=13, labelpad=10)
    plt.plot(range(numDays), np.ones((numDays)), color='k', ls=':') # chance line.... max error = 1
    
plt.xlim([-1, len(days)])
lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.45), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax)
ymin, ymax = ax.get_ylim()

##%% Average across days
ax = plt.subplot(gs[-5:-3])
#x =[0,1]
#labels = ['Data', 'Shfl']
#plt.errorbar(x, [np.mean(av_l2_test_d), np.mean(av_l2_test_s)], yerr = [np.std(av_l2_test_d), np.std(av_l2_test_s)], marker='o', fmt=' ', color='k')
x =[0,1,2]
labels = ['Corr', 'Incorr', 'Shfl']
plt.errorbar(x, [np.mean(av_l2_test_d), np.mean(av_l2_incorr_d), np.mean(np.concatenate((av_l2_test_s,av_l2_incorr_s)))], yerr = [np.std(av_l2_test_d), np.std(av_l2_incorr_d), np.std(np.concatenate((av_l2_test_s,av_l2_incorr_s)))], marker='o', fmt=' ', color='k')
if doSVM==0:
    plt.plot(range(numDays), np.ones((numDays)), color='k', ls=':')
    
plt.xlim([x[0]-1, x[-1]+1])
plt.ylim([ymin, ymax])
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical', fontsize=13)    
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
ax.axes.get_yaxis().set_visible(False)
#ax.yaxis.label.set_color('white')
ax.spines['left'].set_color('white')
#ax.set_axis_off()
plt.subplots_adjust(wspace=1)
makeNicePlots(ax)
    
if savefigs:#% Save the figure
    if doSVM:
        dd = 'chAl_svm'
    else:
        dd = 'chAl_svr'
        
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    



#%% stim-aligned traces
 
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################


#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName

def setSVMname(pnevFileName, trialHistAnalysis, doSVM, epAllStim):
    import glob

    if trialHistAnalysis:
#        svmn = 'svmPrevChoice_corrIncorr_chAl_*'
#        svmn = 'svrPrevChoice_corrIncorr_chAl_*'
        if doSVM:
            svmn = 'svmPrevChoice_corrIncorr_stAl_*'    
        else:
            if epAllStim:
                svmn = 'svrPrevChoice_corrIncorr_stAl_epAllStim_*'        
            else:        
                svmn = 'svrPrevChoice_corrIncorr_stAl_*'
        
    else:
#        svmn = 'svmCurrChoice_corrIncorr_chAl_*'
#        svmn = 'svrCurrChoice_corrIncorr_chAl_*'
        if doSVM:
            svmn = 'svmCurrChoice_corrIncorr_stAl_*'    
        else:
            if epAllStim:
                svmn = 'svrCurrChoice_corrIncorr_stAl_epAllStim_*'        
            else:        
                svmn = 'svrCurrChoice_corrIncorr_stAl_*'
    
    
    svmn = svmn + os.path.basename(pnevFileName) #pnevFileName[-32:]    
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
    svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    svmName = svmName[0] # get the latest file
    
    return svmName
    
    
    
#%%
'''
#####################################################################################################################################################   
############################ stimulus-aligned, SVR (L2,corr,incorr) ###################################################################################################     
#####################################################################################################################################################
'''

#%% Loop over days    
for epAllStim in epall:
    # test in the names below means cv (cross validated or testing data!)    
    
    classErr_bestC_test_data_all = np.full((numSamp, len(days)), np.nan)
    classErr_bestC_test_chance_all = np.full((numSamp, len(days)), np.nan)
    classErr_bestC_incorr_data_all = np.full((numSamp, len(days)), np.nan)
    classErr_bestC_incorr_chance_all = np.full((numSamp, len(days)), np.nan)
    #l2_wnon0_all = np.full((1, len(days)), np.nan).flatten()
    #l2_b_all = np.full((1, len(days)), np.nan).flatten()
    
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
        
        svmName = setSVMname(pnevFileName, trialHistAnalysis, doSVM, epAllStim) # latest is l1, then l2. we use [] to get both files. # after that you again ran analysis with cross validation shuffles (l1).
    
        print os.path.basename(svmName)    
            
            
        #%% Load vars
            
        Data = scio.loadmat(svmName, variable_names=['cvect','cbest','perClassErrorTest','perClassErrorTest_chance','perClassErrorTest_incorr','perClassErrorTest_incorr_chance'])
        cvect = Data.pop('cvect')
        cbest = Data.pop('cbest')
        perClassErrorTest = Data.pop('perClassErrorTest')
        perClassErrorTest_chance = Data.pop('perClassErrorTest_chance')
        perClassErrorTest_incorr = Data.pop('perClassErrorTest_incorr')
        perClassErrorTest_incorr_chance = Data.pop('perClassErrorTest_incorr_chance')
    #    wAllC = Data.pop('wAllC')
    #    bAllC = Data.pop('bAllC')
            
    
        #%%
        indBestC = np.in1d(cvect, cbest)
        
    #    w_bestc_data = wAllC[:,indBestC,:].squeeze() # numSamps x neurons
    #    b_bestc_data = bAllC[:,indBestC]
        
        classErr_bestC_test_data = perClassErrorTest[:,indBestC].squeeze() #numSamps
        classErr_bestC_test_chance = perClassErrorTest_chance[:,indBestC].squeeze()
        classErr_bestC_incorr_data = perClassErrorTest_incorr[:,indBestC].squeeze()
        classErr_bestC_incorr_chance = perClassErrorTest_incorr_chance[:,indBestC].squeeze()
     
        # bc l2 was used, we dont need to worry about all weights being 0       
    #    a = (abs(w_bestc_data)>eps) # non-zero weights
    #    b = np.mean(a, axis=(1)) # Fraction of non-zero weights (averaged across shuffles)
    
        classErr_bestC_test_data_all[:, iday] = classErr_bestC_test_data
        classErr_bestC_test_chance_all[:, iday] = classErr_bestC_test_chance
        classErr_bestC_incorr_data_all[:, iday] = classErr_bestC_incorr_data
        classErr_bestC_incorr_chance_all[:, iday] = classErr_bestC_incorr_chance
    
    
    
    #%% Average and std across shuffles
    
    if doSVM:
        av_l2_test_d = 100-np.nanmean(classErr_bestC_test_data_all, axis=0) # numDays
        av_l2_test_s = 100-np.nanmean(classErr_bestC_test_chance_all, axis=0) 
        
        av_l2_incorr_d = 100-np.nanmean(classErr_bestC_incorr_data_all, axis=0) # numDays
        av_l2_incorr_s = 100-np.nanmean(classErr_bestC_incorr_chance_all, axis=0) 
    else:
        av_l2_test_d = np.nanmean(classErr_bestC_test_data_all, axis=0) # numDays
        av_l2_test_s = np.nanmean(classErr_bestC_test_chance_all, axis=0) 
        
        av_l2_incorr_d = np.nanmean(classErr_bestC_incorr_data_all, axis=0) # numDays
        av_l2_incorr_s = np.nanmean(classErr_bestC_incorr_chance_all, axis=0) 
    
    
    sd_l2_test_d = np.nanstd(classErr_bestC_test_data_all, axis=0) / np.sqrt(numSamp)  
    sd_l2_test_s = np.nanstd(classErr_bestC_test_chance_all, axis=0) / np.sqrt(numSamp)  
    
    sd_l2_incorr_d = np.nanstd(classErr_bestC_incorr_data_all, axis=0) / np.sqrt(numSamp)  
    sd_l2_incorr_s = np.nanstd(classErr_bestC_incorr_chance_all, axis=0) / np.sqrt(numSamp)
    
    #_,p = stats.ttest_ind(l1_err_test_data, l1_err_test_shfl, nan_policy = 'omit')
    #p
    
    
    ######################## PLOTS ########################
    
    #################################################################
    #################### Testing data: corr, inorr #########################
    #################################################################
    #%% Plot average across shuffles for each day
    
    # L2;
    plt.figure(figsize=(6,2.5))
    gs = gridspec.GridSpec(1,10)#, width_ratios=[2, 1]) 
    
    ax = plt.subplot(gs[0:-5])
    #ax = plt.subplot(121)
    plt.errorbar(range(numDays), av_l2_test_d, yerr = sd_l2_test_d, color='g', label='Correct')
    plt.errorbar(range(numDays), av_l2_incorr_d, yerr = sd_l2_incorr_d, color='c', label='Incorrect')
    plt.errorbar(range(numDays), av_l2_test_s, yerr = sd_l2_test_s, color='k', label='Shuffled')
    plt.errorbar(range(numDays), av_l2_incorr_s, yerr = sd_l2_incorr_s, color=[.5,.5,.5])#, label='Shuffled')
    plt.xlabel('Days', fontsize=13, labelpad=10)
    if doSVM:
        plt.ylabel('Classification accuracy (%)\n(cross-validated)', fontsize=13, labelpad=10)
    else:
        plt.ylabel('Prediction error\n(cross-validated)', fontsize=13, labelpad=10)
        plt.plot(range(numDays), np.ones((numDays)), color='k', ls=':') # chance line.... max error = 1
        
    plt.xlim([-1, len(days)])
    lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.45), frameon=False)
    #leg.get_frame().set_linewidth(0.0)
    makeNicePlots(ax)
    ymin, ymax = ax.get_ylim()
#    plt.ylim([40,100])
    
    ##%% Average across days
    ax = plt.subplot(gs[-5:-3])
    #x =[0,1]
    #labels = ['Data', 'Shfl']
    #plt.errorbar(x, [np.mean(av_l2_test_d), np.mean(av_l2_test_s)], yerr = [np.std(av_l2_test_d), np.std(av_l2_test_s)], marker='o', fmt=' ', color='k')
    x =[0,1,2]
    labels = ['Corr', 'Incorr', 'Shfl']
    plt.errorbar(x, [np.mean(av_l2_test_d), np.mean(av_l2_incorr_d), np.mean(np.concatenate((av_l2_test_s,av_l2_incorr_s)))], yerr = [np.std(av_l2_test_d), np.std(av_l2_incorr_d), np.std(np.concatenate((av_l2_test_s,av_l2_incorr_s)))], marker='o', fmt=' ', color='k')
    if doSVM==0:
        plt.plot(range(numDays), np.ones((numDays)), color='k', ls=':')
        
    plt.xlim([x[0]-1, x[-1]+1])
    plt.ylim([ymin, ymax])
#    plt.ylim([40,100])
    #plt.ylabel('Classification error (%) - testing data')
    plt.xticks(x, labels, rotation='vertical', fontsize=13)    
    #plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
    ax.axes.get_yaxis().set_visible(False)
    #ax.yaxis.label.set_color('white')
    ax.spines['left'].set_color('white')
    #ax.set_axis_off()
    plt.subplots_adjust(wspace=1)
    makeNicePlots(ax)
        
    if savefigs:#% Save the figure
        if doSVM:
            dd = 'stAl_svm'
        else:
            if epAllStim:                
                dd = 'stAl_svr_epAllStim'
            else:                
                dd = 'stAl_svr'
            
        fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
        plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    

