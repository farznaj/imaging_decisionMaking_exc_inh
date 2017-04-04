# -*- coding: utf-8 -*-
"""
SVM trained on stim-aligned traces (averaged from stim onset to go tone, equal number of corr and incorr trials) 
to train stim category (the earlier stimCateg mat file was trained to decode choice using same traces; but the later mat file
is for decoding stim category). Here we compute classification accuracy.

Created on Sun Mar 12 15:12:29 2017
@author: farznaj
"""


mousename = 'fni17' #'fni17'

trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  
execfile("svm_plots_setVars.py")  

savefigs = 0
thNumRemCorr = 0 #20 # we need at least this number of remainingCorr trials for a session to include it into analysis
numSamp = 50 # 500 perClassErrorTest_shfl.shape[0]  # number of shuffles for doing cross validation (ie number of random sets of test/train trials.... You changed the following: in mainSVM_notebook.py this is set to 100, so unless you change the value inside the code it should be always 100.)
nRandCorrSel = 50
doPlots= 1 #1 # plot c path of each day 


#%% Set vars

dnow = '/classAccur/'+mousename+'/stimCategDecode/'
kfold = 10    


#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName

def setSVMname(pnevFileName, trialHistAnalysis):
    import glob

    if trialHistAnalysis:
        svmn = 'svmPrevChoice_stimCateg_stAl_epAllStim_*'
    else:
        svmn = 'svmCurrChoice_stimCateg_stAl_epAllStim_*'

    
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

    
#%% 
'''
#####################################################################################################################################################   
############################ stimulus-aligned, SVR (trained on trials of 1 choice to avoid variations in neural response due to animal's choice... we only want stimulus rate to be different) ###################################################################################################     
#####################################################################################################################################################
'''
cbestCV = 0
cbestRemCorr = 0
cbestBoth = 1

            
#%% Loop over days    

#for epAllStim in epall:
# test in the names below means cv (cross validated or testing data!)    

classErr_bestC_train_data_all = np.full((numSamp*nRandCorrSel, len(days)), np.nan)
classErr_bestC_test_data_all = np.full((numSamp*nRandCorrSel, len(days)), np.nan)
classErr_bestC_test_chance_all = np.full((numSamp*nRandCorrSel, len(days)), np.nan)
classErr_bestC_remCorr_data_all = np.full((numSamp*nRandCorrSel, len(days)), np.nan)
classErr_bestC_remCorr_chance_all = np.full((numSamp*nRandCorrSel, len(days)), np.nan)
classErr_bestC_both_data_all = np.full((numSamp*nRandCorrSel, len(days)), np.nan)
classErr_bestC_both_chance_all = np.full((numSamp*nRandCorrSel, len(days)), np.nan)
classErr_bestC_both_shfl_all = np.full((numSamp*nRandCorrSel, len(days)), np.nan)


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


    #%%            
    ######################################################################################################################################                        

    svmName = setSVMname(pnevFileName, trialHistAnalysis) # latest is l1, then l2. we use [] to get both files. # after that you again ran analysis with cross validation shuffles (l1).

#    if svmName=='':
#        print 'No svm file found for', days[iday], '... skipping this day!'
#    else:
#        print os.path.basename(svmName)    

    if len(svmName)<2:
        sys.exit('<2 mat files; needs work!')            
    else:
        svmName = svmName[0]
        print os.path.basename(svmName)    

            
        #%% Load vars
            
        # compute number of trials that went into cv and remCorr datasets.
        Data = scio.loadmat(svmName, variable_names=['trsExcluded'])
        trsExcluded = Data.pop('trsExcluded').astype('bool').squeeze()    
    
        Data = scio.loadmat(postName, variable_names=['outcomes']) # 'allResp_HR_LR'
        outcomes = (Data.pop('outcomes').astype('float'))[0,:]
        outcomes = outcomes[~trsExcluded]
    
    #    compute number of trials that went into cv and remCorr datasets.
        nc = (outcomes==1).sum()
        ni = (outcomes==0).sum()
        
        numRemCorr = nc - ni
        
        no = ni*2
        len_test = no - int((kfold-1.)/kfold*no)
        numCV = len_test
        
        print numCV, numRemCorr

        if numRemCorr >= thNumRemCorr:
            #######################################################################            
            Data = scio.loadmat(svmName, variable_names=['cvect','cbest','cbest_remCorr','cbest_both','perClassErrorTrain','perClassErrorTest','perClassErrorTest_chance','perClassErrorTestRemCorr','perClassErrorTestRemCorr_chance','perClassErrorTestBoth','perClassErrorTestBoth_chance','perClassErrorTestBoth_shfl'])
            cvect = Data.pop('cvect').squeeze()
            cbest = Data.pop('cbest').squeeze()
            cbest_remCorr = Data.pop('cbest_remCorr').squeeze()
            cbest_both = Data.pop('cbest_both').squeeze() 
        #    trsExcluded_svr = Data.pop('trsExcluded_svr').astype('bool').squeeze()
                    
            perClassErrorTrain = Data.pop('perClassErrorTrain') # numSamples x len(cvect) x nRandCorrSel
            perClassErrorTest = Data.pop('perClassErrorTest')
            perClassErrorTest_chance = Data.pop('perClassErrorTest_chance')
            perClassErrorTestRemCorr = Data.pop('perClassErrorTestRemCorr')
            perClassErrorTestRemCorr_chance = Data.pop('perClassErrorTestRemCorr_chance')
            perClassErrorTestBoth = Data.pop('perClassErrorTestBoth')
            perClassErrorTestBoth_chance = Data.pop('perClassErrorTestBoth_chance')
            perClassErrorTestBoth_shfl = Data.pop('perClassErrorTestBoth_shfl')
        
    #        Data = scio.loadmat(svmName, variable_names=['wAllC','bAllC'])
    #        wAllC = Data.pop('wAllC')
    #        bAllC = Data.pop('bAllC')
    #     
            #'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'trsTrainedTestedInds':trsTrainedTestedInds, 'trsRemCorrInds':trsRemCorrInds,
        #    Data = scio.loadmat(svmName, variable_names=['trsTrainedTestedInds','trsRemCorrInds'])
        #    trsTrainedTestedInds = Data.pop('trsTrainedTestedInds')
        #    trsRemCorrInds = Data.pop('trsRemCorrInds')
                    
                          
            #%% Set vars for C path: Compute average of class errors across numSamples
             
            numSamples = perClassErrorTest.shape[0]
            nRandCorrSel = perClassErrorTest.shape[2]
        
        
            #%% Compute average of class errors across numSamples and nRandCorrSel
            
            meanPerClassErrorTrain = np.mean(perClassErrorTrain, axis = (0,2));
            semPerClassErrorTrain = np.std(perClassErrorTrain, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);
            
            meanPerClassErrorTest = np.mean(perClassErrorTest, axis = (0,2));
            semPerClassErrorTest = np.std(perClassErrorTest, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);    
        #    meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl, axis = (0,2));
        #    semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);    
            meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance, axis = (0,2));
            semPerClassErrorTest_chance = np.std(perClassErrorTest_chance, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);
            
            meanPerClassErrorTestRemCorr = np.mean(perClassErrorTestRemCorr, axis = (0,2));
            semPerClassErrorTestRemCorr = np.std(perClassErrorTestRemCorr, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);    
        #    meanPerClassErrorTestRemCorr_shfl = np.mean(perClassErrorTestRemCorr_shfl, axis = (0,2));
        #    semPerClassErrorTestRemCorr_shfl = np.std(perClassErrorTestRemCorr_shfl, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);    
            meanPerClassErrorTestRemCorr_chance = np.mean(perClassErrorTestRemCorr_chance, axis = (0,2));
            semPerClassErrorTestRemCorr_chance = np.std(perClassErrorTestRemCorr_chance, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);
            
            meanPerClassErrorTestBoth = np.mean(perClassErrorTestBoth, axis = (0,2));
            semPerClassErrorTestBoth = np.std(perClassErrorTestBoth, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);    
            meanPerClassErrorTestBoth_shfl = np.mean(perClassErrorTestBoth_shfl, axis = (0,2));
            semPerClassErrorTestBoth_shfl = np.std(perClassErrorTestBoth_shfl, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);    
            meanPerClassErrorTestBoth_chance = np.mean(perClassErrorTestBoth_chance, axis = (0,2));
            semPerClassErrorTestBoth_chance = np.std(perClassErrorTestBoth_chance, axis = (0,2))/np.sqrt(numSamples+nRandCorrSel);
            
        
            #%% Pick best c for choice 0 decoders                
        
            if cbestCV: # find bestc on the same choice testing dataset               
                indBestC = np.in1d(cvect, cbest)
                
            elif cbestRemCorr: # find bestc on other choice testing dataset                               
                indBestC = np.in1d(cvect, cbest_remCorr)
                
            elif cbestBoth: # Allow bestc to take different values for each testing dataset                             
                indBestC = np.in1d(cvect, cbest_both)
            
           
        #    w_bestc_data = wAllC[:,indBestC,:,:].squeeze() # numSamps x neurons
        #    b_bestc_data = bAllC[:,indBestC,:]
        
            # Normalize wAllC
        #    nw = np.linalg.norm(w_bestc_data, axis=1) # numSamps x len(cvect); 2-norm of weights 
        #    wAllC_n = (w_bestc_data.T/nw).T    
        
            
            classErr_bestC_train_data = perClassErrorTrain[:,indBestC,:].squeeze() # numSamples x nRandCorrSel     
            classErr_bestC_test_data = perClassErrorTest[:,indBestC,:].squeeze()
            classErr_bestC_test_chance = perClassErrorTest_chance[:,indBestC,:].squeeze()    
        #    classErr_bestC_test_shfl = perClassErrorTest_shfl[:,indBestC,:].squeeze()        
            classErr_bestC_remCorr_data = perClassErrorTestRemCorr[:,indBestC,:].squeeze()
            classErr_bestC_remCorr_chance = perClassErrorTestRemCorr_chance[:,indBestC,:].squeeze()
        #    classErr_bestC_remCorr_shfl = perClassErrorTestRemCorr_shfl[:,indBestC,:].squeeze()        
            classErr_bestC_both_data = perClassErrorTestBoth[:,indBestC,:].squeeze()
            classErr_bestC_both_chance = perClassErrorTestBoth_chance[:,indBestC,:].squeeze()        
            classErr_bestC_both_shfl = perClassErrorTestBoth_shfl[:,indBestC,:].squeeze()        
        
            classErr_bestC_train_data_all[:, iday] = classErr_bestC_train_data.reshape((-1,)) # pool numSamples and nRandCorrSel
            classErr_bestC_test_data_all[:, iday] = classErr_bestC_test_data.reshape((-1,))
            classErr_bestC_test_chance_all[:, iday] = classErr_bestC_test_chance.reshape((-1,))
            classErr_bestC_remCorr_data_all[:, iday] = classErr_bestC_remCorr_data.reshape((-1,))
            classErr_bestC_remCorr_chance_all[:, iday] = classErr_bestC_remCorr_chance.reshape((-1,))
            classErr_bestC_both_data_all[:, iday] = classErr_bestC_both_data.reshape((-1,))
            classErr_bestC_both_chance_all[:, iday] = classErr_bestC_both_chance.reshape((-1,))        
            classErr_bestC_both_shfl_all[:, iday] = classErr_bestC_both_shfl.reshape((-1,))    
            
        #    del trsExcluded_svr,trsExcluded_svr_ch1,trsExcluded_incorr,perClassErrorTrain,perClassErrorTest,perClassErrorTest_chance,perClassErrorTest_incorr,perClassErrorTest_incorr_chance,perClassErrorTest_otherChoice, perClassErrorTest_otherChoice_chance        
                
          
            #%% plot C path
                   
            if doPlots:
        #        print 'Best c (inverse of regularization parameter) = %.2f' %cbest
                plt.figure()
                plt.subplot(1,2,1)
                plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
                plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
                plt.fill_between(cvect, meanPerClassErrorTestRemCorr-semPerClassErrorTestRemCorr, meanPerClassErrorTestRemCorr+ semPerClassErrorTestRemCorr, alpha=0.5, edgecolor='g', facecolor='g')        
                plt.fill_between(cvect, meanPerClassErrorTestBoth-semPerClassErrorTestBoth, meanPerClassErrorTestBoth+ semPerClassErrorTestBoth, alpha=0.5, edgecolor='m', facecolor='m')        
            
            #    plt.fill_between(cvect, meanPerClassErrorTestRemCorr_chance-semPerClassErrorTestRemCorr_chance, meanPerClassErrorTestRemCorr_chance+ semPerClassErrorTestRemCorr_chance, alpha=0.5, edgecolor='m', facecolor='m')        
            #    plt.fill_between(cvect, meanPerClassErrorTestRemCorr_shfl-semPerClassErrorTestRemCorr_shfl, meanPerClassErrorTestRemCorr_shfl+ semPerClassErrorTestRemCorr_shfl, alpha=0.5, edgecolor='c', facecolor='c')        
            #    plt.fill_between(cvect, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
            #    plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
                
                plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
                plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation: %d' %(numCV))
                plt.plot(cvect, meanPerClassErrorTestRemCorr, 'g', label = 'rem corr: %d' %(numRemCorr))        
                plt.plot(cvect, meanPerClassErrorTestBoth, 'm', label = 'cv+remCorr')        
            
            #    plt.plot(cvect, meanPerClassErrorTestRemCorr_chance, 'm', label = 'remCorr-chance')            
            #    plt.plot(cvect, meanPerClassErrorTestRemCorr_shfl, 'c', label = 'incorr-shfl')            
            #    plt.plot(cvect, meanPerClassErrorTest_chance, 'b', label = 'cv-chance')       
            #    plt.plot(cvect, meanPerClassErrorTest_shfl, 'y', label = 'corr-shfl')            
                plt.plot(cvect, meanPerClassErrorTestBoth_chance, 'b', label = 'cv+remCorr-chance')       
            
                plt.plot(cvect[cvect==cbest], meanPerClassErrorTest[cvect==cbest], 'bo')
                plt.plot(cvect[cvect==cbest_remCorr], meanPerClassErrorTestRemCorr[cvect==cbest_remCorr], 'bo')
                plt.plot(cvect[cvect==cbest_both], meanPerClassErrorTestBoth[cvect==cbest_both], 'bo')
             
                plt.xlim([cvect[1], cvect[-1]])
                plt.xscale('log')
                plt.xlabel('c (inverse of regularization parameter)')
                plt.ylabel('classification error (%)')
                plt.legend(loc='center left', bbox_to_anchor=(1, .7))
                plt.tight_layout()
    

      
######################################################################################################################################################    
######################################################################################################################################################          
#%%

#%% Average and std across shuffles

av_l2_test_d = 100-np.nanmean(classErr_bestC_test_data_all, axis=0) # numDays
av_l2_test_s = 100-np.nanmean(classErr_bestC_test_chance_all, axis=0) 

av_l2_remCorr_d = 100-np.nanmean(classErr_bestC_remCorr_data_all, axis=0) # numDays
av_l2_remCorr_s = 100-np.nanmean(classErr_bestC_remCorr_chance_all, axis=0) 

av_l2_both_d = 100-np.nanmean(classErr_bestC_both_data_all, axis=0) # numDays
av_l2_both_s = 100-np.nanmean(classErr_bestC_both_chance_all, axis=0) 
av_l2_both_sh = 100-np.nanmean(classErr_bestC_both_shfl_all, axis=0) 


sd_l2_test_d = np.nanstd(classErr_bestC_test_data_all, axis=0) / np.sqrt(numSamp)  
sd_l2_test_s = np.nanstd(classErr_bestC_test_chance_all, axis=0) / np.sqrt(numSamp)  

sd_l2_remCorr_d = np.nanstd(classErr_bestC_remCorr_data_all, axis=0) / np.sqrt(numSamp)  
sd_l2_remCorr_s = np.nanstd(classErr_bestC_remCorr_chance_all, axis=0) / np.sqrt(numSamp)

sd_l2_both_d = np.nanstd(classErr_bestC_both_data_all, axis=0) / np.sqrt(numSamp)  
sd_l2_both_s = np.nanstd(classErr_bestC_both_chance_all, axis=0) / np.sqrt(numSamp)
sd_l2_both_sh = np.nanstd(classErr_bestC_both_shfl_all, axis=0) / np.sqrt(numSamp)

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
#plt.errorbar(range(numDays), av_l2_test_d, yerr = sd_l2_test_d, color='g', label='CV')
#plt.errorbar(range(numDays), av_l2_remCorr_d, yerr = sd_l2_remCorr_d, color='c', label='remCorr')
plt.errorbar(range(numDays), av_l2_both_d, yerr = sd_l2_both_d, color='c', label='Data')
plt.errorbar(range(numDays), av_l2_both_s, yerr = sd_l2_both_s, color='k', label='Chance')
plt.errorbar(range(numDays), av_l2_both_sh, yerr = sd_l2_both_sh, color=[.5,.5,.5], label='Shuffled')
plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('Classification accuracy (%)\n(cross-validated)', fontsize=13, labelpad=10)   
plt.xlim([-1, len(days)])
lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.45), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax)
ymin, ymax = ax.get_ylim()

##%% Average across days
ax = plt.subplot(gs[-5:-3])
x =[0,1]
labels = ['Data', 'Shfl']
plt.errorbar(x, [np.nanmean(av_l2_test_d), np.nanmean(av_l2_test_s)], yerr = [np.nanstd(av_l2_test_d), np.nanstd(av_l2_test_s)], marker='o', fmt=' ', color='k')
#x =[0,1,2]
#labels = ['Corr', 'Incorr', 'Shfl']
#plt.errorbar(x, [np.mean(av_l2_test_d), np.mean(av_l2_both_d), np.mean(np.concatenate((av_l2_test_s,av_l2_both_s)))], yerr = [np.std(av_l2_test_d), np.std(av_l2_both_d), np.std(np.concatenate((av_l2_test_s,av_l2_both_s)))], marker='o', fmt=' ', color='k')
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
    dd = 'stAl_svm'
    d = os.path.join(svmdir+dnow)
    if not os.path.exists(d):
        print 'creating folder'
        os.makedirs(d)
            
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    

    