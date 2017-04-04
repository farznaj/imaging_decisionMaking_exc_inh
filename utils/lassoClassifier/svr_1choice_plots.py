# -*- coding: utf-8 -*-
"""
Compute class accuracy for stim-aligned traces, SVR decoder trained to decode stimulus using only trials of one of the choices (to avoid correlations with choice decoder)

Created on Sun Feb 19 03:13:36 2017
@author: farznaj
"""


mousename = 'fni16' #'fni17'
trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  

execfile("svm_plots_setVars.py")  
savefigs = 1



#%% Set vars

doPlots= 1 # plot c path of each day 
numSamp = 100 # 500 perClassErrorTest_shfl.shape[0]  # number of shuffles for doing cross validation (ie number of random sets of test/train trials.... You changed the following: in mainSVM_notebook.py this is set to 100, so unless you change the value inside the code it should be always 100.)
dnow = '/classAccur/'+mousename+'/chAl_stAl/'


#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName

def setSVMname(pnevFileName, trialHistAnalysis):
    import glob

    if trialHistAnalysis:
        svmn = 'svrPrevChoice_choice*Trained_stAl_epAllStim_*'
    else:
        svmn = 'svrCurrChoice_choice*Trained_stAl_epAllStim_*'   
    
    svmn = svmn + os.path.basename(pnevFileName) #pnevFileName[-32:]    
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
    svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
#    svmName = svmName[0] # get the latest file
    
    return svmName
    
    
    
#%% 
'''
#####################################################################################################################################################   
############################ stimulus-aligned, SVR (trained on trials of 1 choice to avoid variations in neural response due to animal's choice... we only want stimulus rate to be different) ###################################################################################################     
#####################################################################################################################################################
'''

cbestSameChoice = 0 # find bestc on the same choice testing dataset               
cbestOtherChoice = 0 # find bestc on other choice testing dataset                               
cbestAny = 1 # Allow bestc to take different values for each testing dataset   

            
#%% Loop over days    

#for epAllStim in epall:
# test in the names below means cv (cross validated or testing data!)    


classErr_bestC_train_data_all_ch0 = np.full((numSamp, len(days)), np.nan)
classErr_bestC_test_data_all_ch0 = np.full((numSamp, len(days)), np.nan)
classErr_bestC_incorr_data_all_ch0 = np.full((numSamp, len(days)), np.nan)
classErr_bestC_otherChoice_data_all_ch0 = np.full((numSamp, len(days)), np.nan)

classErr_bestC_train_data_all_ch1 = np.full((numSamp, len(days)), np.nan)
classErr_bestC_test_data_all_ch1 = np.full((numSamp, len(days)), np.nan)
classErr_bestC_incorr_data_all_ch1 = np.full((numSamp, len(days)), np.nan)
classErr_bestC_otherChoice_data_all_ch1 = np.full((numSamp, len(days)), np.nan)

AveAngDecodTrainedOnCh01 = np.full((len(days)), np.nan)

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


#%%            
######################################################################################################################################                        
######################################################################################################################################                        
######################################################################################################################################                        
################################### Analyze decoders trained on choice 0 (LR) ###################################################################            
######################################################################################################################################                        
######################################################################################################################################                        

    ch0Analys = 1 # if 1, analyze choice 0, if 0 analyze choice 1
    svmFound_ch0 = 1 # default is that there exists a mat file.
    svmName = setSVMname(pnevFileName, trialHistAnalysis) # latest is l1, then l2. we use [] to get both files. # after that you again ran analysis with cross validation shuffles (l1).

    if len(svmName)==4: # the 1st 2 files were ran with minTrPerChoice=1, and earliestGoTone=0 by mistake (The later 2 files were ran with minTrPerChoice=15, earliestGoTone=50)
        # remove the 2 earlier files
        svmName.pop(-1)
        svmName.pop(-1)
        svmNameAll = svmName

        if ch0Analys==1: # analyze choice1 or choice 0 data (ie svm trained on what choice)
            svmName = svmNameAll[1]  # choice 0: 2nd mat file
        else:
            svmName = svmNameAll[0]  # choice 1: 1st mat file
        
    else: # this happens because some days did not have enough number of trials per choice (you set the threshold to 15 for min number of trials per choice for the analysis to be run)
        if len(svmName)==0:
            print 'no mat file exists for day %s' %days[iday]
            svmFound_ch0 = 0
            
        elif len(svmName)==1: # there is 1 mat file... figure out for choice 0 or choice 1
            if ch0Analys==1:  # analyze choice 0
                if str.find(svmName[0], 'choice0Trained')>-1:
                    svmName = svmName[0]
                else:
                    print 'no ch0 mat file exists for day %s' %days[iday]
                    svmFound_ch0 = 0
            else: # analyze choice 1
                if str.find(svmName[0], 'choice1Trained')>-1:
                    svmName = svmName[0]
                else:
                    print 'no ch1 mat file exists for day %s' %days[iday]
                    svmFound_ch0 = 0
        else:
            sys.exit('<4 mat files; needs work!')
    
    if svmFound_ch0==1:                         
        print os.path.basename(svmName)    
        
        
    #%% Load vars
        
    if svmFound_ch0==1:                
        Data = scio.loadmat(svmName, variable_names=['cvect','cbest','cbest_incorr','cbest_otherChoice','trsExcluded_svr_ch0','trsExcluded_svr_ch1','trsExcluded_incorr','perClassErrorTrain','perClassErrorTest','perClassErrorTest_chance','perClassErrorTest_incorr','perClassErrorTest_incorr_chance','perClassErrorTest_otherChoice', 'perClassErrorTest_otherChoice_chance'])
        cvect_ch0 = Data.pop('cvect').squeeze()
        cbest_ch0 = Data.pop('cbest').squeeze()
        cbest_incorr_ch0 = Data.pop('cbest_incorr').squeeze()
        cbest_otherChoice_ch0 = Data.pop('cbest_otherChoice').squeeze() 

        trsExcluded_svr_ch0 = Data.pop('trsExcluded_svr_ch0').astype('bool').squeeze()
        trsExcluded_svr_ch1 = Data.pop('trsExcluded_svr_ch1').astype('bool').squeeze()
        trsExcluded_incorr = Data.pop('trsExcluded_incorr').astype('bool').squeeze()        
                
        perClassErrorTrain = Data.pop('perClassErrorTrain')
        perClassErrorTest = Data.pop('perClassErrorTest')
        perClassErrorTest_chance = Data.pop('perClassErrorTest_chance')
        perClassErrorTest_incorr = Data.pop('perClassErrorTest_incorr')
        perClassErrorTest_incorr_chance = Data.pop('perClassErrorTest_incorr_chance')
        perClassErrorTest_otherChoice = Data.pop('perClassErrorTest_otherChoice')
        perClassErrorTest_otherChoice_chance = Data.pop('perClassErrorTest_otherChoice_chance')

        Data = scio.loadmat(svmName, variable_names=['wAllC','bAllC'])
        wAllC = Data.pop('wAllC')
        bAllC = Data.pop('bAllC')
 
          
        #%% C path: Compute average of class errors across numSamples
         
        numSamples = perClassErrorTest.shape[0]
         
        meanPerClassErrorTrain_ch0 = np.mean(perClassErrorTrain, axis = 0);
        semPerClassErrorTrain_ch0 = np.std(perClassErrorTrain, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_ch0 = np.mean(perClassErrorTest, axis = 0);
        semPerClassErrorTest_ch0 = np.std(perClassErrorTest, axis = 0)/np.sqrt(numSamples);
        meanPerClassErrorTest_chance_ch0 = np.mean(perClassErrorTest_chance, axis = 0);
        semPerClassErrorTest_chance_ch0 = np.std(perClassErrorTest_chance, axis = 0)/np.sqrt(numSamples);       
        
        meanPerClassErrorTest_incorr_ch0 = np.mean(perClassErrorTest_incorr, axis = 0);
        semPerClassErrorTest_incorr_ch0 = np.std(perClassErrorTest_incorr, axis = 0)/np.sqrt(numSamples);        
        meanPerClassErrorTest_incorr_chance_ch0 = np.mean(perClassErrorTest_incorr_chance, axis = 0);
        semPerClassErrorTest_incorr_chance_ch0 = np.std(perClassErrorTest_incorr_chance, axis = 0)/np.sqrt(numSamples);
               
        
        meanPerClassErrorTest_otherChoice_ch0 = np.mean(perClassErrorTest_otherChoice, axis = 0);
        semPerClassErrorTest_otherChoice_ch0 = np.std(perClassErrorTest_otherChoice, axis = 0)/np.sqrt(numSamples);
        meanPerClassErrorTest_otherChoice_chance_ch0 = np.mean(perClassErrorTest_otherChoice_chance, axis = 0);
        semPerClassErrorTest_otherChoice_chance_ch0 = np.std(perClassErrorTest_otherChoice_chance, axis = 0)/np.sqrt(numSamples);            
        

        #%% Pick best c for choice 0 decoders                

        if cbestSameChoice: # find bestc on the same choice testing dataset               
            indBestC = np.in1d(cvect_ch0, cbest_ch0)
            indBestC_incorr = indBestC
            indBestC_otherCh = indBestC
            
        elif cbestOtherChoice: # find bestc on other choice testing dataset                               
            indBestC = np.in1d(cvect_ch0, cbest_otherChoice_ch0)
            indBestC_incorr = indBestC
            indBestC_otherCh = indBestC
            
        elif cbestAny: # Allow bestc to take different values for each testing dataset                             
            indBestC = np.in1d(cvect_ch0, cbest_ch0)
            indBestC_incorr = np.in1d(cvect_ch0, cbest_incorr_ch0)
            indBestC_otherCh = np.in1d(cvect_ch0, cbest_otherChoice_ch0)
        
        w_bestc_data_ch0 = wAllC[:,indBestC,:].squeeze() # numSamps x neurons
    #    b_bestc_data = bAllC[:,indBestC]
        # Normalize wAllC
        nw = np.linalg.norm(w_bestc_data_ch0, axis=1) # numSamps x len(cvect); 2-norm of weights 
        wAllC_n_ch0 = (w_bestc_data_ch0.T/nw).T
        
        
        classErr_bestC_train_data = perClassErrorTrain[:,indBestC].squeeze() #numSamps # same indBestC as the same choice indBestC
        classErr_bestC_test_data = perClassErrorTest[:,indBestC].squeeze() #numSamps
#        classErr_bestC_test_chance = perClassErrorTest_chance[:,indBestC].squeeze()
        classErr_bestC_incorr_data = perClassErrorTest_incorr[:,indBestC_incorr].squeeze()
#        classErr_bestC_incorr_chance = perClassErrorTest_incorr_chance[:,indBestC].squeeze()
        classErr_bestC_otherChoice_data = perClassErrorTest_otherChoice[:,indBestC_otherCh].squeeze()
        
        # bc l2 was used, we dont need to worry about all weights being 0       
    #    a = (abs(w_bestc_data)>eps) # non-zero weights
    #    b = np.mean(a, axis=(1)) # Fraction of non-zero weights (averaged across shuffles)
    
        classErr_bestC_train_data_all_ch0[:, iday] = classErr_bestC_train_data
        classErr_bestC_test_data_all_ch0[:, iday] = classErr_bestC_test_data
#        classErr_bestC_test_chance_all[:, iday] = classErr_bestC_test_chance
        classErr_bestC_incorr_data_all_ch0[:, iday] = classErr_bestC_incorr_data
#        classErr_bestC_incorr_chance_all[:, iday] = classErr_bestC_incorr_chance
        classErr_bestC_otherChoice_data_all_ch0[:, iday] = classErr_bestC_otherChoice_data
        
        
        del trsExcluded_svr_ch0,trsExcluded_svr_ch1,trsExcluded_incorr,perClassErrorTrain,perClassErrorTest,perClassErrorTest_chance,perClassErrorTest_incorr,perClassErrorTest_incorr_chance,perClassErrorTest_otherChoice, perClassErrorTest_otherChoice_chance        
            
            
#%%            
######################################################################################################################################                        
######################################################################################################################################                        
######################################################################################################################################                        
################################### Analyze decoders trained on choice 1 (HR) ###################################################################            
######################################################################################################################################                        
######################################################################################################################################                        

    ch0Analys = 0 # if 0, analyze choice 1     
    svmFound_ch1 = 1     
    svmName = setSVMname(pnevFileName, trialHistAnalysis) # latest is l1, then l2. we use [] to get both files. # after that you again ran analysis with cross validation shuffles (l1).

    if len(svmName)==4: # the 1st 2 files were ran with minTrPerChoice=1, and earliestGoTone=0 by mistake (The later 2 files were ran with minTrPerChoice=15, earliestGoTone=50)
        # remove the 2 earlier files
        svmName.pop(-1)
        svmName.pop(-1)
        svmNameAll = svmName

        if ch0Analys==1: # analyze choice1 or choice 0 data (ie svm trained on what choice)
            svmName = svmNameAll[1]  # choice 0: 2nd mat file
        else:
            svmName = svmNameAll[0]  # choice 1: 1st mat file
        
    else:
        if len(svmName)==0:
            print 'no mat file exists for day %s' %days[iday]
            svmFound_ch1 = 0
            
        elif len(svmName)==1: # there is 1 mat file... figure out for choice 0 or choice 1
            if ch0Analys==1:  # analyze choice 0
                if str.find(svmName[0], 'choice0Trained')>-1:
                    svmName = svmName[0]
                else:
                    print 'no ch0 mat file exists for day %s' %days[iday]
                    svmFound_ch1 = 0
            else: # analyze choice 1
                if str.find(svmName[0], 'choice1Trained')>-1:
                    svmName = svmName[0]
                else:
                    print 'no ch1 mat file exists for day %s' %days[iday]
                    svmFound_ch1 = 0
        else:
            sys.exit('<4 mat files; needs work!')
    
    if svmFound_ch1==1:                         
        print os.path.basename(svmName)    
        
        
    #%% Load vars
        
    if svmFound_ch1==1:                
        Data = scio.loadmat(svmName, variable_names=['cvect','cbest','cbest_incorr','cbest_otherChoice','trsExcluded_svr_ch0','trsExcluded_svr_ch1','trsExcluded_incorr','perClassErrorTrain','perClassErrorTest','perClassErrorTest_chance','perClassErrorTest_incorr','perClassErrorTest_incorr_chance','perClassErrorTest_otherChoice', 'perClassErrorTest_otherChoice_chance'])
        cvect_ch1 = Data.pop('cvect').squeeze()
        cbest_ch1 = Data.pop('cbest').squeeze()
        cbest_incorr_ch1 = Data.pop('cbest_incorr').squeeze()
        cbest_otherChoice_ch1 = Data.pop('cbest_otherChoice').squeeze() 

        trsExcluded_svr_ch0 = Data.pop('trsExcluded_svr_ch0').astype('bool').squeeze()
        trsExcluded_svr_ch1 = Data.pop('trsExcluded_svr_ch1').astype('bool').squeeze()
        trsExcluded_incorr = Data.pop('trsExcluded_incorr').astype('bool').squeeze()        
                
        perClassErrorTrain = Data.pop('perClassErrorTrain')
        perClassErrorTest = Data.pop('perClassErrorTest')
        perClassErrorTest_chance = Data.pop('perClassErrorTest_chance')
        perClassErrorTest_incorr = Data.pop('perClassErrorTest_incorr')
        perClassErrorTest_incorr_chance = Data.pop('perClassErrorTest_incorr_chance')
        perClassErrorTest_otherChoice = Data.pop('perClassErrorTest_otherChoice')
        perClassErrorTest_otherChoice_chance = Data.pop('perClassErrorTest_otherChoice_chance')
        
        Data = scio.loadmat(svmName, variable_names=['wAllC','bAllC'])
        wAllC = Data.pop('wAllC')
        bAllC = Data.pop('bAllC')

     
        #%% C path: Compute average of class errors across numSamples
         
        numSamples = perClassErrorTest.shape[0]
         
        meanPerClassErrorTrain_ch1 = np.mean(perClassErrorTrain, axis = 0);
        semPerClassErrorTrain_ch1 = np.std(perClassErrorTrain, axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_ch1 = np.mean(perClassErrorTest, axis = 0);
        semPerClassErrorTest_ch1 = np.std(perClassErrorTest, axis = 0)/np.sqrt(numSamples);
        meanPerClassErrorTest_chance_ch1 = np.mean(perClassErrorTest_chance, axis = 0);
        semPerClassErrorTest_chance_ch1 = np.std(perClassErrorTest_chance, axis = 0)/np.sqrt(numSamples);       
        
        meanPerClassErrorTest_incorr_ch1 = np.mean(perClassErrorTest_incorr, axis = 0);
        semPerClassErrorTest_incorr_ch1 = np.std(perClassErrorTest_incorr, axis = 0)/np.sqrt(numSamples);        
        meanPerClassErrorTest_incorr_chance_ch1 = np.mean(perClassErrorTest_incorr_chance, axis = 0);
        semPerClassErrorTest_incorr_chance_ch1 = np.std(perClassErrorTest_incorr_chance, axis = 0)/np.sqrt(numSamples);
               
        
        meanPerClassErrorTest_otherChoice_ch1 = np.mean(perClassErrorTest_otherChoice, axis = 0);
        semPerClassErrorTest_otherChoice_ch1 = np.std(perClassErrorTest_otherChoice, axis = 0)/np.sqrt(numSamples);
        meanPerClassErrorTest_otherChoice_chance_ch1 = np.mean(perClassErrorTest_otherChoice_chance, axis = 0);
        semPerClassErrorTest_otherChoice_chance_ch1 = np.std(perClassErrorTest_otherChoice_chance, axis = 0)/np.sqrt(numSamples);            
            
            
        #%% Pick best c for choice 1 decoders                
                    
        if cbestSameChoice: # find bestc on the same choice testing dataset               
            indBestC = np.in1d(cvect_ch1, cbest_ch1)
            indBestC_incorr = indBestC
            indBestC_otherCh = indBestC
            
        elif cbestOtherChoice: # find bestc on other choice testing dataset                               
            indBestC = np.in1d(cvect_ch1, cbest_otherChoice_ch1)
            indBestC_incorr = indBestC
            indBestC_otherCh = indBestC
            
        elif cbestAny: # Allow bestc to take different values for each testing dataset                             
            indBestC = np.in1d(cvect_ch1, cbest_ch1)
            indBestC_incorr = np.in1d(cvect_ch1, cbest_incorr_ch1)
            indBestC_otherCh = np.in1d(cvect_ch1, cbest_otherChoice_ch1)
            
            
        w_bestc_data_ch1 = wAllC[:,indBestC,:].squeeze() # numSamps x neurons
    #    b_bestc_data = bAllC[:,indBestC]
        # Normalize wAllC
        nw = np.linalg.norm(w_bestc_data_ch1, axis=1) # numSamps x len(cvect); 2-norm of weights 
        wAllC_n_ch1 = (w_bestc_data_ch1.T/nw).T
        
        
        classErr_bestC_train_data = perClassErrorTrain[:,indBestC].squeeze() #numSamps
        classErr_bestC_test_data = perClassErrorTest[:,indBestC].squeeze() #numSamps
#        classErr_bestC_test_chance = perClassErrorTest_chance[:,indBestC].squeeze()
        classErr_bestC_incorr_data = perClassErrorTest_incorr[:,indBestC_incorr].squeeze()
#        classErr_bestC_incorr_chance = perClassErrorTest_incorr_chance[:,indBestC].squeeze()
        classErr_bestC_otherChoice_data = perClassErrorTest_otherChoice[:,indBestC_otherCh].squeeze()
        
        # bc l2 was used, we dont need to worry about all weights being 0       
    #    a = (abs(w_bestc_data)>eps) # non-zero weights
    #    b = np.mean(a, axis=(1)) # Fraction of non-zero weights (averaged across shuffles)
    
        classErr_bestC_train_data_all_ch1[:, iday] = classErr_bestC_train_data
        classErr_bestC_test_data_all_ch1[:, iday] = classErr_bestC_test_data
#        classErr_bestC_test_chance_all[:, iday] = classErr_bestC_test_chance
        classErr_bestC_incorr_data_all_ch1[:, iday] = classErr_bestC_incorr_data
#        classErr_bestC_incorr_chance_all[:, iday] = classErr_bestC_incorr_chance
        classErr_bestC_otherChoice_data_all_ch1[:, iday] = classErr_bestC_otherChoice_data
        

        del trsExcluded_svr_ch0,trsExcluded_svr_ch1,trsExcluded_incorr,perClassErrorTrain,perClassErrorTest,perClassErrorTest_chance,perClassErrorTest_incorr,perClassErrorTest_incorr_chance,perClassErrorTest_otherChoice, perClassErrorTest_otherChoice_chance        
    
    
    #%%       
    ############################################################################################################################################
    ############################################################################################################################################
    ############################### plot C path for choice 0 and choice 1 trained decoders #########################################################    
    ############################################################################################################################################
    ############################################################################################################################################

    if doPlots:
#            print ' Number of trials: %d choice0; %d choice1; %d incorr \n Best c: CV = %.2g; otherChoice = %.2g; incorr = %.2g' %(sum(~trsExcluded_svr_ch0), sum(~trsExcluded_svr_ch1), sum(~trsExcluded_incorr), cbest, cbest_otherChoice, cbest_incorr)
#            print 'Best c: CV = %.2g; otherChoice = %.2g; incorr = %.2g' %(cbest, cbest_otherChoice, cbest_incorr)
        plt.figure()
        
        ######################################## choice 0 ########################################
        if svmFound_ch0==1:
            plt.subplot(1,4,(1,2))
            plt.fill_between(cvect_ch0, meanPerClassErrorTrain_ch0-semPerClassErrorTrain_ch0, meanPerClassErrorTrain_ch0+ semPerClassErrorTrain_ch0, alpha=0.5, edgecolor='k', facecolor='k')
            plt.fill_between(cvect_ch0, meanPerClassErrorTest_ch0-semPerClassErrorTest_ch0, meanPerClassErrorTest_ch0+ semPerClassErrorTest_ch0, alpha=0.5, edgecolor='r', facecolor='r')
            plt.fill_between(cvect_ch0, meanPerClassErrorTest_incorr_ch0-semPerClassErrorTest_incorr_ch0, meanPerClassErrorTest_incorr_ch0+ semPerClassErrorTest_incorr_ch0, alpha=0.5, edgecolor='g', facecolor='g')        
            plt.fill_between(cvect_ch0, meanPerClassErrorTest_otherChoice_ch0-semPerClassErrorTest_otherChoice_ch0, meanPerClassErrorTest_otherChoice_ch0+ semPerClassErrorTest_otherChoice_ch0, alpha=0.5, edgecolor='k', facecolor='k')        
    #            plt.fill_between(cvect_ch0, meanPerClassErrorTest_incorr_chance-semPerClassErrorTest_incorr_chance, meanPerClassErrorTest_incorr_chance+ semPerClassErrorTest_incorr_chance, alpha=0.5, edgecolor='m', facecolor='m')        
    #            plt.fill_between(cvect_ch0, meanPerClassErrorTest_incorr_shfl-semPerClassErrorTest_incorr_shfl, meanPerClassErrorTest_incorr_shfl+ semPerClassErrorTest_incorr_shfl, alpha=0.5, edgecolor='c', facecolor='c')        
    #            plt.fill_between(cvect_ch0, meanPerClassErrorTest_corr_shfl-semPerClassErrorTest_corr_shfl, meanPerClassErrorTest_corr_shfl+ semPerClassErrorTest_corr_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
    #            plt.fill_between(cvect_ch0, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
#            plt.fill_between(cvect_ch0, meanPerClassErrorTest_otherChoice_chance-semPerClassErrorTest_otherChoice_chance, meanPerClassErrorTest_otherChoice_chance+ semPerClassErrorTest_otherChoice_chance, alpha=0.5, edgecolor='b', facecolor='b')        
    #                plt.fill_between(cvect_ch0, meanPerClassErrorTest_otherChoice_shfl-semPerClassErrorTest_otherChoice_shfl, meanPerClassErrorTest_otherChoice_shfl+ semPerClassErrorTest_otherChoice_shfl, alpha=0.5, edgecolor='b', facecolor='b')        
            
            plt.plot(cvect_ch0, meanPerClassErrorTrain_ch0, 'k', label = 'training')
            plt.plot(cvect_ch0, meanPerClassErrorTest_ch0, 'r', label = 'validation')
            plt.plot(cvect_ch0, meanPerClassErrorTest_incorr_ch0, 'g', label = 'incorr')        
            plt.plot(cvect_ch0, meanPerClassErrorTest_otherChoice_ch0, 'k', label = 'other choice')        
    #                plt.plot(cvect_ch0, meanPerClassErrorTest_incorr_chance, 'm', label = 'incorr-chance')            
    #                plt.plot(cvect_ch0, meanPerClassErrorTest_incorr_shfl, 'c', label = 'incorr-shfl')            
    #            plt.plot(cvect_ch0, meanPerClassErrorTest_corr_shfl, 'y', label = 'corr-shfl')            
    #            plt.plot(cvect_ch0, meanPerClassErrorTest_chance, 'b', label = 'corr-chance')        
#            plt.plot(cvect_ch0, meanPerClassErrorTest_otherChoice_chance, 'b', label = 'other choice-chance')    
    #                plt.plot(cvect_ch0, meanPerClassErrorTest_otherChoice_shfl, 'b', label = 'other choice-shfl')
            
            plt.plot(cvect_ch0[cvect_ch0==cbest_ch0], meanPerClassErrorTest_ch0[cvect_ch0==cbest_ch0], 'ro')
            plt.plot(cvect_ch0[cvect_ch0==cbest_incorr_ch0], meanPerClassErrorTest_incorr_ch0[cvect_ch0==cbest_incorr_ch0], 'go')
            plt.plot(cvect_ch0[cvect_ch0==cbest_otherChoice_ch0], meanPerClassErrorTest_otherChoice_ch0[cvect_ch0==cbest_otherChoice_ch0], 'ko')
    
            plt.xlim([cvect_ch0[1], cvect_ch0[-1]])
            plt.xscale('log')
#                plt.title(' Number of trials: %d choice0; %d choice1; %d incorr \n Best c: CV = %.2g; otherChoice = %.2g; incorr = %.2g' %(sum(~trsExcluded_svr_ch0), sum(~trsExcluded_svr_ch1), sum(~trsExcluded_incorr), cbest, cbest_otherChoice, cbest_incorr))
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('prediction error (normalized sum of square error)')
#                plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            plt.tight_layout()
        #    plt.ylim([-10,110])
        
        

        ######################################## choice 1 ########################################

#            print ' Number of trials: %d choice0; %d choice1; %d incorr \n Best c: CV = %.2g; otherChoice = %.2g; incorr = %.2g' %(sum(~trsExcluded_svr_ch0), sum(~trsExcluded_svr_ch1), sum(~trsExcluded_incorr), cbest, cbest_otherChoice, cbest_incorr)
#            print 'Best c: CV = %.2g; otherChoice = %.2g; incorr = %.2g' %(cbest, cbest_otherChoice, cbest_incorr)
#                plt.figure()
        plt.subplot(1,4,(3,4))
        if svmFound_ch1==1:
            plt.fill_between(cvect_ch1, meanPerClassErrorTrain_ch1-semPerClassErrorTrain_ch1, meanPerClassErrorTrain_ch1+ semPerClassErrorTrain_ch1, alpha=0.5, edgecolor='k', facecolor='k')
            plt.fill_between(cvect_ch1, meanPerClassErrorTest_ch1-semPerClassErrorTest_ch1, meanPerClassErrorTest_ch1+ semPerClassErrorTest_ch1, alpha=0.5, edgecolor='r', facecolor='r')
            plt.fill_between(cvect_ch1, meanPerClassErrorTest_incorr_ch1-semPerClassErrorTest_incorr_ch1, meanPerClassErrorTest_incorr_ch1+ semPerClassErrorTest_incorr_ch1, alpha=0.5, edgecolor='g', facecolor='g')        
            plt.fill_between(cvect_ch1, meanPerClassErrorTest_otherChoice_ch1-semPerClassErrorTest_otherChoice_ch1, meanPerClassErrorTest_otherChoice_ch1+ semPerClassErrorTest_otherChoice_ch1, alpha=0.5, edgecolor='k', facecolor='k')        
    #            plt.fill_between(cvect_ch1, meanPerClassErrorTest_incorr_chance-semPerClassErrorTest_incorr_chance, meanPerClassErrorTest_incorr_chance+ semPerClassErrorTest_incorr_chance, alpha=0.5, edgecolor='m', facecolor='m')        
    #            plt.fill_between(cvect_ch1, meanPerClassErrorTest_incorr_shfl-semPerClassErrorTest_incorr_shfl, meanPerClassErrorTest_incorr_shfl+ semPerClassErrorTest_incorr_shfl, alpha=0.5, edgecolor='c', facecolor='c')        
    #            plt.fill_between(cvect_ch1, meanPerClassErrorTest_corr_shfl-semPerClassErrorTest_corr_shfl, meanPerClassErrorTest_corr_shfl+ semPerClassErrorTest_corr_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
    #            plt.fill_between(cvect_ch1, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
#            plt.fill_between(cvect_ch1, meanPerClassErrorTest_otherChoice_chance-semPerClassErrorTest_otherChoice_chance, meanPerClassErrorTest_otherChoice_chance+ semPerClassErrorTest_otherChoice_chance, alpha=0.5, edgecolor='b', facecolor='b')        
    #                plt.fill_between(cvect_ch1, meanPerClassErrorTest_otherChoice_shfl-semPerClassErrorTest_otherChoice_shfl, meanPerClassErrorTest_otherChoice_shfl+ semPerClassErrorTest_otherChoice_shfl, alpha=0.5, edgecolor='b', facecolor='b')        
            
            plt.plot(cvect_ch1, meanPerClassErrorTrain_ch1, 'k', label = 'training')
            plt.plot(cvect_ch1, meanPerClassErrorTest_ch1, 'r', label = 'validation')
            plt.plot(cvect_ch1, meanPerClassErrorTest_incorr_ch1, 'g', label = 'incorr')        
            plt.plot(cvect_ch1, meanPerClassErrorTest_otherChoice_ch1, 'k', label = 'other choice')        
    #                plt.plot(cvect_ch1, meanPerClassErrorTest_incorr_chance, 'm', label = 'incorr-chance')            
    #                plt.plot(cvect_ch1, meanPerClassErrorTest_incorr_shfl, 'c', label = 'incorr-shfl')            
    #            plt.plot(cvect_ch1, meanPerClassErrorTest_corr_shfl, 'y', label = 'corr-shfl')            
    #            plt.plot(cvect_ch1, meanPerClassErrorTest_chance, 'b', label = 'corr-chance')        
#            plt.plot(cvect_ch1, meanPerClassErrorTest_otherChoice_chance, 'b', label = 'other choice-chance')    
    #                plt.plot(cvect_ch1, meanPerClassErrorTest_otherChoice_shfl, 'b', label = 'other choice-shfl')
            
            plt.plot(cvect_ch1[cvect_ch1==cbest_ch1], meanPerClassErrorTest_ch1[cvect_ch1==cbest_ch1], 'ro')
            plt.plot(cvect_ch1[cvect_ch1==cbest_incorr_ch1], meanPerClassErrorTest_incorr_ch1[cvect_ch1==cbest_incorr_ch1], 'go')
            plt.plot(cvect_ch1[cvect_ch1==cbest_otherChoice_ch1], meanPerClassErrorTest_otherChoice_ch1[cvect_ch1==cbest_otherChoice_ch1], 'ko')
    
            plt.xlim([cvect_ch1[1], cvect_ch1[-1]])
            plt.xscale('log')
#                plt.title(' Number of trials: %d choice0; %d choice1; %d incorr \n Best c: CV = %.2g; otherChoice = %.2g; incorr = %.2g' %(sum(~trsExcluded_svr_ch0), sum(~trsExcluded_svr_ch1), sum(~trsExcluded_incorr), cbest, cbest_otherChoice, cbest_incorr))
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('prediction error (normalized sum of square error)')
#                plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            plt.tight_layout()
        #    plt.ylim([-10,110])
            
            
    #%% Angle between decoders found on choice 0 and on choice 1
         
    if svmFound_ch0==1 and svmFound_ch1==1:             
        # angle between decocers of all shuffles                         
        angc = np.full((numSamples, numSamples), np.nan)
        for i in range(numSamples):
            for j in range(numSamples):
                angc[i,j] = np.arccos(abs(np.dot(wAllC_n_ch0[i,:], wAllC_n_ch1[j,:].transpose())))*180/np.pi

#        np.diagonal(angc)
        
#        angc0 = angc + 0
        # set diagonal elements to nan
#        np.fill_diagonal(angc, np.nan)

        if doPlots:
            plt.figure()            
            plt.subplot(121)
            plt.imshow(angc, cmap='jet_r')
            plt.colorbar()
            
#            plt.subplot(122)
#            plt.imshow(angc, cmap='jet_r')
#            plt.colorbar()
    
        # average angle between the decoder trained on ch0 and that of ch1, averaged across all combination of shuffles.                
    #    a = np.tril(angc)
        # set repitative values to nan
        angc[np.triu_indices(angc.shape[0],1)] = np.nan
#        plt.imshow(angc, cmap='jet_r')
        AveAngDecodTrainedOnCh01[iday] = np.nanmean(angc.reshape((-1,)))        
        
        
#%% 
############################################################################################################################################
############################################################################################################################################
############################ Optimal SVR decoders trained on each choice: take average and std across samples ###############################################################################
############################################################################################################################################
############################################################################################################################################    
        
av_train_d_ch0 = (np.mean(classErr_bestC_train_data_all_ch0,axis=0))
av_train_d_ch1 = (np.mean(classErr_bestC_train_data_all_ch1,axis=0))
sd_train_d_ch0 = (np.std(classErr_bestC_train_data_all_ch0,axis=0)) / np.sqrt(numSamp)  
sd_train_d_ch1 = (np.std(classErr_bestC_train_data_all_ch1,axis=0)) / np.sqrt(numSamp)  

av_test_d_ch0 = (np.mean(classErr_bestC_test_data_all_ch0,axis=0))
av_test_d_ch1 = (np.mean(classErr_bestC_test_data_all_ch1,axis=0))
sd_test_d_ch0 = (np.std(classErr_bestC_test_data_all_ch0,axis=0)) / np.sqrt(numSamp)  
sd_test_d_ch1 = (np.std(classErr_bestC_test_data_all_ch1,axis=0)) / np.sqrt(numSamp)    

av_otherChoice_d_ch0 = (np.mean(classErr_bestC_otherChoice_data_all_ch0,axis=0))
av_otherChoice_d_ch1 = (np.mean(classErr_bestC_otherChoice_data_all_ch1,axis=0))
sd_otherChoice_d_ch0 = (np.std(classErr_bestC_otherChoice_data_all_ch0,axis=0)) / np.sqrt(numSamp)    
sd_otherChoice_d_ch1 = (np.std(classErr_bestC_otherChoice_data_all_ch1,axis=0)) / np.sqrt(numSamp)    

av_incorr_d_ch0 = (np.mean(classErr_bestC_incorr_data_all_ch0,axis=0))
av_incorr_d_ch1 = (np.mean(classErr_bestC_incorr_data_all_ch1,axis=0))
sd_incorr_d_ch0 = (np.std(classErr_bestC_incorr_data_all_ch0,axis=0)) / np.sqrt(numSamp)    
sd_incorr_d_ch1 = (np.std(classErr_bestC_incorr_data_all_ch1,axis=0)) / np.sqrt(numSamp)    




######################## PLOTS ########################
#################################################################
#################################################################
#%% Plot average across shuffles for each day

# L2;
plt.figure(figsize=(6,5))
gs = gridspec.GridSpec(2,10)#, width_ratios=[2, 1]) 


################ choice 0 ################

ax = plt.subplot(gs[0,0:-5])
#ax = plt.subplot(121)
plt.errorbar(range(numDays), av_train_d_ch0, yerr = sd_train_d_ch0, color='g', label='Train')
plt.errorbar(range(numDays), av_test_d_ch0, yerr = sd_test_d_ch0, color='c', label='Test')
plt.errorbar(range(numDays), av_otherChoice_d_ch0, yerr = sd_otherChoice_d_ch0, color='k', label='OtherChoice')
plt.errorbar(range(numDays), av_incorr_d_ch0, yerr = sd_incorr_d_ch0, color=[.5,.5,.5], label='Incorr')
   
plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('Ch0 prediction error\n(cross-validated)', fontsize=13, labelpad=10)
plt.plot(range(numDays), np.ones((numDays)), color='k', ls=':') # chance line.... max error = 1
    
plt.xlim([-1, len(days)])
lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.45), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax)
ymin, ymax = ax.get_ylim()
#    plt.ylim([40,100])

##%% Average across days
ax = plt.subplot(gs[0,-5:-3])
#x =[0,1]
#labels = ['Data', 'Shfl']
#plt.errorbar(x, [np.mean(av_l2_test_d), np.mean(av_l2_test_s)], yerr = [np.std(av_l2_test_d), np.std(av_l2_test_s)], marker='o', fmt=' ', color='k')
x =[0,1,2,3]
labels = ['Train', 'Test', 'OtherChoice', 'Incorr']
plt.errorbar(x, [np.nanmean(av_train_d_ch0), np.nanmean(av_test_d_ch0), np.nanmean(av_otherChoice_d_ch0), np.nanmean(av_incorr_d_ch0)], yerr = [np.nanstd(av_train_d_ch0), np.nanstd(av_test_d_ch0), np.nanstd(av_otherChoice_d_ch0), np.nanstd(av_incorr_d_ch0)], marker='o', fmt=' ', color='k')    

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
    
    
    
    
################ choice 1 ################
    
ax = plt.subplot(gs[1,0:-5])
#ax = plt.subplot(121)
# ch1
plt.errorbar(range(numDays), av_train_d_ch1, yerr = sd_train_d_ch1, color='g', label='Train')
plt.errorbar(range(numDays), av_test_d_ch1, yerr = sd_test_d_ch1, color='c', label='Test')
plt.errorbar(range(numDays), av_otherChoice_d_ch1, yerr = sd_otherChoice_d_ch1, color='k', label='OtherChoice')
plt.errorbar(range(numDays), av_incorr_d_ch1, yerr = sd_incorr_d_ch1, color=[.5,.5,.5], label='Incorr')

plt.xlabel('Days', fontsize=13, labelpad=10)
plt.ylabel('Ch1 prediction error\n(cross-validated)', fontsize=13, labelpad=10)
plt.plot(range(numDays), np.ones((numDays)), color='k', ls=':') # chance line.... max error = 1

plt.xlim([-1, len(days)])
#    lgd = plt.legend(loc='upper left', bbox_to_anchor=(-.05,1.45), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax)
ymin, ymax = ax.get_ylim()
#    plt.ylim([40,100])

##%% Average across days
ax = plt.subplot(gs[1,-5:-3])
#x =[0,1]
#labels = ['Data', 'Shfl']
#plt.errorbar(x, [np.mean(av_l2_test_d), np.mean(av_l2_test_s)], yerr = [np.std(av_l2_test_d), np.std(av_l2_test_s)], marker='o', fmt=' ', color='k')
x =[0,1,2,3]
labels = ['Train', 'Test', 'OtherChoice', 'Incorr']
# ch1
plt.errorbar(x, [np.nanmean(av_train_d_ch1), np.nanmean(av_test_d_ch1), np.nanmean(av_otherChoice_d_ch1), np.nanmean(av_incorr_d_ch1)], yerr = [np.nanstd(av_train_d_ch1), np.nanstd(av_test_d_ch1), np.nanstd(av_otherChoice_d_ch1), np.nanstd(av_incorr_d_ch1)], marker='o', fmt=' ', color='k')    

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
    dd = 'stAl_svr_1choice_epAllStim'
    
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    
    
#%% How different are SVR decoders trained on choice 0 vs choice 1?
    
plt.figure()            
plt.plot(AveAngDecodTrainedOnCh01)
plt.xlabel('Days')
plt.ylabel('Ave angle btwn decoders trained on ch0 and ch1')


if savefigs:#% Save the figure    
    dd = 'stAl_svr_1choice_epAllStim_ang'
    
    fign = os.path.join(svmdir+dnow, suffn[0:5]+dd+'.'+fmt[0])
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    
    
#%% Print some summary numbers
    
print np.nanmean(np.mean(classErr_bestC_train_data_all_ch0,axis=0)), np.nanmean(np.mean(classErr_bestC_train_data_all_ch1,axis=0))
print np.nanmean(np.mean(classErr_bestC_test_data_all_ch0,axis=0)), np.nanmean(np.mean(classErr_bestC_test_data_all_ch1,axis=0))
print np.nanmean(np.mean(classErr_bestC_otherChoice_data_all_ch0,axis=0)), np.nanmean(np.mean(classErr_bestC_otherChoice_data_all_ch1,axis=0))        
print np.nanmean(np.mean(classErr_bestC_incorr_data_all_ch0,axis=0)), np.nanmean(np.mean(classErr_bestC_incorr_data_all_ch1,axis=0))
print np.nanmean(AveAngDecodTrainedOnCh01)


### p values relative to 1 (max error)
# train
_,p_ch0 = stats.ttest_1samp(av_train_d_ch0, 1, nan_policy = 'omit')
_,p_ch1 = stats.ttest_1samp(av_train_d_ch1, 1, nan_policy = 'omit')
print p_ch0, p_ch1    

# test
_,p_ch0 = stats.ttest_1samp(av_test_d_ch0, 1, nan_policy = 'omit')
_,p_ch1 = stats.ttest_1samp(av_test_d_ch1, 1, nan_policy = 'omit')
print p_ch0, p_ch1


