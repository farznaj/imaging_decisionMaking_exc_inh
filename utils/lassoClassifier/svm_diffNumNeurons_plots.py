# -*- coding: utf-8 -*-
"""
Plot the output of svm_diffNumNeurons.py, which gives us class accuracy for decoders trained using different number of neurons.
A very high bestc (cbest = 10**6) was used so no regularization was done for any of the decoders.
For each number of neurons (x axis of the plots) we trained the decoder for several times to get random sets of neurons.

vars are provided by svm_diffNumNeurons.py (to be run on the cluster)



Created on Thu Dec 15 17:10:43 2016
@author: farznaj
"""

# Go to svm_plots_setVars and define vars!
execfile("svm_plots_setVars.py")    


#%%
dnow = '/classAccur_diffNumNeurons'


#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName
def setSVMname(pnevFileName, trialHistAnalysis, ntName, itiName='all'):
    import glob
    
    if trialHistAnalysis:
        svmn = 'diffNumNeurons_svmPrevChoice_%sN_%sITIs_ep*-*ms_*' %(ntName, itiName) # C2 has test/trial shuffles as well. C doesn't have it.
    else:
        svmn = 'diffNumNeurons_svmCurrChoice_%sN_ep%d-%dms_*' %(ntName, ep_ms[0], ep_ms[-1])   
    
    svmn = svmn + pnevFileName[-32:]    
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
    svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    svmName = svmName[0] # get the latest file
    
    return svmName
    

#%%    
perClassErrorTrain_data_nN_allD = []
perClassErrorTrain_shfl_nN_allD = []
perClassErrorTest_data_nN_allD = []
perClassErrorTest_shfl_nN_allD = []
    
for iday in range(len(days)):
    
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
    svmName = setSVMname(pnevFileName, trialHistAnalysis, ntName, itiName) # latest is l1, then l2. we use [] to get both files. # after that you again ran analysis with cross validation shuffles (l1).
#    svmNameAll = setSVMname(pnevFileName, trialHistAnalysis, ntName, np.nan, itiName, 0) # go w latest for 500 shuffles and bestc including at least one non0 w.
    print os.path.basename(svmName)
        
    '''    
    Data = scio.loadmat(svmName, variable_names=['w'])
    w = Data.pop('w')[0,:]
    
    if abs(w.sum()) < eps:  # I think it is wrong to do this. When ws are all 0, it means there was no decoder for that day, ie no info about the choice.
        print '\tAll weights in the allTrials-trained decoder are 0 ... '            
#    else:    
    '''        
    Data = scio.loadmat(svmName, variable_names=['perClassErrorTrain_data_nN_all', 'perClassErrorTrain_shfl_nN_all', 'perClassErrorTest_data_nN_all', 'perClassErrorTest_shfl_nN_all'])
    perClassErrorTrain_data_nN_all = Data.pop('perClassErrorTrain_data_nN_all').squeeze()
    perClassErrorTrain_shfl_nN_all = Data.pop('perClassErrorTrain_shfl_nN_all').squeeze()
    perClassErrorTest_data_nN_all = Data.pop('perClassErrorTest_data_nN_all').squeeze()
    perClassErrorTest_shfl_nN_all = Data.pop('perClassErrorTest_shfl_nN_all').squeeze()

    # pool all neural and trials shuffles for each element of the above arrays 
    perClassErrorTrain_data_nN_all = np.array([np.ravel(perClassErrorTrain_data_nN_all[i], order='F') for i in range(len(perClassErrorTrain_data_nN_all))]) # For each element, pool all neural and trial shuffles.
    perClassErrorTrain_shfl_nN_all = np.array([np.ravel(perClassErrorTrain_shfl_nN_all[i], order='F') for i in range(len(perClassErrorTrain_shfl_nN_all))])
    perClassErrorTest_data_nN_all = np.array([np.ravel(perClassErrorTest_data_nN_all[i], order='F') for i in range(len(perClassErrorTest_data_nN_all))]) # For each element, pool all neural and trial shuffles.
    perClassErrorTest_shfl_nN_all = np.array([np.ravel(perClassErrorTest_shfl_nN_all[i], order='F') for i in range(len(perClassErrorTest_shfl_nN_all))])    
    
    # keep results from all days
    perClassErrorTrain_data_nN_allD.append(perClassErrorTrain_data_nN_all) 
    perClassErrorTrain_shfl_nN_allD.append(perClassErrorTrain_shfl_nN_all) 
    perClassErrorTest_data_nN_allD.append(perClassErrorTest_data_nN_all) 
    perClassErrorTest_shfl_nN_allD.append(perClassErrorTest_shfl_nN_all) 
    
    
#%% class accuracy: Average across neuron shuffles for each element of perClassErrorTest_shfl_nN_allD
perClassErrorTrain_data_nN_avN = [] #[[]]*np.shape(days)[0] #np.empty(np.shape(days), dtype=float)# np.full((np.shape(days)), np.nan)
perClassErrorTrain_shfl_nN_avN = []
perClassErrorTest_data_nN_avN = []
perClassErrorTest_shfl_nN_avN = []
perClassErrorTrain_data_nN_sdN = [] #[[]]*np.shape(days)[0] #np.empty(np.shape(days), dtype=float)# np.full((np.shape(days)), np.nan)
perClassErrorTrain_shfl_nN_sdN = []
perClassErrorTest_data_nN_sdN = []
perClassErrorTest_shfl_nN_sdN = []

for iday in range(len(days)):
    perClassErrorTrain_data_nN_avN.append([100-perClassErrorTrain_data_nN_allD[iday][ine].mean() for ine in range(len(perClassErrorTrain_data_nN_allD[iday]))])
    perClassErrorTrain_shfl_nN_avN.append([100-perClassErrorTrain_shfl_nN_allD[iday][ine].mean() for ine in range(len(perClassErrorTrain_shfl_nN_allD[iday]))])
    perClassErrorTest_data_nN_avN.append([100-perClassErrorTest_data_nN_allD[iday][ine].mean() for ine in range(len(perClassErrorTest_data_nN_allD[iday]))])
    perClassErrorTest_shfl_nN_avN.append([100-perClassErrorTest_shfl_nN_allD[iday][ine].mean() for ine in range(len(perClassErrorTest_shfl_nN_allD[iday]))])    

    perClassErrorTrain_data_nN_sdN.append([perClassErrorTrain_data_nN_allD[iday][ine].std() for ine in range(len(perClassErrorTrain_data_nN_allD[iday]))])
    perClassErrorTrain_shfl_nN_sdN.append([perClassErrorTrain_shfl_nN_allD[iday][ine].std() for ine in range(len(perClassErrorTrain_shfl_nN_allD[iday]))])
    perClassErrorTest_data_nN_sdN.append([perClassErrorTest_data_nN_allD[iday][ine].std() for ine in range(len(perClassErrorTest_data_nN_allD[iday]))])
    perClassErrorTest_shfl_nN_sdN.append([perClassErrorTest_shfl_nN_allD[iday][ine].std() for ine in range(len(perClassErrorTest_shfl_nN_allD[iday]))])    



#%% Plot class accur vs num neurons in the decoder for each day separately (for both testing and training data)

plt.figure(figsize=(3,80))

for iday in range(len(days)):  
    plt.subplot(len(days)*2,1,iday*2+1)
    
    x = range(len(perClassErrorTrain_data_nN_avN[iday]))
#    plt.errorbar(x, perClassErrorTrain_data_nN_avN[iday], perClassErrorTrain_data_nN_sdN[iday], label='data')
#    plt.errorbar(x, perClassErrorTrain_shfl_nN_avN[iday], perClassErrorTrain_shfl_nN_sdN[iday], label='data')
    plt.plot(perClassErrorTrain_data_nN_avN[iday], label='shfl')
    plt.plot(perClassErrorTrain_shfl_nN_avN[iday], label='shfl')
    lgd = plt.legend(frameon=False)
    plt.title('%s' %(days[iday]))
    plt.xlabel('# neurons in decoder')    
    plt.ylabel('Class accuracy')    
#    plt.title('%s\ntraining' %(days[iday]))
    makeNicePlots(plt.gca())
    
    plt.subplot(len(days)*2,1,iday*2+2)
#    plt.errorbar(x, perClassErrorTest_data_nN_avN[iday], perClassErrorTest_data_nN_sdN[iday], label='data')
#    plt.errorbar(x, perClassErrorTest_shfl_nN_avN[iday], perClassErrorTest_shfl_nN_sdN[iday], label='data')
    plt.plot(perClassErrorTest_data_nN_avN[iday], label='data')    
    plt.plot(perClassErrorTest_shfl_nN_avN[iday], label='shfl')
#    plt.title('testing')
    makeNicePlots(plt.gca())
      
plt.subplots_adjust(hspace=.5)
   

if savefigs:#% Save the figure
    fign = os.path.join(svmdir+dnow, suffn+'classAccur_diffNN_eachDay'+'.'+fmt[0])
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')



#%%
Find a way to average results across days
Also it seems like days w more neurons are doing better.... remember no regularization... what does it mean?
Comp exc vs inh