# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:16:54 2017

@author: farznaj
"""

mousename = 'fni17'

if mousename == 'fni18': #set one of the following to 1:
    allDays = 1# all 7 days will be used (last 3 days have z motion!)
    noZmotionDays = 0 # 4 days that dont have z motion will be used.
    noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!
elif mousename == 'fni19':    
    allDays = 1
    noExtraStimDays = 0   
else:
    import numpy as np
    allDays = np.nan
    noZmotionDays = np.nan
    noZmotionDays_strict = np.nan
    noExtraStimDays = np.nan

    
corrTrained = 1 
ch_st_goAl = [1,0,0] # whether do analysis on traces aligned on choice, stim or go tone. #chAl = 1 # If 1, analyze SVM output of choice-aligned traces, otherwise stim-aligned traces. 
chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
stAl = ch_st_goAl[1]
goToneAl = ch_st_goAl[2]
trialHistAnalysis = 0

execfile("defFuns.py")
#execfile("svm_plots_setVars_n.py")  
days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays)

frameLength = 1000/30.9; # sec.
regressBins = int(np.round(100/frameLength)) # must be same regressBins used in svm_eachFrame. 100ms # set to nan if you don't want to downsample.


#%%
for iday in np.arange(0,len(days)):  

    ##%%            
    print '___________________'
    imagingFolder = days[iday][0:6]; #'151013'
    mdfFileNumber = map(int, (days[iday][7:]).split("-")); #[1,2] 
        
    ##%% Set .mat file names
    pnev2load = [] #[] [3] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.
    signalCh = [2] # since gcamp is channel 2, should be always 2.
    postNProvided = 1; # If your directory does not contain pnevFile and instead it contains postFile, set this to 1 to get pnevFileName
       
    imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load, postNProvided=postNProvided)
    
    postName = os.path.join(os.path.dirname(pnevFileName), 'post_'+os.path.basename(pnevFileName))
    moreName = os.path.join(os.path.dirname(pnevFileName), 'more_'+os.path.basename(pnevFileName))    
    print(os.path.basename(imfilename))


    for idi in range(3):
        
        doInhAllexcEqexc = np.full((3), False)
        doInhAllexcEqexc[idi] = True 
        
        # find the svm file to be renamed  (which was in fact corrTrained but doesnt have corr)
        svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc, regressBins)
#        print svmName    
        a = svmName[0] # this is in fact corr, should be renamed
        print a
    
        # make sure it is really the corrTrained file... by checking its number of trials with a known corrTrained file
        dd = scio.loadmat(a,variable_names=['trsExcluded'])
        dd = dd['trsExcluded'].flatten()
        dds = sum(dd)
    		
        acorr = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc, regressBins, corrTrained=1, shflTrsEachNeuron=1)[0]		
        dc = scio.loadmat(acorr,variable_names=['trsExcluded'])
        dc = dc['trsExcluded'].flatten()
        dcs = sum(dc)
    
        if dds!=dcs:
            print dds,dcs
            sys.exit('something wrong') 
        else:
            ## rename (add corr)       
            f = a.find('excInh_SVMtrained_eachFrame_')
            e = f + len('excInh_SVMtrained_eachFrame_')
            
            an = a[f:e]+'corr_'
            
            a = a.replace(a[f:e], an)
            print a
            
            os.rename(svmName[0],a)
    
    