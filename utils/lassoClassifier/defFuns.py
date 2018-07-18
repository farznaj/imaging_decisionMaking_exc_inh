## Load commonly used libraries, and define commanly used functions

trialHistAnalysis = 0;
iTiFlg = 2; # Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.  

#%%
frameLength = 1000/30.9; # sec.  # np.diff(time_aligned_stim)[0];
palpha = .05 # p <= palpha is significant


##%% Import modules
import sys
eps = sys.float_info.epsilon #10**-10 # tiny number below which weight is considered 0
import os
import glob
import numpy as np   
import numpy.random as rng
import scipy as sci
import scipy.io as scio
import scipy.stats as stats
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import copy

#import sys
#sys.path.append('/home/farznaj/Documents/trial_history/imaging/')  
#from setImagingAnalysisNamesP import *
#neuronType = 2; # 0: excitatory, 1: inhibitory, 2: all types.    


#%% Set svm and figure directories; also some vars related to saving figures

#plt.rc('font', family='helvetica')    
fmt = ['pdf', 'svg', 'eps'] #'png', 'pdf': preserve transparency # Format of figures for saving

#figsDir = '/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/' # Directory for saving figures.
import platform
if platform.system()=='Linux':
#    figsDir = '/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/'
    figsDir = '/home/farznaj/Dropbox/ChurchlandLab/Projects/inhExcDecisionMaking/'
elif platform.system()=='Darwin':
    figsDir = '/Users/Farzaneh/Dropbox/ChurchlandLab/Farzaneh_Gamal/' #    svmfold = ''
    figsDir = '/Users/Farzaneh/Dropbox/ChurchlandLab//Projects/inhExcDecisionMaking/'

# Make folder named SVM to save figures inside it
svmdir = os.path.join(figsDir, 'SVM') #if not os.path.exists(svmdir): #    os.makedirs(svmdir)    


if trialHistAnalysis: #     ep_ms = np.round((ep-eventI)*frameLength) #    th_stim_dur = []
    suffn = 'prev_'  #    suffnei = 'prev_%sITIs_excInh_' %(itiName)
else:
    suffn = 'curr_'  #    suffnei = 'curr_%s_epVar_' %('excInh')   #print '\n', suffn[:-1], ' - ', suffnei[:-1]


if trialHistAnalysis==1:    
    if iTiFlg==0:
        itiName = 'short'
    elif iTiFlg==1:
        itiName = 'long'
    elif iTiFlg==2:
        itiName = 'all'
else: # wont be used, only not to get error when running setSVMname
    itiName = 'all'



#%%
########################################################################
## All commanly used functions
########################################################################

#%%
# LOOKING FOR A DAY: #days.index('151009_1')   # name of this function should change to set_days!
def svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained=1, trialHistAnalysis=0, iTiFlg=1, allDays=1, noZmotionDays=0, noZmotionDays_strict=0, noExtraStimDays=0, loadYtest=0, decodeStimCateg=0):
     
    ##%% Define days that you want to analyze
    '''
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
    '''
        
    ##%%        
    if mousename=='fni16':
        # 0817,0818: don't have excInh files
        # 0817,0818,0819: don't have eachFrame, stAl. Also their FOV more superficial that the rest of the days
        if 'ch_st_goAl' in locals(): # eachFrame analysis
            if ch_st_goAl[0]==1: # chAl   
#                if 'corrTrained' in locals() and corrTrained==1: # svm_eachFrame trained only on corr trials: exclude '150824_1-2' (no LR trials)
                days = ['150817_1', '150818_1', '150819_1', '150820_1', '150821_1-2', '150825_1-2-3', '150826_1', '150827_1', '150828_1-2', '150831_1-2', '150901_1', '150903_1', '150904_1', '150915_1', '150916_1-2', '150917_1', '150918_1-2-3-4', '150921_1', '150922_1', '150923_1', '150924_1', '150925_1-2-3', '150928_1-2', '150929_1-2', '150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2']; # '150914_1-2' : don't analyze!
#                else:
#                    days = ['150817_1', '150818_1', '150819_1', '150820_1', '150821_1-2', '150824_1-2', '150825_1-2-3', '150826_1', '150827_1', '150828_1-2', '150831_1-2', '150901_1', '150903_1', '150904_1', '150915_1', '150916_1-2', '150917_1', '150918_1-2-3-4', '150921_1', '150922_1', '150923_1', '150924_1', '150925_1-2-3', '150928_1-2', '150929_1-2', '150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2']; # '150914_1-2' : don't analyze!
            
            elif ch_st_goAl[1]==1: # stAl           
                days = ['150820_1', '150821_1-2', '150824_1-2', '150825_1-2-3', '150826_1', '150827_1', '150828_1-2', '150831_1-2', '150901_1', '150903_1', '150904_1', '150915_1', '150916_1-2', '150917_1', '150918_1-2-3-4', '150921_1', '150922_1', '150923_1', '150924_1', '150925_1-2-3', '150928_1-2', '150929_1-2', '150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2']; # '150914_1-2' : don't analyze!
    
        if 'loadInhAllexcEqexc' in locals(): # excInh_eachFrame analysis
            days = ['150819_1', '150820_1', '150821_1-2', '150824_1-2', '150825_1-2-3', '150826_1', '150827_1', '150828_1-2', '150831_1-2', '150901_1', '150903_1', '150904_1', '150915_1', '150916_1-2', '150917_1', '150918_1-2-3-4', '150921_1', '150922_1', '150923_1', '150924_1', '150925_1-2-3', '150928_1-2', '150929_1-2', '150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2']; # '150914_1-2' : don't analyze!
        
        if trialHistAnalysis == 1 and iTiFlg == 1:
            days.remove('151001_1') # this day has only 8 long ITI trials, 1 of which is hr ... so not enough trials for the HR class to train the classifier!
    
    
    
    elif mousename=='fni17': 
        # '150814_2, '150820_1', '150821_1', '150825_1': dont have enough trials. ('150821_1' and '150825_1' only have the eachFrame, stAl svm file)    
        # '150814_1', '150817_1', '150818_1': cant be run for eachFrame, stAl
        # days = ['151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2']     # Done on 6Jan2017: reverse the order, so it is from early days to last days
        if 'loadYtest' in locals() and loadYtest==1 and ch_st_goAl[0]==1 and 'corrTrained' in locals() and corrTrained==1: # chAl, svm_eachFrame trained only on corr trials: exclude '150818_1', '150819_1-2' (too few HR trials)
            # '150817_1' and '150824_1' only have 6 trials (when using equal hr and lr)... 
            days = ['150814_1', '150826_1', '150827_1', '150828_1', '150831_1', '150901_1', '150902_1-2', '150903_1', '150908_1', '150909_1', '150910_1', '150914_1', '150915_1-2', '150916_1', '150917_1-2', '150918_1', '150921_1-2-3', '150922_1-2', '150923_1-2-3', '150924_1-2', '150925_1-2', '150928_1-2', '150930_1-2-3-4', '151001_1', '151002_1-2', '151005_1-2', '151006_1', '151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2'];
            
        elif ch_st_goAl[0]==1: # and 'corrTrained' in locals() and corrTrained==1: # chAl, svm_eachFrame trained only on corr trials: exclude '150818_1', '150819_1-2' (too few HR trials)
            days = ['150814_1', '150817_1', '150824_1', '150826_1', '150827_1', '150828_1', '150831_1', '150901_1', '150902_1-2', '150903_1', '150908_1', '150909_1', '150910_1', '150914_1', '150915_1-2', '150916_1', '150917_1-2', '150918_1', '150921_1-2-3', '150922_1-2', '150923_1-2-3', '150924_1-2', '150925_1-2', '150928_1-2', '150930_1-2-3-4', '151001_1', '151002_1-2', '151005_1-2', '151006_1', '151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2'];
    #        days = ['151015_1']   # eg class accur
    #        days = ['151101_1']   # eg svm proj
            # 150910 : eg exc,inh FRs
#        else: # all days
#            days = ['150814_1', '150817_1', '150818_1', '150819_1-2', '150824_1', '150826_1', '150827_1', '150828_1', '150831_1', '150901_1', '150902_1-2', '150903_1', '150908_1', '150909_1', '150910_1', '150914_1', '150915_1-2', '150916_1', '150917_1-2', '150918_1', '150921_1-2-3', '150922_1-2', '150923_1-2-3', '150924_1-2', '150925_1-2', '150928_1-2', '150930_1-2-3-4', '151001_1', '151002_1-2', '151005_1-2', '151006_1', '151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2'];
    
    
    
    elif mousename=='fni18': # eachFrame, stimAl files couldnt be run for '151211_1', '151214_1-2', '151215_1-2', '151216_1'
        if allDays:
            days = ['151209_1', '151210_1', '151211_1', '151214_1-2', '151215_1-2', '151216_1', '151217_1-2'] # alldays
        elif noZmotionDays:        
            days = ['151209_1', '151210_1', '151211_1', '151214_1-2'] # following days removed bc of z motion: '151215_1-2', '151216_1', '151217_1-2'
        elif noZmotionDays_strict:        
            days = ['151209_1', '151210_1', '151211_1'] # even '151214_1-2' is suspicious about z motion, so remove it!
        
        if decodeStimCateg:
            days.remove('151209_1')
    
    
    elif mousename=='fni19':
        if allDays:
            if 'ch_st_goAl' in locals() and ch_st_goAl[1]==1:  # eachFrame, stimAL: '150904_1' doesnt have enough trials. taken out from below.       
                days = ['150901_1', '150903_1', '150914_1', '150915_1', '150916_1', '150917_1', '150918_1', '150921_1', '150922_1', '150923_1', '150924_1-2', '150925_1-2', '150928_4', '150929_3', '150930_1', '151001_1', '151002_1', '151005_1-2', '151006_1', '151007_1', '151008_1-2', '151009_1-3', '151012_1-2-3', '151013_1', '151015_2', '151016_1', '151019_1', '151020_1', '151022_1-2', '151023_1', '151026_1-2-3', '151027_1', '151028_1-2', '151029_1-2-3', '151101_1'];
            else: # chAl
#                if 'corrTrained' in locals() and corrTrained==1: # svm_eachFrame trained only on corr trials: exclude '150901' (few HR trials)
                days = ['150903_1', '150904_1', '150914_1', '150915_1', '150916_1', '150917_1', '150918_1', '150921_1', '150922_1', '150923_1', '150924_1-2', '150925_1-2', '150928_4', '150929_3', '150930_1', '151001_1', '151002_1', '151005_1-2', '151006_1', '151007_1', '151008_1-2', '151009_1-3', '151012_1-2-3', '151013_1', '151015_2', '151016_1', '151019_1', '151020_1', '151022_1-2', '151023_1', '151026_1-2-3', '151027_1', '151028_1-2', '151029_1-2-3', '151101_1'];                            
#                else:
#                    days = ['150901_1', '150903_1', '150904_1', '150914_1', '150915_1', '150916_1', '150917_1', '150918_1', '150921_1', '150922_1', '150923_1', '150924_1-2', '150925_1-2', '150928_4', '150929_3', '150930_1', '151001_1', '151002_1', '151005_1-2', '151006_1', '151007_1', '151008_1-2', '151009_1-3', '151012_1-2-3', '151013_1', '151015_2', '151016_1', '151019_1', '151020_1', '151022_1-2', '151023_1', '151026_1-2-3', '151027_1', '151028_1-2', '151029_1-2-3', '151101_1'];
                            
        elif noExtraStimDays:
            days = ['151001_1', '151002_1', '151005_1-2', '151006_1', '151007_1', '151008_1-2', '151009_1-3', '151012_1-2-3', '151013_1', '151015_2', '151016_1', '151019_1', '151020_1', '151022_1-2', '151023_1', '151026_1-2-3', '151027_1', '151028_1-2', '151029_1-2-3', '151101_1']
            
    daysOrig = days
    numDays = len(days)
    print 'Analyzing mouse',mousename,'-', len(days), 'days'
    
    return days, numDays    



#%%
def svm_setDays_allMice(mice, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, regressBins, useEqualTrNums, shflTrsEachNeuron, thTrained):
    
    ##%%
    chAl = ch_st_goAl[0] # If 1, use choice-aligned traces; otherwise use stim-aligned traces for trainign SVM. 
    stAl = ch_st_goAl[1]
    goToneAl = ch_st_goAl[2]


    #%% Set alldays, good days and mn_corr for all mice
    
    days_allMice = [] 
    numDaysAll = np.full(len(mice), np.nan, dtype=int)
    
    daysGood_allMice = []
    dayinds_allMice = []
    mn_corr_allMice = []
    
    allDays = np.nan
    noZmotionDays = np.nan
    noZmotionDays_strict = np.nan
    noExtraStimDays = np.nan
    
    for im in range(len(mice)):
        
        #%%            
        mousename = mice[im] # mousename = 'fni16' #'fni17'
        if mousename == 'fni18': #set one of the following to 1:
            allDays = 1# all 7 days will be used (last 3 days have z motion!)
            noZmotionDays = 0 # 4 days that dont have z motion will be used.
            noZmotionDays_strict = 0 # 3 days will be used, which more certainly dont have z motion!
        if mousename == 'fni19':    
            allDays = 1
            noExtraStimDays = 0   
            
    #    execfile("svm_plots_setVars_n.py")      #    execfile("svm_plots_setVars.py")      
        days, numDays = svm_plots_setVars_n(mousename, ch_st_goAl, corrTrained, trialHistAnalysis, iTiFlg, allDays, noZmotionDays, noZmotionDays_strict, noExtraStimDays)
    
        days_allMice.append(days)
        numDaysAll[im] = len(days)
        
        
        #%% Set number of hr, lr trials that were used for svm training on each day
        
        corr_hr_lr = np.full((len(days),2), np.nan) # number of hr, lr correct trials for each day    
    
        for iday in range(len(days)): 
        
            imagingFolder = days[iday][0:6]; #'151013'
            mdfFileNumber = map(int, (days[iday][7:]).split("-")); #[1,2] 
                
            imfilename, pnevFileName, dataPath = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=[2], pnev2load=[], postNProvided=1)        
            postName = os.path.join(os.path.dirname(pnevFileName), 'post_'+os.path.basename(pnevFileName))
    #        moreName = os.path.join(os.path.dirname(pnevFileName), 'more_'+os.path.basename(pnevFileName))       
    #        print(os.path.basename(imfilename))    
            svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, [1,0,0], regressBins, useEqualTrNums, corrTrained, shflTrsEachNeuron)[0]   
            
            ##%% Set number of hr, lr trials that were used for svm training        
            corr_hr, corr_lr = set_corr_hr_lr(postName, svmName)    
            corr_hr_lr[iday,:] = [corr_hr, corr_lr]    
    
    
        #%% Set good days (days with enough trials)
            
        mn_corr = np.min(corr_hr_lr,axis=1) # number of trials of each class. 90% of this was used for training, and 10% for testing.
    
        print 'num days to be excluded with few svm-trained trs:', sum(mn_corr < thTrained)    
        print np.array(days)[mn_corr < thTrained]
        
        numGoodDays = sum(mn_corr>=thTrained)    
        numOrigDays = numDaysAll[im].astype(int)
        
        dayinds = np.arange(numOrigDays)
        dayinds = np.delete(dayinds, np.argwhere(mn_corr < thTrained))
    
        dayinds_allMice.append(dayinds)
        daysGood_allMice.append(np.array(days_allMice[im])[dayinds])        
        mn_corr_allMice.append(mn_corr)
        
    
    return days_allMice, numDaysAll, daysGood_allMice, dayinds_allMice, mn_corr_allMice
        
    
    


#%% Extend the built in two tailed ttest function to one-tailed
# 'right': ### a>b
# 'left':  ### a<b
        
def ttest2(a, b, **tailOption):
    import scipy.stats as stats
    import numpy as np
    
    a = np.array(a)[~np.isnan(a)]
    b = np.array(b)[~np.isnan(b)]
    
    h, p = stats.ttest_ind(a, b)
    
    d = np.mean(a)-np.mean(b)
    
    if tailOption.get('tail'):
        tail = tailOption.get('tail').lower()
        if tail == 'right': ### a>b
            p = p/2.*(d>0) + (1-p/2.)*(d<0)
        elif tail == 'left':  ### a<b
            p = (1-p/2.)*(d>0) + p/2.*(d<0)
    if d==0:
        p = 1;
    return p
	
	

#%%
# optional inputs:
#postNProvided = 1; # Default:0; If your directory does not contain pnevFile and instead it contains postFile, set this to 1 to get pnevFileName
#signalCh = 2 # # since gcamp is channel 2, should be 2.
#pnev2load = [] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.

def setImagingAnalysisNamesP(mousename, imagingFolder=[], mdfFileNumber=1, **options):

    if options.get('signalCh'):
        signalCh = options.get('signalCh');    
    else:
        signalCh = 0
        
    if options.get('pnev2load'):
        pnev2load = options.get('pnev2load');    
    else:
        pnev2load = []
        
    if options.get('postNProvided'):
        postNProvided = options.get('postNProvided');    
    else:
        postNProvided = 0
        
    if options.get('nOuts'):
        nOuts = options.get('nOuts');    
    else:
        nOuts = 2


    ##%%
    import numpy as np
    import platform
    import glob
    import os.path
        
    if len(pnev2load)==0:
        pnev2load = [0];
            
    ##%%
    if platform.system()=='Linux':
        if os.getcwd().find('grid')!=-1: # server # sonas
            dataPath = '/sonas-hs/churchland/nlsas/data/data/'
            altDataPath = '/sonas-hs/churchland/hpc/home/space_managed_data/'            
        else: # office linux
#            dataPath = '/home/farznaj/Shares/Churchland/data/'
            dataPath = '/home/farznaj/Shares/Churchland_nlsas_data/data/'
            altDataPath = '/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/' # the new space-managed server (wos, to which data
    elif platform.system()=='Darwin':
        dataPath = '/Volumes/My Stu_win/ChurchlandLab/'
    else:
        dataPath = '/Users/gamalamin/git_local_repository/Farzaneh/data/'
        
    
    ##%%        
    if type(imagingFolder)==str:    
        tifFold = os.path.join(dataPath+mousename,'imaging',imagingFolder)

        if any([mousename in s for s in ['fni18','fni19']]): # fni18, fni19
            tifFold = os.path.join(altDataPath+mousename, 'imaging', imagingFolder)
            dataPath = altDataPath    
#        if not os.path.exists(tifFold):
#            if 'altDataPath' in locals():
#                tifFold = os.path.join(altDataPath+mousename, 'imaging', imagingFolder)
#                dataPath = altDataPath
#            else:
#                sys.exit('Data directory does not exist!')
    
        
    #    print mdfFileNumber, type(mdfFileNumber)
    #    mdfFileNumber = np.array(mdfFileNumber).astype('int')
    #    print mdfFileNumber, type(mdfFileNumber), np.shape(mdfFileNumber)
    #    print np.shape(mdfFileNumber)[0]
        r = '%03d-'*np.shape(mdfFileNumber)[0] #len(mdfFileNumber)
        r = r[:-1]
        rr = r % (tuple(mdfFileNumber))
        
        date_major = imagingFolder+'_'+rr
        imfilename = os.path.join(tifFold,date_major+'.mat')
        
        ##%%
        if signalCh>0:
            if postNProvided:
                pnevFileName = 'post_'+date_major+'_ch'+str(signalCh)+'-Pnev*'
            else:
                pnevFileName = date_major+'_ch'+str(signalCh)+'-Pnev*'
                
            pnevFileName = glob.glob(os.path.join(tifFold,pnevFileName))   
            # sort pnevFileNames by date (descending)
            pnevFileName = sorted(pnevFileName, key=os.path.getmtime)
            pnevFileName = pnevFileName[::-1] # so the latest file is the 1st one.
            '''
            array = []
            for idx in range(0, len(pnevFileName)):
                array.append(os.path.getmtime(pnevFileName[idx]))
    
            inds = np.argsort(array)
            inds = inds[::-1]
            pnev2load = inds[pnev2load]
            '''    
            if len(pnevFileName)==0:
                c = ("No Pnev file was found"); print("%s\n" % c)
                pnevFileName = ''
            else:
                pnevFileName = pnevFileName[pnev2load[0]]
                if postNProvided:
                    p = os.path.basename(pnevFileName)[5:]
                    pnevFileName = os.path.join(tifFold,p)
        else:
            pnevFileName = ''
        
        ##%%
#        if nOuts==2:
#            return imfilename, pnevFileName
#        else:
        return imfilename, pnevFileName, dataPath
 
    else:
        
        if any([mousename in s for s in ['fni16','fni17']]):
            imagingDir = os.path.join(dataPath+mousename,'imaging')
            
        elif any([mousename in s for s in ['fni18','fni19']]): # these 2 mice are saved on hpc_home
            imagingDir = os.path.join(altDataPath+mousename,'imaging')
            
        return imagingDir
        
        
        
#%% Define perClassError: percent difference between Y and Yhat, ie classification error

def perClassError(Y, Yhat):
    import numpy as np
    perClassEr = np.sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
    return perClassEr


#%% Deine prediction error for SVR .... 

def perClassError_sr(y,yhat):
    import numpy as np
#    err = (np.linalg.norm(yhat - y)**2)/len(y)
    err = np.linalg.norm(yhat - y)**2
    maxerr = np.linalg.norm(y+1e-10)**2 # Gamal:  try adding a small number just to make the expression stable in case y is zero    
#    maxerr = np.linalg.norm(y)**2
#    ce = err
    ce = err/ maxerr    
#    ce = np.linalg.norm(yhat - y)**2 / len(y)
    return ce

#since there is no class in svr, we need to change this mean to square error.
#eps0 = 10**-5
#def perClassError_sr(Y,Yhat,eps0=10**-5):    
#    ce = np.mean(np.logical_and(abs(Y-Yhat) > eps0 , ~np.isnan(Yhat - Y)))*100
#    return ce
  
    
#%% Function to predict class labels     
# Lets check how predict works.... Result: for both SVM and SVR, classfier.predict equals xw+b. For SVM, xw+b gives -1 and 1, that should be changed to 0 and 1 to match svm.predict.

def predictMan(X,w,b,th=0): # set th to nan if you are predicting on svr (since for svr we dont compare with a threshold)
    
    # np.dot(a,b) is a sum product over the last axis of a and the second-to-last axis of b:
    
    # X: trials x units
    # w: units x samples
    # b: samples
    """
    # if the above sizes are not the case, we will transpose matrices below to make the code work!    
    # make sure last dim of X is units
    xud = np.argwhere(np.in1d(X.shape, w.shape)).flatten() # dimenstion of X that corresponds to units
    xu = X.shape[xud] # number of units
    if xud!=np.ndim(X)-1:
        remd = np.delete(range(np.ndim(X)), xud)
        X = np.transpose(X, (int(remd), int(xud)))
            
    # make sure second to last dim of w is units
    if np.ndim(w)>1:
        wud = np.argwhere(np.in1d(w.shape, xu)).flatten() # dimenstion of w that corresponds to units
        if wud!=np.ndim(w)-2:
            remd = np.delete(range(np.ndim(w)), wud)
            w = np.transpose(w, (int(wud), int(remd)))
    """
    
    yhat = np.dot(X, w.T) + b # it gives -1 and 1... change -1 to 0... so it matches svm.predict
    
    if np.isnan(th)==0:
        yhat[yhat<th] = 0
        yhat[yhat>th] = 1    
    
    return yhat


#%% Function to only show left and bottom axes of plots, make tick directions outward, remove every other tick label if requested.

def makeNicePlots(ax, rmv2ndXtickLabel=0, rmv2ndYtickLabel=0):
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    # Make tick directions outward    
    ax.tick_params(direction='out')    
    # Tweak spacing between subplots to prevent labels from overlapping
    #plt.subplots_adjust(hspace=0.5)
#    ymin, ymax = ax.get_ylim()

    # Remove every other tick label
    if rmv2ndXtickLabel:
        [label.set_visible(False) for label in ax.xaxis.get_ticklabels()[::2]]
        
    if rmv2ndYtickLabel:
        [label.set_visible(False) for label in ax.yaxis.get_ticklabels()[::2]]
#        a = np.array(ax.yaxis.get_ticklabels())[np.arange(0,len(ax.yaxis.get_ticklabels()),2).astype(int).flatten()]
#        [label.set_visible(False) for label in a]
        
    # gap between tick labeles and axis
#    ax.tick_params(axis='x', pad=30)

#    plt.xticks(x, labels, rotation='vertical')
    #ax.xaxis.label.set_color('red')    
#    plt.gca().spines['left'].set_color('white')
    #plt.gca().yaxis.set_visible(False)


#%%    
     #import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)    
   


#%% smooth (got from internet)
    
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

   
#%% Plot histogram of verctors a and b on axes h1 and h2

def histerrbar(h1,h2,a,b,binEvery,p,lab,colors = ['g','k'],ylab='Fraction',lab1='exc',lab2='inh',plotCumsum=0,doSmooth=0):
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

    ################### hist
    # set bins
    bn = np.arange(np.min(np.concatenate((a,b))), np.max(np.concatenate((a,b))), binEvery)
    bn[-1] = np.max([np.max(a),np.max(b)])#+binEvery/10. # unlike digitize, histogram doesn't count the right most value
    
    # plt hist of a
    hist, bin_edges = np.histogram(a, bins=bn)
    hist = hist/float(np.sum(hist))    
    if plotCumsum:
        hist = np.cumsum(hist)
    if doSmooth!=0:        
        hist = smooth(hist, doSmooth)
    ax1 = plt.subplot(h1) #(gs[0,0:2])
    # plot the center of bins
    plt.plot(bin_edges[0:-1]+binEvery/2., hist, color=colors[0], label=lab1)    #    plt.bar(bin_edges[0:-1], hist, binEvery, color=colors[0], alpha=.4, label=lab1)    
#    plt.xscale('log')
    
    # plot his of b
    hist, bin_edges = np.histogram(b, bins=bn)
    hist = hist/float(np.sum(hist));     #d = stats.mode(np.diff(bin_edges))[0]/float(2)
    if plotCumsum:
        hist = np.cumsum(hist)
    if doSmooth!=0:
        hist = smooth(hist, doSmooth)
    plt.plot(bin_edges[0:-1]+binEvery/2., hist, color=colors[1], label=lab2)        #    plt.bar(bin_edges[0:-1], hist, binEvery, color=colors[1], alpha=.4, label=lab2)
#    plt.xscale('log')
    
    # set labels, etc
    yl = plt.gca().get_ylim()
    ry = np.diff(yl)
    plt.ylim([yl[0]-ry/20 , yl[1]])   
    #
    xl = plt.gca().get_xlim()
    rx = np.diff(xl)
    
    plt.xlim([xl[0]-rx/20 , xl[1]])   # comment this if you want to plot in log scale
            
    plt.legend(loc=0, frameon=False)
    plt.ylabel(ylab) #('Prob (all days & N shuffs at bestc)')
#    plt.title('mean diff= %.3f, p=%.3f' %(np.mean(a)-np.mean(b), p))
    plt.title('mean diff= %.3f' %(np.mean(a)-np.mean(b)))
    #plt.xlim([-.5,.5])
    plt.xlabel(lab)
    makeNicePlots(ax1,0,1)

    
    ################ errorbar: mean and st error
    ax2 = plt.subplot(h2) #(gs[0,2:3])
    plt.errorbar([0,1], [a.mean(),b.mean()], [a.std()/np.sqrt(len(a)), b.std()/np.sqrt(len(b))], marker='o',color='k', fmt='.')
    plt.xlim([-1,2])
#    plt.title('%.3f, %.3f' %(a.mean(), b.mean()))
    plt.xticks([0,1], (lab1, lab2), rotation='vertical')
    plt.ylabel(lab)
    plt.title('p=%.3f' %(p))
    makeNicePlots(ax2,0,1)
#    plt.tick_params
    yl = plt.gca().get_ylim()
    r = np.diff(yl)
    plt.ylim([yl[0], yl[1]+r/10.])
    
    plt.subplots_adjust(wspace=1, hspace=.5)
    return ax1,ax2


    
#%% each element of a and b contains vars of a days. Here we compute and plot ave and se across those vars vs. days
    
def errbarAllDays(a,b,p,gs,colors = ['g','k'],lab1='exc',lab2='inh',lab='ave(CA)', h1=[], h2=[]):
    
    if type(h1)==list: # h1 is not provided; otherwise h1 is provided as a matplotlib.gridspec.SubplotSpec
        h1 = gs[1,0:2]
    if type(h2)==list:
        h2 = gs[1,2:3]
    
    if np.ndim(a)==1:
        eav = [np.mean(a[i]) for i in range(len(a))] #np.nanmean(a, axis=1) # average across shuffles
        iav = [np.mean(b[i]) for i in range(len(b))] #np.nanmean(b, axis=1)
        ele = [len(a[i]) for i in range(len(a))] #np.shape(a)[1] - np.sum(np.isnan(a),axis=1) # number of non-nan shuffles of each day
        ile = [len(b[i]) for i in range(len(b))] #np.shape(b)[1] - np.sum(np.isnan(b),axis=1) # number of non-nan shuffles of each day
        esd = np.divide([np.std(a[i]) for i in range(len(a))], np.sqrt(ele))
        isd = np.divide([np.std(b[i]) for i in range(len(b))], np.sqrt(ile))
    else:
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
    
    ax1 = plt.subplot(h1)
    plt.errorbar(x, eav, esd, color=colors[0])
    plt.errorbar(x, iav, isd, color=colors[1])
    plt.plot(x, pp, marker='*',color='r', markeredgecolor='r', linestyle='', markersize=3)
    plt.xlim([-1, x[-1]+1])
    plt.xlabel('Days')
    plt.ylabel(lab)
    makeNicePlots(ax1,0,1)

    ax2 = plt.subplot(h2)
    plt.errorbar(0, np.nanmean(eav), np.nanstd(eav)/np.sqrt(len(eav)), marker='o', color='k', markeredgecolor='none')
    plt.errorbar(1, np.nanmean(iav), np.nanstd(iav)/np.sqrt(len(eav)), marker='o', color='k', markeredgecolor='none')
    plt.xticks([0,1], (lab1, lab2), rotation='vertical')
    plt.xlim([-1,2])
    makeNicePlots(ax2,0,1)

    _, p = stats.ttest_ind(eav, iav, nan_policy='omit')
    plt.title('p=%.3f' %(p))

    plt.subplots_adjust(wspace=1, hspace=.5)
    
    return ax1,ax2
    

#%% ave mice time course: Plot time course of whatever var u r interested in for exc and inh: take average across days for each mouse; then plot ave+/-se across mice

def plotTimecourse_avMice(time_al_final, avFRAllMs, daysDim, colors, labs, lab, alph=1, doErrbar=1, dnow0='', fnam='', tlab=''):
   
#    colors = ['b','r']; lab1='exc'; lab2='inh'; lab='Stability (deg.)'

    def avMice(avFRexc_allMice_al, daysDim):
        # average across days for each mouse
        avD_exc = np.array([np.nanmean(avFRexc_allMice_al[im],axis=daysDim) for im in range(len(avFRexc_allMice_al))]) # nMice x alignedFrames (across mice)
        # average and se across mice
        av_exc = np.nanmean(avD_exc,axis=0)
        se_exc = np.nanstd(avD_exc,axis=0)/np.sqrt(avD_exc.shape[0]) 
        
        return avD_exc, av_exc, se_exc
    
    ##################    
#    avFRexc_allMice_al, avFRinh_allMice_al
    #    plt.figure(figsize=(3,2))
    avD_excinh = []
    for i in range(len(avFRAllMs)):
        avD_exc, av_exc, se_exc = avMice(avFRAllMs[i], daysDim)
        avD_excinh.append(avD_exc)
#        avD_inh, av_inh, se_inh = avMice(avFRinh_allMice_al, daysDim)

        if doErrbar:    # plot error bar
            plt.errorbar(time_al_final, av_exc, se_exc, color=colors[i], label=labs[i], alpha=alph)
    #        plt.errorbar(time_al_final, av_inh, se_inh, color=colors[1], label=lab2, alpha=alph)
        else: # plot patches... boundedline...
            plt.fill_between(time_al_final, av_exc - se_exc, av_exc + se_exc, alpha=alph, edgecolor=colors[i], facecolor=colors[i])
            plt.plot(time_al_final, av_exc, colors[i], label=labs[i])
    #        plt.fill_between(time_al_final, av_inh - se_inh, av_inh + se_inh, alpha=0.5, edgecolor=colors[1], facecolor=colors[1])
    #        plt.plot(time_al_final, av_inh, colors[1], label=lab2)        
    
    
    
    if len(avFRAllMs)>1:
        _,p = stats.ttest_ind(avD_excinh[0], avD_excinh[1], nan_policy='omit')    
    
        yl = plt.gca().get_ylim()
        pp = p
        pp[p>.05] = np.nan
        pp[p<=.05] = yl[1]
        plt.plot(time_al_final, pp, marker='*',color='r', markeredgecolor='r', linestyle='', markersize=3)
    
    plt.ylabel(lab)        
    plt.xlabel('Time relative to choice onset')
    plt.title(tlab, position=[.5,1.05])
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)    
    makeNicePlots(plt.gca(),1,0)        
    

#    #% Save the figure           
#    if savefigs:
#        if chAl:
#            cha = 'chAl_'
#        else:
#            cha = 'stAl_'    
#            
#        d = os.path.join(svmdir+dnow0)        
#        if not os.path.exists(d):
#            print 'creating folder'
#            os.makedirs(d)
#        
#        fign = os.path.join(d, suffn[0:5]+cha+ fnam +'.'+fmt[0])
##        fnam = 'FR_timeCourse_aveMice_aveDays_allTrsNeursPooled'+dpp+dp+dpm0
#        plt.savefig(fign, bbox_inches='tight') 

       

#%% each mouse time course: Plot time course of whatever var u r interested in for exc and inh      

#time_al = time_aligned_final; avFRe = stabScoreExc0_al[im]; avFRi = stabScoreInh0_allMice_al[im]
def plotTimecourse_eachMouse(time_al, avFRs, daysDim, colors, labs, lab, linstyl='-', alph=1, dnow='', fnam='', tlab=''): 
    
    for i in range(len(avFRs)):    
        # average and se across days         
        avFR = avFRs[i]+0
        av = np.nanmean(avFR, axis=daysDim) 
        sd = np.nanstd(avFR, axis=daysDim)/np.sqrt(avFR.shape[daysDim])     
        plt.errorbar(time_al, av, sd, color=colors[i], label=labs[i], linestyle=linstyl, alpha=alph)
           

    if len(avFRs)>1: # plot p vals    
        if daysDim==0: # then frames are dim 1, so we transpose it to have frames as dim 0
            avFRs = avFRs[0].T, avFRs[1].T

        p = np.full((avFRs[0].shape[0]), np.nan)
        for fr in range(avFRs[0].shape[0]):
            _,p[fr] = stats.ttest_ind(avFRs[0][fr,:], avFRs[1][fr,:], nan_policy='omit')    
        
        yl = plt.gca().get_ylim()
        pp = p
        pp[p>.05] = np.nan
        pp[p<=.05] = yl[1]
        plt.plot(time_al, pp, marker='*',color='r', markeredgecolor='r', linestyle='', markersize=3)
        

    plt.ylabel(lab)        
    plt.xlabel('Time relative to choice onset')
    plt.title(tlab, position=[.5,1.05])
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)    
    makeNicePlots(plt.gca(),1,0)        
    
    
#    #% Save the figure           
#    if savefigs:
#        if chAl:
#            cha = 'chAl_'
#        else:
#            cha = 'stAl_'
#            
#        d = os.path.join(svmdir+dnow)        
#        if not os.path.exists(d):
#            print 'creating folder'
#            os.makedirs(d)
#        
#        fign = os.path.join(d, suffn[0:5]+cha+ fnam +'.'+fmt[0])
##        fnam = 'FR_timeCourse_aveDays_allTrsNeursPooled'+dp+dpmAllm[im]
#        plt.savefig(fign, bbox_inches='tight') 
        


#%%   
def setTo50classErr(classError, w, thNon0Ws = .05, thSamps = 10, eps = 1e-10):
#            classError = perClassErrorTest_data_inh 
#            w = w_data_inh
#            thNon0Ws = .05; thSamps = 10; eps = 1e-10
#            thNon0Ws = 2 # For samples with <2 non0 weights, we manually set their class error to 50 ... the idea is that bc of difference in number of HR and LR trials, in these samples class error is not accurately computed!
#            thSamps = 10  # Days that have <thSamps samples that satisfy >=thNon0W non0 weights will be manually set to 50 (class error of all their samples) ... bc we think <5 samples will not give us an accurate measure of class error of a day.

#    d = scio.loadmat(svmName, variable_names=[wname])
#    w = d.pop(wname)            
    a = abs(w) > eps # samps x neurons x frames # whether a weight is non0 or not
    # average abs(w) across neurons:
    if w.ndim==4: #exc : average across excShuffles
        a = np.mean(a, axis=0)
    aa = np.mean(a, axis=1) # samples x frames; shows fraction neurons with non0 weights for each sample and frame 
#    plt.imshow(aa), plt.colorbar() # good plot to visualize fraction non0 weights for each sammp at each frame
#    plt.imshow(classError); plt.colorbar() # compare the above figure with classErr at each frame and samp
    goodSamps = aa > thNon0Ws # samples x frames # samples with >.05 of neurons having non-0 weight 
#        sum(goodSamps) # for each frame, it shows number of samples with >.05 of neurons having non-0 weight        
    
    if sum(sum(~goodSamps))>0:
        print 'All frames together have %d bad samples (samples w < %.2f non-0 weights)... setting their classErr to 50' %(sum(sum(~goodSamps)), thNon0Ws)
        
        if sum(sum(goodSamps)<thSamps)>0:
            print 'There are %d frames with < %d good samples... setting all samples classErr of these frames to 50' %(sum(sum(goodSamps)<thSamps), thSamps)
    
        if w.ndim==3: 
            classError[~goodSamps] = 50 # set to 50 class error of samples which have <=.05 of non-0-weight neurons
            classError[:,sum(goodSamps)<thSamps] = 50 # if fewer than 10 samples contributed to a frame, set the perClassError of all samples for that frame to 50...       
        elif w.ndim==4: #exc : average across excShuffles
            classError[:,~goodSamps] = 50 # set to 50 class error of samples which have <=.05 of non-0-weight neurons
            classError[:,:,sum(goodSamps)<thSamps] = 50 # if fewer than 10 samples contributed to a frame, set the perClassError of all samples for that frame to 50...       
 
   
    modClassError = classError+0
    
    return modClassError
      
  
  
#%% function to set best c (svm already trained) and get the class error values at best C
# this function is a combination of the 2 functions below, where best c is set in a separate function and setting classErr at best c is done in a separate function too.
  
def setBestC_classErr(perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance, cvect, regType, doPlots=0, doIncorr=0, wAllC=[], bAllC=[], shflTrLabs=0):
    
    numSamples = perClassErrorTest.shape[0] 
    numFrs = perClassErrorTest.shape[2]

    #########################################################################################
    ################################ Set cbest for all frames ###############################
    #########################################################################################        
    if shflTrLabs==0: # if shflTrLabs=1, you have to load cbestFrs from the svm file ran without shuffling tr labels (shflTrLabs=0)        
        cbestFrs = np.full((numFrs), np.nan)
        for ifr in range(numFrs):                    
            ###%% Compute average of class errors across numSamples        
            meanPerClassErrorTrain = np.mean(perClassErrorTrain[:,:,ifr], axis = 0);
            semPerClassErrorTrain = np.std(perClassErrorTrain[:,:,ifr], axis = 0)/np.sqrt(numSamples);            
            meanPerClassErrorTest = np.mean(perClassErrorTest[:,:,ifr], axis = 0);
            semPerClassErrorTest = np.std(perClassErrorTest[:,:,ifr], axis = 0)/np.sqrt(numSamples);            
            meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl[:,:,ifr], axis = 0);
    #        semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl[:,:,ifr], axis = 0)/np.sqrt(numSamples);            
            meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance[:,:,ifr], axis = 0);
    #        semPerClassErrorTest_chance = np.std(perClassErrorTest_chance[:,:,ifr], axis = 0)/np.sqrt(numSamples);
            
            ###### Identify best c        
            # Use all range of c... it may end up a value at which all weights are 0.
            ix = np.argmin(meanPerClassErrorTest)
            if smallestC==1:
                cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
                cbest = cbest[0]; # best regularization term based on minError+SE criteria
                cbestAll = cbest
            else:
                cbestAll = cvect[ix]
            print 'best c = %.10f' %cbestAll
            
            ###### Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)
            if regType == 'l1': # in l2, we don't really have 0 weights!
                sys.exit('Needs work! below wAllC has to be for 1 frame') 
                
                a = abs(wAllC)>eps # non-zero weights
                b = np.mean(a, axis=(0,2,3)) # Fraction of non-zero weights (averaged across shuffles)
                c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle
                cvectnow = cvect[c1stnon0:]                
                meanPerClassErrorTestnow = np.mean(perClassErrorTest[:,c1stnon0:,ifr], axis = 0);
                semPerClassErrorTestnow = np.std(perClassErrorTest[:,c1stnon0:,ifr], axis = 0)/np.sqrt(numSamples);
                ix = np.argmin(meanPerClassErrorTestnow)
                if smallestC==1:
                    cbest = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
                    cbest = cbest[0]; # best regularization term based on minError+SE criteria    
                else:
                    cbest = cvectnow[ix]        
                print 'best c (at least 1 non-0 weight) = ', cbest                
            else:
                cbest = cbestAll            
        
            cbestFrs[ifr] = cbest
        
            #################### Plot C path                           
            if doPlots:
        #            print 'Best c (inverse of regularization parameter) = %.2f' %cbest
                plt.figure()
                plt.subplot(1,2,1)
                plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
                plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
            #    plt.fill_between(cvect, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
            #    plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
                
                plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
                plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation')
                plt.plot(cvect, meanPerClassErrorTest_chance, 'b', label = 'cv-chance')       
                plt.plot(cvect, meanPerClassErrorTest_shfl, 'y', label = 'cv-shfl')            
            
                plt.plot(cvect[cvect==cbest], meanPerClassErrorTest[cvect==cbest], 'bo')
                
                plt.xlim([cvect[1], cvect[-1]])
                plt.xscale('log')
                plt.xlabel('c (inverse of regularization parameter)')
                plt.ylabel('classification error (%)')
                plt.legend(loc='center left', bbox_to_anchor=(1, .7))
                
                cl = 'r' if ifr==eventI_ds else 'k' # show frame number of eventI in red :)
                plt.title('Frame %d' %(ifr), color=cl)
                plt.tight_layout()                 
        
        
        
    #########################################################################################        
    #################### Set w and class errors at best c for each frame #########################        
    #########################################################################################
    classErr_bestC_train_data = np.full((numSamples, numFrs), np.nan)
    classErr_bestC_test_data = np.full((numSamples, numFrs), np.nan)
    classErr_bestC_test_shfl = np.full((numSamples, numFrs), np.nan)
    classErr_bestC_test_chance = np.full((numSamples, numFrs), np.nan)
#    cbestFrs = np.full((numFrs), np.nan)
#    numNon0SampData = np.full((numFrs), np.nan)       
    classErr_bestC_test_incorr = np.full((numSamples, numFrs), np.nan)
    classErr_bestC_test_incorr_shfl = np.full((numSamples, numFrs), np.nan)
    classErr_bestC_test_incorr_chance = np.full((numSamples, numFrs), np.nan)
    if len(wAllC)>0:
        w_bestc_data = np.full((numSamples, wAllC.shape[2], numFrs), np.nan)
        b_bestc_data = np.full((numSamples, numFrs), np.nan)
    else:
        w_bestc_data = []
        b_bestc_data = []                

    for ifr in range(numFrs):
        if shflTrLabs:
            indBestC = 0
        else:
            indBestC = np.in1d(cvect, cbestFrs[ifr])
        # you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
        # we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
        classErr_bestC_train_data[:,ifr] = perClassErrorTrain[:,indBestC,ifr].squeeze() # numSamps           
        classErr_bestC_test_data[:,ifr] = perClassErrorTest[:,indBestC,ifr].squeeze()
        classErr_bestC_test_shfl[:,ifr] = perClassErrorTest_shfl[:,indBestC,ifr].squeeze()
        classErr_bestC_test_chance[:,ifr] = perClassErrorTest_chance[:,indBestC,ifr].squeeze()        
        if doIncorr==1: #'perClassErrorTest_incorr' in locals() and doIncorr==1:
            classErr_bestC_test_incorr[:,ifr] = perClassErrorTest_incorr[:,indBestC,ifr].squeeze()        
            classErr_bestC_test_incorr_shfl[:,ifr] = perClassErrorTest_incorr_shfl[:,indBestC,ifr].squeeze()        
            classErr_bestC_test_incorr_chance[:,ifr] = perClassErrorTest_incorr_chance[:,indBestC,ifr].squeeze()                
        if len(wAllC)>0:
            w_bestc_data[:,:,ifr] = wAllC[:,indBestC,:,ifr].squeeze() # numSamps x neurons
            b_bestc_data[:,ifr] = bAllC[:,indBestC,ifr].squeeze() # numSamps           
       # check the number of non-0 weights
    #       for ifr in range(numFrs):
    #        w_data = wAllC[:,indBestC,:,ifr].squeeze()
    #        print np.sum(w_data > eps,axis=1)
    #        ada = np.sum(w_data > eps,axis=1) < thNon0Ws # samples w fewer than 2 non-0 weights        
    #        ada = ~ada # samples w >=2 non0 weights
    #        numNon0SampData[ifr] = ada.sum() # number of cv samples with >=2 non-0 weights
    
           
    if doIncorr==1: #'perClassErrorTest_incorr' in locals():
        return classErr_bestC_train_data, classErr_bestC_test_data, classErr_bestC_test_shfl, classErr_bestC_test_chance, cbestFrs, w_bestc_data, b_bestc_data, classErr_bestC_test_incorr, classErr_bestC_test_incorr_shfl, classErr_bestC_test_incorr_chance
    else:
        return classErr_bestC_train_data, classErr_bestC_test_data, classErr_bestC_test_shfl, classErr_bestC_test_chance, cbestFrs, w_bestc_data, b_bestc_data
    
    
    
#%% function to find best c from testing class error ran on values of cvect

def findBestC(perClassErrorTest, cvect, regType, smallestC=1):
    
    numSamples = perClassErrorTest.shape[0] 
    numFrs = perClassErrorTest.shape[2]
    
    ###%% Compute average of class errors across numSamples
    
    cbestFrs = np.full((numFrs), np.nan)
       
    for ifr in range(numFrs):
        
        meanPerClassErrorTest = np.mean(perClassErrorTest[:,:,ifr], axis = 0);
        semPerClassErrorTest = np.std(perClassErrorTest[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
       
        ###%% Identify best c       
        
        # Use all range of c... it may end up a value at which all weights are 0.
        ix = np.argmin(meanPerClassErrorTest)
        if smallestC==1:
            cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
            cbest = cbest[0]; # best regularization term based on minError+SE criteria
            cbestAll = cbest
        else:
            cbestAll = cvect[ix]
        print 'best c = %.10f' %cbestAll
        
        #### Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)
        if regType == 'l1': # in l2, we don't really have 0 weights!
            sys.exit('Needs work! below wAllC has to be for 1 frame') 
            
            a = abs(wAllC)>eps # non-zero weights
            b = np.mean(a, axis=(0,2,3)) # Fraction of non-zero weights (averaged across shuffles)
            c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle
            cvectnow = cvect[c1stnon0:]
            
            meanPerClassErrorTestnow = np.mean(perClassErrorTest[:,c1stnon0:,ifr], axis = 0);
            semPerClassErrorTestnow = np.std(perClassErrorTest[:,c1stnon0:,ifr], axis = 0)/np.sqrt(numSamples);
            ix = np.argmin(meanPerClassErrorTestnow)
            if smallestC==1:
                cbest = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
                cbest = cbest[0]; # best regularization term based on minError+SE criteria    
            else:
                cbest = cvectnow[ix]
    
            print 'best c (at least 1 non-0 weight) = ', cbest            
        else:
            cbest = cbestAll            
        
        
        cbestFrs[ifr] = cbest
#        indBestC = np.in1d(cvect, cbest)      
    return cbestFrs
    
    
    
    
#%% function to set the class error values (and W) at best C (svm is already trained and we have the class errors)

import numpy as np
def setClassErrAtBestc(cbestFrs, cvect, doPlots, perClassErrorTrain=np.nan, perClassErrorTest=np.nan, perClassErrorTest_shfl=np.nan, perClassErrorTest_chance=np.nan, wAllC=np.nan, bAllC=np.nan):
    import numpy as np
    '''
    if np.isnan(bAllC).all():
        numSamples = perClassErrorTest.shape[0] 
        numFrs = perClassErrorTest.shape[2]
    else:
        numSamples = bAllC.shape[0] 
        numFrs = bAllC.shape[2]
        classErr_bestC_train_data = []
        classErr_bestC_test_data = []
        classErr_bestC_test_shfl = []
        classErr_bestC_test_chance = []
            
    if np.isnan(perClassErrorTest).all():
        numSamples = bAllC.shape[0] 
        numFrs = bAllC.shape[2]
    else:
        numSamples = perClassErrorTest.shape[0] 
        numFrs = perClassErrorTest.shape[2]        
        w_bestc_data = [] 
        b_bestc_data = []
    '''        
        
    if ~np.isnan(bAllC).all(): # bAllC is valid
        numSamples = bAllC.shape[0] 
        numFrs = bAllC.shape[2]
            
    elif ~np.isnan(perClassErrorTest).all(): # perClassErrorTest is valid
        numSamples = perClassErrorTest.shape[0] 
        numFrs = perClassErrorTest.shape[2]        


    if np.isnan(bAllC).all(): # bAllC is nan
        w_bestc_data = [] 
        b_bestc_data = []    

    if np.isnan(perClassErrorTest).all(): # perClassErrorTest is nan
        classErr_bestC_train_data = []
        classErr_bestC_test_data = []
        classErr_bestC_test_shfl = []
        classErr_bestC_test_chance = []

    
    ####%% Compute average of class errors across numSamples
    '''
    if np.isnan(perClassErrorTest).all():
        w_bestc_data = np.full((numSamples, wAllC.shape[2], numFrs), np.nan)
        b_bestc_data = np.full((numSamples, numFrs), np.nan)
    
    if np.isnan(bAllC).all():    
        classErr_bestC_train_data = np.full((numSamples, numFrs), np.nan)
        classErr_bestC_test_data = np.full((numSamples, numFrs), np.nan)
        classErr_bestC_test_shfl = np.full((numSamples, numFrs), np.nan)
        classErr_bestC_test_chance = np.full((numSamples, numFrs), np.nan)
    '''
#    cbestFrs = np.full((numFrs), np.nan)
#    numNon0SampData = np.full((numFrs), np.nan)       
       
    if ~np.isnan(perClassErrorTest).all():  # perClassErrorTest is valid
        classErr_bestC_train_data = np.full((numSamples, numFrs), np.nan)
        classErr_bestC_test_data = np.full((numSamples, numFrs), np.nan)
        classErr_bestC_test_shfl = np.full((numSamples, numFrs), np.nan)
        classErr_bestC_test_chance = np.full((numSamples, numFrs), np.nan)
    
    if ~np.isnan(bAllC).all():  # bAllC is valid  
        w_bestc_data = np.full((numSamples, wAllC.shape[2], numFrs), np.nan)
        b_bestc_data = np.full((numSamples, numFrs), np.nan)

       
       
    ####%%        
    for ifr in range(numFrs):
        
        ####%% Set the decoder and class errors at best c         
        cbest = cbestFrs[ifr]        
                
        # you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
        # we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
        indBestC = np.in1d(cvect, cbest)

        if ~np.isnan(bAllC).all():  # bAllC is valid   # np.isnan(perClassErrorTest).all():        
            w_bestc_data[:,:,ifr] = wAllC[:,indBestC,:,ifr].squeeze() # numSamps x neurons
            b_bestc_data[:,ifr] = bAllC[:,indBestC,ifr].squeeze()           
     
     
        if ~np.isnan(perClassErrorTest).all():  # perClassErrorTest is valid  # np.isnan(bAllC).all():  
            classErr_bestC_train_data[:,ifr] = perClassErrorTrain[:,indBestC,ifr].squeeze() # numSamps           
            classErr_bestC_test_data[:,ifr] = perClassErrorTest[:,indBestC,ifr].squeeze()
            classErr_bestC_test_shfl[:,ifr] = perClassErrorTest_shfl[:,indBestC,ifr].squeeze()
            classErr_bestC_test_chance[:,ifr] = perClassErrorTest_chance[:,indBestC,ifr].squeeze()        
    
       # check the number of non-0 weights
    #       for ifr in range(numFrs):
    #        w_data = wAllC[:,indBestC,:,ifr].squeeze()
    #        print np.sum(w_data > eps,axis=1)
    #        ada = np.sum(w_data > eps,axis=1) < thNon0Ws # samples w fewer than 2 non-0 weights        
    #        ada = ~ada # samples w >=2 non0 weights
    #        numNon0SampData[ifr] = ada.sum() # number of cv samples with >=2 non-0 weights
    
        
        ####%% Plot C path           
        
        if doPlots:
            meanPerClassErrorTrain = np.mean(perClassErrorTrain[:,:,ifr], axis = 0);
            semPerClassErrorTrain = np.std(perClassErrorTrain[:,:,ifr], axis = 0)/np.sqrt(numSamples);
            
            meanPerClassErrorTest = np.mean(perClassErrorTest[:,:,ifr], axis = 0);
            semPerClassErrorTest = np.std(perClassErrorTest[:,:,ifr], axis = 0)/np.sqrt(numSamples);
            
            meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl[:,:,ifr], axis = 0);
            semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl[:,:,ifr], axis = 0)/np.sqrt(numSamples);
            
            meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance[:,:,ifr], axis = 0);
            semPerClassErrorTest_chance = np.std(perClassErrorTest_chance[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
    #            print 'Best c (inverse of regularization parameter) = %.2f' %cbest
            plt.figure()
            plt.subplot(1,2,1)
            plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
            plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
        #    plt.fill_between(cvect, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
        #    plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
            
            plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
            plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation')
            plt.plot(cvect, meanPerClassErrorTest_chance, 'b', label = 'cv-chance')       
            plt.plot(cvect, meanPerClassErrorTest_shfl, 'y', label = 'cv-shfl')            
        
            plt.plot(cvect[cvect==cbest], meanPerClassErrorTest[cvect==cbest], 'bo')
            
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('classification error (%)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            
            cl = 'r' if ifr==eventI_ds else 'k' # show frame number of eventI in red :)
            plt.title('Frame %d' %(ifr), color=cl)
            plt.tight_layout()          
            
           
    return classErr_bestC_train_data, classErr_bestC_test_data, classErr_bestC_test_shfl, classErr_bestC_test_chance, w_bestc_data, b_bestc_data
        
        
        
#%% Train SVM and set bestc

####%%##### Function for identifying the best regularization parameter
#     Perform 10-fold cross validation to obtain the best regularization parameter
#         More specifically: "crossValidateModel" divides data into training and test datasets. It calls linearSVM.py, which does linear SVM using XTrain, and returns percent class loss for XTrain and XTest.
#     This procedure gets repeated for numSamples (100 times) for each value of regulariazation parameter. 
#     An average across all 100 samples is computed to find the minimum test class loss.
#     Best regularization parameter is defined as the smallest regularization parameter whose test-dataset class loss is within mean+sem of minimum test class loss.

# X is trials x units
def setbesc(X,Y,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs):
    import numpy as np
    from crossValidateModel import crossValidateModel
    from linearSVM import linearSVM
    # numSamples = 10; # number of iterations for finding the best c (inverse of regularization parameter)
    # if you don't want to regularize, go with a very high cbest and don't run the section below.
    # cbest = 10**6
    
    # regType = 'l1'
    # kfold = 10;
    if regType == 'l1':
        print '\nRunning l1 svm classification\r' 
        # cvect = 10**(np.arange(-4, 6,0.2))/numTrials;
        cvect = 10**(np.arange(-4, 6,0.2))/numDataPoints;
    elif regType == 'l2':
        print '\nRunning l2 svm classification\r' 
        cvect = 10**(np.arange(-6, 6,0.2))/numDataPoints;
    
    print 'try the following regularization values: \n', cvect
    # formattedList = ['%.2f' % member for member in cvect]
    # print 'try the following regularization values = \n', formattedList
    
    wAllC = np.ones((numSamples, len(cvect), X.shape[1]))+np.nan;
    bAllC = np.ones((numSamples, len(cvect)))+np.nan;
    
    perClassErrorTrain = np.ones((numSamples, len(cvect)))+np.nan;
    perClassErrorTest = np.ones((numSamples, len(cvect)))+np.nan;
    
    perClassErrorTest_shfl = np.ones((numSamples, len(cvect)))+np.nan;
    perClassErrorTest_chance = np.ones((numSamples, len(cvect)))+np.nan
    
    
    ##############
    hrn = (Y==1).sum()
    lrn = (Y==0).sum()    
    
    if useEqualTrNums and hrn!=lrn: # if the HR and LR trials numbers are not the same, pick equal number of trials of the 2 classes!        
        trsn = min(lrn,hrn)
        if hrn > lrn:
            print 'Subselecting HR trials so both classes have the same number of trials!'
        elif lrn > hrn:
            print 'Subselecting LR trials so both classes have the same number of trials!'
                
    X0 = X + 0
    Y0 = Y + 0


                
    ############## Train SVM numSamples times to get numSamples cross-validated datasets.
    for s in range(numSamples):        
        print 'Iteration %d' %(s)
        
        ############ Make sure both classes have the same number of trials when training the classifier
        if useEqualTrNums and hrn!=lrn: # if the HR and LR trials numbers are not the same, pick equal number of trials of the 2 classes!                    
            if hrn > lrn:
                randtrs = np.argwhere(Y0==1)[rng.permutation(hrn)[0:trsn]].squeeze()
                trsnow = np.sort(np.concatenate((randtrs , np.argwhere(Y0==0).squeeze())))
            elif lrn > hrn:
                randtrs = np.argwhere(Y0==0)[rng.permutation(lrn)[0:trsn]].squeeze() # random sample of the class with more trials
                trsnow = np.sort(np.concatenate((randtrs , np.argwhere(Y0==1).squeeze()))) # all trials of the class with fewer trials + the random sample set above for the other class
                            
            X = X0[trsnow,:]
            Y = Y0[trsnow]
      
            numTrials, numNeurons = X.shape[0], X.shape[1]
            print 'FINAL: %d trials; %d neurons' %(numTrials, numNeurons)                
        
        ######################## Setting chance Y        
        no = Y.shape[0]
        len_test = no - int((kfold-1.)/kfold*no)    
        permIxs = rng.permutation(len_test)    
    
        Y_chance = np.zeros(len_test)
        if rng.rand()>.5:
            b = rng.permutation(len_test)[0:np.floor(len_test/float(2)).astype(int)]
        else:
            b = rng.permutation(len_test)[0:np.ceil(len_test/float(2)).astype(int)]
        Y_chance[b] = 1



        ######################## Loop over different values of regularization
        for i in range(len(cvect)): # train SVM using different values of regularization parameter
            
            if regType == 'l1':                       
                summary, shfl =  crossValidateModel(X, Y, linearSVM, kfold = kfold, l1 = cvect[i], shflTrs = shuffleTrs)
            elif regType == 'l2':
                summary, shfl =  crossValidateModel(X, Y, linearSVM, kfold = kfold, l2 = cvect[i], shflTrs = shuffleTrs)
                        
            wAllC[s,i,:] = np.squeeze(summary.model.coef_); # weights of all neurons for each value of c and each shuffle
            bAllC[s,i] = np.squeeze(summary.model.intercept_);
    
            # classification errors                    
            perClassErrorTrain[s,i] = summary.perClassErrorTrain;
            perClassErrorTest[s,i] = summary.perClassErrorTest;                
            
            # Testing correct shuffled data: 
            # same decoder trained on correct trials, make predictions on correct with shuffled labels.
            perClassErrorTest_shfl[s,i] = perClassError(summary.YTest[permIxs], summary.model.predict(summary.XTest));
            perClassErrorTest_chance[s,i] = perClassError(Y_chance, summary.model.predict(summary.XTest));
            
    
    

    ######################### Find bestc for each frame, and plot the c path    
        
    #######%% Compute average of class errors across numSamples        
    meanPerClassErrorTrain = np.mean(perClassErrorTrain, axis = 0);
    semPerClassErrorTrain = np.std(perClassErrorTrain, axis = 0)/np.sqrt(numSamples);
    
    meanPerClassErrorTest = np.mean(perClassErrorTest, axis = 0);
    semPerClassErrorTest = np.std(perClassErrorTest, axis = 0)/np.sqrt(numSamples);
    
    meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl, axis = 0);
    semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl, axis = 0)/np.sqrt(numSamples);
    
    meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance, axis = 0);
    semPerClassErrorTest_chance = np.std(perClassErrorTest_chance, axis = 0)/np.sqrt(numSamples);
    
    
    #######%% Identify best c        
#        smallestC = 0 # if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
    if smallestC==1:
        print 'bestc = smallest c whose cv error is less than 1se of min cv error'
    else:
        print 'bestc = c that gives min cv error'
    #I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.
    
    
    # Use all range of c... it may end up a value at which all weights are 0.
    ix = np.argmin(meanPerClassErrorTest)
    if smallestC==1:
        cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
        cbest = cbest[0]; # best regularization term based on minError+SE criteria
        cbestAll = cbest
    else:
        cbestAll = cvect[ix]
    print '\t = ', cbestAll
    
    
    ####### Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)
    if regType == 'l1': # in l2, we don't really have 0 weights!
        sys.exit('Needs work! below wAllC has to be for 1 frame') 
        
        a = abs(wAllC)>eps # non-zero weights
        b = np.mean(a, axis=(0,2,3)) # Fraction of non-zero weights (averaged across shuffles)
        c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle
        cvectnow = cvect[c1stnon0:]
        
        meanPerClassErrorTestnow = np.mean(perClassErrorTest[:,c1stnon0:], axis = 0);
        semPerClassErrorTestnow = np.std(perClassErrorTest[:,c1stnon0:], axis = 0)/np.sqrt(numSamples);
        ix = np.argmin(meanPerClassErrorTestnow)
        if smallestC==1:
            cbest = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
            cbest = cbest[0]; # best regularization term based on minError+SE criteria    
        else:
            cbest = cvectnow[ix]
        
        print 'best c (at least 1 non-0 weight) = ', cbest
    else:
        cbest = cbestAll
            
    
    ########%% Set the decoder and class errors at best c (for data)
    """
    # you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
    # we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
    indBestC = np.in1d(cvect, cbest)
    
    w_bestc_data = wAllC[:,indBestC,:,ifr].squeeze() # numSamps x neurons
    b_bestc_data = bAllC[:,indBestC,ifr]
    
    classErr_bestC_train_data = perClassErrorTrain[:,indBestC,ifr].squeeze()
    
    classErr_bestC_test_data = perClassErrorTest[:,indBestC,ifr].squeeze()
    classErr_bestC_test_shfl = perClassErrorTest_shfl[:,indBestC,ifr].squeeze()
    classErr_bestC_test_chance = perClassErrorTest_chance[:,indBestC,ifr].squeeze()
    """
    
    
    ########### Plot C path    
    if doPlots:              
#        print 'Best c (inverse of regularization parameter) = %.2f' %cbest
        plt.figure()
        plt.subplot(1,2,1)
        plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
        plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
    #    plt.fill_between(cvect, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
    #    plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
        
        plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
        plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation')
        plt.plot(cvect, meanPerClassErrorTest_chance, 'b', label = 'cv-chance')       
        plt.plot(cvect, meanPerClassErrorTest_shfl, 'y', label = 'cv-shfl')            
    
        plt.plot(cvect[cvect==cbest], meanPerClassErrorTest[cvect==cbest], 'bo')
        
        plt.xlim([cvect[1], cvect[-1]])
        plt.xscale('log')
        plt.xlabel('c (inverse of regularization parameter)')
        plt.ylabel('classification error (%)')
        plt.legend(loc='center left', bbox_to_anchor=(1, .7))
        
#        plt.title('Frame %d' %(ifr))
        plt.tight_layout()
    
    

    ##############
    return perClassErrorTrain, perClassErrorTest, wAllC, bAllC, cbestAll, cbest, cvect, perClassErrorTest_shfl, perClassErrorTest_chance







#%% Train SVM and set best c; Same as above but when X is frames x units x trials
# Remember each numSamples will have a different set of training and testing dataset, however for each numSamples, the same set of testing/training dataset
# will be used for all frames and all values of c (unless shuffleTrs is 1, in which case different frames and c values will have different training/testing datasets.)

def setbesc_frs(X,Y,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs):
    
    import numpy as np
    import numpy.random as rng
    from crossValidateModel import crossValidateModel
    from linearSVM import linearSVM
    
    def perClassError(Y, Yhat):
#        import numpy as np
        perClassEr = np.sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
        return perClassEr
    # numSamples = 10; # number of iterations for finding the best c (inverse of regularization parameter)
    # if you don't want to regularize, go with a very high cbest and don't run the section below.
    # cbest = 10**6
    
    # regType = 'l1'
    # kfold = 10;
        
    if regType == 'l1':
        print '\n-------------- Running l1 svm classification --------------\r' 
        # cvect = 10**(np.arange(-4, 6,0.2))/numTrials;
        cvect = 10**(np.arange(-4, 6,0.2))/numDataPoints;
    elif regType == 'l2':
        print '\n-------------- Running l2 svm classification --------------\r' 
        cvect = 10**(np.arange(-6, 6,0.2))/numDataPoints;    
#    print 'try the following regularization values: \n', cvect
    # formattedList = ['%.2f' % member for member in cvect]
    # print 'try the following regularization values = \n', formattedList
    
#    smallestC = 0 # if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
    if smallestC==1:
        print 'bestc = smallest c whose cv error is less than 1se of min cv error'
    else:
        print 'bestc = c that gives min cv error'
    #I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.
        
        
        
    wAllC = np.ones((numSamples, len(cvect), X.shape[1], np.shape(X)[0]))+np.nan;
    bAllC = np.ones((numSamples, len(cvect), np.shape(X)[0]))+np.nan;
    
    perClassErrorTrain = np.ones((numSamples, len(cvect), np.shape(X)[0]))+np.nan;
    perClassErrorTest = np.ones((numSamples, len(cvect), np.shape(X)[0]))+np.nan;
    
    perClassErrorTest_shfl = np.ones((numSamples, len(cvect), np.shape(X)[0]))+np.nan;
    perClassErrorTest_chance = np.ones((numSamples, len(cvect), np.shape(X)[0]))+np.nan
    
    
    ##############
    hrn = (Y==1).sum()
    lrn = (Y==0).sum()    
    
    if useEqualTrNums and hrn!=lrn: # if the HR and LR trials numbers are not the same, pick equal number of trials of the 2 classes!        
        trsn = min(lrn,hrn)
        if hrn > lrn:
            print 'Subselecting HR trials so both classes have the same number of trials!'
            n = lrn*2
        elif lrn > hrn:
            print 'Subselecting LR trials so both classes have the same number of trials!'
            n = lrn*2
        print 'FINAL: %d trials; %d neurons' %(n, X.shape[1])
            
    X0 = X + 0
    Y0 = Y + 0
    

                
    ############## Train SVM numSamples times to get numSamples cross-validated datasets.
    for s in range(numSamples):        
        print 'Iteration %d' %(s)
        
        ############ Make sure both classes have the same number of trials when training the classifier
        if useEqualTrNums and hrn!=lrn: # if the HR and LR trials numbers are not the same, pick equal number of trials of the 2 classes!                    
            if hrn > lrn:
                randtrs = np.argwhere(Y0==1)[rng.permutation(hrn)[0:trsn]].squeeze()
                trsnow = np.sort(np.concatenate((randtrs , np.argwhere(Y0==0).squeeze())))
            elif lrn > hrn:
                randtrs = np.argwhere(Y0==0)[rng.permutation(lrn)[0:trsn]].squeeze() # random sample of the class with more trials
                trsnow = np.sort(np.concatenate((randtrs , np.argwhere(Y0==1).squeeze()))) # all trials of the class with fewer trials + the random sample set above for the other class
                            
            X = X0[:,:,trsnow]
            Y = Y0[trsnow]
      
            numTrials, numNeurons = X.shape[2], X.shape[1]
#            print 'FINAL: %d trials; %d neurons' %(numTrials, numNeurons)                
        
        
        ######################## Shuffle trial orders, so the training and testing datasets are different for each numSamples (we only do this if shuffleTrs is 0, so crossValidateModel does not shuffle trials, so we have to do it here, otherwise all numSamples will have the same set of testing and training datasets.)
        if shuffleTrs==0: # shuffle trials to break any dependencies on the sequence of trails 
            shfl = rng.permutation(np.arange(0, numTrials));
            Y = Y[shfl];
            X = X[:,:,shfl]; 

            
        ######################## Setting chance Y: same length as Y for testing data, and with equal number of classes 0 and 1.
        no = Y.shape[0]
        len_test = no - int((kfold-1.)/kfold*no)    
        permIxs = rng.permutation(len_test)    
    
        Y_chance = np.zeros(len_test)
        if rng.rand()>.5:
            b = rng.permutation(len_test)[0:np.floor(len_test/float(2)).astype(int)]
        else:
            b = rng.permutation(len_test)[0:np.ceil(len_test/float(2)).astype(int)]
        Y_chance[b] = 1



        ############# Start training SVM
        for ifr in range(X.shape[0]): # train SVM on each frame
            print '\tFrame %d' %(ifr)  
            ######################## Loop over different values of regularization
            for i in range(len(cvect)): # train SVM using different values of regularization parameter
                
                if regType == 'l1':                       
                    summary,_ =  crossValidateModel(X[ifr,:,:].transpose(), Y, linearSVM, kfold = kfold, l1 = cvect[i], shflTrs = shuffleTrs)
                elif regType == 'l2':
                    summary,_ =  crossValidateModel(X[ifr,:,:].transpose(), Y, linearSVM, kfold = kfold, l2 = cvect[i], shflTrs = shuffleTrs)
                            
                wAllC[s,i,:,ifr] = np.squeeze(summary.model.coef_); # weights of all neurons for each value of c and each shuffle
                bAllC[s,i,ifr] = np.squeeze(summary.model.intercept_);
        
                # classification errors                    
                perClassErrorTrain[s,i,ifr] = summary.perClassErrorTrain;
                perClassErrorTest[s,i,ifr] = summary.perClassErrorTest;                
                
                # Testing correct shuffled data: 
                # same decoder trained on correct trials, make predictions on correct with shuffled labels.
                perClassErrorTest_shfl[s,i] = perClassError(summary.YTest[permIxs], summary.model.predict(summary.XTest));
                perClassErrorTest_chance[s,i] = perClassError(Y_chance, summary.model.predict(summary.XTest));
            
    
    

    ######################### Find bestc for each frame, and plot the c path 
    print '--------------- Identifying best c ---------------' 
    cbestFrs = np.full((X.shape[0]), np.nan)  
    cbestAllFrs = np.full((X.shape[0]), np.nan)  
    for ifr in range(X.shape[0]):    
        #######%% Compute average of class errors across numSamples        
        meanPerClassErrorTrain = np.mean(perClassErrorTrain[:,:,ifr], axis = 0);
        semPerClassErrorTrain = np.std(perClassErrorTrain[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest = np.mean(perClassErrorTest[:,:,ifr], axis = 0);
        semPerClassErrorTest = np.std(perClassErrorTest[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl[:,:,ifr], axis = 0);
        semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance[:,:,ifr], axis = 0);
        semPerClassErrorTest_chance = np.std(perClassErrorTest_chance[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        
        #######%% Identify best c                
        # Use all range of c... it may end up a value at which all weights are 0.
        ix = np.argmin(meanPerClassErrorTest)
        if smallestC==1:
            cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
            cbest = cbest[0]; # best regularization term based on minError+SE criteria
            cbestAll = cbest
        else:
            cbestAll = cvect[ix]
        print '\tFrame %d: %f' %(ifr,cbestAll)
        cbestAllFrs[ifr] = cbestAll
        
        ####### Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)
        if regType == 'l1': # in l2, we don't really have 0 weights!
            sys.exit('Needs work! below wAllC has to be for 1 frame') 
            
            a = abs(wAllC)>eps # non-zero weights
            b = np.mean(a, axis=(0,2,3)) # Fraction of non-zero weights (averaged across shuffles)
            c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle
            cvectnow = cvect[c1stnon0:]
            
            meanPerClassErrorTestnow = np.mean(perClassErrorTest[:,c1stnon0:,ifr], axis = 0);
            semPerClassErrorTestnow = np.std(perClassErrorTest[:,c1stnon0:,ifr], axis = 0)/np.sqrt(numSamples);
            ix = np.argmin(meanPerClassErrorTestnow)
            if smallestC==1:
                cbest = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
                cbest = cbest[0]; # best regularization term based on minError+SE criteria    
            else:
                cbest = cvectnow[ix]
            
            print 'best c (at least 1 non-0 weight) = ', cbest
        else:
            cbest = cbestAll
                
        cbestFrs[ifr] = cbest
        
        
        ########%% Set the decoder and class errors at best c (for data)
        """
        # you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
        # we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
        indBestC = np.in1d(cvect, cbest)
        
        w_bestc_data = wAllC[:,indBestC,:,ifr].squeeze() # numSamps x neurons
        b_bestc_data = bAllC[:,indBestC,ifr]
        
        classErr_bestC_train_data = perClassErrorTrain[:,indBestC,ifr].squeeze()
        
        classErr_bestC_test_data = perClassErrorTest[:,indBestC,ifr].squeeze()
        classErr_bestC_test_shfl = perClassErrorTest_shfl[:,indBestC,ifr].squeeze()
        classErr_bestC_test_chance = perClassErrorTest_chance[:,indBestC,ifr].squeeze()
        """
        
        
        ########### Plot C path    
        if doPlots:              
    #        print 'Best c (inverse of regularization parameter) = %.2f' %cbest
            plt.figure()
            plt.subplot(1,2,1)
            plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
            plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
        #    plt.fill_between(cvect, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
        #    plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
            
            plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
            plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation')
            plt.plot(cvect, meanPerClassErrorTest_chance, 'b', label = 'cv-chance')       
            plt.plot(cvect, meanPerClassErrorTest_shfl, 'y', label = 'cv-shfl')            
        
            plt.plot(cvect[cvect==cbest], meanPerClassErrorTest[cvect==cbest], 'bo')
            
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('classification error (%)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            
            plt.title('Frame %d' %(ifr))
            plt.tight_layout()
    
    

    ##############
    return perClassErrorTrain, perClassErrorTest, wAllC, bAllC, cbestAllFrs, cbestFrs, cvect, perClassErrorTest_shfl, perClassErrorTest_chance



#%% Change color order to jet 

def colorOrder(nlines=30):
    ##%% Define a colormap   
    from numpy import linspace
    from matplotlib import cm
    cmtype = cm.jet # jet; what kind of colormap?
    
    start = 0.0
    stop = 1.0
    number_of_lines = nlines #len(days)
    cm_subsection = linspace(start, stop, number_of_lines) 
    colorsm = [ cmtype(x) for x in cm_subsection ]
    
    #% Change color order to jet 
#    from cycler import cycler
#    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
    
#    a = plt.scatter(y, y2, c=np.arange(len(y)), cmap=cm.jet, edgecolors='face')#, label='class accuracy (% correct testing trials)')
            
    return colorsm


#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName

def setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc=[], regressBins=3, useEqualTrNums=1, corrTrained=0, shflTrsEachNeuron=0, shflTrLabs=0):
    import glob
    import sys
    
    if len(doInhAllexcEqexc)==4:
        if doInhAllexcEqexc[3]==-1:
            if shflTrsEachNeuron:
                testIncorr = 0
            else:
                testIncorr = 1
            decodeStimCateg = 0
            addNs_rand = 0
            addNs_roc = 0        
        elif doInhAllexcEqexc[3]==0:
            decodeStimCateg = 1        
            addNs_rand = 0
            addNs_roc = 0
            corrTrained = 0
            testIncorr = 0
#            print 'Decoding stimulus category (HR, LR)'
        elif doInhAllexcEqexc[3]==3:
            addNs_rand = 1        
            addNs_roc = 0
            decodeStimCateg = 0
            testIncorr = 0
#            print 'Adding neurons randomly 1 by 1 for SVM analysis'
        else: 
            addNs_roc = 1
            addNs_rand = 0
            decodeStimCateg = 0
            testIncorr = 0
#            if doInhAllexcEqexc[3]==1:
#                print 'Adding neurons 1 by 1 from high to low ROC (choice tuning) for SVM analysis' 
#            elif doInhAllexcEqexc[3]==2:
#                print 'Adding neurons 1 by 1 from low to high ROC (choice tuning) for SVM analysis' 
    else:
        addNs_rand = 0
        addNs_roc = 0
        decodeStimCateg = 0
        testIncorr = 0
#        print 'Decoding left, right choice'
 
    
    if addNs_roc:
        diffn = 'diffNumNsROC_'
        if doInhAllexcEqexc[3]==1:
            h2l = '' #'hi2loROC_'
        else:
            h2l = 'lo2hiROC_'        
    elif addNs_rand:        
        diffn = 'diffNumNsRand_'
        h2l = ''
    elif decodeStimCateg:
        diffn = 'decodeStimCateg_'
        h2l = ''
    else:
        diffn = ''
        h2l = ''
        
        
        
    if testIncorr:
        inc = 'testIncorr_'
    else:
        inc = ''    
    
    
    if chAl==1:
        al = 'chAl'
    else:
        al = 'stAl'
    
    if corrTrained:
        o2a = 'corr_'
    else:
        o2a = ''

    if shflTrsEachNeuron:
        shflname = 'shflTrsPerN_'
    else:
        shflname = ''

    if shflTrLabs:
        shflTrLabs_n = '_shflTrLabs'
    else:
        shflTrLabs_n = ''        
            
    ''' 
    if len(doInhAllexcEqexc)==0: # 1st run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc at the same time, also for all days (except a few days of fni18), inhRois was used (not the new inhRois_pix)       
        if trialHistAnalysis:
            if useEqualTrNums:
                svmn = 'excInh_SVMtrained_eachFrame_prevChoice_%s_ds%d_eqTrs_*' %(al,regressBins)
            else:
                svmn = 'excInh_SVMtrained_eachFrame_prevChoice_%s_ds%d_*' %(al,regressBins)
        else:
            if useEqualTrNums:
                svmn = 'excInh_SVMtrained_eachFrame_currChoice_%s_ds%d_eqTrs_*' %(al,regressBins)
            else:
                svmn = 'excInh_SVMtrained_eachFrame_currChoice_%s_ds%d_*' %(al,regressBins)
        
    else: # 2nd run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc separately; also for all days the new vector inhRois_pix was used (not the old inhRois)       
    '''
    if doInhAllexcEqexc[0] == 1:
        ntype = 'inh'
    elif doInhAllexcEqexc[1] == 1:
        ntype = 'allExc'
    elif doInhAllexcEqexc[1] == 2:
        ntype = 'allN'        
    elif doInhAllexcEqexc[2] == 1:
        ntype = 'eqExc' 
    elif doInhAllexcEqexc[2]==2:
        ntype = 'excInhHalf'     
    elif doInhAllexcEqexc[2]==3:
        ntype = 'allExc2inhSize'         
          
        
    if trialHistAnalysis:
        if useEqualTrNums:
            svmn = '%s%sexcInh_SVMtrained_eachFrame_%s%s%s%s%s_prevChoice_%s_ds%d_eqTrs_*' %(diffn, h2l, o2a, inc, shflname, ntype, shflTrLabs_n, al,regressBins)
        else:
            svmn = '%s%sexcInh_SVMtrained_eachFrame_%s%s%s%s%s_prevChoice_%s_ds%d_*' %(diffn, h2l, o2a, inc, shflname, ntype, shflTrLabs_n, al,regressBins)
    else:
        if useEqualTrNums:
            svmn = '%s%sexcInh_SVMtrained_eachFrame_%s%s%s%s%s_currChoice_%s_ds%d_eqTrs_*' %(diffn, h2l, o2a, inc, shflname, ntype, shflTrLabs_n, al,regressBins)
        else:
            svmn = '%s%sexcInh_SVMtrained_eachFrame_%s%s%s%s%s_currChoice_%s_ds%d_*' %(diffn, h2l, o2a, inc, shflname, ntype, shflTrLabs_n, al,regressBins)
        
        
    svmn = svmn + os.path.basename(pnevFileName) #pnevFileName[-32:]    
    svmn = os.path.join(os.path.dirname(pnevFileName), 'svm', svmn)    
    svmName = glob.glob(svmn)
    if len(svmName)>0:
        svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    else:
        print svmn
#        sys.exit('No such mat file!')

    return svmName
    


#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName

def setSVMname_allN_eachFrame(pnevFileName, trialHistAnalysis, chAl, regressBins=3, corrTrained=0, shflTrsEachNeuron=0, shflTrLabs=0):
    import glob

    if chAl==1:
        al = 'chAl'
    else:
        al = 'stAl'

    if corrTrained==1: 
        o2a = '_corr'
    else:
        o2a = '' 

    if shflTrsEachNeuron:
    	shflname = '_shflTrsPerN'
    else:
    	shflname = ''        
    
    if shflTrLabs:
        shflTrLabs_n = 'shflTrLabs_'
    else:
        shflTrLabs_n = ''
    
    if trialHistAnalysis:
        svmn = 'svmPrevChoice_eachFrame_%s%s%s%s_ds%d_*' %(shflTrLabs_n, al,o2a,shflname,regressBins)
    else:
        svmn = 'svmCurrChoice_eachFrame_%s%s%s%s_ds%d_*' %(shflTrLabs_n, al,o2a,shflname,regressBins)
    
    svmn = svmn + os.path.basename(pnevFileName) #pnevFileName[-32:]    
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
    svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.
    
#    if len(svmName)>0:
#        svmName = svmName[0] # get the latest file
#    else:
#        svmName = ''
#    
    return svmName
    
    
    

    
#%% Load matlab vars to set eventI_ds (downsampled eventI)
    
def setEventIds(postName, chAl, regressBins=3, trialHistAnalysis=0):
            
    if chAl==1:    #%% Use choice-aligned traces 
        # Load 1stSideTry-aligned traces, frames, frame of event of interest
        # use firstSideTryAl_COM to look at changes-of-mind (mouse made a side lick without committing it)
        Data = scio.loadmat(postName, variable_names=['firstSideTryAl'],squeeze_me=True,struct_as_record=False)
    #    traces_al_1stSide = Data['firstSideTryAl'].traces.astype('float')
        time_aligned_1stSide = Data['firstSideTryAl'].time.astype('float')
        time_trace = time_aligned_1stSide
        eventI = Data['firstSideTryAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
        
    else:   #%% Use stimulus-aligned traces           
        # Load stim-aligned_allTrials traces, frames, frame of event of interest
        if trialHistAnalysis==0:
            Data = scio.loadmat(postName, variable_names=['stimAl_noEarlyDec'],squeeze_me=True,struct_as_record=False)
#            eventI = Data['stimAl_noEarlyDec'].eventI - 1 # remember difference indexing in matlab and python!
#            traces_al_stimAll = Data['stimAl_noEarlyDec'].traces.astype('float')
            time_aligned_stim = Data['stimAl_noEarlyDec'].time.astype('float')        
        else:
            Data = scio.loadmat(postName, variable_names=['stimAl_allTrs'],squeeze_me=True,struct_as_record=False)
#            eventI = Data['stimAl_allTrs'].eventI - 1 # remember difference indexing in matlab and python!
#            traces_al_stimAll = Data['stimAl_allTrs'].traces.astype('float')
            time_aligned_stim = Data['stimAl_allTrs'].time.astype('float')
            # time_aligned_stimAll = Data['stimAl_allTrs'].time.astype('float') # same as time_aligned_stim        
        
        time_trace = time_aligned_stim        

    print(np.shape(time_trace))
#    eventI_allDays[iday] = eventI        
    
    
    ############%% Downsample traces: average across multiple times (downsampling, not a moving average. we only average every regressBins points.)    
    if np.isnan(regressBins)==0: # set to nan if you don't want to downsample.
        print 'Downsampling traces ....'    
       
        # preivous method ... earlier svm files are downsampled using this method!   
        '''
        T1 = time_trace.shape[0]
        tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X            

        time_trace = time_trace[0:regressBins*tt]
        time_trace = np.round(np.mean(np.reshape(time_trace, (regressBins, tt), order = 'F'), axis=0), 2)
        print time_trace.shape
    
        eventI_ds = np.argwhere(np.sign(time_trace)>0)[0] # frame in downsampled trace within which event_I happened (eg time1stSideTry)    
        '''
        # new method (frames before frame0 downsampled, and frames after (including frame0) separately)
        ##### time_trace
        # set frames before frame0 (not including it)
        f = (np.arange(eventI - regressBins*np.floor(eventI/float(regressBins)) , eventI)).astype(int) # 1st frame until 1 frame before frame0 (so that the total length is a multiplicaion of regressBins)
        
        T1 = f.shape[0]
        eventI_ds = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames before frame0         # eventI_ds computed here is of type int (ie scalar), but the once computed above is of type list (ie array); to conver scalar to list do [scalar]    

        '''
        ### the parts below are not really needed! unless you want to run a sanity check below
        ############# 
        x = time_trace[f] # time_trace including frames before frame0
        T1 = x.shape[0]
        tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames before frame0 # same as eventI_ds
        xdb = np.mean(np.reshape(x, (regressBins, tt), order = 'F'), axis=0) # downsampled X_svm inclusing frames before frame0
                
        # set frames after frame0 (including it)
        if lastTimeBinMissed==0: # if 0, things were run fine; if 1: by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
            f = (np.arange(eventI , eventI + regressBins * np.floor((time_trace.shape[0] - eventI) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins    
        else: # by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
            f = (np.arange(eventI , eventI + regressBins * np.floor((time_trace.shape[0] - (eventI+1)) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins            
        x = time_trace[f] # X_svm including frames after frame0
        T1 = x.shape[0]
        tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames after frame0
        xda = np.mean(np.reshape(x, (regressBins, tt), order = 'F'), axis=0) # downsampled X_svm inclusing frames after frame0
        
        # set the final downsampled time_trace: concatenate downsampled X at frames before frame0, with x at frames after (including) frame0
        time_trace_d = np.concatenate((xdb, xda))   # time_traceo[eventI] will be an array if eventI is an array, but if you load it from matlab as int, it wont be an array and you have to do [time_traceo[eventI]] to make it a list so concat works below:
    #    time_trace_d = np.concatenate((xdb, [time_traceo[eventI]], xda))    
        print 'time trace size--> original:',time_trace.shape, 'downsampled:', time_trace_d.shape    
        time_trace = time_trace_d
    
        # below is same as what's computed above
        eventI_ds = np.argwhere(np.sign(time_trace_d)>0)[0]  # frame in downsampled trace within which event_I happened (eg time1stSideTry)    
        #############        
        '''
        
    else:
        print 'Not downsampling traces ....'        
#        eventI_ch = Data['firstSideTryAl'].eventI - 1 # remember to subtract 1! matlab vs python indexing!   
#        eventI_ds = eventI_ch
    
#    eventI_ds_allDays[iday] = eventI_ds
    
    return eventI, eventI_ds
    
    

#%% Get number of hr, lr trials that were used for svm training
'''
corr_hr_lr = np.full((len(days),2), np.nan) # number of hr, lr correct trials for each day    

for iday in range(len(days)): 

    imagingFolder = days[iday][0:6]; #'151013'
    mdfFileNumber = map(int, (days[iday][7:]).split("-")); #[1,2] 
        
    imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=[2], pnev2load=[], postNProvided=1)        
    postName = os.path.join(os.path.dirname(pnevFileName), 'post_'+os.path.basename(pnevFileName))
#        moreName = os.path.join(os.path.dirname(pnevFileName), 'more_'+os.path.basename(pnevFileName))       
#        print(os.path.basename(imfilename))    
    svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, [1,0,0], regressBins, useEqualTrNums, corrTrained, shflTrsEachNeuron)[0]   
'''
    
def set_corr_hr_lr(postName, svmName, doIncorr=0):
    # codes above show how to get postName and svmName

    Data = scio.loadmat(postName, variable_names=['allResp_HR_LR'])    
    allResp_HR_LR = np.array(Data.pop('allResp_HR_LR')).flatten().astype('float')
#    print '%d correct choices; %d incorrect choices' %(sum(outcomes==1), sum(outcomes==0))
    
    
#    svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, [1,0,0], regressBins)[0]   
    Data = scio.loadmat(svmName, variable_names=['trsExcluded'])
    trsExcluded = Data.pop('trsExcluded').flatten().astype('bool')
    
    corr_hr = sum(np.logical_and(allResp_HR_LR==1 , ~trsExcluded)).astype(int)
    corr_lr = sum(np.logical_and(allResp_HR_LR==0 , ~trsExcluded)).astype(int)    
#    print min(corr_hr, corr_lr) # number of trials of each class used in svm training
#    corr_hr_lr[iday,:] = [corr_hr, corr_lr]        

    ##%% Get number of hr, lr trials that were used for incorr testing    
    if doIncorr: # incorrect trials
        Y_incorr0 = allResp_HR_LR+0
        Y_incorr0[outcomes!=0] = np.nan; # analyze only incorrect trials.
    #    print '\tincorrect trials: %d HR; %d LR' %((Y_incorr0==1).sum(), (Y_incorr0==0).sum())    
        trsExcluded_incorr = (np.isnan(np.sum(traces_al_1stSide, axis=(0,1))) + np.isnan(Y_incorr0)) != 0
        
        incorr_hr = sum(np.logical_and(allResp_HR_LR==1 , ~trsExcluded_incorr)).astype(int)
        incorr_lr = sum(np.logical_and(allResp_HR_LR==0 , ~trsExcluded_incorr)).astype(int)
    #    print incorr_hr, incorr_lr # number of trials of incorr, hr and lr classes used for incorr testing
#        incorr_hr_lr[iday,:] = [incorr_hr, incorr_lr]    
    
    if doIncorr:
        return corr_hr, corr_lr, incorr_hr, incorr_lr
    else:
        return corr_hr, corr_lr


    
#%% Find number of frames before and after eventI for each day, and the the min numbers across days; 
# this is to find common eventI (among a number of session)

##%% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
# By common eventI, we  mean the index on which all traces will be aligned.

def set_nprepost(trace, eventI_ds_allDays, mn_corr=np.nan, thTrained=10., regressBins=3, traceAl=0):
    # if traceAl=1, it means trace is already aligned across days.. so totLen = len(trace)
    # trace: each element is for one day
    # first element of trace[iday] should be frames
    
    ###### if days of each mouse are already aligned, you can stilluse this function to align all mice:
    # set_nprepost(avFRexc_allMice, eventI_ds_allMice); where eventI_ds_allMice is computed as :
#    time_al, nPreMin, nPostMin = set_nprepost(Xinh_allDays_allMice[im], eventI_ds_allDays_allMice[im], mn_corr, thTrained=10, regressBins=3)    
#    time_al_allMice.append(time_al)
#    nPreMin_allMice.append(nPreMin)
    
#    time_al, nPreMin, nPostMin = set_nprepost(Xinh_allDays_allMice[im], eventI_ds_allDays_allMice[im], mn_corr, thTrained=10, regressBins=3)    
#    eventI_ds_allMice.append(nPreMin)    
    
    eventI_ds_allDays = np.array(eventI_ds_allDays).flatten()
    numDays = len(trace)        
    if np.isnan(mn_corr).all():
        mn_corr = np.full((numDays), thTrained+1) # so all days are analyzed


    nPreMin = np.nanmin(eventI_ds_allDays[mn_corr >= thTrained]).astype('int') # number of frames before the common eventI, also the index of common eventI.     
    
    
    if traceAl==1: # the input trace is already aligned across days
        nPostMin  = len(trace[0]) - nPreMin - 1
    else: 
        nPost = (np.ones((numDays,1))+np.nan).flatten()
        for iday in range(numDays):
            if mn_corr[iday] >= thTrained: # dont include days with too few svm trained trials.
                nPost[iday] = len(trace[iday]) - eventI_ds_allDays[iday] - 1
        nPostMin = np.nanmin(nPost).astype('int')
    
        
    ## Set the time array for the across-day aligned traces
    totLen = nPreMin + nPostMin +1
    print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)

    # Get downsampled time trace, without using the non-downsampled eventI
    # you need nPreMin and totLen
    a = frameLength*np.arange(-regressBins*nPreMin,0)
    b = frameLength*np.arange(0,regressBins*(totLen-nPreMin))
    aa = np.mean(np.reshape(a,(regressBins,nPreMin), order='F'), axis=0)
    bb = np.mean(np.reshape(b,(regressBins,totLen-nPreMin), order='F'), axis=0)
    time_al = np.concatenate((aa,bb))

    return time_al, nPreMin, nPostMin
   


#%% Set time_aligned and nPreMin for aligning traces of all mice

def set_nprepost_allMice(trace_allMice, eventI_ds_allDays_allMice, corr_hr_lr_allMice, thTrained=10., regressBins=3):

# trace_allMice[im].shape[1] must be frames
# trace_allMice has len(mice) elements, each element is averaged across days;
# nPreMin_allMice indicates the frame of eventI for each mouse (after aligning its traces from all days) and it is 
# computed using eventI_ds_allDays_allMice which indicates the frame of eventI for each day of each mouse.

    nPreMin_allMice = np.full((len(trace_allMice)), np.nan, dtype=int)
    
    for im in range(len(trace_allMice)):
        
        if np.ndim(corr_hr_lr_allMice[im])==2:
            mn_corr = np.min(corr_hr_lr_allMice[im], axis=1) # number of trials of each class. 90% of this was used for training, and 10% for testing.
        elif np.ndim(corr_hr_lr_allMice[im])==1: # mn_corr is already provided; we dont need to take te min..
            mn_corr = corr_hr_lr_allMice[im]
        
        nPreMin = np.nanmin(eventI_ds_allDays_allMice[im][mn_corr >= thTrained]).astype('int') # number of frames before the common eventI, also the index of common eventI.     
        totLen = trace_allMice[im].shape[1] #nPreMin + nPostMin +1         # trace_allMice = classAccurTMS_allExc[im]
    
        nPreMin_allMice[im] = nPreMin

#    time_al_final, nPreMin_final, nPostMin_final = set_nprepost(trace_allMice, nPreMin_allMice)
    nPreMin_final = np.min(nPreMin_allMice).astype(int) # you need to do the following to get the final common eventI of all mice: nPreMin_final = np.min(nPreMin_allMice)
    
    nPost = [trace_allMice[im].shape[1] - nPreMin_allMice[im] - 1 for im in range(len(trace_allMice))]
    nPostMin_final = np.nanmin(nPost).astype('int')
    
    ## Set the time array for the across-day aligned traces
    totLen = nPreMin_final + nPostMin_final +1
    print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin_final)


    a = frameLength*np.arange(-regressBins*nPreMin_final,0)
    b = frameLength*np.arange(0,regressBins*(totLen-nPreMin_final))
    aa = np.mean(np.reshape(a,(regressBins,nPreMin_final), order='F'), axis=0)
    bb = np.mean(np.reshape(b,(regressBins,totLen-nPreMin_final), order='F'), axis=0)
    time_aligned_final = np.concatenate((aa,bb))
    

    return time_aligned_final, nPreMin_final, nPostMin_final, nPreMin_allMice 


#%% Align traces of each day on the common eventI

def alTrace(trace, eventI_ds_allDays, nPreMin, nPostMin, mn_corr=np.nan, thTrained=10, frsDim=0):    
    # first element of trace[iday] should be frames (frsDim0=1), unless frsDim0=0
    # if days of each mouse is already aligned, you can use this function to align all mice:
    # trace will be trace[im] which is [dayAlignedFrames, days], and you will turn it to [mouseAlignedFrames, days]
    # in this case iday (in below) will be like im... 

    # mn_corr: min number of HR and LR trials 
    numDays = len(trace)        

    # frames are not dimension 0, so transpose traces of each day (to have frames as dimention 0)
    if frsDim!=0: 
        allDims = range(np.ndim(trace[0]))
        nonFrDim = np.argwhere(~np.in1d(allDims, frsDim)).flatten() # this dimension in trace[iday] is not frs 
        trace = [np.transpose(trace[iday], np.concatenate(([frsDim], nonFrDim))) for iday in range(numDays)]

    if np.isnan(mn_corr).any():
        mn_corr = np.full((numDays), thTrained+1, dtype=int) # so all days are analyzed
        
    trace = np.array(trace)
    trace_aligned = []
#    trace_aligned = np.ones((nPreMin + nPostMin + 1, trace.shape[0])) + np.nan # frames x days, aligned on common eventI (equals nPreMin)     
    for iday in range(numDays):
        if mn_corr[iday] >= thTrained: # dont include days with too few svm trained trials.
            trace_aligned.append(trace[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1])
#            trace_aligned[:, iday] = trace[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]    
#    trace_aligned = np.array(trace_aligned)
    return trace_aligned


#%% Align traces of dimensions frs x frs (eg angles) of all days on the common eventI

def alTrace_frfr(trace, eventI_ds_allDays, nPreMin, nPostMin, mn_corr=np.nan, thTrained=10, frsDim=np.nan):
    
    # can be used for aligning matrices of size frs x frs across days, or across mice    
    # trace: each element is for a mouse or a day ... and it has size frs x frs
    # frsDim = nan --> default: dimensions 0 and 1 of trace are frames. otherwise frsDim 
    # the first 2 dimensions of trace must be frames (unless frsDim other than nan)
    # output: days x alignedFrs x alignedFrs
    
    # if each element of trace is for each mouse, and it includes traces of all days, mn_corr must be nan, ie u must first remove bad days ... bc u cannot do it here.

    numDays = len(trace)

    # frames are not dimensions 0 and 1, so transpose traces of each day (to have frames as dimentions 0 and 1)
    if np.isnan(frsDim).any()==0: 
        allDims = range(np.ndim(trace[0]))
        nonFrDim = np.argwhere(~np.in1d(allDims, frsDim)).flatten() # this dimension in trace[iday] is not frs 
        trace = [np.transpose(trace[im], ([frsDim[0],frsDim[1],nonFrDim])) for im in range(numDays)]

    
    if type(mn_corr)==float and np.isnan(mn_corr).any():
        mn_corr = np.full((numDays), thTrained+1, dtype=int) # so all days are analyzed    
    
#    totLen = nPreMin + nPostMin + 1  # len(time_aligned) # nPreMin + nPostMin + 1   
    trace_aligned = []
    
    for iday in range(numDays): # or looping through days
        
        if mn_corr[iday] >= thTrained: # dont include days with too few svm trained trials.
    #        trace_aligned = np.ones((totLen,totLen, trace[iday].shape[2])) + np.nan
        #    angleInh_aligned[:, iday] = angleInh_all[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1  ,  eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]
            inds = np.arange(eventI_ds_allDays[iday] - nPreMin,  eventI_ds_allDays[iday] + nPostMin + 1)
#            inds = np.arange(nPreMin[iday] - nPreMin,  nPreMin[iday] + nPostMin + 1)
            
            trace_aligned.append(trace[iday][inds][:,inds])
    
    if len(np.unique([np.shape(trace_aligned[iday])[-1] for iday in range(sum(mn_corr >= thTrained))])) == 1: # if it is fr x fr x days we can't do it bc diff mice have diff num days! but if it is frs x frs or frs x frs x samps we can turn it into an array
        trace_aligned = np.array(trace_aligned)
    
    return trace_aligned


#%% Set the time array for the across-day aligned traces

# totLen_ds = len(av_l2_test_diday)
def set_time_al(totLen_ds, eventI, lastTimeBinMissed, regressBins=3):
    # totLen_ds: length of downsample trace
    # eventI : eventI on the original trace (non-downsampled)
#    eventI = eventI_allDaysiday #np.argwhere(time_trace==0).flatten()
    time_trace = frameLength * (np.arange(0, np.ceil(regressBins*(totLen_ds+1))) - eventI) # time_trace = time_aligned_1stSide
#    time_trace = frameLength * (np.arange(0, np.ceil(regressBins*totLen_ds)) - eventI) # time_trace = time_aligned_1stSide
    
    f = (np.arange(eventI - regressBins*np.floor(eventI/float(regressBins)) , eventI)).astype(int) # 1st frame until 1 frame before frame0 (so that the total length is a multiplicaion of regressBins)
    x = time_trace[f] # time_trace including frames before frame0
    T1 = x.shape[0]
    tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames before frame0 # same as eventI_ds
    xdb = np.mean(np.reshape(x, (regressBins, tt), order = 'F'), axis=0) # downsampled X_svm inclusing frames before frame0
    
    
    # set frames after frame0 (including it)
    if lastTimeBinMissed==0: # if 0, things were run fine; if 1: by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
        f = (np.arange(eventI , eventI + regressBins * np.floor((time_trace.shape[0] - eventI) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins    
    else: # by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
        f = (np.arange(eventI , eventI + regressBins * np.floor((time_trace.shape[0] - (eventI+1)) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins            
#    f = (np.arange(eventI+1 , eventI+1+regressBins * np.floor((time_trace.shape[0] - (eventI+1)) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins    
    x = time_trace[f] # X_svm including frames after frame0
    T1 = x.shape[0]
    tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames after frame0
    xda = np.mean(np.reshape(x, (regressBins, tt), order = 'F'), axis=0) # downsampled X_svm inclusing frames after frame0
    
    time_trace_d = np.concatenate((xdb, xda))    
    # set the final downsampled time_trace: concatenate downsampled X at frames before frame0, with x at frame0, and x at frames after frame0
#    time_trace_d = np.concatenate((xdb, [0], xda))    
    time_al = time_trace_d[0:int(totLen_ds)] 

    return time_al




#%%
def downsampXsvmTime(X_svm, time_trace, eventI, regressBins, lastTimeBinMissed):

    #regressBins = 2 # number of frames to average for downsampling X
    #regressBins = int(np.round(100/frameLength)) # 100ms        
    if np.isnan(regressBins)==0: # set to nan if you don't want to downsample.
        print 'Downsampling traces ....'            
        
        # below is problematic when it comes to aligning all sessions... they have different number of frames before eventI, so time bin 0 might be average of frames eventI-1:eventI+1 or eventI:eventI+2, etc.
        # on 10/4/17 I added the version below, where we average every 3 frames before eventI, also we average every 3 frames including eventI and after. Then we concat them.
        # this way we make sure that time bin 0, always includes eventI and 2 frames after. and time bin -1 always includes average of 3 frames before eventI.
        
        '''
        # X_svm
        T1, N1, C1 = X_svm.shape
        tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X
        X_svm = X_svm[0:regressBins*tt,:,:]
        #X_svm_d.shape
        
        X_svm = np.mean(np.reshape(X_svm, (regressBins, tt, N1, C1), order = 'F'), axis=0)
        print 'downsampled choice-aligned trace: ', X_svm.shape
            
            
        time_trace = time_trace[0:regressBins*tt]
    #    print time_trace_d.shape
        time_trace = np.round(np.mean(np.reshape(time_trace, (regressBins, tt), order = 'F'), axis=0), 2)
    #    print time_trace.shape
    
        eventI_ds = np.argwhere(np.sign(time_trace)>0)[0] # frame in downsampled trace within which event_I happened (eg time1stSideTry)    
        '''
    
        # new method, started on 10/4/17
    
        ##### x_svm
        # set frames before frame0 (not including it)
        f = (np.arange(eventI - regressBins*np.floor(eventI/float(regressBins)) , eventI)).astype(int) # 1st frame until 1 frame before frame0 (so that the total length is a multiplicaion of regressBins)
        x = X_svm[f,:,:] # X_svmo including frames before frame0
        T1, N1, C1 = x.shape
        tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames before frame0
        xdb = np.mean(np.reshape(x, (regressBins, tt, N1, C1), order = 'F'), axis=0) # downsampled X_svmo inclusing frames before frame0        
        
        # set frames after frame0 (including it)
        if lastTimeBinMissed==0: # if 0, things were run fine; if 1: by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)        
            f = (np.arange(eventI , eventI+regressBins * np.floor((X_svm.shape[0] - (eventI)) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins    
        else: # by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)        
            f = (np.arange(eventI , eventI+regressBins * np.floor((X_svm.shape[0] - (eventI+1)) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins    
        x = X_svm[f,:,:] # X_svmo including frames after frame0
        T1, N1, C1 = x.shape
        tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames after frame0
        xda = np.mean(np.reshape(x, (regressBins, tt, N1, C1), order = 'F'), axis=0) # downsampled X_svmo inclusing frames after frame0
        
        # set the final downsampled X_svmo: concatenate downsampled X at frames before frame0, with x at frames after (and including) frame0
        X_svm_d = np.concatenate((xdb, xda))    
        print 'trace size--> original:',X_svm.shape, 'downsampled:', X_svm_d.shape
        X_svm = X_svm_d
        
        
        
        ##### time_trace
        if len(time_trace)>0:
            # set frames before frame0 (not including it)
            f = (np.arange(eventI - regressBins*np.floor(eventI/float(regressBins)) , eventI)).astype(int) # 1st frame until 1 frame before frame0 (so that the total length is a multiplicaion of regressBins)
            x = time_trace[f] # time_trace including frames before frame0
            T1 = x.shape[0]
            tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames before frame0 # same as eventI_ds
            xdb = np.mean(np.reshape(x, (regressBins, tt), order = 'F'), axis=0) # downsampled X_svm inclusing frames before frame0      
    
            # set frames after frame0 (including it)
            if lastTimeBinMissed==0: # if 0, things were run fine; if 1: by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
                f = (np.arange(eventI , eventI + regressBins * np.floor((time_trace.shape[0] - eventI) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins    
            else: # by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)
                f = (np.arange(eventI , eventI + regressBins * np.floor((time_trace.shape[0] - (eventI+1)) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins            
            x = time_trace[f] # X_svm including frames after frame0
            T1 = x.shape[0]
            tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames after frame0
            xda = np.mean(np.reshape(x, (regressBins, tt), order = 'F'), axis=0) # downsampled X_svm inclusing frames after frame0
            
            # set the final downsampled time_trace: concatenate downsampled X at frames before frame0, with x at frames after (including) frame0
            time_trace_d = np.concatenate((xdb, xda))   # time_traceo[eventI] will be an array if eventI is an array, but if you load it from matlab as int, it wont be an array and you have to do [time_traceo[eventI]] to make it a list so concat works below:
        #    time_trace_d = np.concatenate((xdb, [time_traceo[eventI]], xda))    
            print 'time trace size--> original:',time_trace.shape, 'downsampled:', time_trace_d.shape    
            time_trace = time_trace_d
            
            #####
            eventI_ds = np.argwhere(np.sign(time_trace_d)>0)[0]  # frame in downsampled trace within which event_I happened (eg time1stSideTry)    
        else:
            eventI_ds = []

        ################################### x_svm, incorrect trials ###################################
        """
        if testIncorr:
            # set frames before frame0 (not including it)
            f = (np.arange(eventI - regressBins*np.floor(eventI/float(regressBins)) , eventI)).astype(int) # 1st frame until 1 frame before frame0 (so that the total length is a multiplicaion of regressBins)
            x = X_svm_incorr[f,:,:] # X_svmo including frames before frame0
            T1, N1, C1 = x.shape
            tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames before frame0
            xdb = np.mean(np.reshape(x, (regressBins, tt, N1, C1), order = 'F'), axis=0) # downsampled X_svmo inclusing frames before frame0
            
            
            # set frames after frame0 (including it)
            if lastTimeBinMissed==0: # if 0, things were run fine; if 1: by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)        
                f = (np.arange(eventI , eventI+regressBins * np.floor((X_svm_incorr.shape[0] - (eventI)) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins    
            else: # by mistake you subtracted eventI+1 instead of eventI, so x_svm misses the last time bin (3 frames) in most of the days! (analyses done on the week of 10/06/17 and before)        
                f = (np.arange(eventI , eventI+regressBins * np.floor((X_svm_incorr.shape[0] - (eventI+1)) / float(regressBins)))).astype(int) # total length is a multiplicaion of regressBins                           
            x = X_svm_incorr[f,:,:] # X_svmo including frames after frame0
            T1, N1, C1 = x.shape
            tt = int(np.floor(T1 / float(regressBins))) # number of time points in the downsampled X including frames after frame0
            xda = np.mean(np.reshape(x, (regressBins, tt, N1, C1), order = 'F'), axis=0) # downsampled X_svmo inclusing frames after frame0
            
            # set the final downsampled X_svmo: concatenate downsampled X at frames before frame0, with x at frames after (and including) frame0
            X_svm_incorr_d = np.concatenate((xdb, xda))
            print 'trace size--> original:',X_svm_incorr.shape, 'downsampled:', X_svm_incorr_d.shape
            X_svm_incorr = X_svm_incorr_d      
        """    
                
    
    else:
        print 'Not downsampling traces ....'

 
    return X_svm, time_trace, eventI_ds


        
#%%    
def rmvNans(a): # remove nans
    mask = ~np.isnan(a)
    a = np.array([d[m] for d, m in zip(a, mask)])
    return a

 
#%% compute p value between vectors a,b (wilcoxon, MW, ttest)  
 
def pwm(a,b,whichP=0): # if whichP=2, only compute ttest

    _,p2  = sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)])

    if whichP!=2: 
        _,p0  = sci.stats.ranksums(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)]); 
        _,p1  = sci.stats.mannwhitneyu(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)])

    if whichP!=2:
        p = [p0,p1,p2]
    else:
        p = p2
    return p    
   
   
#%% Compute p value between each column of matrix a and each column of matrix b (wilcoxon, MW, ttest)  
# different columns could be different days or different frames
   # a,b:   samps x days    or      samps x frames
   # output: 3 x days    or     3 x frames
   
def pwmt_frs(a,b):
    nfr = a.shape[1]
    
    a = rmvNans(a)
    b = rmvNans(b)    
    
    if np.shape(a)[1]>0 and np.shape(b)[1]>0:
        p0  = np.array([sci.stats.ranksums(a[:,f], b[:,f])[1] for f in range(b.shape[1])])
        p1  = np.array([sci.stats.mannwhitneyu(a[:,f], b[:,f])[1] for f in range(b.shape[1])])
        _,p2  = sci.stats.ttest_ind(a, b)
        p = [p0,p1,p2]
        p = np.array(p)
    else:
        p = np.full((3,nfr),np.nan)
    
    return p
        


#%% Load SVM vars - eachFrame, allN
    
def loadSVM_allN(svmName, doPlots, doIncorr, loadWeights, shflTrLabs=0):
    # loadWeights: 
    # 0: don't load weights, only load class accur
    # 1 : load weights and class cccur
    # 2 : load only weights and not class accur 


    Data = scio.loadmat(svmName, variable_names=['regType','cvect'])
    regType = Data.pop('regType').astype('str')
    cvect = Data.pop('cvect').squeeze()
        
    if loadWeights!=2:                   
        Data = scio.loadmat(svmName, variable_names=['regType','cvect','perClassErrorTrain','perClassErrorTest','perClassErrorTest_chance','perClassErrorTest_shfl'])  
        perClassErrorTrain = Data.pop('perClassErrorTrain') # numSamples x len(cvect) x nFrames
        perClassErrorTest = Data.pop('perClassErrorTest')
        perClassErrorTest_chance = Data.pop('perClassErrorTest_chance')
        perClassErrorTest_shfl = Data.pop('perClassErrorTest_shfl')
#    else:
#        perClassErrorTrain = []
#        perClassErrorTest = []
#        perClassErrorTest_chance = []
#        perClassErrorTest_shfl = []
        
#    numSamples = perClassErrorTest.shape[0] 
#    numFrs = perClassErrorTest.shape[2]

    if doIncorr==1 and loadWeights!=2:        
        Data = scio.loadmat(svmName, variable_names=['perClassErrorTest_incorr','perClassErrorTest_incorr_chance','perClassErrorTest_incorr_shfl'])        
        perClassErrorTest_incorr = Data.pop('perClassErrorTest_incorr') # numSamples x len(cvect) x nFrames
        perClassErrorTest_incorr_chance = Data.pop('perClassErrorTest_incorr_chance')
        perClassErrorTest_incorr_shfl = Data.pop('perClassErrorTest_incorr_shfl')

    if loadWeights>=1:            
        Data = scio.loadmat(svmName, variable_names=['wAllC', 'bAllC']) #,])
        wAllC = Data.pop('wAllC')  # samps x len(cvect) x neurons x nFrs
        bAllC = Data.pop('bAllC')
    else:
         wAllC = []
         bAllC = []
         
        
    #####################%% Set class error values at best C (also find bestc for each frame and plot c path)    
    if loadWeights!=2:        
        if doIncorr==1:
            classErr_bestC_train_data, classErr_bestC_test_data, classErr_bestC_test_shfl, classErr_bestC_test_chance, cbestFrs, w_bestc_data, b_bestc_data, classErr_bestC_test_incorr, classErr_bestC_test_incorr_shfl, classErr_bestC_test_incorr_chance = setBestC_classErr(perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance, cvect, regType, doPlots, doIncorr, wAllC, bAllC, shflTrLabs)
        else:
            classErr_bestC_train_data, classErr_bestC_test_data, classErr_bestC_test_shfl, classErr_bestC_test_chance, cbestFrs, w_bestc_data, b_bestc_data = setBestC_classErr(perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance, cvect, regType, 0, doIncorr, wAllC, bAllC, shflTrLabs)
    else: # only weights
        Data = scio.loadmat(svmName, variable_names=['perClassErrorTest'])  
        perClassErrorTest = Data.pop('perClassErrorTest') # numSamples x len(cvect) x nFrames
        cbestFrs = findBestC(perClassErrorTest, cvect, regType, smallestC=0) # nFrames     	
        #####%% Set class error values at best C (if desired plot c path) (funcion is defined in defFuns.py)            
        _, _, _, _, w_bestc_data, b_bestc_data = setClassErrAtBestc(cbestFrs, cvect, 0, np.nan, np.nan, np.nan, np.nan, wAllC, bAllC)        
        classErr_bestC_train_data = [] 
        classErr_bestC_test_data = [] 
        classErr_bestC_test_shfl = []
        classErr_bestC_test_chance = []
        
    '''    
    # compare cbest between frames
    plt.figure(figsize=(3,2))    
    plt.plot(cbestFrs); plt.xlabel('frames'); plt.ylabel('best C'); 
    r = max(cbestFrs)-min(cbestFrs)
    plt.ylim([-r/10, max(cbestFrs)+r/10])
    plt.vlines(eventI_ds, -r/10, max(cbestFrs)+r/10, color='r') # mark eventI_ds
    '''
#    print 'CE values for testing data:', np.unique(classErr_bestC_test_data.flatten())
#    print 'CE values for incorr:', np.unique(classErr_bestC_test_incorr.flatten())
    
    
    ######%% Set class errors to 50 if less than .05 fraction of neurons in a sample have non-0 weights, and set all samples class error to 50, if less than 10 samples satisfy this condition.

    if loadWeights==1:            
        thNon0Ws = .05; # fraction non0 neurons to accept a sample (ie good sample, which we wont turn all its CA values to 50) 
        thSamps = 10; # min number of good samples to accept a frame (ie not to turn all its CA values to 50)
        eps = 1e-10 # below which is considered 0
        classErr_bestC_test_data = setTo50classErr(classErr_bestC_test_data, w_bestc_data, thNon0Ws, thSamps, eps)
        classErr_bestC_test_shfl = setTo50classErr(classErr_bestC_test_shfl, w_bestc_data, thNon0Ws, thSamps, eps)
        if doIncorr==1:
            classErr_bestC_test_incorr = setTo50classErr(classErr_bestC_test_incorr, w_bestc_data, thNon0Ws, thSamps, eps)
            classErr_bestC_test_incorr_shfl = setTo50classErr(classErr_bestC_test_incorr_shfl, w_bestc_data, thNon0Ws, thSamps, eps)
            
    
    if doIncorr==1:
        return classErr_bestC_train_data, classErr_bestC_test_data, classErr_bestC_test_shfl, classErr_bestC_test_chance, cbestFrs, w_bestc_data, b_bestc_data, classErr_bestC_test_incorr, classErr_bestC_test_incorr_shfl, classErr_bestC_test_incorr_chance
    else:
        return classErr_bestC_train_data, classErr_bestC_test_data, classErr_bestC_test_shfl, classErr_bestC_test_chance, cbestFrs, w_bestc_data, b_bestc_data
    
         
#%% Load exc,inh SVM vars
        
def loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, doPlots, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0, loadYtest=0, testIncorr=0):
    # loadWeights: 
    # 0: don't load weights, only load class accur
    # 1 : load weights and class cccur
    # 2 : load only weights and not class accur
    # shflTrLabs=0; loadYtest=1
    
    '''
    if loadInhAllexcEqexc==0: # 1st run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc at the same time, also for all days (except a few days of fni18), inhRois was used (not the new inhRois_pix)       
        svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, [], regressBins)
        svmName = svmName[0]
        print os.path.basename(svmName)    

        Data = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_inh', 'perClassErrorTest_shfl_inh', 'perClassErrorTest_chance_inh',    
                                                 'perClassErrorTest_data_allExc', 'perClassErrorTest_shfl_allExc', 'perClassErrorTest_chance_allExc',    
                                                 'perClassErrorTest_data_exc', 'perClassErrorTest_shfl_exc', 'perClassErrorTest_chance_exc',
                                                 'w_data_inh', 'w_data_allExc', 'w_data_exc'])        
        Datai = Dataae = Datae = Data                                                 

    else:  # 2nd run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc separately; also for all days the new vector inhRois_pix was used (not the old inhRois)       
    '''
    
#    if doAllN==1:
#        smallestC = 0 # Identify best c: if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
#        if smallestC==1:
#            print 'bestc = smallest c whose cv error is less than 1se of min cv error'
#        else:
#            print 'bestc = c that gives min cv error'
            
    svmName_allN = ''            
    svmName_excInh = []
    for idi in range(3):
        
        print '------'
        doInhAllexcEqexc = np.full((4), 0)        
        if np.logical_or(testIncorr, shflTrsEachNeuron):
            doInhAllexcEqexc[-1] = -1
        
        doInhAllexcEqexc[idi] = 1

        
        #################################### Set svm file name ############################
        if idi==1 and doAllN: # plot allN, instead of allExc
            # old:
#            svmName = setSVMname_allN_eachFrame(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, shflTrsEachNeuron, shflTrLabs)[0] # for chAl: the latest file is with soft norm; earlier file is 
            # new svm_excInh_trainDecoder_eachFrame.py computes allN too... 
            doInhAllexcEqexc[idi] = 2
            svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc, regressBins, useEqualTrNums, corrTrained, shflTrsEachNeuron, shflTrLabs)[0]
            svmName_allN = svmName
        else:        
            svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc, regressBins, useEqualTrNums, corrTrained, shflTrsEachNeuron, shflTrLabs)[0]
            svmName_excInh.append(svmName)
#        svmName = svmName[0] # use [0] for the latest file; use [-1] for the earliest file
        print os.path.basename(svmName)    


        #################################### Load vars ####################################
        ######### inh ######### 
        if doInhAllexcEqexc[0] == 1: 
            if loadWeights==1:
                Datai = scio.loadmat(svmName, variable_names=['trsExcluded', 'perClassErrorTest_data_inh', 'perClassErrorTest_shfl_inh', 'perClassErrorTest_chance_inh', 'w_data_inh', 'b_data_inh'])
            elif loadWeights==2:                
                Datai = scio.loadmat(svmName, variable_names=['trsExcluded', 'w_data_inh','b_data_inh'])
            else:
                Datai = scio.loadmat(svmName, variable_names=['trsExcluded', 'perClassErrorTest_data_inh', 'perClassErrorTest_shfl_inh', 'perClassErrorTest_chance_inh'])
            if loadYtest:
                Datai2 = scio.loadmat(svmName, variable_names=['trsnow_allSamps','testTrInds_allSamps_inh', 'Ytest_allSamps_inh', 'Ytest_hat_allSampsFrs_inh'])
            if testIncorr:    
                Datai3 = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_inh_incorr', 'perClassErrorTest_shfl_inh_incorr'])
                
        ######### allN or allExc ######### 
        elif np.logical_or(doInhAllexcEqexc[1] == 1, doInhAllexcEqexc[1] == 2): 
#            if doAllN: # plot allN, instead of allExc
#                _, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, _, w_data_allExc, b_data_allExc = loadSVM_allN(svmName, doPlots, doIncorr, 1, shflTrLabs)   
#                
#            else: # plot allExc
            if loadWeights==1:
                Dataae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_allExc', 'perClassErrorTest_shfl_allExc', 'perClassErrorTest_chance_allExc', 'w_data_allExc', 'b_data_allExc'])
            elif loadWeights==2:                
                Dataae = scio.loadmat(svmName, variable_names=['w_data_allExc', 'b_data_allExc'])
            else:
                Dataae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_allExc', 'perClassErrorTest_shfl_allExc', 'perClassErrorTest_chance_allExc'])
            if loadYtest:
                Dataae2 = scio.loadmat(svmName, variable_names=['trsnow_allSamps','testTrInds_allSamps_allExc', 'Ytest_allSamps_allExc', 'Ytest_hat_allSampsFrs_allExc'])
            if testIncorr:    
                Dataae3 = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_allExc_incorr', 'perClassErrorTest_shfl_allExc_incorr'])    
        
        ######### eqExc ######### 
        elif doInhAllexcEqexc[2] == 1: 
            if loadWeights==1:
                Datae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_exc', 'perClassErrorTest_shfl_exc', 'perClassErrorTest_chance_exc', 'w_data_exc', 'b_data_exc'])                                                 
            elif loadWeights==2:                                
                Datae = scio.loadmat(svmName, variable_names=['w_data_exc', 'b_data_exc'])                                                 
            else:
                Datae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_exc', 'perClassErrorTest_shfl_exc', 'perClassErrorTest_chance_exc'])
            if loadYtest:
                Datae2 = scio.loadmat(svmName, variable_names=['trsnow_allSamps','testTrInds_allSamps_exc', 'Ytest_allSamps_exc', 'Ytest_hat_allSampsFrs_exc'])
            if testIncorr:    
                Datae3 = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_exc_incorr', 'perClassErrorTest_shfl_exc_incorr'])    

        
    
    #### inh #### 
    if loadWeights!=2:   # 2: only download weights, no CA                                     
        perClassErrorTest_data_inh = Datai.pop('perClassErrorTest_data_inh')
        perClassErrorTest_shfl_inh = Datai.pop('perClassErrorTest_shfl_inh')
        perClassErrorTest_chance_inh = Datai.pop('perClassErrorTest_chance_inh') 
        trsExcluded = Datai.pop('trsExcluded')
    if loadWeights>=1:
        w_data_inh = Datai.pop('w_data_inh') 
        b_data_inh = Datai.pop('b_data_inh')         
    else:
        w_data_inh = []
        b_data_inh = []
    
    
    #### allExc or allN #### 
#    if doAllN==0:
    if loadWeights!=2:
        perClassErrorTest_data_allExc = Dataae.pop('perClassErrorTest_data_allExc')
        perClassErrorTest_shfl_allExc = Dataae.pop('perClassErrorTest_shfl_allExc')
        perClassErrorTest_chance_allExc = Dataae.pop('perClassErrorTest_chance_allExc')   
    if loadWeights>=1:
        w_data_allExc = Dataae.pop('w_data_allExc') 
        b_data_allExc = Dataae.pop('b_data_allExc') 
    else:
        w_data_allExc = []
        b_data_allExc = []
    
    
    #### exc #### 
    if loadWeights!=2:
        perClassErrorTest_data_exc = Datae.pop('perClassErrorTest_data_exc')    
        perClassErrorTest_shfl_exc = Datae.pop('perClassErrorTest_shfl_exc')
        perClassErrorTest_chance_exc = Datae.pop('perClassErrorTest_chance_exc')
    if loadWeights>=1:    
        w_data_exc = Datae.pop('w_data_exc') 
        b_data_exc = Datae.pop('b_data_exc') 
    else:
        w_data_exc = []
        b_data_exc = []
    
    
    #######
    if loadYtest:
        testTrInds_allSamps_inh = Datai2.pop('testTrInds_allSamps_inh').astype(int)
        Ytest_allSamps_inh = Datai2.pop('Ytest_allSamps_inh')
        Ytest_hat_allSampsFrs_inh = Datai2.pop('Ytest_hat_allSampsFrs_inh')
        trsnow_allSamps_inh = np.array(Datai2.pop('trsnow_allSamps')).astype(int) # index of trials after picking random hr (or lr) in order to make sure both classes have the same number in the final Y (on which svm was run)

        testTrInds_allSamps_allExc = Dataae2.pop('testTrInds_allSamps_allExc').astype(int)
        Ytest_allSamps_allExc = Dataae2.pop('Ytest_allSamps_allExc')
        Ytest_hat_allSampsFrs_allExc = Dataae2.pop('Ytest_hat_allSampsFrs_allExc')
        trsnow_allSamps_allExc = np.array(Dataae2.pop('trsnow_allSamps')).astype(int) 
        
        testTrInds_allSamps_exc = Datae2.pop('testTrInds_allSamps_exc').astype(int)
        Ytest_allSamps_exc = Datae2.pop('Ytest_allSamps_exc')
        Ytest_hat_allSampsFrs_exc = Datae2.pop('Ytest_hat_allSampsFrs_exc')
        trsnow_allSamps_exc = np.array(Datae2.pop('trsnow_allSamps')).astype(int)
    else:
        testTrInds_allSamps_inh = []
        Ytest_allSamps_inh = []
        Ytest_hat_allSampsFrs_inh = []
        testTrInds_allSamps_allExc = []
        Ytest_allSamps_allExc = []
        Ytest_hat_allSampsFrs_allExc = []        
        testTrInds_allSamps_exc = []
        Ytest_allSamps_exc = []
        Ytest_hat_allSampsFrs_exc = []
        trsnow_allSamps_inh = []
        trsnow_allSamps_allExc = []
        trsnow_allSamps_exc = []
        
    
    #######    
    if testIncorr:
        perClassErrorTest_data_inh_incorr = Datai3.pop('perClassErrorTest_data_inh_incorr')
        perClassErrorTest_shfl_inh_incorr = Datai3.pop('perClassErrorTest_shfl_inh_incorr')

        perClassErrorTest_data_allExc_incorr = Dataae3.pop('perClassErrorTest_data_allExc_incorr')
        perClassErrorTest_shfl_allExc_incorr = Dataae3.pop('perClassErrorTest_shfl_allExc_incorr')
        
        perClassErrorTest_data_exc_incorr = Datae3.pop('perClassErrorTest_data_exc_incorr')
        perClassErrorTest_shfl_exc_incorr = Datae3.pop('perClassErrorTest_shfl_exc_incorr')
    else:
        perClassErrorTest_data_inh_incorr = []
        perClassErrorTest_shfl_inh_incorr = []

        perClassErrorTest_data_allExc_incorr = []
        perClassErrorTest_shfl_allExc_incorr = []
        
        perClassErrorTest_data_exc_incorr = []
        perClassErrorTest_shfl_exc_incorr = []
        
        
    #### sanity check
    '''
    if (len(time_trace) == perClassErrorTest_data_inh.shape[-1] == perClassErrorTest_data_exc.shape[-1] == perClassErrorTest_data_allExc.shape[-1])==False:
        print '%d len(time_trace)\n%d perClassErrorTest_data_inh.shape[-1]\n%d perClassErrorTest_data_exc.shape[-1]\n%d perClassErrorTest_data_allExc.shape[-1]' %(len(time_trace), perClassErrorTest_data_inh.shape[-1], perClassErrorTest_data_exc.shape[-1], perClassErrorTest_data_allExc.shape[-1])
        sys.exit('something wrong!')
    '''    
       
    ######%% Set class errors to 50 if less than .05 fraction of neurons in a sample have non-0 weights, and set all samples class error to 50, if less than 10 samples satisfy this condition.
   
    if loadWeights==1:
        perClassErrorTest_data_inh = setTo50classErr(perClassErrorTest_data_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_shfl_inh = setTo50classErr(perClassErrorTest_shfl_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_chance_inh = setTo50classErr(perClassErrorTest_chance_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
        
        perClassErrorTest_data_allExc = setTo50classErr(perClassErrorTest_data_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_shfl_allExc = setTo50classErr(perClassErrorTest_shfl_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_chance_allExc = setTo50classErr(perClassErrorTest_chance_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
        
        perClassErrorTest_data_exc = setTo50classErr(perClassErrorTest_data_exc, w_data_exc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_shfl_exc = setTo50classErr(perClassErrorTest_shfl_exc, w_data_exc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_chance_exc = setTo50classErr(perClassErrorTest_chance_exc, w_data_exc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
    
        ##%% Get number of inh and exc        
#        numInh[iday] = w_data_inh.shape[1]
#        numAllexc[iday] = w_data_allExc.shape[1]

    if loadWeights==2:   # 2: only download weights, no CA                                     
        perClassErrorTest_data_inh = []
        perClassErrorTest_shfl_inh = []
        perClassErrorTest_chance_inh = []
        perClassErrorTest_data_allExc = []
        perClassErrorTest_shfl_allExc = []
        perClassErrorTest_chance_allExc = []
        perClassErrorTest_data_exc = []
        perClassErrorTest_shfl_exc = []
        perClassErrorTest_chance_exc = []
    
    if loadYtest:
        return perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, \
        w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN, trsExcluded, \
        testTrInds_allSamps_inh, Ytest_allSamps_inh, Ytest_hat_allSampsFrs_inh, trsnow_allSamps_inh, \
        testTrInds_allSamps_allExc, Ytest_allSamps_allExc, Ytest_hat_allSampsFrs_allExc, trsnow_allSamps_allExc, \
        testTrInds_allSamps_exc, Ytest_allSamps_exc, Ytest_hat_allSampsFrs_exc, trsnow_allSamps_exc
        
    else:
        return perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, \
        w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN, trsExcluded, \
        perClassErrorTest_data_inh_incorr, perClassErrorTest_shfl_inh_incorr, \
        perClassErrorTest_data_allExc_incorr, perClassErrorTest_shfl_allExc_incorr, \
        perClassErrorTest_data_exc_incorr, perClassErrorTest_shfl_exc_incorr



#%% Load exc,inh SVM vars
        
def loadSVM_excInh_decodeStimCateg(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, doPlots, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0, loadYtest=0):
    # loadWeights: 
    # 0: don't load weights, only load class accur
    # 1 : load weights and class cccur
    # 2 : load only weights and not class accur
    # shflTrLabs=0; loadYtest=0
    
    '''
    if loadInhAllexcEqexc==0: # 1st run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc at the same time, also for all days (except a few days of fni18), inhRois was used (not the new inhRois_pix)       
        svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, [], regressBins)
        svmName = svmName[0]
        print os.path.basename(svmName)    

        Data = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_inh', 'perClassErrorTest_shfl_inh', 'perClassErrorTest_chance_inh',    
                                                 'perClassErrorTest_data_allExc', 'perClassErrorTest_shfl_allExc', 'perClassErrorTest_chance_allExc',    
                                                 'perClassErrorTest_data_exc', 'perClassErrorTest_shfl_exc', 'perClassErrorTest_chance_exc',
                                                 'w_data_inh', 'w_data_allExc', 'w_data_exc'])        
        Datai = Dataae = Datae = Data                                                 

    else:  # 2nd run of the svm_excInh_trainDecoder_eachFrame code: you ran inh,exc,allExc separately; also for all days the new vector inhRois_pix was used (not the old inhRois)       
    '''
    
#    if doAllN==1:
#        smallestC = 0 # Identify best c: if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
#        if smallestC==1:
#            print 'bestc = smallest c whose cv error is less than 1se of min cv error'
#        else:
#            print 'bestc = c that gives min cv error'
            
    svmName_allN = ''            
    svmName_excInh = []
    for idi in range(3):
        
        doInhAllexcEqexc = np.full((4), 0, dtype=int)
        doInhAllexcEqexc[-1] = 0
        doInhAllexcEqexc[idi] = 1
        
        ########## set svm file name ##########
        if idi==1 and doAllN: # plot allN, instead of allExc
            # old:
#            svmName = setSVMname_allN_eachFrame(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, shflTrsEachNeuron, shflTrLabs)[0] # for chAl: the latest file is with soft norm; earlier file is 
            # new svm_excInh_trainDecoder_eachFrame.py computes allN too... 
            doInhAllexcEqexc[1] = 2
            svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc, regressBins, useEqualTrNums, corrTrained, shflTrsEachNeuron, shflTrLabs)[0]
            svmName_allN = svmName
        else:        
            svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc, regressBins, useEqualTrNums, corrTrained, shflTrsEachNeuron, shflTrLabs)[0]
            svmName_excInh.append(svmName)
#        svmName = svmName[0] # use [0] for the latest file; use [-1] for the earliest file
        print os.path.basename(svmName)    


        ######### inh ######### 
        if doInhAllexcEqexc[0] == 1: 
            if loadWeights==1:
                Datai = scio.loadmat(svmName, variable_names=['trsExcluded', 'perClassErrorTest_data_inh', 'perClassErrorTest_shfl_inh', 'perClassErrorTest_chance_inh', 'w_data_inh', 'b_data_inh'])
            elif loadWeights==2:                
                Datai = scio.loadmat(svmName, variable_names=['trsExcluded', 'w_data_inh','b_data_inh'])
            else:
                Datai = scio.loadmat(svmName, variable_names=['trsExcluded', 'perClassErrorTrain_data_inh', 'perClassErrorTest_data_inh', 'perClassErrorTest_shfl_inh', 'perClassErrorTest_chance_inh', 'perClassErrorTestRemCorr_inh', 'perClassErrorTestRemCorr_shfl_inh', 'perClassErrorTestRemCorr_chance_inh', 'perClassErrorTestBoth_inh', 'perClassErrorTestBoth_shfl_inh', 'perClassErrorTestBoth_chance_inh'])
            if loadYtest:
                Datai2 = scio.loadmat(svmName, variable_names=['trsnow_allSamps','testTrInds_allSamps_inh', 'Ytest_allSamps_inh', 'Ytest_hat_allSampsFrs_inh'])
                
                
        ######### allN or allExc ######### 
        elif np.logical_or(doInhAllexcEqexc[1] == 1, doInhAllexcEqexc[1] == 2): 
#            if doAllN: # plot allN, instead of allExc
#                _, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, _, w_data_allExc, b_data_allExc = loadSVM_allN(svmName, doPlots, doIncorr, 1, shflTrLabs)   
#                
#            else: # plot allExc
            if loadWeights==1:
                Dataae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_allExc', 'perClassErrorTest_shfl_allExc', 'perClassErrorTest_chance_allExc', 'w_data_allExc', 'b_data_allExc'])
            elif loadWeights==2:                
                Dataae = scio.loadmat(svmName, variable_names=['w_data_allExc', 'b_data_allExc'])
            else:
                Dataae = scio.loadmat(svmName, variable_names=['perClassErrorTrain_data_allExc', 'perClassErrorTest_data_allExc', 'perClassErrorTest_shfl_allExc', 'perClassErrorTest_chance_allExc', 'perClassErrorTestRemCorr_allExc', 'perClassErrorTestRemCorr_shfl_allExc', 'perClassErrorTestRemCorr_chance_allExc', 'perClassErrorTestBoth_allExc', 'perClassErrorTestBoth_shfl_allExc', 'perClassErrorTestBoth_chance_allExc'])
            if loadYtest:
                Dataae2 = scio.loadmat(svmName, variable_names=['trsnow_allSamps','testTrInds_allSamps_allExc', 'Ytest_allSamps_allExc', 'Ytest_hat_allSampsFrs_allExc'])

        
        ######### eqExc ######### 
        elif doInhAllexcEqexc[2] == 1: 
            if loadWeights==1:
                Datae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_exc', 'perClassErrorTest_shfl_exc', 'perClassErrorTest_chance_exc', 'w_data_exc', 'b_data_exc'])                                                 
            elif loadWeights==2:                                
                Datae = scio.loadmat(svmName, variable_names=['w_data_exc', 'b_data_exc'])                                                 
            else:
                Datae = scio.loadmat(svmName, variable_names=['perClassErrorTrain_data_exc', 'perClassErrorTest_data_exc', 'perClassErrorTest_shfl_exc', 'perClassErrorTest_chance_exc', 'perClassErrorTestRemCorr_exc', 'perClassErrorTestRemCorr_shfl_exc', 'perClassErrorTestRemCorr_chance_exc', 'perClassErrorTestBoth_exc', 'perClassErrorTestBoth_shfl_exc', 'perClassErrorTestBoth_chance_exc'])
            if loadYtest:
                Datae2 = scio.loadmat(svmName, variable_names=['trsnow_allSamps','testTrInds_allSamps_exc', 'Ytest_allSamps_exc', 'Ytest_hat_allSampsFrs_exc'])

        
    
    #### inh #### 
    if loadWeights!=2:   # 2: only download weights, no CA                                     
        perClassErrorTest_data_inh = Datai.pop('perClassErrorTest_data_inh')
        perClassErrorTest_shfl_inh = Datai.pop('perClassErrorTest_shfl_inh')
        perClassErrorTest_chance_inh = Datai.pop('perClassErrorTest_chance_inh') 
        perClassErrorTestRemCorr_inh = Datai.pop('perClassErrorTestRemCorr_inh')
        perClassErrorTestRemCorr_shfl_inh = Datai.pop('perClassErrorTestRemCorr_shfl_inh')
        perClassErrorTestRemCorr_chance_inh = Datai.pop('perClassErrorTestRemCorr_chance_inh') 
        perClassErrorTestBoth_inh = Datai.pop('perClassErrorTestBoth_inh')
        perClassErrorTestBoth_shfl_inh = Datai.pop('perClassErrorTestBoth_shfl_inh')
        perClassErrorTestBoth_chance_inh = Datai.pop('perClassErrorTestBoth_chance_inh') 
        trsExcluded = Datai.pop('trsExcluded')
        # take care of days with equal corr and incorr, for which the CA vars of Both were set to nan... here we set them equal to testing data (bc there was no remCorr for these days, so Both will be only testing data)
        if np.all(np.isnan(perClassErrorTestBoth_inh)):
            print 'CA_Both_inh is all nan, corr & incorr must be equal, setting it same as CA_test'
            perClassErrorTestBoth_inh = perClassErrorTest_data_inh        
            perClassErrorTestBoth_shfl_inh = perClassErrorTest_shfl_inh
            perClassErrorTestBoth_chance_inh = perClassErrorTest_chance_inh
    if loadWeights>=1:
        w_data_inh = Datai.pop('w_data_inh') 
        b_data_inh = Datai.pop('b_data_inh')         
    else:
        w_data_inh = []
        b_data_inh = []
    
    
    #### allExc or allN #### 
#    if doAllN==0:
    if loadWeights!=2:
        perClassErrorTest_data_allExc = Dataae.pop('perClassErrorTest_data_allExc')
        perClassErrorTest_shfl_allExc = Dataae.pop('perClassErrorTest_shfl_allExc')
        perClassErrorTest_chance_allExc = Dataae.pop('perClassErrorTest_chance_allExc')  
        perClassErrorTestRemCorr_allExc = Dataae.pop('perClassErrorTestRemCorr_allExc')
        perClassErrorTestRemCorr_shfl_allExc = Dataae.pop('perClassErrorTestRemCorr_shfl_allExc')
        perClassErrorTestRemCorr_chance_allExc = Dataae.pop('perClassErrorTestRemCorr_chance_allExc') 
        perClassErrorTestBoth_allExc = Dataae.pop('perClassErrorTestBoth_allExc')
        perClassErrorTestBoth_shfl_allExc = Dataae.pop('perClassErrorTestBoth_shfl_allExc')
        perClassErrorTestBoth_chance_allExc = Dataae.pop('perClassErrorTestBoth_chance_allExc')         
        # take care of days with equal corr and incorr, for which the CA vars of Both were set to nan... here we set them equal to testing data (bc there was no remCorr for these days, so Both will be only testing data)
        if np.all(np.isnan(perClassErrorTestBoth_allExc)):
            print 'CA_Both_allExc is all nan, corr & incorr must be equal, setting it same as CA_test'
            perClassErrorTestBoth_allExc = perClassErrorTest_data_allExc
            perClassErrorTestBoth_shfl_allExc = perClassErrorTest_shfl_allExc
            perClassErrorTestBoth_chance_allExc = perClassErrorTest_chance_allExc
    if loadWeights>=1:
        w_data_allExc = Dataae.pop('w_data_allExc') 
        b_data_allExc = Dataae.pop('b_data_allExc') 
    else:
        w_data_allExc = []
        b_data_allExc = []
    
    
    #### exc #### 
    if np.logical_and('Datae' in locals(), loadWeights!=2):
        perClassErrorTest_data_exc = Datae.pop('perClassErrorTest_data_exc')    
        perClassErrorTest_shfl_exc = Datae.pop('perClassErrorTest_shfl_exc')
        perClassErrorTest_chance_exc = Datae.pop('perClassErrorTest_chance_exc')
        perClassErrorTestRemCorr_exc = Datae.pop('perClassErrorTestRemCorr_exc')
        perClassErrorTestRemCorr_shfl_exc = Datae.pop('perClassErrorTestRemCorr_shfl_exc')
        perClassErrorTestRemCorr_chance_exc = Datae.pop('perClassErrorTestRemCorr_chance_exc') 
        perClassErrorTestBoth_exc = Datae.pop('perClassErrorTestBoth_exc')
        perClassErrorTestBoth_shfl_exc = Datae.pop('perClassErrorTestBoth_shfl_exc')
        perClassErrorTestBoth_chance_exc = Datae.pop('perClassErrorTestBoth_chance_exc')
        # take care of days with equal corr and incorr, for which the CA vars of Both were set to nan... here we set them equal to testing data (bc there was no remCorr for these days, so Both will be only testing data)
        if np.all(np.isnan(perClassErrorTestBoth_exc)):
            print 'CA_Both_exc is all nan, corr & incorr must be equal, setting it same as CA_test'
            perClassErrorTestBoth_exc = perClassErrorTest_data_exc        
            perClassErrorTestBoth_shfl_exc = perClassErrorTest_shfl_exc
            perClassErrorTestBoth_chance_exc = perClassErrorTest_chance_exc        
    else:
        perClassErrorTest_data_exc = []
        perClassErrorTest_shfl_exc = []
        perClassErrorTest_chance_exc = []
        perClassErrorTestRemCorr_exc = []
        perClassErrorTestRemCorr_shfl_exc = []
        perClassErrorTestRemCorr_chance_exc = []
        perClassErrorTestBoth_exc = []
        perClassErrorTestBoth_shfl_exc = []
        perClassErrorTestBoth_chance_exc = []
    if loadWeights>=1:    
        w_data_exc = Datae.pop('w_data_exc') 
        b_data_exc = Datae.pop('b_data_exc') 
    else:
        w_data_exc = []
        b_data_exc = []
    
    
    #######
    if loadYtest:
        testTrInds_allSamps_inh = Datai2.pop('testTrInds_allSamps_inh').astype(int)
        Ytest_allSamps_inh = Datai2.pop('Ytest_allSamps_inh')
        Ytest_hat_allSampsFrs_inh = Datai2.pop('Ytest_hat_allSampsFrs_inh')
        trsnow_allSamps_inh = np.array(Datai2.pop('trsnow_allSamps_inh')).astype(int) # index of trials after picking random hr (or lr) in order to make sure both classes have the same number in the final Y (on which svm was run)

        testTrInds_allSamps_allExc = Dataae2.pop('testTrInds_allSamps_allExc').astype(int)
        Ytest_allSamps_allExc = Dataae2.pop('Ytest_allSamps_allExc')
        Ytest_hat_allSampsFrs_allExc = Dataae2.pop('Ytest_hat_allSampsFrs_allExc')
        trsnow_allSamps_allExc = np.array(Dataae2.pop('trsnow_allSamps_allExc')).astype(int) 
        
        testTrInds_allSamps_exc = Datae2.pop('testTrInds_allSamps_exc').astype(int)
        Ytest_allSamps_exc = Datae2.pop('Ytest_allSamps_exc')
        Ytest_hat_allSampsFrs_exc = Datae2.pop('Ytest_hat_allSampsFrs_exc')
        trsnow_allSamps_exc = np.array(Datae2.pop('trsnow_allSamps_exc')).astype(int)
    else:
        testTrInds_allSamps_inh = []
        Ytest_allSamps_inh = []
        Ytest_hat_allSampsFrs_inh = []
        testTrInds_allSamps_allExc = []
        Ytest_allSamps_allExc = []
        Ytest_hat_allSampsFrs_allExc = []        
        testTrInds_allSamps_exc = []
        Ytest_allSamps_exc = []
        Ytest_hat_allSampsFrs_exc = []
        trsnow_allSamps_inh = []
        trsnow_allSamps_allExc = []
        trsnow_allSamps_exc = []
        
    #### sanity check
    '''
    if (len(time_trace) == perClassErrorTest_data_inh.shape[-1] == perClassErrorTest_data_exc.shape[-1] == perClassErrorTest_data_allExc.shape[-1])==False:
        print '%d len(time_trace)\n%d perClassErrorTest_data_inh.shape[-1]\n%d perClassErrorTest_data_exc.shape[-1]\n%d perClassErrorTest_data_allExc.shape[-1]' %(len(time_trace), perClassErrorTest_data_inh.shape[-1], perClassErrorTest_data_exc.shape[-1], perClassErrorTest_data_allExc.shape[-1])
        sys.exit('something wrong!')
    '''    
       
    ######%% Set class errors to 50 if less than .05 fraction of neurons in a sample have non-0 weights, and set all samples class error to 50, if less than 10 samples satisfy this condition.
   
    if loadWeights==1:
        perClassErrorTest_data_inh = setTo50classErr(perClassErrorTest_data_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_shfl_inh = setTo50classErr(perClassErrorTest_shfl_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_chance_inh = setTo50classErr(perClassErrorTest_chance_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
        
        perClassErrorTest_data_allExc = setTo50classErr(perClassErrorTest_data_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_shfl_allExc = setTo50classErr(perClassErrorTest_shfl_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_chance_allExc = setTo50classErr(perClassErrorTest_chance_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
        
        perClassErrorTest_data_exc = setTo50classErr(perClassErrorTest_data_exc, w_data_exc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_shfl_exc = setTo50classErr(perClassErrorTest_shfl_exc, w_data_exc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_chance_exc = setTo50classErr(perClassErrorTest_chance_exc, w_data_exc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
    
        ##%% Get number of inh and exc        
#        numInh[iday] = w_data_inh.shape[1]
#        numAllexc[iday] = w_data_allExc.shape[1]

    if loadWeights==2:   # 2: only download weights, no CA                                     
        perClassErrorTest_data_inh = []
        perClassErrorTest_shfl_inh = []
        perClassErrorTest_chance_inh = []
        perClassErrorTest_data_allExc = []
        perClassErrorTest_shfl_allExc = []
        perClassErrorTest_chance_allExc = []
        perClassErrorTest_data_exc = []
        perClassErrorTest_shfl_exc = []
        perClassErrorTest_chance_exc = []
    
    return perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, \
    perClassErrorTestRemCorr_inh, perClassErrorTestRemCorr_shfl_inh, perClassErrorTestRemCorr_chance_inh, \
    perClassErrorTestBoth_inh, perClassErrorTestBoth_shfl_inh, perClassErrorTestBoth_chance_inh, \
    perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, \
    perClassErrorTestRemCorr_allExc, perClassErrorTestRemCorr_shfl_allExc, perClassErrorTestRemCorr_chance_allExc, \
    perClassErrorTestBoth_allExc, perClassErrorTestBoth_shfl_allExc, perClassErrorTestBoth_chance_allExc, \
    perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, \
    perClassErrorTestRemCorr_exc, perClassErrorTestRemCorr_shfl_exc, perClassErrorTestRemCorr_chance_exc, \
    perClassErrorTestBoth_exc, perClassErrorTestBoth_shfl_exc, perClassErrorTestBoth_chance_exc, \
    w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh, svmName_allN, trsExcluded, \
    testTrInds_allSamps_inh, Ytest_allSamps_inh, Ytest_hat_allSampsFrs_inh, trsnow_allSamps_inh, \
    testTrInds_allSamps_allExc, Ytest_allSamps_allExc, Ytest_hat_allSampsFrs_allExc, trsnow_allSamps_allExc, \
    testTrInds_allSamps_exc, Ytest_allSamps_exc, Ytest_hat_allSampsFrs_exc, trsnow_allSamps_exc
    



#%% Load exc,inh SVM vars for excInhHalf (ie when the population consists of half exc and half inh) and allExc2inhSize (ie when populatin consists of allExc but same size as 2*inh size)
        
def loadSVM_excInh_excInhHalf(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, loadWeights, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0):
    # loadWeights: 
    # 0: don't load weights, only load class accur
    # 1 : load weights and class cccur
    # 2 : load only weights and not class accur

    svmName_excInh = []
    for idi in [2,3]:
        
        doInhAllexcEqexc = np.full((3), 0, dtype=int)
        doInhAllexcEqexc[2] = idi 

        svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc, regressBins, useEqualTrNums, corrTrained, shflTrsEachNeuron, shflTrLabs)[0]
        svmName_excInh.append(svmName)
#        svmName = svmName[0] # use [0] for the latest file; use [-1] for the earliest file
        print os.path.basename(svmName)    

        ######### excInhHalf
        if doInhAllexcEqexc[2] == 2: 
            if loadWeights==1:
                Datai = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_excInhHalf', 'perClassErrorTest_shfl_excInhHalf', 'perClassErrorTest_chance_excInhHalf', 'w_data_excInhHalf', 'b_data_excInhHalf'])
            elif loadWeights==2:                
                Datai = scio.loadmat(svmName, variable_names=['w_data_excInhHalf','b_data_excInhHalf'])
            else:
                Datai = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_excInhHalf', 'perClassErrorTest_shfl_excInhHalf', 'perClassErrorTest_chance_excInhHalf'])
        
        ######### allExc2inhSize
        elif doInhAllexcEqexc[2] == 3: 
            if loadWeights==1:
                Dataae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_allExc2inhSize', 'perClassErrorTest_shfl_allExc2inhSize', 'perClassErrorTest_chance_allExc2inhSize', 'w_data_allExc2inhSize', 'b_data_allExc2inhSize'])
            elif loadWeights==2:                
                Dataae = scio.loadmat(svmName, variable_names=['w_data_allExc2inhSize', 'b_data_allExc2inhSize'])
            else:
                Dataae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_allExc2inhSize', 'perClassErrorTest_shfl_allExc2inhSize', 'perClassErrorTest_chance_allExc2inhSize'])
        

        
    ###%%             
    if loadWeights!=2:   # 2: only download weights, no CA                                     
        perClassErrorTest_data_inh = Datai.pop('perClassErrorTest_data_excInhHalf')
        perClassErrorTest_shfl_inh = Datai.pop('perClassErrorTest_shfl_excInhHalf')
        perClassErrorTest_chance_inh = Datai.pop('perClassErrorTest_chance_excInhHalf') 
    if loadWeights>=1:
        w_data_inh = Datai.pop('w_data_excInhHalf') 
        b_data_inh = Datai.pop('b_data_excInhHalf')         
    else:
        w_data_inh = []
        b_data_inh = []

        
    if loadWeights!=2:
        perClassErrorTest_data_allExc = Dataae.pop('perClassErrorTest_data_allExc2inhSize')
        perClassErrorTest_shfl_allExc = Dataae.pop('perClassErrorTest_shfl_allExc2inhSize')
        perClassErrorTest_chance_allExc = Dataae.pop('perClassErrorTest_chance_allExc2inhSize')   
    if loadWeights>=1:
        w_data_allExc = Dataae.pop('w_data_allExc2inhSize') 
        b_data_allExc = Dataae.pop('b_data_allExc2inhSize') 
    else:
        w_data_allExc = []
        b_data_allExc = []
    

       
    ######%% Set class errors to 50 if less than .05 fraction of neurons in a sample have non-0 weights, and set all samples class error to 50, if less than 10 samples satisfy this condition.
   
    if loadWeights==1:
        perClassErrorTest_data_inh = setTo50classErr(perClassErrorTest_data_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_shfl_inh = setTo50classErr(perClassErrorTest_shfl_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_chance_inh = setTo50classErr(perClassErrorTest_chance_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
        
        perClassErrorTest_data_allExc = setTo50classErr(perClassErrorTest_data_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_shfl_allExc = setTo50classErr(perClassErrorTest_shfl_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_chance_allExc = setTo50classErr(perClassErrorTest_chance_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
        
    
        ##%% Get number of inh and exc        
#        numInh[iday] = w_data_inh.shape[1]
#        numAllexc[iday] = w_data_allExc.shape[1]

    if loadWeights==2:   # 2: only download weights, no CA                                     
        perClassErrorTest_data_inh = []
        perClassErrorTest_shfl_inh = []
        perClassErrorTest_chance_inh = []
        perClassErrorTest_data_allExc = []
        perClassErrorTest_shfl_allExc = []
        perClassErrorTest_chance_allExc = []
    
    return perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, w_data_inh, w_data_allExc, b_data_inh, b_data_allExc, svmName_excInh
    
    
    

#%% Load exc,inh SVM vars for the following analysis : add neurons 1 by 1 to the decoder based on their tuning strength to see how the decoder performance increases.
        
def loadSVM_excInh_addNs1by1(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, loadWeights, useEqualTrNums, shflTrsEachNeuron, shflTrLabs=0, h2l=1):
    # loadWeights: 
    # 0: don't load weights, only load class accur
    # 1 : load weights and class cccur
    # 2 : load only weights and not class accur

    svmName_excInh = []
    for idi in [0,1,2]:
        
        doInhAllexcEqexc = np.full((4), 0, dtype=int)
        doInhAllexcEqexc[idi] = 1
        if h2l:
            doInhAllexcEqexc[3] = 1 # high to low AUC
        else:
            doInhAllexcEqexc[3] = 2 # low to high AUC

        svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc, regressBins, useEqualTrNums, corrTrained, shflTrsEachNeuron, shflTrLabs)[0]
        svmName_excInh.append(svmName)
#        svmName = svmName[0] # use [0] for the latest file; use [-1] for the earliest file
        print os.path.basename(svmName)    

        ######### inh
        if doInhAllexcEqexc[0] == 1: 
            if loadWeights==1:
                Datai = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_inh_numNs', 'perClassErrorTest_shfl_inh_numNs', 'perClassErrorTest_chance_inh_numNs', 'w_data_inh_numNs', 'b_data_inh_numNs'])
            elif loadWeights==2:                
                Datai = scio.loadmat(svmName, variable_names=['w_data_inh_numNs','b_data_inh_numNs'])
            else:
                Datai = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_inh_numNs', 'perClassErrorTest_shfl_inh_numNs', 'perClassErrorTest_chance_inh_numNs'])
        
        ######### allExc
        elif doInhAllexcEqexc[1] == 1: 
            if loadWeights==1:
                Dataae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_allExc_numNs', 'perClassErrorTest_shfl_allExc_numNs', 'perClassErrorTest_chance_allExc_numNs', 'w_data_allExc_numNs', 'b_data_allExc_numNs'])
            elif loadWeights==2:                
                Dataae = scio.loadmat(svmName, variable_names=['w_data_allExc_numNs', 'b_data_allExc_numNs'])
            else:
                Dataae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_allExc_numNs', 'perClassErrorTest_shfl_allExc_numNs', 'perClassErrorTest_chance_allExc_numNs'])

        ######### eqExc
        elif doInhAllexcEqexc[2] == 1: 
            if loadWeights==1:
                Datae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_exc_numNs', 'perClassErrorTest_shfl_exc_numNs', 'perClassErrorTest_chance_exc_numNs', 'w_data_exc_numNs', 'b_data_exc_numNs'])                                                 
            elif loadWeights==2:                                
                Datae = scio.loadmat(svmName, variable_names=['w_data_exc_numNs', 'b_data_exc_numNs'])                                                 
            else:
                Datae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_exc_numNs', 'perClassErrorTest_shfl_exc_numNs', 'perClassErrorTest_chance_exc_numNs'])
        

        
    ###%%             
    if loadWeights!=2:   # 2: only download weights, no CA                                     
        perClassErrorTest_data_inh = Datai.pop('perClassErrorTest_data_inh_numNs')
        perClassErrorTest_shfl_inh = Datai.pop('perClassErrorTest_shfl_inh_numNs')
        perClassErrorTest_chance_inh = Datai.pop('perClassErrorTest_chance_inh_numNs') 
    if loadWeights>=1:
        w_data_inh = Datai.pop('w_data_inh_numNs') 
        b_data_inh = Datai.pop('b_data_inh_numNs')         
    else:
        w_data_inh = []
        b_data_inh = []

        
    if loadWeights!=2:
        perClassErrorTest_data_allExc = Dataae.pop('perClassErrorTest_data_allExc_numNs')
        perClassErrorTest_shfl_allExc = Dataae.pop('perClassErrorTest_shfl_allExc_numNs')
        perClassErrorTest_chance_allExc = Dataae.pop('perClassErrorTest_chance_allExc_numNs')   
    if loadWeights>=1:
        w_data_allExc = Dataae.pop('w_data_allExc_numNs') 
        b_data_allExc = Dataae.pop('b_data_allExc_numNs') 
    else:
        w_data_allExc = []
        b_data_allExc = []
    

    if loadWeights!=2:
        perClassErrorTest_data_exc = Datae.pop('perClassErrorTest_data_exc_numNs')    
        perClassErrorTest_shfl_exc = Datae.pop('perClassErrorTest_shfl_exc_numNs')
        perClassErrorTest_chance_exc = Datae.pop('perClassErrorTest_chance_exc_numNs')
    if loadWeights>=1:    
        w_data_exc = Datae.pop('w_data_exc_numNs') 
        b_data_exc = Datae.pop('b_data_exc_numNs') 
    else:
        w_data_exc = []
        b_data_exc = []
        
        
    ######%% Set class errors to 50 if less than .05 fraction of neurons in a sample have non-0 weights, and set all samples class error to 50, if less than 10 samples satisfy this condition.
   
    if loadWeights==1:
        perClassErrorTest_data_inh = setTo50classErr(perClassErrorTest_data_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_shfl_inh = setTo50classErr(perClassErrorTest_shfl_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_chance_inh = setTo50classErr(perClassErrorTest_chance_inh, w_data_inh, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
        
        perClassErrorTest_data_allExc = setTo50classErr(perClassErrorTest_data_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_shfl_allExc = setTo50classErr(perClassErrorTest_shfl_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_chance_allExc = setTo50classErr(perClassErrorTest_chance_allExc, w_data_allExc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    

        perClassErrorTest_data_exc = setTo50classErr(perClassErrorTest_data_exc, w_data_exc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_shfl_exc = setTo50classErr(perClassErrorTest_shfl_exc, w_data_exc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)
        perClassErrorTest_chance_exc = setTo50classErr(perClassErrorTest_chance_exc, w_data_exc, thNon0Ws = .05, thSamps = 10, eps = 1e-10)    
        
        
        ##%% Get number of inh and exc        
#        numInh[iday] = w_data_inh.shape[1]
#        numAllexc[iday] = w_data_allExc.shape[1]

    if loadWeights==2:   # 2: only download weights, no CA                                     
        perClassErrorTest_data_inh = []
        perClassErrorTest_shfl_inh = []
        perClassErrorTest_chance_inh = []
        perClassErrorTest_data_allExc = []
        perClassErrorTest_shfl_allExc = []
        perClassErrorTest_chance_allExc = []
        perClassErrorTest_data_exc = []
        perClassErrorTest_shfl_exc = []
        perClassErrorTest_chance_exc = []        
            

#    return perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, w_data_inh, w_data_allExc, b_data_inh, b_data_allExc, svmName_excInh    
    return perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc, svmName_excInh
    

    
#%% Average and st error of class accuracies across CV samples ... for each day

def av_se_CA_trsamps(numD, perClassErrorTest_data_inh_all, perClassErrorTest_shfl_inh_all, perClassErrorTest_chance_inh_all, perClassErrorTest_data_exc_all, perClassErrorTest_shfl_exc_all, perClassErrorTest_chance_exc_all, perClassErrorTest_data_allExc_all, perClassErrorTest_shfl_allExc_all, perClassErrorTest_chance_allExc_all):
    
#    numD = len(eventI_allDays)
    numSamples = np.shape(perClassErrorTest_data_inh_all[0])[0]
    numExcSamples = np.shape(perClassErrorTest_data_exc_all[0])[0]
    
    #### inh
    av_test_data_inh = np.array([100-np.nanmean(perClassErrorTest_data_inh_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_data_inh = np.array([np.nanstd(perClassErrorTest_data_inh_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  
    
    av_test_shfl_inh = np.array([100-np.nanmean(perClassErrorTest_shfl_inh_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_shfl_inh = np.array([np.nanstd(perClassErrorTest_shfl_inh_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  
    
    av_test_chance_inh = np.array([100-np.nanmean(perClassErrorTest_chance_inh_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_chance_inh = np.array([np.nanstd(perClassErrorTest_chance_inh_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  
    
    
    #### exc (average across cv samples and exc shuffles)
    av_test_data_exc = np.array([100-np.nanmean(perClassErrorTest_data_exc_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
    sd_test_data_exc = np.array([np.nanstd(perClassErrorTest_data_exc_all[iday], axis=(0,1)) / np.sqrt(numSamples+numExcSamples) for iday in range(numD)])  
    
    av_test_shfl_exc = np.array([100-np.nanmean(perClassErrorTest_shfl_exc_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
    sd_test_shfl_exc = np.array([np.nanstd(perClassErrorTest_shfl_exc_all[iday], axis=(0,1)) / np.sqrt(numSamples+numExcSamples) for iday in range(numD)])  
    
    av_test_chance_exc = np.array([100-np.nanmean(perClassErrorTest_chance_exc_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
    sd_test_chance_exc = np.array([np.nanstd(perClassErrorTest_chance_exc_all[iday], axis=(0,1)) / np.sqrt(numSamples+numExcSamples) for iday in range(numD)])  
    
    
    #### allExc
    av_test_data_allExc = np.array([100-np.nanmean(perClassErrorTest_data_allExc_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_data_allExc = np.array([np.nanstd(perClassErrorTest_data_allExc_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  
    
    av_test_shfl_allExc = np.array([100-np.nanmean(perClassErrorTest_shfl_allExc_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_shfl_allExc = np.array([np.nanstd(perClassErrorTest_shfl_allExc_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  
    
    av_test_chance_allExc = np.array([100-np.nanmean(perClassErrorTest_chance_allExc_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_chance_allExc = np.array([np.nanstd(perClassErrorTest_chance_allExc_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  
    
    return numSamples, numExcSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, \
        av_test_data_exc, sd_test_data_exc, av_test_shfl_exc, sd_test_shfl_exc, av_test_chance_exc, sd_test_chance_exc, \
        av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc


#%%
        
def av_se_CA_trsamps_decodeStimCateg(numD, perClassErrorTest_data_inh_all, perClassErrorTest_shfl_inh_all, perClassErrorTest_chance_inh_all, \
                                     perClassErrorTestRemCorr_inh_all, perClassErrorTestRemCorr_shfl_inh_all, perClassErrorTestRemCorr_chance_inh_all, \
                                     perClassErrorTestBoth_inh_all, perClassErrorTestBoth_shfl_inh_all, perClassErrorTestBoth_chance_inh_all, \
                                     perClassErrorTest_data_exc_all, perClassErrorTest_shfl_exc_all, perClassErrorTest_chance_exc_all, \
                                     perClassErrorTestRemCorr_exc_all, perClassErrorTestRemCorr_shfl_exc_all, perClassErrorTestRemCorr_chance_exc_all, \
                                     perClassErrorTestBoth_exc_all, perClassErrorTestBoth_shfl_exc_all, perClassErrorTestBoth_chance_exc_all, \
                                     perClassErrorTest_data_allExc_all, perClassErrorTest_shfl_allExc_all, perClassErrorTest_chance_allExc_all, \
                                     perClassErrorTestRemCorr_allExc_all, perClassErrorTestRemCorr_shfl_allExc_all, perClassErrorTestRemCorr_chance_allExc_all, \
                                     perClassErrorTestBoth_allExc_all, perClassErrorTestBoth_shfl_allExc_all, perClassErrorTestBoth_chance_allExc_all):
    

#    numD = len(eventI_allDays)
    numSamples = np.shape(perClassErrorTest_data_inh_all[0])[0]
    numExcSamples = np.shape(perClassErrorTest_data_exc_all[0])[0]
    
    #### inh
    av_test_data_inh = np.array([100-np.nanmean(perClassErrorTest_data_inh_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_data_inh = np.array([np.nanstd(perClassErrorTest_data_inh_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])      
    av_test_shfl_inh = np.array([100-np.nanmean(perClassErrorTest_shfl_inh_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_shfl_inh = np.array([np.nanstd(perClassErrorTest_shfl_inh_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])      
    av_test_chance_inh = np.array([100-np.nanmean(perClassErrorTest_chance_inh_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_chance_inh = np.array([np.nanstd(perClassErrorTest_chance_inh_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  
    av_test_remCorr_data_inh = np.array([100-np.nanmean(perClassErrorTestRemCorr_inh_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_remCorr_data_inh = np.array([np.nanstd(perClassErrorTestRemCorr_inh_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])      
    av_test_remCorr_shfl_inh = np.array([100-np.nanmean(perClassErrorTestRemCorr_shfl_inh_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_remCorr_shfl_inh = np.array([np.nanstd(perClassErrorTestRemCorr_shfl_inh_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])      
    av_test_remCorr_chance_inh = np.array([100-np.nanmean(perClassErrorTestRemCorr_chance_inh_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_remCorr_chance_inh = np.array([np.nanstd(perClassErrorTestRemCorr_chance_inh_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  
    av_test_both_data_inh = np.array([100-np.nanmean(perClassErrorTestBoth_inh_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_both_data_inh = np.array([np.nanstd(perClassErrorTestBoth_inh_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])      
    av_test_both_shfl_inh = np.array([100-np.nanmean(perClassErrorTestBoth_shfl_inh_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_both_shfl_inh = np.array([np.nanstd(perClassErrorTestBoth_shfl_inh_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])      
    av_test_both_chance_inh = np.array([100-np.nanmean(perClassErrorTestBoth_chance_inh_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_both_chance_inh = np.array([np.nanstd(perClassErrorTestBoth_chance_inh_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  
    
    
    #### exc (average across cv samples and exc shuffles)
    if len(perClassErrorTest_data_exc_all[0])>0:
        av_test_data_exc = np.array([100-np.nanmean(perClassErrorTest_data_exc_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
        sd_test_data_exc = np.array([np.nanstd(perClassErrorTest_data_exc_all[iday], axis=(0,1)) / np.sqrt(numSamples+numExcSamples) for iday in range(numD)])  
        av_test_shfl_exc = np.array([100-np.nanmean(perClassErrorTest_shfl_exc_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
        sd_test_shfl_exc = np.array([np.nanstd(perClassErrorTest_shfl_exc_all[iday], axis=(0,1)) / np.sqrt(numSamples+numExcSamples) for iday in range(numD)])  
        av_test_chance_exc = np.array([100-np.nanmean(perClassErrorTest_chance_exc_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
        sd_test_chance_exc = np.array([np.nanstd(perClassErrorTest_chance_exc_all[iday], axis=(0,1)) / np.sqrt(numSamples+numExcSamples) for iday in range(numD)])  
        av_test_remCorr_data_exc = np.array([100-np.nanmean(perClassErrorTestRemCorr_exc_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
        sd_test_remCorr_data_exc = np.array([np.nanstd(perClassErrorTestRemCorr_exc_all[iday], axis=(0,1)) / np.sqrt(numSamples) for iday in range(numD)])      
        av_test_remCorr_shfl_exc = np.array([100-np.nanmean(perClassErrorTestRemCorr_shfl_exc_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
        sd_test_remCorr_shfl_exc = np.array([np.nanstd(perClassErrorTestRemCorr_shfl_exc_all[iday], axis=(0,1)) / np.sqrt(numSamples) for iday in range(numD)])      
        av_test_remCorr_chance_exc = np.array([100-np.nanmean(perClassErrorTestRemCorr_chance_exc_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
        sd_test_remCorr_chance_exc = np.array([np.nanstd(perClassErrorTestRemCorr_chance_exc_all[iday], axis=(0,1)) / np.sqrt(numSamples) for iday in range(numD)])  
        av_test_both_data_exc = np.array([100-np.nanmean(perClassErrorTestBoth_exc_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
        sd_test_both_data_exc = np.array([np.nanstd(perClassErrorTestBoth_exc_all[iday], axis=(0,1)) / np.sqrt(numSamples) for iday in range(numD)])      
        av_test_both_shfl_exc = np.array([100-np.nanmean(perClassErrorTestBoth_shfl_exc_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
        sd_test_both_shfl_exc = np.array([np.nanstd(perClassErrorTestBoth_shfl_exc_all[iday], axis=(0,1)) / np.sqrt(numSamples) for iday in range(numD)])      
        av_test_both_chance_exc = np.array([100-np.nanmean(perClassErrorTestBoth_chance_exc_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
        sd_test_both_chance_exc = np.array([np.nanstd(perClassErrorTestBoth_chance_exc_all[iday], axis=(0,1)) / np.sqrt(numSamples) for iday in range(numD)])  

    else:
        av_test_data_exc = []
        sd_test_data_exc = []        
        av_test_shfl_exc = []
        sd_test_shfl_exc = []        
        av_test_chance_exc = []
        sd_test_chance_exc = []
        av_test_remCorr_data_exc = []
        sd_test_remCorr_data_exc = []
        av_test_remCorr_shfl_exc = []
        sd_test_remCorr_shfl_exc = []
        av_test_remCorr_chance_exc = []
        sd_test_remCorr_chance_exc = []
        av_test_both_data_exc = []
        sd_test_both_data_exc = []
        av_test_both_shfl_exc = []
        sd_test_both_shfl_exc = []
        av_test_both_chance_exc = []
        sd_test_both_chance_exc = []
    
    #### allExc
    av_test_data_allExc = np.array([100-np.nanmean(perClassErrorTest_data_allExc_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_data_allExc = np.array([np.nanstd(perClassErrorTest_data_allExc_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])    
    av_test_shfl_allExc = np.array([100-np.nanmean(perClassErrorTest_shfl_allExc_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_shfl_allExc = np.array([np.nanstd(perClassErrorTest_shfl_allExc_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  
    av_test_chance_allExc = np.array([100-np.nanmean(perClassErrorTest_chance_allExc_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_chance_allExc = np.array([np.nanstd(perClassErrorTest_chance_allExc_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  
    av_test_remCorr_data_allExc = np.array([100-np.nanmean(perClassErrorTestRemCorr_allExc_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_remCorr_data_allExc = np.array([np.nanstd(perClassErrorTestRemCorr_allExc_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])      
    av_test_remCorr_shfl_allExc = np.array([100-np.nanmean(perClassErrorTestRemCorr_shfl_allExc_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_remCorr_shfl_allExc = np.array([np.nanstd(perClassErrorTestRemCorr_shfl_allExc_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])      
    av_test_remCorr_chance_allExc = np.array([100-np.nanmean(perClassErrorTestRemCorr_chance_allExc_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_remCorr_chance_allExc = np.array([np.nanstd(perClassErrorTestRemCorr_chance_allExc_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  
    av_test_both_data_allExc = np.array([100-np.nanmean(perClassErrorTestBoth_allExc_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_both_data_allExc = np.array([np.nanstd(perClassErrorTestBoth_allExc_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])      
    av_test_both_shfl_allExc = np.array([100-np.nanmean(perClassErrorTestBoth_shfl_allExc_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_both_shfl_allExc = np.array([np.nanstd(perClassErrorTestBoth_shfl_allExc_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])      
    av_test_both_chance_allExc = np.array([100-np.nanmean(perClassErrorTestBoth_chance_allExc_all[iday], axis=0) for iday in range(numD)]) # numDays
    sd_test_both_chance_allExc = np.array([np.nanstd(perClassErrorTestBoth_chance_allExc_all[iday], axis=0) / np.sqrt(numSamples) for iday in range(numD)])  
    
    return numSamples, numExcSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, \
        av_test_remCorr_data_inh, sd_test_remCorr_data_inh, av_test_remCorr_shfl_inh, sd_test_remCorr_shfl_inh, av_test_remCorr_chance_inh, sd_test_remCorr_chance_inh, \
        av_test_both_data_inh, sd_test_both_data_inh, av_test_both_shfl_inh, sd_test_both_shfl_inh, av_test_both_chance_inh, sd_test_both_chance_inh, \
        av_test_data_exc, sd_test_data_exc, av_test_shfl_exc, sd_test_shfl_exc, av_test_chance_exc, sd_test_chance_exc, \
        av_test_remCorr_data_exc, sd_test_remCorr_data_exc, av_test_remCorr_shfl_exc, sd_test_remCorr_shfl_exc, av_test_remCorr_chance_exc, sd_test_remCorr_chance_exc, \
        av_test_both_data_exc, sd_test_both_data_exc, av_test_both_shfl_exc, sd_test_both_shfl_exc, av_test_both_chance_exc, sd_test_both_chance_exc, \
        av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc, \
        av_test_remCorr_data_allExc, sd_test_remCorr_data_allExc, av_test_remCorr_shfl_allExc, sd_test_remCorr_shfl_allExc, av_test_remCorr_chance_allExc, sd_test_remCorr_chance_allExc, \
        av_test_both_data_allExc, sd_test_both_data_allExc, av_test_both_shfl_allExc, sd_test_both_shfl_allExc, av_test_both_chance_allExc, sd_test_both_chance_allExc
    
    

#%% Same as above but for the excInhHalf case: Average and st error of class accuracies across CV samples ... for each day

def av_se_CA_trsamps_excInhHalf(numD, perClassErrorTest_data_inh_all, perClassErrorTest_shfl_inh_all, perClassErrorTest_chance_inh_all, 
                                perClassErrorTest_data_allExc_all, perClassErrorTest_shfl_allExc_all, perClassErrorTest_chance_allExc_all):
    
#    numD = len(eventI_allDays)
    numSamples = np.shape(perClassErrorTest_data_allExc_all[0])[1]
    numExcSamples = np.shape(perClassErrorTest_data_allExc_all[0])[0]
    
    #### excInhHalf (average across cv samples and exc shuffles)
    av_test_data_inh = np.array([100-np.nanmean(perClassErrorTest_data_inh_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
    sd_test_data_inh = np.array([np.nanstd(perClassErrorTest_data_inh_all[iday], axis=(0,1)) / np.sqrt(numSamples+numExcSamples) for iday in range(numD)])  
    
    av_test_shfl_inh = np.array([100-np.nanmean(perClassErrorTest_shfl_inh_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
    sd_test_shfl_inh = np.array([np.nanstd(perClassErrorTest_shfl_inh_all[iday], axis=(0,1)) / np.sqrt(numSamples+numExcSamples) for iday in range(numD)])  
    
    av_test_chance_inh = np.array([100-np.nanmean(perClassErrorTest_chance_inh_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
    sd_test_chance_inh = np.array([np.nanstd(perClassErrorTest_chance_inh_all[iday], axis=(0,1)) / np.sqrt(numSamples+numExcSamples) for iday in range(numD)])  
     
    
    #### allExc2inhSize (average across cv samples and exc shuffles)
    av_test_data_allExc = np.array([100-np.nanmean(perClassErrorTest_data_allExc_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
    sd_test_data_allExc = np.array([np.nanstd(perClassErrorTest_data_allExc_all[iday], axis=(0,1)) / np.sqrt(numSamples+numExcSamples) for iday in range(numD)])  
    
    av_test_shfl_allExc = np.array([100-np.nanmean(perClassErrorTest_shfl_allExc_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
    sd_test_shfl_allExc = np.array([np.nanstd(perClassErrorTest_shfl_allExc_all[iday], axis=(0,1)) / np.sqrt(numSamples+numExcSamples) for iday in range(numD)])  
    
    av_test_chance_allExc = np.array([100-np.nanmean(perClassErrorTest_chance_allExc_all[iday], axis=(0,1)) for iday in range(numD)]) # numDays
    sd_test_chance_allExc = np.array([np.nanstd(perClassErrorTest_chance_allExc_all[iday], axis=(0,1)) / np.sqrt(numSamples+numExcSamples) for iday in range(numD)])  
    
    
    return numSamples, numExcSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, \
        av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc



#%% Same as above but for the analysis of adding neurons 1 by 1

def av_se_CA_trsamps_addNs1by1(numD, perClassErrorTest_data_inh_all, perClassErrorTest_shfl_inh_all, perClassErrorTest_chance_inh_all, 
                                perClassErrorTest_data_exc_all, perClassErrorTest_shfl_exc_all, perClassErrorTest_chance_exc_all, 
                                perClassErrorTest_data_allExc_all, perClassErrorTest_shfl_allExc_all, perClassErrorTest_chance_allExc_all, fr2an, eventI_ds_allDays):
        
#    numD = len(eventI_allDays)
    numSamples = np.shape(perClassErrorTest_data_allExc_all[0])[1] # number of neurons in the decoder x nSamps x nFrs
    numExcSamples = np.shape(perClassErrorTest_data_exc_all[0])[0] # numShufflesExc x number of neurons in the decoder x numSamples x nFrs
    
    #### inh
    av_test_data_inh = np.array([100-np.nanmean(perClassErrorTest_data_inh_all[iday][:,:, eventI_ds_allDays[iday]+fr2an], axis=1) for iday in range(numD)]) # numDays; each day: number of neurons in the decoder
    sd_test_data_inh = np.array([np.nanstd(perClassErrorTest_data_inh_all[iday][:,:, eventI_ds_allDays[iday]+fr2an], axis=1) / np.sqrt(numSamples) for iday in range(numD)])  
    
    av_test_shfl_inh = np.array([100-np.nanmean(perClassErrorTest_shfl_inh_all[iday][:,:, eventI_ds_allDays[iday]+fr2an], axis=1) for iday in range(numD)]) 
    sd_test_shfl_inh = np.array([np.nanstd(perClassErrorTest_shfl_inh_all[iday][:,:, eventI_ds_allDays[iday]+fr2an], axis=1) / np.sqrt(numSamples) for iday in range(numD)])  
    
    av_test_chance_inh = np.array([100-np.nanmean(perClassErrorTest_chance_inh_all[iday][:,:, eventI_ds_allDays[iday]+fr2an], axis=1) for iday in range(numD)]) # 
    sd_test_chance_inh = np.array([np.nanstd(perClassErrorTest_chance_inh_all[iday][:,:, eventI_ds_allDays[iday]+fr2an], axis=1) / np.sqrt(numSamples) for iday in range(numD)])  
 

    #### exc (average across cv samples and exc shuffles)
    av_test_data_exc = np.array([100-np.nanmean(perClassErrorTest_data_exc_all[iday][:,:,:, eventI_ds_allDays[iday]+fr2an], axis=(0,2)) for iday in range(numD)]) # numDays
    sd_test_data_exc = np.array([np.nanstd(perClassErrorTest_data_exc_all[iday][:,:,:, eventI_ds_allDays[iday]+fr2an], axis=(0,2)) / np.sqrt(numSamples+numExcSamples) for iday in range(numD)])  
    
    av_test_shfl_exc = np.array([100-np.nanmean(perClassErrorTest_shfl_exc_all[iday][:,:,:, eventI_ds_allDays[iday]+fr2an], axis=(0,2)) for iday in range(numD)]) # numDays
    sd_test_shfl_exc = np.array([np.nanstd(perClassErrorTest_shfl_exc_all[iday][:,:,:, eventI_ds_allDays[iday]+fr2an], axis=(0,2)) / np.sqrt(numSamples+numExcSamples) for iday in range(numD)])  
    
    av_test_chance_exc = np.array([100-np.nanmean(perClassErrorTest_chance_exc_all[iday][:,:,:, eventI_ds_allDays[iday]+fr2an], axis=(0,2)) for iday in range(numD)]) # numDays
    sd_test_chance_exc = np.array([np.nanstd(perClassErrorTest_chance_exc_all[iday][:,:,:, eventI_ds_allDays[iday]+fr2an], axis=(0,2)) / np.sqrt(numSamples+numExcSamples) for iday in range(numD)])  
    # 1 exc subsample
    '''
    shf = rng.permutation(numExcSamples)
    av_test_data_exc = np.array([100-np.nanmean(perClassErrorTest_data_exc_all[iday][shf[iday],:,:, eventI_ds_allDays[iday]+fr2an], axis=1) for iday in range(numD)]) # numDays
    sd_test_data_exc = np.array([np.nanstd(perClassErrorTest_data_exc_all[iday][shf[iday],:,:, eventI_ds_allDays[iday]+fr2an], axis=1) / np.sqrt(numSamples) for iday in range(numD)])  
    
    av_test_shfl_exc = np.array([100-np.nanmean(perClassErrorTest_shfl_exc_all[iday][shf[iday],:,:, eventI_ds_allDays[iday]+fr2an], axis=1) for iday in range(numD)]) # numDays
    sd_test_shfl_exc = np.array([np.nanstd(perClassErrorTest_shfl_exc_all[iday][shf[iday],:,:, eventI_ds_allDays[iday]+fr2an], axis=1) / np.sqrt(numSamples) for iday in range(numD)])  
    
    av_test_chance_exc = np.array([100-np.nanmean(perClassErrorTest_chance_exc_all[iday][shf[iday],:,:, eventI_ds_allDays[iday]+fr2an], axis=1) for iday in range(numD)]) # numDays
    sd_test_chance_exc = np.array([np.nanstd(perClassErrorTest_chance_exc_all[iday][shf[iday],:,:, eventI_ds_allDays[iday]+fr2an], axis=1) / np.sqrt(numSamples) for iday in range(numD)])  
    '''
    
    #### allExc
    av_test_data_allExc = np.array([100-np.nanmean(perClassErrorTest_data_allExc_all[iday][:,:, eventI_ds_allDays[iday]+fr2an], axis=1) for iday in range(numD)]) # 
    sd_test_data_allExc = np.array([np.nanstd(perClassErrorTest_data_allExc_all[iday][:,:, eventI_ds_allDays[iday]+fr2an], axis=1) / np.sqrt(numSamples) for iday in range(numD)])  
    
    av_test_shfl_allExc = np.array([100-np.nanmean(perClassErrorTest_shfl_allExc_all[iday][:,:, eventI_ds_allDays[iday]+fr2an], axis=1) for iday in range(numD)]) # 
    sd_test_shfl_allExc = np.array([np.nanstd(perClassErrorTest_shfl_allExc_all[iday][:,:, eventI_ds_allDays[iday]+fr2an], axis=1) / np.sqrt(numSamples) for iday in range(numD)])  
    
    av_test_chance_allExc = np.array([100-np.nanmean(perClassErrorTest_chance_allExc_all[iday][:,:, eventI_ds_allDays[iday]+fr2an], axis=1) for iday in range(numD)]) # 
    sd_test_chance_allExc = np.array([np.nanstd(perClassErrorTest_chance_allExc_all[iday][:,:, eventI_ds_allDays[iday]+fr2an], axis=1) / np.sqrt(numSamples) for iday in range(numD)])  


    return numSamples, numExcSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, \
        av_test_data_exc, sd_test_data_exc, av_test_shfl_exc, sd_test_shfl_exc, av_test_chance_exc, sd_test_chance_exc, \
        av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc    
#    return numSamples, av_test_data_inh, sd_test_data_inh, av_test_shfl_inh, sd_test_shfl_inh, av_test_chance_inh, sd_test_chance_inh, \
#        av_test_data_allExc, sd_test_data_allExc, av_test_shfl_allExc, sd_test_shfl_allExc, av_test_chance_allExc, sd_test_chance_allExc


#%% Set extent for imshow, so x axis has values corresponding to time_aligned

def setExtent_imshow(time_aligned, y=np.nan):
    
    real_x = time_aligned
    if np.isnan(y).all(): # y has same extent as x
        real_y = real_x[::-1] + 0
    else:
        real_y = y
    dx = (real_x[1]-real_x[0])/2.
    dy = (real_y[1]-real_y[0])/2.
    extent = [real_x[0]-dx, real_x[-1]+dx, real_y[0]-dy, real_y[-1]+dy]  # extent=[time_aligned[0], time_aligned[-1], time_aligned[-1], time_aligned[0]])

    return extent
    

#%% Plot heatmap, eg. showing angle between decoders of two time points, or showing class accuracy of a decoder trained on a time point, and tested on a different time point    

def plotAng(top, time_aligned, nPreMin=0, lab='', cmin=0, cmax=0, cmap='jet', cblab='', xl='', yl=''):
#    
#    totLen = len(time_aligned) 
#    step = 4
#    x = (np.unique(np.concatenate((np.arange(np.argwhere(time_aligned>=0)[0], -.5, -step), 
#               np.arange(np.argwhere(time_aligned>=0)[0], totLen, step))))).astype(int)
#    #x = np.arange(0,totLen,step)
   
    if cmin==0:
        cmin = np.nanmin(top)
        cmax = np.nanmax(top)
        
    extent = setExtent_imshow(time_aligned) #, y)      
#    extent = setExtent_imshow(time_aligned, np.arange(0, top.shape[0])[::-1]) # mark y axis every 5 days.         
    nFrs = len(time_aligned)
           
    plt.imshow(top, cmap, extent=extent) #, interpolation='nearest') #, extent=time_aligned)
    
#    plt.xticks(x, np.round(time_aligned[x]).astype(int))
#    plt.yticks(x, np.round(time_aligned[x]).astype(int))
    #ax.set_xticklabels(np.round(time_aligned[np.arange(0,60,10)]))
#    plt.xlim([0, len(time_aligned)])
#    plt.ylim([0, len(time_aligned)][::-1]) #    plt.ylim([0, len(time_aligned)])
     
     # mark time 0
#    plt.axhline(0, 0, nFrs, color='r')
#    plt.axvline(0, 0, nFrs, color='r')
    
    plt.title(lab)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.colorbar(label=cblab)
    plt.clim(cmin, cmax)
#    plt.gca().set_xlim([extent[0]-10 , extent[1]+10])
    
    makeNicePlots(plt.gca(), 1, 1)



#%% Plot heatmaps of Angles averaged across days for data, shfl, shfl-data (plot angle between decoders at different time points)        

def plotAngsAll(nPreMin, time_aligned, angInh_av, angExc_av, angAllExc_av, angExc_excsh_av, angInhS_av, angExcS_av, angAllExcS_av, angExc_excshS_av, pei, dnow, fnam, cblab = 'Angle between decoders', CA=0):
    
    # Make a large figure including 3 subfigues: data, shfl, data-shfl; each subfigure has 3 subplots for exc, inh, allN
    
    # CA: if 1, plotting class accur, if 0, plotting angles.
    # set angInhS_av to 0 if you want to only plot data
    # set angInh_av to 0 if you want to only plot data-shfl
    
    ################## set vars ##################
    
    if 'doExcSamps' not in locals():
        doExcSamps = 1 # exc was first averaged across trial samps, then across exc samps... for each mouse
        
    totLen = len(time_aligned) 

    if chAl:
        cha = 'chAl_'
    else:
        cha = 'stAl_'
    
    if CA==1:
        cmap = 'jet'
#        cblab = 'Class accuracy (%)'        
        xl = 'Testing t (ms)'
        yl = 'Decoder training t (ms)'
        cblabd = 'Class accuracy (data-shfl)'
        cblabdd = 'Inh-Exc: class accur'        
#        cmax = 100
        cmax = np.ceil(np.max([np.nanmax(angInh_av), np.nanmax(angExc_excsh_av), np.nanmax(angAllExc_av)])*100)/100
    else:
        cmap = 'jet_r' # lower angles: more aligned: red
#        cblab = 'Angle between decoders'
        xl = 'Time since choice onset (ms)'
        yl = xl
        cblabd = 'Alignment rel. shuffle'
        cblabdd = 'Inh-Exc: angle between decoders'        
        cmax = 90
        
    ################## start plotting ##################
        
    if type(angInh_av)==int or type(angInhS_av)==int: # we set angInh_av to 0, when angInhS_av is in fact shfl-data and we only want to plot shfl-data; # similary we set angInhS_av to 0 when we want to only plot data!        
        # plot either just data, or just data-shfl        
        plt.figure(figsize=(8,8))  
        rows = 2; cols = 2; 
        sp11 = 1; sp13 = 3; sp12 = 2; sp14 = 4  # for data - shfl        
    else: # plot data, shfl, data-shfl
        plt.figure(figsize=(8,22))
        rows = 6; cols = 2; 
        sp11 = 1+8; sp13 = 3+8; sp12 = 2+8; sp14 = 4+8   
    sp1 = 1; sp3 = 3; sp2 = 2; sp4 = 4        # for data        
    
    ########################### data figure ###########################    
    if type(angInh_av)!=int: 
        if doExcSamps:
            ex = angExc_excsh_av
        else:
            ex = angExc_av        
        
        cmin = np.floor(np.min([np.nanmin(angInh_av), np.nanmin(ex), np.nanmin(angAllExc_av)])*100)/100
        
        plt.subplot(rows,cols,sp1); lab = 'inh (data)'; 
        top = angInh_av; 
        plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
        plt.xlabel(xl); plt.ylabel(yl)
        
        plt.subplot(rows,cols,sp3); lab = 'exc (data)'; 
        top = ex; 
        plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
        plt.xlabel(xl); plt.ylabel(yl)
        
        plt.subplot(rows,cols,sp2); lab = labAll+' (data)'; 
        top = angAllExc_av; 
        plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
        plt.xlabel(xl); plt.ylabel(yl)
        
        plt.subplots_adjust(hspace=.2, wspace=.3)        
                

    ########################### shfl figure ###########################
    if type(angInhS_av)!=int:         
        if doExcSamps:
            ex = angExc_excshS_av
        else:
            ex = angExcS_av        
    #    plt.figure(figsize=(8,8))
        cmin = np.floor(np.min([np.nanmin(angInhS_av), np.nanmin(ex), np.nanmin(angAllExcS_av)]))
#        cmax = 90
#        cmap = 'jet_r' # lower angles: more aligned: red
#        cblab = 'Angle between decoders'
    
        plt.subplot(625); lab = 'inh (shfl)'; 
        top = angInhS_av; 
        plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
        plt.xlabel(xl); plt.ylabel(yl)
    
        plt.subplot(627); lab = 'exc (shfl)'; 
        top = ex; 
        plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
        plt.xlabel(xl); plt.ylabel(yl)
    
        plt.subplot(626); lab = labAll+' (shfl)'; 
        top = angAllExcS_av; 
        plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblab)
        plt.xlabel(xl); plt.ylabel(yl)
    
        plt.subplots_adjust(hspace=.2, wspace=.3)
        
    
    
    ########################### shfl-data figure (most useful plot) ###########################    
    if type(angInhS_av)!=int:         
        if doExcSamps:
            ex = angExc_excshS_av - angExc_excsh_av
        else:
            ex = angExcS_av - angExc_av     
        if CA==1:
            ex = -ex # data-shfl (instead of shfl-data which is what we do for angles)
    
        inhd = angInhS_av-angInh_av
        if CA==1:
            inhd = -inhd
    
        allNd = angAllExcS_av-angAllExc_av
        if CA==1:
            allNd = -allNd
    
    #    plt.figure(figsize=(8,8))        
        cmin = np.floor(np.min([np.nanmin(inhd), np.nanmin(ex), np.nanmin(allNd)]))
        cmax = np.floor(np.max([np.nanmax(inhd), np.nanmax(ex), np.nanmax(allNd)]))
        cmap = 'jet' # larger value: lower angle for real: more aligned: red
        
        plt.subplot(rows,cols,sp11);  lab = 'inh (shfl-data)';     
        top = inhd #    top[np.triu_indices(len(top),k=0)] = np.nan # plot only the lower triangle (since values are rep)
        plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblabd)
        plt.xlabel(xl); plt.ylabel(yl)
        
        plt.subplot(rows,cols,sp13); lab = 'exc (shfl-data)'; 
        top = ex #    top[np.triu_indices(len(top),k=0)] = np.nan
        plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblabd)
        plt.xlabel(xl); plt.ylabel(yl)
            
        plt.subplot(rows,cols,sp12); lab = labAll+' (shfl-data)'; 
        top = allNd #    top[np.triu_indices(len(top),k=0)] = np.nan
        plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblabd)
        plt.xlabel(xl); plt.ylabel(yl)        
            
        # last subplot: inh(shfl-real) - exc(shfl-real)
        cmin = np.nanmin(inhd - ex) 
        cmax = np.nanmax(inhd - ex)
    #    cmap = 'jet' # more diff: higher inh: lower angle for inh: inh more aligned: red    
        
        plt.subplot(rows,cols,sp14); lab = 'inh-exc'; 
        top = inhd - ex #    top[np.triu_indices(len(top),k=0)] = np.nan
        plotAng(top, time_aligned, nPreMin, lab, cmin, cmax, cmap, cblabdd)
        plt.xlabel(xl); plt.ylabel(yl)
            
        # Is inh and exc stability different? 
        # for each population (inh,exc) subtract shuffle-averaged angles from real angles... do ttest across days for each pair of time points
    #    _,pei = stats.ttest_ind(angleInh_aligned - angleInhS_aligned_avSh , angleExc_aligned - angleExcS_aligned_avSh, axis=-1)     
    #    _,pei = stats.ttest_ind(angInh_av_allMice_al - angInhS_av_allMice_al ,   angExc_excsh_av_allMice_al - angExc_excshS_av_allMice_al, axis=0) 
    #    pei[np.triu_indices(len(pei),k=0)] = np.nan
        # mark sig points by a dot:
        for f1 in range(totLen):
            for f2 in range(totLen): #np.delete(range(totLen),f1): #
                if pei[f1,f2]<.05:
                    plt.plot(f2,f1, marker='*', color='b', markersize=5)
    
    
    
    plt.subplots_adjust(hspace=.2, wspace=.5)
    
    ################## save the figure ##################
    if savefigs:
        d = os.path.join(svmdir+dnow) #,mousename)       
        if not os.path.exists(d):
            print 'creating folder'
            os.makedirs(d)            
            
        fign = os.path.join(d, suffn[0:5]+cha+fnam+'.'+fmt[0])    
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
        



#%% Plot heatmaps of a meaure showing all days : day vs time(during trial)

def plotStabScore(top, lab, cmins, cmaxs, cmap='jet', cblab='', ax=plt, xl = 'Time since choice onset (ms)'):
#    img = ax.imshow(top, cmap)
    extent = setExtent_imshow(time_aligned, np.arange(0, top.shape[0])[::-1]) # mark y axis every 5 days.
    img = ax.imshow(top, cmap, vmin=cmins, vmax=cmaxs, extent=extent)
    
#    plt.plot([nPreMin, nPreMin], [0, len(dayinds)], color='r')
#            ax.set_xlim([-1, len(time_aligned)])
#            ax.set_ylim([-2, len(dayinds)][::-1])
#    ax.autoscale(False)
#            ax.axvline(x=nPreMin, c='w', lw=1)
    ax.axvline(x=0, c='w', lw=1)
#    fig.colorbar(label=cblab);     
#    plt.clim(cmins, cmaxs)
#            ax.set_xticks(x)
#            ax.set_xticklabels(np.round(time_aligned[x]).astype(int))
    ax.set_ylabel('Days')
    ax.set_xlabel(xl)
    ax.set_title(lab)
    makeNicePlots(ax)
    return img
 


#%% Compute the change in CA from the actual case to shflTrsEachNeuron case ... aveaged across those population sizes that are significantly different between actual and shflTrsEachN
# for each day do ttest across samples for each of the population sizes to see if shflTrsEachN is differnt fromt eh eactual case (ie neuron numbers in the decoder)
    
def changeCA_shflTrsEachN(lastPopSize=0, onlySigPops=1):    

#    lastPopSize = 0 # if 1, compute CA changes on the last population size (ie when all neurons are included in the population). If 0, average change in CA across all population sizes
#    onlySigPops = 1 # if 1, only use population sizes that show sig difference between original and shuffled. If 0, include all population sizes, regardless of significancy

    if onlySigPops==0:  # include all regardless of sig
        alph = 1
        thN = -1        
    else:
        if mousename == 'fni18':
            alph = .05
            thN = 0 # at least have 3 population sizes that are sig diff btwn act and shflTrsEachN, in order to compute the change in CA after breaking noise corrs.
        elif mousename == 'fni19':
            alph = .001
            thN = 6#3 # at least have 3 population sizes that are sig diff btwn act and shflTrsEachN, in order to compute the change in CA after breaking noise corrs.            
        else:
            alph = .001
            thN = 3 # at least have 3 population sizes that are sig diff btwn act and shflTrsEachN, in order to compute the change in CA after breaking noise corrs.
    
    thE = 5 # when setting average across exc samps: only use days that have more than 5 valid exc samples
    
   
    dav_allExc = np.full((numD), np.nan)
    dav_inh = np.full((numD), np.nan)
    dav_exc = np.full((numD, numExcSamples), np.nan)
    
    for iday in range(numD):
        if mn_corr[iday] >= thTrained:
               
            ### allExc
            a = perClassErrorTest_data_allExc_all_shflTrsEachNn[iday]#[:,:,eventI_ds_allDays[iday]+fr2an] # nNeurons x nSamps
            b = perClassErrorTest_data_allExc_alln[iday]#[:,:,eventI_ds_allDays[iday]+fr2an]
            if lastPopSize:
                a = a[-1]; b = b[-1]; ax = 0  # use the last population size
            else:
#                a = a[-15:-5]; b = b[-15:-5]; 
                ax = 1            
            # ttest across samples for each population size
            p = sci.stats.ttest_ind(a, b, axis=ax)[1] # nNeurons
            if lastPopSize:
                check = p<=alph
            else:
                check = sum(p<=alph) >= thN
            if check:
                aav = av_test_data_allExc_shflTrsEachN[iday].flatten()
                bav = av_test_data_allExc[iday].flatten()
                if lastPopSize:
                    dav_allExc[iday] = aav[-1] - bav[-1] # use the last population size
                else:
                    d = (aav - bav)[p<=alph]
                    dav_allExc[iday] = np.nanmean(d)
    
    
            ### inh
            a = perClassErrorTest_data_inh_all_shflTrsEachNn[iday]#[:,:,eventI_ds_allDays[iday]+fr2an] # nNeurons x nSamps
            b = perClassErrorTest_data_inh_alln[iday]#[:,:,eventI_ds_allDays[iday]+fr2an]
            if lastPopSize:
                a = a[-1]; b = b[-1]; ax = 0
            else:
#                a = a[-15:-5]; b = b[-15:-5]; 
                ax = 1                        
            # ttest across samples for each population size
            p = sci.stats.ttest_ind(a, b, axis=ax)[1] # nNeurons
            if lastPopSize:
                check = p<=alph
            else:
                check = sum(p<=alph) >= thN            
            if check:#sum(p<=alph)>=thN:
                aav = av_test_data_inh_shflTrsEachN[iday].flatten()
                bav = av_test_data_inh[iday].flatten()
                if lastPopSize:
                    dav_inh[iday] = aav[-1] - bav[-1]
                else:
                    d = (aav - bav)[p<=alph]
                    dav_inh[iday] = np.nanmean(d)
    
    
            ### exc
            a = np.transpose(perClassErrorTest_data_exc_all_shflTrsEachNn[iday], (1,0,2))#[:,:,:,eventI_ds_allDays[iday]+fr2an] # nExcSamps x nNeurons x nSamps
            b = np.transpose(perClassErrorTest_data_exc_alln[iday], (1,0,2))#[:,:,:,eventI_ds_allDays[iday]+fr2an]
            if lastPopSize:
                a = a[:,-1,:]; b = b[:,-1,:]; ax = 1
            else:
#                a = a[:,-15:-5,:]; b = b[:,-15:-5,:];
                ax = 2
            # ttest across samples for each population size
            p = sci.stats.ttest_ind(a, b, axis=ax)[1] # nExcSamps x nNeurons
            for iexc in range(numExcSamples):
                if lastPopSize:
                    check = p[iexc]<=alph
                else:
                    check = sum(p[iexc]<=alph) >= thN
                if check:#sum(p[iexc]<=alph)>=thN:
                    aav = av_test_data_exc_shflTrsEachN_excSamp[iday][iexc] # numShufflesExc x number of neurons in the decoder
                    bav = av_test_data_exc_excSamp[iday][iexc]
                    if lastPopSize:
                        dav_exc[iday, iexc] = aav[-1] - bav[-1]
                    else:
                        d = (aav - bav)[p[iexc]<=alph] # pooled neurons of all exc samps with sig difference between shflTrsEachN and the actual case
                        dav_exc[iday, iexc] = np.nanmean(d)
    
    
    # Average across exc samps: only use days that have more than 5 valid exc samples
    dav_exc[np.sum(~np.isnan(dav_exc), axis=1)<thE] = np.full((dav_exc[np.sum(~np.isnan(dav_exc), axis=1)<5].shape), np.nan)
    dav_exc_av = np.nanmean(dav_exc, axis=1)        
    #    dav_exc_av = np.nanmean(dav_exc[np.sum(~np.isnan(dav_exc), axis=1)>=5], axis=1)
    print np.nanmean(dav_allExc), np.nanmean(dav_inh), np.nanmean(dav_exc_av)    
    print sci.stats.ttest_ind(dav_exc_av, dav_inh, nan_policy='omit')[1]        
    
    return dav_allExc, dav_inh, dav_exc_av, dav_exc
    



#%% Compute number of neurons to reach plateau for each day
    
# It makes sense to call it plateau if the highest ~15th percentile of CA are within 1 s.e.m of each other (hence no statistical change in CA). 
# then, once this condition of being plateau is met, use what you are doing below, ie finding the first point where the curve reaches the mean of CA of the
# highest 15th percentile, to find the number of neurons to reach plateau.
    
def numNeurPlateau():
    
    platN_allExc = np.full((numD), np.nan)
    platN_inh = np.full((numD), np.nan)
    platN_exc = np.full((numD), np.nan)
    platN_exc_excSamp = np.full((numD, numExcSamples), np.nan)
    
    for iday in range(numD):
        if mn_corr[iday] >= thTrained:
            
            #### allExc
            av = av_test_data_allExc[iday].flatten()          
            avs = av_test_shfl_allExc[iday].flatten()
#            plt.plot(av); plt.plot(avs)
            tp = ttest2(av, avs, tail='right')
            if tp <= alph: # only set plateau if CA is sig different from chance               
#                p = np.nanpercentile(av, thpa)
#                th = (av[av >= p]).mean()
                th = av[min(len(av)-50, 50): len(av)].mean() # max(len(av)-0, len(av))  # th = av[len(av)-100: len(av)].mean()
                platN_allExc[iday] = np.argwhere(np.diff(np.argwhere(av < th).flatten()) > nNsContHiCA)[0]
    
            #### inh
            av = av_test_data_inh[iday].flatten()                    
            avs = av_test_shfl_inh[iday].flatten()
#            plt.plot(av); plt.plot(avs)
            tp = ttest2(av, avs, tail='right')
            if tp <= alph: # only set plateau if CA is sig different from chance
#                p = np.percentile(av, thpi)
#                th = (av[av >= p]).mean() 
                th = av[len(av)-5: len(av)].mean() # th = av[min(len(av)-10, 30): max(len(av)-0, len(av))].mean()
                a0 = np.argwhere(av < th).flatten()
                a0 = np.unique(np.concatenate((a0, [len(av)]))) # add the last element
                a = np.argwhere(np.diff(a0) > nNsContHiCA)
                if len(a)==0:
                    platN_inh[iday] = np.nan #a0[-1]
                else:
                    platN_inh[iday] = a[0]
            
            #### exc
            for iexc in range(numExcSamples):
                av = av_test_data_exc_excSamp[iday][iexc].flatten()        
                avs = av_test_shfl_exc_excSamp[iday][iexc].flatten()
#                plt.plot(av); plt.plot(avs)
                tp = ttest2(av, avs, tail='right')
                if tp <= alph: # only set plateau if CA is sig different from chance                    
#                    p = np.percentile(av, thpe)
#                    th = (av[av >= p]).mean()                    
                    th = av[len(av)-5: len(av)].mean()
                    a0 = np.argwhere(av < th).flatten()
                    a0 = np.unique(np.concatenate((a0, [len(av)]))) # add the last element
                    a = np.argwhere(np.diff(a0) > nNsContHiCA)
                    if len(a)==0:
                        platN_exc_excSamp[iday, iexc] = np.nan #a0[-1]
                    else:
                        platN_exc_excSamp[iday, iexc] = a[0]
            # average across exc samps (if >5 valid exc samps)
            if sum(~np.isnan(platN_exc_excSamp[iday])) >= 5: # how many valid samples
                platN_exc[iday] = np.nanmean(platN_exc_excSamp[iday])
                
                
    return platN_allExc, platN_inh, platN_exc, platN_exc_excSamp



                