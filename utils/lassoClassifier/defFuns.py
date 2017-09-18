## All commanly used functions

#%% Extend the built in two tailed ttest function to one-tailed
def ttest2(a, b, **tailOption):
    import scipy.stats as stats
    import numpy as np
    h, p = stats.ttest_ind(a, b)
    d = np.mean(a)-np.mean(b)
    if tailOption.get('tail'):
        tail = tailOption.get('tail').lower()
        if tail == 'right':
            p = p/2.*(d>0) + (1-p/2.)*(d<0)
        elif tail == 'left':
            p = (1-p/2.)*(d>0) + p/2.*(d<0)
    if d==0:
        p = 1;
    return p
	
	

#%%
# optional inputs:
#postNProvided = 1; # Default:0; If your directory does not contain pnevFile and instead it contains postFile, set this to 1 to get pnevFileName
#signalCh = 2 # # since gcamp is channel 2, should be 2.
#pnev2load = [] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.

def setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, **options):

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


    #%%
    import numpy as np
    import platform
    import glob
    import os.path
        
    if len(pnev2load)==0:
        pnev2load = [0];
            
    #%%
    if platform.system()=='Linux':
        if os.getcwd().find('grid')!=-1: # server # sonas
            dataPath = '/sonas-hs/churchland/nlsas/data/data/'
            altDataPath = '/sonas-hs/churchland/hpc/home/space_managed_data/'            
        else: # office linux
            dataPath = '/home/farznaj/Shares/Churchland/data/'
            altDataPath = '/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/' # the new space-managed server (wos, to which data
    elif platform.system()=='Darwin':
        dataPath = '/Volumes/My Stu_win/ChurchlandLab/'
    else:
        dataPath = '/Users/gamalamin/git_local_repository/Farzaneh/data/'
        
    #%%        
    tifFold = os.path.join(dataPath+mousename,'imaging',imagingFolder)

    if not os.path.exists(tifFold):
        if 'altDataPath' in locals():
            tifFold = os.path.join(altDataPath+mousename, 'imaging', imagingFolder)
        else:
            sys.exit('Data directory does not exist!')

    
#    print mdfFileNumber, type(mdfFileNumber)
#    mdfFileNumber = np.array(mdfFileNumber).astype('int')
#    print mdfFileNumber, type(mdfFileNumber), np.shape(mdfFileNumber)
#    print np.shape(mdfFileNumber)[0]
    r = '%03d-'*np.shape(mdfFileNumber)[0] #len(mdfFileNumber)
    r = r[:-1]
    rr = r % (tuple(mdfFileNumber))
    
    date_major = imagingFolder+'_'+rr
    imfilename = os.path.join(tifFold,date_major+'.mat')
    
    #%%
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
    
    #%%
    if nOuts==2:
        return imfilename, pnevFileName
    else:
        return imfilename, pnevFileName, dataPath
    
    
#%%
#imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh, pnev2load)


    
    
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
    
    # gap between tick labeles and axis
#    ax.tick_params(axis='x', pad=30)

#    plt.xticks(x, labels, rotation='vertical')
    

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
#    plt.bar(bin_edges[0:-1], hist, binEvery, color=colors[0], alpha=.4, label=lab1)
    plt.plot(bin_edges[0:-1], hist, color=colors[0], label=lab1)    
    
    # set his of b
    hist, bin_edges = np.histogram(b, bins=bn)
    hist = hist/float(np.sum(hist));     #d = stats.mode(np.diff(bin_edges))[0]/float(2)
    # Plot hist of b
#    plt.bar(bin_edges[0:-1], hist, binEvery, color=colors[1], alpha=.4, label=lab2)
    plt.plot(bin_edges[0:-1], hist, color=colors[1], label=lab2)    
    
    
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


    
#%%
def errbarAllDays(a,b,p):
    if a.ndim==1:
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
      
  
  
  
#%% function to find best c from testing class error ran on values of cvect

def findBestC(perClassErrorTest, cvect, regType, smallestC=1):
    
    numSamples = perClassErrorTest.shape[0] 
    numFrs = perClassErrorTest.shape[2]
    
    #%% Compute average of class errors across numSamples
    
    cbestFrs = np.full((numFrs), np.nan)
       
    for ifr in range(numFrs):
        
        meanPerClassErrorTest = np.mean(perClassErrorTest[:,:,ifr], axis = 0);
        semPerClassErrorTest = np.std(perClassErrorTest[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
       
        #%% Identify best c       
        
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

    
    #%% Compute average of class errors across numSamples
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

       
       
    #%%        
    for ifr in range(numFrs):
        
        #%% Set the decoder and class errors at best c         
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
    
        
        #%% Plot C path           
        
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

#%%##### Function for identifying the best regularization parameter
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







#%% Same as above but when X is frames x units x trials
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



#%% Define a colormap

from matplotlib import cm
from numpy import linspace
start = 0.0
stop = 1.0
number_of_lines= len(days)
cm_subsection = linspace(start, stop, number_of_lines) 
colors = [ cm.jet(x) for x in cm_subsection ]


        
