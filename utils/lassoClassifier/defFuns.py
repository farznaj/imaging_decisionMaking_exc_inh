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
            dataPath = '/home/farznaj/Shares/Churchland/data/'
            altDataPath = '/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/' # the new space-managed server (wos, to which data
    elif platform.system()=='Darwin':
        dataPath = '/Volumes/My Stu_win/ChurchlandLab/'
    else:
        dataPath = '/Users/gamalamin/git_local_repository/Farzaneh/data/'
        
    ##%%        
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
    if nOuts==2:
        return imfilename, pnevFileName
    else:
        return imfilename, pnevFileName, dataPath
 
        
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
   
   
#%% Plot histogram of verctors a and b on axes h1 and h2

def histerrbar(h1,h2,a,b,binEvery,p,lab,colors = ['g','k'],ylab='Fraction',lab1='exc',lab2='inh',plotCumsum=0):
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
    bn[-1] = np.max([np.max(a),np.max(b)]) # unlike digitize, histogram doesn't count the right most value
    
    # plt hist of a
    hist, bin_edges = np.histogram(a, bins=bn)
    hist = hist/float(np.sum(hist))    
    if plotCumsum:
        hist = np.cumsum(hist)
    ax1 = plt.subplot(h1) #(gs[0,0:2])
    # plot the center of bins
    plt.plot(bin_edges[0:-1]+binEvery/2., hist, color=colors[0], label=lab1)    #    plt.bar(bin_edges[0:-1], hist, binEvery, color=colors[0], alpha=.4, label=lab1)
    
    # plot his of b
    hist, bin_edges = np.histogram(b, bins=bn)
    hist = hist/float(np.sum(hist));     #d = stats.mode(np.diff(bin_edges))[0]/float(2)
    if plotCumsum:
        hist = np.cumsum(hist)
    plt.plot(bin_edges[0:-1]+binEvery/2., hist, color=colors[1], label=lab2)        #    plt.bar(bin_edges[0:-1], hist, binEvery, color=colors[1], alpha=.4, label=lab2)

    # set labels, etc
    yl = plt.gca().get_ylim()
    ry = np.diff(yl)
    plt.ylim([yl[0]-ry/20 , yl[1]])   
    #
    xl = plt.gca().get_xlim()
    rx = np.diff(xl)
    plt.xlim([xl[0]-rx/20 , xl[1]])   
    #    
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
    
    plt.subplots_adjust(wspace=1, hspace=.5)
    return ax1,ax2


    
#%% each element of a and b contras vars for a days. Here we compute and plot ave and se across those vars.
    
def errbarAllDays(a,b,p,gs,colors = ['g','k'],lab1='exc',lab2='inh',lab='ave(CA)', h1=[], h2=[]):
    
    if type(h1)==list: # h1 is not provided; otherwise h1 is provided as a matplotlib.gridspec.SubplotSpec
        h1 = gs[1,0:2]
    if type(h1)==list:
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
    plt.errorbar(0, np.nanmean(eav), np.nanstd(eav)/np.sqrt(len(eav)), marker='o', color='k')
    plt.errorbar(1, np.nanmean(iav), np.nanstd(iav)/np.sqrt(len(eav)), marker='o', color='k')
    plt.xticks([0,1], (lab1, lab2), rotation='vertical')
    plt.xlim([-1,2])
    makeNicePlots(ax2,0,1)

    _, p = stats.ttest_ind(eav, iav, nan_policy='omit')
    plt.title('p=%.3f' %(p))

    plt.subplots_adjust(wspace=1, hspace=.5)
    
    return ax1,ax2
    

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
  
def setBestC_classErr(perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance, cvect, regType, doPlots=0, doIncorr=0, wAllC=[], bAllC=[]):
    
    numSamples = perClassErrorTest.shape[0] 
    numFrs = perClassErrorTest.shape[2]
    
    ###%% Compute average of class errors across numSamples
    
    classErr_bestC_train_data = np.full((numSamples, numFrs), np.nan)
    classErr_bestC_test_data = np.full((numSamples, numFrs), np.nan)
    classErr_bestC_test_shfl = np.full((numSamples, numFrs), np.nan)
    classErr_bestC_test_chance = np.full((numSamples, numFrs), np.nan)
    cbestFrs = np.full((numFrs), np.nan)
#    numNon0SampData = np.full((numFrs), np.nan)       
    classErr_bestC_test_incorr = np.full((numSamples, numFrs), np.nan)
    classErr_bestC_test_incorr_shfl = np.full((numSamples, numFrs), np.nan)
    classErr_bestC_test_incorr_chance = np.full((numSamples, numFrs), np.nan)
    if len(wAllC)>0:
        w_bestc_data = np.full((numSamples, wAllC.shape[2], numFrs), np.nan)
    else:
        w_bestc_data = []
    
    for ifr in range(numFrs):
                
        meanPerClassErrorTrain = np.mean(perClassErrorTrain[:,:,ifr], axis = 0);
        semPerClassErrorTrain = np.std(perClassErrorTrain[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest = np.mean(perClassErrorTest[:,:,ifr], axis = 0);
        semPerClassErrorTest = np.std(perClassErrorTest[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl[:,:,ifr], axis = 0);
#        semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance[:,:,ifr], axis = 0);
#        semPerClassErrorTest_chance = np.std(perClassErrorTest_chance[:,:,ifr], axis = 0)/np.sqrt(numSamples);
        
        
        
        ################### Identify best c        
        # Use all range of c... it may end up a value at which all weights are 0.
        ix = np.argmin(meanPerClassErrorTest)
        if smallestC==1:
            cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
            cbest = cbest[0]; # best regularization term based on minError+SE criteria
            cbestAll = cbest
        else:
            cbestAll = cvect[ix]
        print 'best c = %.10f' %cbestAll
        
        
        
        #################### Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)
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
        
        
        
        #################### Set the decoder and class errors at best c         
        # you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
        # we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
        indBestC = np.in1d(cvect, cbest)

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
            b_bestc_data = bAllC[:,indBestC,ifr]            

       # check the number of non-0 weights
    #       for ifr in range(numFrs):
    #        w_data = wAllC[:,indBestC,:,ifr].squeeze()
    #        print np.sum(w_data > eps,axis=1)
    #        ada = np.sum(w_data > eps,axis=1) < thNon0Ws # samples w fewer than 2 non-0 weights        
    #        ada = ~ada # samples w >=2 non0 weights
    #        numNon0SampData[ifr] = ada.sum() # number of cv samples with >=2 non-0 weights
    
        
        
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
    from matplotlib import cm
    from numpy import linspace
    start = 0.0
    stop = 1.0
    number_of_lines = nlines #len(days)
    cm_subsection = linspace(start, stop, number_of_lines) 
    colors = [ cm.jet(x) for x in cm_subsection ]
    
    #% Change color order to jet 
    from cycler import cycler
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)


        

#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName

def setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc=[], regressBins=3, useEqualTrNums=1, corrTrained=0, shflTrsEachNeuron=0):
    import glob

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
    elif doInhAllexcEqexc[2] == 1:
        ntype = 'eqExc'           
        
    if trialHistAnalysis:
        if useEqualTrNums:
            svmn = 'excInh_SVMtrained_eachFrame_%s%s%s_prevChoice_%s_ds%d_eqTrs_*' %(o2a, shflname, ntype, al,regressBins)
        else:
            svmn = 'excInh_SVMtrained_eachFrame_%s%s%s_prevChoice_%s_ds%d_*' %(o2a, shflname, ntype, al,regressBins)
    else:
        if useEqualTrNums:
            svmn = 'excInh_SVMtrained_eachFrame_%s%s%s_currChoice_%s_ds%d_eqTrs_*' %(o2a, shflname, ntype, al,regressBins)
        else:
            svmn = 'excInh_SVMtrained_eachFrame_%s%s%s_currChoice_%s_ds%d_*' %(o2a, shflname, ntype, al,regressBins)
        
        
        
    svmn = svmn + os.path.basename(pnevFileName) #pnevFileName[-32:]    
    svmName = glob.glob(os.path.join(os.path.dirname(pnevFileName), 'svm', svmn))
    svmName = sorted(svmName, key=os.path.getmtime)[::-1] # so the latest file is the 1st one.

    return svmName
    


#%% Function to get the latest svm .mat file corresponding to pnevFileName, trialHistAnalysis, ntName, roundi, itiName

def setSVMname_allN_eachFrame(pnevFileName, trialHistAnalysis, chAl, regressBins=3, corrTrained=0, shflTrsEachNeuron=0):
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
    
    if trialHistAnalysis:
        svmn = 'svmPrevChoice_eachFrame_%s%s%s_ds%d_*' %(al,o2a,shflname,regressBins)
    else:
        svmn = 'svmCurrChoice_eachFrame_%s%s%s_ds%d_*' %(al,o2a,shflname,regressBins)
    
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
    
def set_corr_hr_lr(postName, svmName, doIncorr=0):
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

def set_nprepost(trace, eventI_ds_allDays, mn_corr, thTrained=10, regressBins=3):
    # trace: each element is for one day
            
    numDays = len(trace)
    nPost = (np.ones((numDays,1))+np.nan).flatten()
    for iday in range(numDays):
        if mn_corr[iday] >= thTrained: # dont include days with too few svm trained trials.
            nPost[iday] = (len(trace[iday]) - eventI_ds_allDays[iday] - 1)
    nPostMin = np.nanmin(nPost).astype('int')
    
    nPreMin = np.nanmin(eventI_ds_allDays).astype('int') # number of frames before the common eventI, also the index of common eventI.     
    print 'Number of frames before = %d, and after = %d the common eventI' %(nPreMin, nPostMin)
    
    ## Set the time array for the across-day aligned traces
    totLen = nPreMin + nPostMin +1

    # Get downsampled time trace, without using the non-downsampled eventI
    # you need nPreMin and totLen
    a = frameLength*np.arange(-regressBins*nPreMin,0)
    b = frameLength*np.arange(0,regressBins*(totLen-nPreMin))
    aa = np.mean(np.reshape(a,(regressBins,nPreMin), order='F'), axis=0)
    bb = np.mean(np.reshape(b,(regressBins,totLen-nPreMin), order='F'), axis=0)
    time_al = np.concatenate((aa,bb))

    return time_al, nPreMin, nPostMin
        

#%% Align traces of each day on the common eventI

def alTrace(trace, eventI_ds_allDays, nPreMin, nPostMin, mn_corr, thTrained=10):    
    # mn_corr: min number of HR and LR trials 
    
    trace= np.array(trace)
    trace_aligned = []
#    trace_aligned = np.ones((nPreMin + nPostMin + 1, trace.shape[0])) + np.nan # frames x days, aligned on common eventI (equals nPreMin)     
    for iday in range(trace.shape[0]):
        if mn_corr[iday] >= thTrained: # dont include days with too few svm trained trials.
            trace_aligned.append(trace[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1])
#            trace_aligned[:, iday] = trace[iday][eventI_ds_allDays[iday] - nPreMin  :  eventI_ds_allDays[iday] + nPostMin + 1]    
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
    
        
        eventI_ds = np.argwhere(np.sign(time_trace_d)>0)[0]  # frame in downsampled trace within which event_I happened (eg time1stSideTry)    
        
    
    else:
        print 'Not downsampling traces ....'

    
    return X_svm, time_trace, eventI_ds
    
    
        
#%%    
def rmvNans(a): # remove nans
    mask = ~np.isnan(a)
    a = np.array([d[m] for d, m in zip(a, mask)])
    return a

 
#%% compute p value between vectors a,b (wilcoxon, MW, ttest)  
 
def pwm(a,b):
    _,p0  = sci.stats.ranksums(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)]); 
    _,p1  = sci.stats.mannwhitneyu(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)])
    _,p2  = sci.stats.ttest_ind(np.array(a)[~np.isnan(a)], np.array(b)[~np.isnan(b)])
    p = [p0,p1,p2]
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
    
def loadSVM_allN(svmName, doPlots, doIncorr, loadWeights):
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
         
        
    #######%% Set class error values at best C (also find bestc for each frame and plot c path)
    
    if loadWeights!=2:        
        if doIncorr==1:
            classErr_bestC_train_data, classErr_bestC_test_data, classErr_bestC_test_shfl, classErr_bestC_test_chance, cbestFrs, w_bestc_data, b_bestc_data, classErr_bestC_test_incorr, classErr_bestC_test_incorr_shfl, classErr_bestC_test_incorr_chance = setBestC_classErr(perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance, cvect, regType, doPlots, doIncorr, wAllC, bAllC)
        else:
            classErr_bestC_train_data, classErr_bestC_test_data, classErr_bestC_test_shfl, classErr_bestC_test_chance, cbestFrs, w_bestc_data, b_bestc_data = setBestC_classErr(perClassErrorTrain, perClassErrorTest, perClassErrorTest_shfl, perClassErrorTest_chance, cvect, regType, 0, doIncorr, wAllC, bAllC)
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
        
def loadSVM_excInh(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, doPlots, doIncorr, loadWeights, doAllN, useEqualTrNums, shflTrsEachNeuron):
    # loadWeights: 
    # 0: don't load weights, only load class accur
    # 1 : load weights and class cccur
    # 2 : load only weights and not class accur
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
    for idi in range(3):
        
        doInhAllexcEqexc = np.full((3), False)
        doInhAllexcEqexc[idi] = True 
        if idi==1 and doAllN: # plot allN, instead of allExc
            svmName = setSVMname_allN_eachFrame(pnevFileName, trialHistAnalysis, chAl, regressBins, corrTrained, shflTrsEachNeuron) # for chAl: the latest file is with soft norm; earlier file is 
        else:        
            svmName = setSVMname_excInh_trainDecoder(pnevFileName, trialHistAnalysis, chAl, doInhAllexcEqexc, regressBins, useEqualTrNums, corrTrained, shflTrsEachNeuron)        
        svmName = svmName[0] # use [0] for the latest file; use [-1] for the earliest file
        print os.path.basename(svmName)    

        ######### inh
        if doInhAllexcEqexc[0] == 1: 
            if loadWeights==1:
                Datai = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_inh', 'perClassErrorTest_shfl_inh', 'perClassErrorTest_chance_inh', 'w_data_inh', 'b_data_inh'])
            elif loadWeights==2:                
                Datai = scio.loadmat(svmName, variable_names=['w_data_inh','b_data_inh'])
            else:
                Datai = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_inh', 'perClassErrorTest_shfl_inh', 'perClassErrorTest_chance_inh'])
        
        ######### allN or allExc
        elif doInhAllexcEqexc[1] == 1: 
            if doAllN: # plot allN, instead of allExc
                _, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, _, w_data_allExc, b_data_allExc = loadSVM_allN(svmName, doPlots, doIncorr, 1)   
            else: # plot allExc
                if loadWeights==1:
                    Dataae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_allExc', 'perClassErrorTest_shfl_allExc', 'perClassErrorTest_chance_allExc', 'w_data_allExc', 'b_data_allExc'])
                elif loadWeights==2:                
                    Dataae = scio.loadmat(svmName, variable_names=['w_data_allExc', 'b_data_allExc'])
                else:
                    Dataae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_allExc', 'perClassErrorTest_shfl_allExc', 'perClassErrorTest_chance_allExc'])
        
        ######### eqExc
        elif doInhAllexcEqexc[2] == 1: 
            if loadWeights==1:
                Datae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_exc', 'perClassErrorTest_shfl_exc', 'perClassErrorTest_chance_exc', 'w_data_exc', 'b_data_exc'])                                                 
            elif loadWeights==2:                                
                Datae = scio.loadmat(svmName, variable_names=['w_data_exc', 'b_data_exc'])                                                 
            else:
                Datae = scio.loadmat(svmName, variable_names=['perClassErrorTest_data_exc', 'perClassErrorTest_shfl_exc', 'perClassErrorTest_chance_exc'])

        
    ###%%             
    if loadWeights!=2:   # 2: only download weights, no CA                                     
        perClassErrorTest_data_inh = Datai.pop('perClassErrorTest_data_inh')
        perClassErrorTest_shfl_inh = Datai.pop('perClassErrorTest_shfl_inh')
        perClassErrorTest_chance_inh = Datai.pop('perClassErrorTest_chance_inh') 
    if loadWeights>=1:
        w_data_inh = Datai.pop('w_data_inh') 
        b_data_inh = Datai.pop('b_data_inh')         
    else:
        w_data_inh = []
        b_data_inh = []
        
    if doAllN==0:
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
    
    return perClassErrorTest_data_inh, perClassErrorTest_shfl_inh, perClassErrorTest_chance_inh, perClassErrorTest_data_allExc, perClassErrorTest_shfl_allExc, perClassErrorTest_chance_allExc, perClassErrorTest_data_exc, perClassErrorTest_shfl_exc, perClassErrorTest_chance_exc, w_data_inh, w_data_allExc, w_data_exc, b_data_inh, b_data_allExc, b_data_exc
    
    

    
