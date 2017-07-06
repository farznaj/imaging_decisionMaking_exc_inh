# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:59:12 2016

@author: farznaj

This is Farzaneh's first Python code :-) She is very happy and pleased about it :D

example call:

mousename = 'fni17'
imagingFolder = '151021'
mdfFileNumber = [1] #(1,2)

# optional inputs:
postNProvided = 1; # Default:0; If your directory does not contain pnevFile and instead it contains postFile, set this to 1 to get pnevFileName
signalCh = [2] # since gcamp is channel 2, should be 2.
pnev2load = [] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.

from setImagingAnalysisNamesP import *

imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load, postNProvided=postNProvided)

imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber)

"""

#%%
def setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, **options):

    if options.get('signalCh'):
        signalCh = options.get('signalCh');    
    else:
        signalCh = []
        
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
            altDataPath = '/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/' # the new space-managed server (wos, to which data is migrated from grid)
    elif platform.system()=='Darwin':
        dataPath = '/Volumes/My Stu_win/ChurchlandLab'
    else:
        dataPath = '/Users/gamalamin/git_local_repository/Farzaneh/data/'
        
    #%%        
    tifFold = os.path.join(dataPath+mousename,'imaging',imagingFolder)
#    print os.path.exists(tifFold)
#    print 'altDataPath' in locals()
#    print altDataPath
	
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
    if len(signalCh)>0:
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

