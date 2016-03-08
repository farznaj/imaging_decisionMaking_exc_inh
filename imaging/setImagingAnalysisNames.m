function [imfilename, pnevFileName, tifFold, date_major] = setImagingAnalysisNames(mousename, imagingFolder, mdfFileNumber, signalCh)
% [imfilename, pnevFileName, tifFold, date_major] = setImagingAnalysisNames(mousename, imagingFolder, mdfFileNumber, signalCh)
% E.g.
% mousename = 'fni17';
% imagingFolder = '151102'; % '151021';
% mdfFileNumber = 1; % or tif major
% signalCh = 2;

%%
if ismac
    dataPath = '/Users/Farzaneh/Desktop/Farzaneh/data'; % macbook
elseif isunix
    dataPath = '/sonas-hs/churchland/nlsas/data/data'; % server
elseif ispc
    dataPath = '\\sonas-hs.cshl.edu\churchland\data'; % lab PC
end


tifFold = fullfile(dataPath, mousename, 'imaging', imagingFolder);
date_major = sprintf('%s_%03d', imagingFolder(1:6), mdfFileNumber);
imfilename = fullfile(tifFold, date_major);
pnevFileName = [date_major, '_ch', num2str(signalCh),'-Pnev*'];
pnevFileName = dir(fullfile(tifFold, pnevFileName));
pnevFileName = pnevFileName.name;
pnevFileName = fullfile(tifFold, pnevFileName);


