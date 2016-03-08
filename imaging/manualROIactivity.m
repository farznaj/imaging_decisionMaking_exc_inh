function [activity, rois] = manualROIactivity(mousename, imagingFolder, mdfFileNumber, signalCh)
% activity = manualROIactivity(mousename, imagingFolder, mdfFileNumber, signalCh)
%
% Computes activity (ie mean pixel intensity of each ROI for each frame)
% for the ROIs specified by the ROI.zip file (manually identified and saved
% in fiji) and the movie specified by mdfFileNumber.
% It will also append activity to the date_major (e.g. "151102_001") mat file.
%
% Example inputs:
% mousename = 'fni17';
% imagingFolder = '151102';
% mdfFileNumber = 1; 
% signalCh = 2;

%% Set some initial parameters
%
% mousename = P.mousename; 
% imagingFolder = P.imagingFolder;
% mdfFileNumber = P.mdfFileNumber; 
% signalCh = P.signalCh;

if isunix
    dataPath = '/sonas-hs/churchland/nlsas/data/data';
elseif ispc
    dataPath = '\\sonas-hs.cshl.edu\churchland\data';
end

tifFold = fullfile(dataPath, mousename, 'imaging', imagingFolder);
pathToROIZip = fullfile(tifFold, sprintf('RoiSet_%s_%03d_ch%d_MCM.zip', imagingFolder, mdfFileNumber, signalCh));
%

%{
load(fullfile('improcparams', paramsFileName))
tifFold = params.tifFold; 
date_major = sprintf('%06d_%03d', params.tifNums(1, 1:2));
pathToROIZip = fullfile(tifFold, sprintf('RoiSet_%s_ch%d_MCM.zip', date_major, params.gcampCh));
files = dir(fullfile(tifFold, sprintf('%s_*_ch%d_MCM.TIF', date_major, signalCh)));
%}

% set the tif files corresponding to mdf file mdfFileNumber
files = dir(fullfile(tifFold, sprintf('%s_%03d_*_ch%d_MCM.TIF', imagingFolder, mdfFileNumber, signalCh)));
% tifList = {files.name}
tifList = cell(1, length(files));
for itif = 1:length(files)
    tifList{itif} = fullfile(tifFold, files(itif).name);
end
% tifList;


%% Compute mean pixel intensity of each ROI for each frame 
[activity, rois] = applyFijiROIsToTifs(pathToROIZip, tifList);
% activity = activity'; % you are doing this for compatibility with Eft results (num comps x num frames)
% activity(pmtOffFrames{signalCh},:) = NaN; % we are not doing this here, instead we will do it in 


%% Append activity to the date_major mat file.
save(fullfile(tifFold, sprintf('%s_%03d', imagingFolder, mdfFileNumber)), '-append', 'activity', 'rois')


% activity(pmtOffFrames{signalCh},:) = NaN; % we are not doing this here, instead we will do it in 


