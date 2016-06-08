function activity_man_eftMask = manualROIactivityFromEftMask(mouse, imagingFolder, mdfFileNumber, ch2ana)
% Manually computes activity (ie mean pixel intensity of each ROI in every
% frame) for the ROIs identified by Eftychios's algorithm (using A, spatial
% components). (The goal is to compare the manual activity with Eftychios's
% temporal components for the same set of ROIs.
%
% This function calls manualROIactivityFromEftMaskMovieMC; If you have
% already movieMC (ie you don't need to read tifs), directly use
% manualROIactivityFromEftMaskMovieMC. 
%
%
% Example inputs:
% mousename = 'fni17';
% imagingFolder = '151102';
% mdfFileNumber = 1;
% ch2ana = 2;

% if ~exist('roiCh', 'var')
%     roiCh = ch2ana;
% end


%% Set some initial parameters
%
% mousename = P.mousename;
% imagingFolder = P.imagingFolder;
% mdfFileNumber = P.mdfFileNumber;
% ch2ana = P.signalCh;

if isunix
    dataPath = '/sonas-hs/churchland/nlsas/data/data';
elseif ispc
    dataPath = '\\sonas-hs.cshl.edu\churchland\data';
end
tifFold = fullfile(dataPath, mouse, 'imaging', imagingFolder);
% pathToROIZip = fullfile(tifFold, sprintf('RoiSet_%s_%03d_ch%d_MCM.zip', imagingFolder, mdfFileNumber, roiCh));
%

%{
load(fullfile('improcparams', paramsFileName))
tifFold = params.tifFold;
date_major = sprintf('%06d_%03d', params.tifNums(1, 1:2));
pathToROIZip = fullfile(tifFold, sprintf('RoiSet_%s_ch%d_MCM.zip', date_major, params.gcampCh));
files = dir(fullfile(tifFold, sprintf('%s_*_ch%d_MCM.TIF', date_major, ch2ana)));
%}

% Set the tif files corresponding to mdf file mdfFileNumber and channel ch2ana
files = dir(fullfile(tifFold, sprintf('%s_%03d_*_ch%d_MCM.TIF', imagingFolder, mdfFileNumber, ch2ana)));
% tifList = {files.name}
tifList = cell(1, length(files));
for itif = 1:length(files)
    tifList{itif} = fullfile(tifFold, files(itif).name);
end
% showcell(tifList')


%% Read tif files into movieMC

movieMC = [];
for t = 1:length(tifList)
    fprintf('Reading tif file %s\n', tifList{t})
    movieMC = cat(3, movieMC, bigread2(tifList{t}));
end
        
        
        
%% Load Eftychios's spatial component (A)

signalCh = 2; % because you get A from channel 2, I think this should be always 2.
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh);
[~,f] = fileparts(pnevFileName);
disp(f)

load(imfilename, 'imHeight', 'imWidth')

load(pnevFileName, 'A') % load('demo_results_fni17-151102_001.mat', 'A2')
% load(fullfile(tifFold, 'test_aftInitSpUp_multiTrs'), 'A')
   

%% Compute activity

activity_man_eftMask = manualROIactivityFromEftMaskMovieMC(movieMC, A, imHeight, imWidth);

nam2save = ['activity_man_eftMask_ch', num2str(ch2ana)];
eval([nam2save ' = activity_man_eftMask;'])


%% Create mask for Efty's results by applying contours on A.
%{
% set contours for Eft ROIs
contour_threshold = .95;
[CCorig, ~, ~] = ROIContoursPnev(A, imHeight, imWidth, contour_threshold); % P.d1, P.d2
CC = ROIContoursPnev_cleanCC(CCorig);
% set masks for Eft ROIs
mask = maskSet(CC, imHeight, imWidth); % mask_eft


%% Compute mean pixel intensity of each ROI for each frame

activity = applyMaskROIsToTifs(mask, tifList);
activity_man_eftMask = activity;
% [activity_custom2, rois_custom2] = applyFijiROIsToTifs_customROIs(pathToROIZip, tifList);
% activity = activity'; % you are doing this for compatibility with Eft results (num comps x num frames)
% activity(pmtOffFrames{ch2ana},:) = NaN; % we are not doing this here, instead we will do it in
%}

%% Append activity to the date_major mat file.

save(pnevFileName, '-append', nam2save)

% save(fullfile(tifFold, sprintf('%s_%03d', imagingFolder, mdfFileNumber)), '-append', 'activity_custom2', 'rois_custom2')
% save(fullfile(tifFold, sprintf('%s_%03d', imagingFolder, mdfFileNumber)), '-append', 'activity_man_eftMask')
% save(fullfile(tifFold, 'test_aftInitSpUp_multiTrs'), '-append', 'activity_man_eftMask')

% activity(pmtOffFrames{ch2ana},:) = NaN; % we are not doing this here, instead we will do it in


