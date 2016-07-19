function activity_custom = manualROIactivityFromCustomROIs(mouse, imagingFolder, mdfFileNumber, signalCh)
% activity = manualROIactivity(mousename, imagingFolder, mdfFileNumber, signalCh)
%
% This function calls applyCustomROIsToTifs to compute activity (mean
% fluorescent intensity) for custom ROIs. To generate custom ROIs, use the
% codes in this function.
% It will also append activity to the date_major (e.g. "151102_001") mat file.
%
% Example inputs:
% mousename = 'fni17';
% imagingFolder = '151102';
% mdfFileNumber = 1;
% signalCh = 2;

% if ~exist('roiCh', 'var')
%     roiCh = signalCh;
% end


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
tifFold = fullfile(dataPath, mouse, 'imaging', imagingFolder);
% pathToROIZip = fullfile(tifFold, sprintf('RoiSet_%s_%03d_ch%d_MCM.zip', imagingFolder, mdfFileNumber, roiCh));
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


%%
%%%%%%%%%%%%%%%%%% Make custom ROIs %%%%%%%%%%%%%%%%%%
%{
%% Plot Efty's ROIs

pnev2load = [];
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
load(pnevFileName, 'A')
load(imfilename, 'sdImage', 'imHeight', 'imWidth')
% im = sdImage{2};
im = mean(cat(3, sdImage{1}, sdImage{2}), 3); % look at sdImage of both channels

contour_threshold = .95;
% im = sdImage{2};
[CC, ~, ~, mask] = setCC_cleanCC_plotCC_setMask(spatialComp, imHeight, imWidth, contour_threshold, im);


%% Manually set custom ROIs (eg if you want to choose dark ROIs). 

clear rois_custom
disp('Each time select 3 points that identify x and y of a rectangular ROI. By default 20 ROIs will be made. Abort if done earlier.')

for i = 10:20
    fprintf('ROI %i\n', i)
    a = round(ginput(3));
    x = min(a(:,1)) : max(a(:,1));
    y = min(a(:,2)) : max(a(:,2));
    
    [X,Y] = meshgrid(x, y);
    rois_custom{i}.mnCoordinates = [X(:), Y(:)];
end


%% Plot the custom ROIs 

[CC2, mask2] = setCC_mask_manual(rois_custom, im);


%% Save custom ROIs
save(imfilename, '-append', 'rois_custom')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}


%% Compute mean pixel intensity of each ROI for each frame

imfilename = fullfile(tifFold, sprintf('%s_%03d', imagingFolder, mdfFileNumber));

activity_custom = applyCustomROIsToTifs(imfilename, tifList);


%% Append activity to the date_major mat file.

save(imfilename, '-append', 'activity_custom')




