function activity_man_eftMask = manualROIactivityFromEftMaskMovieMC(movieMC, A, imHeight, imWidth)
%
% Manually computes activity (ie mean pixel intensity of each ROI in every
% frame) for the ROIs identified by Eftychios's algorithm (using the spatial
% components (A)). (The goal is to compare the manual activity with Eftychios's
% temporal components for the same set of ROIs.)
%
% If you don't have movieMC, and you need to read tifs, use the following
% function instead: 
% activity = manualROIactivityFromEftMask(mouse, imagingFolder, mdfFileNumber, signalCh)
%

%% Set the mask

% set contours for Eft ROIs
contour_threshold = .95;
CCorig = ROIContoursPnev(A, imHeight, imWidth, contour_threshold); % P.d1, P.d2
CC = ROIContoursPnev_cleanCC(CCorig);
% set masks for Eft ROIs
mask = maskSet(CC, imHeight, imWidth); % mask_eft


%% Compute mean pixel for each mask

nFrames = size(movieMC,3);
activity = zeros(nFrames, size(mask,3));

for rr = 1:size(activity,2)
    thisUnit = uint16(mask(:,:,rr));
    unitMovie = movieMC .* repmat(thisUnit, [1,1,size(movieMC,3)]);
    activity(:, rr) = squeeze(sum(sum(unitMovie,1),2)) / sum(thisUnit(:));
end

activity_man_eftMask = activity;

