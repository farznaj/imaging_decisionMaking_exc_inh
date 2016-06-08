function [activity, rois] = applyCustomROIsToTifs(imfilename, tifList)
% activity = applyFijiROIsToTifs(pathToROIZip, tifList)
%
% Apply custom ROIs saved in imfilename to the tif files in the cell
% array tifList. Returns an array activity, which is the mean fluorescence
% over time (rows) in each ROI (columns).
%
% This function gets called in manualROIactivityFromCustomROIs. Use the
% codes below in the section below (also copied in
% manualROIactivityFromCustomROIs) to manually generate custom ROIs.


%%
%%%%%%%%%%%%%%%%%% Make custom ROIs %%%%%%%%%%%%%%%%%%
%{
%% Plot Efty's ROIs

pnev2load = [];
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
load(pnevFileName, 'A')
load(imfilename, 'sdImage', 'imHeight', 'imWidth')
im = sdImage{2};

contour_threshold = .95;
% im = sdImage{2};
[CC, ~, ~, mask] = setCC_cleanCC_plotCC_setMask(A, imHeight, imWidth, contour_threshold, im);


%% Manually set custom ROIs (eg if you want to choose dark ROIs). 

clear rois_custom
disp('Each time select 3 points that identify x and y of a rectangular ROI. By default 20 ROIs will be made. Abort if done earlier.')

for i = 1:20
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


%% Load ROIs

load(imfilename, 'rois_custom')
rois = rois_custom;
clear rois_custom

% Read ROIs
% rois = ReadImageJROI(pathToROIZip);


%% Preallocate

roiMeans = cell(1, length(tifList));


%% Loop through files

for t = 1:length(tifList)
  
  fprintf('Applying ROIs to file %d/%d\n', t, length(tifList));
  
  %% Prep for reading file
  
  tifInfo = imfinfo(tifList{t});
  nFrames = length(tifInfo);
  
  %% If this is the first file, set up the ROI masks
  
  if t == 1
    [X,Y] = meshgrid(1:tifInfo(1).Width, 1:tifInfo(1).Height);
    
    mask = false(tifInfo(1).Height, tifInfo(1).Width, length(rois));
    for rr = 1:length(rois)
      mask(:, :, rr) = inpolygon(X, Y, rois{rr}.mnCoordinates(:, 1), rois{rr}.mnCoordinates(:, 2));
    end
  end
  
  
  %% Apply the masks to each frame
  
  roiMeans{t} = zeros(nFrames, length(rois));
  
  for fr = 1:nFrames
    if mod(fr, 100) == 0
      fprintf('%d ', fr);
    end
    if mod(fr, 1000) == 0
      fprintf('\n');
    end

    im = imread(tifList{t}, 'Index', fr, 'Info', tifInfo);
    
    for rr = 1:length(rois)
      roiMeans{t}(fr, rr) = mean(im(mask(:, :, rr)));
    end
  end
end


%% Concatenate results from all files

activity = vertcat(roiMeans{:});

fprintf('\nDone.\n');


%%
%{
clear rois
i = 1; 
[X,Y] = meshgrid(66:90, 1:45);
rois{i}.mnCoordinates = [Y(:), X(:)];

i = 2; 
[X,Y] = meshgrid(373:398, 1:30);
rois{i}.mnCoordinates = [Y(:), X(:)];

i = 3; 
[X,Y] = meshgrid(195:210, 270:295);
rois{i}.mnCoordinates = [Y(:), X(:)];

i = 4; 
[X,Y] = meshgrid(244:260, 409:440);
rois{i}.mnCoordinates = [Y(:), X(:)];
%}
