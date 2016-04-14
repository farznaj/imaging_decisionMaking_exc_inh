function activity = applyFijiROIsToTifs(pathToROIZip, tifList)
% activity = applyFijiROIsToTifs(pathToROIZip, tifList)
%
% Apply the ROIs generated in Fiji/ImageJ to the tif files in the cell
% array tifList. Returns an array activity, which is the mean fluorescence
% over time (rows) in each ROI (columns).


%% Read ROIs

rois = ReadImageJROI(pathToROIZip);


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
