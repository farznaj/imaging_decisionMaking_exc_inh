function activity = applyFijiROIsToTifs_fn(pathToROIZip, day, filenumber, numberOfParts, imWidth, imHeight)
% activity = applyFijiROIsToTifs(pathToROIZip, tifList)
%
% Apply the ROIs generated in Fiji/ImageJ to the tif files in the cell
% array tifList. Returns an array activity, which is the mean fluorescence
% over time (rows) in each ROI (columns).

% day is the date, eg '150617'.
% filenumber is the session to which the list of tif files belong, eg '001'. 
% numberOfParts is the number of tif files for a given session, eg 6.

%%
borderWidth = 55;

%% Read ROIs

rois = ReadImageJROI(pathToROIZip);


%% Preallocate

roiMeans = cell(1, numberOfParts);


%% Loop through files

for t = 1:numberOfParts
  
  fprintf('Applying ROIs to file %d/%d\n', t, numberOfParts);
        
  imfilename_part = strcat(day, '_', filenumber, '_', num2str(t));
  im_matfile = strcat(imfilename_part, '.mat');
  
  
  load(im_matfile, 'movieMC', 'maskBounds')
  nFrames = size(movieMC,3);
  
  % making sure all movieMC files have the same size.
  
  begRows = NaN(maskBounds(3)-1 , size(movieMC,2) , nFrames);
  endRows = NaN(imHeight-maskBounds(4) , size(movieMC,2) , nFrames);
  
  movieMC = vertcat(begRows , movieMC , endRows);
  
  clear begRows endRows
  
  begCols = NaN(size(movieMC,1) , maskBounds(1)-1 , nFrames);
  endCols = NaN(size(movieMC,1) , imWidth-maskBounds(2) , nFrames);
  
  movieMC = [begCols , movieMC , endCols];
  
  clear begCols endCols
  
  %%%%%
  
  %% If this is the first file, set up the ROI masks
    
  if t == 1  
    [X,Y] = meshgrid(1:size(movieMC,2), 1:size(movieMC,1));
    
    mask = false(size(movieMC,1), size(movieMC,2), length(rois));
    for rr = 1:length(rois)
%       mask(:, :, rr) = inpolygon(X, Y, rois{rr}.mnCoordinates(:, 1), rois{rr}.mnCoordinates(:, 2));
      mask(:, :, rr) = inpolygon(X, Y, rois{rr}.mnCoordinates(:, 1)-borderWidth, rois{rr}.mnCoordinates(:, 2)); % assuming that ROIs were identified on the original movie (512x512).
    end    
  end
  
  %% Apply the masks to the movie
  
  roiMeans{t} = zeros(nFrames, length(rois));
  
  for rr = 1:length(rois)
      thisUnit = uint16(mask(:,:,rr));
      unitMovie = movieMC .* repmat(thisUnit, [1,1,size(movieMC,3)]);
      roiMeans{t}(:, rr) = squeeze(sum(sum(unitMovie,1),2)) / sum(thisUnit(:));
  end

  clear movieMC thisUnit unitMovie
  
end


%% Concatenate results from all files

activity = vertcat(roiMeans{:});

fprintf('\nDone.\n');
