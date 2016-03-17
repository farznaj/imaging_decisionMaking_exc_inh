function [CC, CR, COMs] = ROIContoursPnev(A, imHeight, imWidth, thr)
% [contourOutlines, contourPixels, COMs] = ROIContoursPnev(spatialBasis, imHeight, imWidth [, thresh])
%
% Return information about the locations of ROIs found using Eftychios's
% code. Contours are found at the (fractional) level thresh. For purposes
% of this function (but not for actual activity inference), these ROIs are
% then median filtered with a 3 x 3 box. This will get tend to rid of
% isolated or tiny features, and fill in holes.
%
% INPUTS:
%   spatialBasis    -- from applyPnevPaninskiCaSourceSep() (probably run
%                      using processCaImagingMCPnev() )
%   imHeight        -- height of each frame (dimension 1)
%   imWidth         -- width of each frame (dimension 2)
%   thresh          -- optional. Threshold for contour level. Default 0.99
%
% OUTPUTS:
%   contourOutlines -- cell array of size nROIs x 1. Each cell contains the
%                      contours for that ROI at the level given by thresh,
%                      nearly as returned by contourc(). Important
%                      difference: the rows have been swapped, to be
%                      consistent with image conventions (so that the first
%                      index is height and the second is width, ie [y x] pairs).
%   contourPixels   -- cell array of size nROIs x 2. The first column of cells
%                      contains an array of size nPixels x 2, where each
%                      column is a [y x] pair of where a pixel in the ROI
%                      is located. The second column contains an array of size
%                      nPixels x 1, which contains the weight for that
%                      pixel.
%   COMs            -- array of size nROIs x 2. The center of mass for each
%                      ROI, as [y x] pairs.
%
% Based on Eftychios's plot_contours: 
%       [Coor, json_file] = plot_contours(A, sdImage{2},
%       contour_threshold); Coor is similar (but not identical) to CC with
%       rows swapped. json_file has fileds coordinates, values and centriod
%       which are identical to CR(:,1), CR(:,2) and COMs. json_file has [y
%       x] pairs. Coor has [x y] pairs.
%
%
% Contours may be plotted with plotPnevROIs()
%
% Example code:
%
% contourThresh = 0.95;
% [CC, CR, COMs] = ROIContoursPnev(spatialBasis, imHeight, imWidth, contourThresh);
% 
% im = sdImage;
% 
% % Plot all the ROIs
% plotPnevROIs(im, CC);
% 
% % Plot 10 ROIs per image
% nROIsPerPlot = 10;
% for set = 1:ceil(length(CC) / nROIsPerPlot)
%   plotPnevROIs(im, CC(min((set-1)*nROIsPerPlot + (1:nROIsPerPlot), length(CC))));
% end
% 
% % Plot the COMs over the image
% figure; imagesc(im);
% colormap('bone'); hold on;
% plot(COMs(:, 2), COMs(:, 1), 'ro');
% 
% % Plot using the pixels output
% figure; imagesc(im);
% colormap('bone'); hold on;
% i = 1;
% plot(CR{i, 1}(2, :), CR{i, 1}(1, :), '.');
% 
% % Plot a heatmap of pixel weights
% hMap = zeros(size(im));
% for u = 1:size(CR, 1)
%   inds = sub2ind(size(im), CR{u, 1}(1, :), CR{u, 1}(2, :));
%   hMap(inds) = hMap(inds) + CR{u, 2};
% end
% figure;
% pcolor(hMap);
% shading flat;
% set(gca, 'YDir', 'reverse');


if ~exist('thr', 'var')
  thr = 0.99;
end

% [imHeight, imWidth] = size(C);

CC = cell(size(A, 2), 1);
CR = cell(size(A, 2), 2);
for roiI = 1:size(A, 2)
  % Median filter with a 3 x 3 box
  % Is this a good idea?
  A_temp = full(reshape(A(:, roiI), imHeight, imWidth));
  A_temp = medfilt2(A_temp, [3 3]);
  A_temp = A_temp(:);
  
  % Sort then threshold
  [temp, ind] = sort(A_temp(:) .^ 2);
  temp = cumsum(temp);
  ff = find(temp > (1 - thr) * temp(end), 1);
  
  if ~isempty(ff)
    % Contour
    % Swap the rows so that everything is [y x], since this is in order of
    % indices and consistent with the other outputs
    CC{roiI} = flipud(contourc(reshape(A_temp, imHeight, imWidth), [0 0] + A_temp(ind(ff))));
    
    % Pixels
    fp = find(A_temp >= A_temp(ind(ff)));
    [ii, jj] = ind2sub([imHeight, imWidth], fp);
    CR{roiI, 1} = [ii, jj]';
    CR{roiI, 2} = A_temp(fp)';
  end
end

% Center of mass
COMs = com(A, imHeight, imWidth);
