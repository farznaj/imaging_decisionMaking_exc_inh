function [activity, mask] = activityFromSpatialComp(mousename, imagingFolder, mdfFileNumber, ch2read, spatialComp, contour_threshold)
% compute mean fluorescence signal of a Tif movie for each ROI.
% mask is driven from spatialComp.


% mousename = 'fni17';
% imagingFolder = '151102';
% mdfFileNumber = 1; % or tif major
% ch2read = 2;
% contour_threshold = .95;


%% Read tiff files. % FN
tifList = tifListSet(mousename, imagingFolder, mdfFileNumber, ch2read);
movieMC = [];
for t = 1:length(tifList)
    fprintf('Reading tif file %s\n', tifList{t})
    movieMC = cat(3, movieMC, bigread2(tifList{t}));
end


%% Set the masks for the spatial components
[~, mask] = setCC_cleanCC_plotCC_setMask(spatialComp, size(movieMC, 1), size(movieMC, 2), contour_threshold);


%% Apply the masks to the movie
activity = zeros(size(movieMC, 3), size(mask, 3));

for rr = 1:size(mask, 3)
    thisUnit = uint16(mask(:,:,rr));
    unitMovie = movieMC .* repmat(thisUnit, [1,1,size(movieMC,3)]);
    activity(:, rr) = squeeze(sum(sum(unitMovie,1),2)) / sum(thisUnit(:));
end


