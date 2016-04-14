function regImage = makeCaImagingRegImage(tifName, frames, trimBorders)
% regImage = makeCaImagingRegImage(tifName, frames [,trimBorders])
%
% Make a median image to use as a reference (registration image) for motion
% correction. tifName should be the name of the tiff file to use, and
% frames should contain a vector of frame numbers (within that file).
%
% If using the MScan option to correct for sinusoidal motion of the fast
% mirror, there will be black bars on the left and right of the image. The
% optional trimBorders argument lets you throw those away (assumes 55 pixel
% border width, which is correct for a 512 pixel wide image). Must agree
% with option used in motionCorrectCaImagingFile(). Default true.



%% Parameters

% This is how wide the black borders on the left and right sides of the
% image are, when using the MScan option to correct for the sinusoidal
% movement of the mirrors and a horizontal resolution of 512. These borders
% will get chopped off.
borderWidth = 55;


%% Optional arguments

if ~exist('trimBorders', 'var')
  trimBorders = 1;
end


%% Read metadata

fprintf('Reading tiff for registration image\n');

tifInfo = imfinfo(tifName);
imWidth = tifInfo(1).Width;
imHeight = tifInfo(1).Height;


%% Read the chosen frames out of the tiff

% Pre-allocate movie
movie = zeros(imHeight, imWidth, length(frames), 'uint16');

% Read frames
for f = 1:length(frames)
  movie(:, :, f) = imread(tifName, 'Index', frames(f), 'Info', tifInfo);
end


%% Take the median to get regImage

regImage = median(movie, 3);


%% Trim borders

if trimBorders
  validPixels = [false(1, borderWidth) true(1, imWidth - 2*borderWidth) false(1, borderWidth)];
  regImage = regImage(:, validPixels);
end

