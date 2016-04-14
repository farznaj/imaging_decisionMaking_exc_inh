function [movieMC, badFrames] = preprocessCaMovies(tifList, regTif, regFrameNums, outSuffix, maxMaskWidth)
% [movieMC, badFrames] = preprocessCaMovies(tifList, regTif, regFrameNums [, outSuffix] [, maxMaskWidth])
% 
% Takes the names of tif files and information about how to motion correct,
% then performs motion correction, masks off the edges of the movie (which
% are contaminated by motion correction), and returns the movie and which
% frames required too much motion correction.
%
% The motion correction algorithm used is the upsampled discrete Fourier
% transform algorithm in dftregistration.m
%
% INPUTS
%   tifList      -- cell array of strings containing the full path and
%                   filename of each .tif to include
%   regTif       -- string containing the .tif to use for the registration
%                   image (for motion correction)
%   regFrameNums -- vector of frame numbers to use from regTif in making
%                   the registration image
%   outSuffix    -- optional. If present and non-empty, the resulting movie
%                   will be saved to a series of .tif files, corresponding
%                   to the original .tif files, but with this suffix added
%                   to the filenames. Recommended: '_MCM' (for "motion
%                   corrected and masked"). Default: empty (no saving)
%   maxMaskWidth -- optional. The mask will not be wider than this number
%                   of pixels.
%
% OUTPUTS
%   movieMC      -- the motion corrected and masked movie, including all
%                   .tif files given
%   badFrames    -- logical vector, indicating whether each frame was
%                   motion corrected by more than maxMaskWidth


%% Optional arguments

if ~exist('outSuffix', 'var')
  outSuffix = '';
end

if ~exist('maxMaskWidth', 'var')
  maxMaskWidth = 20;
end


%% Get the registration image

regImage = makeCaImagingRegImage(regTif, regFrameNums);


%% Figure out how big the resulting movie will be

nFramesPerMovie = NaN(1, length(tifList));

for t = 1:length(tifList)
  tifInfo = imfinfo(tifList{t});
  nFramesPerMovie(t) = length(tifInfo);
end

imWidth = size(regImage, 2);
imHeight = size(regImage, 1);

totalFrames = sum(nFramesPerMovie);


%% Motion correct each movie, gather them together

movieMC = uint16(zeros(imHeight, imWidth, totalFrames));
pixelShifts = NaN(totalFrames, 2);

frame = 0;
for t = 1:length(tifList)
  fprintf('Motion correcting file: %s\n', tifList{t});
  
  % Perform the motion correction
  frames = frame + 1 : frame + nFramesPerMovie(t);
  [movieMC(:, :, frames), pixelShifts(frames, :)] = motionCorrectCaImagingFile(tifList{t}, regImage);
  
  frame = frame + nFramesPerMovie(t);
end


%% Mask result, to get rid of edges that are affected by motion correction

if maxMaskWidth > 0
  [maskBounds, badFrames] = determineMovieMaskBounds(pixelShifts, [imWidth imHeight], maxMaskWidth);
  fprintf('Masking off pixels: %d on left, %d on right, %d on top, %d on bottom\n', ...
    maskBounds(1)-1, imWidth-maskBounds(2), maskBounds(3)-1, imHeight-maskBounds(4));
  movieMC = maskMovie(movieMC, maskBounds);
else
  badFrames = true(size(movieMC, 3), 1);
end


%% Write movie to file, if requested

if ~isempty(outSuffix)
  % The apparent brightness changes, but I think this is just a scaling issue
  % from a header parameter I can't change
  
  frame = 0;
  for t = 1:length(tifList)
    % Figure out filename
    [fPath, fStem, fExt] = fileparts(tifList{t});
    outFile = fullfile(fPath, [fStem outSuffix fExt]);
    
    fprintf('Writing file %s (%d/%d)\n', outFile, t, length(tifList));

    % Figure out frames
    frames = frame + 1 : frame + nFramesPerMovie(t);
    
    imwrite(movieMC(:, :, frames(1)), outFile, 'TIF', ...
      'Resolution', [size(movieMC, 2) size(movieMC, 1)], 'Compression', 'none');
  
    if length(frames) > 1
      for f = 2:length(frames)
        if mod(f, 100) == 0
          fprintf('%d ', f);
        end
        if mod(f, 1000) == 0
          fprintf('\n');
        end
        
        imwrite(movieMC(:, :, frames(f)), outFile, 'TIF', ...
          'Resolution', [size(movieMC, 2) size(movieMC, 1)], 'Compression', 'none', ...
          'WriteMode', 'append');
      end
    end
    
    frame = frame + nFramesPerMovie(t);
  end
end

fprintf('\nDone.\n');
