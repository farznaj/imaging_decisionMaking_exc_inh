function [movieMC, badFrames, maskBounds, outputsDFT, imWidth, imHeight] = preprocessCaMovies_fn(regImage, tifList, outSuffix, maxMaskWidth, channelForMotionCorrection, channels2write)
%(tifList, regTif, regFrameNums, outSuffix, maxMaskWidth, trimBorders, channelForMotionCorrection)

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

% outputsDFT =  [error,diffphase,net_row_shift,net_col_shift]
% error     Translation invariant normalized RMS error between f and g
% diffphase     Global phase difference between the two images (should be
%               zero if images are non-negative).
% net_row_shift net_col_shift   Pixel shifts between images


%% Optional arguments

if ~exist('outSuffix', 'var')
  outSuffix = '';
end

if ~exist('maxMaskWidth', 'var')
  maxMaskWidth = 20;
end


%% Get the registration image % Farz moved it out of this function and put it in imaging_prep, bc you run tif files one at a time, and you don't want to get the regImage everytime.
% regImage = makeCaImagingRegImage_fn(regTif, regFrameNums, trimBorders, channelForMotionCorrection);


%% Figure out number of recorded channels and how big the resulting movie will be

nFramesPerMovie = NaN(1, length(tifList));
tifInfo = cell(1, length(tifList));

for t = 1:length(tifList)
    fprintf('\nReading info of tif file: %s\n\n', tifList{t});
    tifInfo{t} = imfinfo(tifList{t});
  
    channelsSaved = [];
    if ~isempty(strfind(tifInfo{t}(1).ImageDescription, 'Channel 1: Saved'))
        channelsSaved = [channelsSaved, 1];
    end

    if ~isempty(strfind(tifInfo{t}(1).ImageDescription, 'Channel 2: Saved'))
        channelsSaved = [channelsSaved, 2];
    end

    if ~isempty(strfind(tifInfo{t}(1).ImageDescription, 'Channel 3: Saved'))
        channelsSaved = [channelsSaved, 3];
    end

    if ~isempty(strfind(tifInfo{t}(1).ImageDescription, 'Channel 4: Saved'))
        channelsSaved = [channelsSaved, 4];
    end

    nFramesPerMovie(t) = length(tifInfo{t}) / length(channelsSaved);
end

imWidth = size(regImage{1}, 2);
imHeight = size(regImage{1}, 1);

totalFrames = sum(nFramesPerMovie);


%% Motion correct each movie, gather them together

movieMC = cell(1,length(channelsSaved));
pixelShifts = cell(1,length(channelsSaved));
outputsDFT = cell(1,length(channelsSaved));
maskBounds = cell(1,length(channelsSaved));
badFrames = cell(1,length(channelsSaved));
for ch = 1:length(channelsSaved)
    movieMC{ch} = uint16(zeros(imHeight, imWidth, totalFrames));
    pixelShifts{ch} = NaN(totalFrames, 2);
    outputsDFT{ch} = NaN(totalFrames, 4);
end
otherChannels = channelsSaved(~ismember(channelsSaved, channelForMotionCorrection));

frame = 0;
for t = 1:length(tifList)
%     fprintf('Motion correcting file: %s\n\n', tifList{t});
    
    % Perform the motion correction
    frames = frame + 1 : frame + nFramesPerMovie(t);
    [regMovie, outputs] = motionCorrectCaImagingFile_fn(tifList{t}, regImage, tifInfo{t}, channelForMotionCorrection);
    %     [movieMC{ch}(:, :, frames), pixelShifts(frames, :)]   
    
    % dftregistration is done only for channelForMotionCorrection.
    % the same dftoutputs are used for registering otherChannels.
    if ~isempty(otherChannels)
        regMovie_other = motionCorrectCaImagingFile_2ch_fn(outputs, channelForMotionCorrection, tifList{t}, tifInfo{t}, otherChannels);
    end

    for ch = channelsSaved % channelForMotionCorrection; % 1:length(channelForMotionCorrection)
        if ~isempty(regMovie{ch})
            movieMC{ch}(:, :, frames) = regMovie{ch};
            pixelShifts{ch}(frames, :) = outputs{ch}(:, 3:4);
            outputsDFT{ch}(frames, :) = outputs{ch};
        end
        
        if ~isempty(regMovie_other{ch})
            movieMC{ch}(:, :, frames) = regMovie_other{ch};
        end
    end
    
    clear regMovie outputs regMovie_other
    
    frame = frame + nFramesPerMovie(t);
end


if ~isempty(otherChannels)
    for ch = otherChannels
        pixelShifts{ch} = pixelShifts{channelForMotionCorrection};
        outputsDFT{ch} = outputsDFT{channelForMotionCorrection};
    end        
end


%% Mask result, to get rid of edges that are affected by motion correction

for ch = channelsSaved % 1:length(channelForMotionCorrection)
    if maxMaskWidth > 0
      [maskBounds{ch}, badFrames{ch}] = determineMovieMaskBounds_fn(pixelShifts{ch}, [imWidth imHeight], maxMaskWidth); % maskBounds shows bad pixels on left, right, top and bottom, respectively; so it corresponds to columns and rows, respectively.

      fprintf('Masking off pixels (channel %d): %d on left, %d on right, %d on top, %d on bottom\n', ...
        ch, maskBounds{ch}(1)-1, imWidth-maskBounds{ch}(2), maskBounds{ch}(3)-1, imHeight-maskBounds{ch}(4));

    
        movieMC{ch} = maskMovie_fn(movieMC{ch}, maskBounds{ch});

    else
        
      badFrames{ch} = true(size(movieMC{ch}, 3), 1);
    end
end


%% Write movie to file, if requested
    
if ~isempty(outSuffix)
    
    if ~exist('channels2write', 'var')
        channels2write = channelsSaved;
    end
    
    for ch = channels2write % unlike the raw tifs, the motion corrected tifs will have channel 1 and channel 2 in separate files, instead of alternating frames in the same file. (FN)
      % The apparent brightness changes, but I think this is just a scaling issue
      % from a header parameter I can't change

      frame = 0;
      for t = 1:length(tifList)
          
        % Figure out filename
        [fPath, fStem, fExt] = fileparts(tifList{t});
%         outFile = fullfile(fPath, [fStem outSuffix fExt]);
        outFile = fullfile(fPath, [[fStem,'_ch',num2str(ch)] outSuffix fExt]);

        fprintf('Writing file %s (%d/%d)\n', outFile, t, length(tifList));

        % Figure out frames
        frames = frame + 1 : frame + nFramesPerMovie(t);

        imwrite(movieMC{ch}(:, :, frames(1)), outFile, 'TIF', ...
          'Resolution', [size(movieMC{ch}, 2) size(movieMC{ch}, 1)], 'Compression', 'none');

        if length(frames) > 1
          for f = 2:length(frames)
            if mod(f, 100) == 0
              fprintf('%d ', f);
            end
            if mod(f, 1000) == 0
              fprintf('\n');
            end

            imwrite(movieMC{ch}(:, :, frames(f)), outFile, 'TIF', ...
              'Resolution', [size(movieMC{ch}, 2) size(movieMC{ch}, 1)], 'Compression', 'none', ...
              'WriteMode', 'append');
          end
        end

        frame = frame + nFramesPerMovie(t);
      end
      
    end
end

fprintf('\n---------- Done ----------.\n\n\n');
    
