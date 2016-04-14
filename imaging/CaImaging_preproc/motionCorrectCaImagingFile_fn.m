function [regMovie, outputsDFT] = motionCorrectCaImagingFile_fn(tifName, regImage, tifInfo, channelForMotionCorrection, framesToUse, trimBorders)
% [regMovie, pixelShifts] = ...
%       motionCorrectCaImagingFile(tifName, regImage [, framesToUse] [, trimBorders])
%
% Use the efficient subpixel registration algorithm to motion correct a
% calcium imaging movie.  Uses 10-fold upsampling.
%
% INPUTS
%   tifName      -- name of the file to operate on
%   regImage     -- image to register to (get from makeCaImagingRegImage() )
%   framesToUse  -- optional. If provided and not NaN, will extract only
%                   these frames
%   trimBorders  -- optional (default true). If using the MScan option to
%                   correct for sinusoidal motion of the fast mirror, there
%                   will be black bars on the left and right of the image.
%                   This option lets you throw those away (assumes 55 pixel
%                   border width, which is correct for a 512 pixel wide
%                   image).
%
% OUTPUTS
%   regMovie     -- The motion corrected movie
%   pixelShifts  -- an nFrames x 2 vector containing how far each image was
%                   moved during correction. Each row is [x y].
%
% To trim off edges after motion correction, use maskMovie()


%% Parameters

% This is how wide the black borders on the left and right sides of the
% image are, when using the MScan option to correct for the sinusoidal
% movement of the mirrors and a horizontal resolution of 512. These borders
% will get chopped off.
borderWidth = 55;

% Upsampling factor for subpixel motion correction. 10 seems likes more
% than enough.
usFac = 10;



%% Optional arguments

if ~exist('framesToUse', 'var')
  framesToUse = NaN;
end

if ~exist('trimBorders', 'var')
  trimBorders = 1;
end


%% Read tiff metadata

% tifInfo = imfinfo(tifName);
channelsSaved = [];
if ~isempty(strfind(tifInfo(1).ImageDescription, 'Channel 1: Saved'))
    channelsSaved = [channelsSaved, 1];
end

if ~isempty(strfind(tifInfo(1).ImageDescription, 'Channel 2: Saved'))
    channelsSaved = [channelsSaved, 2];
end

if ~isempty(strfind(tifInfo(1).ImageDescription, 'Channel 3: Saved'))
    channelsSaved = [channelsSaved, 3];
end

if ~isempty(strfind(tifInfo(1).ImageDescription, 'Channel 4: Saved'))
    channelsSaved = [channelsSaved, 4];
end
    
nFrames = length(tifInfo);
imWidth = tifInfo(1).Width;
imHeight = tifInfo(1).Height;

% Prepare to trim borders
if trimBorders
  validPixels = [false(1, borderWidth) true(1, imWidth - 2*borderWidth) false(1, borderWidth)];
else
  validPixels = true(1, imWidth);
end


% if isnan(framesToUse)
%     framesToUse = 1:nFrames;
% end

%%
regMovie = cell(1,length(channelsSaved)); % channelsSaved
outputsDFT = cell(1,length(channelsSaved));
% pixelShifts = cell(1,length(channelForMotionCorrection));

tic;
for ch = channelForMotionCorrection % 1:length(channelForMotionCorrection) % 
    
    framesToUse = ch : length(channelsSaved) : nFrames; % channelForMotionCorrection(ch)
%     framesToUse = channelsSaved(ch) : length(channelsSaved) : nFrames;

    %% Read all the images out of the tiff and trim borders.
    
    fprintf('Reading tiff, channel %d\n', ch);
    
    % Pre-allocate movie
    movie = zeros(imHeight, sum(validPixels), length(framesToUse), 'uint16');

    % Read frames, throwing away borders
    for f = 1:length(framesToUse)
      if mod(f, 100) == 0
        fprintf('%d ', f);
      end
      if mod(f, 1000) == 0
        fprintf('\n');
      end  
      rawFrame = imread(tifName, 'Index', framesToUse(f), 'Info', tifInfo);
      movie(:, :, f) = rawFrame(:, validPixels);
    end
    fprintf('\n');
    
    
    %%
    clear rawFrame
    

    %% Motion correction / registration

    fprintf('Correcting motion, channel %d\n', ch);

    % Get FFT of reference registration image
    fftRef = fft2(regImage{ch});

    % Pre-allocate result
    regMovie{ch} = uint16(zeros(size(movie)));
    outputsDFT{ch} = zeros(size(movie, 3), 4);

    % Pre-allocate for saving the shift magnitudes
    dftOutputs = zeros(size(movie, 3), 4);

    % Do the registration
%     tic;
    for f = 1:size(movie, 3)
      % Display progress
      if mod(f, 100) == 0
        fprintf('%d ', f);
      end
      if mod(f, 1000) == 0
        fprintf('\n');
      end

      [dftOutputs(f, :), Greg] = dftregistration(fftRef, fft2(movie(:, :, f)), usFac);
      regMovie{ch}(:, :, f) = uint16(abs(ifft2(Greg)));
    end
    fprintf('\n');
%     fprintf('\n\nRegistering %d frames took %0.1f s\n\n', size(movie, 3), toc);

    %%% pixelsShifts output
%     pixelShifts{ch} = dftOutputs(:, 3:4);
    outputsDFT{ch} = dftOutputs;

    
    %%
    clear fftRef dftOutputs Greg movie
    
end

fprintf('Registering %d frames from %d channel(s) took %0.1f s\n\n', length(framesToUse), length(channelForMotionCorrection), toc);

    
