function [regImage, frames] = makeCaImagingRegImage_fn(tifName, framesCh, trimBorders, channelForMotionCorrection, file2read, noMotionTrs, playMovie)
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

% channelNum: what channels will be used for image correction. If only 1 channel was saved, that will be used for motion correction. (FN).

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

if ~exist('playMovie', 'var')
    playMovie = 0;
end


%% Read metadata

% fprintf('Reading tiff for registration image\n\n');

tifInfo = imfinfo(tifName);
imWidth = tifInfo(1).Width;
imHeight = tifInfo(1).Height;


%% Channels to use for motion correction (FN).
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

if length(channelsSaved)==1
    channelForMotionCorrection = channelsSaved;
end


%% if framesCh is not specified either use noMotionTrs or play the movie to find no motion frames.
playTag = 0;
if isempty(framesCh) && ~isempty(noMotionTrs)
    framesCh = regFrameNums_set(file2read, noMotionTrs);
    framesCh(framesCh > length(tifInfo)/length(channelsSaved)) = []; % assuming that regTif is the 1st tif file.
    if isempty(framesCh)
        framesCh = 1:50;
        warning('Arbitrary frames 1:50 are used for motion correction.')
    end
%     length(regFrameNums)

elseif isempty(framesCh) && isempty(noMotionTrs)
    playTag = 1;
    framesCh = 1:min(1000, length(tifInfo)/length(channelsSaved));
end


%%
regImage = cell(1, length(channelForMotionCorrection));

for ch = 1:length(channelForMotionCorrection)
  
    frames = framesCh*length(channelsSaved) - (length(channelsSaved) - channelForMotionCorrection(ch));
  
    %% Read the chosen frames out of the tiff

    % Pre-allocate movie
    movie = zeros(imHeight, imWidth, length(frames), 'uint16');

    fprintf('Reading reference tif file %s, %d frames....\n', tifName, length(frames))
    % Read frames
    for f = 1:length(frames)
      movie(:, :, f) = imread(tifName, 'Index', frames(f), 'Info', tifInfo);
    end

    
    %%
    if playTag
        implay(movie)
        frames = input('What are the no-motion frames? ');
        movie = movie(:,:,frames);
    end
    

    %% Take the median to get regImage

    regImage{ch} = median(movie, 3);


    %% Trim borders

    if trimBorders
      validPixels = [false(1, borderWidth) true(1, imWidth - 2*borderWidth) false(1, borderWidth)];
      regImage{ch} = regImage{ch}(:, validPixels);
    end

end

if playMovie
    implay(movie)
end


