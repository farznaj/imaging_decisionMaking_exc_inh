function [regMovie] = motionCorrectCaImagingFile_2ch_fn(outputsDFT, channelForMotionCorrection, tifName, tifInfo, otherChannels, trimBorders)


%% Parameters

% This is how wide the black borders on the left and right sides of the
% image are, when using the MScan option to correct for the sinusoidal
% movement of the mirrors and a horizontal resolution of 512. These borders
% will get chopped off.
borderWidth = 55;

% Upsampling factor for subpixel motion correction. 10 seems likes more
% than enough.
% usFac = 10;



%% Optional arguments
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
regMovie = cell(1,length(channelsSaved));

tic;
for ch = otherChannels; % 1:length(otherChannels)
    
    framesToUse = ch : length(channelsSaved) : nFrames; % otherChannels(ch)
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
    
    for f = 1:size(movie, 3)
        % Display progress
        if mod(f, 100) == 0
            fprintf('%d ', f);
        end
        if mod(f, 1000) == 0
            fprintf('\n');
        end

        %%
        buf2ft = fft2(movie(:, :, f));
        %     output=[error,diffphase,row_shift,col_shift];
        diffphase = outputsDFT{channelForMotionCorrection}(f,2);
        row_shift = outputsDFT{channelForMotionCorrection}(f,3);
        col_shift = outputsDFT{channelForMotionCorrection}(f,4);
        
        
        %%
        [nr,nc]=size(buf2ft);
        Nr = ifftshift([-fix(nr/2):ceil(nr/2)-1]);
        Nc = ifftshift([-fix(nc/2):ceil(nc/2)-1]);
        [Nc,Nr] = meshgrid(Nc,Nr);
        Greg = buf2ft.*exp(i*2*pi*(-row_shift*Nr/nr-col_shift*Nc/nc));
        Greg = Greg*exp(i*diffphase);
        
        
        %%
        regMovie{ch}(:, :, f) = uint16(abs(ifft2(Greg)));
        
    end
    fprintf('\n');
    
end

fprintf('Registering %d frames from %d channel(s) took %0.1f s\n\n', length(framesToUse), length(otherChannels), toc);


%%
%{
figure; 
for f = 1:3600
subplot(1,2,1)
     imagesc(regMovie{1}(:,:,f))
     axis image
     colormap gray

%      pause(1e-5)
%      pause
% end


% figure; 
% for f = 1:500
subplot(1,2,2)
     imagesc(movieMC{2}(:,:,f))
     axis image
     colormap gray
     set(gcf,'name',num2str(f))
     pause(1e-3)
end
%}
