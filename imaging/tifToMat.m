function movieMC = tifToMat(mousename, imagingFolder, mdfFileNumber, ch2read, convert2double)
% movieMC = tifToMat(mousename, imagingFolder, mdfFileNumber, ch2read, convert2double)
%
% output: movieMC will be a cell array with each element corresponding to a
% channel of ch2read.
%
% reads channel(s) ch2read of all tif files and put them in a single movie mat file.
%
% if convert2boule is true, movieMC will be of class double, otherwise it
% will be of class unit16.


%% set params
% mousename = 'fni17';
% imagingFolder = '151102';
% mdfFileNumber = 1; % or tif major
outName = [mousename,'-',imagingFolder, '-', num2str(mdfFileNumber)];
PP = struct;
PP.signalCh = 2;
PP.saveParams = false; % if 0, you don't need outName.
params = writeCaProcessParams(outName, mousename, imagingFolder, mdfFileNumber, PP);
% clear P

%% Read tif files to movieMC and convert it to double.
% tiffilename = sprintf('%s_%03d_*_ch2_MCM.TIF*', imagingFolder, mdfFileNumber);
% files = dir(fullfile(tifFold, tiffilename));

chAll = ch2read; % 2; % channelsToRead
movieMC = cell(1, max([chAll', params.dftRegCh, params.gcampCh])); 
for ch = chAll' 
  % Get list of MCM tif files corresponding to channel ch.
  tifNumsCh = params.tifNums(params.tifNums(:,4)==ch,:);
  tifList = cell(1, size(tifNumsCh,1));
  for f = 1:length(tifList)
      tifList{f} = fullfile(params.tifFold, assembleCaImagingTifName(tifNumsCh(f, :), params.oldTifName(f))); 
  end

  % Read tif files.
  movieMC{ch} = readTiffSet(tifList);
end
clear tifList  


%% Convert to double
if convert2double && ~isa(movieMC{params.gcampCh}, 'double')
  fprintf('Converting to double...\n');
  tic;  
  for ch = 1:length(movieMC)
      movieMC{ch} = double(movieMC{ch});
  end
  
  fprintf('%0.2fs\n', toc);
end





%% old version

%{
%%
if isunix
    dataPath = '/sonas-hs/churchland/nlsas/data/data';
elseif ispc
    dataPath = '\\sonas-hs.cshl.edu\churchland\data'; % FN
end

mouse = 'fni17';
imagingFold = '151102';
tifMajor = 1;
channelsSaved = 2; [1 2];

tifFold = fullfile(dataPath, mouse, 'imaging', imagingFold);

%%
tifList = cell(1,2);
movieMC = cell(1,2); % max(channelsSaved));

for ch = channelsSaved
    imfilename = sprintf('%s_%03d_*_ch%d_MCM.TIF*', imagingFold, tifMajor, ch);
    
    files = dir(fullfile(tifFold, imfilename));
%     files = files(cellfun(@(x)ismember(length(x),[17,25]) && ~isnan(str2double(x(12:13))), {files.name})); % make sure tif file name is of format : YYMMDD_mmm_nn.TIF or YYMMDD_mmm_nn_ch#_MCM.TIF
    tifList = {files.name};
    
    
    %% Set some parameters.
    cd(tifFold)
    
    nFramesPerMovie = NaN(1, length(tifList));
    tifInfo = cell(1, length(tifList));
    
    for t = 1:length(tifList)
        fprintf('\nReading info of tif file: %s\n\n', tifList{t});
        tifInfo{t} = imfinfo(tifList{t});
        %     channelsSaved = setImagedChannels(tifInfo{t}(1));
        
        nFramesPerMovie(t) = length(tifInfo{t}); % / length(channelsSaved);
    end
    
    totalFrames = sum(nFramesPerMovie);
    imWidth = tifInfo{1}(1).Width;
    imHeight = tifInfo{1}(1).Height;
    
    
    %%
    movieMC{ch} = uint16(zeros(imHeight, imWidth, totalFrames));
    
    
    %%
    frame = 0;
    
    for t = 1:length(tifList)
        frames = frame + 1 : frame + nFramesPerMovie(t);
        framesThisTif = 1:length(tifInfo{t});
        
        %% Read all the images out of the tiff and trim borders.
        
        fprintf('Reading tiff %s, channel %d\n', tifList{t}, ch);
        
        % Read frames
        for f = 1:length(framesThisTif)
            if mod(f, 100) == 0
                fprintf('%d ', f);
            end
            if mod(f, 1000) == 0
                fprintf('\n');
            end
            
            rawFrame = imread(tifList{t}, 'Index', framesThisTif(f), 'Info', tifInfo{t});
            
            movieMC{ch}(:, :, frames(f)) = rawFrame;
            
        end
        fprintf('\n');
        
        frame = frame + nFramesPerMovie(t);
        
    end
    
end


%% Produce averaged images : Median, std dev, max, range
medImage = cell(1,max(channelsSaved));
sdImage = cell(1,max(channelsSaved));
maxImage = cell(1,max(channelsSaved));
rangeImage = cell(1,max(channelsSaved));

for ch = 1:max(channelsSaved)
    %     goodMovie = movieMC{ch}(:, :, ~badFrames{ch});
    goodMovie = movieMC{ch};
    maxImage{ch} = max(goodMovie, [], 3);
    try % in case matlab runs out of memory
        medImage{ch} = median(goodMovie, 3);
    catch
    end
    
    try
        rangeImage{ch} = range(goodMovie, 3);
    catch
    end
    
    try
        sdImage{ch} = uint16(std(double(goodMovie), 0, 3));
    catch
    end
    
    clear goodMovie;
end

imHeight = size(movieMC{1}, 1);
imWidth = size(movieMC{1}, 2);


%% Save results
nowStr = datestr(now, 'yymmdd-HHMMSS');
save(fullfile(tifFold, ['PnevPanResults-' nowStr]), ...
    'imHeight', 'imWidth', 'medImage', 'sdImage', 'maxImage', 'rangeImage');

%%
% imwrite(maxImage{2}, 'MAX_151016_001_ch2_MCM.TIF')
%}


