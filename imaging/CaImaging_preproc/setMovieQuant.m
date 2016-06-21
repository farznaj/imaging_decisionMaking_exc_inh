function quantImage = setMovieQuant(mouse, imagingFolder, mdfFileNumber, ch2ana, quantToUse)

% quantToUse = .003;


%%
if isunix
    dataPath = '/sonas-hs/churchland/nlsas/data/data';
elseif ispc
    dataPath = '\\sonas-hs.cshl.edu\churchland\data';
end
tifFold = fullfile(dataPath, mouse, 'imaging', imagingFolder);
% pathToROIZip = fullfile(tifFold, sprintf('RoiSet_%s_%03d_ch%d_MCM.zip', imagingFolder, mdfFileNumber, roiCh));
%

%{
load(fullfile('improcparams', paramsFileName))
tifFold = params.tifFold;
date_major = sprintf('%06d_%03d', params.tifNums(1, 1:2));
pathToROIZip = fullfile(tifFold, sprintf('RoiSet_%s_ch%d_MCM.zip', date_major, params.gcampCh));
files = dir(fullfile(tifFold, sprintf('%s_*_ch%d_MCM.TIF', date_major, ch2ana)));
%}

% Set the tif files corresponding to mdf file mdfFileNumber and channel ch2ana
files = dir(fullfile(tifFold, sprintf('%s_%03d_*_ch%d_MCM.TIF', imagingFolder, mdfFileNumber, ch2ana)));
% tifList = {files.name}
tifList = cell(1, length(files));
for itif = 1:length(files)
    tifList{itif} = fullfile(tifFold, files(itif).name);
end
% showcell(tifList')


%% Read motion corrected tif files of channel ch2ana into movieMC

movieMC = [];
for t = 1:length(tifList)
    fprintf('Reading tif file %s\n', tifList{t})
    movieMC = cat(3, movieMC, bigread2(tifList{t}));
end
        

%% Remove badFrames and pmtOffFrames

imfilename = fullfile(tifFold, sprintf('%s_%03d', imagingFolder, mdfFileNumber));
load(imfilename, 'badFrames', 'pmtOffFrames')

y = movieMC(:, :, ~badFrames{ch2ana} & ~pmtOffFrames{ch2ana});
clear movieMC


%% Compute quantile

quantImage = cell(1,2);
quantImage{ch2ana} = double(quantile(y, quantToUse, 3));


%% Save

save(imfilename, '-append', 'quantImage')


