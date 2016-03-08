%% fni17, 8/26, 90:190

clearvars; home

mousename = 'FNI17'; fprintf(['\n====================',mousename,'====================\n'])
day = '150908';
filenumber = '001'; % corresponds to what mdf file?
imfilename = strcat(day, '_', filenumber);

% regFrameNums:
% if set to nonempty, noMotionTrs won't matter anymore.
% set to empty, if you want to use noMotionTrs to set regFrameNums.
% if both regFrameNums and noMotionTrs are set to empty, movie will be
% played and you will be asked to specify frame nums.
% frame nums of a given channel (not the original movie that includes all channels).

% noMotionTrs must correspond to the 1st tif file. 
exist_refImage = 1; % if 1, info from already saved filenumber 001 will be used for registration.
regFrameNums = []; % don't use the 1st frame bc it is NaN on a few top rows. 
noMotionTrs = [2]; % [9, 13, 14, 15, 18, 19, 23, 39, 77, 83, 103, 104, 105, 118, 127 ]; % [12 13 16]; 
playRefMovie = 1; % if 1, implay window will open for all frames whose median formed the reference image.
files2analyze = []; % which tif files you want to analyze. set it to [] if you want to analyze all tif files.

regTif = strcat(day, '_001_01.TIF'); % regTif = strcat(day, '_', filenumber, '_1.TIF');
channelForMotionCorrection = 1; % [1,2]; % channel numbers to use for motion correction (1:red, 2:green). If only 1 channel was saved, that will be used for motion correction regardless of the input. (FN).
outSuffix = ''; % '_MCM'; % set to '' if you don't want save the motion corrected movie as tif. Otherwise set to '_MCM' (it will be the suffix of the tif file.)
channels2write = 2; % imaging channels to write to tif file. It wont take effect if outSuffix is ''.

trimBorders = 1; % set it to 1 if using MScan option to correct for sinusoidal motion of the fast mirror.
maxMaskWidth = 20; % The mask will not be wider than this number of pixels.
headerBug = 1;



% set some variables.
imdir_server = fullfile('\\sonas-hs.cshl.edu\churchland\data', mousename, 'imaging', day);
dir0 = 'C:\Users\fnajafi\Documents\Data\';
imdir = fullfile(dir0, mousename, day, 'imaging');
behdir = fullfile(dir0, mousename, day, 'behavior');
anadir = fullfile(dir0, mousename, day, 'analysis');
if ~exist(imdir, 'dir'), mkdir(imdir), end
if ~exist(behdir, 'dir'), mkdir(behdir), end
if ~exist(anadir, 'dir'), mkdir(anadir), end

cd(imdir_server)
if ~exist(regTif,'file')
    regTif = [regTif(1:end-6),'1.TIF'];
end


im_matfile_all = [imfilename,'_all.mat'];
binFilename = fullfile(imdir, [imfilename, '.bin']);
framecountFilename = fullfile(imdir, ['framecounts_', filenumber, '.txt']);
pathToROIZip = fullfile(anadir, strcat(day, '_', filenumber, '_1_RoiSet.zip'));

cd(imdir_server)
tif_files = dir([imfilename,'_*.tif']);
tifList_all = {tif_files.name};
numberOfParts = length(tifList_all);
if isempty(files2analyze)
    files2analyze = 1:numberOfParts; 
end

% numberOfParts = length(regexp([tif_files(:).name], [imfilename,'_[0-9][0-9]?.TIF'])); % number of tif parts (of a session).
% b = [tif_files(:).name];
% i = regexp(b, [imfilename,'_[0-9][0-9]?.TIF']);
% ii = bsxfun(@plus, i, repmat((0:length(imfilename)+5)',[1,length(i)]));
% tifList_all = cellstr(b(ii'))


%% copy tif, bin, and text files from the server to your PC.
specificFiles = tifList_all(files2analyze); % {tif_files.name};
if exist_refImage
    specificFiles = [specificFiles, regTif];
end

copyIfNotCopied(imdir_server, imdir, 'tif', specificFiles)
copyIfNotCopied(imdir_server, imdir, 'bin')
copyIfNotCopied(imdir_server, imdir, 'txt')
fprintf('\n')


%%
clear tif_files a b i ii mousename imdir_server


%% Get the reference image % Farz moved it out of preprocessCaMovies_fn function and put it in imaging_prep, bc you run tif files one at a time, and you don't want to get the regImage everytime.
if exist_refImage
    cd(anadir)
    load(strcat(day, '_001_all.mat'), 'regTif', 'regFrameNums')
    if isempty(regFrameNums)
        regFrameNums = 1:50;
    end
        
    playRefMovie = 0;
end

file2read = fullfile(imdir, 'framecounts_001.txt');
% regFrameNums = regFrameNums_set(file2read, noMotionTrs);
cd(imdir)
[refImage, regFrameNums] = makeCaImagingRegImage_fn(regTif, regFrameNums, trimBorders, channelForMotionCorrection, file2read, noMotionTrs, playRefMovie);

if isempty(regFrameNums)
    error('no frames specified for motion correction.')
end

if ~exist_refImage
    figure('name', regTif);
    imagesc(refImage{channelForMotionCorrection}), axis image, colormap gray
end

save(fullfile(anadir, im_matfile_all), 'regTif', 'regFrameNums')


%%
clear regFrameNums


%% Motion correct and mask the movie. 
%%%%%%% IMPORTANT: if you want to write the MCM movie into tif files, make
%%%%%%% sure the tif folder location is not open. %%%%%%% 

for tifnum = files2analyze
    
%     imfilename_part = strcat(day, '_', filenumber, '_', num2str(tifnum));
%     im_matfile = strcat(imfilename_part, '.mat') 
%     tifList = {strcat(imfilename_part, '.TIF')}

    tifList = tifList_all(tifnum);
    [~, fStem] = fileparts(tifList{1});
    im_matfile = strcat(fStem, '.mat');
    
    %%
    cd(imdir)
    [movieMC, badFrames, maskBounds, outputsDFT, imWidth, imHeight] = preprocessCaMovies_fn(refImage, tifList, outSuffix, maxMaskWidth, channelForMotionCorrection, channels2write);   %     [movieMC, badFrames, maskBounds, imWidth, imHeight] = preprocessCaMovies_fn(tifList, regTif, regFrameNums, outSuffix, maxMaskWidth, trimBorders, channelForMotionCorrection);
    % badFrames: when a frame has a larger pixel shift during motion correction than the width of the mask, it is marked as bad (true).
    
    %%
    save(fullfile(anadir, im_matfile), 'movieMC', 'badFrames', 'maskBounds', 'outputsDFT', '-v7.3')
%     save(fullfile(anadir, im_matfile), 'movieMC', 'badFrames', 'maskBounds', 'outputsDFT', 'regFrameNums', '-v7.3')
%     save(fullfile(anadir, im_matfile), '-append', 'movieMC', 'badFrames', 'maskBounds', 'outputsDFT', 'regFrameNums', '-v7.3')
%     save(im_matfile, 'maskBounds', '-append')

    %%
    clear movieMC badFrames maskBounds outputsDFT
    
end


%% delete files from the local computer
cd(imdir)
delete([imfilename,'_*.tif'])
delete(regTif)
delete('*.bin')
delete('*.txt')

%{
think about this if there is movement, then finding rois on the average is a bad idea.
compute rois on the mcm movie; if so, then save the tif file of mcm movie.
then correct for the imcoordinates
rois{rr}.mnCoordinates(:, 1) + maskBounds(3) -1 , rois{rr}.mnCoordinates(:, 2) + maskBounds(1) -1

% check if mnCoordinates(:, 2) is computed from bottom or top. if bottom then:
% + imHeight - maskBounds(2)
%}

%% Choose RIOs on the original (512x512) movie.
disp('Now choose RIOs on the original (512x512) movie.')

% Load (the original movie) file in Fiji. Take Z-projection of median or
% SD, change color scheme, choose ROIs. Save ROIs.

% open a tif file (you do the first part) in fiji.

% Analyze/set scale : to change the coordinates to pixels. (click remove
% scale)

% Image/stack/z project : to take std or median (remember to exclude
% pmtOffFrames).

% use free hand to draw ROIs.
% Analyze/ tools/ ROI manager : to manage the rois.
% after selecting all ROIs, save the tif file.
% on the ROI manager window, click More, and Save to save the ROIs.
% also click Flattern to save an RGB version of the figure.
% to see rois: open the tif file; then go to Image/Overlay/To ROI manager.


%% Get the mean raw fluorecense for each neuron using the motion corrected movie.
activity = applyFijiROIsToTifs_fn(pathToROIZip, day, filenumber, numberOfParts, imWidth, imHeight);


%% mean fluorescence for the entire session and all neurons (nFramesTotal x nUnits).
% save(im_matfile, 'activity','-append')
save(fullfile(anadir, im_matfile_all), '-append', 'activity')


%% Trialization
% load(fullfile(anadir, im_matfile_all), 'activity')

[framesPerTrial, trialNumbers, frame1RelToStartOff, badAlignments, trialStartMissing, framesPerTrial_galvo] = ...
    framesPerTrialStopStart3An_fn(binFilename, framecountFilename, headerBug);

if any(badAlignments)
    error('Farzaneh Note: this variable is not used in mergeActivityIntoAlldata. see if you need it ... ')
end


%% load behavioral data
list_files = dir(behdir);
list_files = list_files(3:end);
[~,b] = sort([list_files(:).datenum]);
list_files = list_files(b);

beh_matfile = list_files(str2double(filenumber)).name

load(fullfile(behdir, beh_matfile), 'all_data')
length(fieldnames(all_data))
clear list_files b


%% find trials that were not recorded in mscan, and see if it is bc of short iti.
trmiss = find(~ismember(1:length(trialNumbers), trialNumbers));

if ~isempty(trmiss)
    fprintf('Index of trials not recorded in MScan: %d\n', trmiss)
    
    % duration of no scopeTTL, preceding the trial
    dur_nottl = NaN(1,length(all_data)-1);
    for tr = 2:length(all_data)-1
        dur_nottl(tr) = all_data(tr).parsedEvents.states.start_rotary_scope(1)*1000 - all_data(tr-1).parsedEvents.states.stop_rotary_scope(1)*1000;
    end
    fprintf('noTTL duration of trials not recorded in MScan: %d\n', dur_nottl(trmiss))
end


%% compare frame numbers per trial driven from bcontrol states with frameCounts in the text file and frame numbers driven from the galvo analog signal.
frameLength = 1000 / 30.9;
nfrs = NaN(1,length(all_data)-1);
for tr = 1:length(all_data)-1
    % duration of a trial in mscan (ie duration of scopeTTL being sent).
    durtr = all_data(tr).parsedEvents.states.stop_rotary_scope(1)*1000 + 500 - ...
        all_data(tr).parsedEvents.states.start_rotary_scope(1)*1000; % 500 is added bc the duration of stop_rotary_scope is 500ms.
    nfrs(tr) = durtr/ frameLength;
end
nfrs(trmiss) = [];
framesBcontrol = floor(nfrs);

[(1:length(framesPerTrial))' framesBcontrol' framesPerTrial' framesPerTrial_galvo(1:length(framesPerTrial))' [framesBcontrol - framesPerTrial]']


%%
save(fullfile(anadir, im_matfile_all), 'framesPerTrial', 'trialNumbers', 'frame1RelToStartOff', 'badAlignments', '-append')



%% Set vars for mergeActivityIntoAlldata
% load imaging data
load(fullfile(anadir, im_matfile_all), 'activity', 'framesPerTrial', 'trialNumbers', 'frame1RelToStartOff')

% concatenate badFrames from all tif parts. (badFrames: frames with lots of
% motion).
r = 0;
badFrames_all = NaN(size(activity,1),1);
for tifnum = 1:numberOfParts
    
    imfilename_part = strcat(day, '_', filenumber, '_', num2str(tifnum));
    im_matfile = strcat(imfilename_part, '.mat');
    load(fullfile(anadir, im_matfile), 'badFrames')
    
    badFrames_all(r+1 : r+length(badFrames)) = badFrames;
    r = r+length(badFrames);
end
   
clear badFrames

%% set dFOF for each trial. Add it to a new field in all_data.
all_data = mergeActivityIntoAlldata_fn(all_data, activity, framesPerTrial, ...
  trialNumbers, frame1RelToStartOff, badFrames_all);


%% append to im mat file all.
save(fullfile(anadir, im_matfile_all), 'all_data', '-append')
% save(fullfile(behdir, beh_matfile), 'all_data', '-append')


%% for further analysis
avetrialAlign.m


%
remember to copy analysis files to the server 
delete tif files (and perhaps analysis files) from local computer.


%%
%{
t = [7 0 2 2];
t = [9 0 1 3];
t = [10 0 1 3];
t = [13 0 2 3];
t = [12 0 1 4];

maskBounds(1) = t(1)+1;
maskBounds(2) = 402-t(2);
maskBounds(3) = t(3)+1;
maskBounds(4) = 512-t(4)

% save('150617_001_6.mat', '-append', 'maskBounds')

movieMC2 = NaN(512, 402, size(movieMC,3));
movieMC2(maskBounds(3):maskBounds(4), maskBounds(1):maskBounds(2), :) = movieMC;
%}

