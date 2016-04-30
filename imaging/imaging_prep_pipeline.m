% writeCaProcessParams(outName, mouse, imagingFold, tifMajor, signalCh, regFileNums, regFrameNums, behavName, behavFcn, headerBug,...
%     channelsToWrite, maxMaskWidth, motionCorrDone, analysisFolder)

hpc_dir = '\\sonas-hs.cshl.edu\churchland-hpc-home\fnajafi\matlab';
if ispc, cd(hpc_dir), end

%%
mousename = 'fni17';
imagingFolder = '150908';
mdfFileNumber = 1; % or tif major
outName = [mousename,'-',imagingFolder, '-', num2str(mdfFileNumber)]

P = struct;
P.saveParams = true; % if 0, you don't need outName then.

P.pnevActivity = 1; % 1 % whether to run Eftychios's algorithm or not.
    P.poolsize = 8; % 0 for default
    P.limit_threads = 16; %2; % 8 % 0 for default
    P.multiTrs = 1; % remember if this is 1 you need to run the trialization part below to save cs_frtrs and Nnan to imfilename.
    P.ARmodelOrder = 0; % 2;
    P.orderROI_extractDf = 0; % true;
    P.maxFrsForMinPsn = 16000; % 10000 %[] for default % min(Y) and P.sn will be computed on the first maxFrsForMinPsn frames.

    P.numComps = 500; % remember to set to 30 or smaller for tests!
    P.tempSub = 3;
    P.spaceSub = 2;

P.motionCorrDone = 1; % if 0, the below matters.
    % if 1
    P.channelsToRead = 2; %[1,2]; % 2 % []; % set to both channels if you want to save goodMovieStats % it will be only used when MC is done. if motion correction is done, specify what channels to read (which later will be used for computing average images and eftychios's algorithm).
    % if 0
    P.regFrameNums = {2}; % {2}; % noMotionTr
    P.regFileNums = [1 1 1]; %[2 1 1] % file to use for motion correction major, minor, channel % params.dftRegCh = P.regFileNums(3); % channel to perform dftregistration on.

P.saveGoodMovieStats = 0;
    

P.pmt_th = []; % 1400;
P.tifMinor = []; %[]; % set to [] if you want all tif minor files to be analyzed.
P.channelsToWrite = [1,2];
P.signalCh = 2; % 2 % channel whose signal activity you want to analyze (normally 2 for gcamp channel). % 
P.headerBug = 0;
P.maxMaskWidth = 30;
P.analysisFolder = 0;
P.behavName = '';


P.save_merging_vars = true;
% P.parallelTempUpdate = true; % false;
P.save4debug = false; % false;
P.doPlots = false;

P.finalRoundMCMC = false;
P.MCMC_B = 200; % 300
P.MCMC_Nsamples = 200; % 400
P.MCMC_prec = 1e-2; % 1e-2 --> 5e-2 needs test.
P.search_dist = 3; % set to a smaller number, e.g., 2 to prevent the components to grow too much and pick up neighboring pixels (Eft).

        
%{
behavdir = dir(fullfile(dataPath, mousename, 'behavior', [mousename, '_', datestr(datenum(imagingFolder, 'yymmdd')), '*.mat']));
behavName = {behavdir.name};
%}


%%
% remember to cd to the foler that contains the improcparams folder!

params = writeCaProcessParams(outName, mousename, imagingFolder, mdfFileNumber, P);

% writeCaProcessParams(outName, mousename, imagingFolder, mdfFileNumber, signalCh, regFileNums, noMotionTr,....
%     behavName, '', headerBug, channelsToWrite, maxMaskWidth, motionCorrDone, analysisFolder, pmt_th, channelsToRead, saveGoodMovieStats, pnevActivity, tifMinor);

% day = num2str(params.tifNums(1,1));
% day_number = datenum(day, 'yymmdd');
clearvars -except params outName 



%%
processCaImagingMCPnev(outName)




%% Set behavFileName, binaryfileName, frameCountFileName

% load params file
% hpc_dir = '\\sonas-hs.cshl.edu\churchland-hpc-home\fnajafi\matlab';
% outName = 'fni17-151016';
% load(fullfile(hpc_dir, 'improcparams',outName))
dataPath = '\\sonas-hs.cshl.edu\churchland\data'; % lab PC

mousename = 'fni17';
imagingFolder = '150904'; % '151021';
mdfFileNumber = 1; % or tif major
signalCh = 2;

[imfilename, pnevFileName, tifFold, date_major] = setImagingAnalysisNames(mousename, imagingFolder, mdfFileNumber, signalCh);
cd(tifFold)

% Set behavFile, binFiles, and framecountFiles (Related to behavior and
% trialization).
params.headerBug = 0;
%{
behavName = dir(fullfile(dataPath, mousename, 'behavior', [mousename, '_', datestr(datenum(imagingFolder(1:6), 'yymmdd')), '*.mat']));
behavName = {behavName.name};
behavName = behavName{mdfFileNumber};
pacrams.behavFile = fullfile(dataPath, mousename, 'behavior', behavName);
%}
% set filenames
[alldata_fileNames, ~] = setBehavFileNames(mousename, {datestr(datenum(imagingFolder, 'yymmdd'))});
% sort it
[~,fn] = fileparts(alldata_fileNames{1});
a = alldata_fileNames(cellfun(@(x)~isempty(x),cellfun(@(x)strfind(x, fn(1:end-4)), alldata_fileNames, 'uniformoutput', 0)))';
[~, isf] = sort(cellfun(@(x)x(end-25:end), a, 'uniformoutput', 0));
alldata_fileNames = alldata_fileNames(isf);
params.behavFile = alldata_fileNames{mdfFileNumber};
% showcell(params.behavFile)

params.binFiles = {};
params.framecountFiles = {};
for m = mdfFileNumber
    params.binFiles{end+1} = fullfile(tifFold, sprintf('%s_%03d.bin', imagingFolder(1:6), m));
    params.framecountFiles{end+1} = fullfile(tifFold, sprintf('framecounts_%03d.txt', m));
end
showcell(params.binFiles)
showcell(params.framecountFiles)

   
%% Load alldata

% load(params.behavFile, 'all_data')
excludeLastTr = 0; % don't exclude the last trial. You need it this way for alignment purposes with the imaging data.
[all_data, ~] = loadBehavData(alldata_fileNames(mdfFileNumber), [], [], [], excludeLastTr);
fprintf('Number of behavioral trials: %d\n', length(all_data))
% load(imfilename, 'all_data')  % this one has the additinal imaging and mouse helped fields.


%% Get trialization-related parameters

%   fname = 'framesPerTrialStopStart3An_fn';
%   eval([fname,'(params.binFiles{f}, params.framecountFiles{f}, params.headerBug, all_data)'])
  
framesPerTrial = cell(1, length(params.binFiles));
trialNumbers = cell(1, length(params.binFiles));
frame1RelToStartOff = cell(1, length(params.binFiles));
badAlignTrStartCode = cell(1, length(params.binFiles));
trialStartMissing = cell(1, length(params.binFiles));
framesPerTrial_galvo = cell(1, length(params.binFiles));
trialCodeMissing = cell(1, length(params.binFiles));

% you combined framesPerTrialStopStart3An_fn and
% framesPerTrialStopStart3An_fn2, so no need to check for data anymore!

% if str2double(imagingFolder) < 151101
    for f = 1:length(params.binFiles)
        [framesPerTrial{f}, trialNumbers{f}, frame1RelToStartOff{f}, badAlignTrStartCode{f}, framesPerTrial_galvo{f}, trialStartMissing{f}, trialCodeMissing{f}] = ...
            framesPerTrialStopStart3An_fn(params.binFiles{f}, params.framecountFiles{f}, params.headerBug, all_data);
    end
% else % starting from 151101 you added the state trial_start_rot_scope which lasts for 36ms and sends the trialStart signal. so you can use simpler codes for finding framesPerTrial.
%     for f = 1:length(params.binFiles)
%         [framesPerTrial{f}, trialNumbers{f}, frame1RelToStartOff{f}, badAlignTrStartCode{f}, framesPerTrial_galvo{f}] = ...
%             framesPerTrialStopStart3An_fn2(params.binFiles{f}, params.framecountFiles{f}, params.headerBug, all_data);
%     end    
% end

framesPerTrial = [framesPerTrial{:}];
trialNumbers = [trialNumbers{:}]; % trial number found by the trialStart signal (sent from bcontrol to mscan).
frame1RelToStartOff = [frame1RelToStartOff{:}];
badAlignTrStartCode = [badAlignTrStartCode{:}];
trialStartMissing = [trialStartMissing{:}];
framesPerTrial_galvo = [framesPerTrial_galvo{:}];
trialCodeMissing = [trialCodeMissing{:}];
  
nansum(framesPerTrial)
size(framesPerTrial)
size(trialNumbers)
size(trialStartMissing)
% size(trialCodeMissing)
% size(badAlignTrStartCode)
find(trialStartMissing)
find(trialCodeMissing)
% find(badAlignTrStartCode)


%%
save(imfilename, '-append', 'framesPerTrial', 'trialNumbers', 'frame1RelToStartOff', 'badAlignTrStartCode', 'trialStartMissing', 'trialCodeMissing')

%{
if ~exist([imfilename, '.mat'], 'file')
    save(imfilename, 'framesPerTrial', 'trialNumbers', 'frame1RelToStartOff', 'badAlignTrStartCode', 'trialStartMissing', 'trialCodeMissing')
end
%}

%% Get vars for running Efythios algorith for the multi-trial case
%
% Remember you need badFrames to get total number of recorded frames.

% p.trialCodeMissing = trialCodeMissing;
[cs_frtrs, Nnan_nanBeg_nanEnd] = update_tempcomps_multitrs_setvars...
    (mousename, imagingFolder, mdfFileNumber); % , p);
% [cs_frtrs, Nnan] = update_tempcomps_multitrs_setvars(mousename, imagingFolder, mdfFileNumber, allTifMinors, tifMinor);
size(Nnan_nanBeg_nanEnd)
size(cs_frtrs)


%%
save(imfilename, '-append', 'cs_frtrs', 'Nnan_nanBeg_nanEnd')
%}

%% Some tests for frame numbers per trial

% find trials that were not recorded in mscan, and see if it is bc of short iti.
trmiss = find(~ismember(1:length(trialNumbers), trialNumbers));

% compare frame numbers per trial driven from bcontrol states with frameCounts in the text file and frame numbers driven from the galvo analog signal.
frameLength = 1000 / 30.9;
nfrs = NaN(1, min(length(all_data)-1, sum(~isnan(framesPerTrial))));
for tr = 1 : min(length(all_data)-1, sum(~isnan(framesPerTrial)))
    % duration of a trial in mscan (ie duration of scopeTTL being sent).
    durtr = all_data(tr).parsedEvents.states.stop_rotary_scope(1)*1000 + 500 - ...
        all_data(tr).parsedEvents.states.start_rotary_scope(1)*1000; % 500 is added bc the duration of stop_rotary_scope is 500ms.
    nfrs(tr) = durtr/ frameLength;
end
nfrs(trmiss) = [];
framesBcontrol = floor(nfrs);
framesBcontrol = [framesBcontrol NaN];

minl = min(length(framesPerTrial), length(framesPerTrial_galvo));
trNum_framsBcontrol_framesMscan_framesGalvo_bcontrolMinusMscan = [(1:minl)', ...
    framesBcontrol(1:minl)', framesPerTrial(1:minl)', framesPerTrial_galvo(1:minl)', [framesBcontrol(1:minl) - framesPerTrial(1:minl)]']

figure; plot(trNum_framsBcontrol_framesMscan_framesGalvo_bcontrolMinusMscan(:,2:end-1));


if ~isempty(trmiss)
    fprintf('Index of trials not recorded in MScan: %d\n', trmiss)
    
    % duration of no scopeTTL, preceding the trial
    dur_nottl = NaN(1,length(all_data)-1);
    for tr = 2:length(all_data)-1
        dur_nottl(tr) = all_data(tr).parsedEvents.states.start_rotary_scope(1)*1000 - all_data(tr-1).parsedEvents.states.stop_rotary_scope(1)*1000;
    end
    fprintf('noTTL duration of trials not recorded in MScan: %d\n', dur_nottl(trmiss))
end




%% manual RIO selection  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Write to tif the average images %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

image2load = 'medImage';
load(imfilename, image2load) % ('maxImage', 'medImage', 'rangeImage', 'sdImage')

outFile = fullfile(tifFold, [image2load, '_', date_major]);
ch2write = signalCh;
writeTif(eval(image2load), outFile, ch2write)

%
% Write to tif the average images (med,sd,etc) for openning in imageJ. both channels will be written to the same tif file.
%{
load(imfilename, 'medImage', 'maxImage', 'rangeImage', 'sdImage')
aveImages = {'medImage', 'maxImage', 'rangeImage', 'sdImage'};

for ia = 1:length(aveImages)
    outFile = fullfile(tifFold, sprintf('%s_%s_%03d', aveImages{ia}, imagingFolder(1:6), mdfFileNumber));    
    fprintf('Writing %s\n', aveImages{ia})

    imageVar = eval([aveImages{ia}, '{1}']);
    imwrite(uint16(imageVar), [outFile,'.TIF'], ...
        'Resolution', [size(imageVar, 2) size(imageVar, 1)], 'Compression', 'none');
    
    for ch = 2:length(eval(aveImages{ia}))
        imageVar = eval([aveImages{ia}, '{', num2str(ch), '}']);
        imwrite(uint16(imageVar), [outFile,'.TIF'], ...
            'Resolution', [size(imageVar, 2) size(imageVar, 1)], 'Compression', 'none', ...
            'WriteMode', 'append');
    end
end
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Select RIOs in fiji   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

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

% on the ROI manager window, click More, and Save to save the ROI zip file.

% also click Flattern to save an RGB version of the figure.

% to see ROIs: open the tif file; then go to Image/Overlay/To ROI manager.


%% compute activity for manually selected ROIs
roiCh = 2; % channel on which ROIs was found. you can find activity on signalCh=1 using rois found roiCh=2;
[activity, rois] = manualROIactivity(mousename, imagingFolder, mdfFileNumber, signalCh, roiCh);
% activity(pmtOffFrames{signalCh},:) = NaN; % we are not doing this here, instead we will do it in 

% plot ROIs found manually on the sdImage
figure; hold on;
imagesc(sdImage{signalCh}); 
 
colors = hot(2*size(A2,2));
colors = colors(end:-1:1,:);
for ir = 1:length(rois)
    plot(rois{ir}.mnCoordinates(:,1), rois{ir}.mnCoordinates(:,2), '-', 'color', colors(ir,:))
end



%%
load(imfilename, 'activity') % manual activity
load('demo_results_fni17-151102_001', 'C2', 'C_df', 'S_df')
activity = C2';
dFOF = C_df(1:end-1,:)'; % excluding the last unit which is the background.
load(imfilename, 'badFrames', 'pmtOffFrames', 'framesPerTrial', 'trialNumbers', 'frame1RelToStartOff')
load(imfilename, 'all_data')

%% merge imaging into alldata. set dFOF for each trial. Add it to a new field in all_data.

 % remember that for badAlignTrStartCode, trialStartMissing you are
      % still setting alldata.frameTimes but you need to take that into
      % account when analyzing the data!
      
% size of the window over which minimum will be computed in the running min
% step of computing F in order to compute DF/F
minPts = 7000; %800;
[all_data, mscanLag] = mergeActivityIntoAlldata_fn(all_data, activity, framesPerTrial, ...
  trialNumbers, frame1RelToStartOff, badFrames{signalCh}, pmtOffFrames{signalCh}, minPts, dFOF, S_df');
  
% alldata = mergeActivityIntoAlldata_fn(alldata, C2', framesPerTrial, ...
%   trialNumbers, frame1RelToStartOff, badFrames{signalCh}, pmtOffFrames{signalCh}, minPts, C_df', S_df');

%% save alldata
save(imfilename, 'all_data', 'mscanLag', '-append')


%% for further analysis
avetrialAlign

% remember to copy analysis files to the server 
% delete tif files (and perhaps analysis files) from local computer.


%%
%{
if ispc    
    hpc_dir = '\\sonas-hs.cshl.edu\churchland-hpc-home';
else
    hpc_dir = '/sonas-hs/churchland/hpc/home';
end

username = 'fnajafi';
outName = [mousename,'-',imagingFolder];
load(fullfile(username, 'matlab', 'improcparams', outName), 'params');
%}




%% If ch1 average images were saved separately from ch2 average images, combine them into the same var and save them again.

date_major
load([date_major,'_ch1'])

meda = medImage;
ranga = rangeImage;
maxa = maxImage;
sda = sdImage;

load(date_major, 'medImage', 'maxImage', 'rangeImage', 'sdImage')
medImage{1} = meda{1};
rangeImage{1} = ranga{1};
maxImage{1} = maxa{1};
sdImage{1} = sda{1};


aveImages = {'medImage', 'maxImage', 'rangeImage', 'sdImage'};
for i=1:length(aveImages)
    top = eval(aveImages{i});
    figure; 
    for ii=1:2,
        subplot(2,1,ii)
        imagesc(top{ii})
    end
end


%%
save(date_major, '-append', 'medImage', 'maxImage', 'rangeImage', 'sdImage')

