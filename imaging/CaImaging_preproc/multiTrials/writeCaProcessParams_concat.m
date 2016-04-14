function params = writeCaProcessParams(outName, mouse, imagingFold, tifMajor, P)
% params = writeCaProcessParams(outName, mouse, imagingFold, tifMajor, P)
%
% INPUTS
%
% outName       -- string, name of parameters file to output. Will be
%                  placed in ./improcparams/
% mouse         -- string, name of mouse imagingFold   -- string, name of
% folder containing imaging data relative
%                  to dataPath/imaging
% tifMajor      -- tif "major" numbers, from MScan. May have length > 1
%
% P             -- structure with the following fields. All are optional,
%                  except for signalCh.
% signalCh      -- channel whose signal activity we want to analyse (usually gcamp channel)
% regFileNums   -- optional, 1 x 3 numeric array, containing info about which file to
%                  use for motion correction. Should be: [major minor
%                  channel]. Registration will be skipped if empty or NaN.
% regFrameNums  -- optional, numeric array or cell array. If numeric array, it
%                  indicates frames to use from reg file for motion
%                  correction. If cell array it indiates trial(s) to use
%                  for motion correciton. For simplicity, this trial must
%                  be chosen from the 1st tif file, and regTif must be the
%                  1st tif file (ie minor=1).
% behavName     -- optional. Name of behavioral file to merge into. Ignored
%                  if empty string (i.e., merging will not be attempted if
%                  not specified or empty)
% behavFcn      -- optional. Function to be called on behavioral data for
%                  cleanup and merge. Ignored if empty string.
% headerBug     -- optional, default 0. Whether the analog channel was from
%                  the buggy version of MScan that messes up the header.
% maxMaskWidth  -- optional, default 20. Max number of pixels that will be
%                  masked from each side of a frame after motion
%                  correction. Frames that required more pixel shifts in
%                  order to be registered will be marked as badFrames.
% channelsToWrite -- optional. Movie from these channels will be written to
%                  tif files. If not provided or if empty, both signalCh
%                  and dftRegCh channels will be written.
% analysisFolder  -- optional. If 1, files resulting from the analysis will
%                  be saved in a separate folder named "analysis". If not
%                  provided or 0, they will be saved in the same folder as
%                  the
%                  imaging folder.
% motionCorrDone -- optional. If 1, indicates motion correction files are
%                  already saved and it wont be performed again.
% pmt_th         -- optional. If provided, it determins the average pixel
%                  threshold for finding pmtOff frames (ie frames during
%                  which PMT was off). The pixel values of these frame will
%                  be turned to NaN (since movieMC is uint16, NaN will be
%                  converted to 0, so pmtOff frames will be all 0). If not
%                  provided, no frame will be marked as pmtOffFrame.
% channelsToRead -- Channels to load in case of motionCorrDone. Default:
%                  all saved channels.
% saveGoodMovieStats -- Whether or not to save stats (max, sd, median,
%                  range) of goodMovie (movie ignoring badFrames and
%                  pmtOffFrames). Default: true.
% pnevActivity -- optional, default true, indicating that Eftychios's
%                  algorithm will be used to drive ativity traces. If
%                  provided and 0, Eftychios's algorithm will not be executed.
% tifMinor     -- optional, default all tif files of a mdf file will be
%                  included in params. If provided and non-empty, only
%                  tifMinor files will be included in params for analysis.
% saveParams   -- optional, default true. If true, params will be saved to
%                  improcess folder.
%
% The following optional fields are related to Eftychios's algorithm
% numComps           -- number of components to be found, default:200
% tempSub            -- temporal subsampling for greedy initiation, set to 1 for no down sampling. default:3
% spaceSub           -- spatial subsampling for greedy initiation, set to 1 for no down sampling. default:2
% parallelTempUpdate -- do parallel temporal updating, default:false
% finalRoundMCMC     -- do a final round of MCMC method, default:false (if false, after merging 2 iterations of const foopsi will be done. If true, after merging 1 iter of const foopsi and 1 iter of MCMC will be done.)
% doPlots;           -- make some figures and a movie, default:false
% save4debug;        -- save Eftychios's variables (eg A,C,etc) after each step for debug purposes. default:false
% MCMC_B             -- MCMC deconvolution method, number of burn in samples. default 300
% MCMC_Nsamples      -- MCMC deconvolution method, number of samples after burn in. default 400
% MCMC_prec          -- MCMC deconvolution method, the extent of discarding the long slowly decaying tales of the ca response. default 1e-2
% save_merging_vars  -- whether to save some vars related to merging components; allows offline assessment if desired.
% search_dist        -- search distance when updating spatial components.
% concatTempUpdate   -- default:true, whether to do temporal updating on
%                       the concatenated trace of all trials, or to do it
%                       on the trace of each trial individually and then
%                       concatenate trials. The latter gives better
%                       estimate of time constants and removes artificial
%                       spikes at the begining of each trial.
%
% Goal: make it easy to write a parameters file for use with
% processCaImagingMCPnev()
%
% Example call:
%
% writeCaProcessParams('pnev-150901', 'mk32', '150901-1', 1, 2, [1 3 1], ...
%   700:800, '01-Sep-15_1321.mat', 'cleanupSeesaw', 1);
%
% This would output a parameters file at improcparams/pnev-150901.mat. It
% would look for files with '001' as their 'major' number (using all
% 'minor' numbers, e.g., 001_01, 001_02, etc.), and will use Eftychios's
% algorithm on channel 2. For motion correction, it will use the
% file with major/minor numbers 001_03, channel 1, frames 700 to 800 for the reference
% image. Since a behavior file is specified, it will try to merge it at the
% end, after calling cleanupSeesaw on that file. It will process the analog
% channels file (the .bin) assuming that the header was damaged by MScan.


%%
fprintf('Mouse: %s\n', mouse)

if isunix
    dataPath = '/sonas-hs/churchland/nlsas/data/data';
elseif ispc
    dataPath = '\\sonas-hs.cshl.edu\churchland\data'; % FN
end


%% Imaging tif files

params.tifFold = fullfile(dataPath, mouse, 'imaging', imagingFold);

if ~isfield(P, 'analysisFolder') || ~P.analysisFolder
    params.analysisFold = params.tifFold;
elseif P.analysisFolder
    params.analysisFold = fullfile(dataPath, mouse, 'analysis', imagingFold);
end

%{
imfilename = sprintf('%s_%03d_*.TIF*', imagingFold, tifMajor);
files = dir(fullfile(params.tifFold, imfilename));
files = files(cellfun(@(x)ismember(length(x),[17,25]) && ~isnan(str2double(x(12:13))), {files.name})); % make sure tif file name is of format : YYMMDD_mmm_nn.TIF or YYMMDD_mmm_nn_ch#_MCM.TIF
%}

% Handle folders named by YYYYMMDD or YYMMDD, or YYYYMMDD-1 or YYMMDD-1
% tokens = simpleTokenize(imagingFold, '-');
% if length(tokens{1}) == 8
%   tifStem = imagingFold(3:8);
% else
%   tifStem = imagingFold(1:6);
% end

%
files = dir(params.tifFold);
files = files(~[files.isdir]);
% filenames = {files.name};

% Parse all the filenames, save results into files struct array
for f = 1:length(files)
    [nums, valid, renameTif] = parseCaImagingTifName(files(f).name);
    files(f).nums = nums;
    files(f).valid = valid;
    files(f).renameTif = renameTif;
end

% Find the valid files, with the right major number
files = files([files.valid] & arrayfun(@(f) ismember(f.nums(2), tifMajor), files'));

if isfield(P, 'tifMinor') && ~isempty(P.tifMinor)
    files = files(arrayfun(@(f) ismember(f.nums(3), P.tifMinor), files'));
    params.allTifMinors = false; % all tif minors are included in params.
else
    params.allTifMinors = true;
end

if isempty(files)
    error('No files found!');
end

% Extract the numbers from each tif filename
params.tifNums = NaN(length(files), 4);
for f = 1:length(files)
    params.tifNums(f, :) = parseCaImagingTifName(files(f).name); % [date, major, minor, channel]
    %   params.tifFileStems{end+1} = fullfile(tifFold, sprintf('%s_%03d_%02d', tifStem, files(f).nums(2), files(f).nums(3)));
end

params.oldTifName = [files.renameTif];

% Display
fprintf('Found %d tif files. Showing filenames:\n', length(files));
for f = 1:length(files)
    fprintf('%s\n', files(f).name);
end

fprintf('Looking for signal on channel %d\n', P.signalCh);

params.activityCh = P.signalCh;


%% Registration-related info

if ~isfield(P, 'motionCorrDone') || ~P.motionCorrDone
    motionCorrDone = false;
elseif P.motionCorrDone
    motionCorrDone = true;
end
params.motionCorrDone = motionCorrDone;

if ~isfield(P, 'maxMaskWidth')
    params.maxMaskWidth = 20;
else
    params.maxMaskWidth = P.maxMaskWidth;
end


if motionCorrDone || ~isfield(P,'regFileNums') || isempty(P.regFileNums) || any(isnan(P.regFileNums))
    if ~isfield(P, 'channelsToRead') % Channels to load in case of motionCorrDone. Default: all saved channels
        params.channelsToRead = [];
    else
        params.channelsToRead = P.channelsToRead;
    end
    
    params.dftRegCh = P.signalCh; % channel to perform dftregistration on.
    fprintf('Registration info not specified. Assuming motion correction already completed. If not, will cause error only when running main processing\n');
    
else
    
    params.dftRegCh = P.regFileNums(3); % channel to perform dftregistration on.
    
    % FN: if the trial number without motion is provided (instead of the frames
    % without motion), find its corresponding frames. (you need to add the
    % following directory to your path for this to work: repoland\utils\Farzaneh)
    if iscell(P.regFrameNums)
        noMotionTrs = P.regFrameNums{1};
        file2read = fullfile(dataPath, mouse, 'imaging', imagingFold, sprintf('framecounts_%03d.txt', P.regFileNums(1)));
        P.regFrameNums = frameNumsSet(file2read, noMotionTrs);
        %     regFrameNums(regFrameNums > length(tifInfo)/length(channelsSaved)) = []; % regTif must be the 1st tif file.
    end
    
    
    if length(P.regFileNums) ~= 3
        error('regFileNums should have length 3 (tif major number, minor number, and channel)');
    end
    
    regTifFile = assembleCaImagingTifName([params.tifNums(1,1) P.regFileNums(1:2) NaN], params.oldTifName(P.regFileNums(2))); % regFileNums(2) is the tif minor name.
    fprintf('Using channel %d of %s for motion registration\n', P.regFileNums(3), regTifFile);
    
    params.regTifFile = fullfile(params.tifFold, regTifFile);
    %   fullfile(tifFold, sprintf('%s_%03d_%02d_ch%d.TIF', tifStem, regFileNums(1), regFileNums(2), regFileNums(3)));
    params.regFrameNums = P.regFrameNums;
    
    % Check file exists
    if ~exist(params.regTifFile, 'file')
        error('Tif file for registration does not exist: %s', params.regTifFile);
    end
    
    % channelsToWrite
    if ~isfield(P, 'channelsToWrite') || isempty(P.channelsToWrite)
        params.channelsToWrite = unique([params.dftRegCh, P.signalCh]);
    else
        params.channelsToWrite = P.channelsToWrite;
    end
    
    % threshold for pmtOffFrames
    if ~isfield(P, 'pmt_th') || isempty(P.pmt_th)
        params.pmtOffThr = NaN;
    else
        params.pmtOffThr = P.pmt_th; % if the average pixel intensity of a frame is below pmt_th that frame will be marked as a pmtOffFrame and its pixel values will be turned to NaN (since movieMC is uint16, NaN will be converted to 0, so pmtOffFrames will be all 0).
    end
    
end


%%
if ~isfield(P, 'pnevActivity')
    params.pnevActivity = true;
else
    params.pnevActivity = P.pnevActivity;
end


%%
if ~isfield(P, 'saveGoodMovieStats') % Whether to save stats (max, sd, median, range) of goodMovie (movie ignoring badFrames and pmtOffFrames or not). Default: true.
    params.saveGoodMovieStats = true;
else
    params.saveGoodMovieStats = logical(P.saveGoodMovieStats);
end


%% Merged filename

mergedName = [mouse '_' imagingFold '_Eft'];
params.mergedName = fullfile(dataPath, mouse, 'merged', mergedName);


%% headerBug

if isfield(P, 'headerBug')
    params.headerBug = P.headerBug;
else
    params.headerBug = 0;
end


%% ========  Behavior-related portion  ==========

params.binFiles = {};
params.framecountFiles = {};
params.behavFile = '';
params.behavFcn = '';

if isfield(P, 'behavName') && ~isempty(P.behavName)
    
    %% Behavior file for merging
    
    params.behavFile = fullfile(dataPath, mouse, 'behavior', P.behavName);
    
    
    %% Analog channel files, framecounts files
    
    for m = tifMajor
        params.binFiles{end+1} = fullfile(params.tifFold, sprintf('%d_%03d.bin', params.tifNums(1, 1), m));
        params.framecountFiles{end+1} = fullfile(params.tifFold, sprintf('framecounts_%03d.txt', m));
    end
    
    
    %% Verify that files exist
    
    if ~exist(params.behavFile, 'file') && ~exist([params.behavFile '.mat'], 'var')
        error('Behavior file does not exist: %s', params.behavFile);
    end
    
    for f = 1:length(params.binFiles)
        if ~exist(params.binFiles{f}, 'file')
            error('Analog channel binary file does not exist: %s', params.binFiles{f});
        end
        if ~exist(params.framecountFiles{f}, 'file')
            error('Framecounts file does not exist: %s', params.framecountFiles{f});
        end
    end
    
end


if isfield(P, 'behavFcn') && ~isempty(P.behavFcn)
    params.behavFcn = P.behavFcn;
end


%% Eftychio's algorithm for identifying ROIs and activity
if ~isfield(P, 'numComps')
    P.numComps = 200;
end
params.numComps = P.numComps;


if ~isfield(P, 'tempSub')
    P.tempSub = 3;
end
params.tempSub = P.tempSub;


if ~isfield(P, 'spaceSub')
    P.spaceSub = 2;
end
params.spaceSub = P.spaceSub;


% if ~isfield(P, 'parallelTempUpdate') % FN commented. In Eftychios V0.3.3 options.temporal_parallel takes care of it and it will be by default true if parallel processing exists on the machine.
%     P.parallelTempUpdate = false;
% end
% params.parallelTempUpdate = P.parallelTempUpdate;


if ~isfield(P, 'finalRoundMCMC')
    P.finalRoundMCMC = false;
end
params.finalRoundMCMC = P.finalRoundMCMC;


if ~isfield(P, 'doPlots')
    P.doPlots = false;
end
params.doPlots = P.doPlots;


if ~isfield(P, 'save4debug')
    P.save4debug = false;
end
params.save4debug = P.save4debug;


if ~isfield(P, 'MCMC_B')
    P.MCMC_B = 300;                             
end
params.MCMC_B = P.MCMC_B;


if ~isfield(P, 'MCMC_Nsamples')
    P.MCMC_Nsamples = 400;               
end
params.MCMC_Nsamples = P.MCMC_Nsamples;


if ~isfield(P, 'MCMC_prec')
    P.MCMC_prec = 1e-2;                       
end
params.MCMC_prec = P.MCMC_prec;


if ~isfield(P, 'save_merging_vars')
    P.save_merging_vars = false;
end
params.save_merging_vars = P.save_merging_vars;


if ~isfield(P, 'search_dist')
    P.search_dist = 3;
end
params.search_dist = P.search_dist;


if ~isfield(P, 'concatTempUpdate')
    P.concatTempUpdate = true;
end
params.concatTempUpdate = P.concatTempUpdate;


%% Save

if isfield(P, 'saveParams')
    saveParams = P.saveParams;
else
    saveParams = true;
end

if saveParams
    save(fullfile('improcparams', outName), 'params');
end


%% Copy to workspace

assignin('base', 'params', params);
