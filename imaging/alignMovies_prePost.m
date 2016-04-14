function alignMovies_prePost(mousename, imagingFolder, mdfFileNumber, ch2read)
% Make average aligned movies, ie align movies of each trial on specific
% trial events, then get average across aligned movies.
%
% mousename = 'fni17';
% imagingFolder = '151102';
% mdfFileNumber = 1; % or tif major
% ch2read = 2;


%% Read tif files

tifList = tifListSet(mousename, imagingFolder, mdfFileNumber, ch2read);

Y = [];
for t = 1:length(tifList)
    fprintf('Reading tif file %s\n', tifList{t})
    Y = cat(3, Y, bigread2(tifList{t}));
end

if ~isa(Y,'double');    Y = double(Y);  end         % convert to double


%% load framesPerTrial

if isunix
    dataPath = '/sonas-hs/churchland/nlsas/data/data';
elseif ispc
    dataPath = '\\sonas-hs.cshl.edu\churchland\data'; % FN
end
tifFold = fullfile(dataPath, mousename, 'imaging', imagingFolder);
date_major = sprintf('%s_%03d', imagingFolder, mdfFileNumber);

load(fullfile(tifFold, date_major), 'framesPerTrial', 'alldata', 'frameLength', 'trs2rmv')

if ~exist('frameLength', 'var')
    frameLength = 1000/30.9;
end


%% Set movie_trials: cell array, each cell contains movie for a trial, size pix x pix x frames

% movieSt = []; %4; % movieEn
% cs_frtrs = framesPerTrialMovie(mousename, imagingFolder, mdfFileNumber, movieSt, movieEn);
framesPerTrialNoNaN = framesPerTrial(~isnan(framesPerTrial));
cs_frtrs = unique([0 cumsum(framesPerTrialNoNaN)]); % you are not including frames at the end that don't have a corresponding behavior.

movie_trials = cell(1, length(cs_frtrs)-1);
for itr = 1:length(cs_frtrs)-1
    frs = cs_frtrs(itr)+1 : cs_frtrs(itr+1);
    movie_trials{itr} = Y(:,:,frs);
end


%% Set vars for alignment

traceTimeVec = {alldata.frameTimes}; % time vector of the trace that you want to realign.

defaultPrePostFrames = 2; % default value for nPre and postFrames, if their computed values < 1.
shiftTime = 0; % event of interest will be centered on time 0 which corresponds to interval [-frameLength/2  +frameLength/2]
scaleTime = frameLength;

flag_traces = false; % if 1, 'traces' contains the temporal traces for each trial. if 0, it contains the movie for each trial.


%% Set event times (ms) relative to when bcontrol starts sending the scope TTL. event times will be set to NaN for trs2rmv.

[timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, ...
    time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks] = ...
    setEventTimesRelBcontrolScopeTTL(alldata, trs2rmv);


%%
alignedEvent = 'trialBeg'; % align on frame 1
[traces_aligned_fut, time_aligned_trialBeg, eventI_trialBeg] = alignTraces_prePost_allCases...
    (alignedEvent, movie_trials, traceTimeVec, frameLength, defaultPrePostFrames, shiftTime, scaleTime, timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, trs2rmv, flag_traces);

% do averaging across trs
movieAligned_trialBeg = nanmean(traces_aligned_fut, 4);


%% traces_aligned_fut : pix x pix x frs x trs
alignedEvent = 'initTone';
[traces_aligned_fut, time_aligned_initTone, eventI_initTone] = alignTraces_prePost_allCases...
    (alignedEvent, movie_trials, traceTimeVec, frameLength, defaultPrePostFrames, shiftTime, scaleTime, timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, trs2rmv, flag_traces);

movieAligned_initTone = nanmean(traces_aligned_fut, 4);


%%
alignedEvent = 'stimOn';
[traces_aligned_fut, time_aligned_stimOn, eventI_stimOn] = alignTraces_prePost_allCases...
    (alignedEvent, movie_trials, traceTimeVec, frameLength, defaultPrePostFrames, shiftTime, scaleTime, timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, trs2rmv, flag_traces);

movieAligned_stimOn = nanmean(traces_aligned_fut, 4);


%%
alignedEvent = 'goTone';
[traces_aligned_fut, time_aligned_goTone, eventI_goTone] = alignTraces_prePost_allCases...
    (alignedEvent, movie_trials, traceTimeVec, frameLength, defaultPrePostFrames, shiftTime, scaleTime, timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, trs2rmv, flag_traces);

movieAligned_goTone = nanmean(traces_aligned_fut, 4);


%%
alignedEvent = '1stSideTry';
[traces_aligned_fut, time_aligned_1stSideTry, eventI_1stSideTry] = alignTraces_prePost_allCases...
    (alignedEvent, movie_trials, traceTimeVec, frameLength, defaultPrePostFrames, shiftTime, scaleTime, timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, trs2rmv, flag_traces);

movieAligned_1stSideTry = nanmean(traces_aligned_fut, 4);


%%
alignedEvent = 'reward';
[traces_aligned_fut, time_aligned_reward, eventI_reward] = alignTraces_prePost_allCases...
    (alignedEvent, movie_trials, traceTimeVec, frameLength, defaultPrePostFrames, shiftTime, scaleTime, timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, trs2rmv, flag_traces);

movieAligned_reward = nanmean(traces_aligned_fut, 4);


%%
save('movieAligned', 'movieAligned_trialBeg', 'movieAligned_initTone', 'movieAligned_stimOn', 'movieAligned_goTone', 'movieAligned_1stSideTry', 'movieAligned_reward', ...
    'eventI_trialBeg', 'eventI_initTone', 'eventI_stimOn', 'eventI_goTone', 'eventI_1stSideTry', 'eventI_reward')


