function alignMovies_prePost(mouse, imagingFolder, mdfFileNumber, ch2read)
% Make average aligned movies, ie align movies of each trial on specific
% trial events, then get average across aligned movies.
%
% mousename = 'fni17';
% imagingFolder = '151102';
% mdfFileNumber = 1; % or tif major
% ch2read = 2;

frameLength = 1000/30.9;
shiftTime = frameLength/2; % For consistency with your other figures you changed it back from 0 to frameLength --> % 0; % event of interest will be centered on time 0 which corresponds to interval [-frameLength/2  +frameLength/2]
scaleTime = frameLength;

flag_traces = false; % if 1, the 1st input to triggerAlignTraces_prepost contains the temporal traces for each trial. if 0, it contains the movie for each trial.
nPreFrames = []; % 10; 
nPostFrames = []; % 15;


%% Read tif files

tifList = tifListSet(mouse, imagingFolder, mdfFileNumber, ch2read);

Y = [];
for t = 1:length(tifList)
    fprintf('Reading tif file %s\n', tifList{t})
    Y = cat(3, Y, bigread2(tifList{t}));
end
% if ~isa(Y,'double');    Y = double(Y);  end         % convert to double


%% Set imfilename and load a few vars from it (framesPerTrial, etc).
%{
if isunix
    if isempty(strfind(pwd, 'grid')) %isempty(strfind(pwd, 'sonas')) % Farzaneh's Unix in the office
        dataPath = '~/Shares/Churchland/data';
    else
        dataPath = '/sonas-hs/churchland/nlsas/data/data';        
    end
elseif ispc
    dataPath = '\\sonas-hs.cshl.edu\churchland\data'; % FN
end
tifFold = fullfile(dataPath, mouse, 'imaging', imagingFolder);
date_major = sprintf('%s_%03d', imagingFolder, mdfFileNumber);
imfilename = fullfile(tifFold, date_major);
%}

signalCh = 2; % because you get A from channel 2, I think this should be always 2.
pnev2load = [];
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
[pd, pnev_n] = fileparts(pnevFileName);
% disp(pnev_n)
% cd(fileparts(imfilename))
% moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));
postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));


%%
% load(fullfile(tifFold, date_major), 'framesPerTrial', 'alldata', 'frameLength', 'trs2rmv')
load(imfilename, 'framesPerTrial', 'trialCodeMissing') %, 'alldata_frameTimes'
load(postName, 'timeStimOnset', 'alldata_frameTimes')%, 'outcomes') % 'timeStop', 


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

% trials that miss trialCode signal.
% trs_noTrialCode = find(trialCodeMissing==1);

% trials that were not imaged. (I believe trs_noTrialCode is a subset
% of trs_notScanned).
% trs_notScanned = find([alldata.hasActivity]==0);

%%%%%%%%%%%%%%%%% you need to take care of the following (trialCodeMissing)
%%%%%%%%%%%%%%%%% but for now:
if sum(trialCodeMissing)>0
    error('take care of trialCodeMissing')
end

alignedEvent = timeStimOnset;

%{
% exclude from alignedEvent and alldata_frameTimes trials that were not triggered in mscan. 
alignedEvent = timeStop(trialCodeMissing==0);
alldata_frameTimes = alldata_frameTimes(trialCodeMissing==0);
%}
if length(alignedEvent) ~= size(movie_trials,2)
    error('Length of alignedEvent does not match the size of movie_trials! Perhaps something wrong with excluding non-triggered trials!')
end

traceTimeVec = alldata_frameTimes;
eventInds_f = eventTimeToIdx(alignedEvent, traceTimeVec);


%% Perform the alignment

% traces_aligned_fut : uint16, pix x pix x frames x trials : event-aligned movies for all trials.
[traces_aligned_fut, time_aligned, eventI, nPreFrames, nPostFrames] = triggerAlignTraces_prepost...
    (movie_trials, eventInds_f, nPreFrames, nPostFrames,  shiftTime, scaleTime, flag_traces);

movieAligned_stimOnset = nanmean(traces_aligned_fut, 4); 

%{
outcomes = outcomes(trialCodeMissing==0);
% movieAligned_timeStop : double, pix x pix x frames: average movies of event-aligned movies across all trials
movieAligned_timeStop_outcome_0 = nanmean(traces_aligned_fut(:,:,:,outcomes==0), 4); 
movieAligned_timeStop_outcome_1 = nanmean(traces_aligned_fut(:,:,:,outcomes==1), 4); 
movieAligned_timeStop_outcome_n1 = nanmean(traces_aligned_fut(:,:,:,outcomes==-1), 4); 
% implay(movieAligned_timeStop/max(movieAligned_timeStop(:)))

eventI_timeStop = eventI;
%}

%% Save average movie of event-aligned trial movies (pix x pix x frames)

% save([imfilename, '_movieAligned'], 'movieAligned_timeStop_outcome_0', 'movieAligned_timeStop_outcome_1', 'movieAligned_timeStop_outcome_n1', 'eventI_timeStop')
save([imfilename, '_movieAligned_ch', num2str(ch2read)], 'movieAligned_stimOnset', 'eventI')


% if exist([imfilename, '_movieAligned.mat'], 'file')==2
%     save([imfilename, '_movieAligned'], '-append', 'movieAligned_timeStop', 'eventI_timeStop')
% else
%     save([imfilename, '_movieAligned'], 'movieAligned_timeStop', 'eventI_timeStop')
% end


%% See the movie
%{
movie_play(movieAligned_stimOnset)

%%%
a = movieAligned_stimOnset_ch1;
figure;
for i=1:size(a,3)
    imagesc(a(:,:,i))
    set(gcf, 'name', num2str(i))
    pause
end

%%%
aa = a/max(a(:));
writeTif(aa, 'movieAligned_stimOnset_ch1.TIF', 1)

aa = nan(size(a));
softImMax = quantile(a(:), 0.995);
aa = a/softImMax;
mn = min(a(:));

for i=1:size(a,3)
    im = a(:,:,i);
    im = im - mn;
    im = im / softImMax;
    im(im > 1) = 1; 
    
    aa(:,:,i) = im;
end

writeTif(aa, 'movieAligned_stimOnset_ch1_softmax.TIF', 1)

%}


%% Use below if you want to align on all trial events. Read the note below
% this needs work: event times that you get below have indeces based on
% alldata, ie all trials even the ones non-triggered in mscan. But
% movie_trials only includes movie of trials that were triggered. So you
% need to exclude non-triggered trials from event times before aligning the
% movie traces.

%{
traceTimeVec = {alldata.frameTimes}; % time vector of the trace that you want to realign.
defaultPrePostFrames = 2; % default value for nPre and postFrames, if their computed values < 1.


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
%}
