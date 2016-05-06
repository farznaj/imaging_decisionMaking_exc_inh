function [traces_al_sm, time_aligned_stimOn, eventI_stimOn] = alignTraces_prePost_filt(traces, traceTimeVec, alignedEvent, frameLength, dofilter, timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, nPreFrames, nPostFrames)
% calls functions for aligning traces on a particular trial event, and does
% gaussian filtering if desired. (Pre_post method is used for alignment, as
% opposed to nan method.)
%
% example inputs:
% traces = alldataSpikesGood; % alldataSpikesGood; %  % traces to be aligned.
% traceTimeVec = {alldata.frameTimes}; % time vector of the trace that you want to realign.
% alignedEvent = 'stimOn'; % align the traces on stim onset. % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward'
% dofilter = true;


%%
defaultPrePostFrames = 2; % default value for nPre and postFrames, if their computed values < 1.
shiftTime = 0; % event of interest will be centered on time 0 which corresponds to interval [-frameLength/2  +frameLength/2]
scaleTime = frameLength;


%% align the traces on stim onset.
trs2rmv = []; 
flag_traces = 1;
[traces_aligned_fut_stimOn, time_aligned_stimOn, eventI_stimOn] = alignTraces_prePost_allCases...
    (alignedEvent, traces, traceTimeVec, frameLength, defaultPrePostFrames, shiftTime, scaleTime, timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, trs2rmv, flag_traces, nPreFrames, nPostFrames);

traces_al = traces_aligned_fut_stimOn;
fprintf('Size of aligned trces: %d  %d  %d (fr x unit x tr)\n', size(traces_al))


%% set the filter to smooth the traces 
if dofilter
    siz = 5;
    sig = 1;
    x = -floor(siz/2) : floor(siz/2);
    H = exp(-(x.^2/ (2 * sig^2)));
    H = H/sum(H);
end
% figure; plot(H)


%% Smooth the traces with a gaussian kernel
if dofilter
    traces_al_sm = NaN(size(traces_al));
    for in = 1:size(traces_al, 2)
        for itr = 1:size(traces_al,3)
            traceNow = traces_al(:,in,itr);
            traces_al_sm(:,in,itr) = conv(traceNow, H', 'same');
        end
    end
else
    traces_al_sm = traces_al;
end

