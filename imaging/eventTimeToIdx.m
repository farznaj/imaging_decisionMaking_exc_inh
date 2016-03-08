function eventInds = eventTimeToIdx(eventTime, traceTimeVec)
% eventInds = eventTimeToIdx(eventTime, traceTimeVec)
%
% eventInds: index of eventTime on traceTimeVec array for each trial.

if iscell(eventTime)
    eventTime = cellfun(@(x)x(1), eventTime);
end


% frametimes and eventTime are relatie to bcontrol ttl onset; so we compare them to find on what frame eventTime happened.
eventInds = findFrame0(traceTimeVec, eventTime); 


%%
%{
function [eventBinIdx, traceEventAlign, timeEventAlign] = eventAlign(eventTime, traceTimeVec, traces, shiftTime, scaleTime)
% [eventBinIdx, traceEventAlign, timeEventAlign] = eventAlign(eventTime, traceTimeVec, traces, shiftTime, scaleTime)
%
% This function does 2 things:
% 1. Finds the index of the event on traceTimeVec array for each trial --> eventBinIdx
% 2. Aligns traces on the event --> traceEventAlign, timeEventAlign
% are event aligned trace and times (time 0 will be the event time).

% traceTimeVec = framet;
% frame0s = eventBinIdx;

if iscell(eventTime)
    eventTime = cellfun(@(x)x(1), eventTime);
end

%% Find the bin index of the event (on traceTimeVec array) for each trial
% frametimes and eventTime are relatie to bcontrol ttl onset. so we compare them to find on what frame eventTime happened.
eventBinIdx = findFrame0(traceTimeVec, eventTime); 


%% Align traces on the event.
[traceEventAlign, timeEventAlign] = triggerAlignTraces(traces, eventBinIdx, shiftTime, scaleTime);
%}