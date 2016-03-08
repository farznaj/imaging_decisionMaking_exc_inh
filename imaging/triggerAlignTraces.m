function [traceEventAlign, timeEventAlign] = triggerAlignTraces(traceOriginal, frame0s, shiftTime, scaleTime)
% INPUTS
% frame0s -- the indeces of the events (triggeres) on which the traces will be aligned.
% traceOriginal -- if an array, dimensions must be frames x units x trials.
% if a cell, dimensions of each cell must be frames x units.

% OUTPUTS
% traceEventAlig -- frames x units x trials. 
% timeEventAlign -- 1 x frames. 0 indicates the time of the event.


if isempty(shiftTime)
    shiftTime = 0;
end
if isempty(scaleTime)
    scaleTime = 1;
end


if iscell(traceOriginal)    
    framesPerTrial = cellfun(@(x)size(x,1), traceOriginal);
    numUnits = size(traceOriginal{1},2);
    numTrials = length(traceOriginal);
else
    framesPerTrial = size(traceOriginal, 1);
    numUnits = size(traceOriginal, 2);
    numTrials = size(traceOriginal, 3);
end

maxNumFramesPerTrial = max(framesPerTrial);

%% set the trig-aligned dFOF trace
% frames x neurons x trials
traceEventAlign = NaN(max(frame0s) + maxNumFramesPerTrial, numUnits, numTrials);
m = max(frame0s);

if iscell(traceOriginal)
    for tr = 1:numTrials
        if ~isnan(frame0s(tr))
            offset = m - frame0s(tr); % add this to each index of frameTimes to have all trials aligned on frame0.
            traceEventAlign(offset+(1:framesPerTrial(tr)), :, tr) = traceOriginal{tr};
        end
    end
else
    for tr = 1:numTrials
        if ~isnan(frame0s(tr))
            offset = m - frame0s(tr); % add this to each index of frameTimes to have all trials aligned on frame0.
            traceEventAlign(offset+(1:framesPerTrial(tr)), :, tr) = traceOriginal(:,:,tr);
        end
    end    
end


%% set the trig-aligned time trace
% frame0 (ie the frame to align trials on) will be 0+frameLength/2.
timeEventAlign = scaleTime * ((1:size(traceEventAlign, 1)) - m) + shiftTime;


