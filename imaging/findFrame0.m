function frame0s = findFrame0(traceTimeVec, eventTime)
% find the frame during which the eventTime happened. remember you are not
% simply using frame = ceil(eventTime/frameLength), because eventTimes are
% relative to scopeTTL and not imaging onset.... in other words you need to
% correct for mscanDelay.
% (frameTimes is also relative to scopeTTL.)
%
%{
% Check your computation
% binfun = @(t)(t==0)+ceil(t/frameLength)
for itr = 1:length(traceTimeVec)
    del = traceTimeVec{itr}(1)-frameLength/2;
    % the following 2 should be identical
    if ~isnan(timeStimOnset(itr)) && binfun(timeStimOnset(itr) - del) ~= eventInds_f(itr)
        itr, binfun(timeStimOnset(itr) - del), eventInds_f(itr)
        error('the 2 measures should be identical. what is wrong?')       
    end
end
%}


frame0s = NaN(size(eventTime));
for tr = 1:length(frame0s)
    [~, frame0s(tr)] = min(abs(eventTime(tr) - traceTimeVec{tr})); % since traceTime includes the center time of frames, eventTime is within [-frameLength/2  +frameLength/2] of frame0  % eventTime is closest in time to frame0's center time (ie time at the middle of the frame)
end
frame0s(isnan(eventTime)) = NaN;

