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
    [~, frame0s(tr)] = min(abs(eventTime(tr) - traceTimeVec{tr}));
end
frame0s(isnan(eventTime)) = NaN;

