function frame0s = findFrame0(framet, eventTime)
% find the frame during which the eventTime happened.

frame0s = NaN(size(eventTime));
for tr = 1:length(frame0s)
    [~, frame0s(tr)] = min(abs(eventTime(tr) - framet{tr}));
end
frame0s(isnan(eventTime)) = NaN;

