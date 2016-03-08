function frame0s = findFrame0(framet, eventTime)

frame0s = NaN(size(eventTime));
for tr = 1:length(frame0s)
    [~, frame0s(tr)] = min(abs(eventTime(tr) - framet{tr}));
end
frame0s(isnan(eventTime)) = NaN;

