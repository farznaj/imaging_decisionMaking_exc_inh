function y2 = shiftScaleY(y)

% yshift = y - mean(y(1:3));
% yshift = y - min(y);
s = sort(y);
yshift = y - mean(s(1:ceil(length(y)*.2)));
y2 = yshift / max(yshift);

