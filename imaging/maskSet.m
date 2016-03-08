function mask = maskSet(CC, imHeight, imWidth)
% mask = maskSet(CC, imHeight, imWidth)
%
% Set ROI masks for contours with coordinates CC
%
% CC: cell array, each element contains the [y x] coordinates of an ROI
% contour.
% mask(:,:,i) shows the spatial location of ROI i in
% the field of view.

[X,Y] = meshgrid(1:imWidth, 1:imHeight);

mask = false(imHeight, imWidth, length(CC));

for rr = 1:length(CC)
    x = CC{rr}(2,:);
    y = CC{rr}(1,:);
    mask(:, :, rr) = inpolygon(X, Y, x, y);
end

