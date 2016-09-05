function plotCompTpSp(i, trace, A, im, CC, imHeight, imWidth)
%{
% if no CC:
% CC = ROIContoursPnevCC(A, imHeight, imWidth, .95);
im = sdImage{2};
i = 288;
plotCompTpSp(i, C_df, A, im, CC, imHeight, imWidth)

% COMs = fastCOMsA(A, [imHeight, imWidth]);
% nearbyROIs = findNearbyROIs(COMs, COMs(i,:), 5)
%}

figure;

subplot(2,2,[1,2])
plot(trace(i,:))

subplot(223)
imagesc(reshape(A(:,i), imHeight, imWidth))

subplot(224)
imagesc(im)
hold on
plot(CC{i}(2,:), CC{i}(1,:), 'r')