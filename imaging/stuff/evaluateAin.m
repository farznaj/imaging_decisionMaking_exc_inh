% find index of a particular ROI with given [y,x] as COM: 
% y = 165.1; x = 50.22; x_y = [x, y]

% x_y = ginput;

th = 5;
roiI = [];
for rr = 1:size(x_y,1)
    x = x_y(rr,1);
    y = x_y(rr,2);
    roiI{rr} = find(abs(COMs(:,1)-y) < th & abs(COMs(:,2)-x) < th);
end
roiI


%%
spComp = A; % merging_vars.A; % A;

plotCOMs = 0; %1; 

im = sdImage{2}; % std(Y,[],3);


%%
COMs = fastCOMsA(spComp, [imHeight, imWidth]);

if ~plotCOMs
    CC = ROIContoursPnevCC(spComp, imHeight, imWidth, .95);
%     CC = ROIContoursPnev_cleanCC(CC);
end


%%
figure;
% imagesc(im);
imagesc(log(im));
% imagesc(medImspCompge{2}./(normingMedispCompnImspCompge))

hold on;
% colormap gray
colors = distinguishable_colors(size(spComp,2), 'k');

for rr = 1:size(spComp,2)
    if plotCOMs
        plot(COMs(rr,2), COMs(rr,1), 'r.')
        
    else
        %[CC, ~, COMs] = setCC_cleanCC_plotCC_setMask(spComp, imHeight, imWidth, contour_threshold, im);        
        
        if ~isempty(CC{rr})
            plot(CC{rr}(2,:), CC{rr}(1,:), 'color', colors(rr, :))
        else
            fprintf('Contour of ROI %i is empty!\n', rr)
        end
    end
end