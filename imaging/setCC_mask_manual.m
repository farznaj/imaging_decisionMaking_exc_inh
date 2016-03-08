function [CC_rois, mask_manual] = setCC_mask_manual(rois, im)
% [CC_rois, mask_manual] = setCC_mask_manual(rois, im);
% 
% set CC_rois (in the same format as Eftychios's CC) and mask for the
% manual method.



%% Set the coordinates for the manual method in [y x] format for each ROI. (similar to CC of Eftychios method.)
CC_rois = cell(1, length(rois));
for rr = 1:length(rois)
    CC_rois{rr} = [rois{rr}.mnCoordinates(:, 2)'; rois{rr}.mnCoordinates(:, 1)'];
end


%% Plot ROI contours on the image
if exist('im', 'var')
    colors = hot(2*size(spatialComp,2));
    colors = colors(end:-1:1,:);
    figure;
    imagesc(im);
    hold on;
    % in rois, 1st column is x and 2nd column is y.
    for rr = 1:length(rois)
        %     plot(rois{rr}.mnCoordinates(:,1), rois{rr}.mnCoordinates(:,2), '-', 'color', colors(rr,:))
        plot(CC_rois{rr}(2,:), CC_rois{rr}(1,:), 'color', colors(rr, :))
    end
end


%% Set the ROI masks for manually found ROIs
mask_manual = maskSet(CC_rois, imHeight, imWidth);
if (sum(mask_manual(:)>1))
    fprintf('mask has values >1... weird... check it!!\n')
end


