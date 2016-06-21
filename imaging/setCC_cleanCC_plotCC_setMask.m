function [CC, CR, COMs, mask_eft] = setCC_cleanCC_plotCC_setMask(spatialComp, imHeight, imWidth, contour_threshold, im)
% [CC, CCorig] = setPnevContours_cleanCC_plotCC(spatialComp, imHeight, imWidth, contour_threshold, im)
%
% im = sdImage{2};


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% CONTOURS %%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set ROI contours of the Spatial components found by Eftychios's algorithm.
% contour_threshold = .95;

% display_numbers = 0;
% figure;
% [CC, jsf] = plot_contours(spatialComp, reshape(P.sn, imHeight, imWidth), contour_threshold, display_numbers);% jsf has [y x] pairs. CC has [x y] pairs.
% CC_swap = CC:

[CC, CR, COMs] = ROIContoursPnev(spatialComp, imHeight, imWidth, contour_threshold); % CC_swap, CR, COMs are all [y x] pairs.

% plotPnevROIs(im, CC_swap, colors);
CCorig = CC;


%% In CC set the column of metadata to NaN, so plotting contours would be easy.
CC = ROIContoursPnev_cleanCC(CCorig);


%% Plot ROIs found by Eftychios's algorithm on the sdImage
% im = reshape(P.sn, imHeight, imWidth);
% im = sdImage{2};
if exist('im', 'var') && ~isempty(im)
    colors = hot(2*size(spatialComp,2));
    colors = colors(end:-1:1,:);
    
    figure;
%     imagesc(im);
    imagesc(log(im));
    hold on;
    colormap gray
    
    for rr = 1:length(CC)
        plot(CC{rr}(2,:), CC{rr}(1,:), 'color', colors(rr, :))
%         pause
    end
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% MASKS %%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set the ROI masks for Eftichios's ROIs
if nargout==4
    mask_eft = maskSet(CC, imHeight, imWidth);
    
    if (sum(mask_eft(:)>1))
        fprintf('mask has values >1... weird... check it!!\n')
    end
end



