function [inhibitRois, roi2surr_sig, sigTh] = inhibit_excit_setVars(imfilename, pnevFileName, sigTh, showResults)
% identify inhibitory neurons.

% sigTh = 1.2;
if ~exist('showResults', 'var')
    showResults = false; % set to true so u can evaluate if sigTh is doing a good job.
end


%% set vars to identify inhibitory neurons: get ROI contours for the gcamp channel.

% [imfilename, pnevFileName] = setImagingAnalysisNames(mousename, imagingFolder, mdfFileNumber, signalCh);

load(imfilename, 'imHeight', 'imWidth', 'sdImage')
load(pnevFileName, 'A') % pnevFileName should contain Eft results after merging-again has been performed.
spatialComp = A; 
clear A

if showResults
%     im = medImage{2};
    im = sdImage{2};
else
    im = [];
end

contour_threshold = .95;
fprintf('Setting the mask for the gcamp channel....\n')
[CC, ~, COMs, mask] = setCC_cleanCC_plotCC_setMask(spatialComp, imHeight, imWidth, contour_threshold, im);
title('ROIs shown on the sdImage of channel 2')
% size(CC), % size(mask)


%% Load manual activity computed for ch1 and ch2, and compute their DF/F

load(pnevFileName, 'activity_man_eftMask_ch1', 'activity_man_eftMask_ch2')
load(imfilename, 'pmtOffFrames')

smoothPts = 6; minPts = 7000; %800;
activity_man_eftMask_ch1 = konnerthDeltaFOverF(activity_man_eftMask_ch1, pmtOffFrames{1}, smoothPts, minPts);
activity_man_eftMask_ch2 = konnerthDeltaFOverF(activity_man_eftMask_ch2, pmtOffFrames{2}, smoothPts, minPts);

    
%% identify inhibitory neurons.

load(imfilename, 'medImage'), im2 = medImage{1};
% load(imfilename, 'quantImage'), im2 = quantImage{1};

[inhibitRois, roi2surr_sig, sigTh] = inhibitROIselection(mask, medImage, sigTh, CC, showResults, gcf, COMs, activity_man_eftMask_ch1, activity_man_eftMask_ch2); % an array of length all neurons, with 1s for inhibit. and 0s for excit. neurons


% Show the results:
im2p = sdImage{1};
colors = hot(2*size(spatialComp,2));
colors = colors(end:-1:1,:);
% plot inhibitory ROIs on the image of inhibit channel.
figure('name', 'sdImage of inhibit channel'); 
subplot(211)
imagesc(im2p)
colormap gray
hold on
for rr = find(inhibitRois==1)
    plot(CC{rr}(2,:), CC{rr}(1,:), 'color', colors(rr, :))
end
title('gcamp ROIs idenetified as inhibitory');

% plot excitatory ROIs on the image of inhibit channel.
subplot(212)
imagesc(im2p)
colormap gray
hold on
for rr = find(inhibitRois==0)
    plot(CC{rr}(2,:), CC{rr}(1,:), 'color', colors(rr, :))
end
title('gcamp ROIs identified as excitatory');




%% set good_inhibit and good_excit neurons. (run avetrialAlign_setVars to get goodinds and traces.)
%{
% goodinds: an array of length of all neurons, with 1s indicating good and 0s bad neurons.
good_inhibit = inhibitRois(goodinds); % an array of length of good neurons, with 1s for inhibit. and 0s for excit. neurons.
good_excit = ~inhibitRois(goodinds); % an array of length of good neurons, with 1s for excit. and 0s for inhibit neurons.

% you can use the codes below if you want to be safe.
% good_inhibit = inhibitRois(goodinds & roi2surr_sig >= 1.3);
% good_excit = ~inhibitRois(goodinds & roi2surr_sig <= 1.1);

fprintf('Fract inhibit in all, good & bad Ns = %.3f  %.3f  %.3f\n', [...
    nanmean(inhibitRois)
    nanmean(inhibitRois(goodinds))
    nanmean(inhibitRois(~goodinds))])
% it seems good neurons are biased to include more excit neurons bc for
% some reason tdtomato neruons have low quality on the red channel.
%}



