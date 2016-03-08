function [inhibitRois, good_inhibit, good_excit] = inhibit_excit_setVars(imfilename, pnevFileName, sigTh, goodinds, showResults)
% identify inhibitory neurons.

% sigTh = 1.2;
if ~exist('showResults', 'var')
    showResults = false; % set to true so u can evaluate if sigTh is doing a good job.
end


%% set vars to identify inhibitory neurons.

% [imfilename, pnevFileName] = setImagingAnalysisNames(mousename, imagingFolder, mdfFileNumber, signalCh);

load(imfilename, 'imHeight', 'imWidth', 'medImage')
load(pnevFileName, 'A') % pnevFileName should contain Eft results after merging-again has been performed.
spatialComp = A; 
clear A

im = medImage{2};
if ~showResults
    im = [];
end

contour_threshold = .95;
[CC, ~, ~, mask] = setCC_cleanCC_plotCC_setMask(spatialComp, imHeight, imWidth, contour_threshold, im);
% size(CC)
% size(mask)


%% identify inhibitory neurons.

im2 = medImage{1};
inhibitRois = inhibitROIselection(mask, im2, sigTh, CC, showResults); % an array of length all neurons, with 1s for inhibit. and 0s for excit. neurons


%% set good_inhibit and good_excit neurons. (run avetrialAlign_setVars to get goodinds and traces.)

% goodinds: an array of length of all neurons, with 1s indicating good and 0s bad neurons.
good_inhibit = (inhibitRois(goodinds)); % an array of length of good neurons, with 1s for inhibit. and 0s for excit. neurons.
good_excit = (~inhibitRois(goodinds)); % an array of length of good neurons, with 1s for excit. and 0s for inhibit neurons.


fprintf('Fract inhibit in all, good & bad Ns = %.3f  %.3f  %.3f\n', [...
    nanmean(inhibitRois)
    nanmean(inhibitRois(goodinds))
    nanmean(inhibitRois(~goodinds))])
% it seems good neurons are biased to include more excit neurons bc for
% some reason tdtomato neruons have low quality on the red channel.




