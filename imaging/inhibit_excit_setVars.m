function [inhibitRois, roi2surr_sig, sigTh] = inhibit_excit_setVars(imfilename, pnevFileName, quantTh, showResults, keyEval)
% identify inhibitory neurons.

% sigTh = 1.2;
if ~exist('showResults', 'var')
    showResults = false; % set to true so u can evaluate if sigTh is doing a good job.
end


%% Set vars to identify inhibitory neurons: mask and ROI contours for the gcamp channel.

% [imfilename, pnevFileName] = setImagingAnalysisNames(mousename, imagingFolder, mdfFileNumber, signalCh);

load(imfilename, 'imHeight', 'imWidth', 'sdImage')
load(pnevFileName, 'A') % pnevFileName should contain Eft results after merging-again has been performed.
spatialComp = A; 
clear A

fprintf('Setting mask for the gcamp channel....\n')
if showResults
    im = []; % normImage(sdImage{2}); % medImage{2};
    contour_threshold = .95;    
    [CC, ~, COMs, mask] = setCC_cleanCC_plotCC_setMask(spatialComp, imHeight, imWidth, contour_threshold, im);        
    % COMs = fastCOMsA(A, [imHeight, imWidth]); % size(medImage{2})
    % CC = ROIContoursPnevCC(spatialComp, imHeight, imWidth, contour_threshold);
    % mask = maskSet(spatialComp, imHeight, imWidth);
else
%     im = [];
    mask = maskSet(spatialComp, imHeight, imWidth);    
    COMs = fastCOMsA(spatialComp, [imHeight, imWidth]); % size(medImage{2})
    CC = [];
end




%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Green channel to red channel bleed-through correction
% Model: red_ch1 = offset + slope * green_ch2

% If you don't want correction of bleedthrough:
% inhibitImage = medImage{1}; % aveImage{1};


%% Load vars 

save_common_slope = 1; % if 1, results will be appended to pnevFile

load(pnevFileName, 'activity_man_eftMask_ch1', 'activity_man_eftMask_ch2')
load(imfilename, 'aveImage')

%{
load(imfilename, 'pmtOffFrames')
smoothPts = 6; minPts = 7000; %800;
activity_man_eftMask_ch1 = konnerthDeltaFOverF(activity_man_eftMask_ch1, pmtOffFrames{1}, smoothPts, minPts);
activity_man_eftMask_ch2 = konnerthDeltaFOverF(activity_man_eftMask_ch2, pmtOffFrames{2}, smoothPts, minPts);
%}
    

%% Compute slope and offsets (model: red_ch1 = offset + slope * green_ch2)

[d,pnev_n] = fileparts(pnevFileName);
finame = sprintf('inhibitROIs_%s', pnev_n); % save inhibit vars under this name.
        
% a = matfile(pnevFileName);
cd(d)
if exist([finame, '.mat'], 'file') % isprop(a, 'slope_common')
    fprintf('Loading slope_common and offsets_ch1...\n')
    load(finame, 'slope_common', 'offsets_ch1')
    
else
    inhibit_remove_bleedthrough

    % Save the slope and the offset    
    if save_common_slope
        fprintf('Saving slope_common and offsets_ch1...\n')
        
        save(finame, 'slope_common', 'offsets_ch1')
%         save(pnevFileName, '-append', 'slope_common', 'offsets_ch1')
    end

end


%% Create the bleedthrough-corrected image; model: red_ch1 = offset + slope * green_ch2

origImage = aveImage; % medImage;

inhibitImage = origImage{1} - slope_common*origImage{2};


%% Show images : ch1, ch2, ch1 - commonSlope*ch2

normims = 0;
warning('off', 'MATLAB:nargchk:deprecated')
ax1 = [];
figure('name', sprintf('model: red = offset + slope * green, commonSlope = %.2f', slope_common));
% figure('name', sprintf('model: red = offset + slope * green, commonSlope = %.2f, cost = %.2f', slope_common, cost_common));
ha = tight_subplot(2,2,[.05],[.05],[.05]);

if normims,	im = normImage(origImage{1}); else im = origImage{1}; end
axes(ha(1)); imagesc(im), freezeColors, colorbar, title('ch1')
ax1 = [ax1, gca];

if normims,	im = normImage(origImage{2}); else im = origImage{2}; end
axes(ha(2)); imagesc(im), freezeColors, colorbar, title('ch2')
ax1 = [ax1, gca];
hold on, plotCOMsCC(COMs)

% md2 is what I will use to identify inhibit neurons... this is supposedly
% the image that is free of the effect of bleedthrough.
% md2 = medImage{1} - mean(slope)*medImage{2};
if normims, im = normImage(inhibitImage); else im = inhibitImage; end
axes(ha(3)); imagesc(im), freezeColors, colorbar, title('corrected image: ch1 - commonSlope*ch2')
ax1 = [ax1, gca];

%
% medImageInhibit = workingImage; % medImageInhibit(medImageInhibit < 0) = 0;
% if normims, im = normImage(medImageInhibit); else im = medImageInhibit; end
im = origImage{1} - inhibitImage;
axes(ha(4)); imagesc(im), freezeColors, colorbar, title('ch1 - corrected image')
ax1 = [ax1, gca];
% hold on, plotCOMsCC(COMs)
%}
linkaxes(ax1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Identify inhibitory neurons.

%{
% load(imfilename, 'medImage'), workingImage = medImage;
load(imfilename, 'aveImage'), workingImage = aveImage;
% load(imfilename, 'quantImage'), im2 = quantImage{1};
%}
% quantTh = .8; % .5; % .1; % threshold for finding inhibit neurons will be sigTh = quantile(roi2surr_sig, quantTh);
% keyEval = 0; % if 1, you will use key presses to evaluate ROIs. % Linux hangs with getKey... so make sure this is set to 0! % if 0 you will simply go though ROIs one by one, otherwise it will go to getKey and you will be able to change neural classification.    
ch2Image = sdImage{2};

[inhibitRois, roi2surr_sig, sigTh] = inhibitROIselection(mask, inhibitImage, quantTh, showResults, keyEval, CC, ch2Image, COMs, activity_man_eftMask_ch1, activity_man_eftMask_ch2); % an array of length all neurons, with 1s for inhibit. and 0s for excit. neurons


%% Show the results

if showResults
    
    im2p = normImage(sdImage{1});
    colors = hot(2*size(spatialComp,2));
    colors = colors(end:-1:1,:);

    % plot inhibitory ROIs on the image of inhibit channel.
    figure('name', 'sdImage of inhibit channel', 'position', [288     1   560   972]); 
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

end

%%%%
figure; 
subplot(221), plot(mean(activity_man_eftMask_ch2(:, inhibitRois==1), 2)), title('ch2, inhibit')
subplot(223),  plot(mean(activity_man_eftMask_ch2(:, inhibitRois==0), 2)), title('ch2, excit')
subplot(222), plot(mean(activity_man_eftMask_ch1(:, inhibitRois==1), 2)), title('ch1, inhibit')
subplot(224),  plot(mean(activity_man_eftMask_ch1(:, inhibitRois==0), 2)), title('ch1, excit')
%{
subplot(222), plot(mean(C(inhibitRois==1,:)))
subplot(224),  plot(mean(C(inhibitRois==0,:)))
%}



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



