function [inhibitRois, roi2surr_sig, sigTh_IE, x_all, cost_all, th_pix] = inhibit_excit_setVars(imfilename, pnevFileName, manThSet, assessClass_unsure_inh_excit, keyEval, identifInh, do2dGauss)
% identify inhibitory neurons (only on good neurons (not badROIs))

% sigTh = 1.2;
if ~exist('assessClass_unsure_inh_excit', 'var')
    assessClass_unsure_inh_excit = false(1,3); % set to true so u can evaluate if sigTh is doing a good job.
end
save_common_slope = 1; % if 1, results will be appended to pnevFile


%%
[pd,pnev_n] = fileparts(pnevFileName);
moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));

load(moreName, 'mask', 'CC', 'badROIs01')
a = load(moreName, 'val_EP_AG_size_tau_tempCorr_hiLight_hiLightDB'); 
if isempty(fieldnames(a))
    val_EP_AG_size_tau_tempCorr_hiLight_hiLightDB = [];
else
    val_EP_AG_size_tau_tempCorr_hiLight_hiLightDB = a.val_EP_AG_size_tau_tempCorr_hiLight_hiLightDB(~badROIs01,:);
end

load(imfilename, 'imHeight', 'imWidth', 'sdImage', 'aveImage')
load(pnevFileName, 'A') % pnevFileName should contain Eft results after merging-again has been performed.

A = A(:, ~badROIs01);

COMs = fastCOMsA(A, [imHeight, imWidth]); % size(medImage{2})

% Remove bad components
mask = mask(:,:,~badROIs01);
CC = CC(~badROIs01);
% COMs = COMs(~badROIs01,:);
%

%{
load(imfilename, 'pmtOffFrames')
smoothPts = 6; minPts = 7000; %800;
activity_man_eftMask_ch1 = konnerthDeltaFOverF(activity_man_eftMask_ch1, pmtOffFrames{1}, smoothPts, minPts);
activity_man_eftMask_ch2 = konnerthDeltaFOverF(activity_man_eftMask_ch2, pmtOffFrames{2}, smoothPts, minPts);
%}

%{
fprintf('Setting mask for the gcamp channel....\n')
if showResults
    im = []; % normImage(sdImage{2}); % medImage{2};
    contour_threshold = .95;
    [CC, ~, COMs, mask] = setCC_cleanCC_plotCC_setMask(A, imHeight, imWidth, contour_threshold, im);
    % COMs = fastCOMsA(A, [imHeight, imWidth]); % size(medImage{2})
    % CC = ROIContoursPnevCC(A, imHeight, imWidth, contour_threshold);
    % mask = maskSet(A, imHeight, imWidth);
else
%     im = [];
    mask = maskSet(A, imHeight, imWidth);
    COMs = fastCOMsA(A, [imHeight, imWidth]); % size(medImage{2})
    CC = [];
end
%}



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Green channel to red channel bleed-through correction
% Model: red_ch1 = offset + slope * green_ch2

% If you don't want correction of bleedthrough:
% inhibitImage = medImage{1}; % aveImage{1};


%% Load or compute slope and offsets (model: red_ch1 = offset + slope * green_ch2)

% [d,pnev_n] = fileparts(pnevFileName);
% finame = sprintf('inhibitROIs_%s', pnev_n); % save inhibit vars under this name.
% cd(d)

a = matfile(moreName);

if isprop(a, 'slope_common_pix') %'slope_common') % exist([finame, '.mat'], 'file')
    
    fprintf('Loading slope_common and offsets_ch1...\n')
    load(moreName, 'slope_common_pix', 'offsets_ch1_pix')
%     load(moreName, 'slope_common', 'offsets_ch1')    
    
else
    fprintf('Setting slope_common and offsets_ch1...\n')
    
    %{
    load(pnevFileName, 'activity_man_eftMask_ch1', 'activity_man_eftMask_ch2')
    load(imfilename, 'pmtOffFrames')
    activity_man_eftMask_ch1 = activity_man_eftMask_ch1(~pmtOffFrames{1}, ~badROIs01);
    activity_man_eftMask_ch2 = activity_man_eftMask_ch2(~pmtOffFrames{2}, ~badROIs01);
    
    [slope_common, offsets_ch1] = inhibit_remove_bleedthrough(activity_man_eftMask_ch1, activity_man_eftMask_ch2);
    %}
    
    % New method: to find bleedthrough fraction use pixels (averaged over
    % frames) of the 2 channel, instead of activity traces (ie averaged
    % pixels in each frame).
    [ch1Pixs, ch2Pixs] = inhibit_remove_bleedthrough_setVars(A, imHeight, imWidth, aveImage);
    
%     inhibit_remove_bleedthrough
    [slope_common, offsets_ch1] = inhibit_remove_bleedthrough(ch1Pixs, ch2Pixs);
    slope_common_pix = slope_common;
    offsets_ch1_pix = offsets_ch1;
    
    % Save the slope and the offset
    if save_common_slope
        fprintf('Saving slope_common and offsets_ch1...\n')
        save(moreName, '-append', 'slope_common_pix', 'offsets_ch1_pix') % save to more_pnevFile...        
%         save(moreName, '-append', 'slope_common', 'offsets_ch1') % save to more_pnevFile...
        %         save(pnevFileName, '-append', 'slope_common', 'offsets_ch1')
    end    
end
slope_common = slope_common_pix;


%% Create the bleedthrough-corrected image; model: red_ch1 = offset + slope * green_ch2

origImage = aveImage; % medImage;

inhibitImage = origImage{1} - slope_common*origImage{2};
% inhibitImage = origImage{1}; % use this if you dont want to correct for bleedthrough

a = matfile(moreName);
if ~isprop(a, 'inhibitImageCorrcted_pix') %'inhibitImageCorrcted')
    inhibitImageCorrcted_pix = inhibitImage;
    save(moreName, '-append', 'inhibitImageCorrcted_pix')
%     save(moreName, '-append', 'inhibitImageCorrcted')   
end


%% Does the correction make the actCh1 channel become constant!

load(pnevFileName, 'activity_man_eftMask_ch1', 'activity_man_eftMask_ch2')
load(imfilename, 'pmtOffFrames')
activity_man_eftMask_ch1 = activity_man_eftMask_ch1(~pmtOffFrames{1}, ~badROIs01);
activity_man_eftMask_ch2 = activity_man_eftMask_ch2(~pmtOffFrames{2}, ~badROIs01);

actCh1Corr = activity_man_eftMask_ch1 - slope_common*activity_man_eftMask_ch2;

figure; 
subplot(211),plot(mean(activity_man_eftMask_ch1,2)); a1=gca; title('ch1 activity')
subplot(212),plot(mean(actCh1Corr,2)); a2=gca; title('ch1 activity corrected using slopeCommon')
linkaxes([a1,a2],'x')



%% Show images : ch1, ch2, ch1 - commonSlope*ch2

normims = 0;
warning('off', 'MATLAB:nargchk:deprecated')
ax1 = [];
figure('name', sprintf('Average images. Model: red = offset + slope * green, commonSlope = %.2f', slope_common), 'position', [680   270   843   706]);
% figure('name', sprintf('model: red = offset + slope * green, commonSlope = %.2f, cost = %.2f', slope_common, cost_common));
ha = tight_subplot(2,2,[.05],[.05],[.05]);

if normims,	im = normImage(origImage{1}); else im = origImage{1}; end
axes(ha(1)); imagesc(im), freezeColors, colorbar, title('ch1')
ax1 = [ax1, gca];
hold on, plotCOMsCC(COMs)

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
hold on, plotCOMsCC(COMs)

%
% medImageInhibit = workingImage; % medImageInhibit(medImageInhibit < 0) = 0;
% if normims, im = normImage(medImageInhibit); else im = medImageInhibit; end
im = origImage{1} - inhibitImage;
axes(ha(4)); imagesc(im), freezeColors, colorbar, title('ch1 - corrected image')
ax1 = [ax1, gca];
hold on, plotCOMsCC(COMs)
% hold on, plotCOMsCC(COMs)
%}
linkaxes(ax1)

[pd,pnev_n] = fileparts(pnevFileName);
figdir = fullfile(pd, 'figs');
if ~exist(figdir, 'dir')
    mkdir(figdir) % save the following 3 figures in a folder named "figs"
end

savefig(fullfile(figdir, 'inhIdentCorrImg')) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Identify inhibitory neurons.

if identifInh
    %{
% load(imfilename, 'medImage'), workingImage = medImage;
load(imfilename, 'aveImage'), workingImage = aveImage;
% load(imfilename, 'quantImage'), im2 = quantImage{1};
    %}
    % quantTh = .8; % .5; % .1; % threshold for finding inhibit neurons will be sigTh = quantile(roi2surr_sig, quantTh);
    % keyEval = 0; % if 1, you will use key presses to evaluate ROIs. % Linux hangs with getKey... so make sure this is set to 0! % if 0 you will simply go though ROIs one by one, otherwise it will go to getKey and you will be able to change neural classification.
    ch2Image = sdImage{2};
    
    % if showResults
    load(pnevFileName, 'C')
    C = C(~badROIs01,:);
    % else
    %     C = [];
    % end
    
    % sigTh_IE: threshold value on roi2surr_sig for identifying inhibit and excit neurons: 1st element for inhibit, 2nd element for excit.
    

    %%
    fprintf('Identifying inhibitory neurons....\n')    
    [inhibitRois, roi2surr_sig, sigTh_IE, x_all, cost_all, th_pix] = inhibitROIselection(mask, inhibitImage, manThSet, assessClass_unsure_inh_excit, keyEval, CC, ch2Image, COMs, C, A, do2dGauss, val_EP_AG_size_tau_tempCorr_hiLight_hiLightDB, figdir); % an array of length all neurons, with 1s for inhibit. and 0s for excit. neurons
    
    
    %% Show the results
    
    % if showResults
    
    % plot inhibitory and excitatory ROIs on the image of inhibit channel.
    
%     im2p = normImage(sdImage{1}); % original inh
%     fn = 'Normalized sdImage of inhibit channel';
    im2p = normImage(inhibitImage); % corrected inh
    fn = 'Normalized bCorrected image of inhibit channel';
    
    colors = hot(2*length(CC));
    colors = colors(end:-1:1,:);
    
    % plot inhibitory ROIs on the image of inhibit channel.
    froi = figure('name', fn, 'position', [288          75        1083         898]);
    subplot(221)
    imagesc(im2p)
    colormap gray
    hold on
    for rr = find(inhibitRois==1)
        plot(CC{rr}(2,:), CC{rr}(1,:), 'color', colors(rr, :))
    end
    title(sprintf('gcamp ROIs idenetified as inhibitory (n=%d)', sum(inhibitRois==1)));
    
    % plot excitatory ROIs on the image of inhibit channel.
    subplot(222)
    imagesc(im2p)
    colormap gray
    hold on
    for rr = find(inhibitRois==0)
        plot(CC{rr}(2,:), CC{rr}(1,:), 'color', colors(rr, :))
    end
    title(sprintf('gcamp ROIs idenetified as excitatory (n=%d)', sum(inhibitRois==0)));
    

    % plot unsure ROIs on the image of inhibit channel.
    subplot(223)
    imagesc(im2p)
    colormap gray
    hold on
    for rr = find(isnan(inhibitRois))
        plot(CC{rr}(2,:), CC{rr}(1,:), 'color', colors(rr, :))
    end
    title(sprintf('gcamp ROIs idenetified as unsure (n=%d)', sum(isnan(inhibitRois))));    

    savefig(froi, fullfile(figdir, 'inhIdentROIs')) 
    
   
    %% Compare spike rates
    
    load(pnevFileName, 'S')
    S = S(~badROIs01,:);
    top = S;
    meanS_uns_inh_exc = [mean(mean(top(isnan(inhibitRois),:))), mean(mean(top(inhibitRois==1,:))), mean(mean(top(inhibitRois==0,:)))]  
    
    
    %% Compare average C and S of inhibit and excit neurons
    
    ftra = figure('position', [675          77        1019         899]);
    top = C;
    subplot(421), hold on;
    plot(mean(top(isnan(inhibitRois),:)),'c'); plot(mean(top(inhibitRois==1,:)),'r'); plot(mean(top(inhibitRois==0,:)),'color',[0,127,0]/255); legend('uns', 'inh', 'exc'); xlim([1,length(top)]); title('C')
    subplot(423), plot(mean(top(inhibitRois==1,:))), title('inhibit'); xlim([1,length(top)])
    subplot(425),  plot(mean(top(inhibitRois==0,:))), title('excit'); xlim([1,length(top)])
    subplot(427),  plot(mean(top(isnan(inhibitRois),:))), title('unsure'); xlim([1,length(top)])
    

    top = S;
    subplot(422), hold on;
    plot(mean(top(isnan(inhibitRois),:)),'c'); plot(mean(top(inhibitRois==1,:)),'r'); plot(mean(top(inhibitRois==0,:)),'color',[0,127,0]/255); legend('uns', 'inh', 'exc'); xlim([1,length(top)]); title(sprintf('S\nmeanS: uns %.1f; inh %.1f; exc %.1f', meanS_uns_inh_exc))
    subplot(424), plot(mean(top(inhibitRois==1,:))), title('inhibit'); xlim([1,length(top)])
    subplot(426),  plot(mean(top(inhibitRois==0,:))), title('excit'); xlim([1,length(top)])
    subplot(428),  plot(mean(top(isnan(inhibitRois),:))), title('unsure'); xlim([1,length(top)])

    savefig(ftra, fullfile(figdir, 'inhIdentAveTraces'))
    %{
    subplot(221), plot(mean(activity_man_eftMask_ch2(:, inhibitRois==1), 2)), title('ch2, inhibit')
    subplot(223),  plot(mean(activity_man_eftMask_ch2(:, inhibitRois==0), 2)), title('ch2, excit')
    subplot(222), plot(mean(activity_man_eftMask_ch1(:, inhibitRois==1), 2)), title('ch1, inhibit')
    subplot(224),  plot(mean(activity_man_eftMask_ch1(:, inhibitRois==0), 2)), title('ch1, excit')
    %}    
    % end
    
else
    inhibitRois= []; roi2surr_sig = []; sigTh_IE = [];
    
end


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



