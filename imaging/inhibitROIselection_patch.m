% you tried this but then decided to go with inhibitROIselection (instaed
% of using noSpikeTime patches and only corr with A to find inh ROIs)


%% Set mat file names

signalCh = 2; % because you get A from channel 2, I think this should be always 2.
pnev2load = [];
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
[pd, pnev_n] = fileparts(pnevFileName);
% disp(pnev_n)
% cd(fileparts(imfilename))
moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));
inhName = fullfile(pd, sprintf('inhPatch_%s.mat', pnev_n));


%% Load vars
%{
load(inhName, 'pad')
load(imfilename, 'imHeight', 'imWidth', 'sdImage')
load(pnevFileName, 'A', 'highlightPatchAvg', 'highlightCorrROI')
load(moreName, 'badROIs01', 'inhibitImageCorrcted', 'inhibitRois', 'CC')
% load(inhName, 'inhPatch', 'Apatch_th', 'noSpikeTimes', 'pad')

% Apatch_th = Apatch_th(~badROIs01);
% inhPatch = inhPatch(~badROIs01);
% inhPatchMov = inhPatchMov(~badROIs01);
CC = CC(~badROIs01);
A = A(:,~badROIs01);
highlightPatchAvg = highlightPatchAvg(~badROIs01);
highlightCorrROI = highlightCorrROI(~badROIs01);

siz = [imHeight, imWidth];
%}



%%
% inhName = fullfile(pd, sprintf('inhPatch_%s.mat', pnev_n));
load(inhName, 'pad', 'patch_ch1', 'patch_ch2')
patch_ch1 = patch_ch1(~badROIs01);
patch_ch2 = patch_ch2(~badROIs01);

load(pnevFileName, 'highlightPatchAvg', 'highlightCorrROI')
highlightPatchAvg = highlightPatchAvg(~badROIs01);
highlightCorrROI = highlightCorrROI(~badROIs01);

siz = [imHeight, imWidth];


%% Set rawA, ie A including only pixels within thr contour

thr = .8; %.95; %.8; % contour_threshold; % FN: at thr 95% of the elements in sorted A_temp.^2 will be included. See the comment below for why we use A_temp.^2 to find thr (since there are lots of 0s in A_temp)....

d1 = imHeight;
d2 = imWidth;
K = size(A,2);

%     CR = cell(K,2);    % store the information of the cell contour
rawA = zeros(size(A));          % spatial contour for raw signal
for idx=1:K
    A_temp = full(reshape(A(:,idx),d1,d2));
    A_temp = medfilt2(A_temp,[3,3]);
    A_temp = A_temp(:);
    [temp,ind] = sort(A_temp(:).^2,'ascend');
    temp =  cumsum(temp); % FN: The idea is that we find percentile not on a linear scale, instead on cumsum(A_temp.^2) which spans [0 1] however it's highly exponential. As a result, the index at which cumsum(A_temp.^2)=.05 percentile will be very high, hence sorted(A_temp) will be high (and not 0) at this high index. If we were going to use the regular percentile, because it uses a linear scale, the index corresponding to .05 percentile will be very low, hence A_temp will also be very low (almost always 0 regardless of percentile value!). Regular percentile:  pctl = @(v,p) interp1(linspace(0.5/length(v), 1-0.5/length(v), length(v))', sort(v), p*0.01, 'spline')
    ff = find(temp > (1-thr)*temp(end),1,'first'); % temp = cumsum(sorted_A_temp.^2)
    if ~isempty(ff)
        fp = find(A_temp >= A_temp(ind(ff)));
%         [ii,jj] = ind2sub([d1,d2],fp);
        %             CR{idx,1} = [ii,jj]';
        %             CR{idx,2} = A_temp(fp)';
        rawA(fp,idx) = A(fp,idx);
    end
end

%%%%%
% rawA(rawA~=1) = 0; % FN added to only keep values = 1 and set the rest to 0!


%% Set Apatch_th (ie patch of A for each ROI)

Apatch0 = cell(1, size(A,2)); % not-thresholded ... all non-0 values of A are included
Apatch_th = cell(1, size(A,2)); % thresholded (only pixels within thr contour of A are included)
for rr = 1:length(Apatch_th)    
    
    % All non0 A: Find non-0 pixels in A
    Anow = A(:,rr); 
    AMat = reshape(Anow, [imHeight, imWidth]);
    [i, j] = find(AMat);
    xRange = [max(min(j)-pad, 1) min(max(j)+pad, imWidth)];
    yRange = [max(min(i)-pad, 1) min(max(i)+pad, imHeight)];
    %
    % Clip A to only include non-0 values (use all non-0 pixels in A)
    AClip = AMat(yRange(1):yRange(2), xRange(1):xRange(2));     
    % set 0s to nans
    AClip(~AClip) = NaN;    
    Apatch0{rr} = AClip;
    
    
    % Thresholded A: Clip thresholded A to only include non-0 values (use only the good pixels in A(within thr contour))
    Anow = rawA(:,rr); 
    AMat = reshape(Anow, [imHeight, imWidth]);
    AClipTh = AMat(yRange(1):yRange(2), xRange(1):xRange(2));    
    % set 0s to nans
    AClipTh(~AClipTh) = NaN;    
    Apatch_th{rr} = AClipTh;
end

% number of non-0 pixels in Apatch_th; same as sizeROIhiPix
szApatch = cellfun(@(x)sum(~isnan(x(:))), Apatch_th); 
figure; plot(szApatch)


%% Set the mask of patch_ch1, patch_ch2 (ie in each patch set pixels outside ROI to nan; if Apatch_th is computed from the thresholded A, then all pixels outside the 95% contour of ROI will be set to nan for each patch)

patch_ch1m_th = cell(1, length(Apatch_th));
patch_ch2m_th = cell(1, length(Apatch_th));
for rr = 1:length(Apatch_th)    
    
%     AClip = Apatch0{rr}; % if you want patch_ch1m_th to include all non-0 pixels of A
    AClip = Apatch_th{rr}; % if you want patch_ch1m_th to include thresholded pixels of A
    
    % inhibit channel
    % First set roiIm, ie the ROI on the inhibit channel
    roiMask = AClip;
    roiMask(~isnan(AClip)) = true;
    roiMask(isnan(AClip)) = false;
    
    % use the non-bleedthrough corrected inh patch
    inhClip = patch_ch1{rr} .* roiMask; % image of ch2 ROI on ch1. % figure; imagesc(roiIm)
    inhClip(~inhClip) = NaN;
    patch_ch1m_th{rr} = inhClip;
    
    inhClip = patch_ch2{rr} .* roiMask; % image of ch2 ROI on ch1. % figure; imagesc(roiIm)
    inhClip(~inhClip) = NaN;
    patch_ch2m_th{rr} = inhClip;    
end




%% Set slope for removing bleedthrough using pixel values of each ROI on the averaged patch (of noSpikeTimes frames), instead of using manActNoSpike (ie average of pixels in each noSpikeTimes frame, the part commented above).

% Take the mask of patch_ch1, patch_ch2... then for each ROI run regress between pixel values of the two images!

% onlyRegress = 1; % commonSlope solution works worse than regular regression after I removed spikes from macAct_ch2.... 

a = cellfun(@(x)full(x(:)), patch_ch1m_th, 'uniformoutput', 0); % all pixels of each ROI
b = cellfun(@(x)full(x(:)), patch_ch2m_th, 'uniformoutput', 0);
% remove nans (pixels not belonging to ROI)
a = cellfun(@(x)x(~isnan(x)), a, 'uniformoutput', 0);
b = cellfun(@(x)x(~isnan(x)), b, 'uniformoutput', 0);

% a = cellfun(@(x)x(:), patch_ch1, 'uniformoutput', 0); % all pixels of each ROI
% b = cellfun(@(x)x(:), patch_ch2, 'uniformoutput', 0);

% only use the mask pixels.

tic
[slope_common, offsets_ch1] = inhibit_remove_bleedthrough(a, b); %, onlyRegress);
t = toc;
disp(t)


%% Remove bleedthrough from each inh patch

bCorrPatch1 = cell(1, length(patch_ch1));
bCorrPatch1m_th = cell(1, length(patch_ch1)); % values outside ROI are set to nan (if Apatch_th was thresholded (95% contour), bCorrPatch1m_th will be also thresholded).
for ni = 1:length(patch_ch1)
    bCorrPatch1{ni} = patch_ch1{ni} - slope_common * patch_ch2{ni};
    bCorrPatch1m_th{ni} = patch_ch1m_th{ni} - slope_common * patch_ch2m_th{ni};
end


%% Filter (before computing corr)
%{
gaussDev = [5 5]; %params.brightGaussSDs;
% Filter image using gauss2D
Apatch0_f = cellfun(@(x)imfilter(full(x), gauss2D(gaussDev .* 4, gaussDev), 'symmetric'), Apatch0, 'UniformOutput', 0);

filtWidth = 2;
filtSigma = 1;
imageFilter = fspecial('gaussian',filtWidth,filtSigma);
bCorrPatch1_f = cellfun(@(x)nanconv(full(x),imageFilter, 'nanout'), bCorrPatch1, 'UniformOutput', 0);


fsz = [2,2];
Apatch0_f = cellfun(@(x)medfilt2(full(x),fsz), Apatch0, 'UniformOutput', 0);
bCorrPatch1_f = cellfun(@(x)medfilt2(x,fsz), bCorrPatch1, 'UniformOutput', 0);

Apatch_thf = cellfun(@(x)medfilt2(x,fsz), Apatch_th, 'UniformOutput', 0);
bCorrPatch1m_thf = cellfun(@(x)medfilt2(x,fsz), bCorrPatch1m_th, 'UniformOutput', 0);


figure; 
subplot(221), imagesc(Apatch0{12}); subplot(223), imagesc(Apatch0_f{12})
subplot(222), imagesc(Apatch_th{12}); subplot(224), imagesc(Apatch_thf{12})

figure; 
subplot(221), imagesc(bCorrPatch1{11}); subplot(223), imagesc(bCorrPatch1_f{11})
subplot(222), imagesc(bCorrPatch1m_th{121}); subplot(224), imagesc(bCorrPatch1m_thf{121})
%}


%% Compute corr between bleedtrhough-corrected patch_ch1 and patch_ch2

corr_A_inh_m0 = cellfun(@(x,y)corr(x(:), y(:), 'rows', 'complete'), bCorrPatch1, Apatch0); % corr with original A
corr_A_inh_mth = cellfun(@(x,y)corr(x(:), y(:), 'rows', 'complete'), bCorrPatch1, Apatch_th); % corr with thresholded A
% same as above bc of nans in Apatch_th.
% corr_A_inh_mth = cellfun(@(x,y)corr(x(:), y(:), 'rows', 'complete'), bCorrPatch1m_th, Apatch_th); % corr with thresholded A


% corr_A_inh_m0 = cellfun(@(x,y)corr(x(:), y(:), 'rows', 'complete'), bCorrPatch1_f, Apatch0); % corr with original A
% corr_A_inh_mth = cellfun(@(x,y)corr(x(:), y(:), 'rows', 'complete'), bCorrPatch1_f, Apatch_th); % corr with thresholded A


% pad = 0;
%{
corr_A_inh_m0 = nan(1, length(bCorrPatch1));

for rr = 1:length(bCorrPatch1)        
    %{
    Anow = A(:,rr);
    AMat = reshape(Anow, [imHeight, imWidth]);

    [i, j] = find(AMat);
    xRange = [max(min(j)-pad, 1) min(max(j)+pad, imWidth)];
    yRange = [max(min(i)-pad, 1) min(max(i)+pad, imHeight)];
    % Clip A to only include non-0 values
    AClip = AMat(yRange(1):yRange(2), xRange(1):xRange(2));    
    % set 0s to nans
    AClip(~AClip) = NaN;
    %}
    AClip = Apatch_th{rr};
    % inhibit channel
    % First set roiIm, ie the ROI on the inhibit channel
    roiMask = AClip;
    roiMask(~isnan(AClip)) = true;
    roiMask(isnan(AClip)) = false;
    
    % use the non-bleedthrough corrected inh patch
%     inhClip = patch_ch1{rr} .* roiMask; % image of ch2 ROI on ch1. % figure; imagesc(roiIm)
    % use the bleedthrough corrected inh patch
    inhClip = bCorrPatch1{rr} .* roiMask; % image of ch2 ROI on ch1. % figure; imagesc(roiIm)
    
    % Clip A to only include non-0 values
    %     inhClip = roiIm(yRange(1):yRange(2), xRange(1):xRange(2));
    % set 0s to nans
    inhClip(~inhClip) = NaN;
    
    %     figure; imagesc(AClip); figure; imagesc(inhClip)
    
    % now get the corr btwn the 2
    corr_A_inh_m0(rr) = corr(inhClip(:), full(AClip(:)), 'rows', 'complete'); % corr2(inhClip, AClip);
    
    % above is same as :
    corr(bCorrPatch1m_th{rr}(:), Apatch_th{rr}(:), 'rows', 'complete')
end
%}


%% Compute ave pixel intensity in each patch

sigPatch2m_th = cellfun(@(x)nanmean(full(x(:))), patch_ch2m_th);
sigPatch2 = cellfun(@(x)nanmean(full(x(:))), patch_ch2);
sigPatch1m_th = cellfun(@(x)nanmean(full(x(:))), patch_ch1m_th);
sigPatch1 = cellfun(@(x)nanmean(full(x(:))), patch_ch1);
sigCorrPatch1m_th = cellfun(@(x)nanmean(full(x(:))), bCorrPatch1m_th);
sigCorrPatch1 = cellfun(@(x)nanmean(full(x(:))), bCorrPatch1);
% sigCaPatch = cellfun(@(x)full(mean(x(:))), Apatch_th);
% figure; plot(sigCaPatch)
% sigHlPatch = cellfun(@(x)mean(x(:)), highlightPatchAvg);

wSigCorrPatch1m_th = nan(1, length(Apatch_th));
% wnSigCorrPatch1m = nan(1, length(Apatch_th));
for ni = 1:length(Apatch_th)
    a = Apatch_th{ni}(:) / max(Apatch_th{ni}(:)); a = a(~isnan(a));
    b = bCorrPatch1m_th{ni}(:); b = b(~isnan(b));
    c = a'*b;
    wSigCorrPatch1m_th(ni) = c/sum(~isnan(b)); % weighted by A
%     wnSigCorrPatch1m(ni) = sum(b)/sum(~isnan(b)); % regular mean % same as sigCorrPatch1m
end


%%
figure; 
subplot(411), hold on; plot(sigPatch2), plot(sigPatch2m_th), legend('patch2', 'patch2m_th') %title('patch2m')
subplot(412), hold on; plot(sigPatch1), plot(sigPatch1m_th), legend('patch1', 'patch1m_th') %title('patch1m')
subplot(413), hold on; plot(wSigCorrPatch1m_th); plot(sigCorrPatch1), plot(sigCorrPatch1m_th), legend('weighted by A', 'bCorrPatch1', 'bCorrPatch1m_th') %title('bCorrPatch1m_th')
subplot(414), hold on; plot(corr_A_inh_m0), plot(corr_A_inh_mth); legend('corrAinh', 'corrAinhm')

%
figure; plot(corr_A_inh_mth, sigCorrPatch1m_th, '.')
title(sprintf('corr = %.2f', corr(corr_A_inh_mth', sigCorrPatch1m_th')))


%%
the = .2; %.15
thi = .35; %.4

nExc = sum(corr_A_inh_mth <= the);
nInh = sum(corr_A_inh_mth >= thi);
nUns = sum(corr_A_inh_mth > the & corr_A_inh_mth < thi);
fprintf('Number exc = %d; inh = %d; unsure = %d\n', nExc, nInh, nUns)

fExc = mean(corr_A_inh_mth <= the);
fInh = mean(corr_A_inh_mth >= thi);
fUns = mean(corr_A_inh_mth > the & corr_A_inh_mth < thi);
fprintf('Fraction exc = %.2f; inh = %.2f; unsure = %.2f\n', fExc, fInh, fUns)


%%
load(moreName, 'inhibitRois', 'inhibitImageCorrcted')

sum(inhibitRois==1)

intersect(find(inhibitRois==1), find(corr_A_inh_mth >= thi))


%%
[~,ins] = sort(corr_A_inh_mth, 'descend');
% neurons to plot:
nstop = ins; %find(corr_A_inh_mth>=.25 & corr_A_inh_mth<.3); %find(corr_A_inh_mth>=.4)%99%229%241; %%313; %ins(end:-1:1); %1:length(patch_ch1) % length(Apatch_th):-1:1; % 1:length(Apatch_th)

f0 = figure('position', [24    21   570   882]);
f = figure; imagesc(sdImage{1}); hold on

cnt = 0;
for ni = nstop
    
    cnt = cnt+1;
    
    AMat = reshape(A(:,ni), siz);
    [i, j] = find(AMat);
    xRange = [max(min(j)-pad, 1)  ,  min(max(j)+pad, siz(2))];
    yRange = [max(min(i)-pad, 1)  ,  min(max(i)+pad, siz(1))];
    neighbors = getNeighbors(COMs, xRange, yRange);
    %     neighbors(neighbors == ni) = [];
    
    %%
    % figure; imagesc(reshape(A(:,1), imHeight, imWidth))
    if inhibitRois(ni)==0
        nam = sprintf('ROI %d(%d/%d) is exc', ni, cnt, length(nstop));
    elseif inhibitRois(ni)==1
        nam = sprintf('ROI %d/(%d/%d) is inh', ni, cnt, length(nstop));
    elseif isnan(inhibitRois(ni))
        nam = sprintf('ROI %d/(%d/%d) is unsure', ni, cnt, length(nstop));
    end
    
    
    figure(f0)
    set(gcf, 'name', nam);
    
%     subplot(321); imagesc(highlightPatchAvg{ni}), title(sprintf('hilitePatch, %.2f', sigPatch2(ni)))
    subplot(421); imagesc(patch_ch2{ni}), title(sprintf('noSpike patch2, %.0f', sigPatch2(ni)))
    hold on
    for rr = neighbors
        plot(CC{rr}(2,:)-xRange(1)+1 , CC{rr}(1,:)-yRange(1)+1, 'color', 'r')
    end
    
    subplot(422), imagesc(patch_ch1{ni}), title(sprintf('noSpike patch1, %.0f', sigPatch1(ni)))
    hold on
    for rr = neighbors
        plot(CC{rr}(2,:)-xRange(1)+1 , CC{rr}(1,:)-yRange(1)+1, 'color', 'r')
    end    
       

    
    
    subplot(423); imagesc(highlightPatchAvg{ni}), title(sprintf('hilite patch2, %.2f', highlightCorrROI(ni)))
    hold on
    for rr = neighbors
        plot(CC{rr}(2,:)-xRange(1)+1 , CC{rr}(1,:)-yRange(1)+1, 'color', 'r')
    end

%     subplot(428), imagesc(inhibitImageCorrcted(yRange(1):yRange(2), xRange(1):xRange(2))), title('correctedInh')
    subplot(424), imagesc(bCorrPatch1{ni}), title(sprintf('bCorrectedPatch1, %.2f, %.0f', corr_A_inh_m0(ni), sigCorrPatch1(ni)))
    hold on
    for rr = neighbors
        plot(CC{rr}(2,:)-xRange(1)+1 , CC{rr}(1,:)-yRange(1)+1, 'color', 'r')
    end
    
    

    
    subplot(425), imagesc(Apatch0{ni}), title(sprintf('A, %.0f', szApatch(ni)))
    hold on
    for rr = neighbors
        plot(CC{rr}(2,:)-xRange(1)+1 , CC{rr}(1,:)-yRange(1)+1, 'color', 'r')
    end
    
%     subplot(424), imagesc(bCorrPatch1{ni}), title(sprintf('correctPatch1, %.2f, %.0f', corr_A_inh_m0(ni), sigCorrPatch1(ni)))
    subplot(426), imagesc(bCorrPatch1m_th{ni}), title(sprintf('bCorrectedPatch1mth, %.2f, %.0f', corr_A_inh_mth(ni), sigCorrPatch1m_th(ni)))
    hold on
    for rr = neighbors
        plot(CC{rr}(2,:)-xRange(1)+1 , CC{rr}(1,:)-yRange(1)+1, 'color', 'r')
    end
    
    
        
    
    subplot(427), imagesc(sdImage{2}(yRange(1):yRange(2), xRange(1):xRange(2))), title('sdImage2')
    hold on
    for rr = neighbors
        plot(CC{rr}(2,:)-xRange(1)+1 , CC{rr}(1,:)-yRange(1)+1, 'color', 'r')
    end
    
    subplot(428), imagesc(sdImage{1}(yRange(1):yRange(2), xRange(1):xRange(2))), title('sdImage1')    
    hold on
    for rr = neighbors
        plot(CC{rr}(2,:)-xRange(1)+1 , CC{rr}(1,:)-yRange(1)+1, 'color', 'r')
    end

    
    
    %%
    figure(f)
    h = plot(CC{ni}(2,:), CC{ni}(1,:), 'color', 'r');

    %%
    pause
    figure(f0); clf
    delete(h)
    
end







%% Get manAct traces for noSpikeTimes
%{
inhName = fullfile(pd, sprintf('inhPatch_%s.mat', pnev_n));

load(inhName, 'patch_ch1', 'patch_ch2', 'noSpikeTimes', 'pad')
patch_ch1 = patch_ch1(~badROIs01);
patch_ch2 = patch_ch2(~badROIs01);


load(pnevFileName, 'activity_man_eftMask_ch1', 'activity_man_eftMask_ch2')
load(imfilename, 'pmtOffFrames')
activity_man_eftMask_ch1 = activity_man_eftMask_ch1(~pmtOffFrames{1}, ~badROIs01);
activity_man_eftMask_ch2 = activity_man_eftMask_ch2(~pmtOffFrames{2}, ~badROIs01);


%%
% nFrsNoSpike = size(inhPatchMov{1},3);
thq = .1; % go with the lowest 10% C values.
nFrsNoSpike = ceil(size(activity_man_eftMask_ch2,1)*thq);
randfrs = randperm(nFrsNoSpike);
actManNoSpike_ch1 = nan(nFrsNoSpike, length(patch_ch1));
actManNoSpike_ch2 = nan(nFrsNoSpike, length(patch_ch1));

for ni = 1:length(patch_ch1)
    sortIndex = noSpikeTimes{ni};
    %{
%     Cnew = C(ni,:);
    Cnew = activity_man_eftMask_ch2(:,ni);
    
    [sortedValues, sortIndex] = sort(Cnew(:),'ascend'); % get the first 500 frames with the smallest C values
    sortIndex = sortIndex(1:nFrsNoSpike);
%     sortIndex = sortIndex(randfrs);
%     figure; plot(Cnew); hold on; plot(sortIndex, repmat(0, size(sortIndex)), 'r.')
    %}
    %{
    [sortedValues, sortIndex] = sort(Cnew(:),'ascend'); % get the first 500 frames with the smallest C values
    sortIndex = sortIndex(1:min(length(sortIndex), nFrsNoSpike));
    %}
    actManNoSpike_ch1(:, ni) = activity_man_eftMask_ch1(sortIndex, ni);
    actManNoSpike_ch2(:, ni) = activity_man_eftMask_ch2(sortIndex, ni);
end


figure; 
subplot(211), plot(mean(actManNoSpike_ch1,2))
subplot(212),  plot(mean(actManNoSpike_ch2,2))


%% Set slope for removing bleedthrough

onlyRegress = 1; % commonSlope solution works worse than regular regression after I removed spikes from macAct_ch2.... 

tic
[slope_common, offsets_ch1] = inhibit_remove_bleedthrough(actManNoSpike_ch1, actManNoSpike_ch2, onlyRegress);
t = toc;
print t
%}

