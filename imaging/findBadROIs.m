% Find bad ROI outputs of the CNMF algorithm

mouse = 'fni17';
imagingFolder = '151102'; %'151029'; %  '150916'; % '151021';
mdfFileNumber = [1,2];  % 3; %1; % or tif major


%% Set imfilename, pnevFileName

signalCh = 2; % because you get A from channel 2, I think this should be always 2.
pnev2load = [];

[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
[pd,pnev_n] = fileparts(pnevFileName);
disp(pnev_n)
cd(fileparts(imfilename))

fname = fullfile(pd, sprintf('more_%s.mat', pnev_n));


%% Append A and CC to a mat file named more_pnev created in python... (you have to first create this file in python, it includes outputs of Andrea's evaluate components).

fname = fullfile(pd, sprintf('more_%s.mat', pnev_n));

load(pnevFileName, 'A')
load(imfilename, 'imHeight', 'imWidth')

mask = maskSet(A, imHeight, imWidth);
CC = ROIContoursPnevCC(A, imHeight, imWidth, .95);

if exist(fname,'file')==2
    save(fname, 'mask', 'CC', '-append')
else
    error('You should first creat this mat file in python, otherwise you wont be able to append evaluate_comps vars to it later in python!')
    save(fname, 'mask', 'CC')
end


%%
load(pnevFileName, 'activity_man_eftMask_ch2', 'C', 'P', 'srt_val','highlightCorrROI', 'roiPatch', 'highlightPatchAvg', 'A')
load(imfilename, 'imHeight', 'imWidth', 'sdImage')
load(fname, 'idx_components', 'fitness', 'mask', 'CC')

if ~min(idx_components)
    idx_components = idx_components+1; % bc python's indeces are from 0! % idx_components(i) = j; i: after-sort index; j: before-sort index.
end

highlightCorrROI = highlightCorrROI';
fitness = fitness';
fitnessNow = fitness(idx_components); % turn fitness into an array whose indeces match Efty's outputs.
% srt_val = full(srt_val);


%% Set correlation between C and raw trace

c = corr(C', activity_man_eftMask_ch2); % this is generally higher than corr(C_df', dFOF_man)
temp_corr = diag(c);
size(temp_corr)
% c = corr(C_df', dFOF_man); 
% cc2 = diag(c);

fht = figure;
figure(fht), subplot(321), histogram(temp_corr)
xlabel('Raw vs C temp corr')
ylabel('# Neurons')


%% Set time constants (in ms) from P.gn

frameLength = 1000/30.9; % sec.
tau = nan(size(P.gn,1), 2);
for i = 1:length(tau)
    g = P.gn{i};
    tau(i,:) = tau_d2c(g,frameLength); % tau(:,1) is rise, and tau(:,2) is decay time constant (in ms).
end

figure(fht), subplot(322), histogram(tau(:,2))
xlabel('Tau\_decay (ms)')


%% Highlight-reel vs spatial component correaltion

figure(fht), subplot(323), histogram(highlightCorrROI)
xlabel('highlight-raw vs spatial-comp corr')


%% Size of ROI

mask_numpix = sum(reshape(mask, imHeight*imWidth, []), 1)';
figure(fht), subplot(324), histogram(mask_numpix), xlabel('mask # pixels')


%% Andrea's fitness measure 

figure(fht), subplot(325), histogram(fitnessNow), xlabel('AG fitness')


%% Eftychios srt_val

figure(fht), subplot(326), histogram(srt_val), xlabel('Efty sort\_val')


%% Intensity of the sdImage of the ROI 
%{
im = sdImage{2};
maskn = double(mask);
maskn(mask(:)==0) = nan;

meansdsig = nan(size(mask,3), 1);
for rr = 1:size(mask,3)
    imthis = im .* maskn(:,:,rr);
    meansdsig(rr) = nanmean(imthis(:));
end
%}

%% Plot all the measures for componentes sorted by Eftychios's srt_val

figure; 
subplot(611), plot(srt_val), title('sort value')
subplot(612), plot(fitnessNow), title('fitness')
subplot(613), plot(temp_corr), title('temp corr')
subplot(614), plot(highlightCorrROI), title('spac corr')
subplot(615), plot(tau(:,2)), title('decay tau')
subplot(616), plot(mask_numpix), title('mask # pixels')
% subplot(616), plot(meansdsig), title('meanSdImage')


%% Find nearby neurons.
%{
doeval = 0;
merged_ROIs = mergeROIs_set([], A, C, imHeight, imWidth, [4.8 nan 2], 1, doeval, 0);

% find ROIs near ROI 13
i=131;
nearbyROIs = findNearbyROIs(COMs, COMs(i,:), 5);
%}


%% Identify bad components using a combination of criteria

thAG = -20;
badAG = fitnessNow >= thAG;

numbad = sum(fitnessNow >= thAG);
ss = sort(srt_val);
th_srt_val = ss(numbad); % you are trying to find a good threshold for srt_val (below which ROIs are bad).
badEP = srt_val < th_srt_val;


smallROI = mask_numpix < 15; % increase this to 20 if you want to get rid of neuropils
shortDecayTau = tau(:,2) < 200;
badTempCorr = temp_corr < .4;

badHighlightCorr = highlightCorrROI < .5;
goodSrtvalButbadHighlightCorr = (highlightCorrROI<.5 & srt_val>=1e4); % these have good trace quality but are mostly neuropils. so you decide to add them or not.


% bada = badEP + badAG + smallROI + shortDecayTau + badTempCorr;
badAll = badEP + badAG + smallROI + shortDecayTau + badTempCorr + badHighlightCorr;


badROIs01 = (badAll ~= 0); % any of the above measure is bad.
fprintf('number of bad ROIs = %d, mean = %.2f\n', sum(badROIs01), mean(badROIs01))

badROIs = find(badROIs01);
goodROIs = find(~badROIs01);

%{
% ROIs that are in very low intensity parts of the image.
badsdimage = meansdsig < 1000; % 800
sum(badsdimage)

% add/remove one condition and see how it affects the results.
fth = (badsdimage==1 & badROIs01==0);
sum(fth)
%}




%% Nice figure to evaluate the results
COMs = fastCOMsA(A, [imHeight, imWidth]);


%%
rois2p = find(~badROIs01);
rois2p = nearbyROIs;


%%
rois2p = rois2p(randperm(length(rois2p)));

figure; 
subplot(3,3,[7]); 
imagesc(log(sdImage{2}))

for i = rois2p' % % %; %ag_eb % f%ab_eg; %find(bc)' %find(mask_numpix<15); %nearbyROIs' %1:size(C,1) % fb'; % fb'; %220; %477;     
    [cinall(i) coutall(i)]
    if ismember(i, badROIs)
        col = 'r';
    else
        col = 'k';
    end
%     i
    set(gcf,'name', sprintf('ROI: %i', i))
    hold on
    subplot(3,3,[1,2,3]), h1 = plot(C(i,:)); 
%     title(sprintf('tau = %.2f ms', tau(i,2))),  % title(sprintf('%.2f, %.2f', [temp_corr(i), tau(i,2)])), 
    title(sprintf('fitness = %.2f,  srtval = %.2f', fitnessNow(i), full(srt_val(i))), 'color', col)
    xlim([1 size(C,2)])
    ylabel('C')% (denoised-demixed trace)')
    
    subplot(3,3,[4,5,6]), 
    h2 = plot(activity_man_eftMask_ch2(:,i)); 
%     h2 = plot(yrac(i,:)); 
    title(sprintf('tau = %.2f ms, temp corr = %.2f', tau(i,2), temp_corr(i)), 'color', col)
    xlim([1 size(C,2)])
    ylabel('Raw') % (averaged pixel intensities)')
    
    subplot(3,3,[7]); hold on
    h3 = plot(CC{i}(2,:), CC{i}(1,:), 'r');
    xlim([COMs(i,2)-50  COMs(i,2)+50])
    ylim([COMs(i,1)-50  COMs(i,1)+50])
%     imagesc(reshape(A(:,i), imHeight, imWidth))
    title(sprintf('#pix = %i', mask_numpix(i)), 'color', col)
%     title(sprintf('#pix = %i, fitness = %.2f srtval = %.2f', mask_numpix(i), fitness(i), full(srt_val(i))))
%     title(sprintf('#pix = %i,  meansdsig = %.2f', mask_numpix(i), meansdsig(i)))
    
    plotCorr_FN(roiPatch, highlightPatchAvg, highlightCorrROI, A, CC, COMs, [imHeight, imWidth], i, [3,3,8], [3,3,9])
    h4 = subplot(3,3,8);
    h5 = subplot(3,3,9);
    
    pause
    delete([h1,h2,h3,h4,h5])
end



%% Plot COMs of bad and good components on the medImage.

im = sdImage{2}; % medImage{2};

COMs = fastCOMsA(A, size(im));

% bad components
figure; 
imagesc(im)
hold on
for rr = find(badROIs01')
    plot(COMs(rr,2), COMs(rr,1), 'r.')
end

% good components
figure; 
imagesc(im)
hold on
for rr = find(~badROIs01')
    plot(COMs(rr,2), COMs(rr,1), 'r.')
end


%%
A = A(:, ~badROIs01);
C = C(~badROIs01,:);
C_df = C_df(~badROIs01,:);



%%
%{
%% 
Cnew = C(idx_components, :);
erfcnew = erfc(idx_components, :);

%%
figure; plot(idx_components)
xlabel('after-sort index')
ylabel('before-sort index')

ii = randperm(length(idx_components), 1); 
figure; hold on
plot(C(idx_components(ii),:))
plot(Cnew(ii,:), 'r')


%% Compare Andrea and Efty's measures.

numbad = sum(fitnessNow >= thAG);
figure; plot(sort(fitnessNow))
hold on; plot([1 length(fitnessNow)], [thAG thAG], 'r-')

srtval_bad = zeros(1, length(srt_val));
[ss,is] = sort(srt_val);
srtval_bad(is(1:numbad)) = 1; % srt_val' <= quantile(srt_val, .1);
th_srt_val = max(srt_val(srtval_bad==1)) % you are trying to see what could be a good threshold for srt_val (below which ROIs are bad).

figure; plot(srt_val), hold on; plot([1 length(fitnessNow)], [th_srt_val, th_srt_val], 'r-')


fitness_bad = zeros(1, length(srt_val));

fitness_bad(fitnessNow >= thAG) = 1;
% [ss,is] = sort(fitnessNow);
% fitness_bad(is(end-(numbad-1):end)) = 1; % fitnessNow >= thAG;


ab_eg = find(fitness_bad==1 & srtval_bad==0); % Andrea says bad, Efty says good

ag_eb = find(fitness_bad==0 & srtval_bad==1); % Andrea says good, Efty says bad
%}




%%
% tau seems to be the best measure.
% tau is very related to order_measure, and using f==1 measure is enough
% (you don't need f3==1... it is almost redundant).

% corr: if u want to exclude more comps, use both corr and tau
% measures and a thresh of .4 for corr. You can go down to .3 to exclude fewer comps.
% or just go with tau (u'll exclude fewer neurons).

% remember u can set an upper limit for tau too (too slow time constants).


%%
% [A_or,C_or,S_or,P_or,srt, order_measure] = order_ROIs(A,C,S,P);
% [srt_sn,n_events,n_act] = order_components(YrA,C,[]);


%% You need to load these vars

%{
load('151102_001-002.mat', 'imHeight', 'imWidth', 'sdImage', 'Nnan_nanBeg_nanEnd')
load('151102_001-002_ch2-PnevPanResults-160624-113108.mat', 'A_m', 'C_m', 'P_m', 'badComps', 'activity_man_eftMask_ch2', 'merging_vars_m')

C = C_m;
A = A_m;
P = P_m;

nanBeg =  Nnan_nanBeg_nanEnd(2,:);
nanEnd = Nnan_nanBeg_nanEnd(3,:);
inds2rmv = cell2mat(arrayfun(@(x,y)(x:y), nanBeg, nanEnd, 'uniformoutput', 0)); % index of nan-ITIs (inferred ITIs) on C and S traces.
C(:, inds2rmv) = [];

%}

% merge components in activity_man using mergedROIs (in case merging was done after activity was set)
%{
if size(activity_man_eftMask_ch2,2) ~= size(C,1)
    a = activity_man_eftMask_ch2;
    a = a(:, ~badComps);
    size(a)

    clear am
    for i=1:length(merging_vars_m.merged_ROIs{1})
        am(:,i) = mean(a(:, merging_vars_m.merged_ROIs{1}{i}),2);
    end
    size(am)

    %%%%
    m = cell2mat(merging_vars_m.merged_ROIs{1});
    a(:,m) = [];
    size(a)

    %%%%
    a = cat(2, a, am);
    size(a)

    %%%%
    activity_man_eftMask_ch2 = a;
end
%}

%%
%{


%% Identify bad components

f1 = (tau(:,2) < 100); % 50 % (frameLength*tau(:,2) < 50) | (frameLength*tau(:,2) > 2000)
f2 = temp_corr < .35; % .3; .4; % max([temp_corr,cc2],[],2) < .4; % 
% f3 = full(order_measure < 2000); % very few comps with f1==0 and f3==1... ie if you go with f1==1, then f3 is more of a redundant measure.

% [sum(f1==1) sum(f2==1) sum(f3==1)]
[sum(f1==1) sum(f2==1)]
sum(f1==0 & f2==1)
sum(f1==1 | f2==1)


%%
f1 = (tau(:,2) < 100); % 50 % (frameLength*tau(:,2) < 50) | (frameLength*tau(:,2) > 2000)
f2 = temp_corr < .5; % .3; .4; % max([temp_corr,cc2],[],2) < .4; % 
f3 = highlightCorrROI' < .5;
f4 = mask_numpix' < 15;
% f5 = srt_val <= quantile(srt_val, .1);
srt_val = fitness;
f5 = fitness' > -20;

bc_s = sum([f1, f2, f3, f4, f5], 2);

bc = (bc_s ~= 0); % either of the bad measures was 1.

sum(bc)

%% Define neurons to be excluded

badComps = (f1==1 | f2==1); % logical array same size as number of comps in C, with 1s for bad components.
num_fract_badComps = [sum(badComps) mean(badComps)]


%% Append badComps to pnevFile 

% save(pnevFileName, '-append', 'badComps')


%% Look at tau of good components

fb = find(~badComps);
[n,v] = histcounts(tau(fb,2)); figure; bar(v(1:end-1),n); hold on; plot(median(tau(fb,2)), max(n), 'r*'),  

figure; plot(tau(~badComps,2))


%% Assess the results (look at bad and good components (manual and C traces as well as contours)

fb = find(badComps); % [~,is]= sort(tau(fb,2), 'descend'); fb = fb(is); fb = fb(1:10); % fb = fb(randperm(length(fb))); 
fg = find(~badComps); % fg = fg(tau(fg,2)<600); fg = fg(2:11); % fb = fb(randperm(length(fb)));
% length(fb)
%}

