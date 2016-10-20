function [inhibitRois, roi2surr_sig, sigTh_IE, x_all, cost_all] = inhibitROIselection(maskGcamp, inhibitImage, manThSet, assessClass_unsure_inh_excit, keyEval, CCgcamp, ch2Image, COMs, C)
% inhibitRois = inhibitROIselection(maskGcamp, inhibitImage, sigTh, CCgcamp);
%
% Identifies inhibitory neurons on a gcamp channel (containing both
% excitatory and inhibitory neurons, "maskGcamp") by using the average image
% of inhibitory channel ("inhibitImage").
%
% INPUTS:
% maskGcamp: imHeight x imWidth x units, mask of gcamp channel neurons, that contains both excitatory and inhibitory neurons.
% inhibitImage: median image of channel with inhibitory neurons (ie tdtomato channel).
% sigTh: signal to noise threshold for identifying inhibitory neurons on tdtomato channel. eg. sigTh = 1.2;
% CCgcamp: optional, cell array: 1 x units. coordinates of ROI contours on
%   gcamp channel. If provided, a plot will be made to assess the output of the algorithm.
% assessClass_unsure_inh_excit: 3 elements, each specify whether to assess a specific class or not (unsure, inh, exc). if 1, plots will be made to assess the classification of inhibitory neurons. If 1, you will need the following 2 vars.
% figh: handle to figure that shows all ROIs on the gcamp channel.
% COMs: center of mass of ROIs
%
% OUTPUTS:
% inhibitRois: index of ROIs in gcamp mask that are inhibitory (ie they have a match on inhibitImage.)
%
% How the algorithm works?
% it computes a measure of signal/noise for each ch2 ROI (gcamp channel) on ch1 image (inhibitory channel),
% by applying the contour of ch2 on the medImage of ch1 and computing signal_magnitude_inside_contour / signal_magnitude_outside_contour
%
% e.g:
% [CCgcamp, ~, ~, maskGcamp] = setCC_cleanCC_plotCC_setMask(spatialComp, imHeight, imWidth, contour_threshold, medImage{2});
% medImageInhibit = medImage{1};
% sigTh = 1.2;

% Related to 2d-gaussian fitting method:
% x_all = []; % predicted paramteres of the gaussian : [Amplitude, x0, sigmax, y0, sigmay, angel(in rad)]
% cost_all = []; % normalized cost (0: good, 1: max failure)


%% Set vars

if ~exist('manThSet', 'var')
    manThSet = 0;
end

if ~exist('assessClass_unsure_inh_excit', 'var')
    assessClass_unsure_inh_excit = false(1,3);
end

imHeight = size(maskGcamp,1);
imWidth = size(maskGcamp,2);

% inhibitImage0 = inhibitImage; inhibitImage(inhibitImage<0) = 0;

%% Compute roi2surr_sig, ie average pixel intensity inside ROI relative to its immediate surrounding. This measure will be used to identify inhibitory neurons.

roi2surr_sig = NaN(1, size(maskGcamp,3));
roi2surr_sig_num = NaN(1, size(maskGcamp,3));
surrSig_all = NaN(1, size(maskGcamp,3));
roiSig_all = NaN(1, size(maskGcamp,3));

for rr = 1 : size(maskGcamp,3);
    
    % Compute roiSing: signal magnitude of ch2 ROI (maskGcamp) on ch1 image (inhibitImage).
    
    roiMask0 = maskGcamp(:,:,rr); % mask of ch2 ROI % figure; imagesc(roiMask)
    % set pixels outside the ROI to nan. use this if you are doing roiSig = nanmean(roiIm(:)); below
    roiMask = roiMask0;
    roiMask = double(roiMask);
    roiMask(~roiMask) = NaN;
    
    roiIm = inhibitImage .* roiMask; % image of ch2 ROI on ch1. % figure; imagesc(roiIm)
    ss = roiIm(:)~=0 & ~isnan(roiIm(:));
    s = sum(ss); % number of non-zero pixels in the image of ch2 ROI on ch1.
    %     s = sum(ss) / sum(roiMask(:)>0);
    %     if s > 3
    roiSigN = s;
    %     else, roiSig = 0; end
    roi2surr_sig_num(rr) = roiSigN;
    % if not doing nans for pixels outside ROI, use below.
    %{
    a = roiIm(ss);
%     roiSig = nanmedian(a(:)); % signal magnitude of ch2 ROI on ch1 image.
    roiSig = nanmean(a(:)); % signal magnitude of ch2 ROI on ch1 image.
    %}
    roiSig = nanmean(roiIm(:)); % mean of pixels inside the ROI. All pixels outside the ROI are set to nan.
    if roiSig<0
        roiSig = 0; % nan;
    end
    %     roi2surr_sig(rr) = roiSig; % use this if you don't want to account for surround mask.
    roiSig_all(rr) = roiSig;
    
    
    %% Set surrMask : a square mask surrounding roiMask (mask of ch2 ROI)
    
    surr_sz = 5; % 1; 5; % remember for 151029_003 you used 5.
    xl = [find(sum(roiMask0), 1, 'first')  find(sum(roiMask0), 1, 'last')];
    yl = [find(sum(roiMask0,2), 1, 'first')  find(sum(roiMask0,2), 1, 'last')];
    
    ccn_y = [max(yl(1)-surr_sz, 1)  max(yl(1)-surr_sz, 1)  min(yl(2)+surr_sz, imHeight) min(yl(2)+surr_sz, imHeight)];
    ccn_x = [max(xl(1)-surr_sz, 1)  min(xl(2)+surr_sz, imHeight)  min(xl(2)+surr_sz, imHeight)  max(xl(1)-surr_sz, 1)];
    ccn = [ccn_y ; ccn_x];
    
    maskn = maskSet({ccn}, imHeight, imWidth, 0);
    
    surrMask = maskn - roiMask0;
%     figure; imagesc(surrMask)
%     figure; imagesc(maskn)
%     figure; imagesc(roiMask)
    
    
    %% Compute surrSig: magnitude of ch1 image surrounding ch2 ROI.
    
    roiIm = inhibitImage .* surrMask;  % figure; imagesc(roiIm)
    ss = roiIm(:)~=0;
    %     s = sum(ss); % number of non-zero pixels in the image of ch2 surround ROI on ch1.
    %     surrSig = sum(roiIm(:)~=0);
    a = roiIm(ss);
    %     surrSig = nanmedian(a(:));
    surrSig = nanmean(a(:));
    %     sa = sort(a(:));
    %     surrSig = nanmean(sa(floor(.3*length(sa)) : floor(.7*length(sa))));
       
    if roiSig~=0 && surrSig<=0        
        surrSig = nanmean(a(a(:)>0));
    end
    surrSig_all(rr) = surrSig;
    
    
    %% Compute ratio of roi signal/surr sig.
    
    roi2surr_sig(rr) = roiSig / surrSig;
    %}
    
end

%%
% sum(~roi2surr_sig)
% roi2surr_sig(isnan(roi2surr_sig)) = 0; % ROIs with negative values (bleedthrough corrected can result in this.) % There are ROIs that had 0 signal for both the ROI and its surrounding, so they should be classified as excit.
% sum(~roi2surr_sig)

% q_num = quantile(roi2surr_sig_num, 9)
q_sig_orig = quantile(roi2surr_sig, 9)
% q_sig_nonzero = quantile(roi2surr_sig(roi2surr_sig~=0), 9)

% roi2surr_sig(isnan(surrSig_all)) = roiSig_all(isnan(surrSig_all)) ./ nanmean(surrSig_all)


%% Set to 0 the value of roi2surr_sig if an ROI has few positive pixels.
% you may want to add it back
%{
posPixTh = 10; %15;
cprintf('red', '%i defined as the min num of positive pixels for an ROI to be counted as inhibitory!\n', posPixTh)
roi2surr_sig(roi2surr_sig_num < posPixTh) = 0;

q_sig_aft = quantile(roi2surr_sig, 9)

% warning('you are using .9 quantile to define sigTh')
% sigTh = quantile(roi2surr_sig, .9);
% sigTh = quantile(roi2surr_sig, .8)

% fprintf('%.2f, %.2f, %.2f = upper 0.2, 0.25 and 0.3 quantiles of roi2surr_sig\n', quantile(roi2surr_sig, .8), quantile(roi2surr_sig, .75), quantile(roi2surr_sig, .7))
q_sig_nonzero = quantile(roi2surr_sig(roi2surr_sig~=0), 9)
%}


%% Set threshold for identifying inhibitory neurons

% It seems roi2surr_sig = 1.2 is a good threshold.
% sigTh = 1.2;

qinh = .9; %.9; % do a looser unsure so you assess more w 2d gauss
qexc = .8; %.8;
sigThI = quantile(roi2surr_sig, qinh); 
sigThE = quantile(roi2surr_sig, qexc); 
sigTh_IE = [sigThI, sigThE];

%     cprintf('red', 'Not performing manual evaulation!\n')
cprintf('red', 'inhibit: > %.2f quantile (%.2f)\nexcit: < %.2f quantile (%.2f) of roi2surr_sig\n', qinh, sigThI, qexc, sigThE)    


%%%%%%%%%%%%%%%%%% the following not really needed 
quantTh = .8; % .5; % .1; % threshold for finding inhibit neurons will be sigTh = quantile(roi2surr_sig, quantTh);
sigTh = quantile(roi2surr_sig, quantTh);
% sigTh = quantile(roi2surr_sig(roi2surr_sig~=0), quantTh);

figure('position', [1443         622         453         354]); hold on
h = plot(sort(roi2surr_sig)); set(h, 'handlevisibility', 'off')
plot([0 length(roi2surr_sig)], [sigTh sigTh], 'r')
plot([0 length(roi2surr_sig)], [quantile(roi2surr_sig, .9)  quantile(roi2surr_sig, .9)], 'g')
xlabel('Neuron number')
ylabel('ROI / surround')
legend('.8 quant of roi2surr\_sig', '.9 quantile of roi2surr\_sig')

if manThSet % if 1, you will change the default .8 quantile to whatever you wish :)
    disp('use Data Cursor to find a threshold, then type dbcont to return to the function')
    keyboard % pause and give control to the keyboard
    sigTh = input('What threshold you like? ');
    
end
  
    
%% Set inhibitory neurons

% if ~keyEval & ~manThSet % fully automatic
    
    % Alternative approach that will not require assessing:
    % count ROIs with sig_surr > .9th quantile as inhibit.
    % those with sig_surr < .8th quantile as excit.
    % those with sig_surr between .8 and .9th quantile as unknown (nan).
        
inhibitRois = NaN(1, length(roi2surr_sig));
inhibitRois(roi2surr_sig >= sigThI) = 1;
inhibitRois(roi2surr_sig <= sigThE) = 0;

%{    
else
    % Above .8 quantile is defined as inhibitory and below as excitatory. Evaluate it manually though!
    cprintf('black', 'sigTh defined as %.2f quantile of roi2surr_sig, = %.2f\n', quantTh, sigTh)
    fprintf('Using %.2f as the threshold for finding inhibitory ROIs, = %.2f\n', sigTh, sigTh)
    
    inhibitRois = roi2surr_sig > sigTh; % neurons in ch2 that are inhibitory. (ie present in ch1).
    inhibitRois = double(inhibitRois); % you do this so the class is consistent with when you do manual evaluation (below)
    
    sigTh_IE = [sigTh, sigTh];
    
end
%}

cprintf('blue', '%d inhibitory; %d excitatory; %d unsure neurons in gcamp channel.\n', sum(inhibitRois==1), sum(inhibitRois==0), sum(isnan(inhibitRois)))
cprintf('blue', '%.1f%% inhibitory; %.1f%% excitatory; %.1f%% unsure neurons in gcamp channel.\n', mean(inhibitRois==1)*100, mean(inhibitRois==0)*100, mean(isnan(inhibitRois)*100))


%
%{
sum(inhibitRois(roi2surr_sig_num < posPixTh))
inhibitRois(roi2surr_sig_num < posPixTh) = 0;

fract = nanmean(inhibitRois); % fraction of ch2 neurons also present in ch1.
cprintf('blue', '%d: num, %.3f: fraction of inhibitory neurons in gcamp channel.\n', sum(inhibitRois), fract)
%}




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x_all = []; % predicted paramteres of the gaussian : [Amplitude, x0, sigmax, y0, sigmay, angel(in rad)]
% cost_all = []; % normalized cost (0: good, 1: max failure)

%% Do 2D gaussian fitting on ch1 images of gcamp ROIs. 
% Later based on the cost value and amplitude we decide if ROIs are tdtomato-positive or not.
% Because it is slow we do it only for unsure ROIs that could not be classified as excit or inhibit. 

uns = isnan(inhibitRois);
f = find(uns); 

fprintf('Performing 2D-gaussian fitting on %i unsure ROIs\n', length(f))

cost_all = nan(1, size(maskGcamp,3));
x_all = nan(size(maskGcamp,3), 6);
exitflag_all = nan(1, size(maskGcamp,3));

for rr = f %1 : size(maskGcamp,3);
    
    fprintf('ROI %i\n', rr)
    
    roiMask0 = maskGcamp(:,:,rr); % mask of ch2 ROI % figure; imagesc(roiMask)
    % set pixels outside the ROI to nan. use this if you are doing roiSig = nanmean(roiIm(:)); below
    roiMask = roiMask0;
    roiMask = double(roiMask);
%     roiMask(~roiMask) = NaN;
    
    roiIm = inhibitImage .* roiMask; % image of ch2 ROI on ch1. % figure; imagesc(roiIm)
    
    %%    
    x0 = [100, COMs(rr,2), 4, COMs(rr,1), 4, 0]; %[1,0,50,0,50,0]; %Inital guess parameters [Amplitude, x0, sigmax, y0, sigmay, angel(in rad)]
    
    [X,Y] = meshgrid(1:imWidth, 1:imHeight);
    xdata = zeros(size(X,1),size(Y,2),2);
    xdata(:,:,1) = X;
    xdata(:,:,2) = Y;

    lb = [0, -COMs(rr,2), 0, -COMs(rr,1), 0, -pi/4]; % [Amp,xo,wx,yo,wy,fi]
    ub = [realmax('double'), COMs(rr,2), (COMs(rr,2))^2, COMs(rr,1), (COMs(rr,1))^2, pi/4];

    Z = roiIm;
%     tic
    [x,resnorm,residual,exitflag] = lsqcurvefit(@D2GaussFunctionRot,x0,xdata,Z,lb,ub); % [Amplitude, x0, sigmax, y0, sigmay, angel(in rad)]
%     t(rr) = toc;
    
    x_all(rr,:) = x;
    cost_all(rr) = resnorm / sum(Z(:).^2);    
    exitflag_all(rr) = exitflag;
    % figure; imagesc(residual), colorbar
end


figure; 
subplot(221), [n,v] = histcounts(cost_all, 100); bar(v(1:end-1)+mode(diff(v))/2, n)
title('cost')
subplot(222), [n,v] = histcounts(x_all(:,1), 100); bar(v(1:end-1)+mode(diff(v))/2, n)
title('amplitude')
subplot(223), [n,v] = histcounts(x_all(:,3), 300); bar(v(1:end-1)+mode(diff(v))/2, n)
title('sigmaX')
subplot(224), [n,v] = histcounts(x_all(:,5), 300); bar(v(1:end-1)+mode(diff(v))/2, n)
title('sigmaY')


%{
costQ = quantile(cost_all, 9)
ampQ = quantile(x_all(:,1), 9)
sigmaxQ = quantile(x_all(:,3), 9)
sigmayQ = quantile(x_all(:,5), 9)

figure; plot(roi2surr_sig)
hold on; plot(cost_all)
c = corrcoef(roi2surr_sig, cost_all);
title(sprintf('corr = %.2f', c(2)))

figure;
[n,v] = histcounts(roi2surr_sig, 100); bar(v(1:end-1)+mode(diff(v))/2, n)
xlabel('roi2surr\_sig')
ylabel('counts')
%}

%%%%%%%%%%%%%%%%%%%%%
%{
th_cost = .25;
th_amp = 1000; 
th_sigx = 3;
th_sigy = 3;

cm = cost_all'<th_cost;
am = x_all(:,1)>th_amp;
sxm = x_all(:,3)<th_sigx;
sym = x_all(:,5)<th_sigy;

meas = [cm, am, sxm, sym];
meas = meas(:,[1:2]);

aa = sum(meas,2);
fin = aa==size(meas,2);
sum(fin)

inhibitRois = NaN(1, length(roi2surr_sig));
inhibitRois(fin) = 1;
inhibitRois(~fin) = 0;
%}



%% identify unsure neurons with good Gauss fitting 

uns = isnan(inhibitRois);
f = find(uns); 

th_cost = .2; 
uns1 = cost_all(uns)' < th_cost; % very low cost

th_amp = 5000; 
uns2 = x_all(uns,1) > th_amp; % very hi amp
% uns3 = mean(x_all(uns,[3,5]),2)<th_sig;

% meas = [uns1, uns2]; %, uns3]
meas = [uns1];
ff = f(sum(meas,2)>=1); % if either very low cost or very high amplitude then w will call them inhibitory.
% roi2surr_sig(ff)


%% Assess the high-fit unsure neurons (potential inhibit ROIs)

fprintf('cost_amp_sigx_sigy\n')
if any(assessClass_unsure_inh_excit)
    h2d = figure; imagesc(inhibitImage); hold on
    for rr = ff; %[250   283   295]
        set(gcf, 'name', sprintf('ROI %d', rr))
        h = plot(CCgcamp{rr}(2,:), CCgcamp{rr}(1,:), 'color', 'r');
        fprintf('%.2f, %.2f, %.2f, %.2f\n', [cost_all(rr), x_all(rr,1), x_all(rr,3), x_all(rr,5)])
        pause
        delete(h)
    end
end


%% Re-assign unsure as inhibit

cprintf('magenta', '2D-Gauss reassigned %d ROIs as inhibit\n', length(ff))
inhibitRois(ff) = 1;


%% Identify unsure neurons with bad gauss fitting

th_cost = .35; %.25;
th_amp = 4000; 

uns1 = cost_all(uns)' >= th_cost; % very hi cost
uns2 = x_all(uns,1) <= th_amp; % very low amp
% uns3 = mean(x_all(uns,[3,5]),2)<th_sig;

meas = [uns1, uns2]; %, uns3]
ff = f(sum(meas,2)==size(meas,2));
% roi2surr_sig(ff)


%% Assess the low-fit unsure neurons (potential excit ROIs)

if any(assessClass_unsure_inh_excit)
    figure(h2d); % imagesc(inhibitImage); hold on
    for rr = ff; %[250   283   295]
        set(gcf, 'name', sprintf('ROI %d', rr))
        h = plot(CCgcamp{rr}(2,:), CCgcamp{rr}(1,:), 'color', 'r');
        fprintf('%.2f, %.2f, %.2f, %.2f\n', [cost_all(rr), x_all(rr,1), x_all(rr,3), x_all(rr,5)])
        pause
        delete(h)
    end
end


%% Re-assign unsure as excit

cprintf('magenta', '2DGauss reassigned %d ROIs as excit\n', length(ff))
inhibitRois(ff) = 0;


%%
cprintf('blue', '%d inhibitory; %d excitatory; %d unsure neurons in gcamp channel.\n', sum(inhibitRois==1), sum(inhibitRois==0), sum(isnan(inhibitRois)))
cprintf('blue', '%.1f%% inhibitory; %.1f%% excitatory; %.1f%% unsure neurons in gcamp channel.\n', mean(inhibitRois==1)*100, mean(inhibitRois==0)*100, mean(isnan(inhibitRois)*100))



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Compute correlation in the activity of each ch2 ROI between ch2 movie and ch1 movie.
%{
if showResults
    crr = NaN(1, size(activity_man_eftMask_ch1,2));
    for rr = 1:length(crr)
        t1 = activity_man_eftMask_ch1(:,rr);
        t2 = activity_man_eftMask_ch2(:,rr);
        crr(rr) = corr(t1, t2);
    end
end
%}
%%% Plots related to crr and brightness
%{
plotCrr(crr, inhibitRois, roi2surr_sig)

    
%% Call excitatory those neurons that have high crr (temporal correlation between ch1 and ch2)
% The idea is that these neurons are positive on the red channel due to
% bleedthrough, so they are false positive.

thCrr = .4;
cprintf('magenta', '%.2f= threshold for finding neurons with high correlation between ch1 and ch2\n', thCrr)
cprintf('blue', '%d= number of inhibitory neurons with high correlation\n', sum(inhibitRois(crr > thCrr)))

inhibitRois(crr > thCrr) = 0;

fract = nanmean(inhibitRois); % fraction of ch2 neurons also present in ch1.
cprintf('*magenta*', '%d: num, %.3f: fraction of inhibitory neurons in gcamp channel.\n', sum(inhibitRois), fract)
%}


%% Look at ch2 ROIs on ch1 image, 1 by 1.

if exist('CCgcamp', 'var') && any(assessClass_unsure_inh_excit)
    
    %     keyEval = 0; % Linux hangs with getKey if you click while it is running... so make sure this is set to 0! % if 0 you will simply go though ROIs one by one, otherwise it will go to getKey and you will be able to change neural classification.
    
    
    %% Plot ch2 image
    
    colors = hot(2*length(CCgcamp));
    colors = colors(end:-1:1,:);
    
    figh = figure;
    set(figh, 'position', [1   676   415   300])
    subplot(211), imagesc(ch2Image); hold on
    subplot(212), imagesc(inhibitImage); hold on
    %     imagesc(normImage(ch2Image));
    %     colormap gray
    
    subplot(211)
    for rr = 1:length(CCgcamp)
        %         if plotCOMs
        %             plot(COMs(rr,2), COMs(rr,1), 'r.')
        %
        %         else
        %[CC, ~, COMs] = setCC_cleanCC_plotCC_setMask(Ain, imHeight, imWidth, contour_threshold, im);
        if ~isempty(CCgcamp{rr})
            plot(CCgcamp{rr}(2,:), CCgcamp{rr}(1,:), 'color', colors(rr, :))
        else
            fprintf('Contour of ROI %i is empty!\n', rr)
        end
        %         end
    end
    title('ROIs shown on the sdImage of channel 2')
    
    
    subplot(212)
    for rr = 1:length(CCgcamp)
        if ~isempty(CCgcamp{rr})
            plot(CCgcamp{rr}(2,:), CCgcamp{rr}(1,:), 'color', colors(rr, :))
        else
            fprintf('Contour of ROI %i is empty!\n', rr)
        end
    end
    title('ch1 image')
    
    %     if ~isvalid(figh), figh = figure; imagesc(inhibitImage); end
    
    
    %% Set vars
    
    % Sort rois based on the roi2surr_signal, so first inhibit neurons
    % (from low to high signal, ie from problematic to reliable) are shown.
    % Then excit neurons (from high to low signal, ie from problematic to
    % reliable) are shown.
    
    % inhibit
    a = roi2surr_sig(inhibitRois==1);
    [~, i] = sort(a);
    f = find(inhibitRois==1);
    inhibit_inds_lo2hi = f(i); % indeces of inhibit neurons in the array that includes all neurons, sorted from low to high value of roi2surr_sig
    % first elements (low values of roi2surr_sig) are problematic and may not be really inhibit.
    
    
    % excit
    a = roi2surr_sig(inhibitRois==0);
    [~, i] = sort(a, 'descend');
    f = find(inhibitRois==0);
    excit_inds_hi2lo = f(i); % indeces of excit neurons in the array that includes all neurons, sorted from high to low value of roi2surr_sig
    % first elements (high values of roi2surr_sig) are problematic and may not be really excit.
    
    %     inds_inh_exc = [inhibit_inds_lo2hi, excit_inds_hi2lo]; % 1st see inhibit (ordered from low to high value of roi2surr_sig), then see excit (ordered from high to low value of roi2surr_sig)
    %     inds_inh_exc = [find(inhibitRois), find(~inhibitRois)]; % 1st see inhibit, then excit neurons
    
    
    % unsure
    a = roi2surr_sig(isnan(inhibitRois));  % look at unsure cases.
    [~, i] = sort(a, 'descend');
    f = find(isnan(inhibitRois)); % look at unsure cases.
    unsure_inds_hi2lo = f(i); % indeces of excit neurons in the array that includes all neurons, sorted from high to low value of roi2surr_sig
    
    %     plotInhFirst = 1; % if 1, first inhibitory ROIs will be plotted then excit. If 0, ROIs will be plotted in their original order.
    
    
    inhibitRois_new = double(inhibitRois); % array indicating neurons' class after manual assessment.
    
    
    %{
    disp('=====================')
    disp('Evaluate inhibitory neuron identification. Figure shows medImage of inhibit neurons.')
    if keyEval
        %     disp('Red contours are inhibitory. Yellow: excitatory')
        disp('Esc: quit evaluation.')
        disp('Other keys: keep showing the same ROI.')
        disp('When contour is shown :')
        disp('... press Enter if happy with classification and want to see next ROI.')
        disp('... press 0 if classification is wrong.')
        disp('... press 2 if unsure about classification.')
        cprintf('red', 'DO NOT CLICK OR THE COMPUTER WILL HANG!! only use keyboard keys!!\n')
    else
        disp('Press any key to go through neurons one by one.')
    end
    %}
    
    %% Plot and evaluate unsure neurons
    
    if assessClass_unsure_inh_excit(1)
        cprintf('-blue', '------- Evaluating unsure neurons -------\n')
        if keyEval
            %     disp('Red contours are inhibitory. Yellow: excitatory')
            disp('Esc: quit evaluation.')
            disp('Other keys: keep showing the same ROI.')
            disp('When contour is shown :')
            disp('... press Enter if happy with classification and want to see next ROI.')
            disp('... press 0 if neuron must be excit.')
            disp('... press 1 if neuron must be inhibit.')
            cprintf('red', 'DO NOT CLICK OR THE COMPUTER WILL HANG!! only use keyboard keys!!\n')
        else
            disp('Press any key to go through neurons one by one.')
        end
        
        
        fimag = figure;
        set(fimag, 'position', [1   -18   693   610]); %[28   133   805   658]) % get(groot, 'screensize'))
        set(gca, 'position', [0.0519    0.0656    0.9019    0.8951])
        imagesc(inhibitImage)
        %     imagesc(normImage(inhibitImage))
        hold on
        
        ftrace = figure('position', [383   682   987   236], 'name', 'C is plotted');
        
        unsureEval = NaN(1, length(CCgcamp));
        rr = 1;
        while rr <= length(unsure_inds_hi2lo); % length(CCgcamp)
            
            %         if plotInhFirst
            %             rr2 = inds_inh_exc(rr); % first plot all inhibit neurons, then all excit neurons.
            rr2 = unsure_inds_hi2lo(rr); % rr2 is the index in the array including all neurons
            nearbyROIs = findNearbyROIs(COMs, COMs(rr2,:), 6); nearbyROIs(nearbyROIs==rr2) = [];
            %         else
            %             rr2 = rr; % plot ROIs in the original order
            %         end
            
            set(fimag, 'name', sprintf('ROI %d. medImage of inhibitory channel. Use Esc to quit! ', rr2))
%             set(fimag, 'name', sprintf('ROI %d. Sig/Surr threshold = %.2f. medImage of inhibitory channel. Use Esc to quit! ', rr2, sigTh))
            
            % zoom on the gcamp channel image so you get an idea of the surrounding ROIs.
            figure(figh)
            subplot(211)
            plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'r', 'linewidth', 1.5)
            comd = 20;
            xlim([COMs(rr2,2)-comd  COMs(rr2,2)+comd])
            ylim([COMs(rr2,1)-comd  COMs(rr2,1)+comd])
            
            subplot(212) % plot on inhibitImage, the current ROI and its too nearby ROIs
            plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'r', 'linewidth', 1.5);
            xlim([COMs(rr2,2)-comd  COMs(rr2,2)+comd])
            ylim([COMs(rr2,1)-comd  COMs(rr2,1)+comd])
            if ~isempty(nearbyROIs)
                for nnb = nearbyROIs'
                    plot(CCgcamp{nnb}(2,:), CCgcamp{nnb}(1,:), 'y', 'linewidth', 1.5)
                end
            end
            % axis image
            
            %         ch = 0;
            %         while ch~=13
            
            % lines will be red for neurons identified as inhibitory.
            if isnan(inhibitRois(rr2))
                % Plot the trace
                %             t1 = activity_man_eftMask_ch1(:,rr2);
                %             t2 = activity_man_eftMask_ch2(:,rr2);
                %                 crr = corr(t1, t2);
                figure(ftrace), cla
                plot(C(rr2,:))
                
                % If there are any too close ROIs (which most likely are
                % all the same ROI) plot them.                
                if ~isempty(nearbyROIs)
                    hold on
                    for nnb = nearbyROIs'
                        plot(C(nnb,:))
                        ccf = corrcoef(C(rr2,:), C(nnb,:)); fprintf('corrcoef w %i = %.2f\n', nnb, ccf(2))
                    end
                    fprintf('\n')
                    hold off
                end
                
                %             ht = plot(t1); hold on
                %             ht2 = plot(t2);
                xlim([0  size(C, 2)])
                %             title(crr(rr2))
                
                
                % Plot ROI contour on the image
                figure(fimag)
                h = plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'r');
                title(sprintf('sig/surr = %.2f', roi2surr_sig(rr2)), 'color', 'r')
                
%                 if ~isempty(nearbyROIs)
%                     hold on
%                     hn = plot(COMs(nearbyROIs,2), COMs(nearbyROIs,1), 'r.');
%                     h = [h, hn];
%                 end
                    
                %             title(sprintf('sig/surr = %.2f   corr = %.3f', roi2surr_sig(rr2), crr(rr2)), 'color', 'r')
                
            else
                figure(fimag)
                h = plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'y');
                title(sprintf('sig/surr = %.2f', roi2surr_sig(rr2)), 'color', 'k')
            end
            
            
            if keyEval
                ch = getkey;
                pause(.1); delete(h)
                
                % if enter, then go to next roi
                if ch==13
                    %{
                q = input('happy? ');
                if isempty(q)
                    inhibitEval(rr2) = true;
                end
                    %}
                    rr = rr+1;
                    %                 break
                    
                    
                    % if number 0 pressed, you want to assign this unsure ROI as excit.
                elseif ch==48
                    unsureEval(rr2) = 0; % rr2 is index in the all neurons array (not the inhibit neurons array).
                    fprintf('Reset as excit\n')
                    
                    % if number 1 pressed, you want to assign this unsure ROI as inhibit.
                elseif ch==49
                    unsureEval(rr2) = 1;
                    fprintf('Reset as inhibit\n')
                    
                    % if escape button, stop showing ROIs
                elseif ch==27
                    rr = length(CCgcamp);
                    break
                    
                    % if any key other than enter and escape, then keep deleting and reploting the same roi
                else
                    ch = getkey;
                end
                
            else
                figure(figh), drawnow
                figure(fimag), drawnow
                figure(ftrace), drawnow
                pause
                delete(h)
                rr = rr+1;
            end
            %         end
            %         rr = rr+1;
        end
        
        % now correct unsureRois based on your evaluation.
        fprintf('%i of unsure ROIs are reset as excitatory.\n', sum(unsureEval==0))
        fprintf('%i of unsure ROIs are reset as inhibitory.\n', sum(unsureEval==1))
        inhibitRois_new(unsureEval==0) = 0; % these ROIs are misidentified as unsure, and must be excit.
        inhibitRois_new(unsureEval==1) = 1; % these ROIs are misidentified as unsure, and must be inhibit.
    end
    
    
    %% Plot and evaluate inhibitory neurons
    
    if assessClass_unsure_inh_excit(2)
        cprintf('-blue', '------- Evaluating inhibitory neurons -------\n')
        if keyEval
            %     disp('Red contours are inhibitory. Yellow: excitatory')
            disp('Esc: quit evaluation.')
            disp('Other keys: keep showing the same ROI.')
            disp('When contour is shown :')
            disp('... press Enter if happy with classification and want to see next ROI.')
            disp('... press 0 if ROI must be excit.')
            disp('... press 2 if unsure about classification.')
            cprintf('red', 'DO NOT CLICK OR THE COMPUTER WILL HANG!! only use keyboard keys!!\n')
        else
            disp('Press any key to go through neurons one by one.')
        end
        
        fimag = figure;
        set(fimag, 'position', [1   -18   693   610]); %[28   133   805   658]) % get(groot, 'screensize'))
        set(gca, 'position', [0.0519    0.0656    0.9019    0.8951])
        imagesc(inhibitImage)
        %     imagesc(normImage(inhibitImage))
        hold on
        
        ftrace = figure('position', [383   682   987   236], 'name', 'C is plotted');
        
        inhibitEval = NaN(1, length(CCgcamp));
        rr = 1;
        while rr <= length(inhibit_inds_lo2hi); % length(CCgcamp)
            
            %         if plotInhFirst
            %             rr2 = inds_inh_exc(rr); % first plot all inhibit neurons, then all excit neurons.
            rr2 = inhibit_inds_lo2hi(rr); % rr2 is the index in the array including all neurons
            
%             fprintf('%.2f, %.2f, %.2f, %.2f\n', [cost_all(rr2), x_all(rr2,1), x_all(rr2,3), x_all(rr2,5)])
            
            nearbyROIs = findNearbyROIs(COMs, COMs(rr2,:), 6); nearbyROIs(nearbyROIs==rr2) = [];
            %         else
            %             rr2 = rr; % plot ROIs in the original order
            %         end
            
            set(fimag, 'name', sprintf('ROI %d. medImage of inhibitory channel. Use Esc to quit! ', rr2))
%             set(fimag, 'name', sprintf('ROI %d. Sig/Surr threshold = %.2f. medImage of inhibitory channel. Use Esc to quit! ', rr2, sigTh))
            
            % zoom on the gcamp channel image so you get an idea of the surrounding ROIs.            
            figure(figh)
            subplot(211)
            plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'r', 'linewidth', 1.5)
            comd = 20;
            xlim([COMs(rr2,2)-comd  COMs(rr2,2)+comd])
            ylim([COMs(rr2,1)-comd  COMs(rr2,1)+comd])
            
            subplot(212) % plot on inhibitImage, the current ROI and its too nearby ROIs
            plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'r', 'linewidth', 1.5);
            xlim([COMs(rr2,2)-comd  COMs(rr2,2)+comd])
            ylim([COMs(rr2,1)-comd  COMs(rr2,1)+comd])
            if ~isempty(nearbyROIs)
                for nnb = nearbyROIs'
                    plot(CCgcamp{nnb}(2,:), CCgcamp{nnb}(1,:), 'y', 'linewidth', 1.5)
                end
            end            
            
            %         ch = 0;
            %         while ch~=13
            
            % lines will be red for neurons identified as inhibitory.
            if inhibitRois(rr2)
                % Plot the trace
                %             t1 = activity_man_eftMask_ch1(:,rr2);
                %             t2 = activity_man_eftMask_ch2(:,rr2);
                %                 crr = corr(t1, t2);
                figure(ftrace), cla
                plot(C(rr2,:))
                %             ht = plot(t1); hold on
                %             ht2 = plot(t2);
                xlim([0  size(C, 2)])
                %             title(crr(rr2))
                
                
                % Plot ROI contour on the image
                figure(fimag)
                h = plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'r');
                title(sprintf('sig/surr = %.2f', roi2surr_sig(rr2)), 'color', 'r')
                %             title(sprintf('sig/surr = %.2f   corr = %.3f', roi2surr_sig(rr2), crr(rr2)), 'color', 'r')
                
            else
                figure(fimag)
                h = plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'y');
                title(sprintf('sig/surr = %.2f', roi2surr_sig(rr2)), 'color', 'k')
            end
            
            if keyEval
                ch = getkey;
                delete(h)
                
                % if enter, then go to next roi
                if ch==13
                    %{
                q = input('happy? ');
                if isempty(q)
                    inhibitEval(rr2) = true;
                end
                    %}
                    rr = rr+1;
                    %                 break
                    
                    
                % if number 0 pressed, you want to reassign this ROI as excit.
                elseif ch==48
                    inhibitEval(rr2) = 0; % rr2 is index in the all neurons array (not the inhibit neurons array).
                    fprintf('Reset as excit\n')
                    
                    % if number 2 pressed, you are unsure if this neuron is an inhibit neuron.
                elseif ch==50
                    inhibitEval(rr2) = 2;
                    fprintf('Reset as unsure\n')
                    
                    % if escape button, stop showing ROIs
                elseif ch==27
                    rr = length(CCgcamp);
                    break
                    
                    % if any key other than enter and escape, then keep deleting and reploting the same roi
                else
                    ch = getkey;
                end
                
            else
                figure(figh), drawnow
                figure(fimag), drawnow
                figure(ftrace), drawnow
                pause
                delete(h)
                rr = rr+1;
            end
            %         end
            %         rr = rr+1;
        end
        
        % now correct inhibitRois based on your evaluation.
        %     inhibitRois_new = double(inhibitRois);
        fprintf('%i of inhibitory ROIs are reset as excitatory.\n', sum(inhibitEval==0))
        fprintf('%i of inhibitory ROIs are reset as unknown.\n', sum(inhibitEval==2))
        inhibitRois_new(inhibitEval==0) = 0; % these ROIs are misidentified as inhibit, and must be excit.
        inhibitRois_new(inhibitEval==2) = nan; % we don't know whether to classify these ROIs as excit or inhibit.
    end
    
    
    %% Plot and evaluate excitatory neurons
    
    if assessClass_unsure_inh_excit(3)
        cprintf('-blue', '------- Evaluating excitatory neurons -------\n')
        if keyEval
            %     disp('Red contours are inhibitory. Yellow: excitatory')
            disp('Esc: quit evaluation.')
            disp('Other keys: keep showing the same ROI.')
            disp('When contour is shown :')
            disp('... press Enter if happy with classification and want to see next ROI.')
            disp('... press 1 if ROI must be inhibit.')
            disp('... press 2 if unsure about classification.')
            cprintf('red', 'DO NOT CLICK OR THE COMPUTER WILL HANG!! only use keyboard keys!!\n')
        else
            disp('Press any key to go through neurons one by one.')
        end
        
        fimag = figure;
        set(fimag, 'position', [1   -18   693   610]); %[28   133   805   658]) % get(groot, 'screensize'))
        set(gca, 'position', [0.0519    0.0656    0.9019    0.8951])
        imagesc(inhibitImage)
        hold on
        
        ftrace = figure('position', [383   682   987   236], 'name', 'C is plotted');
        set(gca, 'position', [0.0290    0.1637    0.9524    0.7119])
        
        excitEval = NaN(1, length(CCgcamp));
        rr = 1;
        while rr <= length(excit_inds_hi2lo) % length(CCgcamp)
            
            %         if plotInhFirst
            %             rr2 = inds_inh_exc(rr); % first plot all inhibit neurons, then all excit neurons.
            rr2 = excit_inds_hi2lo(rr);
            
%             fprintf('%.2f, %.2f, %.2f, %.2f\n', [cost_all(rr2), x_all(rr2,1), x_all(rr2,3), x_all(rr2,5)])
            
            nearbyROIs = findNearbyROIs(COMs, COMs(rr2,:), 6); nearbyROIs(nearbyROIs==rr2) = [];
            %         else
            %             rr2 = rr; % plot ROIs in the original order
            %         end
            
            set(fimag, 'name', sprintf('ROI %d. medImage of inhibitory channel. Use Esc to quit! ', rr2))
%             set(fimag, 'name', sprintf('ROI %d. Sig/Surr threshold = %.2f. medImage of inhibitory channel. Use Esc to quit! ', rr2, sigTh))
            
            % zoom on the gcamp channel image so you get an idea of the surrounding ROIs.            
            figure(figh)
            subplot(211)
            plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'r', 'linewidth', 1.5)
            comd = 20;
            xlim([COMs(rr2,2)-comd  COMs(rr2,2)+comd])
            ylim([COMs(rr2,1)-comd  COMs(rr2,1)+comd])
            
            subplot(212) % plot on inhibitImage, the current ROI and its too nearby ROIs
            plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'r', 'linewidth', 1.5);
            xlim([COMs(rr2,2)-comd  COMs(rr2,2)+comd])
            ylim([COMs(rr2,1)-comd  COMs(rr2,1)+comd])
            if ~isempty(nearbyROIs)
                for nnb = nearbyROIs'
                    plot(CCgcamp{nnb}(2,:), CCgcamp{nnb}(1,:), 'y', 'linewidth', 1.5)
                end
            end               
            
            %         ch = 0;
            %         while ch~=13
            
            % lines will be red for neurons identified as inhibitory.
            if inhibitRois(rr2)
                figure(fimag)
                h = plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'r');
                title(sprintf('sig/surr = %.2f', roi2surr_sig(rr2)), 'color', 'r')
            else
                %             t1 = activity_man_eftMask_ch1(:,rr2);
                %             t2 = activity_man_eftMask_ch2(:,rr2);
                %                 crr = corr(t1, t2);
                figure(ftrace), cla
                plot(C(rr2,:))
                %             ht = plot(t1); hold on
                %             ht2 = plot(t2);
                xlim([0  size(C, 2)])
                %             title(crr(rr2))
                
                
                figure(fimag)
                h = plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'y');
                title(sprintf('sig/surr = %.2f', roi2surr_sig(rr2)), 'color', 'k')
                %             title(sprintf('sig/surr = %.2f   corr = %.3f', roi2surr_sig(rr2), crr(rr2)), 'color', 'k')
            end
            
            if(keyEval)
                ch = getkey;
                delete(h)
                
                % if enter, then go to next roi
                if ch==13
                    %{
                q = input('happy? ');
                if isempty(q)
                    inhibitEval(rr2) = true;
                end
                    %}
                    rr = rr+1;
                    %                 break
                    
                    
                % if number 1 pressed, you want to reassign this ROI as inhibit.
                elseif ch==49
                    excitEval(rr2) = 0;
                    fprintf('Reset as inhibit\n')
                    
                    % if number 2 pressed, you are unsure if this neuron is an excit neuron.
                elseif ch==50
                    excitEval(rr2) = 2;
                    fprintf('Reset as unsure\n')
                    
                    % if escape button, stop showing ROIs
                elseif ch==27
                    rr = length(CCgcamp);
                    break
                    
                    % if any key other than enter and escape, then keep deleting and reploting the same roi
                else
                    ch = getkey;
                end
            else
                figure(figh), drawnow
                figure(fimag), drawnow
                figure(ftrace), drawnow
                pause
                delete(h)
                rr = rr+1;
            end
            %         end
            %         rr = rr+1;
        end
        
        % now correct inhibitRois based on your evaluation.
        %%%     inhibitRois_new = inhibitRois;
        fprintf('%i of excitatory ROIs are reset as inhibitory.\n', sum(excitEval==0))
        fprintf('%i of excitatory ROIs are reset as unknown.\n', sum(excitEval==2))
        inhibitRois_new(excitEval==1) = 1; % these ROIs are misidentified as excit, and must be inhibit.
        inhibitRois_new(excitEval==2) = nan; % we don't know whether to classify these ROIs as excit or inhibit.
    end
    
    
    %% Finally reset inhibitRois after evaluation is done.
    
    inhibitRois = inhibitRois_new;
    
%     fract = mean(inhibitRois==1); % fraction of ch2 neurons also present in ch1.
    cprintf('blue', '%d inhibitory; %d excitatory; %d unsure neurons in gcamp channel.\n', sum(inhibitRois==1), sum(inhibitRois==0), sum(isnan(inhibitRois)))

end


%% some good plots

%{
meas = roi2surr_sig';
% meas = bs;
figure; plot(sort(meas))
hold on

thq = .8;
plot([1 length(meas)], [quantile(meas, thq) quantile(meas, thq)], 'g')
% plot([1 length(meas)], [quantile(meas, thq) quantile(meas, thq)], 'g')

sigTh = quantile(meas, thq);
% sigTh = 1.2;
q = [min(meas) sigTh max(meas)+1];
% q = [min(meas) quantile(meas, .75)  sigTh  max(meas)+1];

normims = 1;
im = inhibitImage;
if normims
    im = normImage(im);
end


for iq = length(q)-2 : length(q)-1 % 1:length(q)-1
    f = find(meas >= q(iq) & meas < q(iq+1))'; length(f)
    
    figure;
    set(gcf, 'name', num2str(max(meas(f))))
    imagesc(im), freezeColors, colorbar, title('nonZero: ch1 - gamalSlope*ch2')
    
    for rr = f  % ibs'
%         ax1 = [ax1, gca];
        hold on,
        % a = plot(COMs(rr,2),COMs(rr,1), 'r.');
        a = plot(CCgcamp{rr}(2,:), CCgcamp{rr}(1,:), 'r');
        
        % pause
        % delete(a)
    end
    
end
%}

%%
%{
function plotCrr(crr, inhibitRois, roi2surr_sig)

% compute for each correlation bin the fraction of excitatory neurons with
% that correlation and the fraction of inhibitory neurons with that
% correlation.
figure('position', [1248        -162         560         420]);

[n,v,i] = histcounts(crr,0:.1:1);
ex = NaN(1, length(v));
in = NaN(1, length(v)-1);
for iv = 1:length(v);
    a = inhibitRois(i==iv);
    ex(iv) = sum(a==0)/sum(inhibitRois==0);
    in(iv) = sum(a==1)/sum(inhibitRois);
end

% where in goes above ex, choose that as that threshold for crr of ch1 and
% ch2. this is the threshold that tells us positive on ch1 is most likely
% due to bleedthrough, hence this roi should not be called inhibit.
subplot(211), hold on
plot(v, ex)
plot(v, in)
legend('excit', 'inhibit')
xlabel('correlation btwn ch1 and ch2')
ylabel('fraction of neurons')


%% compute the median of correlation for excit or inhibit neurons for each roiSig2surr bin.

subplot(212); hold on

for ei = 1:2
    
    if ei==1
        b = find(inhibitRois==0); % index of excit neurons in the array that includes all neurons.
    else
        b = find(inhibitRois==1);
    end
    
    a = roi2surr_sig(b); % roi2surr_sig for either excit or inhibit neurons.
    
    % v = [floor(min(a)): .25: 3, max(a)];
    v = linspace(min(a), max(a), 10);
    
    [n,v,i] = histcounts(a, v); % i: bin index of excit or inhibit neurons (in array a). eg i(34)=2, indicates the 34th inhibit (excit) neuron has its roi2surr_sig in the 2nd bin of v.
    
    m = NaN(1, length(v));
    se = NaN(1, length(v));
    for ii = 1:length(v)
        m(ii) = median(crr(b(i==ii)));
%         m(ii) = mean(crr(b(i==ii)));
        se(ii) = std(crr(b(i==ii)));
    end
    
%     plot(v,m)
    errorbar(v,m,se)
    xlabel('roi2surr\_sig')
    ylabel('corr btwn ch1 and ch2 (median of neurons)')
    
end
legend('excit', 'inhibit')
%}
