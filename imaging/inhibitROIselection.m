function [inhibitRois, roi2surr_sig, sigTh] = inhibitROIselection(maskGcamp, medImage, sigTh, CCgcamp, showResults, figh, COMs, activity_man_eftMask_ch1, activity_man_eftMask_ch2)
% inhibitRois = inhibitROIselection(maskGcamp, medImageInhibit, sigTh, CCgcamp);
%
% Identifies inhibitory neurons on a gcamp channel (containing both
% excitatory and inhibitory neurons, "maskGcamp") by using the average image
% of inhibitory channel ("medImageInhibit").
%
% INPUTS:
% maskGcamp: imHeight x imWidth x units, mask of gcamp channel neurons, that contains both excitatory and inhibitory neurons.
% medImageInhibit: median image of channel with inhibitory neurons (ie tdtomato channel).
% sigTh: signal to noise threshold for identifying inhibitory neurons on tdtomato channel. eg. sigTh = 1.2;
% CCgcamp: optional, cell array: 1 x units. coordinates of ROI contours on
%   gcamp channel. If provided, a plot will be made to assess the output of the algorithm.
% showResults: if 1, plots will be made to assess the classification of inhibitory neurons. If 1, you will need the following 2 vars.
% figh: handle to figure that shows all ROIs on the gcamp channel.
% COMs: center of mass of ROIs
%
% OUTPUTS:
% inhibitRois: index of ROIs in gcamp mask that are inhibitory (ie they have a match on medImageInhibit.)
%
% How the algorithm works?
% it computes a measure of signal/noise for each ch2 ROI (gcamp channel) on ch1 image (inhibitory channel),
% by applying the contour of ch2 on the medImage of ch1 and computing signal_magnitude_inside_contour / signal_magnitude_outside_contour
%
% e.g:
% [CCgcamp, ~, ~, maskGcamp] = setCC_cleanCC_plotCC_setMask(spatialComp, imHeight, imWidth, contour_threshold, medImage{2});
% medImageInhibit = medImage{1};
% sigTh = 1.2;


medImageInhibit = medImage{1};


%%
frs = size(activity_man_eftMask_ch2,1);
nn = size(activity_man_eftMask_ch2,2);

Xs = mat2cell(activity_man_eftMask_ch2, frs, ones(1,nn));
Ys = mat2cell(activity_man_eftMask_ch1, frs, ones(1,nn));

[a, bs] = regressCommonSlopeModel(Xs, Ys);


im2 = medImage{1} - a*medImage{2};
medImageInhibit = im2; 
figure; imagesc(medImageInhibit)

medImageInhibit(medImageInhibit < 0) = 0;
figure; imagesc(medImageInhibit)


%% Set vars
 
if ~exist('showResults', 'var')
    showResults = false;
end

imHeight = size(maskGcamp,1);
imWidth = size(maskGcamp,2);
roi2surr_sig = NaN(1, size(maskGcamp,3));
roi2surr_sig_num = NaN(1, size(maskGcamp,3));


%%

for rr = 1 : size(maskGcamp,3)
    
    % Compute roiSing: signal magnitude of ch2 ROI (maskGcamp) on ch1 image (medImageInhibit).
    
    roiMask = maskGcamp(:,:,rr); % mask of ch2 ROI % figure; imagesc(roiMask)
    % set pixels outside the ROI to nan. use this if you are doing roiSig = nanmean(roiIm(:)); below
    roiMask = double(roiMask);
    roiMask(~roiMask) = NaN;
    
    roiIm = medImageInhibit .* roiMask; % image of ch2 ROI on ch1. % figure; imagesc(roiIm)
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
    roi2surr_sig(rr) = roiSig;
    
    
    %% Set surrMask : a square mask surrounding roiMask (mask of ch2 ROI)
    %{
    xl = [find(sum(roiMask), 1, 'first')  find(sum(roiMask), 1, 'last')];
    yl = [find(sum(roiMask,2), 1, 'first')  find(sum(roiMask,2), 1, 'last')];
    
    surr_sz = 5; % 1; 5; % remember for 151029_003 you used 5.
    ccn_y = [max(yl(1)-surr_sz, 1)  max(yl(1)-surr_sz, 1)  min(yl(2)+surr_sz, imHeight) min(yl(2)+surr_sz, imHeight)];
    ccn_x = [max(xl(1)-surr_sz, 1)  min(xl(2)+surr_sz, imHeight)  min(xl(2)+surr_sz, imHeight)  max(xl(1)-surr_sz, 1)];
    ccn = [ccn_y ; ccn_x];
    
    maskn = maskSet({ccn}, imHeight, imWidth, 0);
    
    surrMask = maskn - roiMask;
    %     figure; imagesc(surrMask)
    
    
    %% Compute surrSig: magnitude of ch1 image surrounding ch2 ROI.
    
    roiIm = medImageInhibit .* surrMask;
    ss = roiIm(:)~=0;
    %     s = sum(ss); % number of non-zero pixels in the image of ch2 surround ROI on ch1.
    %     surrSig = sum(roiIm(:)~=0);
    a = roiIm(ss);
    surrSig = nanmedian(a(:));
%     surrSig = nanmean(a(:));
    
    
    %% Compute ratio of roi signal/surr sig.
    
    roi2surr_sig(rr) = roiSig / surrSig;
    %}
    
end

roi2surr_sig(isnan(roi2surr_sig)) = 0; % There are ROIs that had 0 signal for both the ROI and its surrounding, so they should be classified as excit.

q_num = quantile(roi2surr_sig_num, 9)
q_sig_orig = quantile(roi2surr_sig, 9)


%% Set to 0 the value of roi2surr_sig if an ROI has few positive pixels.
%
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


%% Define a threshold for identifying inhibitory neurons

qTh = .5; %.8; % .1;
cprintf('magenta', 'sigTh defined as %.2f quantile of non-zero values in roi2surr_sig\n', qTh)

sigTh = quantile(roi2surr_sig(roi2surr_sig~=0), qTh);
% sigTh = quantile(roi2surr_sig, qTh);
fprintf('Using %.2f as the threshold for finding inhibitory ROIs.\n', sigTh)
% It seems roi2surr_sig = 1.2 is a good threshold.
% sigTh = 1.2;


%% Set inhibitory neurons

inhibitRois = roi2surr_sig > sigTh; % neurons in ch2 that are inhibitory. (ie present in ch1).
inhibitRois = double(inhibitRois); % you do this so the class is consistent with when you do manual evaluation (below)

fract = nanmean(inhibitRois); % fraction of ch2 neurons also present in ch1.
cprintf('blue', '%d: num, %.3f: fraction of inhibitory neurons in gcamp channel.\n', sum(inhibitRois), fract)


%%
sum(inhibitRois(roi2surr_sig_num < posPixTh))
inhibitRois(roi2surr_sig_num < posPixTh) = 0;

fract = nanmean(inhibitRois); % fraction of ch2 neurons also present in ch1.
cprintf('blue', '%d: num, %.3f: fraction of inhibitory neurons in gcamp channel.\n', sum(inhibitRois), fract)


%% Compute correlation in the activity of each ch2 ROI between ch2 movie and ch1 movie.
%
crr = NaN(1, length(CCgcamp));
for rr = 1:length(CCgcamp)
    t1 = activity_man_eftMask_ch1(:,rr);
    t2 = activity_man_eftMask_ch2(:,rr);
    crr(rr) = corr(t1, t2);
end
    

%% Plots related to crr and brightness
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

if exist('CCgcamp', 'var') && showResults
    
    figure('position', [1157 747-200 695 237]); hold on
    plot(sort(roi2surr_sig))
    plot([0 500], [sigTh sigTh], 'r')
    xlabel('Neuron number')
    ylabel('ROI / surround')
    
    if ~isvalid(figh), figh = figure; end
    
    %%
    
    doEval = 0; % Linux hangs with getKey... so make sure this is set to 0! % if 0 you will simply go though ROIs one by one, otherwise it will go to getKey and you will be able to change neural classification.    
    
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
    a = roi2surr_sig(~inhibitRois==1);
    [~, i] = sort(a, 'descend');
    
    f = find(~inhibitRois==1);
    excit_inds_hi2lo = f(i); % indeces of excit neurons in the array that includes all neurons, sorted from low to high value of roi2surr_sig
    % first elements (high values of roi2surr_sig) are problematic and may not be really excit.
    
    %     inds_inh_exc = [inhibit_inds_lo2hi, excit_inds_hi2lo]; % 1st see inhibit (ordered from low to high value of roi2surr_sig), then see excit (ordered from high to low value of roi2surr_sig)
    %     inds_inh_exc = [find(inhibitRois), find(~inhibitRois)]; % 1st see inhibit, then excit neurons
    
    
    plotInhFirst = 1; % if 1, first inhibitory ROIs will be plotted then excit. If 0, ROIs will be plotted in their original order.
    
    set(figh, 'position', [1237           6         560         420])
    
        
    %%
    disp('=====================')
    disp('Evaluate inhibitory neuron identification. Figure shows medImage of inhibit neurons.')
    %     disp('Red contours are inhibitory. Yellow: excitatory')
    disp('Esc: quit evaluation.')
    disp('Other keys: keep showing the same ROI.')
    disp('When contour is shown :')
    disp('... press Enter if happy with classification and want to see next ROI.')
    disp('... press 0 if classification is wrong.')
    disp('... press 2 if unsure about classification.')
    
    
    
    
    %% Plot and evaluate inhibitory neurons
    
    disp('------- evaluating inhibitory neurons -------')
    
    fimag = figure;
    set(fimag, 'position', [-104         -26        1014         820]); %[28   133   805   658]) % get(groot, 'screensize'))
    imagesc(medImageInhibit)
    hold on
    
    ftrace = figure('position', [946   494   987   236]);
    set(gca,'position', [0.0290    0.1637    0.9524    0.7119])
    
    inhibitEval = NaN(1, length(CCgcamp));
    rr = 1;
    while rr <= length(inhibit_inds_lo2hi); % length(CCgcamp)
        
        if plotInhFirst
            %             rr2 = inds_inh_exc(rr); % first plot all inhibit neurons, then all excit neurons.
            rr2 = inhibit_inds_lo2hi(rr); % rr2 is the index in the array including all neurons
        else
            rr2 = rr; % plot ROIs in the original order
        end
        
        set(fimag, 'name', sprintf('ROI %d. Sig/Surr threshold = %.2f. medImage of inhibitory channel. Use Esc to quit! ', rr2, sigTh))
        
        % zoom on the gcamp channel image so you get an idea of the surrounding ROIs.
        figure(figh)
        comd = 20;
        xlim([COMs(rr2,2)-comd  COMs(rr2,2)+comd])
        ylim([COMs(rr2,1)-comd  COMs(rr2,1)+comd])
        % axis image
        
        %         ch = 0;
        %         while ch~=13
        
        % lines will be red for neurons identified as inhibitory.
        if inhibitRois(rr2)
            t1 = activity_man_eftMask_ch1(:,rr2);
            t2 = activity_man_eftMask_ch2(:,rr2);
            %                 crr = corr(t1, t2);
            figure(ftrace), cla, hold on
            ht = plot(t1);
            ht2 = plot(t2);
            xlim([0  size(activity_man_eftMask_ch1, 1)])
            title(crr(rr2))
            
            figure(fimag)
            h = plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'r');
            title(sprintf('sig/surr = %.2f   corr = %.3f', roi2surr_sig(rr2), crr(rr2)), 'color', 'r')
            
        else
            figure(fimag)
            h = plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'y');
            title(sprintf('sig/surr = %.2f', roi2surr_sig(rr2)), 'color', 'k')
        end
        
        if doEval
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
                
                
                % if number 0 pressed, you want to exclude this ROI from inhibit neurons.
            elseif ch==48
                inhibitEval(rr2) = 0; % rr2 is index in the all neurons array (not the inhibit neurons array).
                
                
                % if number 2 pressed, you are unsure if this neuron is an inhibit neuron.
            elseif ch==50
                inhibitEval(rr2) = 2;
                
                
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
    aa = double(inhibitRois);
    fprintf('%i of inhibitory ROIs are reset as excitatory.\n', sum(inhibitEval==0))
    fprintf('%i of inhibitory ROIs are reset as unknown.\n', sum(inhibitEval==2))
    aa(inhibitEval==0) = 0; % these ROIs are misidentified as inhibit, and must be excit.
    aa(inhibitEval==2) = nan; % we don't know whether to classify these ROIs as excit or inhibit.
    
    
    %% Plot and evaluate excitatory neurons
    
    disp('------- evaluating excitatory neurons -------')
    
    fimag = figure;
    set(gcf, 'position', [-104         -26        1014         820]); %, [28   133   805   658]) % get(groot, 'screensize'))
    imagesc(medImageInhibit)
    hold on
    
    ftrace = figure('position', [946   494   987   236]);
    set(gca, 'position', [0.0290    0.1637    0.9524    0.7119])
    
    excitEval = NaN(1, length(CCgcamp));
    rr = 1;
    while rr <= length(excit_inds_hi2lo) % length(CCgcamp)
        
        if plotInhFirst
            %             rr2 = inds_inh_exc(rr); % first plot all inhibit neurons, then all excit neurons.
            rr2 = excit_inds_hi2lo(rr);
        else
            rr2 = rr; % plot ROIs in the original order
        end
        
        set(fimag, 'name', sprintf('ROI %d. Sig/Surr threshold = %.2f. medImage of inhibitory channel. Use Esc to quit! ', rr2, sigTh))
        
        % zoom on the gcamp channel image so you get an idea of the surrounding ROIs.
        figure(figh)
        comd = 20;
        xlim([COMs(rr2,2)-comd  COMs(rr2,2)+comd])
        ylim([COMs(rr2,1)-comd  COMs(rr2,1)+comd])
        % axis image
        
        %         ch = 0;
        %         while ch~=13
        
        % lines will be red for neurons identified as inhibitory.
        if inhibitRois(rr2)
            figure(fimag)
            h = plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'r');
            title(sprintf('sig/surr = %.2f', roi2surr_sig(rr2)), 'color', 'r')
        else
            t1 = activity_man_eftMask_ch1(:,rr2);
            t2 = activity_man_eftMask_ch2(:,rr2);
            %                 crr = corr(t1, t2);
            figure(ftrace), cla, hold on
            ht = plot(t1);
            ht2 = plot(t2);
            xlim([0  size(activity_man_eftMask_ch1, 1)])
            title(crr(rr2))
            
            
            figure(fimag)
            h = plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'y');
            title(sprintf('sig/surr = %.2f   corr = %.3f', roi2surr_sig(rr2), crr(rr2)), 'color', 'k')
        end
        
        if(doEval)
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
                
                
                % if number 0 pressed, you want to exclude this ROI from excit neurons.
            elseif ch==48
                excitEval(rr2) = 0;
                
                
                % if number 2 pressed, you are unsure if this neuron is an excit neuron.
            elseif ch==50
                excitEval(rr2) = 2;
                
                
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
    %%%     aa = inhibitRois;
    fprintf('%i of excitatory ROIs are reset as inhibitory.\n', sum(excitEval==0))
    fprintf('%i of excitatory ROIs are reset as unknown.\n', sum(excitEval==2))
    aa(excitEval==0) = 1; % these ROIs are misidentified as excit, and must be inhibit.
    aa(excitEval==2) = nan; % we don't know whether to classify these ROIs as excit or inhibit.
    
    
    %% Finally reset inhibitRois after evaluation is done.
    
    inhibitRois = aa;
    
    
end



%%
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

