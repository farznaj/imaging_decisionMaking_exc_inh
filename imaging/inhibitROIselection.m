function [inhibitRois, roi2surr_sig] = inhibitROIselection(maskGcamp, medImageInhibit, sigTh, CCgcamp, showResults)
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


%% Set vars
if ~exist('showResults', 'var')
    showResults = false;
end

imHeight = size(maskGcamp,1);
imWidth = size(maskGcamp,2);
roi2surr_sig = NaN(1, size(maskGcamp,3));


%%
for rr = 1 : size(maskGcamp,3)
    
    %% compute roiSing: signal magnitude of ch2 ROI (maskGcamp) on ch1 image (medImageInhibit).
    
    roiMask = maskGcamp(:,:,rr); % mask of ch2 ROI
    
    roiIm = medImageInhibit .* roiMask; % image of ch2 ROI on ch1.
    % sum(roiIm(:)) / sum(roiMask(:))
    a = roiIm(roiIm~=0);
    roiSig = nanmean(a(:)); % signal magnitude of ch2 ROI on ch1 image.
    
    
    %% set surrMask : a square mask surrounding roiMask (mask of ch2 ROI)
    
    xl = [find(sum(roiMask), 1, 'first')  find(sum(roiMask), 1, 'last')];
    yl = [find(sum(roiMask,2), 1, 'first')  find(sum(roiMask,2), 1, 'last')];
    
    surr_sz = 1; % 5; % remember for 151029_003 you used 5.
    ccn_y = [max(yl(1)-surr_sz, 1)  max(yl(1)-surr_sz, 1)  min(yl(2)+surr_sz, imHeight) min(yl(2)+surr_sz, imHeight)];
    ccn_x = [max(xl(1)-surr_sz, 1)  min(xl(2)+surr_sz, imHeight)  min(xl(2)+surr_sz, imHeight)  max(xl(1)-surr_sz, 1)];
    ccn = [ccn_y ; ccn_x];
    
    maskn = maskSet({ccn}, imHeight, imWidth);
    
    surrMask = maskn - roiMask;
%     figure; imagesc(surrMask)
    
    
    %% compute surrSig: magnitude of ch1 image surrounding ch2 ROI.
    
    roiIm = medImageInhibit .* surrMask;
    a = roiIm(roiIm~=0);
    surrSig = nanmean(a(:));
    
    
    %% compute ratio of roi signal/surr sig.
    
    roi2surr_sig(rr) = roiSig / surrSig;

    
end


%% Identify inhibitory neurons.

% It seems roi2surr_sig = 1.2 is a good threshold.
% sigTh = 1.2;
inhibitRois = roi2surr_sig > sigTh; % neurons in ch2 that are inhibitory. (ie present in ch1).
inhibitRois = double(inhibitRois); % you do this so the class is consistent with when you do manual evaluation (below)

fract = nanmean(inhibitRois); % fraction of ch2 neurons also present in ch1.
fprintf('%.2f: fraction of inhibitory neurons in gcamp channel.\n', fract)
  
    
%% Look at ch2 ROIs on ch1 image, 1 by 1.

if exist('CCgcamp', 'var') && showResults
    
    figure; hold on
    plot(roi2surr_sig)
    plot([0 500], [sigTh sigTh], 'r')
    xlabel('Neuron number')
    ylabel('ROI / surround')
    
    
    %%
    plotInhFirst = 1; % if 1, first inhibitory ROIs will be plotted then excit. If 0, ROIs will be plotted in their original order.    
   
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
    
    inds_inh_exc = [inhibit_inds_lo2hi, excit_inds_hi2lo]; % 1st see inhibit (ordered from low to high value of roi2surr_sig), then see excit (ordered from high to low value of roi2surr_sig)
%     inds_inh_exc = [find(inhibitRois), find(~inhibitRois)]; % 1st see inhibit, then excit neurons


    %%
    disp('Evaluate inhibitory neuron identification. Figure shows medImage of inhibit neurons.')
    disp('Red indiciates inhibitory. Yellow: excitatory')
    disp('Enter: show the next ROI.')
    disp('Esc: quit')
    disp('Other keys: keep showing the same ROI.')
    disp('Press 0 if the classification is wrong.')
    disp('Press 2 if unsure about the classification.')
    
    
    %% Plot and evaluate inhibitory neurons
    
    figure;
    set(gcf, 'position', [44  62  1014  846]); %[28   133   805   658]) % get(groot, 'screensize'))    
    imagesc(medImageInhibit)
    hold on
    
    inhibitEval = NaN(1, length(CCgcamp));
    rr = 1;
    while rr <= length(inhibit_inds_lo2hi); % length(CCgcamp)
        
        if plotInhFirst
%             rr2 = inds_inh_exc(rr); % first plot all inhibit neurons, then all excit neurons.
            rr2 = inhibit_inds_lo2hi(rr); % rr2 is the index in the array including all neurons 
        else
            rr2 = rr; % plot ROIs in the original order
        end
        
        set(gcf, 'name', sprintf('ROI %d. Sig/Surr threshold = %.2f. medImage of inhibitory channel. Use Esc to quit! ', rr2, sigTh))
        ch = 0;
        while ch~=13
            
            % lines will be red for neurons identified as inhibitory.
            if inhibitRois(rr2)
                h = plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'r');
                title(sprintf('sig/surr = %.2f', roi2surr_sig(rr2)), 'color', 'r')
            else
                h = plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'y');
                title(sprintf('sig/surr = %.2f', roi2surr_sig(rr2)), 'color', 'k')
            end
                        
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
                break
                
                
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
        end
        
        rr = rr+1;
    end    
    
    % now correct inhibitRois based on your evaluation.
    aa = double(inhibitRois); 
    aa(inhibitEval==0) = 0; % these ROIs are misidentified as inhibit, and must be excit.
    aa(inhibitEval==2) = nan; % we don't know whether to classify these ROIs as excit or inhibit.


    %% Plot and evaluate excitatory neurons
    
    figure;
    set(gcf, 'position', [44  62  1014  846]); %, [28   133   805   658]) % get(groot, 'screensize'))
    
    imagesc(medImageInhibit)
    hold on
    
    excitEval = NaN(1, length(CCgcamp));
    rr = 1;
    while rr <= length(excit_inds_hi2lo) % length(CCgcamp)
        
        if plotInhFirst
%             rr2 = inds_inh_exc(rr); % first plot all inhibit neurons, then all excit neurons.
            rr2 = excit_inds_hi2lo(rr);
        else
            rr2 = rr; % plot ROIs in the original order
        end
        
        set(gcf, 'name', sprintf('ROI %d. Sig/Surr threshold = %.2f. medImage of inhibitory channel. Use Esc to quit! ', rr2, sigTh))
        ch = 0;
        while ch~=13
            
            % lines will be red for neurons identified as inhibitory.
            if inhibitRois(rr2)
                h = plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'r');
                title(sprintf('sig/surr = %.2f', roi2surr_sig(rr2)), 'color', 'r')
            else
                h = plot(CCgcamp{rr2}(2,:), CCgcamp{rr2}(1,:), 'y');
                title(sprintf('sig/surr = %.2f', roi2surr_sig(rr2)), 'color', 'k')
            end
                        
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
                break
                
                
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
        end
        
        rr = rr+1;
    end

    % now correct inhibitRois based on your evaluation.
%%%     aa = inhibitRois; 
    aa(excitEval==0) = 1; % these ROIs are misidentified as excit, and must be inhibit.
    aa(excitEval==2) = nan; % we don't know whether to classify these ROIs as excit or inhibit.

    
    %% Reset inhibitRois after evaluation is done.
    
    inhibitRois = aa;
    
    
end



