% related to highlight patch and ROI patch correlation measure: 
% but instead of correlatin, I am looking at the mean pixel intensity of
% highlightPatch for pixels that fall inside and outside the ROI mask. 
% The idea is that ROIs with high mean hightlightPatch pixel outside the
% mask are problematic. 
%
% The following is great to find bad neurons, but most of it is picked by Andrea's measure as well.
% ft = (aveHighlightOutRoi>=.75); sum(ft)



%% Another way of finding discrepancies between hightlightPatch and roiPatch:
% subtract roiPatch from hightlightPath and take the mean of its abs.

highlightRoiDiff = nan(length(roiPatch), 1);
for i = 1:length(highlightRoiDiff)
    a = highlightPatchAvg{i}; am = max(a(:));
    b = roiPatch{i}; bm = max(b(:));
    %   imagesc(a/am - b/bm)
    c = (a/am - b/bm);
    highlightRoiDiff(i) = mean(abs(c(:)));
%     highlightRoiDiff(i) = mean(c(:));
end
figure; plot(highlightRoiDiff)

%{
ft = (~badAll & highlightRoiDiff>=.5);
% ft = (~badAll & cinall<=.3);
sum(ft)
rois2p = find(ft);
%}


%%
aveHighlightInRoi = nan(length(roiPatch), 1); % average of highlightPatch that is inside the roiPatch
aveHighlightOutRoi = nan(length(roiPatch), 1); % average of highlightPatch that is outside the roiPatch

for rr = 1:length(roiPatch)
    
    roiMask0 = roiPatch{rr}; % mask(:,:,rr); % mask of ch2 ROI 
    %{
    rr = 459;
    figure; subplot(211), imagesc(roiPatch{rr})
    subplot(212), imagesc(highlightPatchAvg{rr})
    %}
    
    % Use the mask instead of the non-zero values in roiPatch (A) to set
    % roiIm_in and roiIm_outs.
    
    %{
    siz = [imHeight, imWidth];
    AMat = reshape(A(:,rr), siz);
    
    [i, j] = find(AMat);
    pad = 1;
    
    xRange = [max(min(j)-pad, 1) min(max(j)+pad, siz(2))];
    yRange = [max(min(i)-pad, 1) min(max(i)+pad, siz(1))];
    
    maskNeur = mask(yRange(1):yRange(2), xRange(1):xRange(2), rr);
    
    roiMask0 = double(maskNeur);
%     figure; imagesc(roiMask0)
    %}
    
    %% highlightPatchAvg signal inside the mask
    
    roiMask = roiMask0;
    % roiMask(roiMask==0) = nan;
    % roiMask(~isnan(roiMask)) = 1;
    roiMask(roiMask~=0) = 1;
    
    roiIm_in = highlightPatchAvg{rr} .* roiMask; % image of ch2 ROI on ch1. %
%     figure; imagesc(roiIm_in)
    
    %% highlightPatchAvg signal outside the mask
    
    roiMask = roiMask0;
    roiMask(roiMask~=0) = NaN;
    roiMask(roiMask==0) = 1;
    roiMask(isnan(roiMask)) = 0;
    
    roiIm_out = highlightPatchAvg{rr} .* roiMask; % image of ch2 ROI on ch1. %
%     figure; imagesc(roiIm_out)    
    
    %% Average of highlightPatchAvg pixels inside and outside the mask
    
%     cin = corr2(roiIm_in, roiMask0); %, 'rows', 'pairwise');
    a = roiIm_in(:);
    a = a/max(highlightPatchAvg{rr}(:));
    cin = mean(a(a~=0));    
    
    
%     cout = corr2(roiIm_out, roiMask0); %, 'rows', 'pairwise');    
    a = roiIm_out(:);
    a = a/max(highlightPatchAvg{rr}(:));
    cout = mean(a(a~=0)); % 

    % [cin, cout] 
    
    %%
    aveHighlightInRoi(rr) = cin;
    aveHighlightOutRoi(rr) = cout;
    
end


%%
figure; 
subplot(211), hold on
plot(aveHighlightInRoi)
plot(aveHighlightOutRoi, 'r')

subplot(212)
plot(aveHighlightInRoi ./ aveHighlightOutRoi)

% a = (cinall ./ coutall);
% sum(a>-1.06)

%% 

ft = (aveHighlightOutRoi>=.75); sum(ft)
ft = (~badAll & aveHighlightOutRoi>=.75);
% ft = (~badAll & highlightRoiDiff>=.5);
% ft = (~badAll & cinall<=.3);
sum(ft)
rois2p = find(ft);

%%
rois2p = find(~badAll & aveHighlightOutRoi > .7 & fitnessNow >= -30 & fitnessNow <- 20); 
length(rois2p)




