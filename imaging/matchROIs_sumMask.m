function [matchedROIidx_maskMethod, maskOverlapMeasure] = matchROIs_sumMask(refMask, toMatchMask)
% matchedROIidx_maskMethod = matchROIs_sumMask(refMask, toMatchMask)
%
% Use sum of masks method to find matching ROIs between 2 sets of masks. It
% computes the sum of mask between all paris of ROIs and identifies the ROI
% whose mask has the max sum value as the matching ROI. 
%
% Output, matchedROIidx_maskMethod, includes the indeces of ROIs in toMatchMask
% that matche each ROI in refMask.
%

%%
maskOverlapMeasure = NaN(size(toMatchMask,3), size(refMask,3));

for iref = 1:size(refMask,3)    
    for itoMatch = 1:size(toMatchMask,3)
        sumMask = sparse(toMatchMask(:,:,itoMatch)) + sparse(refMask(:,:,iref));
        maskOverlapMeasure(itoMatch, iref) = sum(sumMask(:)>1);        
    end
end


%% ROI matches based on the mask method.
[~, matchedROIidx_maskMethod] = max(maskOverlapMeasure);
matchedROIidx_maskMethod(~max(maskOverlapMeasure)) = NaN; % set to NaN those that had no match in Eft ROI.

% figure; plot(matchedROIidx_maskMethod), xlabel('Manual ROI index'), ylabel('matching Eft ROI index')

