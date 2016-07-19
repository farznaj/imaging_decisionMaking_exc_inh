function [matchedROIidx_maskMethod, maskOverlapMeasure] = matchROIs_sumMask_fast(refMask, toMatchMask, COMs)
it needs work so it generalizes to when refMask and toMatchMask are different, hence 2 different COMs should be given as inputs.
% Faster version of matchROIs_sumMask: it uses COMs to find nearby masks
% and then sets sum of masks (which is a slow process) only for those
% nearby masks. In contrast to matchROIs_sumMask which sets sum of masks
% for all pairs of masks.
%
% Use sum of masks method to find matching ROIs between 2 sets of masks. It
% computes the sum of mask between nearby paris of ROIs and identifies the ROI
% whose mask has the max sum value as the matching ROI. 
%
% Output, matchedROIidx_maskMethod, includes the indeces of ROIs in toMatchMask
% that matche each ROI in refMask.
%



maskOverlapMeasure = NaN(size(toMatchMask,3), size(refMask,3));

for iref = 1:size(COMs, 1)
    b = abs(bsxfun(@minus, COMs(iref,:), COMs));
    bb = sum(abs(b), 2);
    %     [diffCOMs(iref), imatch(iref)] = min(bb); % which ROI in mask{2} matched ROI iref in mask{1}
    
    % For ROIs that are near each other use the sum_mask method to find the
    % best matching one.
    f = find(bb<10)'; % this threshold is important
    if ~isempty(f)
%         clear maskOverlapMeasure
        %     maskOverlapMeasure = NaN(size(toMatchMask,3), 1);
        for itoMatch = f
            sumMask = sparse(toMatchMask(:,:,itoMatch)) + sparse(refMask(:,:,iref));
            maskOverlapMeasure(itoMatch, iref) = sum(sumMask(:)>1);
        end
    end
end

% figure; imagesc(maskOverlapMeasure)


%% Set sum of masks for all pairs of masks

maskOverlapMeasure = NaN(size(toMatchMask,3), size(refMask,3));

fprintf('Setting sum of masks for all pairs of ROIs\n')
for iref = 1:size(refMask,3)    
    for itoMatch = 1:size(toMatchMask,3)
        sumMask = sparse(toMatchMask(:,:,itoMatch)) + sparse(refMask(:,:,iref));
        maskOverlapMeasure(itoMatch, iref) = sum(sumMask(:)>1);        
    end
    fprintf('\t%d out of %d\n', iref, size(refMask,3))
end


%% Set the best ROI match as the one with max overlap

% matchedROIidx_maskMethod : shows which toMatch ROI had max overlap with each ref ROI.

% matchedROIidx_maskMethod(i) = j means ROI j of toMatchMask matched ROI i of refMask. 
% matchedROIidx_maskMethod(i) = NaN means no ROI in toMatchMask matched the ref mask.

warning('Instead of using max, you need to define a threshold, otherwise even tiny overlaps will result in a matched ROI!')
[~, matchedROIidx_maskMethod] = max(maskOverlapMeasure);
matchedROIidx_maskMethod(~max(maskOverlapMeasure)) = NaN; % set to NaN those that had no match in Eft ROI.

% figure; plot(matchedROIidx_maskMethod), xlabel('Ref ROI index'), ylabel('toMatch ROI index')
% figure; plot(matchedROIidx_maskMethod), xlabel('Manual ROI index'), ylabel('matching Eft ROI index')

