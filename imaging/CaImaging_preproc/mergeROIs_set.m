function merged_ROIs = mergeROIs_set(mouse, imagingFolder, mdfFileNumber, thMerge, doEvaluate, saveMergedVar)
%
% Newest method for merging ROIs: uses Eftychios's method however instead
% of defining spatial overlap as A'*A, it uses your custom method, ie based
% on fraction overlap of masks driven from A (at .95 contour threshold).
%
% Example inputs:
%{
mouse = 'fni17';
imagingFolder = '151102';
mdfFileNumber = 1; % 1; % or tif major

thMerge = .5; % .4; % min of mean fraction overlap between 2 ROIs to count them as matching ROIs.
saveMergedVar = 0 ;% if 1, merged_ROIs will be appended to pnevFileName.
doEvaluate = 0; % if 1, spatial and temporal components of merged_ROIs will be displayed for evaluation of merging.

%}

%% Set mat file names

signalCh = 2; % because you get A from channel 2, I think this should be always 2.
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh);
% [~,f] = fileparts(pnevFileName);
% disp(f)


%% Load vars

load(imfilename, 'imHeight', 'imWidth')
load(pnevFileName, 'C', 'A')

% A = A_m;
% C = C_m;

%% Set mask and COMs from A

contour_threshold = .95;
if doEvaluate, load(imfilename, 'sdImage'); im = sdImage{2}; else im = []; end
% fprintf('Setting the mask for the gcamp channel....\n')
[CC, ~, COMs, mask] = setCC_cleanCC_plotCC_setMask(A, imHeight, imWidth, contour_threshold, im);
% title('ROIs shown on the sdImage of channel 2')


%% Fast way for computing maskOverlapMeasure

maskOverlapMeasure = NaN(size(mask,3), size(mask,3));

for iref = 1:size(COMs, 1)
    b = abs(bsxfun(@minus, COMs(iref,:), COMs));
    bb = sum(abs(b), 2);
    %     [diffCOMs(iref), imatch(iref)] = min(bb); % which ROI in mask{2} matched ROI iref in mask{1}
    
    % For ROIs that are near each other use the sum_mask method to find the
    % best matching one.
    f = find(bb<10)'; % this threshold is important
    if ~isempty(f)
        %         clear maskOverlapMeasure
        %     maskOverlapMeasure = NaN(size(mask,3), 1);
        for itoMatch = f
            sumMask = sparse(mask(:,:,itoMatch)) + sparse(mask(:,:,iref));
            maskOverlapMeasure(itoMatch, iref) = sum(sumMask(:)>1);
        end
    end
end

% figure; imagesc(maskOverlapMeasure)

%% Set overlapROIs_ratio

maskNumPix = reshape(mask, [], size(mask,3));
maskNumPix = sum(maskNumPix); % number of pixels of each mask

% overlapROIs_ratio(i2,ir) :  fraction of ROI i2 that overlaps with ROI ir.
overlapROIs_ratio = bsxfun(@rdivide, maskOverlapMeasure, maskNumPix');
% figure; imagesc(overlapROIs_ratio)


%% Threshold overlapROIs_ratio

nr = size(overlapROIs_ratio,1);

% overlapROIs_ratio_thed = logical(overlapROIs_ratio);
overlapROIs_ratio_thed = false(size(overlapROIs_ratio));
overlapROIs_ratio_thed(overlapROIs_ratio>0) = true;
overlapROIs_ratio_thed(1:nr+1:nr^2) = false;

% if the fraction overlap for either of the ROIs is <thMerge, those ROIs
% wont be counted as matching:
% overlapROIs_ratio_thed(overlapROIs_ratio <= thMerge) = false;

% if the mean fraction overlap across the 2 ROIs is <thMerge, those ROIs wont be counted as matching.
overlapMean = (overlapROIs_ratio + overlapROIs_ratio')/2;
overlapROIs_ratio_thed(overlapMean <= thMerge) = false;

% figure; imagesc(overlapROIs_ratio_thed)


%% Define FF3 for merging components (needed by Eftychios's method in merge_components)

% thr = 0; % .7;
mx = 50; % maximum merging operations

nr = size(A,2);
% [d,T] = size(Y);
C_corr = corr(full(C(1:nr,:)'));
% FF1 = triu(C_corr)>= thr;                           % find graph of strongly correlated temporal components

% A_corr = triu(A(:,1:nr)'*A(:,1:nr));
% A_corr(1:nr+1:nr^2) = 0;
% FF2 = A_corr > thSpatialOverlap; % 0: Efty's value  % find graph of overlapping spatial components

% FF3 = and(FF1,FF2);                                 % intersect the two graphs

FF3 = overlapROIs_ratio_thed;


%% Use Eftychios's method (in merge_components) to find components that need to be merged.

[l,c] = graph_connected_comp(sparse(FF3+FF3'));     % extract connected components
MC = [];
for i = 1:c
    if length(find(l==i))>1
        MC = [MC,(l==i)'];
    end
end

cor = zeros(size(MC,2),1);
for i = 1:length(cor)
    fm = find(MC(:,i));
    for j1 = 1:length(fm)
        for j2 = j1+1:length(fm)
            cor(i) = cor(i) + C_corr(fm(j1),fm(j2));
        end
    end
end

[~,ind] = sort(cor,'descend');
nm = min(length(ind),mx);   % number of merging operations
merged_ROIs = cell(nm,1);
% A_merged = zeros(d,nm);
% C_merged = zeros(nm,T);
% S_merged = zeros(nm,T);
% if strcmpi(options.deconv_method,'constrained_foopsi')
%     P_merged.gn = cell(nm,1);
%     P_merged.b = cell(nm,1);
%     P_merged.c1 = cell(nm,1);
%     P_merged.neuron_sn = cell(nm,1);
% end
% if ~options.fast_merge
%     Y_res = Y - A*C;
% end

for i = 1:nm
    merged_ROIs{i} = find(MC(:,ind(i)));
end

merged_ROIs
fprintf('Length of merged_ROIs = %d\n', length(merged_ROIs))


%% Evaluate merging.

if doEvaluate
    load(pnevFileName, 'C_df')
    % C_df = C_df_m;
    
    f = figure;
    f3 = figure;
    % ff = figure;
    
    for i = 1:length(merged_ROIs)
        
        c = tril(corr(C_df([merged_ROIs{i}],:)'), -1);
        c(c==0) = nan;
        s = sort(c(c~=0))'; s(isnan(s)) = []; lowestCorrs = s(1:min(length(s),10))
        %     figure(ff); plot(s);
        
        
        figure(f), set(gcf, 'name', ['Merged ROI', num2str(i)]);
        %     imagesc(sum(mask(:,:,merged_ROIs{i}),3)), colorbar
        %     drawnow, pause
        mm = zeros(size(mask(:,:,1)));
        for ii = 1:length(merged_ROIs{i})
            
            mm = mm + mask(:,:,merged_ROIs{i}(ii));
            figure(f)
            imagesc(mm), colorbar
            title(merged_ROIs{i}(ii))
            
            yx = COMs(merged_ROIs{i}(1),:); % yx = COMs_all_sess_new{1}(merged_ROIs{i}(1),:);
            xlim([yx(2)-20  yx(2)+20])
            ylim([yx(1)-20  yx(1)+20])
            
            
            figure(f3), hold on
            h = plot(C_df(merged_ROIs{i}(ii),:)');
            
            drawnow, pause
        end
        figure(f3), cla
    end
    
end


%% Save to pnevFileName

if saveMergedVar
    save(pnevFileName, '-append', 'merged_ROIs')
end






