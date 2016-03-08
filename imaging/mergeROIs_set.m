% find ROIs that need to be merged (Eft algorithm failed to merge them).

%%
thMerge = .35; % fraction of an ROI that is overlapped by the other ROI, above thMerge would be considered for merging ROIs.

mousename = 'fni17';
imagingFolder = '151102'; % '151021';
mdfFileNumber = 1; % or tif major

signalCh = 2; % CC, mask, etc are driven from signalCh_meth1 (usually you use this as Ref, but you can change in pnev_manual_comp_match)

[imfilename, pnevFileName] = setImagingAnalysisNames(mousename, imagingFolder, ...
    mdfFileNumber, signalCh);

load(imfilename, 'sdImage', 'imHeight', 'imWidth')

disp(pnevFileName)
load(pnevFileName, 'A', 'C_df')

nr = size(A,2);


%%
contour_threshold = .99; % .95;
im = sdImage{2};
[CC, ~, COMs, mask] = setCC_cleanCC_plotCC_setMask(A, imHeight, imWidth, contour_threshold, im);


%%
[matchedROI_idx, maskOverlapMeasure] = matchROIs_sumMask(mask, mask);


%%
figure; imagesc(maskOverlapMeasure)

measuLower = tril(maskOverlapMeasure, -1);
% figure; imagesc(measuLower)


%%
maskNumPix = reshape(mask, [], size(mask,3));
maskNumPix = sum(maskNumPix); % number of pixels of each mask

overlapROIs = cell(1, nr);
overlapROIs_ratio_origMask = cell(1, nr);
overlapROIs_ratio_overlapMask = cell(1, nr);
for rr = 1:nr
    % index of overlapping ROIs
    overlapROIs{rr} = find(measuLower(:,rr));
    
    % fraction of the original mask that is covered by the overlapping mask
    overlapROIs_ratio_origMask{rr} = bsxfun(@rdivide, maskOverlapMeasure(overlapROIs{rr},rr), maskOverlapMeasure(rr,rr));
    
    % fraction of the overlapping mask that covers the original mask
    %     ms = maskOverlapMeasure(sub2ind(size(maskOverlapMeasure), overlapROIs{rr}, overlapROIs{rr})); % number of pixels of each overlapping mask
    %     overlapROIs_ratio_overlapMask{rr} = maskOverlapMeasure(overlapROIs{rr},rr) ./ ms;
    overlapROIs_ratio_overlapMask{rr} = maskOverlapMeasure(overlapROIs{rr},rr) ./ maskNumPix(overlapROIs{rr})';
end

% overlapROIs_ratio_origMask2 = bsxfun(@rdivide, maskOverlapMeasure, diag(maskOverlapMeasure)');

overlapROIs_ratio = bsxfun(@rdivide, maskOverlapMeasure, maskNumPix');
figure; imagesc(overlapROIs_ratio)
% overlapROIs_ratio_overlapMask2(i,j) shows fraction of mask i that
% overlaps mask j.
% overlapROIs_ratio_overlapMask2(j,i) shows fraction of mask j that
% overlaps mask i.


%%
%{
ccoef = corrcoef(C_df([rr;rr2],:)');
size(ccoef)
figure; imagesc(ccoef)
%}

%% threshold overlapROIs_ratio
overlapROIs_ratio_thed = logical(overlapROIs_ratio);

overlapROIs_ratio_thed(1:nr+1:nr^2) = false;
overlapROIs_ratio_thed(overlapROIs_ratio <= thMerge) = false;

figure; imagesc(overlapROIs_ratio_thed)

% [l,c] = graph_connected_comp(sparse(overlapROIs_ratio_thed));


%% set merged_ROIs_new
merged_ROIs_rr = cell(1, nr);
for rr = 1:nr
    merged_ROIs_rr{rr} = intersect(find(overlapROIs_ratio_thed(:,rr)), find(overlapROIs_ratio_thed(rr,:))); % merged_ROIs_rr{rr} = ROI indeces that overlap with ROI rr.
    %     merged_ROIs_rr{rr} = find(overlapROIs_ratio_thed(:,rr));
end


clear merged_ROIs_new
cnt = 0;
for rr = 1:nr
    if ~isempty(merged_ROIs_rr{rr})
        cnt = cnt+1;
        i = rr; a = [i, merged_ROIs_rr{i}']; % a shows all ROIs that overlap with ROI i other at >thMerge of their area, also ROI i overlaps with them at >thMerge of its area.
        
        for ii = 1:length(merged_ROIs_rr{rr})
            i = merged_ROIs_rr{rr}(ii);
            b = [i, merged_ROIs_rr{i}']; % b shows all ROIs that overlap with ROI i other at >thMerge of their area, also ROI i overlaps with them at >thMerge of its area.
            a = union(a,b); % union of all of these ROIs set the merged_ROIs_new.
        end
        merged_ROIs_new{cnt} = a;
        [merged_ROIs_rr{a}] = deal({});
    end
end


%% evaluate merged components. remove manually if you don't like some merged ones...
plottraces = true;
pauseAftEachComp = false;
plotCm = true; false; % set to true after you set Cm.

f1 = figure;
imagesc(im)
freezeColors
hold on

if plottraces; f2 = figure('position', [440  572  1288  226]); hold on; end;
% if plotCm; f3 = figure('position', [440  572  1288  226]); end;

for rrs = 1:length(merged_ROIs_new)
    rr = merged_ROIs_new{rrs}(1);
    figure(f1);
    h = plot(CC{rr}(2,:), CC{rr}(1,:), 'color', 'r');
    
    xlim([COMs(rr,2)-60 COMs(rr,2)+60])
    ylim([COMs(rr,1)-60 COMs(rr,1)+60])
    
    if plottraces; figure(f2); h2 = plot(C_df(rr, :), 'r'); end
    
    for rr2 = merged_ROIs_new{rrs}(2:end);
        figure(f1)
        hn = plot(CC{rr2}(2,:), CC{rr2}(1,:), 'k'); %'color', [255 215 0]/255);
        h = [h, hn];
        
        set(gcf, 'name', sprintf(' Neuron %d & %d', rr, rr2));
        title(sprintf('%.2f  %.2f', overlapROIs_ratio(rr2,rr), overlapROIs_ratio(rr,rr2)));
        
        if plottraces
            figure(f2)
            h2n = plot(C_df(rr2, :), 'k');
            h2 = [h2 h2n];
            set(gcf, 'name', sprintf(' Neuron %d', rr2));            
        end
        
        if plotCm
%             figure(f3)
            figure(f2)
            e = size(Cm,1)-1;
            h2n = plot(Cm(e-length(merged_ROIs_new)+rrs, :), 'b');
            h2 = [h2 h2n];
        end
        
        if pauseAftEachComp
            pause
            figure(f1)
            delete(hn)
            
            if plottraces
                figure(f2)
                delete(h2n)
            end
        end
        
    end
    pause
    delete([h, h2])
end



%% run Eft code to get A, C and S after merging
load(pnevFileName, 'A', 'C', 'b', 'f', 'S', 'P', 'options')
[Am,Cm,K_m,~,Pm,Sm] = merge_components_again(merged_ROIs_new,A,b,C,f,P,S,options);

% demo_script_finalMerge --> to finalize Eft results after this new merging.



%% plot all overlapping ROIs without any thresholding.
% for each roi plot its overlapping rois and the overlap measure... find a
% threshold for which rois are parts of the same roi.
plottraces = false;

f1 = figure;
imagesc(im)
freezeColors
hold on

if plottraces; f2 = figure; end;

for rr = 1:nr
    figure(f1)
    h = plot(CC{rr}(2,:), CC{rr}(1,:), 'color', 'r');
    
    xlim([COMs(rr,2)-60 COMs(rr,2)+60])
    ylim([COMs(rr,1)-60 COMs(rr,1)+60])
    
    if plottraces; figure(f2); plot(C_df(rr, :)); end
    
    rr2s = overlapROIs{rr}; % index of all overlapping ROIs
    if ~isempty(rr2s)
        for ir = 1:length(rr2s)
            rr2 = rr2s(ir);
            
            figure(f1)
            hn = plot(CC{rr2}(2,:), CC{rr2}(1,:), 'k'); %'color', [255 215 0]/255);
            h = [h, hn];
            set(gcf, 'name', sprintf(' Neuron %d & %d', rr, rr2));
            
            %             title(sprintf('%.2f', overlapROIs_ratio_origMask{rr}(ir)));
            title(sprintf('%.2f', overlapROIs_ratio_overlapMask{rr}(ir)));
            
            if plottraces
                figure(f2)
                set(gcf, 'name', sprintf(' Neuron %d', rr2));
                h2 = plot(C_df(rr2, :));
            end
            
            pause
            figure(f1)
            delete(hn)
            
            if plottraces
                figure(f2)
                delete(h2)
            end
        end
    end
    
    figure(f1)
    delete(h)
end



%% take a look at overlapping ROIs after some kind of thresholding
f1 = figure;
imagesc(im)
freezeColors
hold on

for rr = 1:nr
    %     rr2 = find(overlapROIs_ratio_thed(:,rr));
    rr2s = intersect(find(overlapROIs_ratio_thed(:,rr)), find(overlapROIs_ratio_thed(rr,:)));
    %     rr2 = union(find(overlapROIs_ratio_thed(:,rr)), find(overlapROIs_ratio_thed(rr,:)));
    %     rr2 = find(l==rr);
    if ~isempty(rr2s)
        h = plot(CC{rr}(2,:), CC{rr}(1,:), 'color', 'r');
        
        xlim([COMs(rr,2)-60 COMs(rr,2)+60])
        ylim([COMs(rr,1)-60 COMs(rr,1)+60])
        
        for ir = 1:length(rr2s)
            rr2 = rr2s(ir);
            
            figure(f1)
            hn = plot(CC{rr2}(2,:), CC{rr2}(1,:), 'k'); %'color', [255 215 0]/255);
            h = [h, hn];
            set(gcf, 'name', sprintf(' Neuron %d & %d', rr, rr2));
            title(sprintf('%.2f', overlapROIs_ratio(rr2,rr)));
            
            
            pause
            figure(f1)
            delete(hn)
        end
        figure(f1)
        delete(h)
    end
end





%%
%{
clear merged_ROIs_new

cnt = 0;
for rr = 1:nr
    rr2 = overlapROIs{rr}; % index of all overlapping ROIs
    if ~isempty(rr2)
        f = (overlapROIs_ratio_overlapMask{rr} > thMerge);
        %         overlapROIs_ratio_overlapMask{rr}(f)
        if any(f)
            cnt = cnt+1;
            merged_ROIs_new{cnt} = [rr; rr2(f)];
            pause
        end
    end
end


fa = merged_ROIs_new{cnt}';
for i = merged_ROIs_new{cnt}'
    a = overlapROIs_ratio_thed(i,:);
    fa = [fa, find(a~=0)];
end
%}


