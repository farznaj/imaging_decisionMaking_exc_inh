% After merging components (in demo_script_finalMerge), use this script to evaluate the results.

% remember indeces in merged_ROIs are on the original A (before merging)
% after removing badComps (ie from A, badComps were removed and then
% merging ROIs were found).

%%
mouse = 'fni17';
imagingFolder = '151102';
mdfFileNumber = [1,2]; % 1; % or tif major


%% Set mat file names

signalCh = 2; % because you get A from channel 2, I think this should be always 2.
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh);
[~,f] = fileparts(pnevFileName);
disp(f)


%%
load(imfilename, 'sdImage', 'imHeight', 'imWidth');
im = sdImage{2};


%%
load(pnevFileName, 'A', 'A_m', 'merging_vars_m', 'badComps', 'C_df', 'C_df_m')
merged_ROIs_m = merging_vars_m.merged_ROIs;


%%
A = A(:, ~badComps);
C_df = C_df(~badComps, :);

%
C_df(end,:) = [];
C_df_m(end,:) = [];


%%
COMs = fastCOMsA(A, [imHeight, imWidth]); % size(medImage{2})
size(COMs)

COMs_m = fastCOMsA(A_m, [imHeight, imWidth]); % size(medImage{2})
size(COMs_m)


%% Plot contours

sp = A_m;
contour_threshold = .95;
plotCOMs = 0;

setCC_cleanCC_plotCC_setMask(sp, imHeight, imWidth, contour_threshold, im, plotCOMs);
% [CC, ~, COMs, mask] = setCC_cleanCC_plotCC_setMask(sp, imHeight, imWidth, contour_threshold, im, plotCOMs);


%% Plot COMs

figure;
imagesc(im) %(medImage{2})
hold on

% Plot all ROIs
for rr = 1:size(COMs,1)
    plot(COMs(rr,2), COMs(rr,1), 'c.')
end


% Plot ROIs that were merged
% col = hot(length(merged_ROIs));
% col = distinguishable_colors(length(merged_ROIs_m));
col = repmat([1 0 0; 1 0 1; .3 0 0], ceil(length(merged_ROIs_m)/3), 1);
for i = 1:length(merged_ROIs_m)
    for rr = merged_ROIs_m{i}
        plot(COMs(rr,2), COMs(rr,1), '.', 'color', col(i,:))
%         plot(COMs(rr,2), COMs(rr,1), '.', 'color', 'r')
    end
    %     pause
end


%%
lt = size(C_df_m,1);
lm = length(merged_ROIs_m);

f = figure;
ff = figure; figure(ff), imagesc(im), hold on
cnt = 0;
for i = lt: -1: lt - lm + 1 % lt - lm + 1 : lt
    cnt = cnt+1;
    %     m = merged_ROIs_m{cnt};
    m = merged_ROIs_m{end-cnt+1};
    
    % plot traces
    figure(f), hold on
    
    % plot merged trace
    plot(C_df_m(i,:), 'linewidth', 1.5, 'color', 'k')
    % plot individual traces before merging
    plot(C_df(m,:)', 'linewidth', .2)
    
    
    % Plot ROIs that got merged.
    figure(ff), hold on
    %         h = plot(COMs_m(i,2), COMs_m(i,1), 'r.');
    hh = plot(COMs(m,2), COMs(m,1), 'r.');
    
    
    drawnow, pause
    figure(f), cla
    figure(ff), delete(h), delete(hh)
end



