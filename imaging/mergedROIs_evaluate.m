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
% merged_ROIs_m = merging_vars_m.merged_ROIs;
merged_ROIs_m = merging_vars_m.merged_ROIs{1};


%%
A = A(:, ~badComps);
C_df = C_df(~badComps, :);

% get rid of the background component in C_df
if size(C_df,1) == size(A,2)+1
    %     bk_df = temporalDf(end,:); % background DF/F
    C_df(end,:) = [];
end

% get rid of the background component in C_df
if size(C_df_m,1) == size(A_m,2)+1
    %     bk_df = temporalDf(end,:); % background DF/F
    C_df_m(end,:) = [];
end


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
% [CC, ~, COMs_m, mask] = setCC_cleanCC_plotCC_setMask(sp, imHeight, imWidth, contour_threshold, im, plotCOMs);



%%
%%%%%%%%%%%%%%%% Single round of merging

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
% find index of merged_ROIs in C_df_m
lt = size(C_df_m,1);
lm = length(merged_ROIs_m);
merged_1st_ind = lt - lm + 1; % index of 1st merged_ROI in C_df_m
merged_inds = (1:lm) + (merged_1st_ind-1); % index of merged_ROIs in C_df_m : merged_inds(i)=j means C_df_m(j,:) is the merged trace of ROIs that are in merged_ROIs{i}

f = figure;
ff = figure; figure(ff), imagesc(im), hold on
cnt = 0;
for i = merged_inds(end:-1:1) % merged_inds
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
    figure(ff), % delete(h), 
    delete(hh)
end



%% Look at unmerged components and compare their traces before and after merging

origInds = 1:size(C_df,1);
origInds(cell2mat(merged_ROIs_m)) = []; % origInds(i) = j: index i in C_df_m is same as index j in C_df
fprintf('Number of unmerged components = %d\n', length(origInds))
% figure; plot(origInds) 

figure; 
for i = 1:length(origInds)
    hold on
    plot(C_df(origInds(i),:))
    plot(C_df_m(i,:))   
    
    pause
    cla
end

% 1. it gets noisier after merging
% 2. the sharp drops (due to trial onset I think) get smoothed after
% merging, ie what I didn't want.





%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Multiple rounds of merging
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
imagesc(im) %(medImage{2})
hold on

% Plot all ROIs
for rr = 1:size(COMs_m,1)
    plot(COMs_m(rr,2), COMs_m(rr,1), 'c.')
end


%%
% merged_ROIs_m{r}{m} = k;
% ROIs k of round r-1 were merged in round r. (index of k is relative to round r-1)
% The merged ROI m is roi ind1all(r)+m-1 in C_df of round r.


nnow = size(C_df,1);
nfinal = size(C_df_m,1);
lmall = cellfun(@length, merged_ROIs_m);
nall = nan(1, length(merged_ROIs_m));
nmall = nan(1, length(merged_ROIs_m));
for r = 1:length(merged_ROIs_m) % round 1 of merging
    mnow = merged_ROIs_m{r};
    lm = length(mnow);
    nm = sum(cellfun(@length, mnow)); 
    nnow = nnow - nm + lm;
    nall(r) = nnow;
    nmall(r) = nm;
end

if ~isequal(nnow, nfinal), error('something wrong'), end

ind1all = nall - lmall + 1; % 1st index of merged ROIs for each round of merging.

%{
for r = 1:length(merged_ROIs_m)
    figure; hold on
    for m = 1:length(merged_ROIs_m{r})
        plot(merged_ROIs_m{r}{m})
    end
end

r = 2;
c = cellfun(@(x)x- ind1all(r-1) + 1, merged_ROIs_m{r}, 'uniformoutput', 0);
cellfun(@(x)x(x>0), c, 'uniformoutput', 0)
cell2mat(cellfun(@(x)x(x>0), c, 'uniformoutput', 0))
%}

%%
r = 1;
newInds{r} = 1:size(C_df, 1);
rmvInds = cell2mat(merged_ROIs_m{r});
newInds{r}(rmvInds) = [];
newInds{r} = [newInds{r}, 1000+ind1all(r) : 1000+ind1all(r)-1+lmall(r)];

for r = 2:3
    newInds{r} = newInds{r-1};
    rmvInds = cell2mat(merged_ROIs_m{r});
    newInds{r}(rmvInds) = [];
    newInds{r} = [newInds{r}, r*1000+ind1all(r) : r*1000+ind1all(r)-1+lmall(r)];
end


%%
col = distinguishable_colors(sum(lmall));
cnt = 0;
ind1all_prev = ind1all - [nan lmall(1:end-1)]; 
arr_all = cell(1, length(merged_ROIs_m));
for r0 = length(merged_ROIs_m):-1:1
    ccnt = 0;
    figure; imagesc(im), hold on
    for m = 1:length(merged_ROIs_m{r0})
        m
        arr = [];
        r = r0;
        while r > 0
            if r==1
                a = merged_ROIs_m{r}{m};
            else
                a = newInds{r-1}(merged_ROIs_m{r}{m});
            end
            arr = [arr; a(a<1000)];
            
            if any(a>2000)
                m = a(a>2000)-2000 - ind1all(r-1) + 1;
                
                b = merged_ROIs_m{2}{m};
                b(b>ind1all(1)) = 1000+b(b>ind1all(1));
                a = b;
                
            elseif any(a>1000)
                m = a(a>1000)-1000 - ind1all(r-1) + 1;
                
                if r==3 % | m < 0 
%                     [m,r]
                    f = newInds{r-1}(find(newInds{r-1}>1000, 1));
                    %                 'now'
                    %                 pause
                    %                 m = a(a>1000)-1000 - ind1all_prev(r-1) + 1;
                    m = a(a>1000)-(f-1); % - ind1all_prev(r-1) + 1;
                    r = r-1;
                end
            else
                break
            end
            r = r-1;
            
        end
        cnt = cnt+1;
        ccnt = ccnt+1;
        arr_all{r0}{ccnt} = arr;
        plot(COMs(arr,2), COMs(arr,1), '.', 'color', col(cnt,:))
%         plot(COMs(arr,2), COMs(arr,1), 'r.')
%         pause
    end
end
arr_all0 = arr_all;

%%
rr0 = 1:length(arr_all);
clear ism ism_any rm
for r1 = 1:length(arr_all);
    rr = rr0;
    rr(r1) = [];
    for m = 1:length(arr_all{r1})
        for r2 = rr
            for m2 = 1:length(arr_all{r2})
                s = sum(ismember(arr_all{r1}{m}, arr_all{r2}{m2}));
                ism{r1,m}(r2,m2) = s;
            end
        end
        [row, col] = find(ism{r1,m});
        if ~isempty(row)
            rm{r1,m} = [[r1,m] ; [row, col]];
        end
%         ism_any(r1,m) = any(ism{r1,m}(:));
    end
end

for imf = 1:length(rmf)
    [~,is] = sortrows(rmf{imf},1);
    rf = rmf{imf}(is(1:end-1),:);
    for ir = 1:size(rf,1) 
        arr_all{rf(ir,1)}{rf(ir,2)}=[]; 
    end
end

arr_all_final = []; 
for ir = 1:length(arr_all), 
    arr_all_final = [arr_all_final, arr_all{ir}(cellfun(@(x)~isempty(x), arr_all{ir}))]; 
end

% everything seems fine except that the following two are not identical...
% I'm too tired to figure this out...
[size(C_df,1) - sum(cellfun(@length,arr_all_final)) + length(arr_all_final)
size(C_df_m,1)]


%% final plot of merged comps

col = distinguishable_colors(length(arr_all_final), 'k');
figure; imagesc(im), colormap(gray), hold on

plot(COMs_m(:,2), COMs_m(:,1), '.', 'color', 'c')

for i = 1:length(arr_all_final)
    k = arr_all_final{i};
    plot(COMs(k,2), COMs(k,1), '.', 'color', col(i,:))
%     plot(COMs(k,2), COMs(k,1), 'r.')
end


%% look at the traces...
% below is not correct... u didn't finish this to find what indeces in
% C_df_m correspond to mergedRois in C_df

af = arr_all_final(end:-1:1);
figure; hold on; 
for i = 0:length(af)
    plot(C_df_m(end-i,:), 'k', 'linewidth', 2)    
    plot(C_df(af{end-i},:)')
    pause
    cla
end


figure; 
for i = 0:39
    k = arr_all{1}{end-i};
    hold on, 
    plot(C_df(k,:)')
%     plot(C_df_m(end-i,:), 'k')
%     plot(C_df_m(end-12-i,:), 'k')
    plot(C_df_m(end-12-12-i,:), 'k')
    pause
    cla
end


%%
%{
figure; imagesc(im), hold on
r0 = 3;
for m0 = 1:length(merged_ROIs_m{r})
    r = r0;
    m = m0;
    arr = [];
    while r>0
        if ~isempty(m)
            mnow = merged_ROIs_m{r}{m};
        else
            break
        end
        if r==1
            arr = [arr; mnow];
        else
            f = find((mnow - ind1all(r-1) + 1) <= 0);
            
            o = newInds{r-1}(merged_ROIs_m{r}{m}(f));
            arr = [arr; o(:)];
            
            ff = find((mnow - ind1all(r-1) + 1) > 0);
            if ~isempty(ff)
                indMergedROIinPrevMerg = mnow(ff) - ind1all(r-1) + 1;
                % mnow = merged_ROIs_m{r-1}{indMergedROIinPrevMerg}
                m = indMergedROIinPrevMerg;
            else
                m = [];
            end
        end
        r = r-1;
    end
    
    arr
    plot(COMs(arr,2), COMs(arr,1), 'r.')
    pause
end
%}
