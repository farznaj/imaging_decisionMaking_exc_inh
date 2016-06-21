% This script uses the same method as Eftychios's merge_components to set
% merged_ROIs, except using different thresholds for A_corr and C_corr. 
%
% It also does plotting to evaluate merged_ROIs

% make sure A,C,S,p,etc are set correctly after merging.

%%
load(imfilename, 'imHeight', 'imWidth', 'sdImage')
% showResults = 1;
im = sdImage{2};
contour_threshold = .95;

fprintf('Setting the mask for the gcamp channel....\n')
[CC, ~, COMs, mask] = setCC_cleanCC_plotCC_setMask(A, imHeight, imWidth, contour_threshold, im);
title('ROIs shown on the sdImage of channel 2')
   
    
%%
load(pnevFileName, 'C', 'C_df', 'A')


%% Using Eftychios's method (in merge_components) to find components that need to be merged.

thr = 0; % .7;
thSpatialOverlap = 20; %5e6; % 1e6; %.1; % 0; % 0: Efty's value

C = C_df_m;
A = A_m;
% A(A~=0) = 1;


u = unique(A_corr(:)); 
figure; plot(sort(u(1:end-1)))
figure; plot(diff(sort(u(1:end-1))))

FF2 = overlapROIs_ratio_thed;
FF3 = overlapROIs_ratio_thed;

%%
mx = 50; % maximum merging operations

nr = size(A,2);
% [d,T] = size(Y);
C_corr = corr(full(C(1:nr,:)'));
FF1 = triu(C_corr)>= thr;                           % find graph of strongly correlated temporal components

A_corr = triu(A(:,1:nr)'*A(:,1:nr));
A_corr(1:nr+1:nr^2) = 0;
FF2 = A_corr > thSpatialOverlap; % 0: Efty's value  % find graph of overlapping spatial components

FF3 = and(FF1,FF2);                                 % intersect the two graphs

%%
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
length(merged_ROIs)


%% evaluate merging.

% C_df = C_df_m;

f = figure;
% ff = figure;
f3 = figure;

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


%% evaluate ROIs that are in the same marged_roi cell but have correlations < thr.

for i0 = 1:length(merged_ROIs)
    
    f = figure;
    ff = figure;
    [rr,cc] = find(c < thr);
    
    for i = 1:size(rr,1)
        r1 = merged_ROIs{i0}(rr(i));
        r2 = merged_ROIs{i0}(cc(i));
        
        %     corr(C_df([r1,r2],:)')
        
        figure(f)
        imagesc(mask(:,:,r1) + mask(:,:,r2))
        colorbar
        title(c(rr(i),cc(i)))
        
        yx = COMs_all_sess_new{1}(r1,:);
        xlim([yx(2)-20  yx(2)+20])
        ylim([yx(1)-20  yx(1)+20])
        
        
        figure(ff)
        plot(C_df([r1,r2],:)')
        
        drawnow
        pause
    end
    
end





%% Run Eft code to get A, C and S after merging

load(pnevFileName, 'A', 'C', 'b', 'f', 'S', 'P', 'options')
[Am,Cm,K_m,~,Pm,Sm] = merge_components_again(merged_ROIs,A,b,C,f,P,S,options);


