%% decide what y (activity_man) you want to use, with or without nan.

activity_man_final = activity_man_eftMask(end-1846:end,:);

%% Add nans for ITIs to activiy_man

tracey = activity_man_eftMask(end-1846:end, :);
activity_man_final = [];
% cnan = [];
for itr = 1:length(cs_frtrs)-1
    frs = cs_frtrs(itr)+1 : cs_frtrs(itr+1);    
    
    activity_man_final = [activity_man_final; tracey(frs,:)];
%     cnan = [cnan, C(:, frs)];
    if itr < length(cs_frtrs)-1
        activity_man_final = [activity_man_final; NaN(Nnan(itr), size(tracey,2))];
%         cnan = [cnan, NaN(size(C,1), Nnan(itr))];
    end
end
size(activity_man_final)
% size(cnan)

%%
N = size(activity_man_final,2);
T = size(activity_man_final,1);


%% Decide what C you want to plot : with estimated vals for ITIs or with them set to nan.
% with estimated vals:
c = C; % Cin;

%% On the NaN-ITI-inserted traces, find the index of ITI start and end, ie nanBeg and nanEnd 
% you will use this below to set to NaN the infered values of C during ITI.

% set cumsum of #frames for each tr after adding nans for ITIs
n_frtrs = diff(cs_frtrs);
ntrs_nan = [n_frtrs(1:end-1) + Nnan, n_frtrs(end)];
cs_frtrs_nan = [0 cumsum(ntrs_nan)];
% cs_frtrs_nan(1:end-1) + ntrs

% % on the traces including nans, find the beg and end index for nans.
beg = cs_frtrs_nan(2:end-1)+1;
nanBeg = beg - Nnan;
nanEnd = beg-1;

% inds2keep = cell2mat(arrayfun(@(x,y)(x:y), [1 nanEnd+1], [nanBeg-1 size(C2,2)], 'uniformoutput', 0));


%% Set to NaN the infered values of C during ITI.

c = C2; % C_multi_trs;
for itr = 1:length(nanBeg)
    c(:, nanBeg(itr) : nanEnd(itr)) = NaN;
end


%% Find non-merged components.
% indsComps = 1:size(c,1);

indsComps = 1:size(activity_man_final,2);
indsComps(cell2mat(merged_ROIs)') = nan;
indsComps = indsComps(~isnan(indsComps))


%% Plot non-merged comps.

cnt = 0;
figure;
for i = indsComps % 1:length(indsComps)
    cnt = cnt+1;
%     T = length(C(i,:));
    subplot(5,ceil(N/5),i); 
%     plot(1:T, y(:,i),  1:T, C(i,:));
%     plot(1:T, shiftScaleY(y(:,i)),  1:T, shiftScaleY(C(i,:))); % scaled
%     plot(1:T, shiftScaleY(y(:,indsComps(i))),  1:T, shiftScaleY(c(i,:))); % scaled
    plot(1:T, shiftScaleY(activity_man_final(:,i)),  1:T, shiftScaleY(c(cnt,:))); % scaled
%     plot(1:T, (y(:,i)),  1:T, (c(cnt,:))); % scaled
end


%% Plot merged comps

figure;
for i = 1:length(merged_ROIs)
%     T = length(C(i,:));
    subplot(5,ceil(N/5),i); 
%     plot(1:T, y(:,i),  1:T, C(i,:));
%     plot(1:T, shiftScaleY(y(:,i)),  1:T, shiftScaleY(C(i,:))); % scaled
    plot(1:T, shiftScaleY(nanmean(activity_man_final(:,merged_ROIs{i}),2)),  1:T, shiftScaleY(c(length(indsComps)+i,:))); % scaled
end



    


