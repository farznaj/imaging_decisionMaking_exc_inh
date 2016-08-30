load('SVM_151029_003_ch2-PnevPanResults-160426-191859.mat')




dataTensor = non_filtered;
all_times = time_aligned;
% % dataTensor = traces_al_1stSideTry;
% % all_times = time_aligned_1stSideTry;

%% preprocess
[T, N, R] = size(dataTensor);

meanN = mean(reshape(permute(dataTensor, [1 3 2]), T*R, N));
stdN = std(reshape(permute(dataTensor, [1 3 2]), T*R, N));
meanN = mean(X);
stdN = std(X);
dataTensor = bsxfun(@times, bsxfun(@minus, dataTensor, meanN), 1./(stdN+sqrt(0)));

%% plot average per decsision
figure;
hold on
plot(all_times, mean(mean(dataTensor(:, :, Y==0), 3), 2), 'b')
plot(all_times, mean(mean(dataTensor(:, :, Y==1), 3), 2), 'r')
xlabel('time (ms)')
ylabel('normalized firing rates')
%% alignment of top variance subspaces
numDim = 2; % define dimensionality of subspace
[PCs_t, Summary] = pca_t(dataTensor, numDim); % identify subspaces
aIx = nan(T, T); % alignement index between subspaces
for i = 1:T
    for j = 1:T
        aIx(i,j) = alignIx(squeeze(PCs_t(i, :, :)), squeeze(PCs_t(j, :, :)));
    end
end
figure;
imagesc(all_times, all_times, aIx);
colormap(colormap('jet'))
title('alignment index')
colorbar
caxis([0 1])
axis square
xlabel('time (ms)')
ylabel('time (ms)')
%% TDR analysis
stim = stimrate(:);
decision = Y;
codedParams = [[stim(:)-min(stim(:))]/range(stim(:)) [decision(:)-mean(decision(:))]/range(decision(:))];

[dRAs, normdRAs, Summary] = runTDR(dataTensor, N, codedParams, [], false);
angle = real(acos(abs(dRAs(:,:,1)*dRAs(:,:,1)')))*180/pi;
angle(:,:,2) = real(acos(abs(dRAs(:,:,2)*dRAs(:,:,2)')))*180/pi;
angle(:,:,3) = real(acos(abs(dRAs(:,:,1)*dRAs(:,:,2)')))*180/pi;

%%
figure;
hold on
plot(all_times, Summary.R2_tk(:,1), 'k')
plot(all_times, Summary.R2_tk(:,2), 'b')
xlabel('time (ms)')
ylabel('signal contribution to rate')

figure;
subplot(131)
imagesc(all_times, all_times, angle(:,:,1));
colormap(flipud(colormap('jet')))
colorbar
title('angle stim')
caxis([0 90])
xlabel('time (ms)')
ylabel('time (ms)')
axis square
subplot(132)
imagesc(all_times, all_times, angle(:,:,2));
colormap(flipud(colormap('jet')))
colorbar
title('angle decision')
caxis([0 90])
xlabel('time (ms)')
ylabel('time (ms)')
axis square

subplot(133)
imagesc(all_times, all_times, angle(:,:,3));
colormap(flipud(colormap('jet')))
colorbar
title('angle stim vs decision')
caxis([0 90])
xlabel('time (ms)')
ylabel('time (ms)')
axis square

%%
dataTensor_proj(:, 1, :) = projectTensor(dataTensor, squeeze(dRAs(end-20, :, 1)).');
dataTensor_proj(:, 2, :) = projectTensor(dataTensor, squeeze(dRAs(end, :, 2)).');

uniqueStim = unique(stim);
uniqueDecision = unique(decision);
S = length(uniqueStim);
D = length(uniqueDecision);
projStim = [];
for s = 1:S
    for d = 1:D
        msk = (stim ==uniqueStim(s)) & (decision ==uniqueDecision(d));
        proj1(:, s, d) = mean(squeeze(dataTensor_proj(: ,1, msk)), 2);
        proj2(:, s, d) = mean(squeeze(dataTensor_proj(: ,2, msk)), 2);
        
    end
end

% % clr = redgreencmap(S, 'interpolation', 'linear');
% % figure;
% % subplot(211)
% % hold on
% % for s = 1:S
% % plot(all_times, proj1(:, s, 1), '--', 'color', clr(s, :))
% % plot(all_times, proj1(:, s, 2), '-', 'color', clr(s, :))
% % end
% % subplot(212)
% % hold on
% % for s = 1:S
% % plot(all_times, proj2(:, s, 1), '--', 'color', clr(s, :))
% % plot(all_times, proj2(:, s, 2), '-', 'color', clr(s, :))
% % end
%%