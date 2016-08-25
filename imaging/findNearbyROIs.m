function nearbyROIs = findNearbyROIs(COMs, coms_this_roi, dist_th)
% coms_this_roi = [y, x];


if ~exist('dist_th', 'var')
    dist_th = 3; % find ROIs in 3 pixel vicinity of ROI with COM = coms_this_roi
end

xdif= (abs(COMs(:,2) - coms_this_roi(2)));
ydif= (abs(COMs(:,1) - coms_this_roi(1)));
nearbyROIs = find(ydif < dist_th  &  xdif < dist_th);


%%
%{
figure; hold on
imagesc(log(sdImage{2}))
for rr = nearbyROIs'
    plot(COMs(rr,2), COMs(rr,1), 'r.')
end
set(gca, 'ydir', 'reverse')
%}

%%
%{
figure; 
subplot(3,3,[7]); 
imagesc(log(sdImage{2}))

% for j = 1:length(merged_ROIs)
    for i = r' % merged_ROIs{j}'
        set(gcf,'name', sprintf('ROI: %i', i))
        hold on

        subplot(3,3,[1,2,3]), h1 = plot(C(i,:)); 
        hold on
        title(sprintf('tau = %.2f ms', tau(i,2))),  % title(sprintf('%.2f, %.2f', [temp_corr(i), tau(i,2)])), 
        xlim([1 size(C,2)])
        ylabel('C')% (denoised-demixed trace)')

        subplot(3,3,[4,5,6]), h2 = plot(activity_man_eftMask_ch2(:,i)); 
        hold on
        title(sprintf('temp corr = %.2f', temp_corr(i))),  
        xlim([1 size(C,2)])
        ylabel('Raw') % (averaged pixel intensities)')

        subplot(3,3,[7]); hold on
        h3 = plot(CC{i}(2,:), CC{i}(1,:), 'r');
        xlim([COMs(i,2)-50  COMs(i,2)+50])
        ylim([COMs(i,1)-50  COMs(i,1)+50])
    %     imagesc(reshape(A(:,i), imHeight, imWidth))
        title(sprintf('mask #pix = %i', mask_numpix(i)))

        plotCorr_FN(roiPatch, highlightPatchAvg, highlightCorrROI, A, CC, COMs, [imHeight, imWidth], i, [3,3,8], [3,3,9])
        h4 = subplot(3,3,8);
        h5 = subplot(3,3,9);
        
        pause
        delete([h4, h5])
    end
% end


%}


%%
%{
figure; hold on
j = 1;
for i = 1:length(merged_ROIs{j})
    r = merged_ROIs{j}(i);
    trac = C(r,:);
    plot(shiftScaleY(trac))
    pause
end


%%
rr = merged_ROIs{j};
r = rr(highlightCorrROI(rr) > .5)
tau(r,2)
tau(rr,2)
%}


