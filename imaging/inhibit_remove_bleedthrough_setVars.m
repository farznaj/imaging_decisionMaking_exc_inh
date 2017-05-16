function [a, b] = inhibit_remove_bleedthrough_setVars(A, imHeight, imWidth, aveImage)
% For each ROI set a cell (each element of a and b) that includes the
% frame-averaged pixel values, only including the pixels within 80% contour
% of A of the given ROI.
%
% Use a high contour (only high probable pixels to be included in rawA), bc
% we want to be conservative (calling pixels to be of the same ROI only if they are highly probable in A) when computing bleedthrough slope.
thr = .8; %.95; %.8; % contour_threshold; % FN: at thr 95% of the elements in sorted A_temp.^2 will be included. See the comment below for why we use A_temp.^2 to find thr (since there are lots of 0s in A_temp)....


%% Set rawA, ie A including only pixels within thr contour

d1 = imHeight;
d2 = imWidth;
K = size(A,2);

%     CR = cell(K,2);    % store the information of the cell contour
rawA = zeros(size(A));          % spatial contour for raw signal
for idx=1:K
    A_temp = full(reshape(A(:,idx),d1,d2));
    A_temp = medfilt2(A_temp,[3,3]);
    A_temp = A_temp(:);
    [temp,ind] = sort(A_temp(:).^2,'ascend');
    temp =  cumsum(temp); % FN: The idea is that we find percentile not on a linear scale, instead on cumsum(A_temp.^2) which spans [0 1] however it's highly exponential. As a result, the index at which cumsum(A_temp.^2)=.05 percentile will be very high, hence sorted(A_temp) will be high (and not 0) at this high index. If we were going to use the regular percentile, because it uses a linear scale, the index corresponding to .05 percentile will be very low, hence A_temp will also be very low (almost always 0 regardless of percentile value!). Regular percentile:  pctl = @(v,p) interp1(linspace(0.5/length(v), 1-0.5/length(v), length(v))', sort(v), p*0.01, 'spline')
    ff = find(temp > (1-thr)*temp(end),1,'first'); % temp = cumsum(sorted_A_temp.^2)
    if ~isempty(ff)
        fp = find(A_temp >= A_temp(ind(ff)));
%         [ii,jj] = ind2sub([d1,d2],fp);
        %             CR{idx,1} = [ii,jj]';
        %             CR{idx,2} = A_temp(fp)';
        rawA(fp,idx) = A(fp,idx);
    end
end

%%%%%
% rawA(rawA~=1) = 0; % FN added to only keep values = 1 and set the rest to 0!


%% For each ROI get aveImage1 and aveImage2 (only for thresholded (80% contour) non0 pixels in A)

ave_ch1m_th = cell(1, size(A,2));
ave_ch2m_th = cell(1, size(A,2));

for rr = 1:size(A,2)

    Anow = rawA(:,rr); 
    AMat = reshape(Anow, [imHeight, imWidth]);
    
%     AClip = Apatch0{rr}; % if you want patch_ch1m_th to include all non-0 pixels of A
%     AClip = Apatch_th{rr}; % if you want patch_ch1m_th to include thresholded pixels of A
    
    % inhibit channel
    % First set roiIm, ie the ROI on the inhibit channel
    roiMask = AMat;
    roiMask(AMat~=0) = true;
    roiMask(~AMat) = nan;
    
    % use the non-bleedthrough corrected inh patch
    roiNow = aveImage{1} .* roiMask; % image of ch2 ROI on ch1. % figure; imagesc(roiIm)
%     inhClip(~inhClip) = NaN;
    ave_ch1m_th{rr} = roiNow;
    
    roiNow = aveImage{2} .* roiMask; % image of ch2 ROI on ch1. % figure; imagesc(roiIm)
%     inhClip(~inhClip) = NaN;
    ave_ch2m_th{rr} = roiNow;    
end


%% Set slope for removing bleedthrough using pixel values of each ROI on the aveImage (instead of using the traces manActNoSpike)

% Take the mask of patch_ch1, patch_ch2... then for each ROI run regress between pixel values of the two images!

maxIter = 100;
% onlyRegress = 1; % commonSlope solution works worse than regular regression after I removed spikes from macAct_ch2.... 

a = cellfun(@(x)full(x(:)), ave_ch1m_th, 'uniformoutput', 0); % all pixels of each ROI
b = cellfun(@(x)full(x(:)), ave_ch2m_th, 'uniformoutput', 0);
% remove nans (pixels not belonging to ROI)
a = cellfun(@(x)x(~isnan(x)), a, 'uniformoutput', 0);
b = cellfun(@(x)x(~isnan(x)), b, 'uniformoutput', 0);

% a = cellfun(@(x)x(:), patch_ch1, 'uniformoutput', 0); % all pixels of each ROI
% b = cellfun(@(x)x(:), patch_ch2, 'uniformoutput', 0);

% only use the mask pixels.


%%
% tic
% [slope_common, offsets_ch1] = inhibit_remove_bleedthrough(a, b, maxIter); %, onlyRegress);
% t = toc;
% disp(t)


% Remove bleedthrough from aveImage1
% corrInh = aveImage{1} - slope_common * aveImage{2};

%{
figure; imagesc(corrInh)
figure; imagesc(aveImage{1})
figure; imagesc(aveImage{2})

hold on
for rr = 1:size(A,2) %find(corr_A_inh_mth >= thi)
    plot(CC{rr}(2,:) , CC{rr}(1,:), 'color', 'r')
end
%}
