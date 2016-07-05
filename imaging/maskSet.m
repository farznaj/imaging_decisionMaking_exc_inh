function mask = maskSet(A_or_contoursYX, imHeight, imWidth, isContPixels)
% Set mask (same as A but with no spatial weighing, ie all pixels of a
% components are 1) using contour outlines or pixels.
%
% A_or_contoursYX: if not a cell, it is A (spatial component); if cell it is
% either contour coordinates (CC) or pixels (CR). Default is that pixels
% are provided.
%
% isContPixels: if 1 (default) contour pixels (CR) are provided, if 0 contour
% outlines (CC) are provided.
%
% CC (contour outlines): cell array, each element contains the [y x]
% coordinates of an ROI contour.
%
% CR (contour pixels)   -- cell array of size nROIs x 2. The first column of cells
%                      contains an array of size nPixels x 2, where each
%                      column is a [y x] pair of where a pixel in the ROI
%                      is located. The second column contains an array of size
%                      nPixels x 1, which contains the weight for that
%                      pixel.
%
% mask(:,:,i) shows the spatial location of ROI i in the field of view.
%


if iscell(A_or_contoursYX)
    Aprovided = 0;
    
    if ~exist('isContPixels', 'var')
        isContPixels = 1;
    end
    
else
    Aprovided = 1; % A (spatial component) is provided, so you will compute the contour pixels.
    A = A_or_contoursYX;
end


%%

if Aprovided
    %% A provided instead of contours, so set the contours.
    
    d1 = imHeight;
    d2 = imWidth;
    K = size(A,2);
    thr = .95; % contour_threshold;
    
    CR = cell(K,2);    % store the information of the cell contour
    rawA = A;          % spatial contour for raw signal
    for idx=1:K
        A_temp = full(reshape(A(:,idx),d1,d2));
        A_temp = medfilt2(A_temp,[3,3]);
        A_temp = A_temp(:);
        [temp,ind] = sort(A_temp(:).^2,'ascend');
        temp =  cumsum(temp);
        ff = find(temp > (1-thr)*temp(end),1,'first');
        if ~isempty(ff)
            fp = find(A_temp >= A_temp(ind(ff)));
            [ii,jj] = ind2sub([d1,d2],fp);
            CR{idx,1} = [ii,jj]';
            CR{idx,2} = A_temp(fp)';
            rawA(fp,idx) = 1;
        end
    end
    
    %%%%%
    rawA(rawA~=1) = 0; % FN added to only keep values = 1!
    
    mask = logical(reshape(rawA, imHeight, imWidth, []));
    
    
else
    %% contours provided
    
    if isContPixels % contour pixels provided
        
        mask = false(imHeight*imWidth, size(A_or_contoursYX,1));
        for rr = 1:size(mask,2)
            lind = sub2ind([imHeight, imWidth], A_or_contoursYX{rr,1}(1,:), A_or_contoursYX{rr,1}(2,:));
            mask(lind, rr) = true;
        end
        mask = reshape(mask, [imHeight, imWidth, size(mask,2)]);
        
        
    else % contour outlines provided
        
        % Below uses CC (contour outlines) and inpolygon to set the masks
        
        [X,Y] = meshgrid(1:imWidth, 1:imHeight);
        
        mask = false(imHeight, imWidth, length(A_or_contoursYX));
        
        for rr = 1:length(A_or_contoursYX)
            x = A_or_contoursYX{rr}(2,:);
            y = A_or_contoursYX{rr}(1,:);
            mask(:, :, rr) = inpolygon(X, Y, x, y);
        end
        
    end
end




