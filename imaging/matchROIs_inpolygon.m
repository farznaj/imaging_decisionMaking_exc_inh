%% Use inpolygon to find matching ROIs between the manual and Eft method.
% a = inpolygon(rois{1}.mnCoordinates(:,1), rois{1}.mnCoordinates(:,2), CC{2}(2,2:end), CC{2}(1,2:end));
commonROI = false(length(CC), length(rois));
inpolyROIMeasure = NaN(length(CC), length(rois));
for iman = 1:length(rois)
    xq = rois{iman}.mnCoordinates(:,1)';
    yq = rois{iman}.mnCoordinates(:,2)';
    
    for ieft = 1:length(CC) % find(commonROI(:,iman))'         
%         xv = CC{ieft}(2,2:end);
%         yv = CC{ieft}(1,2:end);
        regI = 1;
        xv = []; yv = [];
        while regI < size(CC{ieft}, 2)
            nElem = CC{ieft}(1, regI);            
            xv = [xv, CC{ieft}(2, regI+(1:nElem))]; 
            yv = [yv, CC{ieft}(1, regI+(1:nElem))];            
            regI = regI + nElem + 1;
        end
%     hold on, plot(xv,yv)
        commonROI(ieft, iman) = any([inpolygon(xq, yq, xv, yv) , inpolygon(xv, yv, xq, yq)]);        
        
        inpolyROIMeasure(ieft, iman) = max(sum(inpolygon(xq, yq, xv, yv)), sum(inpolygon(xv, yv, xq, yq))); % how many "overlapping" points between manual and eftychios ROIs.
        
    end
end

figure; plot(sum(commonROI)) % shows the number of eft ROIs that match each manual ROI. 


%% ROI matches based on the inpolygon method
% find(sum(commonROI)>1) % these manual ROIs were matched with >1 eft ROI, so they need evaluation.
[m, eftMatchIdx_inpoly] = max(inpolyROIMeasure); % i_eft_match is the index of eftychios ROI that matches each manual ROI based on which Eft ROI had the max inpolygon with the manual ROI.
eftMatchIdx_inpoly(~sum(inpolyROIMeasure)) = NaN; % there are the ROIs that had not match with Eft ROIs.

% allEftMatchMeasure = sparse(inpolyROIMeasure(:, sum(commonROI)>1)); % shows commonROIMeasure for all matched Efy ROIs with each manual ROI. % remember the column index corresponds to the find(sum(commonROI)>1) array (not the manual ROI), since it is a sparce matrix, but the row index is the correct index correponsing to the Eft ROIs.

figure; plot(eftMatchIdx_inpoly), xlabel('Manual ROI index'), ylabel('matching Eft ROI index')

