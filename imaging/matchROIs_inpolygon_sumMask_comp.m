% evaluation of the mask method when there are >1 matches for a single ROI and comparison of its output with the inpolygon method.
figure;

eftMatchIdx_mask = NaN(1, length(rois));
for iman = 1:length(rois)
    f = find(maskOverlapMeasure(:,iman)>0);
    
    if length(f)==1
        eftMatchIdx_mask(iman) = f;
        
    elseif length(f)>1
        
        xq = rois{iman}.mnCoordinates(:,1)';
        yq = rois{iman}.mnCoordinates(:,2)';
        
        hold on
        plot(xq, yq, 'linewidth', 2)    
        
        for ieft = f'
            
            regI = 1;
            xv = []; yv = [];
            while regI < size(CC{ieft}, 2)
                nElem = CC{ieft}(1, regI);            
                xv = [xv, CC{ieft}(2, regI+(1:nElem))]; 
                yv = [yv, CC{ieft}(1, regI+(1:nElem))];            
                regI = regI + nElem + 1;
            end
            
            plot(xv, yv)
            pause
        end
        
        maskMethod = [f maskOverlapMeasure(f,iman)]
        [~,ff] = max(maskOverlapMeasure(:,iman));
        eftMatchIdx_mask(iman) = ff;
        
        ieft = ff;            
        regI = 1;
        xv = []; yv = [];
        while regI < size(CC{ieft}, 2)
            nElem = CC{ieft}(1, regI);
            xv = [xv, CC{ieft}(2, regI+(1:nElem))];
            yv = [yv, CC{ieft}(1, regI+(1:nElem))];
            regI = regI + nElem + 1;
        end        
        plot(xv, yv, 'r', 'linewidth', 2)
            
        % comparison with the polygon method:
        inpolyMethod = [find(inpolyROIMeasure(:,iman)) inpolyROIMeasure(inpolyROIMeasure(:,iman)~=0, iman)]
        [~, inF] = max(inpolyROIMeasure(:,iman));
        
        maskF_inF = [ff, inF]
        
        
        pause
        delete(gca)
    end
end
