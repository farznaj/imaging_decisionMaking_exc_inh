function [bins,ye,yi,x,h1,h2] = plotHist_sp(y1,y2,xlab,ylab,leg, cols, tit, fign, sp,yy, documsum, nBins, bins, doSmooth)
    
    if ~exist('sp', 'var') | isempty(sp)
        sp = subplot(1,1,1);
    end
    
    if ~exist('doSmooth', 'var')
        doSmooth = 0;
    end
    
    if ~exist('tit','var')
        tit = [];
    end
    
    if ~exist('documsum','var')
        documsum = 0;
    end
    
    if ~iscell(cols)
        cols = mat2cell(cols,[1,1]);
    end
    
    if ~exist('nBins', 'var')
        nBins = 10;
    end

        
    if ~exist('bins','var') | isempty(bins)
        ally = [y1(:);y2(:)];
        r1 = round(min(ally)-.05, 1);  %min(ally); %   % round(min(ally), 1); 
        r2 = round(max(ally)+.05, 1);
        bins = r1 : (r2-r1)/nBins : r2;
    end

    [nexc, e] = histcounts(y1(:), bins);
    [ninh, e] = histcounts(y2(:), bins);

    x = mode(diff(bins))/2 + bins; x = x(1:end-1);

    if documsum
        nexc = cumsum(nexc);
        ninh = cumsum(ninh);   
    end
    
    ye = nexc/sum(nexc);
    yi = ninh/sum(ninh);

    if doSmooth
        ye = smooth(ye,5);
        yi = smooth(yi,5);
    end
    
    figure(fign)
    hold(sp,'on')
    h1 = plot(sp, x, ye, 'color', cols{1});
    h2 = plot(sp, x, yi, 'color', cols{2});
    if ~isempty(yy)
        plot(sp, [yy yy],[0 max([ye(:);yi(:)])], 'k:')
    end

    [h,p] = ttest2(y1(:),y2(:));
    if isempty(tit)
        title(sp, sprintf('p=%.3f',round(p,3)))
    else
        if isnumeric(tit)
            title(sp, sprintf('%dms; p=%.3f',tit,round(p,3)))
        else
            title(sp, sprintf('%s; p=%.3f',tit,round(p,3)))
        end
    end
    
    
    
