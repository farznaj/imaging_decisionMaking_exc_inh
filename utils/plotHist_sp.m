function plotHist_sp(y1,y2,xlab,ylab,leg, cols, tit, fign, sp,yy)

    if ~exist('tit','var')
        tit = [];
    end
    
    r1 = round(min([y1(:);y2(:)]),1); 
    r2 = round(max([y1(:);y2(:)])+.05,1);
    bins = r1 : (r2-r1)/10 : r2;

    [nexc, e] = histcounts(y1(:), bins);
    [ninh, e] = histcounts(y2(:), bins);

    x = mode(diff(bins))/2 + bins; x = x(1:end-1);
    ye = nexc/sum(nexc);
    yi = ninh/sum(ninh);
    %     ye = smooth(ye);
    %     yi = smooth(yi);
    
    figure(fign)
    hold(sp,'on')
    plot(sp, x, ye, 'color', cols{1})
    plot(sp, x, yi, 'color', cols{2})
    if ~isempty(yy)
        plot(sp, [yy yy],[0 max([ye,yi])], 'k:')
    end

    [h,p] = ttest2(y1(:),y2(:));
    if isempty(tit)
        title(sp, sprintf('p=%.3f',round(p,3)))
    else
        title(sp, sprintf('%dms; p=%.3f',tit,round(p,3)))
    end
    
    
    
