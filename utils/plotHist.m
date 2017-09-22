function plotHist(y1,y2,xlab,ylab,leg)

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

    figure;
    subplot(211), hold on
    plot(x, ye)
    plot(x, yi)
    xlabel(xlab)
    ylabel(ylab)
    legend(leg)
    plot([.5 .5],[0 max([ye,yi])], 'k:')
    a = gca;
    % print p value
    [h,p] = ttest2(y1(:),y2(:));
    title(sprintf('ttest2 p = %.3f',round(p,3)))
    
    subplot(212), hold on
    plot(x, cumsum(ye))
    plot(x, cumsum(yi))
    xlabel(xlab)
    ylabel('Cumulative sum')
    legend(leg)
    plot([.5 .5],[0 max([ye,yi])], 'k:')
    a = [a, gca];

    linkaxes(a, 'x')
