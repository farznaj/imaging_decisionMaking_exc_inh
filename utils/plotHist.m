function fh = plotHist(y1,y2,xlab,ylab,leg, cols, yy, fh) % draw a vertical line at yy

    if ~exist('fh','var')
        fh = [];
    end
    
    if ~iscell(cols)
        cols = mat2cell(cols,[1,1]);
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

    if isempty(fh)
        fh = figure;
    else
        figure(fh)
    end
    
    subplot(211), hold on
    plot(x, ye, 'color', cols{1})
    plot(x, yi, 'color', cols{2})
    xlabel(xlab)
    ylabel(ylab)
    legend(leg)
    if ~isempty(yy)
        plot([yy yy],[0 max([ye,yi])], 'k:')
    end
    a = gca;
    % print p value
    [h,p] = ttest2(y1(:),y2(:));
    title(sprintf('p(ttest2) = %.3f', round(p,3)))
    
    subplot(212), hold on
    plot(x, cumsum(ye), 'color', cols{1})
    plot(x, cumsum(yi), 'color', cols{2})
    xlabel(xlab)
    ylabel('Cumulative sum')
    legend(leg)
    if ~isempty(yy)
        plot([yy yy],[0 1], 'k:')
    end
    a = [a, gca];

    linkaxes(a, 'x')
