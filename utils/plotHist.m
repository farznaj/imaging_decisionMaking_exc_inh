function [fh,bins] = plotHist(y1,y2,xlab,ylab,leg, cols, yy, fh, nBins, doSmooth, lineStyles, sp) 
    
    % optional:
    % yy: draw a vertical line at yy
    % fh: figure handle
    %
    % leg: legend
    % eg inputs:
    %{
    cols = {'k', 'r'};
    leg = {'y1','y2'};
    ylab = 'Fraction neurons';
    xlab = 'ROC';
    %}
    
    if ~exist('nBins','var')
        nBins = 10;
    end
    
    if ~exist('fh','var')
        fh = [];
    end
    
    if ~exist('yy','var')
        yy = [];
    end
    
    if ~iscell(cols)
        cols = mat2cell(cols,[1,1]);
    end
    
    if ~exist('lineStyles', 'var')
        lineStyles = {'-','-'};
    end
    
    if ~exist('sp', 'var')
        sp = [211, 212];
    end
    
    
    %% set the histogram vars
    
    % set the bins
    ally = [y1(:);y2(:)];
    r1 = round(min(ally)-.05, 1);  %min(ally); %   % round(min(ally), 1); 
    r2 = round(max(ally)+.05, 1);
    bins = r1 : (r2-r1)/nBins : r2;
    
    % get the counts in each bin
    [nexc, e] = histcounts(y1(:), bins);
    [ninh, e] = histcounts(y2(:), bins);
    
    % turn counts to fractions (of total elements that exist in each bin)
    % this is the y for plotting hists
    ye = nexc/sum(nexc);
    yi = ninh/sum(ninh);
    ye_cs = cumsum(ye);
    yi_cs = cumsum(yi);
    
    if doSmooth
        ye = smooth(ye);
        yi = smooth(yi);
        ye_cs = smooth(ye_cs);
        yi_cs = smooth(yi_cs);        
    end

    
    % set x for plotting hists as the center of bins
    x = mode(diff(bins))/2 + bins; x = x(1:end-1);
    
    
    %% plots
    
    if isempty(fh)
        fh = figure;
    else
        figure(fh)
    end
    
    subplot(sp(1)), hold on
    h1=plot(x, ye, 'color', cols{1}, 'linestyle', lineStyles{1});
    h2=plot(x, yi, 'color', cols{2}, 'linestyle', lineStyles{1});
    xlabel(xlab); ylabel(ylab)  %     xlim([r1,r2])
    legend([h1,h2], leg)
    if ~isempty(yy)
        plot([yy yy],[0 max([ye,yi])], 'k:')
    end
    a = gca;
    [h,p] = ttest2(y1(:), y2(:));
%     title(sprintf('p(ttest2) = %.3f', round(p,3)))
    
    %%%%%%%%%% cumsum %%%%%%%%%%
    subplot(sp(2)), hold on
    h1=plot(x, ye_cs, 'color', cols{1}, 'linestyle', lineStyles{1});
    h2=plot(x, yi_cs, 'color', cols{2}, 'linestyle', lineStyles{1});
    xlabel(xlab); ylabel('Cumulative sum')    
    legend([h1,h2], leg)
    if ~isempty(yy)
        plot([yy yy],[0 1], 'k:')
    end
    a = [a, gca];
    % show p value
    title(sprintf('p(ttest2) = %.3f', round(p,3)))
    
    linkaxes(a, 'x')
    
    
