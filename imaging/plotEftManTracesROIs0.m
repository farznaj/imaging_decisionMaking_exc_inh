function plotEftManTracesROIs0(refTrace, refTrace_df, matchedTrace, matchedTrace_df, eftMatchIdx_mask, CC, rois, inds2plot)

figure;

for iref = inds2plot % 1:length(eftMatchIdx_mask)
    
    set(gcf,'name', sprintf('ROI %d', iref))
    imatched = eftMatchIdx_mask(iref);
    %     imatched = eftMatchIdx_inpoly(iref);
    
    %% compare manual and Eft ROI contours    
    subplot(431)
    hold on;
    plot(CC{iref}(2,:), CC{iref}(1,:), 'k')
    
    if ~isnan(imatched)
        plot(rois{imatched}.mnCoordinates(:,1), rois{imatched}.mnCoordinates(:,2), '-', 'color', [77 190 238]/256)
    end
    
    %% raw traces
    subplot(4,1,2)
    hold on;
    if ~isnan(imatched)
        plot(matchedTrace(imatched,:), 'color', [77 190 238]/256)
    end
    plot(refTrace(iref,:), 'k')
    title('Raw traces')
    
    %% scaled and shifted version so the 2 plots overlap.
    subplot(413)
    hold on;
    
    if ~isnan(imatched)
        top = matchedTrace(imatched,:);
        topbs = top - quantile(top, .1);
        topbsn = topbs / max(topbs);        
        plot(topbsn, 'color', [77 190 238]/256)
    end
    
    top = refTrace(iref,:);
    topbs = top - quantile(top, .1);
    topbsn = topbs / max(topbs);    
    plot(topbsn, 'k')
    title('Scaled and shifte raw traces')
    
    %% df/f    
    subplot(414)
    hold on;
    if ~isnan(imatched)
        plot(matchedTrace_df(imatched,:), 'color', [77 190 238]/256)
    end
    plot(refTrace_df(iref,:), 'k')
    title('DF/F traces')
    
    %%
    pause
    subplot(431), clf % delete(gca)
    subplot(412), clf % delete(gca)
    subplot(413), clf % delete(gca)
    subplot(414), clf % delete(gca)

end

