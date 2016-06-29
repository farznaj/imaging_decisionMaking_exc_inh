function traceQualManual = plotEftManTracesROIs(C_df, S_df, dFOF, A2, mask_eft, CC, CC_rois, eftMatchIdx_mask, im, C, inds2plot, manualTraceQual, plothists, im2)
% use the following to evaluate A and C of Efty's algorithm without
% comparing with the manual method:
% inds2plot = 1:size(C,1);
% plotEftManTracesROIs(C_df, S, [], A, [], CC, [], [], im, C, inds2plot, 0, 0, []);
%
% remember if you input A2, then you need to input CC and im too.
% traces are required in format units x frames
%
% Plot ROIs and traces found by Eft and manual methods.
% im = sdImage{2};
% manualTraceQual = 1; % if 1, a question dialogue box willb e shown that asks about traces quality.
%

%%
traceQualManual = NaN(1, length(CC));

% refTrace = C2;
% matchedTrace = activity;
refTrace_df = C_df;
if ~isempty(dFOF)
    matchedTrace_df = dFOF;
    toMatchTrace = true;
else
    toMatchTrace = false;
end

screensize = get( 0, 'Screensize' );
h = figure('position', screensize);
% h = figure('position', [50        -167        1365         973]);
h1_0 = subplot('position', [0.0286    0.6695    0.2145    0.3046]); % [0.0425    0.6588    0.2549    0.3165]
h1_1 = subplot('position', [0.2472    0.6695    0.2145    0.3046]);
h1_2 = subplot('position', [0.4659    0.6695    0.2145    0.3046]);
h2 = subplot('position', [0.7099    0.6987    0.1258    0.2696]); % 0.3241    0.6902    0.2539    0.2862
h3 = subplot('position', [0.8628    0.6987    0.1258    0.2696]); % 0.6029    0.6537    0.1736    0.3211
htr_1 = subplot('position', [0.0529    0.4430    0.9288    0.1708]);
htr_2 = subplot('position', [0.0529    0.2557    0.9288    0.1708]);
htr_3 = subplot('position', [0.0529    0.0671    0.9288    0.1708]);


% plot sdImage (or medImage, etc)
if ~isempty(im)
    
    subplot(h1_0)
    imagesc(im)
    freezeColors
    hold on
    for rr = 1:length(CC)
        plot(CC{rr}(2,:), CC{rr}(1,:), 'color', [255 215 0]/255)
    end
    
    
    subplot(h1_1)
    imagesc(im)
    freezeColors
    set(gca, 'yticklabel', '')
    
end


if exist('im2', 'var')
    subplot(h1_2)
    imagesc(im2)
    freezeColors
    set(gca, 'yticklabel', '')
end


if plothists
    histf = figure;
end


%%
cnt = 0;
for rr = inds2plot
    
    cnt = cnt+1;
    set(gcf, 'name', sprintf('ROI  %d / %d', cnt, length(inds2plot)))
    clear hp
    
    sn = S_df(rr,:);
    sn = sn/max(sn);
    
    if ~isempty(A2)
        spcomp = reshape(A2(:,rr), size(im,1), size(im,2));
        
        xl = [find(sum(spcomp), 1, 'first')  find(sum(spcomp), 1, 'last')];
        yl = [find(sum(spcomp,2), 1, 'first')  find(sum(spcomp,2), 1, 'last')];
    end
    
    if ~isempty(eftMatchIdx_mask)
        imatched = eftMatchIdx_mask(rr);
    end
    
    
    %% compute peaks on the C_df trace.
    bl = quantile(refTrace_df(rr,:), .1);
    [pks,~, ~, prom] = findpeaks(refTrace_df(rr,:), 'minpeakheight', bl+.05); %'minpeakdistance', 3, 'threshold', .005);
    
    measQual = ((mean(pks)/std(sn))-3) .* ((mean(prom)/std(sn))-1);
    a = sign((mean(pks)/std(sn))-3)<0 & sign((mean(prom)/std(sn))-1)<0;
    measQual(a) = -measQual(a);
    
    %     sigAmp = nanmean(pks) - max(0,bl);
    
    %     [mean(pks) mean(p)]
    %     a = sgolayfilt(refTrace_df(rr,:), 7, 21);
    %     [length(locs), sum(S_df(rr,:) > .01)]
    %     a = refTrace_df(rr,:);
    %     s = S_df(rr,:);
    %     sigAmp_stdLessBLplus05_stdLessQuant25_sSmallAve_sAve(rr,:) = [sigAmp, std(a(a<bl+.05)), std(a(a<quantile(a, .25))), ...
    %         nanmean((s>.001 & s<.01)), nanmean(s>.001)];
    %     mean(sigAmp)/std(sn)
    
    
    %% plot the contours on the sdImage
    hp = [];
    
    % plot ref image with all contours. the contour of interest made thick.
    subplot(h1_0), hold on
    hp(1) = plot(CC{rr}(2,:), CC{rr}(1,:), 'r', 'linewidth', 2);
    xlim(xl+[-60 60]), ylim(yl+[-60 60])
    
    % plot ref image w contours
    subplot(h1_1), hold on
    hpn = plot(CC{rr}(2,:), CC{rr}(1,:), 'r:');
    hp = [hp hpn];
    xlim(xl+[-60 60]), ylim(yl+[-60 60])
    
    if ~isempty(mask_eft)
        m = mask_eft(:,:,rr);
        title(['ROI: ', num2str(rr), ' - Mask size: ', num2str(sum(m(:)))])
    end
    
    if exist('imatched', 'var') && ~isnan(imatched) && ~isempty(CC_rois)
        hpn = plot(CC_rois{imatched}(2,:), CC_rois{imatched}(1,:), ':', 'color', [255 215 0]/255);
        hp = [hp hpn];
    end
    
    % plot toMatch image w contours
    if exist('im2', 'var') && ~isempty(im2)
        % overlap the contour on the other image as well.
        subplot(h1_2), hold on
        hpn = plot(CC{rr}(2,:), CC{rr}(1,:), 'r:');
        xlim(xl+[-60 60]), ylim(yl+[-60 60])
        hp = [hp hpn];
        
        if ~isnan(imatched) && ~isempty(CC_rois) 
            hpn = plot(CC_rois{imatched}(2,:), CC_rois{imatched}(1,:), ':', 'color', [255 215 0]/255);
            hp = [hp hpn];
        end
    end
    
    
    
    
    %% plot spatial component and the contour
    if ~isempty(A2)
        
        subplot(h2)
        imagesc(spcomp)
        
        freezeColors
        xlim(xl+[-5 5])
        ylim(yl+[-5 5])
        
        hold on;
        hpn = plot(CC{rr}(2,:), CC{rr}(1,:), 'r');
        hp = [hp hpn];
    end
    
    
    %% compare manual and Eft ROI contours
    if exist('imatched', 'var') && ~isempty(CC_rois)
        if ~isempty(CC)
            
            subplot(h3)
            hold on;
            hpn = plot(CC{rr}(2,:), CC{rr}(1,:), 'k');
            hp = [hp hpn];
            
            if ~isnan(imatched)
                hpn = plot(CC_rois{imatched}(2,:), CC_rois{imatched}(1,:), '-', 'color', [77 190 238]/256);
                %         hpn = plot(rois{imatched}.mnCoordinates(:,1), rois{imatched}.mnCoordinates(:,2), '-', 'color', [77 190 238]/256);
                hp = [hp hpn];
            end
            set(gca, 'ydir', 'reverse')
            
        end
    end
    
    %% plot C (or C_df)
    top = C(rr,:); % C_df(rr,:);
    
    subplot(htr_1);
    hold on
    hpn = plot(top, 'k');
    hp = [hp hpn];
    xlim([0 size(top,2)+1])
    ylim([min(top)-.05  max(top)+.05])
    ylabel('C')
    
    title(sprintf('pks %.2f   prom %.2f   prod %.2f   meas %.2f', mean(pks)/std(sn), mean(prom)/std(sn), ...
        mean(pks)/std(sn)*mean(prom)/std(sn), measQual))
    
    
    %% plot normalized S_df
    subplot(htr_2);
    hpn = plot(sn, 'r');
    hp = [hp hpn];
    xlim([0 size(refTrace_df,2)+1])
    if ~all(isnan(sn))
        ylim([min(sn)-.05  max(sn)+.05])
    end
    ylabel('Normalized S')
    
    
    %% compare Eft trace with manually computed trace
    %%% scaled and shifted version so the 2 plots overlap.
    %{
    subplot(h4)
    hold on;
    if ~isnan(imatched)
        top = matchedTrace(imatched,:);
        topbs = top - quantile(top, .1);
        topbsn = topbs / max(topbs);
        h42 = plot(topbsn, 'color', [77 190 238]/256);
    end
    
    top = refTrace(rr,:);
    topbs = top - quantile(top, .1);
    topbsn = topbs / max(topbs);
    h41 = plot(topbsn, 'k');

    xlim([0 size(C_df,2)+1])
    ylim([min(C_df(rr,:))-.05  max(C_df(rr,:))+.05])
    %}
    
    tr = refTrace_df(rr,:);
    
    %%% matched df/f
    subplot(htr_3)
    hold on;
    if ~isempty(dFOF) && ~isnan(imatched)
        hpn = plot(matchedTrace_df(imatched,:), 'color', [77 190 238]/256);
        hp = [hp hpn];
        
        tr = [tr, matchedTrace_df(imatched,:)];
    end
    
    %%%%% ref df/f
    hpn = plot(refTrace_df(rr,:), 'k');
    hp = [hp hpn];
    
    xlim([0 size(refTrace_df,2)+1])
    ylim([min(tr)-.05  max(tr)+.05])
    ylabel('C\_df')
    
    %%%%%
    if ~isempty(dFOF) && ~isnan(imatched)
        ln = min(length(matchedTrace_df(imatched,:)),  length(refTrace_df(rr,:)));
        c = corr(matchedTrace_df(imatched, 1:ln)' , refTrace_df(rr, 1:ln)');
        handle = title(sprintf('corr: %.2f', c));
        v = axis;
        set(handle,'Position',[v(2)*.5 v(4)-v(4)*.05 0]);
        %     xlabel(sprintf('corr: %.2f', c));
    end
    
    
    %% plot some hists.
    if plothists
        figure(histf)
        hold on
        
        % plot hist of sn (normalized S_df)
        %     edges = [0 .001 .002 .003 .004 .007 .01 .02 .04 .08 .15 .3 .6 1];
        edges = [0 .001 .01:.01:1];
        [N] = histcounts(sn, edges, 'Normalization', 'probability');
        hpn = plot(edges(1:end-1) + mode(diff(edges))/2, N);
        hp = [hp hpn];
        xlim([min(edges)-.05  max(edges)+.05])
        ylim([-.05 1.05])
        
        % plot hist of C_df
        %     edges = [min(C_df(rr,:)) max(C_df(rr,:))];
        %     [N] = histcounts(C_df(rr,:), edges, 'Normalization', 'probability');
        [N, edges] = histcounts(refTrace_df(rr,:), 'Normalization', 'probability');
        hpn = plot(edges(1:end-1) + mode(diff(edges))/2, N, 'colo', 'g');
        hp = [hp hpn];
        mn = min(edges);
        mx = max(edges);
        %     xlim([min(edges)-.05  max(edges)+.05])
        %     ylim([-.05 1.05])
        
        
        [N, edges] = histcounts(pks, 'Normalization', 'probability');
        hpn = plot(edges(1:end-1) + mode(diff(edges))/2, N, 'k');
        hp = [hp hpn];
        mn = min([mn,edges]);
        mx = max([mx,edges]);
        %     xlim([min(edges)-.05  max(edges)+.05])
        %     ylim([-.05 1.05])
        
        [N, edges] = histcounts(prom, 'Normalization', 'probability');
        hpn = plot(edges(1:end-1) + mode(diff(edges))/2, N, 'b');
        hp = [hp hpn];
        mn = min([mn,edges]);
        mx = max([mx,edges]);
        
        xlim([min(mn)-.05  max(mx)+.05])
        ylim([-.05 1.05])
        legend('C\_df','pks','prom')
        
    end
    
    %%
    hAllAxes = findobj(gcf,'type','axes');
    set(hAllAxes, 'tickdir', 'out', 'box', 'off')
    
    if manualTraceQual
        traceQualManual(rr) = bttnChoiseDialog_pos({'Good', 'Ok', 'Bad'}, 'Trace quality?', 'Good', '', ...
            [1;3], [.5, .9, .1, .1]); % (dlgOptions, dlgTitle, defOption, qStr, bttnsOredring, pos)
    else
        drawnow, pause
    end
    %     answer = questdlg('Trace quality? ', '1 Good, 2 Ok, 3 Bad', '1', '2', '3', 'abort', '1');
    
    delete(hp)
    
    
end

