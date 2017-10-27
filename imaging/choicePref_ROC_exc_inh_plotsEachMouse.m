%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Plots of each mouse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

savefigs = eachMouse_do_savefigs(2);
cols = {'b','r'}; % exc,inh, real data
colss = [0,.8,.8; .8,.5,.8]; % exc,inh colors for shuffled data

% im = 1;
for im = 1:length(mice)

    mouse = mice{im};    
    dirn = fullfile(dirn0, mouse);
    cd(dirn)
    
    % these are vars that are aligned across days for each mouse (they are
    % not the ones aligned across all mice).
    choicePref_exc_aligned = choicePref_exc_aligned_allMice{im}; % days; each day: frs x ns
    choicePref_inh_aligned = choicePref_inh_aligned_allMice{im};
    choicePref_exc_aligned_shfl = choicePref_exc_aligned_allMice_shfl{im}; % days; each day: frs x ns
    choicePref_inh_aligned_shfl = choicePref_inh_aligned_allMice_shfl{im};
    time_aligned = time_aligned_allMice{im};
    nowStr = nowStr_allMice{im};
    mnTrNum = mnTrNum_allMice{im};
    days = days_allMice{im};
    corr_ipsi_contra = corr_ipsi_contra_allMice{im};
    set(groot,'defaultAxesColorOrder',cod)    
    
    
    %% Average AUC across neurons for each day and each frame

    aveexc = cellfun(@(x)mean(x,2), choicePref_exc_aligned, 'uniformoutput',0); % average across neurons
    aveexc = cell2mat(aveexc); % frs x days
    seexc = cellfun(@(x)std(x,[],2)/sqrt(size(x,2)), choicePref_exc_aligned, 'uniformoutput',0); % standard error across neurons
    seexc = cell2mat(seexc); % frs x days    

    aveinh = cellfun(@(x)mean(x,2), choicePref_inh_aligned, 'uniformoutput',0);
    aveinh = cell2mat(aveinh);
    seinh = cellfun(@(x)std(x,[],2)/sqrt(size(x,2)), choicePref_inh_aligned, 'uniformoutput',0);
    seinh = cell2mat(seinh); % frs x days        

    % shfl
    aveexc_shfl = cellfun(@(x)mean(x,2), choicePref_exc_aligned_shfl, 'uniformoutput',0); % average across neurons (already averaged across samps)
    aveexc_shfl = cell2mat(aveexc_shfl); % frs x days
    seexc_shfl = cellfun(@(x)std(x,[],2)/sqrt(size(x,2)), choicePref_exc_aligned_shfl, 'uniformoutput',0); % standard error across neurons 
    seexc_shfl = cell2mat(seexc_shfl); % frs x days

    aveinh_shfl = cellfun(@(x)mean(x,2), choicePref_inh_aligned_shfl, 'uniformoutput',0);
    aveinh_shfl = cell2mat(aveinh_shfl);
    seinh_shfl = cellfun(@(x)std(x,[],2)/sqrt(size(x,2)), choicePref_inh_aligned_shfl, 'uniformoutput',0);
    seinh_shfl = cell2mat(seinh_shfl); % frs x days x samps        


    % run ttest across days for each frame
    % ttest: is exc (neuron-averaged ROC pooled across days) ROC
    % different from inh ROC? Do it for each time bin seperately.
    [h,p] = ttest2(aveexc',aveinh'); % 1 x nFrs
    hh0 = h;
    hh0(h==0) = nan;
    

    %% Pool AUC across all neurons of all days (for each frame)

    excall = cell2mat(choicePref_exc_aligned); % nFrs x n_exc_all
    inhall = cell2mat(choicePref_inh_aligned); % nFrs x n_inh_all
    size(excall), size(inhall)
    % shfl
    excall_shfl = cell2mat(choicePref_exc_aligned_shfl); % nFrs x n_exc_all x samps
    inhall_shfl = cell2mat(choicePref_inh_aligned_shfl); % nFrs x n_inh_all x samps        
    size(excall_shfl), size(inhall_shfl)

    % ttest: is exc (single neuron ROC pooled across days) ROC
    % different from inh ROC? Do it for each time bin seperately.
    h = ttest2(excall', inhall'); % h: 1 x nFrs
    hh = h;
    hh(h==0) = nan;
%     h = ttest2(excall_shfl', inhall_shfl')


    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Plot AUC timecourse averaged across days

    fh = figure('position',[10   556   792   383]);

    %%%%%%%%% Average and se across days; each day is already averaged across neurons; done seperately for each time bin
    subplot(121); hold on
    h1 = boundedline(time_aligned, nanmean(aveexc,2), nanstd(aveexc,0,2)/sqrt(nGoodDays(im)), 'b', 'alpha'); % sum(mnTrNum>=thMinTrs)
    h2 = boundedline(time_aligned, nanmean(aveinh,2), nanstd(aveinh,0,2)/sqrt(nGoodDays(im)), 'r', 'alpha');
    % shfl
    h1 = boundedline(time_aligned, nanmean(aveexc_shfl,2), nanstd(aveexc_shfl,0,2)/sqrt(nGoodDays(im)), 'cmap', colss(1,:), 'alpha'); % sum(mnTrNum>=thMinTrs)
    h2 = boundedline(time_aligned, nanmean(aveinh_shfl,2), nanstd(aveinh_shfl,0,2)/sqrt(nGoodDays(im)), 'cmap', colss(2,:), 'alpha');    
    a = get(gca, 'ylim');    
    plot(time_aligned, hh0*(a(2)-.05*diff(a)), 'k.')    
    plot([0,0],a,'k:')
    b = get(gca, 'xlim');
    if ~isempty(yy)
        plot(b, [yy,yy],'k:')
    end
    legend([h1,h2], {'Excitatory', 'Inhibitory'}, 'position', [0.1347    0.8177    0.1414    0.0901]);
    xlabel('Time since choice onset (ms)')
    ylabel(namc)
    title('mean+se days (Ns aved per day)')


    %%%%%%%%%% Average and se across all neurons of all days; done seperately for each time bin
    subplot(122); hold on
    h1 = boundedline(time_aligned, nanmean(excall,2), nanstd(excall,0,2)/sqrt(sum(~isnan(excall(1,:)))), 'b', 'alpha');
    h2 = boundedline(time_aligned, nanmean(inhall,2), nanstd(inhall,0,2)/sqrt(sum(~isnan(inhall(1,:)))), 'r', 'alpha');
    % shfl
    h1 = boundedline(time_aligned, nanmean(excall_shfl,2), nanstd(excall_shfl,0,2)/sqrt(sum(~isnan(excall_shfl(1,:)))), 'cmap', colss(1,:), 'alpha');
    h2 = boundedline(time_aligned, nanmean(inhall_shfl,2), nanstd(inhall_shfl,0,2)/sqrt(sum(~isnan(inhall_shfl(1,:)))), 'cmap', colss(2,:), 'alpha');    
    a = get(gca, 'ylim');
    plot(time_aligned, hh*(a(2)-.05*diff(a)), 'k.')
    plot([0,0],a,'k:')
    b = get(gca, 'xlim');
    if ~isempty(yy)
        plot(b, [yy,yy],'k:')    
    end
    legend([h1,h2], {'Excitatory', 'Inhibitory'}, 'position', [0.5741    0.8281    0.1414    0.0901])
    if chAl==1
        xlabel('Time since choice onset (ms)')
    else
        xlabel('Time since stim onset (ms)')
    end
    ylabel(namc)
    title('mean+se all neurons of all days')
    
    % save figure
    if savefigs        
        savefig(fh, fullfile(dirn, [namc,'_','ROC_curr_chAl_excInh_timeCourse_aveDays_', nowStr,'.fig']))
        print(fh, '-dpdf', fullfile(dirn, [namc,'_','ROC_curr_chAl_excInh_timeCourse_aveDays_', nowStr]))
    end  


    %% Dist of neuron-averaged ROC, pooled across all days and frames, compare exc and inh

    ylab = 'Fraction days*frames';
    xlab = namc;        
    leg = {'exc','inh'};
%     cols = {'b','r'};

    fh = figure;
    % data
    y1 = aveexc(:);
    y2 = aveinh(:);
    fh = plotHist(y1,y2,xlab,ylab,leg, cols, yy,fh);
    % shfl
    y1 = aveexc_shfl(:);
    y2 = aveinh_shfl(:);        
    fh = plotHist(y1,y2,xlab,ylab,leg, colss, yy,fh);

    if savefigs
        savefig(fh, fullfile(dirn, [namc,'_','ROC_curr_chAl_excInh_dist_aveNeurs_frsDaysPooled_', nowStr,'.fig']))
        print(fh, '-dpdf', fullfile(dirn, [namc,'_','ROC_curr_chAl_excInh_dist_aveNeurs_frsDaysPooled_', nowStr]))
    end

    % abs
    %{
    aveexc = cellfun(@(x)mean(abs(x),2), choicePref_exc_aligned, 'uniformoutput',0); % average of neurons across each day
    aveexc = cell2mat(aveexc);

    aveinh = cellfun(@(x)mean(abs(x),2), choicePref_inh_aligned, 'uniformoutput',0); % average of neurons across each day
    aveinh = cell2mat(aveinh);
    %}


    %% Dist of AUC all neurons of all days and all time bins pooled.

    xlab = namc;
    ylab = 'Fraction neurons*days*frames';
    leg = {'exc','inh'};
%     cols = {'b','r'};

    fh = figure;
    % data
    y1 = excall(:);
    y2 = inhall(:);
    fh = plotHist(y1,y2,xlab,ylab,leg, cols, yy,fh);
    % shfl
    y1 = excall_shfl(:);
    y2 = inhall_shfl(:);
    fh = plotHist(y1,y2,xlab,ylab,leg, colss, yy,fh);


    if savefigs        
        savefig(fh, fullfile(dirn, [namc,'_','ROC_curr_chAl_excInh_dist_frsDaysNeursPooled_', nowStr,'.fig']))
        print(fh, '-dpdf', fullfile(dirn, [namc,'_','ROC_curr_chAl_excInh_dist_frsDaysNeursPooled_', nowStr]))
    end            


    %% Same as above but done for each frame separately: Compare exc/inh ROC dist for each frame, all neurons of all days pooled

    xlab = namc;
    ylab = 'Fraction neurons*days';
    leg = {'exc','inh'};
%     cols = {'b','r'};

    fign = figure('position', [680   197   696   779]);
    c = 4;        
    ha = tight_subplot(ceil(length(time_aligned)/c), c, [.1 .04],[.03 .03],[.1 .03]);
    for ifr = 1:length(time_aligned)        
        tit = round(time_aligned(ifr));
        % data
        y1 = excall(ifr,:);
        y2 = inhall(ifr,:);        
        plotHist_sp(y1,y2,xlab,ylab,leg, cols, tit, fign, ha(ifr), yy)
        % shfl
        y1 = excall_shfl(ifr,:);
        y2 = inhall_shfl(ifr,:);        
        plotHist_sp(y1,y2,xlab,ylab,leg, colss, tit, fign, ha(ifr), yy)
        
        if ifr==1
            xlabel(ha(ifr), xlab)
            ylabel(ha(ifr), ylab)
            legend(ha(ifr), leg)
        end
    end

    if savefigs        
        savefig(fign, fullfile(dirn, [namc,'_','ROC_curr_chAl_excInh_dist_eachFr_daysNeursPooled_', nowStr,'.fig']))
        print(fign, '-dpdf', fullfile(dirn, [namc,'_','ROC_curr_chAl_excInh_dist_eachFr_daysNeursPooled_', nowStr]))
    end  

    
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    %%%%%%%%%%% single day plots %%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    %% For each day plot the timecourse of roc for exc and inh

    if isempty(yy)
        fh=figure('name', 'numTrs (ipsi contra)'); 
    else
        fh=figure('name', 'ipsi>contra; numTrs (ipsi contra)'); 
    end
    set(gcf, 'position',[680    85   699   891])
    c = 7;
    minNrows = 3;
    if minNrows > ceil(size(aveexc,2)/c)
        r = minNrows;
        c = ceil(c/minNrows);
    else
        r = ceil(size(aveexc,2)/c);
    end
    ha = tight_subplot(r, c, [.04 .04],[.03 .03],[.1 .03]);

    for iday = 1:size(aveexc,2)        
        if mnTrNum(iday) >= thMinTrs             
%             subplot(ceil(size(aveexc,2)/10), 10, iday)                                    
            [h1] = boundedline(ha(iday), time_aligned, aveexc(:,iday)', seexc(:,iday)', 'b', 'alpha', 'nan', 'remove');
            [h2] = boundedline(ha(iday), time_aligned, aveinh(:,iday)', seinh(:,iday)', 'r', 'alpha', 'nan', 'remove');
            % shfl
            [h1] = boundedline(ha(iday), time_aligned, aveexc_shfl(:,iday)', seexc_shfl(:,iday)', 'cmap', colss(1,:), 'alpha', 'nan', 'remove');
            [h2] = boundedline(ha(iday), time_aligned, aveinh_shfl(:,iday)', seinh_shfl(:,iday)', 'cmap', colss(2,:), 'alpha', 'nan', 'remove');            

            hold(ha(iday),'on')
            yl = get(ha(iday),'ylim');
            plot(ha(iday), [0, 0], [yl(1), yl(2)], 'k:')
            if ~isempty(yy)
                plot(ha(iday), [time_aligned(1), time_aligned(end)], [yy,yy], 'k:')
            end
            ylim(ha(iday), yl)
            da = days{iday};
            s = sprintf('%s (%d %d)', da(1:6), corr_ipsi_contra(iday,1), corr_ipsi_contra(iday,2));
            title(ha(iday), s)            
        end
    end
%     subplot(ceil(size(aveexc,2)/10), 10, 1)
    xlabel(ha(1), 'Time')
    ylabel(ha(1), namc)
    legend(ha(1), [h1,h2], 'exc','inh')


    if savefigs
        savefig(fh, fullfile(dirn, [namc,'_','ROC_curr_chAl_excInh_timeCourse_aveNeurs_eachDay_', nowStr,'.fig']))
        print(fh, '-dpdf', fullfile(dirn, [namc,'_','ROC_curr_chAl_excInh_timeCourse_aveNeurs_eachDay_', nowStr]))
    end


    %% Superimpose time course of ROC for all days

    co = jet(size(aveexc,2));
    set(groot,'defaultAxesColorOrder',co)

    fh=figure('position', [424   253   534   687]);
    
    subplot(221)
    plot(time_aligned, aveexc);     hold on
    if ~isempty(yy)
        plot([time_aligned(1),time_aligned(end)], [yy,yy], 'k:')
    end
    yl = get(gca,'ylim');
    plot([0,0], yl, 'k:')
    ylim(yl)
    title('exc')
    legend(days, 'Position', [0.6434 -0.7195 0.3153 1.3991]);
    xlabel('time')
    ylabel(namc)

    subplot(223)
    plot(time_aligned, aveinh);     hold on
    if ~isempty(yy)
        plot([time_aligned(1),time_aligned(end)], [yy,yy], 'k:')
    end
    yl = get(gca,'ylim');
    plot([0,0], yl, 'k:')
    ylim(yl)    
    title('inh')    
%     figure
%     imagesc(aveexc)


    %%%%% shfl
    subplot(222)
    plot(time_aligned, aveexc_shfl);     hold on
    if ~isempty(yy)
        plot([time_aligned(1),time_aligned(end)], [yy,yy], 'k:')
    end
    yl = get(gca,'ylim');
    plot([0,0], yl, 'k:')
    ylim(yl)
    title('exc\_shfl')
%     legend(days, 'Position', [0.6434 -0.7195 0.3153 1.3991]);
    xlabel('time')
    ylabel(namc)

    subplot(224)
    plot(time_aligned, aveinh_shfl);     hold on
    if ~isempty(yy)
        plot([time_aligned(1),time_aligned(end)], [yy,yy], 'k:')
    end
    yl = get(gca,'ylim');
    plot([0,0], yl, 'k:')
    ylim(yl)    
    title('inh\_shfl')    
    
    
    if savefigs
        savefig(fh, fullfile(dirn, [namc,'_','ROC_curr_chAl_excInh_timeCourse_aveNeurs_allDaysSup_', nowStr,'.fig']))
        print(fh, '-dpdf', fullfile(dirn, [namc,'_','ROC_curr_chAl_excInh_timeCourse_aveNeurs_allDaysSup_', nowStr]))
    end

    % go back to default color order
    set(groot,'defaultAxesColorOrder',cod)


    %% Learning figure: Compare how exc/inh neuron-averaged ROC change across days, do it separately for each frame

    fh=figure('position', [680   131   714   845]); 
    ha = tight_subplot(ceil((size(aveexc,1)+1)/4), 4, [.04 .04],[.03 .03],[.1 .03]);
%     subplot(ceil((size(aveexc,1)+1)/4), 4, 1)
    h = plot(ha(1), corr_ipsi_contra);
    legend(h, 'ipsi', 'contra')
    xlabel(ha(1), 'days')
    ylabel(ha(1), 'Num trials')

    for fr = 1:size(aveexc,1)            %         subplot(ceil((size(aveexc,1)+1)/4), 4, fr+1)        
        % data
        [h1] = boundedline(ha(fr+1), 1:size(aveexc,2), aveexc(fr,:), seexc(fr,:), 'b', 'alpha', 'nan', 'remove');
        [h2] = boundedline(ha(fr+1), 1:size(aveinh,2), aveinh(fr,:), seinh(fr,:), 'r', 'alpha', 'nan', 'remove');
        % shfl
        [h1] = boundedline(ha(fr+1), 1:size(aveexc_shfl,2), aveexc_shfl(fr,:), seexc_shfl(fr,:), 'cmap', colss(1,:), 'alpha', 'nan', 'remove');
        [h2] = boundedline(ha(fr+1), 1:size(aveinh_shfl,2), aveinh_shfl(fr,:), seinh_shfl(fr,:), 'cmap', colss(2,:), 'alpha', 'nan', 'remove');            
        hold(ha(fr+1),'on')
        if ~isempty(yy)
            plot(ha(fr+1), [0,length(days)+1], [yy,yy], 'k:')
        end
        title(ha(fr+1), sprintf('%d ms',round(time_aligned(fr))))
    end
%     subplot(ceil((size(aveexc,1)+1)/4), 4, 2)
    xlabel(ha(2), 'days')
    ylabel(ha(2), namc)
    legend(ha(2), [h1,h2], 'exc','inh')
    title(ha(2), sprintf('time rel2 choice = %d ms',round(time_aligned(1))))

    if savefigs
        savefig(fh, fullfile(dirn, [namc,'_','ROC_curr_chAl_excInh_trainingDays_aveNeurs_eachFrame_', nowStr,'.fig']))
        print(fh, '-dpdf', fullfile(dirn, [namc,'_','ROC_curr_chAl_excInh_trainingDays_aveNeurs_eachFrame_', nowStr]))
    end
    
end



