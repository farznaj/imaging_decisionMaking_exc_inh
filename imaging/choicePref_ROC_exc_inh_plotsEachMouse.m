
dirn0 = '/home/farznaj/Dropbox/ChurchlandLab/Projects/inhExcDecisionMaking/ROC';

savefigs = eachMouse_do_savefigs(2);
cols = {'b','r'}; % exc,inh, real data
colss = [0,.8,.8; .8,.5,.8]; % exc,inh colors for shuffled data


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Plots of each mouse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Scatter plot of AUC_corr vs AUC_incorr for individual neurons; use jet colormap for days (early to late days go from blue to red)
% For this plot you need to download ROC vars of correct trials as well as ROC vars of incorrect trials.
% This allows us to see if neurons represent choice or stim... if for both
% corr and incorr trials, AUC is below 0.5 (contra-choice specific) or above
% 0.5 (ipsi-choice specific), then the neuron represnts choice; But if for corr AUC is below 0.5 and for incorr it is above 0.5, then the neuron represents stimulus. 

if exist('ipsi_contra_corr', 'var') && exist('ipsi_contra_incorr', 'var')
    thROC = .51; % .505; % .5 % threshold to compute number of choice vs stim representing neurons of each side

    nPreMin_corr = find(time_al_corr < 0, 1, 'last');
    nPreMin_incorr = find(time_al_incorr < 0, 1, 'last');
    cc_corr_incorr_exc = nan(1, length(mice));
    cc_corr_incorr_inh = nan(1, length(mice));

    choiceCoded_contraSpecific = nan(1, length(mice));
    choiceCoded_ipsiSpecific = nan(1, length(mice));
    stimCoded_HRSpecific = nan(1, length(mice));
    stimCoded_LRSpecific = nan(1, length(mice));
    choiceCoded = nan(1, length(mice));
    stimCoded = nan(1, length(mice));

    for im = 1:length(mice)

        mnTrNum_corr = min(ipsi_contra_corr{im},[],2); % min number of trials of the 2 class
        mnTrNum_incorr = min(ipsi_contra_incorr{im},[],2);
        ndays = length(choicePref_exc_al_allMice_corr{im});
        cmp = jet(ndays);   
        exc_corr_incorr_avNs = nan(2, ndays);
        exc_corr_incorr_sdNs = nan(2, ndays);
        inh_corr_incorr_avNs = nan(2, ndays);
        inh_corr_incorr_sdNs = nan(2, ndays);
        goodD = (mnTrNum_corr > thMinTrs) & (mnTrNum_incorr > thMinTrs);

        fh = figure('name', mice{im}, 'position', [106   453   562   520]); %15   456   320   520]); 

        for iday = 1:ndays
            if goodD(iday)

                %%% exc %%% 
                x = choicePref_exc_al_allMice_incorr{im}{iday}(nPreMin_incorr,:); % AUC of exc neurons at time bin -1 computed on incorrect trials
                y = choicePref_exc_al_allMice_corr{im}{iday}(nPreMin_corr,:);
                % mean and sd across neurons
                exc_corr_incorr_avNs(:,iday) = [mean(y); mean(x)]; 
                exc_corr_incorr_sdNs(:,iday) = [std(y); std(x)];
                % plot
                subplot(221); hold on;
                plot(x, y, '.', 'color', cmp(iday,:), 'markersize', 5)            

                %%% inh %%%        
                x = choicePref_inh_al_allMice_incorr{im}{iday}(nPreMin_incorr,:);
                y = choicePref_inh_al_allMice_corr{im}{iday}(nPreMin_corr,:);            
                % mean and sd across neurons
                inh_corr_incorr_avNs(:,iday) = [mean(y); mean(x)]; 
                inh_corr_incorr_sdNs(:,iday) = [std(y); std(x)];
                % plot
                subplot(223); hold on;
                plot(x, y, '.', 'color', cmp(iday,:), 'markersize', 5)        
            end
        end

        % pool all neurons of all days 
        exc_incorr_daysPooled_eachMouse = cell2mat(choicePref_exc_al_allMice_incorr{im}(goodD)); 
        exc_incorr_daysPooled_eachMouse = exc_incorr_daysPooled_eachMouse(nPreMin_incorr,:); % take AUC of all neurons of all days at time bin -1
        exc_corr_daysPooled_eachMouse = cell2mat(choicePref_exc_al_allMice_corr{im}(goodD)); 
        exc_corr_daysPooled_eachMouse = exc_corr_daysPooled_eachMouse(nPreMin_incorr,:);
        inh_incorr_daysPooled_eachMouse = cell2mat(choicePref_inh_al_allMice_incorr{im}(goodD)); 
        inh_incorr_daysPooled_eachMouse = inh_incorr_daysPooled_eachMouse(nPreMin_incorr,:);
        inh_corr_daysPooled_eachMouse = cell2mat(choicePref_inh_al_allMice_corr{im}(goodD)); 
        inh_corr_daysPooled_eachMouse = inh_corr_daysPooled_eachMouse(nPreMin_incorr,:);
        % compute corrcoeff between corr and incorr AUCs (all neurons)
        cc_corr_incorr_exc(im) = corr2(exc_incorr_daysPooled_eachMouse, exc_corr_daysPooled_eachMouse);
        cc_corr_incorr_inh(im) = corr2(inh_incorr_daysPooled_eachMouse, inh_corr_daysPooled_eachMouse);
        % compute number of choice vs stim representing neurons of each side
    %     thROC = .51; % .505; % .5
        choiceCoded_contraSpecific(im) = sum((exc_incorr_daysPooled_eachMouse < thROC) & (exc_corr_daysPooled_eachMouse < thROC));
        choiceCoded_ipsiSpecific(im) = sum((exc_incorr_daysPooled_eachMouse > thROC) & (exc_corr_daysPooled_eachMouse > thROC));
        stimCoded_HRSpecific(im) = sum((exc_incorr_daysPooled_eachMouse < thROC) & (exc_corr_daysPooled_eachMouse > thROC));
        stimCoded_LRSpecific(im) = sum((exc_incorr_daysPooled_eachMouse > thROC) & (exc_corr_daysPooled_eachMouse < thROC));
        choiceCoded(im) = choiceCoded_ipsiSpecific(im) + choiceCoded_contraSpecific(im);
        stimCoded(im) = stimCoded_HRSpecific(im) + stimCoded_LRSpecific(im);

        %%%%
        figure(fh)
        subplot(221)
        title(sprintf('Excitatory (R=%.2f)', cc_corr_incorr_exc(im)))
        hline(.5, 'k:'); vline(.5, 'k:')
        xlabel('AUC (incorr)'); ylabel('AUC (corr)')
        xlim([0,1]); ylim([0,1])

        subplot(223)
        title(sprintf('Inhibitory (R=%.2f)', cc_corr_incorr_inh(im)))
        hline(.5, 'k:'); vline(.5, 'k:')
        xlabel('AUC (incorr)'); ylabel('AUC (corr)')    
        xlim([0,1]); ylim([0,1])


        %%%%%%%%%%%%%%%% Plot AUC averaged across neurons for each day %%%%%%%%%%%%%%%%
        % set corrcoefs (across days)
        cc_exc = corr(exc_corr_incorr_avNs', 'rows', 'pairwise'); cc_exc = cc_exc(2);
        cc_inh = corr(inh_corr_incorr_avNs', 'rows', 'pairwise'); cc_inh = cc_inh(2);       
    %     figure('name', mice{im}, 'position', [15   456   320   520]); set(groot,'defaultAxesColorOrder',cmp)    

        %%% exc %%%        
        subplot(222); hold on;
        x = exc_corr_incorr_avNs(2,:);
        y = exc_corr_incorr_avNs(1,:);    
        scatter(x, y, 15, cmp,'filled')
        title(sprintf('Excitatory (R=%.2f)', cc_exc))
        hline(.5, 'k:'); vline(.5, 'k:')
        xlabel('AUC (incorr)'); ylabel('AUC (corr)')
    %     plot(x, y, '.', 'markersize', 5)    
    %     xe = exc_corr_incorr_sdNs(2,:);
    %     ye = exc_corr_incorr_sdNs(1,:); 
    %     errorbarxy(x, y, xe, ye, xe, ye);
    %     xlim([0,1]); ylim([0,1])    

        %%% inh %%%        
        subplot(224); hold on;
        x = inh_corr_incorr_avNs(2,:);
        y = inh_corr_incorr_avNs(1,:);    
        scatter(x, y, 15, cmp,'filled')
        title(sprintf('Inhibitory (R=%.2f)', cc_inh))
        hline(.5, 'k:'); vline(.5, 'k:')
        xlabel('AUC (incorr)'); ylabel('AUC (corr)')    
    %     plot(x, y, '.', 'markersize', 5)
    %     xe = inh_corr_incorr_sdNs(2,:);
    %     ye = inh_corr_incorr_sdNs(1,:);    
    %     errorbarxy(x, y, xe, ye, xe, ye);
    %     xlim([0,1]); ylim([0,1])


        %%%%%%%%%%%%%%%%%% save figure %%%%%%%%%%%%%%%%%% 
        if savefigs        
            mouse = mice{im};    
            dirnFig = fullfile(dirn0, mouse);
            lab = simpleTokenize(namc, '_'); lab = lab{1};
            nowStr = nowStr_allMice{im};
            savefig(fh, fullfile(dirnFig, [lab,'_corrVSincorr_','ROC_curr_chAl_excInh_time-1_allN_aveN_', nowStr,'.fig']))
            print(fh, '-dpdf', fullfile(dirnFig, [lab,'_corrVSincorr_','ROC_curr_chAl_excInh_time-1_allN_aveN_', nowStr]))
        end  

    end

    choiceCoded ./ stimCoded
    choiceCoded_contraSpecific ./ choiceCoded_ipsiSpecific
    stimCoded_HRSpecific ./ stimCoded_LRSpecific
end



%%
% im = 1;
for im = 1:length(mice)

    mouse = mice{im};    
    dirnFig = fullfile(dirn0, mouse);
    
    if strcmp(outcome2ana, '')
        dirnFig = fullfile(dirnFig, 'allOutcome');
    elseif strcmp(outcome2ana, 'corr')
        dirnFig = fullfile(dirnFig, 'corr');
    elseif strcmp(outcome2ana, 'incorr')
        dirnFig = fullfile(dirnFig, 'incorr');
    end
    
    if doshfl
        dirnFig = fullfile(dirnFig, 'shuffled_actual_ROC');
    end
    
    if ~exist(dirnFig, 'dir')
        mkdir(dirnFig)
    end
    
    [~,~,dirn] = setImagingAnalysisNames(mouse, 'analysis', []); 
    cd(dirn)
    
    % these are vars that are aligned across days for each mouse (they are
    % not the ones aligned across all mice).
    choicePref_exc_aligned = choicePref_exc_aligned_allMice{im}; % days; each day: frs x ns
    choicePref_inh_aligned = choicePref_inh_aligned_allMice{im};
    if doshfl
        choicePref_exc_aligned_shfl = choicePref_exc_aligned_allMice_shfl{im}; % days; each day: frs x ns
        choicePref_inh_aligned_shfl = choicePref_inh_aligned_allMice_shfl{im};
        choicePref_exc_aligned_shfl0 = choicePref_exc_aligned_allMice_shfl0{im}; % days; each day: frs x ns x samps
        choicePref_inh_aligned_shfl0 = choicePref_inh_aligned_allMice_shfl0{im};
    end
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

    if doshfl % shfl
        aveexc_shfl = cellfun(@(x)mean(x,2), choicePref_exc_aligned_shfl, 'uniformoutput',0); % average across neurons (already averaged across samps)
        aveexc_shfl = cell2mat(aveexc_shfl); % frs x days
        seexc_shfl = cellfun(@(x)std(x,[],2)/sqrt(size(x,2)), choicePref_exc_aligned_shfl, 'uniformoutput',0); % standard error across neurons 
        seexc_shfl = cell2mat(seexc_shfl); % frs x days

        aveinh_shfl = cellfun(@(x)mean(x,2), choicePref_inh_aligned_shfl, 'uniformoutput',0);
        aveinh_shfl = cell2mat(aveinh_shfl);
        seinh_shfl = cellfun(@(x)std(x,[],2)/sqrt(size(x,2)), choicePref_inh_aligned_shfl, 'uniformoutput',0);
        seinh_shfl = cell2mat(seinh_shfl); % frs x days x samps        
        
        % individual shfl samples
        aveexc_shfl0 = cellfun(@(x)squeeze(mean(x,2)), choicePref_exc_aligned_shfl0, 'uniformoutput',0); % average across neurons
        aveexc_shfl0 = cell2mat(aveexc_shfl0); % frs x (days x samples) 
        aveinh_shfl0 = cellfun(@(x)squeeze(mean(x,2)), choicePref_inh_aligned_shfl0, 'uniformoutput',0); % average across neurons
        aveinh_shfl0 = cell2mat(aveinh_shfl0); % frs x (days x samples) 
        
    end

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
    if doshfl % shfl
        excall_shfl = cell2mat(choicePref_exc_aligned_shfl); % nFrs x n_exc_all
        inhall_shfl = cell2mat(choicePref_inh_aligned_shfl); % nFrs x n_inh_all
        size(excall_shfl), size(inhall_shfl)
        excall_shfl0 = cell2mat(choicePref_exc_aligned_shfl0); % nFrs x n_exc_all x nSamps
        inhall_shfl0 = cell2mat(choicePref_inh_aligned_shfl0); % nFrs x n_inh_all x nSamps
        size(excall_shfl0), size(inhall_shfl0)        
    end
    % ttest: is exc (single neuron ROC pooled across days) ROC
    % different from inh ROC? Do it for each time bin seperately.
    h = ttest2(excall', inhall'); % h: 1 x nFrs
    hh = h;
    hh(h==0) = nan;
%     h = ttest2(excall_shfl', inhall_shfl')


    %% PLOTS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Plot AUC timecourse averaged across days

    fh = figure('position',[10   556   792   383]);

    %%%%%%%%% Average and se across days; each day is already averaged across neurons; done seperately for each time bin
    subplot(121); hold on
    h1 = boundedline(time_aligned, nanmean(aveexc,2), nanstd(aveexc,0,2)/sqrt(numDaysGood(im)), 'b', 'alpha'); % sum(mnTrNum>=thMinTrs)
    h2 = boundedline(time_aligned, nanmean(aveinh,2), nanstd(aveinh,0,2)/sqrt(numDaysGood(im)), 'r', 'alpha');
    if doshfl % shfl
        h1 = boundedline(time_aligned, nanmean(aveexc_shfl,2), nanstd(aveexc_shfl,0,2)/sqrt(numDaysGood(im)), 'cmap', colss(1,:), 'alpha'); % sum(mnTrNum>=thMinTrs)
        h2 = boundedline(time_aligned, nanmean(aveinh_shfl,2), nanstd(aveinh_shfl,0,2)/sqrt(numDaysGood(im)), 'cmap', colss(2,:), 'alpha');    
    end
    a = get(gca, 'ylim');    
    plot(time_aligned, hh0*(a(2)-.05*diff(a)), 'k.')    
    plot([0,0],a,'k:')
    b = get(gca, 'xlim');
    if ~isempty(yy)
        plot(b, [yy,yy],'k:')
    end
    legend([h1,h2], {'Excitatory', 'Inhibitory'}, 'position', [0.1347    0.8177    0.1414    0.0901]);
    xlabel('Time since choice onset (ms)')
    ylab = simpleTokenize(namc, '_'); ylab = ylab{1}; %namc;        
    ylabel(ylab)
    title('mean+se days (Ns aved per day)')


    %%%%%%%%%% Average and se across all neurons of all days; done seperately for each time bin
    subplot(122); hold on
    h1 = boundedline(time_aligned, nanmean(excall,2), nanstd(excall,0,2)/sqrt(sum(~isnan(excall(1,:)))), 'b', 'alpha');
    h2 = boundedline(time_aligned, nanmean(inhall,2), nanstd(inhall,0,2)/sqrt(sum(~isnan(inhall(1,:)))), 'r', 'alpha');
    if doshfl % shfl
        h1 = boundedline(time_aligned, nanmean(excall_shfl,2), nanstd(excall_shfl,0,2)/sqrt(sum(~isnan(excall_shfl(1,:)))), 'cmap', colss(1,:), 'alpha');
        h2 = boundedline(time_aligned, nanmean(inhall_shfl,2), nanstd(inhall_shfl,0,2)/sqrt(sum(~isnan(inhall_shfl(1,:)))), 'cmap', colss(2,:), 'alpha');    
    end
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
    ylabel(ylab)
    title('mean+se all neurons of all days')
    
    % save figure
    if savefigs        
        savefig(fh, fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInh_timeCourse_aveDays_', nowStr,'.fig']))
        print(fh, '-dpdf', fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInh_timeCourse_aveDays_', nowStr]))
    end  


    %% Dist of neuron-averaged ROC, pooled across all frames and days, compare exc and inh

    ylab = 'Fraction days*frames';
    xlab = simpleTokenize(namc, '_'); xlab = xlab{1}; %namc;        
    leg = {'exc','inh'};
    nBins = 50;
    doSmooth = 1;

    fh = figure;
    % data
    y1 = aveexc(:);
    y2 = aveinh(:);
    [fh, bins] = plotHist(y1,y2,xlab,ylab,leg, cols, yy,fh, nBins, doSmooth);
%     fh = plotHist(y1,y2,xlab,ylab,leg, cols, yy, fh, nBins, doSmooth, lineStyles, sp, bins); 
    if doshfl % shfl
        % sample-averaged shfls
%         y1 = aveexc_shfl(:);
%         y2 = aveinh_shfl(:);        
%         fh = plotHist(y1,y2,xlab,ylab,leg, colss, yy,fh, nBins, [],[],[], bins);
        
        % individual shfl samples
        y1 = aveexc_shfl0(:);
        y2 = aveinh_shfl0(:);        
        fh = plotHist(y1,y2,xlab,ylab,leg, colss, yy,fh, nBins, [],[],[], bins);
    end

    if savefigs
        savefig(fh, fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInh_dist_aveNeurs_frsDaysPooled_', nowStr,'.fig']))
        print(fh, '-dpdf', fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInh_dist_aveNeurs_frsDaysPooled_', nowStr]))
    end

    % abs
    %{
    aveexc = cellfun(@(x)mean(abs(x),2), choicePref_exc_aligned, 'uniformoutput',0); % average of neurons across each day
    aveexc = cell2mat(aveexc);

    aveinh = cellfun(@(x)mean(abs(x),2), choicePref_inh_aligned, 'uniformoutput',0); % average of neurons across each day
    aveinh = cell2mat(aveinh);
    %}


    %% Dist of AUC, all neurons of all days and all time bins pooled.

    xlab = simpleTokenize(namc, '_'); xlab = xlab{1}; %namc;        
    ylab = 'Fraction neurons*days*frames';
    leg = {'exc','inh'};
%     cols = {'b','r'};

    fh = figure;
    % data
    y1 = excall(:);
    y2 = inhall(:);
    [fh, bins] = plotHist(y1,y2,xlab,ylab,leg, cols, yy,fh, nBins);
    if doshfl % shfl
        % dist of mean of shfl
%         y1 = excall_shfl(:);
%         y2 = inhall_shfl(:);
%         fh = plotHist(y1,y2,xlab,ylab,leg, colss, yy,fh, nBins, [], [], [], bins);
        % dist of individual shfl samples
        y1 = excall_shfl0(:);
        y2 = inhall_shfl0(:);
        fh = plotHist(y1,y2,xlab,ylab,leg, colss, yy,fh, nBins, [], [], [], bins);        
    end

    if savefigs        
        savefig(fh, fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInh_dist_frsDaysNeursPooled_', nowStr,'.fig']))
        print(fh, '-dpdf', fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInh_dist_frsDaysNeursPooled_', nowStr]))
    end            


    %% Same as above but done for each frame separately: Compare exc/inh ROC dist for each frame, all neurons of all days pooled

    numBins = 50;
    documsum = 0;
    xlab = simpleTokenize(namc, '_'); xlab = xlab{1}; %namc;        
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
        bins = plotHist_sp(y1,y2,xlab,ylab,leg, cols, tit, fign, ha(ifr), yy, documsum, numBins);
       
        if doshfl % shfl
            % dist of mean of shfl
%             y1 = excall_shfl(ifr,:);
%             y2 = inhall_shfl(ifr,:);        
%             plotHist_sp(y1,y2,xlab,ylab,leg, colss, tit, fign, ha(ifr), yy, documsum, numBins, bins);
            % dist of all shfl samples
            y1 = excall_shfl0(ifr,:,:); y1 = y1(:);
            y2 = inhall_shfl0(ifr,:,:); y2 = y2(:);
            plotHist_sp(y1,y2,xlab,ylab,leg, colss, tit, fign, ha(ifr), yy, documsum, numBins, bins);            
        end
        
        if ifr==1
            xlabel(ha(ifr), xlab)
            ylabel(ha(ifr), ylab)
            legend(ha(ifr), leg)
        end
    end

    if savefigs        
        savefig(fign, fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInh_dist_eachFr_daysNeursPooled_', nowStr,'.fig']))
        print(fign, '-dpdf', fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInh_dist_eachFr_daysNeursPooled_', nowStr]))
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
            if doshfl  % shfl
                [h1] = boundedline(ha(iday), time_aligned, aveexc_shfl(:,iday)', seexc_shfl(:,iday)', 'cmap', colss(1,:), 'alpha', 'nan', 'remove');
                [h2] = boundedline(ha(iday), time_aligned, aveinh_shfl(:,iday)', seinh_shfl(:,iday)', 'cmap', colss(2,:), 'alpha', 'nan', 'remove');            
            end
            
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
    ylab = simpleTokenize(namc, '_'); ylab = ylab{1}; %namc;    
    ylabel(ha(1), ylab)
    legend(ha(1), [h1,h2], 'exc','inh')


    if savefigs
        savefig(fh, fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInh_timeCourse_aveSeNeurs_eachDay_', nowStr,'.fig']))
        print(fh, '-dpdf', fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInh_timeCourse_aveSeNeurs_eachDay_', nowStr]))
    end


    %% Superimpose time course of ROC for all days

    co = jet(size(aveexc,2));
    set(groot,'defaultAxesColorOrder',co)

    fh = figure('position', [424   253   534   687]);
    
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
    ylabel(ylab)

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


    if doshfl %%%%% shfl
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
        ylabel(ylab)

        subplot(224)
        plot(time_aligned, aveinh_shfl);     hold on
        if ~isempty(yy)
            plot([time_aligned(1),time_aligned(end)], [yy,yy], 'k:')
        end
        yl = get(gca,'ylim');
        plot([0,0], yl, 'k:')
        ylim(yl)    
        title('inh\_shfl')    
    end
    
    
    if savefigs
        savefig(fh, fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInh_timeCourse_aveNeurs_allDaysSup_', nowStr,'.fig']))
        print(fh, '-dpdf', fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInh_timeCourse_aveNeurs_allDaysSup_', nowStr]))
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
        if doshfl % shfl
            [h1] = boundedline(ha(fr+1), 1:size(aveexc_shfl,2), aveexc_shfl(fr,:), seexc_shfl(fr,:), 'cmap', colss(1,:), 'alpha', 'nan', 'remove');
            [h2] = boundedline(ha(fr+1), 1:size(aveinh_shfl,2), aveinh_shfl(fr,:), seinh_shfl(fr,:), 'cmap', colss(2,:), 'alpha', 'nan', 'remove');            
        end
        hold(ha(fr+1),'on')
        if ~isempty(yy)
            plot(ha(fr+1), [0,length(days)+1], [yy,yy], 'k:')
        end
        title(ha(fr+1), sprintf('%d ms',round(time_aligned(fr))))
    end
%     subplot(ceil((size(aveexc,1)+1)/4), 4, 2)
    xlabel(ha(2), 'days')
    ylab = simpleTokenize(namc, '_'); ylab = ylab{1}; %namc;   
    ylabel(ha(2), ylab)
    legend(ha(2), [h1,h2], 'exc','inh')
    title(ha(2), sprintf('time rel2 choice = %d ms',round(time_aligned(1))))
    ylabel(ha(5), ylab)

    if savefigs
        savefig(fh, fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInh_trainingDays_aveSeNeurs_eachFrame_', nowStr,'.fig']))
        print(fh, '-dpdf', fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInh_trainingDays_aveSeNeurs_eachFrame_', nowStr]))
    end
    
end



