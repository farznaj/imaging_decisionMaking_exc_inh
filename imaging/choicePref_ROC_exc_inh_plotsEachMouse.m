%% 1st run the script choicePref_ROC_exc_inh_plots_setVars.m to set the required vars.

%%
dirn0 = '/home/farznaj/Dropbox/ChurchlandLab/Projects/inhExcDecisionMaking/ROC';

savefigs = eachMouse_do_savefigs(2);
cols = {'b','r'}; % exc,inh, real data
colss = [0,.8,.8; .8,.5,.8]; % exc,inh colors for shuffled data
% nowStr = nowStr_allMice{imfni18}; % use nowStr of mouse fni18 (in case its day 4 was removed, you want that to be indicated in the figure name of summary of all mice).
% dirn00 = fullfile(dirn0, 'allMice');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Plots of each mouse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Scatter plots and errorbars to compare AUC_corr vs AUC_incorr for individual neurons

% For this plot you need to download ROC vars of correct trials as well as ROC vars of incorrect trials.
% This allows us to see if neurons represent choice or stim... if for both
% corr and incorr trials, AUC is below 0.5 (contra-choice specific) or above
% 0.5 (ipsi-choice specific), then the neuron represnts choice; But if for corr AUC is below 0.5 and for incorr it is above 0.5, then the neuron represents stimulus. 

if exist('ipsi_contra_corr', 'var') && exist('ipsi_contra_incorr', 'var')    
    choicePref_ROC_exc_inh_plotsEachMouse_corrIncorr
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
    
   
    time_aligned = time_aligned_allMice{im};
    nowStr = nowStr_allMice{im};
    mnTrNum = mnTrNum_allMice{im};
    days = days_allMice{im};
    corr_ipsi_contra = corr_ipsi_contra_allMice{im};
    set(groot,'defaultAxesColorOrder',cod)    
    
    
    %% Set vars for each mouse
   
    aveexc = aveexc_allMice0{im};
    aveinh = aveinh_allMice0{im};
    aveallN = aveallN_allMice0{im};
    aveuns = aveuns_allMice0{im};
    
    aveexc_shfl = aveexc_shfl_allMice{im};
    aveinh_shfl = aveinh_shfl_allMice{im};
    aveallN_shfl = aveallN_shfl_allMice{im};
    aveuns_shfl = aveuns_shfl_allMice{im};
    
    aveexc_shfl0 = aveexc_shfl0_allMice{im};
    aveinh_shfl0 = aveinh_shfl0_allMice{im};
    aveallN_shfl0 = aveallN_shfl0_allMice{im};
    aveuns_shfl0 = aveuns_shfl0_allMice{im};
    
    seexc = seexc_allMice{im};
    seinh = seinh_allMice{im};
    seallN = seallN_allMice{im};
    seuns = seuns_allMice{im};
    
    seexc_shfl = seexc_shfl_allMice{im};
    seinh_shfl = seinh_shfl_allMice{im};
    seallN_shfl = seallN_shfl_allMice{im};
    seuns_shfl = seuns_shfl_allMice{im};    
    
    excall = excall_allMice{im};
    inhall = inhall_allMice{im};
    allNall = allNall_allMice{im};
    unsall = unsall_allMice{im};

    choicePref_exc_aligned = choicePref_exc_aligned_allMice{im}; % days; each day: frs x ns
    choicePref_inh_aligned = choicePref_inh_aligned_allMice{im};
    choicePref_uns_aligned = choicePref_uns_aligned_allMice{im};
    if doshfl
        choicePref_exc_aligned_shfl0 = choicePref_exc_aligned_allMice_shfl0{im}; % days; each day: frs x ns x samps
        choicePref_inh_aligned_shfl0 = choicePref_inh_aligned_allMice_shfl0{im};
        choicePref_uns_aligned_shfl0 = choicePref_uns_aligned_allMice_shfl0{im};
    end


    % run ttest across days for each frame
    % ttest: is exc (neuron-averaged ROC pooled across days) ROC
    % different from inh ROC? Do it for each time bin seperately.
    [h,p] = ttest2(aveexc',aveinh'); % 1 x nFrs
    hh0 = h;
    hh0(h==0) = nan;


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
    %%% Average AUC across neurons for each day and each frame
    %%% Pool AUC across all neurons of all days (for each frame)

    %% Plot AUC timecourse averaged across days

    fh = figure('position',[10   556   792   383]);

    %%%%%%%%% Average and se across days; each day is already averaged across neurons; done seperately for each time bin
    subplot(121); hold on
    h1 = boundedline(time_aligned, nanmean(aveexc,2), nanstd(aveexc,0,2)/sqrt(numDaysGood(im)), 'b', 'alpha'); % sum(mnTrNum>=thMinTrs)
    h2 = boundedline(time_aligned, nanmean(aveinh,2), nanstd(aveinh,0,2)/sqrt(numDaysGood(im)), 'r', 'alpha');
    h3 = boundedline(time_aligned, nanmean(aveuns,2), nanstd(aveuns,0,2)/sqrt(numDaysGood(im)), 'g', 'alpha');
    if 0%doshfl % shfl
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
    legend([h1,h2,h3], {'Excitatory', 'Inhibitory', 'Unsure'}, 'position', [0.1347    0.8177    0.1414    0.0901]);
    xlabel('Time since choice onset (ms)')
    ylab = simpleTokenize(namc, '_'); ylab = ylab{1}; %namc;        
    ylabel(ylab)
    title('mean+se days (Ns aved per day)')


    %%%%%%%%%% Average and se across all neurons of all days; done seperately for each time bin
    subplot(122); hold on
    h1 = boundedline(time_aligned, nanmean(excall,2), nanstd(excall,0,2)/sqrt(sum(~isnan(excall(1,:)))), 'b', 'alpha');
    h2 = boundedline(time_aligned, nanmean(inhall,2), nanstd(inhall,0,2)/sqrt(sum(~isnan(inhall(1,:)))), 'r', 'alpha');
    h3 = boundedline(time_aligned, nanmean(unsall,2), nanstd(unsall,0,2)/sqrt(sum(~isnan(unsall(1,:)))), 'g', 'alpha');
    if 0%doshfl % shfl
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
    legend([h1,h2,h3], {'Excitatory', 'Inhibitory', 'Unsure'}, 'position', [0.5741    0.8281    0.1414    0.0901])
    if chAl==1
        xlabel('Time since choice onset (ms)')
    else
        xlabel('Time since stim onset (ms)')
    end
    ylabel(ylab)
    title('mean+se all neurons of all days')
    
    % save figure
    if savefigs        
        savefig(fh, fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInhUns_timeCourse_aveSeDays_', nowStr,'.fig']))
        print(fh, '-dpdf', fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInhUns_timeCourse_aveSeDays_', nowStr]))
    end  


    
    %% Plot fraction choice-selective neurons averaged across days
    
    % run ttest across days for each frame
    % ttest: is exc different from inh, in terms of fraction choice selective neurons? Do it for each time bin seperately.
    [h,p] = ttest2(exc_fractSigTuned_eachDay_allFrs{im}', inh_fractSigTuned_eachDay_allFrs{im}'); % 1 x nFrs
    hh0f = h;
    hh0f(h==0) = nan;
    
    fh = figure('position',[9   639   372   299]); %[10   556   792   383]);

    %%%%%%%%% Average and se across days; each day is already averaged across neurons; done seperately for each time bin
    hold on
    h1 = boundedline(time_aligned, nanmean(exc_fractSigTuned_eachDay_allFrs{im},2), nanstd(exc_fractSigTuned_eachDay_allFrs{im},0,2)/sqrt(numDaysGood(im)), 'b', 'alpha'); % sum(mnTrNum>=thMinTrs)
    h2 = boundedline(time_aligned, nanmean(inh_fractSigTuned_eachDay_allFrs{im},2), nanstd(inh_fractSigTuned_eachDay_allFrs{im},0,2)/sqrt(numDaysGood(im)), 'r', 'alpha');
    h3 = boundedline(time_aligned, nanmean(uns_fractSigTuned_eachDay_allFrs{im},2), nanstd(uns_fractSigTuned_eachDay_allFrs{im},0,2)/sqrt(numDaysGood(im)), 'g', 'alpha');
    if 0 %doshfl % shfl
        h1 = boundedline(time_aligned, nanmean(aveexc_shfl,2), nanstd(aveexc_shfl,0,2)/sqrt(numDaysGood(im)), 'cmap', colss(1,:), 'alpha'); % sum(mnTrNum>=thMinTrs)
        h2 = boundedline(time_aligned, nanmean(aveinh_shfl,2), nanstd(aveinh_shfl,0,2)/sqrt(numDaysGood(im)), 'cmap', colss(2,:), 'alpha');    
    end
    
    a = get(gca, 'ylim');    
    plot(time_aligned, hh0f*(a(2)-.05*diff(a)), 'k.')    
    plot([0,0],a,'k:')    
%     b = get(gca, 'xlim');
%     if ~isempty(yy)
%         plot(b, [yy,yy],'k:')
%     end    
    legend([h1,h2,h3], {'Excitatory', 'Inhibitory', 'Unsure'}, 'position', [0.1347    0.8177    0.1414    0.0901]);
    xlabel('Time since choice onset (ms)')
    ylab = 'Fraction choice-Selective Neurons'; %simpleTokenize(namc, '_'); ylab = ylab{1}; %namc;        
    ylabel(ylab)
    title('mean+se days (Ns aved per day)')
    
    % save figure
    if savefigs        
        savefig(fh, fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInhUns_fractSigTunedOfAllN_aveSeDays_', nowStr,'.fig']))
        print(fh, '-dpdf', fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInhUns_fractSigTunedOfAllN_aveSeDays_', nowStr]))
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
    y3 = aveuns(:);
    [fh, ~,~,~,~,h1,h3,hsp] = plotHist(y1,y3,xlab,ylab,nan, {'b', 'g'}, yy,fh, nBins, doSmooth); % unsure will be plotted green       
    [fh, bins,~,~,~,~,h2] = plotHist(y1,y2,xlab,ylab,nan, cols, yy,fh, nBins, doSmooth);
%     fh = plotHist(y1,y2,xlab,ylab,leg, cols, yy, fh, nBins, doSmooth, lineStyles, sp, bins); 
    if doshfl % shfl
        % sample-averaged shfls
%         y1 = aveexc_shfl(:);
%         y2 = aveinh_shfl(:);        
%         fh = plotHist(y1,y2,xlab,ylab,leg, colss, yy,fh, nBins, [],[],[], bins);
        
        % individual shfl samples
        y1 = aveexc_shfl0(:);
        y2 = aveinh_shfl0(:);
        y3 = aveuns_shfl0(:);
        [fh,~,~,~,~,h11,h31] = plotHist(y1,y3,xlab,ylab,nan, [0    0.8000    0.8000; .2, .2, .2], yy,fh, nBins, 0,{'-.', '-.'},[], bins);
        [fh,~,~,~,~,~,h21] = plotHist(y1,y2,xlab,ylab,nan, colss, yy,fh, nBins, 0,{'-.', '-.'},[], bins);
    end
    
    legend(hsp, [h1,h2,h3,h11,h21,h31], {'exc','inh', 'uns', 'exc','inh', 'uns'})
    
    if savefigs
        savefig(fh, fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInhUns_dist_aveNeurs_frsDaysPooled_', nowStr,'.fig']))
        print(fh, '-dpdf', fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInhUns_dist_aveNeurs_frsDaysPooled_', nowStr]))
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
    y3 = unsall(:);
    [fh, ~,~,~,~,h1,h3,hsp] = plotHist(y1,y3,xlab,ylab,nan, {'b','g'}, yy,fh, nBins);
    [fh, bins,~,~,~,~,h2] = plotHist(y1,y2,xlab,ylab,nan, cols, yy,fh, nBins);

    if doshfl % shfl
        % dist of mean of shfl
%         y1 = excall_shfl(:);
%         y2 = inhall_shfl(:);
%         fh = plotHist(y1,y2,xlab,ylab,leg, colss, yy,fh, nBins, [], [], [], bins);
        % dist of individual shfl samples
        y1 = excall_shfl0(:);
        y2 = inhall_shfl0(:);
        y3 = unsall_shfl0(:);
        [fh,~,~,~,~,h11,h31] = plotHist(y1,y3,xlab,ylab,nan, [0    0.8000    0.8000; .2, .2, .2], yy,fh, nBins, 0,{'-.', '-.'}, [], bins);        
        [fh,~,~,~,~,~,h21] = plotHist(y1,y2,xlab,ylab,nan, colss, yy,fh, nBins, 0,{'-.', '-.'}, [], bins);        
    end

    legend(hsp, [h1,h2,h3,h11,h21,h31], {'exc','inh', 'uns', 'exc','inh', 'uns'})

    if savefigs        
        savefig(fh, fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInhUns_dist_frsDaysNeursPooled_', nowStr,'.fig']))
        print(fh, '-dpdf', fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInhUns_dist_frsDaysNeursPooled_', nowStr]))
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
    
    documsum = 0;
    xlab = simpleTokenize(namc, '_'); xlab = xlab{1}; %namc;        
    ylab = 'Fraction neurons*days';
    leg = {'exc','inh'};
    
    %% For each day, plot hist of AUC for inh and exc at time -1.
   
    numBins = 10;
    doSmooth = 1;
    
    fh = figure;
    set(gcf, 'position',[680  85  1364  891])
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
            y1 = choicePref_exc_aligned{iday}(nPreMin_allMice(im),:); % AUC at timebin -1 for all neurons in day iday    
            y2 = choicePref_inh_aligned{iday}(nPreMin_allMice(im),:);
            tit = days{iday}(1:6);
            
            plotHist_sp(y1,y2,xlab,ylab,leg, cols, tit, fh, ha(iday), yy, documsum, numBins, bins, doSmooth);            
%             fh = plotHist(y1,y2,xlab,ylab,leg, cols, yy,fh, nBins, doSmooth, [], [], bins);   
        end
    end
    

    
    %% Same as above, plot hist of AUC for inh and exc, but also shade significant values of AUC, do for an example session      
    % for the paper ... example session
    % fni16, day 151029 
    
    if im==1        
        
        iday = 45; % last day for fni16
    
        %%%%%%%%%%%% Set the significant areas %%%%%%%%%%%%
        
        % Method1: find the min AUC value across neurons that is sig
        a = sort(choicePref_exc_onlySig_allMice{im}{iday});
        minSigIpsiAUC = a(find(a > .5, 1, 'first'));
        maxSigContraAUC = a(find(a < .5, 1, 'last'));
        mse = [maxSigContraAUC, minSigIpsiAUC];
        
        a = sort(choicePref_inh_onlySig_allMice{im}{iday});
        minSigIpsiAUC = a(find(a > .5, 1, 'first'));
        maxSigContraAUC = a(find(a < .5, 1, 'last'));
        msi = [maxSigContraAUC, minSigIpsiAUC];        
        
        a = sort(choicePref_uns_onlySig_allMice{im}{iday});
        minSigIpsiAUC = a(find(a > .5, 1, 'first'));
        maxSigContraAUC = a(find(a < .5, 1, 'last'));
        msu = [maxSigContraAUC, minSigIpsiAUC];

        % Method2: use the pooled shuffled dist of all neurons to identify the threshold values of significancy
        %{
        sh = squeeze(choicePref_exc_aligned_shfl0{iday}(nPreMin_allMice(im),:,:)); % Ns x samps
        sh = sh(:);
        m = nanmean(sh);
        s = nanstd(sh); 
        mse = [m-2*s , m+2*s];

        sh = squeeze(choicePref_inh_aligned_shfl0{iday}(nPreMin_allMice(im),:,:)); % Ns x samps
        sh = sh(:);
        m = nanmean(sh);
        s = nanstd(sh); 
        msi = [m-2*s m+2*s];
        %}

        
        %%%%%%%%%%%%%%%% for example exc and inh neurons, plot dist of shuffled AUC and marke real AUC; do it for example choice-selective and choice-nonselective neurons %%%%%%%%%%%%%%%%
        %          
        nBins = 25;
        doSmooth = 1;
        coln = {'k','r'};        
        fh = figure('position', [680   491   311   485]); 
        
        for isel = [2,1]
            if isel==1   % identify sig choice-selective neurons
                sigExc = exc_prob_dataROC_from_shflDist{im}{iday} <= alpha;
                sigInh = inh_prob_dataROC_from_shflDist{im}{iday} <= alpha;
                tit = [days{iday}(1:6), ' choiceSel'];
            else         % identify non-sig choice-selective neurons
                sigExc = exc_prob_dataROC_from_shflDist{im}{iday} > alpha;
                sigInh = inh_prob_dataROC_from_shflDist{im}{iday} > alpha;
                tit = [days{iday}(1:6), ' nonChoiceSel'];
            end

            % shuffled AUC distributions for an example choice-selective exc and an example choice-selective inh neuron
            % exc
            rn = randperm(sum(sigExc));
            inE = find(sigExc==1); inE = inE(rn(1));
            y1 = squeeze(choicePref_exc_aligned_shfl0{iday}(nPreMin_allMice(im),inE,:));
            % inh
            rn = randperm(sum(sigInh));
            inI = find(sigInh==1); inI = inI(rn(1));
            y2 = squeeze(choicePref_inh_aligned_shfl0{iday}(nPreMin_allMice(im),inI,:));
            
            disp([inE, inI])
            
            % real AUC value
            y1r = choicePref_exc_aligned{iday}(nPreMin_allMice(im),inE);
            y2r = choicePref_inh_aligned{iday}(nPreMin_allMice(im),inI);

            % dist of shuffled AUC and marking real AUC
            sp = subplot(2,1,isel); hold on
            [bins,ye,yi,x,he,hi] = plotHist_sp(y1,y2,xlab,ylab,leg, coln, tit, fh, sp, yy, documsum, nBins, [], doSmooth);          
            plot(y1r, .005, 'k*'); plot(y2r, .005, 'r*')
            % vline(y1r, 'k'); vline(y2r, 'r')
        end
        

        if savefigs
            savefig(fh, fullfile(dirnFig, [namc,'_exampleDayNeuron_',tit(1:6),'_dist_shflROC_timeM1_curr_chAl_excInh_', nowStr,'.fig']))
            print(fh, '-dpdf', fullfile(dirnFig, [namc,'_exampleDayNeuron_',tit(1:6),'_dist_shflROC_timeM1_curr_chAl_excInh_', nowStr]))
        end
        
        %}
        
        
        %%%%%%%%%%%%%%%% plot dist of firing rates for ipsi vs contra choices for the example choice selective neurons %%%%%%%%%%%%%%%%
        %{
        % run corr_excInh_setVars to get FRs
        frE_ipsiTrs = squeeze(fr_exc_aligned_allMice{im}{iday}(nPreMin_allMice(im), inE, ipsiTrs_allDays_allMice{im}{iday}));
        frE_contraTrs = squeeze(fr_exc_aligned_allMice{im}{iday}(nPreMin_allMice(im), inE, contraTrs_allDays_allMice{im}{iday}));
        
        frI_ipsiTrs = squeeze(fr_inh_aligned_allMice{im}{iday}(nPreMin_allMice(im), inI,  ipsiTrs_allDays_allMice{im}{iday}));
        frI_contraTrs = squeeze(fr_inh_aligned_allMice{im}{iday}(nPreMin_allMice(im), inI,  contraTrs_allDays_allMice{im}{iday}));
        
        nBins = 100;
        doSmooth = 1;        
        fh = figure('position', [87   660   390   266]); hold on
        y1 = frE_ipsiTrs;
        y2 = frE_contraTrs;
        [bins,ye,yi,x,he,hi] = plotHist_sp(y1,y2,xlab,ylab,leg, coln, tit, fh, [], yy, documsum, nBins, [], doSmooth);          
        
        fh = figure('position', [87   660   390   266]); hold on
        y1 = frI_ipsiTrs;
        y2 = frI_contraTrs;
        [bins,ye,yi,x,he,hi] = plotHist_sp(y1,y2,xlab,ylab,leg, coln, tit, fh, [], yy, documsum, nBins, [], doSmooth);          
        %}
        
        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%% Plot hist of real AUC for all neurons in the session and mark the sig values %%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        nBins = 25;
        doSmooth = 0; %1;
        coln = {'k','r'};

        y1 = choicePref_exc_aligned{iday}(nPreMin_allMice(im),:); % AUC at timebin -1 for all neurons in day iday    
        y2 = choicePref_inh_aligned{iday}(nPreMin_allMice(im),:);
        y3 = choicePref_uns_aligned{iday}(nPreMin_allMice(im),:);
        tit = days{iday}(1:6);        
        
        fh = figure('position', [87   660   390   266]); hold on
%         [bins,ye,yi,x,he,hi] = plotHist_sp(y1,y3,xlab,ylab,leg, coln, tit, fh, [], yy, documsum, nBins, [], doSmooth); hold on           
        [bins,ye,yi,x,he,hi] = plotHist_sp(y1,y2,xlab,ylab,leg, coln, tit, fh, [], yy, documsum, nBins, [], doSmooth);            
    %     [~,bins,ye,yi,x,he,hi] = plotHist(y1,y2,xlab,ylab,leg, cols, yy,fh, nBins, doSmooth);

        %%%%%%%% mark sig values
    %     subplot(211)
        % E
        H1 = area(x,ye,'FaceColor',[1 1 1], 'facealpha', 0);
        idx = x < mse(1);
        H1 = area(x(idx),ye(idx), 'facecolor', 'k', 'facealpha', .2);
        idx = x > mse(2);
        H = area(x(idx),ye(idx), 'facecolor', 'k', 'facealpha', .2);
        % I
        H11 = area(x,yi,'FaceColor',[1 1 1], 'edgecolor', 'r', 'facealpha', 0);
        idx = x < msi(1);
        H11 = area(x(idx),yi(idx), 'facecolor', 'r', 'edgecolor', 'r', 'facealpha', .2);
        idx = x > msi(2);
        H = area(x(idx),yi(idx), 'facecolor', 'r', 'edgecolor', 'r', 'facealpha', .2);

        xlabel(xlab)
        ylabel('Fraction neurons')
        legend([he,hi], leg)        
        set(gca, 'tickdir', 'out')    

        if savefigs
            savefig(fh, fullfile(dirnFig, [namc,'_exampleDay_noSmooth_',tit,'_dist_timeM1_neursPooled_ROC_curr_chAl_excInh_', nowStr,'.fig']))
            print(fh, '-dpdf', fullfile(dirnFig, [namc,'_exampleDay_',tit,'_dist_timeM1_neursPooled_ROC_curr_chAl_excInh_', nowStr]))
        end

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%% Stairs plot %%%%%%%%%%%%%%%%%% 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        fh = figure('position', [498   658   390   267]); hold on        
        %%%%% exc
        he = stairs(x, ye, 'k');

        idx = x < mse(1);
        x0 = [x(idx);x(idx)];
        y = [ye(idx)';ye(idx)'];
        area(x0([2:end end]),y(1:end), 'facecolor', 'k', 'facealpha', .2, 'edgecolor', 'none')        

        idx = x > mse(2);
        x0 = [x(idx);x(idx)];
        y = [ye(idx)';ye(idx)'];
        area(x0([2:end end]),y(1:end), 'facecolor', 'k', 'facealpha', .2, 'edgecolor', 'none')        
        
        %%%%% inh
        hi = stairs(x, yi, 'r');

        idx = x < msi(1);
        x0 = [x(idx);x(idx)];
        y = [yi(idx)';yi(idx)'];
        area(x0([2:end end]),y(1:end), 'facecolor', 'r', 'facealpha', .2, 'edgecolor', 'none')        

        idx = x > msi(2);
        x0 = [x(idx);x(idx)];
        y = [yi(idx)';yi(idx)'];
        area(x0([2:end end]),y(1:end), 'facecolor', 'r', 'facealpha', .2, 'edgecolor', 'none')        
        
        xlabel(xlab)
        ylabel('Fraction neurons')
        legend([he,hi], leg)        
        set(gca, 'tickdir', 'out') 
        title(tit)

        if savefigs
            savefig(fh, fullfile(dirnFig, [namc,'_exampleDay_noSmooth_',tit,'_distStairs_timeM1_neursPooled_ROC_curr_chAl_excInh_', nowStr,'.fig']))
            print(fh, '-dpdf', fullfile(dirnFig, [namc,'_exampleDay_',tit,'_distStairs_timeM1_neursPooled_ROC_curr_chAl_excInh_', nowStr]))
        end
        
    end
    

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
    
    
    %% Heatmap of AUC: days vs time in the trial. Pixel intensity: averaged AUC across neurons for each day and at each time point
    
    xlab = simpleTokenize(namc, '_'); xlab = xlab{1};

    a = aveexc';
%     a = aveexc_shfl';
%     a = aveexc' - aveexc_shfl';
    aa = sum(isnan(a),2);
    a(aa==size(a,2),:) = [];
    

    ai = aveinh';
%     ai = aveinh_shfl';
%     ai = aveinh' - aveinh_shfl';
    aa = sum(isnan(ai),2);
    ai(aa==size(ai,2),:) = [];
    
    
    fh = figure('position', [21   607   555   369], 'name', mouse); 
    
    % exc
    subplot(121)
    imagesc(a); hold on;
    vline(nPreMin_allMice(im)+1)
    tm = get(gca, 'xtick'); 
    set(gca, 'xticklabel', round(time_aligned_allMice{im}(tm)))
    set(gca,'tickdir','out', 'box','off')
    xlabel('Time rel. choice onset (ms)')
    ylabel('Training days')
    c = colorbar; c.Label.String = xlab;
    title('exc')

    % inh
    subplot(122)
    imagesc(ai); hold on;
    vline(nPreMin_allMice(im)+1)
    tm = get(gca, 'xtick'); 
    set(gca, 'xticklabel', round(time_aligned_allMice{im}(tm)))
    set(gca,'tickdir','out', 'box','off')
    xlabel('Time rel. choice onset (ms)')
    ylabel('Training days')
    c = colorbar; c.Label.String = xlab;
    title('inh')

    
    if savefigs
        savefig(fh, fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInh_trainingDays_heatmaps_aveNeurs_', nowStr,'.fig']))
        print(fh, '-dpdf', fullfile(dirnFig, [namc,'_','ROC_curr_chAl_excInh_trainingDays_heatmaps_aveNeurs_', nowStr]))
    end
    
    
    
end
    
   






