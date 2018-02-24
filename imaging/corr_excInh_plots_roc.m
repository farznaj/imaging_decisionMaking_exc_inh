% First run corr_excInh_plots to get the vars you need here.


%%%%% We dont care about the sign of correlatins... just the magnitude ...
% I dont know what to make of sign in a 100ms time window!
% if u want to care about the sign, remove abs from r values in the codes
% below.

% Compare tuning of the following two groups of inh neurons:
% inh neurons that are equally connected to exc of same and opposite tuning 
% vs.
% inh neurons that are strongly connected to exc of same tuning but weakly connected to exc of opposite tuning


%% 
%%%%%%%%%%%%%%%%%%%%%%%%% Averaged pairwise correlations %%%%%%%%%%%%%%%%%%%%%%%%%  
% Assuming each neuron is connected to all other neurons, to estimate the
% neuron's (N) connectivity strength with a particular population ("P",
% either Eipsi, Econtra, Iipsi, or Icontra), we average pairwise
% correlation of the neuron (N) with all the neurons in that population
% (P). This way we can compare connectivity of N with different populations 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compare tuning of Inh neurons with high corrAve_same and low corrAve_diff
% with those with low corrAve_same and high corrAve_diff
% the former group is expected to have higher tuning.

% thqHi = .8; %thq = .95; % we get closer number of elements for hi vs lo corr with .8 than .95 % to set the threshold corr for identifying hi vs lo corr % to do so we take the .95 quantile of the shuffled corr (separately done for each mouse, all days pooled... assuming that data corrs > .95 of shfl corrs are significant) 
thqLo = .8;

cols = {'k', 'r'};
ylab = 'Fraction neurons';
xlab = 'ROC (absDev)';


%% Set the same corr bins for all mice

rall = cell(1, length(mice));
for im = 1:length(mice)
    rall{im} = [rAve_EI_same{im}{:}, rAve_EI_diff{im}{:}, rAve_EE_same{im}{:}, rAve_EE_diff{im}{:}, rAve_II_same{im}{:}, rAve_II_diff{im}{:}];
end

mn0 = min(cellfun(@min, rall));
mx0 = max(cellfun(@max, rall));
disp([mn0, mx0])
% r1 = round(mn0-.05,1); 
r2 = round(mx0+.05,1);
% bins = r1 : (r2-r1)/10 : r2; 
bins = [0, .02, r2]; % bins = [0, .2, r2]; % we only study positive correlations ... also we use abs for corrs below...
% bins = [0, .02, .02, r2];
hiBin = length(bins)-1;


%%
ROC_I_rAveLoDiff = cell(1, length(mice));
ROC_I_rAveHiDiff = cell(1, length(mice));
ROC_E_rAveLoDiff = cell(1, length(mice));
ROC_E_rAveHiDiff = cell(1, length(mice));

for im = 1:length(mice)
    
    % set the threshold corr for identifying hi vs lo corr: 
    % to do so we take the .95 quantile of the shuffled corr (separately done for each mouse, all days pooled... assuming that data corrs > .95 of shfl corrs are significant) 
    % inh
    binsI = bins;
    a = abs([rAve_Ii_Ei_shfl{im}{:}, rAve_Ii_Ec_shfl{im}{:}, rAve_Ic_Ec{im}{:}, rAve_Ic_Ei{im}{:}]);
    binsI(2) = quantile(a, thqLo); %fprintf('threshold corr = %.3f\n', binsI(2))    
%     binsI(3) = quantile(a, thqHi);
    
    % exc
    binsE = bins;
    a = abs([rAve_Ei_Ii_shfl{im}{:}, rAve_Ei_Ic_shfl{im}{:}, rAve_Ec_Ic_shfl{im}{:}, rAve_Ec_Ii_shfl{im}{:}]);
    binsE(2) = quantile(a, thqLo); %fprintf('threshold corr = %.3f\n', binsE(2))
%     binsI(3) = quantile(a, thqHi);
    
    
    for iday = 1:numDaysAll(im)    
        
        %%%%% EI case %%%%%
        
        %%%%%%%%%%%%%%%%%%%% Inh %%%%%%%%%%%%%%%%%%%%
        %%%%% same
        % ipsi
        [n0, ~, i_rAve_Ii_Ei] = histcounts(abs(rAve_Ii_Ei{im}{iday}), binsI);        
        % contra
        [n0, ~, i_rAve_Ic_Ec] = histcounts(abs(rAve_Ic_Ec{im}{iday}), binsI);        
        
        %%%%% diff
        % ipsi
        [n0, ~, i_rAve_Ii_Ec] = histcounts(abs(rAve_Ii_Ec{im}{iday}), binsI);        % i==1: low corr ; % i==2: high corr        
        % contra
        [n0, ~, i_rAve_Ic_Ei] = histcounts(abs(rAve_Ic_Ei{im}{iday}), binsI);
       
        
        %%% set indeces of neurons that have hi corrAve_same and lo corrAve_diff (or negative corrAve_diff)
        Ii_inds_corrAveLoDiff = (i_rAve_Ii_Ei == hiBin) & (i_rAve_Ii_Ec == 1); %(rAve_Ii_Ec{im}{iday}<0); %
        Ic_inds_corrAveLoDiff = (i_rAve_Ic_Ec == hiBin) & (i_rAve_Ic_Ei == 1); %(rAve_Ic_Ei{im}{iday}<0); %
%         Ii_inds_corrAveHiSameLoDiff = (i_rAve_Ii_Ec == 1); 
%         Ic_inds_corrAveHiSameLoDiff = (i_rAve_Ic_Ei == 1); 
        
        %%% set indeces of neurons that have lo corrAve_same and hi corrAve_diff
        Ii_inds_corrAveHiDiff = (i_rAve_Ii_Ei == hiBin) & (i_rAve_Ii_Ec == hiBin);
        Ic_inds_corrAveHiDiff = (i_rAve_Ic_Ec == hiBin) & (i_rAve_Ic_Ei == hiBin);
%         Ii_inds_corrAveLoSameHiDiff = (i_rAve_Ii_Ec == hiBin);
%         Ic_inds_corrAveLoSameHiDiff = (i_rAve_Ic_Ei==2);
        


        %%%%%%%%%%%%% set ROC for Ii and Ic neurons in this day.. important to compute deviation from chance, since we are pooling ipsi and contra.
        roc_Ii = abs(.5 - roc_inh_timeM1{im}{iday}(inh_ipsi_timeM1{im}{iday}));
        roc_Ic = abs(.5 - roc_inh_timeM1{im}{iday}(inh_contra_timeM1{im}{iday}));
        
        %%% set ROC for neurons that have lo corrAve_same and hi corrAve_diff
        % hi corrAve_same and lo corrAve_diff
        ROC_I_rAveLoDiff{im}{iday} = [roc_Ii(Ii_inds_corrAveLoDiff), roc_Ic(Ic_inds_corrAveLoDiff)];
%         ROC_I_rAveHiSameLoDiff{im}{iday} = roc_Ic(Ic_inds_corrAveHiSameLoDiff);
        
        % lo corrAve_same and lo corrAve_diff
        ROC_I_rAveHiDiff{im}{iday} = [roc_Ii(Ii_inds_corrAveHiDiff), roc_Ic(Ic_inds_corrAveHiDiff)];
%         ROC_I_rAveLoSameHiDiff{im}{iday} = roc_Ic(Ic_inds_corrAveLoSameHiDiff);


        

        %%%%%%%%%%%%%%%%%%%% Exc %%%%%%%%%%%%%%%%%%%%
        %%%%% same
        % ipsi
        [n0, ~, i_rAve_Ei_Ii] = histcounts(abs(rAve_Ei_Ii{im}{iday}), binsE);        
        % contra
        [n0, ~, i_rAve_Ec_Ic] = histcounts(abs(rAve_Ec_Ic{im}{iday}), binsE);
        
        %%%%% diff
        % ispi
        [n0, ~, i_rAve_Ei_Ic] = histcounts(abs(rAve_Ei_Ic{im}{iday}), binsE);
        % contra
        [n0, ~, i_rAve_Ec_Ii] = histcounts(abs(rAve_Ec_Ii{im}{iday}), binsE);        % i==1: low corr ; % i==2: high corr                
       
        
        %%% set indeces of neurons that have hi corrAve_same and lo corrAve_diff (or nagative corrAve_diff)
        Ei_inds_corrAveLoDiff = (i_rAve_Ei_Ii == hiBin) & (i_rAve_Ei_Ic == 1); %(rAve_Ei_Ic{im}{iday}<0); %
        Ec_inds_corrAveLoDiff = (i_rAve_Ec_Ic == hiBin) & (i_rAve_Ec_Ii == 1); %(rAve_Ec_Ii{im}{iday}<0); %
%         Ei_inds_corrAveHiSameLoDiff = (i_rAve_Ei_Ic == 1); 
%         Ec_inds_corrAveHiSameLoDiff = (i_rAve_Ec_Ii == 1); 
        
        %%% set indeces of neurons that have lo corrAve_same and hi corrAve_diff
        Ei_inds_corrAveHiDiff = (i_rAve_Ei_Ii == hiBin) & (i_rAve_Ei_Ic == hiBin);
        Ec_inds_corrAveHiDiff = (i_rAve_Ec_Ic == hiBin) & (i_rAve_Ec_Ii == hiBin);
%         Ei_inds_corrAveLoSameHiDiff = (i_rAve_Ei_Ic == hiBin);
%         Ec_inds_corrAveLoSameHiDiff = (i_rAve_Ec_Ii==2);
        

        %%%%%%%%% set ROC for Ei and Ec neurons in this day.. important to compute deviation from chance, since we are pooling ipsi and contra.
        roc_Ei = abs(.5 - roc_exc_timeM1{im}{iday}(exc_ipsi_timeM1{im}{iday}));
        roc_Ec = abs(.5 - roc_exc_timeM1{im}{iday}(exc_contra_timeM1{im}{iday}));
%         roc_Ei = roc_exc_timeM1{im}{iday}(exc_ipsi_timeM1{im}{iday});
%         roc_Ec = roc_exc_timeM1{im}{iday}(exc_contra_timeM1{im}{iday});
        
        %%%%% Now set ROC for neurons that have lo corrAve_same and hi corrAve_diff
        % hi corrAve_same and lo corrAve_diff
        ROC_E_rAveLoDiff{im}{iday} = [roc_Ei(Ei_inds_corrAveLoDiff), roc_Ec(Ec_inds_corrAveLoDiff)];
%         ROC_E_rAveHiSameLoDiff{im}{iday} = roc_Ec(Ec_inds_corrAveHiSameLoDiff);
        
        % lo corrAve_same and lo corrAve_diff
        ROC_E_rAveHiDiff{im}{iday} = [roc_Ei(Ei_inds_corrAveHiDiff), roc_Ec(Ec_inds_corrAveHiDiff)];        
%         ROC_E_rAveLoSameHiDiff{im}{iday} = roc_Ec(Ec_inds_corrAveLoSameHiDiff);
        
    end
end


%% For each mouse, pool roc values across days

ROC_I_rAveLoDiff_daysPooled = cell(1, length(mice));
ROC_I_rAveHiDiff_daysPooled = cell(1, length(mice));
ROC_E_rAveLoDiff_daysPooled = cell(1, length(mice));
ROC_E_rAveHiDiff_daysPooled = cell(1, length(mice));

for im = 1:length(mice)
    ROC_I_rAveLoDiff_daysPooled{im} = cell2mat(ROC_I_rAveLoDiff{im});
    ROC_I_rAveHiDiff_daysPooled{im} = cell2mat(ROC_I_rAveHiDiff{im});
    ROC_E_rAveLoDiff_daysPooled{im} = cell2mat(ROC_E_rAveLoDiff{im});
    ROC_E_rAveHiDiff_daysPooled{im} = cell2mat(ROC_E_rAveHiDiff{im});    
end
inh_ = [ROC_I_rAveLoDiff_daysPooled ; ROC_I_rAveHiDiff_daysPooled]
exc_ = [ROC_E_rAveLoDiff_daysPooled ; ROC_E_rAveHiDiff_daysPooled]


%% Plot errorbar, showing each mouse, ave +/- se of rAve across days (already averaged across neurons for each day)

x = 1:length(mice);
gp = .2;
marg = .2;

figure('name', 'All mice', 'position', [9   631   516   297]); 
set(gca, 'position', [0.2919    0.1908    0.5229    0.7095])

%%%%%%%%%% Plot EI %%%%%%%%%%%
typeNs = 'Inh';    
subplot(121); hold on
h1 = errorbar(x, cellfun(@mean, ROC_I_rAveHiDiff_daysPooled), cellfun(@std, ROC_I_rAveHiDiff_daysPooled) ./ sqrt(cellfun(@length, ROC_I_rAveHiDiff_daysPooled)), 'r.', 'linestyle', 'none');
h2 = errorbar(x+gp, cellfun(@mean, ROC_I_rAveLoDiff_daysPooled), cellfun(@std, ROC_I_rAveLoDiff_daysPooled) ./ sqrt(cellfun(@length, ROC_I_rAveLoDiff_daysPooled)), 'k.', 'linestyle', 'none');
xlim([x(1)-marg, x(end)+gp+marg])
set(gca,'xtick', x)
set(gca,'xticklabel', mice)
title(typeNs)
ylabel('ROC (absDev; mean +/- se neurons)')
legend([h1,h2], {'hiDiff','loDiff'})


typeNs = 'Exc';    
subplot(122); hold on
h1 = errorbar(x, cellfun(@mean, ROC_E_rAveHiDiff_daysPooled), cellfun(@std, ROC_E_rAveHiDiff_daysPooled) ./ sqrt(cellfun(@length, ROC_E_rAveHiDiff_daysPooled)), 'r.', 'linestyle', 'none');
h2 = errorbar(x+gp, cellfun(@mean, ROC_E_rAveLoDiff_daysPooled), cellfun(@std, ROC_E_rAveLoDiff_daysPooled) ./ sqrt(cellfun(@length, ROC_E_rAveLoDiff_daysPooled)), 'k.', 'linestyle', 'none');
xlim([x(1)-marg, x(end)+gp+marg])
set(gca,'xtick', x)
set(gca,'xticklabel', mice)
title(typeNs)
ylabel('ROC (absDev; mean +/- se neurons)')
% legend([h1,h2], {'hiDiff','loDiff'})


if saveFigs
    namv = sprintf('ROC_loHiCorrFR_avePairwise_aveSeNs_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);
    
    d = fullfile(dirn0fr, 'sumAllMice', nnow);
    fn = fullfile(d, namv);
    
    savefig(gcf, fn)
    
    % print to pdf
%         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
%         figs_adj_poster_ax(fn, axpos)                
end    


%% Plot histograms for each mouse to compare ROC of I neurons with hi corrAveSame and lo corrAveDiff with I neurons with lo corrAveSDame and hi corrAveDiff

nBins = 15;
doSmooth = 1;

leg = {'lowDiff & hiSame avePairwise', 'hiDiff & hiSame avePairwise'};

for im = 1%:length(mice)

    fh = figure('name', mice{im}, 'position', [92   350   869   528]); %[22   399   991   528]);              %fh = figure('name', [mice{im}, ' - EI'], 'position', [27         404        1370         521]);
        
    typeNs = 'Inh';
    sp = [221,223];
    plotHist(ROC_I_rAveLoDiff_daysPooled{im}, ROC_I_rAveHiDiff_daysPooled{im}, xlab, ylab, ...
        leg, cols, [], fh, nBins, doSmooth, linestylesData, sp); 
    subplot(sp(1)), title(typeNs)
    
    
    typeNs = 'Exc';  
    sp = [222,224];
    plotHist(ROC_E_rAveLoDiff_daysPooled{im}, ROC_E_rAveHiDiff_daysPooled{im}, xlab, ylab, ...
        leg, cols, [], fh, nBins, doSmooth, linestylesData, sp); 
    subplot(sp(1)), title(typeNs)
    
    %{
    typeNs = 'inh & exc';  
    sp = [233,236];
    plotHist([ROC_I_rAveHiSameLoDiff_daysPooled{im}, ROC_E_rAveHiSameLoDiff_daysPooled{im}], ...
        [ROC_I_rAveLoSameHiDiff_daysPooled{im}, ROC_E_rAveLoSameHiDiff_daysPooled{im}], xlab, ylab, ...
        leg, cols, [], fh, nBins, doSmooth, linestylesData, sp); 
    subplot(sp(1)), title(typeNs)
    %}
    
    if saveFigs
        namv = sprintf('ROC_loHiCorrFR_avePairwise_distDaysPooled_sameDiffTun_FR%s_ROC%s_%s_curr%s_%s_%s', alFR, al, time2an, o2a, mice{im}, nowStr);
        
        d = fullfile(dirn0fr, mice{im}, nnow);
        fn = fullfile(d, namv);
        
        savefig(gcf, fn)
        
        % print to pdf
%         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
%         figs_adj_poster_ax(fn, axpos)                
    end    
end


%% Plot hist of all mice

inh_ = [length([ROC_I_rAveLoDiff_daysPooled{:}]), length([ROC_I_rAveHiDiff_daysPooled{:}])]
exc_ = [length([ROC_E_rAveLoDiff_daysPooled{:}]), length([ROC_E_rAveHiDiff_daysPooled{:}])]

nBins = 20;
doSmooth = 1;

fh = figure('name', 'All mice', 'position', [22   399   991   528]);              %fh = figure('name', [mice{im}, ' - EI'], 'position', [27         404        1370         521]);

typeNs = 'Inh';
sp = [221,223];
plotHist([ROC_I_rAveLoDiff_daysPooled{:}], [ROC_I_rAveHiDiff_daysPooled{:}], xlab, ylab, ...
    leg, cols, [], fh, nBins, doSmooth, linestylesData, sp);
subplot(sp(1)), title(typeNs)


typeNs = 'Exc';
sp = [222,224];
plotHist([ROC_E_rAveLoDiff_daysPooled{:}], [ROC_E_rAveHiDiff_daysPooled{:}], xlab, ylab, ...
    leg, cols, [], fh, nBins, doSmooth, linestylesData, sp);
subplot(sp(1)), title(typeNs)

%{
    typeNs = 'inh & exc';
    sp = [233,236];
    plotHist([ROC_I_rAveHiSameLoDiff_daysPooled{:}], ROC_E_rAveHiSameLoDiff_daysPooled{:}]], ...
        [ROC_I_rAveLoSameHiDiff_daysPooled{:}], ROC_E_rAveLoSameHiDiff_daysPooled{:}]], xlab, ylab, ...
        leg, cols, [], fh, nBins, doSmooth, linestylesData, sp);
    subplot(sp(1)), title(typeNs)
%}

if saveFigs
    namv = sprintf('ROC_loHiCorrFR_avePairwise_distDaysPooled_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);
    
    d = fullfile(dirn0fr, 'sumAllMice', nnow);
    fn = fullfile(d, namv);
    
    savefig(gcf, fn)
    
    % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)
end






%%
%%%%%%%%%%%%%%% Same analysis as above, now using corr values computed on population-averaged FRs.
%%%
%%%%%%%%%%%%%%%%%%%%%%%%% Correlation between population-averaged FRs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
%%%%% Are strength of tuning and correlations related?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%% see if EI pairs with high corr_diff, have lower tuning...

% get EI pairs with very high corr_diff, then get their ROCs... we expect it to be low
% get EI pairs with very low corr_diff, then get their ROCs... we expect it to be high
% the opposite trend is expected for corr_same: 
% get EI pairs with very high corr_same, then get their ROCs... we expect it to be high
% get EI pairs with very low corr_same, then get their ROCs... we expect it to be low

% remember each pair has 2 rocs... bc there r 2 neurons...

% do a hist on r_EI_diff_allMice ,,, to get high and low corrs...
% the apply the same indices on roc vals... 


%%
% thqHi = .95; % we get closer number of elements for hi vs lo corr with .95 than .8.
thqLo = .95;

rocInh = roc_inh_timeM1;
rocExc = roc_exc_timeM1; % like inh, exc also will be more tuned (ie hi roc) if its corr_E_I_diff is low.

leg = {'lowCorr','hiCorr'};
% set the same corr bins for all mice
mn0 = min([min(cellfun(@min, corr_exci_inhi)), min(cellfun(@min, corr_excc_inhc)), min(cellfun(@min, corr_exci_inhc)), min(cellfun(@min, corr_excc_inhi))]);
mx0 = max([max(cellfun(@max, corr_exci_inhi)), max(cellfun(@max, corr_excc_inhc)), max(cellfun(@max, corr_exci_inhc)), max(cellfun(@max, corr_excc_inhi))]);
% r1 = round(mn0-.05,1); 
r2 = round(mx0+.05,1);
% bins = r1 : (r2-r1)/10 : r2;
bins = [0, .02, r2]; % bins = [0, .2, r2]; % we only study positive correlations ... also we use abs for corrs below...
% bins = [0, .02, .02, r2];
hiBin = length(bins)-1;


%%
roc_inh_loDiff = cell(1, length(mice));
roc_inh_hiDiff = cell(1, length(mice));
roc_exc_loDiff = cell(1, length(mice));
roc_exc_hiDiff = cell(1, length(mice));

for im = 1:length(mice)    
    
    a = abs([corr_exci_inhi_shfl{im} , corr_excc_inhc_shfl{im} , corr_exci_inhc_shfl{im} , corr_excc_inhi_shfl{im}]);
    bins(2) = quantile(a, thqLo); %fprintf('threshold corr = %.3f\n', bins(2))
%     bins(3) = quantile(a, thqHi);
    
    % same
    [n0, ~, i_bin_EiIi_allMice] = histcounts(abs(corr_exci_inhi{im}), bins);
    [n0, ~, i_bin_EcIc_allMice] = histcounts(abs(corr_excc_inhc{im}), bins);
    
    % diff
    [n0, ~, i_bin_EiIc_allMice] = histcounts(abs(corr_exci_inhc{im}), bins);
    [n0, ~, i_bin_EcIi_allMice] = histcounts(abs(corr_excc_inhi{im}), bins);
    
    
    %%%%%%%%%%%%%%%%%%%%% INH %%%%%%%%%%%%%%%%%%%%%
    %%% inh: hi same and low diff
    % ipsi
    days_Ii_loDiff = (i_bin_EiIi_allMice == hiBin) & (i_bin_EcIi_allMice == 1);
    % contra
    days_Ic_loDiff = (i_bin_EcIc_allMice == hiBin) & (i_bin_EiIc_allMice == 1);
    
    %%% inh: hi same and hi diff
    % ipsi
    days_Ii_hiDiff = (i_bin_EiIi_allMice == hiBin) & (i_bin_EcIi_allMice == hiBin);
    % contra
    days_Ic_hiDiff = (i_bin_EcIc_allMice == hiBin) & (i_bin_EiIc_allMice == hiBin);
    
    
    %%%%%%% ROC %%%%%%%
    % compute tuning, ie abs difference from chance (.5)... if we want to pool ipsi and contra tuning we have to do this... bc for contra roc values get lower (from .5) as tuning increases
    %%%%% hiSame_loDiff
    % ipsi
    rocAllI_Ii_loDiff = abs(.5-[rocInh{im}{days_Ii_loDiff}]); % ROC of all inh neurons of all days whose population corr for Ii is hiSame and loDiff
    inds = [inh_ipsi_timeM1{im}{days_Ii_loDiff}]; % index of Ii neurons of all days whose population corr is hiSame and loDiff  
    roc_Ii_loDiff = rocAllI_Ii_loDiff(inds); % ROC of all Ii neurons of all days whose population corr for Ii is hiSame and loDiff
    % contra
    rocAllI_Ic_loDiff = abs(.5-[rocInh{im}{days_Ic_loDiff}]); % ROC of all inh neurons of all days whose population corr for Ic is hiSame and loDiff
    inds = [inh_contra_timeM1{im}{days_Ic_loDiff}]; % index of Ic neurons of all days whose population corr is hiSame and loDiff  
    roc_Ic_loDiff = rocAllI_Ic_loDiff(inds); % ROC of all Ic neurons of all days whose population corr for Ic is hiSame and loDiff
    % pool ipsi and contra
    roc_inh_loDiff{im} = [roc_Ii_loDiff , roc_Ic_loDiff];
    
    %%%%% hiSame_hiDiff
    % ipsi
    rocAllI_Ii_hiDiff = abs(.5-[rocInh{im}{days_Ii_hiDiff}]);
    inds = [inh_ipsi_timeM1{im}{days_Ii_hiDiff}];
    roc_Ii_hiDiff = rocAllI_Ii_hiDiff(inds);
    % contra
    rocAllI_Ic_hiDiff = abs(.5-[rocInh{im}{days_Ic_hiDiff}]);
    inds = [inh_contra_timeM1{im}{days_Ic_hiDiff}];
    roc_Ic_hiDiff = rocAllI_Ic_hiDiff(inds);
    % pool ipsi and contra
    roc_inh_hiDiff{im} = [roc_Ii_hiDiff , roc_Ic_hiDiff];
    
    
    
    %%%%%%%%%%%%%%%%%%%%% EXC %%%%%%%%%%%%%%%%%%%%%
    %%% exc: hi same and low diff
    % ipsi
    days_Ei_loDiff = (i_bin_EiIi_allMice == hiBin) & (i_bin_EiIc_allMice == 1);
    % contra
    days_Ec_loDiff = (i_bin_EcIc_allMice == hiBin) & (i_bin_EcIi_allMice == 1);
    
    %%% exc: hi same and hi diff
    % ipsi
    days_Ei_hiDiff = (i_bin_EiIi_allMice == hiBin) & (i_bin_EiIc_allMice == hiBin);
    % contra
    days_Ec_hiDiff = (i_bin_EcIc_allMice == hiBin) & (i_bin_EcIi_allMice == hiBin);
    
    
    %%%%%%% ROC %%%%%%%
    % compute tuning, ie abs difference from chance (.5)... if we want to pool ipsi and contra tuning we have to do this... bc for contra roc values get lower (from .5) as tuning increases
    %%%%% hiSame_loDiff
    % ipsi
    rocAllE_Ei_loDiff = abs(.5-[rocExc{im}{days_Ei_loDiff}]); % ROC of all exc neurons of all days whose population corr for Ei is hiSame and loDiff
    inds = [exc_ipsi_timeM1{im}{days_Ei_loDiff}]; % index of Ei neurons of all days whose population corr is hiSame and loDiff  
    roc_Ei_loDiff = rocAllE_Ei_loDiff(inds); % ROC of all Ei neurons of all days whose population corr for Ei is hiSame and loDiff
    % contra
    rocAllE_Ec_loDiff = abs(.5-[rocExc{im}{days_Ec_loDiff}]); % ROC of all exc neurons of all days whose population corr for Ec is hiSame and loDiff
    inds = [exc_contra_timeM1{im}{days_Ec_loDiff}]; % index of Ec neurons of all days whose population corr is hiSame and loDiff  
    roc_Ec_loDiff = rocAllE_Ec_loDiff(inds); % ROC of all Ec neurons of all days whose population corr for Ec is hiSame and loDiff
    % pool ipsi and contra
    roc_exc_loDiff{im} = [roc_Ei_loDiff , roc_Ec_loDiff];
    
    %%%%% hiSame_hiDiff
    % ipsi
    rocAllE_Ei_hiDiff = abs(.5-[rocExc{im}{days_Ei_hiDiff}]);
    inds = [exc_ipsi_timeM1{im}{days_Ei_hiDiff}];
    roc_Ei_hiDiff = rocAllE_Ei_hiDiff(inds);
    % contra
    rocAllE_Ec_hiDiff = abs(.5-[rocExc{im}{days_Ec_hiDiff}]);
    inds = [exc_contra_timeM1{im}{days_Ec_hiDiff}];
    roc_Ec_hiDiff = rocAllE_Ec_hiDiff(inds);
    % pool ipsi and contra
    roc_exc_hiDiff{im} = [roc_Ei_hiDiff , roc_Ec_hiDiff];
    
end

inh_ = [roc_inh_loDiff ; roc_inh_hiDiff]
exc_ = [roc_exc_loDiff ; roc_exc_hiDiff]


%% Plot errorbar, showing each mouse, ave +/- se of rAve across days (already averaged across neurons for each day)

x = 1:length(mice);
gp = .2;
marg = .2;

figure('name', 'All mice', 'position', [9   631   516   297]); 
set(gca, 'position', [0.2919    0.1908    0.5229    0.7095])

%%%%%%%%%% Plot EI %%%%%%%%%%%
typeNs = 'Inh';    
subplot(121); hold on
h1 = errorbar(x, cellfun(@mean, roc_inh_hiDiff), cellfun(@std, roc_inh_hiDiff) ./ sqrt(cellfun(@length, roc_inh_hiDiff)), 'r.', 'linestyle', 'none');
h2 = errorbar(x+gp, cellfun(@mean, roc_inh_loDiff), cellfun(@std, roc_inh_loDiff) ./ sqrt(cellfun(@length, roc_inh_loDiff)), 'k.', 'linestyle', 'none');
xlim([x(1)-marg, x(end)+gp+marg])
set(gca,'xtick', x)
set(gca,'xticklabel', mice)
title(typeNs)
ylabel('ROC (absDev; mean +/- se neurons)')
legend([h1,h2], {'hiDiff','loDiff'})


typeNs = 'Exc';    
subplot(122); hold on
h1 = errorbar(x, cellfun(@mean, roc_exc_hiDiff), cellfun(@std, roc_exc_hiDiff) ./ sqrt(cellfun(@length, roc_exc_hiDiff)), 'r.', 'linestyle', 'none');
h2 = errorbar(x+gp, cellfun(@mean, roc_exc_loDiff), cellfun(@std, roc_exc_loDiff) ./ sqrt(cellfun(@length, roc_exc_loDiff)), 'k.', 'linestyle', 'none');
xlim([x(1)-marg, x(end)+gp+marg])
set(gca,'xtick', x)
set(gca,'xticklabel', mice)
title(typeNs)
ylabel('ROC (absDev; mean +/- se neurons)')
% legend([h1,h2], {'hiDiff','loDiff'})


if saveFigs
    namv = sprintf('ROC_loHiCorrFR_popAveFRs_aveSeNs_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);
    
    d = fullfile(dirn0fr, 'sumAllMice', nnow);
    fn = fullfile(d, namv);
    
    savefig(gcf, fn)
    
    % print to pdf
%         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
%         figs_adj_poster_ax(fn, axpos)                
end    


%% Plot histograms for each mouse to compare ROC of I neurons with hi corrAveSame and lo corrAveDiff with I neurons with lo corrAveSDame and hi corrAveDiff

nBins = 20;
doSmooth = 0;

leg = {'lowDiff & hiSame corrPop', 'hiDiff & hiSame corrPop'};

for im = 1:length(mice)

    fh = figure('name', mice{im}, 'position', [22   399   991   528]);   %fh = figure('name', [mice{im}, ' - EI'], 'position', [27         404        1370         521]);
        
    typeNs = 'Inh';
    sp = [221,223];
    plotHist(roc_inh_loDiff{im}, roc_inh_hiDiff{im}, xlab, ylab, leg, cols, [], fh, nBins, doSmooth, linestylesData, sp); 
    subplot(sp(1)), title(typeNs)
    
    
    typeNs = 'Exc';  
    sp = [222,224];
    plotHist(roc_exc_loDiff{im}, roc_exc_hiDiff{im}, xlab, ylab, leg, cols, [], fh, nBins, doSmooth, linestylesData, sp); 
    subplot(sp(1)), title(typeNs)
    
    %{
    typeNs = 'inh & exc';  
    sp = [233,236];
    plotHist([ROC_I_rAveHiSameLoDiff_daysPooled{im}, ROC_E_rAveHiSameLoDiff_daysPooled{im}], ...
        [ROC_I_rAveLoSameHiDiff_daysPooled{im}, ROC_E_rAveLoSameHiDiff_daysPooled{im}], xlab, ylab, ...
        leg, cols, [], fh, nBins, doSmooth, linestylesData, sp); 
    subplot(sp(1)), title(typeNs)
    %}
    
    if saveFigs
        namv = sprintf('ROC_loHiCorrFR_popAveFRs_distDaysPooled_sameDiffTun_FR%s_ROC%s_%s_curr%s_%s_%s', alFR, al, time2an, o2a, mice{im}, nowStr);
        
        d = fullfile(dirn0fr, mice{im}, nnow);
        fn = fullfile(d, namv);
        
        savefig(gcf, fn)
        
        % print to pdf
%         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
%         figs_adj_poster_ax(fn, axpos)                
    end    
end


%% Plot hist of all mice

inh_ = [length([roc_inh_loDiff{:}]), length([roc_inh_hiDiff{:}])]
exc_ = [length([roc_exc_loDiff{:}]), length([roc_exc_hiDiff{:}])]

nBins = 30;
doSmooth = 0;

fh = figure('name', 'All mice', 'position', [22   399   991   528]);              %fh = figure('name', [mice{im}, ' - EI'], 'position', [27         404        1370         521]);

typeNs = 'Inh';
sp = [221,223];
plotHist([roc_inh_loDiff{:}], [roc_inh_hiDiff{:}], xlab, ylab, ...
    leg, cols, [], fh, nBins, doSmooth, linestylesData, sp);
subplot(sp(1)), title(typeNs)


typeNs = 'Exc';
sp = [222,224];
plotHist([roc_exc_loDiff{:}], [roc_exc_hiDiff{:}], xlab, ylab, ...
    leg, cols, [], fh, nBins, doSmooth, linestylesData, sp);
subplot(sp(1)), title(typeNs)

%{
    typeNs = 'inh & exc';
    sp = [233,236];
    plotHist([ROC_I_rAveHiSameLoDiff_daysPooled{:}], ROC_E_rAveHiSameLoDiff_daysPooled{:}]], ...
        [ROC_I_rAveLoSameHiDiff_daysPooled{:}], ROC_E_rAveLoSameHiDiff_daysPooled{:}]], xlab, ylab, ...
        leg, cols, [], fh, nBins, doSmooth, linestylesData, sp);
    subplot(sp(1)), title(typeNs)
%}

if saveFigs
    namv = sprintf('ROC_loHiCorrFR_popAveFRs_distDaysPooled_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);
    
    d = fullfile(dirn0fr, 'sumAllMice', nnow);
    fn = fullfile(d, namv);
    
    savefig(gcf, fn)
    
    % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)
end





%% old stuff

% th_hiCorr = .4; days_hi_corrDiff = find(corr_diffTun{im}>th_hiCorr); roc_inh_hiCorrDiff = [roc_inh_timeM1{im}{days_hi_corrDiff}]; size(roc_inh_hiCorrDiff), [mean(roc_inh_hiCorrDiff), std(roc_inh_hiCorrDiff)]
% th_loCorr = 0; days_lo_corrDiff = find(corr_diffTun{im}<th_loCorr); roc_inh_loCorrDiff = [roc_inh_timeM1{im}{days_lo_corrDiff}]; size(roc_inh_loCorrDiff), [mean(roc_inh_loCorrDiff), std(roc_inh_loCorrDiff)]

%{
doSame = 1; % if 1, compare tuning of E (and I) with low vs high corr_EI_same
            % if 0, compare tuning of E (and I) with low vs high corr_EI_diff

% 1st cell : corr_exci_inhc;
% 2nd cell : corr_excc_inhi;

rocNowlowCorr = cell(1, 2);
rocNowhiCorr = cell(1, 2);
% rocNow_allCorrBin = cell(1, 2);

for corrtype = 1:2
    %{
    if corrtype==1 
        if doSame % EiIi
            corr2an = corr_exci_inhi;
        else % EiIc
            corr2an = corr_exci_inhc;        
        end
        
    else 
        if doSame % EcIc
            corr2an = corr_excc_inhc;
        else % EcIi
            corr2an = corr_excc_inhi;
        end
        
    end
    %}

    %% Discretize corr in bins (ie hist of corr over days)

    % Set the histogram vars across days (each day has one corr value ...
    % find corr_EI_ic (or corr_EI_ci) of each day falls in what bin of
    % correlation)


    % bins_allMice = cell(1,length(mice));
%     i_bin_corr_allMice = cell(1,length(mice));

    for im = 1:length(mice)

%         ally = corr2an{im};
        % get the counts in each bin
        %{
        [n0, e, i] = histcounts(ally(:), bins);
        i_bin_corr_allMice{im} = i;
        %}
        
        thLow = .18;
        thHi = .2;
        % high corr_same
        i_bin_hiCorr_EiIi_allMice{im} = corr_exci_inhi{im} >= thHi; % | (0<=corr_exci_inhi{im} & corr_exci_inhi{im}<thLow)]
        i_bin_hiCorr_EcIc_allMice{im} = corr_excc_inhc{im} >= thHi;
        % pool
        i_bin_hiCorr_EI_same_allMice{im} = i_bin_hiCorr_EiIi_allMice{im} + i_bin_hiCorr_EcIc_allMice{im};
        % low corr_same
        i_bin_loCorr_EiIi_allMice{im} = (0<=corr_exci_inhi{im} & corr_exci_inhi{im}<thLow);
        i_bin_loCorr_EcIc_allMice{im} = (0<=corr_excc_inhc{im} & corr_excc_inhc{im}<thLow);
        % pool
        i_bin_loCorr_EI_same_allMice{im} = i_bin_loCorr_EiIi_allMice{im} + i_bin_loCorr_EcIc_allMice{im};        
        
        % low corr_diff
        i_bin_loCorr_EiIc_allMice{im} = (0<=corr_exci_inhc{im} & corr_exci_inhc{im}<thLow);
        i_bin_loCorr_EcIi_allMice{im} = (0<=corr_excc_inhi{im} & corr_excc_inhi{im}<thLow);
        % pool
        i_bin_loCorr_EI_diff_allMice{im} = i_bin_loCorr_EiIc_allMice{im} + i_bin_loCorr_EcIi_allMice{im};
        % high corr_diff
        i_bin_hiCorr_EiIc_allMice{im} = (corr_exci_inhc{im}>=thHi);
        i_bin_hiCorr_EcIi_allMice{im} = (corr_excc_inhi{im}>=thHi);        
        % pool
        i_bin_hiCorr_EI_diff_allMice{im} = i_bin_hiCorr_EiIc_allMice{im} + i_bin_hiCorr_EcIi_allMice{im};
        
        
        %%%%% Plotting
        %{
        mn = min(ally);
        mx = max(ally);    
        % set the bins
        r1 = round(mn-.05,1); 
        r2 = round(mx+.05,1);
        bins = r1 : (r2-r1)/10 : r2;
        bins_allMice{im} = bins;
        %}        
        %{
        % turn counts to fractions (of total elements that exist in each bin) % this is the y for plotting hists
        ye = n0/sum(n0); %     ye = smooth(ye);
        ye = n0;

        % set x for plotting hists as the center of bins
        x = mode(diff(bins))/2 + bins; x = x(1:end-1);

        %%%% 
        figure; plot(x, ye)
        %}
    end

    for im = 1:length(mice)
        [sum(i_bin_hiCorr_EI_same_allMice{im}>=1 & i_bin_loCorr_EI_diff_allMice{im}>=1), ... % neurons in these days should have high tuning
        sum(i_bin_loCorr_EI_same_allMice{im}>=1 & i_bin_hiCorr_EI_diff_allMice{im}>=1)] % neurons in these days should have low tuning
    end

    bins = [0, thLow, thHi];
    
    
    %% Set ROC of inh neurons whose corr_exci_inhc (computed on the population, so one value per day) is in a particular bin of corrs

    rocExc_corrbin_now = cell(1, length(mice));
    rocInh_corrbin_now = cell(1, length(mice));
    
    for im = 1:length(mice)
        rocExc_corrbin_now{im} = cell(1, length(bins)-1);
        rocInh_corrbin_now{im} = cell(1, length(bins)-1);        
        
        for ii = 1:2  % unique(i_bin_corr_allMice{im}(i_bin_corr_allMice{im}>0))' % days with their corr_exci_inhc in bin number ii
            
%             idaysNow = i_bin_corr_allMice{im}==ii;
            
            if ii==2
                idaysNow = (i_bin_hiCorr_EI_same_allMice{im}>=1 & i_bin_loCorr_EI_diff_allMice{im}>=1);
            elseif ii==1
                idaysNow = (i_bin_loCorr_EI_same_allMice{im}>=1 & i_bin_hiCorr_EI_diff_allMice{im}>=1);
            end
            
            % compute tuning, ie abs difference from chance (.5)... if we
            % want to pool ipsi and contra tuning we have to do this... bc
            % for contra roc values get lower (from .5) as tuning increases
            rocExcCorrDays = abs(.5-[rocExc{im}{idaysNow}]);
            rocInhCorrDays = abs(.5-[rocInh{im}{idaysNow}]); % ROC of all inh neurons whose population corr (on that session) is in bin ii.            
            
            % now get ROC only for ipsi or contra neurons (whose population corr (ie corr for their session) is in bin ii)
            if corrtype==1 % EiIc
                rocNowE = rocExcCorrDays([(exc_ipsi_timeM1{im}{idaysNow})]); 
                if doSame % EiIi
                    rocNowI = rocInhCorrDays([(inh_ipsi_timeM1{im}{idaysNow})]); 
                else
                    rocNowI = rocInhCorrDays([(inh_contra_timeM1{im}{idaysNow})]); 
                end                
                
            else % EcIi
                rocNowE = rocExcCorrDays([(exc_contra_timeM1{im}{idaysNow})]); 
                if doSame
                    rocNowI = rocInhCorrDays([(inh_contra_timeM1{im}{idaysNow})]); 
                else
                    rocNowI = rocInhCorrDays([(inh_ipsi_timeM1{im}{idaysNow})]); 
                end                
            end
            
            rocExc_corrbin_now{im}{ii} = rocNowE; % roc value of all ipsi (or contra depending on corrtype) inh neurons in days whose corr_exci_inhc is in bin ii
            rocInh_corrbin_now{im}{ii} = rocNowI;
        end
    end

    % number of neurons per bin of corr
    %{
    numInh_corrBin = nan(length(bins)-1, length(mice));
    for im = 1:length(mice)
        numInh_corrBin(:,im) = cellfun(@length, rocInh_corrbin_now{im});
    end
    %}

    
    %%
    if corrtype==1 % Ei
        roc_allCorrBin_Ei = rocExc_corrbin_now;
        if doSame
            roc_allCorrBin_Ii = rocInh_corrbin_now;    
        else
            roc_allCorrBin_Ic = rocInh_corrbin_now;
        end
        
    else % Ec
        roc_allCorrBin_Ec = rocExc_corrbin_now;        
        if doSame
            roc_allCorrBin_Ic = rocInh_corrbin_now;        
        else
            roc_allCorrBin_Ii = rocInh_corrbin_now;        
        end
    end

    
    %% Get ROC vals for the first and last valid bins of corr (not necessarily the same bins of corr for all mice)
    %{
    lowCorrNow_rocInhTimeM1 = cell(1, length(mice));
    hiCorrNow_rocInhTimeM1 = cell(1, length(mice));

    for im = 1:length(mice)
        % first non-zero corr bin (ie lowest corr)
        inow = find(numInh_corrBin(:,im)>0 , 1); 
        lowCorrNow_rocInhTimeM1{im} = rocInh_corrbin_now{im}{inow}; % get roc val for the lowest bin of corr       

        % last non-zero corr bin (ie highest corr)
        inow = find(numInh_corrBin(:,im)>0 , 1, 'last'); 
        hiCorrNow_rocInhTimeM1{im} = rocInh_corrbin_now{im}{inow}; % get roc val for the lowest bin of corr       

    end


    %%
    %{
    for im = 1:length(mice)
        fh = figure('name', mice{im}); 
        y1 = lowCorrNow_rocInhTimeM1{im};
        y2 = hiCorrNow_rocInhTimeM1{im};

        fh = plotHist(y1,y2,xlab,ylab,leg, cols, [], fh); 
    end
    %}

    % so clear that inh neurons with lower corr_exci_inhc have higher roc
    % values (compared to those with high corr_exci_inhc values).... these are the ones that are more strongly tuned, hence they
    % should be less correlated with exc neurons with the opposite preferance.


    %%
    
    if corrtype==1 % EiIc
        rocNowlowCorr_EiIc = lowCorrNow_rocInhTimeM1;
        rocNowhiCorr_EiIc = hiCorrNow_rocInhTimeM1;
        
    else % EcIi
        rocNowlowCorr_EcIi = lowCorrNow_rocInhTimeM1;
        rocNowhiCorr_EiIi = hiCorrNow_rocInhTimeM1;
    end
    %}

end


%% average ROC across neurons for each corr bin

rocAve_allCorrBin_Ic = nan(length(bins)-1, length(mice));
rocAve_allCorrBin_Ii = nan(length(bins)-1, length(mice));
rocAve_allCorrBin_Ei = nan(length(bins)-1, length(mice));
rocAve_allCorrBin_Ec = nan(length(bins)-1, length(mice));

for im = 1:length(mice)
    rocAve_allCorrBin_Ic(:,im) = cellfun(@mean, roc_allCorrBin_Ic{im});
    rocAve_allCorrBin_Ii(:,im) = cellfun(@mean, roc_allCorrBin_Ii{im});
    
    rocAve_allCorrBin_Ei(:,im) = cellfun(@mean, roc_allCorrBin_Ei{im});
    rocAve_allCorrBin_Ec(:,im) = cellfun(@mean, roc_allCorrBin_Ec{im});   
end



%% Plot averages

x = bins(1:end-1) + mode(diff(bins))/2;

figure; plot(x, rocAve_allCorrBin_Ec, 'o-'), title('Ec')
figure; plot(x, rocAve_allCorrBin_Ei, 'o-'), title('Ei')
figure; plot(x, rocAve_allCorrBin_Ic, 'o-'), title('Ic')
figure; plot(x, rocAve_allCorrBin_Ii, 'o-'), title('Ii')


%% Now pool ROC of exc (or inh) neurons across ipsi and contra preferring... we can only do this if we turned ROC values to deviation from .5 (chance) in the above section.

rocAve_allCorrBin_Iic = nan(length(bins)-1, length(mice));
rocAve_allCorrBin_Eic = nan(length(bins)-1, length(mice));

for im = 1:length(mice)
    for ibin = 1:length(bins)-1
        rocAve_allCorrBin_Iic(ibin,im) = mean([roc_allCorrBin_Ii{im}{ibin}, roc_allCorrBin_Ic{im}{ibin}]);
        rocAve_allCorrBin_Eic(ibin,im) = mean([roc_allCorrBin_Ei{im}{ibin}, roc_allCorrBin_Ec{im}{ibin}]);
    end
end


figure; plot(x, rocAve_allCorrBin_Iic, 'o-'), title('I')
figure; plot(x, rocAve_allCorrBin_Eic, 'o-'), title('E')




%% Combine both corrs (corr_E_I_diff):
%{
nBins = 15;
doSmooth = 0;

for im = 1:length(mice)
    fh = figure('name', mice{im}); 
    y1 = [rocNowlowCorr_EiIc{im} , rocNowlowCorr_EcIi{im}];
    y2 = [rocNowhiCorr_EiIc{im} , rocNowhiCorr_EcIi{im}];
    [mean(y1), mean(y2)]
    fh = plotHist(y1,y2,xlab,ylab,leg, cols, [], fh, nBins, doSmooth); 
end
%}


%% Get average of ROC of all neurons for each corr_diff bin
%{
% corrtype = 1;
aveROCInh_corrBin = nan(length(bins)-1, length(mice));
for im = 1:length(mice)

    y = [roc_allCorrBin_Ic{im} ; roc_allCorrBin_Ii{im}];
%     ycat = rocNow_allCorrBin{1}{im};
%     ycat = rocNow_allCorrBin{2}{im};
    
    ycat = cell(1,size(y,2));
    for ibin = 1:size(y,2)
        ycat{ibin} = [y{:,ibin}];
    end

    aveROCInh_corrBin(:,im) = cellfun(@mean, ycat);
end
aveROCInh_corrBin


figure; plot(aveROCInh_corrBin)

%}

%}




