%%% copied from corr_excInh_plots, and modified. 
%%% make sure you set the vars as defined below for corr_excInh_setVars and
%%% choicePref_ROC_exc_inh_plots_setVars

% First run corr_excInh_setVars to get FRs
%     alFR = 'chAl'; % 'initAl'; % the firing rate traces were aligned on what
%   OR
%     alFR = 'initAl';
% outcome2ana = 'corr';
%
% Then run choicePref_ROC_exc_inh_plots_setVars to get ROCs (to set ipsi, contra tuned neurons) 
%     outcome2ana = 'corr'; %''; % 'corr'; 'incorr'; '';
%     doChoicePref = 2; 
% Then run this script


%% Set the following vars

nBinsMedFrs = 50; %100; % if nan,dont bin FRs, use all neurons; otherwise only use those inh and exc (for computing corr between all exc and all inh) whose FR is within a small bin. This will not affect corrs of same/diff tuning... it is only for corrs of all neurons.
saveFigs = 1; 
doPlots = 1;
if strcmp(alFR,'chAl') % only if alFR is chAl, set the 2 vars below:
    cprintf('r', 'MAKE SURE you set the 2 vars below!\n')
end
subMeanResp = 1; %1; % For each neuron, mean response across ipsi(contra) trials will be subtracted from the response to each ipsi (contra) trial, so using all trials for computing correlations
ipsi_contra_all_which = [1,0,0]; %[0,1,0]; %  if subMeanResp is 0, and alFR is chAl, we need to analyze either ipsi or contra trials, to make sure neural correlations are not due to the choice.


% What time point to use for setting ipsi, contra tuning (fr2an) and for computing firing rate correlations (fr2an_FR)
fr2an = nPreMin; % for ROC, timebin before the choice
fr2an_FR = nPreMin_fr; % for FR, timebin before the initiation tone (assuming corr_excInh_setVars was run with alFR='initAl') % in case FR traces are aligned on a different event than ROC traces.
time2an = 'timeM1'; % time minus 1, meaning timeBin before whatever the alignment was on

th = .5; % threshold for identifying ipsi-preferred vs contra preferred neurons. % ipsi auc will be above 0.5


%%
dirn0fr = '/home/farznaj/Dropbox/ChurchlandLab/Projects/inhExcDecisionMaking/FR';

if strcmp(alFR,'initAl') % if FRs are aligned on initiTone, we will set subMeanResp to 0. 
    cprintf('blue', 'FRs are aligned on initTone, so using all trials for computing correlations\n')
    subMeanResp = 0;
    ipsi_contra_all = [0,0,1]; % what trials to use: ipsi, contra, all        
    nnow = 'FRinitAl_ROCchAl';
    
elseif strcmp(alFR,'chAl') %fr2an==fr2an_FR % FRs are choice aligned like ROC... so lets make sure we look at only one type of trial
    if subMeanResp
        cprintf('blue', 'For each neuron, mean response across ipsi(contra) trials will be subtracted from the response to each ipsi (contra) trial, so using all trials for computing correlations\n')
        ipsi_contra_all = [0,0,1]; %[1,0,0]; % what trials to use: ipsi, contra, all
        nnow = 'FRchAlMeanSub_ROCchAl';
        
    else
        cprintf('r', 'FRs are aligned on choice, so using only ipsi or only contra trials for computing correlations!\n')
        ipsi_contra_all = ipsi_contra_all_which;
        if ipsi_contra_all(1)==1
            nnow = 'FRchAlOnlyIpsi_ROCchAl';
        elseif ipsi_contra_all(2)==1
            nnow = 'FRchAlOnlyContra_ROCchAl';
        end
    end
end


if strcmp(outcome2ana, 'corr')
    o2a = '_corr'; 
elseif strcmp(outcome2ana, 'incorr')
    o2a = '_incorr';  
else
    o2a = '_allOutcome';
end  

cols = {'k', 'r'}; % same, opposite tuning : this is color for plots of corr between neurons with the same vs opposite tuning.
linestylesShfl = {':',':'};
linestylesData = {'-','-'};
nowStr = datestr(now, 'yymmdd-HHMMSS');


%%
roc_exc_timeM1 = cell(1, length(mice));
roc_inh_timeM1 = cell(1, length(mice));

fr_exc_timeM1 = cell(1, length(mice));
fr_inh_timeM1 = cell(1, length(mice));

roc_exc_timeM1_shfl = cell(1, length(mice));
roc_inh_timeM1_shfl = cell(1, length(mice));

for im = 1:length(mice)
    roc_exc_timeM1{im} = cell(numDaysAll(im),1);
    roc_inh_timeM1{im} = cell(numDaysAll(im),1);
    exc_ipsi_timeM1{im} = cell(numDaysAll(im),1);
    inh_ipsi_timeM1{im} = cell(numDaysAll(im),1);    
    exc_contra_timeM1{im} = cell(numDaysAll(im),1);
    inh_contra_timeM1{im} = cell(numDaysAll(im),1);        
    
    for iday = 1:numDaysAll(im)
        if mnTrNum_allMice{im}(iday) >= thMinTrs
            % ROC values (% ROCs at timebin -1 for all neurons, for each day of each mouse)
            roc_exc_timeM1{im}{iday} = choicePref_exc_al_allMice{im}{iday}(fr2an,:); % nNeurons
            roc_inh_timeM1{im}{iday} = choicePref_inh_al_allMice{im}{iday}(fr2an,:);
            
            roc_exc_timeM1_shfl{im}{iday} = choicePref_exc_al_allMice_shfl{im}{iday}(fr2an,:); % nNeurons
            roc_inh_timeM1_shfl{im}{iday} = choicePref_inh_al_allMice_shfl{im}{iday}(fr2an,:);
            

            % Firing rate of E_ipsi, etc neurons at timebin -1
            frE = squeeze(fr_exc_al_allMice{im}{iday}(fr2an_FR,:,:)); % exc units x trials
            frI = squeeze(fr_inh_al_allMice{im}{iday}(fr2an_FR,:,:)); % inh units x trials


            fr_exc_timeM1{im}{iday} = frE; % numE_ipsi x numTrs        
            fr_inh_timeM1{im}{iday} = frI;
            
        end
    end
end


%% Compute correlatin between paris of neurons

roc_exc_timeM1_sameFR = cell(1, length(mice));
roc_inh_timeM1_sameFR = cell(1, length(mice));
roc_exc_timeM1_sameFR_shfl = cell(1, length(mice));
roc_inh_timeM1_sameFR_shfl = cell(1, length(mice));

for im = 1:length(mice)
    for iday = 1:numDaysAll(im)
        if mnTrNum_allMice{im}(iday) >= thMinTrs
            
            if ipsi_contra_all(1)==1
                trs = ipsiTrs_allDays_allMice{im}{iday};
            elseif ipsi_contra_all(2)==1
                trs = contraTrs_allDays_allMice{im}{iday};
            elseif ipsi_contra_all(3)==1
                trs = true(1, size(fr_exc_timeM1{im}{iday}, 2)); %1:length(ipsiTrs_allDays_allMice{im}{iday});
            end

            if isnan(nBinsMedFrs) % dont bin FRs, use all neurons; otherwise only use those inh and exc (for computing corr between all exc and all inh) whose FR is within a small bin 
                
                ns_e = ones(size(fr_exc_timeM1{im}{iday},1),1);
                ns_i = ones(size(fr_inh_timeM1{im}{iday},1),1);

            else
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%
                e = fr_exc_timeM1{im}{iday}(:,trs); % ns x trs
                i = fr_inh_timeM1{im}{iday}(:,trs); % ns x trs


                %% Bin median of FRs

                y1 = median(e, 2);
                y2 = median(i, 2);

                ally = [y1(:);y2(:)];
                r1 = min(ally);
                r2 = max(ally);
                bins = r1 : (r2-r1)/nBinsMedFrs : r2;

                [nexc, be, ie] = histcounts(y1(:), bins);
                [ninh, bi, ii] = histcounts(y2(:), bins);

                ye = nexc/sum(nexc);
                yi = ninh/sum(ninh);


                %% Get the inh FR bin with most neurons in it, and use those neurons to compute correlations

                if subMeanResp % yi is negative, and smaller values are larger ... we want to pick the value close to 0 because that one has more exc neurons
                    imi = find(yi==max(yi),1,'last');
                else
    %                 imi = find(yi==max(yi),1,'first');
                    % [me,ime] = max(ye); 
                    [mi,imi] = max(yi);
                    % disp([me, mi])
                    % disp([ime, imi])
                end

                % excIndsToUse = (ie==ime); 
                excIndsToUse = (ie==imi); 
                inhIndsToUse = (ii==imi);

                binFRtoUse = bins(imi:imi+1);    % this range of FRs will be used.

                disp([imi, sum(excIndsToUse), sum(inhIndsToUse)])


                ns_e = excIndsToUse;
                ns_i = inhIndsToUse;


                %%
                if iday==1
                    doSmooth = 0;
                    fign = figure;
                    sp = subplot(1,1,1);
                    cols = {'b','r'};

                    x = mode(diff(bins))/2 + bins; x = x(1:end-1);

                    if doSmooth
                        ye = smooth(ye,5);
                        yi = smooth(yi,5);
                    end

                    figure(fign)
                    hold(sp,'on')
                    h1 = plot(sp, x, ye, 'color', cols{1});
                    h2 = plot(sp, x, yi, 'color', cols{2});
                end
            end
            
            
            %% Get ROC values at time -1 only for those exc and inh neurons whose FR is in the same bin (ie they have similiar FRs)
            
            roc_exc_timeM1_sameFR{im}{iday} = roc_exc_timeM1{im}{iday}(ns_e);
            roc_inh_timeM1_sameFR{im}{iday} = roc_inh_timeM1{im}{iday}(ns_i);
            
            roc_exc_timeM1_sameFR_shfl{im}{iday} = roc_exc_timeM1_shfl{im}{iday}(ns_e);
            roc_inh_timeM1_sameFR_shfl{im}{iday} = roc_inh_timeM1_shfl{im}{iday}(ns_i);
            
        end
    end
end



%% Average ROC value at time -1 across all neurons (only exc, inh neurons with similar FRs are included)

aveexc_sameFR_aveDays = nan(1, length(mice));
aveinh_sameFR_aveDays = nan(1, length(mice));
aveexc_sameFR_seDays = nan(1, length(mice));
aveinh_sameFR_seDays = nan(1, length(mice));

aveexc_sameFR_aveDays_shfl = nan(1, length(mice));
aveinh_sameFR_aveDays_shfl = nan(1, length(mice));
aveexc_sameFR_seDays_shfl = nan(1, length(mice));
aveinh_sameFR_seDays_shfl = nan(1, length(mice));

pallm_days = nan(1, length(mice));
hallm_days = nan(1, length(mice));

for im = 1:length(mice)
    % average across neurons for each day
    aveexc_sameFR = cellfun(@(x)mean(x,2), roc_exc_timeM1_sameFR{im}, 'uniformoutput',0); % average across neurons
    aveinh_sameFR = cellfun(@(x)mean(x,2), roc_inh_timeM1_sameFR{im}, 'uniformoutput',0); % average across neurons

    aveexc_sameFR = cell2mat(aveexc_sameFR(cellfun(@length, aveexc_sameFR)==1)); % number of days (after removing 0-size days)
    aveinh_sameFR = cell2mat(aveinh_sameFR(cellfun(@length, aveinh_sameFR)==1));

    % average across days
    aveexc_sameFR_aveDays(im) = mean(aveexc_sameFR);
    aveinh_sameFR_aveDays(im) = mean(aveinh_sameFR);

    % se across days
    aveexc_sameFR_seDays(im) = std(aveexc_sameFR, [], 2) / sqrt(numDaysGood(im));
    aveinh_sameFR_seDays(im) = std(aveinh_sameFR, [], 2) / sqrt(numDaysGood(im));
    
    
    %%%%% shfl
    % average across neurons for each day
    aveexc_sameFR_shfl = cellfun(@(x)mean(x,2), roc_exc_timeM1_sameFR_shfl{im}, 'uniformoutput',0); % average across neurons
    aveinh_sameFR_shfl = cellfun(@(x)mean(x,2), roc_inh_timeM1_sameFR_shfl{im}, 'uniformoutput',0); % average across neurons

    aveexc_sameFR_shfl = cell2mat(aveexc_sameFR_shfl(cellfun(@length, aveexc_sameFR_shfl)==1)); % number of days (after removing 0-size days)
    aveinh_sameFR_shfl = cell2mat(aveinh_sameFR_shfl(cellfun(@length, aveinh_sameFR_shfl)==1));

    % average across days
    aveexc_sameFR_aveDays_shfl(im) = mean(aveexc_sameFR_shfl);
    aveinh_sameFR_aveDays_shfl(im) = mean(aveinh_sameFR_shfl);

    % se across days
    aveexc_sameFR_seDays_shfl(im) = std(aveexc_sameFR_shfl, [], 2) / sqrt(numDaysGood(im));
    aveinh_sameFR_seDays_shfl(im) = std(aveinh_sameFR_shfl, [], 2) / sqrt(numDaysGood(im));    


    %%%%%%%%% ttest for each mouse, exc vs inh across days
    [h,p0] = ttest2(aveexc_sameFR', aveinh_sameFR'); % 1 x nFrs
    pallm_days(:,im) = p0;
    hallm_days(:,im) = h;
end

hh_days = hallm_days;
hh_days(hallm_days==0) = nan;



%% %%%%%%%%%%%%%%% Plot

colss = [0,.8,.8; .8,.5,.8]; % exc,inh colors for shuffled data
dirn00 = fullfile(dirn0, 'allMice');
if strcmp(outcome2ana, '')
    dirn00 = fullfile(dirn00, 'allOutcome');
elseif strcmp(outcome2ana, 'corr')
    dirn00 = fullfile(dirn00, 'corr');
elseif strcmp(outcome2ana, 'incorr')
    dirn00 = fullfile(dirn00, 'incorr');
end
if doshfl
    dirn00 = fullfile(dirn00, 'shuffled_actual_ROC');
end
if ~exist(dirn00, 'dir')
    mkdir(dirn00)
end


%%
eav = aveexc_sameFR_aveDays;
ese = aveexc_sameFR_seDays;
iav = aveinh_sameFR_aveDays;
ise = aveinh_sameFR_seDays;

eavs = aveexc_sameFR_aveDays_shfl;
eses = aveexc_sameFR_seDays_shfl;
iavs = aveinh_sameFR_aveDays_shfl;
ises = aveinh_sameFR_seDays_shfl;

g = 0;
fign = figure('position',[60   593   450   320]); 

subplot(121), hold on;
% errorbar(1:length(mice), uav, use, 'g', 'linestyle','none', 'marker','o')
errorbar(1:length(mice), eav, ese, 'b', 'linestyle','none', 'marker','o')
errorbar((1:length(mice))+g, iav, ise ,'r', 'linestyle','none', 'marker','o')
if doshfl % shfl
%     errorbar(1:length(mice), uavs, uses, 'color', [.2,.2,.2], 'linestyle','none', 'marker','o')
    errorbar(1:length(mice), eavs, eses, 'color', colss(1,:), 'linestyle','none', 'marker','o')
    errorbar((1:length(mice))+g, iavs, ises ,'color', colss(2,:), 'linestyle','none', 'marker','o')
end
xlim([.5, length(mice)+.5])
xlabel('Mice')
ylab = simpleTokenize(namc, '_'); ylab = ylab{1}; %namc;   
ylabel(ylab)
set(gca,'tickdir','out')
set(gca,'xtick',1:length(mice))
yl = get(gca,'ylim');
% ym = yl(2)-range(yl)/20;
ym = iav + ise + range(yl)/20;
plot(1:length(mice), hh_days.*ym, 'k*')
title('aveDays,aveNs')
legend('exc', 'inh')
% legend('unsure','exc', 'inh')



% save figure
if saveFigs        
    fdn = fullfile(dirn00, strcat(namc,'_','ROC_curr_chAl_excInh_sameFR_aveSe_time-1_eachMouse_', dm0, nowStr));
    savefig(fign, fdn)    
    print(fign, '-dpdf', fdn)
end 



