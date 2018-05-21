% First run corr_excInh_setVars to get FRs
%     alFR = 'chAl'; % 'initAl'; % the firing rate traces were aligned on what
%   OR
%     alFR = 'initAl';
%     outcome2ana = '';
%
% Then run choicePref_ROC_exc_inh_plots_setVars to get ROCs (to set ipsi, contra tuned neurons) 
%     outcome2ana = 'corr'; %''; % 'corr'; 'incorr'; '';
%     doChoicePref = 0; %2; 
% Then run this script

% E_ipsi and I_ipsi have ROC > .5 (right before the choice), we want to see
% if E_ipsi activates I_ipsi; subsequently I_ipsi will inhibit E_contra  
% so we want to see if E_ipsi is correlated with I_ipsi
% and perhaps if I_ipsi is anti-correlated with E_contra...

%%% Correlations are computed in 4 different ways:
%{
rAve_EI_same --> average of r(N1-allN)  

r_EI_same_allMice --> pw corrs without any averaging

r_EI_same_aveDays --> pw corrs are averaged for each day (for each population)

rPop_EI_same_allMice --> corr btwn pop-averaged FRs
%}


%% Set the following vars

saveFigs = 0; 
doPlots = 0;
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

exc_ipsi_timeM1 = cell(1, length(mice));
inh_ipsi_timeM1 = cell(1, length(mice));
exc_contra_timeM1 = cell(1, length(mice));
inh_contra_timeM1 = cell(1, length(mice));

fr_exc_ipsi_timeM1 = cell(1, length(mice));
fr_inh_ipsi_timeM1 = cell(1, length(mice));
fr_exc_contra_timeM1 = cell(1, length(mice));
fr_inh_contra_timeM1 = cell(1, length(mice));

fr_exc_ipsi_timeM1_aveNs = cell(1, length(mice));
fr_inh_ipsi_timeM1_aveNs = cell(1, length(mice));
fr_exc_contra_timeM1_aveNs = cell(1, length(mice));
fr_inh_contra_timeM1_aveNs = cell(1, length(mice));

fr_exc_timeM1 = cell(1, length(mice));
fr_inh_timeM1 = cell(1, length(mice));

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

            % is neuron ipsi or contra preferred? (if roc==.5, don't assign the neuron to either class)
            % exc_ipsi refers to ipsi-preferring excitatory neurons.
            % Ipsi-preferring is determined based on ROC values right
            % before the choice. [Not to be confused with trials in which
            % animal made the left choice.]
            exc_ipsi_timeM1{im}{iday} = choicePref_exc_al_allMice{im}{iday}(fr2an,:) > th;
            inh_ipsi_timeM1{im}{iday} = choicePref_inh_al_allMice{im}{iday}(fr2an,:) > th;

            exc_contra_timeM1{im}{iday} = choicePref_exc_al_allMice{im}{iday}(fr2an,:) < th;
            inh_contra_timeM1{im}{iday} = choicePref_inh_al_allMice{im}{iday}(fr2an,:) < th;                


            % Firing rate of E_ipsi, etc neurons at timebin -1
            frE = squeeze(fr_exc_al_allMice{im}{iday}(fr2an_FR,:,:)); % exc units x trials
            frI = squeeze(fr_inh_al_allMice{im}{iday}(fr2an_FR,:,:)); % inh units x trials

            if subMeanResp
                % subtract N resp averaged across all ipsiTrs from N resp to each
                % individual ipsi trial
                % Exc, ipsiTrs
                frE_ipsiTrs = frE(:,ipsiTrs_allDays_allMice{im}{iday});
                frE_ipsiTrsAve = mean(frE_ipsiTrs,2);
                frE_ipsiTrs_minusAveIpsi = frE_ipsiTrs - frE_ipsiTrsAve;
                % Exc, contraTrs
                frE_contraTrs = frE(:,contraTrs_allDays_allMice{im}{iday});
                frE_contraTrsAve = mean(frE_contraTrs,2);
                frE_contraTrs_minusAveIpsi = frE_contraTrs - frE_contraTrsAve;        
                % Exc, pool ipsi and contra trials
                frE = [frE_ipsiTrs_minusAveIpsi, frE_contraTrs_minusAveIpsi];

                % Inh, ipsiTrs
                frI_ipsiTrs = frI(:,ipsiTrs_allDays_allMice{im}{iday});
                frI_ipsiTrsAve = mean(frI_ipsiTrs,2);
                frI_ipsiTrs_minusAveIpsi = frI_ipsiTrs - frI_ipsiTrsAve;        
                % Inh, contraTrs
                frI_contraTrs = frI(:,contraTrs_allDays_allMice{im}{iday});
                frI_contraTrsAve = mean(frI_contraTrs,2);
                frI_contraTrs_minusAveIpsi = frI_contraTrs - frI_contraTrsAve;
                % Inh, pool ipsi and contra trials
                frI = [frI_ipsiTrs_minusAveIpsi, frI_contraTrs_minusAveIpsi];        
            end

            %
            fr_exc_ipsi_timeM1{im}{iday} = frE(exc_ipsi_timeM1{im}{iday}, :); % numE_ipsi x numTrs        
            fr_inh_ipsi_timeM1{im}{iday} = frI(inh_ipsi_timeM1{im}{iday}, :);

            fr_exc_contra_timeM1{im}{iday} = frE(exc_contra_timeM1{im}{iday}, :); % numE_contra x numTrs        
            fr_inh_contra_timeM1{im}{iday} = frI(inh_contra_timeM1{im}{iday}, :);

            fr_exc_timeM1{im}{iday} = frE; % numE_ipsi x numTrs        
            fr_inh_timeM1{im}{iday} = frI;
            
            % Average FRs across neurons (for each trials)
            fr_exc_ipsi_timeM1_aveNs{im}{iday} = mean(fr_exc_ipsi_timeM1{im}{iday},1); % numTrs
            fr_inh_ipsi_timeM1_aveNs{im}{iday} = mean(fr_inh_ipsi_timeM1{im}{iday},1);

            fr_exc_contra_timeM1_aveNs{im}{iday} = mean(fr_exc_contra_timeM1{im}{iday},1);
            fr_inh_contra_timeM1_aveNs{im}{iday} = mean(fr_inh_contra_timeM1{im}{iday},1);
        end
    end
end




%% Show number of neurons in each category (based on their ROC value in the time bin before the choice)

num_exc_ipsi_timeM1 = cell(1, length(mice));
num_inh_ipsi_timeM1 = cell(1, length(mice));
num_exc_contra_timeM1 = cell(1, length(mice));
num_inh_contra_timeM1 = cell(1, length(mice));

for im = 1:length(mice)
    for iday = 1:numDaysAll(im)
        num_exc_ipsi_timeM1{im}(iday) = sum(exc_ipsi_timeM1{im}{iday});
        num_inh_ipsi_timeM1{im}(iday) = sum(inh_ipsi_timeM1{im}{iday});

        num_exc_contra_timeM1{im}(iday) = sum(exc_contra_timeM1{im}{iday});
        num_inh_contra_timeM1{im}(iday) = sum(inh_contra_timeM1{im}{iday});
    end
end

% number of Ei, etc for each mouse
num_Ei = cellfun(@sum, num_exc_ipsi_timeM1);
num_Ec = cellfun(@sum, num_exc_contra_timeM1);
num_Ii = cellfun(@sum, num_inh_ipsi_timeM1);
num_Ic = cellfun(@sum, num_inh_contra_timeM1);

num_Ei_Ec_eachMouse = [num_Ei; num_Ec]
num_Ii_Ic_eachMouse = [num_Ii; num_Ic]

fract_Ei_Ec = [num_Ei; num_Ec] ./ sum(num_Ei_Ec_eachMouse,1);
fract_Ii_Ic = [num_Ii; num_Ic] ./ sum(num_Ii_Ic_eachMouse,1);


% set fraction of ipsi and contra trials for each mouse
numIpsiTrs = cellfun(@(x) sum(vertcat(x{:})), ipsiTrs_allDays_allMice); %, 'uniformoutput', 0);
numContraTrs = cellfun(@(x) sum(vertcat(x{:})), contraTrs_allDays_allMice); %, 'uniformoutput', 0);

FractIpsiTrs = numIpsiTrs ./ sum([numIpsiTrs ; numContraTrs],1);
FractContraTrs = numContraTrs ./ sum([numIpsiTrs ; numContraTrs],1);


%% Plot fraction of exc and inh neurons that are ipsi or contra ; also fraction of trials that are ipsi or contra

if doPlots
    figure; %('position', [1   311   336   420]); 

    subplot(221); hold on; 
    plot(fract_Ei_Ec(1,:), 'k-') % 'color', 'k', 'marker', '.') 
    plot(fract_Ei_Ec(2,:), 'k--') %'color', rgb('gray'), 'marker', '.')
    xlim([.8 4.2])
    set(gca,'xtick', 1:length(mice))
    set(gca,'xticklabel', mice)
    legend('Ipsi','Contra')
    ylabel('Fract neurons')
    title('Exc')

    subplot(223); hold on; 
    plot(fract_Ii_Ic(1,:), 'r-') %'color', 'r', 'marker', '.') 
    plot(fract_Ii_Ic(2,:), 'r--') %'color', rgb('lightsalmon'), 'marker', '.')
    xlim([.8 4.2])
    set(gca,'xtick', 1:length(mice))
    set(gca,'xticklabel', mice)
    legend('Ipsi','Contra')
    ylabel('Fract neurons')
    title('Inh')

    subplot(222); hold on; 
    plot(FractIpsiTrs, 'k-') %'color', 'r', 'marker', '.') 
    plot(FractContraTrs, 'k--') %'color', rgb('lightsalmon'), 'marker', '.')
    xlim([.8 4.2])
    set(gca,'xtick', 1:length(mice))
    set(gca,'xticklabel', mice)
    legend('Ipsi','Contra')
    ylabel('Fract trials')
    title('trials')


    %%%%%% num ipsi - num contra ... hist across days
    %{
    for im = 1:length(mice)
        figure; 

        subplot(211); hold on % exc: ipsi - contra
        histogram(num_exc_ipsi_timeM1{im} - num_exc_contra_timeM1{im})

        subplot(212); hold on % inh: ipsi - contra
        histogram(num_inh_ipsi_timeM1{im} - num_inh_contra_timeM1{im})

    end
    %}
end


%% Compute correlatin between paris of neurons

r_Ei_Ei = cell(1, length(mice));
r_Ec_Ec = cell(1, length(mice));
r_Ei_Ec = cell(1, length(mice));
r_Ii_Ii = cell(1, length(mice));
r_Ic_Ic = cell(1, length(mice));
r_Ii_Ic = cell(1, length(mice));
r_Ei_Ii = cell(1, length(mice));
r_Ec_Ic = cell(1, length(mice));
r_Ei_Ic = cell(1, length(mice));
r_Ec_Ii = cell(1, length(mice));
r_E_E = cell(1, length(mice));
r_I_I = cell(1, length(mice));
r_E_I = cell(1, length(mice));

p_Ei_Ei = cell(1, length(mice));
p_Ec_Ec = cell(1, length(mice));
p_Ei_Ec = cell(1, length(mice));
p_Ii_Ii = cell(1, length(mice));
p_Ic_Ic = cell(1, length(mice));
p_Ii_Ic = cell(1, length(mice));
p_Ei_Ii = cell(1, length(mice));
p_Ec_Ic = cell(1, length(mice));
p_Ei_Ic = cell(1, length(mice));
p_Ec_Ii = cell(1, length(mice));
p_E_E = cell(1, length(mice));
p_I_I = cell(1, length(mice));
p_E_I = cell(1, length(mice));

r_Ei_Ei_shfl = cell(1, length(mice));
r_Ec_Ec_shfl = cell(1, length(mice));
r_Ei_Ec_shfl = cell(1, length(mice));
r_Ii_Ii_shfl = cell(1, length(mice));
r_Ic_Ic_shfl = cell(1, length(mice));
r_Ii_Ic_shfl = cell(1, length(mice));
r_Ei_Ii_shfl = cell(1, length(mice));
r_Ec_Ic_shfl = cell(1, length(mice));
r_Ei_Ic_shfl = cell(1, length(mice));
r_Ec_Ii_shfl = cell(1, length(mice));
r_E_E_shfl = cell(1, length(mice));
r_I_I_shfl = cell(1, length(mice));
r_E_I_shfl = cell(1, length(mice));

for im = 1:length(mice)
    for iday = 1:numDaysAll(im)
        if mnTrNum_allMice{im}(iday) >= thMinTrs
            
            if ipsi_contra_all(1)==1
                trs = ipsiTrs_allDays_allMice{im}{iday};
            elseif ipsi_contra_all(2)==1
                trs = contraTrs_allDays_allMice{im}{iday};
            elseif ipsi_contra_all(3)==1
                trs = true(1, size(fr_exc_ipsi_timeM1{im}{iday}, 2)); %1:length(ipsiTrs_allDays_allMice{im}{iday});
            end

            %%%%% exc,exc: corr between similarly-tuned neurons
            [r_Ei_Ei{im}{iday}, p_Ei_Ei{im}{iday}] = corr(fr_exc_ipsi_timeM1{im}{iday}(:,trs)'); % ns x ns
            [r_Ec_Ec{im}{iday}, p_Ec_Ec{im}{iday}] = corr(fr_exc_contra_timeM1{im}{iday}(:,trs)'); 
            %%%%% exc,exc: corr between oppositely-tuned neurons
            [r_Ei_Ec{im}{iday}, p_Ei_Ec{im}{iday}] = corr(fr_exc_ipsi_timeM1{im}{iday}(:,trs)', fr_exc_contra_timeM1{im}{iday}(:,trs)'); % ns x ns
            %%%%% exc,exc: corr between all neurons
            [r_E_E{im}{iday}, p_E_E{im}{iday}] = corr(fr_exc_timeM1{im}{iday}(:,trs)', fr_exc_timeM1{im}{iday}(:,trs)'); % ns x ns

            %%%%% inh,inh: corr between similarly-tuned neurons
            [r_Ii_Ii{im}{iday}, p_Ii_Ii{im}{iday}] = corr(fr_inh_ipsi_timeM1{im}{iday}(:,trs)');
            [r_Ic_Ic{im}{iday}, p_Ic_Ic{im}{iday}] = corr(fr_inh_contra_timeM1{im}{iday}(:,trs)');
            %%%%% inh,inh: corr between oppositely-tuned neurons
            [r_Ii_Ic{im}{iday}, p_Ii_Ic{im}{iday}] = corr(fr_inh_ipsi_timeM1{im}{iday}(:,trs)', fr_inh_contra_timeM1{im}{iday}(:,trs)'); % ns x ns
            %%%%% inh,inh: corr between all neurons
            [r_I_I{im}{iday}, p_I_I{im}{iday}] = corr(fr_inh_timeM1{im}{iday}(:,trs)', fr_inh_timeM1{im}{iday}(:,trs)'); % ns x ns

            %%%%% exc,inh: corr between similarly-tuned neurons of exc and inh
            [r_Ei_Ii{im}{iday}, p_Ei_Ii{im}{iday}] = corr(fr_exc_ipsi_timeM1{im}{iday}(:,trs)', fr_inh_ipsi_timeM1{im}{iday}(:,trs)');
            [r_Ec_Ic{im}{iday}, p_Ec_Ic{im}{iday}] = corr(fr_exc_contra_timeM1{im}{iday}(:,trs)', fr_inh_contra_timeM1{im}{iday}(:,trs)');

            %%%%% exc,inh: corr between oppositely-tuned neurons of exc and inh
            [r_Ei_Ic{im}{iday}, p_Ei_Ic{im}{iday}] = corr(fr_exc_ipsi_timeM1{im}{iday}(:,trs)', fr_inh_contra_timeM1{im}{iday}(:,trs)');
            [r_Ec_Ii{im}{iday}, p_Ec_Ii{im}{iday}] = corr(fr_exc_contra_timeM1{im}{iday}(:,trs)', fr_inh_ipsi_timeM1{im}{iday}(:,trs)');
            %%%%% exc,inh: corr between all neurons
            [r_E_I{im}{iday}, p_E_I{im}{iday}] = corr(fr_exc_timeM1{im}{iday}(:,trs)', fr_inh_timeM1{im}{iday}(:,trs)'); % ns x ns
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%% shuffled: corr between neurons after shuffling trials %%%%%%%%%%%%%%%%%%%%%%%%%            
            %%%%% exc,exc: corr between similarly-tuned neurons
            ntrs = sum(trs);
            x = fr_exc_ipsi_timeM1{im}{iday}(:,trs)'; % trials x neurons            
            r_Ei_Ei_shfl{im}{iday} = corr(x(randperm(ntrs),:) , x(randperm(ntrs),:)); % ns x ns
            x = fr_exc_contra_timeM1{im}{iday}(:,trs)';
            r_Ec_Ec_shfl{im}{iday} = corr(x(randperm(ntrs),:) , x(randperm(ntrs),:));
            %%%%% exc,exc: corr between oppositely-tuned neurons
            x1 = fr_exc_ipsi_timeM1{im}{iday}(:,trs)';
            x2 = fr_exc_contra_timeM1{im}{iday}(:,trs)';
            r_Ei_Ec_shfl{im}{iday} = corr(x1(randperm(ntrs),:) , x2(randperm(ntrs),:)); % ns x ns
            %%%%% exc,exc: corr between all neurons
            x1 = fr_exc_timeM1{im}{iday}(:,trs)';
            x2 = fr_exc_timeM1{im}{iday}(:,trs)';
            r_E_E_shfl{im}{iday} = corr(x1(randperm(ntrs),:) , x2(randperm(ntrs),:)); % ns x ns

            %%%%% inh,inh: corr between similarly-tuned neurons
            x = fr_inh_ipsi_timeM1{im}{iday}(:,trs)';
            r_Ii_Ii_shfl{im}{iday} = corr(x(randperm(ntrs),:) , x(randperm(ntrs),:));
            x = fr_inh_contra_timeM1{im}{iday}(:,trs)';
            r_Ic_Ic_shfl{im}{iday} = corr(x(randperm(ntrs),:) , x(randperm(ntrs),:));
            %%%%% inh,inh: corr between oppositely-tuned neurons
            x1 = fr_inh_ipsi_timeM1{im}{iday}(:,trs)';
            x2 = fr_inh_contra_timeM1{im}{iday}(:,trs)';
            r_Ii_Ic_shfl{im}{iday} = corr(x1(randperm(ntrs),:) , x2(randperm(ntrs),:)); % ns x ns
            %%%%% inh,inh: corr between all neurons
            x1 = fr_inh_timeM1{im}{iday}(:,trs)';
            x2 = fr_inh_timeM1{im}{iday}(:,trs)';
            r_I_I_shfl{im}{iday} = corr(x1(randperm(ntrs),:) , x2(randperm(ntrs),:)); % ns x ns

            %%%%% exc,inh: corr between similarly-tuned neurons of exc and inh
            x1 = fr_exc_ipsi_timeM1{im}{iday}(:,trs)';
            x2 = fr_inh_ipsi_timeM1{im}{iday}(:,trs)';
            r_Ei_Ii_shfl{im}{iday} = corr(x1(randperm(ntrs),:) , x2(randperm(ntrs),:));
            x1 = fr_exc_contra_timeM1{im}{iday}(:,trs)';
            x2 = fr_inh_contra_timeM1{im}{iday}(:,trs)';
            r_Ec_Ic_shfl{im}{iday} = corr(x1(randperm(ntrs),:) , x2(randperm(ntrs),:));
            %%%%% exc,inh: corr between oppositely-tuned neurons of exc and inh
            x1 = fr_exc_ipsi_timeM1{im}{iday}(:,trs)';
            x2 = fr_inh_contra_timeM1{im}{iday}(:,trs)';
            r_Ei_Ic_shfl{im}{iday} = corr(x1(randperm(ntrs),:) , x2(randperm(ntrs),:));
            x1 = fr_exc_contra_timeM1{im}{iday}(:,trs)';
            x2 = fr_inh_ipsi_timeM1{im}{iday}(:,trs)';
            r_Ec_Ii_shfl{im}{iday} = corr(x1(randperm(ntrs),:) , x2(randperm(ntrs),:));
            %%%%% exc,inh: corr between all neurons of exc and inh
            x1 = fr_exc_timeM1{im}{iday}(:,trs)';
            x2 = fr_inh_timeM1{im}{iday}(:,trs)';
            r_E_I_shfl{im}{iday} = corr(x1(randperm(ntrs),:) , x2(randperm(ntrs),:));
            
        end
    end
end


%% Set the diagonal of Ei_Ei, etc to nan (bc they are symmetric)

for im = 1:length(mice)
    for iday = 1:numDaysAll(im)
        if mnTrNum_allMice{im}(iday) >= thMinTrs
            r_Ei_Ei{im}{iday}(logical(eye(size(r_Ei_Ei{im}{iday})))) = nan; 
            r_Ec_Ec{im}{iday}(logical(eye(size(r_Ec_Ec{im}{iday})))) = nan;
            r_Ii_Ii{im}{iday}(logical(eye(size(r_Ii_Ii{im}{iday})))) = nan;
            r_Ic_Ic{im}{iday}(logical(eye(size(r_Ic_Ic{im}{iday})))) = nan;
            r_E_E{im}{iday}(logical(eye(size(r_E_E{im}{iday})))) = nan;
            r_I_I{im}{iday}(logical(eye(size(r_I_I{im}{iday})))) = nan;
            
            % shfl
            r_Ei_Ei_shfl{im}{iday}(logical(eye(size(r_Ei_Ei{im}{iday})))) = nan; 
            r_Ec_Ec_shfl{im}{iday}(logical(eye(size(r_Ec_Ec{im}{iday})))) = nan;
            r_Ii_Ii_shfl{im}{iday}(logical(eye(size(r_Ii_Ii{im}{iday})))) = nan;
            r_Ic_Ic_shfl{im}{iday}(logical(eye(size(r_Ic_Ic{im}{iday})))) = nan;            
            r_E_E_shfl{im}{iday}(logical(eye(size(r_E_E{im}{iday})))) = nan;
            r_I_I_shfl{im}{iday}(logical(eye(size(r_I_I{im}{iday})))) = nan;            

        end
    end
end




%% Set averaged pairwise correlations 
% Assuming each neuron is connected to all other neurons, to estimate the
% neuron's (N) connectivity strength with a particular population ("P",
% either Eipsi, Econtra, Iipsi, or Icontra), we average pairwise
% correlation of the neuron (N) with all the neurons in that population
% (P). This way we can compare connectivity of N with different populations 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Now average corr of each neuron with all neurons of the ipsi (or contra) population.

rAve_Ei_Ei = cell(1, length(mice));
rAve_Ec_Ec = cell(1, length(mice));
rAve_Ii_Ii = cell(1, length(mice));
rAve_Ic_Ic = cell(1, length(mice));
rAve_Ei_Ec = cell(1, length(mice));
rAve_Ec_Ei = cell(1, length(mice));
rAve_Ii_Ic = cell(1, length(mice));
rAve_Ic_Ii = cell(1, length(mice));
rAve_Ei_Ii = cell(1, length(mice));
rAve_Ii_Ei = cell(1, length(mice));
rAve_Ec_Ic = cell(1, length(mice));
rAve_Ic_Ec = cell(1, length(mice));
rAve_Ei_Ic = cell(1, length(mice));
rAve_Ic_Ei = cell(1, length(mice));
rAve_Ec_Ii = cell(1, length(mice));
rAve_Ii_Ec = cell(1, length(mice));
rAve_EE_same = cell(1, length(mice));
rAve_II_same = cell(1, length(mice));
rAve_EE_diff = cell(1, length(mice));
rAve_II_diff = cell(1, length(mice));
rAve_Ei_Ii_both = cell(1, length(mice));
rAve_Ec_Ic_both = cell(1, length(mice));
rAve_EI_same = cell(1, length(mice));
rAve_IE_same = cell(1, length(mice));
rAve_Ei_Ic_both = cell(1, length(mice));
rAve_Ec_Ii_both = cell(1, length(mice));
rAve_EI_diff = cell(1, length(mice));
rAve_IE_diff = cell(1, length(mice));
rAve_EE = cell(1, length(mice));
rAve_II = cell(1, length(mice));
rAve_EI0 = cell(1, length(mice));
rAve_IE0 = cell(1, length(mice));
rAve_EI = cell(1, length(mice));

% shfl
rAve_Ei_Ei_shfl = cell(1, length(mice));
rAve_Ec_Ec_shfl = cell(1, length(mice));
rAve_Ii_Ii_shfl = cell(1, length(mice));
rAve_Ic_Ic_shfl = cell(1, length(mice));
rAve_Ei_Ec_shfl = cell(1, length(mice));
rAve_Ec_Ei_shfl = cell(1, length(mice));
rAve_Ii_Ic_shfl = cell(1, length(mice));
rAve_Ic_Ii_shfl = cell(1, length(mice));
rAve_Ei_Ii_shfl = cell(1, length(mice));
rAve_Ii_Ei_shfl = cell(1, length(mice));
rAve_Ec_Ic_shfl = cell(1, length(mice));
rAve_Ic_Ec_shfl = cell(1, length(mice));
rAve_Ei_Ic_shfl = cell(1, length(mice));
rAve_Ic_Ei_shfl = cell(1, length(mice));
rAve_Ec_Ii_shfl = cell(1, length(mice));
rAve_Ii_Ec_shfl = cell(1, length(mice));
rAve_EE_same_shfl = cell(1, length(mice));
rAve_II_same_shfl = cell(1, length(mice));
rAve_EE_diff_shfl = cell(1, length(mice));
rAve_II_diff_shfl = cell(1, length(mice));
rAve_Ei_Ii_both_shfl = cell(1, length(mice));
rAve_Ec_Ic_both_shfl = cell(1, length(mice));
rAve_EI_same_shfl = cell(1, length(mice));
rAve_Ei_Ic_both_shfl = cell(1, length(mice));
rAve_Ec_Ii_both_shfl = cell(1, length(mice));
rAve_EI_diff_shfl = cell(1, length(mice));
rAve_EE_shfl = cell(1, length(mice));
rAve_II_shfl = cell(1, length(mice));
rAve_EI0_shfl = cell(1, length(mice));
rAve_IE0_shfl = cell(1, length(mice));
rAve_EI_shfl = cell(1, length(mice));

for im = 1:length(mice)
    for iday = 1:numDaysAll(im)
        if mnTrNum_allMice{im}(iday) >= thMinTrs
            
            %%%%%%%%%%%% EE_same
            rAve_Ei_Ei{im}{iday} = nanmean(r_Ei_Ei{im}{iday}, 1);
            rAve_Ec_Ec{im}{iday} = nanmean(r_Ec_Ec{im}{iday}, 1);
            % pool the two above
            rAve_EE_same{im}{iday} = [rAve_Ei_Ei{im}{iday}, rAve_Ec_Ec{im}{iday}];
            
            %%%%%%%%%%%% II_same
            rAve_Ii_Ii{im}{iday} = nanmean(r_Ii_Ii{im}{iday}, 1);
            rAve_Ic_Ic{im}{iday} = nanmean(r_Ic_Ic{im}{iday}, 1);
            % pool the two above
            rAve_II_same{im}{iday} = [rAve_Ii_Ii{im}{iday}, rAve_Ic_Ic{im}{iday}];
            
            
            %%%%%%%%%%%% EE_diff
            rAve_Ei_Ec{im}{iday} = nanmean(r_Ei_Ec{im}{iday}, 2)'; % corr of each Ei with all Ecs averaged
            rAve_Ec_Ei{im}{iday} = nanmean(r_Ei_Ec{im}{iday}, 1); % corr of each Ec with all Eis averaged
            % pool the two above
            rAve_EE_diff{im}{iday} = [rAve_Ei_Ec{im}{iday}, rAve_Ec_Ei{im}{iday}];
            
            %%%%%%%%%%%% II_diff
            rAve_Ii_Ic{im}{iday} = nanmean(r_Ii_Ic{im}{iday}, 2)'; % corr of each Ii with all Ics averaged
            rAve_Ic_Ii{im}{iday} = nanmean(r_Ii_Ic{im}{iday}, 1); % corr of each Ic with all Iis averaged
            % pool the two above
            rAve_II_diff{im}{iday} = [rAve_Ii_Ic{im}{iday}, rAve_Ic_Ii{im}{iday}];
            
            
            %%%%%%%%%%%% EI_same
            rAve_Ei_Ii{im}{iday} = nanmean(r_Ei_Ii{im}{iday}, 2)'; % corr of each Ei with all Iis averaged
            rAve_Ii_Ei{im}{iday} = nanmean(r_Ei_Ii{im}{iday}, 1); % corr of each Ii with all Eis averaged
            % pool the two above
            rAve_Ei_Ii_both{im}{iday} = [rAve_Ei_Ii{im}{iday}, rAve_Ii_Ei{im}{iday}];
            
            rAve_Ec_Ic{im}{iday} = nanmean(r_Ec_Ic{im}{iday}, 2)'; % corr of each Ec with all Ics averaged
            rAve_Ic_Ec{im}{iday} = nanmean(r_Ec_Ic{im}{iday}, 1); % corr of each Ic with all Ecs averaged
            % pool the two above
            rAve_Ec_Ic_both{im}{iday} = [rAve_Ec_Ic{im}{iday}, rAve_Ic_Ec{im}{iday}];
            
            % pool EI_i and EI_c
            rAve_EI_same{im}{iday} = [rAve_Ei_Ii_both{im}{iday} , rAve_Ec_Ic_both{im}{iday}];
%             rAve_EI_same{im}{iday} = [rAve_Ei_Ii{im}{iday} , rAve_Ec_Ic{im}{iday}];
%             rAve_IE_same{im}{iday} = [rAve_Ii_Ei{im}{iday} , rAve_Ic_Ec{im}{iday}];
            
            
            %%%%%%%%%%%% EI_diff
            rAve_Ei_Ic{im}{iday} = nanmean(r_Ei_Ic{im}{iday}, 2)'; % corr of each Ei with all Ics averaged
            rAve_Ic_Ei{im}{iday} = nanmean(r_Ei_Ic{im}{iday}, 1); % corr of each Ic with all Eis averaged
            % pool the two above
            rAve_Ei_Ic_both{im}{iday} = [rAve_Ei_Ic{im}{iday}, rAve_Ic_Ei{im}{iday}];
            
            rAve_Ec_Ii{im}{iday} = nanmean(r_Ec_Ii{im}{iday}, 2)'; % corr of each Ec with all Iis averaged
            rAve_Ii_Ec{im}{iday} = nanmean(r_Ec_Ii{im}{iday}, 1); % corr of each Ii with all Ecs averaged
            % pool the two above
            rAve_Ec_Ii_both{im}{iday} = [rAve_Ec_Ii{im}{iday}, rAve_Ii_Ec{im}{iday}];
            
            % pool EiIc and EcIi
            rAve_EI_diff{im}{iday} = [rAve_Ei_Ic_both{im}{iday} , rAve_Ec_Ii_both{im}{iday}];
%             rAve_EI_diff{im}{iday} = [rAve_Ei_Ic{im}{iday} , rAve_Ec_Ii{im}{iday}];
%             rAve_IE_diff{im}{iday} = [rAve_Ic_Ei{im}{iday} , rAve_Ii_Ec{im}{iday}];
            

            %%%%%%%%%%%% EE (all neurons)
            rAve_EE{im}{iday} = nanmean(r_E_E{im}{iday}, 2)'; % corr of each E with all Es averaged
            
            %%%%%%%%%%%% II (all neurons)
            rAve_II{im}{iday} = nanmean(r_I_I{im}{iday}, 2)'; % corr of each I with all Is averaged
            
            %%%%%%%%%%%% EI (all neurons)
            rAve_EI0{im}{iday} = nanmean(r_E_I{im}{iday}, 2)'; % corr of each E with all Is averaged
            rAve_IE0{im}{iday} = nanmean(r_E_I{im}{iday}, 1); % corr of each I with all Es averaged
            % pool the two above
            rAve_EI{im}{iday} = [rAve_EI0{im}{iday}, rAve_IE0{im}{iday}];
            
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%% shfl %%%%%%%%%%%
            rAve_Ei_Ei_shfl{im}{iday} = nanmean(r_Ei_Ei_shfl{im}{iday}, 1);
            rAve_Ec_Ec_shfl{im}{iday} = nanmean(r_Ec_Ec_shfl{im}{iday}, 1);
            % pool the two above
            rAve_EE_same_shfl{im}{iday} = [rAve_Ei_Ei_shfl{im}{iday}, rAve_Ec_Ec_shfl{im}{iday}];
            
            rAve_Ii_Ii_shfl{im}{iday} = nanmean(r_Ii_Ii_shfl{im}{iday}, 1);
            rAve_Ic_Ic_shfl{im}{iday} = nanmean(r_Ic_Ic_shfl{im}{iday}, 1);
            % pool the two above
            rAve_II_same_shfl{im}{iday} = [rAve_Ii_Ii_shfl{im}{iday}, rAve_Ic_Ic_shfl{im}{iday}];
            
            
            rAve_Ei_Ec_shfl{im}{iday} = nanmean(r_Ei_Ec_shfl{im}{iday}, 2)'; % corr of each Ei with all Ecs averaged
            rAve_Ec_Ei_shfl{im}{iday} = nanmean(r_Ei_Ec_shfl{im}{iday}, 1); % corr of each Ec with all Eis averaged
            % pool the two above
            rAve_EE_diff_shfl{im}{iday} = [rAve_Ei_Ec_shfl{im}{iday}, rAve_Ec_Ei_shfl{im}{iday}];
            
            rAve_Ii_Ic_shfl{im}{iday} = nanmean(r_Ii_Ic_shfl{im}{iday}, 2)'; % corr of each Ii with all Ics averaged
            rAve_Ic_Ii_shfl{im}{iday} = nanmean(r_Ii_Ic_shfl{im}{iday}, 1); % corr of each Ic with all Iis averaged
            % pool the two above
            rAve_II_diff_shfl{im}{iday} = [rAve_Ii_Ic_shfl{im}{iday}, rAve_Ic_Ii_shfl{im}{iday}];
            
            
            rAve_Ei_Ii_shfl{im}{iday} = nanmean(r_Ei_Ii_shfl{im}{iday}, 2)'; % corr of each Ei with all Iis averaged
            rAve_Ii_Ei_shfl{im}{iday} = nanmean(r_Ei_Ii_shfl{im}{iday}, 1); % corr of each Ii with all Eis averaged
            % pool the two above
            rAve_Ei_Ii_both_shfl{im}{iday} = [rAve_Ei_Ii_shfl{im}{iday}, rAve_Ii_Ei_shfl{im}{iday}];
            
            rAve_Ec_Ic_shfl{im}{iday} = nanmean(r_Ec_Ic_shfl{im}{iday}, 2)'; % corr of each Ec with all Ics averaged
            rAve_Ic_Ec_shfl{im}{iday} = nanmean(r_Ec_Ic_shfl{im}{iday}, 1); % corr of each Ic with all Ecs averaged
            % pool the two above
            rAve_Ec_Ic_both_shfl{im}{iday} = [rAve_Ec_Ic_shfl{im}{iday}, rAve_Ic_Ec_shfl{im}{iday}];
            
            % pool EI_i and EI_c
            rAve_EI_same_shfl{im}{iday} = [rAve_Ei_Ii_both_shfl{im}{iday} , rAve_Ec_Ic_both_shfl{im}{iday}];
            
            
            rAve_Ei_Ic_shfl{im}{iday} = nanmean(r_Ei_Ic_shfl{im}{iday}, 2)'; % corr of each Ei with all Ics averaged
            rAve_Ic_Ei_shfl{im}{iday} = nanmean(r_Ei_Ic_shfl{im}{iday}, 1); % corr of each Ic with all Eis averaged
            % pool the two above
            rAve_Ei_Ic_both_shfl{im}{iday} = [rAve_Ei_Ic_shfl{im}{iday}, rAve_Ic_Ei_shfl{im}{iday}];
            
            rAve_Ec_Ii_shfl{im}{iday} = nanmean(r_Ec_Ii_shfl{im}{iday}, 2)'; % corr of each Ec with all Iis averaged
            rAve_Ii_Ec_shfl{im}{iday} = nanmean(r_Ec_Ii_shfl{im}{iday}, 1); % corr of each Ii with all Ecs averaged
            % pool the two above
            rAve_Ec_Ii_both_shfl{im}{iday} = [rAve_Ec_Ii_shfl{im}{iday}, rAve_Ii_Ec_shfl{im}{iday}];
            
            % pool EiIc and EcIi
            rAve_EI_diff_shfl{im}{iday} = [rAve_Ei_Ic_both_shfl{im}{iday} , rAve_Ec_Ii_both_shfl{im}{iday}];
            
            
            %%%%%%%%%%%% EE (all neurons)
            rAve_EE_shfl{im}{iday} = nanmean(r_E_E_shfl{im}{iday}, 2)'; % corr of each E with all Es averaged
            
            %%%%%%%%%%%% II (all neurons)
            rAve_II_shfl{im}{iday} = nanmean(r_I_I_shfl{im}{iday}, 2)'; % corr of each I with all Is averaged
            
            %%%%%%%%%%%% EI (all neurons)
            rAve_EI0_shfl{im}{iday} = nanmean(r_E_I_shfl{im}{iday}, 2)'; % corr of each E with all Is averaged
            rAve_IE0_shfl{im}{iday} = nanmean(r_E_I_shfl{im}{iday}, 1); % corr of each I with all Es averaged
            % pool the two above
            rAve_EI_shfl{im}{iday} = [rAve_EI0_shfl{im}{iday}, rAve_IE0_shfl{im}{iday}];
            
        end
    end
end


%% Pool all neurons of all mice

rAve_EE_same_allMiceNs = [rAve_EE_same{:}];
rAve_EE_same_allMiceNs = [rAve_EE_same_allMiceNs{:}];
rAve_EE_diff_allMiceNs = [rAve_EE_diff{:}];
rAve_EE_diff_allMiceNs = [rAve_EE_diff_allMiceNs{:}];
rAve_II_same_allMiceNs = [rAve_II_same{:}];
rAve_II_same_allMiceNs = [rAve_II_same_allMiceNs{:}];
rAve_II_diff_allMiceNs = [rAve_II_diff{:}];
rAve_II_diff_allMiceNs = [rAve_II_diff_allMiceNs{:}];
rAve_EI_same_allMiceNs = [rAve_EI_same{:}];
rAve_EI_same_allMiceNs = [rAve_EI_same_allMiceNs{:}];
rAve_EI_diff_allMiceNs = [rAve_EI_diff{:}];
rAve_EI_diff_allMiceNs = [rAve_EI_diff_allMiceNs{:}];
rAve_EE_allMiceNs = [rAve_EE{:}];
rAve_EE_allMiceNs = [rAve_EE_allMiceNs{:}];
rAve_II_allMiceNs = [rAve_II{:}];
rAve_II_allMiceNs = [rAve_II_allMiceNs{:}];
rAve_EI_allMiceNs = [rAve_EI{:}];
rAve_EI_allMiceNs = [rAve_EI_allMiceNs{:}];

% shfl
rAve_EE_same_allMiceNs_shfl = [rAve_EE_same_shfl{:}];
rAve_EE_same_allMiceNs_shfl = [rAve_EE_same_allMiceNs_shfl{:}];
rAve_EE_diff_allMiceNs_shfl = [rAve_EE_diff_shfl{:}];
rAve_EE_diff_allMiceNs_shfl = [rAve_EE_diff_allMiceNs_shfl{:}];
rAve_II_same_allMiceNs_shfl = [rAve_II_same_shfl{:}];
rAve_II_same_allMiceNs_shfl = [rAve_II_same_allMiceNs_shfl{:}];
rAve_II_diff_allMiceNs_shfl = [rAve_II_diff_shfl{:}];
rAve_II_diff_allMiceNs_shfl = [rAve_II_diff_allMiceNs_shfl{:}];
rAve_EI_same_allMiceNs_shfl = [rAve_EI_same_shfl{:}];
rAve_EI_same_allMiceNs_shfl = [rAve_EI_same_allMiceNs_shfl{:}];
rAve_EI_diff_allMiceNs_shfl = [rAve_EI_diff_shfl{:}];
rAve_EI_diff_allMiceNs_shfl = [rAve_EI_diff_allMiceNs_shfl{:}];
rAve_EE_allMiceNs_shfl = [rAve_EE_shfl{:}];
rAve_EE_allMiceNs_shfl = [rAve_EE_allMiceNs_shfl{:}];
rAve_II_allMiceNs_shfl = [rAve_II_shfl{:}];
rAve_II_allMiceNs_shfl = [rAve_II_allMiceNs_shfl{:}];
rAve_EI_allMiceNs_shfl = [rAve_EI_shfl{:}];
rAve_EI_allMiceNs_shfl = [rAve_EI_allMiceNs_shfl{:}];


%% Take average and std of rAve across neurons for each day

rAve_EI_same_avEachDay = cell(1, length(mice));
rAve_EI_diff_avEachDay = cell(1, length(mice));
rAve_EE_same_avEachDay = cell(1, length(mice));
rAve_EE_diff_avEachDay = cell(1, length(mice));
rAve_II_same_avEachDay = cell(1, length(mice));
rAve_II_diff_avEachDay = cell(1, length(mice));
rAve_EI_avEachDay = cell(1, length(mice));
rAve_EE_avEachDay = cell(1, length(mice));
rAve_II_avEachDay = cell(1, length(mice));
% rAve_EI_same_sdEachDay = cell(1, length(mice));
% rAve_EI_diff_sdEachDay = cell(1, length(mice));
% rAve_EE_same_sdEachDay = cell(1, length(mice));
% rAve_EE_diff_sdEachDay = cell(1, length(mice));
% rAve_II_same_sdEachDay = cell(1, length(mice));
% rAve_II_diff_sdEachDay = cell(1, length(mice));
rAve_EI_same_shfl_avEachDay = cell(1, length(mice));
rAve_EI_diff_shfl_avEachDay = cell(1, length(mice));
rAve_EE_same_shfl_avEachDay = cell(1, length(mice));
rAve_EE_diff_shfl_avEachDay = cell(1, length(mice));
rAve_II_same_shfl_avEachDay = cell(1, length(mice));
rAve_II_diff_shfl_avEachDay = cell(1, length(mice));
rAve_EI_shfl_avEachDay = cell(1, length(mice));
rAve_EE_shfl_avEachDay = cell(1, length(mice));
rAve_II_shfl_avEachDay = cell(1, length(mice));

for im = 1:length(mice)
    %%% EI
    rAve_EI_same_avEachDay{im} = cellfun(@mean, rAve_EI_same{im});
    rAve_EI_diff_avEachDay{im} = cellfun(@mean, rAve_EI_diff{im});
%     rAve_EI_same_sdEachDay{im} = cellfun(@std, rAve_EI_same{im});    
%     rAve_EI_diff_sdEachDay{im} = cellfun(@std, rAve_EI_diff{im});
    rAve_EI_avEachDay{im} = cellfun(@mean, rAve_EI{im});
    
    %%% EE
    rAve_EE_same_avEachDay{im} = cellfun(@mean, rAve_EE_same{im});
    rAve_EE_diff_avEachDay{im} = cellfun(@mean, rAve_EE_diff{im});
%     rAve_EE_same_sdEachDay{im} = cellfun(@std, rAve_EE_same{im});
%     rAve_EE_diff_sdEachDay{im} = cellfun(@std, rAve_EE_diff{im});
    rAve_EE_avEachDay{im} = cellfun(@mean, rAve_EE{im});
    
    %%% II
    rAve_II_same_avEachDay{im} = cellfun(@mean, rAve_II_same{im});
    rAve_II_diff_avEachDay{im} = cellfun(@mean, rAve_II_diff{im});
%     rAve_II_same_sdEachDay{im} = cellfun(@std, rAve_II_same{im});
%     rAve_II_diff_sdEachDay{im} = cellfun(@std, rAve_II_diff{im});    
    rAve_II_avEachDay{im} = cellfun(@mean, rAve_II{im});

    
    %%%%%%%% shfl
    %%% EI
    rAve_EI_same_shfl_avEachDay{im} = cellfun(@mean, rAve_EI_same_shfl{im});
    rAve_EI_diff_shfl_avEachDay{im} = cellfun(@mean, rAve_EI_diff_shfl{im});
    rAve_EI_shfl_avEachDay{im} = cellfun(@mean, rAve_EI_shfl{im});
    
    %%% EE
    rAve_EE_same_shfl_avEachDay{im} = cellfun(@mean, rAve_EE_same_shfl{im});
    rAve_EE_diff_shfl_avEachDay{im} = cellfun(@mean, rAve_EE_diff_shfl{im});
    rAve_EE_shfl_avEachDay{im} = cellfun(@mean, rAve_EE_shfl{im});
    
    %%% II
    rAve_II_same_shfl_avEachDay{im} = cellfun(@mean, rAve_II_same_shfl{im});
    rAve_II_diff_shfl_avEachDay{im} = cellfun(@mean, rAve_II_diff_shfl{im});
    rAve_II_shfl_avEachDay{im} = cellfun(@mean, rAve_II_shfl{im});
end


%%%%%%%%%%%% Now take ave and se across days (also pool neurons and days, and then take average) %%%%%%%%%%%%
rAve_EI_diff_same_avDays = nan(2, length(mice));
rAve_EI_diff_same_seDays = nan(2, length(mice));
rAve_EE_diff_same_avDays = nan(2, length(mice));
rAve_EE_diff_same_seDays = nan(2, length(mice));
rAve_II_diff_same_avDays = nan(2, length(mice));
rAve_II_diff_same_seDays = nan(2, length(mice));

rAve_EI_diff_same_avAllNsDays = nan(2, length(mice));
rAve_EI_diff_same_seAllNsDays = nan(2, length(mice));
rAve_EE_diff_same_avAllNsDays = nan(2, length(mice));
rAve_EE_diff_same_seAllNsDays = nan(2, length(mice));
rAve_II_diff_same_avAllNsDays = nan(2, length(mice));
rAve_II_diff_same_seAllNsDays = nan(2, length(mice));

rAve_EI_avDays = nan(1, length(mice));
rAve_EE_avDays = nan(1, length(mice));
rAve_II_avDays = nan(1, length(mice));
rAve_EI_seDays = nan(1, length(mice));
rAve_EE_seDays = nan(1, length(mice));
rAve_II_seDays = nan(1, length(mice));
rAve_EI_avAllNsDays = nan(1, length(mice));
rAve_EE_avAllNsDays = nan(1, length(mice));
rAve_II_avAllNsDays = nan(1, length(mice));
rAve_EI_seAllNsDays = nan(1, length(mice));
rAve_EE_seAllNsDays = nan(1, length(mice));
rAve_II_seAllNsDays = nan(1, length(mice));

% shfl
rAve_EI_diff_same_shfl_avDays = nan(2, length(mice));
rAve_EI_diff_same_shfl_seDays = nan(2, length(mice));
rAve_EE_diff_same_shfl_avDays = nan(2, length(mice));
rAve_EE_diff_same_shfl_seDays = nan(2, length(mice));
rAve_II_diff_same_shfl_avDays = nan(2, length(mice));
rAve_II_diff_same_shfl_seDays = nan(2, length(mice));

rAve_EI_diff_same_shfl_avAllNsDays = nan(2, length(mice));
rAve_EI_diff_same_shfl_seAllNsDays = nan(2, length(mice));
rAve_EE_diff_same_shfl_avAllNsDays = nan(2, length(mice));
rAve_EE_diff_same_shfl_seAllNsDays = nan(2, length(mice));
rAve_II_diff_same_shfl_avAllNsDays = nan(2, length(mice));
rAve_II_diff_same_shfl_seAllNsDays = nan(2, length(mice));

rAve_EI_shfl_avDays = nan(1, length(mice));
rAve_EE_shfl_avDays = nan(1, length(mice));
rAve_II_shfl_avDays = nan(1, length(mice));
rAve_EI_shfl_seDays = nan(1, length(mice));
rAve_EE_shfl_seDays = nan(1, length(mice));
rAve_II_shfl_seDays = nan(1, length(mice));
rAve_EI_shfl_avAllNsDays = nan(1, length(mice));
rAve_EE_shfl_avAllNsDays = nan(1, length(mice));
rAve_II_shfl_avAllNsDays = nan(1, length(mice));
rAve_EI_shfl_seAllNsDays = nan(1, length(mice));
rAve_EE_shfl_seAllNsDays = nan(1, length(mice));
rAve_II_shfl_seAllNsDays = nan(1, length(mice));

for im = 1:length(mice)
    %%% EI
    rAve_EI_diff_same_avDays(:,im) = [nanmean(rAve_EI_diff_avEachDay{im}) ; nanmean(rAve_EI_same_avEachDay{im})];
    rAve_EI_diff_same_seDays(:,im) = [nanstd(rAve_EI_diff_avEachDay{im}) / sqrt(numDaysGood(im)) ; nanstd(rAve_EI_same_avEachDay{im}) / sqrt(numDaysGood(im))];
    rAve_EI_avDays(im) = nanmean(rAve_EI_avEachDay{im});
    rAve_EI_seDays(im) = nanstd(rAve_EI_avEachDay{im}) / sqrt(numDaysGood(im));    
    %%%%%%%%%%%%%% pool all neurons of all days, then take average and sd
    y1 = [rAve_EI_diff{im}{:}];
    y2 = [rAve_EI_same{im}{:}];
    rAve_EI_diff_same_avAllNsDays(:,im) = [nanmean(y1) ; nanmean(y2)];
    rAve_EI_diff_same_seAllNsDays(:,im) = [nanstd(y1) / sqrt(sum(~isnan(y1))) ; nanstd(y2) / sqrt(sum(~isnan(y2)))];
    y1 = [rAve_EI{im}{:}];
    rAve_EI_avAllNsDays(im) = nanmean(y1);
    rAve_EI_seAllNsDays(im) = nanstd(y1) / sqrt(sum(~isnan(y1)));

    
    %%% EE
    rAve_EE_diff_same_avDays(:,im) = [nanmean(rAve_EE_diff_avEachDay{im}) ; nanmean(rAve_EE_same_avEachDay{im})];
    rAve_EE_diff_same_seDays(:,im) = [nanstd(rAve_EE_diff_avEachDay{im}) / sqrt(numDaysGood(im)) ; nanstd(rAve_EE_same_avEachDay{im}) / sqrt(numDaysGood(im))];
    rAve_EE_avDays(im) = nanmean(rAve_EE_avEachDay{im});
    rAve_EE_seDays(im) = nanstd(rAve_EE_avEachDay{im}) / sqrt(numDaysGood(im));      
    %%%%%%%%%%%%%% pool all neurons of all days, then take average and sd
    y1 = [rAve_EE_diff{im}{:}];
    y2 = [rAve_EE_same{im}{:}];
    rAve_EE_diff_same_avAllNsDays(:,im) = [nanmean(y1) ; nanmean(y2)];
    rAve_EE_diff_same_seAllNsDays(:,im) = [nanstd(y1) / sqrt(sum(~isnan(y1))) ; nanstd(y2) / sqrt(sum(~isnan(y2)))];
    y1 = [rAve_EE{im}{:}];
    rAve_EE_avAllNsDays(im) = nanmean(y1);
    rAve_EE_seAllNsDays(im) = nanstd(y1) / sqrt(sum(~isnan(y1)));
    
    
    %%% II
    rAve_II_diff_same_avDays(:,im) = [nanmean(rAve_II_diff_avEachDay{im}) ; nanmean(rAve_II_same_avEachDay{im})];
    rAve_II_diff_same_seDays(:,im) = [nanstd(rAve_II_diff_avEachDay{im}) / sqrt(numDaysGood(im)) ; nanstd(rAve_II_same_avEachDay{im}) / sqrt(numDaysGood(im))];
    rAve_II_avDays(im) = nanmean(rAve_II_avEachDay{im});
    rAve_II_seDays(im) = nanstd(rAve_II_avEachDay{im}) / sqrt(numDaysGood(im));          
    %%%%%%%%%%%%%% pool all neurons of all days, then take average and sd
    y1 = [rAve_II_diff{im}{:}];
    y2 = [rAve_II_same{im}{:}];
    rAve_II_diff_same_avAllNsDays(:,im) = [nanmean(y1) ; nanmean(y2)];
    rAve_II_diff_same_seAllNsDays(:,im) = [nanstd(y1) / sqrt(sum(~isnan(y1))) ; nanstd(y2) / sqrt(sum(~isnan(y2)))];
    y1 = [rAve_II{im}{:}];
    rAve_II_avAllNsDays(im) = nanmean(y1);
    rAve_II_seAllNsDays(im) = nanstd(y1) / sqrt(sum(~isnan(y1)));

    

    %%%%%%%%%%% shfl
    %%% EI
    rAve_EI_diff_same_shfl_avDays(:,im) = [nanmean(rAve_EI_diff_shfl_avEachDay{im}) ; nanmean(rAve_EI_same_shfl_avEachDay{im})];
    rAve_EI_diff_same_shfl_seDays(:,im) = [nanstd(rAve_EI_diff_shfl_avEachDay{im}) / sqrt(numDaysGood(im)) ; nanstd(rAve_EI_same_shfl_avEachDay{im}) / sqrt(numDaysGood(im))];
    rAve_EI_shfl_avDays(im) = nanmean(rAve_EI_shfl_avEachDay{im});
    rAve_EI_shfl_seDays(im) = nanstd(rAve_EI_shfl_avEachDay{im}) / sqrt(numDaysGood(im));    
    %%%%%%%%%%%%%% pool all neurons of all days, then take average and sd
    y1 = [rAve_EI_diff_shfl{im}{:}];
    y2 = [rAve_EI_same_shfl{im}{:}];
    rAve_EI_diff_same_shfl_avAllNsDays(:,im) = [nanmean(y1) ; nanmean(y2)];
    rAve_EI_diff_same_shfl_seAllNsDays(:,im) = [nanstd(y1) / sqrt(sum(~isnan(y1))) ; nanstd(y2) / sqrt(sum(~isnan(y2)))];
    y1 = [rAve_EI_shfl{im}{:}];
    rAve_EI_shfl_avAllNsDays(im) = nanmean(y1);
    rAve_EI_shfl_seAllNsDays(im) = nanstd(y1) / sqrt(sum(~isnan(y1)));
    
    
    %%% EE
    rAve_EE_diff_same_shfl_avDays(:,im) = [nanmean(rAve_EE_diff_shfl_avEachDay{im}) ; nanmean(rAve_EE_same_shfl_avEachDay{im})];
    rAve_EE_diff_same_shfl_seDays(:,im) = [nanstd(rAve_EE_diff_shfl_avEachDay{im}) / sqrt(numDaysGood(im)) ; nanstd(rAve_EE_same_shfl_avEachDay{im}) / sqrt(numDaysGood(im))];
    rAve_EE_shfl_avDays(im) = nanmean(rAve_EE_shfl_avEachDay{im});
    rAve_EE_shfl_seDays(im) = nanstd(rAve_EE_shfl_avEachDay{im}) / sqrt(numDaysGood(im));      
    %%%%%%%%%%%%%% pool all neurons of all days, then take average and sd
    y1 = [rAve_EE_diff_shfl{im}{:}];
    y2 = [rAve_EE_same_shfl{im}{:}];
    rAve_EE_diff_same_shfl_avAllNsDays(:,im) = [nanmean(y1) ; nanmean(y2)];
    rAve_EE_diff_same_shfl_seAllNsDays(:,im) = [nanstd(y1) / sqrt(sum(~isnan(y1))) ; nanstd(y2) / sqrt(sum(~isnan(y2)))];
    y1 = [rAve_EE_shfl{im}{:}];
    rAve_EE_shfl_avAllNsDays(im) = nanmean(y1);
    rAve_EE_shfl_seAllNsDays(im) = nanstd(y1) / sqrt(sum(~isnan(y1)));

    
    %%% II
    rAve_II_diff_same_shfl_avDays(:,im) = [nanmean(rAve_II_diff_shfl_avEachDay{im}) ; nanmean(rAve_II_same_shfl_avEachDay{im})];
    rAve_II_diff_same_shfl_seDays(:,im) = [nanstd(rAve_II_diff_shfl_avEachDay{im}) / sqrt(numDaysGood(im)) ; nanstd(rAve_II_same_shfl_avEachDay{im}) / sqrt(numDaysGood(im))];
    rAve_II_shfl_avDays(im) = nanmean(rAve_II_shfl_avEachDay{im});
    rAve_II_shfl_seDays(im) = nanstd(rAve_II_shfl_avEachDay{im}) / sqrt(numDaysGood(im));      
    %%%%%%%%%%%%%% pool all neurons of all days, then take average and sd
    y1 = [rAve_II_diff_shfl{im}{:}];
    y2 = [rAve_II_same_shfl{im}{:}];
    rAve_II_diff_same_shfl_avAllNsDays(:,im) = [nanmean(y1) ; nanmean(y2)];
    rAve_II_diff_same_shfl_seAllNsDays(:,im) = [nanstd(y1) / sqrt(sum(~isnan(y1))) ; nanstd(y2) / sqrt(sum(~isnan(y2)))];    
    y1 = [rAve_II_shfl{im}{:}];
    rAve_II_shfl_avAllNsDays(im) = nanmean(y1);
    rAve_II_shfl_seAllNsDays(im) = nanstd(y1) / sqrt(sum(~isnan(y1)));    
end



no
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Plots : avePairwise %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if doPlots
    
    %% Plot errorbar, showing each mouse, ave +/- se of rAve across days (already averaged across neurons for each day)
    %{
    x = 1:length(mice);
    gp = .2;
    marg = .2;

    figure('name', 'All mice', 'position', [14   636   661   290]); 
%     set(gca, 'position', [0.2919    0.1908    0.5229    0.7095])

    %%%%%%%%%% Plot EI %%%%%%%%%%%
    typeNs = 'EI';    
    subplot(131); hold on
    h1 = errorbar(x, rAve_EI_diff_same_avDays(1,:), rAve_EI_diff_same_seDays(1,:), 'r.', 'linestyle', 'none');
    h2 = errorbar(x+gp, rAve_EI_diff_same_avDays(2,:), rAve_EI_diff_same_seDays(2,:), 'k.', 'linestyle', 'none');
    errorbar(x, rAve_EI_diff_same_shfl_avDays(1,:), rAve_EI_diff_same_shfl_seDays(1,:), 'color',rgb('lightsalmon'), 'marker', '.', 'linestyle', 'none')
    errorbar(x+gp, rAve_EI_diff_same_shfl_avDays(2,:), rAve_EI_diff_same_shfl_seDays(2,:), 'color',rgb('gray'), 'marker', '.', 'linestyle', 'none')
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    title(typeNs)
    ylabel('corr (mean +/- se days)')
    legend([h1,h2], {'diff','same'})

    %%%%%%%%%% Plot EE %%%%%%%%%%%
    typeNs = 'EE';    
    subplot(132); hold on
    errorbar(x, rAve_EE_diff_same_avDays(1,:), rAve_EE_diff_same_seDays(1,:), 'r.', 'linestyle', 'none')
    errorbar(x+gp, rAve_EE_diff_same_avDays(2,:), rAve_EE_diff_same_seDays(2,:), 'k.', 'linestyle', 'none')
    errorbar(x, rAve_EE_diff_same_shfl_avDays(1,:), rAve_EE_diff_same_shfl_seDays(1,:), 'color',rgb('lightsalmon'), 'marker', '.', 'linestyle', 'none')
    errorbar(x+gp, rAve_EE_diff_same_shfl_avDays(2,:), rAve_EE_diff_same_shfl_seDays(2,:), 'color',rgb('gray'), 'marker', '.', 'linestyle', 'none')
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    title(typeNs)

    %%%%%%%%%% Plot II %%%%%%%%%%%
    typeNs = 'II';    
    subplot(133); hold on
    errorbar(x, rAve_II_diff_same_avDays(1,:), rAve_II_diff_same_seDays(1,:), 'r.', 'linestyle', 'none')
    errorbar(x+gp, rAve_II_diff_same_avDays(2,:), rAve_II_diff_same_seDays(2,:), 'k.', 'linestyle', 'none')
    errorbar(x, rAve_II_diff_same_shfl_avDays(1,:), rAve_II_diff_same_shfl_seDays(1,:), 'color',rgb('lightsalmon'), 'marker', '.', 'linestyle', 'none')
    errorbar(x+gp, rAve_II_diff_same_shfl_avDays(2,:), rAve_II_diff_same_shfl_seDays(2,:), 'color',rgb('gray'), 'marker', '.', 'linestyle', 'none')
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    title(typeNs)


    if saveFigs
        namv = sprintf('corrFR_avePairwise_aveSeDays_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);

        d = fullfile(dirn0fr, 'sumAllMice', nnow);
        if ~exist(d,'dir')
            mkdir(d)
        end
        fn = fullfile(d, namv);

        savefig(gcf, fn)
        % print to pdf
        print('-dpdf', fn)
    end    
    %}

    
    %% Plot errorbar, showing each mouse, ave +/- se of rAve across days (already averaged across neurons for each day)
    %%%%% all pairs, same-tuned pair, diff-tuned pairs
    
    x = 1:length(mice);
    gp = .2; %0; %.2;
    marg = .2;
    mn = min([rAve_EE_shfl_avDays - rAve_EE_shfl_seDays , rAve_EI_shfl_avDays - rAve_EI_shfl_seDays , rAve_II_shfl_avDays - rAve_II_shfl_seDays]); 
    mne = min(rAve_EE_diff_same_shfl_avDays - rAve_EE_diff_same_shfl_seDays); mne = min(mne);
    mnei = min(rAve_EI_diff_same_shfl_avDays - rAve_EI_diff_same_shfl_seDays); mnei = min(mnei);
    mni = min(rAve_II_diff_same_shfl_avDays - rAve_II_diff_same_shfl_seDays); mni = min(mni);
%     mn = min([mn, mne,mnei,mni])
%     mn = mn + mn/5;
    
    figure('name', 'All mice', 'position', [34   661   788   280]); %[39   703   905   236]); 
    
    %%%%%%%%%% Plot EE, II, and EI all pairs %%%%%%%%%%%    
    subplot(141); hold on    
    h1 = errorbar(x, rAve_EE_avDays, rAve_EE_seDays, 'k.', 'linestyle', 'none');
    h2 = errorbar(x+gp, rAve_II_avDays, rAve_II_seDays, 'r.', 'linestyle', 'none');   
    h3 = errorbar(x+2*gp, rAve_EI_avDays, rAve_EI_seDays, 'color', rgb('green'), 'marker', '.', 'linestyle', 'none');
    errorbar(x, rAve_EE_shfl_avDays, rAve_EE_shfl_seDays, 'color',rgb('gray'), 'marker','.', 'linestyle', 'none');
    errorbar(x+gp, rAve_II_shfl_avDays, rAve_II_shfl_seDays, 'color',rgb('lightsalmon'), 'linestyle', 'none');        
    errorbar(x+2*gp, rAve_EI_shfl_avDays, rAve_EI_shfl_seDays, 'color',rgb('lightgreen'), 'linestyle', 'none');        
    xlim([x(1)-marg, x(end)+2*gp+marg])
    yl = get(gca, 'ylim'); ylim([mn-range(yl)/50, yl(2)])
    set(gca,'xtick', x+gp)
    set(gca,'xticklabel', mice)    
    ylabel('corr (mean +/- se days)')
    legend([h1,h2,h3], {'EE','II', 'EI'}, 'location','northoutside')
    set(gca, 'tickdir', 'out')
    
    %%%%%%%%%% Plot EE, same and diff %%%%%%%%%%%
    subplot(142); hold on
    h1 = errorbar(x, rAve_EE_diff_same_avDays(1,:), rAve_EE_diff_same_seDays(1,:), 'color',rgb('gray'), 'marker','.', 'linestyle', 'none');
    h2 = errorbar(x+gp, rAve_EE_diff_same_avDays(2,:), rAve_EE_diff_same_seDays(2,:), 'k.', 'linestyle', 'none');
    errorbar(x, rAve_EE_diff_same_shfl_avDays(1,:), rAve_EE_diff_same_shfl_seDays(1,:), 'color',rgb('gray'), 'marker','.', 'linestyle', 'none');
    errorbar(x+gp, rAve_EE_diff_same_shfl_avDays(2,:), rAve_EE_diff_same_shfl_seDays(2,:), 'k.', 'linestyle', 'none');
    xlim([x(1)-marg, x(end)+gp+marg])
    yl = get(gca, 'ylim'); ylim([mne-range(yl)/50, yl(2)])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    ylabel('corr (mean +/- se days)')
    legend([h1,h2], {'diff','same'}, 'location','northoutside')
    set(gca, 'tickdir', 'out')
    title('EE')
    
    %%%%%%%%%% Plot II, same and diff %%%%%%%%%%%
    subplot(143); hold on
    h1 = errorbar(x, rAve_II_diff_same_avDays(1,:), rAve_II_diff_same_seDays(1,:), 'color',rgb('lightsalmon'), 'marker','.', 'linestyle', 'none');
    h2 = errorbar(x+gp, rAve_II_diff_same_avDays(2,:), rAve_II_diff_same_seDays(2,:), 'r.', 'linestyle', 'none');
    errorbar(x, rAve_II_diff_same_shfl_avDays(1,:), rAve_II_diff_same_shfl_seDays(1,:), 'color',rgb('lightsalmon'), 'marker','.', 'linestyle', 'none');
    errorbar(x+gp, rAve_II_diff_same_shfl_avDays(2,:), rAve_II_diff_same_shfl_seDays(2,:), 'r.', 'linestyle', 'none');            
    xlim([x(1)-marg, x(end)+gp+marg])
    yl = get(gca, 'ylim'); ylim([mni-range(yl)/50, yl(2)])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    ylabel('corr (mean +/- se days)')
    legend([h1,h2], {'diff','same'}, 'location','northoutside')
    set(gca, 'tickdir', 'out')
    title('II')    
    
    %%%%%%%%%% Plot EI, same and diff %%%%%%%%%%%
    subplot(144); hold on
    h1 = errorbar(x, rAve_EI_diff_same_avDays(1,:), rAve_EI_diff_same_seDays(1,:), 'color', rgb('lightgreen'), 'marker','.', 'linestyle', 'none');
    h2 = errorbar(x+gp, rAve_EI_diff_same_avDays(2,:), rAve_EI_diff_same_seDays(2,:), 'color', rgb('green'), 'marker', '.', 'linestyle', 'none');
    errorbar(x, rAve_EI_diff_same_shfl_avDays(1,:), rAve_EI_diff_same_shfl_seDays(1,:), 'color', rgb('lightgreen'), 'marker','.', 'linestyle', 'none');
    errorbar(x+gp, rAve_EI_diff_same_shfl_avDays(2,:), rAve_EI_diff_same_shfl_seDays(2,:), 'color', rgb('green'), 'marker', '.', 'linestyle', 'none');    
    xlim([x(1)-marg, x(end)+gp+marg])
    yl = get(gca, 'ylim'); ylim([mnei-range(yl)/50, yl(2)])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    ylabel('corr (mean +/- se days)')
    legend([h1,h2], {'diff','same'}, 'location','northoutside')
    set(gca, 'tickdir', 'out')
    title('EI')
    
    
    if saveFigs
        namv = sprintf('corrFR_avePairwise_aveSeDays_all_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);

        d = fullfile(dirn0fr, 'sumAllMice', nnow);
        if ~exist(d,'dir')
            mkdir(d)
        end
        fn = fullfile(d, namv);

        savefig(gcf, fn)
        print('-dpdf', fn)
    end
    
    
    
    %% (Same as above but all neurons of all days pooled)... Plot errorbar, showing each mouse, ave +/- se of rAve across all neurons of all days

    x = 1:length(mice);
    gp = 0; 
    marg = .2;

    figure('name', 'All mice', 'position', [14   636   661   290]); 
%     set(gca, 'position', [0.2919    0.1908    0.5229    0.7095])

    %%%%%%%%%% Plot EI %%%%%%%%%%%
    typeNs = 'EI';    
    subplot(131); hold on
    h1 = errorbar(x, rAve_EI_diff_same_avAllNsDays(1,:), rAve_EI_diff_same_seAllNsDays(1,:), 'r.', 'linestyle', 'none');
    h2 = errorbar(x+gp, rAve_EI_diff_same_avAllNsDays(2,:), rAve_EI_diff_same_seAllNsDays(2,:), 'k.', 'linestyle', 'none');
%     errorbar(x, rAve_EI_diff_same_shfl_avAllNsDays(1,:), rAve_EI_diff_same_shfl_seAllNsDays(1,:), 'color',rgb('lightsalmon'), 'marker', '.', 'linestyle', 'none')
%     errorbar(x+gp, rAve_EI_diff_same_shfl_avAllNsDays(2,:), rAve_EI_diff_same_shfl_seAllNsDays(2,:), 'color',rgb('gray'), 'marker', '.', 'linestyle', 'none')
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    title(typeNs)
    ylabel('corr (mean +/- se neurons)')
    legend([h1,h2], {'diff','same'})
    yl1 = get(gca, 'ylim');
    
    %%%%%%%%%% Plot EE %%%%%%%%%%%
    typeNs = 'EE';    
    subplot(132); hold on
    errorbar(x, rAve_EE_diff_same_avAllNsDays(1,:), rAve_EE_diff_same_seAllNsDays(1,:), 'r.', 'linestyle', 'none')
    errorbar(x+gp, rAve_EE_diff_same_avAllNsDays(2,:), rAve_EE_diff_same_seAllNsDays(2,:), 'k.', 'linestyle', 'none')
%     errorbar(x, rAve_EE_diff_same_shfl_avAllNsDays(1,:), rAve_EE_diff_same_shfl_seAllNsDays(1,:), 'color',rgb('lightsalmon'), 'marker', '.', 'linestyle', 'none')
%     errorbar(x+gp, rAve_EE_diff_same_shfl_avAllNsDays(2,:), rAve_EE_diff_same_shfl_seAllNsDays(2,:), 'color',rgb('gray'), 'marker', '.', 'linestyle', 'none')
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    title(typeNs)
    yl2 = get(gca, 'ylim');
    
    %%%%%%%%%% Plot II %%%%%%%%%%%
    typeNs = 'II';    
    subplot(133); hold on
    errorbar(x, rAve_II_diff_same_avAllNsDays(1,:), rAve_II_diff_same_seAllNsDays(1,:), 'r.', 'linestyle', 'none')
    errorbar(x+gp, rAve_II_diff_same_avAllNsDays(2,:), rAve_II_diff_same_seAllNsDays(2,:), 'k.', 'linestyle', 'none')
%     errorbar(x, rAve_II_diff_same_shfl_avAllNsDays(1,:), rAve_II_diff_same_shfl_seAllNsDays(1,:), 'color',rgb('lightsalmon'), 'marker', '.', 'linestyle', 'none')
%     errorbar(x+gp, rAve_II_diff_same_shfl_avAllNsDays(2,:), rAve_II_diff_same_shfl_seAllNsDays(2,:), 'color',rgb('gray'), 'marker', '.', 'linestyle', 'none')
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    title(typeNs)
    yl3 = get(gca, 'ylim');
    
%     mn = min([yl1, yl2]);%, yl3]);
%     mx = max([yl1, yl2]);%, yl3]);
%     subplot(131), ylim([mn,mx])
%     subplot(132), ylim([mn,mx])
%     subplot(133), ylim([mn,mx])

    
    if saveFigs
        namv = sprintf('corrFR_avePairwise_aveSeAllNs_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);

        d = fullfile(dirn0fr, 'sumAllMice', nnow);
        if ~exist(d,'dir')
            mkdir(d)
        end
        fn = fullfile(d, namv);

        savefig(gcf, fn)                
    end    

    
    
    %% Any change in corr_diff and corr_same across days?  compare mean of corr_same and corr_diff across days of training

    for im = 1:length(mice)

        figure('name', mice{im}, 'position', [32         566        1091         408]); 

        %%%% EE, II, EI %%%%
        subplot(244)
        a = plot([rAve_EE_avEachDay{im}; rAve_II_avEachDay{im}; rAve_EI_avEachDay{im}]');
        set(a(1),'color','b')
        set(a(2),'color','r')
        set(a(3),'color','g') 
        legend('EE', 'II', 'EI', 'location', 'best')
        ylabel('corr (avePairWise)')
        xlabel('Training days')        
        xlim([0, numDaysAll(im)+1])
        
        
        %%%%%%%% EI %%%%%%%%
        subplot(241)
        plot([rAve_EI_same_avEachDay{im}; rAve_EI_diff_avEachDay{im}]');
        legend('Same tuning', 'Diff tuning', 'location', 'best')
        ylabel('corr (avePairWise)')
        xlabel('Training days')
        title('EI')
        xlim([0, numDaysAll(im)+1])
        
        subplot(245)
        plot(rAve_EI_same_avEachDay{im} ./ rAve_EI_diff_avEachDay{im});
        ylabel('corrSame/corrDiff')
        xlabel('Training days')
        xlim([0, numDaysAll(im)+1])


        %%%%%%%% EE %%%%%%%%
        subplot(242)
        plot([rAve_EE_same_avEachDay{im}; rAve_EE_diff_avEachDay{im}]');
    %     legend('Same tuning', 'Diff tuning')
        ylabel('corr (avePairWise)')
        xlabel('Training days')
        title('EE')
        xlim([0, numDaysAll(im)+1])
        
        subplot(246)
        plot(rAve_EE_same_avEachDay{im} ./ rAve_EE_diff_avEachDay{im});
        ylabel('corrSame/corrDiff')
        xlabel('Training days')
        xlim([0, numDaysAll(im)+1])


        %%%%%%%% II %%%%%%%%    
        subplot(243)
        plot([rAve_II_same_avEachDay{im}; rAve_II_diff_avEachDay{im}]');
    %     legend('Same tuning', 'Diff tuning')
        ylabel('corr (avePairWise)')
        xlabel('Training days')
        title('II')
        xlim([0, numDaysAll(im)+1])
        
        subplot(247)
        plot(rAve_II_same_avEachDay{im} ./ rAve_II_diff_avEachDay{im});
        ylabel('corrSame/corrDiff')
        xlabel('Training days')
        xlim([0, numDaysAll(im)+1])

        if saveFigs
            namv = sprintf('corrFR_avePairwise_trainingDays_sameDiffTun_FR%s_ROC%s_%s_curr%s_%s_%s', alFR, al, time2an, o2a, mice{im}, nowStr);

            d = fullfile(dirn0fr, mice{im}, nnow);
            if ~exist(d,'dir')
                mkdir(d)
            end
            fn = fullfile(dirn0fr, mice{im}, nnow, namv);

            savefig(gcf, fn)
        end
               
    end


    %% All mice: compare noise corr for low vs high behavioral performance levels 

    perc_thb = [10,90]; %[20,80] # perc_thb = [15,85] # percentiles of behavioral performance for determining low and high performance.

    x = 1:3; %length(mice);
    gp = .2;
    marg = .2;

    figure('name', 'All mice', 'position', [34   661   788   280]);
    
    for im = 1:length(mice)
        mouse = mice{im};    
        
        mn_corr = mnTrNum_allMice{im};
        mn_corr0 = mn_corr; % will be used for the heatmaps of CA for all days; We need to exclude 151023 from fni16, this day had issues! ...  in the mat file of stabTestTrainTimes, its class accur is very high ... and in the mat file of excInh_trainDecoder_eachFrame it is very low ...ANYWAY I ended up removing it from the heatmap of CA for all days!!! but it is included in other analyses!!
%         if strcmp(mouse, 'fni16')
%             tr = find(cellfun(@isempty, strfind(days_allMice{im}, '151023_1'))==0);
%             mn_corr0(tr) = thMinTrs - 1; % # see it will be excluded from analysis!    
%         end
        days2an_heatmap = (mn_corr0 >= thMinTrs);

        %%%%%%%%% set svm_stab mat file name that contains behavioral and class accuracy vars
        [~,~,dirn] = setImagingAnalysisNames(mouse, 'analysis', []);    % dirn = fullfile(dirn0fr, mouse);
        finame = fullfile(dirn, 'svm_stabilityBehCA_*.mat');
        stabBehName = dir(finame);
        [~,i] = sort([stabBehName.datenum], 'descend');
        stabBehName = fullfile(dirn, stabBehName(i).name)

        % load beh vars    
        load(stabBehName, 'behCorr_all')

        %%%%%%%% Compare change in CA (after removing noise corr) for days with low vs high behavioral performance ##################
        a = behCorr_all(days2an_heatmap);    
        thb = prctile(a, perc_thb);

        loBehCorrDays = (a <= thb(1));
        hiBehCorrDays = (a >= thb(2));
        fprintf('%d, %d, num days with low and high beh performance\n', sum(loBehCorrDays), sum(hiBehCorrDays))

        aa = rAve_EE_avEachDay{im}(days2an_heatmap);
        bb = rAve_II_avEachDay{im}(days2an_heatmap);
        cc = rAve_EI_avEachDay{im}(days2an_heatmap);
        
        % set corrs for low and high behavioral performance days
        a = aa(loBehCorrDays); 
        b = bb(loBehCorrDays); 
        c = cc(loBehCorrDays); 
        ah = aa(hiBehCorrDays); % EE
        bh = bb(hiBehCorrDays); % II
        ch = cc(hiBehCorrDays); % EI
        
        % set se of corrs across low and high beh perform days
        as0 = nanstd(a) / sqrt(sum(~isnan(a)));
        bs0 = nanstd(b) / sqrt(sum(~isnan(b)));
        cs0 = nanstd(c) / sqrt(sum(~isnan(c)));
        ahs = nanstd(ah)/ sqrt(sum(~isnan(ah)));
        bhs = nanstd(bh)/ sqrt(sum(~isnan(bh)));
        chs = nanstd(ch)/ sqrt(sum(~isnan(ch)));
        
%         figure; %('name', 'All mice', 'position', [14   636   661   290]); 
%         set(gca, 'position', [0.2919    0.1908    0.5229    0.7095])

        %%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%% 
        %%%%%%%%%%%%%%%%%%%%% PLOT %%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%        
        subplot(1,length(mice),im); 
%         subplot(248)
        hold on

        % EI
        h1 = errorbar(x(1), nanmean(ch), chs, 'color', rgb('green'), 'marker','.', 'linestyle', 'none', 'markersize', 10);
        h2 = errorbar(x(1)+gp, nanmean(c), cs0, 'color', rgb('lightgreen'), 'marker','.', 'linestyle', 'none', 'markersize', 10);        
        % EE
        h11 = errorbar(x(2), nanmean(ah), ahs, 'b.', 'linestyle', 'none', 'markersize', 10);
        h12 = errorbar(x(2)+gp, nanmean(a), as0, 'color', rgb('lightblue'), 'marker','.', 'linestyle', 'none', 'markersize', 10);
        % II
        h11 = errorbar(x(3), nanmean(bh), bhs, 'r.', 'linestyle', 'none', 'markersize', 10);
        h12 = errorbar(x(3)+gp, nanmean(b), bs0, 'color', rgb('lightsalmon'), 'marker','.', 'linestyle', 'none', 'markersize', 10);

        xlim([x(1)-marg, x(end)+gp+marg])
        set(gca,'xtick', x+gp)
        set(gca,'xticklabel', {'EI','EE','II'})
        ylabel('corr (mean +/- se days)')
        legend([h1,h2], {'highBeh','lowBeh'}, 'location', 'northoutside')
        set(gca, 'tickdir', 'out')
        
        
        if saveFigs
            namv = sprintf('corrFR_avePairwise_trainingDays_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);

            d = fullfile(dirn0fr, 'sumAllMice', nnow);
            if ~exist(d,'dir')
                mkdir(d)
            end
            fn = fullfile(d, namv);

            savefig(gcf, fn)
            print('-dpdf', fn)
        end    
    end

    %{
    %%%%% Compare number of pairs with hi_corrDiff and lo_corrDiff across days of training.

    % Set the same corr bins for all mice
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
    bins = [0, .02, r2]; % bins = [0, .2, r2];


    I_inds_corrAveLoDiff = cell(1, length(mice));
    I_inds_corrAveHiDiff = cell(1, length(mice));
    E_inds_corrAveLoDiff = cell(1, length(mice));
    E_inds_corrAveHiDiff = cell(1, length(mice));

    I_inds_corrAveHiSameLoDiff = cell(1, length(mice));
    I_inds_corrAveLoSameHiDiff = cell(1, length(mice));
    E_inds_corrAveHiSameLoDiff = cell(1, length(mice));
    E_inds_corrAveLoSameHiDiff = cell(1, length(mice));

    I_inds_corrAveLoSame = cell(1, length(mice));
    I_inds_corrAveHiSame = cell(1, length(mice));
    E_inds_corrAveLoSame = cell(1, length(mice));
    E_inds_corrAveHiSame = cell(1, length(mice));

    for im = 1:length(mice)    

        dn = mnTrNum_allMice{im} < thMinTrs;
        I_inds_corrAveLoDiff{im}(dn) = nan;
        I_inds_corrAveHiDiff{im}(dn) = nan;
        E_inds_corrAveLoDiff{im}(dn) = nan;
        E_inds_corrAveHiDiff{im}(dn) = nan;

        I_inds_corrAveHiSameLoDiff{im}(dn) = nan;
        I_inds_corrAveLoSameHiDiff{im}(dn) = nan;
        E_inds_corrAveHiSameLoDiff{im}(dn) = nan;
        E_inds_corrAveLoSameHiDiff{im}(dn) = nan;

        I_inds_corrAveLoSame{im}(dn) = nan;
        I_inds_corrAveHiSame{im}(dn) = nan;
        E_inds_corrAveLoSame{im}(dn) = nan;
        E_inds_corrAveHiSame{im}(dn) = nan;


        for iday = 1:numDaysAll(im)
            if mnTrNum_allMice{im}(iday) >= thMinTrs
                % inh
                binsI = bins;
                a = [rAve_Ii_Ei_shfl{im}{iday}, rAve_Ii_Ec_shfl{im}{iday}, rAve_Ic_Ec{im}{iday}, rAve_Ic_Ei{im}{iday}];
                binsI(2) = quantile(a, .95);  fprintf('threshold corr = %.3f\n', binsI(2))    

                % exc
                binsE = bins;
                a = [rAve_Ei_Ii_shfl{im}{iday}, rAve_Ei_Ic_shfl{im}{iday}, rAve_Ec_Ic_shfl{im}{iday}, rAve_Ec_Ii_shfl{im}{iday}];
                binsE(2) = quantile(a, .95);  fprintf('threshold corr = %.3f\n', binsE(2))

                %%%%%%%%%%%%%%%%%%%% Inh %%%%%%%%%%%%%%%%%%%%
                %%%%% Ii
                % same
                [n0, ~, i_rAve_Ii_Ei] = histcounts(rAve_Ii_Ei{im}{iday}, binsI);        
                % diff
                [n0, ~, i_rAve_Ii_Ec] = histcounts(rAve_Ii_Ec{im}{iday}, binsI);
                % i==1: low corr ; % i==2: high corr

                %%%%% Ic
                % same
                [n0, ~, i_rAve_Ic_Ec] = histcounts(rAve_Ic_Ec{im}{iday}, binsI);        
                % diff
                [n0, ~, i_rAve_Ic_Ei] = histcounts(rAve_Ic_Ei{im}{iday}, binsI);


                %%%%%% Now get indeces of neurons that have hi corrAve_same and lo corrAve_diff (or negative corrAve_diff)
                Ii_inds_corrAveLoDiff = sum(i_rAve_Ii_Ec == 1); %(rAve_Ii_Ec{im}{iday}<0); %
                Ic_inds_corrAveLoDiff = sum(i_rAve_Ic_Ei == 1); %(rAve_Ic_Ei{im}{iday}<0); %
                I_inds_corrAveLoDiff{im}(iday) = Ii_inds_corrAveLoDiff + Ic_inds_corrAveLoDiff;

                Ii_inds_corrAveHiSameLoDiff = sum((i_rAve_Ii_Ei == 2) & (i_rAve_Ii_Ec == 1));
                Ic_inds_corrAveHiSameLoDiff = sum((i_rAve_Ic_Ec == 2) & (i_rAve_Ic_Ei == 1)); %(rAve_Ic_Ei{im}{iday}<0); %
                I_inds_corrAveHiSameLoDiff{im}(iday) = Ii_inds_corrAveHiSameLoDiff + Ic_inds_corrAveHiSameLoDiff;

                Ii_inds_corrAveHiSame = sum(i_rAve_Ii_Ei == 2);
                Ic_inds_corrAveHiSame = sum(i_rAve_Ic_Ec == 2); %(rAve_Ic_Ei{im}{iday}<0); %
                I_inds_corrAveHiSame{im}(iday) = Ii_inds_corrAveHiSame + Ic_inds_corrAveHiSame;

                %%%%%% Now get indeces of neurons that have lo corrAve_same and hi corrAve_diff
                Ii_inds_corrAveHiDiff = sum(i_rAve_Ii_Ec == 2);
                Ic_inds_corrAveHiDiff = sum(i_rAve_Ic_Ei == 2);
                I_inds_corrAveHiDiff{im}(iday) = Ii_inds_corrAveHiDiff + Ic_inds_corrAveHiDiff;

                Ii_inds_corrAveLoSameHiDiff = sum((i_rAve_Ii_Ei == 1) & (i_rAve_Ii_Ec == 2));
                Ic_inds_corrAveLoSameHiDiff = sum((i_rAve_Ic_Ec == 1) & (i_rAve_Ic_Ei==2));
                I_inds_corrAveLoSameHiDiff{im}(iday) = Ii_inds_corrAveLoSameHiDiff + Ic_inds_corrAveLoSameHiDiff;

                Ii_inds_corrAveLoSame = sum(i_rAve_Ii_Ei == 1);
                Ic_inds_corrAveLoSame = sum(i_rAve_Ic_Ec == 1);
                I_inds_corrAveLoSame{im}(iday) = Ii_inds_corrAveLoSame + Ic_inds_corrAveLoSame;            


                %%%%%%%%%%%%%%%%%%%% Exc %%%%%%%%%%%%%%%%%%%%
                %%%%% Ei
                % same
                [n0, ~, i_rAve_Ei_Ii] = histcounts(rAve_Ei_Ii{im}{iday}, binsE);        
                % diff
                [n0, ~, i_rAve_Ei_Ic] = histcounts(rAve_Ei_Ic{im}{iday}, binsE);
                % i==1: low corr ; % i==2: high corr

                %%%%% Ec
                % same
                [n0, ~, i_rAve_Ec_Ic] = histcounts(rAve_Ec_Ic{im}{iday}, binsE);        
                % diff
                [n0, ~, i_rAve_Ec_Ii] = histcounts(rAve_Ec_Ii{im}{iday}, binsE);


                %%%%%% Now get indeces of neurons that have hi corrAve_same and lo
                %%%%%% corrAve_diff (or nagative corrAve_diff)
                Ei_inds_corrAveLoDiff = sum(i_rAve_Ei_Ic == 1); %(rAve_Ei_Ic{im}{iday}<0); %
                Ec_inds_corrAveLoDiff = sum(i_rAve_Ec_Ii == 1); %(rAve_Ec_Ii{im}{iday}<0); %
                E_inds_corrAveLoDiff{im}(iday) = Ei_inds_corrAveLoDiff + Ec_inds_corrAveLoDiff;

                Ei_inds_corrAveHiSameLoDiff = sum((i_rAve_Ei_Ii == 2) & (i_rAve_Ei_Ic == 1)); %(rAve_Ei_Ic{im}{iday}<0); %
                Ec_inds_corrAveHiSameLoDiff = sum((i_rAve_Ec_Ic == 2) & (i_rAve_Ec_Ii == 1)); %(rAve_Ec_Ii{im}{iday}<0); %
                E_inds_corrAveHiSameLoDiff{im}(iday) = Ei_inds_corrAveHiSameLoDiff + Ec_inds_corrAveHiSameLoDiff;

                Ei_inds_corrAveHiSame = sum(i_rAve_Ei_Ii == 2); %
                Ec_inds_corrAveHiSame = sum(i_rAve_Ec_Ic == 2); %
                E_inds_corrAveHiSame{im}(iday) = Ei_inds_corrAveHiSame + Ec_inds_corrAveHiSame;


                %%%%%% Now get indeces of neurons that have lo corrAve_same and hi corrAve_diff
                Ei_inds_corrAveHiDiff = sum(i_rAve_Ei_Ic == 2);
                Ec_inds_corrAveHiDiff = sum(i_rAve_Ec_Ii==2);
                E_inds_corrAveHiDiff{im}(iday) = Ei_inds_corrAveHiDiff + Ec_inds_corrAveHiDiff;

                Ei_inds_corrAveLoSameHiDiff = sum((i_rAve_Ei_Ii == 1) & (i_rAve_Ei_Ic == 2));
                Ec_inds_corrAveLoSameHiDiff = sum((i_rAve_Ec_Ic == 1) & (i_rAve_Ec_Ii==2));
                E_inds_corrAveLoSameHiDiff{im}(iday) = Ei_inds_corrAveLoSameHiDiff + Ec_inds_corrAveLoSameHiDiff;

                Ei_inds_corrAveLoSame = sum(i_rAve_Ei_Ii == 1);
                Ec_inds_corrAveLoSame = sum(i_rAve_Ec_Ic == 1);
                E_inds_corrAveLoSame{im}(iday) = Ei_inds_corrAveLoSame + Ec_inds_corrAveLoSame;            

            end
        end
    end


    %%
    for im = 1:length(mice)
    %     fI = figure('name', mice{im}); plot([I_inds_corrAveLoDiff{im};    I_inds_corrAveHiDiff{im}]', 'r.-'); hold on
    %     fE = figure('name', mice{im}); plot([E_inds_corrAveLoDiff{im};    E_inds_corrAveHiDiff{im}]', 'k.-'); hold on

    %     figure(fI); plot([I_inds_corrAveLoSame{im};    I_inds_corrAveHiSame{im}]', 'r.-.'); %); set(gca, 'colororder', ['r','r'])
    %     figure(fE); plot([E_inds_corrAveLoSame{im};    E_inds_corrAveHiSame{im}]', 'k.-.')

    %     figure('name', mice{im}); plot([I_inds_corrAveHiSameLoDiff{im};    I_inds_corrAveLoSameHiDiff{im}]')
    %     figure('name', mice{im}); plot([E_inds_corrAveHiSameLoDiff{im};    E_inds_corrAveLoSameHiDiff{im}]')    

        figure('name', mice{im}); plot([I_inds_corrAveHiSameLoDiff{im}  ./    I_inds_corrAveLoSameHiDiff{im}]')
        hold on; plot([E_inds_corrAveHiSameLoDiff{im}   ./    E_inds_corrAveLoSameHiDiff{im}]')    
        hline(1)

    %     figure('name', mice{im});  hold on;
    %     plot([I_inds_corrAveLoDiff{im} ./ I_inds_corrAveHiDiff{im}]')
    %     plot([E_inds_corrAveLoDiff{im} ./   E_inds_corrAveHiDiff{im}]')

    end


    ILoDiffAve = cellfun(@nanmean, I_inds_corrAveLoDiff);
    IHiDiffAve = cellfun(@nanmean, I_inds_corrAveHiDiff);
    ELoDiffAve = cellfun(@nanmean, E_inds_corrAveLoDiff);
    EHiDiffAve = cellfun(@nanmean, E_inds_corrAveHiDiff);

    ILoSameAve = cellfun(@nanmean, I_inds_corrAveLoSame);
    IHiSameAve = cellfun(@nanmean, I_inds_corrAveHiSame);
    ELoSameAve = cellfun(@nanmean, E_inds_corrAveLoSame);
    EHiSameAve = cellfun(@nanmean, E_inds_corrAveHiSame);

    ILoDiffSd = cellfun(@nanstd, I_inds_corrAveLoDiff);
    IHiDiffSd = cellfun(@nanstd, I_inds_corrAveHiDiff);
    ELoDiffSd = cellfun(@nanstd, E_inds_corrAveLoDiff);
    EHiDiffSd = cellfun(@nanstd, E_inds_corrAveHiDiff);

    ILoSameSd = cellfun(@nanstd, I_inds_corrAveLoSame);
    IHiSameSd = cellfun(@nanstd, I_inds_corrAveHiSame);
    ELoSameSd = cellfun(@nanstd, E_inds_corrAveLoSame);
    EHiSameSd = cellfun(@nanstd, E_inds_corrAveHiSame);

    figure; errorbar([1:4;1:4]', [ILoDiffAve; ILoSameAve]', [ILoDiffSd; ILoSameSd]')
    hold on; errorbar([1:4;1:4]', [IHiDiffAve; IHiSameAve]', [IHiDiffSd; IHiSameSd]')
    %}


    %% Each mouse : plot histograms for each mouse of averaged correlations computed above... comparing corrs for pairs of neurons with same vs opposite tuning for choice

    nBins = 50;
    doSmooth = 0;

    for im = 1:length(mice)
        
        % set the bins ... same for all plots
        ally = [[rAve_EI_same_shfl{im}{:}] , [rAve_EI_diff_shfl{im}{:}] , [rAve_EI_same{im}{:}] , [rAve_EI_diff{im}{:}] , ...
            [rAve_EE_same_shfl{im}{:}] , [rAve_EE_diff_shfl{im}{:}], [rAve_EE_same{im}{:}] , [rAve_EE_diff{im}{:}] , ...
            [rAve_II_same_shfl{im}{:}] , [rAve_II_diff_shfl{im}{:}], [rAve_II_same{im}{:}] , [rAve_II_diff{im}{:}]];
        r1 = round(min(ally)-.05, 1);  %min(ally); %   % round(min(ally), 1); 
        r2 = round(max(ally)+.05, 1);
        bins = r1 : (r2-r1)/nBins : r2;
        
        
        fh = figure('name', mice{im}, 'position', [27         404        1370         521]); %[27   404   419   521]);         

        %%%%%%%%%% Plot EI %%%%%%%%%%%
        typeNs = 'EI';    
        sp = [231,234];
        [~,binsSh] = plotHist([rAve_EI_same_shfl{im}{:}], [rAve_EI_diff_shfl{im}{:}], '', '', {'', ''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins); 
        [~,bins] = plotHist([rAve_EI_same{im}{:}], [rAve_EI_diff{im}{:}], 'Corr coef (avePairWise)', 'Fract Ns', {'same E,I', 'diff E,I'}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
    %     [~,bins] = plotHist([rAve_IE_same{im}{:}], [rAve_IE_diff{im}{:}], 'Corr coef (avePairWise)', 'Fract Ns', {'same E,I', 'diff E,I'}, cols, [], fh, nBins, doSmooth); 
        xlim([binsSh(1),bins(end)])
        subplot(sp(1)), title(typeNs)
        [~, p1] = ttest2([rAve_EI_same{im}{:}], [rAve_EI_diff{im}{:}])
        %{
        if saveFigs
            namv = sprintf('corrFR_avePairwise_%s_sameDiffTun_FR%s_ROC%s_%s_curr%s_%s_%s', typeNs, alFR, al, time2an, o2a, mice{im}, nowStr);
            fn = fullfile(dirn0fr, mice{im}, namv);
            savefig(gcf, fn)

            % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
        end
        %}


        %%%%%%%%%% Plot EE %%%%%%%%%%%
        typeNs = 'EE';
        sp = [232,235];
    %     fh = figure('name', [mice{im}, '-', typeNs], 'position', [27   404   419   521]);         
        [~,binsSh] = plotHist([rAve_EE_same_shfl{im}{:}], [rAve_EE_diff_shfl{im}{:}], '', '', {'', ''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins); 
        [~,bins] = plotHist([rAve_EE_same{im}{:}], [rAve_EE_diff{im}{:}], 'Corr coef (avePairWise)', 'Fract Ns', {'same E,E', 'diff E,E'}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
        xlim([binsSh(1),bins(end)])
        subplot(sp(1)), title(typeNs)
        [~, p2] = ttest2([rAve_EE_same{im}{:}], [rAve_EE_diff{im}{:}])
        %{
        if saveFigs
            namv = sprintf('corrFR_avePairwise_%s_sameDiffTun_FR%s_ROC%s_%s_curr%s_%s_%s', typeNs, alFR, al, time2an, o2a, mice{im}, nowStr);
            fn = fullfile(dirn0fr, mice{im}, namv);
            savefig(gcf, fn)

            % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
        end    
        %}


        %%%%%%%%%% Plot II %%%%%%%%%%%
        typeNs = 'II';
        sp = [233,236];
    %     fh = figure('name', [mice{im}, '-', typeNs], 'position', [27   404   419   521]);         
        [~,binsSh] = plotHist([rAve_II_same_shfl{im}{:}], [rAve_II_diff_shfl{im}{:}], '', '', {'', ''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins); 
        [~,bins] = plotHist([rAve_II_same{im}{:}], [rAve_II_diff{im}{:}], 'Corr coef (avePairWise)', 'Fract Ns', {'same I,I', 'diff I,I'}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
        xlim([binsSh(1),bins(end)])
        subplot(sp(1)), title(typeNs)
        [~, p3] = ttest2([rAve_II_same{im}{:}], [rAve_II_diff{im}{:}])
        title([p1,p2,p3])
        
        if saveFigs
            namv = sprintf('corrFR_avePairwise_distDaysPooled_sameDiffTun_FR%s_ROC%s_%s_curr%s_%s_%s', alFR, al, time2an, o2a, mice{im}, nowStr);

            d = fullfile(dirn0fr, mice{im}, nnow);
            if ~exist(d,'dir')
                mkdir(d)
            end
            fn = fullfile(dirn0fr, mice{im}, nnow, namv);

            savefig(gcf, fn)
            % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
        end    
    end


    %% Plot EE, II, EI: all neurons : Each mouse : plot histograms for each mouse of averaged correlations computed above... comparing corrs for pairs of neurons with same vs opposite tuning for choice

    nBins = 100;
    doSmooth = 1;
    
    for im = 1:length(mice)
        
        % Set the bins ... same for all plots
%         ally = [[rAve_EI_shfl{im}{:}] , [rAve_EI{im}{:}] , [rAve_EE_shfl{im}{:}] , [rAve_EE{im}{:}] , [rAve_II_shfl{im}{:}] , [rAve_II{im}{:}]];
        ally = [[rAve_EI{im}{:}] , [rAve_EE{im}{:}] , [rAve_II{im}{:}]];
        r1 = round(min(ally)-.05, 1);  %min(ally); %   % round(min(ally), 1); 
        r2 = round(max(ally)+.05, 1);
        bins = r1 : (r2-r1)/nBins : r2;
        
        figure('name', mice{im}, 'position', [679   665   355   232]); hold on    
%         fh = figure('name', mice{im}, 'position', [27         404        1370         521]); %[27   404   419   521]);         

        %%%%%%%%%% Plot EE, II, EI: all neurons %%%%%%%%%%%        
        sp = [231,234];
%         [~,binsSh] = plotHist([rAve_EE_shfl{im}{:}], [rAve_II_shfl{im}{:}], '', '', {'', ''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins); 
        [~,bins,ye,yi,x,he,hi] = plotHist([rAve_EE{im}{:}], [rAve_II{im}{:}], 'Corr coef (avePairWise)', 'Fract Ns', {'EE', 'II'}, cols, [], nan, nBins, doSmooth, linestylesData, sp, bins);     
%         [~,binsSh] = plotHist([rAve_EI_shfl{im}{:}], [], '', '', {'', ''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins); 
        [~,~,yei,~,x,hei] = plotHist([rAve_EI{im}{:}], [], 'Corr coef (avePairWise)', 'Fract Ns', {'EI', ''}, [[.5, .5, .5];[1,1,1]], [], nan, nBins, doSmooth, linestylesData, sp, bins); 
%         legend([he,hi,hei], {'EE', 'II', 'EI'})        
%         xlim([binsSh(1),bins(end)])        
%         [~, p1] = ttest2([rAve_EE{im}{:}], [rAve_II{im}{:}])
        
        
        %%%%%%%%%%% Plot hists and shade the significant areas %%%%%%%%%%%
        sh = [rAve_EE_shfl{im}{:}]; 
        m = nanmean(sh);
        s = nanstd(sh); 
        mse = [m-2*s , m+2*s];   
        
        sh = [rAve_II_shfl{im}{:}]; 
        m = nanmean(sh);
        s = nanstd(sh); 
        msi = [m-2*s m+2*s];
        
        sh = [rAve_EI_shfl{im}{:}]; 
        m = nanmean(sh);
        s = nanstd(sh); 
        msei = [m-2*s , m+2*s];         
    
        % EE
        H1 = area(x,ye,'FaceColor',[1 1 1], 'facealpha', 0);
        idx = x < mse(1);
        H1 = area(x(idx),ye(idx), 'facecolor', 'k', 'facealpha', .2);
        idx = x > mse(2);
        H = area(x(idx),ye(idx), 'facecolor', 'k', 'facealpha', .2);
        % II
        H11 = area(x,yi,'FaceColor',[1 1 1], 'edgecolor', 'r', 'facealpha', 0);
        idx = x < msi(1);
        H11 = area(x(idx),yi(idx), 'facecolor', 'r', 'edgecolor', 'r', 'facealpha', .2);
        idx = x > msi(2);
        H = area(x(idx),yi(idx), 'facecolor', 'r', 'edgecolor', 'r', 'facealpha', .2);
        % EI
        H12 = area(x,yei,'FaceColor',[1 1 1], 'edgecolor', rgb('gray'), 'facealpha', 0);
        idx = x < msei(1);
        H = area(x(idx),yei(idx), 'facecolor', rgb('gray'), 'edgecolor', rgb('gray'), 'facealpha', .2);
        idx = x > msei(2);
        H12 = area(x(idx),yei(idx), 'facecolor', rgb('gray'), 'edgecolor', rgb('gray'), 'facealpha', .2);
        
        xlabel('Corr coef (avePairWise)')
        ylabel('Fract Ns');
        legend([H1, H11, H12], {'EE', 'II', 'EI'}, 'box', 'off')
        set(gca, 'tickdir', 'out')
        
        if saveFigs
            namv = sprintf('corrFR_avePairwise_distDaysPooled_allEE_allII_FR%s_ROC%s_%s_curr%s_%s_%s', alFR, al, time2an, o2a, mice{im}, nowStr);

            d = fullfile(dirn0fr, mice{im}, nnow);
            if ~exist(d,'dir')
                mkdir(d)
            end
            fn = fullfile(dirn0fr, mice{im}, nnow, namv);

            savefig(gcf, fn)
            % print to pdf
            print('-dpdf', fn)
        end   
    end
    
    
    
    
    %% All mice: plot histograms of all mice neurons, comparing corr between same vs oppositely tuned neurons

    nBins = 50;
    doSmooth = 0;

    % set the bins ... same for all plots
    ally = [rAve_EI_same_allMiceNs_shfl, rAve_EI_diff_allMiceNs_shfl , rAve_EI_same_allMiceNs, rAve_EI_diff_allMiceNs , rAve_EE_same_allMiceNs_shfl, rAve_EE_diff_allMiceNs_shfl , rAve_EE_same_allMiceNs, rAve_EE_diff_allMiceNs , rAve_II_same_allMiceNs_shfl, rAve_II_diff_allMiceNs_shfl , rAve_II_same_allMiceNs, rAve_II_diff_allMiceNs];
    r1 = round(min(ally)-.05, 1);  %min(ally); %   % round(min(ally), 1);
    r2 = round(max(ally)+.05, 1);
    bins = r1 : (r2-r1)/nBins : r2;
     
    
    fh = figure('name', 'All mice', 'position', [27  404  1370  521]);

    typeNs = 'EI';
    % fh = figure('name', ['All Mice-',typeNs], 'position', [27   404   419   521]);         
    sp = [231,234];
    [~,binsSh] = plotHist(rAve_EI_same_allMiceNs_shfl, rAve_EI_diff_allMiceNs_shfl, '', '', {'', ''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins); 
    [~,bins] = plotHist(rAve_EI_same_allMiceNs, rAve_EI_diff_allMiceNs, 'Corr coef (avePairWise)', 'Fract Ns', {'same E,I', 'diff E,I'}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
    xlim([binsSh(1),bins(end)])
    subplot(sp(1)), title(typeNs)
    [~, p1] = ttest2(rAve_EI_same_allMiceNs, rAve_EI_diff_allMiceNs);
    %{
    if saveFigs
        namv = sprintf('corrFR_avePairwise_%s_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', typeNs, alFR, al, time2an, o2a, nowStr);
        fn = fullfile(dirn0fr, namv);
        savefig(gcf, fn)
        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end    
    %}

    typeNs = 'EE';
    % fh = figure('name', ['All Mice-',typeNs], 'position', [27   404   419   521]);         
    sp = [232,235];
    [~,binsSh] = plotHist(rAve_EE_same_allMiceNs_shfl, rAve_EE_diff_allMiceNs_shfl, '', '', {'', ''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins); 
    [~,bins] = plotHist(rAve_EE_same_allMiceNs, rAve_EE_diff_allMiceNs, 'Corr coef (avePairWise)', 'Fract Ns', {'same E,E', 'diff E,E'}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
    xlim([binsSh(1),bins(end)])
    subplot(sp(1)), title(typeNs)
    [~, p2] = ttest2(rAve_EE_same_allMiceNs, rAve_EE_diff_allMiceNs);
    %{
    if saveFigs
        namv = sprintf('corrFR_avePairwise_%s_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', typeNs, alFR, al, time2an, o2a, nowStr);
        fn = fullfile(dirn0fr, namv);
        savefig(gcf, fn)
        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end    
    %}

    typeNs = 'II';
    % fh = figure('name', ['All Mice-',typeNs], 'position', [27   404   419   521]);         
    sp = [233,236];
    [~,binsSh] = plotHist(rAve_II_same_allMiceNs_shfl, rAve_II_diff_allMiceNs_shfl, '', '', {'', ''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins); 
    [~,bins] = plotHist(rAve_II_same_allMiceNs, rAve_II_diff_allMiceNs, 'Corr coef (avePairWise)', 'Fract Ns', {'same I,I', 'diff I,I'}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
    xlim([binsSh(1),bins(end)])
    subplot(sp(1)), title(typeNs)
    [~, p3] = ttest2(rAve_II_same_allMiceNs, rAve_II_diff_allMiceNs);
    title([p1,p2,p3])

    if saveFigs
        namv = sprintf('corrFR_avePairwise_distMiceDaysPooled_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);

        d = fullfile(dirn0fr, 'sumAllMice', nnow);
        fn = fullfile(d, namv);

        savefig(gcf, fn)
    end    

    
    %% Plot EE, II, EI: all neurons :  All mice: plot histograms of all mice neurons, comparing corr between same vs oppositely tuned neurons

    nBins = 100;
    doSmooth = 1;

    % Set the bins ... same for all plots
    ally = [rAve_EI_allMiceNs, rAve_EE_allMiceNs, rAve_II_allMiceNs];
    r1 = round(min(ally)-.05, 1);  %min(ally); %   % round(min(ally), 1); 
    r2 = round(max(ally)+.05, 1);
    bins = r1 : (r2-r1)/nBins : r2;
    
    fh = figure('name', 'All mice', 'position', [679   665   355   232]); hold on

    %%%%%%%%%% Plot EE, II, EI: all neurons %%%%%%%%%%%        
    sp = [231,234];
%         [~,binsSh] = plotHist([rAve_EE_shfl{im}{:}], [rAve_II_shfl{im}{:}], '', '', {'', ''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins); 
    [~,bins,ye,yi,x,he,hi] = plotHist(rAve_EE_allMiceNs, rAve_II_allMiceNs, 'Corr coef (avePairWise)', 'Fract Ns', {'EE', 'II'}, cols, [], nan, nBins, doSmooth, linestylesData, sp, bins);     
%         [~,binsSh] = plotHist([rAve_EI_shfl{im}{:}], [], '', '', {'', ''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins); 
    [~,~,yei,~,x,hei] = plotHist(rAve_EI_allMiceNs, [], 'Corr coef (avePairWise)', 'Fract Ns', {'EI', ''}, [[.5, .5, .5];[1,1,1]], [], nan, nBins, doSmooth, linestylesData, sp, bins); 
%         legend([he,hi,hei], {'EE', 'II', 'EI'})        
%         xlim([binsSh(1),bins(end)])        
%         [~, p1] = ttest2([rAve_EE{im}{:}], [rAve_II{im}{:}])


    %%%%%%%%%%% Plot hists and shade the significant areas %%%%%%%%%%%
    s = [rAve_EE_allMiceNs_shfl]; 
    m = nanmean(s);
    s = nanstd(s); 
    mse = [m-2*s , m+2*s];         
    s = [rAve_II_allMiceNs_shfl]; 
    m = nanmean(s);
    s = nanstd(s); 
    msi = [mi-2*si mi+2*si];
    s = [rAve_EI_allMiceNs_shfl]; 
    m = nanmean(s);
    s = nanstd(s); 
    msei = [m-2*s , m+2*s];         

    % EE
    H1 = area(x,ye,'FaceColor',[1 1 1], 'facealpha', 0);
    idx = x < mse(1);
    H1 = area(x(idx),ye(idx), 'facecolor', 'k', 'facealpha', .2);
    idx = x > mse(2);
    H = area(x(idx),ye(idx), 'facecolor', 'k', 'facealpha', .2);
    % II
    H11 = area(x,yi,'FaceColor',[1 1 1], 'edgecolor', 'r', 'facealpha', 0);
    idx = x < msi(1);
    H11 = area(x(idx),yi(idx), 'facecolor', 'r', 'edgecolor', 'r', 'facealpha', .2);
    idx = x > msi(2);
    H = area(x(idx),yi(idx), 'facecolor', 'r', 'edgecolor', 'r', 'facealpha', .2);
    % EI
    H12 = area(x,yei,'FaceColor',[1 1 1], 'edgecolor', rgb('gray'), 'facealpha', 0);
    idx = x < msei(1);
    H = area(x(idx),yei(idx), 'facecolor', rgb('gray'), 'edgecolor', rgb('gray'), 'facealpha', .2);
    idx = x > msei(2);
    H12 = area(x(idx),yei(idx), 'facecolor', rgb('gray'), 'edgecolor', rgb('gray'), 'facealpha', .2);

    xlabel('Corr coef (avePairWise)')
    ylabel('Fract Ns');
    legend([H1, H11, H12], {'EE', 'II', 'EI'}, 'box', 'off')
    set(gca, 'tickdir', 'out')

    if saveFigs
        namv = sprintf('corrFR_avePairwise_distMiceDaysPooled_allEE_allII_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);

        d = fullfile(dirn0fr, 'sumAllMice', nnow);
        fn = fullfile(d, namv);

        savefig(gcf, fn)
        % print to pdf
        print('-dpdf', fn)               
    end  
        
        
end





%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%  Set pairwise correlations between FR of pairs of neurons  %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We don't average pairwise correlations (unlike what we did above). 
% We get a lot of negative correlations in this way... neurons are not
% firing much so makes sense that corrs will become negative.


%% Set the upper triangular of Ei_Ei, etc to nan (bc they are symmetric)
% remember you don't want to have the upper triu as nan for the above
% sections ... but here you change the values of r_Ei_Ei, etc ...

for im = 1:length(mice)
    for iday = 1:numDaysAll(im)
        if mnTrNum_allMice{im}(iday) >= thMinTrs

            r_Ei_Ei{im}{iday} = triu_nan(r_Ei_Ei{im}{iday});
            r_Ec_Ec{im}{iday} = triu_nan(r_Ec_Ec{im}{iday});
            r_Ii_Ii{im}{iday} = triu_nan(r_Ii_Ii{im}{iday});
            r_Ic_Ic{im}{iday} = triu_nan(r_Ic_Ic{im}{iday});
            % set diagonal to nan
%             r_Ei_Ei{im}{iday}(logical(eye(size(r_Ei_Ei{im}{iday})))) = nan; 
%             r_Ec_Ec{im}{iday}(logical(eye(size(r_Ec_Ec{im}{iday})))) = nan;
%             r_Ii_Ii{im}{iday}(logical(eye(size(r_Ii_Ii{im}{iday})))) = nan;
%             r_Ic_Ic{im}{iday}(logical(eye(size(r_Ic_Ic{im}{iday})))) = nan;

            %%%% shfl
            r_Ei_Ei_shfl{im}{iday} = triu_nan(r_Ei_Ei_shfl{im}{iday});
            r_Ec_Ec_shfl{im}{iday} = triu_nan(r_Ec_Ec_shfl{im}{iday});
            r_Ii_Ii_shfl{im}{iday} = triu_nan(r_Ii_Ii_shfl{im}{iday});
            r_Ic_Ic_shfl{im}{iday} = triu_nan(r_Ic_Ic_shfl{im}{iday});            
        end
    end
end



%% Pool the pairwise corrs into a vertical vector (that includes all possible pairs of neurons)

r_Ei_Ei_alln = cell(1,length(mice));
r_Ec_Ec_alln = cell(1,length(mice));
r_Ei_Ec_alln = cell(1,length(mice));
r_Ii_Ii_alln = cell(1,length(mice));
r_Ic_Ic_alln = cell(1,length(mice));
r_Ii_Ic_alln = cell(1,length(mice));
r_Ei_Ii_alln = cell(1,length(mice));
r_Ec_Ic_alln = cell(1,length(mice));
r_Ei_Ic_alln = cell(1,length(mice));
r_Ec_Ii_alln = cell(1,length(mice));

r_EI_same_alln = cell(1,length(mice));
r_EI_diff_alln = cell(1,length(mice));
r_EE_same_alln = cell(1,length(mice));
r_EE_diff_alln = cell(1,length(mice));
r_II_same_alln = cell(1,length(mice));
r_II_diff_alln = cell(1,length(mice));
            
p_Ei_Ei_alln = cell(1,length(mice));
p_Ec_Ec_alln = cell(1,length(mice));
p_Ei_Ec_alln = cell(1,length(mice));
p_Ii_Ii_alln = cell(1,length(mice));
p_Ic_Ic_alln = cell(1,length(mice));
p_Ii_Ic_alln = cell(1,length(mice));
p_Ei_Ii_alln = cell(1,length(mice));
p_Ec_Ic_alln = cell(1,length(mice));
p_Ei_Ic_alln = cell(1,length(mice));
p_Ec_Ii_alln = cell(1,length(mice));

% shfl
r_Ei_Ei_alln_shfl = cell(1,length(mice));
r_Ec_Ec_alln_shfl = cell(1,length(mice));
r_Ei_Ec_alln_shfl = cell(1,length(mice));
r_Ii_Ii_alln_shfl = cell(1,length(mice));
r_Ic_Ic_alln_shfl = cell(1,length(mice));
r_Ii_Ic_alln_shfl = cell(1,length(mice));
r_Ei_Ii_alln_shfl = cell(1,length(mice));
r_Ec_Ic_alln_shfl = cell(1,length(mice));
r_Ei_Ic_alln_shfl = cell(1,length(mice));
r_Ec_Ii_alln_shfl = cell(1,length(mice));

r_EI_same_alln_shfl = cell(1,length(mice));
r_EI_diff_alln_shfl = cell(1,length(mice));
r_EE_same_alln_shfl = cell(1,length(mice));
r_EE_diff_alln_shfl = cell(1,length(mice));
r_II_same_alln_shfl = cell(1,length(mice));
r_II_diff_alln_shfl = cell(1,length(mice));

for im = 1:length(mice)
    for iday = 1:numDaysAll(im)
        if mnTrNum_allMice{im}(iday) >= thMinTrs
            r_Ei_Ei_alln{im}{iday} = r_Ei_Ei{im}{iday}(:)';
            r_Ec_Ec_alln{im}{iday} = r_Ec_Ec{im}{iday}(:)';
            r_Ei_Ec_alln{im}{iday} = r_Ei_Ec{im}{iday}(:)';
            r_Ii_Ii_alln{im}{iday} = r_Ii_Ii{im}{iday}(:)';
            r_Ic_Ic_alln{im}{iday} = r_Ic_Ic{im}{iday}(:)';
            r_Ii_Ic_alln{im}{iday} = r_Ii_Ic{im}{iday}(:)';            
            r_Ei_Ii_alln{im}{iday} = r_Ei_Ii{im}{iday}(:)';
            r_Ec_Ic_alln{im}{iday} = r_Ec_Ic{im}{iday}(:)';
            r_Ei_Ic_alln{im}{iday} = r_Ei_Ic{im}{iday}(:)';
            r_Ec_Ii_alln{im}{iday} = r_Ec_Ii{im}{iday}(:)';            
            % pool same, diff
            r_EI_same_alln{im}{iday} = [r_Ei_Ii_alln{im}{iday}, r_Ec_Ic_alln{im}{iday}];
            r_EI_diff_alln{im}{iday} = [r_Ei_Ic_alln{im}{iday}, r_Ec_Ii_alln{im}{iday}];  
            r_EE_same_alln{im}{iday} = [r_Ei_Ei_alln{im}{iday}, r_Ec_Ec_alln{im}{iday}];
            r_EE_diff_alln{im}{iday} = [r_Ei_Ec_alln{im}{iday}];            
            r_II_same_alln{im}{iday} = [r_Ii_Ii_alln{im}{iday}, r_Ic_Ic_alln{im}{iday}];
            r_II_diff_alln{im}{iday} = [r_Ii_Ic_alln{im}{iday}];                        
            
            p_Ei_Ei_alln{im}{iday} = p_Ei_Ei{im}{iday}(:)';
            p_Ec_Ec_alln{im}{iday} = p_Ec_Ec{im}{iday}(:)';
            p_Ei_Ec_alln{im}{iday} = p_Ei_Ec{im}{iday}(:)';
            p_Ii_Ii_alln{im}{iday} = p_Ii_Ii{im}{iday}(:)';
            p_Ic_Ic_alln{im}{iday} = p_Ic_Ic{im}{iday}(:)';
            p_Ii_Ic_alln{im}{iday} = p_Ii_Ic{im}{iday}(:)';            
            p_Ei_Ii_alln{im}{iday} = p_Ei_Ii{im}{iday}(:)';
            p_Ec_Ic_alln{im}{iday} = p_Ec_Ic{im}{iday}(:)';
            p_Ei_Ic_alln{im}{iday} = p_Ei_Ic{im}{iday}(:)';
            p_Ec_Ii_alln{im}{iday} = p_Ec_Ii{im}{iday}(:)';  
            
            
            %%%%% shfl
            r_Ei_Ei_alln_shfl{im}{iday} = r_Ei_Ei_shfl{im}{iday}(:)';
            r_Ec_Ec_alln_shfl{im}{iday} = r_Ec_Ec_shfl{im}{iday}(:)';
            r_Ei_Ec_alln_shfl{im}{iday} = r_Ei_Ec_shfl{im}{iday}(:)';
            r_Ii_Ii_alln_shfl{im}{iday} = r_Ii_Ii_shfl{im}{iday}(:)';
            r_Ic_Ic_alln_shfl{im}{iday} = r_Ic_Ic_shfl{im}{iday}(:)';
            r_Ii_Ic_alln_shfl{im}{iday} = r_Ii_Ic_shfl{im}{iday}(:)';            
            r_Ei_Ii_alln_shfl{im}{iday} = r_Ei_Ii_shfl{im}{iday}(:)';
            r_Ec_Ic_alln_shfl{im}{iday} = r_Ec_Ic_shfl{im}{iday}(:)';
            r_Ei_Ic_alln_shfl{im}{iday} = r_Ei_Ic_shfl{im}{iday}(:)';
            r_Ec_Ii_alln_shfl{im}{iday} = r_Ec_Ii_shfl{im}{iday}(:)';                        
            
            % pool same, diff
            r_EI_same_alln_shfl{im}{iday} = [r_Ei_Ii_alln_shfl{im}{iday}, r_Ec_Ic_alln_shfl{im}{iday}];
            r_EI_diff_alln_shfl{im}{iday} = [r_Ei_Ic_alln_shfl{im}{iday}, r_Ec_Ii_alln_shfl{im}{iday}];  
            r_EE_same_alln_shfl{im}{iday} = [r_Ei_Ei_alln_shfl{im}{iday}, r_Ec_Ec_alln_shfl{im}{iday}];
            r_EE_diff_alln_shfl{im}{iday} = [r_Ei_Ec_alln_shfl{im}{iday}];            
            r_II_same_alln_shfl{im}{iday} = [r_Ii_Ii_alln_shfl{im}{iday}, r_Ic_Ic_alln_shfl{im}{iday}];
            r_II_diff_alln_shfl{im}{iday} = [r_Ii_Ic_alln_shfl{im}{iday}];              
        end
    end
end


%% 
%%%%%%%%%%%%%%% Pairwair corrs averaged per day %%%%%%%%%%%%%%%

%% For each day, average corr across all neuron pairs, done for each population seperately.

r_EI_same_aveDays = cell(1,length(mice));
r_EI_diff_aveDays = cell(1,length(mice));
r_EE_same_aveDays = cell(1,length(mice));
r_EE_diff_aveDays = cell(1,length(mice));
r_II_same_aveDays = cell(1,length(mice));
r_II_diff_aveDays = cell(1,length(mice));
% r_EI_same_seDays = cell(1,length(mice));
% r_EI_diff_seDays = cell(1,length(mice));
% r_EE_same_seDays = cell(1,length(mice));
% r_EE_diff_seDays = cell(1,length(mice));
% r_II_same_seDays = cell(1,length(mice));
% r_II_diff_seDays = cell(1,length(mice));

% shfl
r_EI_same_shfl_aveDays = cell(1,length(mice));
r_EI_diff_shfl_aveDays = cell(1,length(mice));
r_EE_same_shfl_aveDays = cell(1,length(mice));
r_EE_diff_shfl_aveDays = cell(1,length(mice));
r_II_same_shfl_aveDays = cell(1,length(mice));
r_II_diff_shfl_aveDays = cell(1,length(mice));
% r_EI_same_shfl_seDays = cell(1,length(mice));
% r_EI_diff_shfl_seDays = cell(1,length(mice));
% r_EE_same_shfl_seDays = cell(1,length(mice));
% r_EE_diff_shfl_seDays = cell(1,length(mice));
% r_II_same_shfl_seDays = cell(1,length(mice));
% r_II_diff_shfl_seDays = cell(1,length(mice));

for im = 1:length(mice)
    r_EI_same_aveDays{im} = cellfun(@nanmean, r_EI_same_alln{im});
    r_EI_diff_aveDays{im} = cellfun(@nanmean, r_EI_diff_alln{im});
    r_EE_same_aveDays{im} = cellfun(@nanmean, r_EE_same_alln{im});
    r_EE_diff_aveDays{im} = cellfun(@nanmean, r_EE_diff_alln{im});
    r_II_same_aveDays{im} = cellfun(@nanmean, r_II_same_alln{im});
    r_II_diff_aveDays{im} = cellfun(@nanmean, r_II_diff_alln{im});
    
%     r_EI_same_seDays{im} = cellfun(@nanstd, r_EI_same_alln{im}) / sqrt(numDaysGood(im));
%     r_EI_diff_seDays{im} = cellfun(@nanstd, r_EI_diff_alln{im}) / sqrt(numDaysGood(im));
%     r_EE_same_seDays{im} = cellfun(@nanstd, r_EE_same_alln{im}) / sqrt(numDaysGood(im));
%     r_EE_diff_seDays{im} = cellfun(@nanstd, r_EE_diff_alln{im}) / sqrt(numDaysGood(im));
%     r_II_same_seDays{im} = cellfun(@nanstd, r_II_same_alln{im}) / sqrt(numDaysGood(im));
%     r_II_diff_seDays{im} = cellfun(@nanstd, r_II_diff_alln{im}) / sqrt(numDaysGood(im));
     
    % shfl
    r_EI_same_shfl_aveDays{im} = cellfun(@nanmean, r_EI_same_alln_shfl{im});
    r_EI_diff_shfl_aveDays{im} = cellfun(@nanmean, r_EI_diff_alln_shfl{im});
    r_EE_same_shfl_aveDays{im} = cellfun(@nanmean, r_EE_same_alln_shfl{im});
    r_EE_diff_shfl_aveDays{im} = cellfun(@nanmean, r_EE_diff_alln_shfl{im});
    r_II_same_shfl_aveDays{im} = cellfun(@nanmean, r_II_same_alln_shfl{im});
    r_II_diff_shfl_aveDays{im} = cellfun(@nanmean, r_II_diff_alln_shfl{im});    

%     r_EI_same_shfl_seDays{im} = cellfun(@nanstd, r_EI_same_alln_shfl{im}) / sqrt(numDaysGood(im));
%     r_EI_diff_shfl_seDays{im} = cellfun(@nanstd, r_EI_diff_alln_shfl{im}) / sqrt(numDaysGood(im));
%     r_EE_same_shfl_seDays{im} = cellfun(@nanstd, r_EE_same_alln_shfl{im}) / sqrt(numDaysGood(im));
%     r_EE_diff_shfl_seDays{im} = cellfun(@nanstd, r_EE_diff_alln_shfl{im}) / sqrt(numDaysGood(im));
%     r_II_same_shfl_seDays{im} = cellfun(@nanstd, r_II_same_alln_shfl{im}) / sqrt(numDaysGood(im));
%     r_II_diff_shfl_seDays{im} = cellfun(@nanstd, r_II_diff_alln_shfl{im}) / sqrt(numDaysGood(im));    

end


%%
%%%%%%%%%%%%%%%%%%%%% Plots : pairwise averaged per day %%%%%%%%%%%%%%%%%%%%%%%%%%%
if doPlots
    
    %%% Plot errorbar, showing each mouse, ave +/- se of r_aveDays across days (already averaged across PW corrs for each day)

    x = 1:length(mice);
    gp = .2;
    marg = .2;

    figure('name', 'All mice', 'position', [14   636   661   290]); 
    set(gca, 'position', [0.2919    0.1908    0.5229    0.7095])

    %%%%%%%%%% Plot EI %%%%%%%%%%%
    typeNs = 'EI';    
    subplot(131); hold on
    h1 = errorbar(x, cellfun(@nanmean, r_EI_diff_aveDays), cellfun(@nanstd, r_EI_diff_aveDays)./sqrt(numDaysGood), 'r.', 'linestyle', 'none');
    h2 = errorbar(x+gp, cellfun(@nanmean, r_EI_same_aveDays), cellfun(@nanstd, r_EI_same_aveDays)./sqrt(numDaysGood), 'k.', 'linestyle', 'none');
    % shfl
    errorbar(x, cellfun(@nanmean, r_EI_diff_shfl_aveDays), cellfun(@nanstd, r_EI_diff_shfl_aveDays)./sqrt(numDaysGood), 'color',rgb('lightsalmon'), 'marker', '.', 'linestyle', 'none')
    errorbar(x+gp, cellfun(@nanmean, r_EI_same_shfl_aveDays), cellfun(@nanstd, r_EI_same_shfl_aveDays)./sqrt(numDaysGood), 'color',rgb('gray'), 'marker', '.', 'linestyle', 'none')
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    title(typeNs)
    ylabel('corr (mean +/- se days)')
    legend([h1,h2], {'diff','same'})

    %%%%%%%%%% Plot EE %%%%%%%%%%%
    typeNs = 'EE';    
    subplot(132); hold on
    errorbar(x, cellfun(@nanmean, r_EE_diff_aveDays), cellfun(@nanstd, r_EE_diff_aveDays)./sqrt(numDaysGood), 'r.', 'linestyle', 'none')
    errorbar(x+gp, cellfun(@nanmean, r_EE_same_aveDays), cellfun(@nanstd, r_EE_same_aveDays)./sqrt(numDaysGood), 'k.', 'linestyle', 'none')
    % shfl
    errorbar(x, cellfun(@nanmean, r_EE_diff_shfl_aveDays), cellfun(@nanstd, r_EE_diff_shfl_aveDays)./sqrt(numDaysGood), 'color',rgb('lightsalmon'), 'marker', '.', 'linestyle', 'none')
    errorbar(x+gp, cellfun(@nanmean, r_EE_same_shfl_aveDays), cellfun(@nanstd, r_EE_same_shfl_aveDays)./sqrt(numDaysGood), 'color',rgb('gray'), 'marker', '.', 'linestyle', 'none')
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    title(typeNs)

    %%%%%%%%%% Plot II %%%%%%%%%%%
    typeNs = 'II';    
    subplot(133); hold on
    errorbar(x, cellfun(@nanmean, r_II_diff_aveDays), cellfun(@nanstd, r_II_diff_aveDays)./sqrt(numDaysGood), 'r.', 'linestyle', 'none')
    errorbar(x+gp, cellfun(@nanmean, r_II_same_aveDays), cellfun(@nanstd, r_II_same_aveDays)./sqrt(numDaysGood), 'k.', 'linestyle', 'none')
    % shfl
    errorbar(x, cellfun(@nanmean, r_II_diff_shfl_aveDays), cellfun(@nanstd, r_II_diff_shfl_aveDays)./sqrt(numDaysGood), 'color',rgb('lightsalmon'), 'marker', '.', 'linestyle', 'none')
    errorbar(x+gp, cellfun(@nanmean, r_II_same_shfl_aveDays), cellfun(@nanstd, r_II_same_shfl_aveDays)./sqrt(numDaysGood), 'color',rgb('gray'), 'marker', '.', 'linestyle', 'none')
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    title(typeNs)


    if saveFigs
        namv = sprintf('corrFR_pairwise_rsAvedPerDay_aveSeDays_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);

        d = fullfile(dirn0fr, 'sumAllMice', nnow);
        if ~exist(d,'dir')
            mkdir(d)
        end
        fn = fullfile(d, namv);

        savefig(gcf, fn)

        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end    


    %% Each mouse : plot histogram of correlations  (averaged across pair of neurons for each day), all days pooled

    nBins = 50;
    doSmooth = 0;
    ylab = 'Fraction days';
    xlab = 'Corr coef (avePerDay)';

    for im = 1:length(mice)    

        fh = figure('name', mice{im}, 'position', [27         404        1370         521]);

        % set the bins ... same for all plots
        ally = [r_EI_same_shfl_aveDays{im}, r_EI_diff_shfl_aveDays{im}, r_EI_same_aveDays{im}, r_EI_diff_aveDays{im}, ...
            r_EE_same_shfl_aveDays{im}, r_EE_diff_shfl_aveDays{im}, r_EE_same_aveDays{im}, r_EE_diff_aveDays{im}, ...
            r_II_same_shfl_aveDays{im}, r_II_diff_shfl_aveDays{im}, r_II_same_aveDays{im}, r_II_diff_aveDays{im}];
        r1 = round(min(ally)-.05, 1);  %min(ally); %   % round(min(ally), 1); 
        r2 = round(max(ally)+.05, 1);
        bins = r1 : (r2-r1)/nBins : r2;
        
        
        % 'EI\_same','EI\_diff'
        sp = [231,234];
        plotHist(r_EI_same_shfl_aveDays{im}, r_EI_diff_shfl_aveDays{im},'','', {'',''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins); 
        plotHist(r_EI_same_aveDays{im}, r_EI_diff_aveDays{im},xlab,ylab, {'EI\_same','EI\_diff'}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
        % mark corr significancy value 
    %     corrSig = min([min(abs(r_EI_same_aveDays{im}(p_EI_same_aveDays{im}<=.05))), min(abs(r_EI_diff_aveDays{im}(p_EI_diff_aveDays{im}<=.05)))]);
    %     subplot(sp(1)), title('EI')
    %     vline(corrSig, 'k:')
    %     vline(-corrSig, 'k:')

        % 'EE\_same','EE\_diff'
        sp = [232,235];
        plotHist(r_EE_same_shfl_aveDays{im}, r_EE_diff_shfl_aveDays{im},'','', {'',''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins);     
        plotHist(r_EE_same_aveDays{im}, r_EE_diff_aveDays{im},xlab,ylab, {'EE\_same','EE\_diff'}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
        % mark corr significancy value 
    %     corrSig = min([min(abs(r_EE_same_aveDays{im}(p_EE_same_aveDays{im}<=.05))), min(abs(r_EE_diff_aveDays{im}(p_EE_diff_aveDays{im}<=.05)))]);
    %     subplot(sp(1)), title('EE')
    %     vline(corrSig, 'k:')
    %     vline(-corrSig, 'k:')

        % 'II\_same','II\_diff'
        sp = [233,236];
        plotHist(r_II_same_shfl_aveDays{im}, r_II_diff_shfl_aveDays{im},'','', {'',''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins);         
        plotHist(r_II_same_aveDays{im}, r_II_diff_aveDays{im},xlab,ylab, {'II\_same','II\_diff'}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins);   
        % mark corr significancy value     
    %     corrSig = min([min(abs(r_II_same_aveDays{im}(p_II_same_aveDays{im}<=.05))), min(abs(r_II_diff_aveDays{im}(p_II_diff_aveDays{im}<=.05)))]);
    %     subplot(sp(1)), title('II')
    %     vline(corrSig, 'k:')
    %     vline(-corrSig, 'k:')


        if saveFigs
            namv = sprintf('corrFR_pairwise_rsAvedPerDay_distDaysPooled_sameDiffTun_FR%s_ROC%s_%s_curr%s_%s', alFR, al, time2an, o2a, nowStr);

            d = fullfile(dirn0fr, mice{im}, nnow);
            fn = fullfile(dirn0fr, mice{im}, nnow, namv); 

            savefig(gcf, fn)
        end      
    end



    %% All mice: plot histogram of correlations (averaged across pair of neurons for each day), all days of all mice pooled

    nBins = 50;
    doSmooth = 0;
    ylab = 'Fraction days (all mice)';

    % set the bins ... same for all plots
    ally = [[r_EI_same_shfl_aveDays{:}], [r_EI_diff_shfl_aveDays{:}], [r_EI_same_aveDays{:}], [r_EI_diff_aveDays{:}], ...
        [r_EE_same_shfl_aveDays{:}], [r_EE_diff_shfl_aveDays{:}], [r_EE_same_aveDays{:}], [r_EE_diff_aveDays{:}], ...
        [r_II_same_shfl_aveDays{:}], [r_II_diff_shfl_aveDays{:}], [r_II_same_aveDays{:}], [r_II_diff_aveDays{:}]];
    r1 = round(min(ally)-.05, 1);  %min(ally); %   % round(min(ally), 1);
    r2 = round(max(ally)+.05, 1);
    bins = r1 : (r2-r1)/nBins : r2;
        
    fh = figure('name', 'All mice', 'position', [27         404        1370         521]);

    typeNs = 'EI';
    leg = {'EI\_same','EI\_diff'};
    y1 = [r_EI_same_aveDays{:}];
    y2 = [r_EI_diff_aveDays{:}];
    sp = [231,234];
    % fh = figure('name', ['All Mice-',typeNs], 'position', [27   404   419   521]);         
    plotHist([r_EI_same_shfl_aveDays{:}], [r_EI_diff_shfl_aveDays{:}],'','',{'',''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins); 
    plotHist(y1, y2, xlab,ylab,leg, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
    % mark corr significancy value     
    % corrSig = min([min(abs(y1([p_EI_same_aveDays{:}]<=.05))), min(abs(y2([p_EI_diff_aveDays{:}]<=.05)))]);
    % subplot(sp(1)), title('EI')
    % vline(corrSig, 'k:')
    % vline(-corrSig, 'k:')
    % figure; errorbar(1:2, [nanmean(y1), mean(y2)], [nanstd(y1), std(y2)])
    % xlim([0,3])
    %{
    if saveFigs
        namv = sprintf('corrFR_pairwise_%s_sameDiffTun_FR%s_ROC%s_%s_curr%s_aveDays_%s', typeNs, alFR, al, time2an, o2a, nowStr);
        fn = fullfile(dirn0fr, namv);
        savefig(gcf, fn)
        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end  
    %}


    typeNs = 'EE';
    leg = {'EE\_same','EE\_diff'};
    y1 = [r_EE_same_aveDays{:}];
    y2 = [r_EE_diff_aveDays{:}];
    sp = [232,235];
    % fh = figure('name', ['All Mice-',typeNs], 'position', [27   404   419   521]);         
    plotHist([r_EE_same_shfl_aveDays{:}], [r_EE_diff_shfl_aveDays{:}],'','',{'',''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins); 
    plotHist(y1, y2, xlab,ylab,leg, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
    % mark corr significancy value     
    % corrSig = min([min(abs(y1([p_EE_same_aveDays{:}]<=.05))), min(abs(y2([p_EE_diff_aveDays{:}]<=.05)))]);
    % subplot(sp(1)), title('EE')
    % vline(corrSig, 'k:')
    % vline(-corrSig, 'k:')
    % figure; errorbar(1:2, [nanmean(y1), mean(y2)], [nanstd(y1), std(y2)])
    % xlim([0,3])
    %{
    if saveFigs
        namv = sprintf('corrFR_pairwise_%s_sameDiffTun_FR%s_ROC%s_%s_curr%s_aveDays_%s', typeNs, alFR, al, time2an, o2a, nowStr);
        fn = fullfile(dirn0fr, namv);
        savefig(gcf, fn)
        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end  
    %}


    typeNs = 'II';
    leg = {'II\_same','II\_diff'};
    y1 = [r_II_same_aveDays{:}];
    y2 = [r_II_diff_aveDays{:}];
    sp = [233,236];
    % fh = figure('name', ['All Mice-',typeNs], 'position', [27   404   419   521]);         
    plotHist([r_II_same_shfl_aveDays{:}], [r_II_diff_shfl_aveDays{:}],'','',{'',''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins); 
    plotHist(y1, y2, xlab,ylab,leg, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
    % mark corr significancy value     
    % corrSig = min([min(abs(y1([p_II_same_aveDays{:}]<=.05))), min(abs(y2([p_II_diff_aveDays{:}]<=.05)))]);
    % subplot(sp(1)), title('II')
    % vline(corrSig, 'k:')
    % vline(-corrSig, 'k:')
    % figure; errorbar(1:2, [nanmean(y1), mean(y2)], [nanstd(y1), std(y2)])
    % xlim([0,3])


    %%%
    if saveFigs
        namv = sprintf('corrFR_pairwise_rsAvedPerDay_distMiceDaysPooled_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);

        d = fullfile(dirn0fr, 'sumAllMice', nnow);
        if ~exist(d,'dir')
            mkdir(d)
        end
        fn = fullfile(d, namv);

        savefig(gcf, fn)
        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end    


    %% Any change in corr_diff and corr_same across days?  compare mean of corr_same and corr_diff across days of training
    % figures are almost same as corrAve figures we got from the avePairWise method above... makes sense,
    % we are averaging pw corrs in both cases! ... so even though hists look
    % different between corrAve and pw corr cases, ave figres (like this one)
    % look the same.
    %{
    for im = 1:length(mice)

        figure('name', mice{im}); 

        %%%%%%%% EI %%%%%%%%
        subplot(231)
        plot([r_EI_same_aveDays{im}; r_EI_diff_aveDays{im}]');
        legend('Same tuning', 'Diff tuning')
        ylabel('corr (pairWise)')
        xlabel('Training days')
        title('EI')

        subplot(234)
        plot(r_EI_same_aveDays{im} ./ r_EI_diff_aveDays{im});
        ylabel('corrSame/corrDiff')
        xlabel('Training days')


        %%%%%%%% EE %%%%%%%%   
        subplot(232)
        plot([r_EE_same_aveDays{im}; r_EE_diff_aveDays{im}]');
    %     legend('Same tuning', 'Diff tuning')
        ylabel('corr (pairWise)')
        xlabel('Training days')
        title('EE')

        subplot(235)
        plot(r_EE_same_aveDays{im} ./ r_EE_diff_aveDays{im});
        ylabel('corrSame/corrDiff')
        xlabel('Training days')


        %%%%%%%% II %%%%%%%%    
        subplot(233)
        plot([r_II_same_aveDays{im}; r_II_diff_aveDays{im}]');
    %     legend('Same tuning', 'Diff tuning')
        ylabel('corr (pairWise)')
        xlabel('Training days')
        title('II')

        subplot(236)
        plot(r_II_same_aveDays{im} ./ r_II_diff_aveDays{im});
        ylabel('corrSame/corrDiff')
        xlabel('Training days')


        if saveFigs
            namv = sprintf('corrFR_pairwise_trainingDays_sameDiffTun_FR%s_ROC%s_%s_curr%s_%s_%s', alFR, al, time2an, o2a, mice{im}, nowStr);

            d = fullfile(dirn0fr, mice{im}, nnow);
            fn = fullfile(d, namv);

            savefig(gcf, fn)
        end      
    end
    %}
end


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Pairwair corrs without any averaging %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Pool across days (for each mouse) corrs for same-tuning and opposite-tunning neurons

r_EI_same_allMice = cell(1,length(mice));
r_EI_diff_allMice = cell(1,length(mice));
r_EE_same_allMice = cell(1,length(mice));
r_EE_diff_allMice = cell(1,length(mice));
r_II_same_allMice = cell(1,length(mice));
r_II_diff_allMice = cell(1,length(mice));

p_EI_same_allMice = cell(1,length(mice));
p_EI_diff_allMice = cell(1,length(mice));
p_EE_same_allMice = cell(1,length(mice));
p_EE_diff_allMice = cell(1,length(mice));
p_II_same_allMice = cell(1,length(mice));
p_II_diff_allMice = cell(1,length(mice));

% shfl
r_EI_same_shfl_allMice = cell(1,length(mice));
r_EI_diff_shfl_allMice = cell(1,length(mice));
r_EE_same_shfl_allMice = cell(1,length(mice));
r_EE_diff_shfl_allMice = cell(1,length(mice));
r_II_same_shfl_allMice = cell(1,length(mice));
r_II_diff_shfl_allMice = cell(1,length(mice));

for im = 1:length(mice)    

    % 'EI\_same','EI\_diff'
    r_EI_same_allMice{im} = [r_Ei_Ii_alln{im}{:}, r_Ec_Ic_alln{im}{:}];
    r_EI_diff_allMice{im} = [r_Ei_Ic_alln{im}{:}, r_Ec_Ii_alln{im}{:}];    

    p_EI_same_allMice{im} = [p_Ei_Ii_alln{im}{:}, p_Ec_Ic_alln{im}{:}];
    p_EI_diff_allMice{im} = [p_Ei_Ic_alln{im}{:}, p_Ec_Ii_alln{im}{:}];    
    
    
    % 'EE\_same','EE\_diff'
    r_EE_same_allMice{im} = [r_Ei_Ei_alln{im}{:}, r_Ec_Ec_alln{im}{:}];
    r_EE_diff_allMice{im} = [r_Ei_Ec_alln{im}{:}];
    
    p_EE_same_allMice{im} = [p_Ei_Ei_alln{im}{:}, p_Ec_Ec_alln{im}{:}];
    p_EE_diff_allMice{im} = [p_Ei_Ec_alln{im}{:}];
    
    
    % 'II\_same','II\_diff'
    r_II_same_allMice{im} = [r_Ii_Ii_alln{im}{:}, r_Ic_Ic_alln{im}{:}];
    r_II_diff_allMice{im} = [r_Ii_Ic_alln{im}{:}];
    
    p_II_same_allMice{im} = [p_Ii_Ii_alln{im}{:}, p_Ic_Ic_alln{im}{:}];
    p_II_diff_allMice{im} = [p_Ii_Ic_alln{im}{:}];

    
    % shfl
    r_EI_same_shfl_allMice{im} = [r_Ei_Ii_alln_shfl{im}{:}, r_Ec_Ic_alln_shfl{im}{:}];
    r_EI_diff_shfl_allMice{im} = [r_Ei_Ic_alln_shfl{im}{:}, r_Ec_Ii_alln_shfl{im}{:}];    
    
    r_EE_same_shfl_allMice{im} = [r_Ei_Ei_alln_shfl{im}{:}, r_Ec_Ec_alln_shfl{im}{:}];
    r_EE_diff_shfl_allMice{im} = [r_Ei_Ec_alln_shfl{im}{:}];    
    
    r_II_same_shfl_allMice{im} = [r_Ii_Ii_alln_shfl{im}{:}, r_Ic_Ic_alln_shfl{im}{:}];
    r_II_diff_shfl_allMice{im} = [r_Ii_Ic_alln_shfl{im}{:}];    
    
end



if doPlots
    %% Plot errorbar, showing each mouse, ave +/- se of r across all PW corrs

    x = 1:length(mice);
    gp = 0; %.2;
    marg = .2;

    figure('name', 'All mice', 'position', [14   636   661   290]); 
    set(gca, 'position', [0.2919    0.1908    0.5229    0.7095])

    %%%%%%%%%% Plot EI %%%%%%%%%%%
    typeNs = 'EI';    
    subplot(131); hold on
    h1 = errorbar(x, cellfun(@nanmean, r_EI_diff_allMice), cellfun(@nanstd, r_EI_diff_allMice)./sqrt(cellfun(@(x)sum(~isnan(x)), r_EI_diff_allMice)), 'r.', 'linestyle', 'none');
    h2 = errorbar(x+gp, cellfun(@nanmean, r_EI_same_allMice), cellfun(@nanstd, r_EI_same_allMice)./sqrt(cellfun(@(x)sum(~isnan(x)), r_EI_same_allMice)), 'k.', 'linestyle', 'none');
    % shfl
    errorbar(x, cellfun(@nanmean, r_EI_diff_shfl_allMice), cellfun(@nanstd, r_EI_diff_shfl_allMice)./sqrt(cellfun(@(x)sum(~isnan(x)), r_EI_diff_allMice)), 'color',rgb('lightsalmon'), 'marker', '.', 'linestyle', 'none')
    errorbar(x+gp, cellfun(@nanmean, r_EI_same_shfl_allMice), cellfun(@nanstd, r_EI_same_shfl_allMice)./sqrt(cellfun(@(x)sum(~isnan(x)), r_EI_same_allMice)), 'color',rgb('gray'), 'marker', '.', 'linestyle', 'none')
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    title(typeNs)
    ylabel('corr (mean +/- se neurons)')
    legend([h1,h2], {'diff','same'})

    %%%%%%%%%% Plot EE %%%%%%%%%%%
    typeNs = 'EE';    
    subplot(132); hold on
    errorbar(x, cellfun(@nanmean, r_EE_diff_allMice), cellfun(@nanstd, r_EE_diff_allMice)./sqrt(cellfun(@(x)sum(~isnan(x)), r_EE_diff_allMice)), 'r.', 'linestyle', 'none')
    errorbar(x+gp, cellfun(@nanmean, r_EE_same_allMice), cellfun(@nanstd, r_EE_same_allMice)./sqrt(cellfun(@(x)sum(~isnan(x)), r_EE_same_allMice)), 'k.', 'linestyle', 'none')
    % shfl
    errorbar(x, cellfun(@nanmean, r_EE_diff_shfl_allMice), cellfun(@nanstd, r_EE_diff_shfl_allMice)./sqrt(cellfun(@(x)sum(~isnan(x)), r_EE_diff_allMice)), 'color',rgb('lightsalmon'), 'marker', '.', 'linestyle', 'none')
    errorbar(x+gp, cellfun(@nanmean, r_EE_same_shfl_allMice), cellfun(@nanstd, r_EE_same_shfl_allMice)./sqrt(cellfun(@(x)sum(~isnan(x)), r_EE_same_allMice)), 'color',rgb('gray'), 'marker', '.', 'linestyle', 'none')
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    title(typeNs)

    %%%%%%%%%% Plot II %%%%%%%%%%%
    typeNs = 'II';    
    subplot(133); hold on
    errorbar(x, cellfun(@nanmean, r_II_diff_allMice), cellfun(@nanstd, r_II_diff_allMice)./sqrt(cellfun(@(x)sum(~isnan(x)), r_II_diff_allMice)), 'r.', 'linestyle', 'none')
    errorbar(x+gp, cellfun(@nanmean, r_II_same_allMice), cellfun(@nanstd, r_II_same_allMice)./sqrt(cellfun(@(x)sum(~isnan(x)), r_II_same_allMice)), 'k.', 'linestyle', 'none')
    % shfl
    errorbar(x, cellfun(@nanmean, r_II_diff_shfl_allMice), cellfun(@nanstd, r_II_diff_shfl_allMice)./sqrt(cellfun(@(x)sum(~isnan(x)), r_II_diff_allMice)), 'color',rgb('lightsalmon'), 'marker', '.', 'linestyle', 'none')
    errorbar(x+gp, cellfun(@nanmean, r_II_same_shfl_allMice), cellfun(@nanstd, r_II_same_shfl_allMice)./sqrt(cellfun(@(x)sum(~isnan(x)), r_II_same_allMice)), 'color',rgb('gray'), 'marker', '.', 'linestyle', 'none')
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    title(typeNs)


    if saveFigs
        namv = sprintf('corrFR_pairwise_aveSeAllNs_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);

        d = fullfile(dirn0fr, 'sumAllMice', nnow);
        if ~exist(d,'dir')
            mkdir(d)
        end
        fn = fullfile(d, namv);

        savefig(gcf, fn)

        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end    


    %% Each mouse : plot histogram of correlations, all neuron pairs pooled

    nBins = 500;
    doSmooth = 0;
    ylab = 'Fraction neuron pairs';
    xlab = 'Corr coef';

    for im = 1:length(mice)    

        fh = figure('name', mice{im}, 'position', [27         404        1370         521]);

        % 'EI\_same','EI\_diff'
        sp = [231,234];
        plotHist(r_EI_same_shfl_allMice{im}, r_EI_diff_shfl_allMice{im},'','', {'',''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp); 
        plotHist(r_EI_same_allMice{im}, r_EI_diff_allMice{im},xlab,ylab, {'EI\_same','EI\_diff'}, cols, [], fh, nBins, doSmooth, linestylesData, sp); 
        % mark corr significancy value 
        corrSig = min([min(abs(r_EI_same_allMice{im}(p_EI_same_allMice{im}<=.05))), min(abs(r_EI_diff_allMice{im}(p_EI_diff_allMice{im}<=.05)))]);
        subplot(sp(1)), title('EI')
        vline(corrSig, 'k:')
        vline(-corrSig, 'k:')

        % 'EE\_same','EE\_diff'
        sp = [232,235];
        plotHist(r_EE_same_shfl_allMice{im}, r_EE_diff_shfl_allMice{im},'','', {'',''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp);     
        plotHist(r_EE_same_allMice{im}, r_EE_diff_allMice{im},xlab,ylab, {'EE\_same','EE\_diff'}, cols, [], fh, nBins, doSmooth, linestylesData, sp); 
        % mark corr significancy value 
        corrSig = min([min(abs(r_EE_same_allMice{im}(p_EE_same_allMice{im}<=.05))), min(abs(r_EE_diff_allMice{im}(p_EE_diff_allMice{im}<=.05)))]);
        subplot(sp(1)), title('EE')
        vline(corrSig, 'k:')
        vline(-corrSig, 'k:')

        % 'II\_same','II\_diff'
        sp = [233,236];
        plotHist(r_II_same_shfl_allMice{im}, r_II_diff_shfl_allMice{im},'','', {'',''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp);         
        plotHist(r_II_same_allMice{im}, r_II_diff_allMice{im},xlab,ylab, {'II\_same','II\_diff'}, cols, [], fh, nBins, doSmooth, linestylesData, sp);   
        % mark corr significancy value     
        corrSig = min([min(abs(r_II_same_allMice{im}(p_II_same_allMice{im}<=.05))), min(abs(r_II_diff_allMice{im}(p_II_diff_allMice{im}<=.05)))]);
        subplot(sp(1)), title('II')
        vline(corrSig, 'k:')
        vline(-corrSig, 'k:')


        if saveFigs
            namv = sprintf('corrFR_pairwise_distNsDaysPooled_sameDiffTun_FR%s_ROC%s_%s_curr%s_%s', alFR, al, time2an, o2a, nowStr);

            d = fullfile(dirn0fr, mice{im}, nnow);
            fn = fullfile(dirn0fr, mice{im}, nnow, namv); 

            savefig(gcf, fn)
        end      
    end



    %% All mice: plot histogram of correlations for all mice, all neuron pairs pooled

    nBins = 500;
    doSmooth = 0;

    fh = figure('name', 'All mice', 'position', [27         404        1370         521]);

    typeNs = 'EI';
    leg = {'EI\_same','EI\_diff'};
    y1 = [r_EI_same_allMice{:}];
    y2 = [r_EI_diff_allMice{:}];
    sp = [231,234];
    % fh = figure('name', ['All Mice-',typeNs], 'position', [27   404   419   521]);         
    plotHist([r_EI_same_shfl_allMice{:}], [r_EI_diff_shfl_allMice{:}],'','',{'',''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp); 
    plotHist(y1, y2, xlab,ylab,leg, cols, [], fh, nBins, doSmooth, linestylesData, sp); 
    % mark corr significancy value     
    corrSig = min([min(abs(y1([p_EI_same_allMice{:}]<=.05))), min(abs(y2([p_EI_diff_allMice{:}]<=.05)))]);
    subplot(sp(1)), title('EI')
    vline(corrSig, 'k:')
    vline(-corrSig, 'k:')
    % figure; errorbar(1:2, [nanmean(y1), mean(y2)], [nanstd(y1), std(y2)])
    % xlim([0,3])
    %{
    if saveFigs
        namv = sprintf('corrFR_pairwise_%s_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', typeNs, alFR, al, time2an, o2a, nowStr);
        fn = fullfile(dirn0fr, namv);
        savefig(gcf, fn)
        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end  
    %}


    typeNs = 'EE';
    leg = {'EE\_same','EE\_diff'};
    y1 = [r_EE_same_allMice{:}];
    y2 = [r_EE_diff_allMice{:}];
    sp = [232,235];
    % fh = figure('name', ['All Mice-',typeNs], 'position', [27   404   419   521]);         
    plotHist([r_EE_same_shfl_allMice{:}], [r_EE_diff_shfl_allMice{:}],'','',{'',''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp); 
    plotHist(y1, y2, xlab,ylab,leg, cols, [], fh, nBins, doSmooth, linestylesData, sp); 
    % mark corr significancy value     
    corrSig = min([min(abs(y1([p_EE_same_allMice{:}]<=.05))), min(abs(y2([p_EE_diff_allMice{:}]<=.05)))]);
    subplot(sp(1)), title('EE')
    vline(corrSig, 'k:')
    vline(-corrSig, 'k:')
    % figure; errorbar(1:2, [nanmean(y1), mean(y2)], [nanstd(y1), std(y2)])
    % xlim([0,3])
    %{
    if saveFigs
        namv = sprintf('corrFR_pairwise_%s_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', typeNs, alFR, al, time2an, o2a, nowStr);
        fn = fullfile(dirn0fr, namv);
        savefig(gcf, fn)
        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end  
    %}


    typeNs = 'II';
    leg = {'II\_same','II\_diff'};
    y1 = [r_II_same_allMice{:}];
    y2 = [r_II_diff_allMice{:}];
    sp = [233,236];
    % fh = figure('name', ['All Mice-',typeNs], 'position', [27   404   419   521]);         
    plotHist([r_II_same_shfl_allMice{:}], [r_II_diff_shfl_allMice{:}],'','',{'',''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp); 
    plotHist(y1, y2, xlab,ylab,leg, cols, [], fh, nBins, doSmooth, linestylesData, sp); 
    % mark corr significancy value     
    corrSig = min([min(abs(y1([p_II_same_allMice{:}]<=.05))), min(abs(y2([p_II_diff_allMice{:}]<=.05)))]);
    subplot(sp(1)), title('II')
    vline(corrSig, 'k:')
    vline(-corrSig, 'k:')
    % figure; errorbar(1:2, [nanmean(y1), mean(y2)], [nanstd(y1), std(y2)])
    % xlim([0,3])


    if saveFigs
        namv = sprintf('corrFR_pairwise_distMiceDaysNsPooled_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);

        d = fullfile(dirn0fr, 'sumAllMice', nnow);
        if ~exist(d,'dir')
            mkdir(d)
        end
        fn = fullfile(d, namv);

        savefig(gcf, fn)
        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end    

end





%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% Correlation between population-averaged FRs %%%%%%%%%%%%%%%%%%%%%%%
% For each day, we average FR across neurons of each population
% (Ei,Ec,Ii,Ic), then we compute correlations between different
% populations. So for each day we get one number (for each correlation).
% non-specific connectivity: both E_ipsi and E_contra should have corr with
% I...  (though we already know we have I_ipsi and I_contra...)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Compute correlations in population-averaged FR between exc and inh (FR averaged across neurons of each population) for each day

corr_exci_inhi = cell(1, length(mice));
corr_excc_inhc = cell(1, length(mice));
corr_exci_inhc = cell(1, length(mice));
corr_excc_inhi = cell(1, length(mice));

corr_exci_exci = cell(1, length(mice));
corr_excc_excc = cell(1, length(mice));
corr_exci_excc = cell(1, length(mice));
corr_excc_exci = cell(1, length(mice));

corr_inhi_inhi = cell(1, length(mice));
corr_inhc_inhc = cell(1, length(mice));
corr_inhi_inhc = cell(1, length(mice));
corr_inhc_inhi = cell(1, length(mice));

rPop_EI_same_allMice = cell(1, length(mice));
rPop_EI_diff_allMice = cell(1, length(mice));
rPop_EE_same_allMice = cell(1, length(mice));
rPop_EE_diff_allMice = cell(1, length(mice));
rPop_II_same_allMice = cell(1, length(mice));
rPop_II_diff_allMice = cell(1, length(mice));
rPop_EI_same_shfl_allMice = cell(1, length(mice));
rPop_EI_diff_shfl_allMice = cell(1, length(mice));
rPop_EE_same_shfl_allMice = cell(1, length(mice));
rPop_EE_diff_shfl_allMice = cell(1, length(mice));
rPop_II_same_shfl_allMice = cell(1, length(mice));
rPop_II_diff_shfl_allMice = cell(1, length(mice));

p_corr_exci_inhi = cell(1, length(mice));
p_corr_excc_inhc = cell(1, length(mice));
p_corr_exci_inhc = cell(1, length(mice));
p_corr_excc_inhi = cell(1, length(mice));

p_corr_exci_exci = cell(1, length(mice));
p_corr_excc_excc = cell(1, length(mice));
p_corr_exci_excc = cell(1, length(mice));
p_corr_excc_exci = cell(1, length(mice));

p_corr_inhi_inhi = cell(1, length(mice));
p_corr_inhc_inhc = cell(1, length(mice));
p_corr_inhi_inhc = cell(1, length(mice));
p_corr_inhc_inhi = cell(1, length(mice));

% shfl
corr_exci_inhi_shfl = cell(1, length(mice));
corr_excc_inhc_shfl = cell(1, length(mice));
corr_exci_inhc_shfl = cell(1, length(mice));
corr_excc_inhi_shfl = cell(1, length(mice));

corr_exci_exci_shfl = cell(1, length(mice));
corr_excc_excc_shfl = cell(1, length(mice));
corr_exci_excc_shfl = cell(1, length(mice));
corr_excc_exci_shfl = cell(1, length(mice));

corr_inhi_inhi_shfl = cell(1, length(mice));
corr_inhc_inhc_shfl = cell(1, length(mice));
corr_inhi_inhc_shfl = cell(1, length(mice));
corr_inhc_inhi_shfl = cell(1, length(mice));


for im = 1:length(mice)
    for iday = 1:numDaysAll(im)
        
        if mnTrNum_allMice{im}(iday) >= thMinTrs
            
            if ipsi_contra_all(1)==1
                trs = ipsiTrs_allDays_allMice{im}{iday};
            elseif ipsi_contra_all(2)==1
                trs = contraTrs_allDays_allMice{im}{iday};
            elseif ipsi_contra_all(3)==1
                trs = true(1, size(fr_exc_ipsi_timeM1{im}{iday}, 2)); %1:length(ipsiTrs_allDays_allMice{im}{iday});
            end

            %%%%%%%%%% EI %%%%%%%%%%
            % E_ipsi , I_ipsi
            [corr_exci_inhi{im}(iday), p_corr_exci_inhi{im}(iday)] = corr(fr_exc_ipsi_timeM1_aveNs{im}{iday}(trs)' , fr_inh_ipsi_timeM1_aveNs{im}{iday}(trs)');
            % E_contra , I_contra
            [corr_excc_inhc{im}(iday), p_corr_excc_inhc{im}(iday)] = corr(fr_exc_contra_timeM1_aveNs{im}{iday}(trs)' , fr_inh_contra_timeM1_aveNs{im}{iday}(trs)');
            %%% pool (set corr for same and oppositely tuned populations (each day has one corr valule for same and one corr value for oppositely tuned populations))
            rPop_EI_same_allMice{im}(iday) = mean([corr_exci_inhi{im}(iday), corr_excc_inhc{im}(iday)]);
            
            % E_ipsi , I_contra
            [corr_exci_inhc{im}(iday), p_corr_exci_inhc{im}(iday)] = corr(fr_exc_ipsi_timeM1_aveNs{im}{iday}(trs)' , fr_inh_contra_timeM1_aveNs{im}{iday}(trs)');           
            % E_contra , I_ipsi
            [corr_excc_inhi{im}(iday), p_corr_excc_inhi{im}(iday)] = corr(fr_exc_contra_timeM1_aveNs{im}{iday}(trs)' , fr_inh_ipsi_timeM1_aveNs{im}{iday}(trs)');
            %%% pool
            rPop_EI_diff_allMice{im}(iday) = mean([corr_exci_inhc{im}(iday), corr_excc_inhi{im}(iday)]);
            
            
            %%%%%%%%%% EE %%%%%%%%%%
            % well EE_same will obviously be 1!! (corr between the same popultion!)
            % E_ipsi , E_ipsi
            [corr_exci_exci{im}(iday), p_corr_exci_exci{im}(iday)] = corr(fr_exc_ipsi_timeM1_aveNs{im}{iday}(trs)' , fr_exc_ipsi_timeM1_aveNs{im}{iday}(trs)');
            % E_contra , E_contra
            [corr_excc_excc{im}(iday), p_corr_excc_excc{im}(iday)] = corr(fr_exc_contra_timeM1_aveNs{im}{iday}(trs)' , fr_exc_contra_timeM1_aveNs{im}{iday}(trs)');
            %%% pool
            rPop_EE_same_allMice{im}(iday) = mean([corr_exci_exci{im}(iday), corr_excc_excc{im}(iday)]);

            % E_ipsi , E_contra
            [rPop_EE_diff_allMice{im}(iday), p_corr_exci_excc{im}(iday)] = corr(fr_exc_ipsi_timeM1_aveNs{im}{iday}(trs)' , fr_exc_contra_timeM1_aveNs{im}{iday}(trs)');
            

            %%%%%%%%%% II %%%%%%%%%%
            % well II_same will obviously be 1!! (corr between the same popultion!)
            % I_ipsi , I_ipsi
            [corr_inhi_inhi{im}(iday), p_corr_inhi_inhi{im}(iday)] = corr(fr_inh_ipsi_timeM1_aveNs{im}{iday}(trs)' , fr_inh_ipsi_timeM1_aveNs{im}{iday}(trs)');
            % I_contra , I_contra
            [corr_inhc_inhc{im}(iday), p_corr_inhc_inhc{im}(iday)] = corr(fr_inh_contra_timeM1_aveNs{im}{iday}(trs)' , fr_inh_contra_timeM1_aveNs{im}{iday}(trs)');
            %%% pool
            rPop_II_same_allMice{im}(iday) = mean([corr_inhi_inhi{im}(iday), corr_inhc_inhc{im}(iday)]);

            % I_ipsi , I_contra
            [rPop_II_diff_allMice{im}(iday), p_corr_inhi_inhc{im}(iday)] = corr(fr_inh_ipsi_timeM1_aveNs{im}{iday}(trs)' , fr_inh_contra_timeM1_aveNs{im}{iday}(trs)');
            

            
            %%%%%%%%%%%%%%%%%%%%%%%%% shuffled: corr between neurons after shuffling trials %%%%%%%%%%%%%%%%%%%%%%%%%            
            %%%%% exc,exc: corr between similarly-tuned neurons
            ntrs = sum(trs);
            
            %%%%%%%%%% EI %%%%%%%%%%
            % E_ipsi , I_ipsi
            [corr_exci_inhi_shfl{im}(iday)] = corr(fr_exc_ipsi_timeM1_aveNs{im}{iday}(randperm(ntrs))' , fr_inh_ipsi_timeM1_aveNs{im}{iday}(randperm(ntrs))');            
            % E_contra , I_contra
            [corr_excc_inhc_shfl{im}(iday)] = corr(fr_exc_contra_timeM1_aveNs{im}{iday}(randperm(ntrs))' , fr_inh_contra_timeM1_aveNs{im}{iday}(randperm(ntrs))');
            %%% pool
            rPop_EI_same_shfl_allMice{im}(iday) = mean([corr_exci_inhi_shfl{im}(iday), corr_excc_inhc_shfl{im}(iday)]);            
            
            % E_ipsi , I_contra
            [corr_exci_inhc_shfl{im}(iday)] = corr(fr_exc_ipsi_timeM1_aveNs{im}{iday}(randperm(ntrs))' , fr_inh_contra_timeM1_aveNs{im}{iday}(randperm(ntrs))');            
            % E_contra , I_ipsi
            [corr_excc_inhi_shfl{im}(iday)] = corr(fr_exc_contra_timeM1_aveNs{im}{iday}(randperm(ntrs))' , fr_inh_ipsi_timeM1_aveNs{im}{iday}(randperm(ntrs))');
            %%% pool
            rPop_EI_diff_shfl_allMice{im}(iday) = mean([corr_exci_inhc_shfl{im}(iday), corr_excc_inhi_shfl{im}(iday)]);
            
            
            %%%%%%%%%% EE %%%%%%%%%%
            % E_ipsi , E_ipsi
            [corr_exci_exci_shfl{im}(iday)] = corr(fr_exc_ipsi_timeM1_aveNs{im}{iday}(randperm(ntrs))' , fr_exc_ipsi_timeM1_aveNs{im}{iday}(randperm(ntrs))');
            % E_contra , E_contra
            [corr_excc_excc_shfl{im}(iday)] = corr(fr_exc_contra_timeM1_aveNs{im}{iday}(randperm(ntrs))' , fr_exc_contra_timeM1_aveNs{im}{iday}(randperm(ntrs))');
            %%% pool
            rPop_EE_same_shfl_allMice{im}(iday) = mean([corr_exci_exci_shfl{im}(iday), corr_excc_excc_shfl{im}(iday)]);
            
            % E_ipsi , E_contra
            [rPop_EE_diff_shfl_allMice{im}(iday)] = corr(fr_exc_ipsi_timeM1_aveNs{im}{iday}(randperm(ntrs))' , fr_exc_contra_timeM1_aveNs{im}{iday}(randperm(ntrs))');
            

            %%%%%%%%%% II %%%%%%%%%%
            % I_ipsi , I_ipsi
            [corr_inhi_inhi_shfl{im}(iday)] = corr(fr_inh_ipsi_timeM1_aveNs{im}{iday}(randperm(ntrs))' , fr_inh_ipsi_timeM1_aveNs{im}{iday}(randperm(ntrs))');
            % I_contra , I_contra
            [corr_inhc_inhc_shfl{im}(iday)] = corr(fr_inh_contra_timeM1_aveNs{im}{iday}(randperm(ntrs))' , fr_inh_contra_timeM1_aveNs{im}{iday}(randperm(ntrs))');
            %%% pool
            rPop_II_same_shfl_allMice{im}(iday) = mean([corr_inhi_inhi_shfl{im}(iday), corr_inhc_inhc_shfl{im}(iday)]);
            
            % I_ipsi , I_contra
            [rPop_II_diff_shfl_allMice{im}(iday)] = corr(fr_inh_ipsi_timeM1_aveNs{im}{iday}(randperm(ntrs))' , fr_inh_contra_timeM1_aveNs{im}{iday}(randperm(ntrs))');
         
        else
            rPop_EI_same_allMice{im}(iday) = nan;
            rPop_EI_diff_allMice{im}(iday) = nan;
            rPop_EE_same_allMice{im}(iday) = nan;
            rPop_EE_diff_allMice{im}(iday) = nan;
            rPop_II_same_allMice{im}(iday) = nan;
            rPop_II_diff_allMice{im}(iday) = nan;
            rPop_EI_same_shfl_allMice{im}(iday) = nan;
            rPop_EI_diff_shfl_allMice{im}(iday) = nan;
            rPop_EE_same_shfl_allMice{im}(iday) = nan;
            rPop_EE_diff_shfl_allMice{im}(iday) = nan;
            rPop_II_same_shfl_allMice{im}(iday) = nan;
            rPop_II_diff_shfl_allMice{im}(iday) = nan;            
        end
    end
end




%% Plot errorbar, showing each mouse, ave +/- se of rPop across days

if doPlots
    
    x = 1:length(mice);
    gp = .2;
    marg = .2;

    figure('name', 'All mice', 'position', [14   636   661   290]); 
    set(gca, 'position', [0.2919    0.1908    0.5229    0.7095])

    %%%%%%%%%% Plot EI %%%%%%%%%%%
    typeNs = 'EI';    
    subplot(131); hold on
    h1 = errorbar(x, cellfun(@nanmean, rPop_EI_diff_allMice), cellfun(@nanstd, rPop_EI_diff_allMice)./sqrt(numDaysGood), 'r.', 'linestyle', 'none');
    h2 = errorbar(x+gp, cellfun(@nanmean, rPop_EI_same_allMice), cellfun(@nanstd, rPop_EI_same_allMice)./sqrt(numDaysGood), 'k.', 'linestyle', 'none');
    % shfl
    errorbar(x, cellfun(@nanmean, rPop_EI_diff_shfl_allMice), cellfun(@nanstd, rPop_EI_diff_shfl_allMice)./sqrt(numDaysGood), 'color',rgb('lightsalmon'), 'marker', '.', 'linestyle', 'none')
    errorbar(x+gp, cellfun(@nanmean, rPop_EI_same_shfl_allMice), cellfun(@nanstd, rPop_EI_same_shfl_allMice)./sqrt(numDaysGood), 'color',rgb('gray'), 'marker', '.', 'linestyle', 'none')
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    title(typeNs)
    ylabel('corr (mean +/- se days)')
    legend([h1,h2], {'diff','same'})

    %%%%%%%%%% Plot EE %%%%%%%%%%%
    typeNs = 'EE';    
    subplot(132); hold on
    errorbar(x, cellfun(@nanmean, rPop_EE_diff_allMice), cellfun(@nanstd, rPop_EE_diff_allMice)./sqrt(numDaysGood), 'r.', 'linestyle', 'none')
    errorbar(x+gp, cellfun(@nanmean, rPop_EE_same_allMice), cellfun(@nanstd, rPop_EE_same_allMice)./sqrt(numDaysGood), 'k.', 'linestyle', 'none')
    % shfl
    errorbar(x, cellfun(@nanmean, rPop_EE_diff_shfl_allMice), cellfun(@nanstd, rPop_EE_diff_shfl_allMice)./sqrt(numDaysGood), 'color',rgb('lightsalmon'), 'marker', '.', 'linestyle', 'none')
    errorbar(x+gp, cellfun(@nanmean, rPop_EE_same_shfl_allMice), cellfun(@nanstd, rPop_EE_same_shfl_allMice)./sqrt(numDaysGood), 'color',rgb('gray'), 'marker', '.', 'linestyle', 'none')
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    title(typeNs)

    %%%%%%%%%% Plot II %%%%%%%%%%%
    typeNs = 'II';    
    subplot(133); hold on
    errorbar(x, cellfun(@nanmean, rPop_II_diff_allMice), cellfun(@nanstd, rPop_II_diff_allMice)./sqrt(numDaysGood), 'r.', 'linestyle', 'none')
    errorbar(x+gp, cellfun(@nanmean, rPop_II_same_allMice), cellfun(@nanstd, rPop_II_same_allMice)./sqrt(numDaysGood), 'k.', 'linestyle', 'none')
    % shfl
    errorbar(x, cellfun(@nanmean, rPop_II_diff_shfl_allMice), cellfun(@nanstd, rPop_II_diff_shfl_allMice)./sqrt(numDaysGood), 'color',rgb('lightsalmon'), 'marker', '.', 'linestyle', 'none')
    errorbar(x+gp, cellfun(@nanmean, rPop_II_same_shfl_allMice), cellfun(@nanstd, rPop_II_same_shfl_allMice)./sqrt(numDaysGood), 'color',rgb('gray'), 'marker', '.', 'linestyle', 'none')
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    title(typeNs)


    if saveFigs
        namv = sprintf('corrFR_popAveFRs_aveSeDays_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);

        d = fullfile(dirn0fr, 'sumAllMice', nnow);
        if ~exist(d,'dir')
            mkdir(d)
        end
        fn = fullfile(d, namv);

        savefig(gcf, fn)

        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end    


    %% Each mouse: plot histogram of corr_same, corr_diff (corr of population-averaged FR)

    nBins = 20;
    doSmooth = 0;
    ylab = 'Fraction days';
    xlab = 'Corr coef (population)';

    for im = 1:length(mice)    

        fh = figure('name', mice{im}, 'position', [27         404        1370         521]);
        
        % set the bins ... same for all plots
        ally = [rPop_EI_same_shfl_allMice{im}, rPop_EI_diff_shfl_allMice{im}, rPop_EI_same_allMice{im}, rPop_EI_diff_allMice{im},...
            rPop_EE_same_shfl_allMice{im}, rPop_EE_diff_shfl_allMice{im}, rPop_EE_same_allMice{im}, rPop_EE_diff_allMice{im}, ...
            rPop_II_same_shfl_allMice{im}, rPop_II_diff_shfl_allMice{im}, rPop_II_same_allMice{im}, rPop_II_diff_allMice{im}];
        r1 = round(min(ally)-.05, 1);  %min(ally); %   % round(min(ally), 1);
        r2 = round(max(ally)+.05, 1);
        bins = r1 : (r2-r1)/nBins : r2;
    
        typeNs = 'EI';
        sp = [231,234];
        [~,binsSh] = plotHist(rPop_EI_same_shfl_allMice{im}, rPop_EI_diff_shfl_allMice{im},'','', {'',''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins); 
        [~,bins] = plotHist(rPop_EI_same_allMice{im}, rPop_EI_diff_allMice{im},xlab,ylab, {'EI\_same','EI\_diff'}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
        xlim([binsSh(1),bins(end)])
        subplot(sp(1)), title(typeNs)    
        % mark corr significancy value 
        corrSig = min([min(abs(corr_exci_inhi{im}(p_corr_exci_inhi{im}<=.05))), min(abs(corr_excc_inhc{im}(p_corr_excc_inhc{im}<=.05))), ...
            min(abs(corr_exci_inhc{im}(p_corr_exci_inhc{im}<=.05))), min(abs(corr_excc_inhi{im}(p_corr_excc_inhi{im}<=.05)))]);
        vline(corrSig, 'k:')
        vline(-corrSig, 'k:')

        typeNs = 'EE';
        sp = [232,235];
        [~,binsSh] = plotHist(rPop_EE_same_shfl_allMice{im}, rPop_EE_diff_shfl_allMice{im},'','', {'',''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins);     
        [~,bins] = plotHist(rPop_EE_same_allMice{im}, rPop_EE_diff_allMice{im},xlab,ylab, {'EE\_same','EE\_diff'}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
        xlim([binsSh(1),bins(end)])
        subplot(sp(1)), title(typeNs)    
        % mark corr significancy value 
        corrSig = min([min(abs(corr_exci_exci{im}(p_corr_exci_exci{im}<=.05))), min(abs(corr_excc_excc{im}(p_corr_excc_excc{im}<=.05))), ...
            min(abs(rPop_EE_diff_allMice{im}(p_corr_exci_excc{im}<=.05)))]);
        vline(corrSig, 'k:')
        vline(-corrSig, 'k:')

        typeNs = 'II';
        sp = [233,236];
        [~,binsSh] = plotHist(rPop_II_same_shfl_allMice{im}, rPop_II_diff_shfl_allMice{im},'','', {'',''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins);         
        [~,bins] = plotHist(rPop_II_same_allMice{im}, rPop_II_diff_allMice{im},xlab,ylab, {'II\_same','II\_diff'}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins);   
        xlim([binsSh(1),bins(end)])
        subplot(sp(1)), title(typeNs)    
        % mark corr significancy value     
        corrSig = min([min(abs(corr_inhi_inhi{im}(p_corr_inhi_inhi{im}<=.05))), min(abs(corr_inhc_inhc{im}(p_corr_inhc_inhc{im}<=.05))), ...
            min(abs(rPop_II_diff_allMice{im}(p_corr_inhi_inhc{im}<=.05)))]);
        vline(corrSig, 'k:')
        vline(-corrSig, 'k:')


        if saveFigs
            namv = sprintf('corrFR_popAveFRs_distDaysPooled_sameDiffTun_FR%s_ROC%s_%s_curr%s_%s', alFR, al, time2an, o2a, nowStr);

            d = fullfile(dirn0fr, mice{im}, nnow);
            fn = fullfile(dirn0fr, mice{im}, nnow, namv); 

            savefig(gcf, fn)
        end      

    end

    % hist of corrs, (EiEi is not pooled with EiEc, etc)
    %{
    for im = 1:length(mice)
        figure('name', mice{im}); 

        subplot(221); hold on
        histogram(corr_exci_inhi{im})
        plot(nanmean(corr_exci_inhi{im}), 1, 'r*')
        title('E\_ipsi , I\_ipsi')
        xlabel('corr coef')
        ylabel('num days')

        subplot(223); hold on
        histogram(corr_excc_inhc{im})
        plot(nanmean(corr_excc_inhc{im}), 1, 'r*')
        title('E\_contra , I\_contra')
        xlabel('corr coef')
        ylabel('num days')

        %
        subplot(222); hold on
        histogram(corr_exci_inhc{im})
        plot(nanmean(corr_exci_inhc{im}), 1, 'r*')
        title('E\_ipsi , I\_contra')
        xlabel('corr coef')
        ylabel('num days')

        subplot(224); hold on
        histogram(corr_excc_inhi{im})    
        plot(nanmean(corr_excc_inhi{im}), 1, 'r*')
        title('E\_contra , I\_ipsi')
        xlabel('corr coef')
        ylabel('num days')
    end
    %}


    %% All mice: pool days of all mice and get a histogram...

    nBins = 30;
    doSmooth = 0;

    % set the bins ... same for all plots
    ally = [[rPop_EI_same_shfl_allMice{:}], [rPop_EI_diff_shfl_allMice{:}], [rPop_EI_same_allMice{:}], [rPop_EI_diff_allMice{:}], ...
        [rPop_EE_same_shfl_allMice{:}], [rPop_EE_diff_shfl_allMice{:}], [rPop_EE_same_allMice{:}], [rPop_EE_diff_allMice{:}], ...
        [rPop_II_same_shfl_allMice{:}], [rPop_II_diff_shfl_allMice{:}], [rPop_II_same_allMice{:}], [rPop_II_diff_allMice{:}]];
    r1 = round(min(ally)-.05, 1);  %min(ally); %   % round(min(ally), 1);
    r2 = round(max(ally)+.05, 1);
    bins = r1 : (r2-r1)/nBins : r2;
        
    fh = figure('name', 'All mice', 'position', [27         404        1370         521]);

    typeNs = 'EI';
    leg = {'EI\_same','EI\_diff'};
    y1 = [rPop_EI_same_allMice{:}];
    y2 = [rPop_EI_diff_allMice{:}];
    sp = [231,234];
    % fh = figure('name', ['All Mice-',typeNs], 'position', [27   404   419   521]);         
    [~,binsSh] = plotHist([rPop_EI_same_shfl_allMice{:}], [rPop_EI_diff_shfl_allMice{:}],'','',{'',''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins); 
    [~,bins] = plotHist(y1, y2, xlab,ylab,leg, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
    xlim([binsSh(1),bins(end)])
    subplot(sp(1)), title(typeNs)    
    %{
    % mark corr significancy value     
    corrSig = min([min(abs(y1([p_EI_same_allMice{:}]<=.05))), min(abs(y2([p_EI_diff_allMice{:}]<=.05)))]);
    subplot(sp(1)), title('EI')
    vline(corrSig, 'k:')
    vline(-corrSig, 'k:')
    %}
    % figure; errorbar(1:2, [nanmean(y1), mean(y2)], [nanstd(y1), std(y2)])
    % xlim([0,3])
    %{
    if saveFigs
        namv = sprintf('corrFR_pairwise_%s_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', typeNs, alFR, al, time2an, o2a, nowStr);
        fn = fullfile(dirn0fr, namv);
        savefig(gcf, fn)
        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end  
    %}


    typeNs = 'EE';
    leg = {'EE\_same','EE\_diff'};
    y1 = [rPop_EE_same_allMice{:}];
    y2 = [rPop_EE_diff_allMice{:}];
    sp = [232,235];
    % fh = figure('name', ['All Mice-',typeNs], 'position', [27   404   419   521]);         
    [~,binsSh] = plotHist([rPop_EE_same_shfl_allMice{:}], [rPop_EE_diff_shfl_allMice{:}],'','',{'',''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins); 
    [~,bins] = plotHist(y1, y2, xlab,ylab,leg, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
    xlim([binsSh(1),bins(end)])
    subplot(sp(1)), title(typeNs)    
    %{
    % mark corr significancy value     
    corrSig = min([min(abs(y1([p_EE_same_allMice{:}]<=.05))), min(abs(y2([p_EE_diff_allMice{:}]<=.05)))]);
    subplot(sp(1)), title('EE')
    vline(corrSig, 'k:')
    vline(-corrSig, 'k:')
    %}
    % figure; errorbar(1:2, [nanmean(y1), mean(y2)], [nanstd(y1), std(y2)])
    % xlim([0,3])
    %{
    if saveFigs
        namv = sprintf('corrFR_pairwise_%s_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', typeNs, alFR, al, time2an, o2a, nowStr);
        fn = fullfile(dirn0fr, namv);
        savefig(gcf, fn)
        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end  
    %}


    typeNs = 'II';
    leg = {'II\_same','II\_diff'};
    y1 = [rPop_II_same_allMice{:}];
    y2 = [rPop_II_diff_allMice{:}];
    sp = [233,236];
    % fh = figure('name', ['All Mice-',typeNs], 'position', [27   404   419   521]);         
    [~,binsSh] = plotHist([rPop_II_same_shfl_allMice{:}], [rPop_II_diff_shfl_allMice{:}],'','',{'',''}, cols, [], fh, nBins, doSmooth, linestylesShfl, sp, bins); 
    [~,bins] = plotHist(y1, y2, xlab,ylab,leg, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
    xlim([binsSh(1),bins(end)])
    subplot(sp(1)), title(typeNs)    
    %{
    % mark corr significancy value     
    corrSig = min([min(abs(y1([p_II_same_allMice{:}]<=.05))), min(abs(y2([p_II_diff_allMice{:}]<=.05)))]);
    subplot(sp(1)), title('II')
    vline(corrSig, 'k:')
    vline(-corrSig, 'k:')
    %}
    % figure; errorbar(1:2, [nanmean(y1), mean(y2)], [nanstd(y1), std(y2)])
    % xlim([0,3])


    if saveFigs
        namv = sprintf('corrFR_popAveFRs_distMiceDaysPooled_sameDiffTun_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);

        d = fullfile(dirn0fr, 'sumAllMice', nnow);
        if ~exist(d,'dir')
            mkdir(d)
        end
        fn = fullfile(d, namv);

        savefig(gcf, fn)
        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end    

end





%% How tuning of neurons changes with the strength of correlations
% run the following script: 
% corr_excInh_plots_roc




%% Average FRs across trials for each neuron

frAv_Ei_eachDay = cell(1, length(mice));
frAv_Ec_eachDay = cell(1, length(mice));
frAv_Ii_eachDay = cell(1, length(mice));
frAv_Ic_eachDay = cell(1, length(mice));
frAv_E_eachDay = cell(1, length(mice));
frAv_I_eachDay = cell(1, length(mice));
frAv_ipsi_eachDay = cell(1, length(mice));
frAv_contra_eachDay = cell(1, length(mice));

frAv_Ei_eachDay_ipsiTrs = cell(1, length(mice));
frAv_Ec_eachDay_ipsiTrs = cell(1, length(mice));
frAv_Ii_eachDay_ipsiTrs = cell(1, length(mice));
frAv_Ic_eachDay_ipsiTrs = cell(1, length(mice));
frAv_E_eachDay_ipsiTrs = cell(1, length(mice));
frAv_I_eachDay_ipsiTrs = cell(1, length(mice));
frAv_ipsi_eachDay_ipsiTrs = cell(1, length(mice));
frAv_contra_eachDay_ipsiTrs = cell(1, length(mice));

frAv_Ei_eachDay_contraTrs = cell(1, length(mice));
frAv_Ec_eachDay_contraTrs = cell(1, length(mice));
frAv_Ii_eachDay_contraTrs = cell(1, length(mice));
frAv_Ic_eachDay_contraTrs = cell(1, length(mice));
frAv_E_eachDay_contraTrs = cell(1, length(mice));
frAv_I_eachDay_contraTrs = cell(1, length(mice));
frAv_ipsi_eachDay_contraTrs = cell(1, length(mice));
frAv_contra_eachDay_contraTrs = cell(1, length(mice));

for im = 1 : length(mice)
    for iday = 1:numDaysAll(im)        
        if mnTrNum_allMice{im}(iday) >= thMinTrs               
            %{
            if ipsi_contra_all(1)==1
                trs = ipsiTrs_allDays_allMice{im}{iday};
            elseif ipsi_contra_all(2)==1
                trs = contraTrs_allDays_allMice{im}{iday};
            elseif ipsi_contra_all(3)==1
                trs = true(1, size(fr_exc_ipsi_timeM1{im}{iday}, 2)); %1:length(ipsiTrs_allDays_allMice{im}{iday});
            end
            %}
            
            %%%% average across all trials
            trs = true(1, size(fr_exc_ipsi_timeM1{im}{iday}, 2)); %1:length(ipsiTrs_allDays_allMice{im}{iday});
            frAv_Ei_eachDay{im}{iday} = mean(fr_exc_ipsi_timeM1{im}{iday}(:,trs), 2); % nNeurons
            frAv_Ec_eachDay{im}{iday} = mean(fr_exc_contra_timeM1{im}{iday}(:,trs), 2);
            frAv_Ii_eachDay{im}{iday} = mean(fr_inh_ipsi_timeM1{im}{iday}(:,trs), 2); % nNeurons
            frAv_Ic_eachDay{im}{iday} = mean(fr_inh_contra_timeM1{im}{iday}(:,trs), 2);

            frAv_E_eachDay{im}{iday} = [frAv_Ei_eachDay{im}{iday}; frAv_Ec_eachDay{im}{iday}];
            frAv_I_eachDay{im}{iday} = [frAv_Ii_eachDay{im}{iday}; frAv_Ic_eachDay{im}{iday}];
            
            frAv_ipsi_eachDay{im}{iday} = [frAv_Ei_eachDay{im}{iday}; frAv_Ii_eachDay{im}{iday}];
            frAv_contra_eachDay{im}{iday} = [frAv_Ec_eachDay{im}{iday}; frAv_Ic_eachDay{im}{iday}];  
            
            
            if size(fr_exc_ipsi_timeM1{im}{iday},2) ~= length(ipsiTrs_allDays_allMice{im}{iday})
                a = [ipsiTrs_allDays_allMice{im}{iday} , contraTrs_allDays_allMice{im}{iday}]; 
                f = find(sum(a,2)==0);
                ipsiTrs_allDays_allMice{im}{iday}(f) = [];
                contraTrs_allDays_allMice{im}{iday}(f) = [];
            end
            
            
            %%%% average across ipsi trials
            trs = ipsiTrs_allDays_allMice{im}{iday};
            frAv_Ei_eachDay_ipsiTrs{im}{iday} = mean(fr_exc_ipsi_timeM1{im}{iday}(:,trs), 2); % nNeurons
            frAv_Ec_eachDay_ipsiTrs{im}{iday} = mean(fr_exc_contra_timeM1{im}{iday}(:,trs), 2);
            frAv_Ii_eachDay_ipsiTrs{im}{iday} = mean(fr_inh_ipsi_timeM1{im}{iday}(:,trs), 2); % nNeurons
            frAv_Ic_eachDay_ipsiTrs{im}{iday} = mean(fr_inh_contra_timeM1{im}{iday}(:,trs), 2);

            frAv_E_eachDay_ipsiTrs{im}{iday} = [frAv_Ei_eachDay_ipsiTrs{im}{iday}; frAv_Ec_eachDay_ipsiTrs{im}{iday}];
            frAv_I_eachDay_ipsiTrs{im}{iday} = [frAv_Ii_eachDay_ipsiTrs{im}{iday}; frAv_Ic_eachDay_ipsiTrs{im}{iday}];
            
            frAv_ipsi_eachDay_ipsiTrs{im}{iday} = [frAv_Ei_eachDay_ipsiTrs{im}{iday}; frAv_Ii_eachDay_ipsiTrs{im}{iday}];
            frAv_contra_eachDay_ipsiTrs{im}{iday} = [frAv_Ec_eachDay_ipsiTrs{im}{iday}; frAv_Ic_eachDay_ipsiTrs{im}{iday}];            
            
            
            %%%% average across contra trials
            trs = contraTrs_allDays_allMice{im}{iday};
            frAv_Ei_eachDay_contraTrs{im}{iday} = mean(fr_exc_ipsi_timeM1{im}{iday}(:,trs), 2); % nNeurons
            frAv_Ec_eachDay_contraTrs{im}{iday} = mean(fr_exc_contra_timeM1{im}{iday}(:,trs), 2);
            frAv_Ii_eachDay_contraTrs{im}{iday} = mean(fr_inh_ipsi_timeM1{im}{iday}(:,trs), 2); % nNeurons
            frAv_Ic_eachDay_contraTrs{im}{iday} = mean(fr_inh_contra_timeM1{im}{iday}(:,trs), 2);

            frAv_E_eachDay_contraTrs{im}{iday} = [frAv_Ei_eachDay_contraTrs{im}{iday}; frAv_Ec_eachDay_contraTrs{im}{iday}];
            frAv_I_eachDay_contraTrs{im}{iday} = [frAv_Ii_eachDay_contraTrs{im}{iday}; frAv_Ic_eachDay_contraTrs{im}{iday}];
            
            frAv_ipsi_eachDay_contraTrs{im}{iday} = [frAv_Ei_eachDay_contraTrs{im}{iday}; frAv_Ii_eachDay_contraTrs{im}{iday}];
            frAv_contra_eachDay_contraTrs{im}{iday} = [frAv_Ec_eachDay_contraTrs{im}{iday}; frAv_Ic_eachDay_contraTrs{im}{iday}];
            
        end
    end
end


%% Pool all days for each mouse

%%%%%%%% all trials %%%%%%%%
frAv_Ei_allDays = cell(1, length(mice));
frAv_Ec_allDays = cell(1, length(mice));
frAv_Ii_allDays = cell(1, length(mice));
frAv_Ic_allDays = cell(1, length(mice));
frAv_E_allDays = cell(1, length(mice));
frAv_I_allDays = cell(1, length(mice));
frAv_ipsi_allDays = cell(1, length(mice));
frAv_contra_allDays = cell(1, length(mice));

for im = 1 : length(mice)
    frAv_Ei_allDays{im} = vertcat(frAv_Ei_eachDay{im}{:});
    frAv_Ec_allDays{im} = vertcat(frAv_Ec_eachDay{im}{:});
    frAv_Ii_allDays{im} = vertcat(frAv_Ii_eachDay{im}{:});
    frAv_Ic_allDays{im} = vertcat(frAv_Ic_eachDay{im}{:});
    % E vs I
    frAv_E_allDays{im} = [frAv_Ei_allDays{im} ; frAv_Ec_allDays{im}];
    frAv_I_allDays{im} = [frAv_Ii_allDays{im} ; frAv_Ic_allDays{im}];
    % ipsi vs contra
    frAv_ipsi_allDays{im} = [frAv_Ei_allDays{im} ; frAv_Ii_allDays{im}];
    frAv_contra_allDays{im} = [frAv_Ec_allDays{im} ; frAv_Ic_allDays{im}];        
end


%%%%%%%% ipsi trials %%%%%%%%
frAv_Ei_allDays_ipsiTrs = cell(1, length(mice));
frAv_Ec_allDays_ipsiTrs = cell(1, length(mice));
frAv_Ii_allDays_ipsiTrs = cell(1, length(mice));
frAv_Ic_allDays_ipsiTrs = cell(1, length(mice));
frAv_E_allDays_ipsiTrs = cell(1, length(mice));
frAv_I_allDays_ipsiTrs = cell(1, length(mice));
frAv_ipsi_allDays_ipsiTrs = cell(1, length(mice));
frAv_contra_allDays_ipsiTrs = cell(1, length(mice));

for im = 1 : length(mice)
    frAv_Ei_allDays_ipsiTrs{im} = vertcat(frAv_Ei_eachDay_ipsiTrs{im}{:});
    frAv_Ec_allDays_ipsiTrs{im} = vertcat(frAv_Ec_eachDay_ipsiTrs{im}{:});
    frAv_Ii_allDays_ipsiTrs{im} = vertcat(frAv_Ii_eachDay_ipsiTrs{im}{:});
    frAv_Ic_allDays_ipsiTrs{im} = vertcat(frAv_Ic_eachDay_ipsiTrs{im}{:});
    % E vs I
    frAv_E_allDays_ipsiTrs{im} = [frAv_Ei_allDays_ipsiTrs{im} ; frAv_Ec_allDays_ipsiTrs{im}];
    frAv_I_allDays_ipsiTrs{im} = [frAv_Ii_allDays_ipsiTrs{im} ; frAv_Ic_allDays_ipsiTrs{im}];
    % ipsi vs contra
    frAv_ipsi_allDays_ipsiTrs{im} = [frAv_Ei_allDays_ipsiTrs{im} ; frAv_Ii_allDays_ipsiTrs{im}];
    frAv_contra_allDays_ipsiTrs{im} = [frAv_Ec_allDays_ipsiTrs{im} ; frAv_Ic_allDays_ipsiTrs{im}];        
end


%%%%%%%% contra trials %%%%%%%%
frAv_Ei_allDays_contraTrs = cell(1, length(mice));
frAv_Ec_allDays_contraTrs = cell(1, length(mice));
frAv_Ii_allDays_contraTrs = cell(1, length(mice));
frAv_Ic_allDays_contraTrs = cell(1, length(mice));
frAv_E_allDays_contraTrs = cell(1, length(mice));
frAv_I_allDays_contraTrs = cell(1, length(mice));
frAv_ipsi_allDays_contraTrs = cell(1, length(mice));
frAv_contra_allDays_contraTrs = cell(1, length(mice));

for im = 1 : length(mice)
    frAv_Ei_allDays_contraTrs{im} = vertcat(frAv_Ei_eachDay_contraTrs{im}{:});
    frAv_Ec_allDays_contraTrs{im} = vertcat(frAv_Ec_eachDay_contraTrs{im}{:});
    frAv_Ii_allDays_contraTrs{im} = vertcat(frAv_Ii_eachDay_contraTrs{im}{:});
    frAv_Ic_allDays_contraTrs{im} = vertcat(frAv_Ic_eachDay_contraTrs{im}{:});
    % E vs I
    frAv_E_allDays_contraTrs{im} = [frAv_Ei_allDays_contraTrs{im} ; frAv_Ec_allDays_contraTrs{im}];
    frAv_I_allDays_contraTrs{im} = [frAv_Ii_allDays_contraTrs{im} ; frAv_Ic_allDays_contraTrs{im}];
    % ipsi vs contra
    frAv_ipsi_allDays_contraTrs{im} = [frAv_Ei_allDays_contraTrs{im} ; frAv_Ii_allDays_contraTrs{im}];
    frAv_contra_allDays_contraTrs{im} = [frAv_Ec_allDays_contraTrs{im} ; frAv_Ic_allDays_contraTrs{im}];        
end




%% Pool all mice

%%% all trials
frAv_Ei = vertcat(frAv_Ei_allDays{:});
frAv_Ec = vertcat(frAv_Ec_allDays{:});
frAv_Ii = vertcat(frAv_Ii_allDays{:});
frAv_Ic = vertcat(frAv_Ic_allDays{:});

%%% ispi trials
frAv_Ei_ipsiTrs = vertcat(frAv_Ei_allDays_ipsiTrs{:});
frAv_Ec_ipsiTrs = vertcat(frAv_Ec_allDays_ipsiTrs{:});
frAv_Ii_ipsiTrs = vertcat(frAv_Ii_allDays_ipsiTrs{:});
frAv_Ic_ipsiTrs = vertcat(frAv_Ic_allDays_ipsiTrs{:});

%%% contra trials
frAv_Ei_contraTrs = vertcat(frAv_Ei_allDays_contraTrs{:});
frAv_Ec_contraTrs = vertcat(frAv_Ec_allDays_contraTrs{:});
frAv_Ii_contraTrs = vertcat(frAv_Ii_allDays_contraTrs{:});
frAv_Ic_contraTrs = vertcat(frAv_Ic_allDays_contraTrs{:});


%%
if doPlots    
    
    %% Plot errorbar, showing each mouse, ave +/- se of FR across neurons of all days

    marg = .2;

    figure('name', 'All mice', 'position', [14   636   661   290]); 
    set(gca, 'position', [0.2919    0.1908    0.5229    0.7095])

    
    %%%%%%%%%% Plot E,I,ipsi,contra %%%%%%%%%%%
    x = 1:1.5:length(mice)*1.5;
    gp = .2;
    typeNs = 'EI';    
    subplot(131); hold on
    h1 = errorbar(x, cellfun(@mean, frAv_Ei_allDays), cellfun(@std, frAv_Ei_allDays) ./ sqrt(cellfun(@length, frAv_Ei_allDays)), 'k.', 'linestyle', 'none');
    h2 = errorbar(x+gp, cellfun(@mean, frAv_Ec_allDays), cellfun(@std, frAv_Ec_allDays) ./ sqrt(cellfun(@length, frAv_Ec_allDays)), 'color',rgb('gray'), 'marker', '.', 'linestyle', 'none');
    h1 = errorbar(x+3*gp, cellfun(@mean, frAv_Ii_allDays), cellfun(@std, frAv_Ii_allDays) ./ sqrt(cellfun(@length, frAv_Ii_allDays)), 'r.', 'linestyle', 'none');
    h2 = errorbar(x+4*gp, cellfun(@mean, frAv_Ic_allDays), cellfun(@std, frAv_Ic_allDays) ./ sqrt(cellfun(@length, frAv_Ic_allDays)), 'color',rgb('lightsalmon'), 'marker', '.', 'linestyle', 'none');    
%     xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+2*gp)
    set(gca,'xticklabel', mice)
    ylabel('corr (mean +/- se days)')
    legend({'Ei','Ec', 'Ii', 'Ic'})
    xlim([x(1)-marg, x(end)+4*gp+marg])


    %%%%%%%%%% Plot E vs I %%%%%%%%%%%
    gp = .2;
    x = 1:length(mice);
    subplot(132); hold on
    h1 = errorbar(x, cellfun(@mean, frAv_E_allDays), cellfun(@std, frAv_E_allDays) ./ sqrt(cellfun(@length, frAv_E_allDays)), 'k.', 'linestyle', 'none');
    h2 = errorbar(x+gp, cellfun(@mean, frAv_I_allDays), cellfun(@std, frAv_I_allDays) ./ sqrt(cellfun(@length, frAv_I_allDays)), 'r.', 'linestyle', 'none');
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    legend('exc', 'inh')

    
    %%%%%%%%%% Plot ispi vs contra %%%%%%%%%%%
    subplot(133); hold on
    h1 = errorbar(x, cellfun(@mean, frAv_ipsi_allDays), cellfun(@std, frAv_ipsi_allDays) ./ sqrt(cellfun(@length, frAv_ipsi_allDays)), 'k.', 'linestyle', 'none');
    h2 = errorbar(x+gp, cellfun(@mean, frAv_contra_allDays), cellfun(@std, frAv_contra_allDays) ./ sqrt(cellfun(@length, frAv_contra_allDays)), 'r.', 'linestyle', 'none');
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    legend('ipsi', 'contra')


    if saveFigs
        namv = sprintf('FR_trAved_aveSeAllNs_EI_ipsiContra_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);

        d = fullfile(dirn0fr, 'sumAllMice', nnow);
        fn = fullfile(d, namv);

        savefig(gcf, fn)
        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end    

    
    %% Histogram; each mouse: pool days of all mice and get a histogram...
    
    linestylesE = {'--','--'};
    nBins = 50;
    doSmooth = 0;
    xlab = 'FR (trial-averaged)';
    ylab = 'Fract Ns of all days';

    for im = 1 : length(mice)
        fh = figure('name', mice{im}, 'position', [27         404        1370         521]);
        
        % set the bins ... same for all plots
        ally = [frAv_Ei_allDays{im}; frAv_Ec_allDays{im}; frAv_Ii_allDays{im}; frAv_Ic_allDays{im}];
        r1 = round(min(ally)-.05, 1);  %min(ally); %   % round(min(ally), 1);
        r2 = round(max(ally)+.05, 1);
        bins = r1 : (r2-r1)/nBins : r2;
    
        sp = [231,234];
        plotHist(frAv_Ei_allDays{im}, frAv_Ec_allDays{im}, '', '', {'',''}, cols, [], fh, nBins, doSmooth, linestylesE, sp, bins); 
        plotHist(frAv_Ii_allDays{im}, frAv_Ic_allDays{im}, xlab, ylab, {'',''}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
        xlim([bins(1),bins(end)])
        legend('Ei','Ec','Ii','Ic')
        subplot(sp(1)), legend('Ei','Ec','Ii','Ic')

        % exc vs inh
        sp = [232,235];
        [~,bins] = plotHist([frAv_Ei_allDays{im} ; frAv_Ec_allDays{im}] , [frAv_Ii_allDays{im} ; frAv_Ic_allDays{im}], xlab, ylab, {'E','I'}, {'k','k'}, [], fh, nBins, doSmooth, {'--','-'}, sp, bins); 

        % ipsi vs contra    
        sp = [233,236];
        [~,bins] = plotHist([frAv_Ei_allDays{im} ; frAv_Ii_allDays{im}] , [frAv_Ec_allDays{im} ; frAv_Ic_allDays{im}], xlab, ylab, {'ipsi','contra'}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 


        if saveFigs
            namv = sprintf('FR_trAved_distNsDaysPooled_EI_ipsiContra_FR%s_ROC%s_%s_curr%s_%s', alFR, al, time2an, o2a, nowStr);

            d = fullfile(dirn0fr, mice{im}, nnow);
            fn = fullfile(dirn0fr, mice{im}, nnow, namv); 

            savefig(gcf, fn)
        end      
    end


    %% Histogram; All mice: pool days of all mice and get a histogram...

    nBins = 50;
    doSmooth = 0;

    fh = figure('name', 'All mice', 'position', [27         404        1370         521]);

    % set the bins ... same for all plots
    ally = [frAv_Ei ; frAv_Ec ; frAv_Ii ; frAv_Ic];
    r1 = min(ally); %round(min(ally)-.05, 1);  %min(ally); %   % round(min(ally), 1);
    r2 = round(max(ally)+.05, 1);
    bins = r1 : (r2-r1)/nBins : r2;
        
    sp = [231,234];
    plotHist(frAv_Ei, frAv_Ec, '', '', {'',''}, cols, [], fh, nBins, doSmooth, linestylesE, sp, bins); 
    plotHist(frAv_Ii, frAv_Ic, xlab, ylab, {'',''}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
%     xlim([binsE(1),bins(end)])
    legend('Ei','Ec','Ii','Ic')
    subplot(sp(1)), legend('Ei','Ec','Ii','Ic')

    % exc vs inh
    sp = [232,235];
    plotHist([frAv_Ei ; frAv_Ec] , [frAv_Ii ; frAv_Ic], xlab, ylab, {'E','I'}, {'k','k'}, [], fh, nBins, doSmooth, {'--','-'}, sp, bins); 

    % ipsi vs contra    
    sp = [233,236];
    plotHist([frAv_Ei ; frAv_Ii] , [frAv_Ec ; frAv_Ic], xlab, ylab, {'ipsi','contra'}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 


    if saveFigs
        namv = sprintf('FR_trAved_distNsDaysPooled_EI_ipsiContra_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);

        d = fullfile(dirn0fr, 'sumAllMice', nnow);
        fn = fullfile(d, namv);

        savefig(gcf, fn)
        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end    



    %% Compare FR change across training days: each mouse

    for im = 1 : length(mice)
        fh = figure('name', mice{im}, 'position', [21   134   505   740]);

        subplot(311); hold on
        plot(cellfun(@mean, frAv_Ei_eachDay{im}), 'k')
        plot(cellfun(@mean, frAv_Ec_eachDay{im}), 'color', rgb('lightsalmon'))
        plot(cellfun(@mean, frAv_Ii_eachDay{im}), 'r')
        plot(cellfun(@mean, frAv_Ic_eachDay{im}), 'color', rgb('gray'))
        xlabel('Training days')
        ylabel('FR')

        subplot(312); hold on
        plot(cellfun(@mean, frAv_E_eachDay{im}), 'k')
        plot(cellfun(@mean, frAv_I_eachDay{im}), 'r')
        xlabel('Training days')
        ylabel('FR')

        subplot(313); hold on
        plot(cellfun(@mean, frAv_ipsi_eachDay{im}), 'k')
        plot(cellfun(@mean, frAv_contra_eachDay{im}), 'r')
        xlabel('Training days')
        ylabel('FR')        
        
        if saveFigs
            namv = sprintf('FR_trAved_trainingDays_EI_ipsiContra_FR%s_ROC%s_%s_curr%s_%s_%s', alFR, al, time2an, o2a, mice{im}, nowStr);

            d = fullfile(dirn0fr, mice{im}, nnow);
            fn = fullfile(dirn0fr, mice{im}, nnow, namv);

            savefig(gcf, fn)
        end                    
    end
    
    
    
    
    %%
    %%%%%%%%%%%%%%%% same as above but for ipsi trials %%%%%%%%%%%%%%%%
    
    %% Plot errorbar, showing each mouse, ave +/- se of FR across neurons of all days

    marg = .2;

    figure('name', 'All mice', 'position', [14   636   661   290]); 
    set(gca, 'position', [0.2919    0.1908    0.5229    0.7095])

    
    %%%%%%%%%% Plot E,I,ipsi,contra %%%%%%%%%%%
    x = 1:1.5:length(mice)*1.5;
    gp = .2;
    typeNs = 'EI';    
    subplot(131); hold on
    h1 = errorbar(x, cellfun(@mean, frAv_Ei_allDays_ipsiTrs), cellfun(@std, frAv_Ei_allDays_ipsiTrs) ./ sqrt(cellfun(@length, frAv_Ei_allDays_ipsiTrs)), 'k.', 'linestyle', 'none');
    h2 = errorbar(x+gp, cellfun(@mean, frAv_Ec_allDays_ipsiTrs), cellfun(@std, frAv_Ec_allDays_ipsiTrs) ./ sqrt(cellfun(@length, frAv_Ec_allDays_ipsiTrs)), 'color',rgb('gray'), 'marker', '.', 'linestyle', 'none');
    h1 = errorbar(x+3*gp, cellfun(@mean, frAv_Ii_allDays_ipsiTrs), cellfun(@std, frAv_Ii_allDays_ipsiTrs) ./ sqrt(cellfun(@length, frAv_Ii_allDays_ipsiTrs)), 'r.', 'linestyle', 'none');
    h2 = errorbar(x+4*gp, cellfun(@mean, frAv_Ic_allDays_ipsiTrs), cellfun(@std, frAv_Ic_allDays_ipsiTrs) ./ sqrt(cellfun(@length, frAv_Ic_allDays_ipsiTrs)), 'color',rgb('lightsalmon'), 'marker', '.', 'linestyle', 'none');    
%     xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+2*gp)
    set(gca,'xticklabel', mice)
    ylabel('corr (mean +/- se days)')
    legend({'Ei','Ec', 'Ii', 'Ic'})
    xlim([x(1)-marg, x(end)+4*gp+marg])


    %%%%%%%%%% Plot E vs I %%%%%%%%%%%
    gp = .2;
    x = 1:length(mice);
    subplot(132); hold on
    h1 = errorbar(x, cellfun(@mean, frAv_E_allDays_ipsiTrs), cellfun(@std, frAv_E_allDays_ipsiTrs) ./ sqrt(cellfun(@length, frAv_E_allDays_ipsiTrs)), 'k.', 'linestyle', 'none');
    h2 = errorbar(x+gp, cellfun(@mean, frAv_I_allDays_ipsiTrs), cellfun(@std, frAv_I_allDays_ipsiTrs) ./ sqrt(cellfun(@length, frAv_I_allDays_ipsiTrs)), 'r.', 'linestyle', 'none');
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    legend('exc', 'inh')

    
    %%%%%%%%%% Plot ispi vs contra %%%%%%%%%%%
    subplot(133); hold on
    h1 = errorbar(x, cellfun(@mean, frAv_ipsi_allDays_ipsiTrs), cellfun(@std, frAv_ipsi_allDays_ipsiTrs) ./ sqrt(cellfun(@length, frAv_ipsi_allDays_ipsiTrs)), 'k.', 'linestyle', 'none');
    h2 = errorbar(x+gp, cellfun(@mean, frAv_contra_allDays_ipsiTrs), cellfun(@std, frAv_contra_allDays_ipsiTrs) ./ sqrt(cellfun(@length, frAv_contra_allDays_ipsiTrs)), 'r.', 'linestyle', 'none');
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    legend('ipsi', 'contra')


    if saveFigs
        namv = sprintf('FR_trAved_aveSeAllNs_EI_ipsiContra_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);

        d = fullfile(dirn0fr, 'sumAllMice', nnow);
        fn = fullfile(d, namv);

        savefig(gcf, fn)
        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end    

    
    %% Histogram; each mouse: pool days of all mice and get a histogram...
    
    linestylesE = {'--','--'};
    nBins = 50;
    doSmooth = 0;
    xlab = 'FR (trial-averaged)';
    ylab = 'Fract Ns of all days';

    for im = 1 : length(mice)
        fh = figure('name', mice{im}, 'position', [27         404        1370         521]);
        
        % set the bins ... same for all plots
        ally = [frAv_Ei_allDays_ipsiTrs{im}; frAv_Ec_allDays_ipsiTrs{im}; frAv_Ii_allDays_ipsiTrs{im}; frAv_Ic_allDays_ipsiTrs{im}];
        r1 = round(min(ally)-.05, 1);  %min(ally); %   % round(min(ally), 1);
        r2 = round(max(ally)+.05, 1);
        bins = r1 : (r2-r1)/nBins : r2;
    
        sp = [231,234];
        plotHist(frAv_Ei_allDays_ipsiTrs{im}, frAv_Ec_allDays_ipsiTrs{im}, '', '', {'',''}, cols, [], fh, nBins, doSmooth, linestylesE, sp, bins); 
        plotHist(frAv_Ii_allDays_ipsiTrs{im}, frAv_Ic_allDays_ipsiTrs{im}, xlab, ylab, {'',''}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
        xlim([bins(1),bins(end)])
        legend('Ei','Ec','Ii','Ic')
        subplot(sp(1)), legend('Ei','Ec','Ii','Ic')

        % exc vs inh
        sp = [232,235];
        [~,bins] = plotHist([frAv_Ei_allDays_ipsiTrs{im} ; frAv_Ec_allDays_ipsiTrs{im}] , [frAv_Ii_allDays_ipsiTrs{im} ; frAv_Ic_allDays_ipsiTrs{im}], xlab, ylab, {'E','I'}, {'k','k'}, [], fh, nBins, doSmooth, {'--','-'}, sp, bins); 

        % ipsi vs contra    
        sp = [233,236];
        [~,bins] = plotHist([frAv_Ei_allDays_ipsiTrs{im} ; frAv_Ii_allDays_ipsiTrs{im}] , [frAv_Ec_allDays_ipsiTrs{im} ; frAv_Ic_allDays_ipsiTrs{im}], xlab, ylab, {'ipsi','contra'}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 


        if saveFigs
            namv = sprintf('FR_trAved_distNsDaysPooled_EI_ipsiContra_FR%s_ROC%s_%s_curr%s_%s', alFR, al, time2an, o2a, nowStr);

            d = fullfile(dirn0fr, mice{im}, nnow);
            fn = fullfile(dirn0fr, mice{im}, nnow, namv); 

            savefig(gcf, fn)
        end      
    end


    %% Histogram; All mice: pool days of all mice and get a histogram...

    nBins = 50;
    doSmooth = 0;

    fh = figure('name', 'All mice', 'position', [27         404        1370         521]);

    % set the bins ... same for all plots
    ally = [frAv_Ei_ipsiTrs ; frAv_Ec_ipsiTrs ; frAv_Ii_ipsiTrs ; frAv_Ic_ipsiTrs];
    r1 = min(ally); %round(min(ally)-.05, 1);  %min(ally); %   % round(min(ally), 1);
    r2 = round(max(ally)+.05, 1);
    bins = r1 : (r2-r1)/nBins : r2;
        
    sp = [231,234];
    plotHist(frAv_Ei_ipsiTrs, frAv_Ec_ipsiTrs, '', '', {'',''}, cols, [], fh, nBins, doSmooth, linestylesE, sp, bins); 
    plotHist(frAv_Ii_ipsiTrs, frAv_Ic_ipsiTrs, xlab, ylab, {'',''}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
%     xlim([binsE(1),bins(end)])
    legend('Ei','Ec','Ii','Ic')
    subplot(sp(1)), legend('Ei','Ec','Ii','Ic')

    % exc vs inh
    sp = [232,235];
    plotHist([frAv_Ei_ipsiTrs ; frAv_Ec_ipsiTrs] , [frAv_Ii_ipsiTrs ; frAv_Ic_ipsiTrs], xlab, ylab, {'E','I'}, {'k','k'}, [], fh, nBins, doSmooth, {'--','-'}, sp, bins); 

    % ipsi vs contra    
    sp = [233,236];
    plotHist([frAv_Ei_ipsiTrs ; frAv_Ii_ipsiTrs] , [frAv_Ec_ipsiTrs ; frAv_Ic_ipsiTrs], xlab, ylab, {'ipsi','contra'}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 


    if saveFigs
        namv = sprintf('FR_trAved_distNsDaysPooled_EI_ipsiContra_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);

        d = fullfile(dirn0fr, 'sumAllMice', nnow);
        fn = fullfile(d, namv);

        savefig(gcf, fn)
        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end    



    %% Compare FR change across training days: each mouse

    for im = 1 : length(mice)
        fh = figure('name', mice{im}, 'position', [21   134   505   740]);

        subplot(311); hold on
        plot(cellfun(@mean, frAv_Ei_eachDay_ipsiTrs{im}), 'k')
        plot(cellfun(@mean, frAv_Ec_eachDay_ipsiTrs{im}), 'color', rgb('lightsalmon'))
        plot(cellfun(@mean, frAv_Ii_eachDay_ipsiTrs{im}), 'r')
        plot(cellfun(@mean, frAv_Ic_eachDay_ipsiTrs{im}), 'color', rgb('gray'))
        xlabel('Training days')
        ylabel('FR')

        subplot(312); hold on
        plot(cellfun(@mean, frAv_E_eachDay_ipsiTrs{im}), 'k')
        plot(cellfun(@mean, frAv_I_eachDay_ipsiTrs{im}), 'r')
        xlabel('Training days')
        ylabel('FR')

        subplot(313); hold on
        plot(cellfun(@mean, frAv_ipsi_eachDay_ipsiTrs{im}), 'k')
        plot(cellfun(@mean, frAv_contra_eachDay_ipsiTrs{im}), 'r')
        xlabel('Training days')
        ylabel('FR')        
        
        if saveFigs
            namv = sprintf('FR_trAved_trainingDays_EI_ipsiContra_FR%s_ROC%s_%s_curr%s_%s_%s', alFR, al, time2an, o2a, mice{im}, nowStr);

            d = fullfile(dirn0fr, mice{im}, nnow);
            fn = fullfile(dirn0fr, mice{im}, nnow, namv);

            savefig(gcf, fn)
        end                    
    end
    
    
    
    
    %%
    %%%%%%%%%%%%%%%% same as above but for contra trials %%%%%%%%%%%%%%%%
    
    %% Plot errorbar, showing each mouse, ave +/- se of FR across neurons of all days

    marg = .2;

    figure('name', 'All mice', 'position', [14   636   661   290]); 
    set(gca, 'position', [0.2919    0.1908    0.5229    0.7095])

    
    %%%%%%%%%% Plot E,I,ipsi,contra %%%%%%%%%%%
    x = 1:1.5:length(mice)*1.5;
    gp = .2;
    typeNs = 'EI';    
    subplot(131); hold on
    h1 = errorbar(x, cellfun(@mean, frAv_Ei_allDays_contraTrs), cellfun(@std, frAv_Ei_allDays_contraTrs) ./ sqrt(cellfun(@length, frAv_Ei_allDays_contraTrs)), 'k.', 'linestyle', 'none');
    h2 = errorbar(x+gp, cellfun(@mean, frAv_Ec_allDays_contraTrs), cellfun(@std, frAv_Ec_allDays_contraTrs) ./ sqrt(cellfun(@length, frAv_Ec_allDays_contraTrs)), 'color',rgb('gray'), 'marker', '.', 'linestyle', 'none');
    h1 = errorbar(x+3*gp, cellfun(@mean, frAv_Ii_allDays_contraTrs), cellfun(@std, frAv_Ii_allDays_contraTrs) ./ sqrt(cellfun(@length, frAv_Ii_allDays_contraTrs)), 'r.', 'linestyle', 'none');
    h2 = errorbar(x+4*gp, cellfun(@mean, frAv_Ic_allDays_contraTrs), cellfun(@std, frAv_Ic_allDays_contraTrs) ./ sqrt(cellfun(@length, frAv_Ic_allDays_contraTrs)), 'color',rgb('lightsalmon'), 'marker', '.', 'linestyle', 'none');    
%     xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+2*gp)
    set(gca,'xticklabel', mice)
    ylabel('corr (mean +/- se days)')
    legend({'Ei','Ec', 'Ii', 'Ic'})
    xlim([x(1)-marg, x(end)+4*gp+marg])


    %%%%%%%%%% Plot E vs I %%%%%%%%%%%
    gp = .2;
    x = 1:length(mice);
    subplot(132); hold on
    h1 = errorbar(x, cellfun(@mean, frAv_E_allDays_contraTrs), cellfun(@std, frAv_E_allDays_contraTrs) ./ sqrt(cellfun(@length, frAv_E_allDays_contraTrs)), 'k.', 'linestyle', 'none');
    h2 = errorbar(x+gp, cellfun(@mean, frAv_I_allDays_contraTrs), cellfun(@std, frAv_I_allDays_contraTrs) ./ sqrt(cellfun(@length, frAv_I_allDays_contraTrs)), 'r.', 'linestyle', 'none');
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    legend('exc', 'inh')

    
    %%%%%%%%%% Plot ispi vs contra %%%%%%%%%%%
    subplot(133); hold on
    h1 = errorbar(x, cellfun(@mean, frAv_ipsi_allDays_contraTrs), cellfun(@std, frAv_ipsi_allDays_contraTrs) ./ sqrt(cellfun(@length, frAv_ipsi_allDays_contraTrs)), 'k.', 'linestyle', 'none');
    h2 = errorbar(x+gp, cellfun(@mean, frAv_contra_allDays_contraTrs), cellfun(@std, frAv_contra_allDays_contraTrs) ./ sqrt(cellfun(@length, frAv_contra_allDays_contraTrs)), 'r.', 'linestyle', 'none');
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp/2)
    set(gca,'xticklabel', mice)
    legend('ipsi', 'contra')


    if saveFigs
        namv = sprintf('FR_trAved_aveSeAllNs_EI_ipsiContra_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);

        d = fullfile(dirn0fr, 'sumAllMice', nnow);
        fn = fullfile(d, namv);

        savefig(gcf, fn)
        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end    

    
    %% Histogram; each mouse: pool days of all mice and get a histogram...
    
    linestylesE = {'--','--'};
    nBins = 50;
    doSmooth = 0;
    xlab = 'FR (trial-averaged)';
    ylab = 'Fract Ns of all days';

    for im = 1 : length(mice)
        fh = figure('name', mice{im}, 'position', [27         404        1370         521]);
        
        % set the bins ... same for all plots
        ally = [frAv_Ei_allDays_contraTrs{im}; frAv_Ec_allDays_contraTrs{im}; frAv_Ii_allDays_contraTrs{im}; frAv_Ic_allDays_contraTrs{im}];
        r1 = round(min(ally)-.05, 1);  %min(ally); %   % round(min(ally), 1);
        r2 = round(max(ally)+.05, 1);
        bins = r1 : (r2-r1)/nBins : r2;
    
        sp = [231,234];
        plotHist(frAv_Ei_allDays_contraTrs{im}, frAv_Ec_allDays_contraTrs{im}, '', '', {'',''}, cols, [], fh, nBins, doSmooth, linestylesE, sp, bins); 
        plotHist(frAv_Ii_allDays_contraTrs{im}, frAv_Ic_allDays_contraTrs{im}, xlab, ylab, {'',''}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
        xlim([bins(1),bins(end)])
        legend('Ei','Ec','Ii','Ic')
        subplot(sp(1)), legend('Ei','Ec','Ii','Ic')

        % exc vs inh
        sp = [232,235];
        [~,bins] = plotHist([frAv_Ei_allDays_contraTrs{im} ; frAv_Ec_allDays_contraTrs{im}] , [frAv_Ii_allDays_contraTrs{im} ; frAv_Ic_allDays_contraTrs{im}], xlab, ylab, {'E','I'}, {'k','k'}, [], fh, nBins, doSmooth, {'--','-'}, sp, bins); 

        % ipsi vs contra    
        sp = [233,236];
        [~,bins] = plotHist([frAv_Ei_allDays_contraTrs{im} ; frAv_Ii_allDays_contraTrs{im}] , [frAv_Ec_allDays_contraTrs{im} ; frAv_Ic_allDays_contraTrs{im}], xlab, ylab, {'ipsi','contra'}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 


        if saveFigs
            namv = sprintf('FR_trAved_distNsDaysPooled_EI_ipsiContra_FR%s_ROC%s_%s_curr%s_%s', alFR, al, time2an, o2a, nowStr);

            d = fullfile(dirn0fr, mice{im}, nnow);
            fn = fullfile(dirn0fr, mice{im}, nnow, namv); 

            savefig(gcf, fn)
        end      
    end


    %% Histogram; All mice: pool days of all mice and get a histogram...

    nBins = 50;
    doSmooth = 0;

    fh = figure('name', 'All mice', 'position', [27         404        1370         521]);

    % set the bins ... same for all plots
    ally = [frAv_Ei_contraTrs ; frAv_Ec_contraTrs ; frAv_Ii_contraTrs ; frAv_Ic_contraTrs];
    r1 = min(ally); %round(min(ally)-.05, 1);  %min(ally); %   % round(min(ally), 1);
    r2 = round(max(ally)+.05, 1);
    bins = r1 : (r2-r1)/nBins : r2;
        
    sp = [231,234];
    plotHist(frAv_Ei_contraTrs, frAv_Ec_contraTrs, '', '', {'',''}, cols, [], fh, nBins, doSmooth, linestylesE, sp, bins); 
    plotHist(frAv_Ii_contraTrs, frAv_Ic_contraTrs, xlab, ylab, {'',''}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 
%     xlim([binsE(1),bins(end)])
    legend('Ei','Ec','Ii','Ic')
    subplot(sp(1)), legend('Ei','Ec','Ii','Ic')

    % exc vs inh
    sp = [232,235];
    plotHist([frAv_Ei_contraTrs ; frAv_Ec_contraTrs] , [frAv_Ii_contraTrs ; frAv_Ic_contraTrs], xlab, ylab, {'E','I'}, {'k','k'}, [], fh, nBins, doSmooth, {'--','-'}, sp, bins); 

    % ipsi vs contra    
    sp = [233,236];
    plotHist([frAv_Ei_contraTrs ; frAv_Ii_contraTrs] , [frAv_Ec_contraTrs ; frAv_Ic_contraTrs], xlab, ylab, {'ipsi','contra'}, cols, [], fh, nBins, doSmooth, linestylesData, sp, bins); 


    if saveFigs
        namv = sprintf('FR_trAved_distNsDaysPooled_EI_ipsiContra_FR%s_ROC%s_%s_curr%s_allMice_%s', alFR, al, time2an, o2a, nowStr);

        d = fullfile(dirn0fr, 'sumAllMice', nnow);
        fn = fullfile(d, namv);

        savefig(gcf, fn)
        % print to pdf
    %         axpos{1} = [0.2578    0.6276    0.4948    0.3089];     axpos{2} = [0.2578    0.1651    0.4948    0.3089];
    %         figs_adj_poster_ax(fn, axpos)                
    end    



    %% Compare FR change across training days: each mouse

    for im = 1 : length(mice)
        fh = figure('name', mice{im}, 'position', [21   134   505   740]);

        subplot(311); hold on
        plot(cellfun(@mean, frAv_Ei_eachDay_contraTrs{im}), 'k')
        plot(cellfun(@mean, frAv_Ec_eachDay_contraTrs{im}), 'color', rgb('lightsalmon'))
        plot(cellfun(@mean, frAv_Ii_eachDay_contraTrs{im}), 'r')
        plot(cellfun(@mean, frAv_Ic_eachDay_contraTrs{im}), 'color', rgb('gray'))
        xlabel('Training days')
        ylabel('FR')

        subplot(312); hold on
        plot(cellfun(@mean, frAv_E_eachDay_contraTrs{im}), 'k')
        plot(cellfun(@mean, frAv_I_eachDay_contraTrs{im}), 'r')
        xlabel('Training days')
        ylabel('FR')

        subplot(313); hold on
        plot(cellfun(@mean, frAv_ipsi_eachDay_contraTrs{im}), 'k')
        plot(cellfun(@mean, frAv_contra_eachDay_contraTrs{im}), 'r')
        xlabel('Training days')
        ylabel('FR')        
        
        if saveFigs
            namv = sprintf('FR_trAved_trainingDays_EI_ipsiContra_FR%s_ROC%s_%s_curr%s_%s_%s', alFR, al, time2an, o2a, mice{im}, nowStr);

            d = fullfile(dirn0fr, mice{im}, nnow);
            fn = fullfile(dirn0fr, mice{im}, nnow, namv);

            savefig(gcf, fn)
        end                    
    end    
    
end


%% FR per trial type(ipsi) (contra)








%%
no
%%%%%%%%%%%%%%%% below didn't really get anywhere.... 
% Bin firing rates ... and look at corr vs FR for each bin of FR

%%
nBins = 30;

% set the bins
ally = [frAv_Ei(:); frAv_Ec(:); frAv_Ii(:); frAv_Ic(:)];
r1 = min(ally); %round(min(ally),1); 
r2 = max(ally); %round(max(ally)+.05,1);
bins = r1 : (r2-r1)/nBins : r2;
bins(end) = r2 + .001;
% set x for plotting hists as the center of bins
x = mode(bins(2)-bins(1))/2 + bins; x = x(1:end-1);

% get the counts in each bin
[n_Ei0, e, b_Ei] = histcounts(frAv_Ei, bins);
[n_Ec0, e, b_Ec] = histcounts(frAv_Ec, bins);
[n_Ii0, e, b_Ii] = histcounts(frAv_Ii, bins);
[n_Ic0, e, b_Ic] = histcounts(frAv_Ic, bins);

%
n_Ei = n_Ei0/sum(n_Ei0); 
n_Ec = n_Ec0/sum(n_Ec0); 
n_Ii = n_Ii0/sum(n_Ii0); 
n_Ic = n_Ic0/sum(n_Ic0);
% ye = smooth(ye);

figure(); hold on
plot(x, n_Ei, 'b')
plot(x, n_Ec, 'g')
plot(x, n_Ii, 'r')
plot(x, n_Ic, 'm')
legend(sprintf('Ei (n=%d)', sum(n_Ei0)), sprintf('Ec (n=%d)', sum(n_Ec0)), sprintf('Ii (n=%d)', sum(n_Ii0)), sprintf('Ic (n=%d)', sum(n_Ic0)))
xlabel('FR (trial-averaged)')
ylabel('Fraction neurons')

if saveFigs
    namv = sprintf('FR_EI_ipsiContra_aveTr_allMiceDaysPooled_curr_%s%s_%s.fig', alFR, o2a, nowStr);
    savefig(gcf, fullfile(dirn0fr, namv))
end


%{
% compare means of Ei_Ii, Ec_Ic with 
% Ei_Ic, Ec_Ii
[mean([frAv_Ei; frAv_Ii]), mean([frAv_Ec; frAv_Ic])];
[mean([frAv_Ei; frAv_Ic]), mean([frAv_Ec; frAv_Ii])];

% is FR of Ei and Ii different from the FR of Ei and Ic
[n_EiIi, e, b_EiIi] = histcounts([frAv_Ei; frAv_Ii], bins);
[n_EiIc, e, b_EiIc] = histcounts([frAv_Ei; frAv_Ic], bins);
n_EiIi = n_EiIi/sum(n_EiIi);
n_EiIc = n_EiIc/sum(n_EiIc);
figure(); hold on
plot(x, n_EiIi)
plot(x, n_EiIc)
[h,p] = ttest2([frAv_Ei; frAv_Ii], [frAv_Ei; frAv_Ic])

% is EcIc FR different from EcIi FR
[n_EcIc, e, b_EcIc] = histcounts([frAv_Ec; frAv_Ic], bins);
[n_EcIi, e, b_EcIi] = histcounts([frAv_Ec; frAv_Ii], bins);
n_EcIc = n_EcIc/sum(n_EcIc);
n_EcIi = n_EcIi/sum(n_EcIi);
figure(); hold on
plot(x, n_EcIc)
plot(x, n_EcIi)
[h,p] = ttest2([frAv_Ec; frAv_Ic], [frAv_Ec; frAv_Ii])
%}

%%%%%%%%%%%%%%%%%%%%
%%
corrAveBin_Ei_Ei = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrAveBin_Ec_Ec = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrAveBin_Ei_Ec = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);

corrAveBin_Ii_Ii = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrAveBin_Ic_Ic = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrAveBin_Ii_Ic = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);

corrAveBin_Ei_Ii = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrAveBin_Ec_Ic = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrAveBin_Ei_Ic = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrAveBin_Ec_Ii = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);

% median
corrMedBin_Ei_Ei = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrMedBin_Ec_Ec = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrMedBin_Ei_Ec = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);

corrMedBin_Ii_Ii = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrMedBin_Ic_Ic = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrMedBin_Ii_Ic = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);

corrMedBin_Ei_Ii = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrMedBin_Ec_Ic = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrMedBin_Ei_Ic = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrMedBin_Ec_Ii = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
    
for im = 1 : length(mice)
    for iday = 1:numDaysAll(im)
        if mnTrNum_allMice{im}(iday) >= thMinTrs

            frAv_Ei0 = mean(fr_exc_ipsi_timeM1{im}{iday}, 2);
            frAv_Ec0 = mean(fr_exc_contra_timeM1{im}{iday}, 2);
            frAv_Ii0 = mean(fr_inh_ipsi_timeM1{im}{iday}, 2); % nNeurons
            frAv_Ic0 = mean(fr_inh_contra_timeM1{im}{iday}, 2);
            
            % get the counts in each bin
			[n_Ei, e, b_Ei] = histcounts(frAv_Ei0, bins);
			[n_Ec, e, b_Ec] = histcounts(frAv_Ec0, bins);
			[n_Ii, e, b_Ii] = histcounts(frAv_Ii0, bins);
			[n_Ic, e, b_Ic] = histcounts(frAv_Ic0, bins);

            for ibin1 = 1:length(bins)-1
                for ibin2 = 1:length(bins)-1

                    % b==ibin; % index of ns whose FR is in bin ibin
                    % Take a subset of corrs ... in which neurons FR is in bins ibin1 and ibin2
                    a_Ei_Ei = r_Ei_Ei{im}{iday}(b_Ei==ibin1 , b_Ei==ibin2);
                    a_Ec_Ec = r_Ec_Ec{im}{iday}(b_Ec==ibin1 , b_Ec==ibin2);
                    a_Ei_Ec = r_Ei_Ec{im}{iday}(b_Ei==ibin1 , b_Ec==ibin2);

                    a_Ii_Ii = r_Ii_Ii{im}{iday}(b_Ii==ibin1 , b_Ii==ibin2);
                    a_Ic_Ic = r_Ic_Ic{im}{iday}(b_Ic==ibin1 , b_Ic==ibin2);
                    a_Ii_Ic = r_Ii_Ic{im}{iday}(b_Ii==ibin1 , b_Ic==ibin2);    

                    a_Ei_Ii = r_Ei_Ii{im}{iday}(b_Ei==ibin1 , b_Ii==ibin2);
                    a_Ec_Ic = r_Ec_Ic{im}{iday}(b_Ec==ibin1 , b_Ic==ibin2);
                    a_Ei_Ic = r_Ei_Ic{im}{iday}(b_Ei==ibin1 , b_Ic==ibin2);
                    a_Ec_Ii = r_Ec_Ii{im}{iday}(b_Ec==ibin1 , b_Ii==ibin2);

                    
                    if ibin1==ibin2 % it's symmetric so set upper triu to nan
                        a_Ei_Ei = triu_nan(a_Ei_Ei);
                        a_Ec_Ec = triu_nan(a_Ec_Ec);

                        a_Ii_Ii = triu_nan(a_Ii_Ii);
                        a_Ic_Ic = triu_nan(a_Ic_Ic);
                    end

                    %{ 
                    % not sure if you want to apply this condition!
                    if sum(~isnan(a_Ei_Ei(:))) < thNpairs % we need at least 10 pairs of neurons for each combination of FR bins to rely on their average corr value.
                        a_Ei_Ei = nan;
                        a_Ec_Ec = nan;
                        a_Ei_Ec = nan;

                        a_Ii_Ii = nan;
                        a_Ic_Ic = nan;
                        a_Ii_Ic = nan;

                        a_Ei_Ii = nan;
                        a_Ec_Ic = nan;
                        a_Ei_Ic = nan;                        
                        a_Ec_Ii = nan;                        
                    end
                    %}
                    % get mean corr of these neurons (each neuron has its FR in a specific bin)
                    corrAveBin_Ei_Ei{im}{iday}(ibin1, ibin2) = nanmean(a_Ei_Ei(:));
                    corrAveBin_Ec_Ec{im}{iday}(ibin1, ibin2) = nanmean(a_Ec_Ec(:));
                    corrAveBin_Ei_Ec{im}{iday}(ibin1, ibin2) = nanmean(a_Ei_Ec(:));

                    corrAveBin_Ii_Ii{im}{iday}(ibin1, ibin2) = nanmean(a_Ii_Ii(:));
                    corrAveBin_Ic_Ic{im}{iday}(ibin1, ibin2) = nanmean(a_Ic_Ic(:));
                    corrAveBin_Ii_Ic{im}{iday}(ibin1, ibin2) = nanmean(a_Ii_Ic(:));

                    corrAveBin_Ei_Ii{im}{iday}(ibin1, ibin2) = nanmean(a_Ei_Ii(:));
                    corrAveBin_Ec_Ic{im}{iday}(ibin1, ibin2) = nanmean(a_Ec_Ic(:));
                    corrAveBin_Ei_Ic{im}{iday}(ibin1, ibin2) = nanmean(a_Ei_Ic(:));
                    corrAveBin_Ec_Ii{im}{iday}(ibin1, ibin2) = nanmean(a_Ec_Ii(:));


                    % median
                    corrMedBin_Ei_Ei{im}{iday}(ibin1, ibin2) = nanmedian(a_Ei_Ei(:));
                    corrMedBin_Ec_Ec{im}{iday}(ibin1, ibin2) = nanmedian(a_Ec_Ec(:));
                    corrMedBin_Ei_Ec{im}{iday}(ibin1, ibin2) = nanmedian(a_Ei_Ec(:));

                    corrMedBin_Ii_Ii{im}{iday}(ibin1, ibin2) = nanmedian(a_Ii_Ii(:));
                    corrMedBin_Ic_Ic{im}{iday}(ibin1, ibin2) = nanmedian(a_Ic_Ic(:));
                    corrMedBin_Ii_Ic{im}{iday}(ibin1, ibin2) = nanmedian(a_Ii_Ic(:));

                    corrMedBin_Ei_Ii{im}{iday}(ibin1, ibin2) = nanmean(a_Ei_Ii(:));
                    corrMedBin_Ec_Ic{im}{iday}(ibin1, ibin2) = nanmean(a_Ec_Ic(:));
                    corrMedBin_Ei_Ic{im}{iday}(ibin1, ibin2) = nanmean(a_Ei_Ic(:));
                    corrMedBin_Ec_Ii{im}{iday}(ibin1, ibin2) = nanmean(a_Ec_Ii(:));

                end
            end
        end
    end
end


% plot corrs of individual populations
%{
figure; 
subplot(211); plot(x, corrAveBin_Ei_Ei, 'o-')
subplot(212);  plot(x, corrMedBin_Ei_Ei, 'o-')

figure; 
subplot(211); plot(x, corrAveBin_Ec_Ec, 'o-')
subplot(212);  plot(x, corrMedBin_Ec_Ec, 'o-')

figure; 
subplot(211); plot(x, corrAveBin_Ei_Ec, 'o-')
subplot(212);  plot(x, corrMedBin_Ei_Ec, 'o-')

% inh
figure; 
subplot(211); plot(x, corrAveBin_Ii_Ii, 'o-')
subplot(212);  plot(x, corrMedBin_Ii_Ii, 'o-')

figure; 
subplot(211); plot(x, corrAveBin_Ic_Ic, 'o-')
subplot(212);  plot(x, corrMedBin_Ic_Ic, 'o-')

figure; 
subplot(211); plot(x, corrAveBin_Ii_Ic, 'o-')
subplot(212);  plot(x, corrMedBin_Ii_Ic, 'o-')

%
figure; 
subplot(211); plot(x, corrAveBin_Ei_Ii, 'o-')
subplot(212);  plot(x, corrMedBin_Ei_Ii, 'o-')

figure; 
subplot(211); plot(x, corrAveBin_Ec_Ic, 'o-')
subplot(212);  plot(x, corrMedBin_Ec_Ic, 'o-')

figure; 
subplot(211); plot(x, corrAveBin_Ei_Ic, 'o-')
subplot(212);  plot(x, corrMedBin_Ei_Ic, 'o-')

figure; 
subplot(211); plot(x, corrAveBin_Ec_Ii, 'o-')
subplot(212); plot(x, corrMedBin_Ec_Ii, 'o-')
%}

% pool all different neurons
%{
a = [corrAveBin_Ei_Ei; corrAveBin_Ec_Ec; corrAveBin_Ei_Ec; corrAveBin_Ii_Ii; corrAveBin_Ic_Ic; corrAveBin_Ii_Ic; corrAveBin_Ei_Ii; corrAveBin_Ec_Ic; corrAveBin_Ei_Ic; corrAveBin_Ec_Ii];
figure; 
subplot(211); plot(x, a, 'o')
xlabel('FR'); ylabel('corr coef')
subplot(212); plot(x, nanmean(a), 'bo')
%}

%%%%%%%%%%% For one day of one mouse:
%{
a = cat(3, corrAveBin_Ei_Ei, corrAveBin_Ec_Ec, corrAveBin_Ei_Ec, corrAveBin_Ii_Ii, corrAveBin_Ic_Ic, corrAveBin_Ii_Ic, corrAveBin_Ei_Ii, corrAveBin_Ec_Ic, corrAveBin_Ei_Ic, corrAveBin_Ec_Ii);
aa = nanmean(a,3);
figure; imagesc(aa)

figure; plot(x, nanmean(aa(1:10,:),1), 'bo')
figure; plot(x, nanmean(aa(end-20:end,:),1), 'bo')

% both neurons in the same bin of FR
figure; plot(x, diag(aa), 'bo'), xlabel('FR'); ylabel('corr coef')

% low FR .. both neurons in the low FR bin (not necessarily the same bin though)
a0 = aa(1:10,1:10);
% how FR .. both neurons in the low FR bin (not necessarily the same bin though)
a00 = aa(end-20:end , end-20:end);
[nanmean(a0(:)), nanmean(a00(:))]
%}




%%%%% Pool all days of all mice
corrAveBin_Ei_Ei_allDaysPooledAve = nanmean(reshape(cell2mat(vertcat([corrAveBin_Ei_Ei{:}])), 50,50,[]), 3);
corrAveBin_Ec_Ec_allDaysPooledAve = nanmean(reshape(cell2mat(vertcat([corrAveBin_Ec_Ec{:}])), 50,50,[]), 3);
corrAveBin_Ei_Ec_allDaysPooledAve = nanmean(reshape(cell2mat(vertcat([corrAveBin_Ei_Ec{:}])), 50,50,[]), 3);

corrAveBin_Ii_Ii_allDaysPooledAve = nanmean(reshape(cell2mat(vertcat([corrAveBin_Ii_Ii{:}])), 50,50,[]), 3);
corrAveBin_Ic_Ic_allDaysPooledAve = nanmean(reshape(cell2mat(vertcat([corrAveBin_Ic_Ic{:}])), 50,50,[]), 3);
corrAveBin_Ii_Ic_allDaysPooledAve = nanmean(reshape(cell2mat(vertcat([corrAveBin_Ii_Ic{:}])), 50,50,[]), 3);

corrAveBin_Ei_Ii_allDaysPooledAve = nanmean(reshape(cell2mat(vertcat([corrAveBin_Ei_Ii{:}])), 50,50,[]), 3);
corrAveBin_Ec_Ic_allDaysPooledAve = nanmean(reshape(cell2mat(vertcat([corrAveBin_Ec_Ic{:}])), 50,50,[]), 3);
corrAveBin_Ei_Ic_allDaysPooledAve = nanmean(reshape(cell2mat(vertcat([corrAveBin_Ei_Ic{:}])), 50,50,[]), 3);
corrAveBin_Ec_Ii_allDaysPooledAve = nanmean(reshape(cell2mat(vertcat([corrAveBin_Ec_Ii{:}])), 50,50,[]), 3);

% pool all different population corrs
a = cat(3, corrAveBin_Ei_Ei_allDaysPooledAve, corrAveBin_Ec_Ec_allDaysPooledAve, corrAveBin_Ei_Ec_allDaysPooledAve, corrAveBin_Ii_Ii_allDaysPooledAve, corrAveBin_Ic_Ic_allDaysPooledAve, corrAveBin_Ii_Ic_allDaysPooledAve, corrAveBin_Ei_Ii_allDaysPooledAve, corrAveBin_Ec_Ic_allDaysPooledAve, corrAveBin_Ei_Ic_allDaysPooledAve, corrAveBin_Ec_Ii_allDaysPooledAve);
% a = cat(3, corrAveBin_Ei_Ei_allDaysPooledAve, corrAveBin_Ec_Ec_allDaysPooledAve, corrAveBin_Ii_Ii_allDaysPooledAve, corrAveBin_Ic_Ic_allDaysPooledAve, corrAveBin_Ei_Ii_allDaysPooledAve, corrAveBin_Ec_Ic_allDaysPooledAve);
% a = cat(3, corrAveBin_Ei_Ec_allDaysPooledAve, corrAveBin_Ii_Ic_allDaysPooledAve, corrAveBin_Ei_Ic_allDaysPooledAve, corrAveBin_Ec_Ii_allDaysPooledAve);
% a = cat(3, corrAveBin_Ei_Ei_allDaysPooledAve, corrAveBin_Ec_Ec_allDaysPooledAve, corrAveBin_Ii_Ii_allDaysPooledAve, corrAveBin_Ic_Ic_allDaysPooledAve);
% a = cat(3, corrAveBin_Ei_Ic_allDaysPooledAve, corrAveBin_Ec_Ii_allDaysPooledAve);
% a = corrAveBin_Ii_Ic_allDaysPooledAve;

figure('name', 'avePooledDays, E-I'); 
a = cat(3, corrAveBin_Ei_Ii_allDaysPooledAve, corrAveBin_Ec_Ic_allDaysPooledAve);
aa = nanmean(a,3);
subplot(224); imagesc(aa); % set(gca, 'xticklabel'
subplot(221); hold on; plot(x, nanmean(aa(1:10,:),1), 'o')
subplot(223); hold on; plot(x, nanmean(aa(end-20:end,:),1), 'o')
% both neurons in the same bin of FR
subplot(222); hold on; plot(x, diag(aa), 'o'), xlabel('FR'); ylabel('corr coef')

a = cat(3, corrAveBin_Ei_Ic_allDaysPooledAve, corrAveBin_Ec_Ii_allDaysPooledAve);
aa = nanmean(a,3);
subplot(224); imagesc(aa); % set(gca, 'xticklabel'
subplot(221); hold on; plot(x, nanmean(aa(1:10,:),1), 'o')
subplot(223); hold on; plot(x, nanmean(aa(end-20:end,:),1), 'o')
% both neurons in the same bin of FR
subplot(222); hold on; plot(x, diag(aa), 'o'), xlabel('FR'); ylabel('corr coef')


figure('name', 'avePooledDays, E-E'); 
a = cat(3, corrAveBin_Ei_Ei_allDaysPooledAve, corrAveBin_Ec_Ec_allDaysPooledAve);
aa = nanmean(a,3);
subplot(224); imagesc(aa); % set(gca, 'xticklabel'
subplot(221); hold on; plot(x, nanmean(aa(1:10,:),1), 'o')
subplot(223); hold on; plot(x, nanmean(aa(end-20:end,:),1), 'o')
% both neurons in the same bin of FR
subplot(222); hold on; plot(x, diag(aa), 'o'), xlabel('FR'); ylabel('corr coef')

a = corrAveBin_Ei_Ec_allDaysPooledAve;
aa = nanmean(a,3);
subplot(224); imagesc(aa); % set(gca, 'xticklabel'
subplot(221); hold on; plot(x, nanmean(aa(1:10,:),1), 'o')
subplot(223); hold on; plot(x, nanmean(aa(end-20:end,:),1), 'o')
% both neurons in the same bin of FR
subplot(222); hold on; plot(x, diag(aa), 'o'), xlabel('FR'); ylabel('corr coef')


figure('name', 'avePooledDays, I-I'); 
a = cat(3, corrAveBin_Ii_Ii_allDaysPooledAve, corrAveBin_Ic_Ic_allDaysPooledAve);
aa = nanmean(a,3);
subplot(224); imagesc(aa); % set(gca, 'xticklabel'
subplot(221); hold on; plot(x, nanmean(aa(1:10,:),1), 'o')
subplot(223); hold on; plot(x, nanmean(aa(end-20:end,:),1), 'o')
% both neurons in the same bin of FR
subplot(222); hold on; plot(x, diag(aa), 'o'), xlabel('FR'); ylabel('corr coef')

a = corrAveBin_Ii_Ic_allDaysPooledAve;
aa = nanmean(a,3);
subplot(224); imagesc(aa); % set(gca, 'xticklabel'
subplot(221); hold on; plot(x, nanmean(aa(1:10,:),1), 'o')
subplot(223); hold on; plot(x, nanmean(aa(end-20:end,:),1), 'o')
% both neurons in the same bin of FR
subplot(222); hold on; plot(x, diag(aa), 'o'), xlabel('FR'); ylabel('corr coef')



% low FR .. both neurons in the low FR bin (not necessarily the same bin though)
a0 = aa(1:10,1:10);
% how FR .. both neurons in the low FR bin (not necessarily the same bin though)
a00 = aa(end-20:end , end-20:end);
[nanmean(a0(:)), nanmean(a00(:))]




% average across days for each mouse
%%%%%%%%%%%%%%%%%%%%
corrAveBin_Ei_Ei_avDays = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrAveBin_Ec_Ec_avDays = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrAveBin_Ei_Ec_avDays = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);

corrAveBin_Ii_Ii_avDays = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrAveBin_Ic_Ic_avDays = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrAveBin_Ii_Ic_avDays = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);

corrAveBin_Ei_Ii_avDays = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrAveBin_Ec_Ic_avDays = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrAveBin_Ei_Ic_avDays = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);
corrAveBin_Ec_Ii_avDays = cell(1, length(mice)); % nan(length(bins)-1 , length(bins)-1);

for im = 1 : length(mice)
    % concat all days, then take an average across days
    corrAveBin_Ei_Ei_avDays{im} = nanmean(reshape(cell2mat(corrAveBin_Ei_Ei{im}), nBins,nBins,[]),3);
    corrAveBin_Ec_Ec_avDays{im} = nanmean(reshape(cell2mat(corrAveBin_Ec_Ec{im}), nBins,nBins,[]),3);
    corrAveBin_Ei_Ec_avDays{im} = nanmean(reshape(cell2mat(corrAveBin_Ei_Ec{im}), nBins,nBins,[]),3);
    
    corrAveBin_Ii_Ii_avDays{im} = nanmean(reshape(cell2mat(corrAveBin_Ii_Ii{im}), nBins,nBins,[]),3);
    corrAveBin_Ic_Ic_avDays{im} = nanmean(reshape(cell2mat(corrAveBin_Ic_Ic{im}), nBins,nBins,[]),3);
    corrAveBin_Ii_Ic_avDays{im} = nanmean(reshape(cell2mat(corrAveBin_Ii_Ic{im}), nBins,nBins,[]),3);

    corrAveBin_Ei_Ii_avDays{im} = nanmean(reshape(cell2mat(corrAveBin_Ei_Ii{im}), nBins,nBins,[]),3);
    corrAveBin_Ec_Ic_avDays{im} = nanmean(reshape(cell2mat(corrAveBin_Ec_Ic{im}), nBins,nBins,[]),3);
    corrAveBin_Ei_Ic_avDays{im} = nanmean(reshape(cell2mat(corrAveBin_Ei_Ic{im}), nBins,nBins,[]),3);    
    corrAveBin_Ec_Ii_avDays{im} = nanmean(reshape(cell2mat(corrAveBin_Ec_Ii{im}), nBins,nBins,[]),3);    
end

% average across mice
corrAveBin_Ei_Ei_avMice = nanmean(reshape(cell2mat(corrAveBin_Ei_Ei_avDays), nBins,nBins,[]),3);
corrAveBin_Ec_Ec_avMice = nanmean(reshape(cell2mat(corrAveBin_Ec_Ec_avDays), nBins,nBins,[]),3);
corrAveBin_Ei_Ec_avMice = nanmean(reshape(cell2mat(corrAveBin_Ei_Ec_avDays), nBins,nBins,[]),3);

corrAveBin_Ii_Ii_avMice = nanmean(reshape(cell2mat(corrAveBin_Ii_Ii_avDays), nBins,nBins,[]),3);
corrAveBin_Ic_Ic_avMice = nanmean(reshape(cell2mat(corrAveBin_Ic_Ic_avDays), nBins,nBins,[]),3);
corrAveBin_Ii_Ic_avMice = nanmean(reshape(cell2mat(corrAveBin_Ii_Ic_avDays), nBins,nBins,[]),3);

corrAveBin_Ei_Ii_avMice = nanmean(reshape(cell2mat(corrAveBin_Ei_Ii_avDays), nBins,nBins,[]),3);
corrAveBin_Ec_Ic_avMice = nanmean(reshape(cell2mat(corrAveBin_Ec_Ic_avDays), nBins,nBins,[]),3);
corrAveBin_Ei_Ic_avMice = nanmean(reshape(cell2mat(corrAveBin_Ei_Ic_avDays), nBins,nBins,[]),3);
corrAveBin_Ec_Ii_avMice = nanmean(reshape(cell2mat(corrAveBin_Ec_Ii_avDays), nBins,nBins,[]),3);

% pool all different population corrs
a = cat(3, corrAveBin_Ei_Ei_avMice, corrAveBin_Ec_Ec_avMice, corrAveBin_Ei_Ec_avMice, corrAveBin_Ii_Ii_avMice, corrAveBin_Ic_Ic_avMice, corrAveBin_Ii_Ic_avMice, corrAveBin_Ei_Ii_avMice, corrAveBin_Ec_Ic_avMice, corrAveBin_Ei_Ic_avMice, corrAveBin_Ec_Ii_avMice);
aa = nanmean(a,3);
figure('name', 'aveMice'); 
subplot(224); imagesc(aa)

subplot(221); plot(x, nanmean(aa(1:10,:),1), 'bo')
subplot(223); plot(x, nanmean(aa(end-20:end,:),1), 'bo')

% both neurons in the same bin of FR
subplot(222); plot(x, diag(aa), 'bo'), xlabel('FR'); ylabel('corr coef')

% low FR .. both neurons in the low FR bin (not necessarily the same bin though)
a0 = aa(1:10,1:10);
% how FR .. both neurons in the low FR bin (not necessarily the same bin though)
a00 = aa(end-20:end , end-20:end);
[nanmean(a0(:)), nanmean(a00(:))]



%%%%%%%%%%%% each mouse
for im = 1:length(mice)
    % pool all different population corrs
    a = cat(3, corrAveBin_Ei_Ei_avDays{im}, corrAveBin_Ec_Ec_avDays{im}, corrAveBin_Ei_Ec_avDays{im}, corrAveBin_Ii_Ii_avDays{im}, corrAveBin_Ic_Ic_avDays{im}, corrAveBin_Ii_Ic_avDays{im}, corrAveBin_Ei_Ii_avDays{im}, corrAveBin_Ec_Ic_avDays{im}, corrAveBin_Ei_Ic_avDays{im}, corrAveBin_Ec_Ii_avDays{im});
    aa = nanmean(a,3);
    figure('name',mice{im}); 
    subplot(224); imagesc(aa)

%     figure('name',mice{im}); 
    subplot(221); plot(x, nanmean(aa(1:10,:),1), 'bo')
    subplot(223);  plot(x, nanmean(aa(end-20:end,:),1), 'bo')

    % both neurons in the same bin of FR
    subplot(222);  plot(x, diag(aa), 'bo'), xlabel('FR'); ylabel('corr coef')

    % low FR .. both neurons in the low FR bin (not necessarily the same bin though)
    a0 = aa(1:10,1:10);
    % how FR .. both neurons in the low FR bin (not necessarily the same bin though)
    a00 = aa(end-20:end , end-20:end);
    [nanmean(a0(:)), nanmean(a00(:))]
end


%%%%%%%%%%%% each day
for im = 1:length(mice)
    % pool all different population corrs
    a = cat(3, corrAveBin_Ei_Ei{im}{iday}, corrAveBin_Ec_Ec{im}{iday}, corrAveBin_Ei_Ec{im}{iday}, corrAveBin_Ii_Ii{im}{iday}, corrAveBin_Ic_Ic{im}{iday}, corrAveBin_Ii_Ic{im}{iday}, corrAveBin_Ei_Ii{im}{iday}, corrAveBin_Ec_Ic{im}{iday}, corrAveBin_Ei_Ic{im}{iday}, corrAveBin_Ec_Ii{im}{iday});
    aa = nanmean(a,3);
    figure('name',mice{im}); 
    subplot(224); imagesc(aa)

%     figure('name',mice{im}); 
    subplot(221); plot(x, nanmean(aa(1:10,:),1), 'bo')
    subplot(223);  plot(x, nanmean(aa(end-20:end,:),1), 'bo')

    % both neurons in the same bin of FR
    subplot(222);  plot(x, diag(aa), 'bo'), xlabel('FR'); ylabel('corr coef')

    % low FR .. both neurons in the low FR bin (not necessarily the same bin though)
    a0 = aa(1:10,1:10);
    % how FR .. both neurons in the low FR bin (not necessarily the same bin though)
    a00 = aa(end-20:end , end-20:end);
    [nanmean(a0(:)), nanmean(a00(:))]
end


%{
figure;
for iN = 1:length(y1) % corr of n1 with n2
    n1corrs = r_Ei_Ei{im}{iday}(:,iN);

    % what's their average FR
%     y1([1,2])

    % or see if N1's corr with other ns depends on their fr?
    subplot(211); plot(y1, n1corrs, '.'); hold on; plot(y1(iN), 0, 'r*')
    subplot(212); plot(y1/y1(iN), n1corrs, '.'); hold on; plot(1, 0, 'r*')
    pause
    subplot(211); cla
    subplot(212); cla
end
%}


