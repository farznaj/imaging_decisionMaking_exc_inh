%%% This script is called inside: choicePref_ROC_exc_inh_plotsEachMouse


startDayFni19 = 16; %1 % this will affect the corrCoeff ave se across days plot! % what day is the start day for mouse fni19: amazing changes during learning. I believe he was more posterior. In early days he represents stimulus category, in later days he represents choice. So we go with his choice days for this summary.

dirn0 = '/home/farznaj/Dropbox/ChurchlandLab/Projects/inhExcDecisionMaking/ROC';
savefigs = eachMouse_do_savefigs(2);

nowStr = nowStr_allMice{imfni18}; % use nowStr of mouse fni18 (in case its day 4 was removed, you want that to be indicated in the figure name of summary of all mice).
dirn00 = fullfile(dirn0, 'allMice');


%% Compare AUC_corr vs AUC_incorr for individual neurons

% For this plot you need to download ROC vars of correct trials as well as ROC vars of incorrect trials.
% This allows us to see if neurons represent choice or stim... if for both
% corr and incorr trials, AUC is below 0.5 (contra-choice specific) or above
% 0.5 (ipsi-choice specific), then the neuron represnts choice; But if for corr AUC is below 0.5 and for incorr it is above 0.5, then the neuron represents stimulus. 

%%%%%% Set vars %%%%%%

nPreMin_corr = find(time_al_corr < 0, 1, 'last');
nPreMin_incorr = find(time_al_incorr < 0, 1, 'last');

cc_corr_incorr_exc_eachDay = cell(1, length(mice));
cc_corr_incorr_inh_eachDay = cell(1, length(mice));
cc_corr_incorr_exc = nan(1, length(mice));
cc_corr_incorr_inh = nan(1, length(mice));
choiceCoded_contraSpecific = nan(1, length(mice));
choiceCoded_ipsiSpecific = nan(1, length(mice));
stimCoded_HRSpecific = nan(1, length(mice));
stimCoded_LRSpecific = nan(1, length(mice));
choiceCoded = nan(1, length(mice));
stimCoded = nan(1, length(mice));
exc_corr_incorr_avNs_allMice = cell(1, length(mice));
exc_corr_incorr_sdNs_allMice = cell(1, length(mice));
inh_corr_incorr_avNs_allMice = cell(1, length(mice));
inh_corr_incorr_sdNs_allMice = cell(1, length(mice));
goodD_allM = cell(1, length(mice));    

for im = 1:length(mice)

    mnTrNum_corr = min(ipsi_contra_corr{im},[],2); % min number of trials of the 2 class
    mnTrNum_incorr = min(ipsi_contra_incorr{im},[],2);
    goodD = (mnTrNum_corr > thMinTrs) & (mnTrNum_incorr > thMinTrs);
    goodD_allM{im} = goodD;
    ndays = length(choicePref_exc_al_allMice_corr{im});        
    days2an = 1:ndays; % ndays;

    cc_corr_incorr_exc_eachDay{im} = nan(1,ndays);
    cc_corr_incorr_inh_eachDay{im} = nan(1,ndays);
    exc_corr_incorr_avNs = nan(2, ndays);
    exc_corr_incorr_sdNs = nan(2, ndays);
    inh_corr_incorr_avNs = nan(2, ndays);
    inh_corr_incorr_sdNs = nan(2, ndays);                

    for iday = days2an
        if goodD(iday)
            %%%%%% exc %%%%%% 
            x = choicePref_exc_al_allMice_incorr{im}{iday}(nPreMin_incorr,:); % AUC of exc neurons at time bin -1 computed on incorrect trials
            y = choicePref_exc_al_allMice_corr{im}{iday}(nPreMin_corr,:);
            % corrcoef between corr and incorr AUCs, for each day                
            cc_corr_incorr_exc_eachDay{im}(iday) = corr2(x,y);                
            % mean and sd across neurons
            exc_corr_incorr_avNs(:,iday) = [mean(y); mean(x)]; 
            exc_corr_incorr_sdNs(:,iday) = [std(y); std(x)];

            %%%%%% inh %%%%%%
            x = choicePref_inh_al_allMice_incorr{im}{iday}(nPreMin_incorr,:);
            y = choicePref_inh_al_allMice_corr{im}{iday}(nPreMin_corr,:);
            % corrcoef between corr and incorr AUCs, for each day                
            cc_corr_incorr_inh_eachDay{im}(iday) = corr2(x,y);                
            % mean and sd across neurons
            inh_corr_incorr_avNs(:,iday) = [mean(y); mean(x)]; 
            inh_corr_incorr_sdNs(:,iday) = [std(y); std(x)];
        end
    end
    exc_corr_incorr_avNs_allMice{im} = exc_corr_incorr_avNs;
    exc_corr_incorr_sdNs_allMice{im} = exc_corr_incorr_sdNs;
    inh_corr_incorr_avNs_allMice{im} = inh_corr_incorr_avNs;
    inh_corr_incorr_sdNs_allMice{im} = inh_corr_incorr_sdNs;


    %%%%%%%%%%%%%%%% compute corr coeff on pooled days data
    days2an = goodD;
    % pool all neurons of all days 
    exc_incorr_daysPooled_eachMouse = cell2mat(choicePref_exc_al_allMice_incorr{im}(days2an)); 
    exc_incorr_daysPooled_eachMouse = exc_incorr_daysPooled_eachMouse(nPreMin_incorr,:); % take AUC of all neurons of all days at time bin -1
    exc_corr_daysPooled_eachMouse = cell2mat(choicePref_exc_al_allMice_corr{im}(days2an)); 
    exc_corr_daysPooled_eachMouse = exc_corr_daysPooled_eachMouse(nPreMin_corr,:);

    inh_incorr_daysPooled_eachMouse = cell2mat(choicePref_inh_al_allMice_incorr{im}(days2an)); 
    inh_incorr_daysPooled_eachMouse = inh_incorr_daysPooled_eachMouse(nPreMin_incorr,:);
    inh_corr_daysPooled_eachMouse = cell2mat(choicePref_inh_al_allMice_corr{im}(days2an)); 
    inh_corr_daysPooled_eachMouse = inh_corr_daysPooled_eachMouse(nPreMin_corr,:);

    % compute corrcoeff between corr and incorr AUCs (all neurons)
    cc_corr_incorr_exc(im) = corr2(exc_incorr_daysPooled_eachMouse, exc_corr_daysPooled_eachMouse);
    cc_corr_incorr_inh(im) = corr2(inh_incorr_daysPooled_eachMouse, inh_corr_daysPooled_eachMouse);


    %%%%%%%%%% compute number of choice vs stim representing neurons of each side
    % to do this properly, you need to use shuffles to decide which
    % neurons are significant, instead of using the arbitrary threshold below!
    thROC = .51; % .505; % .5 % threshold to compute number of choice vs stim representing neurons of each side
    choiceCoded_contraSpecific(im) = sum((exc_incorr_daysPooled_eachMouse < thROC) & (exc_corr_daysPooled_eachMouse < thROC));
    choiceCoded_ipsiSpecific(im) = sum((exc_incorr_daysPooled_eachMouse > thROC) & (exc_corr_daysPooled_eachMouse > thROC));
    stimCoded_HRSpecific(im) = sum((exc_incorr_daysPooled_eachMouse < thROC) & (exc_corr_daysPooled_eachMouse > thROC));
    stimCoded_LRSpecific(im) = sum((exc_incorr_daysPooled_eachMouse > thROC) & (exc_corr_daysPooled_eachMouse < thROC));
    choiceCoded(im) = choiceCoded_ipsiSpecific(im) + choiceCoded_contraSpecific(im);
    stimCoded(im) = stimCoded_HRSpecific(im) + stimCoded_LRSpecific(im);           
end

choiceCoded ./ stimCoded
choiceCoded_contraSpecific ./ choiceCoded_ipsiSpecific
stimCoded_HRSpecific ./ stimCoded_LRSpecific


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Plot %%%%%%%%%%%%%%%%%        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Example in Supp Fig 2C: im=1; iday=45;

%%%%%%%%%%%% Plots of each mouse: Scatter plot of AUC_corr vs AUC_incorr for individual neurons; use jet colormap for days (early to late days go from blue to red) %%%%%%%%%%%%

for im = 1:length(mice)  
    goodD = goodD_allM{im};
    ndays = length(choicePref_exc_al_allMice_corr{im});        
    days2an = 1:ndays; % ndays;
    cmp = jet(ndays);   %repmat([0,0,0],[ndays,1]); %

    fh = figure('name', mice{im}, 'position', [106   453   562   520]); %15   456   320   520]); 
    for iday = days2an
        if goodD(iday)
            %%%%%% exc %%%%%% 
            x = choicePref_exc_al_allMice_incorr{im}{iday}(nPreMin_incorr,:); % AUC of exc neurons at time bin -1 computed on incorrect trials
            y = choicePref_exc_al_allMice_corr{im}{iday}(nPreMin_corr,:);
%             [r,p] = corrcoef(x,y); disp([r(2), p(2)])
            % plot
            subplot(221); hold on;
            plot(x, y, '.', 'color', cmp(iday,:), 'markersize', 5)            

            %%%%%% inh %%%%%%
            x = choicePref_inh_al_allMice_incorr{im}{iday}(nPreMin_incorr,:);
            y = choicePref_inh_al_allMice_corr{im}{iday}(nPreMin_corr,:);            
%             [r,p] = corrcoef(x,y); disp([r(2), p(2)])
            % plot
            subplot(223); hold on;
            plot(x, y, '.', 'color', cmp(iday,:), 'markersize', 5)        
        end
    end

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
    cc_exc = corr(exc_corr_incorr_avNs_allMice{im}', 'rows', 'pairwise'); cc_exc = cc_exc(2);
    cc_inh = corr(inh_corr_incorr_avNs_allMice{im}', 'rows', 'pairwise'); cc_inh = cc_inh(2);       
%     figure('name', mice{im}, 'position', [15   456   320   520]); set(groot,'defaultAxesColorOrder',cmp)    

    %%% exc %%%        
    subplot(222); hold on;
    x = exc_corr_incorr_avNs_allMice{im}(2,:);
    y = exc_corr_incorr_avNs_allMice{im}(1,:);    
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
    x = inh_corr_incorr_avNs_allMice{im}(2,:);
    y = inh_corr_incorr_avNs_allMice{im}(1,:);    
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



%% 

cc_corr_incorr_exc_eachDay0 = cc_corr_incorr_exc_eachDay;
cc_corr_incorr_inh_eachDay0 = cc_corr_incorr_inh_eachDay;

cc_corr_incorr_exc_eachDay{4} = cc_corr_incorr_exc_eachDay0{4}(startDayFni19:end);
cc_corr_incorr_inh_eachDay{4} = cc_corr_incorr_inh_eachDay0{4}(startDayFni19:end);


%% %%%%%%%%%%%%%%%%%%% Plots of all mice: average and se of corrCoeff_AUC_corrIncorr acorr days %%%%%%%%%%%%%%%%%%%

numDaysGood_corrIncorr = cellfun(@sum, goodD_allM);

cc_corr_incorr_exc_eachDay_avDays = cellfun(@nanmean, cc_corr_incorr_exc_eachDay);
cc_corr_incorr_exc_eachDay_seDays = cellfun(@nanstd, cc_corr_incorr_exc_eachDay) ./ sqrt(numDaysGood_corrIncorr);

cc_corr_incorr_inh_eachDay_avDays = cellfun(@nanmean, cc_corr_incorr_inh_eachDay);
cc_corr_incorr_inh_eachDay_seDays = cellfun(@nanstd, cc_corr_incorr_inh_eachDay) ./ sqrt(numDaysGood_corrIncorr);

%     p_allM = nan(1,length(mice));
%     for im=1:length(mice)
%         [~,p_allM(im)] = ttest2(exc_fractSigTuned_eachDay{im} , inh_fractSigTuned_eachDay{im});
%     end

x = (1:length(mice))';
gp = .1; 

% all preferences (ipsi and contra)
fign = figure('position', [17   714   207   251]);

y = [cc_corr_incorr_exc_eachDay_avDays; cc_corr_incorr_inh_eachDay_avDays]';
ye = [cc_corr_incorr_exc_eachDay_seDays; cc_corr_incorr_inh_eachDay_seDays]';

b = errorbar([x,x+gp], y, ye, 'linestyle', 'none', 'marker', '.', 'markersize', 9);
b(1).Color = 'b'; %'b';
b(2).Color = 'r'; %rgb('lightblue');
set(gca,'xtick',x+gp/2)
set(gca,'xticklabel',x)
xlabel('Mice')
ylabel('Average corrCoeff') % significant
title('AUC: corr vs incorr')
set(gca, 'tickdir', 'out', 'box', 'off')
xlim([.5,4.5])
% mark mice with sig diff btwn exc and inh
%     hold on
%     yl = get(gca,'ylim');
%     plot(find(p_allM<=.05)+gp/2, yl(2)-range(diff(yl))/20, 'k*')

legend(b, 'exc', 'inh')

% save figure
if savefigs        
    fdn = fullfile(dirn00, strcat('AUC_corrIncorr_corrCoeff_aveSeDays_time-1_ROC_curr_chAl_excInh_', dm0, nowStr));
    savefig(fign, fdn)    
    print(fign, '-dpdf', fdn)
end 





