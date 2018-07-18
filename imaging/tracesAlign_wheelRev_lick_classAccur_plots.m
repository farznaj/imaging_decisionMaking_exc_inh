% Load vars (alrady saved in tracesAlign_wheelRev_lick), aligns them across
% days. Saves vars of all mice in
% /home/farznaj/Shares/Churchland_hpc_home/space_managed_data/fni_allMice.
%
% Also make plots of all mice and save them in '/home/farznaj/Dropbox/ChurchlandLab/Projects/inhExcDecisionMaking/Behavior/wheelRev_Lick'


%%
savefigs = 0;

outcome2ana = 'corr'; % 'all'; % 'corr'
mice = {'fni16','fni17','fni18','fni19'};
% thMinTrs = 10; % days with fewer than this number of trials wont be analyzed.

dira = '/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/fni_allMice';
if strcmp(outcome2ana, 'corr')
    ocn = 'corr_allMice_';
else
    ocn = 'allOutcomes_allMice_';
end
nowStr = datestr(now, 'yymmdd-HHMMSS');

%%% plots:
cod = get(groot,'defaultAxesColorOrder'); 
% ylabd = 'Travelled distance (mm)'; % ylab = 'Wheel revolution';
% ylabsp = 'Running speed (mm/sec)';

ylabd = 'Travelled distance (mm)'; % ylab = 'Wheel revolution';
ylabsp = 'Running speed (mm/sec)';

dirn0 = '/home/farznaj/Dropbox/ChurchlandLab/Projects/inhExcDecisionMaking/Behavior/wheelRev_Lick';


%% Load wheelRev and lick vars

fn = ['wheelRev_lick_', ocn, '*.mat'];
fnam = dir(fullfile(dira, fn));
fnam = fnam(end);
disp(fnam.name)

load(fullfile(dira, fnam.name))
%{
('time_aligned_lick_allMice', 'nPreMin_lick_allMice', 'lick_aligned_allMice', 'lick_aligned_allMice_lrResp', 'lick_aligned_allMice_hrResp', ...
    'time_aligned_wheelRev_allMice', 'nPreMin_wheelRev_allMice', 'wheelRev_aligned_allMice', 'wheelRev_aligned_allMice_lrResp', 'wheelRev_aligned_allMice_hrResp', ...
    'td_allMice', 'nPreMin_wheelRev_allMice_ds', 'speed_d_allMice', 'speed_d_allMice_lrResp', 'speed_d_allMice_hrResp', ...
    'downSampBin', 'numTrs_all', 'mnTrNum_allMice')
%}


%% Load CA vars (saved in python) 

av_test_data_inh_aligned_allMice = cell(1, length(mice));
av_test_data_exc_aligned_allMice = cell(1, length(mice));
av_test_data_allExc_aligned_allMice = cell(1, length(mice));
behCorr_all_allMice = cell(1, length(mice));
nPreMin_allMice = nan(1, length(mice));
days2an_heatmap_allMice = cell(1, length(mice));

for im = 1:length(mice)

    mouse = mice{im};    
    [~,~,dirn] = setImagingAnalysisNames(mouse, 'analysis', []);
    
    fn = dir(fullfile(dirn, 'svm_CA_decodeChoice_*'));
    load(fullfile(dirn, fn.name), 'days2an_heatmap', 'av_test_data_inh_aligned', 'nPreMin', 'av_test_data_exc_aligned', 'av_test_data_allExc_aligned', 'behCorr_all')
   
    av_test_data_inh_aligned_allMice{im} = av_test_data_inh_aligned;    % frs x days
    av_test_data_exc_aligned_allMice{im} = av_test_data_exc_aligned;
    av_test_data_allExc_aligned_allMice{im} = av_test_data_allExc_aligned;
    behCorr_all_allMice{im} = behCorr_all;
    nPreMin_allMice(im) = nPreMin;
    days2an_heatmap_allMice{im} = days2an_heatmap;    
    
end





%% Correlation of lick traces preceding hr vs lr choice for each day 

win = [-250 0]; % [win(1) wind(2)] ms relative to choice onset: window over which to compute corr between lick of lr and hr

lickCorr_lr_hr = cell(1,length(mice));
p_lickCorr_lr_hr = cell(1,length(mice));
for im = 1:length(mice)
    r = nPreMin_lick_allMice(im) + win(1) : nPreMin_lick_allMice(im) + win(2);
    aa = lick_aligned_allMice_lrResp{im}(days2an_heatmap_allMice{im}, r);
    bb = lick_aligned_allMice_hrResp{im}(days2an_heatmap_allMice{im}, r);    
    [cc, pp] = corr(aa', bb'); 
    lickCorr_lr_hr{im} = diag(cc);
    p_lickCorr_lr_hr{im} = diag(pp);
end


% perc_thb = [.10,.90];
perc_thb = [.20,.80];


%% Set low and high behavioral performance days, and comapre lick patterns for low vs high behavioral performance days, also set corr between lr and hr lick traces 

lickCorr_lo_vs_hiBehDays = nan(1, length(mice));
lick_aveLoHiBehDays = cell(1, length(mice));       
lickCorr_lr_hr_loBehDays = cell(1, length(mice));
lickCorr_lr_hr_hiBehDays = cell(1, length(mice));       

for im = 1:length(mice)
    
    % set low and high beh days
    a = behCorr_all_allMice{im}(days2an_heatmap_allMice{im});
    thb = quantile(a, perc_thb);    
    loBehCorrDays = (a <= thb(1));
    hiBehCorrDays = (a >= thb(2));
    disp([sum(loBehCorrDays), sum(hiBehCorrDays)])
    
    % set average of lick traces across low and high beh days
    aa = lick_aligned_allMice{im}(days2an_heatmap_allMice{im},:);    
    a = mean(aa(loBehCorrDays,:),1);
    ah = mean(aa(hiBehCorrDays,:),1);   
    tr = [a;ah];
    lick_aveLoHiBehDays{im} = tr;
    
    % compute corr between lick of low and high beh days
    c = corrcoef(tr');
    lickCorr_lo_vs_hiBehDays(im) = c(2);   
    

    %%%%%%%%%%% get corr between lr and hr lick traces (across window win). Do it for low beh days as well as high beh days
    aa = lick_aligned_allMice_lrResp{im}(days2an_heatmap_allMice{im},:);
    bb = lick_aligned_allMice_hrResp{im}(days2an_heatmap_allMice{im},:);
    r = nPreMin_lick_allMice(im) + win(1) : nPreMin_lick_allMice(im) + win(2);
    
    a = aa(loBehCorrDays,r);
    b = bb(loBehCorrDays,r);    
    cc = corr(a', b'); 
    lickCorr_lr_hr_loBehDays{im} = diag(cc); % for each of low beh days, compute corr of lr lick and hr lick traces.

    ah = aa(hiBehCorrDays,r);
    bh = bb(hiBehCorrDays,r);    
    cch = corr(ah', bh'); 
    lickCorr_lr_hr_hiBehDays{im} = diag(cch); % for each of high beh days, compute corr of lr lick and hr lick traces.

end



%% Set significancy for whether HR vs LR traces (lick or wheelRev) are different.

alph = .05;

%%%% wheelRev
sig_wheelRev_hr_lr_aligned_allMice = cell(1, length(mice));
for im = 1:length(mice)
%     figure; plot(p_wheelRev_hr_lr_aligned_allMice{im}')
%     figure; imagesc(p_wheelRev_hr_lr_aligned_allMice{im})

    sig_wheelRev_hr_lr_aligned_allMice{im} = nan(size(p_wheelRev_hr_lr_aligned_allMice{im}));
    for iday = 1:size(sig_wheelRev_hr_lr_aligned_allMice{im},1)
        sig_wheelRev_hr_lr_aligned_allMice{im}(iday, p_wheelRev_hr_lr_aligned_allMice{im}(iday,:)<=alph) = 1;    
        sig_wheelRev_hr_lr_aligned_allMice{im}(iday, p_wheelRev_hr_lr_aligned_allMice{im}(iday,:)>alph) = 0;    
    end
end

% % Remove days with few trials.
% for im = 1:length(mice)
%     sig_wheelRev_hr_lr_aligned_allMice{im}(mnTrNum_allMice{im} < thMinTrs,:) = [];
% end


%%%%% lick


sig_lick_hr_lr_aligned_allMice = cell(1, length(mice));
for im = 1:length(mice)
%     figure; plot(p_lick_hr_lr_aligned_allMice{im}')
%     figure; imagesc(p_lick_hr_lr_aligned_allMice{im})

    sig_lick_hr_lr_aligned_allMice{im} = nan(size(p_lick_hr_lr_aligned_allMice{im}));
    for iday = 1:size(sig_lick_hr_lr_aligned_allMice{im},1)
        sig_lick_hr_lr_aligned_allMice{im}(iday, p_lick_hr_lr_aligned_allMice{im}(iday,:)<=alph) = 1;    
        sig_lick_hr_lr_aligned_allMice{im}(iday, p_lick_hr_lr_aligned_allMice{im}(iday,:)>alph) = 0;    
    end
end

% % Remove days with few trials.
% for im = 1:length(mice)
%     sig_lick_hr_lr_aligned_allMice{im}(mnTrNum_allMice{im} < thMinTrs,:) = [];
% end



no

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plots of licks and running %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Plots of wheel revolution

for doDist = [0,1] % if 1, plot running distance; if 0, plot running speed
    
    for all_lr_hr = [0,1,2] % plot average of all trials, lr trials, hr trials
        
        if all_lr_hr==0                
            fan = 'All trials';
        elseif all_lr_hr==1
            fan = 'LR-choice trials';
        elseif all_lr_hr==2
            fan = 'HR-choice trials';
        end
        
        fa = figure('position', [43   346   779   552]);
        fh = figure('position', [822   263   354   635]);        
        set(fa, 'name', fan)
        set(fh, 'name', fan)
        
        for im = 1:length(mice)

            if doDist
                timenow = time_aligned_wheelRev_allMice{im};
                nprenow = nPreMin_wheelRev_allMice(im);
                ds = 1;
                ylab = ylabd;
                if all_lr_hr==0                
                    tracenow = wheelRev_aligned_allMice{im};                
                elseif all_lr_hr==1
                    tracenow = wheelRev_aligned_allMice_lrResp{im};                                
                elseif all_lr_hr==2
                    tracenow = wheelRev_aligned_allMice_hrResp{im};                                
                end
            else    
                timenow = td_allMice{im}(1:end-1);
                nprenow = nPreMin_wheelRev_allMice_ds(im);                
                ds = downSampBin;
                ylab = ylabsp;
                if all_lr_hr==0
                    tracenow = speed_d_allMice{im};                    
                elseif all_lr_hr==1                    
                    tracenow = speed_d_allMice_lrResp{im};                    
                elseif all_lr_hr==2
                    tracenow = speed_d_allMice_hrResp{im};                    
                end
                
            end


            %%%%%%%%%%%% line plot showing all days %%%%%%%%%%%% 

            figure(fa)
            co = jet(size(tracenow,1)); % parula
            set(groot,'defaultAxesColorOrder',co)

            %%% average of days
            subplot(2,4,im); hold on

            plot(timenow, nanmean(tracenow,1), 'k')
            vline(0, 'r-.') % nPreMin_wheelRev_allMice{im}
            xlim([timenow(1), timenow(end)])
            if im==1
                xlabel('Time rel. choice (ms)')
                ylabel(ylab)
            end

            %%% single days
            subplot(2,4,4+im); hold on

            plot(timenow, tracenow')
            vline(0, 'r-.') % nPreMin_wheelRev_allMice{im}
            xlim([timenow(1), timenow(end)])
            if im==1
                xlabel('Time rel. choice (ms)')
                ylabel(ylab)
            end



            %%%%%%%%%%%% heatmap showing all days %%%%%%%%%%%% 

            figure(fh)
            subplot(4,1,im); hold on

            mxClim = quantile(tracenow(:), .9); % use this instead of max if you want to see better the values
    %         mxClim = max(tracenow(:));

            imagesc(tracenow, [0, mxClim]);
            vline(nprenow, 'r-.') % 
            x = sort(unique([nprenow: -30/ds : -30/ds , nprenow : 30/ds : size(tracenow,2)]));
            x(x<0) = [];  % x = x(2:end);
            set(gca, 'xtick', x)
            set(gca, 'xticklabel', timenow(x))       
            xlabel('Time rel. choice (ms)')
            ylabel('Days')
            set(gca,'ydir','reverse')
            c = colorbar;
            if im==1, c.Label.String = ylab; end
            xlim([1, size(tracenow,2)])  % ([timenow(1), timenow(end)]) %
            ylim([1, size(tracenow,1)])

        end



        if savefigs        
            if doDist
                dd = 'distance_';
            else
                dd = 'speed_';
            end

            if all_lr_hr==0                
                dd = [dd, 'aveAllTrs_'];
            elseif all_lr_hr==1
                dd = [dd, 'aveLrTrs_'];
            elseif all_lr_hr==2
                dd = [dd, 'aveHrTrs_'];
            end

            fdn = fullfile(dirn0, strcat('wheelRev_',dd,'chAl_lineplot_timeCourse_allMice_', nowStr));
            savefig(fa, fdn)
            print(fa, '-dpdf', fdn)

            fdn = fullfile(dirn0, strcat('wheelRev_',dd,'chAl_heatmap_timeCourse_allMice_', nowStr));
            savefig(fh, fdn)
            print(fh, '-dpdf', fdn)
        end

    end
end



%% Plots of licks

for all_lr_hr = [0,1,2,3] % plot average of all trials, lr trials, hr trials, abs(hr-lr)

    if all_lr_hr==0                
        fan = 'All trials';
    elseif all_lr_hr==1
        fan = 'LR choice trials';
    elseif all_lr_hr==2
        fan = 'HR choice trials';
    elseif all_lr_hr==3
        fan = 'abs(HR-LR) choice trials';        
    end
        
    fa = figure('position', [43   346   779   552]);
    fh = figure('position', [817    17   816   902]);
    set(fa, 'name', fan)
    set(fh, 'name', fan)
        
    for im = 1:length(mice)
        
        timenow = time_aligned_lick_allMice{im};
        nprenow = nPreMin_lick_allMice(im);
        if all_lr_hr==0
            tracenow = lick_aligned_allMice{im};  
        elseif all_lr_hr==1
            tracenow = lick_aligned_allMice_lrResp{im};  
        elseif all_lr_hr==2
            tracenow = lick_aligned_allMice_hrResp{im};  
        elseif all_lr_hr==3
            tracenow = lick_aligned_allMice_hrResp{im} - lick_aligned_allMice_lrResp{im};              
        end
        
        co = jet(size(tracenow,1)); % parula
        set(groot,'defaultAxesColorOrder',co)

        %%% average of days
        figure(fa)
        subplot(2,4,im); hold on

        plot(timenow, nanmean(tracenow,1), 'k')
    %     vline(0, 'r-.') % nprenow{im}
        xlim([timenow(1), timenow(end)])
        yl = get(gca, 'ylim');
        ylim([-.02, yl(2)])
        if im==1
            xlabel('Time rel. choice (ms)')
            ylabel('Fraction licks')
        end

        %%% single days
        subplot(2,4,4+im); hold on

        plot(timenow, tracenow')
    %     vline(0, 'r-.') % nprenow{im}
        xlim([timenow(1), timenow(end)])
        if im==1
            xlabel('Time rel. choice (ms)')
            ylabel('Fraction licks')
        end



        %%%%%%%%%%% heatmap showing all days %%%%%%%%%%%

        figure(fh)    
        subplot(4,1,im); hold on
        
        if all_lr_hr==3
            tracenow = abs(tracenow);
        end
        
        mxClim = quantile(tracenow(:), .9); % use this instead of max if you want to see better the values
        mnClim = 0; % quantile(tracenow(:), 0); %
        if mxClim==0
            mxClim = max(tracenow(:));
        end

        imagesc(tracenow, [mnClim, mxClim]); colorbar   
        x = sort(unique([nprenow: -249 : 0 , nprenow : 251 : size(tracenow,2)]));
        set(gca, 'xtick', x)
        set(gca, 'xticklabel', timenow(x)) 
        xlim([0, size(tracenow,2)])    %     xlim([nprenow(im)-1000 , nprenow(im)+251])    
        ylim([1, size(tracenow,1)])
    %     vline(nprenow{im}, 'r:') % 
        xlabel('Time rel. choice (ms)')
        ylabel('Days')
        set(gca,'ydir','reverse')
        c = colorbar;
        if im==1, c.Label.String = 'Fraction licks'; end

    end


    if savefigs        
        if all_lr_hr==0                
            dd = 'aveAllTrs_';
        elseif all_lr_hr==1
            dd = 'aveLrTrs_';
        elseif all_lr_hr==2
            dd = 'aveHrTrs_';
        end
            
        fdn = fullfile(dirn0, strcat('lick_',dd,'chAl_lineplot_timeCourse_allMice_', nowStr));
        savefig(fa, fdn)
        print(fa, '-dpdf', fdn)

        fdn = fullfile(dirn0, strcat('lick_',dd,'chAl_heatmap_timeCourse_allMice_', nowStr));
        savefig(fh, fdn)
        print(fh, '-dpdf', fdn)
    end

end



%% Figure of the corrs of lick traces for hr vs lr choice shown for each day 

figure('position', [43   704   905   228]);
for im = 1:length(mice)
    subplot(1,4,im); hold on
    plot(lickCorr_lr_hr{im})    
%     pn = nan(size(p_lickCorr_lr_hr{im}));
%     pn(p_lickCorr_lr_hr{im}<=.05) = 1;
%     plot(1:length(p_lickCorr_lr_hr{im}), 0*pn, 'r.')
    
    xlabel('Training day')
    if im==1
        ylabel(sprintf('Corrcoeff lick before \nhr vs lr choice'))
    end
    title(mice{im})
    box off
    set(gca,'tickdir','out')
end

if savefigs
    fdn = fullfile(dirn0, sprintf('lick_corrHrLrTraces_eachDays_win%d_%dms_allMice_%s', win(1), win(2),nowStr));
    savefig(gcf, fdn)
    print(gcf, '-dpdf', fdn)
end



%% How does the corr of wheelRev for hr vs lr choices change with training?

aaw_allM = cell(1,length(mice));
a_allM = cell(1,length(mice));

figure('position', [57   113   328   835]);
for im = 1:length(mice) 
    % corr_wheelRev_hrVSlr
    r = nPreMin_wheelRev_allMice(im) + win(1)/10 : nPreMin_wheelRev_allMice(im) + win(2)/10; % compute corr of wheelRev in this window
    aw = corr(wheelRev_aligned_allMice_hrResp{im}(:, r)' , wheelRev_aligned_allMice_lrResp{im}(:, r)'); 
    aaw = diag(aw);     
    aaw_allM{im} = aaw;
    
    % CA
    a = nanmean(av_test_data_allExc_aligned_allMice{im}(nPreMin_allMice(im)-5: nPreMin_allMice(im), :), 1); 
    a_allM{im} = a;

    % do days with more difference between hr vs lr wheelRev, have higher
    % CA? (I am using minus of aaw to use dis-similarity instead of
    % similarity)
    cc = corrcoef(-aaw, a, 'rows','complete'); 
    
%     figure; 
    subplot(4,1,im)
    plot(aaw); 
    title(sprintf('corr\n(CA,-corrWheelHRLR) %.2f', cc(2))); 
    if im==1, ylabel('Corr HR vs LR wheelRev'), end
    xlabel('Days')
    box off
    set(gca,'tickdir','out')
end

if savefigs
    fdn = fullfile(dirn0, sprintf('wheelRev_corrHrLrTraces_eachDays_win%d_%dms_allMice_%s', win(1), win(2),nowStr));
    savefig(gcf, fdn)
    print(gcf, '-dpdf', fdn)
end



%% Plot correlation of lr-lick with hr-lick for low vs high beh days 

fc = figure('position', [23    34   251   903]);

for im = 1:length(mice)
    [~,p] = ttest2(lickCorr_lr_hr_loBehDays{im}, lickCorr_lr_hr_hiBehDays{im});
    
    figure(fc);     
    subplot(4,1,im);
    
    boxplot(lickCorr_lr_hr_loBehDays{im}, 'position',1); yl1 = get(gca, 'ylim');
    hold on; 
    boxplot(lickCorr_lr_hr_hiBehDays{im}, 'position',2); yl2 = get(gca, 'ylim');
    set(gca,'xtick',[1,2])
    set(gca,'xticklabel',{'loBeh','hiBeh'})
    xlim([0,3])
    ylim([min([yl1,yl2]) , max([yl1,yl2])])
    title(sprintf('p %.2f', p))
    box off
    ylabel(sprintf('Corr \nHR- vs LR-lick trace')); 
%     if im==2; ylabel(sprintf('Corr \nHR- vs LR-lick trace')); end
end

if savefigs
    fdn = fullfile(dirn0, sprintf('lick_corrHrLrTraces_loHiBehDays_win%d_%dms_allMice_%s', win(1), win(2),nowStr));
    savefig(gcf, fdn)
    print(gcf, '-dpdf', fdn)
end




%% Heatmap of average across lick trace across low beh days vs high beh days 
%%%% This plot cannot help with the nature of the signal in the decoder (because it is not separating licks (wheelRev) for HR vs LR choices.

fh = figure('position', [817   359   816   560]);

for im = 1:length(mice)    
    tr = lick_aveLoHiBehDays{im};
    
    figure(fh)    
    subplot(4,1,im); hold on    
    mxClim = quantile(tr(:), .9); % use this instead of max if you want to see better the values
%     mxClim = max(lick_aligned_allMice{im}(:));
    
    imagesc(tr, [0,.05]); % plot average across lick trace across low beh days vs high beh days
    
    x = sort(unique([nPreMin_lick_allMice(im): -249 : 0 , nPreMin_lick_allMice(im) : 251 : size(lick_aligned_allMice{im},2)]));
    set(gca, 'xtick', x)
    set(gca, 'xticklabel', time_aligned_lick_allMice{im}(x))  
    xlim([0, size(lick_aligned_allMice{im},2)])    
    ylim([1, size(tr,1)])
%     vline(nPreMin_lick_allMice{im}, 'r:') % 
    xlabel('Time rel. choice (ms)')
    ylabel('Days')
    set(gca,'ydir','reverse')
    c = colorbar;
    if im==1, c.Label.String = 'Fraction licks'; end    
    title(sprintf('corr (lo vs hi beh): %.2f', lickCorr_lo_vs_hiBehDays(im)))
    
end

if savefigs
    fdn = fullfile(dirn0, strcat('lick_aveLoHiBehDays_heatmap_timeCourse_allMice_', nowStr));
    savefig(fh, fdn)
    print(fh, '-dpdf', fdn)
end



%% Scatter plots of SVM class accuracy vs. wheel travelled distance, speed and licks; show each day, and compute correlation. Do it for time bin -1 relative to choice.
%%%% This plot cannot help with the nature of the signal in the decoder (because it is not separating licks (wheelRev) for HR vs LR choices.

aw = 300; % 100 % average licks in what window. ... you can perhaps get a bout of licks if you go back to ~150ms.... varies between mice!
    
ca_wheel_timeM1_corr = nan(1,length(mice));
ca_speed_timeM1_corr = nan(1,length(mice));
ca_lick_timeM1_corr = nan(1,length(mice));

figure('position', [1         191        1197         721]); 
% co = jet(max(cellfun(@sum, days2an_heatmap_allMice))); % parula
% set(groot,'defaultAxesColorOrder',co)

for im = 1:length(mice)

    %%%%% CA at time -1 for all days
    ca = av_test_data_allExc_aligned_allMice{im};
    np = nPreMin_allMice(im);
    ds = days2an_heatmap_allMice{im};
    catm1 = ca(np, ds)'; % CA at time -1 % days
    
    
    %%%%% wheel rev    
    w = wheelRev_aligned_allMice{im}; % days x timebins
    npw = nPreMin_wheelRev_allMice(im); % each bin is 10ms, CA is 100ms, so average 10 bins before (and including) nPreMin!
    % first 10-bin before the choice
    tm1 = npw-9:npw; 
    wtm1 = w(ds, tm1); % days x 10        % get w at tm1 bins
    wtm1 = mean(wtm1,2); % days % average 10 bins    
    % second 10-bin
    tm2 = npw-19:npw-10;
    wtm2 = w(ds, tm2); % days x 10    
    wtm2 = mean(wtm2,2); % days % average 10 bins
    % change in travelled distance from time -2 to time -1
    wtm12 = wtm1 - wtm2;


   
    %%%% speed of wheel rev
    s = speed_d_allMice{im}; % days x downsampTimeBins
    nps = nPreMin_wheelRev_allMice_ds(im); % each bin is 100ms
    stm1 = s(ds, nps); % speed at time -1 % days
    

    
    %%%% fraction licks averaged in 150ms window
    l = lick_aligned_allMice{im}; % days x times
    npl = nPreMin_lick_allMice(im); % each bin is 1ms
    ltm1 = mean(l(ds, npl-aw+1:npl),2); % speed at time -1 % days
    
    
    %%%%%% Correlations: set corr between CA and wheelRev or speed or licks
    ca_wheel_timeM1_corr(im) = corr(catm1, wtm12);    
    ca_speed_timeM1_corr(im) = corr(catm1, stm1);
    ca_lick_timeM1_corr(im) = corr(catm1, ltm1);

    
    
    %%%%% Plots 
    
    % wheelRev change from time -2 to -1    
    co = jet(length(catm1)); % max(cellfun(@sum, days2an_heatmap_allMice))); % parula    
    subplot(3,4,im); hold on
    for i=1:length(wtm12)
        plot(wtm12(i), catm1(i), '.', 'color', co(i,:))
    end    
%     plot(catm1, wtm1, 'k.')
    title(sprintf('%s, corr: %.2f', mice{im}, ca_wheel_timeM1_corr(im)))
    xlabel('Travelled distance (mm)')
    ylabel('Class accuracy %')
    
    
    % speed of wheel rev at time -1
    subplot(3,4,4+im); hold on    
    for i=1:length(wtm12)
        plot(stm1(i), catm1(i), '.', 'color', co(i,:))
    end    
%     plot(catm1, wtm1, 'k.')
    title(sprintf('%s, corr: %.2f', mice{im}, ca_speed_timeM1_corr(im)))
    xlabel('Running speed (mm/sec)')
    ylabel('Class accuracy %')
    


    % licks at 150ms before the choice 
    subplot(3,4,2*4+im); hold on    
    for i=1:length(wtm12)
        plot(ltm1(i), catm1(i), '.', 'color', co(i,:))
    end    
%     plot(catm1, wtm1, 'k.')
    title(sprintf('%s, corr: %.2f', mice{im}, ca_lick_timeM1_corr(im)))
    xlabel('Fraction licks')
    ylabel('Class accuracy %')    
    
end





%% Low vs high behavioral days, how does corr of wheelRev for hr vs lr change?
%{
for im=1:length(mice)
    
    a = behCorr_all_allMice{im}(days2an_heatmap_allMice{im});
    thb = quantile(a, perc_thb);
    
    loBehCorrDays = (a <= thb(1));
    hiBehCorrDays = (a >= thb(2));
    disp([sum(loBehCorrDays), sum(hiBehCorrDays)])
    
    aa = aaw_allM{im}(days2an_heatmap_allMice{im},:);
    
    a = mean(aa(loBehCorrDays,:),1);  % average across low beh days
    ah = mean(aa(hiBehCorrDays,:),1);
    [a,ah]
end
    
    
for im = 1:4
    a0 = aaw_allM{im} > .5; .7;
    ah = nanmedian(a_allM{im}(a0));
    a1 = aaw_allM{im} <= .5; 0;
    al = nanmedian(a_allM{im}(a1));
    [ah, al]
    [~, p] = ttest2(a_allM{im}(a0),a_allM{im}(a1))
end
%}


%% Compute CA at time -1 for days with low vs high corr_lrHr_lick 
%{
for im = 1:length(mice)

    %%%%% CA at time -1 for all days
    ca = av_test_data_allExc_aligned_allMice{im};
    np = nPreMin_allMice(im);
    ds = days2an_heatmap_allMice{im};
    catm1 = ca(np, ds)'; % CA at time -1 % days
    
    lickCorr_lr_hr
end
%}



%% Set heatmap of significancy for wheelRev and licks (HR vs LR)

%%% wheelRev
figure('position', [680    60   240   914]); 
for im = 1:length(mice)
    timenow = time_aligned_wheelRev_allMice{im};
    nprenow = nPreMin_wheelRev_allMice(im);        
    ds = 1;
    subplot(4,1,im); 
    imagesc(sig_wheelRev_hr_lr_aligned_allMice{im})
    hold on
    vline(nPreMin_wheelRev_allMice(im))
    
    x = sort(unique([nprenow: -30/ds : -30/ds , nprenow : 30/ds : size(sig_wheelRev_hr_lr_aligned_allMice{im},2)]));
    x(x<0) = [];  % x = x(2:end);
    set(gca, 'xtick', x)
    set(gca, 'xticklabel', timenow(x))       
    xlabel('Time rel. choice (ms)')
    ylabel('Days')
end

if savefigs        
    fdn = fullfile(dirn0, strcat('wheelRev_sigHRvsLR_chAl_heatmap_timeCourse_allMice_', nowStr));
    savefig(gcf, fdn)
    print(gcf, '-dpdf', fdn)
end



%%%% Lick
figure('position', [68    52   580   921]); 
for im = 1:length(mice)
    timenow = time_aligned_lick_allMice{im};
    nprenow = nPreMin_lick_allMice(im);

    subplot(4,1,im); 
    imagesc(sig_lick_hr_lr_aligned_allMice{im})
    hold on
    vline(nPreMin_lick_allMice(im))

    x = sort(unique([nprenow: -249 : 0 , nprenow : 251 : size(sig_lick_hr_lr_aligned_allMice{im},2)]));
    set(gca, 'xtick', x)
    set(gca, 'xticklabel', timenow(x))
    xlabel('Time rel. choice (ms)')
    ylabel('Days')    
end


if savefigs        
    fdn = fullfile(dirn0, strcat('lick_sigHRvsLR_chAl_heatmap_timeCourse_allMice_', nowStr));
    savefig(gcf, fdn)
    print(gcf, '-dpdf', fdn)
end


%% Set Class accuracy for days that running is not sig diff between lr and hr trials

% you also need the shuffled CA to show the CA of data from days with no
% sig change in running between lr and hr is still sig!

winn = [-9,0]; % average across last 10 bins = 100 ms

CA_sameWheelRevHrLrDays = cell(1, length(mice));
CA_diffWheelRevHrLrDays = cell(1, length(mice));
for im = 1:length(mice)
    r = nPreMin_wheelRev_allMice(im) + winn(1) : nPreMin_wheelRev_allMice(im) + winn(2);
    sig = mean(sig_wheelRev_hr_lr_aligned_allMice{im}(:, r),2); % average across last 10 bins = 100 ms
    CA_sameWheelRevHrLrDays{im} = av_test_data_allExc_aligned_allMice{im}(nPreMin_allMice(im), sig==0);    
    CA_diffWheelRevHrLrDays{im} = av_test_data_allExc_aligned_allMice{im}(nPreMin_allMice(im), sig==1);    
end


%%%% same HR- vs LR-wheel trac
yl1 = nan(2, length(mice));
fc = figure;
for im = 1:length(mice)
    figure(fc);     
    boxplot(CA_sameWheelRevHrLrDays{im}, 'position',im); 
    yl1(:,im) = get(gca, 'ylim');
    hold on; 
%     title(sprintf('p %.2f', p))
    box off
    if im==1, ylabel(sprintf('Class accuracy of days with \nsame HR- vs LR-wheel trace')); end
end
set(gca,'xtick',1:length(mice)) %     set(gca,'xtick',[1,2])
set(gca,'xticklabel', (1:length(mice))) %     set(gca,'xticklabel',{'loBeh','hiBeh'})
ylim([min(yl1(:)) , max(yl1(:))])
xlim([0,im+1])
title('Same HR and LR wheel trace')

if savefigs        
    fdn = fullfile(dirn0, sprintf('wheelRev_CA_sameHRvsLR_win%d_%dms_allMice_%s', winn(1)*10, winn(2)*10, nowStr));
    savefig(gcf, fdn)
    print(gcf, '-dpdf', fdn)
end



%%%% different HR- vs LR-wheel trace
yl2 = nan(2, length(mice));
fc = figure;
for im = 1:length(mice)
    figure(fc)
    boxplot(CA_diffWheelRevHrLrDays{im}, 'position',im); 
    yl2(:,im) = get(gca, 'ylim');
    hold on; 
%     title(sprintf('p %.2f', p))
    box off
    if im==1, ylabel(sprintf('Class accuracy of days with \ndifferent HR- vs LR-wheel trace')); end

end
set(gca,'xtick',1:length(mice)) %     set(gca,'xtick',[1,2])
set(gca,'xticklabel', (1:length(mice))) %     set(gca,'xticklabel',{'loBeh','hiBeh'})
ylim([min(yl2(:)) , max(yl2(:))])
xlim([0,im+1])
title('Different HR and LR wheel trace')

if savefigs        
    fdn = fullfile(dirn0, sprintf('wheelRev_CA_diffHRvsLR_win%d_%dms_allMice_%s', winn(1)*10, winn(2)*10, nowStr));
    savefig(gcf, fdn)
    print(gcf, '-dpdf', fdn)
end


%%
%{
get p values on the downsampled wheelRev trace

I would summarize it for time -1:
get the difference of wheelRev plots (for each day, hr - lr).
see how this difference at time -1 covaries with CA at time -1, across training days
report this corr value it for all mice (I believe fni16, 17 and perhaps 18 are correlated, but fni19 is not).
also show in low vs hi beh days, diff_wheelRev_hrVSlr and see if hi beh days have higher diff_wheelRev... 
conclusion: learning induces changes in diff_wheelRev in a couple of mice, hence in those mice it is likely population activity is representing locomotion correlated with the choice.
%}
