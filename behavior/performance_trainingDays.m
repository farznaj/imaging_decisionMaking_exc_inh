% Plot performance vs training days for each mouse

%% Compute HRchoicePerc for each session of each mouse.

miceNames = {'fni16', 'fni17', 'fni18', 'fni19'}; %{'fn03', 'fn04', 'fn05', 'fn06', 'fni16', 'fni17', 'fni18', 'fni19', 'hni01', 'hni04'};
allowCorrectResp = 'change';
uncommittedResp = 'nothing'; % 'change'; %'remove'; % % 'remove', 'change', 'nothing';
excludeShortWaitDur = true; % waitdur_th = .032; % sec  % trials w waitdur less than this will be excluded.
excludeExtraStim = false;

doplots = 0;
plotPMF = false;
shownumtrs = false; %true;
% only use easy stimuli
clear vecr
vecr{1} = [4 6];
vecr{2} = [26 28];
% wd = 2;
% vec_rates = sort([cb : -wd : 4  ,  cb+wd : wd : 28]);
% vec_rates(end) = vec_rates(end)+1;

% performance_allmice = NaN(length(vec_rates) - 1, length(miceNames));
HRchoicePerc_allSess_allMice = cell(1, length(miceNames));


%%
for imouse = 1:length(miceNames)
    
    % Remember data are loaded from the latest day to the earliest day.
    
    fprintf('--------------------------\n')
    mouse = miceNames{imouse};
    [day, dayLast, days2exclude] = setMouseAnalysisDays(mouse);
    
    [alldata_fileNames, days_all] = setBehavFileNames(mouse, day, dayLast, days2exclude);
    fprintf('Total number of session: %d\n', length(alldata_fileNames))
    
    [alldata, trials_per_session] = loadBehavData(alldata_fileNames);
    fprintf('Total number of trials: %d\n', length(alldata))
    trials_per_mouse(imouse) = length(alldata);
    
    
    %%
    [stimrate, y] = stimrate_choice_set...
        (alldata, trials_per_session, uncommittedResp, allowCorrectResp, ...
        excludeExtraStim, excludeShortWaitDur, mouse);
    
    % to get pmf:
    %{
    [stimrate, y, HRchoicePerc, vec_rates, up, lo, nSamples] = stimrate_choice_set...
        (alldata, trials_per_session, uncommittedResp, allowCorrectResp, ...
        excludeExtraStim, excludeShortWaitDur, mouse);
    %}
    
    
    %%    
    cs = [0 cumsum(trials_per_session)];
%     HRchoicePerc_allSess = NaN(length(cs)-1, length(vecr)-1);
    HRchoicePerc_allSess = NaN(length(cs)-1, length(vecr));
    
    for isess = 1:length(cs)-1
        
        trRange = cs(isess)+1 : cs(isess+1);
        good_corr_incorr = ~isnan(y(trRange)');
        
        if sum(good_corr_incorr) > 10 % at least 10 trials per session needed to compute HRchoicePerc
            
            %% compute prop HR for each bin of vec_rates
            
%             cb = alldata(1).categoryBoundaryHz;
%             HRchoicePerc = PMF_set_plot(stimrate(trRange), y(trRange), cb, vec_rates, plotPMF, shownumtrs);
            
            HRchoicePerc = NaN(1, length(vecr));
            nSamples = NaN(1, length(vecr)); % number of trials in each bin of vec_rates
            
            for ri = 1:length(vecr)                
                allResp_HR_LR = y(trRange);
                allout = allResp_HR_LR((stimrate(trRange) >= vecr{ri}(1)) & (stimrate(trRange) <= vecr{ri}(2))); % choice of mouse on all trials whose stim rate was in bin ri
                validout = allout(allout>=0); % only consider trials that were correct, incorrect, noDecision, and noSideLickAgain trials. (exclude non valid trials).
                nSamples(ri) = length(validout);                
%                 if length(validout)>th
                    HRchoicePerc(ri) = nanmean(validout); 
%                 end
            end
            
            HRchoicePerc_allSess(isess,:) = HRchoicePerc; % HRchoicePerc(1:end-1);           
            
        end
    end
    
    
    %%
    HRchoicePerc_allSess_allMice{imouse} = HRchoicePerc_allSess;
%     performance_allmice(:, icount) = nanmean(HRchoicePerc_allSess, 1);
%     performance_allmice(icount) = nanmean([1 - HRchoicePerc(1:3) , HRchoicePerc(end-2:end)]);
    

end


%% Plot performance vs training days for each mouse

l = max(cellfun(@(x)size(x,1), HRchoicePerc_allSess_allMice));
perf_ave_allmice = NaN(l, length(miceNames));
for imouse = 1 : length(miceNames)
    p = HRchoicePerc_allSess_allMice{imouse};
    perf = [1 - p(:,1) , p(:,end)];
    perf_ave_allmice(1:size(perf,1), imouse) = nanmean(perf,2);
end
perf_ave_allmice = perf_ave_allmice(end:-1:1,:); % so its chronological

figure; plot(perf_ave_allmice) 
figure;
for imouse = 1 : length(miceNames)
    plot(perf_ave_allmice(:, imouse))
    pause
end




%% Plot for each mouse training days until peforming ~ >=20 above chance
%{
fn04: 9 d
fn05: 10 d
fni11: didn't learn
fni16: 1 month
fni17: 3 w
fni18: 1 month
fni19: 3.5 weeks
hni01: 2.5 weeks
hni04: 1 month
%}
% d = [9, 10, 30, 21, 25, 24, 16, 28];
% d = [d, 14]; % missing animals
d = [7, 8, 24, 17, 21, 20, 14, 23, 12]; % excluding weekends
% figure; errorbar(mean(d), std(d), 'ro')

figure; hold on
[x,y] = plotcone(d,1);
plot(x, y,'marker','d','markerfacecolor','k','markeredgecolor','k','markersize',2,'linestyle','none')
plot(1, nanmean(y), 'r*')
xlim([.5 1.5])
ylim([min(y)-2 max(y)+2])
set(gca, 'xticklabel', '')
ylabel('Training days') % until peforming >=20% above chance
set(gca,'tickdir','out')
% legend('mouse')




