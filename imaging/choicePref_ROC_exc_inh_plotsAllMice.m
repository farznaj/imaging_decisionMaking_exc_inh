%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Summary plots of all mouse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dirn0 = '/home/farznaj/Dropbox/ChurchlandLab/Projects/inhExcDecisionMaking/ROC';

savefigs = sumMice_do_savefigs(2); % savefigs = 0;
cols = {'b','r'}; % exc,inh, real data
colss = [0,.8,.8; .8,.5,.8]; % exc,inh colors for shuffled data
nowStr = nowStr_allMice{imfni18}; % use nowStr of mouse fni18 (in case its day 4 was removed, you want that to be indicated in the figure name of summary of all mice).

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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Plot average timecourse of ROC across mice %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Plot average and se across mice (ROC is already averaged across days (and neurons))

fh = figure; hold on
h1 = boundedline(time_al, nanmean(exc_avDays_eachMouse,2), nanstd(exc_avDays_eachMouse,[],2)/sqrt(size(exc_avDays_eachMouse,2)), 'b', 'alpha');
h2 = boundedline(time_al, nanmean(inh_avDays_eachMouse,2), nanstd(inh_avDays_eachMouse,[],2)/sqrt(size(inh_avDays_eachMouse,2)), 'r', 'alpha');
if doshfl % shfl
    h11 = boundedline(time_al, nanmean(exc_avDays_eachMouse_shfl,2), nanstd(exc_avDays_eachMouse_shfl,[],2)/sqrt(size(exc_avDays_eachMouse_shfl,2)), 'cmap', colss(1,:), 'alpha');
    h12 = boundedline(time_al, nanmean(inh_avDays_eachMouse_shfl,2), nanstd(inh_avDays_eachMouse_shfl,[],2)/sqrt(size(inh_avDays_eachMouse_shfl,2)), 'cmap', colss(2,:), 'alpha');
end
a = get(gca, 'ylim');    
plot(time_al, hh0_allm*(a(2)-.05*diff(a)), 'k.')    
plot([0,0],a,'k:')
b = get(gca, 'xlim');
if ~isempty(yy)
    plot(b, [yy,yy],'k:')
end
legend([h1,h2], {'Excitatory', 'Inhibitory'}, 'position', [0.1347    0.8177    0.1414    0.0901]);
xlabel('Time since choice onset (ms)')
ylab = simpleTokenize(namc, '_'); ylab = ylab{1}; %namc;   
ylabel(ylab)
% title('mean+se mice (Ns aved per day;days aved per mouse)')
xlim([time_al(1)-50, time_al(end)+50])

% figs_adj_poster_ax(fh)

if savefigs
    fdn = fullfile(dirn00, strcat(namc,'_','ROC_curr_chAl_excInh_timeCourse_aveSeMice_aveDays_aveNs_', dm0, nowStr));
    savefig(fh, fdn)
    print(fh, '-dpdf', fdn)
end
        

%% Plot average and se of pooled days (of all mice; they are already averaged across neurons for each day)

fh = figure; hold on
h1 = boundedline(time_al, nanmean(exc_allDaysPooled_allMice,2), nanstd(exc_allDaysPooled_allMice,[],2)/sqrt(sum(numDaysGood)), 'b', 'alpha');
h2 = boundedline(time_al, nanmean(inh_allDaysPooled_allMice,2), nanstd(inh_allDaysPooled_allMice,[],2)/sqrt(sum(numDaysGood)), 'r', 'alpha');
if doshfl % shfl
    h11 = boundedline(time_al, nanmean(exc_allDaysPooled_allMice_shfl,2), nanstd(exc_allDaysPooled_allMice_shfl,[],2)/sqrt(sum(numDaysGood)), 'cmap', colss(1,:), 'alpha');
    h12 = boundedline(time_al, nanmean(inh_allDaysPooled_allMice_shfl,2), nanstd(inh_allDaysPooled_allMice_shfl,[],2)/sqrt(sum(numDaysGood)), 'cmap', colss(2,:), 'alpha');
end
a = get(gca, 'ylim');    
plot(time_al, hh_allm*(a(2)-.05*diff(a)), 'k.')    
plot([0,0],a,'k:')
b = get(gca, 'xlim');
if ~isempty(yy)
    plot(b, [yy,yy],'k:')
end
legend([h1,h2], {'Excitatory', 'Inhibitory'}, 'position', [0.1347    0.8177    0.1414    0.0901]);
xlabel('Time since choice onset (ms)')
ylab = simpleTokenize(namc, '_'); ylab = ylab{1}; %namc;   
ylabel(ylab)
% title('mean+se pooled days (Ns aved per day)')

% figs_adj_poster_ax(fh)

if savefigs
    fdn = fullfile(dirn00, strcat(namc,'_','ROC_curr_chAl_excInh_timeCourse_aveSePooledDaysAllMice_aveNs_', dm0, nowStr));
    savefig(fh, fdn)
    print(fh, '-dpdf', fdn)
end
        

%%
%%%%%%%%%%%%%%%%%%%%%%% Distributions of individual neurons %%%%%%%%%%%%%%%%%%%%%%%

%% Compare exc/inh ROC dist for each frame, all days pooled

numBins = 25;
documsum = 0;
xlab = simpleTokenize(namc, '_'); xlab = xlab{1}; %namc;        
ylab = 'Fraction neurons,days,mice';
leg = {'exc','inh'};
% cols = {'b','r'};

fign = figure('position', [680   197   696   779]);
c = 4;        
ha = tight_subplot(ceil(length(time_al)/c), c, [.1 .04],[.03 .03],[.1 .03]);

for ifr = 1:length(time_al)        
    tit = round(time_al(ifr));
    
    y1 = exc_allNsDaysMicePooled(ifr,:);
    y2 = inh_allNsDaysMicePooled(ifr,:);
    bins = plotHist_sp(y1,y2,xlab,ylab,leg, cols, tit, fign, ha(ifr), yy, documsum, numBins);
        
    if doshfl % sfhl
        % average of shfl samples
%         y1 = exc_allNsDaysMicePooled_shfl(ifr,:);
%         y2 = inh_allNsDaysMicePooled_shfl(ifr,:);   
%         plotHist_sp(y1,y2,xlab,ylab,leg, colss, tit, fign, ha(ifr), yy, documsum, numBins, bins)
        
        % individual shlf sampls
        y1s = exc_allNsDaysMicePooled_shfl0(ifr,:,:);
        y2s = inh_allNsDaysMicePooled_shfl0(ifr,:,:);
        % pool all samples
        y1s = y1s(:);
        y2s = y2s(:);
        plotHist_sp(y1s,y2s,xlab,ylab,leg, colss, tit, fign, ha(ifr), yy, documsum, numBins, bins)
    end

    if ifr==1
        xlabel(ha(ifr), xlab)
        ylabel(ha(ifr), ylab)
        legend(ha(ifr), leg)
    end
end

if savefigs        
    fdn = fullfile(dirn00, strcat(namc,'_','ROC_curr_chAl_excInh_dist_eachFr_NsDaysMicePooled_', dm0, nowStr));
    savefig(fign, fdn)    
    print(fign, '-dpdf', fdn)
end   


%% Get dist of roc for exc and inh at time bin -1

time2an = -1;

% frame before choice on the mouse-aligned traces
e = nPreMin+1;
fr = e+time2an;

nBins = 50;
doSmooth = 1;
xlab = simpleTokenize(namc, '_'); xlab = xlab{1}; %namc;        
ylab = 'Fraction neurons,days,mice';

fign = figure; % f = nan(1,1); f(1) = axes();

y1 = exc_allNsDaysMicePooled(fr,:);
y2 = inh_allNsDaysMicePooled(fr,:);
plotHist(y1,y2,xlab,ylab,leg, cols, yy, fign, nBins, doSmooth)%, lineStyles, sp, bins); 
if doshfl % shfl
    % average of shfl samples .... I don't think this is informative...
    % this is showing the dist of mean of shfls but the one below shows the
    % dist of all shfl samps... and we are comparing it with the dist of
    % all data vals!
%     y1s = exc_allNsDaysMicePooled_shfl(fr,:);
%     y2s = inh_allNsDaysMicePooled_shfl(fr,:);
%     plotHist(y1s,y2s,xlab,ylab,leg, colss, yy,fign, nBins, doSmooth);
    
    % individual shlf sampls
    y1s = exc_allNsDaysMicePooled_shfl0(fr,:,:);
    y2s = inh_allNsDaysMicePooled_shfl0(fr,:,:);
    % pool all samples
    y1s = y1s(:);
    y2s = y2s(:);
    plotHist(y1s,y2s,xlab,ylab,leg, colss, yy,fign, nBins, doSmooth);
end


if savefigs        
    fdn = fullfile(dirn00, strcat(namc,'_','ROC_curr_chAl_excInh_dist_time-1_NsDaysMicePooled_', dm0, nowStr));
    savefig(fign, fdn)    
    print(fign, '-dpdf', fdn)
end 

% fign = figure; % f = nan(1,1); f(1) = axes();
% plotHist_sp(y1,y2,xlab,ylab,leg, cols, tit, fign, gca, yy, documsum)
% xlabel(xlab)
% ylabel(ylab)
% legend(leg)


%%
%%%%%%%%%%%%%%%%%%%%%%% single mouse plots %%%%%%%%%%%%%%%%%%%%%%%

%% Error bar, time bin -1: 

% frame before choice on the mouse-aligned traces
e = nPreMin+1;
fr = e+time2an;


%%%%%%%%%%%%%%%%%%%%%%%% show ave across days for each mouse  %%%%%%%%%%%%%%%%%%%%%%%%
% (traces are already averaged across neurons)

% for each mouse, day-averaged value at time bin -1
eav = exc_avDays_eachMouse(fr,:);
ese = exc_seDays_eachMouse(fr,:);

iav = inh_avDays_eachMouse(fr,:);
ise = inh_seDays_eachMouse(fr,:);

% ttest for each mouse, exc vs inh across days
pallm_days = nan(length(time_al), length(mice));
hallm_days = nan(length(time_al), length(mice));
for im=1:length(mice)
    [h,p0] = ttest2(aveexc_allMice{im}', aveinh_allMice{im}'); % 1 x nFrs
    pallm_days(:,im) = p0;
    hallm_days(:,im) = h;
end
hh_days = hallm_days;
hh_days(hallm_days==0) = nan;
pallm_days(fr,:)


if doshfl  % shfl
    eavs = exc_avDays_eachMouse_shfl(fr,:);
    eses = exc_seDays_eachMouse_shfl(fr,:);

    iavs = inh_avDays_eachMouse_shfl(fr,:);
    ises = inh_seDays_eachMouse_shfl(fr,:);
end


%%%%%%%%%%%%%%% Plot
g = 0;
fign = figure('position',[60   593   450   320]); 

subplot(121), hold on;
errorbar(1:length(mice), eav, ese, 'b', 'linestyle','none', 'marker','o')
errorbar((1:length(mice))+g, iav, ise ,'r', 'linestyle','none', 'marker','o')
if doshfl % shfl
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
plot(1:length(mice), hh_days(fr,:).*ym, 'k*')
title('aveDays,aveNs')



%%%%%%%%%%%%%%%%%%%%%%%% show ave across pooled neurons of all days for each mouse %%%%%%%%%%%%%%%%%%%%%%%%

% number of valid neurons (from all days) for each mouse
exc_nValidNs = nan(1,length(mice));
inh_nValidNs = nan(1,length(mice));
for im=1:length(mice)
    exc_nValidNs(im) = sum(~isnan(exc_allNsDaysPooled_eachMouse{im}(1,:)));
    inh_nValidNs(im) = sum(~isnan(inh_allNsDaysPooled_eachMouse{im}(1,:)));
end

eav = cellfun(@(x)nanmean(x(fr,:),2), exc_allNsDaysPooled_eachMouse);
ese = cellfun(@(x)nanstd(x(fr,:),[],2), exc_allNsDaysPooled_eachMouse)/sqrt(exc_nValidNs(im));

iav = cellfun(@(x)nanmean(x(fr,:),2), inh_allNsDaysPooled_eachMouse);
ise = cellfun(@(x)nanstd(x(fr,:),[],2), inh_allNsDaysPooled_eachMouse)/sqrt(inh_nValidNs(im));


% ttest for each mouse, exc vs inh across pooled neurons of all days
pallm_pooledN = nan(length(time_al), length(mice));
hallm_pooledN = nan(length(time_al), length(mice));
for im=1:length(mice)
    [h,p0] = ttest2(exc_allNsDaysPooled_eachMouse{im}', inh_allNsDaysPooled_eachMouse{im}'); % 1 x nFrs
    pallm_pooledN(:,im) = p0;
    hallm_pooledN(:,im) = h;
end
hh_pooledN = hallm_pooledN;
hh_pooledN(hallm_pooledN==0) = nan;
pallm_pooledN(fr,:)


if doshfl  % shfl
    eavs = cellfun(@(x)nanmean(x(fr,:),2), exc_allNsDaysPooled_eachMouse_shfl);
    eses = cellfun(@(x)nanstd(x(fr,:),[],2), exc_allNsDaysPooled_eachMouse_shfl)/sqrt(exc_nValidNs(im));

    iavs = cellfun(@(x)nanmean(x(fr,:),2), inh_allNsDaysPooled_eachMouse_shfl);
    ises = cellfun(@(x)nanstd(x(fr,:),[],2), inh_allNsDaysPooled_eachMouse_shfl)/sqrt(inh_nValidNs(im));
end


%%%%% Plot
g = 0;
% figure('position',[60   593   239   320]); 

subplot(122), hold on;
errorbar(1:length(mice), eav, ese, 'b', 'linestyle','none', 'marker','o')
errorbar((1:length(mice))+g, iav, ise ,'r', 'linestyle','none', 'marker','o')
if doshfl  % shfl
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
plot(1:length(mice), hh_pooledN(fr,:).*ym, 'k*')
title('avePooledNs')


% save figure
if savefigs        
    fdn = fullfile(dirn00, strcat(namc,'_','ROC_curr_chAl_excInh_aveSe_time-1_eachMouse_', dm0, nowStr));
    savefig(fign, fdn)    
    print(fign, '-dpdf', fdn)
end 




%% Plot fractions of significantly choice-tuned neurons, compare between exc and inh

fign = figure;

% all preferences (ipsi and contra)
x = 1:length(mice);
y = [exc_fractSigTuned; inh_fractSigTuned]';
subplot(221)
b = bar(x, y);
b(1).FaceColor = cols{1};
b(2).FaceColor = cols{2};
xlabel('Mice')
ylabel('Fraction choice-tuned') % significant
legend('exc', 'inh')
title('All neurons')
set(gca, 'tickdir', 'out', 'box', 'off')


% ipsi-preferring neurons
y = [exc_fractSigTuned_ipsi; inh_fractSigTuned_ipsi]';
subplot(222)
b = bar(x, y);
b(1).FaceColor = cols{1};
b(2).FaceColor = cols{2};
xlabel('Mice')
ylabel('Fraction choice-tuned') % significant
title('Ns with AUC > 0.5')
set(gca, 'tickdir', 'out', 'box', 'off')


% contra-preferring neurons
y = [exc_fractSigTuned_contra; inh_fractSigTuned_contra]';
subplot(224)
b = bar(x, y);
b(1).FaceColor = cols{1};
b(2).FaceColor = cols{2};
xlabel('Mice')
ylabel('Fraction choice-tuned') % significant
title('Ns with AUC < 0.5')
set(gca, 'tickdir', 'out', 'box', 'off')


% ipsi and contra
y = [exc_fractSigTuned_ipsi; exc_fractSigTuned_contra; inh_fractSigTuned_ipsi; inh_fractSigTuned_contra]';
subplot(223); hold on
b = bar(x, y);
b(1).FaceColor = 'b'; % ipsi
b(2).FaceColor = rgb('lightblue'); % contra
b(3).FaceColor = 'r'; % ipsi
b(4).FaceColor = rgb('lightsalmon'); % contra

xlabel('Mice')
ylabel('Fraction choice-tuned') % significant
xlim([0, length(mice)+1])
legend('exc-ipsi', 'exc-contra', 'inh-ipsi', 'inh-contra')
set(gca, 'tickdir', 'out', 'box', 'off')


% save figure
if savefigs        
    fdn = fullfile(dirn00, strcat(namc,'_','ROC_curr_chAl_excInh_fractSigTuned_time-1_', dm0, nowStr));
    savefig(fign, fdn)    
    print(fign, '-dpdf', fdn)
end 


%%%%%%%%%%%%%%%%%%%%%%%%%
fign = figure;

% ipsi-preferring neurons
y = [exc_fractSigIpsiTuned; exc_fractSigContraTuned]';
subplot(222)
b = bar(x, y);
b(1).FaceColor = 'b';
b(2).FaceColor = rgb('lightblue');
xlabel('Mice')
ylabel('Fraction choice-tuned') % significant
legend({'Ipsi-preferring', 'Contra-preferring'}, 'position', [0.1655    0.7861    0.2786    0.0821])
set(gca, 'tickdir', 'out', 'box', 'off')
title('Excitatory')

% contra-preferring neurons
y = [inh_fractSigIpsiTuned; inh_fractSigContraTuned]';
subplot(224)
b = bar(x, y);
b(1).FaceColor = 'r';
b(2).FaceColor = rgb('lightsalmon');
xlabel('Mice')
ylabel('Fraction choice-tuned') % significant
legend({'Ipsi-preferring', 'Contra-preferring'}, 'position', [0.1655    0.2861    0.2786    0.0821])
set(gca, 'tickdir', 'out', 'box', 'off')
title('Inhibitory')

% save figure
if savefigs        
    fdn = fullfile(dirn00, strcat(namc,'_','ROC_curr_chAl_excInh_fractIpsiContraSigTuned_time-1_', dm0, nowStr));
    savefig(fign, fdn)    
    print(fign, '-dpdf', fdn)
end 



%% Plot hist of probabilities of data ROC coming from the same distribution as shuffled (assuming shuffled is a normal distribution with mu and sigma equal to mean and std of shuffled ROC values for each neuron).

xlab = 'Prob(data from shfl dist)';
ylab = 'Fraction neurons,days';
nBins = 100;
alpha = .05;

for im = 1:length(mice)
    y1 = exc_prob_dataROC_from_shflDist_eachMouseDaysPooled{im};
    y2 = inh_prob_dataROC_from_shflDist_eachMouseDaysPooled{im};   
   
    fh = figure('name', mice{im});
    [fh,bins] = plotHist(y1,y2,xlab,ylab,leg, cols, alpha, fh, nBins); %, doSmooth, lineStyles, sp, bins);
end



