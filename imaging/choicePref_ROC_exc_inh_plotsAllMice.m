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
h3 = boundedline(time_al, nanmean(uns_avDays_eachMouse,2), nanstd(uns_avDays_eachMouse,[],2)/sqrt(size(uns_avDays_eachMouse,2)), 'g', 'alpha');
if doshfl % shfl
    h11 = boundedline(time_al, nanmean(exc_avDays_eachMouse_shfl,2), nanstd(exc_avDays_eachMouse_shfl,[],2)/sqrt(size(exc_avDays_eachMouse_shfl,2)), 'cmap', colss(1,:), 'alpha');
    h12 = boundedline(time_al, nanmean(inh_avDays_eachMouse_shfl,2), nanstd(inh_avDays_eachMouse_shfl,[],2)/sqrt(size(inh_avDays_eachMouse_shfl,2)), 'cmap', colss(2,:), 'alpha');
    h13 = boundedline(time_al, nanmean(uns_avDays_eachMouse_shfl,2), nanstd(uns_avDays_eachMouse_shfl,[],2)/sqrt(size(uns_avDays_eachMouse_shfl,2)), 'cmap', [.2,.2,.2], 'alpha');
end
a = get(gca, 'ylim');    
plot(time_al, hh0_allm*(a(2)-.05*diff(a)), 'k.')    
plot([0,0],a,'k:')
b = get(gca, 'xlim');
if ~isempty(yy)
    plot(b, [yy,yy],'k:')
end
legend([h1,h2,h3], {'Excitatory', 'Inhibitory', 'Unsure'}, 'position', [0.1347    0.8177    0.1414    0.0901]);
xlabel('Time since choice onset (ms)')
ylab = simpleTokenize(namc, '_'); ylab = ylab{1}; %namc;   
ylabel(ylab)
% title('mean+se mice (Ns aved per day;days aved per mouse)')
xlim([time_al(1)-50, time_al(end)+50])

% figs_adj_poster_ax(fh)

if savefigs
    fdn = fullfile(dirn00, strcat(namc,'_','ROC_curr_chAl_excInhUns_timeCourse_aveSeMice_aveDays_aveNs_', dm0, nowStr));
    savefig(fh, fdn)
    print(fh, '-dpdf', fdn)
end
        

%% Plot average and se of pooled days (of all mice; they are already averaged across neurons for each day)

fh = figure; hold on
h1 = boundedline(time_al, nanmean(exc_allDaysPooled_allMice,2), nanstd(exc_allDaysPooled_allMice,[],2)/sqrt(sum(numDaysGood)), 'b', 'alpha');
h2 = boundedline(time_al, nanmean(inh_allDaysPooled_allMice,2), nanstd(inh_allDaysPooled_allMice,[],2)/sqrt(sum(numDaysGood)), 'r', 'alpha');
h3 = boundedline(time_al, nanmean(uns_allDaysPooled_allMice,2), nanstd(uns_allDaysPooled_allMice,[],2)/sqrt(sum(numDaysGood)), 'g', 'alpha');
if doshfl % shfl
    h11 = boundedline(time_al, nanmean(exc_allDaysPooled_allMice_shfl,2), nanstd(exc_allDaysPooled_allMice_shfl,[],2)/sqrt(sum(numDaysGood)), 'cmap', colss(1,:), 'alpha');
    h12 = boundedline(time_al, nanmean(inh_allDaysPooled_allMice_shfl,2), nanstd(inh_allDaysPooled_allMice_shfl,[],2)/sqrt(sum(numDaysGood)), 'cmap', colss(2,:), 'alpha');
    h13 = boundedline(time_al, nanmean(uns_allDaysPooled_allMice_shfl,2), nanstd(uns_allDaysPooled_allMice_shfl,[],2)/sqrt(sum(numDaysGood)), 'cmap', [.2,.2,.2], 'alpha');
end
a = get(gca, 'ylim');    
plot(time_al, hh_allm*(a(2)-.05*diff(a)), 'k.')    
plot([0,0],a,'k:')
b = get(gca, 'xlim');
if ~isempty(yy)
    plot(b, [yy,yy],'k:')
end
legend([h1,h2,h3], {'Excitatory', 'Inhibitory','Unsure'}, 'position', [0.1347    0.8177    0.1414    0.0901]);
xlabel('Time since choice onset (ms)')
ylab = simpleTokenize(namc, '_'); ylab = ylab{1}; %namc;   
ylabel(ylab)
% title('mean+se pooled days (Ns aved per day)')

% figs_adj_poster_ax(fh)

if savefigs
    fdn = fullfile(dirn00, strcat(namc,'_','ROC_curr_chAl_excInhUns_timeCourse_aveSePooledDaysAllMice_aveNs_', dm0, nowStr));
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
    y3 = uns_allNsDaysMicePooled(ifr,:);
    [~,~,~,~,h1,h3] = plotHist_sp(y1,y3,xlab,ylab,leg, {'b','g'}, tit, fign, ha(ifr), yy, documsum, numBins);
    [bins,~,~,~,~,h2] = plotHist_sp(y1,y2,xlab,ylab,leg, cols, tit, fign, ha(ifr), yy, documsum, numBins);
        
    if doshfl % sfhl
        % average of shfl samples
%         y1 = exc_allNsDaysMicePooled_shfl(ifr,:);
%         y2 = inh_allNsDaysMicePooled_shfl(ifr,:);   
%         plotHist_sp(y1,y2,xlab,ylab,leg, colss, tit, fign, ha(ifr), yy, documsum, numBins, bins)
        
        % individual shlf sampls
        y1s = exc_allNsDaysMicePooled_shfl0(ifr,:,:);
        y2s = inh_allNsDaysMicePooled_shfl0(ifr,:,:);
        y3s = uns_allNsDaysMicePooled_shfl0(ifr,:,:);
        % pool all samples
        y1s = y1s(:);
        y2s = y2s(:);
        y3s = y3s(:);
        plotHist_sp(y1s,y3s,xlab,ylab,leg, [0    0.8000    0.8000; .2,.2,.2], tit, fign, ha(ifr), yy, documsum, numBins, bins);
        plotHist_sp(y1s,y2s,xlab,ylab,leg, colss, tit, fign, ha(ifr), yy, documsum, numBins, bins);
    end
    
    legend(ha(1), [h1,h2,h3], {'Exc','Inh','Unsure'})

    if ifr==1
        xlabel(ha(ifr), xlab)
        ylabel(ha(ifr), ylab)
        legend(ha(ifr), leg)
    end
end

if savefigs        
    fdn = fullfile(dirn00, strcat(namc,'_','ROC_curr_chAl_excInhUns_dist_eachFr_NsDaysMicePooled_', dm0, nowStr));
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
y3 = uns_allNsDaysMicePooled(fr,:);
[~,~,~,~,~,h1,h3,hsp] = plotHist(y1,y3,xlab,ylab,leg, {'b','g'}, yy, fign, nBins, doSmooth); %, lineStyles, sp, bins); 
[~,~,~,~,~,~,h2,hsp] = plotHist(y1,y2,xlab,ylab,leg, cols, yy, fign, nBins, doSmooth); %, lineStyles, sp, bins); 
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
    y3s = uns_allNsDaysMicePooled_shfl0(fr,:,:);
    % pool all samples
    y1s = y1s(:);
    y2s = y2s(:);
    y3s = y3s(:);
    plotHist(y1s,y3s,xlab,ylab,leg, colss, yy,fign, nBins, doSmooth);
    plotHist(y1s,y2s,xlab,ylab,leg, colss, yy,fign, nBins, doSmooth);
end

legend(hsp, [h1,h2,h3], {'Exc','Inh','Unsure'})

if savefigs        
    fdn = fullfile(dirn00, strcat(namc,'_','ROC_curr_chAl_excInhUns_dist_time-1_NsDaysMicePooled_', dm0, nowStr));
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

uav = uns_avDays_eachMouse(fr,:);
use = uns_seDays_eachMouse(fr,:);

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

    uavs = uns_avDays_eachMouse_shfl(fr,:);
    uses = uns_seDays_eachMouse_shfl(fr,:);
end


%%%%%%%%%%%%%%% Plot
g = 0;
fign = figure('position',[60   593   450   320]); 

subplot(121), hold on;
errorbar(1:length(mice), uav, use, 'g', 'linestyle','none', 'marker','o')
errorbar(1:length(mice), eav, ese, 'b', 'linestyle','none', 'marker','o')
errorbar((1:length(mice))+g, iav, ise ,'r', 'linestyle','none', 'marker','o')
if doshfl % shfl
    errorbar(1:length(mice), uavs, uses, 'color', [.2,.2,.2], 'linestyle','none', 'marker','o')
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
legend('unsure','exc', 'inh')


%%%%%%%%%%%%%%%%%%%%%%%% show ave across pooled neurons of all days for each mouse %%%%%%%%%%%%%%%%%%%%%%%%

% number of valid neurons (from all days) for each mouse
exc_nValidNs = nan(1,length(mice));
inh_nValidNs = nan(1,length(mice));
uns_nValidNs = nan(1,length(mice));
for im=1:length(mice)
    exc_nValidNs(im) = sum(~isnan(exc_allNsDaysPooled_eachMouse{im}(1,:)));
    inh_nValidNs(im) = sum(~isnan(inh_allNsDaysPooled_eachMouse{im}(1,:)));
    uns_nValidNs(im) = sum(~isnan(uns_allNsDaysPooled_eachMouse{im}(1,:)));
end

eav = cellfun(@(x)nanmean(x(fr,:),2), exc_allNsDaysPooled_eachMouse);
ese = cellfun(@(x)nanstd(x(fr,:),[],2), exc_allNsDaysPooled_eachMouse)/sqrt(exc_nValidNs(im));

iav = cellfun(@(x)nanmean(x(fr,:),2), inh_allNsDaysPooled_eachMouse);
ise = cellfun(@(x)nanstd(x(fr,:),[],2), inh_allNsDaysPooled_eachMouse)/sqrt(inh_nValidNs(im));

uav = cellfun(@(x)nanmean(x(fr,:),2), uns_allNsDaysPooled_eachMouse);
use = cellfun(@(x)nanstd(x(fr,:),[],2), uns_allNsDaysPooled_eachMouse)/sqrt(uns_nValidNs(im));

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

    uavs = cellfun(@(x)nanmean(x(fr,:),2), uns_allNsDaysPooled_eachMouse_shfl);
    uses = cellfun(@(x)nanstd(x(fr,:),[],2), uns_allNsDaysPooled_eachMouse_shfl)/sqrt(uns_nValidNs(im));
end


%%%%% Plot
g = 0;
% figure('position',[60   593   239   320]); 

subplot(122), hold on;
errorbar(1:length(mice), uav, use, 'g', 'linestyle','none', 'marker','o')
errorbar(1:length(mice), eav, ese, 'b', 'linestyle','none', 'marker','o')
errorbar((1:length(mice))+g, iav, ise ,'r', 'linestyle','none', 'marker','o')
if doshfl  % shfl
    errorbar(1:length(mice), uavs, uses, 'color', [.2,.2,.2], 'linestyle','none', 'marker','o')
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
    fdn = fullfile(dirn00, strcat(namc,'_','ROC_curr_chAl_excInhUns_aveSe_time-1_eachMouse_', dm0, nowStr));
    savefig(fign, fdn)    
    print(fign, '-dpdf', fdn)
end 



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Fractions of significantly choice-tuned neurons %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% Average and se across days %%%%%%%%%%%%%%%%%%%%

%% Fract significantly choice tuned neurons, compare between exc and inh

exc_fractSigTuned_eachDay_avDays = cellfun(@nanmean, exc_fractSigTuned_eachDay);
exc_fractSigTuned_eachDay_seDays = cellfun(@nanstd, exc_fractSigTuned_eachDay) ./ sqrt(numDaysGood);

inh_fractSigTuned_eachDay_avDays = cellfun(@nanmean, inh_fractSigTuned_eachDay);
inh_fractSigTuned_eachDay_seDays = cellfun(@nanstd, inh_fractSigTuned_eachDay) ./ sqrt(numDaysGood);

uns_fractSigTuned_eachDay_avDays = cellfun(@nanmean, uns_fractSigTuned_eachDay);
uns_fractSigTuned_eachDay_seDays = cellfun(@nanstd, uns_fractSigTuned_eachDay) ./ sqrt(numDaysGood);

p_allM = nan(1,length(mice));
for im=1:length(mice)
    [~,p_allM(im)] = ttest2(exc_fractSigTuned_eachDay{im} , inh_fractSigTuned_eachDay{im});
end

x = (1:length(mice))';
gp = .1; 

% all preferences (ipsi and contra)
fign = figure('position', [17   714   207   251]);

y = [exc_fractSigTuned_eachDay_avDays; inh_fractSigTuned_eachDay_avDays; uns_fractSigTuned_eachDay_avDays]';
ye = [exc_fractSigTuned_eachDay_seDays; inh_fractSigTuned_eachDay_seDays; uns_fractSigTuned_eachDay_seDays]';

b = errorbar([x, x+gp, x+2*gp], y, ye, 'linestyle', 'none', 'marker', '.', 'markersize', 9);
b(1).Color = 'b'; %'b';
b(2).Color = 'r'; %rgb('lightblue');
b(3).Color = 'g';
set(gca,'xtick',x+gp/2)
set(gca,'xticklabel',x)
xlabel('Mice')
ylabel('Fraction choice-tuned') % significant
title('All neurons')
set(gca, 'tickdir', 'out', 'box', 'off')
xlim([.5,4.5])
% mark mice with sig diff btwn exc and inh
hold on
yl = get(gca,'ylim');
plot(find(p_allM<=.05)+gp/2, yl(2)-range(diff(yl))/20, 'k*')

legend(b, 'exc', 'inh', 'unsure')

% save figure
if savefigs        
    fdn = fullfile(dirn00, strcat(namc,'_','fractSigTunedOfAllN_aveSeDays_time-1_ROC_curr_chAl_excInhUns_', dm0, nowStr));
    savefig(fign, fdn)    
    print(fign, '-dpdf', fdn)
end 


%% In significantly choice selective neurons, what fraction are ipsi and what fraction are contra, show it separately for exc and inh

if doChoicePref==0
    % set average and se across days
    exc_fractSigIpsiTuned_eachDay_avDays = cellfun(@nanmean, exc_fractSigIpsiTuned_eachDay);
    exc_fractSigContraTuned_eachDay_avDays = cellfun(@nanmean, exc_fractSigContraTuned_eachDay); % this is 1 - exc_fractSigIpsiTuned_eachDay_avDays
    exc_fractSigIpsiTuned_eachDay_seDays = cellfun(@nanstd, exc_fractSigIpsiTuned_eachDay) ./ sqrt(numDaysGood);
    exc_fractSigContraTuned_eachDay_seDays = cellfun(@nanstd, exc_fractSigContraTuned_eachDay) ./ sqrt(numDaysGood);

    inh_fractSigIpsiTuned_eachDay_avDays = cellfun(@nanmean, inh_fractSigIpsiTuned_eachDay);
    inh_fractSigContraTuned_eachDay_avDays = cellfun(@nanmean, inh_fractSigContraTuned_eachDay); % this is 1 - inh_fractSigIpsiTuned_eachDay_avDays
    inh_fractSigIpsiTuned_eachDay_seDays = cellfun(@nanstd, inh_fractSigIpsiTuned_eachDay) ./ sqrt(numDaysGood);
    inh_fractSigContraTuned_eachDay_seDays = cellfun(@nanstd, inh_fractSigContraTuned_eachDay) ./ sqrt(numDaysGood);

    uns_fractSigIpsiTuned_eachDay_avDays = cellfun(@nanmean, uns_fractSigIpsiTuned_eachDay);
    uns_fractSigContraTuned_eachDay_avDays = cellfun(@nanmean, uns_fractSigContraTuned_eachDay); % this is 1 - uns_fractSigIpsiTuned_eachDay_avDays
    uns_fractSigIpsiTuned_eachDay_seDays = cellfun(@nanstd, uns_fractSigIpsiTuned_eachDay) ./ sqrt(numDaysGood);
    uns_fractSigContraTuned_eachDay_seDays = cellfun(@nanstd, uns_fractSigContraTuned_eachDay) ./ sqrt(numDaysGood);

    x = (1:length(mice))';
    gp = .25; 

    fign = figure('position', [41   483   385   470]);

    % excitatory neurons
    y = [exc_fractSigIpsiTuned_eachDay_avDays; exc_fractSigContraTuned_eachDay_avDays]';
    ye = [exc_fractSigIpsiTuned_eachDay_seDays; exc_fractSigContraTuned_eachDay_seDays]';
    subplot(222)
    b = errorbar([x,x+gp], y, ye, 'linestyle', 'none', 'marker', '.', 'markersize', 9);
    b(1).Color = 'k'; %'b';
    b(2).Color = 'g'; %rgb('lightblue');
    set(gca,'xtick',x+gp/2)
    set(gca,'xticklabel',x)
    xlabel('Mice')
    ylabel('Fraction choice-tuned') % significant
    legend({'Ipsi-pref', 'Contra-pref'})%, 'position', [0.1655    0.7861    0.2786    0.0821])
    set(gca, 'tickdir', 'out', 'box', 'off')
    title('Excitatory')
    xlim([.5,4.5])

    % inhibitory neurons
    y = [inh_fractSigIpsiTuned_eachDay_avDays; inh_fractSigContraTuned_eachDay_avDays]';
    ye = [inh_fractSigIpsiTuned_eachDay_seDays; inh_fractSigContraTuned_eachDay_seDays]';
    subplot(224)
    b = errorbar([x,x+gp], y, ye, 'linestyle', 'none', 'marker', '.', 'markersize', 9);
    b(1).Color = 'k'; %'r';
    b(2).Color = 'g'; %rgb('lightsalmon');
    set(gca,'xtick',x+gp/2)
    set(gca,'xticklabel',x)
    xlabel('Mice')
    ylabel('Fraction choice-tuned') % significant
    % legend({'Ipsi-pref', 'Contra-pref'}) %, 'position', [0.1655    0.2861    0.2786    0.0821])
    set(gca, 'tickdir', 'out', 'box', 'off')
    title('Inhibitory')
    xlim([.5,4.5])


    % ipsi-selective
    y = [exc_fractSigIpsiTuned_eachDay_avDays; inh_fractSigIpsiTuned_eachDay_avDays; uns_fractSigIpsiTuned_eachDay_avDays]';
    ye = [exc_fractSigIpsiTuned_eachDay_seDays; inh_fractSigIpsiTuned_eachDay_seDays; uns_fractSigIpsiTuned_eachDay_seDays]';
    subplot(221); hold on
    b = errorbar([x,x+gp,x+2*gp], y, ye, 'linestyle', 'none', 'marker', '.', 'markersize', 9);
    b(1).Color = 'b'; % ipsi
    b(2).Color = 'r'; % ipsi
    b(3).Color = 'g'; % ipsi
    set(gca,'xtick',x+gp/2)
    set(gca,'xticklabel',x)
    xlabel('Mice')
    ylabel('Fraction choice-tuned') % significant
    legend({'Exc', 'Inh', 'Unsure'})%, 'position', [0.1655    0.2861    0.2786    0.0821])
    set(gca, 'tickdir', 'out', 'box', 'off')
    title('Ipsi-selective')
    xlim([.5,4.5])


    % contra-selective
    y = [exc_fractSigContraTuned_eachDay_avDays; inh_fractSigContraTuned_eachDay_avDays; uns_fractSigContraTuned_eachDay_avDays]';
    ye = [exc_fractSigContraTuned_eachDay_seDays; inh_fractSigContraTuned_eachDay_seDays; uns_fractSigContraTuned_eachDay_seDays]';
    subplot(223); hold on
    b = errorbar([x,x+gp,x+2*gp], y, ye, 'linestyle', 'none', 'marker', '.', 'markersize', 9);
    b(1).Color = 'b'; %rgb('lightblue'); % contra
    b(2).Color = 'r'; %rgb('lightsalmon'); % contra
    b(3).Color = 'g';
    set(gca,'xtick',x+gp/2)
    set(gca,'xticklabel',x)
    xlabel('Mice')
    ylabel('Fraction choice-tuned') % significant
    % legend({'Exc', 'Inh'})%, 'position', [0.1655    0.2861    0.2786    0.0821])
    set(gca, 'tickdir', 'out', 'box', 'off')
    title('Contra-selective')
    xlim([.5,4.5])


    % save figure
    if savefigs        
        fdn = fullfile(dirn00, strcat(namc,'_','fractIpsiContraOfSigTuned_aveSeDays_time-1_ROC_curr_chAl_excInhUns_', dm0, nowStr));
        savefig(fign, fdn)    
        print(fign, '-dpdf', fdn)
    end 

end




%%
%%%%%%%%%%%%%%%%%%%%%%% Same as above, but vars were computed on pooled
%%%%%%%%%%%%%%%%%%%%%%% days... so no error bar for sd across days !! 

%% Plot fractions of significantly choice-tuned neurons, compare between exc and inh

%%% what fraction of neurons are significant... all neurons, neurons with AUC>.5 and neurons with AUC<.5

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
    fdn = fullfile(dirn00, strcat(namc,'_','fractSigTunedOfAllN_time-1_ROC_curr_chAl_excInh_', dm0, nowStr));
    savefig(fign, fdn)    
    print(fign, '-dpdf', fdn)
end 


%% In significantly choice selective neurons, what fraction are ipsi and what fraction are contra, show it separately for exc and inh

fign = figure;

x = 1:length(mice);
% excitatory neurons
y = [exc_fractSigIpsiTuned; exc_fractSigContraTuned]';
subplot(222)
b = bar(x, y);
b(1).FaceColor = 'b';
b(2).FaceColor = rgb('lightblue');
b(1).EdgeColor = 'none';
b(2).EdgeColor = 'none';
xlabel('Mice')
ylabel('Fraction choice-tuned') % significant
legend({'Ipsi-pref', 'Contra-pref'})%, 'position', [0.1655    0.7861    0.2786    0.0821])
set(gca, 'tickdir', 'out', 'box', 'off')
title('Excitatory')


% inhibitory neurons
y = [inh_fractSigIpsiTuned; inh_fractSigContraTuned]';
subplot(224)
b = bar(x, y);
b(1).FaceColor = 'r';
b(2).FaceColor = rgb('lightsalmon');
b(1).EdgeColor = 'none';
b(2).EdgeColor = 'none';
xlabel('Mice')
ylabel('Fraction choice-tuned') % significant
legend({'Ipsi-pref', 'Contra-pref'}) %, 'position', [0.1655    0.2861    0.2786    0.0821])
set(gca, 'tickdir', 'out', 'box', 'off')
title('Inhibitory')


% ipsi-selective
y = [exc_fractSigIpsiTuned; inh_fractSigIpsiTuned]';
subplot(221); hold on
b = bar(x, y);
b(1).FaceColor = 'b'; % ipsi
b(2).FaceColor = 'r'; % ipsi
b(1).EdgeColor = 'none';
b(2).EdgeColor = 'none';
xlabel('Mice')
ylabel('Fraction choice-tuned') % significant
legend({'Exc', 'Inh'})%, 'position', [0.1655    0.2861    0.2786    0.0821])
set(gca, 'tickdir', 'out', 'box', 'off')
title('Ipsi-selective')


% contra-selective
y = [exc_fractSigContraTuned; inh_fractSigContraTuned]';
subplot(223); hold on
b = bar(x, y);
b(1).FaceColor = rgb('lightblue'); % contra
b(2).FaceColor = rgb('lightsalmon'); % contra
b(1).EdgeColor = 'none';
b(2).EdgeColor = 'none';
xlabel('Mice')
ylabel('Fraction choice-tuned') % significant
legend({'Exc', 'Inh'})%, 'position', [0.1655    0.2861    0.2786    0.0821])
set(gca, 'tickdir', 'out', 'box', 'off')
title('Contra-selective')

% y = [exc_fractSigIpsiTuned; exc_fractSigContraTuned; inh_fractSigIpsiTuned; inh_fractSigContraTuned]';


% save figure
if savefigs        
    fdn = fullfile(dirn00, strcat(namc,'_','fractIpsiContraOfSigTuned_time-1_ROC_curr_chAl_excInh_', dm0, nowStr));
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






%% Learing figure: All mice: Plot fraction of choice selective neurons per day

figure; 
for im = 1:length(mice)
    e = exc_fractSigTuned_eachDay{im};
    e(isnan(e)) = [];
    
    i = inh_fractSigTuned_eachDay{im};
    i(isnan(i)) = [];

    u = uns_fractSigTuned_eachDay{im};
    u(isnan(u)) = [];
    
    subplot(2,2,im); hold on
    plot(e, 'b')
    plot(i, 'r')
    plot(u, 'g')
    xlim([1, length(e)])
    
    set(gca,'tickdir','out', 'box','off')
    title(mouse)
    xlabel('Days')
    if im==1
        ylabel('Fraction choice selective neurons')
    end
end

if savefigs
    savefig(gcf, fullfile(dirn00, [namc,'_','ROC_curr_chAl_excInhUns_trainindDays_fractSigTunedOfAllN_', nowStr,'.fig']))
    print(gcf, '-dpdf', fullfile(dirn00, [namc,'_','ROC_curr_chAl_excInhUns_trainingDays_heatmaps_aveNeurs_', nowStr]))
end
    

%% Learning figure: All mice: compare fraction of choice selective neurons for low vs high behavioral performance days

perc_thb = [20,80]; %[10,90]; % # perc_thb = [15,85] # percentiles of behavioral performance for determining low and high performance.

x = 1:3;
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

    
    %%%%%%%%%%%%%%%%%%%
    aa = exc_fractSigTuned_eachDay{im}(days2an_heatmap);
    bb = inh_fractSigTuned_eachDay{im}(days2an_heatmap);
    cc = allN_fractSigTuned_eachDay{im}(days2an_heatmap);

    % set corrs for low and high behavioral performance days
    a = aa(loBehCorrDays); 
    b = bb(loBehCorrDays); 
    c = cc(loBehCorrDays); 
    ah = aa(hiBehCorrDays); % EE
    bh = bb(hiBehCorrDays); % II
    ch = cc(hiBehCorrDays); % II

    % set se of corrs across low and high beh perform days
    as0 = nanstd(a) / sqrt(sum(~isnan(a)));
    bs0 = nanstd(b) / sqrt(sum(~isnan(b)));
    cs0 = nanstd(c) / sqrt(sum(~isnan(c)));
    ahs = nanstd(ah)/ sqrt(sum(~isnan(ah)));
    bhs = nanstd(bh)/ sqrt(sum(~isnan(bh)));
    chs = nanstd(ch)/ sqrt(sum(~isnan(ch)));

%         figure; %('name', 'All mice', 'position', [14   636   661   290]); 
%         set(gca, 'position', [0.2919    0.1908    0.5229    0.7095])

%     disp([ttest(a, ah), ttest(b, bh)])
    
    
    %%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%% 
    %%%%%%%%%%%%%%%%%%%%% PLOT %%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%        
    subplot(1,length(mice),im); 
%         subplot(248)
    hold on

    % E
    h11 = errorbar(x(1), nanmean(ah), ahs, 'b.', 'linestyle', 'none', 'markersize', 10);
    h12 = errorbar(x(1)+gp, nanmean(a), as0, 'color', rgb('lightblue'), 'marker','.', 'linestyle', 'none', 'markersize', 10);
    % mark sig (low vs high beh)
%     yl = get(gca, 'ylim');    
    yl = [min(nanmean(ah)-ahs, nanmean(bh)-bhs) ,  max(nanmean(ah)+ahs, nanmean(bh)+bhs)];
    at = ttest2(a, ah);
    at(at==0) = nan;
    ylv = at * yl(2)+range(yl)/5;
    plot(x(1), ylv, 'k*'); 
    
    % I
    h11 = errorbar(x(2), nanmean(bh), bhs, 'r.', 'linestyle', 'none', 'markersize', 10);
    h12 = errorbar(x(2)+gp, nanmean(b), bs0, 'color', rgb('lightsalmon'), 'marker','.', 'linestyle', 'none', 'markersize', 10);
    % mark sig (low vs high beh)
    at = ttest2(b, bh);
    at(at==0) = nan;
    ylv = at * yl(2)+range(yl)/5;
    plot(x(2), ylv, 'k*'); 
    
    % allN
    h11 = errorbar(x(3), nanmean(ch), chs, 'k.', 'linestyle', 'none', 'markersize', 10);
    h12 = errorbar(x(3)+gp, nanmean(c), cs0, 'color', rgb('gray'), 'marker','.', 'linestyle', 'none', 'markersize', 10);
    % mark sig (low vs high beh)
    at = ttest2(c, ch);
    at(at==0) = nan;
    ylv = at * yl(2)+range(yl)/5;
    plot(x(3), ylv, 'k*'); 
    
    
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp)
    set(gca,'xticklabel', {'Exc','Inh', 'All'})
    ylabel('Fract choice selective')
    legend([h11,h12], {'highBeh','lowBeh'}, 'location', 'northoutside')
    set(gca, 'tickdir', 'out')
    title(mouse)

end


if savefigs
    savefig(gcf, fullfile(dirn00, [namc,'_','ROC_curr_chAl_excInh_trainindDays_fractSigTunedOfAllN_earlyLateDays_', nowStr,'.fig']))
    print(gcf, '-dpdf', fullfile(dirn00, [namc,'_','ROC_curr_chAl_excInh_trainindDays_fractSigTunedOfAllN_earlyLateDays_', nowStr]))
end






%% Compare choice selectivity for low vs high behavioral performance days

onlySig = 0; % use average AUC of only sig choice selective neurons
perc_thb = [20,80]; %[10,90]; % # perc_thb = [15,85] # percentiles of behavioral performance for determining low and high performance.

xlab = simpleTokenize(namc, '_'); xlab = xlab{1};
x = 1:3;
gp = .2;
marg = .2;
fnow = figure('name', 'All mice', 'position', [34   661   788   280]);

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
    
    
    %% 
    %%% Average AUC across neurons for each day and each frame
    %%% Pool AUC across all neurons of all days (for each frame)
    
    if ~onlySig
        aveexc = aveexc_allMice{im};
        aveinh = aveinh_allMice{im};
        aveallN = aveallN_allMice{im};
    end
    %{
    aveexc_shfl = aveexc_shfl_allMice{im};
    aveinh_shfl = aveinh_shfl_allMice{im};
    aveallN_shfl = aveallN_shfl_allMice{im};
    
    aveexc_shfl0 = aveexc_shfl0_allMice{im};
    aveinh_shfl0 = aveinh_shfl0_allMice{im};
    aveallN_shfl0 = aveallN_shfl0_allMice{im};
    
    seexc = seexc_allMice{im};
    seinh = seinh_allMice{im};
    seallN = seallN_allMice{im};
    
    seexc_shfl = seexc_shfl_allMice{im};
    seinh_shfl = seinh_shfl_allMice{im};
    seallN_shfl = seallN_shfl_allMice{im};
    %}
    

    % run ttest across days for each frame
    %{
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
    %}
    
    %% Learning figure: compare magnitude of choice selectivity at time bin -1 for low vs high behavioral performance days

    
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

    
    %%

    if onlySig % use average AUC of only sig choice selective neurons
        aa = choicePref_exc_onlySig_aveNs_allMice{im}(days2an_heatmap);
        bb = choicePref_inh_onlySig_aveNs_allMice{im}(days2an_heatmap);
        cc = choicePref_allN_onlySig_aveNs_allMice{im}(days2an_heatmap);
        figure; hold on; plot(aa, 'b'); plot(bb,'r'); plot(cc,'k')
    else
        aa = aveexc(nPreMin_allMice(im), days2an_heatmap);
        bb = aveinh(nPreMin_allMice(im), days2an_heatmap);
        cc = aveallN(nPreMin_allMice(im), days2an_heatmap);
    end
    
    
    % set corrs for low and high behavioral performance days
    a = aa(loBehCorrDays); 
    b = bb(loBehCorrDays); 
    c = cc(loBehCorrDays); 
    ah = aa(hiBehCorrDays); % E
    bh = bb(hiBehCorrDays); % I
    ch = cc(hiBehCorrDays); % allN

    % set se of corrs across low and high beh perform days
    as0 = nanstd(a) / sqrt(sum(~isnan(a)));
    bs0 = nanstd(b) / sqrt(sum(~isnan(b)));
    cs0 = nanstd(c) / sqrt(sum(~isnan(c)));
    ahs = nanstd(ah)/ sqrt(sum(~isnan(ah)));
    bhs = nanstd(bh)/ sqrt(sum(~isnan(bh)));
    chs = nanstd(ch)/ sqrt(sum(~isnan(ch)));

%         figure; %('name', 'All mice', 'position', [14   636   661   290]); 
%         set(gca, 'position', [0.2919    0.1908    0.5229    0.7095])

%     disp([ttest(a, ah), ttest(b, bh)])
    
    
    %%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%% 
    %%%%%%%%%%%%%%%%%%%%% PLOT %%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%           
    
    figure(fnow)
    subplot(1,length(mice),im); 
    hold on

    % E
    h11 = errorbar(x(1), nanmean(ah), ahs, 'b.', 'linestyle', 'none', 'markersize', 10);
    h12 = errorbar(x(1)+gp, nanmean(a), as0, 'color', rgb('lightblue'), 'marker','.', 'linestyle', 'none', 'markersize', 10);
    % mark sig (low vs high beh)
%     yl = get(gca, 'ylim');    
    yl = [min(nanmean(ah)-ahs, nanmean(bh)-bhs) ,  max(nanmean(ah)+ahs, nanmean(bh)+bhs)];
    at = ttest2(a, ah);
    at(at==0) = nan;
    ylv = at * yl(2)+range(yl)/5;
    plot(x(1), ylv, 'k*'); 
    
    % I
    h11 = errorbar(x(2), nanmean(bh), bhs, 'r.', 'linestyle', 'none', 'markersize', 10);
    h12 = errorbar(x(2)+gp, nanmean(b), bs0, 'color', rgb('lightsalmon'), 'marker','.', 'linestyle', 'none', 'markersize', 10);
    % mark sig (low vs high beh)
    at = ttest2(b, bh);
    at(at==0) = nan;
    ylv = at * yl(2)+range(yl)/5;
    plot(x(2), ylv, 'k*'); 
    
    % AllN
    h11 = errorbar(x(3), nanmean(ch), chs, 'k.', 'linestyle', 'none', 'markersize', 10);
    h12 = errorbar(x(3)+gp, nanmean(c), cs0, 'color', rgb('gray'), 'marker','.', 'linestyle', 'none', 'markersize', 10);
    % mark sig (low vs high beh)
    at = ttest2(c, ch);
    at(at==0) = nan;
    ylv = at * yl(2)+range(yl)/5;
    plot(x(3), ylv, 'k*'); 
    
    
    xlim([x(1)-marg, x(end)+gp+marg])
    set(gca,'xtick', x+gp)
    set(gca,'xticklabel', {'Exc','Inh', 'All'})
    ylabel(xlab)
    legend([h11,h12], {'highBeh','lowBeh'}, 'location', 'northoutside')
    set(gca, 'tickdir', 'out')
    title(mouse)

    
end

if savefigs
    if onlySig
        osig = 'aveNeursOnlySig';
    else
        osig = 'aveNeurs';
    end
    
    savefig(gcf, fullfile(dirn00, [namc,'_','ROC_curr_chAl_excInh_trainindDays_',osig,'_earlyLateDays_', nowStr,'.fig']))
    print(gcf, '-dpdf', fullfile(dirn00, [namc,'_','ROC_curr_chAl_excInh_trainindDays_',osig,'_earlyLateDays_', nowStr]))
end


% Note: with training, the fraction of choice selective neurons goes up,
% but surprisingly the magnitude of choice selectivity goes down.... my
% concern is that in earlier days the AUC of shuffled dist is also higher
% than in later days (I think because earelier days have fewere trials,
% their AUC estimate is noisier), so even if I only take the sig neurons to
% compute magnitude of choice selectivity, it is not a surprise that in
% early days the AUC magnitude is higher, just how the magnitude of shuffle
% AUC is higher... so I think it makes more sense to only use the fraction
% of significantly choice selective neurons, and not their magnitude when
% comparing early vs late days (and the effect of training on single
% neurons)!









