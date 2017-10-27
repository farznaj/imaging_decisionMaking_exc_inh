%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Summary plots of all mouse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

savefigs = sumMice_do_savefigs(2);
cols = {'b','r'}; % exc,inh, real data
colss = [0,.8,.8; .8,.5,.8]; % exc,inh colors for shuffled data
nowStr = nowStr_allMice{imfni18}; % use nowStr of mouse fni18 (in case its day 4 was removed, you want that to be indicated in the figure name of summary of all mice).


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Plot average timecourse of ROC across mice %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Plot average and se across mice (ROC is already averaged across days (and neurons))

fh = figure; hold on
h1 = boundedline(time_al, nanmean(exc_avDays_eachMouse,2), nanstd(exc_avDays_eachMouse,[],2)/sqrt(size(exc_avDays_eachMouse,2)), 'b', 'alpha');
h2 = boundedline(time_al, nanmean(inh_avDays_eachMouse,2), nanstd(inh_avDays_eachMouse,[],2)/sqrt(size(inh_avDays_eachMouse,2)), 'r', 'alpha');
% shfl
h11 = boundedline(time_al, nanmean(exc_avDays_eachMouse_shfl,2), nanstd(exc_avDays_eachMouse_shfl,[],2)/sqrt(size(exc_avDays_eachMouse_shfl,2)), 'cmap', colss(1,:), 'alpha');
h12 = boundedline(time_al, nanmean(inh_avDays_eachMouse_shfl,2), nanstd(inh_avDays_eachMouse_shfl,[],2)/sqrt(size(inh_avDays_eachMouse_shfl,2)), 'cmap', colss(2,:), 'alpha');
a = get(gca, 'ylim');    
plot(time_al, hh0_allm*(a(2)-.05*diff(a)), 'k.')    
plot([0,0],a,'k:')
b = get(gca, 'xlim');
if ~isempty(yy)
    plot(b, [yy,yy],'k:')
end
legend([h1,h2], {'Excitatory', 'Inhibitory'}, 'position', [0.1347    0.8177    0.1414    0.0901]);
xlabel('Time since choice onset (ms)')
ylabel(namc)
% title('mean+se mice (Ns aved per day;days aved per mouse)')
xlim([time_al(1)-50, time_al(end)+50])

figs_adj_poster_ax

if savefigs
    fdn = fullfile(dirn0, strcat(namc,'_','ROC_curr_chAl_excInh_timeCourse_aveMice_aveDays_aveNs_', dm0, nowStr));
    savefig(fh, fdn)
    print(fh, '-dpdf', fdn)
end
        

%% Plot average and se of pooled days (of all mice; they are already averaged across neurons for each day)

fh = figure; hold on
h1 = boundedline(time_al, nanmean(exc_allDaysPooled_allMice,2), nanstd(exc_allDaysPooled_allMice,[],2)/sqrt(sum(nGoodDays)), 'b', 'alpha');
h2 = boundedline(time_al, nanmean(inh_allDaysPooled_allMice,2), nanstd(inh_allDaysPooled_allMice,[],2)/sqrt(sum(nGoodDays)), 'r', 'alpha');
% shfl
h11 = boundedline(time_al, nanmean(exc_allDaysPooled_allMice_shfl,2), nanstd(exc_allDaysPooled_allMice_shfl,[],2)/sqrt(sum(nGoodDays)), 'cmap', colss(1,:), 'alpha');
h12 = boundedline(time_al, nanmean(inh_allDaysPooled_allMice_shfl,2), nanstd(inh_allDaysPooled_allMice_shfl,[],2)/sqrt(sum(nGoodDays)), 'cmap', colss(2,:), 'alpha');
a = get(gca, 'ylim');    
plot(time_al, hh_allm*(a(2)-.05*diff(a)), 'k.')    
plot([0,0],a,'k:')
b = get(gca, 'xlim');
if ~isempty(yy)
    plot(b, [yy,yy],'k:')
end
legend([h1,h2], {'Excitatory', 'Inhibitory'}, 'position', [0.1347    0.8177    0.1414    0.0901]);
xlabel('Time since choice onset (ms)')
ylabel(namc)
% title('mean+se pooled days (Ns aved per day)')

figs_adj_poster_ax

if savefigs
    fdn = fullfile(dirn0, strcat(namc,'_','ROC_curr_chAl_excInh_timeCourse_avePooledDaysAllMice_aveNs_', dm0, nowStr));
    savefig(fh, fdn)
    print(fh, '-dpdf', fdn)
end
        

%%
%%%%%%%%%%%%%%%%%%%%%%% Dists of individual neurons %%%%%%%%%%%%%%%%%%%%%%%

%% Compare exc/inh ROC dist for each frame, all days pooled

documsum = 0;
xlab = namc;
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
    plotHist_sp(y1,y2,xlab,ylab,leg, cols, tit, fign, ha(ifr), yy, documsum)
        
    % sfhl
    y1 = exc_allNsDaysMicePooled_shfl(ifr,:);
    y2 = inh_allNsDaysMicePooled_shfl(ifr,:);   
    plotHist_sp(y1,y2,xlab,ylab,leg, colss, tit, fign, ha(ifr), yy, documsum)

    if ifr==1
        xlabel(ha(ifr), xlab)
        ylabel(ha(ifr), ylab)
        legend(ha(ifr), leg)
    end
end

if savefigs        
    fdn = fullfile(dirn0, strcat(namc,'_','ROC_curr_chAl_excInh_dist_eachFr_NsDaysMicePooled_', dm0, nowStr));
    savefig(fign, fdn)    
    print(fign, '-dpdf', fdn)
end   


%% Get dist of roc for exc and inh at time bin -1

time2an = -1;

% frame before choice on the mouse-aligned traces
e = nPreMin+1;
fr = e+time2an;

fign = figure; % f = nan(1,1); f(1) = axes();

y1 = exc_allNsDaysMicePooled(fr,:);
y2 = inh_allNsDaysMicePooled(fr,:);
plotHist(y1,y2,xlab,ylab,leg, cols, yy,fign);
% shfl
y1s = exc_allNsDaysMicePooled_shfl(fr,:);
y2s = inh_allNsDaysMicePooled_shfl(fr,:);
plotHist(y1s,y2s,xlab,ylab,leg, colss, yy,fign);


if savefigs        
    fdn = fullfile(dirn0, strcat(namc,'_','ROC_curr_chAl_excInh_dist_time-1_NsDaysMicePooled_', dm0, nowStr));
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


% shfl
eavs = exc_avDays_eachMouse_shfl(fr,:);
eses = exc_seDays_eachMouse_shfl(fr,:);

iavs = inh_avDays_eachMouse_shfl(fr,:);
ises = inh_seDays_eachMouse_shfl(fr,:);


%%%%%%%%%%%%%%% Plot
g = 0;
fign = figure('position',[60   593   450   320]); 

subplot(121), hold on;
errorbar(1:length(mice), eav, ese, 'b', 'linestyle','none', 'marker','o')
errorbar((1:length(mice))+g, iav, ise ,'r', 'linestyle','none', 'marker','o')
% shfl
errorbar(1:length(mice), eavs, eses, 'color', colss(1,:), 'linestyle','none', 'marker','o')
errorbar((1:length(mice))+g, iavs, ises ,'color', colss(2,:), 'linestyle','none', 'marker','o')

xlim([.5, length(mice)+.5])
xlabel('Mice')
ylabel(namc)
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


% shfl
eavs = cellfun(@(x)nanmean(x(fr,:),2), exc_allNsDaysPooled_eachMouse_shfl);
eses = cellfun(@(x)nanstd(x(fr,:),[],2), exc_allNsDaysPooled_eachMouse_shfl)/sqrt(exc_nValidNs(im));

iavs = cellfun(@(x)nanmean(x(fr,:),2), inh_allNsDaysPooled_eachMouse_shfl);
ises = cellfun(@(x)nanstd(x(fr,:),[],2), inh_allNsDaysPooled_eachMouse_shfl)/sqrt(inh_nValidNs(im));



%%%%% Plot
g = 0;
% figure('position',[60   593   239   320]); 

subplot(122), hold on;
errorbar(1:length(mice), eav, ese, 'b', 'linestyle','none', 'marker','o')
errorbar((1:length(mice))+g, iav, ise ,'r', 'linestyle','none', 'marker','o')
% shfl
errorbar(1:length(mice), eavs, eses, 'color', colss(1,:), 'linestyle','none', 'marker','o')
errorbar((1:length(mice))+g, iavs, ises ,'color', colss(2,:), 'linestyle','none', 'marker','o')

xlim([.5, length(mice)+.5])
xlabel('Mice')
ylabel(namc)
set(gca,'tickdir','out')
set(gca,'xtick',1:length(mice))
yl = get(gca,'ylim');
% ym = yl(2)-range(yl)/20;
ym = iav + ise + range(yl)/20;
plot(1:length(mice), hh_pooledN(fr,:).*ym, 'k*')
title('avePooledNs')


% save figure
if savefigs        
    fdn = fullfile(dirn0, strcat(namc,'_','ROC_curr_chAl_excInh_aveSe_time-1_eachMouse_', dm0, nowStr));
    savefig(fign, fdn)    
    print(fign, '-dpdf', fdn)
end 


