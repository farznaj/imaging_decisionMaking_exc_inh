clc
% clear all
% close all

ratname = 'am053';
folder = fullfile('Z:','data', ratname, 'behavior');
cd(folder)


% cd '~/Churchland Lab/repoland/playgrounds/Kachi/data'
cd 'C:\Users\fnajafi\Dropbox\ChurchlandLab\data for trial history'

% [filename, pathname] = uigetfile({'*.mat'},'File Selector');

%try to implement sliding window for calculating b values for each day?

% mice = {'am008', 'am010','am012', 'am013','am014','am015','am016'};
mice = {'am049', 'am052','am053', 'am054','am055'};
% day = '2-Feb-2013'; % '2-Sep_2014';
day = '19-Sep_2014';
n_days_back = 10; 20;

str = ' all filtered data.mat';
bsMat = zeros(length(mice),2);
bfMat = zeros(size(bsMat));
trialMat = zeros(length(mice));

mod = 1; % remove auditory trials
for i = 1: length(mice) % length(filename_all); % length(mice)
%     figure;
%     filename = [mice{i} str];    
    ratname = mice{i};
    [bs bf b0 mousename numtrials] = mouse_prob_choice_file(ratname, day, n_days_back, mod);
%     [bs bf b0 mousename numtrials] = mouse_prob_choice_file(filename_all{i},mod);
    
    bsMat(i,:) = bs;
    bfMat(i,:) = bf;
    trialMat(i,:) = numtrials;
    
end

%%
colors = jet(length(mice)); % [1 0 1; 0 1 1; 1 0 0; 0 0 1; 0 0 0];

figure; %(6); 

y = (-1:0.05:1);
x = zeros(size(y));
dash = scatter(x,y,2,[0 0 0],'o'); hold on; scatter(y,x,2,[0 0 0],'o');hold on;
plot([0 0],[-1 1],':', 'color', [.6 .6 .6])
plot([-1 1],[0 0],':', 'color', [.6 .6 .6])
miceplot = size(colors,1);

for k = 1:size(colors,1)
% miceplot(k) = scatter(bsMat(k,1),bfMat(k,1),20,colors(k,:),'o','filled'); hold on
% h = errorbar(bsMat(k,1),bfMat(k,1),bfMat(k,2), 'color' ,'k');
% errorbar_tick(h,80)
% h = herrorbar(bsMat(k,1),bfMat(k,1),bsMat(k,2),'k');
% errorbar_tick(h,80)

h = errorbarxy(bsMat(k,1),bfMat(k,1),bsMat(k,2),bsMat(k,2),bfMat(k,2),bfMat(k,2));
set(h,'color', 'k')
hold on
% text(bfMat(k,1),bsMat(k,1),['\leftarrow ' mice{k}], 'FontSize',10);
miceplot(k) = scatter(bsMat(k,1),bfMat(k,1),30,'k','o','filled'); hold on

end
% ylim([-1 1])
% xlim([-1 1])
ylim([-.57 .57])
xlim([-.57 .57])
ylabel('b_{f}')
xlabel('b_{s}')

% legend(miceplot,mice)
title('pooled across modalities')

set(gca,'tickdir', 'out')
set(gca,'ticklength',[.05 .05])
axis square
% cd '~/Churchland Lab/Dropbox/Kachi/trial history data/'
% saveas(6,'success and failures all modalities.jpg');
