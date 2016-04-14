clc
clear all
close all


% cd '~/Churchland Lab/repoland/playgrounds/Kachi/data'
cd 'C:\Users\fnajafi\Dropbox\ChurchlandLab\data for trial history'

% [filename, pathname] = uigetfile({'*.mat'},'File Selector');

%try to implement sliding window for calculating b values for each day?


mice = {'am008', 'am010','am012', 'am013','am014','am015','am016'};

str = ' all filtered data.mat';
bsMat = zeros(length(mice),2);
bfMat = zeros(size(bsMat));
trialMat = zeros(length(mice));

mod = 1; % remove auditory trials
for i = 1: length(mice)
%     figure;
    filename = [mice{i} str];
    
    
    [bs bf b0 mousename numtrials] = mouse_prob_choice_file(filename,mod);
    
    bsMat(i,:) = bs;
    bfMat(i,:) = bf;
    trialMat(i,:) = numtrials;
    
end

colors = jet(length(mice)); % [1 0 1; 0 1 1; 1 0 0; 0 0 1; 0 0 0];
figure(6);
y = (-1:0.05:1);
x = zeros(size(y));
dash = scatter(x,y,2,[0 0 0],'o'); hold on; scatter(y,x,2,[0 0 0],'o');hold on;
miceplot = size(colors,1);

for k = 1:size(colors,1)
miceplot(k) = scatter(bsMat(k,1),bfMat(k,1),20,colors(k,:),'o','filled'); hold on
errorbar(bsMat(k,1),bfMat(k,1),bfMat(k,2))
herrorbar(bsMat(k,1),bfMat(k,1),bsMat(k,2))
% text(bfMat(k,1),bsMat(k,1),['\leftarrow ' mice{k}], 'FontSize',10);

end
ylim([-1 1])
xlim([-1 1])
ylabel('bf')
xlabel('bs')

legend(miceplot,mice)
title('pooled across modalities')

% cd '~/Churchland Lab/Dropbox/Kachi/trial history data/'
% saveas(6,'success and failures all modalities.jpg');
