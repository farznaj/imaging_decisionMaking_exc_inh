


clear all

close all;
allmice = 'am008';

all_bs = [];
all_bf = [];
all_b0 = [];

numDays = 17;

myDate = datenum('01-Feb-2013');

for i_mouse = 1:numDays

    datestr(myDate)
    
[bs bf b0] = mouse_prob_choice_komac(allmice(1,:),1,datestr(myDate),0); 


all_bs(i_mouse,:) = bs;
all_bf(i_mouse,:) = bf;
%all_b0(i_mouse,:) = b0;
myDate = myDate -1;
end;

% [bs bf b0] = mouse_prob_choice_komac(allmice(1,:),numDays,datestr(myDate),1);

figure

xax = fliplr([1:1:numDays]);


plot(xax,all_bs(:,1),'k-'); hold on
plot(xax,all_bf(:,1),'k--'); hold on




errorplot([xax' all_bs(:,1) all_bs(:,2)],'k',1)
errorplot([xax' all_bf(:,1) all_bf(:,2)],'k',1)

plot(xax,all_bs(:,1),'ko','markerface','k'); hold on
plot(xax,all_bf(:,1),'ko','markerface','w'); hold on

axprefs(gca);

ylabel('magnitude of bias parameter')
xlabel('training day')

set(gcf,'paperposition',[0.25 5 6 4]);
cd '~/Churchland Lab/Dropbox/behavdata/mice data/mouseFigures'

print(gcf, '-dpdf', [allmice ' Bias History.pdf'])


