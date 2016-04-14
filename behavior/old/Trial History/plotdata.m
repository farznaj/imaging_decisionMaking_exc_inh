
clc
clear all
close all

cd '~/Churchland Lab/repoland/playgrounds/Kachi/data/filterapproach1/'

[filename, pathname] = uigetfile({'*.mat'},'File Selector');

datastruct = load(filename);

dat = datastruct.filter_dataMat;
numtrials = length(dat);
subject = datastruct.subject;



ClickRates = unique([dat.this_vis_click_rate]);

for rate = 1:length(ClickRates)
    
    ind_trials_for_rate  = ([dat.this_vis_click_rate] == ClickRates(rate));   % indices for trials w/ current trials w
    
    trials_for_rate = dat(ind_trials_for_rate);            %selected trials for current rate
    
    right_choice_trials = ([trials_for_rate.resp] == 2);   %indices for right choice trials
    
    
    numchooseright(rate) = sum(right_choice_trials);
    numtrialsrate(rate) = sum(ind_trials_for_rate);
    

end


proportion_right = numchooseright./ numtrialsrate;

figure; plot(ClickRates, proportion_right,'bo-'); hold on

% [x,y,b,d] = psych_fit_plot(ClickRates,numchooseright, numtrialsrate);

