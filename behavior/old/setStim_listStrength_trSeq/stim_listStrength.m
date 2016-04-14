% generate a random number of events for the stimulus on the current trial
% given strength_prop, which corresponds to the proportion of 
% hard, medium-hard, medium-easy, and easy trials, respectively.

vis_strength_prop = value(VisStrengthPropHME);

%%
% min and max number of events for the given stimulus duration.
min_evnum = round(stimdur/(max_iei_accepted+evdur))+1; % this formula works well.
max_evnum = floor(stimdur/(evdur+gapdur));

categ_bound_Hz = 16; % category boundary in Hz, i.e. number of events for 1000ms stim.
categ_bound_num = categ_bound_Hz*stimdur/1000; % number of events for the current stimulus.

% generate a vector in order to group number of events into hard, medium-hard, medium-easy, easy
vec_allStrength = floor(categ_bound_num) : ceil(range([floor(categ_bound_num) max_evnum])/4) : max_evnum;
vec_allStrength = [vec_allStrength, max_evnum];

%%
% pick a strength group (hard, medium-hard, medium-easy, easy) from
% list_prop (probability that a particular group happens is given by list_prop)
v = [0 cumsum(vis_strength_prop)];
v(end) = v(end)+.01;

r = rand;
[~,i] = histc(r,v);

hiS = vec_allStrength(i)+1:vec_allStrength(i+1);
loS = min_evnum + max_evnum - hiS;
strengthGroup = sort([loS, hiS]);

% pick an evnum (number of events) from the strength group (probability
% that a particular evnum happens is equal for all evnums in strengthGroup)
vthisS = 0: 1/length(strengthGroup) :1;

r = 1;
while r==1
    r = rand;
end

[~,i] = histc(r,vthisS);

evnum = strengthGroup(i)



%%
%{
% high rates:
range_high_rate = range([floor(categ_bound) max_evnum])-2;

hard_rates = [categ_bound+1 : categ_bound+2]; % .25 of the remaining values
easy_rates = [max_evnum - floor(range_high_rate*.5) + 1  :  max_evnum ]; % 2 lowest values
medhard_rates = [hard_rates(end)+1 : hard_rates(end)+floor(range_high_rate*.25)]; % .25 of the remaining values
medeasy_rates = [medhard_rates(end)+1 : easy_rates(1)-1]; % .5 of the remaining values

% low rates: 
hard_rates = [categ_bound-2 : categ_bound-1];
medhard_rates = [hard_rates(1)-2 : hard_rates(1)-1];
medeasy_rates = [medhard_rates(1)-2 : medhard_rates(1)-1];
easy_rates = [medeasy_rates(1)-2 : medeasy_rates(1)-1];
%}
