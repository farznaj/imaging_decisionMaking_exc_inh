% all_data = all_data(1:end-1);
[data, array_days_loaded, trials_per_day] = loadRatBehavioralData_fn (subject, day, n_days_back);
data_backup = data;

%%
rates_all = [data.nVisualEvents];
rates = unique(rates_all);
responses_all = [data.responseSide];

ew = [data.earlyWithdrawal];
dnc = [data.didNotChoose];
[nanmean(ew), nanmean(dnc)]


% in the array [data.success] set those trials that are earlyWithdrawal or didNotChoose to NaN.
success_all = [data.success];
nanmean(success_all)

success_all = double(success_all);
success_all(ew) = NaN;
success_all(dnc) = NaN;
nanmean(success_all)

%% percentage of early withdrawals for each stimulus rate.
ew_rate = NaN(length(rates),1);
for r = 1:length(rates)
    ew_rate(r) = nanmean(ew(rates_all==rates(r)));
end
[rates' ew_rate]

%% waitduration for each rate and whether the trial was a success or not. (mouse am053, 25Sep to 10 days back: LR stimuli have longer wait times on correct trials; but HR stimuli have longer wait times on incorrect trials... puzzling!)
waitdur_all = [data.timeInCenter]/1000;
waitdur_eachrate_sf = NaN(length(rates),2);
for r = 1:length(rates)
    waitdur_eachrate_sf(r,:) = [nanmean(waitdur_all(rates_all==rates(r) & success_all==1)),...
        nanmean(waitdur_all(rates_all==rates(r) & success_all==0))];
end

[rates' waitdur_eachrate_sf diff(waitdur_eachrate_sf,[],2)]

%% time2choose (movement duration): compare t2choose left side when it was correct, with t2choose left
% side when it was incorrect, with
% t2choose right when correct, and t2choose right when incorrect.

% (mouse am053, 25Sep to 10 days back: it takes quite longer to go to the
% left side compared to the right side when the choice is correct; but when
% the choice is incorrect, it takes longer to go to the right side)
t2choose = NaN(1,length(data));
t2choose_l = NaN(1,length(data));
t2choose_r = NaN(1,length(data));
for itr = 1:length(data)
    d = data(itr).parsedEvents.states.left_on;
    if ~isempty(d)
        t2choose_l(itr) = diff(d); % animal was supposed to go to left (based on the stimulus rate).
    end
    
    d = data(itr).parsedEvents.states.right_on;
    if ~isempty(d)
        t2choose_r(itr) = diff(d); % animal was supposed to go to right (based on the stimulus rate).
    end
    
end
% compare time2choose left side when it was correct, with t2choose left
% side when it was incorrect, with
% t2choose right when correct, and t2choose right when incorrect.
[nanmean(t2choose_l(success_all==1)) nanmean(t2choose_r(success_all==0))...
    nanmean(t2choose_r(success_all==1)) nanmean(t2choose_l(success_all==0))]

% compare t2choose for correct, incorrect, left, right choices, at each
% rate
t2choose_s_rate = NaN(1,length(rates)); % shows the average t2choose at each rate for success trials.
t2choose_f_rate = NaN(1,length(rates));  % shows the average t2choose at each rate for failure trials.
for r = 1:length(rates)
    t2choose_s_rate(r) = nanmean(t2choose_l(success_all==1 & rates_all==rates(r)));
    if isnan(t2choose_s_rate(r))
        t2choose_s_rate(r) = nanmean(t2choose_r(success_all==1 & rates_all==rates(r)));
    end
    
    t2choose_f_rate(r) = nanmean(t2choose_l(success_all==0 & rates_all==rates(r)));
    if isnan(t2choose_f_rate(r))
        t2choose_f_rate(r) = nanmean(t2choose_r(success_all==0 & rates_all==rates(r)));
    end
end


% compare t2choose for the same type of stimulus (lr or hr), but different
% choices
[nanmean(t2choose_l(success_all==1)) nanmean(t2choose_l(success_all==0))...
    nanmean(t2choose_r(success_all==1)) nanmean(t2choose_r(success_all==0))]



%{
% another way of getting the above results, in a more intuitive way.
t2choose_l_s = NaN(1,length(data));
t2choose_l_f = NaN(1,length(data));
t2choose_r_s = NaN(1,length(data));
t2choose_r_f = NaN(1,length(data));
for itr = 1:length(data)
    if data(itr).responseSide==1
        if data(itr).success==1 % animal correctly chose left
            t2choose_l_s(itr) = diff(data(itr).parsedEvents.states.left_on);
        elseif data(itr).success==0 % animal incorrectly chose left
            t2choose_l_f(itr) = diff(data(itr).parsedEvents.states.right_on);
        end
    elseif data(itr).responseSide==2
        if data(itr).success==1 % animal correctly chose right
            t2choose_r_s(itr) = diff(data(itr).parsedEvents.states.right_on);
        elseif data(itr).success==0 % animal incorrectly chose right
            t2choose_r_f(itr) = diff(data(itr).parsedEvents.states.left_on);
        end
    end
end
% compare time2choose left side when it was correct, with t2choose left
% side when it was incorrect, with
% t2choose right when correct, and t2choose right when incorrect.
[nanmean(t2choose_l_s) nanmean(t2choose_l_f) nanmean(t2choose_r_s) nanmean(t2choose_r_f)]
%}

%%
[rates_all ; responses_all ; success_all]'



%% similar to pmf_modality
success_rate = NaN(1,length(rates));
for r = 1:length(rates)
    success_rate(r) = nanmean(success_all(rates_all==rates(r)));
end
success_rate 

%% similar to numtrs_eachrate_modal
for r = 1:length(rates)
    sum(~isnan(success_all(rates_all == rates(r))))
end
