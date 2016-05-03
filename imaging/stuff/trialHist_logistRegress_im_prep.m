% preparing neural data for trial history analysis.

%%
traces_stimAve = squeeze(nanmean(traces_al_sm,1)); % units x trials.

traces_aveTrs = nanmean(traces_stimAve,2); % units x 1 % response of each neuron averaged across stim and trials.

% figure; plot(traces_aveTrs)

thAct = 1e-3; % could be a good th for excluding neurons w no activity.
nonActiveNs = traces_aveTrs < thAct; 
sum(nonActiveNs)

%% ok u want to include same trials for both imaging and behavior... now the
% 2 scripts need to merge.

d = diff(trialNumbers); 
u = unique(d);
if length(u)>1
    error('A gap in imaged trials. Trial history analysis will be problematic! Look into it!')
end

X = X(trialNumbers,:);


%% ok now what? set the y vector. that's the neurons response during stim
% presentation at each trial. of course u don't know what timepoint to use.
% 1) do the analysis on every frame (there are ~30 frames)? 2) do the
% analysis on the average across frames. lets go with the 2nd one for now.

in = 1;
if ~nonActiveNs(in)
    y_ca = traces_stimAve(in, :)';
end

% decide about what trials u wanna analyze... if you have the choice term
% in the predictor matrix u'll be limited to corr and incorr trials (ie
% completed trials.) if not, decide what you want.

% since in early decision trials the stimulus ends at the middle, u don't
% want to include them in ur analysis. 
% if u're also going to include choice in X then u'll be limited to succ
% and fail trials.

% ok now an important question: u'll use logistic regression. the output
% vector, if u're doing logistic regressoin and binomial, should be 0 or 1,
% for HR or LR, or something similar. o o ok now? 

% for the output vecotr, u were going to use y_ca values for each trial, they are like
% the average spike activity during stim presentation in a neuron. 

% but it seems if u want to do logistic regression... do you want to?
% logistic regress is about categorization... eg you want to see how much a
% bunch of predictors determin the category of something 







