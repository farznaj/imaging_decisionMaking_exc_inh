function [rates, hrchoices, numtrs_eachrate_modal] = set_pmf(data,toplot)
% Compute the stimulus rates and percentage high-rate choices for each stimulus
% modality in the structure "data". Also compute the number of trials for
% each stimulus rate.
%
% data: use the following code to set it.
% [data, array_days_loaded, trials_per_day] = loadRatBehavioralData_fn (subject, day, n_days_back);
%
% toplot: 1 or 0; if 1, percent HR vs. rate will be plotted.

%% Remove early withdrawals and didnotchoose trials.
if nargin < 2
    toplot = 0;
end

data_backup = data;
data = data([data(:).earlyWithdrawal] == 0 & [data(:).didNotChoose] == 0);

%% set the index (1 or 2) of the highRatePort for this animal (i.e. port associated with high rate stimuli)
% remember Left is 1, Right is 2. highRateChoicePort for an animal could be
% left or right, but it is fixed per animal.

if ~isfield(data,'highRateChoicePort')
    highRatePort = 2;
else
    if strcmp(data(1).highRateChoicePort, 'L')
        highRatePort = 1;
    else
        highRatePort = 2;
    end
end

rates = unique([data.nVisualEvents])';

%% Compute percentage of HR choices at each stimulus rate for each modality
hrchoices = NaN(length(rates),3);
numtrs_eachrate_modal = NaN(length(rates),3);
for c = 1:3 % auditory, multi-sensory, visual
  % Grab data for this modality
  this_data = data([data(:).visualOrAuditory] == c-2);
  
  if ~isempty(this_data)
%       rates = unique([this_data(:).nVisualEvents this_data(:).nAuditoryEvents]);
%       ns = zeros(1, length(rates));
%       nRightResponses = zeros(1, length(rates));
      % nCorrectResponses = zeros(1, length(rates));

      % Compute PMF (percentage of HR choices at each stimulus rate).
      pmf = nan(1,length(rates));
      nt = nan(1,length(rates));
      for r = 1:length(rates)
        this_rate = rates(r);
        this_rate_data = this_data([this_data(:).nVisualEvents] == this_rate);
        nt(r) = length(this_rate_data); 
        
        pmf(r) = nanmean([this_rate_data(:).responseSide] == highRatePort);
        
        % nCorrectResponses(r) = sum([this_rate_data(:).success] == 1);
%         nRightResponses(r) = sum([this_rate_data(:).responseSide] == highRatePort);
%         ns(r) = length(this_rate_data);              
%         pmf(r) = nRightResponses(r) / ns(r);
      end
      hrchoices(:,c) = pmf';
      numtrs_eachrate_modal(:,c) = nt;
  end
  
end

%% plot percent HR choice vs. rate
if toplot
    th = mean(rates([1,end]));
    figure; hold on, 
    plot(rates, hrchoices,'.'), 

    plot([th,th],[0 1],'k:'), 
    plot([rates(1),rates(end)],[.5 .5],'k:'),
    xlim([rates(1)-1 rates(end)+1])
end


% [[data([data.nVisualEvents]==rates(r)).responseSide]' , [data([data.nVisualEvents]==rates(r)).success]' [data([data.nVisualEvents]==rates(r)).didNotChoose]' [data([data.nVisualEvents]==rates(r)).earlyWithdrawal]']
