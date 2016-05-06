% Preliminary analysis for the Lick paradigm.
% Set vars, plot PMF and waitDur histogram.

%%
% clear all; home
% clear n_days

subject = 'hni04';
days = {'15-Jan-2016'};
savefigs = 0;
n_days = 1; %15; % specify number of days back (from days{1}) that you want to analyze. if you don't specify this, only days mentioned in days will be analyzed.

exclude_Exclude = 1; % whether you want to exclude trials set to Exclude during experiment.
exclude_shortWaitdur = 1; %1; % whether you want to exclude shortWaitDur trials.
waitdur_th = 0; %0; % waitdur below this value will be discarded. % you can use stimdur instead, but i think it makes more sense to use waitdur...

thbeg = 0; %6; % exclude 10 initial trials of each session.
shownumtrs = 1; % if 1, number of trials that went into each stim rate will be shown on PMF.
fit_sig = 1; % fit a sigmoid?
checks = 0; % if 1, excluded trials and choice on eary decision trials will be printed.

if ispc
    %     server_address = '\\sonas-hs.cshl.edu\churchland-norepl\data not likely to be used';
    server_address = '\\sonas-hs.cshl.edu\churchland\data'; % office PC
elseif isunix
    server_address = '/Volumes/Churchland/data'; % your mac
end

if ~exist('hpmf', 'var') || ~ishandle(hpmf)
    hpmf = figure;
end
figure(hpmf), clf


%% Go to the directory that contains the behavioral data

folder = fullfile(server_address, subject,'behavior');
cd(folder)

% get the list of all files in that directory
list_files = dir(folder);
list_files = list_files(3:end);
[~,b] = sort([list_files(:).datenum]);
files_ordered = list_files(b);
files = {files_ordered.name};

% exclude files that start with ._
bbb = cellfun(@(x)strfind(x,'._'), files, 'uniformoutput', 0);
files = files(cellfun(@isempty, bbb));
% only include .mat files
bbb = cellfun(@(x)strfind(x,'.mat'), files, 'uniformoutput', 0);
files = files(cellfun(@(x)~isempty(x), bbb));

alldata = [];
all_ntrs = [];

if ~exist('n_days','var')
    n_days = length(days);
else
    day = days{1};
end

array_days_loaded = zeros(1, n_days);
days_loaded = {};


%%

for iday = 1:n_days
    
    if length(days)>1 % if days includes specifically all the days that you want to analyze. otherwise start with days (single element) and go back by n_days (similar to n_days_back).
        day = days{iday};
    end
    
    % find the files that correspond to day
    a = cellfun(@(x)strfind(x,day), files, 'uniformoutput', 0);
    tempf = files(~cellfun(@isempty, a)); % tempf includes all files whose name includes day
    finalf = tempf;
    
    if ~isempty(finalf)
        days_loaded = [days_loaded {day}];
    end
    
    %% Load the behavioral data corresponding to day "day".
    
    for isess = 1:length(finalf)
        
        thisf = fullfile(folder, finalf{isess});
        
        % Load the all_data file and remove the last trial
        fprintf('Loading... %s\n\n', finalf{isess})
        load(thisf)
        array_days_loaded(iday) = 1;
        
        all_data = removeBegEndTrs(all_data, thbeg);
        
        %% Clean fields
        
        cleanFields_behavior_analysis
        
        %% Concatenate the all_data file of all sessions
        
        alldata = [alldata, all_data];
        all_ntrs = [all_ntrs, length(all_data)];
        
    end
    
    %% Go one day back
    
    day = datestr(datenum(day,'dd-mmm-yyyy')-1); % Farz
    
end


%%
if isempty(alldata)
    warning('no files were loaded')
    return
end

cb = alldata(1).categoryBoundaryHz; % categ boundary in hz


%% Set event numbers and trial types: multi-sensory, only visual, only audio

% set stimulus rate for each trial
[nevents, stimdur, stimrate, stimtype] = setStimRateType(alldata);

multisens = stimtype(:,1);
onlyvis = stimtype(:,2);
onlyaud = stimtype(:,3);

stimrate_0 = stimrate;
num_multi_vis_aud = [sum(multisens) sum(onlyvis) sum(onlyaud)];


%% Set outcome and account for allow correction.

allowCorrectResp = 'change';
uncommittedResp = 'nothing'; % 'change'; %'remove'; % % 'remove', 'change', 'nothing';
[outcomes, allResp, allResp_HR_LR] = set_outcomes_allResp(alldata, uncommittedResp, allowCorrectResp);

%{
outcomes = [alldata.outcome];

% set to failure the outcome of correct trials on which the mouse used allow correction (bc we want to consider the outcome of the first choice)
a = arrayfun(@(x)x.parsedEvents.states.punish_allowcorrection, alldata, 'uniformoutput', 0);
allowCorrectEntered = ~cellfun(@isempty, a); % trials that the mouse entered the allow correction state.
% outcome_0(allowCorrectEntered & outcomes_0==1) = 0;
outcomes(allowCorrectEntered) = 0; % it was first the above code, then you took outcomes_0==1 out, bc mouse may go to allowcorrect, then he may correct it but not commit it (outcome=-5), I'd still call these an incorrect choice.

%{
for tri = 1:length(alldata)
    if ~isempty(alldata(tri).parsedEvents.states.punish_allowcorrection) % this trial entered state punish_allowcorrection
        outcome(tri) = 0;
    end
end
%}
%}


%%
ntrs_eachSession_sum = [all_ntrs sum(all_ntrs)]


%% Exclude some trials from analysis

waitdur = [alldata.waitDuration];
invalid = sum(ismember(outcomes, [-3 -4 -1]));
noChoice = sum(ismember(outcomes, [-2 -5]));
% sanity1 = invalid+noChoice == sum(outcome_0~=0 & outcome_0~=1);
% if sanity1~=1, error('something wrong'), end
invalid_noChoice_shortDur_exclude = [invalid  noChoice  sum(waitdur<waitdur_th)  sum([alldata.exclude])]
% length(outcome_0)

outcomes(outcomes~=0 & outcomes~=1) = NaN; % only analyze correct and incorrect trials (not invalid or noChoice).

if exclude_shortWaitdur
    outcomes(waitdur<waitdur_th) = NaN;
    allResp_HR_LR(waitdur<waitdur_th) = NaN;
    stimrate_0(waitdur<waitdur_th) = NaN;
end

if exclude_Exclude
    outcomes([alldata.exclude]==1) = NaN; % exclude trials that were marked as exclude during experiment.
    allResp_HR_LR([alldata.exclude]==1) = NaN;
    stimrate_0([alldata.exclude]==1) = NaN;
end

%{
trsdiff = find((nvis~=0 & naud~=0) & (nvis-naud)~=0); % these are trials that different number of events were presented for the audio and visual parts of the multi-sensory stimulus.
[nvis(trsdiff)  naud(trsdiff)  outcome_0(trsdiff)']

outcome(trsdiff) = NaN;
%}

num_fract_excluded = [sum(isnan(allResp_HR_LR))  nanmean(isnan(outcomes))]

num_LR_HR_CB_ratio = [sum(stimrate_0 < cb) , sum(stimrate_0 > cb) , ...
    sum(stimrate_0 == cb) , sum(stimrate_0 < cb)/sum(stimrate_0 > cb)]; % LR, HR, boundary

num_multisens_vis_aud = [sum(multisens) sum(onlyvis) sum(onlyaud)]


%% Plot psychometric functions

wd = 2;
vec_rates = sort([cb : -wd : min(stimrate_0)-wd  ,  cb+wd : wd : max(stimrate_0)+wd]); % make sure categ boundary doesn't go at the middle of a bin!

col = [1 0 0; 0 0 1; 0 1 0];
hmodal_all = NaN(1,length(col));
legstring = {'multi-sensory', 'visual', 'auditory'};
nss_all = NaN(length(vec_rates) , length(legstring));
dolabs = 0;

for imodality = 1:3 % {'multi', 'onlyvis', 'onlyaud'}
    
    switch imodality
        case 1
            allRespM = allResp_HR_LR(multisens);
            outcome = outcomes(multisens);
            stimrate = stimrate_0(multisens);
        case 2
            allRespM = allResp_HR_LR(onlyvis);
            outcome = outcomes(onlyvis);
            stimrate = stimrate_0(onlyvis);
        case 3
            allRespM = allResp_HR_LR(onlyaud);
            outcome = outcomes(onlyaud);
            stimrate = stimrate_0(onlyaud);
    end
    
    
    %%
    if ~isempty(allRespM) && ~all(isnan(allRespM))
        
        dolabs = 1;
        figure(hpmf); subplot(211)
        shownumtrs = true;
        
        % you started using this instead of the lines below on Feb 2 2016, so
        % all PMF plots made before this date and put on evernote need the following attention:
        % prop HR for stim rate = category boundary was not accurate (but accurate for all other rates).
        
        [HRchoicePerc, vec_rates, up, lo, nSamples, xx, yy, ee, hmodal] = PMF_set_plot...
            (stimrate, allRespM, cb, vec_rates, 1, shownumtrs, col(imodality,:));
        
        nss_all(:,imodality) = nSamples;
        hmodal_all(imodality) = hmodal;
        
        %% fit a sigmoid
        
        if fit_sig
            rates = xx; % vec_rates(~isnan(HRchoicePerc)) + wd/2;
            pmf = yy; % HRchoicePerc(~isnan(HRchoicePerc));
            ns = nSamples(~isnan(HRchoicePerc));
            
            c = 1;
            % sigmoid = 'logistic';
            sigmoid = 'invgauss';  % can be 'invgauss' or 'logistic', invgauss uses probit
            
            % Sigmoid
            if length(rates) > 2
                xs = (rates(1) : 0.1 : rates(end))';
                % GLM
                switch sigmoid
                    case 'invgauss'
                        [b, junk, stats] = glmfit(rates', pmf', 'binomial', 'link', 'probit');
                        bs(c) = 1/b(2);
                        b_ses(c) = stats.se(2);
                        yfit = glmval(b, xs, 'probit');
                        % Compute PSE. Since with probit the GLM is norminv(mu) = Xb, and
                        % b(1) is for the constant term, the PSE is
                        % (norminv(0.5) - b(1)) / b(2)
                        pses(c) = (norminv(0.5, 0, 1) - b(1)) / b(2);
                        
                    case 'logistic'
                        [b, junk, stats] = glmfit(rates', [pmf'.*ns' ns'], 'binomial');
                        bs(c) = 1/b(2);
                        b_ses(c) = stats.se(2);
                        yfit = glmval(b, xs, 'logit');
                        % Compute PSE. Since with logit the GLM is log(mu/(1-mu)) = Xb, and
                        % b(1) is for the constant term, the equation is
                        % log(0.5 / (1-0.5)) = Xb,  so...
                        % log(1) = Xb
                        % 0 = Xb
                        % ...and the PSE is
                        % -b(1) / b(2)
                        pses(c) = -b(1) / b(2);
                        
                    otherwise
                        error('sigmoid parameter must be ''invgauss'' or ''logistic''');
                end
                
                plot(xs, yfit, 'color', col(imodality,:), 'LineWidth', 1);
                
                %     bestParams = fitSigConstrained(rates, pmf, ns);
                %     sigRates = rates(1)-0.5 : 0.1 : rates(end)+0.5;
                %     plot(sigRates, logistic(sigRates, bestParams), 'color', colors(c,:));
                
            end
        end
    end
end


%% add figure name and title.

% a = [days_loaded{:}];
% a(:) % shows all files that were loaded.
% day1 = a{1}(6:end-9);
day1 = days_loaded{1};
nd = sum(array_days_loaded);
if n_days>1
    figtitle = [subject, '\_', day1, '\_', num2str(nd), 'daysTotal'];
    figname = [subject, '_', day1, '_', num2str(nd), 'daysTotal'];
    %     set(gcf,'name', [subject, '_', days{1}, '_', num2str(n_days), 'daysTotal'])
    set(gcf,'name', figname)
else
    figtitle = [subject, '\_', day1];
    figname = [subject, '_', day1];
    %     set(gcf,'name', [subject, '_', days{1}])
    set(gcf,'name', figname)
end

title(figtitle)


%% add labels to the figure

if dolabs
    legstring_now = legstring(num_multisens_vis_aud~=0);
    legend(hmodal_all(~isnan(hmodal_all)), legstring_now, 'location', 'northwest')
    
    h = plot([cb cb],[0 1],'k:');
    hh = plot([vec_rates(1) vec_rates(end)],[.5 .5],'k:');
    
    % a = vec_rates(~isnan(HRchoicePerc));
    xl = round([min(stimrate_0) max(stimrate_0)]);
    xlim([xl(1)-1   xl(2)+1])
    ylim([-.05 1.05])
    xlabel('Stimulus rate')
    ylabel(['% HR choice', ' (', alldata(1).highRateChoicePort, ')'])
    
    set(gca,'tickdir','out')
    
    %%
    final_num_multi_vis_aud = sum(nss_all)
    if ~isequal(sum(all_ntrs), nansum(nss_all(:)) + num_fract_excluded(1))
        warning('something wrong with the number of trials.')
    end
    
    num_LR_HR_CB_ratio
    lr = nansum(sum(nss_all(vec_rates < cb,:)));
    hr = nansum(sum(nss_all(vec_rates > cb,:)));
    cbn = nansum(sum(nss_all(vec_rates == cb,:)));
    findal_num_LR_HR_CB_ratio = [lr hr cbn lr/hr]
    
end


%% histogram of waitdur

waitdurs = [alldata.waitDuration]'*1000; % in ms
r = range(waitdurs);
vec_waitdurs = min(waitdurs):r/10:max(waitdurs);
[n,i] = histc(waitdurs, vec_waitdurs);
% n = n/sum(n);
figure(hpmf); subplot(212)
plot(vec_waitdurs, n)
xlabel('Wait duration (ms)')
% ylabel('Fraction of trials')
ylabel('Number of trials')
set(gca,'tickdir','out')
box off


%%
if checks
    % look at excluded trials
    [(1:length(alldata))' [alldata.exclude]' [alldata.outcome]']
    
    
    % look at animal's choice on early decision trials
    respi = NaN(size(alldata'));
    ed = find(outcomes==-1);
    for tri = ed
        ll = alldata(tri).parsedEvents.pokes.L(:);
        rl = alldata(tri).parsedEvents.pokes.R(:);
        if ~isempty(alldata(tri).parsedEvents.states.early_decision0)
            edtime = alldata(tri).parsedEvents.states.early_decision0(1);
            if ismember(edtime, ll)
                respi(tri) = 1;
                
            elseif ismember(edtime, rl)
                respi(tri) = 2;
            end
        end
    end
    
    correctside = [alldata.correctSideIndex]';
    [respi(ed) correctside(ed) nevents(ed) stimrate_0(ed)]
    
    [nanmean(respi(ed) == correctside(ed))  nanmean(respi(ed)==1) nanmean(correctside(ed)==1)]
    
end


%% check the length of the arudino wheel signal and the trial.

%{
% I guess you are doing this test here to see if having
% alldata(1).wheelPostTrialToRecord = 1sec is a problem... basically you
% want to know if by the time rotary stops, the next trial has already
% stopped...
if isfield(alldata, 'wheelRev')
    for tri = 2:length(all_data)-1; if (all_data(tri).parsedEvents.states.start_rotary_scope(1) - (all_data(tri-1).parsedEvents.states.stop_rotary_scope(1)+.5))<=1, tri, end, end
end
%}


%%
if savefigs
    if ispc
        fold = 'C:\Users\fnajafi\Desktop'; % PC
    elseif isunix
        fold = '/Users/Farzaneh'; % mac
    end
    %     cd(fullfile(fold, 'Dropbox', 'ChurchlandLab', 'analysis_behavior'))
    cd(fold)
    
    fign = [get(gcf,'name')];
    
    %     saveas(gca, fign, 'fig')
    saveas(gca, fign, 'jpg')
end
% saveas(gca, fign, 'pdf')





%% set the PMF (pshychometric function).
%{
    th = 0; 2; % min number of trials in a stim rate bin in order to compute %HR
    % wd = 2;
    % vec_rates = sort([cb : -wd : min(stimrate)-wd  ,  cb+wd : wd : max(stimrate)+wd]); % make sure categ boundary doesn't go at the middle of a bin!

    [n, bin_rate] = histc(stimrate, vec_rates);
    % [vec_rates' n']
    
    HRchoicePerc = NaN(1, length(vec_rates));
    
    for ri = 1:length(vec_rates) % allbins
        
        allout = outcome(bin_rate == ri); % outcome of all trials whose stim rate was in bin ri
        validout = allout(allout>=0); % only consider trials that were correct, incorrect
    
        if length(validout)>th
            HRchoicePerc(ri) = nanmean(validout==1);
        end
    end
    
    
    % success rate for LR choices indicates percent LR. since we want to plot percent HR, then for LR choices it will be eqal to 1-success rate.
    HRchoicePerc(vec_rates < cb) = 1-HRchoicePerc(vec_rates < cb);
    
    
    %% compute confidence bounds
    
%     CI_alpha = 0.05; % alpha corresponding to CI=.95
%     perc_normal = 1 - CI_alpha/2; % .975
%     z = norminv(perc_normal, 0, 1); % 1.96; z corresponding to probability .975 in a standard normal distribution.
    z = 1; % 1SEM
    
    up = NaN(1, length(HRchoicePerc));
    lo = NaN(1, length(HRchoicePerc));
    nSamples = NaN(1, length(HRchoicePerc));
    for r = 1:length(HRchoicePerc)
        nSamples(r) = sum(~isnan(outcome(bin_rate==r)));
        
        [up(r), lo(r)] = wilsonBinomialConfidenceInterval(HRchoicePerc(r), z, nSamples(r));
    end
    nss_all(:,imodality) = nSamples;
    
    
    %% plot percentage HR vs ev rate.
    
    e = [(HRchoicePerc-lo)', (up-HRchoicePerc)'];
    e = e(~isnan(HRchoicePerc),:);
    x = vec_rates(~isnan(HRchoicePerc)) + wd/2;
    y = HRchoicePerc(~isnan(HRchoicePerc));
    
    figure(hpmf); subplot(211)
    hold on
    
    % plot(vec_rates + wd/2, HRchoicePerc, 'k.')
    % h = errorbar(vec_rates + wd/2, HRchoicePerc, HRchoicePerc-lo, up-HRchoicePerc, 'color', 'k','linestyle','none');
    
    hmodal(imodality) = plot(x,y,'.','color', col(imodality,:));
    h = errorbar(x, y, e(:,1), e(:,2), 'color', col(imodality,:),'linestyle','none');
    errorbar_tick(h,200)
    
%{
    [h1,hp] = boundedline(x, y, e, 'alpha', 'transparency', .05);
    set(h1, 'color', 'k', 'marker', 'o');
    set(hp, 'facecolor', 'k');
%}
    
    
    % print the number of trials that went into each stim rate.
    if shownumtrs
%         nn = n;
%         nn(isnan(HRchoicePerc)) = NaN;
        for r = 1:length(vec_rates)
            text(vec_rates(r)+wd/2, HRchoicePerc(r)+.05, num2str(nSamples(r)),'color',col(imodality,:));
        end
    end
%}
