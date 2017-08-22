function avetrialAlign_plotAve_noTrGroup_licks(evT, outcome2ana, stimrate2ana, strength2ana, outcomes, stimrate, cb, alldata, alldataDfofGood, alldataSpikesGood, frameLength, nPreFrames, nPostFrames, centerLicks, leftLicks, rightLicks, excludeLicksPrePost)
% Align ca imaging traces (and wheel traces) on center, left and right
% licks. [Similar to avetrialAlign_plotAve_noTrGroup.m, but the events are
% licks here, instead of trial events.)
%
% Example input variables:
%{
nPreFrames = 5;
nPostFrames = 20;
outcome2ana = 'all'; % 'all'; 1: success, 0: failure, -1: early decision, -2: no decision, -3: wrong initiation, -4: no center commit, -5: no side commit
stimrate2ana = 'all'; % 'all'; 'HR'; 'LR';
strength2ana = 'all'; % 'all'; 'eary'; 'medium'; 'hard';
evT = {'centerLicks', 'leftLicks', 'rightLicks'}; % times are relative to scopeTTL onset, hence negative values are licks that happened before that (during iti states).
excludeLicksPrePost = 'none'; % 'none'; 'pre'; 'post'; 'both';
% excludeLicksPrePost: whether you want to include in the analysis all licks, or just
% licks that lack any other lick during nPre, or during nPost, or
% during both.


%}


%%
thStimStrength = 3; % 2; % threshold of stim strength for defining hard, medium and easy trials.

% Don't be surprised if you only want to analyze LR, correct and then you
% see some traces for HR; this is because evT includes licks throughout the
% trial (so the HR licks happen perhaps during the ITI).
s = (stimrate-cb)';
allStrn = unique(abs(s));
switch strength2ana
    case 'easy'
        str2ana = (abs(s) >= (max(allStrn) - thStimStrength));
    case 'hard'
        str2ana = (abs(s) <= thStimStrength);
    case 'medium'
        str2ana = ((abs(s) > thStimStrength) & (abs(s) < (max(allStrn) - thStimStrength))); % intermediate strength
    otherwise
        str2ana = true(1, length(outcomes));
end

if strcmp(outcome2ana, 'all')
    os = sprintf('%s', outcome2ana);
    outcome2ana = -5:1;
else
    os = sprintf('%i', outcome2ana);
end

switch stimrate2ana
    case 'HR'
        sr2ana = s > 0;
    case 'LR'
        sr2ana = s < 0;
    otherwise
        sr2ana = true(1, length(outcomes));
end

trs2ana = (ismember(outcomes, outcome2ana)) & str2ana & sr2ana;
fprintf('Analyzing %s outcomes, %s strengths, %s stimulus: %i trials.\n', os, strength2ana, stimrate2ana, sum(trs2ana))



%%
if length(evT)==3
    col = {'k', 'r', 'g'}; % center, left, right
else
    col = {'r', 'g'}; % left, right
end

h = NaN(1, length(evT));

alldatanow = alldata(trs2ana);

wheelTimeRes = alldata(1).wheelSampleInt;
[traces_wheel, times_wheel] = wheelInfo(alldata);


figure('name', sprintf('%s outcomes, %s strengths, %s stimulus', os, strength2ana, stimrate2ana), 'position', [514   351   246   595]); %[66   550   235   420]);


%%

for i = 1:length(evT) % loop over center, left and right licks.
    
    disp(['------- ', evT{i}, ' -------'])
    
    eventTime = eval(evT{i});
    
    % Take only trials in trs2ana for analysis
    if length(eventTime)>1
        eventTime = eventTime(trs2ana);
    end
    
    
    %% Set eventInds_f_all_clean: for each trial, find imaging frames during which licks happened.
    
    % framesPerTrial = cellfun(@(x)size(x,1), traces);
    % a = cell2mat(eventTime');
    eventInds_f_all = cell(1, length(alldatanow));
    eventInds_f_all(:) = {NaN};
    eventInds_f_all_clean = cell(1, length(alldatanow));
    eventInds_f_all_clean(:) = {NaN};
    % mm = []; % mm2 = [];
    
    disp('setting lick events for each trial....')
    for tr = 1:length(alldatanow)
        et = eventTime{tr}(eventTime{tr}>0)';  % only look at licks that happened after scopeTTL was sent.
        
        if ~all(isnan(et))
            a = repmat(alldatanow(tr), 1, length(et));
            %                         a = repmat(alldatanow(tr), 1, length(eventTime{tr}));
            traceTimeVec = {a.frameTimes}; % time vector of the trace that you want to realign.
            eventInds_f = eventTimeToIdx(et, traceTimeVec); % index of eventTime on traceTimeVec array for each trial.
            
            u = unique(eventInds_f);
            u = u(~isnan(u));
            
            eventInds_f_all{tr} = eventInds_f;
            eventInds_f_all_clean{tr} = u; % no rep, no nan
            
            %             mm = max([mm, eventInds_f]);
            %
            %             mm0 = max(framesPerTrial(tr) - eventInds_f);
            %             mm2 = max([mm2, mm0]);
        end
    end
    
    
    %% Take care of excludeLicksPrePost
    
    %%%%% Choose one of the cases below, whether you want to include in
    %%%%% the analysis all licks, or just licks that lack any other
    %%%%% lick during nPre, or during nPost, or during both.
    
    switch excludeLicksPrePost
        case 'none'
            eventInds_f_all_clean_final = eventInds_f_all_clean;
        case 'pre' %         For each trial, exclude licks that are preceded by another lick in <= nPreFrames.
            eventInds_f_all_clean_final = cellfun(@(x) x([0,diff(x) <= nPre]==0), eventInds_f_all_clean, 'uniformoutput', 0);
        case 'post'
            eventInds_f_all_clean_final = cellfun(@(x) x([diff(x) <= nPost, 0]==0), eventInds_f_all_clean, 'uniformoutput', 0);
        case 'both'
            eventInds_f_all_clean_final = cellfun(@(x) x([0,diff(x) <= nPre]==0 & [diff(x) <= nPost, 0]==0), eventInds_f_all_clean, 'uniformoutput', 0);
    end
    
    % old stuff
    % identify trials with licks happening at intervals shorter than nPreFrames, we later exclude them, bc we want to make sure during nPreFrames (ie baseline of the traces) there has not been any licks.
    % a = cellfun(@(x)(diff(x) < nPreFrames), eventInds_f_all_clean,
    % 'uniformoutput', 0);
    %         lint = cellfun(@(x)sum(diff(x) <= nPreFrames), eventInds_f_all_clean); % trials with licks happening with interval < nPreFrames
    % For each trial, exclude licks that are preceded by another lick in <= nPreFrames.
    
    
    
    %% Set the traces
    
    for isub = 1:2 % plot DF, spikes        
        
        subplot(3,1,isub)
        
        switch isub
            
            case 1  % DF/F
                traces = alldataDfofGood; % alldataSpikesGood; %  traces to be aligned.
                title('DF/F')
                
                shiftTime = frameLength / 2;
                scaleTime = frameLength;
                
                nPre = nPreFrames;
                nPost = nPostFrames;
                
                disp('setting DF/F traces')
                
            case 2 % Spikes
                traces = alldataSpikesGood;
                title('Spiking')
                
                shiftTime = frameLength / 2;
                scaleTime = frameLength;
                
                nPre = nPreFrames;
                nPost = nPostFrames;
                
                disp('setting spike traces')
        end
        
        traces = traces(trs2ana);
        
        
        %% Align traces on licks, ie set tracesa: cell array, each element for 1 trial, and of size frs x units x licks
        
        % clear timea
        %         tracesa = cell(1, length(alldatanow));
        %         nvalida = cell(1, length(alldatanow));
        tracesa = NaN(nPre+nPost+1, size(traces{1},2), max(cellfun(@length, eventInds_f_all_clean_final)), length(alldatanow));
        
        for tr = 1:length(alldatanow)
            %             if ~lint(tr) && ~all(isnan(eventTime{tr})) % && ~isempty(eventTime{tr}(eventTime{tr}>0))
            if ~all(isnan(eventTime{tr})) % && ~isempty(eventTime{tr}(eventTime{tr}>0))
                %             disp(tr)
                %             tracesa{tr} = NaN(mm+mm2, size(traces{1},2), length(eventInds_f_all_clean{tr}));
                %             nvalida{tr} = NaN(mm+mm2, length(eventInds_f_all_clean{tr}));
                
                %                 licks = eventInds_f_all_clean{tr};
                
                licks = eventInds_f_all_clean_final{tr};
                % only get those licks that have enough number of frames before and after the aligned event.
                licks = licks(licks > nPre  &  licks <= size(traces{tr},1) - nPost);
                if ~isempty(licks)
                    
                    %                     tracesa{tr} = NaN(nPreFrames+nPostFrames+1, size(traces{1},2), length(licks));
                    % loop through licks to align traces(tr) on them one by one.
                    for il = 1:length(licks) % eventInds_f_all_clean{tr})
                        %                     [traceEventAlign, timeEventAlign, nvalidtrs] = triggerAlignTraces(traces(tr), eventInds_f_all_clean{tr}(il), shiftTime, scaleTime, mm, mm2); % frames x units x trials.
                        %                     tracesa{tr}(:,:,il) = traceEventAlign;
                        %                     nvalida{tr}(:,il) = nvalidtrs;
                        
                        [traces_aligned_fut, time_aligned, eventI] =...
                            triggerAlignTraces_prepost(traces(tr), licks(il), nPre, nPost, shiftTime, scaleTime, 1);
                        
                        %                         tracesa{tr}(:,:,il) = traces_aligned_fut; % frames x units x licks
                        tracesa(:,:,il,tr) = traces_aligned_fut; % frames x units x licks x trials
                        
                    end
                    
                    %                 traces0 = repmat(traces(tr), 1, length(eventInds_f_all_clean{tr}));
                    %
                    %                 [traceEventAlign, timeEventAlign, nvalidtrs] = triggerAlignTraces(traces0, eventInds_f_all_clean{tr}, shiftTime, scaleTime, mm, mm2); % frames x units x trials.
                    %                 tracesa{tr} = traceEventAlign;
                    %                 timea{tr} = timeEventAlign;
                end
            end
        end
        
        
        %         Fract_trs_with_licks = nanmean(cellfun(@(x)~isempty(x), tracesa)) % percentage of trials with lick aligned traces.
        
        
        %% For each neuron concatenate all licks of all trials, ie set traces_licks_ful, size: frs x units x licks
        
        traces_licks_ful = NaN(nPre + nPost + 1, size(traces{1},2), size(tracesa,3) * size(tracesa,4));
        
        for in = 1:size(traces{1},2)
            a = squeeze(tracesa(:, in, :, :));
            traces_licks_ful(:,in,:) = reshape(a, size(a,1), []);
        end
        % below is not correct bc eventInds_f_all_clean_final still
        % includes licks that don't satisfy the following: (licks > nPreFrames  &  licks <= size(traces{tr},1) - nPostFrames);
        %         fprintf('Total number of licks in all trials = %i\n', sum(cellfun(@(x)sum(~isnan(x)), eventInds_f_all_clean_final)))
        
        f = find(~isnan(sum(traces{1},1)), 1); % find the 1st non-nan neuron
        fprintf('\ttotal number of licks in all trials = %i\n', sum(~isnan(traces_licks_ful(1,f,:))))
        %         figure; plot(~isnan(nanmean(squeeze(traces_licks_ful(:,f,:)),1)))
        %         a = squeeze(nanmean(traces_licks_ful,1));
        %         aa = sum(~isnan(a),2);
        %         figure; plot(aa)
        
        %{
%         total_num_licks = sum(cellfun(@(x)size(x,2), eventInds_f_all_clean_final))
        a = cellfun(@(x)size(x,3), tracesa);
        total_num_licks = sum(a(cellfun(@(x)~isempty(x), tracesa)))
        
        traces_licks_ful = NaN(nPreFrames + nPostFrames + 1, size(traces{1},2), total_num_licks);
        for in = 1:size(traces{1},2)
            if ~all(isnan(traces{1}(:,in)))
                %             disp(in)
                a = [];
                for tr = 1:size(traces,2)
                    if ~isempty(tracesa{tr})
                        a = cat(2, a, squeeze(tracesa{tr}(:,in,:)));
                    end
                end
                traces_licks_ful(:,in,:) = a; % frames x units x total licks
            end
        end
        %}
        
        
        %% Plot
        
        av = nanmean(traces_licks_ful, 3); % average across licks
        top = nanmean(av,2); % average across neurons.
        tosd = nanstd(av, [], 2);
        tosd = tosd / sqrt(size(av, 2)); % plot se
        
        % figure;
        e = nPre+1; % find(time_aligned >= 0, 1);
        h(i) = boundedline((1:length(top))-e, top, tosd, col{i}, 'alpha');
        boundedline((1:length(top))-e, top, tosd, col{i}, 'alpha');
        hold on
        %     plot([nPreFrames+1 nPreFrames+1], [min(top) max(top)], 'k:')
        plot([0 0], [min(top) max(top)], 'k:')
        xlim([1 nPre+nPost+1]-e)
        
        
    end
end




%% wheel

for i = 1:length(evT) % loop over center, left and right licks.
    
    disp(['------- ', evT{i}, ' -------'])
    
    eventTime = eval(evT{i});
    
    % Take only trials in trs2ana for analysis
    if length(eventTime)>1
        eventTime = eventTime(trs2ana);
    end
    
    
    %% Set eventInds_f_all_clean: for each trial, find imaging frames during which licks happened.
    
    % framesPerTrial = cellfun(@(x)size(x,1), traces);
    % a = cell2mat(eventTime');
    eventInds_f_all = cell(1, length(alldatanow));
    eventInds_f_all(:) = {NaN};
    eventInds_f_all_clean = cell(1, length(alldatanow));
    eventInds_f_all_clean(:) = {NaN};
    % mm = []; % mm2 = [];
    
    disp('setting lick events for each trial....')
    for tr = 1:length(alldatanow)
        et = eventTime{tr}(eventTime{tr}>0)';  % only look at licks that happened after scopeTTL was sent.
        
        if ~all(isnan(et))
            traceTimeVec = repmat(times_wheel(tr), 1, length(et));
            eventInds_f = eventTimeToIdx(et, traceTimeVec); % index of eventTime on traceTimeVec array for each trial.
            %                         eventInds_f = eventTimeToIdx(et, times_wheel); % index of eventTime on traceTimeVec array for each trial.
            
            
            u = unique(eventInds_f);
            u = u(~isnan(u));
            
            eventInds_f_all{tr} = eventInds_f;
            eventInds_f_all_clean{tr} = u; % no rep, no nan
            
            %             mm = max([mm, eventInds_f]);
            %
            %             mm0 = max(framesPerTrial(tr) - eventInds_f);
            %             mm2 = max([mm2, mm0]);
        end
    end
    
    
    %% Take care of excludeLicksPrePost
    
    %%%%% Choose one of the cases below, whether you want to include in
    %%%%% the analysis all licks, or just licks that lack any other
    %%%%% lick during nPre, or during nPost, or during both.
    
    switch excludeLicksPrePost
        case 'none'
            eventInds_f_all_clean_final = eventInds_f_all_clean;
        case 'pre' %         For each trial, exclude licks that are preceded by another lick in <= nPreFrames.
            eventInds_f_all_clean_final = cellfun(@(x) x([0,diff(x) <= nPre]==0), eventInds_f_all_clean, 'uniformoutput', 0);
        case 'post'
            eventInds_f_all_clean_final = cellfun(@(x) x([diff(x) <= nPost, 0]==0), eventInds_f_all_clean, 'uniformoutput', 0);
        case 'both'
            eventInds_f_all_clean_final = cellfun(@(x) x([0,diff(x) <= nPre]==0 & [diff(x) <= nPost, 0]==0), eventInds_f_all_clean, 'uniformoutput', 0);
    end
    
    % old stuff
    % identify trials with licks happening at intervals shorter than nPreFrames, we later exclude them, bc we want to make sure during nPreFrames (ie baseline of the traces) there has not been any licks.
    % a = cellfun(@(x)(diff(x) < nPreFrames), eventInds_f_all_clean,
    % 'uniformoutput', 0);
    %         lint = cellfun(@(x)sum(diff(x) <= nPreFrames), eventInds_f_all_clean); % trials with licks happening with interval < nPreFrames
    % For each trial, exclude licks that are preceded by another lick in <= nPreFrames.
    
    
    
    %% Set the traces
    
    disp('setting traces for the wheel revolution')
    
    subplot(3,1,3)
    
    traces = traces_wheel;
    title('Wheel revolution')
    
    shiftTime = wheelTimeRes / 2;
    scaleTime = wheelTimeRes;
    
    nPre = round(nPreFrames*frameLength/wheelTimeRes);
    nPost = round(nPostFrames*frameLength/wheelTimeRes);
    
    
    
    traces = traces(trs2ana);
    
    
    %% Align traces on licks, ie set tracesa: cell array, each element for 1 trial, and of size frs x units x licks
    
    % clear timea
    %         tracesa = cell(1, length(alldatanow));
    %         nvalida = cell(1, length(alldatanow));
    tracesa = NaN(nPre+nPost+1, size(traces{1},2), max(cellfun(@length, eventInds_f_all_clean_final)), length(alldatanow));
    
    for tr = 1:length(alldatanow)
        %             if ~lint(tr) && ~all(isnan(eventTime{tr})) % && ~isempty(eventTime{tr}(eventTime{tr}>0))
        if ~all(isnan(eventTime{tr})) % && ~isempty(eventTime{tr}(eventTime{tr}>0))
            %             disp(tr)
            %             tracesa{tr} = NaN(mm+mm2, size(traces{1},2), length(eventInds_f_all_clean{tr}));
            %             nvalida{tr} = NaN(mm+mm2, length(eventInds_f_all_clean{tr}));
            
            %                 licks = eventInds_f_all_clean{tr};
            
            licks = eventInds_f_all_clean_final{tr};
            % only get those licks that have enough number of frames before and after the aligned event.
            licks = licks(licks > nPre  &  licks <= size(traces{tr},1) - nPost);
            if ~isempty(licks)
                
                %                     tracesa{tr} = NaN(nPreFrames+nPostFrames+1, size(traces{1},2), length(licks));
                % loop through licks to align traces(tr) on them one by one.
                for il = 1:length(licks) % eventInds_f_all_clean{tr})
                    %                     [traceEventAlign, timeEventAlign, nvalidtrs] = triggerAlignTraces(traces(tr), eventInds_f_all_clean{tr}(il), shiftTime, scaleTime, mm, mm2); % frames x units x trials.
                    %                     tracesa{tr}(:,:,il) = traceEventAlign;
                    %                     nvalida{tr}(:,il) = nvalidtrs;
                    
                    [traces_aligned_fut, time_aligned, eventI] =...
                        triggerAlignTraces_prepost(traces(tr), licks(il), nPre, nPost, shiftTime, scaleTime, 1);
                    
                    %                         tracesa{tr}(:,:,il) = traces_aligned_fut; % frames x units x licks
                    tracesa(:,:,il,tr) = traces_aligned_fut; % frames x units x licks x trials
                    
                end
                
                %                 traces0 = repmat(traces(tr), 1, length(eventInds_f_all_clean{tr}));
                %
                %                 [traceEventAlign, timeEventAlign, nvalidtrs] = triggerAlignTraces(traces0, eventInds_f_all_clean{tr}, shiftTime, scaleTime, mm, mm2); % frames x units x trials.
                %                 tracesa{tr} = traceEventAlign;
                %                 timea{tr} = timeEventAlign;
            end
        end
    end
    
    
    %         Fract_trs_with_licks = nanmean(cellfun(@(x)~isempty(x), tracesa)) % percentage of trials with lick aligned traces.
    
    
    %% For each neuron concatenate all licks of all trials, ie set traces_licks_ful, size: frs x units x licks
    
    traces_licks_ful = NaN(nPre + nPost + 1, size(traces{1},2), size(tracesa,3) * size(tracesa,4));
    
    for in = 1:size(traces{1},2)
        a = squeeze(tracesa(:, in, :, :));
        traces_licks_ful(:,in,:) = reshape(a, size(a,1), []);
    end
    % below is not correct bc eventInds_f_all_clean_final still
    % includes licks that don't satisfy the following: (licks > nPreFrames  &  licks <= size(traces{tr},1) - nPostFrames);
    %         fprintf('Total number of licks in all trials = %i\n', sum(cellfun(@(x)sum(~isnan(x)), eventInds_f_all_clean_final)))
    
    f = find(~isnan(sum(traces{1},1)), 1); % find the 1st non-nan neuron
    fprintf('\ttotal number of licks in all trials = %i\n', sum(~isnan(traces_licks_ful(1,f,:))))
    %         figure; plot(~isnan(nanmean(squeeze(traces_licks_ful(:,f,:)),1)))
    %         a = squeeze(nanmean(traces_licks_ful,1));
    %         aa = sum(~isnan(a),2);
    %         figure; plot(aa)
    
    %{
%         total_num_licks = sum(cellfun(@(x)size(x,2), eventInds_f_all_clean_final))
        a = cellfun(@(x)size(x,3), tracesa);
        total_num_licks = sum(a(cellfun(@(x)~isempty(x), tracesa)))
        
        traces_licks_ful = NaN(nPreFrames + nPostFrames + 1, size(traces{1},2), total_num_licks);
        for in = 1:size(traces{1},2)
            if ~all(isnan(traces{1}(:,in)))
                %             disp(in)
                a = [];
                for tr = 1:size(traces,2)
                    if ~isempty(tracesa{tr})
                        a = cat(2, a, squeeze(tracesa{tr}(:,in,:)));
                    end
                end
                traces_licks_ful(:,in,:) = a; % frames x units x total licks
            end
        end
    %}
    
    
    %% Plot
    
    av = nanmean(traces_licks_ful, 3); % average across licks
    top = nanmean(av,2); % average across neurons.
    tosd = nanstd(traces_licks_ful, [], 3);
    tosd = tosd / sqrt(size(traces_licks_ful, 3)); % plot se    
%     tosd = nanstd(av, [], 2);
%     tosd = tosd / sqrt(size(av, 2)); % plot se
    
    % figure;
    e = nPre+1; % find(time_aligned >= 0, 1);
    h(i) = boundedline((1:length(top))-e, top, tosd, col{i}, 'alpha');
    hold on
    
    %     plot([nPreFrames+1 nPreFrames+1], [min(top) max(top)], 'k:')
    plot([0 0], [min(top) max(top)], 'k:')
    xlim([1 nPre+nPost+1]-e)
    
end

legend(h, evT, 'box', 'off', 'location', 'northeast')






%%
%{
for isub = 1:3 % plot DF, spikes, wheel
    
    subplot(3,1,isub)
    
    switch isub
        case 3 % wheel
            shiftTime = wheelTimeRes / 2;
            scaleTime = wheelTimeRes;
            
            nPre = round(nPreFrames*frameLength/wheelTimeRes);
            nPost = round(nPostFrames*frameLength/wheelTimeRes);
            
        otherwise % imaging
            shiftTime = frameLength / 2;
            scaleTime = frameLength;
            
            nPre = nPreFrames;
            nPost = nPostFrames;
    end
    
    
    %%
    for i = 1:3 % loop over center, left and right licks.
        
        switch isub
            case 1
                traces = alldataDfofGood; % alldataSpikesGood; %  traces to be aligned.
                ylabel('DF/F')
            case 2
                traces = alldataSpikesGood;
                ylabel('Spiking')
            case 3
                traces = traces_wheel;
                ylabel('Wheel revolution')
        end
        
        
        disp(['------- ', evT{i}, ' -------'])
        
        eventTime = eval(evT{i});
        
        % Take only trials in trs2ana for analysis
        if length(eventTime)>1
            eventTime = eventTime(trs2ana);
        end
        traces = traces(trs2ana);
        
        
        %% For each trial, find imaging frames during which licks happened.
        
        % framesPerTrial = cellfun(@(x)size(x,1), traces);
        % a = cell2mat(eventTime');
        eventInds_f_all = cell(1, length(alldatanow));
        eventInds_f_all(:) = {NaN};
        eventInds_f_all_clean = cell(1, length(alldatanow));
        eventInds_f_all_clean(:) = {NaN};
        % mm = []; % mm2 = [];
        for tr = 1:length(alldatanow)
            et = eventTime{tr}(eventTime{tr}>0)';  % only look at licks that happened after scopeTTL was sent.
            
            if ~all(isnan(et))
                switch isub
                    case 3 % wheel
                        traceTimeVec = repmat(times_wheel(tr), 1, length(et));
                        eventInds_f = eventTimeToIdx(et, traceTimeVec); % index of eventTime on traceTimeVec array for each trial.
                        %                         eventInds_f = eventTimeToIdx(et, times_wheel); % index of eventTime on traceTimeVec array for each trial.
                        
                    otherwise
                        a = repmat(alldatanow(tr), 1, length(et));
                        %                         a = repmat(alldatanow(tr), 1, length(eventTime{tr}));
                        traceTimeVec = {a.frameTimes}; % time vector of the trace that you want to realign.
                        eventInds_f = eventTimeToIdx(et, traceTimeVec); % index of eventTime on traceTimeVec array for each trial.
                end
                
                u = unique(eventInds_f);
                u = u(~isnan(u));
                
                eventInds_f_all{tr} = eventInds_f;
                eventInds_f_all_clean{tr} = u; % no rep, no nan
                
                %             mm = max([mm, eventInds_f]);
                %
                %             mm0 = max(framesPerTrial(tr) - eventInds_f);
                %             mm2 = max([mm2, mm0]);
            end
        end
        
        
        %% Take care of excludeLicksPrePost       
       
        %%%%% Choose one of the cases below, whether you want to include in
        %%%%% the analysis all licks, or just licks that lack any other
        %%%%% lick during nPre, or during nPost, or during both.
        
        switch excludeLicksPrePost
            case 'none'
                eventInds_f_all_clean_final = eventInds_f_all_clean;
            case 'pre' %         For each trial, exclude licks that are preceded by another lick in <= nPreFrames.
                eventInds_f_all_clean_final = cellfun(@(x) x([0,diff(x) <= nPre]==0), eventInds_f_all_clean, 'uniformoutput', 0);
            case 'post'
                eventInds_f_all_clean_final = cellfun(@(x) x([diff(x) <= nPost, 0]==0), eventInds_f_all_clean, 'uniformoutput', 0);
            case 'both'
                eventInds_f_all_clean_final = cellfun(@(x) x([0,diff(x) <= nPre]==0 & [diff(x) <= nPost, 0]==0), eventInds_f_all_clean, 'uniformoutput', 0);
        end
        
        % old stuff
        % identify trials with licks happening at intervals shorter than nPreFrames, we later exclude them, bc we want to make sure during nPreFrames (ie baseline of the traces) there has not been any licks.
        % a = cellfun(@(x)(diff(x) < nPreFrames), eventInds_f_all_clean,
        % 'uniformoutput', 0);
        %         lint = cellfun(@(x)sum(diff(x) <= nPreFrames), eventInds_f_all_clean); % trials with licks happening with interval < nPreFrames
        % For each trial, exclude licks that are preceded by another lick in <= nPreFrames.
        
        
        %% Align traces on licks, ie set tracesa: cell array, each element for 1 trial, and of size frs x units x licks
        
        % clear timea
        %         tracesa = cell(1, length(alldatanow));
        %         nvalida = cell(1, length(alldatanow));
        tracesa = NaN(nPre+nPost+1, size(traces{1},2), max(cellfun(@length, eventInds_f_all_clean_final)), length(alldatanow));
        for tr = 1:length(alldatanow)
            %             if ~lint(tr) && ~all(isnan(eventTime{tr})) % && ~isempty(eventTime{tr}(eventTime{tr}>0))
            if ~all(isnan(eventTime{tr})) % && ~isempty(eventTime{tr}(eventTime{tr}>0))
                %             disp(tr)
                %             tracesa{tr} = NaN(mm+mm2, size(traces{1},2), length(eventInds_f_all_clean{tr}));
                %             nvalida{tr} = NaN(mm+mm2, length(eventInds_f_all_clean{tr}));
                
                %                 licks = eventInds_f_all_clean{tr};
                
                licks = eventInds_f_all_clean_final{tr};
                % only get those licks that have enough number of frames before and after the aligned event.
                licks = licks(licks > nPre  &  licks <= size(traces{tr},1) - nPost);
                if ~isempty(licks)
                    
                    %                     tracesa{tr} = NaN(nPreFrames+nPostFrames+1, size(traces{1},2), length(licks));
                    % loop through licks to align traces(tr) on them one by one.
                    for il = 1:length(licks) % eventInds_f_all_clean{tr})
                        %                     [traceEventAlign, timeEventAlign, nvalidtrs] = triggerAlignTraces(traces(tr), eventInds_f_all_clean{tr}(il), shiftTime, scaleTime, mm, mm2); % frames x units x trials.
                        %                     tracesa{tr}(:,:,il) = traceEventAlign;
                        %                     nvalida{tr}(:,il) = nvalidtrs;
                        
                        [traces_aligned_fut, time_aligned, eventI] =...
                            triggerAlignTraces_prepost(traces(tr), licks(il), nPre, nPost, shiftTime, scaleTime, 1);
                        
                        %                         tracesa{tr}(:,:,il) = traces_aligned_fut; % frames x units x licks
                        tracesa(:,:,il,tr) = traces_aligned_fut; % frames x units x licks x trials
                        
                    end
                    
                    %                 traces0 = repmat(traces(tr), 1, length(eventInds_f_all_clean{tr}));
                    %
                    %                 [traceEventAlign, timeEventAlign, nvalidtrs] = triggerAlignTraces(traces0, eventInds_f_all_clean{tr}, shiftTime, scaleTime, mm, mm2); % frames x units x trials.
                    %                 tracesa{tr} = traceEventAlign;
                    %                 timea{tr} = timeEventAlign;
                end
            end
        end
        
        
        %         Fract_trs_with_licks = nanmean(cellfun(@(x)~isempty(x), tracesa)) % percentage of trials with lick aligned traces.
        
        
        %% For each neuron concatenate all licks of all trials, ie set traces_licks_ful, size: frs x units x licks
        
        traces_licks_ful = NaN(nPre + nPost + 1, size(traces{1},2), size(tracesa,3) * size(tracesa,4));
        for in = 1:size(traces{1},2)
            a = squeeze(tracesa(:, in, :, :));
            traces_licks_ful(:,in,:) = reshape(a, size(a,1), []);
        end
        % below is not correct bc eventInds_f_all_clean_final still
        % includes licks that don't satisfy the following: (licks > nPreFrames  &  licks <= size(traces{tr},1) - nPostFrames);
        %         fprintf('Total number of licks in all trials = %i\n', sum(cellfun(@(x)sum(~isnan(x)), eventInds_f_all_clean_final)))
        
        f = find(~isnan(sum(traces{1},1)), 1); % find the 1st non-nan neuron
        fprintf('Total number of licks in all trials = %i\n', sum(~isnan(traces_licks_ful(1,f,:))))
        %         figure; plot(~isnan(nanmean(squeeze(traces_licks_ful(:,f,:)),1)))
        %         a = squeeze(nanmean(traces_licks_ful,1));
        %         aa = sum(~isnan(a),2);
        %         figure; plot(aa)
        
        %{
%         total_num_licks = sum(cellfun(@(x)size(x,2), eventInds_f_all_clean_final))
        a = cellfun(@(x)size(x,3), tracesa);
        total_num_licks = sum(a(cellfun(@(x)~isempty(x), tracesa)))
        
        traces_licks_ful = NaN(nPreFrames + nPostFrames + 1, size(traces{1},2), total_num_licks);
        for in = 1:size(traces{1},2)
            if ~all(isnan(traces{1}(:,in)))
                %             disp(in)
                a = [];
                for tr = 1:size(traces,2)
                    if ~isempty(tracesa{tr})
                        a = cat(2, a, squeeze(tracesa{tr}(:,in,:)));
                    end
                end
                traces_licks_ful(:,in,:) = a; % frames x units x total licks
            end
        end
        %}
        
        
        %% Plot
        
        av = nanmean(traces_licks_ful, 3); % average across licks
        top = nanmean(av,2); % average across neurons.
        tosd = nanstd(av, [], 2);
        tosd = tosd / sqrt(size(av, 2)); % plot se
        
        % figure;
        e = nPre+1; % find(time_aligned >= 0, 1);
        h(i) = boundedline((1:length(top))-e, top, tosd, col{i}, 'alpha');
        
    end
    
    %     plot([nPreFrames+1 nPreFrames+1], [min(top) max(top)], 'k:')
    plot([0 0], [min(top) max(top)], 'k:')
    xlim([1 nPre+nPost+1]-e)
    
end

legend(h, evT, 'box', 'off', 'location', 'northeast')
%}

%%
%{
clearvars -except alldata alldataSpikesGood alldataDfofGood goodinds good_excit good_inhibit outcomes allResp allResp_HR_LR ...
        trs2rmv stimdur stimrate stimtype cb timeNoCentLickOnset timeNoCentLickOffset timeInitTone time1stCenterLick ...
        timeStimOnset timeStimOffset timeCommitCL_CR_Gotone time1stSideTry time1stCorrectTry time1stIncorrectTry timeReward timeCommitIncorrResp time1stCorrectResponse timeStop centerLicks leftLicks rightLicks imfilename
%}
