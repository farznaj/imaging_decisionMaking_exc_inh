% similar to avetrialAlign_plotAve_noTrGroup.m, but for aligning ca imaging
% traces (and wheel traces) on center, left and right licks.


%%
%{
clearvars -except alldata alldataSpikesGood alldataDfofGood goodinds good_excit good_inhibit outcomes allResp allResp_HR_LR ...
        trs2rmv stimdur stimrate stimtype cb timeNoCentLickOnset timeNoCentLickOffset timeInitTone time1stCenterLick ...
        timeStimOnset timeStimOffset timeCommitCL_CR_Gotone time1stSideTry time1stCorrectTry time1stIncorrectTry timeReward timeCommitIncorrResp time1stCorrectResponse timeStop centerLicks leftLicks rightLicks imfilename
%}


%%
nPreFrames = 2;
nPostFrames = 20;
outcome2ana = 'all'; % 'all'; 1: success, 0: failure, -1: early decision, -2: no decision, -3: wrong initiation, -4: no center commit, -5: no side commit
strength2ana = 'all'; % 'all'; 'eary'; 'medium'; 'hard';


%%
evT = {'centerLicks', 'leftLicks', 'rightLicks'};

switch strength2ana
    case 'easy'
        str2ana = (s >= (max(allStrn) - thStimStrength));
    case 'hard'
        str2ana = (s <= thStimStrength);
    case 'medium'
        str2ana = ((s > thStimStrength) & (s < (max(allStrn) - thStimStrength))); % intermediate strength
    otherwise
        str2ana = true(1, length(outcomes));
end

if strcmp(outcome2ana, 'all')
    trs2ana = str2ana;
    fprintf('Analyzing outcome %s, %s strengths, including %i trials.\n', outcome2ana, strength2ana, sum(trs2ana))
else
    trs2ana = (outcomes==outcome2ana) & str2ana;
    fprintf('Analyzing outcome %i, %s strengths, including %i trials.\n', outcome2ana, strength2ana, sum(trs2ana))
end



col = {'k', 'r', 'g'};
h = NaN(1, length(evT));

alldatanow = alldata(trs2ana);
wheelTimeRes = alldata(1).wheelSampleInt;
[traces_wheel, times_wheel] = wheelInfo(alldatanow);


figure;


%%
for isub = 1:3
    
    subplot(3,1,isub)
    
    for i = 1:3
        
        disp(['------- ', evT{i}, ' -------'])
        
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
        
        switch isub
            case 3 % wheel
                shiftTime = wheelTimeRes / 2;
                scaleTime = wheelTimeRes;
            otherwise % imaging
                shiftTime = frameLength / 2;
                scaleTime = frameLength;
        end
        
        
        eventTime = eval(evT{i});        
        
        % Take only trials in trs2ana for analysis
        if length(eventTime)>1
            eventTime = eventTime(trs2ana);
        end
        traces = traces(trs2ana);
        
        
        %% Find frame of licks for each trial.
        
        % framesPerTrial = cellfun(@(x)size(x,1), traces);
        % a = cell2mat(eventTime');
        eventInds_f_all = cell(1, length(alldatanow));
        eventInds_f_all_clean = cell(1, length(alldatanow));
        % mm = [];
        % mm2 = [];
        for tr = 1:length(alldatanow)
            if ~all(isnan(eventTime{tr}))
                
                et = eventTime{tr}(eventTime{tr}>0)';                
                switch isub
                    case 3 % wheel
                        eventInds_f = eventTimeToIdx(et, times_wheel); % index of eventTime on traceTimeVec array for each trial.
                        
                    otherwise
                        a = repmat(alldatanow(tr), 1, length(eventTime{tr}));
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
        
        
        %% Set trials with lick intervals shorter than nPreFrames
        
        % a = cellfun(@(x)(diff(x) < nPreFrames), eventInds_f_all_clean,
        % 'uniformoutput', 0);
        lint = cellfun(@(x)sum(diff(x) < nPreFrames), eventInds_f_all_clean); % trials with licks happening with interval < nPreFrames
        
        
        %% Align traces on licks, ie set tracesa: cell array, each element for 1 trial, and of size frs x units x licks
        
        % clear timea
        tracesa = cell(1, length(alldatanow));
        nvalida = cell(1, length(alldatanow));
        for tr = 1:length(alldatanow)
            if ~lint(tr) && ~all(isnan(eventTime{tr})) % && ~isempty(eventTime{tr}(eventTime{tr}>0))
                %             disp(tr)
                %             tracesa{tr} = NaN(mm+mm2, size(traces{1},2), length(eventInds_f_all_clean{tr}));
                %             nvalida{tr} = NaN(mm+mm2, length(eventInds_f_all_clean{tr}));
                
                licks = eventInds_f_all_clean{tr};
                licks = licks(licks > nPreFrames  &  licks <= size(traces{tr},1) - nPostFrames); % only get those licks that have enough number of frames before and after.
                if ~isempty(licks)
                    
                    tracesa{tr} = NaN(nPreFrames+nPostFrames+1, size(traces{1},2), length(licks));
                    %                 disp(tr)
                    %                 disp(licks)
                    for il = 1:length(licks) % eventInds_f_all_clean{tr})
                        %                     [traceEventAlign, timeEventAlign, nvalidtrs] = triggerAlignTraces(traces(tr), eventInds_f_all_clean{tr}(il), shiftTime, scaleTime, mm, mm2); % frames x units x trials.
                        %                     tracesa{tr}(:,:,il) = traceEventAlign;
                        %                     nvalida{tr}(:,il) = nvalidtrs;
                        
                        [traces_aligned_fut, time_aligned, eventI] =...
                            triggerAlignTraces_prepost(traces(tr), licks(il), nPreFrames, nPostFrames, shiftTime, scaleTime, 1); % frames x units x trials
                        
                        tracesa{tr}(:,:,il) = traces_aligned_fut;
                        
                    end
                    
                    %                 traces0 = repmat(traces(tr), 1, length(eventInds_f_all_clean{tr}));
                    %
                    %                 [traceEventAlign, timeEventAlign, nvalidtrs] = triggerAlignTraces(traces0, eventInds_f_all_clean{tr}, shiftTime, scaleTime, mm, mm2); % frames x units x trials.
                    %                 tracesa{tr} = traceEventAlign;
                    %                 timea{tr} = timeEventAlign;
                end
            end
        end
        
        
        Fract_trs_with_licks = nanmean(cellfun(@(x)~isempty(x), tracesa)) % percentage of trials with lick aligned traces.
        
        
        %% For each neuron concatenate all licks of all trials, ie set traces_licks_ful, size: frs x units x licks
        
        %     total_num_licks = sum(cellfun(@(x)size(x,2), eventInds_f_all_clean(lint==0)))
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
                traces_licks_ful(:,in,:) = a;
            end
        end
        
        
        %% Plot
        
        av = nanmean(traces_licks_ful, 3); % average across licks
        top = nanmean(av,2); % average across neurons.
        tosd = nanstd(av, [], 2);
        tosd = tosd / sqrt(size(av, 2)); % plot se
        
        % figure;
        h(i) = boundedline(1:length(top), top, tosd, col{i}, 'alpha');        
        
    end    
    
    plot([nPreFrames+1 nPreFrames+1],[min(top) max(top)], 'k:')
    xlim([1 nPreFrames+nPostFrames+1])    
end

legend(h, evT, 'box', 'off', 'location', 'northeast')
