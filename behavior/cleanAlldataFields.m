function all_data = cleanAlldataFields(all_data, alldata_fileNam)
% Take care of absent fields
% If alldata_fileNam is provided all_data will be overwritten after adding
% the absent fields if fields were absent.

savealldata = false;

if ~isfield(all_data, 'waitDurWarmup')
    [all_data.waitDurWarmup] = deal(0);
    savealldata = true;
end

if ~isfield(all_data, 'adaptiveDurs')
    [all_data.adaptiveDurs] = deal(0);
    savealldata = true;
end

%%
if ~isfield(all_data, 'startScopeDur')
%     v = input('What is the value of startScopeDur? ');
    [all_data.startScopeDur] = deal(NaN); % deal(.5); % deal(NaN);
    savealldata = true;
end
   
if ~isfield(all_data, 'stopScopeDur')
    [all_data.stopScopeDur] = deal(NaN);
    savealldata = true;
end
   
if ~isfield(all_data, 'durTrialCode')
    [all_data.durTrialCode] = deal(NaN);
    savealldata = true;
end

%%
if ~isfield(all_data, 'centerAgainDur')
    [all_data.centerAgainDur] = deal(NaN);
    savealldata = true;
end

if ~isfield(all_data, 'sideAgainDur')
    [all_data.sideAgainDur] = deal(NaN);
    savealldata = true;
end    
    
%%
if ~isfield(all_data, 'wheelRev')
    [all_data.wheelRev] = deal(NaN);
    savealldata = true;
end

if ~isfield(all_data, 'wheelSampleInt')
    [all_data.wheelSampleInt] = deal(NaN);
    savealldata = true;
end

if ~isfield(all_data, 'wheelPostTrialToRecord')
    [all_data.wheelPostTrialToRecord] = deal(NaN);
    savealldata = true;
end


%%
if isfield(all_data, 'didNotInitiate')
    [all_data.wrongInitiation] = all_data.didNotInitiate;
    all_data = rmfield(all_data, 'didNotInitiate');
    savealldata = true;
end

if ~isfield(all_data, 'didNotSideLickAgain')
    [all_data.didNotSideLickAgain] = deal(NaN);
    savealldata = true;
end

if ~isfield(all_data, 'stimDur_diff')
    [all_data.stimDur_diff] = deal(0);
    savealldata = true;
end

if ~isfield(all_data, 'exclude')
    [all_data.exclude] = deal(NaN);
    savealldata = true;
end

if ~isfield(all_data, 'animalWorking_waitd')
    [all_data.animalWorking_waitd] = deal(NaN);
    savealldata = true;
end

if ~isfield(all_data, 'animalWorking_sideld')
    [all_data.animalWorking_sideld] = deal(NaN);
    savealldata = true;
end

if ~isfield(all_data, 'centerPoke_amount')
    [all_data.centerPoke_amount] = deal(NaN);
    savealldata = true;
end

if ~isfield(all_data, 'centerPoke_when')
    [all_data.centerPoke_when] = deal([NaN NaN]);
    savealldata = true;
end

if ~isfield(all_data, 'percentAllow')
    [all_data.percentAllow] = deal(NaN);
    savealldata = true;
end

if ~isfield(all_data, 'propOnlyAudio')
    [all_data.propOnlyAudio] = deal(NaN);
    savealldata = true;
end

if ~isfield(all_data, 'propOnlyVisual')
    [all_data.propOnlyVisual] = deal(NaN);
    savealldata = true;
end


if ~isfield(all_data, 'outcome')
    wrongStart = [all_data.wrongInitiation];
    noCenterLickAgain = [all_data.didNotLickagain];
    earlyDecision = [all_data.earlyDecision];
    noDecision = [all_data.didNotChoose];
    noSideLickAgain = [all_data.didNotSideLickAgain];
    success = [all_data.success];
    
    % set outcome for each trial
    %    1: success, 0: failure, -1: early decision, -2: no decision, -3: wrong initiation, 
    %   -4: no center commit, -5: no side commit%
    outcome = zeros(1,length(all_data)); % failure will remain 0, all other trial values will be changes according to their outcome.
    outcome(wrongStart==1) = -3;
    outcome(noCenterLickAgain==1) = -4;
    outcome(earlyDecision==1) = -1;
    outcome(noDecision==1) = -2;
    outcome(noSideLickAgain==1) = -5;
    outcome(success==1) = 1;
    
    r = num2cell(outcome);
    [all_data.outcome] = deal(r{:});
    
    savealldata = true;
%     for tri = 1:length(all_data)
%         all_data(tri).outcome = outcome(tri);
%     end
end



%%    
if exist('alldata_fileNam', 'var') && savealldata
    warning('Overwriting all_data!!')
    save(alldata_fileNam, 'all_data', '-append')
end

