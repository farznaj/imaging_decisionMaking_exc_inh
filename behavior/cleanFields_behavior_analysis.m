if isfield(all_data, 'didNotInitiate')
    [all_data.wrongInitiation] = all_data.didNotInitiate;
    all_data = rmfield(all_data, 'didNotInitiate');
end

if ~isfield(all_data, 'didNotSideLickAgain')
    [all_data.didNotSideLickAgain] = deal(NaN);
end

if ~isfield(all_data, 'stimDur_diff')
    [all_data.stimDur_diff] = deal(0);
end

if ~isfield(all_data, 'exclude')
    [all_data.exclude] = deal(NaN);
end

if ~isfield(all_data, 'animalWorking_waitd')
    [all_data.animalWorking_waitd] = deal(NaN);
end

if ~isfield(all_data, 'animalWorking_sideld')
    [all_data.animalWorking_sideld] = deal(NaN);
end

if ~isfield(all_data, 'centerPoke_amount')
    [all_data.centerPoke_amount] = deal(NaN);
end

if ~isfield(all_data, 'centerPoke_when')
    [all_data.centerPoke_when] = deal([NaN NaN]);
end

if ~isfield(all_data, 'percentAllow')
    [all_data.percentAllow] = deal(NaN);
end

if ~isfield(all_data, 'propOnlyAudio')
    [all_data.propOnlyAudio] = deal(NaN);
end

if ~isfield(all_data, 'propOnlyVisual')
    [all_data.propOnlyVisual] = deal(NaN);
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
    
%     for tri = 1:length(all_data)
%         all_data(tri).outcome = outcome(tri);
%     end
end
