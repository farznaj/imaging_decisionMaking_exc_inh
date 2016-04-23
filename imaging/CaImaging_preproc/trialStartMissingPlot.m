% This script helps to identify the trials for which recording trialStart
% went wrong. (due to it being short!). In a different script you will take
% care of them. (here you just identify them).

%% Look at when trialStart and trialCodes happen! 
% figure; plot(trialStart); 
% hold on; plot(1:length(volt), trialCodes)


%%
if isfield(all_data(1).parsedEvents.states, 'trial_start_rot_scope')
    %%   

    % diffs is computed from trialStart and codeTimes.
    % startRotScope_trialCode1_dur is the same thing but computed from
    % bcontrol states.
    % so we expect the two to match.
    
    % indeces 1:length(startOffSamples) and trialNumbers may need more work.
    trialStartOff_trialCodeOn_dur = (startOffSamples + 1) - codeTimes(1:length(startOffSamples)); % interval between the end of trialStart and the beginning of trial code.
    ltrcode = length(trialStartOff_trialCodeOn_dur);
    diffAlign = abs(startRotScope_trialCode1_dur(1:ltrcode) + trialStartOff_trialCodeOn_dur);  % I don't really expect the 2 values to be >1 different.
    firstBad = find(diffAlign > 1, 1);

    figure; plot(startRotScope_trialCode1_dur(1:ltrcode) + trialStartOff_trialCodeOn_dur)
    title(sprintf('First bad trial: %i', firstBad))
    
    
else
    %% use alldata to compute number of times that the animal entered start_rotary_scope

    alldata_len = NaN(1,length(all_data)-1); 
    for tr = 1:length(all_data)-1
        alldata_len(tr) = size(all_data(tr).parsedEvents.states.start_rotary_scope,1);
    end
    
    % Don't do this: since you are using loadBehavData.m to load all_data, the last trial is already removed.
%     alldata_len = NaN(1,length(all_data)-1);
%     for tr = 1:length(all_data)-1
%         alldata_len(tr) = size(all_data(tr).parsedEvents.states.start_rotary_scope,1);
%     end
    
    
    %% use trialStart signal to compute number of times that the animal entered start_rotary_scope
    
    d = diff(istr);
    fn1 = find(d==-1);
    f1 = find(d==1);
    
    len = diff([fn1',f1'], [], 2)'+1;
    
    istrlen = double(istr);
    istrlen(f1) = len;
    istrlen(f1+1) = NaN;
    istrlen(istrlen==0 | isnan(istrlen)) = [];
    
    %%
    figure('name','number of times  mouse entered start_rotary_scope'); hold on; plot(alldata_len), plot(istrlen)
    legend('alldata-computed','trialStart-computed')
    xlabel('trial number')
    
    
    %% figure out the problematic trs!
    %{
    % use the figure plotted above to find out missing trs.
    missingTr = 64;
    %% add the missing trial from trialStart
    istrlen = [istrlen(1:missingTr-1) NaN istrlen(missingTr:end)];

    %% find problemtrs, ie. trs for which at least one of the start_rotarty_scope states was not recorded in trialStart.
    % a trial in which the 1st start_rotary_scope is recorded in trialStart but
    % one of the subsequent ones is missing will be fine for future imaging-behavior alignment.
    trialStartMissing = find(istrlen~=alldata_len);

    dur_s = NaN(1,length(trialStartMissing));
    cnt = 0;
    for it = trialStartMissing
        cnt = cnt+1;
        dur_s(cnt) = diff(all_data(it).parsedEvents.states.start_rotary_scope(1,:));
    end
    %}
        
    
end





%% Now use a different way for finding trs for which trialStart went wrong(less conclusive than what you used above), but still useful:

% find trials for which you find a very long mscan lag, ie trials for which trialStart went wrong.

ff = frame1RelToStartOff;

% add the trial missed from trialStart
% (you can find it by looking at the plot of lag and seeing from where it
% start getting and staying high)
% ff = [ff(1:missingTr-1) NaN ff(missingTr:end)];

mscanLag = NaN(1,nTrials);
for tr = 1:nTrials
    if isfield(all_data(tr).parsedEvents.states, 'trial_start_rot_scope')
        firstStateDur = diff(all_data(tr).parsedEvents.states.trial_start_rot_scope * 1000); % _abs indicates that the time is relative to the beginning of the session (not the begining of the trial).
    else
        firstStateDur = diff(all_data(tr).parsedEvents.states.start_rotary_scope(1,:) * 1000);
    end
    mscanLag(tr) = firstStateDur + ff(tr); % firstStateDur: actual duration of state with trialStart % ff: trialStart duration measured from mscan.
end

figure('name','mscan lag. <-1 and >32 ms are problematic!'); plot(mscanLag)
xlabel('trial number'), ylabel('mscan lag (ms)')
longlagtrs = find(mscanLag<-1 | mscanLag >32) % you may need to modify these numbers. these are just estimated of how much the lag should be.




%%
% trialStartMissing = [34,64,112]; % or customize it, ie remove those problemtrs whose 1st trialStart was fine.

