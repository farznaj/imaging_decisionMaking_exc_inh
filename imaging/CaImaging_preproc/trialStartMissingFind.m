%% figure out the problematic trs!
% missedTr is the tr for which no trialStart was recorded. trialStartMissing are the trs which had more than one start_rotary_scope but one did not send trialStart signal to mscan.

%%
if isfield(all_data(1).parsedEvents.states, 'trial_start_rot_scope')
    % use the figure plotted above to find out missing trs.
    missedTr = input('what are the missing trs based on the plots? ');
    missedInds = missedTr;
    % missingTr = 64;
    
    
    %%
else
    % add the missing trial from trialStart
    istrlen = [istrlen(1:missedTr-1) NaN istrlen(missedTr:end)];
    
    
    %% find problemtrs, ie. trs for which at least one of the start_rotarty_scope states was not recorded in trialStart.
    
    % a trial in which the 1st start_rotary_scope is recorded in trialStart but
    % one of the subsequent ones is missing will be fine for future imaging-behavior alignment.
    
    missedInds = find(istrlen~=alldata_len)
    
    dur_s = NaN(1,length(missedInds));
    cnt = 0;
    for it = missedInds
        cnt = cnt+1;
        dur_s(cnt) = diff(all_data(it).parsedEvents.states.start_rotary_scope(1,:));
    end
    dur_s
    
    
    %% now use a different way for finding trs for which trialStart went wrong (less conclusive than what you used above), but still useful:
    
    % find trials for which you find a very long mscan lag, ie trials for which trialStart went wrong.
    ff = frame1RelToStartOff;
    
    % add the trial missed from trialStart
    % (you can find it by looking at the plot of lag and seeing from where it
    % start getting and staying high)
    ff = [ff(1:missedTr-1) NaN ff(missedTr:end)];
    
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
    longlagtrs = find(mscanLag<-1 | mscanLag >32); % you may need to modify these numbers. these are just estimated of how much the lag should be.
    
    
    
    %%
    lag_missedTrs = mscanLag(missedInds)
    toremove = missedInds(-1 < lag_missedTrs  &  lag_missedTrs < 33);
    fprintf('Trial %d will be removed from missedInds because its lag is appropriate, so it should have the 1st trialStart.\n', toremove)
    missedInds(-1<lag_missedTrs & lag_missedTrs<33) = []
    
    Q = input('ok with missedInds (enter:yes, 0:no)? ');
    if ~Q
        missedInds = input('enter missedInds ');
    end
    
end

