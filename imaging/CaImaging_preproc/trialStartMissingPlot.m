% this script helps to find the trils for which recording trialStart went
% wrong. (due to it being short!)

%% use alldata to compute number of times that the animal entered start_rotary_scope
alldata_len = NaN(1,length(all_data)-1); for tr=1:length(all_data)-1, alldata_len(tr) = size(all_data(tr).parsedEvents.states.start_rotary_scope,1); end

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




%%%%%%%%%%%%%%%%%%%%%%%%%%
%% now use a different way for finding trs for which trialStart went wrong(less conclusive than what you used above), but still useful:
% find trials for which you find a very long mscan lag, ie trils for which trialStart went wrong.
ff = frame1RelToStartOff;

% add the trial missed from trialStart
% (you can find it by looking at the plot of lag and seeing from where it
% start getting and staying high)
% ff = [ff(1:missingTr-1) NaN ff(missingTr:end)]; 

mscanLag = NaN(1,nTrials); for tr=1:nTrials, mscanLag(tr) = diff(all_data(tr).parsedEvents.states.start_rotary_scope(1,:))*1000 + ff(tr); end

figure('name','mscan lag. <-1 and >32 ms are problematic!'); plot(mscanLag)
xlabel('trial number'), ylabel('mscan lag (ms)')
longlagtrs = find(mscanLag<-1 | mscanLag >32); % you may need to modify these numbers. these are just estimated of how much the lag should be.


%%
% trialStartMissing = [34,64,112]; % or customize it, ie remove those problemtrs whose 1st trialStart was fine.

