function [init_spec_aligned, stim_spec_aligned, go_spec_aligned, choice_spec_aligned, rew_spec_aligned, incorr_spec_aligned, ...
        time_alignedi, time_aligneds, time_alignedg, time_alignedc, time_alignedr, time_alignedp] = ...
    selHrLrAlign(init_time, stim_time, go_time, choice_time, rew_time, incorr_time, ...
        init_spec, stim_spec, go_spec, choice_spec, rew_spec, incorr_spec, regressBins)

% Align traces of all days 

if ~exist('regressBins', 'var')
    regressBins = 3;
end

eventIi = cellfun(@(x)find(sign(x)>0,1), init_time); % frame 0 for each day
eventIs = cellfun(@(x)find(sign(x)>0,1), stim_time);
eventIg = cellfun(@(x)find(sign(x)>0,1), go_time);
eventIc = cellfun(@(x)find(sign(x)>0,1), choice_time);
eventIr = cellfun(@(x)find(sign(x)>0,1), rew_time);
eventIp = cellfun(@(x)find(sign(x)>0,1), incorr_time);

[nPostMini, nPreMini, time_alignedi] = setnpostpre(init_time, eventIi, regressBins);
[nPostMins, nPreMins, time_aligneds] = setnpostpre(stim_time, eventIs, regressBins);
[nPostMing, nPreMing, time_alignedg] = setnpostpre(go_time, eventIg, regressBins);
[nPostMinc, nPreMinc, time_alignedc] = setnpostpre(choice_time, eventIc, regressBins);
[nPostMinr, nPreMinr, time_alignedr] = setnpostpre(rew_time, eventIr, regressBins);
[nPostMinp, nPreMinp, time_alignedp] = setnpostpre(incorr_time, eventIp, regressBins);


%%%
nresamp = size(init_spec{iday},2);
nday = length(init_time);
init_spec_aligned = nan(nPreMini + nPostMini + 1, nresamp, nday); %frames x numBootstrapSamps x numDays
stim_spec_aligned = nan(nPreMins + nPostMins + 1, nresamp, nday); 
go_spec_aligned = nan(nPreMing + nPostMing + 1, nresamp, nday); 
choice_spec_aligned = nan(nPreMinc + nPostMinc + 1, nresamp, nday); 
rew_spec_aligned = nan(nPreMinr + nPostMinr + 1, nresamp, nday); 
incorr_spec_aligned = nan(nPreMinp + nPostMinp + 1, nresamp, nday); 

for iday = 1:nday
    ri = eventIi(iday) - nPreMini : eventIi(iday) + nPostMini;
    rs = eventIs(iday) - nPreMins : eventIs(iday) + nPostMins;
    rg = eventIg(iday) - nPreMing : eventIg(iday) + nPostMing;
    rc = eventIc(iday) - nPreMinc : eventIc(iday) + nPostMinc;
    rr = eventIr(iday) - nPreMinr : eventIr(iday) + nPostMinr;
    rp = eventIp(iday) - nPreMinp : eventIp(iday) + nPostMinp;
    
    init_spec_aligned(:,:,iday) = init_spec{iday}(ri,:);
    stim_spec_aligned(:,:,iday) = stim_spec{iday}(rs,:);
    go_spec_aligned(:,:,iday) = go_spec{iday}(rg,:);
    choice_spec_aligned(:,:,iday) = choice_spec{iday}(rc,:);
    rew_spec_aligned(:,:,iday) = rew_spec{iday}(rr,:);    
    incorr_spec_aligned(:,:,iday) = incorr_spec{iday}(rp,:);
end


%% Set time_aligned (I prefer this to the function below, bc 0 was not at the center of the 3 bins that were averaged.)

% below can be set using any of the days.
time_alignedi = init_time{iday}(ri); 
time_aligneds = stim_time{iday}(rs); 
time_alignedg = go_time{iday}(rg); 
time_alignedc = choice_time{iday}(rc); 
time_alignedr = rew_time{iday}(rr); 
time_alignedp = incorr_time{iday}(rp); 


%%
function [nPostMins, nPreMins, time_aligneds] = setnpostpre(stim_time, eventIs, regressBins)

frameLength = 1000 / 30.9;
nPosts = nan(1,length(stim_time));
for iday = 1:length(stim_time)
    nPosts(iday) = length(stim_time{iday}) - eventIs(iday); 
end
nPostMins = min(nPosts);
nPreMins = min(eventIs)-1;
%%%
a = -(frameLength*regressBins) * (0:nPreMins); a = a(end:-1:1);
b = (frameLength*regressBins) * (1:nPostMins);
time_aligneds = [a,b];

end

% a = -(np.asarray(frameLength*regressBins) * range(nPreMin+1)[::-1])
% b = (np.asarray(frameLength*regressBins) * range(1, nPostMin+1))

end

