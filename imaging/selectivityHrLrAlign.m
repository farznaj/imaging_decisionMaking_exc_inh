function [init_spec_aligned, stim_spec_aligned, go_spec_aligned, choice_spec_aligned, rew_spec_aligned, incorr_spec_aligned, ...
        time_alignedi, time_aligneds, time_alignedg, time_alignedc, time_alignedr, time_alignedp] = ...
    selectivityHrLrAligh(init_time, stim_time, go_time, choice_time, rew_time, incorr_time, ...
        init_spec, stim_spec, go_spec, choice_spec, rew_spec, incorr_spec)

% Align traces of all days 

eventIi = cellfun(@(x)find(sign(x)>0,1), init_time); % frame 0 for each day
eventIs = cellfun(@(x)find(sign(x)>0,1), stim_time);
eventIg = cellfun(@(x)find(sign(x)>0,1), go_time);
eventIc = cellfun(@(x)find(sign(x)>0,1), choice_time);
eventIr = cellfun(@(x)find(sign(x)>0,1), rew_time);
eventIp = cellfun(@(x)find(sign(x)>0,1), incorr_time);

[nPostMini, nPreMini, time_alignedi] = setnpostpre(init_time, eventIi);
[nPostMins, nPreMins, time_aligneds] = setnpostpre(stim_time, eventIs);
[nPostMing, nPreMing, time_alignedg] = setnpostpre(go_time, eventIg);
[nPostMinc, nPreMinc, time_alignedc] = setnpostpre(choice_time, eventIc);
[nPostMinr, nPreMinr, time_alignedr] = setnpostpre(rew_time, eventIr);
[nPostMinp, nPreMinp, time_alignedp] = setnpostpre(incorr_time, eventIp);


%%%
init_spec_aligned = cell(1,length(days)); % nan(nPreMin + nPostMin + 1, length(days));
stim_spec_aligned = cell(1,length(days)); 
go_spec_aligned = cell(1,length(days)); 
choice_spec_aligned = cell(1,length(days)); 
rew_spec_aligned = cell(1,length(days)); 
incorr_spec_aligned = cell(1,length(days)); 

for iday = 1:length(days)
    ri = eventIi(iday) - nPreMini : eventIi(iday) + nPostMini;
    rs = eventIs(iday) - nPreMins : eventIs(iday) + nPostMins;
    rg = eventIg(iday) - nPreMing : eventIg(iday) + nPostMing;
    rc = eventIc(iday) - nPreMinc : eventIc(iday) + nPostMinc;
    rr = eventIr(iday) - nPreMinr : eventIr(iday) + nPostMinr;
    rp = eventIp(iday) - nPreMinp : eventIp(iday) + nPostMinp;
    
    init_spec_aligned{iday} = init_spec{iday}(ri);
    stim_spec_aligned{iday} = stim_spec{iday}(rs);
    go_spec_aligned{iday} = go_spec{iday}(rg);
    choice_spec_aligned{iday} = choice_spec{iday}(rc);
    rew_spec_aligned{iday} = rew_spec{iday}(rr);    
    incorr_spec_aligned{iday} = incorr_spec{iday}(rp);
end




%%
function [nPostMins, nPreMins, time_aligneds] = setnpostpre(stim_time, eventIs)

frameLength = 1000 / 30.9;
nPosts = nan(1,length(stim_time));
for iday = 1:length(stim_time)
    nPosts(iday) = length(stim_time{iday}) - eventIs(iday); 
end
nPostMins = min(nPosts);
nPreMins = min(eventIs)-1;
%%%
a = -frameLength * (0:nPreMins); a = a(end:-1:1);
b = frameLength * (1:nPostMins);
time_aligneds = [a,b];

end


end

