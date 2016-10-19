% Identify pmtOffFrames by using any of the traces, eg trace f, C, manual, etc.

pmtOffFrames
sum(pmtOffFrames{1})

%{
figure; hold on
plot(f)
plot([0 length(f)],[th th],'m')
%}


%%
th = mean(f)-3*std(f);
flfrs = [find(f<th,1), find(f<th,1,'last')];
fprintf('1st and last frame of pmt off: %d %d\n', flfrs)

% figure; plot(f<th)

for ch = 1:2
    pmtOffFrames{ch} = (f<th);
end
pmtOffFrames
sum(pmtOffFrames{1})


%% Identify the trials during which pmt was off
% The following is exactly like find([all_data.anyPmtOffFrames]) which gets
% set in mergeActivityIntoAlldata_fn. so we don't set it anymore
% [remember though pmtOffTrials indeces below is on imaged trials; but
% above is on alldata indeces)

%{
itr1 = find((cs_frtrs - flfrs(1))>=0, 1)-1;
itr2 = find((cs_frtrs - flfrs(2))>=0, 1)-1;

fprintf('pmt was off from trial %d to trial %d\n', itr1, itr2)
fprintf('ie from frame %d to frame %d\n', cs_frtrs(itr_1)+1, cs_frtrs(itr_2+1))

% the following frames should match flfrs

pmtOffTrials = itr1 : itr2;
%}

%% Save pmtOffFrame and pmtOffTrials

% save(imfilename, '-append', 'pmtOffFrames') % , 'pmtOffTrials')

%%
load(pnevFil


