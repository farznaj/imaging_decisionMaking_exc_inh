tri = 1;

event = all_data(tri).eventDuration;
shorti = all_data(tri).shortInterval;
longi = all_data(tri).longInterval;

%%
nr_shorti = sum(all_data(tri).auditoryIsis==1);
nr_longi = sum(all_data(tri).auditoryIsis==2);

actual_duration = nr_longi * (longi + event) + nr_shorti * (shorti + event) + event;

actual_duration


%% duration of a session
(all_data(end).parsedEvents.states.state_0(2) - all_data(1).parsedEvents.states.state_0(2))/3600

%% inter-trial interval
begt = NaN(1,length(all_data)); % the time when a trial was started.
for tri=1:length(all_data)
    begt(tri) = all_data(tri).parsedEvents.states.state_0(2);
end
iti_all = diff(begt);

figure; plot(iti_all)

%% ending state
endstate_all = cell(1,length(all_data)); % when a trial was started
for tri=1:length(all_data)
    endstate_all{tri} = [all_data(tri).parsedEvents.states.ending_state];
end
unique(endstate_all)

%%
figure; plot([all_data.movementDuration])
figure; plot([all_data.timeInCenter])

timeincenterDur_all = [all_data.timeInCenter];
movementDur_all = [all_data.movementDuration];
success_all = [all_data.success];



