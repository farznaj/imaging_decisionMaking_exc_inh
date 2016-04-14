% reaction time

successtrs = outcome==1; % this doesn't work, just write a loop.

a = [alldata(1).parsedEvents.states]

getfield(alldata, {[6,10]}, 'parsedEvents', 'states', 'correctlick_again', {2})

a = [alldata.parsedEvents.states];
[alldata([1:4]).parsedEvents.states.correctlick_again(2)]

alldata(200).parsedEvents.states.errorlick_again(2)


%%
% find different trial types: multi-sensory, only visual, only audio

nvis = [all_data.nVisualEvents]';
naud = [all_data.nAuditoryEvents]';

% [(1:length(nvis))' nvis naud eq(nvis, naud)]
    
trs2rmv = [nvis(find(naud~=0 & nvis~=naud)) naud(find(naud~=0 & nvis~=naud))]; % these are trials that different number of events were presented for the audio and visual parts of the multi-sensory stimulus.
alldata(trs2rmv) = [];

%
multisens = (nvis~=0 & naud~=0);
onlyvis = (nvis~=0 & naud==0); 
onlyaud = (nvis==0 & naud~=0); 

[sum(multisens)
sum(onlyvis)
sum(onlyaud)]



%%
% bad days which should not be analyzed.
% 'fn05' , '03-Apr-2015', '06-Apr-2015'
% 'fn06' , '03-Apr-2015', '09-Apr-2015', '14-Apr-2015'
% 'fn04' , '10-Apr-2015', '13-Apr-2015',



%%
%{
allresponse = [alldata.responseSideIndex];
sortrows([ratev' outcome' stimdur' evnum_v' allresponse'],1);
%}
 % Right:2
% if strcmp(alldata(1).highRateChoicePort, 'R')
%     side2plot = 2;
% else
%     side2plot = 1;
% end


%% look at the licks
% all_data(1).parsedEvents.states

% compute time from the first center lick after start tone until go tone. (
% this should be equal to preStimDelay + goTone).

% compute time from the first center lick after start tone until first side
% lick (correct or incorrect).


% time from the first center lick after start tone : (both values are the
% same)
% all_data(1).parsedEvents.states.wait_for_initiation(2)
% all_data(1).parsedEvents.pokes.C(1)
% figure; % isp = 0; isp = isp+1;


%{
tri = 1; % trial number
if ~alldata(tri).didNotInitiate % the trial was initiated.
    initLick = alldata(tri).parsedEvents.states.stim_delay(1);
    alldata(tri).parsedEvents.pokes.C

    responsewin_start = alldata(tri).parsedEvents.states.center_reward(2); 
    
    licks = [0 100 170 230]; 
    trnumv = tri*ones(1,length(licks));
    figure; hold on
    plot(licks, trnumv, 'k.')

end
%}


%%
%{
% days = {'09-Mar-2015','10-Mar-2015','11-Mar-2015','12-Mar-2015','13-Mar-2015', '16-Mar-2015','17-Mar-2015','18-Mar-2015','19-Mar-2015','20-Mar-2015','23-Mar-2015', '24-Mar-2015'};
days = {'12-Mar-2015','13-Mar-2015', '16-Mar-2015','17-Mar-2015','18-Mar-2015','19-Mar-2015','20-Mar-2015','23-Mar-2015', '24-Mar-2015'}; % choose for fn03
% days = {'13-Mar-2015', '16-Mar-2015','17-Mar-2015','18-Mar-2015','19-Mar-2015','20-Mar-2015','23-Mar-2015', '24-Mar-2015'};
% days = {'16-Mar-2015','17-Mar-2015','18-Mar-2015','19-Mar-2015','20-Mar-2015','23-Mar-2015', '24-Mar-2015'};
% days = {'17-Mar-2015','18-Mar-2015','19-Mar-2015','20-Mar-2015','23-Mar-2015', '24-Mar-2015'};
days = {'18-Mar-2015','19-Mar-2015','20-Mar-2015','23-Mar-2015', '24-Mar-2015'}; % choose for fn06
days = {'19-Mar-2015','20-Mar-2015','23-Mar-2015', '24-Mar-2015'}; % choose for fn04
days = {'20-Mar-2015','23-Mar-2015', '24-Mar-2015'};
% days = {'23-Mar-2015', '24-Mar-2015'}; % choose for fn05
%}
