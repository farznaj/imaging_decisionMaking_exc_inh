% same as set_stim, but just the function form of it.
% geenrate the stimulus using a a poisson distribution
% script set_stim_iti does the same, but it generates the stimulus using
% ITIs that are generated from a poisson distribution. 
% this scrip is faster than set_stim_iti

%{
visdur = 1000; % value(WaitDuration) % length of visual stimulation in ms. % in your new protocol visdur will be same as waitdur.
evdur = 5; % event duration in ms
gapdur = 30; % 15
evnum = 5; 28; 8; % 600ms stim: 8-17; 1000ms: 10-24 (?)
max_iei_accepted = 250; % 100 % in ms, no inter event interval will be above this value.
%}

%%
function ieis = set_stim_fn(visdur, evdur, gapdur, evnum, max_iei_accepted)

evdur_temp = 1; % event duration in ms
visdur_temp = visdur - evnum*(evdur+gapdur-evdur_temp);

lambda = evnum/visdur_temp;


%%
vis_stim = 0;
while sum(diff([0 vis_stim 0])==-1)~=evnum
    % generate a poisson sequence.
    evorno = poissrnd(lambda, 1, visdur*10);
    evorno((evorno>1)) = 1; % in case there are some values bigger than 1, turn them to 1
    evorno = evorno(find(evorno,1):end); % we want to have an event at the beginning of the stimulus.
    % figure; plot(evorno)

    % make sure there are no events in the poisson process with
    % iei>max_iei_accepted. if there are any, remove those events(and
    % their ieis.)
    evd = evorno;
    evd(end+1) = 1;
    f1 = find(evd==1);
    evdist = diff(f1)-1;

    f11 = f1(find(evdist > max_iei_accepted-gapdur )); % remember each event (element=1) in the poisson (evorno) represents 1event + 1gap in the vis_stim.
    f12 = f1(find(evdist > max_iei_accepted-gapdur)+1);
    f_ldist = [f11;f12]';
    for fi = 1:size(f_ldist,1)
        evorno(f_ldist(fi,1)+1:f_ldist(fi,2)) = NaN;        
    end
    evorno(isnan(evorno)) = [];
    %

    % set your vis_stim using the poisson sequence.
    vis_stim = zeros(1,visdur); 
    i_poiss = 1; 
    i_vstim = 1;
%         i_poiss1 = 1;
    while i_vstim <= length(vis_stim)-(evdur-1)
        %{
        % here you make sure there are no ieis>max_iei_accepted. but this method is not good bc it results in an increase in the number of ieis at 200ms. 
        if i_poiss > i_poiss1+(max_iei_accepted-gapdur)+1
            i_poiss = i_poiss + find(evorno(i_poiss:i_poiss+500)==1, 1)-1;
        end
        %}
        if evorno(i_poiss)==1 % make sure the event lasts for evdur, and followed by a gap of duration gapdur. 
%                 i_poiss1 = i_poiss;
            vis_stim(i_vstim : i_vstim+(evdur-1)) = 1;
            vis_stim(i_vstim+evdur : i_vstim+(evdur+gapdur-1)) = 0;
            i_vstim = i_vstim+(evdur+gapdur);
        else
            i_vstim = i_vstim+1;
        end
        i_poiss = i_poiss+1; % go through the poisson sequence you generated.
%             [i_poiss1, i_poiss, i_vstim]
%             pause
    end
end
vis_stim(visdur+1:end) = [];

%     plot(vis_stim), % pause
%     numev_all(iit) = sum(diff([0 vis_stim 0])==-1);

%%% find inter-event intervals
a = [1 diff(vis_stim) 1];
ieis = diff(find(a==1))-evdur;

