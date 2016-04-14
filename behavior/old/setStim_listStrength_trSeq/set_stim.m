% geenrate the stimulus using a a poisson distribution
% script set_stim_iti does the same, but it generates the stimulus using
% ITIs that are generated from a poisson distribution. 
% this scrip is faster than set_stim_iti

% in your new protocol visdur will be same as waitdur.
visdur = 1000; 1000; % value(WaitDuration) % length of visual stimulation in ms.
evnum = 4; 5; 28; 8; % 600ms stim: 8-17; 1000ms: 10-24 (?)
max_iei_accepted = 250; % 100 % in ms, no inter event interval will be above this value.

evdur = 5; % event duration in ms
gapdur = 32; 30; % 15

evdur_temp = 1; % event duration in ms
% visdur_temp = visdur - evnum*(evdur-evdur_temp); % 600; % value(WaitDuration) % length of visual stimulation in ms.
visdur_temp = visdur - evnum*(evdur+gapdur-evdur_temp);

lambda = evnum/visdur_temp;
%{
% average number of events in 1ms.
% lambda = (min_evnum+max_evnum)/2/visdur; 
% lambda = lambda*gapdur/evdur;
% lambda = .025;
lambda = evnum/visdur; % 8/value(WaitDuration)
%}

%%
min_evnum = round(visdur/(max_iei_accepted+evdur))+1; % this formula works well.
max_evnum = floor(visdur/(evdur+gapdur));

%%
% clear t
% for ii=1:1E2

figure;
numiter = 1; % 1E1;
ieis_all = cell(1,numiter);
ieis_all(:) = {1001};
numev_all = NaN(1,numiter);
t = NaN(1,numiter);
for iit = 1:numiter %:1E3% :1E4
%     tic    
    vis_stim = 0;
%     while max(ieis_all{iit}) > max_iei_accepted
%     while (max(ieis_all{iit})>200 | sum(diff([0 vis_stim 0])==-1)~=evnum) % repeat until vis_stim includes the number of events that you wish.
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
    numev_all(iit) = sum(diff([0 vis_stim 0])==-1);
    
    %% find inter-event intervals
    
    a = [1 diff(vis_stim) 1];
    ieis = diff(find(a==1))-evdur;

    ieis_all{iit} = ieis;
%     end
    
    
%     t(iit) = toc;
end


% end
% %%

% figure; plot(numev_all)
% [min(numev_all) max(numev_all) nanmean(numev_all)]

% figure;
subplot(221)
plot(vis_stim)
xlim([1 length(vis_stim)])
title([numev_all(iit) max(ieis_all{iit})])
% title([nanmean(evorno), nanmean(abs(diff(vis_stim)))/2 , t, numev_all(iit)])

% t =toc



%%% plot distribution of Inter-event intervals.
% figure; 
subplot(222), 
[n,v] = hist(cell2mat(ieis_all),50);
n = n/sum(n);
plot(v,n)

subplot(224),  
[n,v] = hist(cell2mat(cellfun(@(x)x(1:end-1), ieis_all,'uniformoutput',0)),50);
n = n/sum(n);
plot(v,n)




%%

%{

%% a faster way to get high number of events.

% tic
for iter = 1:1E4
%     iter
evnum = 18; % it seems 22 is max for 600ms.
visdur = 600;

extra_gap = evdur+randi(gapdur)-1; % visdur - (evnum*evdur + (evnum-1)*gapdur);

% figure; 
clf
% '_________________'

vis_stim = zeros(1,visdur);
for i = 1 : evdur+gapdur : visdur    
    vis_stim(i : i+evdur-1) = 1;
    vis_stim(i+evdur : i+evdur+gapdur-1) = 0;
end


% while sum(diff([0 vis_stim 0])==-1) ~= evnum
% vis_stim = zeros(1,visdur);
% for i = 1 : evdur+gapdur : visdur    
%     vis_stim(i : i+evdur-1) = 1;
%     vis_stim(i+evdur : i+evdur+gapdur-1) = 0;
% end
% plot(vis_stim)


while sum(diff([0 vis_stim 0])==-1) ~= evnum %+1 % extra_gap>0
    if extra_gap<=0
        extra_gap = evdur+randi(gapdur)-1; % evdur; % *(sum(diff([0 vis_stim 0])==-1) - evnum);
    end
    
    gap_now = randi(extra_gap, 1);
    
    
    ind2r = 1;
    while ind2r<evdur | ind2r>visdur-(evdur-1)
        ind2r = randi(visdur,1);
    end
    
%     while ind2r>visdur-(evdur-1)
%         ind2r = randi(visdur,1);
%     end
    
    ind2r
    vis_stim(ind2r)

%     pause
    
    if vis_stim(ind2r)==1
%         ind = find((vis_stim(ind2r) - vis_stim(ind2r : -1 : ind2r-evdur))==1)
%         vis_stim(ind+1 : ind+evdur + gap_now) = 0;

        iii = 1 - vis_stim(ind2r-(evdur-1) : ind2r+(evdur-1));
        ibeg = ind2r-evdur + find(iii==0,1); % index of the first element of the event.
        ien = ind2r-evdur + find(iii==0,1, 'last');
        gap_now = ien-ibeg-1;
%         vis_stim(ibeg : ien+extra_gap) = 0;
        vis_stim(ibeg : ien) = 0;
        gap_now = ien-ibeg-1;
        
%         vis_stim(ind2r-(evdur-1) : ind2r+(evdur-1)) = 0;
%         extra_gap = extra_gap-(2*evdur-1);
    else
        vis_stim = [vis_stim(1:ind2r) , zeros(1, gap_now) , vis_stim(ind2r+1:end)];
%         extra_gap = extra_gap-gap_now;
        %{
        if vis_stim(ind2r+gap_now-1)==1
%             if gap_now < evdur-1
                vis_stim(ind2r : ind2r+ gap_now-1+ evdur-1) = 0;
                extra_gap = extra_gap-(gap_now-1+ evdur);
%             else
%                 vis_stim(ind2r : ind2r+ gap_now-1+ evdur-1) = 0;
%             end
            
        else
            vis_stim(ind2r : ind2r+gap_now-1) = 0;
            extra_gap = extra_gap-gap_now;
        end
        %}
    end
    extra_gap = extra_gap-gap_now;
    extra_gap
    
%     if extra_gap<0
%         extra_gap = 0;
%     end
%     
%     if sum(diff([0 vis_stim 0])==-1) == evnum && extra_gap==0
%         pause
%     end
%     pause
    
    vis_stim(visdur+1:end) = [];
    if sum(vis_stim(visdur-(evdur-1):end))~=0 && sum(vis_stim(visdur-(evdur-1):end))~=evdur
        vis_stim(visdur-(evdur-1):end) = 0;
    end
    
    
%     hold on; 
%     ha = plot(vis_stim, 'g');
% title(sum(diff([0 vis_stim 0])==-1))
% 
% pause
% delete(ha)

end

% end

hold on; plot(vis_stim, 'g')
title(sum(diff([0 vis_stim 0])==-1))

pause% (.0001)
end
% t = toc





%%






%%

[nanmean(evorno), nanmean(abs(diff(vis_stim)))/2]


%%
figure; plot(vis_stim)
ylim([0 2])


% ieis 
a = [-1 diff(evorno) 1];
fb = find(a==1); 
fall = [find(a==-1)' fb(2:end)'];
diff(fall,[],2)

% 
% 
%     a = diff(vis_stim);
%     ieis = diff(find(a==1))-evdur+1;
%     ieis = [find(vis_stim==1, 1)-1 ieis];
%     ieis = [ieis length(vis_stim) - find(vis_stim==1, 1, 'last')];

%%
% figure; 
clf
evnum = 20;
vis_stim = zeros(1,visdur);
for i = 1 : evdur+gapdur : visdur    
    vis_stim(i : i+evdur-1) = 1;
    vis_stim(i+evdur : i+evdur+gapdur-1) = 0;
end
plot(vis_stim)

% generate variation in the stimulus, by adding 15ms (extra_gap= visdur-actual length of the stimulus) to the gap after a
% random event.
extra_gap = visdur - (max_evnum*evdur + (max_evnum-1)*gapdur);
beg_gap = evdur : evdur+gapdur : visdur-(evdur+gapdur); 
% find the random index for adding the extra_gap
rand_isel = beg_gap(randi(length(beg_gap), 1));

vis_stim = [vis_stim(1:rand_isel) , zeros(1,extra_gap) , vis_stim(rand_isel+1:end-extra_gap)];


beg_ev = find(diff([0 vis_stim])==1); % index of events begining
beg_ev(1) = []; % we don't want to remove the first event.

% find the random index for removing an event
rand_isel = beg_ev(randperm(length(beg_ev), max_evnum-evnum));
% remove the event
for ir = 1:length(rand_isel)
    vis_stim(rand_isel(ir) : rand_isel(ir)+evdur-1) = 0;
end


hold on; plot(vis_stim, 'g')
title(sum(diff([0 vis_stim 0])==-1))


%}


