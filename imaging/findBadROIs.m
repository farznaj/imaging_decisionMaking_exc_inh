% Find Eftychios's ROIs that need to be excluded.

% tau seems to be the best measure.
% tau is very related to order_measure, and using f==1 measure is enough
% (you don't need f3==1... it is almost redundant).

% corr: if u want to exclude more comps, use both corr and tau
% measures and a thresh of .4 for corr. You can go down to .3 to exclude fewer comps.
% or just go with tau (u'll exclude fewer neurons).

% remember u can set an upper limit for tau too (too slow time constants).


%%
% [A_or,C_or,S_or,P_or,srt, order_measure] = order_ROIs(A,C,S,P);
% [srt_sn,n_events,n_act] = order_components(YrA,C,[]);

%%
c = corr(C', activity_man); % this is generally higher than corr(C_df', dFOF_man)
cc = diag(c);
% c = corr(C_df', dFOF_man); 
% cc2 = diag(c);

%%
f1 = (tau(:,2) < 100); % 50 % (frameLength*tau(:,2) < 50) | (frameLength*tau(:,2) > 2000)
f2 = cc < .35; % .3; .4; % max([cc,cc2],[],2) < .4; % 
% f3 = full(order_measure < 2000); % very few comps with f1==0 and f3==1... ie if you go with f1==1, then f3 is more of a redundant measure.

% [sum(f1==1) sum(f2==1) sum(f3==1)]
[sum(f1==1) sum(f2==1)]
sum(f1==0 & f2==1)
sum(f1==1 | f2==1)


%% Define neurons to be excluded

badComps = (f1==1 | f2==1); % logical array same size as number of comps in C, with 1s for bad components.
num_fract_badComps = [sum(badComps) mean(badComps)]


%% Append badComps to pnevFile 

% save(pnevFileName, '-append', 'badComps')


%% Plot COMs of bad and good components on the medImage.

COMs = fastCOMsA(A, size(medImage{2}));

% bad components
figure; 
imagesc(medImage{2})
hold on
for rr = find(badComps')
    plot(COMs(rr,2), COMs(rr,1), 'r.')
end

% good components
figure; 
imagesc(medImage{2})
hold on
for rr = find(~badComps')
    plot(COMs(rr,2), COMs(rr,1), 'r.')
end


%%
A = A(:, ~badComps);
C = C(~badComps,:);
C_df = C_df(~badComps,:);


%%
%{
frmv = sum([f1,f2,f3], 2); % or just go with tau and correlation.

s = [];
for i=0:3
    s = [s, sum(frmv==i)];
end
s

excl = find(frmv==2); % bad components
excl = find(frmv==0); % good components.


excl = find(f1==1); % & frmv~=2)
%}

