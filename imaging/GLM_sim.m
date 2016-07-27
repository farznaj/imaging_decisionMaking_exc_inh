% includes simulation of GLM (stimulus filter and spike history filter).

%%
load('exampleData_FN.mat')
frameLength = 1000/30.9; % sec.

%%

stimCovar = cell(1, length(trial));
wheelCovar = cell(1, length(trial));
lickLeftCovar = cell(1, length(trial));
lickRightCovar = cell(1, length(trial));

covariateAll = cell(1, length(trial));

spAll = cell(1, length(trial));

for itr = 1:length(trial)
    % scale it by stimRate
    stimCovar{itr} = zeros(trial(itr).duration, 1);
    stimCovar{itr}(trial(itr).timeStimOnset : trial(itr).timeStimOffset) = trial(itr).stimRate; % true;
    
    wheelCovar{itr} = trial(itr).wheelRev'; % revolution rel to begining of trial
%     wheelCovar{itr} = [diff(trial(itr).wheelRev), nan]'; % speed
    
    % add offset... figure out which one is choice 
    lickLeftCovar{itr} = zeros(trial(itr).duration, 1);
    lickLeftCovar{itr}(trial(itr).leftLicks) = true;
    
    lickRightCovar{itr} = zeros(trial(itr).duration, 1);
    lickRightCovar{itr}(trial(itr).leftLicks) = true;    
    
    covariateAll{itr} = [wheelCovar{itr}'  stimCovar{itr}'  lickLeftCovar{itr}'  lickRightCovar{itr}'];
   
    
    spAll{itr} = zeros(trial(itr).duration, 1);
    spAll{itr}(trial(itr).sptrain) = true;
end

stimC = cell2mat(stimCovar');
wheelC = cell2mat(wheelCovar');
sp = cell2mat(spAll');


%%
%{
itr = 43;

n = floor(200/frameLength); % ceil(1000/frameLength);
S_stim = makeStimRows(stimCovar{itr}, n); %, flag);
figure; imagesc(S_stim)

% n = 150; % ceil(1000/frameLength);
S_wheel = makeStimRows(wheelCovar{itr}, n); %, flag);
figure; imagesc(S_wheel)

sp = spAll{itr};
%}

%
n = floor(200/frameLength); % ceil(1000/frameLength);
S_stim = makeStimRows(stimC, n); %, flag);
figure; subplot(211), imagesc(S_stim)

n = 50; % ceil(1000/frameLength);
S_wheel = makeStimRows(wheelC, n); %, flag);
subplot(212); imagesc(S_wheel)
% S_wheel(end,:) = [];


%%

X0 = [S_stim , S_wheel];
X = zscore(X0);
% X = bsxfun(@minus, X0, mean(X0));
% X = bsxfun(@rdivide, X, std(X0));

figure; imagesc(X)
figure; imagesc(X'*X)
% figure; plot(sum(X))
any(sum(X)==0)


%%
% X = S_stim;
W = (X'*X) \ (X'*sp);
% W = (X'*X + 10e3*eye(size(X,2))) \ (X'*sp); % if W is noisy, use this to
% penalize parameters.
figure; plot(W)


%%
%{
DTsim = .01; % Bin size for simulating model & computing likelihood (in units of stimulus frames)
nkt = n; % 20;    % Number of time bins in stimulus filter
ttk = [-nkt+1:0]';  % time relative to spike of stim filter taps
ggsim = makeSimStruct_GLM(nkt,DTsim); % Create GLM structure with default params

W = [ggsim.k; .1*ones(n,1)];
W = [ggsim.k; .1*ggsim.k];
figure; plot(W)
%}

%%
sp_hat = X*W;

figure; 
subplot(211), plot(sp)
subplot(212), plot(sp_hat)

% filter spikes

%% bin sp_hat to get probability of spikes

nb = 20;
[n, v, ed] = histcounts(sp_hat, nb);
figure; plot(v(1:end-1),n)
figure; plot(n)

p_sp = nan(1, nb);
for ib = unique(ed)'
    p_sp(ib) = nanmean(sp(ed==ib));
end

figure; plot(p_sp)
% looks like exponential

%% non linearity
n = length(W);
X = makeStimRows(stimC, n); %, flag);

B = glmfit(X, sp, 'poisson', 'link', 'log'); % inverse of exponential is log, poisson is the dist of spike counts.
% B = glmfit(X, sp_exp);%, 'poisson', 'link', 'identity'); % inverse of exponential is log, poisson is the dist of spike counts.

figure; plot(B(2:end))


%% digging a bit into neural response (with Yunshu)

aveFR = nan(1, length(spAll));
aveFR_bl = nan(1, length(spAll));
for itr = 1:length(spAll)
    if ~isempty(spAll{itr})
    aveFR(itr) = mean(spAll{itr}(trial(itr).timeStimOnset: trial(itr).timeStimOffset));
    aveFR_bl(itr) = mean(spAll{itr}(1: 2));
    end
end

sr = [trial.stimRate];
% figure; plot([trial.stimRate], aveFR)

[n,v,ed] = histcounts(sr, 10);


aveFR_sr = nan(1, 10);
for i = unique(ed)
    aveFR_sr(i) = nanmean(aveFR(ed==i));
end

figure; hist(aveFR_bl, 100)
figure; plot(v(1:end-1), aveFR_sr)
median(aveFR_bl)



%% SIMULAITON
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% generate a gaussian filter

n = 20;

mu = -3.5; 
s = 1;
x = linspace(mu-s*5, mu+s*5, n); 
W = gaussmf(x, [s mu]); %/1e5; % gausspdf

figure; plot(W)
size(W)
W = W';


%% generate stimulus vector

stimC = randn(1e4,1); % randn(size(stimC));

% figure; plot(stimC)
% size(stimC)

%
%{
% size(S_stim)

sp_hat_stim = S_stim * W;

size(sp_hat_stim)

figure; plot(sp_hat_stim)
%}


%% convolve stimulus with filter

% Vstm = sameconv(stimC, W); % causal % linear response. X*W (X is the design matrix that includes stimulus at different lags)
linear_xw = filter(W, 1, stimC); % same as above.

% linear_xw = linear_xw - max(linear_xw); % so exp later goes to 1
% v2 = conv(stimC, W, 'same'); % from Jonathan: 'full', take the 1st T bins... maybe also
% flipped
% figure; plot(v2)
% figure; plot(linear_xw)
% figure; plot(linear_xw - v2)

%{
% linear_xw is same as xw
S_stim = makeStimRows(stimC, n); %, flag);
X = S_stim;

xw = X*W;
%}
% W = (X'*X) \ (X'*sp);
% figure; plot(W)
% figure; plot(xw - linear_xw) % they are the same


%% nonlinearity 

% sp_exp = sigmoid(linear_xw);
sp_exp = exp(linear_xw);
% sp_exp = linear_xw;

% figure; plot((sp_exp-min(sp_exp)) / range(sp_exp))
% normalize so max is 1
% sp_exp = (sp_exp-min(sp_exp)) / range(sp_exp); % sp_exp / max(sp_exp);


% figure; plot(sp_exp)


%% poisson process to find spike probability

deltat = .01;
% [sp, pspike] = glm_poiss_binar(sp_exp, deltat);

sp = poissrnd(sp_exp * deltat);
sp0 = sp;


%% plot

figure; 
subplot(411), plot(stimC), a = gca;
subplot(412), plot(linear_xw), a = [a, gca];
subplot(413), plot(sp_exp), a = [a, gca];
% subplot(514), plot(pspike), a = [a, gca];
subplot(414), plot(sp), a = [a, gca]; %ylim([-.1 1.1])
linkaxes(a, 'x')


%% predict W using glmfit and compare with the built W.

n = length(W);
X = makeStimRows(stimC, n); %, flag);

B = glmfit(X, sp, 'poisson', 'link', 'log'); % inverse of exponential is log, poisson is the dist of spike counts.
% B = glmfit(X, sp_exp);%, 'poisson', 'link', 'identity'); % inverse of exponential is log, poisson is the dist of spike counts.

figure; 
subplot(211), plot(W)
subplot(212), plot(B(2:end))


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% spike history
%% make the spike history filter

dt = .24; %.5; .24;

ihbasprs.ncols = 5;  % Number of basis vectors for post-spike kernel
ihbasprs.hpeaks = [.1 2];  % Peak location for first and last vectors
ihbasprs.b = .5;  % How nonlinear to make spacings
ihbasprs.absref = .1; % absolute refractory period 
[iht,ihbas,ihbasis] = makeBasis_PostSpike(ihbasprs,dt);
ih = ihbasis * [-10 -1 0 1 -2]';  % h current % ihbasis*randn(ihbasprs.ncols,1)
figure;
subplot(311); plot(iht, ihbasis);
subplot(312); plot(iht, iht*0, 'k--', iht, ih);
subplot(313); plot(iht, iht*0, 'k--', iht, exp(ih));

length(ih)
Whist = ih;


%% spike history

sp = sp0;

a = (1:length(Whist))';
ra = bsxfun(@(x,y) (x+y), repmat(a,[1,length(sp)-length(Whist)]), (0:length(sp)-length(Whist)-1));

sphist = zeros(size(sp));
linear_aft_sp_hist = zeros(size(sp)); 
sp_exp_hist = zeros(size(sp));

for t = length(Whist)+1 : length(sp)
%     r = t-length(Whist) : t-1; 
    r = ra(:, t-length(Whist)); % just like r above, but I thought faster speed bc I first form ra.
    yhist_tminus1 = sp(r)'; % spikes in the window preceding time t, ie window t-length(Whist) : t-1
    sphist(t) = yhist_tminus1 * Whist;
    
    linear_aft_sp_hist(t) = sphist(t) + linear_xw(t);
    sp_exp_hist(t) = exp(linear_aft_sp_hist(t));    
    sp(t) = poissrnd(sp_exp_hist(t) * deltat);
end


% figure;
% subplot(211), plot(sp), a = gca; title('sp')
% subplot(212), plot(sphist), a = [a, gca]; title('yhist*Whist')
% linkaxes(a, 'x')


% % add spike history to the linear projection
% linear_aft_sp_hist = sphist + linear_xw;
% % do nonlinearity
% sp_exp_hist = exp(linear_aft_sp_hist);
% % poisson process to find spike probability
% deltat = 1;
% [sp_aft_hist, pspike_hist] = glm_poiss_binar(sp_exp_hist, deltat);


%% plot

figure; 
subplot(511), plot(stimC), a = gca;
subplot(512), plot(linear_xw), a = [a, gca];
subplot(513), plot(sp_exp), a = [a, gca];
subplot(514), plot(sp0), a = [a, gca]; % ylim([-.1 1.1])
subplot(515), plot(sp), a = [a, gca]; % ylim([-.1 1.1])
linkaxes(a, 'x')

% figure;
% subplot(411), plot(linear_aft_sp_hist), a = [a, gca];
% subplot(412), plot(sp_exp_hist), a = [a, gca];
% subplot(413), plot(pspike_hist), a = [a, gca];
% subplot(414), plot(sp_aft_hist), a = [a, gca]; ylim([-.1 1.1])
% linkaxes(a, 'x')


%% predict W using glmfit and compare with the built W.

Xstim = makeStimRows(stimC, length(W)); %, flag); % design matrix
% size(Xstim)
Bstim = glmfit(Xstim, sp, 'poisson', 'link', 'log'); % inverse of exponential is log, poisson is the dist of spike counts.

sp_shift = [0; sp(1:end-1)]; %[sp(2:end); nan];
Xhist = makeStimRows(sp_shift, length(Whist));
% size(Xhist)
Bhist = glmfit(Xhist, sp, 'poisson', 'link', 'log'); % inverse of exponential is log, poisson is the dist of spike counts.


X = [Xstim, Xhist];
% size(X)

B = glmfit(X, sp, 'poisson', 'link', 'log'); % inverse of exponential is log, poisson is the dist of spike counts.
% B = glmfit(X, sp_exp);%, 'poisson', 'link', 'identity'); % inverse of exponential is log, poisson is the dist of spike counts.


%% compare original and predicted wights

figure; 
subplot(221), plot(W)
subplot(222), plot((B(2:size(Xstim,2)+1)))
subplot(223), plot(exp(Whist))
subplot(224), plot(exp(B(size(Xhist,2)+2:end)))



figure; 
subplot(221), plot(W)
subplot(222), plot(Bstim(2:end))
subplot(223), plot(Whist)
subplot(224), plot(Bhist(2:end))

