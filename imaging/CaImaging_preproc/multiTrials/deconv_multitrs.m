function [C_multi_trs, f_multi_trs, P, S_multi_trs] = deconv_multitrs(cs_frtrs, Nnan, YrA, C, f, P, options)


%%
Ntrs = length(cs_frtrs)-1;
method = options.deconv_method;


%% Set YrACL ie C2 + YrA

% C or C2
YrAC = YrA + C; % add C2 to residual. C2 corresponds to running demo_script with P=0 so no time constants are estimated.


%% Set YtrsAll, ie YrA for each trial in a cell, you need this to do estimate_g_multi_trs

YtrsAll = cell(1, length(cs_frtrs)-1);
for itr = 1:Ntrs
    frs = cs_frtrs(itr)+1 : cs_frtrs(itr+1);
    YtrsAll{itr} = YrAC(:,frs); % Ytemp(frs,jj);
end


%% Add NaNs to YrA for ITIs, you need this to run const_foopsi on it.

YrACnan = [];
f_multi_trs = []; % f with nans, so f and C will be the same size.
for itr = 1:Ntrs
    frs = cs_frtrs(itr)+1 : cs_frtrs(itr+1);
    
    YrACnan = [YrACnan, YrAC(:,frs)];
    f_multi_trs = [f_multi_trs, f(frs)];
    if itr < Ntrs
        YrACnan = [YrACnan, NaN(size(YrAC,1), Nnan(itr))];
        f_multi_trs = [f_multi_trs, NaN(1, Nnan(itr))];
    end
end
% size(YrACnan)
K = size(YrACnan,1);
T = size(YrACnan,2); % sum(Nnan) + size(YrA,2); % length of traces after adding nans.


%% Loop over components to do deconvolution

if options.temporal_parallel
    
    C_multi_trs = zeros(K, T);
    S_multi_trs = zeros(K, T);
    btemp = zeros(K,1);
    c1temp = btemp;
    sntemp = btemp;
    gtemp = cell(K,1);
    
    
    %%
    parfor jj = 1:K
        
        Ytrs = cellfun(@(x)x(jj,:)', YtrsAll, 'uniformoutput', 0);
        
        %% Estimate g considering multiple trials
        
        [g,snn] = estimate_g_multi_trials(Ytrs,[],[],1); % provided by Eftychios
        gd = max(roots([1;-g(:)])); % decay time constant for initial concentration
        
        
        %% Do constrained foopsi
        
        [cc,cb,c1,gn,sn,spk] = constrained_foopsi(YrACnan(jj,:),[],[],g,snn,options);
        
        
        %% Set C, S and a few related fields of P
        
        gd_vec = gd.^((0:T-1));
        C_multi_trs(jj,:) = full(cc(:)' + cb + c1*gd_vec);
        S_multi_trs(jj,:) = spk(:)';
        btemp(jj) = cb;
        c1temp(jj) = c1;
        sntemp(jj) = sn;
        gtemp{jj} = gn(:)'; % Eft takes transpose but it makes it inconsist w serial version.
        
        
        %% 
%         this needs work because of parallel updating.
        if mod(jj,10) == 0
            fprintf('Temporal component %i updated \n',jj);
        end
        
    end
    
    
    %%
    %     if strcmpi(method,'constrained_foopsi') || strcmpi(method,'MCMC');
    P.b = num2cell(btemp);
    P.c1 = num2cell(c1temp);
    P.neuron_sn = num2cell(sntemp);
    P.gn = gtemp;
    
    if strcmpi(method,'MCMC');
        P.samples_mcmc(O{jo}) = samples_mcmc; % FN added, a useful parameter to have.
    end
    
    
else % serial updating
    
    % Preallocate C, S, and some fields of P
    
    C_multi_trs = zeros(K, T);
    S_multi_trs = zeros(K, T);
    
    if strcmpi(method,'constrained_foopsi') || strcmpi(method,'MCEM_foopsi')
        P.gn = cell(K,1);
        P.b = cell(K,1);
        P.c1 = cell(K,1);
        P.neuron_sn = cell(K,1);
    end
    
    
    %%
    for jj = 1:K
        
        Ytrs = cellfun(@(x)x(jj,:)', YtrsAll, 'uniformoutput', 0);
        
        
        %% Estimate g considering multiple trials
        
        [g,snn] = estimate_g_multi_trials(Ytrs,[],[],1);
        gd = max(roots([1;-g(:)])); % decay time constant for initial concentration
        
        
        %% Do constrained foopsi
        
        % options.bas_nonneg = 0;
        [cc,cb,c1,gn,sn,spk] = constrained_foopsi(YrACnan(jj,:),[],[],g,snn,options);
        
        
        %% Set C, S and a few related fields of P
        
        gd_vec = gd.^((0:length(cc)-1));
        C_multi_trs(jj,:) = full(cc(:)' + cb + c1*gd_vec);
        S_multi_trs(jj,:) = spk(:)';
        % YrA(:,jj) = YrA(:,jj) - C(jj,:)';
        P.b{jj} = cb;
        P.c1{jj} = c1;
        P.neuron_sn{jj} = sn;
        P.gn{jj} = gn;
        
        
        %%
        if mod(jj,10) == 0
            fprintf('%i out of total %i temporal components updated \n',jj,K);
        end
        
    end
    
end





%%
%{
% g = estimate_g_multi_trials(Ytrs,[],[],1);
gt = nan(100,2);
grt = nan(100,2);
for i=1:100
    [g,snn] = estimate_g_multi_trials(Ytrs,[],[],1);
    gr = roots([1;-g(:)]);
    gt(i,:) = g;
    grt(i,:) = gr;
end
%}