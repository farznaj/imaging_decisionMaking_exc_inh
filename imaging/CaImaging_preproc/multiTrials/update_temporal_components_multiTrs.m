function [C,f,P,S,YrA] = update_temporal_components_multiTrs(Y,A,b,Cin,fin,P,options, cs_frtrs)

% update temporal components and background given spatial components
% A variety of different methods can be used and are separated into 2 classes:

% 1-d approaches, where for each component a 1-d trace is computed by removing
% the effect of all the other components and then averaging with the corresponding 
% spatial footprint. Then each trace is denoised. This corresponds to a block-coordinate approach
% 4 different 1-d approaches are included, and any custom method
% can be easily incorporated:
% 'project':                The trace is projected to satisfy the constraints by the (known) calcium indicator dynamics
% 'constrained_foopsi':     The noise constrained deconvolution approach is used. Time constants can be re-estimated (default)
% 'MCEM_foopsi':            Alternating between constrained_foopsi and a MH approach for re-estimating the time constants
% 'MCMC':                   A fully Bayesian method (slowest, but usually most accurate)

% multi-dimensional approaches: (slowest)
% 'noise_constrained': 
% C(j,:) = argmin_{c_j} sum(G*c_j), 
%           subject to:   G*c_j >= 0
%                         ||Y(i,:) - A*C - b*f|| <= sn(i)*sqrt(T)

% The update can happen either in parallel (default) or serial by tuning options.temporal_parallel. 
% In the case of parallel implementation the methods 'MCEM_foopsi' and 'noise_constrained' are not supported

% INPUTS:
% Y:        raw data ( d X T matrix)
% A:        spatial footprints  (d x nr matrix)
% b:        spatial background  (d x 1 vector)
% Cin:      current estimate of temporal components (nr X T matrix)
% fin:      current estimate of temporal background (1 x T vector)
% P:        struct for neuron parameters
% options:  struct for algorithm parameters

% LD:       Lagrange multipliers (needed only for 'noise_constrained' method).
% 
% OUTPUTS:
% C:        temporal components (nr X T matrix)
% f:        temporal background (1 x T vector)
% P:        struct for neuron parameters
% S:        deconvolved activity

% Written by: 
% Eftychios A. Pnevmatikakis, Simons Foundation, 2015

[d,T] = size(Y);
if isempty(P) || nargin < 6
    active_pixels = find(sum(A,2));                                 % pixels where the greedy method found activity
    unsaturated_pixels = find_unsaturatedPixels(Y);                 % pixels that do not exhibit saturation
    options.pixels = intersect(active_pixels,unsaturated_pixels);   % base estimates only on unsaturated, active pixels                
end

defoptions = CNMFSetParms;
if nargin < 7 || isempty(options); options = []; end
if ~isfield(options,'deconv_method') || isempty(options.deconv_method); method = defoptions.deconv_method; else method = options.deconv_method; end  % choose method
if ~isfield(options,'restimate_g') || isempty(options.restimate_g); restimate_g = defoptions.restimate_g; else restimate_g = options.restimate_g; end % re-estimate time constant (only with constrained foopsi)
if ~isfield(options,'temporal_iter') || isempty(options.temporal_iter); ITER = defoptions.temporal_iter; else ITER = options.temporal_iter; end           % number of block-coordinate descent iterations
if ~isfield(options,'bas_nonneg'); options.bas_nonneg = defoptions.bas_nonneg; end
if ~isfield(options,'fudge_factor'); options.fudge_factor = defoptions.fudge_factor; end
if ~isfield(options,'temporal_parallel'); options.temporal_parallel = defoptions.temporal_parallel; end

% FN added the following fields.
if strcmpi(method,'MCMC')
    if ~isfield(options, 'MCMC_B'); options.MCMC_B = 300; end
    if ~isfield(options, 'MCMC_Nsamples'); options.MCMC_Nsamples = 400; end
    if ~isfield(options, 'MCMC_prec'); options.MCMC_prec = 1e-2; end
end

if isfield(P,'interp'); Y_interp = P.interp; else Y_interp = sparse(d,T); end        % missing data
if isfield(P,'unsaturatedPix'); unsaturatedPix = P.unsaturatedPix; else unsaturatedPix = 1:d; end   % saturated pixels

mis_data = find(Y_interp);              % interpolate any missing data before deconvolution
Y(mis_data) = Y_interp(mis_data);

if (strcmpi(method,'noise_constrained') || strcmpi(method,'project')) && ~isfield(P,'g')
    options.flag_g = 1;
    if ~isfield(P,'p') || isempty(P.p); P.p = 2; end; 
    p = P.p;
    P = arpfit(Yr,p,options,P.sn);
    if ~iscell(P.g)
        G = make_G_matrix(T,P.g);
    end
else
    G = speye(T);
end

ff = find(sum(A)==0);
if ~isempty(ff)
    A(:,ff) = [];
    if exist('Cin','var')
        if ~isempty(Cin)
            Cin(ff,:) = [];
        end
    end
end

% estimate temporal (and spatial) background if they are not present
if isempty(fin) || nargin < 5   % temporal background missing
    bk_pix = (sum(A,2)==0);     % pixels with no active neurons
    if isempty(b) || nargin < 3
        fin = mean(Y(bk_pix,:));
        fin = fin/norm(fin);
        b = max(Y*fin',0);
    else
        fin = max(b(bk_pix)'*Y(bk_pix,:),0)/norm(b(bk_pix))^2;
    end
end

if isempty(Cin) || nargin < 4    % estimate temporal components if missing
    Cin = max((A'*A)\(A'*Y - (A'*b)*fin),0);
    ITER = max(ITER,3);
end

if  isempty(b) || isempty(fin) || nargin < 5  % re-estimate temporal background
    if isempty(b) || nargin < 3
        [b,fin] = nnmf(max(Y - A*Cin,0),1);
    else
        fin = max((b'*Y - (b'*A)*Cin)/norm(b)^2,0);
    end
end

saturatedPix = setdiff(1:d,unsaturatedPix);     % remove any saturated pixels
Ysat = Y(saturatedPix,:);
Asat = A(saturatedPix,:);
bsat = b(saturatedPix,:);
Y = Y(unsaturatedPix,:);
A = A(unsaturatedPix,:);
b = b(unsaturatedPix,:);
d = length(unsaturatedPix);

K = size(A,2);
A = [A,b];
S = zeros(size(Cin));
Cin = [Cin;fin];
C = Cin;

if strcmpi(method,'noise_constrained')
    Y_res = Y - A*Cin;
    mc = min(d,15);  % number of constraints to be considered
    LD = 10*ones(mc,K);
else
    nA = sum(A.^2);
    AA = A'*A/spdiags(nA(:),0,length(nA),length(nA));
    YA = Y'*A/spdiags(nA(:),0,length(nA),length(nA));
    YrA = (YA - Cin'*AA);
    if strcmpi(method,'constrained_foopsi') || strcmpi(method,'MCEM_foopsi')
        P.gn = cell(K,1);
        P.b = cell(K,1);
        P.c1 = cell(K,1);           
        P.neuron_sn = cell(K,1);
    end
    % FN added the following if statement
    if strcmpi(method,'MCMC')
        params.B = options.MCMC_B; % 300; % FN modified
        params.Nsamples = options.MCMC_Nsamples; % 400; % FN modified        
        params.prec = options.MCMC_prec; % FN added 
        params.p = P.p; 
    else
        params = [];
    end
end
p = P.p;
if options.temporal_parallel
    for iter = 1:ITER
        [O,lo] = update_order(A(:,1:K));
%         O
        for jo = 1:length(O)
%             jo, O{jo}
            Ytemp = YrA(:,O{jo}(:)) + Cin(O{jo},:)';
            Ctemp = zeros(length(O{jo}),T);
            Stemp = zeros(length(O{jo}),T);
            btemp = zeros(length(O{jo}),1);
            sntemp = btemp;
            c1temp = btemp;
            gtemp = cell(length(O{jo}),1);
            % FN added the part below in order to save SAMPLES as a field of P
            if strcmpi(method,'MCMC')
                clear samples_mcmc
                samples_mcmc(length(O{jo})) = struct();
                [samples_mcmc.Cb] = deal(zeros(params.Nsamples,1));
                [samples_mcmc.Cin] = deal(zeros(params.Nsamples,1));
                [samples_mcmc.sn2] = deal(zeros(params.Nsamples,1));
                [samples_mcmc.ns] = deal(zeros(params.Nsamples,1));
                [samples_mcmc.ss] = deal(cell(params.Nsamples,1));
                [samples_mcmc.ld] = deal(zeros(params.Nsamples,1));
                [samples_mcmc.Am] = deal(zeros(params.Nsamples,1));
                [samples_mcmc.g] = deal(zeros(params.Nsamples,1));
                [samples_mcmc.params] = deal(struct('lam_', [], 'spiketimes_', [], 'A_', [], 'b_', [], 'C_in', [], 'sg', [], 'g', []));
            end
            % framesPerTrialMovie
            Ytrs = cell(1, length(cs_frtrs)-1);
            for itr = 1:length(cs_frtrs)-1
                frs = cs_frtrs(itr)+1 : cs_frtrs(itr+1);
                Ytrs{itr} = Ytemp(frs,:); % Ytemp(frs,jj);
            end
%             clear gtrs
%             for jj = 1:30
%                 Ytrsnow = cellfun(@(x)x(:,jj), Ytrs, 'uniformoutput', 0);
%                 gtrs(:,jj) = estimate_g_multi_trials(Ytrsnow); % ,p,lags)
%             end
            
            parfor jj = 1:length(O{jo})
                if p == 0   % p = 0 (no dynamics assumed)
                    cc = max(Ytemp(:,jj),0);
                    Ctemp(jj,:) = full(cc');
                    Stemp(jj,:) = C(jj,:);
                else
                    switch method
                        case 'project'
                            cc = plain_foopsi(Ytemp(:,jj),G);
                            Ctemp(jj,:) = full(cc');
                            Stemp(jj,:) = Ctemp(jj,:)*G';
                        case 'constrained_foopsi'
                            % framesPerTrialMovie
%                             Ytrs = cell(1, length(cs_frtrs)-1);
%                             for itr = 1:length(cs_frtrs)-1
%                                 frs = cs_frtrs(itr)+1 : cs_frtrs(itr+1);
%                                 Ytrs{itr} = Ytemp(frs,jj);
%                             end
%                             gtrs = estimate_g_multi_trials(Ytrs); % ,p,lags)
%                             jj
                            Ytrsnow = cellfun(@(x)x(:,jj), Ytrs, 'uniformoutput', 0);
                            gtrs = estimate_g_multi_trials(Ytrsnow); % ,p,lags)
                            gd = max(roots([1,-gtrs]));  % decay time constant for initial concentration
%                             gtrs
%                             [cc,cb,c1,gn,sn,spk] = constrained_foopsi(Ytemp(:,jj),[],[],gtrs,[],options);
%                             cc = nan(size(YrA,1),1);
                            Ctrs = cell(size(Ytrsnow));
                            spk = nan(size(YrA,1),1);
                            cball = nan(length(cs_frtrs)-1, 1);
                            c1all = nan(length(cs_frtrs)-1, 1);
                            snall = nan(length(cs_frtrs)-1, 1);
                            gnall = nan(length(cs_frtrs)-1, 2);
                            for itr = 1:length(cs_frtrs)-1
                                frs = cs_frtrs(itr)+1 : cs_frtrs(itr+1);
%                                 [ccnow,cb,c1,gn,sn,spknow] = constrained_foopsi(Ytemp(frs,jj),[],[],gtrs,[],options);

                                [ccnow,cb,c1,gn,sn,spknow] = constrained_foopsi(Ytrsnow{itr},[],[],gtrs,[],options);

                                Tnow = length(Ytrsnow{itr});
%                                 gd = max(roots([1,-gn']));  % decay time constant for initial concentration
                                gd_vec = gd.^((0:Tnow-1));
                                Ctrs{itr} = full(ccnow(:)' + cb + c1*gd_vec);
%                                 Ctemp(jj,frs) = full(ccnow(:)' + cb + c1*gd_vec);
%                                 Stemp(jj,frs) = spknow(:)';
                                
%                                 cc(frs) = ccnow;
                                spk(frs) = spknow;
                                cball(itr) = cb;
                                c1all(itr) = c1;
                                snall(itr) = sn;
                                gnall(itr,:) = gn;
                            end
                            %if restimate_g
%                             [cc,cb,c1,gn,sn,spk] = constrained_foopsi(Ytemp(:,jj),[],[],[],[],options);
                            %else
                            %    [cc,cb,c1,gn,sn,spk] = constrained_foopsi(Ytemp(:,jj)/nA(jj),[],[],P.g,[],options);
                            %end
%                             gd = max(roots([1,-gn']));  % decay time constant for initial concentration
%                             gd_vec = gd.^((0:T-1));
%                             Ctemp(jj,:) = full(cc(:)' + cb + c1*gd_vec);
                            Ctemp(jj,:) = cell2mat(Ctrs)';
                            Stemp(jj,:) = spk(:)';
                            Ytemp(:,jj) = Ytemp(:,jj) - Ctemp(jj,:)';
                            btemp(jj) = cb;
                            c1temp(jj) = c1;
                            sntemp(jj) = sn;
                            gtemp{jj} = gn(:)';
                        case 'MCMC'
                            SAMPLES = cont_ca_sampler(Ytemp(:,jj),params);
                            Ctemp(jj,:) = make_mean_sample(SAMPLES,Ytemp(:,jj));
                            Stemp(jj,:) = mean(samples_cell2mat(SAMPLES.ss,T));
                            btemp(jj) = mean(SAMPLES.Cb);
                            c1temp(jj) = mean(SAMPLES.Cin);
                            sntemp(jj) = sqrt(mean(SAMPLES.sn2));
                            gtemp{jj} = mean(exp(-1./SAMPLES.g))';
                            samples_mcmc(jj) = SAMPLES; % FN added.
                    end
                end
            end
            if p > 0
                if strcmpi(method,'constrained_foopsi') || strcmpi(method,'MCMC');
                    P.b(O{jo}) = num2cell(btemp);
                    P.c1(O{jo}) = num2cell(c1temp);
                    P.neuron_sn(O{jo}) = num2cell(sntemp);
                    for jj = 1:length(O{jo})
                        P.gn(O{jo}(jj)) = gtemp(jj);
                    end
                    YrA = YrA - (Ctemp-C(O{jo}(:),:))'*AA(O{jo}(:),:);
                    C(O{jo}(:),:) = Ctemp;
                    S(O{jo}(:),:) = Stemp;                   
                    if strcmpi(method,'MCMC');
                        P.samples_mcmc(O{jo}) = samples_mcmc; % FN added, a useful parameter to have.
                    end                
                end
            else
                YrA = YrA - (Ctemp-C(O{jo}(:),:))'*AA(O{jo}(:),:);
                C(O{jo}(:),:) = Ctemp;
                S(O{jo}(:),:) = Stemp;
                %YrA = (YA - C'*AA)/spdiags(nA(:),0,length(nA),length(nA));
            end
            fprintf('%i out of %i components updated \n',sum(lo(1:jo)),K);
        end
        ii = K + 1;
        %YrA(:,ii) = YrA(:,ii) + Cin(ii,:)';
        cc = full(max(YrA(:,ii)'+Cin(ii,:),0));
        YrA = YrA - (cc-C(ii,:))'*AA(ii,:);
        C(ii,:) = cc; %full(cc');
        %YrA(:,ii) = YrA(:,ii) - C(ii,:)';
        %YrA(:,end) = (YA(:,end) - C'*AA(:,end)); %/nA(end); %spdiags(nA(:),0,length(nA),length(nA));
        
        %YrA = (YA - Cin'*AA)/spdiags(nA(:),0,length(nA),length(nA));
        if norm(Cin - C,'fro')/norm(C,'fro') <= 1e-3
            % stop if the overall temporal component does not change by much
            break;
        else
            Cin = C;
        end
    end
else
    for iter = 1:ITER
    perm = randperm(K+size(b,2));
        for jj = 1:K
            ii = perm(jj);
            if ii<=K
%                 ii
                if P.p == 0   % p = 0 (no dynamics assumed)
                    YrA(:,ii) = YrA(:,ii) + Cin(ii,:)';
                    cc = max(YrA(:,ii),0);
                    C(ii,:) = full(cc');
                    YrA(:,ii) = YrA(:,ii) - C(ii,:)';
                    S(ii,:) = C(ii,:);
                else
                    switch method
                        case 'project'
                            YrA(:,ii) = YrA(:,ii) + Cin(ii,:)';
                            cc = plain_foopsi(YrA(:,ii),G);
                            C(ii,:) = full(cc');
                            YrA(:,ii) = YrA(:,ii) - C(ii,:)';
                            S(ii,:) = C(ii,:)*G';
                        case 'constrained_foopsi'
                            YrA(:,ii) = YrA(:,ii) + Cin(ii,:)';
                            
                            % framesPerTrialMovie
                            Ytrs = cell(1, length(cs_frtrs)-1);
                            for itr = 1:length(cs_frtrs)-1
                                frs = cs_frtrs(itr)+1 : cs_frtrs(itr+1);
                                Ytrs{itr} = YrA(frs,ii); % Ytemp(frs,jj);
                            end
%                             save Ytrs Ytrs
                            %                             Ytrsnow = cellfun(@(x)x(:,jj), Ytrs, 'uniformoutput', 0);
                            gtrs = estimate_g_multi_trials(Ytrs); % ,p,lags)
                            gd = max(roots([1,-gtrs]));  % decay time constant for initial concentration
%                             gtrs
%                             [cc,cb,c1,gn,sn,spk] = constrained_foopsi(YrA(:,jj),[],[],gtrs,[],options);
%                             cc = nan(size(YrA,1),1);
%                             spk = nan(size(YrA,1),1);
                            cball = nan(length(cs_frtrs)-1, 1);
                            c1all = nan(length(cs_frtrs)-1, 1);
                            snall = nan(length(cs_frtrs)-1, 1);
                            gnall = nan(length(cs_frtrs)-1, 2);
                            for itr = 1:length(cs_frtrs)-1
                                frs = cs_frtrs(itr)+1 : cs_frtrs(itr+1);
                                [ccnow,cb,c1,gn,sn,spknow] = constrained_foopsi(YrA(frs,ii),[],[],gtrs,[],options);
                                
%                                 gd = max(roots([1,-gn']));  % decay time constant for initial concentration
                                Tnow = length(frs);
                                gd_vec = gd.^((0:Tnow-1));
                                C(ii,frs) = full(ccnow(:)' + cb + c1*gd_vec);
                                S(ii,frs) = spknow(:)';
                            
%                                 cc(frs) = ccnow;
%                                 spk(frs) = spknow;
                                cball(itr) = cb;
                                c1all(itr) = c1;
                                snall(itr) = sn;
                                gnall(itr,:) = gn;
                                
                            end                            
                            
%                             if restimate_g
%                                 [cc,cb,c1,gn,sn,spk] = constrained_foopsi(YrA(:,ii),[],[],[],[],options);
%                                 P.gn{ii} = gn;
%                             else
%                                 [cc,cb,c1,gn,sn,spk] = constrained_foopsi(YrA(:,ii),[],[],P.g,[],options);
%                             end
%                             gd = max(roots([1,-gn']));  % decay time constant for initial concentration
%                             gd_vec = gd.^((0:T-1));
%                             C(ii,:) = full(cc(:)' + cb + c1*gd_vec);
%                             S(ii,:) = spk(:)';
                            YrA(:,ii) = YrA(:,ii) - C(ii,:)';
                            P.b{ii} = cb;
                            P.c1{ii} = c1;           
                            P.neuron_sn{ii} = sn;
                            P.gn{ii} = gn;
                        case 'MCEM_foopsi'
                            options.p = length(P.g);
                            YrA(:,ii) = YrA(:,ii) + Cin(ii,:)';
                            [cc,cb,c1,gn,sn,spk] = MCEM_foopsi(YrA(:,ii),[],[],P.g,[],options);
                            gd = max(roots([1,-gn.g(:)']));
                            gd_vec = gd.^((0:T-1));
                            C(ii,:) = full(cc(:)' + cb + c1*gd_vec);
                            S(ii,:) = spk(:)';
                            YrA(:,ii) = YrA(:,ii) - C(ii,:)';
                            P.b{ii} = cb;
                            P.c1{ii} = c1;           
                            P.neuron_sn{ii} = sn;
                            P.gn{ii} = gn.g;
                        case 'MCMC'
                            params.B = 300;
                            params.Nsamples = 400;
                            params.p = P.p; %length(P.g);
                            YrA(:,ii) = YrA(:,ii) + Cin(ii,:)';
                            SAMPLES = cont_ca_sampler(YrA(:,ii),params);
                            C(ii,:) = make_mean_sample(SAMPLES,YrA(:,ii));
                            S(ii,:) = mean(samples_cell2mat(SAMPLES.ss,T));
                            YrA(:,ii) = YrA(:,ii) - C(ii,:)';
                            P.b{ii} = mean(SAMPLES.Cb);
                            P.c1{ii} = mean(SAMPLES.Cin);
                            P.neuron_sn{ii} = sqrt(mean(SAMPLES.sn2));
                            P.gn{ii} = mean(exp(-1./SAMPLES.g));
                            P.samples_mcmc(ii) = SAMPLES; % FN added, a useful parameter to have.
                        case 'noise_constrained'
                            Y_res = Y_res + A(:,ii)*Cin(ii,:);
                            [~,srt] = sort(A(:,ii),'descend');
                            ff = srt(1:mc);
                            [cc,LD(:,ii)] = lagrangian_foopsi_temporal(Y_res(ff,:),A(ff,ii),T*P.sn(unsaturatedPix(ff)).^2,G,LD(:,ii));        
                            C(ii,:) = full(cc');
                            Y_res = Y_res - A(:,ii)*cc';
                            S(ii,:) = C(ii,:)*G';
                    end
                end
            else
                YrA(:,ii) = YrA(:,ii) + Cin(ii,:)';
                cc = max(YrA(:,ii),0);
                C(ii,:) = full(cc');
                YrA(:,ii) = YrA(:,ii) - C(ii,:)';
            end
            if mod(jj,10) == 0
                fprintf('%i out of total %i temporal components updated \n',jj,K);
            end
        end
        if norm(Cin - C,'fro')/norm(C,'fro') <= 1e-3
            % stop if the overall temporal component does not change by much
            break;
        else
            Cin = C;
        end
    end
end
f = C(K+1:end,:);
C = C(1:K,:);
YrA = YrA(:,1:K)';

