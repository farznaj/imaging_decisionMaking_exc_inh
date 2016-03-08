Y = Y(unsaturatedPix,:);
A = [A,b];
Cin = [Cin;fin];

nA = sum(A.^2); % 1 x comp
YrA = Y'*A - Cin'*(A'*A);

YrA(:,ii) = YrA(:,ii) + nA(ii)*Cin(ii,:)'; % fr x comp

SAMPLES = cont_ca_sampler(YrA(:,ii)/nA(ii),params);

C(ii,:) = make_mean_sample(SAMPLES,YrA(:,ii)/nA(ii));
S(ii,:) = mean(samples_cell2mat(SAMPLES.ss,T));
YrA(:,ii) = YrA(:,ii) - nA(ii)*C(ii,:)'; % why is he computing this?

% in his new suggestion how do I get f? (background) below is how it was in
% update_temporal_components
% this part will be performed on the background component (FN).
YrA(:,ii) = YrA(:,ii) + nA(ii)*Cin(ii,:)';
cc = max(YrA(:,ii)/nA(ii),0);
C(ii,:) = full(cc');
YrA(:,ii) = YrA(:,ii) - nA(ii)*C(ii,:)';


%%
nA = sum(A2.^2);
Y_r = diag(nA(:))\(A2'*(Y - [A2,b2]*[C2;f2])) + C2;



% what is B and NSamples about?

% how is C and S different in MCMC vs foopsi, what does he mean by "real"
% spikes? (S_mcmc matches its C_mcmc, but I like C2 better, should I run
% mcmc codes then instead of update_temp right after merging?) -> as a test
% run update_temp one more time to see its resulting C will be similart to
% C_mcmc.

% what is the logic behind project, MCEM_foopsi and noise_constrained
% methods?

% what do MCMC and foopsi stand for?



% Warning: Matrix is singular to working precision.
% > In cont_ca_sampler (line 237)
% Warning: Matrix is singular to working precision.
% > In cont_ca_sampler (line 238)

[temp,~] = HMC_exact2(eye(3), -lb, L, mu_post, 1, Ns, x_in);
[Xs, bounce_count] = HMC_exact2(F, g, M, mu_r, cov, L, initial_X);


initial_X =

1.0e+03 *

Inf
1.5546
0.6536



if isinf(A_) || all(isnan(L(:))) % isinf(A_) is added to avoid hmc reject.
    Am(i) = NaN;
    Cb(i) = NaN;
    Cin(i) = NaN';
else
    [temp,~] = HMC_exact2(eye(3), -lb, L, mu_post, 1, Ns, x_in);
    Am(i) = temp(1,Ns);
    Cb(i) = temp(2,Ns);
    Cin(i) = temp(3,Ns)';
end

%%
nA = sum(A.^2);
YrA = Y'*A - Cin'*(A'*A);

YrA(:,ii) = YrA(:,ii) + nA(ii)*Cin(ii,:)';
SAMPLES = cont_ca_sampler(YrA(:,ii)/nA(ii),params);


