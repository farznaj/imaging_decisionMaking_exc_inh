dataPath = '/sonas-hs/churchland/nlsas/data/data';

%% Load A2,b2,C2,f2
fname = 'demo_results_fni17-151102_001.mat';
fname = fullfile(dataPath, 'fni17', 'imaging', '151102', fname);
load(fname, 'A2', 'b2', 'C2', 'f2', 'P')

%% set params and read tiff files into movieMC.
mousename = 'fni17';
imagingFolder = '151102';
mdfFileNumber = 1; % or tif major
ch2read = 2;
convert2double = 1;

movieMC = tifToMat(mousename, imagingFolder, mdfFileNumber, ch2read, convert2double);


%% Reshape Y
[d1,d2,T] = size(movieMC{params.gcampCh});          % dimensions of dataset
d = d1*d2;                                          % total number of pixels
Yr = reshape(movieMC{params.gcampCh},d,T);

%% Update temporal components using 'MCMC' method..
P.method = 'MCMC';            % choice of method for deconvolution
P.temporal_iter = 1;                        % number of iterations for block coordinate descent
% P.fudge_factor = 0.98;                      % fudge factor to reduce time constant estimation bias

[C_mcmc,f_mcmc,Y_res,P_mcmc,S_mcmc] = update_temporal_components(Yr,A2,b2,C2,f2,P);
[C_mcmc_df,~,~] = extract_DF_F(Yr,[A2,b2],[C_mcmc;f_mcmc],S_mcmc, size(A2,2)+1);


%%
save(fname, '-append', 'C_mcmc', 'C_mcmc_df', 'f_mcmc', 'S_mcmc', 'P_mcmc')


%%
%{
nA = sum(A2.^2);
% EP: This gives you the noisy traces for each neuron after the contribution of all the other neurons has been removed.
Y_r = diag(nA(:))\(A2'*(Yr - [A2,b2]*[C2;f2])) + C2; % comp x frames


%%
params.B = 1;
params.Nsamples = 3; % don't use 2, you'll get error in make_mean_sample bc marginalized sampler gets 1 but it fails to compute 
params.p = length(P.g);
[d,T] = size(Yr); % pix x frames

C_mcmc = NaN(size(C2));
S_mcmc = NaN(size(C2));
P_mcmc = struct;
SAMPLES = [];

%%
% r = randperm(size(Y_r,1));
for ii = 1:size(Y_r,1) % 1:size(Y_r,1)
    fprintf('Updating of temporal component %i started \n', ii);
    SAMPLES = [SAMPLES, cont_ca_sampler(Y_r(ii,:), params)];
    
    %
%     SAMPLES = cont_ca_sampler(Y_r(ii,:), params);
    %     C(ii,:) = make_mean_sample(SAMPLES,YrA(:,ii)/nA(ii));
    C_mcmc(ii,:) = make_mean_sample(SAMPLES(ii), Y_r(ii,:)); % FN: C_mcmc = mean(C_rec); where : C_rec(rep,:) = SAMPLES.Cb(rep) + SAMPLES.Am(rep)*Gs + (ge*SAMPLES.Cin(rep,:)');
    S_mcmc(ii,:) = mean(samples_cell2mat(SAMPLES(ii).ss,T));
    %     YrA(:,ii) = YrA(:,ii) - nA(ii)*C(ii,:)';
    P_mcmc.b{ii} = mean(SAMPLES(ii).Cb);
    P_mcmc.c1{ii} = mean(SAMPLES(ii).Cin);
    P_mcmc.neuron_sn{ii} = sqrt(mean(SAMPLES(ii).sn2));
    P_mcmc.gn{ii} = mean(exp(-1./SAMPLES(ii).g));
    %
end


%%
save('demo_results_Fni17_151102_001_001_j7266345.mat', '-append', 'C_mcmc', 'S_mcmc', 'P_mcmc', 'SAMPLES')
% save('demo_results_Fni17_151102_001_001_j7266345.mat', '-append', 'SAMPLES')

%%
% plot_continuous_samples(SAMP,Y);

%}

