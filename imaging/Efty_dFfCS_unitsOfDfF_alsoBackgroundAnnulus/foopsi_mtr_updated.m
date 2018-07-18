function [c,c1,sp,sn,max_val] = foopsi_mtr_updated(y,sn,g,lam_pr,spk_SNR,c1_est)

% The function implements the FOOPSI algorithm in a multi-trial setup. 
% It assumes that the input is DF/F normalized and as such the baseline is
% always constrained to be zero. Moreover, the shape of the calcium
% transient has maximum value 1 and as such the values of the spikes
% amplitudes correspond to the max DF/F change that each spike causes.

% The deconvolution is performed by using the quadratic programming
% function from the matlab optimization toolbox.

% INPUTS:
% y (cell):       % input DF/F traces, each cell entry corresponds to a
%                 % different trial
% sn:             % noise level of trace
% g:              % discrete time constants
% lam_pr:         % false discovery rate
% spk_SNR:        % minimum SNR for each trace
% c1_est:         % flag for estimating an initial value for each trial

% OUTPUTS:
% 

% implementation of multi-trial foopsi based on quad_prog
% Written by Eftychios Pnevmatikakis

% Updated in 02/2018 to incorporate hard thresholds, spike normalizartion, 
% and better code explanation.

    if ~exist('lam_pr','var'); lam_pr = 0.995; end
    if ~exist('spk_SNR','var'); spk_SNR = 0.5; end
    if ~exist('c1_est','var'); c1_est = false; end
    
    Ntr = length(y);                % number of trials    
    T = zeros(Ntr,1);    
    for tr = 1:Ntr
        T(tr) = length(y{tr});      % length of each trial
    end
    sT = sum(T);                    % total length
    gd = max(roots([1,-g(:)']));    % roots of time constants
    if c1_est; c1 = zeros(Ntr,1); end
    sp = zeros(sT,1);
    c = zeros(sT,1);
    cnt = 0;
    z1 = norminv(lam_pr);
    %G = make_G_matrix(1e4,g);
    h = filter(1,[1,-g(:)'],[1;zeros(999,1)]');  % shape of impulse response
    nK = norm(h);                   % norm of impulse response (used for penalizing spikes)             
    max_val = max(h);               % max value of impulse response (used for normalization)
    y_vec = zeros(sT,1);
    bas_est = 0;                    % no estimation of baseline (assumed zero)
    b_lb = zeros(Ntr,1);
    c1 = zeros(Ntr,1);
    for tr = 1:Ntr
        gd_vec = gd.^((0:T(tr)-1)');
        G = make_G_matrix(T(tr),g);
        A = [speye(T(tr)),ones(T(tr),bas_est),repmat(gd_vec,1,c1_est)];
        AA = A'*A;
        Ay = A'*y{tr}(:);        
        wG = sum(G)';
        Imat = [-G,sparse(T(tr),bas_est+c1_est)];
        opts = optimoptions(@quadprog,'Display','none');
        x = quadprog(AA,-Ay+sn*nK*z1*[wG;zeros(bas_est+c1_est,1)],Imat,zeros(T(tr),1),[],[],[-Inf(T(tr),1);b_lb(tr)*ones(bas_est);zeros(c1_est)],[],[],opts);    
        c(cnt+(1:T(tr))) = A*x;        
        sp_temp = G*x(1:T(tr));
        if spk_SNR > 0    % hard threshold small spikes
            sp_temp(sp_temp<spk_SNR*sn/max_val) = 0;
            x(1:T(tr)) = G\sp_temp;
            c(cnt+(1:T(tr))) = A*x;
        end
        sp(cnt+(1:T(tr))) = sp_temp*max_val;
        %if bas_est; b(tr) = x(T(tr)+bas_est); end
        if c1_est; c1(tr) = x(end); end
        y_vec(cnt+(1:T(tr))) = y{tr}(:);
        cnt = cnt + T(tr);
    end        
    sn = norm(y_vec - c)/sqrt(sT);
    c = c';
    sp = sp';
end