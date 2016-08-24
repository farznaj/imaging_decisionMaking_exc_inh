
%% dataTensor: data tensor at times x neurons x conditions
%% codedParams: predictors matrix of size conditions x predictors. If baseline is one of the parameters one should include it in that matrix
%% Betas: Beta coefficient tensor of size times x neurons x predictors
%% trCountmtx: matrix of size neurons x conditions of number of trial count per condition for each neuron
function [f_t, R2_t, R2_tk] = TDRobjectiveFn(dataTensor, codedParams, Betas, trCountmtx)
[T, N, C] = size(dataTensor);
K = size(codedParams, 2);

f_t = nan(T,1);
R2_t = nan(T,1); % R2 of the model
R2_tk = nan(T, K); % R2 of each of the predictors
sqrtTrCountmtx = sqrt(trCountmtx);
for t = 1:T
    scaledRbar_t = reshape(dataTensor(t, :, :), N, C).*sqrtTrCountmtx; % true average firing rates scaled by the number of trials at time t
    estScaledRbar_t = zeros(N, C); % estimated average firing rates scaled by the number of trials at time t
    for k = 1:K
        Betas_tk = reshape(Betas(t, :, k), N, 1);
        estScaledRbar_tk = (((ones(N, 1)*codedParams(:,k)') .* sqrtTrCountmtx) .* (Betas_tk*ones(C,1)')); % contribution of signal K to estimate scaled firing rates
        estScaledRbar_t = estScaledRbar_t+estScaledRbar_tk;
        R2_tk(t, k) = (estScaledRbar_tk(:)'*estScaledRbar_tk(:))/(scaledRbar_t(:)'*scaledRbar_t(:));
    end
    Er_t = scaledRbar_t - estScaledRbar_t; % Error term
    f_t(t) = (Er_t(:)'*Er_t(:))/(scaledRbar_t(:)'*scaledRbar_t(:)); % objective function
    R2_t(t) = 1 - f_t(t);
end

end

% % % f = norm(Rbar.*sqrt(Tmtx) - ((ones(N, 1)*P(1,:)) .* sqrt(Tmtx)) .* (Beta0*ones(C,1)')...
% % %                  - ((ones(N, 1)*P(2,:)) .* sqrt(Tmtx)) .* (Beta1*ones(C,1)')...
% % %                  - ((ones(N, 1)*P(3,:)) .* sqrt(Tmtx)) .* (Beta2*ones(C,1)')...
% % %                  - ((ones(N, 1)*P(4,:)) .* sqrt(Tmtx)) .* (Beta3*ones(C,1)'),'fro')^2./norm(Rbar.*sqrt(Tmtx),'fro')^2;
             
             
             
             
 