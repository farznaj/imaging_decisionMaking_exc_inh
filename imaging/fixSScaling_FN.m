function [S, peakScalings] = fixSScaling_FN(S, P, scalePeak)
% Adopted from Matt's code: (the nA part is removed bc Eftychios corrected
% that in order_ROIs)
%
% [S, nA] = fixSScaling(S, C, P)
% [S, nA, peakScalings] = fixSScaling(S, C, P, scalePeak)
% 
% Recover scaling factors that are used in order_ROIs. Use them to rescale
% S.
% 
% If scalePeak = 1, S will be further rescaled so that a value of 1 means
% that it evokes a change in C with peak = 1. That is, this scales S such
% that it relates directly to evoked fluorescence. Default scalePeak = 1.
% 
% If using scalePeak = 1, it is recommended to provide C_df instead of C.
% In this case, however, the values of nA returned are useless.
% 
% The actual scaling performed is nA .* peakScalings. But if scalePeak =
% 0, peakScalings will all be 1.


% Tolerance to determine whether we got the scalings right
% tol = 1e-8;

% Check for using C_df, which has an extra entry for the background
if size(S, 1) == length(P.gn) + 1
  S = S(1:end-1, :);
end

if size(S, 1) ~= length(P.gn)
  error('fixSScaling: Do not remove units from S before using fixSScaling');
end

if ~exist('scalePeak', 'var')
  scalePeak = 1;
end


%% Compute the scaling factors

[nUnits, T] = size(S);

% nA = zeros(nUnits, 1);
peakScalings = ones(nUnits, 1);

for u = 1:nUnits
  % Compute G, the matrix such that S = G*C
  g = P.gn{u};
  rg = roots([1; -g(:)]);
  %{
  G = make_G_matrix(T, g);
  
  % Vector that absorbs the onset decay
  gd_vec = max(rg) .^ (0:T-1);
  
  % Reconstruct C
  c = G \ S(u, :)';
  CNew = full(c(:)' + P.b{u} + P.c1{u} * gd_vec);
  
  % Get scaling for each time point
  allNA = C(u, :) ./ CNew;
  
  % Make sure scalings are all the same for this unit
  if any(abs(allNA - allNA(1)) > tol)
    error(['fixSScaling: Not all scaling factors were the same. ' ...
    'This probably indicates that the time constants are not correctly matched with the units']);
  end
  
  nA(u) = mean(allNA);
  %}
  if scalePeak
    taus(1) = -1 / log(rg(1)); % decay time constant
    taus(2) = -1 / log(rg(2));

    % Time lag to peak response (starting from the equation for a
    % second-order autoregressive process):
    %
    % e^(-x/t1) - e^(-x/t2)
    % dy/dx = -1/t1*e^(-x/t1) + 1/t2*e^(-x/t2) = 0
    % t2*e^(-x/t1) = t1*e^(-x/t2)
    % ln(t2) + x/t2 = ln(t1) + x/t1
    % x/t1 - x/t2 = ln(t2) - ln(t1)
    % t2*x - t1*x = t1*t2*(ln(t2) - ln(t1))
    % x = t1*t2*(ln(t2) - ln(t1)) / (t2 - t1)
    peakT = taus(1) * taus(2) * (-log(taus(2)) + log(taus(1))) / (taus(2) - taus(1));
    % Peak response is scaled funny, because autoregressive process makes
    % the value at the first lag come out to 1
    peakVal = exp(-peakT / taus(1)) - exp(-peakT / taus(2));
    initVal = exp(-1 / taus(1)) - exp(-1 / taus(2));
    peakScalings(u) = peakVal / initVal;
  end
end


%% Do the scaling

% S = bsxfun(@times, S, nA .* peakScalings);
S = bsxfun(@times, S, peakScalings);


