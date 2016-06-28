N = 10;     % number of neurons
Tmin = 50;  % minimum number of frames
Tmax = 100; % maximum number of frames
bs_true = rand(N, 1);
a_true = rand;
Xs = cell(N, 1);
Ys = cell(N,1);
for j = 1:N
    Xs{j} = gaussFilter(10, randn(randi([Tmin Tmax]), 1)); % create some smoothly varying signal
    Ys{j} = a_true*Xs{j} + bs_true(j) + 0.3*randn(size(Xs{j}));
end
%% solution using the old method
bs1 = nan(N,1);
for j = 1:N
    p = [Xs{j} ones(size(Xs{j}))]\Ys{j};
    a1(j) = p(1);
    bs1(j,1) = p(2);
end 
a1 = mean(a1);
f1 = sum(objFnRegressCommonSlopeModel(Xs, Ys, a1, bs1))./norm(vertcat(Ys{:}))^2*100;  % objective cost in %
aEr1 = norm(a_true - a1)^2./norm(a_true)^2*100;                                       % error % in a 
bEr1 = norm(bs_true - bs1)^2./norm(bs_true)^2*100;                                    % error % in bs 
%% solution using the new method
[a2, bs2] = regressCommonSlopeModel(Xs, Ys);
f2 = sum(objFnRegressCommonSlopeModel(Xs, Ys, a2, bs2))./norm(vertcat(Ys{:}))^2*100;  % objective cost in %
aEr2 = norm(a_true - a2)^2./norm(a_true)^2*100;                                       % error % in a 
bEr2 = norm(bs_true - bs2)^2./norm(bs_true)^2*100;                                    % error % in bs 
%%
[f1 f2]
[aEr1 aEr2]
[bEr1 bEr2]

