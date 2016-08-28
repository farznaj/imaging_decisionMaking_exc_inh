% Solve the model: red_ch1 = offset + slope * green_ch2

%{
% you need the following vars:
activity_man_eftMask_ch2, activity_man_eftMask_ch1
%}

%% Set the traces

frs = size(activity_man_eftMask_ch2,1);
nn = size(activity_man_eftMask_ch2,2);

Xs = mat2cell(activity_man_eftMask_ch2, frs, ones(1,nn));
Ys = mat2cell(activity_man_eftMask_ch1, frs, ones(1,nn));


%% Get the slope and offset for the common-slope model

[slope_common, offsets_ch1] = regressCommonSlopeModel(Xs, Ys);


%% Compute cost value for the common-slope model

objc_common = objFnRegressCommonSlopeModel(Xs, Ys, slope_common, offsets_ch1);

% normalized cost value per neuron
normterm = cellfun(@norm, Ys)'.^2; % From Gamal: if prediction is impossible, then best values for a and b (in model y = a*x + b) would be 0, hence cost value (sum(y-ax-b)^2) would become sum(y^2).
objc_common_n = 100 * objc_common ./ normterm; % 1: complete failure and 0: complete success

% total cost value
cost_common = sum(objc_common)./norm(vertcat(Ys{:}))^2*100;  % objective cost in %
% f = sum(objFnRegressCommonSlopeModel(Xs, Ys, a, offsets_ch1))./norm(vertcat(Ys{:}))^2*100;  % objective cost in %


%% Get the slope and offset for the Regression model

bs1 = nan(nn,1);
for j = 1:nn
    p = [Xs{j} ones(size(Xs{j}))]\Ys{j};
    a1(j) = p(1);
    bs1(j,1) = p(2);
end
a1 = mean(a1);


%% Compute cost value for the regression model

objc_regress = objFnRegressCommonSlopeModel(Xs, Ys, a1, bs1);

% normalized cost value per neuron
% normterm = cellfun(@norm, Ys)'.^2; % From Gamal: if prediction is impossible, then best values for a and b (in model y = a*x + b) would be 0, hence cost value (sum(y-ax-b)^2) would become sum(y^2).
objc_regress_n = 100 * objc_regress ./ normterm; % 1: complete failure and 0: complete success

% total cost value
f1 = sum(objc_regress)./norm(vertcat(Ys{:}))^2*100; % norm(vertcat(Ys{:}))^2 is same as sum(normterm). % FN: I think you can alternatively get mean(objc_regress_n) to get mean cost across neurons, but here Gamal is computing pooled cost across neurons.


%% Compare the results of the common-slope model with the regression model

cprintf('blue', 'Perc cost: commonSlope: %.2f . Regress: %.2f\n', [cost_common, f1])
cprintf('blue', 'Slope: commonSlope: %.2f . Regress: %.2f\n', [slope_common, a1])

figure; 
subplot(311), hold on
plot(offsets_ch1), plot(bs1)
legend('commonSlope', 'regress')
ylabel('offset')

subplot(312)
plot(offsets_ch1 - bs1)
legend('commonSlope - regress')
ylabel('offset')

subplot(313), hold on
plot(objc_common_n)
plot(objc_regress_n)
legend('commonSlope', 'regress')
ylabel('Percent cost')



