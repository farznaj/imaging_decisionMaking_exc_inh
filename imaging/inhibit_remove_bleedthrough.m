% Solve the model: red_ch1 = offset + slope * green_ch2

%{
% you need the following vars:
activity_man_eftMask_ch2, activity_man_eftMask_ch1, aveImage, A, imHeight, imWidth
COMs = fastCOMsA(A, [imHeight, imWidth]); % size(medImage{2})
% CC = ROIContoursPnevCC(A, imHeight, imWidth, .95);
%}

%% Set the traces

frs = size(activity_man_eftMask_ch2,1);
nn = size(activity_man_eftMask_ch2,2);

Xs = mat2cell(activity_man_eftMask_ch2, frs, ones(1,nn));
Ys = mat2cell(activity_man_eftMask_ch1, frs, ones(1,nn));


%% Common-slope model

[slope_common, offsets_ch1] = regressCommonSlopeModel(Xs, Ys);

objc_common = objFnRegressCommonSlopeModel(Xs, Ys, slope_common, offsets_ch1);
cost_common = sum(objc_common)./norm(vertcat(Ys{:}))^2*100;  % objective cost in %
% f = sum(objFnRegressCommonSlopeModel(Xs, Ys, a, offsets_ch1))./norm(vertcat(Ys{:}))^2*100;  % objective cost in %


%% Regression model

bs1 = nan(nn,1);
for j = 1:nn
    p = [Xs{j} ones(size(Xs{j}))]\Ys{j};
    a1(j) = p(1);
    bs1(j,1) = p(2);
end
a1 = mean(a1);
objc_regress = objFnRegressCommonSlopeModel(Xs, Ys, a1, bs1);
f1 = sum(objc_regress)./norm(vertcat(Ys{:}))^2*100;


%% Compare the results of the common slope model with the regression model

cprintf('blue', 'Perc cost: common model: %.2f . Regress model: %.2f\n', [cost_common, f1])
cprintf('blue', 'Slope: common model: %.2f . Regress model: %.2f\n', [slope_common, a1])

figure; 
subplot(311), hold on
plot(offsets_ch1), plot(bs1)
legend('commonSlope', 'regress model')
ylabel('offset')

subplot(312)
plot(offsets_ch1 - bs1)
legend('common - regress')
ylabel('offset')

subplot(313), hold on
plot(objc_common)
plot(objc_regress)
legend('commonSlope', 'regress model')
ylabel('cost of common')

