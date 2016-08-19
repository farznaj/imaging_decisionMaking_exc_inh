% you need 
%{
activity_man_eftMask_ch2, activity_man_eftMask_ch1, medImage, A, imHeight, imWidth
COMs = fastCOMsA(A, [imHeight, imWidth]); % size(medImage{2})
CC = ROIContoursPnevCC(A, imHeight, imWidth, .95);
%}

%% Find the slope and offset of the model: red = offset + slope * green

% common slope model

frs = size(activity_man_eftMask_ch2,1);
nn = size(activity_man_eftMask_ch2,2);

Xs = mat2cell(activity_man_eftMask_ch2, frs, ones(1,nn));
Ys = mat2cell(activity_man_eftMask_ch1, frs, ones(1,nn));

[slope_common, bs] = regressCommonSlopeModel(Xs, Ys);

objc = objFnRegressCommonSlopeModel(Xs, Ys, slope_common, bs);
f = sum(objc)./norm(vertcat(Ys{:}))^2*100;  % objective cost in %


%% Compare the objective cost of the common slope model with the regression model

% f = sum(objFnRegressCommonSlopeModel(Xs, Ys, a, bs))./norm(vertcat(Ys{:}))^2*100;  % objective cost in %

% regression model
bs1 = nan(nn,1);
for j = 1:nn
    p = [Xs{j} ones(size(Xs{j}))]\Ys{j};
    a1(j) = p(1);
    bs1(j,1) = p(2);
end 
a1 = mean(a1);
f1 = sum(objFnRegressCommonSlopeModel(Xs, Ys, a1, bs1))./norm(vertcat(Ys{:}))^2*100;

fprintf('%.1f  %.1f = Percent cost for regress and common slope models\n', [f, f1])

