function [slope_common, offsets_ch1] = inhibit_remove_bleedthrough(Ys, Xs, maxIter, onlyRegress)
% Solve the model: red_ch1 = offset + slope * green_ch2
% you need the following vars:
% trace_ch2, trace_ch1

if ~exist('maxIter', 'var')
    maxIter = 1000; %100;
end

if ~exist('onlyRegress', 'var')
    onlyRegress = 0; % if 1, the commonSlope solution wont be performed, and we'll go with simple regression.
end


%% Set the traces

if ~iscell(Xs)
    frs = size(trace_ch2,1);
    nn = size(trace_ch2,2); % number of neurons

    Xs = mat2cell(trace_ch2, frs, ones(1,nn));
    Ys = mat2cell(trace_ch1, frs, ones(1,nn));
else
    nn = length(Xs);
end
    
normterm = cellfun(@norm, Ys)'.^2; % From Gamal: if prediction is impossible, then best values for a and b (in model y = a*x + b) would be 0, hence cost value (sum(y-ax-b)^2) would become sum(y^2).


%% Get the slope and offset for the Regression model

bs1 = nan(nn,1);
a10 = nan(nn,1);
for j = 1:nn
    p = [Xs{j} ones(size(Xs{j}))]\Ys{j};
    a10(j) = p(1); % slope
    bs1(j) = p(2); % offset
end
a1 = mean(a10);


%% Compute cost value for the regression model

objc_regress = objFnRegressCommonSlopeModel(Xs, Ys, a1, bs1);

% normalized cost value per neuron
% normterm = cellfun(@norm, Ys)'.^2; % From Gamal: if prediction is impossible, then best values for a and b (in model y = a*x + b) would be 0, hence cost value (sum(y-ax-b)^2) would become sum(y^2).
objc_regress_n = 100 * objc_regress ./ normterm; % 1: complete failure and 0: complete success

% total cost value
f1 = sum(objc_regress)./norm(vertcat(Ys{:}))^2*100; % norm(vertcat(Ys{:}))^2 is same as sum(normterm). % FN: I think you can alternatively get mean(objc_regress_n) to get mean cost across neurons, but here Gamal is computing pooled cost across neurons.



%%
if onlyRegress==0
    kpdo = 1;
%     maxIter = 1000; % start w 100, if cost of common slope model is more than regress model, increase it to 1000.
    ik = 0;
    
    while kpdo 
        
        ik = ik+1;
        if ik>2, break, end
        
        
        %% Get the slope and offset for the common-slope model
        
        fprintf('Rep %d, Solving common slope model; maxIter= %d...\n', ik, maxIter)
        [slope_common, offsets_ch1] = regressCommonSlopeModel(Xs, Ys, maxIter);
        
        
        %% Compute cost value for the common-slope model
        
        objc_common = objFnRegressCommonSlopeModel(Xs, Ys, slope_common, offsets_ch1);
        
        % normalized cost value per neuron
        %     normterm = cellfun(@norm, Ys)'.^2; % From Gamal: if prediction is impossible, then best values for a and b (in model y = a*x + b) would be 0, hence cost value (sum(y-ax-b)^2) would become sum(y^2).
        objc_common_n = 100 * objc_common ./ normterm; % 1: complete failure and 0: complete success
        
        % total cost value
        cost_common = sum(objc_common)./norm(vertcat(Ys{:}))^2*100;  % objective cost in %
        % f = sum(objFnRegressCommonSlopeModel(Xs, Ys, a, offsets_ch1))./norm(vertcat(Ys{:}))^2*100;  % objective cost in %
        
        
        %% Compare the results of the common-slope model with the regression model
        
        cprintf('blue', 'Perc cost: commonSlope: %.2f . Regress: %.2f\n', [cost_common, f1])
        cprintf('blue', 'Slope: commonSlope: %.2f . Regress: %.2f\n', [slope_common, a1])
        
        if cost_common > 6 %5 
            warning('Cost of common-slope model > 5% ... increasing maxIter to 2000')
            maxIter = 2000;
        elseif cost_common > f1
            warning('Regress model has lower cost that common-slope model. Increasing maxIter to 2000')
            maxIter = 2000;
        else
            kpdo = 0;            
        end
        
    end
    
else
    slope_common = a1;
    offsets_ch1 = bs1;
    
    cost_common = f1;
    objc_common_n = objc_regress_n;
end


%% Plots

figure;
subplot(311), hold on
plot(offsets_ch1), plot(bs1)
legend('commonSlope', 'regress')
ylabel('offset')
title({sprintf('%%cost: commonSlope: %.2f . Regress: %.2f', [cost_common, f1]), sprintf('slope: commonSlope: %.2f . Regress: %.2f', [slope_common, a1])})

subplot(312)
plot(offsets_ch1 - bs1)
legend('commonSlope - regress')
ylabel('offset')

subplot(313), hold on
plot(objc_common_n)
plot(objc_regress_n)
legend('commonSlope', 'regress')
ylabel('Percent cost')
xlabel('Neurons (only good neurons)')


