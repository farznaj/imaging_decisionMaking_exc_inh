% Use this script to understand how some of the matlab functions related to SVM classification work.

%% Train a support vector machine classifier : fitcsvm

% cnam = [0,1]; % LR: negative ; HR: positive
% X = spikeAveEp; Y = choiceVec;

load ionosphere % an example from Matlab
cnam = {'b','g'};

SVMModel = fitcsvm(X, Y, 'standardize', 1, 'ClassNames', cnam); % 'KernelFunction'. 'BoxConstraint'


% svmtrain: compute coefficient and bias
%{
SVMStruct = svmtrain(spikeAveEp, choiceVec, 'autoscale', 1);

Bsvmt = SVMStruct.Alpha' * SVMStruct.SupportVectors;
bias = SVMStruct.Bias;
% bias
% figure; hold on; plot(Bsvmt)
% G = svmclassify(SVMStruct, spikeAveEp);
% find(abs(choiceVec - G)>0)
%}


%% How SVMModel.Beta is computed?

% SVMModel.Beta is the coefficients of hyperplane % computed as :
% alpha times y times x (y the response vector with + and -  values; x the support
% vectores; alpha the Lagrangian multipliers.)

al = SVMModel.Alpha .* SVMModel.SupportVectorLabels; % alpha times y
av = al' * SVMModel.SupportVectors; % w = alpha times y times x
% av = av ./SVMModel.Sigma; % you don't need it bc when standardize is on, matlab already adjusts the Alpha output of svm

b = SVMModel.Bias;
% b = b + av * (-SVMModel.Mu'); % you don't need it bc when standardize is on, matlab already adjusts the Alpha output of svm

% [b, SVMModel.Bias]
figure; plot(av)
hold on, plot(SVMModel.Beta)


%% SVMModel.Prior is the probability of each choice in the Y response vector

fprintf('Prior probs = %.3f  %.3f\n', SVMModel.Prior)
isequal([nanmean(SVMModel.Y==0)  nanmean(SVMModel.Y==1)],  SVMModel.Prior)

% SVMModel.SupportVectorLabels
%{
Y = SVMModel.Y; % choiceVec(~isnan(choiceVec));
Y(Y==cnam(1)) = -1;  % 1st element is negative
Y(Y==cnam(2)) = 1;  % 2nd element is positive
isequal(SVMModel.SupportVectorLabels, Y(SVMModel.IsSupportVector))
%}


%% fitcsvm normalizes Weights so that the elements of W within a particular class sum up to the prior probability of that class.

% The following are almost idential:
[sum(SVMModel.W(SVMModel.Y==0))  sum(SVMModel.W(SVMModel.Y==1))] - SVMModel.Prior


%% Score : the score that an observation belongs to a particular class.

% "predict" gives you the scores; but if its input classification model has
% a ScoreTransform function (ie function to map scores to posterior
% probs) it will compute posterior probabilities instead of scores.

[label, score] = resubPredict(SVMModel);
% [label, score] = predict(SVMModel, SVMModel.X); % does the same thing as the code above.


%% How are scores computed? f(x)=(x/s)'*beta + b.

x = bsxfun(@minus, SVMModel.X, SVMModel.Mu);
x = bsxfun(@rdivide, x, SVMModel.Sigma);
x = x / SVMModel.KernelParameters.Scale;
s = x * SVMModel.Beta + SVMModel.Bias;
% mean(abs(s - score(:,2))) 

figure; hold on, plot([s, score(:,2)]), plot([0 size(score,1)], [0 0])
xlabel('Trial')
ylabel('Score of belonging to the positive class')


%% Compute posterior probabilities of training data belonging to the positive class

% If SVMModel is a ClassificationSVM classifier, then the software
% estimates the optimal transformation function by 10-fold cross validation
% as outlined in [1]. Otherwise, SVMModel must be a
% ClassificationPartitionedModel classifier. SVMModel specifies the
% cross-validation method.

[ScoreSVMModel, ScoreParameters] = fitPosterior(SVMModel); % or fitSVMPosterior
[~,postProbs] = resubPredict(ScoreSVMModel);
% [~,postProbs] = predict(ScoreSVMModel, SVMModel.X); % does the same thing as the code above.

table(SVMModel.Y(1:10), label(1:10), score(1:10,2), postProbs(1:10,2),'VariableNames',...
    {'TrueLabel','PredictedLabel','Score','PosteriorProbability'})

%{
CompactSVMModel = compact(SVMModel);
r = randsample(size(SVMModel.X,1), 10);
[ScoreCSVMModel,ScoreParameters] = fitPosterior(CompactSVMModel, SVMModel.X(r,:), SVMModel.Y(r));

[labels, postProbs] = predict(ScoreCSVMModel, SVMModel.X);
%}

figure; plot(postProbs(:,2))
hold on, plot([0 size(postProbs,1)], [.5 .5])
xlabel('Trial')
ylabel('Post prob of belonging to the positive class')

% if you have been careful with using equal number of trials for hr and lr,
% their priors will be .5 and the posterios and scores will follow the same
% trend, ie positie class (in our case choiceVec==1) will have score>0 and
% posterior>.5
if ~isequal(score>0 , postProbs>.5) | ~isequal(score>0 , postProbs>.5), 
    error('something wrong with score and postProb'),end


%% How posterior probs are computed?

% remember postProb wont be exactly same as pp bc matlab computes postProb
% by doing 10 fold cross validation. 
% for the same reason matlab's fitPosterior wont give the same answer each
% time you run it.
pp = NaN(1, size(score,1) ); 
for i = 1:size(score,1) 
    pp(i) = 1 / (1+exp(ScoreParameters.Slope*score(i,2) + ScoreParameters.Intercept)); 
end

plot(pp)



%% Cross-validated classification model

% Evaluate the Classification Error of the SVM classifier using cross validation

CVSVMModel = crossval(SVMModel); % CVSVMModel.Trained{1}: model 1 --> there will be KFold of these models. (by default KFold=10);

% Classification loss (by default the fraction of misclassified data) for observations not used for training
classLoss = kfoldLoss(CVSVMModel);  % average cross-validated classification error 
fprintf('Average cross-validated classification error = %.3f\n', classLoss)

% Estimate cross-validation predicted labels and scores.
% For every fold, kfoldPredict predicts class labels for in-fold
% observations using a model trained on out-of-fold observations.
[elabel, escore] = kfoldPredict(CVSVMModel);

% Estimate the out-of-sample posterior probabilities
[ScoreCVSVMModel, ScoreParameters] = fitSVMPosterior(CVSVMModel);
[elabel, epostp] = kfoldPredict(ScoreCVSVMModel);


% How claassLoss is computed? I think: classLoss = 1 - mean(Y == elabel)   % 1 - mean(label == elabel)
% a = diff([classLoss, mean(label ~= elabel)]); % This is what you had,
% but I think it is wrong and you need to use the code below. Double check!
a = diff([classLoss, mean(Y ~= elabel)]);
fprintf('%.3f = Difference in classLoss computed using kfoldLoss vs manually using kfoldPredict labels. This value is expected to be very small!\n', a)
if a > 1e-10
    a
    error('Why is there a mismatch? Read your comments above about "a".')
end


%% Example using Holdout for Cross-validated classification model.

% remember KFold will be 1 here.

rng(1) % For reproducibility
% Train an SVM classifier. Specify a 15% holdout sample for testing. It is good practice to specify the class order and standardize the data.
CVSVMModel = fitcsvm(SVMModel.X, SVMModel.Y, 'Holdout', 0.15, 'ClassNames', cnam, 'Standardize',true);

% compute the scores :
% method 1
[~,OOscore] = kfoldPredict(CVSVMModel); % it will be non-nan for test samples and nan for all the samples used for training.

% method 2
testInds = test(CVSVMModel.Partition);   % Extract the test indices
XTest = SVMModel.X(testInds,:);
% YTest = SVMModel.Y(testInds,:);
CompactSVMModel = CVSVMModel.Trained{1}; % Extract trained, compact classifier
[label,OOscore] = predict(CompactSVMModel, XTest);


% compute the posterior probabilities:
% Estimate the optimal score function for mapping observation scores to posterior probabilities
[ScoreCVSVMModel, ScoreParameters] = fitSVMPosterior(CVSVMModel);

% Estimate the out-of-sample posterior probabilities for the positive
% class. (Posterior Probabilities for Test Samples)
[~,OOSPostProbs] = kfoldPredict(ScoreCVSVMModel); % OOSPostProbs is nan for the 85% of data that were used for training.
indx = ~isnan(OOSPostProbs(:,2));




