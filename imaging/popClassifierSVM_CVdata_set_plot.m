% Set and plot the projection traces for the test observations in the CV
% model, ie we project the neural traces of test observations in each CV
% model onto its decoder.
% Also set and plot the classification accuracy for all time points using
% the SVM decoder trained on ep.

% (instead of simply getting them from the trained dataset which
% wont be informative because obviously trained dataset will have nicely
% separated projections between the two states (hr vs lr))

% popClassifier will give the vars you need here.


dataType = {'actual', 'shuffled'};


%% Set average weights for CV SVM models across all iterations.

% it gives the normalized average of normalized weights.
if usePooledWeights
    popClassifierSVM_CVdata_setWeights
end


%%
frameTrProjOnBeta_hr_rep_all_alls_dataVSshuff = cell(1,length(dataType));
frameTrProjOnBeta_lr_rep_all_alls_dataVSshuff = cell(1,length(dataType));
avePerf_cv_dataVSshuff = cell(1,length(dataType));


for id = 1:length(dataType)
    
    fprintf('Setting CV data projections for %s data\n', dataType{id})
    
    %%
    % set projections (of neural responses on the hyperplane defined by the
    % decoder's weights) for the test observations in the cross-validated SVM
    % model.
    % frameTrProjOnBeta_hr_rep_all = [];
    % frameTrProjOnBeta_lr_rep_all = [];
    
    % use this if you don't want to pool across folds
    % frameTrProjOnBeta_hr_rep_all_alls = [];
    % frameTrProjOnBeta_lr_rep_all_alls = [];
    
    % use this if you want to pool across folds
    % each cell of frameTrProjOnBeta_hr_rep_all_alls includes frames x test trials for each iteration of s (ie each CVSVMModel you computed in popClassifier).
    frameTrProjOnBeta_hr_rep_all_alls = cell(1, length(CVSVMModel_s_all));
    frameTrProjOnBeta_lr_rep_all_alls = cell(1, length(CVSVMModel_s_all));
    frameTrProjOnBeta_rep_all_alls = cell(1, length(CVSVMModel_s_all));
    
    traces_bef_proj_nf_all_alls = cell(1, length(CVSVMModel_s_all));
    traces_bef_proj_f_all_alls = cell(1, length(CVSVMModel_s_all));
    
    avePerf_cv = [];
    
    % trsUsedInSVM = find(~mskNan);
    
    
    %%
    for s = 1:length(CVSVMModel_s_all)
        
        % Set the CVSVM model and the order of trials that were used for it.
        
        % Trials for this iteration
        trs_s = shflTrials_alls(:, s);
        
        switch dataType{id}
            case 'actual'
                svm_model_now = CVSVMModel_s_all(s).cv; % CV data
                %                 svm_model_now = SVMModel_s_all(s).cv; % trained data
            case 'shuffled'
                svm_model_now = CVSVMModelChance_all(s).cv; % shuffled CV data
                %                 svm_model_now = SVMModelChance_all(s).cv; % shuffled trained data
        end
        
        % use this if you don't want to pool across folds
        %     clear frameTrProjOnBeta_hr_rep_all
        %     clear frameTrProjOnBeta_lr_rep_all
        
        % use this if you want to pool across folds
        frameTrProjOnBeta_hr_rep_all = [];
        frameTrProjOnBeta_lr_rep_all = [];
        frameTrProjOnBeta_rep_all = [];
        
        traces_bef_proj_nf_all = [];
        traces_bef_proj_f_all = [];
        
        
        %% Loop through the 10 folds of each CV SVM model.
        
        % pool the projections across the 10 folds of SV model for each shuffle iteration (s)
        for ik = 1:svm_model_now.KFold % repition ik of svm_model_now.Partition
            
            % Set the test trials beta and traces for kfold ik
            
            trTest = test(svm_model_now.Partition, ik); % index of test trials (in the non-nan array of trials) in repition ik of svm_model_now.Partition
            trs2use = trs_s(trTest); % this is crucial in case you have shuffled trials when computing svm_model_now (in those s iterations).
            
            if ~usePooledWeights
                wNsHrLrAve_rep = svm_model_now.Trained{ik}.Beta; % weights for repitition ik.
                % normalize it to the vector length
                wNsHrLrAve_rep = wNsHrLrAve_rep / norm(wNsHrLrAve_rep);
            end
            
            
            % non-smoothed traces
            traces_bef_proj_nf = non_filtered(:, :, trs2use); % frames x units x trials. % traces_al_sm(:, ~NsExcluded, trsUsedInSVM(trs2use))
            % smoothed traces (with a window of size ep, moving average)
            if pcaFlg
                traces_bef_proj_f = filtered_s(:, :, trs2use);
            else
                traces_bef_proj_f = filtered(:, :, trs2use);
            end
            
            
            %% Set the projections for CV trials.
            
            % project the neural traces of test samples (ie observations not used for
            % training) onto the decoder. This gives the projection traces
            % named frameTrProjOnBeta_rep.
            
            if usePooledWeights
                % project non-filtered traces on the average normalized vector across all iterations and folds
                frameTrProjOnBeta_rep = einsum(traces_bef_proj_nf, w_norm_ave_dataVSshuff(:,id), 2, 1); % frames x sum(trTest) (ie number of test trials) % (fr x u x tr) * (u x 1) --> (fr x tr)
            else
                % project non-filtered traces on the weights from a single model.
                frameTrProjOnBeta_rep = einsum(traces_bef_proj_nf, wNsHrLrAve_rep, 2, 1); % frames x sum(trTest) (ie number of test trials) % (fr x u x tr) * (u x 1) --> (fr x tr)
            end
            
            
            % project filtered traces
            %             frameTrProjOnBeta_rep = einsum(traces_bef_proj_f, wNsHrLrAve_rep, 2, 1); % frames x sum(trTest) (ie number of test trials) % (fr x u x tr) * (u x 1) --> (fr x tr)
            %         size(frameTrProjOnBeta_rep) % frames x sum(trTest) (ie number of test trials)
            
            % now separate frameTrProjOnBeta_rep into hr and lr groups.
            %         frameTrProjOnBeta_hr_rep = frameTrProjOnBeta_rep(:, choiceVec0(trsUsedInSVM(trTest))==1); % frames x trials % HR
            %         frameTrProjOnBeta_lr_rep = frameTrProjOnBeta_rep(:, choiceVec0(trsUsedInSVM(trTest))==0); % frames x trials % LR
            frameTrProjOnBeta_hr_rep = frameTrProjOnBeta_rep(:, Y(trs2use)==1); % frames x trials % HR
            frameTrProjOnBeta_lr_rep = frameTrProjOnBeta_rep(:, Y(trs2use)==0); % frames x trials % LR
            
            % use this if you don't want to pool across folds
            %         frameTrProjOnBeta_hr_rep_all{ik} = frameTrProjOnBeta_hr_rep;
            %         frameTrProjOnBeta_lr_rep_all{ik} = frameTrProjOnBeta_lr_rep;
            
            % use this if you want to pool across folds
            frameTrProjOnBeta_hr_rep_all = cat(2, frameTrProjOnBeta_hr_rep_all, frameTrProjOnBeta_hr_rep);
            frameTrProjOnBeta_lr_rep_all = cat(2, frameTrProjOnBeta_lr_rep_all, frameTrProjOnBeta_lr_rep);
            frameTrProjOnBeta_rep_all = cat(2, frameTrProjOnBeta_rep_all, frameTrProjOnBeta_rep);
            
            traces_bef_proj_nf_all = cat(3, traces_bef_proj_nf_all, traces_bef_proj_nf);
            traces_bef_proj_f_all = cat(3, traces_bef_proj_f_all, traces_bef_proj_f);
            
            
            %% Compute classification accuracy for test trials at all time points using the decoder trained for ep.
            
            traceNow = traces_bef_proj_f;
            %         traceNow = traces_bef_proj_nf;
            corrClass = NaN(size(traceNow,1), size(traceNow,3)); % frames x trials
            Y_tr_test = Y(trs2use);
            
            % loop over test trials.
            for itr = 1 : size(traceNow, 3) % size(traces_bef_proj_nf_all_alls{1},3)
                a = traceNow(:, :, itr); % frames x neurons
                
                if any(isnan(a(:)))
                    if ~all(isnan(a(:))), error('how did it happen?'), end
                    
                elseif ~isnan(Y_tr_test(itr)) % ~isnan(choiceVec0(itr))
                    l = predict(svm_model_now.Trained{ik}, a);
                    corrClass(:, itr) = (l==Y_tr_test(itr)); % (l==choiceVec0(itr)); % frames x testTrials.
                end
            end
            
            % average performance (correct classification) across trials.
            avePerf_cv = cat(2, avePerf_cv, nanmean(corrClass, 2));  % frames x (s x kfold)
            
            
        end
        
        % sanity check for the number of trials
        if ~isequal(size(frameTrProjOnBeta_hr_rep_all,2) + size(frameTrProjOnBeta_lr_rep_all,2), ...
                sum(svm_model_now.Partition.TestSize))
            error('something wrong!')
        end
        
        
        %%
        % use this if you don't want to pool across folds
        %     frameTrProjOnBeta_hr_rep_all_alls = cat(2, frameTrProjOnBeta_hr_rep_all_alls, frameTrProjOnBeta_hr_rep_all);
        %     frameTrProjOnBeta_lr_rep_all_alls = cat(2, frameTrProjOnBeta_lr_rep_all_alls, frameTrProjOnBeta_lr_rep_all);
        
        % use this if you want to pool across folds for each iteration.
        frameTrProjOnBeta_hr_rep_all_alls{s} = frameTrProjOnBeta_hr_rep_all; % frames x trials (these are all test trials)
        frameTrProjOnBeta_lr_rep_all_alls{s} = frameTrProjOnBeta_lr_rep_all;
        frameTrProjOnBeta_rep_all_alls{s} = frameTrProjOnBeta_rep_all;
        
        traces_bef_proj_nf_all_alls{s} = traces_bef_proj_nf_all;
        traces_bef_proj_f_all_alls{s} = traces_bef_proj_f_all;
        
        
    end
    
    frameTrProjOnBeta_hr_rep_all_alls_dataVSshuff{id} = frameTrProjOnBeta_hr_rep_all_alls;
    frameTrProjOnBeta_lr_rep_all_alls_dataVSshuff{id} = frameTrProjOnBeta_lr_rep_all_alls;
    avePerf_cv_dataVSshuff{id} = avePerf_cv;
    
    
    %% Print the size of matrices
    
    if id==1
        fprintf('Data: %s\n', dataType{id})
        % frameTrProjOnBeta_hr_rep_all_alls % frameTrProjOnBeta_lr_rep_all_alls
        % frameTrProjOnBeta_rep_all_alls
        % traces_bef_proj_nf_all_alls
        
        size_projection_testTrs = size(frameTrProjOnBeta_rep_all_alls{randi(length(CVSVMModel_s_all))});
        fprintf(['size_projection_testTrs (frs x trs): ', repmat('%i  ', 1, length(size_projection_testTrs)), '\n'], size_projection_testTrs)
        
        size_projection_hr_testTrs = size(frameTrProjOnBeta_hr_rep_all_alls{randi(length(CVSVMModel_s_all))});
        fprintf(['size_projection_hr_testTrs (frs x trs): ', repmat('%i  ', 1, length(size_projection_hr_testTrs)), '\n'], size_projection_hr_testTrs)
        
        size_projection_lr_testTrs = size(frameTrProjOnBeta_lr_rep_all_alls{randi(length(CVSVMModel_s_all))});
        fprintf(['size_projection_lr_testTrs (frs x trs): ', repmat('%i  ', 1, length(size_projection_lr_testTrs)), '\n'], size_projection_lr_testTrs)
        
        size_orig_traces_testTrs = size(traces_bef_proj_nf_all_alls{randi(length(CVSVMModel_s_all))});
        fprintf(['size_orig_traces_testTrs (frs x ns x trs): ', repmat('%i  ', 1, length(size_orig_traces_testTrs)), '\n'], size_orig_traces_testTrs)
        
        size_avePerf_cv = size(avePerf_cv);
        fprintf(['size_avePerf_cv (frs x shffls): ', repmat('%i  ', 1, length(size_avePerf_cv)), '\n'], size_avePerf_cv)
        
    end
    
    if ~isequal(size(avePerf_cv, 2), length(CVSVMModel_s_all) * svm_model_now.KFold)
        error('something wrong')
    end
    
    
end



%%
%%%%%%%%%%%%%%%%%%%%%%   PLOTS    %%%%%%%%%%%%%%%%%%%%

%% Plots related to CV dataset

% col = {'b', 'r'};
col = {'k', 'g'};
figure('name', 'Only cross-validated trials. Top: actual data. Bottom: shuffled data');
pb = round((- eventI)*frameLength);
pe = round((length(time_aligned) - eventI)*frameLength);
st = round((epStart - eventI)*frameLength);
en = round((epEnd - eventI)*frameLength);

mxa = [];
clear hh

for id = 1:length(dataType)
    
    % compute the average trace across all test trials for each CVSVMModel.
    % then convert the average trace of all CVSVMModels into a single matrix, this is called av11 (or av22).
    % frames x number of shuffle iters (s)
    av11 = cell2mat(cellfun(@(x)nanmean(x,2), frameTrProjOnBeta_hr_rep_all_alls_dataVSshuff{id}, 'uniformoutput', 0)); % frames x number of shuffle iters (s)
    av22 = cell2mat(cellfun(@(x)nanmean(x,2), frameTrProjOnBeta_lr_rep_all_alls_dataVSshuff{id}, 'uniformoutput', 0));
    
    av1 = nanmean(av11, 2); % average across s
    av2 = nanmean(av22, 2); % average across s
    se1 = nanstd(av11, 0, 2) / sqrt(size(av11, 2));
    se2 = nanstd(av22, 0, 2) / sqrt(size(av22, 2));
    
    % av1 = nanmean(frameTrProjOnBeta_hr_rep_all, 2);
    % av2 = nanmean(frameTrProjOnBeta_lr_rep_all, 2);
    
    
    mn = min([av1;av2]);
    mx = max([av1;av2]);
    
    switch dataType{id}
        case 'actual'
            subplot(221)
        case 'shuffled'
            subplot(223)
    end
    
    hold on
    % plot([0 0], [mn mx], 'k:', 'handleVisibility', 'off') % [eventI eventI]
    plot([st st], [mn mx], 'k:', 'handleVisibility', 'off') % [epStart epStart]
    plot([en en], [mn mx], 'k:', 'handleVisibility', 'off') % [epEnd epEnd]
    
    % plot(time_aligned, av1)
    % plot(time_aligned, av2)
    
    h11 = boundedline(time_aligned, av1, se1, 'b', 'alpha');
    h12 = boundedline(time_aligned, av2, se2, 'r', 'alpha');
    
    xlabel('Time since stim onset (ms)')
    ylabel({'Weighted average of', 'neural responses'})
    xlim([pb pe])
    
    legend([h11 h12], {'HR','LR'}, 'location', 'northwest')
    legend boxoff
    
    
    %% Decoder performance at each time point (decoder trained on epoch ep)
    
    top = nanmean(avePerf_cv_dataVSshuff{id}, 2); % average performance across all iters.
    top_sd = nanstd(avePerf_cv_dataVSshuff{id}, 0, 2) / sqrt(size(avePerf_cv_dataVSshuff{id},2));
    
    mn = min(top(:));
    mx = max(top(:));
    mxa = [mxa, mx];
    
    switch dataType{id}
        case 'actual'
            subplot(222)
        case 'shuffled'
            subplot(222)
            %         subplot(224)
    end
    hold on
    
    % plot([0 0], [mn mx], 'k:', 'handleVisibility', 'off') % [eventI eventI]
    plot([st st], [mn mx], 'k:', 'handleVisibility', 'off') % [epStart epStart]
    plot([en en], [mn mx], 'k:', 'handleVisibility', 'off') % [epEnd epEnd]
    plot([time_aligned(1) time_aligned(end)], [.5 .5], 'k:', 'handleVisibility', 'off')
    
    % plot(time_aligned, top) % average across all iters
    hh(id) = boundedline(time_aligned, top, top_sd, col{id}, 'alpha');
    
    xlabel('Time since stim onset (ms)')
    ylabel('Correct classification')
    xlim([pb pe])
    
    
end


%%% mark frames with significant different in class accuracy between
%%% shuffled and actual data
p_classAcc = NaN(1, size(avePerf_cv,1));
h_classAcc = NaN(1, size(avePerf_cv,1));
for fr = 1:length(p_classAcc)
    [h_classAcc(fr), p_classAcc(fr)]= ttest2(avePerf_cv_dataVSshuff{1}(fr,:), avePerf_cv_dataVSshuff{2}(fr,:),...
        'tail', 'right', 'vartype', 'unequal'); % Test the alternative hypothesis that the population mean of actual data is less than the population mean of shuffled data.
end

subplot(222)
hnew = h_classAcc;
hnew(~hnew) = nan;
plot(time_aligned, hnew * max(mxa)+.05, 'k')

legend(hh, 'data', 'shuffled', 'location', 'northwest')
legend boxoff

%{
p_classAcc = NaN(1, size(avePerf_cv,1));
h_classAcc = NaN(1, size(avePerf_cv,1));
for fr = 1:length(p_classAcc)
    [h_classAcc(fr), p_classAcc(fr)]= ttest(avePerf_cv_dataVSshuff{2}(fr,:), .5, 'tail', 'left');
end
hnew = h_classAcc;
hnew(~hnew) = nan;
plot(time_aligned, hnew * max(mxa)-.05, 'g')

fr = 30;
[h,p,ci,stats] = ttest(avePerf_cv_dataVSshuff{2}(fr,:), .5, 'tail', 'left')
%}



%% Plot and compare distributions of classification loss between shuffled and actual datasets. Do this for both training dataset and CV dataset.

[~, p0]= ttest2(classLossTrain, classLossChanceTrain, 'tail', 'left', 'vartype', 'unequal'); % Test the alternative hypothesis that the population mean of actual data is less than the population mean of shuffled data.
[~, p]= ttest2(classLossTest, classLossChanceTest, 'tail', 'left', 'vartype', 'unequal'); % Test the alternative hypothesis that the population mean of actual data is less than the population mean of shuffled data.
[~, pboth] = ttest2(classLossTest, classLossChanceTest, 'vartype', 'unequal');
[~, pright] = ttest2(classLossTest, classLossChanceTest, 'tail', 'right', 'vartype', 'unequal');
fprintf('pval both tails = %.2f\n', pboth)
fprintf('pval left tail (actual<shuff) = %.2f\n', p)
fprintf('pval right tail (actual>shuff) = %.2f\n', pright)

% figure;
%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAINING DATASET (shuffled and actual) %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subplot(211)
subplot(224)
hold on
% Shuffled (chance) dataset
hc = hist(classLossChanceTrain, 0:0.02:1);
%     hd = hist(classLossTrain, 0:0.02:1);
if strfind(version, 'R2016')
    bar(0:0.02:1, hc, 'facecolor', [153 255 153]/255, 'edgecolor', 'none', 'Facealpha', 0.7', 'barwidth', 1);
else
    bar(0:0.02:1, hc, 'facecolor', [153 255 153]/255, 'edgecolor', 'none'); % , 'Facealpha', 0.7', 'barwidth', 1);
end
% Shuffled (chance) training dataset
h11 = plot(mean(classLossChanceTrain), 0, 'go','markerfacecolor', [153 255 153]/255, 'markersize', 6);
% Actual training dataset
h12 = plot(mean(classLossTrain), 0, 'ko','markerfacecolor', [.6 .6 .6], 'markersize', 6);
% ylabel('Count')
xlabel('training loss')
xlim([0 1])
% legend([h11, h12], 'trained\_shuffled', 'trained\_data', 'location', 'northwest')
% legend boxoff



%%%%%%%%%%%%%%%%%%%%%%%%%%%% CV DATASET (shuffled and actual) %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subplot(212)
hold on
% Shuffled (chance) CV dataset
hc = hist(classLossChanceTest, 0:0.02:1);
% Actual CV dataset
hd = hist(classLossTest, 0:0.02:1);
if strfind(version, 'R2016')
    % Shuffled (chance) CV dataset
    bar(0:0.02:1, hc, 'facecolor', 'g', 'edgecolor', 'none', 'Facealpha', 0.7', 'barwidth', 1);
    % Actual CV dataset
    bar(0:0.02:1, hd, 'facecolor', 'k', 'edgecolor', 'none', 'Facealpha', 0.7', 'barwidth', 1);
else
    bar(0:0.02:1, hc, 'facecolor', 'g', 'edgecolor', 'none'); %, 'Facealpha', 0.7', 'barwidth', 1);
    bar(0:0.02:1, hd, 'facecolor', 'k', 'edgecolor', 'none'); %, 'Facealpha', 0.7', 'barwidth', 1);
end
% Shuffled (chance) CV dataset
h1 = plot(mean(classLossChanceTest), 0, 'ko','markerfacecolor', 'g', 'markersize', 6);
% Actual CV dataset
h2 = plot(mean(classLossTest), 0, 'ko','markerfacecolor', 'k', 'markersize', 6);
ylabel('Count')
xlabel('Training- and CV-data loss at the training timepoint')
xlim([0 1])
title({['p-value lower tail, training = ' num2str(p0)], ['CV = ' num2str(p)]})
% legend([h1 h2], 'shuffled', 'data', 'location', 'northwest')
legend([h11, h12, h1 h2], 'trained\_shuffled', 'trained\_data', 'CV\_shuffled', 'CV\_data', 'location', 'northwest')
legend boxoff



