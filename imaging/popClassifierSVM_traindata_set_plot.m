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
    popClassifierSVM_traindata_setWeights
end


%%
makePlots_here = 0; % 1;

frameTrProjOnBeta_hr_rep_all_alls_dataVSshuff_train = cell(1,length(dataType));
frameTrProjOnBeta_lr_rep_all_alls_dataVSshuff_train = cell(1,length(dataType));
avePerf_cv_dataVSshuff_train = cell(1,length(dataType));


for id = 2; % 1:length(dataType) % there is very little variability for the actual trained data among different iterations... so it makes sense not to go though the actual trained dataset.
    
    fprintf('Setting training data projections for %s data\n', dataType{id})
    
    %%
    % set projections (of neural responses on the hyperplane defined by the
    % decoder's weights) for the test observations in the cross-validated SVM
    % model.
    
    % each cell of frameTrProjOnBeta_hr_rep_all_alls includes frames x test trials for each iteration of s (ie each CVSVMModel you computed in popClassifier).
    frameTrProjOnBeta_hr_rep_all_alls_train = cell(1, length(CVSVMModel_s_all));
    frameTrProjOnBeta_lr_rep_all_alls_train = cell(1, length(CVSVMModel_s_all));
    frameTrProjOnBeta_rep_all_alls_train = cell(1, length(CVSVMModel_s_all));
    
    traces_bef_proj_nf_all_alls_train = cell(1, length(CVSVMModel_s_all));
    traces_bef_proj_f_all_alls_train = cell(1, length(CVSVMModel_s_all));
    
    avePerf_cv = [];
    
    % trsUsedInSVM = find(~mskNan);
    
    
    %%
    for s = 1:length(SVMModel_s_all)
        
        trs2use = shflTrials_alls(:, s);
        
        switch dataType{id}
            case 'actual'
                %                 svm_model_now = CVSVMModel_s_all(s).cv; % CV data
                svm_model_now = SVMModel_s_all(s).cv; % trained data
            case 'shuffled'
                %                 svm_model_now = CVSVMModelChance_all(s).cv; % shuffled CV data
                svm_model_now = SVMModelChance_all(s).cv; % shuffled trained data
        end
        
        if ~usePooledWeights
            wNsHrLrAve_rep = svm_model_now.Beta;
            % normalize it to the vector length
            wNsHrLrAve_rep = wNsHrLrAve_rep / norm(wNsHrLrAve_rep);
        end
        
        
        % Set the test trials beta and traces for kfold ik
        
        %             trTest = test(svm_model_now.Partition, ik); % index of test trials (in the non-nan array of trials) in repition ik of svm_model_now.Partition
        %             wNsHrLrAve_rep = svm_model_now.Trained{ik}.Beta; % weights for repitition ik.
        %             trs2use = trs_s(trTest); % this is crucial in case you have shuffled trials when computing svm_model_now (in those s iterations).
        
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
            frameTrProjOnBeta_rep = einsum(traces_bef_proj_nf, w_norm_ave_train_dataVSshuff(:,id), 2, 1); % frames x sum(trTest) (ie number of test trials) % (fr x u x tr) * (u x 1) --> (fr x tr)
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
                l = predict(svm_model_now, a);
                corrClass(:, itr) = (l==Y_tr_test(itr)); % (l==choiceVec0(itr)); % frames x testTrials.
            end
        end
        
        % average performance (correct classification) across trials.
        avePerf_cv = cat(2, avePerf_cv, nanmean(corrClass, 2));  % frames x (s x kfold)
        
        
        %%
        frameTrProjOnBeta_hr_rep_all_alls_train{s} = frameTrProjOnBeta_hr_rep; % frameTrProjOnBeta_hr_rep_all; % frames x trials (these are all test trials)
        frameTrProjOnBeta_lr_rep_all_alls{s} = frameTrProjOnBeta_lr_rep; % _all;
        frameTrProjOnBeta_rep_all_alls_train{s} = frameTrProjOnBeta_rep; %_all;
        
        traces_bef_proj_nf_all_alls_train{s} = traces_bef_proj_nf; %_all;
        traces_bef_proj_f_all_alls_train{s} = traces_bef_proj_f; %_all;
        
        
    end
    
    frameTrProjOnBeta_hr_rep_all_alls_dataVSshuff_train{id} = frameTrProjOnBeta_hr_rep_all_alls_train;
    frameTrProjOnBeta_lr_rep_all_alls_dataVSshuff_train{id} = frameTrProjOnBeta_lr_rep_all_alls;
    avePerf_cv_dataVSshuff_train{id} = avePerf_cv;
    
    
    %% Print the size of matrices
    
    if id==1
        fprintf('Data: %s\n', dataType{id})
        % frameTrProjOnBeta_hr_rep_all_alls % frameTrProjOnBeta_lr_rep_all_alls
        % frameTrProjOnBeta_rep_all_alls
        % traces_bef_proj_nf_all_alls
        
        size_projection_testTrs = size(frameTrProjOnBeta_rep_all_alls_train{randi(length(CVSVMModel_s_all))});
        fprintf(['size_projection_testTrs (frs x trs): ', repmat('%i  ', 1, length(size_projection_testTrs)), '\n'], size_projection_testTrs)
        
        size_projection_hr_testTrs = size(frameTrProjOnBeta_hr_rep_all_alls_train{randi(length(CVSVMModel_s_all))});
        fprintf(['size_projection_hr_testTrs (frs x trs): ', repmat('%i  ', 1, length(size_projection_hr_testTrs)), '\n'], size_projection_hr_testTrs)
        
        size_projection_lr_testTrs = size(frameTrProjOnBeta_lr_rep_all_alls{randi(length(CVSVMModel_s_all))});
        fprintf(['size_projection_lr_testTrs (frs x trs): ', repmat('%i  ', 1, length(size_projection_lr_testTrs)), '\n'], size_projection_lr_testTrs)
        
        size_orig_traces_testTrs = size(traces_bef_proj_nf_all_alls_train{randi(length(CVSVMModel_s_all))});
        fprintf(['size_orig_traces_testTrs (frs x ns x trs): ', repmat('%i  ', 1, length(size_orig_traces_testTrs)), '\n'], size_orig_traces_testTrs)
        
        size_avePerf_cv = size(avePerf_cv);
        fprintf(['size_avePerf_cv (frs x shffls): ', repmat('%i  ', 1, length(size_avePerf_cv)), '\n'], size_avePerf_cv)
        
    end
    
end



if makePlots_here
    %%
    %%%%%%%%%%%%%%%%%%%%%%   PLOTS    %%%%%%%%%%%%%%%%%%%%
    
    %% Plots related to CV dataset
    
%     col = {'b', 'r'};
    col = {'k', 'g'};
    figure('name', 'Trained data. Top: actual data. Bottom: shuffled data');
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
        av11 = cell2mat(cellfun(@(x)nanmean(x,2), frameTrProjOnBeta_hr_rep_all_alls_dataVSshuff_train{id}, 'uniformoutput', 0)); % frames x number of shuffle iters (s)
        av22 = cell2mat(cellfun(@(x)nanmean(x,2), frameTrProjOnBeta_lr_rep_all_alls_dataVSshuff_train{id}, 'uniformoutput', 0));
        
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
        
        boundedline(time_aligned, av1, se1, 'b', 'alpha')
        boundedline(time_aligned, av2, se2, 'r', 'alpha')
        
        xlabel('Time since stim onset (ms)')
        ylabel({'Weighted average of', 'neural responses'})
        xlim([pb pe])
        
        
        %% Decoder performance at each time point (decoder trained on epoch ep)
        
        top = nanmean(avePerf_cv_dataVSshuff_train{id}, 2); % average performance across all iters.
        top_sd = nanstd(avePerf_cv_dataVSshuff_train{id}, 0, 2) / sqrt(size(avePerf_cv_dataVSshuff_train{id},2));
        
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
        hh(id) = boundedline(time_aligned, top, top_sd, col2{id}, 'alpha');
        
        xlabel('Time since stim onset (ms)')
        ylabel('Correct classification')
        xlim([pb pe])
        
        
    end
    
    
    %%% mark frames with significant different in class accuracy between
    %%% shuffled and actual data
    p_classAcc = NaN(1, size(avePerf_cv,1));
    h_classAcc = NaN(1, size(avePerf_cv,1));
    for fr = 1:length(p_classAcc)
        [h_classAcc(fr), p_classAcc(fr)]= ttest2(avePerf_cv_dataVSshuff_train{1}(fr,:)', avePerf_cv_dataVSshuff_train{2}(fr,:)',...
            'tail', 'right', 'vartype', 'unequal'); % Test the alternative hypothesis that the population mean of actual data is less than the population mean of shuffled data.
    end
    
    subplot(222)
    hnew = h_classAcc;
    hnew(~hnew) = nan;
    plot(time_aligned, hnew * max(mxa)+.05, 'k')
    
    legend(hh, 'data', 'shuffled', 'location', 'northwest')
    legend boxoff
    
    
end