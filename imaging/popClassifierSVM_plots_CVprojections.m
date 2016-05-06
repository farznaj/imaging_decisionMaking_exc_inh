% here we get the projection traces for the test observations in the CV
% model. (instead of simply getting them from the trained dataset which
% wont be informative because obviously trained dataset will have nicely
% separated projections between the two states (hr vs lr))

% popClassifier_GE will give the vars you need here.


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
frameTrProjOnBeta_hr_rep_all_alls = cell(1, length(CVSVMModel_s_all));
frameTrProjOnBeta_lr_rep_all_alls = cell(1, length(CVSVMModel_s_all));

trsUsedInSVM = find(~mskNan);

for s = 1:length(CVSVMModel_s_all)
    
    CVSVMModel_s = CVSVMModel_s_all(s).cv;
    % use this if you don't want to pool across folds
%     clear frameTrProjOnBeta_hr_rep_all
%     clear frameTrProjOnBeta_lr_rep_all

    % use this if you want to pool across folds 
    frameTrProjOnBeta_hr_rep_all = [];
    frameTrProjOnBeta_lr_rep_all = [];

    % pool the projections across the 10 folds of SV model for each shuffle iteration (s)
    for ik = 1:CVSVMModel_s.KFold % repition ik of CVSVMModel_s.Partition
        
        trTest = test(CVSVMModel_s.Partition, ik); % index of test trials (in the non-nan array of trials) in repition ik of CVSVMModel_s.Partition
        wNsHrLrAve_rep = CVSVMModel_s.Trained{ik}.Beta; % weights for repitition ik.
        frameTrProjOnBeta_rep = einsum(traces_al_sm(:, ~NsExcluded, trsUsedInSVM(trTest)), wNsHrLrAve_rep, 2, 1); % (fr x u x tr) * (u x 1) --> (fr x tr)
        
        %     disp('______')
        % %     sum(trTest)
        %     size(frameTrProjOnBeta_rep) % frames x trials
        %     sum(choiceVec0(trsUsedInSVM(trTest))==1)
        %     sum(choiceVec0(trsUsedInSVM(trTest))==0)
        
        frameTrProjOnBeta_hr_rep = frameTrProjOnBeta_rep(:, choiceVec0(trsUsedInSVM(trTest))==1); % frames x trials % HR
        frameTrProjOnBeta_lr_rep = frameTrProjOnBeta_rep(:, choiceVec0(trsUsedInSVM(trTest))==0); % frames x trials % LR
        
        % use this if you don't want to pool across folds
%         frameTrProjOnBeta_hr_rep_all{ik} = frameTrProjOnBeta_hr_rep;
%         frameTrProjOnBeta_lr_rep_all{ik} = frameTrProjOnBeta_lr_rep;
        
        % use this if you want to pool across folds 
        frameTrProjOnBeta_hr_rep_all = cat(2, frameTrProjOnBeta_hr_rep_all, frameTrProjOnBeta_hr_rep);
        frameTrProjOnBeta_lr_rep_all = cat(2, frameTrProjOnBeta_lr_rep_all, frameTrProjOnBeta_lr_rep);
    end
    
    % sanity check for number of trials
    isequal(size(frameTrProjOnBeta_hr_rep_all,2) + size(frameTrProjOnBeta_lr_rep_all,2), ...
        sum(CVSVMModel_s.Partition.TestSize))
    
    % use this if you don't want to pool across folds
%     frameTrProjOnBeta_hr_rep_all_alls = cat(2, frameTrProjOnBeta_hr_rep_all_alls, frameTrProjOnBeta_hr_rep_all);
%     frameTrProjOnBeta_lr_rep_all_alls = cat(2, frameTrProjOnBeta_lr_rep_all_alls, frameTrProjOnBeta_lr_rep_all);
    
    % use this if you want to pool across folds 
    frameTrProjOnBeta_hr_rep_all_alls{s} = frameTrProjOnBeta_hr_rep_all;
    frameTrProjOnBeta_lr_rep_all_alls{s} = frameTrProjOnBeta_lr_rep_all;
end


%% Plot the projections

av11 = cell2mat(cellfun(@(x)nanmean(x,2), frameTrProjOnBeta_hr_rep_all_alls, 'uniformoutput', 0)); % frames x number of shuffle iters (s)
av22 = cell2mat(cellfun(@(x)nanmean(x,2), frameTrProjOnBeta_lr_rep_all_alls, 'uniformoutput', 0));

av1 = nanmean(av11, 2); % average across s
av2 = nanmean(av22, 2); % average across s
se1 = nanstd(av11, 0, 2); % / sqrt(size(av11, 2));
se2 = nanstd(av22, 0, 2); % / sqrt(size(av22, 2));

% av1 = nanmean(frameTrProjOnBeta_hr_rep_all, 2);
% av2 = nanmean(frameTrProjOnBeta_lr_rep_all, 2);


mn = min([av1;av2]);
mx = max([av1;av2]);


figure;
subplot(221), hold on
plot([0 0], [mn mx], 'k:', 'handleVisibility', 'off') % [eventI_stimOn eventI_stimOn]
plot([st st], [mn mx], 'k:', 'handleVisibility', 'off') % [epStart epStart]
plot([en en], [mn mx], 'k:', 'handleVisibility', 'off') % [epEnd epEnd]

% plot(time_aligned_stimOn, av1)
% plot(time_aligned_stimOn, av2)

boundedline(time_aligned_stimOn, av1, se1, 'b') 
boundedline(time_aligned_stimOn, av2, se2, 'r') 
xlabel('Time since stim onset (ms)')
ylabel({'Weighted average of', 'neural responses'})
xlim([pb pe])

