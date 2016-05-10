% Set average weights for CV SVM models across all iterations.
% it gives the normalized average of normalized weights.

%% Set the decoder weights for each SVM model

wNsHrLrAve_alls_dataVSshuff = cell(1, length(dataType));

for id = 1:length(dataType)
    
    wNsHrLrAve_alls = NaN(size(X,2), length(CVSVMModel_s_all) * CVSVMModel.KFold);
    cnt = 0;

    for s = 1:length(CVSVMModel_s_all)               
                
        switch dataType{id}
            case 'actual'
                svm_model_now = CVSVMModel_s_all(s).cv; % CV data
                %                 svm_model_now = SVMModel_s_all(s).cv; % trained data
            case 'shuffled'
                svm_model_now = CVSVMModelChance_all(s).cv; % shuffled CV data
                %                 svm_model_now = SVMModelChance_all(s).cv; % shuffled trained data
        end
        
        %% Loop through the 10 folds of each CV SVM model.        
        
        for ik = 1:svm_model_now.KFold % repition ik of svm_model_now.Partition
            cnt = cnt+1;
            wNsHrLrAve_rep = svm_model_now.Trained{ik}.Beta;            
            wNsHrLrAve_alls(:, cnt) = wNsHrLrAve_rep;
        end
        
    end
    
    wNsHrLrAve_alls_dataVSshuff{id} = wNsHrLrAve_alls;
    
end


%% Normalize vector of weights

% get the length of each vecot of weights
w_len = cellfun(@(x)sqrt(sum(x.^2)), wNsHrLrAve_alls_dataVSshuff, 'uniformoutput', 0);

% divide each vector of weithts by its length
w_norm = cell(1, length(dataType));
for id = 1:length(dataType)
    w_norm{id} = bsxfun(@(x,y)rdivide(x,y), wNsHrLrAve_alls_dataVSshuff{id}, w_len{id});
end

% get the mean across all interations and kfols
w_norm_ave_dataVSshuff = cell2mat(cellfun(@(x)mean(x,2), w_norm, 'uniformoutput', 0));

% normalize average normalized weights of data and shuffled by their length
for id = 1:length(dataType)
    w_norm_ave_dataVSshuff(:,id) = w_norm_ave_dataVSshuff(:,id) / norm(w_norm_ave_dataVSshuff(:,id));
end





