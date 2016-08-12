if pnev_inputParams.multiTrs && pnev_inputParams.multiSessions
    
    
    mouse = 'fni17';
    imagingFolder = '151022';
    mdfFileNumber = [1,2]; % or tif major

    P = struct; P.signalCh = 2; % The only required field of P; % channel whose signal activity you want to analyze (normally 2 for gcamp channel). %

    params = writeCaProcessParams('', mouse, imagingFolder, mdfFileNumber, P);

    
    %% Compute number of frames per tif file (nFrsMov) and also for each session (nFrsSess) using badFramesTif mat files.
    
    warning('off', 'MATLAB:MatFile:OlderFormat')
    tifNumsOrig = params.tifNums;
    
    % compute number of frames per movie for each session (ie each mdf file aka tif major)
    sess = unique(tifNumsOrig(:,2));
    nFrsMov = cell(1, length(sess));
    date_major_se = cell(1, length(sess));
    for ise = 1:length(sess)
        s = tifNumsOrig(:,2)==sess(ise);
        tifMinor = unique(tifNumsOrig(s, 3))';
        date_major_se{ise} = sprintf('%06d_%03d', tifNumsOrig(find(s,1), 1:2)); % if there is only one mdfFile in tifNums, date_major_se will be same as date_major.
        for itm = 1:length(tifMinor)
            if ~any(params.oldTifName)
                a = dir(fullfile(params.tifFold, [date_major_se{ise}, '_00', num2str(tifMinor(itm)), '.mat']));
            elseif all(params.oldTifName)
                a = dir(fullfile(params.tifFold, [date_major_se{ise}, '_0', num2str(tifMinor(itm)), '.mat']));
            else
                disp('some tif minors are named 00 and some are named 0... work on your codes.')
            end
                
            a = matfile(a.name);
            nFrsMov{ise}(itm) = cellfun(@length, a.badFramesTif(1,1));
        end
    end
    
    nFrsSess = cellfun(@sum, nFrsMov);
    
    
    
    
    
    %% Now run update_tempcomps_multitrs for each session separately.
    % remember YrA, C and f were computed on the concatenated movie of all
    % sessions, now you take the frames that correspond to each session to
    % do update_tempcomps_multitrs. This is because in
    % update_tempcomps_multitrs, you need framesPerTrial, which is saved
    % for each session separately.
    
    C_all = C;
    f_all = f;
%     YrA_all = YrA;
%     Yr_all = Yr;
    
    csnfrs = [0 cumsum(nFrsSess)];   
    
    for ise = 1:length(nFrsSess)
        s = tifNumsOrig(:,2)==sess(ise);
        params.tifNums = tifNumsOrig(s,:); % only includes tifNums of session ise.
        
        r = csnfrs(ise)+1 : csnfrs(ise+1);
        
        [A, C, S, C_df, S_df, Df, srt, P] = update_tempcomps_multitrs(C_all(:,r), f_all(r), A, b, YrA(:,r), Yr(:,r), P, options, params);
        % how about you concat C here, but in ur post proc analyses u
        % divide it up... think...
        
        fprintf('Saving Pnev results.\n')
        save(fullfile(params.tifFold, [date_major_se{ise}, '_ch', num2str(params.activityCh), '-PnevPanResults-', nowStr]), ...
            'A', 'C', 'S', 'C_df', 'S_df', 'Df', 'b', 'f', 'srt', 'Ain', 'options', 'P', 'pnev_inputParams', 'merging_vars', 'Cin', 'bin', 'fin', 'YrA', 'nFrsSess');
    end
    
    
    
end

