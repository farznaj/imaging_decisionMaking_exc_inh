%% Compute number of frames per tif file (nFrsMov) and also for each session (nFrsSess) using badFramesTif mat files.

% mouse = 'fni17';
% imagingFolder = '151022';
% mdfFileNumber = [1,2]; % or tif major
PP = struct; PP.signalCh = 2; % The only required field of P; % channel whose signal activity you want to analyze (normally 2 for gcamp channel). %
params = writeCaProcessParams('', mouse, imagingFolder, mdfFileNumber, PP);


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
%             a = dir(fullfile(params.tifFold, [date_major_se{ise}, '_00', num2str(tifMinor(itm)), '.mat']));
            a = dir(fullfile(params.tifFold, [date_major_se{ise}, ['_', repmat('0', 1, 3-length(num2str(tifMinor(itm))))], num2str(tifMinor(itm)), '.mat']));
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
fprintf(['# frames per session: ', repmat('%i ', 1, length(nFrsSess)), 'Total: ', '%i', '\n'], nFrsSess, sum(nFrsSess))
