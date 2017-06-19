for iday = 1:length(days)
    
    disp('__________________________________________________________________')
    dn = simpleTokenize(days{iday}, '_');
    
    imagingFolder = dn{1};
    mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));   

    fprintf('Analyzing day %s, sessions %s\n', imagingFolder, dn{2})   

    %%
    signalCh = 2; % because you get A from channel 2, I think this should be always 2.
    pnev2load = [];
    [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
    [pd, pnev_n] = fileparts(pnevFileName);
    
    moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));
    postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));
    svmName1 = fullfile(pd, 'svm', '*SVM*');
    svmName2 = fullfile(pd, 'svm', 'svm*');

    %%
    dest = '/media/farznaj/My Stu_win/ChurchlandLab';
    destFold = fullfile(dest, mouse, 'imaging', imagingFolder);
    mkdir(destFold)
    %{
    destFold = fullfile(dest, mouse, 'imaging', imagingFolder, 'svm');    
    mkdir(destFold)
    pd = fullfile(dest, mouse, 'imaging', imagingFolder);
    svmName1 = fullfile(pd, '*SVM*');
    svmName2 = fullfile(pd, 'svm*');    
    movefile(svmName1, destFold)
    movefile(svmName2, destFold)    
    %}
    %%
    copyfile(moreName, destFold)
    copyfile(postName, destFold)
    copyfile(svmName1, destFold)
    copyfile(svmName2, destFold)    
    copyfile([imfilename, '.mat'], destFold)
    
end


