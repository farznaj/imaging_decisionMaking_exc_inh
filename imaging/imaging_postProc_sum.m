% After running imaging_postproc, open figures saved in dir "figs" of each session, and publish them in a
% pdf file.

%%
%{

signalCh = 2; % because you get A from channel 2, I think this should be always 2.
pnev2load = [];
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
[pd, date_major] = fileparts(imfilename);
figd = fullfile(pd, 'figs');
% cd(figd) %%% copyfile(fullfile('/home/farznaj/Documents/trial_history/imaging','imaging_postProc_html.m'),'.','f')

publish('imaging_postProc_sum.m', 'format', 'pdf')

close all

savedir = fullfile(['~/Dropbox/ChurchlandLab/Farzaneh_Gamal/postprop_sum',mouse,date_major]);
movefile('html/*_sum*', savedir)

%}



%% PLOTS

disp(figd)

aa = dir(fullfile(figd, '*.fig')); %dir('*.fig');
% aa = a(~[a.isdir]);
[~,s] = sort([aa.datenum]);
b = {aa.name}; b = b(s);

fi = find(ismember(b, 'caTraces_aveAllNeurons.fig')); % plot a few figures at different x so you can evaluate it well.
fi2 = find(ismember(b, 'behav_motCorr_sum.fig')); % move its legend location

for i = 1:length(b)

    open(fullfile(figd, b{i})) % open(b{i})
    
    if i==fi2 % behav_motCorr_sum
        pause(1)
        a = get(gcf,'children'); 
        l = a(end-1); %legend of subplot 1
        lp = get(l,'position'); 
        set(l, 'position', [.4,.8,lp(3:4)])
        
        l = a(end-3); %legend of subplot 3
        lp = get(l,'position'); 
        set(l, 'position', [.4,.3,lp(3:4)])        
    end
    
    %
    if i==fi % caTraces_aveAllNeurons
        pause(3)
        xl = get(gca,'xlim');        
        % open 10 figures zoomed in at different x 
        totfs = unique([1 : floor((floor(xl(2)/.1e4)+1)/10) : floor(xl(2)/.1e4)+1, floor(xl(2)/.1e4)+1]);
%         disp(totfs)
        for rr = totfs; %1:floor(xl(2)/.1e4)+1            
            open(fullfile(figd, b{i})) %open(b{i})
            r2 = rr * .1e4;
            r1 = r2 - .1e4;
%             disp([r1,r2])
%         end
            xlim([r1 r2])
            pause(1)
%             open(fullfile(figd, b{i})) %open(b{i})
        end
    end    
    %
    
end



%% DIARY FILES

a = dir(fullfile(pd,'diary*'));
% cd(pd)
for i = 1:length(a)
    fn = fullfile(pd, a(i).name);
    disp('________________________________________________________________________________________________')
    disp('________________________________________________________________________________________________')
    disp('________________________________________________________________________________________________')
    fprintf('Reading diary file %s\n', a(i).name)
    disp('________________________________________________________________________________________________')
    disp('________________________________________________________________________________________________')
    disp('________________________________________________________________________________________________')
    
%     notes = textread(fn, '%s', 'whitespace', '');    
    fid = fopen(fn);
    notes = textscan(fid, '%s', 'whitespace', ''); 
    fclose(fid);
    
    celldisp(notes)
end




