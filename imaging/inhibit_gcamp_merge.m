function inhibit_gcamp_merge(mouse, imagingFolder, mdfFileNumber, savefigs, plotA, removeBadA, qhg)

% Merge red (after bleedthrough correction) and green channels, and
% superimpose contours of identified inhibitory neurons.

if ~exist('savefigs', 'var')
    savefigs = 0; % if 1, figure will be saved in the figs folder.
    plotA = 0; % if 1, green channel will be ROIs identified in A; if 0, it will be the average image of green channel.
    removeBadA = 1; % used only when plotA is 1; if removeBadA = 1, green channel will only show good quality ROIs, otherwise it will show all ROIs.
end



% red channel contrast enhancement values for normImage function
qlr = .2; % quantile to find min of image
qhr = .995; % quantile to find max of image

% green channel contrast enhancement values for normImage function
qlg = .2;
if ~exist('qhg', 'var')
    if plotA
        qhg = .995; % green channel: quantile to find max of image when doing contrast enhancement
    else
        qhg = .985;
    end
end

    
%%
signalCh = 2; % because you get A from channel 2, I think this should be always 2.
pnev2load = [];
% 
% for iday = 1:length(days)
%     
%     dn = simpleTokenize(days{iday}, '_');
%     
%     imagingFolder = dn{1};
%     mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));
    
    
    %%
    [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
    [pd, pnev_n] = fileparts(pnevFileName);
    %     disp(pnev_n)
    %     cd(fileparts(imfilename))
    
    moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));
    
    
    %%
    a = matfile(moreName);
    if isprop(a, 'inhibitRois_pix')
        load(moreName, 'inhibitRois_pix', 'inhibitImageCorrcted_pix')
        inhibitRois = inhibitRois_pix;
        inhibitImageCorrcted = inhibitImageCorrcted_pix;
    else
        load(moreName, 'inhibitRois', 'inhibitImageCorrcted')
    end
    load(moreName, 'badROIs01', 'CC')
    CC = CC(~badROIs01);
    
    if plotA
        load(pnevFileName, 'A')
        load(imfilename, 'imHeight', 'imWidth')
        if removeBadA
            A = A(:, ~badROIs01);
        end
    else
        load(imfilename, 'sdImage')
        load(imfilename, 'aveImage')
        load(imfilename, 'maxImage')
    end
    
    
    %%
    
    %%%%% Red channel
    tdTomatoImg = inhibitImageCorrcted;
    % tdTomatoImg = normImage(tdTomatoImg);
    tdTomatoImg = normImage(tdTomatoImg, qhr, quantile(tdTomatoImg(:), qlr));
    
    
    
    %%%%% Green channel
    if plotA
        gcampImg = full(reshape(mean(A,2), imHeight, imWidth));
    else
        gcampImg = sdImage{2};
        %         gcampImg = aveImage{2};
        %         gcampImg = maxImage{2};
    end
    %     gcampImg = normImage(gcampImg);
    gcampImg = normImage(gcampImg, qhg, quantile(gcampImg(:), qlg));
    
    
    
    %%%% Merged red and green channel
    rgImg = cat(3, tdTomatoImg, gcampImg, zeros(size(gcampImg)));
    
    
    %%
    % colors = hot(3*length(CC));
    % colors = colors(end:-1:1,:);
    
    figure;
    imagesc(rgImg)
    %     imagesc(gcampImg)
    % imagesc(tdTomatoImg)
    hold on
    axis image
    for rr = find(inhibitRois==1)
        plot(CC{rr}(2,:), CC{rr}(1,:), 'color', 'b') %colors(rr, :))
    end
    
    %     title(sprintf('%s - %d', imagingFolder, mdfFileNumber))
    title(imagingFolder)
        
    
    %%
    if savefigs
        if plotA            
            if removeBadA
                savefig(fullfile(pd, 'figs','red_green_A_goodNeurons'))  
            else
                savefig(fullfile(pd, 'figs','red_green_A_allNeurons'))  
            end
        else
            savefig(fullfile(pd, 'figs','red_green'))  
        end
    end    
    
    %%
    clearvars -except signalCh pnev2load days mouse savefigs plotA removeBadA
        
% end



%%
% figure; imagesc(rgImg(:,:,1)) %+ rgImg(:,:,2))

%{
    figure;
    % imagesc(gcampImg)
    imagesc(normImage(gcampImg))
    axis image

    figure;
    % imagesc(tdTomatoImg)
    imagesc(normImage(tdTomatoImg))
    axis image
%}

%{
    figure;
    imagesc(tdTomatoImg)
    figure; imagesc(normImage(tdTomatoImg,.995, quantile(tdTomatoImg(:), .1))), colorbar
    figure; imagesc(normImage(tdTomatoImg,.995, quantile(tdTomatoImg(:), .15))), colorbar

    figure; imagesc(normImage(tdTomatoImg)), colorbar

    figure; imagesc(log(gcampImg)), colorbar
    figure; imagesc(log(aveImage{2})), colorbar
    figure; imagesc(tdTomatoImg), colorbar
%}