function [rgImg, gcampImg, tdTomatoImg] = inhibit_gcamp_merge(mouse, imagingFolder, mdfFileNumber, savefigs, plotA, removeBadA, qhg)

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
    
    
    %% Set mat file names
    
    [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
    [pd, pnev_n] = fileparts(pnevFileName);
    %     disp(pnev_n)
    %     cd(fileparts(imfilename))
    
    moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));
    
    
    %% Load mat files
    
    a = matfile(moreName);
    if isprop(a, 'inhibitRois_pix')
        load(moreName, 'inhibitRois_pix', 'inhibitImageCorrcted_pix')
        inhibitRois = inhibitRois_pix;
        inhibitImageCorrcted = inhibitImageCorrcted_pix;
    else
        load(moreName, 'inhibitRois', 'inhibitImageCorrcted')
    end
    load(moreName, 'badROIs01', 'CC')
    CCb = CC(badROIs01);
    CC = CC(~badROIs01);
    
    if plotA
        load(pnevFileName, 'A')
        load(imfilename, 'imHeight', 'imWidth')
        if removeBadA
            A = A(:, ~badROIs01);
        end
    else
        load(imfilename, 'sdImage')
%         load(imfilename, 'aveImage')
%         load(imfilename, 'maxImage')
    end
    
    
    %% Set images for red and green channels, also the merged image
    
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
    
    
    %% Plot
    
    % colors = hot(3*length(CC)); % colors = colors(end:-1:1,:);    
    col = 'c'; %'b';
    lw = 1;
    figure('units','normalized','outerposition',[0 0 1 1], 'position',[0.1802         0    0.7786    0.9028])
    ha = tight_subplot(1,7,[.03,.01],[.03,.03]);
    
    axes(ha(1)), imagesc(cat(3, zeros(size(gcampImg)), gcampImg, zeros(size(gcampImg)))); axis image; axis off; if plotA, title('gcamp'), else title('gcamp (sdImage)'), end; a = gca;
    axes(ha(2)), imagesc(cat(3, tdTomatoImg, zeros(size(tdTomatoImg)), zeros(size(tdTomatoImg)))); axis image; axis off; title('tdTomato (bleedthrough corrected)'); a = [a, gca];
    axes(ha(3)), imagesc(cat(3, tdTomatoImg, gcampImg, zeros(size(gcampImg)))); axis image; axis off; title('merged'); a= [a, gca];
    
    axes(ha(4)), imagesc(rgImg); hold on; axis image; axis off
    for rr = find(inhibitRois==1)
        plot(CC{rr}(2,:), CC{rr}(1,:), 'color', col, 'linewidth', lw); a = [a, gca]; %colors(rr, :))
    end 
    title(sprintf('%d inhibitory neurons', sum(inhibitRois==1)))
    
    axes(ha(5)), imagesc(rgImg); hold on; axis image; axis off
    for rr = find(inhibitRois==0)
        plot(CC{rr}(2,:), CC{rr}(1,:), 'color', col, 'linewidth', lw); a= [a, gca]; %colors(rr, :))
    end        
    title(sprintf('%d excitatory neurons', sum(inhibitRois==0)))
    
    axes(ha(6)), imagesc(rgImg); hold on; axis image; axis off
    for rr = find(isnan(inhibitRois))
        plot(CC{rr}(2,:), CC{rr}(1,:), 'color', col, 'linewidth', lw); a = [a, gca]; %colors(rr, :))
    end         
    title(sprintf('%d unsure neurons', sum(isnan(inhibitRois))))
    
    axes(ha(7)), imagesc(rgImg); hold on; axis image; axis off
    for rr = 1:length(CCb)
        if ~isempty(CCb{rr})
            plot(CCb{rr}(2,:), CCb{rr}(1,:), 'color', col, 'linewidth', lw); a = [a, gca]; %colors(rr, :))
        else
            cprintf('-blue', 'ROI %d has empty contour!\n', rr)
        end        
    end         
    title(sprintf('%d bad ROIs', length(CCb)))
    
    set(gcf, 'name', imagingFolder) %     title(sprintf('%s - %d', imagingFolder, mdfFileNumber))
    linkaxes(a)
    zoom
    
    
    %% Save
    
    if savefigs
        if ~exist(fullfile(pd, 'figs'), 'dir')
            mkdir(fullfile(pd, 'figs'))
        end
        if plotA            
            if removeBadA
                savefig(fullfile(pd, 'figs','red_green_A_goodNeurons'))  
            else
                savefig(fullfile(pd, 'figs','red_green_A_allNeurons'))  
            end
        else
            savefig(fullfile(pd, 'figs','red_green'))  
            saveas(gca, fullfile(pd, 'figs','red_green'), 'tif')
        end
    end    
    
    
    %%
%     clearvars -except signalCh pnev2load days mouse savefigs plotA removeBadA
        
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