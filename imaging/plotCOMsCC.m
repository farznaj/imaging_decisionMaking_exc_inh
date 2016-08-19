function plotCOMsCC(inp, colors)
% plot COMs or contours of ROIs 
% Example:
% figure; imagesc(medImage{2}, hold on, plotCOMsCC(CC)


if iscell(inp)
    plotCOMs = 0;
    CC = inp;
else
    plotCOMs = 1;
    COMs = inp;
end

if ~exist('colors', 'var')
    colors = hot(2*length(inp));
    colors = colors(end:-1:1,:);
end

%{
figure;
imagesc(im);
%     imagesc(log(im));
hold on;
%     colormap gray
%}

for rr = 1:length(inp)
    if plotCOMs
        plot(COMs(rr,2), COMs(rr,1), 'r.')
        
    else
        %[CC, ~, COMs] = setCC_cleanCC_plotCC_setMask(Ain, imHeight, imWidth, contour_threshold, im);
        if ~isempty(CC{rr})
            plot(CC{rr}(2,:), CC{rr}(1,:), 'color', colors(rr, :))
        else
            fprintf('Contour of ROI %i is empty!\n', rr)
        end
    end
end