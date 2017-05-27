function movie_play(movie, frames, pauseDur, doimadj, figPar)
%
% frames -- optional, if not provided all frames will be played
% pauseDur -- optional, if a non-numeric value, pause will be manual, ie to
% play the movie you have to press the keyboard frame by frame.
% figNum_subplot = [1, 211]; % figure gcf and subplot number

%%
if ~exist('frames', 'var') || isempty(frames)
    frames = 1:size(movie,3);
end

if ~exist('pauseDur', 'var')
    pauseDur = 1E-1;
end
if length(frames)==1
    pauseDur = nan;
end

if ~exist('doimadj', 'var')
    doimadj = 0;
end

if ~exist('figPar', 'var')
    figPar = [];
end


cLims = [min(movie(:)) max(movie(:))];

%%

if isempty(figPar)
    figure;
else
    figure(figPar.num)
    subplot(figPar.subplot)
end


for f = frames
    
    thisfr = movie(:,:,f);
    if doimadj
        thisfr = imadjust(thisfr);
    end
    
    imagesc(thisfr, cLims)
    axis image
    colormap gray
    set(gcf,'name', num2str(f))
    
    if ~isnan(pauseDur)
        if ~isnumeric(pauseDur)
            pause
        else
            pause(pauseDur)
        end
    end
    
end



