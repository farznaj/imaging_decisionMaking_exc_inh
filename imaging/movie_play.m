function movie_play(movie, frames, pauseDur)

% frames -- optional, if not provided all frames will be played
% pauseDur -- optional, if a non-numeric value, pause will be manual, ie to
% play the movie you have to press the keyboard frame by frame.

%%
if ~exist('frames', 'var')
    frames = 1:size(movie,3);
end

if ~exist('pauseDur', 'var')
    pauseDur = 1E-3;
end

cLims = [min(movie(:)) max(movie(:))];

%%
figure;
for f = frames
    
    imagesc(movie(:,:,f), cLims)
    axis image
    colormap gray
    set(gcf,'name', num2str(f))
    
    if ~isnumeric(pauseDur)
        pause
    else
        pause(pauseDur)
    end
    
end



