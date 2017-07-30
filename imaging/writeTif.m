function writeTif(movie, outFile, channels2write)
% writeTif(movie, outFile, channels2write)
% Write mat to tif file

% tifList = {'150825_002_1.tif'};
% outSuffix = '_MCM';

if ~iscell(movie)
    a{channels2write} = movie;
    movie = a;
end

if ~exist('channels2write', 'var') || isempty(channels2write)
    channels2write = find(~cellfun(@isempty, movie)); % 1:size(movieMC);
end


%%

for ch = channels2write % unlike the raw tifs, the motion corrected tifs will have channel 1 and channel 2 in separate files, instead of alternating frames in the same file. (FN)
    % The apparent brightness changes, but I think this is just a scaling issue
    % from a header parameter I can't change
    
    outFile = sprintf('%s_ch%d.TIF', outFile, ch);
    movie2write = uint16(movie{ch});
    nFramesPerMovie = size(movie2write,3);    
    
%     frame = 0;
%     t = 1;
    %       for t = 1:length(tifList)
    
    %{
    % Figure out filename
    [fPath, fStem, fExt] = fileparts(tifList{t});
    %         outFile = fullfile(fPath, [fStem outSuffix fExt]);
    outFile = fullfile(fPath, [[fStem,'_ch',num2str(ch)] outSuffix fExt]);
    fprintf('Writing file %s (%d/%d)\n', outFile, t, length(tifList));
    % Figure out frames
    frames = frame + 1 : frame + nFramesPerMovie(t);
    %}
    
    fprintf('Writing file %s\n', outFile);   
    frames = 1 : nFramesPerMovie;
    
    imwrite(movie2write(:, :, frames(1)), outFile, ...
        'Resolution', [size(movie2write, 2) size(movie2write, 1)], 'Compression', 'none');
    
    if length(frames) > 1
        for f = 2:length(frames)
            if mod(f, 100) == 0
                fprintf('%d ', f);
            end
            if mod(f, 1000) == 0
                fprintf('\n');
            end
            
            imwrite(movie2write(:, :, frames(f)), outFile, ...
                'Resolution', [size(movie2write, 2) size(movie2write, 1)], 'Compression', 'none', ...
                'WriteMode', 'append');
        end
    end   
end

fprintf('\nDone.\n');

