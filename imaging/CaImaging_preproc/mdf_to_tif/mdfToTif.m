function mdfToTif(mfile, tifName, framenums, channelsSaved, ImgDescription)
% Reads mdf file in matlab (using MCSX library), and uses imwrite to write
% it to tif.
%
% INPUTS:
% mfile: mdf openned in matlab using MCSX library.
% framenums: frame numbers of the MDF file that will be written as tif.

%%
if ~exist('ImgDescription', 'var')
    ImgDescription = '';
end

%%
for framenum = framenums
    
    %% print out the framnum that is being written.
    if mod(framenum, 100) == 0
        fprintf('%d ', framenum);
    end
    if mod(framenum, 1000) == 0
        fprintf('\n');
    end
    
    %% write channels alternatingly.
    for chToRead = channelsSaved
        %
        % use mcsx library to read mdf file in matlab.
        
        % frame = invoke(mfile,'ReadFrame', chI, framenum);
        rawframe = mfile.ReadFrame(chToRead, framenum);
        rawframe = rawframe';
        
        % class(rawframe) % int16... read mdf files have negative values.
        % figure; imagesc(rawframe, [0 max(rawframe(:))]);
        % axis image, colormap(gray)
        
        %% scale and convert rawframe to match mview_converted tif files.
        
        rawframe_scaled_u = uint16(double(rawframe) * 65535 / 2047);
        
        % figure('name', 'rawframe_scaled_u'); imagesc(rawframe_scaled_u, [min(rawframe_scaled_u(:)) max(rawframe_scaled_u(:))]);
        % axis image, colormap(gray);
        
        %%%
%         rawframe_scaled_u = movie(:,:,framenum, chToRead);
        
        %% imwrite as tif the mdf-read, scaled and converted mat file.
        
        if framenum == 1 && chToRead == 1
            imwrite(rawframe_scaled_u, [tifName, '.TIF'], 'Compression', 'none', 'Description', ImgDescription)
            
        else
            imwrite(rawframe_scaled_u, [tifName, '.TIF'], 'Compression', 'none', 'WriteMode', 'append', 'Description', ImgDescription);
        end
        
    end
    
    
end


%% read all frames and save movie as mat file.
%{
imWidth = mfile.ReadParameter('Frame Width');
imHeight = mfile.ReadParameter('Frame Height');
movie = int16(zeros(imHeight, imWidth, length(framenums), length(channelsSaved))); 

for chToRead = channelsSaved
    for framenum = framenums        
        % use mcsx library to read mdf file in matlab.        
        rawframe = mfile.ReadFrame(chToRead, framenum);
        movie(:, :, framenum, chToRead) = rawframe';
    end
end
movie = uint16(double(movie) * 65535 / 2047);

save(tifName(1:end-4), 'movie')
%}
