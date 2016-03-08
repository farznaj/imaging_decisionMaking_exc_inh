%%
mousename = 'FNI17'; fprintf(['\n====================',mousename,'====================\n'])
day = '151013';
filenumber = '001'; % indicates mdf file number.
numCh = 2; % number of saved channels

dir_local = 'C:\Users\fnajafi\Documents\Data\';


%% Set some parameters.
imdir_server = fullfile('\\sonas-hs.cshl.edu\churchland\data', mousename, 'imaging', day);

imfilename = strcat(day, '_', filenumber);
imdir = fullfile(dir_local, mousename, day, 'imaging');
if ~exist(imdir, 'dir'), mkdir(imdir), end


%% Copy mdf file from server to your local PC.
mdfFileName = [day, '_', filenumber, '.mdf'];

cd(imdir)
if ~exist(mdfFileName, 'file')
    s = fullfile('\\sonas-hs.cshl.edu\churchland\data', mousename, 'imaging', day, mdfFileName);
    copyfile(s, imdir)
end


%% Open the mdf file using MCSX library.
[mfile, OpenResult] = openMDF(mdfFileName);

if ~OpenResult
    error('Unable to open MDF file!!')
end


%% Read some params from the mdf file.
frameCount = mfile.ReadParameter('Frame Count');
frameCount = str2double(frameCount)

imWidth = mfile.ReadParameter('Frame Width')
imHeight = mfile.ReadParameter('Frame Height')


%% Set input params for mdfToTif function (total frame nums.
numFramesEachTif = 8000/numCh; % each tif file can have ~8000 frames.
tifList = cell(1, ceil(frameCount/numFramesEachTif));
for itif = 1:length(tifList)
    tifList{itif} = sprintf('%s_%02d', imfilename, itif);
end
showcell(tifList)

frameArr = 1:numFramesEachTif:numFramesEachTif*length(tifList);
frameArr = [frameArr frameCount+1]


%% Convert mdf file to all its tif files.
mdfToTif(mfile, tifList, frameArr)


%% The analog data is not read accurately.
%{
% AI0 : trial code
% AI1 : slow mirror
% AI2 : trial start

analogData = NaN(frameCount,3);

for chI = 1:3
%     analogData(:,chI) = mfile.ReadAnalog(chI, frameCount, 0);    
    analogData(:,chI) = invoke(mfile,'ReadAnalog', chI, frameCount, 0);
end

figure; 
plot(analogData)
%}


%% use mcsx library to read mdf file in matlab.
chToRead = 1;
% movie = 
for framenum = 1; % frameCount
    % frame = invoke(mfile,'ReadFrame', chI, framenum);
    rawframe = mfile.ReadFrame(chToRead, framenum);
    rawframe = rawframe';
end

% class(rawframe) % int16... read mdf files have negative values.

% figure; imagesc(rawframe, [0 max(rawframe(:))]);
% axis image, colormap(gray)

%% scale and convert rawframe to match mview_converted_tif

rawframe_scaled_u = uint16(double(rawframe) * 65535 / 2047);

% figure('name', 'rawframe_scaled_u'); imagesc(rawframe_scaled_u, [min(rawframe_scaled_u(:)) max(rawframe_scaled_u(:))]);
% axis image, colormap(gray);

%% imwrite as tif the mdf-read, scaled and converted mat file. 
imwrite(rawframe_scaled_u, 'rawframe_scaled_u.TIF', 'Compression', 'none') % 'WriteMode', 'append'


%% imread a converted tif file

% cd(imdir)
% tif_files = dir([imfilename,'_*.tif']);
% tifList = {tif_files.name};

tifList = {'mviewConvertedTif.TIF'}; % tifList = {'mdf_uint16.TIF'};  
ff = readTiffSet(tifList);

% only look at the frame under study (framenum)
imread_mviewConvertedTif = ff(:,:,framenum);
class(imread_mviewConvertedTif)
size(imread_mviewConvertedTif)

figure('name', 'imread mviewConvertedTif'); imagesc(imread_mviewConvertedTif, [min(imread_mviewConvertedTif(:)) max(imread_mviewConvertedTif(:))]);
axis image, colormap(gray)


%% final test: The following are equal: mviewConvertedTif  and  mdfRead_scaled_uint16Converted
isequal(imread_mviewConvertedTif, rawframe_scaled_u)


%% Important and useful summaries 

% clim and caxis do the same thing. they change only the visualization of
% the data, not the actual data pixel values. 
% colormap also only changes the visualization of data, not the actual
% pixel values.

%  what Aaron said is true when comparing pixel values of the following:
% a. mdf converted to tif openned in imageJ
% b. mdf read by mview, converted to uint16, imwrite to tiff, openned in
% imagej
%  a = b * (65535 / 2047) % remember when mview converts to tif, it scales.
%  mdf files go upto 2047. but converted tif files go upto 65535

% imreading a converted TIF is quite safe; ie it does not change pixel
% values.


% the below test is done and all is approved and safe. so go ahead and
% convert mdf to tif through matlab. it is just like using mview option for
% doing so.

% now what about your new method of creating tif files: 
% use mcsx to read mdf in matlab. 
% do the conversion and scaling:
    % convert to double and scale by 65535 / 2047.
    % convert to uint16. 
% imwrite it
% make sure it is same as the imread converted tif file.


% The following is done and confirmed. matt: At some point, we should probably make absolutely sure that changing the histo in the MScan interface doesn't change the values saved in the tif. It shouldn't, but it's worth checking.
% I have not checked the following: matt in preprocessCaMovies :% The apparent brightness changes, but I think this is just a scaling issue from a header parameter I can't change.



% to do:
% ask Aaron why mscan max pixel is 2047 instead of 2^16/2.
% finalize the code so the automatic conversion is done.

%%
% when you read mdf, you convert it to uint16, write it as tiff, open it in
% imageJ, the pixel values are the same as what you see in mview, except for negative values in mview, which turn to 0 in tif file. 

%{
rawframeu = uint16(rawframe);

imwrite(rawframeu, 'mdfRead_uint16.TIF')

% figure; imagesc(rawframeu, [0 100])
% % caxis([0 100]) 
% c = gray(64); % gray(double(max(rawframeu(:))));
% axis image, colormap(c);

figure; imagesc(rawframeu, [min(rawframeu(:)) max(rawframeu(:))]);
axis image, colormap(gray);
%}


%%
% you tested what you wrote below. matlab imread certainly does not change
% values. They are exactly same as imageJ values!!

% figure out why matlab imread screws up pixel values of tif file... does
% it simply scale them?? 

% in matlab imread a converted tif file
% in imageJ save the pixel intensities of the converted tif file as a text file. read the text file.
% compare pixel intensities of the 2 methods above to see if they match.

