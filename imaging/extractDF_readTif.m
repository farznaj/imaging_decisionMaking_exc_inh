mousename = 'fni17';
imagingFolder = '151102';
mdfFileNumber = 1; % or tif major
ch2read = 2;
convert2double = 1;

Y = tifToMat(mousename, imagingFolder, mdfFileNumber, ch2read, convert2double);
Y = Y{ch2read};

[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;                                          % total number of pixels

Yr = reshape(Y,d,T);
clear Y;

%%
dataPath = '/sonas-hs/churchland/nlsas/data/data';
tifFold = fullfile(dataPath, mousename, 'imaging', imagingFolder);

load(fullfile(tifFold, 'demo_script_class_fni17_151102_001'))

%%
[C_df, ~, S_df] = extractDF_F(obj, Yr);

%%
save('demo_script_class_fni17_151102_001_newDF', 'obj')

