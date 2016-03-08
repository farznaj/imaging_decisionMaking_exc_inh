function tifList = tifListSet(mousename, imagingFolder, mdfFileNumber, ch2read)
% tifList = tifListSet(mousename, imagingFolder, mdfFileNumber, ch2read)

%% set params
% mousename = 'fni17';
% imagingFolder = '151102';
% mdfFileNumber = 1; % or tif major
outName = [mousename,'-',imagingFolder, '-', num2str(mdfFileNumber)];
PP = struct;
PP.signalCh = 2;
PP.saveParams = false; % if 0, you don't need outName.
params = writeCaProcessParams(outName, mousename, imagingFolder, mdfFileNumber, PP);
% clear P

%% Read tif files to movieMC and convert it to double.
% tiffilename = sprintf('%s_%03d_*_ch2_MCM.TIF*', imagingFolder, mdfFileNumber);
% files = dir(fullfile(tifFold, tiffilename));

chAll = ch2read; % 2; % channelsToRead
for ch = chAll' 
  % Get list of MCM tif files corresponding to channel ch.
  tifNumsCh = params.tifNums(params.tifNums(:,4)==ch,:);
  tifList = cell(1, size(tifNumsCh,1));
  for f = 1:length(tifList)
      tifList{f} = fullfile(params.tifFold, assembleCaImagingTifName(tifNumsCh(f, :), params.oldTifName(f))); 
  end
  
end

