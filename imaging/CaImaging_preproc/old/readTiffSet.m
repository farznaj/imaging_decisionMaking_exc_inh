function movie = readTiffSet(tifList)
% movie = readTiffSet(tifList)
%
% tifList should be a cell array containing the paths to tif files, in the
% order you want them read. movie will be a uint16 array of size imHeight x
% imWidth x nFrames, containing all the frames from all the tif files in
% tifList.


%% Figure out the number of frames in each tif file

tifInfo = cell(1, length(tifList));
nFramesPerMovie = NaN(1, length(tifList));

for t = 1:length(tifList)
  tifInfo{t} = imfinfo(tifList{t});
  nFramesPerMovie(t) = length(tifInfo{t});
end


%% Pre-allocate the movie

imWidth = tifInfo{t}(1).Width;
imHeight = tifInfo{t}(1).Height;

totalFrames = sum(nFramesPerMovie);

movie = uint16(zeros(imHeight, imWidth, totalFrames));


%% Read the tif files

frame = 0;
for t = 1:length(tifList)
  frames = frame + 1 : frame + nFramesPerMovie(t);
  
  for f = 1:length(frames)
    if mod(f, 1000) == 0
      fprintf('%d ', f);
    end
    
    movie(:, :, frames(f)) = imread(tifList{t}, f, 'Info', tifInfo{t});
    
  end
  fprintf('\n');
  
  frame = frame + nFramesPerMovie(t);
end




