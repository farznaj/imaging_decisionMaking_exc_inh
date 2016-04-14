function movie = maskMovie(movie, maskBounds)
% movie = maskMovie(movie, maskBounds)
% 
% Trim off the edges of a movie. This is desirable after motion correction.
%
% The input movie may be either a 3D array containing the movie data, or
% the path to a single .tif file. The output movie will always be a 3D
% array.
%
% maskBounds should be [x1 x2 y1 y2]. You probably want to get them by
% passing the pixelsShifts output of motionCorrectCaImagingFile() to
% determineMovieMaskBounds().


%% Read movie if necessary

if ischar(movie)
  tifName = movie;
  
  fprintf('Reading tiff\n');
  
  tifInfo = imfinfo(tifName);
  nFrames = length(tifInfo);
  imWidth = tifInfo(1).Width;
  imHeight = tifInfo(1).Height;
  
  % Read all the images out of the tiff
  
  % Pre-allocate movie
  movie = zeros(imHeight, imWidth, nFrames, 'uint16');
  
  % Read frames
  for f = 1:length(framesToUse)
    if mod(f, 100) == 0
      fprintf('%d ', f);
    end
    if mod(f, 1000) == 0
      fprintf('\n');
    end
    movie(:, :, f) = imread(tifName, 'Index', f, 'Info', tifInfo);
  end
end


%% Apply mask

movie = movie(maskBounds(3):maskBounds(4), maskBounds(1):maskBounds(2), :);

