function im = normImage(im)
% from matt
im = im - min(im(:));
softImMax = quantile(im(:), 0.995);
im = im / softImMax;
im(im > 1) = 1; 