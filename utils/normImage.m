function im = normImage(im, qt, lv)

if ~exist('qt','var')
    qt = 0.995;
end

if ~exist('lv','var')
    lv = min(im(:));
end


% from matt
im = im - lv;
softImMax = quantile(im(:), qt);
im = im / softImMax;
im(im > 1) = 1; 

im(im < 0) = 0; 