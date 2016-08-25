gaussF = fspecial('gaussian', 10, 2);
medianReturn = imfilter(sdImage{2},gaussF,'symmetric');
figure; imagesc(watershed(-medianReturn));
a = (watershed(-medianReturn));
b = medianReturn;
b(a==0) = 0;
figure; imagesc(b)