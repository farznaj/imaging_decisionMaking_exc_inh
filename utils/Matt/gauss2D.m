function h = gauss2D(siz, sigma)
% h = gauss2D(filtSize, sigma)
% 
% Create a 2D Gaussian filter appropriate for use with imfilter(). Much
% like fspecial('gaussian', ...), but takes sigma with length 2 and permits
% the two sigmas to be different.
% 
% Since this is intended for images, gauss2D expects first value to be "y",
% and second to be "x" for both arguments.
% 
% Code mostly copied from fspecial with the one key modification.

siz = (siz - 1) / 2;

[x, y] = meshgrid(-siz(2):siz(2), -siz(1):siz(1));
arg = -(x .* x) / (2 * sigma(2) * sigma(2)) - (y .* y) / (2 * sigma(1) * sigma(1));

h = exp(arg);
h(h < eps * max(h(:))) = 0;

sumh = sum(h(:));
if sumh ~= 0,
  h = h / sumh;
end
