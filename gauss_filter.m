function [stack_filtered, filter_mask] = gauss_filter(stack,height,hp,lp)
F = @(x) fftshift(fft2(ifftshift(x)));
iF = @(x) fftshift(ifft2(ifftshift(x)));
[m,n] = size(stack,1:2);
[x,y] = ndgrid(1:m,1:n);
center_x = floor(m/2);
center_y = floor(n/2);
if lp == 0
    filter_mask = -height.*exp(-(((center_x-x)./hp).^2+((center_y-y)./hp).^2)/2);
elseif hp == 0
    filter_mask = height.*exp(-(((center_x-x)./lp).^2+((center_y-y)./lp).^2)/2);
else
    filter_mask = height.*exp(-(((center_x-x)./lp).^2+((center_y-y)./lp).^2)/2) -height.*exp(-(((center_x-x)./hp).^2+((center_y-y)./hp).^2)/2);
end
% figure;imshow(filter_mask,[]); title 'filter mask'
% figure;imshow(abs(F(filter_mask)),[]); title 'filter mask'

stack_filtered = zeros(size(stack));
for i = 1:size(stack,3)
    stack_filtered(:,:,i) = abs(iF(F(stack(:,:,i)).*(filter_mask)));
end

% figure(1);
% imshow(mean(stack,3),[]); title 'pre-filtered'
% figure(2);
% imshow(mean(stack_filtered,3),[]); title 'filtered'

