
function [lp] = low_pass_calculation(image_stack,threshold_value,ps,Flambda,mag,NA_obj,NAs)



F = @(x) fftshift(fft2(ifftshift(x)));
iF = @(x) fftshift(ifft2(ifftshift(x)));

img = image_stack;
[N,M,Nimg] = size(img);


fx = (-M/2:(M/2-1))./(ps*M); 
fy = (-N/2:(N/2-1))./(ps*N); % fourier space--spatial frequency
NAx = fx*Flambda; 
NAy = fy*Flambda;
[fxx,fyy] = meshgrid(fx,fy);

Pupil_obj = zeros(N,M);
r_obj     = (fxx.^2+fyy.^2).^(1/2); % spatial frequency of obj
Pupil_obj(r_obj<NA_obj/(Flambda))=1;
T_incoherent = abs(F(abs(iF(Pupil_obj)).^2));

Mean_PSF  = abs(iF(T_incoherent));
Mean_PSF = im2uint16(Mean_PSF);

% imwrite(Mean_PSF,'matlab_PSF.tif');

% figure; imshow(T_incoherent,[])

intensity_plot = T_incoherent(N/2-1,:);
intensity_plot = intensity_plot./max(intensity_plot);
plot(intensity_plot);title 'intensity profile of OTF'


% threshold_value = 0.5; % set the threshold for the supporting frequency

high_threshold_idx = find(intensity_plot>threshold_value);
lp = high_threshold_idx(1,size(high_threshold_idx,2)) - high_threshold_idx(1,1);
lp = lp/2;