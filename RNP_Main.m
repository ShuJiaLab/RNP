%% Image Dictionary and Setting
clear all, close all, clc;

img_dir = "E:\Data_Summary_For_Paper\Figure_1_Setup_Alg_Simulation\Algo";
cd(img_dir)
addpath('D:\Dropbox (GaTech)\Jia Lab\Code\RNP_Final');
addpath('D:\Dropbox (GaTech)\Jia Lab\Code\RNP_Final\RobustPCA');
% System setting -- calculating the size of filter in Fourier domain
Flambda = .680; 
k=2*pi/Flambda; % wavelength (micons) and wave number
mag = 10;
ps = 6.5/mag; % Pixels size (microns)
NA_obj = 0.3;
NAs = 0.3;

%% Image loading and filtering
image_stack = tiffreadVolume('6-4beads-thin-ring-1.tif');
image_stack = double(image_stack);

[G H K] = size(image_stack);
crop_size = min(G,H);
crop_size = 300; % crop image to designed size. Note this code now only support square image input
% for example G = H
win1 = centerCropWindow2d([G H],[crop_size crop_size]);
x = win1.XLimits;
y = win1.YLimits;
image_stack_cropped = image_stack(y(1):y(2),x(1):x(2),:);


threshold_value = 0.7;% set the threshold for the supporting frequency
lp = low_pass_calculation(image_stack_cropped,threshold_value,ps,Flambda,mag,NA_obj,NAs);
hp = 2; % this number usually is between 1-3
height = 1;
% Image cropping, skip this step if there is no need to crop



n = 1; % 1 -- filter; 0-- no filter

if n == 1
    [image_stack_filtered, filter_mask] = gauss_filter(image_stack_cropped,height,hp,lp);
    figure; subplot(1, 2, 1);
    imshow(mean(image_stack_cropped,3),[]);
    title('Pre-filtered');

    subplot(1, 2, 2);
    imshow(mean(image_stack_filtered,3),[]);
    title('Filtered');

else
    image_stack_filtered = image_stack_cropped;
    imshow(mean(image_stack_filtered,3),[]);
    title('No filter processing');
end
%%
image_stack_filtered = imresize(image_stack_filtered,0.5,"nearest");
% large input images will lead to long processing time
% You can shrink the image size to short it but it will cause loss of
% resolution and inaccuracy of reconstruction
%% Feature Extraction
PCA_Sparsity = 10; 
% FLFsub_num_dia = 1;

[G,H,Img_index] = size(image_stack_filtered);
Image_Index = [1,Img_index]';
idxPd = 1;
%
Frame_ini = Image_Index(idxPd);
Frame_end = Image_Index(idxPd+1);
Frame_Ind = Frame_ini:Frame_end;
Frame_Len = length(Frame_Ind);

Img_Period =  image_stack_filtered;
Img_reshaplowrank = zeros(Frame_Len,G.^2,1,'uint16');
Img_sparse = zeros(Frame_Len,G.^2,1,'uint16');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Img_Period_GPU = gpuArray( Img_Period );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Sparse decomposition
Img_reshap = reshape(permute(squeeze(Img_Period_GPU(1:G,1:G,:)),[3,1,2]),Frame_Len,G.^2);

PCA_Lambda = 1/sqrt(max( [Frame_Len,G.^2] )) /PCA_Sparsity;
PCA_Mu     = PCA_Lambda*10;
tic
[Img_reshaplowrank(:,:,1),Img_sparse(:,:,1)] = ...
    RobustPCA( single(Img_reshap), PCA_Lambda, PCA_Mu,1e-5, 5000);
toc

% save([Savepath_UniN,'_',num2str(Frame_ini),'-',num2str(Frame_end),Form_Dat],'FLFsub_reshaplowrank','FLFsub_reshap_sparse');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Decomposed matrix reshaped to FLFsubimage stacks
FLFsubLowrank = reshape(permute(reshape(Img_reshaplowrank,Frame_Len,G.^2,1,1), ...
    [2,4,3,1]),G,G,1,Frame_Len);
FLFsub_Sparse = reshape(permute(reshape(Img_sparse,Frame_Len,G.^2,1,1), ...
    [2,4,3,1]),G,G,1,Frame_Len);
FLFsubLowrank(FLFsubLowrank < 0) = 0;
FLFsub_Sparse(FLFsub_Sparse < 0) = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% put together elementary images and save
FLFvidLowrank = gpuArray( zeros(G,G,Frame_Len,'uint16') );
FLFvid_Sparse = gpuArray( zeros(G,G,Frame_Len,'uint16') );

FLFvidLowrank(1:G,1:G,:) = uint16(FLFsubLowrank(:,:,1,:));
FLFvid_Sparse(1:G,1:G,:) = uint16(FLFsub_Sparse(:,:,1,:));

Sparse_matrix = gather(FLFvid_Sparse);
low_rank = gather(FLFvidLowrank);
% figure;
% imshow(mean(Sparse_matrix,3),[]);
Sparse_matrix = double(Sparse_matrix);
figure; 
subplot(1, 3, 1);
imshow(mean(image_stack,3),[]);
title('Pre-filtered');

subplot(1, 3, 2);
imshow(mean(image_stack_filtered,3),[]);
title('Filtered');

subplot(1, 3, 3);

imshow(mean(Sparse_matrix,3),[]);
title('Filtered + RPCA');
%% save images
saveMultipageTiff(Sparse_matrix,'Sparse10.tif')
%
saveMultipageTiff(low_rank,'LowRankImg.tif')
%
saveMultipageTiff(image_stack_filtered,'filtered image.tif')
%% Rank estimation
Sparse_matrix = gather(FLFvid_Sparse);
Sparse_matrix = double(Sparse_matrix);
ref_stack = zeros(size(Sparse_matrix,3),size(Sparse_matrix,1).^2);
for i = 1:size(Sparse_matrix,3) 
    % temp = imfilter(FluoMATfilt_t(:,:,i),high_pass_kernal,'conv','replicate','same');
    temp = Sparse_matrix(:,:,i);
    ref_stack(i,:) = reshape(temp,1,[]);
    % disp(i);
end
% Rank estimation
ini_rank = 18;
final_rank = 22;
FRO_NORM = zeros(1,final_rank-ini_rank);
Hstore = cell(final_rank,1);
step_size = 1; % since RNP has some tolerence in rank estimation, you can set different step size to find the optimum rank
% If you want to find the best rank, you can use large step size to find a
% range, then set step size to 1 and run this section again
for i = ini_rank:step_size:final_rank
    fprintf(['current rank is %d.\n'],i)
    opt = statset('MaxIter',30,'Display','final');
    [W0,H0] = nnmf(ref_stack,i,'Replicates',5,...
        'options',opt,'algorithm','mult');
    opt = statset('MaxIter',1000,'Display','iter','TolFun',1e-6);
    [Wtemp,Htemp,Dtemp] = nnmf(ref_stack,i,'W0',W0,'H0',H0,...
        'options',opt,'algorithm','als');
    FRO_NORM(1,i-ini_rank+1) = Dtemp;
    Hstore{i,1} = Htemp;
    if i == ini_rank
        D = Dtemp;
        W = Wtemp;
        H = Htemp;
        EstimatedNumber = i;
        fprintf(['Initial RMSE: %d\n' ...
            'Initial rank: %f\n'],D,ini_rank)
    else
        if Dtemp < D % update the RMSE and the new matrix
            D = Dtemp;
            W = Wtemp;
            H = Htemp;
            EstimatedNumber = i;
            fprintf(['Updated RMSE: %d\n' ...
                'Updated rank: %f\n'],D,i)
           
        else
            fprintf('The RMSE is not decreasing.\n')
            fprintf('The current estimated rank is %d.\n',EstimatedNumber)
        end
    end
end

fprintf('The rough estimate rank is %d.\n',EstimatedNumber)
%% 
% Previous section test multiple ranks, and they are stored in Hstore
% if you don't run this section, the default rank used in next section will
% be the rank with minimum RMSE
EstimatedNumber = 14;
H = Hstore{EstimatedNumber,1};
%%
global xpixel ypixel
xpixel = 2*size(Sparse_matrix,2); % the size of the image in the X
ypixel = 2*size(Sparse_matrix,1); % the size of the image in the Y 
% xpixel = 66; % the size of the image in the X direction Lcam-2
% ypixel = 74; % the size of the image in the Y direction Ccam
M = cell(EstimatedNumber,1);% create the empty matrix to store the fingerprint
figure;
for kk=1:EstimatedNumber
    tempFP = reshape(H(kk,:),xpixel/2,ypixel/2);
    % [tempFP, filter_mask] = gauss_filter(tempFP,height,hp,lp);
    tempFP = padarray(tempFP,[xpixel/4 ypixel/4],0);
    M{kk} = tempFP;
    %     figure;
    %     imagesc(M{kk}),daspect([1 1 1]), title('Fluo speckle - raw data'), colormap hot;colorbar;
    %     pause(0.5)
    subplot(floor(sqrt(EstimatedNumber)+1),floor(sqrt(EstimatedNumber)+1),kk),imshow(M{kk},[]);colormap hot; title(['FingerPrint_ #',num2str(kk)]), colormap hot;
    % pause(0.1)
end
%% Calculating the relative position among emitters
close all;
O = cell(EstimatedNumber,1);%Create the empty for the partial iamges
recon_image = zeros(xpixel,ypixel,EstimatedNumber);
Maximum_intensity = zeros(EstimatedNumber,EstimatedNumber);
xx = zeros(EstimatedNumber,EstimatedNumber);
yy = zeros(EstimatedNumber,EstimatedNumber);
noise_lvl = 1; % for wiener filter deconv.

for k=1:EstimatedNumber
    PSF = M{k};
    for i = 1:EstimatedNumber
        recon_image(:,:,i) = deconvwnr(M{i},PSF,noise_lvl);
        threshold_intensity = max(max(recon_image(:,:,i))).*0.3; % clean the image by using threshold;
        binary_mask = recon_image(:,:,i)>threshold_intensity; % create the binary mask to filter out background intensity;
        recon_image(:,:,i) = recon_image(:,:,i).*binary_mask; % filter out the background


        [xx_temp,yy_temp] = find(recon_image(:,:,i) == max(max(recon_image(:,:,i))));% Record the position of maximum value of each image
        if size(xx_temp,1)>1 || size(yy_temp,1)>1
            xx(i,k) = NaN;
            yy(i,k) = NaN;
        else
            xx(i,k) = floor(mean(xx_temp(:)));
            yy(i,k) = floor(mean(yy_temp(:)));
        end
        Maximum_intensity(i,k) = max(max(recon_image(:,:,i)));% Record the maximum value of each image

    end
    O{k}=sum(recon_image,3);% the partial image O_{k} by summing up
    % All results of using k th frame as the PSF
end
% display the different partial images
figure;
for jj=1:EstimatedNumber
    subplot(floor(sqrt(EstimatedNumber)+1),floor(sqrt(EstimatedNumber)+1),jj),imshow(O{jj},[]);colormap hot; title(['O_ #',num2str(jj)]), colormap hot;
    % pause(0.1)
end
%%
Size = 1500;
First_pattern = 18; % check the partial images, don't use noisy image as the first_pattern
[Reached_Pattern,Global_image,SNormlized_Maximum_intensity,num] = mergeimages(Maximum_intensity,O,EstimatedNumber,xx,yy,xpixel,ypixel,First_pattern,Size);
% # the first pattern should not be noise pattern
figure(1);

% imshow(Global_image(250+(-100:100),250+(-100:10   0)),[]);colormap gray;
imshow(Global_image(Size/2+(-Size/5:Size/5),Size/2+(-Size/5:Size/5)),[]);colormap gray;
% # second reconstruction: it is used to reconstruct the full object
[Reached_Pattern,Global_image,SNormlized_Maximum_intensity,num] = mergeimages(Maximum_intensity,O,EstimatedNumber,xx,yy,xpixel,ypixel,Reached_Pattern(end),Size);
figure(2);
imshow(Global_image(Size/2+(-Size/5:Size/5),Size/2+(-Size/5:Size/5)),[]);colormap hot;
% imshow(Global_image(250+(-100:100),250+(-100:100)),[]);colormap gray;