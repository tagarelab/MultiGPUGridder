clc
close all
clear obj 

addpath('./src')
addpath('./utils')
addpath('./bin') % The compiled mex file is stored in the bin folder


disp("Resetting devices...")
for i = 1:4
    reset(gpuDevice(i));
end

%% Create a volume 
% Initialize parameters
tic

nBatches = 1;
nGPUs = 1;
nStreams = 4;
volSize = 32;
n1_axes = 50;
n2_axes = 5;

kernelHWidth = 2;

interpFactor = 2.0;

origSize   = volSize;
volCenter  = volSize/2  + 1;
origCenter = origSize/2 + 1;
origHWidth = origCenter - 1;

%% Fuzzy sphere
disp("fuzzymask()...")
vol=fuzzymask(origSize,3,origSize*.25,2,origCenter*[1 1 1]);
size(vol)

% Use the example matlab MRI image to take projections of
% load mri;
% img = squeeze(D);
% img = imresize3(img,[volSize, volSize, volSize]);
% vol = single(img);
% easyMontage(vol,1);


% vol = padarray(vol,[floor(volSize/2) floor(volSize/2)],0,'pre');
% vol = padarray(vol,[floor(volSize/2) floor(volSize/2)],0,'post');

% padded_vol = zeros(size(vol) * 2);
% 
% padded_vol(size(vol,1)/2 : size(vol,1)*1.5-1, size(vol,2)/2 : size(vol,2)*1.5-1, size(vol,3)/2 : size(vol,3)*1.5-1) = vol;

% vol = padded_vol;
% size(vol)
% imagesc(vol(:,:,1))



%% Define the projection directions
coordAxes=single([1 0 0 0 1 0 0 0 1]');
coordAxes=[coordAxes create_uniform_axes(n1_axes,n2_axes,0,10)];
coordAxes = coordAxes(:);
nCoordAxes = length(coordAxes)/9;

%% MATLAB pre-processing to covert vol to CASVol

% interpBoc and fftinfo are needed for plotting the results
disp("MATLAB Vol_Preprocessing()...")
tic
[CASVol, CASBox, origBox, interpBox, fftinfo] = Vol_Preprocessing(vol, interpFactor);
toc

disp("Volume size: " + num2str(volSize))
disp("Number of coordinate axes: " + num2str(nCoordAxes))
 
%% Initialize the multi GPU gridder
obj = MultiGPUGridder_Matlab_Class();
obj.SetNumberBatches(nBatches);
obj.SetNumberGPUs(nGPUs);
obj.SetNumberStreams(nStreams);
obj.SetMaskRadius(single(size(vol,1)*interpFactor/2 - 1));


disp("SetVolume()...")
tic
obj.setVolume(single(CASVol))
toc

% CUDA_Vol = obj.GetVolume();
% CUDA_Vol = CUDA_Vol / 4;

% CUDA_Vol = obj.CUDA_Return('gpuVol_0');
% 
% figure
% slice = 10;
% h(1) = subplot(1,3,1)
% imagesc(CUDA_Vol(:,:,slice))
% title("GPU Processed Input Volume")
% h(2) = subplot(1,3,2)
% imagesc(CASVol(:,:,slice))
% title("Matlab Processed Input Volume")
% h(3) = subplot(1,3,3)
% imagesc(CUDA_Vol(:,:,slice) - CASVol(:,:,slice))
% title("Difference")
% colorbar
% linkaxes(h, 'xy')
% zoom on
% 

disp("SetAxes()...")
obj.SetAxes(coordAxes)

disp("SetImgSize()...")
obj.SetImgSize(int32([size(vol,1)*interpFactor, size(vol,1)*interpFactor,nCoordAxes]))

% This is the size of the non-zero padded projection images
% obj.SetImgSize(int32([size(vol,1)/interpFactor, size(vol,1)/interpFactor,nCoordAxes]))





%% Run the forward projection kernel
% clc
disp("Forward_Project()...")
obj.forwardProject()

disp("Displaying allocated memory()...")
obj.CUDA_disp_mem('all')
obj.disp_mem('all')

% disp("GetImgs()...")
InterpCASImgs = obj.GetImgs();

disp("imgsFromCASImgs()...")
imgs=imgsFromCASImgs(InterpCASImgs, interpBox, fftinfo); 

size(InterpCASImgs)
max(InterpCASImgs(:))

% Check to see if all the projections are there
for i = 1:size(imgs,3)
    temp = imgs(:,:,i);
   if (max(temp(:)) <= 0)
       disp("No projection for slice " + num2str(i))
   end
end

easyMontage(imgs,1);
colormap gray


% disp("easyMontage()...")
% easyMontage(imgs,2);


%% Run the back projection kernel
disp("ResetVolume()...")
obj.ResetVolume()

CAS_projection_imgs = CASImgsFromImgs(imgs,interpBox,fftinfo);
obj.SetImages(CAS_projection_imgs)


% 
% slice = 10
% figure
% imagesc(InterpCASImgs(:,:,slice) - CAS_test_imgs(:,:,10))
% colorbar



disp("Back_Project()...")
obj.Back_Project()

disp("Get_Volume()...") % Get the volumes from all the GPUs added together
volCAS = obj.GetVolume();

 
% vol_0 = obj.CUDA_Return('gpuVol_0');
% vol_1 = obj.CUDA_Return('gpuVol_1');
% vol_2 = obj.CUDA_Return('gpuVol_2');
% vol_3 = obj.CUDA_Return('gpuVol_3');
% 
% volCAS = vol_0 + vol_1 + vol_2 + vol_3;
% 
% figure
% imagesc(volCAS_1(:,:,10) - volCAS(:,:,10))
% colorbar

%% Get the density of inserted planes by backprojecting CASimages of values equal to one
disp("Get Plane Density()...")
interpImgs=ones([interpBox.size interpBox.size size(coordAxes,1)/9],'single');
obj.ResetVolume();
obj.SetImages(interpImgs)
obj.Back_Project()
volWt = obj.GetVolume();

%% Normalize the back projection result with the plane density
% Divide the previous volume with the plane density volume
volCAS=volCAS./(volWt+1e-6);
                
% Reconstruct the volume from CASVol
disp("volFromCAS()...")
volReconstructed=volFromCAS(volCAS,CASBox,interpBox,origBox,kernelHWidth);
% volReconstructed = real(fftshift(ifftn(fftshift(volCAS))));

% volReconstructed = volReconstructed./(size(vol,1)*size(vol,1));

disp("easyMontage()...")
easyMontage(volReconstructed,2);

% easyMontage(vol,4);
% easyMontage(vol - volReconstructed,5);

%% Compare with the previous GPU gridder code
% 
x = load("gpuGridderResults.mat");
gpuGridder = x.gpuGridder


close all
easyMontage(gpuGridder.forward_project,3);
easyMontage(imgs,4);

easyMontage(gpuGridder.forward_project - imgs,5);

close all
slice = 215;

figure
for slice = 1:size(imgs,3)
    
    h(1) = subplot(2,3,1)
    imagesc(imgs(:,:,slice))
    title("Multi GPU Forward Projection")
    h(2) = subplot(2,3,2)
    imagesc(gpuGridder.forward_project (:,:,slice))
    title("gpuGridder Forward")
    h(3) = subplot(2,3,3)
    diff_img = imgs(:,:,slice) - gpuGridder.forward_project(:,:,slice);
%     diff_img(diff_img < 0.00001) = 0; % Clean up the visualization to better debug
    imagesc(diff_img)
    title("Difference")
    colorbar
    linkaxes(h, 'xy')
    zoom on
    


    h(4) = subplot(2,3,4)
    imagesc(volReconstructed(:,:,slice))
    title("Multi GPU Back Projection With Weighing")
    h(5) = subplot(2,3,5)
    imagesc(gpuGridder.back_project(:,:,slice))
    title("gpuGridder Back Projection With Weighing")
    h(6) = subplot(2,3,6)
    imagesc(volReconstructed(:,:,slice) - gpuGridder.back_project(:,:,slice))
    title("Difference")
    
    colorbar
    linkaxes(h, 'xy')
    zoom on
    
    pause(0.005)

end

diff_forward = imgs - gpuGridder.forward_project;
max(diff_forward(:))

max_diff = volReconstructed - gpuGridder.back_project;
max_diff = max(max_diff(:))

max(gpuGridder.forward_project(:))
max(imgs(:))


%% Free the memory
% obj.CUDA_Free('all')
clear obj
clear all

toc