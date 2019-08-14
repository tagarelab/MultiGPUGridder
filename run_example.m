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

nBatches = 4;
nGPUs = 4;
nStreams = 8;
volSize = 64;
n1_axes = 500;
n2_axes = nBatches * nStreams;

kernelHWidth = 2;

interpFactor = 1.0;

origSize   = volSize;
volCenter  = volSize/2  + 1;
origCenter = origSize/2 + 1;
origHWidth = origCenter - 1;

%% Fuzzy sphere
disp("fuzzymask()...")
vol=fuzzymask(origSize,3,origSize*.25,2,origCenter*[1 1 1]);
size(vol)

% Use the example matlab MRI image to take projections of
load mri;
img = squeeze(D);
img = imresize3(img,[volSize, volSize, volSize]);
vol = single(img);
% easyMontage(vol,1);


% vol = padarray(vol,[floor(volSize/2) floor(volSize/2)],0,'pre');
% vol = padarray(vol,[floor(volSize/2) floor(volSize/2)],0,'post');

padded_vol = zeros(size(vol) * 2);

padded_vol(size(vol,1)/2 : size(vol,1)*1.5-1, size(vol,2)/2 : size(vol,2)*1.5-1, size(vol,3)/2 : size(vol,3)*1.5-1) = vol;

vol = padded_vol;
size(vol)
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
% obj.SetMaskRadius(single((size(vol,1) * interpFactor)/2 - 1)); 
obj.SetMaskRadius(single(5)); 
% obj.SetMaskRadius(single((size(vol,1)/2 - 10))); 


disp("SetVolume()...")
tic
obj.SetVolume(single(CASVol))
toc

disp("SetAxes()...")
obj.SetAxes(coordAxes)

disp("SetImgSize()...")
obj.SetImgSize(int32([size(vol,1) * interpFactor, size(vol,1) * interpFactor,nCoordAxes]))





%% Run the forward projection kernel
% clc
disp("Forward_Project()...")
obj.Forward_Project()

disp("Displaying allocated memory()...")
obj.CUDA_disp_mem('all')
obj.disp_mem('all')


% test_imgs = obj.mem_Return('CASImgs_CPU_Pinned');
% size(test_imgs)
% max(test_imgs(:))

% close all
% imagesc(test_imgs(:,:,1))
% colormap gray

% disp("GetImgs()...")
InterpCASImgs = obj.GetImgs();
size(InterpCASImgs)
max(InterpCASImgs(:))

% Check to see if all the projections are there
for i = 1:size(InterpCASImgs,3)
    temp = InterpCASImgs(:,:,i);
   if (max(temp(:)) <= 0)
       disp("No projection for slice " + num2str(i))
   end
end

% InterpCASImgs = InterpCASImgs(:,:,1);
easyMontage(InterpCASImgs,2);
colormap gray

% disp("imgsFromCASImgs()...")
% imgs=imgsFromCASImgs(InterpCASImgs(:,:,1:10), interpBox, fftinfo); 

% disp("easyMontage()...")
% easyMontage(imgs,2);

% 
% %% Run the back projection kernel
% disp("ResetVolume()...")
% obj.ResetVolume()
% 
% disp("Back_Project()...")
% obj.Back_Project()
% 
% disp("Get_Volume()...") % Get the volumes from all the GPUs added together
% volCAS = obj.GetVolume();
% 
% %% Get the density of inserted planes by backprojecting CASimages of values equal to one
% disp("Get Plane Density()...")
% interpImgs=ones([interpBox.size interpBox.size size(coordAxes,1)/9],'single');
% obj.ResetVolume();
% obj.SetImages(interpImgs)
% obj.Back_Project()
% volWt = obj.GetVolume();
% 
% %% Normalize the back projection result with the plane density
% % Divide the previous volume with the plane density volume
% volCAS=volCAS./(volWt+1e-6);
%                 
% % Reconstruct the volume from CASVol
% disp("volFromCAS()...")
% volReconstructed=volFromCAS(volCAS,CASBox,interpBox,origBox,kernelHWidth);
% 
% disp("easyMontage()...")
% easyMontage(volReconstructed,3);


%% Free the memory
obj.CUDA_Free('all')
clear obj
clear all

toc