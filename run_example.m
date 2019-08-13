clc
close all
clear obj 

addpath('./src')
addpath('./utils')
addpath('./bin') % The compiled mex file is stored in the bin folder


reset(gpuDevice());

%% Create a volume 
% Initialize parameters
tic

nBatches = 1;
nGPUs = 4;
nStreams = 8;
volSize = 128;
n1_axes = 100;
n2_axes = 10;

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
obj.SetMaskRadius(single(50)); 

disp("SetVolume()...")
tic
% CASVol = permute(CASVol, [2 1 3]);
% obj.SetVolume(single(CASVol))
obj.SetVolume(single(vol))
toc
% 
% outputVol = obj.GetVolume();
% outputVol = outputVol / 4;
% % outputVol = outputVol / (volSize*volSize);
% outputVol(1:10)
% 
% CASVol(1:10)
% 
% close all
% slice = 2;
% subplot(1,3,1)
% imagesc(CASVol(:,:,slice))
% title("Matlab CASVol")
% subplot(1,3,2)
% imagesc(outputVol(:,:,slice))
% title("CUDA FFT CASVol")
% subplot(1,3,3)
% imagesc(CASVol(:,:,slice) - outputVol(:,:,slice))
% title("Subtraction")
% colormap gray
% colorbar

% obj.CUDA_Free('all')
% clear obj
% clear all

% imagesc(volCAS(:,:,1))

%  410.0000 -239.5747 -566.6325 -462.6787 -118.3519  239.0186  133.5732   63.5933  599.1982  112.2435
%  CASVol(1:10)

%  410.0000 -324.7330 -431.9106 -422.8362 -526.1659 -134.4101  420.3719  533.6011  259.4940 -272.3445
% CASVol(1:10)
% CASVol = permute(CASVol, [2 1 3]);
% outputVol(1:10)
% CASVol(1:10)

% interpCAS=fftshift(fftn(fftshift(vol)));
% interpCAS(1:10)


% CAS_Vol[0]:  410 -324.733 -431.91 -422.836 -526.167 -134.411 420.372 533.601 259.494 -272.345
% interpCAS=ToCAS(fftshift(fftn(fftshift(vol))));
% interpCAS = permute(interpCAS, [2 1 3]);
% interpCAS(1:10)

% obj.CUDA_Free('all')
% clear obj
% clear all
% abc
% FFTSHIFT h_complex_array: 74 + 0
% 76 + 0
% 75 + 0
% 71 + 0
% 69 + 0
% 70 + 0
% 75 + 0
% 60 + 0
% 59 + 0
% 74 + 0


% x = fftshift(vol);
% x(end-10:end)'



% FFT FFTSHIFT h_complex_array: 4.58015e+07 + 0
% -1.5745e+07 + 7.49552e+06
% -4.35522e+06 + 3.68014e+06
% -1.41856e+06 + 981196
% -137044 + -520446
% -414718 + 141614
% -246986 + -469657
% -11574.9 + -270589
% -333057 + -214348
% -124647 + -376950


% x = fftn(fftshift(vol));
% % x = permute(x, [2 1 3]);
% x(1:10)

% fft_vol=fftshift(fftn(fftshift(vol)));
% 
% 
% interpCAS=ToCAS(fft_vol);
% interpCAS = permute(interpCAS, [2 1 3]);
% interpCAS(1:10)
% 
% 
% vol(1:10)
% x=fftshift(vol);
% x(1:10)
% 
% fft_vol=fftn(fftshift(vol));
% fft_vol(1:10)

% 
% fft_vol=fftn(vol);
% interpCAS=ToCAS(fft_vol);
% interpCAS = permute(interpCAS, [2 1 3]);
% interpCAS(1:10)
% fft_vol(1:10)
% CASvol=ToCAS(fft_vol);
% CASvol(1:10)

% obj.CUDA_Free('all')
% clear obj
% clear all
% 
% abc

disp("SetAxes()...")
obj.SetAxes(coordAxes)

disp("SetImgSize()...")
obj.SetImgSize(int32([size(vol,1) * interpFactor, size(vol,1) * interpFactor,nCoordAxes]))

disp("Displaying allocated memory()...")
obj.CUDA_disp_mem('all')
obj.disp_mem('all')

%% Run the forward projection kernel
% clc
disp("Forward_Project()...")
obj.Forward_Project()

test_imgs = obj.mem_Return('CASImgs_CPU_Pinned');
size(test_imgs)
max(test_imgs(:))

close all
% imagesc(test_imgs(:,:,1))
% colormap gray

% disp("GetImgs()...")
InterpCASImgs = obj.GetImgs();
size(InterpCASImgs)
InterpCASImgs = InterpCASImgs(:,:,1:10);
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