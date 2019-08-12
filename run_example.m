clc
close all
clear obj 

addpath('./src')
addpath('./utils')
addpath('./bin') % The compiled mex file is stored in the bin folder

recompile = 0;
if (recompile == true)
    cd('src')

    fprintf('Compiling MultiGPUGridder_Matlab_Class mex file \n');

    % Compile the forward projection CUDA kernel first
    status = system("nvcc -c -shared -Xcompiler -fPIC -lcudart -lcuda gpuForwardProjectKernel.cu -I'/usr/local/cuda/tarets/x86_64-linux/include/' ", '-echo')


    if status ~= 0
    error("Failed to compile");
    end

    % Compile the back projection CUDA kernel first
    status = system("nvcc -c -shared -Xcompiler -fPIC -lcudart -lcuda gpuBackProjectKernel.cu -I'/usr/local/cuda/tarets/x86_64-linux/include/' ", '-echo')

    if status ~= 0
    error("Failed to compile");
    end

    % Compile the mex files second
    % -lnvToolsExt
    clc; mex GCC='/usr/bin/gcc-6' -I'/usr/local/cuda/targets/x86_64-linux/include/' -L"/usr/local/cuda/lib64/" -lcudart -lcuda  -DMEX mexFunctionWrapper.cpp MultiGPUGridder.cpp MemoryManager.cpp gpuForwardProjectKernel.o gpuBackProjectKernel.o

    cd('..')
end


reset(gpuDevice());

%% Create a volume 
% Initialize parameters
tic

nBatches = 2;
nGPUs = 4;
nStreams = 64;
volSize = 128;
n1_axes = 100;
n2_axes = 100;

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
[CASVol, CASBox, origBox, interpBox, fftinfo] = Vol_Preprocessing(vol, interpFactor);

disp("Volume size: " + num2str(volSize))
disp("Number of coordinate axes: " + num2str(nCoordAxes))
 
%% Initialize the multi GPU gridder
obj = MultiGPUGridder_Matlab_Class();
obj.SetNumberBatches(nBatches);
obj.SetNumberGPUs(nGPUs);
obj.SetNumberStreams(nStreams);
obj.SetMaskRadius(single((size(vol,1) * interpFactor)/2 - 1)); 

disp("SetVolume()...")
obj.SetVolume(single(CASVol))

disp("SetAxes()...")
obj.SetAxes(coordAxes)

disp("SetImgSize()...")
obj.SetImgSize(int32([size(vol,1) * interpFactor, size(vol,1) * interpFactor,nCoordAxes]))

disp("Displaying allocated memory()...")
obj.CUDA_disp_mem('all')
obj.disp_mem('all')

%% Run the forward projection kernel
clc
disp("Forward_Project()...")
obj.Forward_Project()

obj.CUDA_Free('all')
clear obj
clear all


x = transpose(reshape(0:24, [5 5]));
fft2(x)

abc


disp("GetImgs()...")
InterpCASImgs = obj.GetImgs();

disp("imgsFromCASImgs()...")
imgs=imgsFromCASImgs(InterpCASImgs(:,:,1:10), interpBox, fftinfo); 

disp("easyMontage()...")
easyMontage(imgs,2);

%% Run the back projection kernel
disp("ResetVolume()...")
obj.ResetVolume()

disp("Back_Project()...")
obj.Back_Project()

disp("Get_Volume()...") % Get the volumes from all the GPUs added together
volCAS = obj.GetVolume();

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

disp("easyMontage()...")
easyMontage(volReconstructed,3);


%% Free the memory
obj.CUDA_Free('all')
clear obj
clear all

toc