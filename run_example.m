% How to run this script from the terminal (needed in order to use the NVIDIA profiling tools)
% "/usr/local/MATLAB/R2018a/bin/glnxa64/MATLAB" -nodisplay -nosplash -nodesktop -r "run('/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj/CUDA_mex_files/run_example.m');exit;"

% -nodisplay -nosplash -nodesktop -r "run('/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj/testGpuObj.m');exit;"


% "C:\<a long path here>\matlab.exe" -nodisplay -nosplash -nodesktop -r "run('C:\<a long path here>\mfile.m');
% sudo '/usr/local/cuda/NsightSystems-2019.3/Host-x86_64/nsight-sys' 
% /usr/local/MATLAB/R2018a/bin/glnxa64/MATLAB

% To launch Nsight Systems
% sudo /usr/local/cuda/NsightSystems-2019.3/Host-x86_64/nsight-sys

% To launch nvidia visual profiler
% sudo /usr/local/cuda/bin/nvvp

% To launch nvidia nsight compute
% sudo /usr/local/cuda/NsightCompute-2019.3/nv-nsight-cu
% "/usr/local/MATLAB/R2018a/bin/glnxa64/MATLAB" -nodisplay -nosplash -nodesktop -r run('/home/brent/Documents/MATLAB/CUDA/Tutorials/July 15 2019/Example 1/Brent_Script.m')

% To watch the GPU usage
% watch nvidia-smi


clc
close all
clear obj 

addpath('./CUDA_mex_files')
addpath('./utils')

recompile = 1;
if (recompile == true)
    cd('CUDA_mex_files')

    fprintf('Compiling MultiGPUGridder_Matlab_Class mex file \n');

    % Compile the forward projection CUDA kernel first
    status = system("nvcc -c -shared -Xcompiler -fPIC -lcudart -lcuda gpuForwardProjectKernel.cu -I'/usr/local/MATLAB/R2018a/extern/include/' -I'/usr/local/cuda/tarets/x86_64-linux/include/' ", '-echo')

    if status ~= 0
        error("Failed to compile");
    end

    % Compile the back projection CUDA kernel first
    status = system("nvcc -c -shared -Xcompiler -fPIC -lcudart -lcuda gpuBackProjectKernel.cu -I'/usr/local/MATLAB/R2018a/extern/include/' -I'/usr/local/cuda/tarets/x86_64-linux/include/' ", '-echo')

    if status ~= 0
        error("Failed to compile");
    end
    
    % Compile the mex files second
    clc; mex GCC='/usr/bin/gcc-6' -I'/usr/local/cuda/targets/x86_64-linux/include/' -L"/usr/local/cuda/lib64/" -lcudart -lcuda  -lnvToolsExt -DMEX mexFunctionWrapper.cpp MultiGPUGridder.cpp MemoryManager.cpp gpuForwardProjectKernel.o gpuBackProjectKernel.o
    
    cd('..')
end


reset(gpuDevice());

%% Create a volume 
% Initialize parameters
tic

nBatches = 2;
nGPUs = 4;
nStreams = 64;
volSize = 128; %256;%256%128;%64;
n1_axes = 50;
n2_axes = 10;

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
nCoordAxes = length(coordAxes)/9

%% MATLAB pre-processing to covert vol to CASVol

% interpBoc and fftinfo are needed for plotting the results
disp("MATLAB Vol_Preprocessing()...")
[CASVol, CASBox, origBox, interpBox, fftinfo] = Vol_Preprocessing(vol, interpFactor);

disp(["Volume size: " + num2str(volSize)])
disp(["Number of coordinate axes: " + num2str(nCoordAxes)])
 
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

disp("Forward_Project()...")
obj.Forward_Project()

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