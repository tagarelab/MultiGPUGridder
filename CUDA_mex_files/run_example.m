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

addpath('/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj')
addpath('/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj/utils')


cd("/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj/CUDA_mex_files")

recompile = 1;
if (recompile == true)
    % cd('mex_files')

    fprintf('Compiling CUDA_Gridder mex file \n');

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
    clc; mex GCC='/usr/bin/gcc-6' -I'/usr/local/cuda/targets/x86_64-linux/include/' -L"/usr/local/cuda/lib64/" -lcudart -lcuda  -lnvToolsExt -DMEX mexFunctionWrapper.cpp CUDA_Gridder.cpp CPU_CUDA_Memory.cpp gpuForwardProjectKernel.o gpuBackProjectKernel.o

end


reset(gpuDevice());

%% Create a volume 
% Initialize parameters
tic

volSize = 128;%256;%256%128;%64;
n1_axes = 100;
n2_axes = 50;
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
% Change the sphere a bit so the projections are not all the same
% vol(:,:,1:volSize/2) = 2 * vol(:,:,1:volSize/2);


% Use the example matlab MRI image to take projections of
load mri;
img = squeeze(D);
img = imresize3(img,[volSize, volSize, volSize]);
vol = single(img);
easyMontage(vol,1);
%% Define the projection directions
coordAxes=single([1 0 0 0 1 0 0 0 1]');
coordAxes=[coordAxes create_uniform_axes(n1_axes,n2_axes,0,10)];
coordAxes = coordAxes(:);
nCoordAxes = length(coordAxes)/9

%% MATLAB pre-processing to covert vol to CASVol

% interpBoc and fftinfo are needed for plotting the results
disp("MATLAB Vol_Preprocessing()...")
[CASVol, CASBox, origBox, interpBox, fftinfo] = Vol_Preprocessing(vol, interpFactor);

size(CASVol)

%% Display some information to the user before running the forward projection kernel

disp(["Volume size: " + num2str(volSize)])
disp(["Number of coordinate axes: " + num2str(nCoordAxes)])
 
%% Run the forward projection kernel

obj = CUDA_Gridder_Matlab_Class();
obj.SetNumberBatches(1);
obj.SetNumberGPUs(4);
obj.SetNumberStreams(32);
obj.SetMaskRadius(single((size(vol,1) * interpFactor)/2 - 1)); 

disp("SetVolume()...")
obj.SetVolume(single(CASVol))

disp("SetAxes()...")
obj.SetAxes(coordAxes)

disp("SetImgSize()...")
obj.SetImgSize(int32([size(vol,1) * interpFactor, size(vol,1) * interpFactor,nCoordAxes]))

disp("Projection_Initilize()...")
obj.Projection_Initilize()

tic
disp("Forward_Project()...")
obj.Forward_Project()
toc

% Return the resulting projection images
disp("mem_Return()...")
InterpCASImgs  = obj.mem_Return('CASImgs_CPU_Pinned');


disp("imgsFromCASImgs()...")
imgs=imgsFromCASImgs(InterpCASImgs(:,:,:), interpBox, fftinfo); 

easyMontage(imgs,2);


%% Run the back projection kernel
disp("ResetVolume()...")
obj.ResetVolume()

tic
disp("Back_Project()...")
obj.Back_Project()
toc

% Get the volumes from all the GPUs and add them together
volCAS  = zeros(size(CASVol));
for i = 0:3
    volCAS  = volCAS + obj.CUDA_Return(char("gpuVol_" + num2str(i)));
end

% Get the density of inserted planes by backprojecting CASimages of values equal to one
nAxes = size(coordAxes,1)/9;
interpImgs=ones([interpBox.size interpBox.size nAxes],'single');
obj.ResetVolume();
obj.SetImages(interpImgs)

tic
obj.Back_Project()
toc

% Get the resulting volume from all the GPUs and add them together
volWt  = zeros(size(CASVol));
for i = 0:3
    volWt  = volWt + obj.CUDA_Return(char("gpuVol_" + num2str(i)));
end

% Divide the previous volume with the plane density volume
volCAS=volCAS./(volWt+1e-6);

% Reconstruct the volume from CASVol
volReconstructed=volFromCAS(volCAS,CASBox,interpBox,origBox,kernelHWidth);


% figure
% imagesc(vol(:,:,floor(size(vol,3)/2)))
% colormap gray
% axis square
% 
easyMontage(volReconstructed,3);

easyMontage(vol - volReconstructed,4);
colorbar

obj.CUDA_Free('all')
clear obj












%%
% 
% %%
% abc
% 
% % obj.CUDA_disp_mem('all')
% % obj.disp_mem('all')
% 
% % Return the resulting projection images
% disp("mem_Return()...")
% InterpCASImgs  = obj.mem_Return('CASImgs_CPU_Pinned');
% 
% 
% disp("imgsFromCASImgs()...")
% tic
% imgs=imgsFromCASImgs(InterpCASImgs(:,:,1), interpBox, fftinfo); 
% toc
% 
% 
% obj.CUDA_Free('all')
% clear obj
% 
% clearvars -except imgs
% 
% clc
%  
% gpuGridder = load("gpuGridder_vol128.mat");
% % gpuGridder = load("gpuGridder_InterpCASImgs.mat");
% % gpuGridder = load("gpuGridder_gpuVol.mat");
% 
% 
% img_slice = 1;
% 
% close all
% figure('Color', [1 1 1])
% h(1) = subplot(1,3,1);
% imagesc(gpuGridder.img(:,:,img_slice));
% title("gpuGridder")
% axis square
% 
% h(2) = subplot(1,3,2)
% imagesc(imgs(:,:,img_slice));
% title("C++ Gridder")
% axis square
% 
% h(3) = subplot(1,3,3);
% imagesc(gpuGridder.img(:,:,img_slice) - imgs(:,:,img_slice))
% title("Subtraction")
% colorbar
% colormap jet
% axis square
% linkaxes(h, 'xy')
% 
% 
% sum(sum(gpuGridder.img(:,:,img_slice) - imgs(:,:,img_slice)))
% 
% display_imgs = 0;
% 
% if display_imgs  == 1
%     % How many images to plot?
%     numImgsPlot = 10;
% 
%     % Make sure we have that many images first
%     numImgsPlot = min(numImgsPlot, size(InterpCASImgs,3));
% 
%     imgs=imgsFromCASImgs(InterpCASImgs(:,:,1:numImgsPlot), interpBox, fftinfo);
%     easyMontage(imgs,1);
%     % 
%     % % 
%     % imgs=imgsFromCASImgs(InterpCASImgs(:,:,end-numImgsPlot:end), interpBox, fftinfo);
%     % easyMontage(imgs,2);
%     colormap jet
% end
% 
% disp('Done!');
% 
% 
% clear all


% 
% % Compare with the gpuGridder matlab version
% gpuGridder = load("gpuGriddervolR.mat")
% 
% 
% img_slice = 32;
% 
% close all
% figure('Color', [1 1 1])
% h(1) = subplot(1,3,1);
% imagesc(gpuGridder.volR(:,:,img_slice));
% title("gpuGridder - Back Projection")
% axis square
% 
% h(2) = subplot(1,3,2)
% imagesc(vol(:,:,img_slice));
% title("Multi GPU Gridder - Back Projection")
% axis square
% 
% h(3) = subplot(1,3,3);
% imagesc(gpuGridder.volR(:,:,img_slice) - vol(:,:,img_slice))
% title("Subtraction")
% colorbar
% colormap jet
% axis square
% linkaxes(h, 'xy')











%%

% obj.CUDA_disp_mem('all')
% obj.disp_mem('all')

% Clear all the variables except for InterpCASImgs
% clearvars -except InterpCASImgs

% InterpCASImgs = obj.CUDA_Return('gpuCASImgs_0');






% obj.CUDA_Free('all')

% obj.CUDA_disp_mem('all')






%%
% reset(gpuDevice(1));
% 
% 
% input_data = load('Forward_Project_Input.mat')
% input_data = input_data.x
% 
% obj = CUDA_Gridder_Matlab_Class();
% 
% % Allocate GPU CUDA memory
% obj.CUDA_alloc('gpuVol', 'float', int32(size(input_data.gpuVol)), 0);
% obj.CUDA_alloc('gpuCASImgs', 'float', int32(size(input_data.gpuCASImgs)), 0);
% obj.CUDA_alloc('gpuCoordAxes', 'float', int32([2034, 1, 1]), 0);
% obj.CUDA_alloc('gpuKerTbl', 'float', int32([501, 1, 1]), 0);
% 
% % Allocate CPU memory
% obj.mem_alloc('CASBox_size', 'int', int32([1 1 1]));
% obj.mem_alloc('imgSize', 'int', int32([1 1 1]));
% obj.mem_alloc('nAxes', 'int', int32([1 1 1]));
% obj.mem_alloc('rMax', 'float', int32([1 1 1]));
% obj.mem_alloc('kerTblSize', 'int', int32([1 1 1]));
% obj.mem_alloc('kerHWidth', 'float', int32([1 1 1]));
% 
% % Copy Matab array to CUDA array 
% obj.CUDA_Copy('gpuVol', input_data.gpuVol);
% obj.CUDA_Copy('gpuCASImgs', input_data.gpuCASImgs);
% obj.CUDA_Copy('gpuCoordAxes', input_data.gpuCoordAxes);
% obj.CUDA_Copy('gpuKerTbl', input_data.gpuKerTbl);
% 
% % Copy Matab array to CPU array 
% obj.mem_Copy('CASBox_size', int32(input_data.CASBox_size));
% obj.mem_Copy('imgSize', int32(input_data.imgSize));
% obj.mem_Copy('nAxes', int32(input_data.nAxes));
% obj.mem_Copy('rMax', input_data.rMax);
% obj.mem_Copy('kerTblSize', int32(input_data.kerTblSize));
% obj.mem_Copy('kerHWidth', input_data.kerHWidth);
% 
% 
% 
% obj.CUDA_disp_mem('all')
% obj.disp_mem('all');
% 
% obj.Forward_Project( ...
%     'gpuVol', 'gpuCASImgs', 'gpuCoordAxes', 'gpuKerTbl', ...
%     'CASBox_size', 'imgSize', 'nAxes', 'rMax', 'kerTblSize', 'kerHWidth')
% 
% 
% InterpCASImgs = obj.CUDA_Return('gpuCASImgs');
% 
% max(InterpCASImgs(:))
% 
% % imgs=imgsFromCASImgs(InterpCASImgs, input_data.interpBox, input_data.fftinfo);
% % easyMontage(imgs,1);
% 
% obj.mem_Free('CASBox_size')
% obj.mem_Free('imgSize')
% obj.mem_Free('nAxes')
% obj.mem_Free('rMax')
% obj.mem_Free('kerTblSize')
% obj.mem_Free('kerHWidth')
% 
% 
% 
% obj.CUDA_Free('gpuVol');
% obj.CUDA_Free('gpuCASImgs');
% obj.CUDA_Free('gpuCoordAxes');
% obj.CUDA_Free('gpuKerTbl');
% 
% 
% 
% 





% clear obj % Clear calls the delete method

%%


% 
% obj = CUDA_Gridder_Matlab_Class();
% 
% mat_size = int32([ 5, 5, 1]);
% obj.mem_alloc('arr_1', 'float', mat_size);
% obj.mem_alloc('arr_2', 'int', mat_size);
% obj.disp_mem('all')
% 
% 
% 
% obj.Forward_Project('arr_2')
% 
% obj.mem_Free('arr_1')
% obj.mem_Free('arr_2')
% 
% clear obj % Clear calls the delete method
% %%
% reset(gpuDevice(1));
% 
% 
% mat_size = int32([ 5, 5, 1]);
% 
% 
% num_ints = prod(mat_size);%32000000*5;
% num_bytes = num_ints * 4;
% num_GB = num_bytes / (10^9)
% 
% if (2147483647 - num_ints ) < 0
%     error('Too many elements')
% end
% 
% 
% data = ones(mat_size);
% 
% for i = 1:length(data(:))
%    data(i) = i; 
% end
% 
% data = single(data);
% 
% 
% fprintf('Using the example interface\n');
% obj = CUDA_Gridder_Matlab_Class();
% 
% % tic 
% obj.mem_alloc('arr_1', 'float', mat_size);
% obj.mem_alloc('arr_2', 'int', mat_size);
% obj.disp_mem('all')

% toc
% 
% tic
% pin_mem(obj, 'arr_1');
% pin_mem(obj, 'arr_2');
% toc

% obj.CUDA_alloc
% 
% obj.CUDA_alloc('gpuVol_1', 'float', mat_size, 0);
% obj.CUDA_alloc('gpuVol_2', 'int', mat_size, 1);
% obj.CUDA_alloc('gpuVol_3', 'float', mat_size, 2);
% obj.CUDA_alloc('gpuVol_4', 'float', mat_size, 3);
% 
% obj.CUDA_disp_mem('all')
% obj.CUDA_Copy('gpuVol_1', data);
% obj.CUDA_Copy('gpuVol_2', int32(data));
% 
% disp('Returning CUDA arrays...')
% tic
% test = obj.CUDA_Return('gpuVol_1')
% toc
% tic
% test = obj.CUDA_Return('gpuVol_2')
% toc



% % Call the kernel
% obj.gpuCASImgs=feval(obj.cudaFPKer,...
%                 obj.gpuVol, obj.CASBox.size,...
%                 obj.gpuCASImgs,obj.imgSize,...
%                 obj.gpuCoordAxes, nAxes,single(obj.rMax),...
%                 obj.gpuKerTbl, int32(obj.kerTblSize), single(obj.kerHWidth));



% Forward_Project(obj, 'gpuVol', 'CASBoxsize',


% 
% obj.CUDA_Free('gpuVol_1');
% obj.CUDA_Free('gpuVol_2');
% obj.CUDA_Free('gpuVol_3');
% obj.CUDA_Free('gpuVol_4');

%     
% tic
% mem_Copy(obj, 'arr_1', data);
% mem_Copy(obj, 'arr_2', int32(data));
% toc
% 
% disp_mem(obj, 'arr_1');
% disp_mem(obj, 'arr_2');
% 
% disp_mem(obj, 'all');
% 
% tic
% x = mem_Return(obj, 'arr_1');
% y = mem_Return(obj, 'arr_2');
% toc
% 
% mem_Free(obj, 'arr_1')
% mem_Free(obj, 'arr_2')

% 
% sum(x(:) - data(:))

% Make very large matlab array to check for page fault
% x = zeros(20*num_ints,1);


% disp_mem(obj, 'arr_1');

% 
% clear obj % Clear calls the delete method
% 
% clear;
% 
% mat_size = int32([ 5, 5, 1]);
% obj.mem_alloc('arr_1', 'float', mat_size);
% obj.mem_alloc('arr_2', 'int', mat_size);
% obj.disp_mem('all')
% 
% 
% 
% obj.Forward_Project('arr_2')
% 
% obj.mem_Free('arr_1')
% obj.mem_Free('arr_2')
% 
% clear obj % Clear calls the delete method
% %%
% reset(gpuDevice(1));
% 
% 
% mat_size = int32([ 5, 5, 1]);
% 
% 
% num_ints = prod(mat_size);%32000000*5;
% num_bytes = num_ints * 4;
% num_GB = num_bytes / (10^9)
% 
% if (2147483647 - num_ints ) < 0
%     error('Too many elements')
% end
% 
% 
% data = ones(mat_size);
% 
% for i = 1:length(data(:))
%    data(i) = i; 
% end
% 
% data = single(data);
% 
% 
% fprintf('Using the example interface\n');
% obj = CUDA_Gridder_Matlab_Class();
% 
% % tic 
% obj.mem_alloc('arr_1', 'float', mat_size);
% obj.mem_alloc('arr_2', 'int', mat_size);
% obj.disp_mem('all')

% toc
% 
% tic
% pin_mem(obj, 'arr_1');
% pin_mem(obj, 'arr_2');
% toc

% obj.CUDA_alloc
% 
% obj.CUDA_alloc('gpuVol_1', 'float', mat_size, 0);
% obj.CUDA_alloc('gpuVol_2', 'int', mat_size, 1);
% obj.CUDA_alloc('gpuVol_3', 'float', mat_size, 2);
% obj.CUDA_alloc('gpuVol_4', 'float', mat_size, 3);
% 
% obj.CUDA_disp_mem('all')
% obj.CUDA_Copy('gpuVol_1', data);
% obj.CUDA_Copy('gpuVol_2', int32(data));
% 
% disp('Returning CUDA arrays...')
% tic
% test = obj.CUDA_Return('gpuVol_1')
% toc
% tic
% test = obj.CUDA_Return('gpuVol_2')
% toc



% % Call the kernel
% obj.gpuCASImgs=feval(obj.cudaFPKer,...
%                 obj.gpuVol, obj.CASBox.size,...
%                 obj.gpuCASImgs,obj.imgSize,...
%                 obj.gpuCoordAxes, nAxes,single(obj.rMax),...
%                 obj.gpuKerTbl, int32(obj.kerTblSize), single(obj.kerHWidth));



% Forward_Project(obj, 'gpuVol', 'CASBoxsize',


% 
% obj.CUDA_Free('gpuVol_1');
% obj.CUDA_Free('gpuVol_2');
% obj.CUDA_Free('gpuVol_3');
% obj.CUDA_Free('gpuVol_4');

%     
% tic
% mem_Copy(obj, 'arr_1', data);
% mem_Copy(obj, 'arr_2', int32(data));
% toc
% 
% disp_mem(obj, 'arr_1');
% disp_mem(obj, 'arr_2');
% 
% disp_mem(obj, 'all');
% 
% tic
% x = mem_Return(obj, 'arr_1');
% y = mem_Return(obj, 'arr_2');
% toc
% 
% mem_Free(obj, 'arr_1')
% mem_Free(obj, 'arr_2')

% 
% sum(x(:) - data(:))

% Make very large matlab array to check for page fault
% x = zeros(20*num_ints,1);


% disp_mem(obj, 'arr_1');

% 
% clear obj % Clear calls the delete method
% 
% clear;