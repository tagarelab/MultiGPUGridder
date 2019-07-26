clc
clear


clear obj 


fprintf('Compiling CUDA_Gridder mex file \n');

% Compile the CUDA kernel first
% status = system("nvcc -c -shared -Xcompiler -fPIC -lcudart -lcuda gpuForwardProjectKernel.cu -I'/usr/local/MATLAB/R2018a/extern/include/' -I'/usr/local/cuda/tarets/x86_64-linux/include/' ", '-echo')

% if status ~= 0
%     error("Failed to compile");
% end

% Compile the mex files second
clc; mex GCC='/usr/bin/gcc-6' -I'/usr/local/cuda/targets/x86_64-linux/include/' -L"/usr/local/cuda/lib64/" -lcudart -lcuda  -lnvToolsExt -DMEX CUDA_Gridder.cpp 

% gpuForwardProjectKernel.o

%
%
%%

reset(gpuDevice(1));


mat_size = int32([ 5, 5, 1]);


num_ints = prod(mat_size);%32000000*5;
num_bytes = num_ints * 4;
num_GB = num_bytes / (10^9)

if (2147483647 - num_ints ) < 0
    error('Too many elements')
end


data = ones(mat_size);

for i = 1:length(data(:))
   data(i) = i; 
end

data = single(data);


fprintf('Using the example interface\n');
obj = MATLAB_CUDA_Memory();

% tic 
obj.mem_alloc('arr_1', 'float', mat_size);
obj.mem_alloc('arr_2', 'int', mat_size);
obj.disp_mem('all')

% toc
% 
% tic
% pin_mem(obj, 'arr_1');
% pin_mem(obj, 'arr_2');
% toc

% obj.CUDA_alloc

obj.CUDA_alloc('gpuVol_1', 'float', mat_size, 0);
obj.CUDA_alloc('gpuVol_2', 'int', mat_size, 1);
obj.CUDA_alloc('gpuVol_3', 'float', mat_size, 2);
obj.CUDA_alloc('gpuVol_4', 'float', mat_size, 3);

obj.CUDA_disp_mem('all')
obj.CUDA_Copy('gpuVol_1', data);
obj.CUDA_Copy('gpuVol_2', int32(data));

disp('Returning CUDA arrays...')
tic
test = obj.CUDA_Return('gpuVol_1')
toc
tic
test = obj.CUDA_Return('gpuVol_2')
toc



% % Call the kernel
% obj.gpuCASImgs=feval(obj.cudaFPKer,...
%                 obj.gpuVol, obj.CASBox.size,...
%                 obj.gpuCASImgs,obj.imgSize,...
%                 obj.gpuCoordAxes, nAxes,single(obj.rMax),...
%                 obj.gpuKerTbl, int32(obj.kerTblSize), single(obj.kerHWidth));



% Forward_Project(obj, 'gpuVol', 'CASBoxsize',



obj.CUDA_Free('gpuVol_1');
obj.CUDA_Free('gpuVol_2');
obj.CUDA_Free('gpuVol_3');
obj.CUDA_Free('gpuVol_4');

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


clear obj % Clear calls the delete method

clear;