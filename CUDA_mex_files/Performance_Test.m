clc
clear
close all

addpath('/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj')
addpath('/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj/utils')



recompile = 1;
if (recompile == true)
    % cd('mex_files')

    fprintf('Compiling CUDA_Gridder mex file \n');

    % Compile the CUDA kernel first
    status = system("nvcc -c -shared -Xcompiler -fPIC -lcudart -lcuda gpuForwardProjectKernel.cu -I'/usr/local/MATLAB/R2018a/extern/include/' -I'/usr/local/cuda/tarets/x86_64-linux/include/' ", '-echo')

    if status ~= 0
        error("Failed to compile");
    end

    % Compile the mex files second
    clc; mex GCC='/usr/bin/gcc-6' -I'/usr/local/cuda/targets/x86_64-linux/include/' -L"/usr/local/cuda/lib64/" -lcudart -lcuda  -lnvToolsExt -DMEX mexFunctionWrapper.cpp CUDA_Gridder.cpp CPU_CUDA_Memory.cpp gpuForwardProjectKernel.o

end
 
% for i = 1:4
%     reset(gpuDevice(i))
% end

%%

volSize = [64]%, 128, 256, 512];
n1_axes = 15;
n2_axes = 15;


timing_measurements = [];

for j = 1:length(volSize)

    timing_measurements(j).volSize = volSize(j);
    timing_measurements(j).nCoordAxes = n1_axes * n2_axes + 1;
    timing_measurements(j).times = RunExample(volSize(j), n1_axes, n2_axes);

end

%% Plot the timing measurements

kernel_times = [];

for i = 1:length(timing_measurements)
    kernel_times = [kernel_times timing_measurements(i).times(end-2)]
end

figure('Color', [1 1 1])
plot(kernel_times, 'b*--')
