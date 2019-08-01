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

% Maximum int32 number 2147483647

2147483647 / (500 * 500)

volSize  = 512;%[64, 128, 256, 512];%[64]%, 128, 256, 512];
n1_axes  = 500%ones(1,5);%12000%[10, 20, 40, 60, 80, 100, 500, 1000];
n2_axes  = 10;

nStreams = 64%[4, 8, 16, 32, 64, 128, 256, 512];

timing_measurements = [];
timing_measurements.volSize = [];
timing_measurements.nCoordAxes = [];
timing_measurements.nStreams = [];
timing_measurements.fuzzymask = [];
timing_measurements.Vol_Preprocessing = [];
timing_measurements.create_uniform_axes = [];
timing_measurements.mem_allocation = [];
timing_measurements.Forward_Project = [];

iter = 1;

for i = 1:length(volSize)
    for j = 1:length(n1_axes)
        for k = 1:length(n2_axes)
            for z = 1:length(nStreams)
                
                % Currently limited by the max int32 size when allocating the pinned CPU memory
                % Leave 5% less of the maximum value
%                 c = round(linspace(10, 2147483647 / (n2_axes(k) * volSize(i)^2 * 4) * 0.95, 5) )


                timing_measurements(iter) = RunExample(volSize(i), n1_axes(j), n2_axes(k), nStreams(z));

                iter = iter + 1;
            end
        end
    end    
end

%% Plot the timing measurements
close all


nProjections = n1_axes * n2_axes;
nProjections = nProjections + 1;
length(unique(nProjections))

numRows = 1; % For plotting

figure('Color', [1 1 1])
hold on

for i = 1:length(timing_measurements)
    
    if timing_measurements(i).volSize == 64
        plot_color = 'b';
    elseif timing_measurements(i).volSize == 128
        plot_color = 'r';
    elseif timing_measurements(i).volSize == 256
        plot_color = 'k';
    elseif timing_measurements(i).volSize == 512
        plot_color = 'g';
    end    
        
    if timing_measurements(i).nStreams == 4
        plot_style = '*';
    elseif timing_measurements(i).nStreams == 8
        plot_style = '+';
    elseif timing_measurements(i).nStreams == 16
        plot_style = 'o';
    elseif timing_measurements(i).nStreams == 32
        plot_style = 'x';
    elseif timing_measurements(i).nStreams == 64
        plot_style = 's';
    end
    
    % Select the subplot based on the number of projections in the current test
    [x ndx] = find(volSize == timing_measurements(i).volSize)
    
%     r = length(nProjections) / ndx 
    
    subplot(numRows, ceil(length(volSize) / numRows), ndx)
    hold on
    
    try
    x = timing_measurements(i).nCoordAxes;
    y = timing_measurements(i).Forward_Project * 1000;
    plot(x, y, [plot_color plot_style])
    
    str_1 = num2str(timing_measurements(i).nCoordAxes);
    str_2 = num2str(timing_measurements(i).nStreams);
    str_3 = string(str_1) + ", " + string(str_2);
    text(x + 0.5,y, str_2)   
    catch
        disp("Failed to plot for " + num2str(i))
    end
    
    

    xlabel('Number of Projections')
    ylabel('Time (msec)')
    title(num2str(volSize(ndx)) + " Volume Size")
    grid on

end

