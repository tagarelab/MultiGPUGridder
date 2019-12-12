clc
close all
clear all

% mex /home/brent/Documents/MATLAB/simple_gpu_gridder_Obj/src/Gridders/CPU/mex_forward_project.c

addpath(genpath('/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj/src/Matlab/'))
addpath(genpath('/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj/'))

% Parameters for creating the volume and coordinate axes
VolumeSize = 64;
interpFactor = 2;

n1_axes = 10;%[1 10 50 100 280];
n2_axes = 100;
nStreams = [2 4 64]

disp("Number of axes: " + num2str([n1_axes*n2_axes]))

RunFFTOnGPU = true;
times = [];
for i = 1:length(nStreams)        
        new_time = MultiGridderTimingTest(VolumeSize, n1_axes, n2_axes, interpFactor, RunFFTOnGPU, nStreams(i));
        times = [times; new_time] 
end


%%
figure('Color', [1 1 1])
% Forward Projection times
plot(nStreams, times(:,1), 'r*--')
hold on

plot(nStreams, times(:,2), 'b*--')
legend("Forward Projection", "Back Projection")

return

%%
figure('Color', [1 1 1])
% Forward Projection times
plot(nStreams, times(:,1), 'r*--')
hold on

% Back Projection times
% plot(nStreams, times(:,2), 'b*--')
legend("Forward Projection", "Back Projection")
xlabel("Number of CUDA Streams")
ylabel("Computation Time (seconds)")
title("Volume size = 128, 10K projections, Computation Time vs. nStreams")

set(gca,'FontSize', 18)
axis square
%%





