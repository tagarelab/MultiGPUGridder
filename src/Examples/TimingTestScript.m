clc
close all
clear all

addpath(genpath('/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj/src/Matlab/'))

% Parameters for creating the volume and coordinate axes
VolumeSize = 128;
interpFactor = 2;
n1_axes = 100;
n2_axes = 100;

nStreams = 1:10:20
% 
% times = [];
% RunFFTOnGPU = false;
% for i = 1:length(nStreams)
%     new_time = MultiGridderTimingTest(VolumeSize, n1_axes, n2_axes, interpFactor, RunFFTOnGPU, nStreams(i));
%     times = [times; new_time]    
% end
% 
% plot(nStreams, times(:,1), 'b*--')

hold on

RunFFTOnGPU = true;
times = [];
for i = 1%:length(nStreams)
    new_time = MultiGridderTimingTest(VolumeSize, n1_axes, n2_axes, interpFactor, RunFFTOnGPU, nStreams(i));
    times = [times; new_time]    
end

plot(nStreams, times(:,1), 'r*--')