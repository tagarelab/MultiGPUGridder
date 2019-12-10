clc
close all
clear all

% mex /home/brent/Documents/MATLAB/simple_gpu_gridder_Obj/src/Gridders/CPU/mex_forward_project.c

addpath(genpath('/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj/src/Matlab/'))
addpath(genpath('/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj/'))

% Parameters for creating the volume and coordinate axes
VolumeSize = 128;
interpFactor = 2;

n1_axes = [1 10 50 100];
n2_axes = 100;



% 
% 
% 
% 
%     % Parameters for creating the volume and coordinate axes
%     VolumeSize = 128;
%     interpFactor = 2;
%     n1_axes = [1 10 50 100];
%     n2_axes = 100;
% 
%     iter = 1;
%         
%     for i=1:length(n1_axes)
%         
%         % Create the volume
%         load mri;
%         MRI_volume = squeeze(D);
%         MRI_volume = imresize3(MRI_volume,[VolumeSize, VolumeSize, VolumeSize]);
%         
%         % Define the projection directions
%         coordAxes = create_uniform_axes(n1_axes(i),n2_axes,0,10);              
%         
%         % Run on the CPU
%         M = size(MRI_volume, 3);
%         rMax = floor(M/2-2);
%         num_projdir = n1_axes(i) * n2_axes;
%         
%         tic
%         CPU_Forward_Project = mex_forward_project(double(MRI_volume), M, coordAxes, num_projdir, rMax);
%         time(iter,1) = toc;
%         
%         tic
%         BackProjected_Volume = mex_back_project(double(CPU_Forward_Project), M, coordAxes, num_projdir, rMax);
%         time(iter,2) = toc;       
%         
%         
%         % Create the gridder object
%         delete gridder
%         
%         gridder = MultiGPUGridder_Matlab_Class(VolumeSize, n1_axes(i) * n2_axes, interpFactor);
%         
%         % Set the volume
%         gridder.setVolume(MRI_volume);
%         
%         % Allocate the memory
%         images = gridder.forwardProject(coordAxes);
%         
%         
%         % Run the forward projection
%         tic
%         images = gridder.forwardProject(coordAxes);
%         time(iter,3) = toc;
%         
%         % Run the back projection
%         gridder.resetVolume();
%         tic
%         gridder.backProject(gridder.Images, coordAxes)
%         time(iter,4) = toc;
%         
%         iter = iter + 1;
%         
%         delete gridder
%         
%         
%     end
% 
% 
% 
% 
% 
















% return
%%

nStreams = 32%:5:20


RunFFTOnGPU = true;
times = [];
for i = 1:length(n1_axes)
%     for NumGPUs = 0:1
        [n1_axes(i)*n2_axes]
        new_time = MultiGridderTimingTest(VolumeSize, n1_axes(i), n2_axes, interpFactor, RunFFTOnGPU, nStreams);
        times = [times; new_time] 
%     end
end

% Forward Projection times
plot(nStreams, times(:,1), 'r*--')
hold on

% Back Projection times
plot(nStreams, times(:,2), 'b*--')
legend("Forward Projection", "Back Projection")

%%





