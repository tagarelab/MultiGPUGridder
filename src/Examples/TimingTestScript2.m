clc
close all
clear all

%% Add the required Matlab file paths
mfilepath=fileparts(which('MultiGPUGridder_Matlab_Class.m'));
addpath(genpath(fullfile(mfilepath)));
% addpath('C:\GitRepositories\MultiGPUGridder\bin\Debug')
% addpath(genpath("C:\GitRepositories\MultiGPUGridder\src\src"))

addpath(genpath('/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj'))

%% Parameters for creating the volume and coordinate axes
VolumeSize = 128;
interpFactor = 2;
n1_axes = 280%[1 10 50 100];
n2_axes = 100;


for i = 1:length(n1_axes)
    n1_axes(i)*n2_axes
%% Create the volume
load mri;
MRI_volume = squeeze(D);
MRI_volume = imresize3(MRI_volume,[VolumeSize, VolumeSize, VolumeSize]);

% Define the projection directions
coordAxes = create_uniform_axes(n1_axes(i),n2_axes,0,10);

%         M = size(MRI_volume, 3);
%         rMax = floor(M/2-2);
%         num_projdir = n1_axes(i) * n2_axes;
%         
%         tic
%         CPU_Forward_Project = mex_forward_project(double(MRI_volume), M, coordAxes, num_projdir, rMax);
%         toc   
%         
%         tic
%         BackProjected_Volume = mex_back_project(double(CPU_Forward_Project), M, coordAxes, num_projdir, rMax);
%         toc
%         
% 
% 
% continue

%% Create the gridder object
gridder = MultiGPUGridder_Matlab_Class(VolumeSize, n1_axes(i) * n2_axes, interpFactor);

%% Set the volume
gridder.setVolume(MRI_volume);

% allocate memory
images = gridder.forwardProject(coordAxes); 
%% Run the forward projection
tic
images = gridder.forwardProject(coordAxes);    
toc
% easyMontage(images, 1)

%% Run the back projection
gridder.resetVolume();
tic
gridder.backProject(gridder.Images, coordAxes)
toc
%% Get the back projection result
% vol=gridder.getVol();
% easyMontage(vol, 2)

%% Reconstruct the volume
% reconstructVol = gridder.reconstructVol();
% easyMontage(reconstructVol, 3)

end