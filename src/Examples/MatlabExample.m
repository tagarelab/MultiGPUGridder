clc
close all
clear all

% Add the required Matlab file paths
mfilepath=fileparts(which('MultiGPUGridder_Matlab_Class.m'));
addpath(genpath(fullfile(mfilepath)));
addpath(genpath("/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj"))

% Parameters for creating the volume and coordinate axes
VolumeSize = 128;
interpFactor = 2;
n1_axes = 100;
n2_axes = 10;

% Create the volume
load mri;
MRI_volume = squeeze(D);
MRI_volume = imresize3(MRI_volume,[VolumeSize, VolumeSize, VolumeSize]);

% Define the projection directions
coordAxes = create_uniform_axes(n1_axes,n2_axes,0,10);

% Create the gridder object
gridder = MultiGPUGridder_Matlab_Class(VolumeSize, n1_axes * n2_axes, interpFactor);

% Set the volume
gridder.setVolume(MRI_volume);

% Run the forward projection
images = gridder.forwardProject(coordAxes);    
easyMontage(images, 1)

% Run the back projection
gridder.resetVolume();
gridder.backProject(gridder.Images, coordAxes)

vol=gridder.getVol();
easyMontage(vol, 2)

% Reconstruct the volume
reconstructVol = gridder.reconstructVol();
easyMontage(reconstructVol, 3)