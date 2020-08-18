% clc
close all
clear all

 
start = tic;

% Add the required Matlab file paths
mfilepath=fileparts(which('MultiGPUGridder_Matlab_Class.m'));
addpath(genpath(fullfile(mfilepath)));
% addpath('C:\GitRepositories\MultiGPUGridder\bin\Debug')
addpath(genpath("C:\GitRepositories\MultiGPUGridder\src\src"))
addpath(genpath("/home/brent/cryo_EM/lib"))

% Parameters for creating the volume and coordinate axes
VolumeSize = 256;
interpFactor = 2;
n1_axes = 10;
n2_axes = 10;

% Create the volume
load mri;
MRI_volume = squeeze(D);
MRI_volume = imresize3(MRI_volume,[VolumeSize, VolumeSize, VolumeSize]);

% Define the projection directions
coordAxes = create_uniform_axes(n1_axes,n2_axes,0,10);
coordAxes = coordAxes(:,1:n1_axes*n2_axes);

% Create the gridder object
gridder = MultiGPUGridder_Matlab_Class(VolumeSize, n1_axes * n2_axes, interpFactor);

% Set the volume
gridder.setVolume(MRI_volume);

% Run the forward projection once to allocate the memory (to get a more accurate timing below)
% images = gridder.forwardProject(coordAxes);    

% Run the forward projection
tic
images = gridder.forwardProject(coordAxes);    
disp("Forward Project: " + toc + " seconds")
easyMontage(images(:,:,1:min(100, size(images,3))), 1)

% x = load("images.mat");
% gridder.Images = x.images;


% Run the back projection
gridder.resetVolume();
tic

gridder.backProject(images, coordAxes)
disp("Back Project: " + toc + " seconds")

tic
vol=gridder.getVol();
disp("Get volume: " + toc + " seconds")
easyMontage(vol, 2)

% easyMontage(gridder.CASVolume)

% Reconstruct the volume
tic
% reconstructVol = gridder.reconstructVol();
% disp("Reconstruct Volume: " + toc + " seconds")
% easyMontage(reconstructVol, 3)

disp("Total time: " + toc(start))
% quit % Needed if running the NVIDIA profiler