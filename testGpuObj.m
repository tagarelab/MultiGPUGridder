%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Test of the GpuObject                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
reset(gpuDevice());

%kernelHWidth=2.0;


%Set the path to the utils directory
addpath(fullfile('.','utils'));

    

%Initialize parameters
volSize=256
n1_axes=100;
n2_axes=50;
interpFactor=2.0;

%Get the gridder
a=gpuBatchGridder(volSize,n1_axes*n2_axes+1,interpFactor);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Volume

origSize=volSize;
volCenter=volSize/2+1;
origCenter=origSize/2+1;
origHWidth= origCenter-1;



%Fuzzy sphere
vol=fuzzymask(origSize,3,origSize*.25,2,origCenter*[1 1 1]);
a.setVolume(vol);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Create Coordaxes

coordAxes=single([1 0 0 0 1 0 0 0 1]');
coordAxes=[coordAxes create_uniform_axes(n1_axes,n2_axes,0,10)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forward Project
%tic
tic
img=a.forwardProject(coordAxes);
toc
easyMontage(img,1);
%toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Back Project

a.resetVolume();
a.setVolume(vol * 0);
%a.setCoordAxes(coordAxes);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% Create Reconstruction volume and ones images
%volR=a.backProject(img,coordAxes);
%tic
a.backProject(img,coordAxes);
%toc
%tic
volR=a.reconstructVol(coordAxes);
easyMontage(volR,2);

% save("gpuGriddervolR.mat", "volR")

figure(10);
hold off
plotProfiles(volR,origCenter,1,10,0);
hold on;
plotProfiles(vol,origCenter,1,10,0);
