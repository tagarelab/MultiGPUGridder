%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Test of the GpuObject                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all;
close all;
reset(gpuDevice());

tic

%kernelHWidth=2.0;


%Set the path to the utils directory
addpath(fullfile('.','utils'));    

%Initialize parameters
volSize=256%128%64;
n1_axes=100;%15;
n2_axes=10;%15
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





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Create Coordaxes

coordAxes=single([1 0 0 0 1 0 0 0 1]');
coordAxes=[coordAxes create_uniform_axes(n1_axes,n2_axes,0,10)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forward Project
tic
for i = 1:11

    vol = vol + 2;

    a.setVolume(vol);

    img=a.forwardProject(coordAxes);


end

toc

% 
% img_slice = 1000;
% 
% figure('Color', [1 1 1])
% subplot(1,3,1)
% imagesc(img(:,:,img_slice))
% title("gpuGridder")
% axis square
% 
% subplot(1,3,2)
% imagesc(brent_imgs(:,:,img_slice))
% title("C++ Gridder")
% axis square
% subplot(1,3,3)
% imagesc(img(:,:,img_slice) - brent_imgs(:,:,img_slice))
% title("Subtraction")
% 
% colormap gray
% axis square

sum(img(:) - brent_imgs(:))

x = load("imgs_brent_gridder.mat")

toc

easyMontage(img,1);

abc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Back Project

a.resetVolume();
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

figure(10);
hold off
plotProfiles(volR,origCenter,1,10,0);
hold on;
plotProfiles(vol,origCenter,1,10,0);
ylim([min(min(vol(:)), min(volR(:))), max(max(vol(:)), max(volR(:)))])
