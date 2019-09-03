clc
close all
clear gridder 
clear all

% bdclose all; % clear all libraries out of memory ( supposedly )
% clear all;   % clear all workspace variables, mex, etc. ( supposedly )
rehash;      % cause all .m files to be reparsed when invoked again


addpath('./src')
addpath('./utils')
addpath('./bin') % The compiled mex file is stored in the bin folder

addpath(genpath("/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj"));
addpath(genpath("/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj_Original"));
addpath(genpath("/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj_Original/utils"));

disp("Resetting devices...")
% for i = 1:4
    reset(gpuDevice());
% end

VolumeSize = 64;
interpFactor = 2;
n1_axes = 200;
n2_axes = 20;

load mri;
img = squeeze(D);
img = imresize3(img,[VolumeSize, VolumeSize, VolumeSize]);
MRI_volume = single(img);
% easyMontage(vol,1);

% Define the projection directions
coordAxes=single([1 0 0 0 1 0 0 0 1]');
coordAxes=[coordAxes create_uniform_axes(n1_axes,n2_axes,0,10)];
% coordAxes  = repmat(coordAxes, [1 n1_axes*n2_axes]);

nCoordAxes = length(coordAxes(:))/9;


KB_Vector = load("KB_Vector.mat");
KBTable = KB_Vector.KB_Vector;

gridder = MultiGPUGridder_Matlab_Class(int32(VolumeSize), int32(nCoordAxes), single(2));
gridder.coordAxes = single(coordAxes(:));
gridder.NumAxes = int32(nCoordAxes);
gridder.VolumeSize = int32(VolumeSize);
gridder.Volume = single(MRI_volume); %ones(gridder.VolumeSize, gridder.VolumeSize, gridder.VolumeSize, 'single');
gridder.CASVolumeSize = repmat(gridder.VolumeSize * gridder.interpFactor + gridder.extraPadding * 2, 1, 3)
gridder.CASVolume = zeros(gridder.CASVolumeSize, 'single');
gridder.CASImages = zeros([VolumeSize*interpFactor, VolumeSize*interpFactor, gridder.NumAxes], 'single');
gridder.ImageSize = [gridder.VolumeSize, gridder.VolumeSize, gridder.NumAxes];
gridder.Images = zeros(gridder.ImageSize(1), gridder.ImageSize(2), gridder.ImageSize(3), 'single');
gridder.KBTable = single(KBTable);

gridder.Set()



% Volume = gridder.Get('Volume');

% slice = 60
% subplot(1,3,1)
% imagesc(gridder.Volume(:,:,slice));
% subplot(1,3,2)
% imagesc(MRI_volume(:,:,slice));
% subplot(1,3,3)
% imagesc(gridder.Volume(:,:,slice) - MRI_volume(:,:,slice));
% colorbar
%  


Volume = gridder.Get('Volume');

disp("ForwardProject...")
tic
gridder.ForwardProject()
toc



% [origBox,interpBox,CASBox]=getSizes(VolumeSize,interpFactor,3);
% 
% imgs = imgsFromCASImgs(gridder.CASImages, interpBox, []); 




% gpuGridder = gpuBatchGridder(VolumeSize,n1_axes*n2_axes+1,interpFactor);
% gpuGridder.setVolume(MRI_volume);
% imgs_GT  = gpuGridder.forwardProject(coordAxes);
% 
% CASImgs_GT = gather(gpuGridder.gridder.gpuCASImgs)
% 
% imagesc(real(fftshift2(fft2(fftshift2(CASImgs_GT(:,:,1))))))


CASVolume = gridder.Get('CASVolume');
Images = gridder.Get('Images');

CASImages = gridder.Get('CASImages');



% easyMontage(gridder.CASVolume(:,:,1:10), 1)
% easyMontage(CASVolume, 1)
% easyMontage(gridder.CASVolume(:,:,1:10), 1)

% colormap gray


% 
% Images = gridder.Get('Images');

% max(gridder.CASVolume(:))
% max(gridder.CASImages(:))
% max(CASImages(:))

% Compare with GT
tic
[CASVol_GT, CASBox, origBox, interpBox, fftinfo] = Vol_Preprocessing(MRI_volume, interpFactor);

gpuGridder = gpuBatchGridder(VolumeSize,n1_axes*n2_axes+1,interpFactor);
gpuGridder.setVolume(MRI_volume);
gpuGridderImg  = gpuGridder.forwardProject(coordAxes);
gpuGridderCASImgs = gather(gpuGridder.gridder.gpuCASImgs);
gpuGridderCASVolume = gather(gpuGridder.gridder.gpuVol);


% easyMontage(gridder.CASVolume - gpuGridderCASVolume, 1)
% colorbar

% easyMontage(gridder.CASImages(:,:,1:10) - gpuGridderCASImgs(:,:,1:10), 2)
% colorbar

toc
 %%
 
 for slice = 5
    
%     
%     slice = 1

    subplot(2,3,1)
    imagesc(gridder.CASVolume(:,:,slice));
    title("Slice " + num2str(slice))
    subplot(2,3,2)
    imagesc(CASVol_GT(:,:,slice));
    title("Slice " + num2str(slice))
    subplot(2,3,3)
    imagesc(gridder.CASVolume(:,:,slice) - CASVol_GT(:,:,slice));
    colormap jet
    title("Slice " + num2str(slice))
    colorbar
 end
 
 %%
for slice = 1
    
    subplot(2,3,4)
    imagesc(gridder.CASImages(:,:,slice));
    title("Slice " + num2str(slice))
    subplot(2,3,5)
    imagesc(gpuGridderCASImgs(:,:,slice));
    title("Slice " + num2str(slice))
    subplot(2,3,6)
    imagesc(gridder.CASImages(:,:,slice) - gpuGridderCASImgs(:,:,slice));
    colormap jet
    title("Slice " + num2str(slice))
    colorbar
    
    
    
    
    pause(0.5)
 end
 
 return
% max(CASVolume(:)) / max(CASVol_GT(:)) 
%%
close all

easyMontage(gridder.Images, 1)
colormap jet

for i = 1:size(gridder.Images,3)
    x = gridder.Images(:,:,i);
    
    if max(x(:)) == 0
        disp(i)
        break
    end
end

return;

%%

GT_Imgs = load("gpuGridderImg.mat");
GT_Imgs = GT_Imgs.gpuGridderImg;

max(gridder.Images(:)) / max(GT_Imgs(:))

GT_CASImgs = load("gpuGridderCASImgs.mat");
GT_CASImgs = GT_CASImgs.gpuGridderCASImgs;

for slice = 1
    
%     
%     slice = 1
    subplot(3,3,1)
    imagesc(gridder.CASVolume(:,:,slice));
    subplot(3,3,2)
    imagesc(CASVol_GT(:,:,slice));
    subplot(3,3,3)
    imagesc(gridder.CASVolume(:,:,slice) - CASVol_GT(:,:,slice));
    colorbar
%     subplot(3,3,4)
%     imagesc(gridder.CASImages(:,:,slice))
%     axis square
%     
%     subplot(3,3,5)
%     imagesc(GT_CASImgs(:,:,slice))
%     axis square
% 
%     subplot(3,3,6)
%     imagesc(gridder.CASImages(:,:,slice) - GT_CASImgs(:,:,slice))
%     axis square
%     colorbar

    subplot(3,3,4)
%     imagesc(imgsFromCASImgs(gridder.CASImages(:,:,slice), interpBox, fftinfo))
    imagesc(gridder.CASImages(:,:,slice))
    
    axis square
    
    subplot(3,3,5)
    imagesc(GT_CASImgs(:,:,slice))
    axis square

    subplot(3,3,6)
    imagesc(gridder.CASImages(:,:,slice) - GT_CASImgs(:,:,slice))
    axis square
    colorbar
    
    
    
    
    
    
    
    
    h(1) = subplot(3,3,7);

%     imagesc(real(fftshift2(fft2(fftshift2(gridder.Images(:,:,slice))))))
    imagesc(gridder.Images(:,:,slice))
    axis square
    colormap jet

    h(2) = subplot(3,3,8);
    imagesc(GT_Imgs(:,:,slice))
    axis square
    colormap jet
    
    h(3) = subplot(3,3,9);
    imagesc(gridder.Images(:,:,slice) - GT_Imgs(:,:,slice))
    axis square
    colormap jet
    colorbar
    linkaxes(h, 'xy')
    zoom on
    pause(0.1)
end


return
%%

gridder.Delete();




clear gridder



return
abc
%% Create a volume 
% Initialize parameters
tic

nBatches = 12;
nGPUs = 4;
nStreams = 16;
volSize = 256;
n1_axes = 100;
n2_axes = 100;

kernelHWidth = 2;

interpFactor = 2.0;

origSize   = volSize;
volCenter  = volSize/2  + 1;
origCenter = origSize/2 + 1;
origHWidth = origCenter - 1;

%% Fuzzy sphere
disp("fuzzymask()...")
vol=fuzzymask(origSize,3,origSize*.25,2,origCenter*[1 1 1]);
size(vol)

% Use the example matlab MRI image to take projections of
% load mri;
% img = squeeze(D);
% img = imresize3(img,[volSize, volSize, volSize]);
% vol = single(img);
% easyMontage(vol,1);


%% Define the projection directions
coordAxes=single([1 0 0 0 1 0 0 0 1]');
coordAxes=[coordAxes create_uniform_axes(n1_axes,n2_axes,0,10)];
coordAxes = coordAxes(:);
nCoordAxes = length(coordAxes)/9;

%% MATLAB pre-processing to covert vol to CASVol
% interpBoc and fftinfo are needed for plotting the results
disp("MATLAB Vol_Preprocessing()...")
[CASVol, CASBox, origBox, interpBox, fftinfo] = Vol_Preprocessing(vol, interpFactor);

disp("Volume size: " + num2str(volSize))
disp("Number of coordinate axes: " + num2str(nCoordAxes))
 
%% Initialize the multi GPU gridder
obj = MultiGPUGridder_Matlab_Class();
obj.SetNumberBatches(nBatches);
obj.SetNumberGPUs(nGPUs);
obj.SetNumberStreams(nStreams);
obj.SetMaskRadius(single(size(vol,1)*interpFactor/2 - 1));

disp("SetVolume()...")
obj.setVolume(single(CASVol))
imgs=imgsFromCASImgs(InterpCASImgs(:,:,1:10), interpBox, fftinfo); 


disp("SetAxes()...")
obj.SetAxes(coordAxes)

disp("SetImgSize()...")
% This is the size of the interpolated (but non-zero padded) projection images
obj.SetImgSize(int32([size(vol,1)*interpFactor, size(vol,1)*interpFactor,nCoordAxes]))

%% Run the forward projection kernel
% clc
disp("Forward_Project()...")
obj.forwardProject()

disp("Displaying allocated memory()...")
obj.CUDA_disp_mem('all')
obj.disp_mem('all')

disp("GetImgs()...")
InterpCASImgs = obj.GetImgs();

disp("imgsFromCASImgs()...")
imgs=imgsFromCASImgs(InterpCASImgs(:,:,1:10), interpBox, fftinfo); 

% Check to see if all the projections are there
for i = 1:size(imgs,3)
    temp = imgs(:,:,i);
   if (max(temp(:)) <= 0)
       disp("No projection for slice " + num2str(i))
   end
end

easyMontage(imgs,1);
colormap gray

clear obj
clear all
return

%% Run the back projection kernel
disp("ResetVolume()...")
obj.ResetVolume()

% Convert the forward projection images back to CAS type and copy to the CPU pinned memory
CAS_projection_imgs = CASImgsFromImgs(imgs,interpBox,fftinfo);
obj.SetImages(CAS_projection_imgs)

disp("Back_Project()...")
obj.Back_Project()

% Get the volumes from all the GPUs added together
disp("Get_Volume()...") 
volCAS = obj.GetVolume();

%% Get the density of inserted planes by backprojecting CASimages of values equal to one
disp("Get Plane Density()...")
interpImgs=ones([interpBox.size interpBox.size size(coordAxes,1)/9],'single');
obj.ResetVolume();
obj.SetImages(interpImgs)
obj.Back_Project()
volWt = obj.GetVolume();


%% Normalize the back projection result with the plane density
% Divide the previous volume with the plane density volume
volCAS=volCAS./(volWt+1e-6);
                
% Reconstruct the volume from CASVol
disp("volFromCAS()...")
volReconstructed=volFromCAS(volCAS,CASBox,interpBox,origBox,kernelHWidth);

disp("easyMontage()...")
easyMontage(volReconstructed,2);

%% Compare with the previous GPU gridder code


comparing_gridders = false;
if comparing_gridders == true
    x = load("gpuGridderResults.mat");
    gpuGridder = x.gpuGridder


    close all
    easyMontage(gpuGridder.forward_project,3);
    easyMontage(imgs,4);

    easyMontage(gpuGridder.forward_project - imgs,5);

    close all
    slice = 215;

    figure
    for slice = 1:10%size(imgs,3)

        h(1) = subplot(2,3,1)
        imagesc(imgs(:,:,slice))
        title("Multi GPU Forward Projection")
        h(2) = subplot(2,3,2)
        imagesc(gpuGridder.forward_project (:,:,slice))
        title("gpuGridder Forward")
        h(3) = subplot(2,3,3)
        diff_img = imgs(:,:,slice) - gpuGridder.forward_project(:,:,slice);
    %     diff_img(diff_img < 0.00001) = 0; % Clean up the visualization to better debug
        imagesc(diff_img)
        title("Difference")
        colorbar
        linkaxes(h, 'xy')
        zoom on



        h(4) = subplot(2,3,4)
        imagesc(volReconstructed(:,:,slice))
        title("Multi GPU Back Projection With Weighing")
        h(5) = subplot(2,3,5)
        imagesc(gpuGridder.back_project(:,:,slice))
        title("gpuGridder Back Projection With Weighing")
        h(6) = subplot(2,3,6)
        imagesc(volReconstructed(:,:,slice) - gpuGridder.back_project(:,:,slice))
        title("Difference")

        colorbar
        linkaxes(h, 'xy')
        zoom on

        pause(0.005)

    end

    diff_forward = imgs - gpuGridder.forward_project;
    max(diff_forward(:))

    max_diff = volReconstructed - gpuGridder.back_project;
    max_diff = max(max_diff(:))

    max(gpuGridder.forward_project(:))
    max(imgs(:))

end

%% Free the memory
% obj.CUDA_Free('all')
clear obj
clear all

toc