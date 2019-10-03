addpath('C:\GitRepositories\MultiGPUGridder\bin\Debug')

VolumeOne = single(round(rand(30,30,20) * 100));
VolumeTwo = single(round(rand(size(VolumeOne))*100));

VolumeOne = VolumeOne + 1e-2;
VolumeTwo = VolumeTwo + 1e-2;

GPU_Device = 0;
reset(gpuDevice(GPU_Device+1));

DividedVolume = mexDivideVolume(...
    single(VolumeOne), ...
    single(VolumeTwo), ...
    int32(size(VolumeOne)), ...
    int32(GPU_Device));

GT_DividedVolume = VolumeOne ./ VolumeTwo;

slice = 2;
subplot(1,3,1)
imagesc(DividedVolume(:,:,slice))
subplot(1,3,2)
imagesc(GT_DividedVolume(:,:,slice))
subplot(1,3,3)
imagesc(DividedVolume(:,:,slice) - GT_DividedVolume(:,:,slice))

isequal(DividedVolume, GT_DividedVolume)