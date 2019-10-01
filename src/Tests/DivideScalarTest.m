addpath('C:\GitRepositories\MultiGPUGridder\bin\Debug')

Volume = single(rand(30,30,2));
Scalar = single(rand(1));

GPU_Device = 0;
reset(gpuDevice(GPU_Device+1));

DividedVolume = mexDivideScalar(...
    single(Volume), ...
    int32(size(Volume)), ...
    single(Scalar), ...
    int32(GPU_Device));

GT_DividedVolume = Volume ./ Scalar;
subplot(1,2,1)
imagesc(DividedVolume(:,:,2))
subplot(1,2,2)
imagesc(GT_DividedVolume(:,:,2))

isequal(DividedVolume, GT_DividedVolume)