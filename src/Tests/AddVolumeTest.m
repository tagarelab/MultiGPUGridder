addpath('C:\GitRepositories\MultiGPUGridder\bin\Debug')

VolumeOne = single(ones(300,300,20));
VolumeTwo = single(ones(300,300,20)*5);


GPU_Device = 0;
reset(gpuDevice(GPU_Device+1));

AddedVolume = mexAddVolume(...
    single(VolumeOne), ...
    single(VolumeTwo), ...
    int32(size(VolumeOne)), ...
    int32(GPU_Device));

imagesc(AddedVolume(:,:,2))

isequal(AddedVolume, VolumeOne + VolumeTwo)