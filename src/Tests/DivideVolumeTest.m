function Result = DivideVolumeTest(VolumeSize, nSlices, GPU_Device)

reset(gpuDevice(GPU_Device+1));

VolumeOne = single(round(rand(VolumeSize,VolumeSize,nSlices) * 100));
VolumeTwo = single(round(rand(size(VolumeOne))*100));

% Avoid zero values (the CUDA kernel doesn't return inf while Matlab does)
VolumeOne = VolumeOne + 1e-2;
VolumeTwo = VolumeTwo + 1e-2;

DividedVolume = mexDivideVolume(...
    single(VolumeOne), ...
    single(VolumeTwo), ...
    int32(size(VolumeOne)), ...
    int32(GPU_Device));

GT_DividedVolume = VolumeOne ./ VolumeTwo;

Result = isequal(DividedVolume, GT_DividedVolume);