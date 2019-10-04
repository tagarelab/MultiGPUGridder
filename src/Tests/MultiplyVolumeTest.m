function Result = MultiplyVolumeTest(VolumeSize, nSlices, GPU_Device)

reset(gpuDevice(GPU_Device+1));

VolumeOne = single(round(rand(VolumeSize,VolumeSize,nSlices)));
VolumeTwo = single(round(rand(size(VolumeOne))));

MultipliedVolume = mexMultiplyVolume(...
    single(VolumeOne), ...
    single(VolumeTwo), ...
    int32(size(VolumeOne)), ...
    int32(GPU_Device));

GT_MultipliedVolume = VolumeOne .* VolumeTwo;

Result = isequal(MultipliedVolume, GT_MultipliedVolume);