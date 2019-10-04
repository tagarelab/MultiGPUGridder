function Result = AddVolumeTest(VolumeSize, nSlices, GPU_Device)



VolumeOne = single(rand(VolumeSize,VolumeSize,nSlices));
VolumeTwo = single(rand(VolumeSize,VolumeSize,nSlices)*5);

reset(gpuDevice(GPU_Device+1));

AddedVolume = mexAddVolume(...
    single(VolumeOne), ...
    single(VolumeTwo), ...
    int32(size(VolumeOne)), ...
    int32(GPU_Device));

Result = isequal(AddedVolume, VolumeOne + VolumeTwo);