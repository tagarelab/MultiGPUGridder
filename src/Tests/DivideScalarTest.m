function Result = DivideScalarTest(VolumeSize, nSlices, Scalar, GPU_Device)

Volume = single(rand(VolumeSize,VolumeSize,nSlices));
Scalar = single(Scalar);

reset(gpuDevice(GPU_Device+1));

DividedVolume = mexDivideScalar(...
    single(Volume), ...
    int32(size(Volume)), ...
    single(Scalar), ...
    int32(GPU_Device));

GT_DividedVolume = Volume ./ Scalar;

Result = isequal(DividedVolume, GT_DividedVolume);