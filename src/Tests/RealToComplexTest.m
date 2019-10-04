function Result = RealToComplexTest(VolumeSize, nSlices, GPU_Device)

reset(gpuDevice(GPU_Device+1));

Volume = single(rand(VolumeSize,VolumeSize,nSlices));

ComplexVolume = mexRealToComplex(...
    single(Volume), ...
    int32(size(Volume)), ...
    int32(GPU_Device));

GT_ComplexVolume = complex(Volume);

Result = isequal(ComplexVolume, GT_ComplexVolume);