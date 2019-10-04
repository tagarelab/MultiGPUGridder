function Result = ComplexToRealTest(VolumeSize, nSlices, GPU_Device)

reset(gpuDevice(GPU_Device+1));

Volume = complex(single(rand(VolumeSize,VolumeSize,nSlices)*100));

RealVolume = mexComplexToReal(...
    complex(Volume), ...
    int32(size(Volume)), ...
    int32(GPU_Device));

GT_RealVolume = real(Volume);

Result = isequal(RealVolume, GT_RealVolume);