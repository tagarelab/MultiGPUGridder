function Result = ComplexToCASTest(VolumeSize, nSlices, GPU_Device)


Volume = complex(single(rand(VolumeSize,VolumeSize,nSlices)*100));
reset(gpuDevice(GPU_Device+1));

CASVolume = mexComplexToCAS(...
    complex(Volume), ...
    int32(size(Volume)), ...
    int32(GPU_Device));

GT_CASVolume = ToCAS(Volume);
                
Result = isequal(CASVolume, GT_CASVolume);