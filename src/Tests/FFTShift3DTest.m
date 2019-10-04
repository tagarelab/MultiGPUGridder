function Result = FFTShift3DTest(VolumeSize, GPU_Device)

reset(gpuDevice(GPU_Device+1));

load mri;
img = squeeze(D);
img = imresize3(img,[VolumeSize, VolumeSize, VolumeSize]);
Volume = single(img);

FFTShiftedVolume = mexFFTShift3D(...
    single(Volume), ...
    int32(size(Volume)), ...
    int32(GPU_Device));

GT_FFTShift3D = fftshift(Volume);

Result = isequal(FFTShiftedVolume, GT_FFTShift3D);