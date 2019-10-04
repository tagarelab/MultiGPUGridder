function Result = FFTShift2DTest(VolumeSize, nSlices, GPU_Device)

reset(gpuDevice(GPU_Device+1));

load mri;
img = squeeze(D);
img = imresize3(img,[VolumeSize, VolumeSize, nSlices]);
Volume = single(img);

FFTShiftedVolume = mexFFTShift2D(...
    single(Volume), ...
    int32(size(Volume)), ...
    int32(GPU_Device));

GT_FFTShift2D = zeros(size(Volume));
for i = 1:size(GT_FFTShift2D,3)
    GT_FFTShift2D(:,:,i) = fftshift(Volume(:,:,i));
end

Result = isequal(FFTShiftedVolume, GT_FFTShift2D);