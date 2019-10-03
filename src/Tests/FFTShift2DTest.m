addpath('C:\GitRepositories\MultiGPUGridder\bin\Debug')
addpath('C:\GitRepositories\MultiGPUGridder\src\src\Matlab\utils')

VolumeSize = 32;


load mri;
img = squeeze(D);
img = imresize3(img,[VolumeSize, VolumeSize, VolumeSize]);
Volume = single(img);


GPU_Device = 0;
reset(gpuDevice(GPU_Device+1));

FFTShiftedVolume = mexFFTShift2D(...
    single(Volume), ...
    int32(size(Volume)), ...
    int32(GPU_Device));

GT_FFTShift2D = zeros(size(Volume));
for i = 1:size(GT_FFTShift2D,3)
    GT_FFTShift2D(:,:,i) = fftshift(Volume(:,:,i));
end

easyMontage(FFTShiftedVolume(:,:,:), 1)
easyMontage(GT_FFTShift2D(:,:,:), 2)

slice = 2;
subplot(1,3,1)
imagesc(FFTShiftedVolume(:,:,slice))
subplot(1,3,2)
imagesc(GT_FFTShift2D(:,:,slice))
subplot(1,3,3)
imagesc(FFTShiftedVolume(:,:,slice) - GT_FFTShift2D(:,:,slice))


isequal(FFTShiftedVolume, GT_FFTShift2D)