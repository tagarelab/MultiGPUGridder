addpath('C:\GitRepositories\MultiGPUGridder\bin\Debug')

VolumeSize = 200;
nSlices = 100;

VolumeOne = single(round(rand(VolumeSize,VolumeSize,nSlices)));
VolumeTwo = single(round(rand(size(VolumeOne))));

VolumeOne = VolumeOne + 1e-2;
VolumeTwo = VolumeTwo + 1e-2;

GPU_Device = 0;
reset(gpuDevice(GPU_Device+1));

MultipliedVolume = mexMultiplyVolume(...
    single(VolumeOne), ...
    single(VolumeTwo), ...
    int32(size(VolumeOne)), ...
    int32(GPU_Device));

GT_MultipliedVolume = VolumeOne .* VolumeTwo;

% for slice = 1:size(VolumeOne,3)
% 
%     subplot(1,3,1)
%     imagesc(MultipliedVolume(:,:,slice))
%     subplot(1,3,2)
%     imagesc(GT_MultipliedVolume(:,:,slice))
%     subplot(1,3,3)
%     imagesc(MultipliedVolume(:,:,slice) - GT_MultipliedVolume(:,:,slice))
%     title(num2str(slice))
%     pause(0.5)
%     
% end


isequal(MultipliedVolume, GT_MultipliedVolume)