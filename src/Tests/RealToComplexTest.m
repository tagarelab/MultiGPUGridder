addpath('C:\GitRepositories\MultiGPUGridder\bin\Debug')

VolumeSize = 20;
nSlices = 10;

Volume = single(rand(VolumeSize,VolumeSize,nSlices));
GPU_Device = 0;
reset(gpuDevice(GPU_Device+1));

ComplexVolume = mexRealToComplex(...
    single(Volume), ...
    int32(size(Volume)), ...
    int32(GPU_Device));

GT_ComplexVolume = complex(Volume);

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


isequal(ComplexVolume, GT_ComplexVolume)