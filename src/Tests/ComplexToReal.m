addpath('C:\GitRepositories\MultiGPUGridder\bin\Debug')

VolumeSize = 50;
nSlices = 20;

Volume = complex(single(rand(VolumeSize,VolumeSize,nSlices)*100));
GPU_Device = 0;
reset(gpuDevice(GPU_Device+1));

RealVolume = mexComplexToReal(...
    complex(Volume), ...
    int32(size(Volume)), ...
    int32(GPU_Device));

GT_RealVolume = real(Volume);

for slice = 1:size(RealVolume,3)

    subplot(1,3,1)
    imagesc(RealVolume(:,:,slice))
    subplot(1,3,2)
    imagesc(GT_RealVolume(:,:,slice))
    subplot(1,3,3)
    imagesc(RealVolume(:,:,slice) - GT_RealVolume(:,:,slice))
    title(num2str(slice))
    pause(0.5)
    
end


isequal(RealVolume, GT_RealVolume)