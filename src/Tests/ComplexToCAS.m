addpath('C:\GitRepositories\MultiGPUGridder\bin\Debug')

VolumeSize = 256;
nSlices = 256;

Volume = complex(single(rand(VolumeSize,VolumeSize,nSlices)*100));
GPU_Device = 0;
reset(gpuDevice(GPU_Device+1));

CASVolume = mexComplexToCAS(...
    complex(Volume), ...
    int32(size(Volume)), ...
    int32(GPU_Device));


GT_CASVolume = ToCAS(Volume);
                
% for slice = 1:size(CASVolume,3)
% 
%     subplot(1,3,1)
%     imagesc(CASVolume(:,:,slice))
%     subplot(1,3,2)
%     imagesc(GT_CASVolume(:,:,slice))
%     subplot(1,3,3)
%     imagesc(CASVolume(:,:,slice) - GT_CASVolume(:,:,slice))
%     title(num2str(slice))
%     pause(0.5)
%     
% end


isequal(CASVolume, GT_CASVolume)