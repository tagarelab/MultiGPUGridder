addpath('C:\GitRepositories\MultiGPUGridder\bin\Debug')

InputVolume = ones(300,300,20);

CropX = 2;
CropY = 10;
CropZ = 5;
GPU_Device = 0;

CroppedVolume = mexCropVolume(...
    single(InputVolume), ...
    int32(size(InputVolume)), ...
    int32(CropX), ...
    int32(CropY), ...
    int32(CropZ), ...
    int32(GPU_Device));

imagesc(CroppedVolume(:,:,2))

GT_CroppedVolume = InputVolume(CropX+1:end-CropX, CropY+1:end-CropY, CropZ+1:end-CropZ);

isequal(GT_CroppedVolume, CroppedVolume)