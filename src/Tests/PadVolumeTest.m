function Result = PadVolumeTest(VolumeSize, nSlices, PaddingX, PaddingY, PaddingZ, GPU_Device)

reset(gpuDevice(GPU_Device+1));
InputVolume = ones(VolumeSize,VolumeSize,nSlices);

PaddedVolume = mexPadVolume(...
    single(InputVolume), ...
    int32(size(InputVolume)), ...
    int32(PaddingX), ...
    int32(PaddingY), ...
    int32(PaddingZ), ...
    int32(GPU_Device));

GT_PaddedVolume = zeros(...
    size(InputVolume,1) + PaddingX * 2, ...
    size(InputVolume,2) + PaddingY * 2, ...
    size(InputVolume,3) + PaddingZ * 2);

GT_PaddedVolume(PaddingX+1:end-PaddingX,PaddingY+1:end-PaddingY,PaddingZ+1:end-PaddingZ) = InputVolume;

Result = isequal(GT_PaddedVolume, PaddedVolume);
