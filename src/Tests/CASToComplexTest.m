addpath('C:\GitRepositories\MultiGPUGridder\bin\Debug')
addpath('C:\GitRepositories\MultiGPUGridder\src\src\Matlab\utils')

VolumeSize = 256;
nSlices = 256;

Volume = single(rand(VolumeSize,VolumeSize,nSlices)*100)+ i * single(rand(VolumeSize,VolumeSize,nSlices)*100);

CASVolume = ToCAS(Volume);




% Calculate the ground truth complex array from the CAS volume
[s1, s2, s3]=size(CASVolume);
o1=min(2,s1);
o2=min(2,s2);
GT_ComplexOutput=complex(zeros(s1,s2,s3,'single'));
GT_ComplexOutput(o1:s1,o2:s2,:)=0.5*(CASVolume(o1:s1,o2:s2,:)+CASVolume(s1:-1:o1,s2:-1:o2,:)...
    +1i*(CASVolume(o1:s1,o2:s2,:)-CASVolume(s1:-1:o1,s2:-1:o2,:)));

GPU_Device = 0;
reset(gpuDevice(GPU_Device+1));

ComplexOutput = mexCASToComplex(...
    CASVolume, ...
    int32(size(CASVolume)), ...
    int32(GPU_Device));


                
for slice = 1%:size(CASVolume,3)

    subplot(1,3,1)
    imagesc(real(ComplexOutput(:,:,slice)))
    subplot(1,3,2)
    imagesc(real(GT_ComplexOutput(:,:,slice)))
    subplot(1,3,3)
    imagesc(real(ComplexOutput(:,:,slice)) - real(GT_ComplexOutput(:,:,slice)))
    title(num2str(slice))
    pause(0.5)
    
end

isequal(ComplexOutput, GT_ComplexOutput)







