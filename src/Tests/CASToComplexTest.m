function Result = CASToComplexTest(VolumeSize, nSlices, GPU_Device)

reset(gpuDevice(GPU_Device+1));

Volume = single(rand(VolumeSize,VolumeSize,nSlices)*100)+ 1i * single(rand(VolumeSize,VolumeSize,nSlices)*100);

CASVolume = ToCAS(Volume);

ComplexOutput = mexCASToComplex(...
    CASVolume, ...
    int32(size(CASVolume)), ...
    int32(GPU_Device));

% Calculate the ground truth complex array from the CAS volume
[s1, s2, s3]=size(CASVolume);
o1=min(2,s1);
o2=min(2,s2);
GT_ComplexOutput=complex(zeros(s1,s2,s3,'single'));
GT_ComplexOutput(o1:s1,o2:s2,:)=0.5*(CASVolume(o1:s1,o2:s2,:)+CASVolume(s1:-1:o1,s2:-1:o2,:)...
    +1i*(CASVolume(o1:s1,o2:s2,:)-CASVolume(s1:-1:o1,s2:-1:o2,:)));
                
Result = isequal(ComplexOutput, GT_ComplexOutput);







