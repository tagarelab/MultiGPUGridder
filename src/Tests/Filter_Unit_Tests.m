classdef Filter_Unit_Tests < matlab.unittest.TestCase
    % SolverTest tests solutions for the CUDA filter kernels

    % Class variables
    properties (TestParameter)  
        %Initialize parameters
        %MaxGPUs = gpuDeviceCount;     
        
        type = {'uint16'};
        GPU_Device = {0,3};
        VolumeSize = {64,256};
        nSlices = {32,128};
    end    

    methods (Test)
        function AddVolumeTest(testCase, VolumeSize, nSlices, GPU_Device)               
            Result = AddVolumeTest(VolumeSize, nSlices, GPU_Device);            
            testCase.verifyTrue(Result, true);
        end
        
        function CASToComplexTest(testCase, VolumeSize, nSlices, GPU_Device)            
            Result = CASToComplexTest(VolumeSize, nSlices, GPU_Device);            
            testCase.verifyTrue(Result, true);
        end

        function ComplexToCASTest(testCase, VolumeSize, nSlices, GPU_Device)            
            Result = ComplexToCASTest(VolumeSize, nSlices, GPU_Device);            
            testCase.verifyTrue(Result, true);
        end        
        
        function ComplexToRealTest(testCase, VolumeSize, nSlices, GPU_Device)            
            Result = ComplexToRealTest(VolumeSize, nSlices, GPU_Device);            
            testCase.verifyTrue(Result, true);
        end    
        
        function CropVolumeTest(testCase, VolumeSize, nSlices, GPU_Device)           
 
            CropX = ceil(VolumeSize*0.1);
            CropY = ceil(VolumeSize*0.2);
            CropZ = ceil(nSlices*0.1);
            
            Result = CropVolumeTest(VolumeSize, nSlices, CropX, CropY, CropZ, GPU_Device);        
            testCase.verifyTrue(Result, true);
        end 
        function DivideScalarTest(testCase, VolumeSize, nSlices, GPU_Device)           
   
            Scalar = 10;

            Result = DivideScalarTest(VolumeSize, nSlices, Scalar, GPU_Device);        
            testCase.verifyTrue(Result, true);
        end 
        function DivideVolumeTest(testCase, VolumeSize, nSlices, GPU_Device)            

            Result = DivideVolumeTest(VolumeSize, nSlices, GPU_Device);        
            testCase.verifyTrue(Result, true);
        end 
        function FFTShift2DTest(testCase, VolumeSize, nSlices, GPU_Device)           
   
            Result = FFTShift2DTest(VolumeSize, nSlices, GPU_Device);        
            testCase.verifyTrue(Result, true);
        end 
        function FFTShift3DTest(testCase, VolumeSize, GPU_Device)           

            Result = FFTShift3DTest(VolumeSize, GPU_Device);        
            testCase.verifyTrue(Result, true);
        end 
        function MultiplyVolumeTest(testCase, VolumeSize, nSlices, GPU_Device)           

            Result = MultiplyVolumeTest(VolumeSize, nSlices, GPU_Device);        
            testCase.verifyTrue(Result, true);
        end 
        function PadVolumeTest(testCase, VolumeSize, nSlices, GPU_Device)          

            PaddingX = ceil(VolumeSize*0.1);
            PaddingY = ceil(VolumeSize*0.2);
            PaddingZ = ceil(nSlices*0.1);
            
            Result = PadVolumeTest(VolumeSize, nSlices, PaddingX, PaddingY, PaddingZ, GPU_Device);        
            testCase.verifyTrue(Result, true);
        end 
        function RealToComplexTest(testCase, VolumeSize, nSlices, GPU_Device)           
 
            Result = RealToComplexTest(VolumeSize, nSlices, GPU_Device);        
            testCase.verifyTrue(Result, true);
        end 
    end    
end