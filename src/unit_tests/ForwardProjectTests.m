classdef ForwardProjectTests < matlab.unittest.TestCase
    % SolverTest tests solutions to the forward project CUDA gridder


    % Class variables
    properties (TestParameter)  
        %Initialize parameters
        %MaxGPUs = gpuDeviceCount;     
        
        type = {'uint16'};
        
        % Parameters for running the CUDA kernels
        GPU_Device = {...
            3,...
            [2,0], ...
            [3,2,1,0]};
        
        nStreams = {1,64};
        
        % Parameters for creating the volume and coordinate axes
        VolumeSize = {64,256};       
        n1_axes = {1,200,400};
        n2_axes = {50};

    end
    

    methods (Test)
        function testForwardProjection_FFTOnGPU(testCase, GPU_Device, VolumeSize, nStreams, n1_axes, n2_axes)

            % Create the fuzzy sphere volume
            origSize=VolumeSize;
            origCenter=origSize/2+1;
            Volume = fuzzymask(origSize,3,origSize*.25,2,origCenter*[1 1 1]);
            
            % Define the projection directions
            coordAxes = create_uniform_axes(n1_axes,n2_axes,0,10);

            % Create the gridder object
            RunFFTOnGPU = true;
            interpFactor = 2;
            gridder = MultiGPUGridder_Matlab_Class(VolumeSize, n1_axes * n2_axes, interpFactor, RunFFTOnGPU);
            
            % Set the number of CUDA streams to use
            gridder.nStreams = nStreams;
            
            % Set the GPU device to use
            gridder.GPUs = int32(GPU_Device);
            
            % Set the volume
            gridder.setVolume(Volume);

            % Run the forward projection
            images = gridder.forwardProject(coordAxes);    
            
            % Create a ground truth by simply summing the MRI volume in the 3 directions
            GT_Projection = squeeze(sum(Volume,3));     
            
            % Calculate the mean difference between the ground truth and the projected images
            MeanDifference = [];
            for i = 1:size(images,3)
                temp_images = images(:,:,i);
                MeanDifference(i) = mean(temp_images(:) - GT_Projection(:));
            end
            
            % Verify that the projections are close to the ground truth
            testCase.verifyLessThanOrEqual(max(abs(MeanDifference(:))), 1);

        end

       function testForwardProjection_FFTOnCPU(testCase, GPU_Device, VolumeSize, nStreams, n1_axes, n2_axes)

            % Create the fuzzy sphere volume
            origSize=VolumeSize;
            origCenter=origSize/2+1;
            Volume = fuzzymask(origSize,3,origSize*.25,2,origCenter*[1 1 1]);
            
            % Define the projection directions
            coordAxes = create_uniform_axes(n1_axes,n2_axes,0,10);

            % Create the gridder object
            RunFFTOnGPU = false;
            interpFactor = 2;
            gridder = MultiGPUGridder_Matlab_Class(VolumeSize, n1_axes * n2_axes, interpFactor, RunFFTOnGPU);
            
            % Set the number of CUDA streams to use
            gridder.nStreams = nStreams;
            
            % Set the GPU device to use
            gridder.GPUs = int32(GPU_Device);
            
            % Set the volume
            gridder.setVolume(Volume);

            % Run the forward projection
            images = gridder.forwardProject(coordAxes);    
            
            % Create a ground truth by simply summing the MRI volume in the 3 directions
            GT_Projection = squeeze(sum(Volume,3));     
            
            % Calculate the mean difference between the ground truth and the projected images
            MeanDifference = [];
            for i = 1:size(images,3)
                temp_images = images(:,:,i);
                MeanDifference(i) = mean(temp_images(:) - GT_Projection(:));
            end
            
            % Verify that the projections are close to the ground truth
            testCase.verifyLessThanOrEqual(max(abs(MeanDifference(:))), 2);

        end

    end
    
end