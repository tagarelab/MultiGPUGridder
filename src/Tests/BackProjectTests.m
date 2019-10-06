classdef BackProjectTests < matlab.unittest.TestCase
    % SolverTest tests solutions to the back project CUDA gridder


    % Class variables
    properties (TestParameter)  
        %Initialize parameters
        
        type = {'uint16'};
        
        % Parameters for running the CUDA kernels
        GPU_Device = {...
            0};
        
        nStreams = {64};
        
        % Parameters for creating the volume and coordinate axes
        VolumeSize = {128};       
        n1_axes = {100},
        n2_axes = {50};

    end    

    methods (Test)
        function testBackProjection_FFTOnGPU(testCase, GPU_Device, VolumeSize, nStreams, n1_axes, n2_axes)

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
            
            % Run the back projection
            gridder.resetVolume();
            gridder.backProject(gridder.Images, coordAxes)
            
            % Reconstruct the volume
            reconstructVol = gridder.reconstructVol();
            
            % Create a ground truth by simply summing the MRI volume in the 3 directions
            GT_Projection = squeeze(sum(Volume,3));     
            
            % Calculate the absolute mean difference between the ground truth and the projected images
            Vol_Difference = abs(reconstructVol - Volume);
            
            % Verify that the projections are close to the ground truth
            testCase.verifyLessThanOrEqual(max(Vol_Difference(:)), 1);

        end

       function testBackProjection_FFTOnCPU(testCase, GPU_Device, VolumeSize, nStreams, n1_axes, n2_axes)

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
            
            % Run the back projection
            gridder.resetVolume();
            gridder.backProject(gridder.Images, coordAxes)
            
            % Reconstruct the volume
            reconstructVol = gridder.reconstructVol();            

            % Calculate the absolute mean difference between the ground truth and the projected images
            Vol_Difference = abs(reconstructVol - Volume);
            
            % Verify that the projections are close to the ground truth
            testCase.verifyLessThanOrEqual(max(Vol_Difference(:)), 1);

        end

    end
    
end