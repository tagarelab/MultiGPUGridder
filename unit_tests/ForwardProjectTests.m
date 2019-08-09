classdef ForwardProjectTests < matlab.unittest.TestCase
    % SolverTest tests solutions to the forward project CUDA gridder

    % Class variables
    properties   

        %Initialize parameters
        nBatches = 2;
        nGPUs = 1;
        MaxGPUs = gpuDeviceCount;
        
        nStreams = 64;
        volSize = 128;
        n1_axes = 50;
        n2_axes = 50;

        kernelHWidth = 2;

        interpFactor = 2.0;

        origSize   = 128; % volSize
        volCenter  = 128/2  + 1;
        origCenter = 128/2  + 1;
        origHWidth = (128/2 + 1)- 1;


    end
    

    methods (Test)
        function testForwardProjection_1_GPU(testCase)

            reset(gpuDevice())
            
            % Use the example matlab MRI image to take projections of
            load mri;
            img = squeeze(D);
            img = imresize3(img,[testCase.volSize, testCase.volSize, testCase.volSize]);
            vol = single(img);

            % Define the projection directions           
            coordAxes  = single([1 0 0 0 1 0 0 0 1]');
            coordAxes  = [coordAxes create_uniform_axes(testCase.n1_axes, testCase.n2_axes,0,10)];
            coordAxes  = coordAxes(:);
            nCoordAxes = length(coordAxes)/9;

            % MATLAB pre-processing to covert vol to CASVol
            [CASVol, CASBox, origBox, interpBox, fftinfo] = Vol_Preprocessing(vol, testCase.interpFactor);

            % Initialize the multi GPU gridder
            obj = MultiGPUGridder_Matlab_Class();
            obj.SetNumberBatches(testCase.nBatches);
            obj.SetNumberGPUs(testCase.nGPUs);
            obj.SetNumberStreams(testCase.nStreams);
            obj.SetMaskRadius(single((size(vol,1) * testCase.interpFactor)/2 - 1)); 

            obj.SetVolume(single(CASVol))

            obj.SetAxes(coordAxes)

            obj.SetImgSize(int32([size(vol,1) * testCase.interpFactor, size(vol,1) * testCase.interpFactor,nCoordAxes]))

            % Run the forward projection kernel
            obj.Forward_Project()

            InterpCASImgs = obj.GetImgs();

            imgs=imgsFromCASImgs(InterpCASImgs(:,:,1), interpBox, fftinfo); % Just use the first projection
            
            % Create a ground truth by simply summing the MRI volume in the 3 directions
            GT_Projection = squeeze(sum(vol,3));       
                     
%             figure
%             subplot(1,3,1)
%             imagesc(imgs)
%             subplot(1,3,2)
%             imagesc(GT_Projection)     
%             subplot(1,3,3)
%             imagesc(imgs - GT_Projection)   
%             colormap gray
%             colorbar
            
            % Calculate the mean difference between the ground truth and the projected image
            MeanDifference = mean(imgs(:) - GT_Projection(:));
            
            % Free the memory
            clear obj
            
            testCase.verifyLessThanOrEqual(MeanDifference, 2);

        end

        function testForwardProjection_Multi_GPU(testCase)
            
            reset(gpuDevice());

            % Use the example matlab MRI image to take projections of
            load mri;
            img = squeeze(D);
            img = imresize3(img,[testCase.volSize, testCase.volSize, testCase.volSize]);
            vol = single(img);

            % Define the projection directions           
            coordAxes  = single([1 0 0 0 1 0 0 0 1]');
            coordAxes  = [coordAxes create_uniform_axes(testCase.n1_axes, testCase.n2_axes,0,10)];
            coordAxes  = coordAxes(:);
            nCoordAxes = length(coordAxes)/9;

            % MATLAB pre-processing to covert vol to CASVol
            [CASVol, CASBox, origBox, interpBox, fftinfo] = Vol_Preprocessing(vol, testCase.interpFactor);


            % Initialize the multi GPU gridder
            obj = MultiGPUGridder_Matlab_Class();
            obj.SetNumberBatches(testCase.nBatches);
            obj.SetNumberGPUs(testCase.MaxGPUs);
            obj.SetNumberStreams(testCase.nStreams);
            obj.SetMaskRadius(single((size(vol,1) * testCase.interpFactor)/2 - 1)); 

            obj.SetVolume(single(CASVol))

            obj.SetAxes(coordAxes)

            obj.SetImgSize(int32([size(vol,1) * testCase.interpFactor, size(vol,1) * testCase.interpFactor,nCoordAxes]))

            % Run the forward projection kernel
            obj.Forward_Project()

            InterpCASImgs = obj.GetImgs();

            imgs=imgsFromCASImgs(InterpCASImgs(:,:,1), interpBox, fftinfo); % Just use the first projection
            
            % Create a ground truth by simply summing the MRI volume in the 3 directions
            GT_Projection = squeeze(sum(vol,3));       
                     
%             figure
%             subplot(1,3,1)
%             imagesc(imgs)
%             subplot(1,3,2)
%             imagesc(GT_Projection)     
%             subplot(1,3,3)
%             imagesc(imgs - GT_Projection)   
%             colormap gray
%             colorbar
            
            % Calculate the mean difference between the ground truth and the projected image
            MeanDifference = mean(imgs(:) - GT_Projection(:));
            
            % Free the memory
            clear obj
            
            testCase.verifyLessThanOrEqual(MeanDifference, 2);


        end

    end
    
end