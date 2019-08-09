classdef BackProjectTests < matlab.unittest.TestCase
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
        
        origSize   = 128; 
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
            

            % Run the back projection kernel
            disp("ResetVolume()...")
            obj.ResetVolume()

            disp("Back_Project()...")
            obj.Back_Project()

            disp("Get_Volume()...") % Get the volumes from all the GPUs added together
            volCAS = obj.GetVolume();

            % Get the density of inserted planes by backprojecting CASimages of values equal to one
            disp("Get Plane Density()...")
            interpImgs=ones([interpBox.size interpBox.size size(coordAxes,1)/9],'single');
            obj.ResetVolume();
            obj.SetImages(interpImgs)
            obj.Back_Project()
            volWt = obj.GetVolume();

            % Normalize the back projection result with the plane density
            % Divide the previous volume with the plane density volume
            volCAS=volCAS./(volWt+1e-6);

            % Reconstruct the volume from CASVol
            disp("volFromCAS()...")
            volReconstructed=volFromCAS(volCAS,CASBox,interpBox,origBox,testCase.kernelHWidth);

            % Free the memory 
            obj.CUDA_Free('all')
            clear obj
            
            testCase.verifyGreaterThanOrEqual(max(volReconstructed(:)), 0);

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

            % Run the back projection kernel
            disp("ResetVolume()...")
            obj.ResetVolume()

            disp("Back_Project()...")
            obj.Back_Project()

            disp("Get_Volume()...") % Get the volumes from all the GPUs added together
            volCAS = obj.GetVolume();

            % Get the density of inserted planes by backprojecting CASimages of values equal to one
            disp("Get Plane Density()...")
            interpImgs=ones([interpBox.size interpBox.size size(coordAxes,1)/9],'single');
            obj.ResetVolume();
            obj.SetImages(interpImgs)
            obj.Back_Project()
            volWt = obj.GetVolume();

            % Normalize the back projection result with the plane density
            % Divide the previous volume with the plane density volume
            volCAS=volCAS./(volWt+1e-6);

            % Reconstruct the volume from CASVol
            disp("volFromCAS()...")
            volReconstructed=volFromCAS(volCAS,CASBox,interpBox,origBox,testCase.kernelHWidth);

            % Free the memory 
            obj.CUDA_Free('all')
            clear obj
            
            testCase.verifyGreaterThanOrEqual(max(volReconstructed(:)), 0);

        end
    end    
end