function [times] = RunBatchTest(volSize, n1_axes, n2_axes, nStreams, nGPUs, nBatches)
    
  


    % Structure containing timing measurements
    times = [];
    times.volSize = volSize;
    times.fuzzymask = [];
    times.nCoordAxes = [];
    times.nStreams = nStreams;
    times.nGPUs = nGPUs;
    times.nBatches = nBatches;
    times.Vol_Preprocessing = [];
    times.create_uniform_axes = [];
    times.mem_allocation = [];
    times.Forward_Project = [];
    
    %% Create a volume 
    % Initialize parameters
%     volSize = 64;%256%128;%64;

    interpFactor = 2.0;

    origSize   = volSize;
    volCenter  = volSize/2 + 1;
    origCenter = origSize/2 + 1;
    origHWidth = origCenter - 1;

    %Fuzzy sphere
    disp("fuzzymask()...")
    tic
    vol=fuzzymask(origSize,3,origSize*.25,2,origCenter*[1 1 1]);
    times.fuzzymask = toc;

    % Change the sphere a bit so the projections are not all the same
%     vol(:,:,1:volSize/2) = 2 * vol(:,:,1:volSize/2);

    % MATLAB pre-processing to covert vol to CASVol
    % interpBoc and fftinfo are needed for plotting the results
    disp("Vol_Preprocessing()...")
    tic
    [CASVol, interpBox, fftinfo] = Vol_Preprocessing(vol, interpFactor);
    times.Vol_Preprocessing = toc;

    %% Define the projection directions
%     n1_axes=15;
%     n2_axes=15;

    coordAxes=single([1 0 0 0 1 0 0 0 1]');
    tic
    coordAxes=[coordAxes create_uniform_axes(n1_axes,n2_axes,0,10)];    
    coordAxes = coordAxes(:);
    nCoordAxes = length(coordAxes)/9;
    times.nCoordAxes = nCoordAxes;
    times.create_uniform_axes = toc;

    %% Display some information to the user before running the forward projection kernel

    disp(["Volume size: " + num2str(volSize)])
    disp(["Number of coordinate axes: " + num2str(nCoordAxes)])

    %% Run the forward projection kernel
    tic
    obj = CUDA_Gridder_Matlab_Class();
%     if (nGPUs ~= 4)
%         obj.SetNumberBatches(3);
%     else 
       obj.SetNumberBatches(nBatches); 
%     end
    
    obj.SetNumberGPUs(nGPUs);
    obj.SetNumberStreams(nStreams);
    
    
    disp("SetVolume()...")
    obj.SetVolume(CASVol)

    disp("SetAxes()...")
    obj.SetAxes(coordAxes)

    disp("SetImgSize()...")        
    obj.SetImgSize(int32([size(vol,1) * interpFactor, size(vol,1) * interpFactor,nCoordAxes]))
       
%     obj.CUDA_disp_mem('all')

    tic
    disp("Forward_Project_Initilize()...") % Allocates the rest of the required memory
    obj.Forward_Project_Initilize()
    times.mem_allocation = toc;

    tic
    disp("Forward_Project()...") % Run a second time to get the kernel running time
    obj.Forward_Project()
    times.Forward_Project = toc;

    % Return the resulting projection images
    InterpCASImgs  = obj.mem_Return('CASImgs_CPU_Pinned');

    clear obj
%     clear all
%     size(InterpCASImgs)
%     max(InterpCASImgs(:))
% 
%     % How many images to plot?
%     numImgsPlot = 10;
% 
%     % Make sure we have that many images first
%     numImgsPlot = min(numImgsPlot, size(InterpCASImgs,3));
% 
%     imgs=imgsFromCASImgs(InterpCASImgs(:,:,1:numImgsPlot), interpBox, fftinfo);
%     easyMontage(imgs,1);
%     colormap gray


    disp('Done!');

end

