function time = MultiGridderTimingTest(VolumeSize, n1_axes, n2_axes, interpFactor, RunFFTOnGPU, nStreams)

    time = [];
    
    % Create the volume
    load mri;
    MRI_volume = squeeze(D);
    MRI_volume = imresize3(MRI_volume,[VolumeSize, VolumeSize, VolumeSize]);
    
    % Define the projection directions
    coordAxes = create_uniform_axes(n1_axes,n2_axes,0,10);

%     
%     if NumGPUs == 0 % Run on the CPU instead
%         M = size(MRI_volume, 3);
%         rMax = floor(M/2-2);
%         num_projdir = n1_axes * n2_axes;
%         
%         tic
%         CPU_Forward_Project = mex_forward_project(double(MRI_volume), M, coordAxes, num_projdir, rMax);
%         time(1) = toc;    
%         
%         tic
%         BackProjected_Volume = mex_back_project(double(CPU_Forward_Project), M, coordAxes, num_projdir, rMax);
%         time(2) = toc;    
%         
% %         easyMontage(CPU_Forward_Project,1)
% %         easyMontage(BackProjected_Volume,2)
% %         pause()
%         
%         return % Skip the GPUs since we ran on the CPU instead
%     end
% %     
%     
%     for i = 1:4
%         disp("Resetting device...")
%         reset(gpuDevice(i))
%     end
    
    % Create the gridder object
    gridder = MultiGPUGridder_Matlab_Class(VolumeSize, n1_axes * n2_axes, interpFactor, RunFFTOnGPU);
    
    % Set the number of streams
    gridder.nStreams = nStreams;
    
    % Set the volume
    gridder.setVolume(MRI_volume);    
    
    % Initialize the memory allocation by running the forward projection once    
    images = gridder.forwardProject(coordAxes);    
    
    % Run the forward projection and time it    
    tic
    images = gridder.forwardProject(coordAxes);    
    time(1) = toc;    
    

    % Run the back projection
    gridder.resetVolume();
    tic
    gridder.backProject(gridder.Images, coordAxes)
    time(2) = toc;
    
    
    delete gridder
    
    return;
    
    tic
    vol=gridder.getVol();
    time(3) = toc;

    % Reconstruct the volume
    tic
    reconstructVol = gridder.reconstructVol();
    time(4) = toc;







end
