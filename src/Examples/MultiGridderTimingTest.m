function time = MultiGridderTimingTest(VolumeSize, n1_axes, n2_axes, interpFactor, RunFFTOnGPU, nStreams)

    time = [];
    
    % Create the volume
    load mri;
    MRI_volume = squeeze(D);
    MRI_volume = imresize3(MRI_volume,[VolumeSize, VolumeSize, VolumeSize]);

    % Define the projection directions
    coordAxes = create_uniform_axes(n1_axes,n2_axes,0,10);

    % Create the gridder object
    gridder = MultiGPUGridder_Matlab_Class(VolumeSize, n1_axes * n2_axes, interpFactor, RunFFTOnGPU);
    
    % Set the number of streams
    gridder.nStreams = nStreams;
    
    % Set the volume
    gridder.setVolume(MRI_volume);    
    
    % Run the forward projection    
    tic
    images = gridder.forwardProject(coordAxes);    
    time(1) = toc;
    
    return;

    % Run the back projection
    gridder.resetVolume();
    tic
    gridder.backProject(gridder.Images, coordAxes)
    time(2) = toc;
    
    tic
    vol=gridder.getVol();
    time(3) = toc;

    % Reconstruct the volume
    tic
    reconstructVol = gridder.reconstructVol();
    time(4) = toc;







end
