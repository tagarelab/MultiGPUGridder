# Multi-GPU Gridding 

## Introduction

For many applications, it is needed to perform many iterations of forward and back projection in the Fourier domain.
Here, we provide a class for fast forward and back projection which utilizes multiple NVIDIA GPUs, CUDA, and C++ along with a 
wrapper for calling all the functions from within Matlab or from within Python.

## Dependencies 

| Library      | Version   | Usage                                                                                       |
|--------------|-----------|---------------------------------------------------------------------------------------------|
| **CUDA**         | >= 10.0   | Used for calling the GPU functions and processing data on the GPU.|
| **NVCC**         | >= 10.0   | NVIDIA CUDA compiler for compiling the GPU code. Should be included with the CUDA download.|
| **C++ Compiler** | >= C++11  | A C++ compiler is needed for compiling the C++ code.|
| **Nvidia GPU**    |  >=3.0  Compute capability       | Between 1 and 12 NVIDIA GPUs are required. Please contact us if you need support for more than 12 GPUs.|
| **Matlab**       | >= R2018a | Optional: If compiling the MATLAB wrappers for calling the class.|
| **Python**       | >= 3.0    | Optional: If compiling the Python wrappers for calling the class.|

## Installation

#### Step 1: Clone or download the GitHub repository to your computer.

#### Step 2: Open the CMake GUI with the source code path to be the "src" folder and the binaries path to be a new folder name "bin".

#### Step 3: Within CMake, click on Configure

#### Step 4: Check or uncheck the optional settings (such as BUILD_TESTS and WITH_MATLAB).

#### Step 5: Click configure again and then click on Generate.

#### Step 6 (Windows): Click on Open Project which should open Visual Studio. Then right click on ALL_BUILD and click on build. Decide between Debug or Release (see the drop down on the top center within Visual Studio).

#### Step 6 (Linux): Close CMake and open a terminal within the bin folder (which was created by CMake). Within the terminal type "make" which will then compile the code.

#### Step 7: Optionally, run the units tests within Matlab and / or Python to check everything is functioning correctly.


## Example

Here is a simple example on running the multi-GPU gridder from within Matlab:

    % Add the required Matlab file paths
    mfilepath=fileparts(which('MultiGPUGridder_Matlab_Class.m'));
    addpath(genpath(fullfile(mfilepath)));

    % Parameters for creating the volume and coordinate axes
    VolumeSize = 128;
    interpFactor = 2;
    n1_axes = 100;
    n2_axes = 100;

    % Create the volume
    load mri;
    MRI_volume = squeeze(D);
    MRI_volume = imresize3(MRI_volume,[VolumeSize, VolumeSize, VolumeSize]);

    % Define the projection directions
    coordAxes = create_uniform_axes(n1_axes,n2_axes,0,10);

    % Create the gridder object
    gridder = MultiGPUGridder_Matlab_Class(VolumeSize, n1_axes * n2_axes, interpFactor);

    % Set the volume
    gridder.setVolume(MRI_volume);

    % Run the forward projection
    images = gridder.forwardProject(coordAxes);    
    easyMontage(images, 1)

    % Run the back projection
    gridder.resetVolume();
    gridder.backProject(gridder.Images, coordAxes)

    vol=gridder.getVol();
    easyMontage(vol, 2)

    % Reconstruct the volume
    reconstructVol = gridder.reconstructVol();
    easyMontage(reconstructVol, 3)

## API Documentation

[Please see the corresponding API documentation for further details.](https://tagarelab.github.io/MultiGPUGridder/MultiGPUGridder_Doxygen/html/index.html)

=======



## Video Tutorial

[Please see the video tutorial for a detailed explanation and example of the software](https://youtu.be/gO2kiizHO4g)
