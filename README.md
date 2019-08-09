# Build Status

[![Build Status](http://130.132.173.30:8080/buildStatus/icon?job=CUDA_Gridding)](http://127.0.0.1:8080/job/CUDA_Gridding/)

# Utilize multiple NVIDIA GPUs for fast forward and back projection

For many applications, it is needed to perform many iterations of forward and back projection in the Fourier domain. Here we provide a class for fast forward and back projection which utilizes multiple NVIDIA GPUs, CUDA, and C++ along with a wrapper for calling all the functions from within Matlab.

# Dependencies

* CUDA Toolkit 10.1
* MATLAB R2018a
* NVCC (NVIDIA CUDA Compiler)
* A compatible Matlab mex C++ compiler 
* At least one NVIDIA graphics card

# Compile CUDA, C++, and Mex Files - Ubuntu 16.04

```sh
%% Within the Matlab Console

% Compile the forward projection CUDA kernel
system("nvcc -c -shared -Xcompiler -fPIC -lcudart -lcuda gpuForwardProjectKernel.cu -I'/usr/local/cuda/tarets/x86_64-linux/include/'")

% Compile the back projection CUDA kernel
system("nvcc -c -shared -Xcompiler -fPIC -lcudart -lcuda gpuBackProjectKernel.cu -I'/usr/local/cuda/tarets/x86_64-linux/include/'")

% Compile the C++ and matlab mex wrappers along with the CUDA kernels
mex GCC='/usr/bin/gcc-6' -I'/usr/local/cuda/targets/x86_64-linux/include/' -L"/usr/local/cuda/lib64/" -lcudart -lcuda  -lnvToolsExt -DMEX mexFunctionWrapper.cpp MultiGPUGridder.cpp MemoryManager.cpp gpuForwardProjectKernel.o gpuBackProjectKernel.o

```

### Example
Please see the run_example.m file for a complete example using an example Matlab MRI dataset.