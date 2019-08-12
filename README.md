# Utilize multiple NVIDIA GPUs for fast forward and back projection

For many applications, it is needed to perform many iterations of forward and back projection in the Fourier domain. Here we provide a class for fast forward and back projection which utilizes multiple NVIDIA GPUs, CUDA, and C++ along with a wrapper for calling all the functions from within Matlab.

# Dependencies
* CMAKE Version >3.10
* CUDA Toolkit 10.1
* NVCC (NVIDIA CUDA Compiler)
* At least one NVIDIA graphics card

* MATLAB R2018a (Matlab wrapper)

* Python 3.* (Python wrapper)
* Numpy (Python wrapper)
* Matplotlib (Python wrapper)


## Matlab Wrapper 
This wrapper allows the MultiGPUGridder functions to be called from within Matlab. 

Please see the run_example.m file for a complete example using the Matlab wrapper.


## Python Wrapper 
This wrapper allows the MultiGPUGridder functions to be called from within Python. 

Please see the MultiGPUGridder_Python.py file for a complete example using the Python wrapper.



# Compile Using CMake - Linux and Windows
## Easy method for compiling CUDA, C++, Python wrapper, and MEX files

* Use CMake to create the project files (Visual Studio for Windows and Make for Linux)
* For Windows, click on Open Project then build using the Visual Studio project
* For Linux, run "make" in the build location



### Compiling fully within Matlab - (Less Robust and only for Ubuntu 16.04)

```sh
%% Within the Matlab Console

% Compile the forward projection CUDA kernel
system("nvcc -c -shared -Xcompiler -fPIC -lcudart -lcuda gpuForwardProjectKernel.cu -I'/usr/local/cuda/tarets/x86_64-linux/include/'")

% Compile the back projection CUDA kernel
system("nvcc -c -shared -Xcompiler -fPIC -lcudart -lcuda gpuBackProjectKernel.cu -I'/usr/local/cuda/tarets/x86_64-linux/include/'")

% Compile the C++ and matlab mex wrappers along with the CUDA kernels
mex GCC='/usr/bin/gcc-6' -I'/usr/local/cuda/targets/x86_64-linux/include/' -L"/usr/local/cuda/lib64/" -lcudart -lcuda  -lnvToolsExt -DMEX mexFunctionWrapper.cpp MultiGPUGridder.cpp MemoryManager.cpp gpuForwardProjectKernel.o gpuBackProjectKernel.o

```

