# MultiGPUGridder Class

# Usage
#### Utilize multiple NVIDIA GPUs for fast forward and back projection

### Compile Mex File - Ubuntu 16.04

```sh
%% Within Matlab Console

% Compile the forward projection CUDA kernel
$ system("nvcc -c -shared -Xcompiler -fPIC -lcudart -lcuda gpuForwardProjectKernel.cu -I'/usr/local/cuda/tarets/x86_64-linux/include/'")

% Compile the back projection CUDA kernel
$ system("nvcc -c -shared -Xcompiler -fPIC -lcudart -lcuda gpuBackProjectKernel.cu -I'/usr/local/cuda/tarets/x86_64-linux/include/'")

% Compile the C++ and matlab mex wrappers along with the CUDA kernels
$mex GCC='/usr/bin/gcc-6' -I'/usr/local/cuda/targets/x86_64-linux/include/' -L"/usr/local/cuda/lib64/" -lcudart -lcuda  -lnvToolsExt -DMEX mexFunctionWrapper.cpp MultiGPUGridder.cpp MemoryManager.cpp gpuForwardProjectKernel.o gpuBackProjectKernel.o

```
