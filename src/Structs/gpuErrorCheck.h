#pragma once

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " in file " << file << " on line " << line << '\n';
   }
}