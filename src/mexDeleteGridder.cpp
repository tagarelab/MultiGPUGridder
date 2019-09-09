#include "mexFunctionWrapper.h"
#include "MultiGPUGridder.h"

#define Log(x) {std::cout << x << '\n';}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    // Destroy the C++ object
    Log("Destroying the MultiGPUGridder class");

    // Return a handle to a new C++ instance
    destroyObject<MultiGPUGridder>(prhs[0]);
    
    // Warn if other commands were ignored
    if (nlhs != 0 || nrhs != 1)
    {
        mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
    }
    
}