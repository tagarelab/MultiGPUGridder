#include "mexFunctionWrapper.h"
#include "MultiGPUGridder.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nrhs != 1)
    {
        mexErrMsgTxt("mexBackProject: There should be 1 input.");
    }

    // std::cout << "mexBackProject()" << '\n';
    
    // Get the class instance pointer from the first input
    MultiGPUGridder *MultiGPUGridderObj = convertMat2Ptr<MultiGPUGridder>(prhs[0]);

    // Run the back projection function
    MultiGPUGridderObj->BackProject();


}