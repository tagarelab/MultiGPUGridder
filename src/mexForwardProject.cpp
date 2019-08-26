#include "mexFunctionWrapper.h"
#include "gpuGridder.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nrhs != 1)
    {
        mexErrMsgTxt("mexForwardProject: There should be 1 input.");
    }

    // Get the class instance pointer from the first input
    gpuGridder *gpuGridderObj = convertMat2Ptr<gpuGridder>(prhs[0]);

    // Run the forward projection function
    gpuGridderObj->ForwardProject();

}