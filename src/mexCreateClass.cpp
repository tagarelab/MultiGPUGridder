#include "mexFunctionWrapper.h"
#include "gpuGridder.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    int VolumeSize = 64;
    int numCoordAxes = 100;
    float interpFactor = 2;

    // Return a handle to a new C++ instance
    // plhs[0] = convertPtr2Mat<gpuGridder>(new gpuGridder( VolumeSize, numCoordAxes, interpFactor ));
    plhs[0] = convertPtr2Mat<gpuGridder>(new gpuGridder(  ));
    


}