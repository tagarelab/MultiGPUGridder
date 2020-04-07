#include "mexFunctionWrapper.h"
#include "MultiGPUGridder.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nrhs != 8)
    {
        mexErrMsgTxt("mexCreateClass: There should be 8 inputs: Volume size, number of coordinate axes, the interpolation factor, amount of extra padding, number of GPUs, a vector of GPU device numbers, a flag to run the FFTs on the GPU or not, and a flag for verbose output.");
    }

    int *VolumeSize = (int *)mxGetData(prhs[0]);
    int *numCoordAxes = (int *)mxGetData(prhs[1]);
    float *interpFactor = (float *)mxGetData(prhs[2]);
    int *extraPadding = (int *)mxGetData(prhs[3]);
    int Num_GPUs = (int)mxGetScalar(prhs[4]);
    int *GPU_Device = (int *)mxGetData(prhs[5]);
    int RunFFTOnDevice = (int)mxGetScalar(prhs[6]);
    bool verboseFlag = (bool)mxGetScalar(prhs[7]);

    // Return a handle to a new C++ instance
    plhs[0] = convertPtr2Mat<MultiGPUGridder>(new MultiGPUGridder(*VolumeSize, *numCoordAxes, *interpFactor, *extraPadding, Num_GPUs, GPU_Device, RunFFTOnDevice, verboseFlag));
}

// If we're on a Windows operating system including the following code
// On Matlab in Windows, std::cout does not output to the Matlab console
// So this class redirects the std::cout messages to mexPrintf
#if defined(_MSC_VER)

class mystream : public std::streambuf
{
protected:
    virtual std::streamsize xsputn(const char *s, std::streamsize n)
    {
        mexPrintf("%.*s", n, s);
        return n;
    }
    virtual int overflow(int c = EOF)
    {
        if (c != EOF)
        {
            mexPrintf("%.1s", &c);
        }
        return 1;
    }
};

// Redirect the std::cout to the Matlab console
class scoped_redirect_cout
{
public:
    scoped_redirect_cout()
    {
        old_buf = std::cout.rdbuf();
        std::cout.rdbuf(&mout);
    }
    ~scoped_redirect_cout() { std::cout.rdbuf(old_buf); }

private:
    mystream mout;
    std::streambuf *old_buf;
};
static scoped_redirect_cout mycout_redirect;

// Redirect the std::cerr to the Matlab console
class scoped_redirect_cerr
{
public:
    scoped_redirect_cerr()
    {
        old_buf = std::cerr.rdbuf();
        std::cerr.rdbuf(&mout);
    }
    ~scoped_redirect_cerr() { std::cerr.rdbuf(old_buf); }

private:
    mystream mout;
    std::streambuf *old_buf;
};
static scoped_redirect_cerr mycerr_redirect;

#endif