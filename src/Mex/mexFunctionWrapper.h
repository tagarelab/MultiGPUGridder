/**
 * @header   mexFunctionWrapper
 * @brief   A class for mex functions
 *
 *
 * This class is used for keeping both the host (i.e. CPU) and the device (i.e. GPU) memory persisant 
 * when going back and forth to Matlab. Essentially, when the MultiGPUGridder object is created,
 * the corresponding memory pointer is converted to a real uint64 scalar and returned to Matlab. 
 * This scalar is remembered within the Matlab class (by the objectHandle member).
 * 
 * Then, when calling a new mex function, the objectHandle (i.e. the real uint64 scalar) is passed from
 * Matlab to the C++ code and then recasted back into the MultiGPUGridder object. This allows us to maintain
 * the memory between Matlab mex function calls.
 * 
 * 
 * 
 * 
 * */

#pragma once

#include <mex.h>
#include <iostream>
#include <stdint.h>
#include <string>
#include <cstring>
#include <typeinfo>

#define CLASS_HANDLE_SIGNATURE 0xFF00F0A5
template <class base>
class class_handle
{
public:
    class_handle(base *ptr) : signature_m(CLASS_HANDLE_SIGNATURE), name_m(typeid(base).name()), ptr_m(ptr) {}
    ~class_handle()
    {
        signature_m = 0;
        delete ptr_m;
    }
    bool isValid() { return ((signature_m == CLASS_HANDLE_SIGNATURE) && !strcmp(name_m.c_str(), typeid(base).name())); }
    base *ptr() { return ptr_m; }

private:
    uint32_t signature_m;
    const std::string name_m;
    base *const ptr_m;
};

template <class base>
inline mxArray *convertPtr2Mat(base *ptr)
{
    mexLock();
    mxArray *out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    mexMakeMemoryPersistent(out); // Make memory persistant
    *((uint64_t *)mxGetData(out)) = reinterpret_cast<uint64_t>(new class_handle<base>(ptr));
    return out;
}

template <class base>
inline class_handle<base> *convertMat2HandlePtr(const mxArray *in)
{
    if (mxGetNumberOfElements(in) != 1 || mxGetClassID(in) != mxUINT64_CLASS || mxIsComplex(in))
        mexErrMsgTxt("Input must be a real uint64 scalar.");
    // class_handle<base> *ptr = reinterpret_cast<class_handle<base> *>(*((uint64_t *)mxGetData(in)));
    class_handle<base> *ptr = reinterpret_cast<class_handle<base> *>(*((uint64_t *)mxGetData(in)));
    if (!ptr->isValid())
        mexErrMsgTxt("Handle not valid.");
    return ptr;
}

template <class base>
inline base *convertMat2Ptr(const mxArray *in)
{
    return convertMat2HandlePtr<base>(in)->ptr();
}

template <class base>
inline void destroyObject(const mxArray *in)
{
    delete convertMat2HandlePtr<base>(in);
    mexUnlock();
}
