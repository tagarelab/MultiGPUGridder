#pragma once

/**
 * @class   AbstractGridder
 * @brief   A class for the gridder interface
 *
 *
 * This class is used as the parent class for all gridders. The AbstractGridder 
 * provides functions for setting the pointer to host (i.e. CPU) memory for 
 * the required arrays. These arrays include 
 * - the volume, 
 * - a set of coordinate axes,
 * - an output array for the projected images,
 * - the Kaiser Bessel lookup table and precompensation array.
 * 
 * The forward and back projection operations are done in-place. In other words,
 * since the pointer to the memory address for the arrays is set in the setter functions of the AbstractGridder,
 * there is no need to return the final arrays. In the Matlab wrapper, the matrices are allocated within Matlab and the
 * memory pointers are passed to the AbstractGridder. Then after the computation, the arrays within Matlab will
 * have the updated outputs without the need for returning the values back to Matlab (which saves on memory transfer time).
*/

#include <stdio.h>
#include <iostream>
#include "HostMemory.h"

class AbstractGridder
{

public:
    // Constructor
    /**
     * The AbstractGridder constructor takes the size of the volume (which must have equal demensions), the number of coordinate axes
     * for forward and back projection, and the interpolation factor for upsampling the volume for the forward and 
     * inverse Fourier transform.
     */
    AbstractGridder(int VolumeSize, int numCoordAxes, float interpFactor, int extraPadding);

    // Deconstructor
    ~AbstractGridder();

    /// Run the forward projection and return the projection images
    void ForwardProject();

    /// Run the back projection and return the volume
    void *BackProject();

    /// Reset the volume to all zeros
    void ResetVolume();

    /// Set the volume by passing a pointer to previously allocated memory and an int vector (of size 3) which has the array dimensions.
    /// The array is then pinned to CPU memory for asynchronous copying to the GPU.
    void SetVolume(float *h_Volume, int *VolumeSize);

    /// Set the Kaiser Bessel lookup table by passing a pointer to previously allocated memory and an int vector (of size 1) which has the array dimensions.
    /// The array is then pinned to CPU memory for asynchronous copying to the GPU.
    void SetKerBesselVector(float *h_KB_Table, int *ArraySize);

    /// Set the output images array (i.e. the result from forward projection) by passing a pointer to previously allocated memory
    /// and an int vector (of size 3) which has the array dimensions.
    /// The array is then pinned to CPU memory for asynchronous copying to the GPU.
    void SetImages(float *h_Imgs, int *ArraySize);

    /// Set the CTFs images array by passing a pointer to previously allocated memory
    /// and an int vector (of size 3) which has the array dimensions.
    /// The array is then pinned to CPU memory for asynchronous copying to the GPU.
    void SetCTFs(float *h_CTFs, int *ArraySize);

    /// Set the coordinate axes array (i.e. for determining the projection angles) by passing a pointer to previously allocated memory
    /// and an int vector (of size 1) which has the array dimensions.
    /// The array is then pinned to CPU memory for asynchronous copying to the GPU.
    void SetCoordAxes(float *h_CoordAxes, int *ArraySize);

    /// Set the Kaiser Bessel pre-compensation array by passing a pointer to previously allocated memory and an int vector (of size 3) which has the volume dimensions.
    /// The array is then pinned to CPU memory for asynchronous copying to the GPU.
    void SetKBPreCompArray(float *h_KBPreComp, int *ArraySize);

    /// OPTIONAL: Set the CAS volume array by passing a pointer to previously allocated memory and an int vector (of size 3) which has the volume dimensions.
    /// The array is then pinned to CPU memory for asynchronous copying to the GPU.
    void SetCASVolume(float *h_CASVolume, int *ArraySize);

    /// OPTIONAL: Set the CAS images array (i.e. the CAS version of the output from the forward projection) by passing a pointer to previously allocated memory
    /// and an int vector (of size 3) which has the volume dimensions.
    /// The array is then pinned to CPU memory for asynchronous copying to the GPU.
    void SetCASImages(float *h_CASImgs, int *ArraySize);

    /// OPTIONAL: Set the plane density array by passing a pointer to previously allocated memory and an int vector (of size 3) which has the volume dimensions.
    /// This is used for the back projection and represents the density of the projection directions and is used for compensating the back projection result.
    /// The array is then pinned to CPU memory for asynchronous copying to the GPU.
    void SetPlaneDensity(float *h_PlaneDensity, int *ArraySize);

    /// OPTIONAL: Set maximum number of coordinate axes (and corresponding number of intermediate arrays) for the forward and back projection.
    /// Although the projection classes estimate the number of coordinate axes based on available memory, this can be useful for limiting to a smaller number.
    void SetMaxAxesToAllocate(int MaxAxesToAllocate) { this->MaxAxesToAllocate = MaxAxesToAllocate; }

    /// The number of coordinate axes to be used in the forward and back projection.
    void SetNumAxes(int numCoordAxes) { this->numCoordAxes = numCoordAxes; }

    /// Interpolation factor for upsampling the volume for forward and inverse Fourier transform. Default is 2.
    void SetInterpFactor(float interpFactor) { this->interpFactor = interpFactor; };

    /// The mask radius for forward and back projection. The default value is the volume size times interpolation factor divided by two minus one.
    void SetMaskRadius(float maskRadius) { this->maskRadius = maskRadius; };

    /// Get the number of corrdinate axes.
    int GetNumAxes() { return this->numCoordAxes; }

    /// Get the pointer to the volume.
    float *GetVolume() { return this->h_Volume->GetPointer(); };

    HostMemory<float> *GetVolumeHostMemory() { return this->h_Volume; };

    /// Get the pointer to the CAS volume.
    float *GetCASVolume() { return this->h_CASVolume->GetPointer(); }

    /// Get the mask radius parameter.
    float GetMaskRadius() { return this->maskRadius; }

    // Create one instance of the following arrays
    // All of these are on the CPU
    HostMemory<float> *h_Imgs;
    HostMemory<float> *h_Volume;
    HostMemory<float> *h_CoordAxes;
    HostMemory<float> *h_KB_Table;
    HostMemory<float> *h_KBPreComp;    // Kaiser Bessel precompensation array (currently set using Matlab getPreComp())
    HostMemory<float> *h_CASVolume;    // Optional inputs
    HostMemory<float> *h_CASImgs;      // Optional inputs
    HostMemory<float> *h_PlaneDensity; // Optional inputs
    HostMemory<float> *h_CTFs;         // Optional inputs

protected:
    // Flag to test that all arrays were allocated successfully
    bool ErrorFlag;

    // Flag to apply the CTFs to the images before back projection
    bool ApplyCTFs;

    // Mask radius for the forward / back projection kernel
    float maskRadius;

    // Interpolation factor for zero padding the volume
    float interpFactor;

    // Extra padding for the volume
    int extraPadding;

    // Maximum number of coordinate axes and output projection images to allocate to memory
    // This aims to limit the amount of memory allocated for these variables
    int MaxAxesToAllocate;

    // Size of the Kaiser bessel vector
    int kerSize;

    // Width of the Kaiser bessel function
    float kerHWidth;

    // Number of coordinate axes
    int numCoordAxes;

    // Convert the volume to a CAS volume
    void VolumeToCAS();

    // Convert the CAS volume back to volume
    void CASToVolume();

    // Set coordinate axes
    void SetAxes(float *coordAxes, int *axesSize);

    // Free all of the allocated memory
    void FreeMemory(){};

private:
    // Flags to see if the host arrays have been initialized already
    bool ImgsInitialized;
    bool CTFsInitialized;
    bool VolumeInitialized;
    bool CASImgsInitialized;
    bool KB_TableInitialized;
    bool KBPreCompInitialized;
    bool CASVolumeInitialized;
    bool CoordAxesInitialized;
    bool PlaneDensityInitialized;
};
