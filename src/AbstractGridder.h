#pragma once

#include <stdio.h>
#include <iostream>
#include "MemoryStruct.h"

class AbstractGridder
{
protected:

    // Volume to use for forward/back projection
    MemoryStruct *Volume;

    // CASVolume to use for forward/back projection
    MemoryStruct *CASVolume;

    // Projection images
    MemoryStruct *imgs;

    // CAS Projection images (on the CPU)
    MemoryStruct *CASimgs;

    // Coordinate axes for forward / back projection
    MemoryStruct *coordAxes;

    // Kaiser bessel window function array of predefined values
    // Provide default vector values (useful if Matlab is not available)
    MemoryStruct * ker_bessel_Vector;

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

    // // Convert the volume to a CAS volume
    // virtual void VolumeToCAS();

    // // Convert the CAS volume back to volume
    // virtual void CASToVolume();

    // Set coordinate axes
    void SetAxes(float *coordAxes, int *axesSize);

    // Mask radius for the forward / back projection kernel
    float maskRadius;

    // Get the images array 
    float *GetImages();

    // Free all of the allocated memory
    //void FreeMemory();

    // Flag to test that all arrays were allocated successfully
    bool ErrorFlag;

public:
    // // Constructor
    AbstractGridder(int VolumeSize, int numCoordAxes, float interpFactor);

    // Deconstructor
    ~AbstractGridder(){};

    // Run the forward projection and return the projection images
    void ForwardProject();

    // Run the back projection and return the volume
    float *BackProject();

    // Set the volume
    void SetVolume(float *Volume, int *VolumeSize);

    // Reset the volume to all zeros
    void ResetVolume();

    // Return the volume
    float *GetVolume();

    // Return the volume size
    int *GetVolumeSize();

    // Return the CAS volume size
    int* GetCASVolumeSize() { return this->CASVolume->size; }

    // Return the CAS images size
    int* GetCASImagesSize() { return this->CASimgs->size; }

    // Return the pointer to the CAS volume
    float *GetCASVolume() { return this->CASVolume->ptr; }

    // Get the image size
    int *GetImgSize() { return this->imgs->size; }

    // Set the kaiser bessel vector
    void SetKerBesselVector(float *ker_bessel_Vector, int* ArraySize);

    // Set the volume pointer
    void SetVolume(float *Volume);

    // Set the CAS volume pointer
    void SetCASVolume(float *CASVolume, int *ArraySize);

    // Set the output projection images array
    void SetImages(float *imgs, int *ArraySize);

    // Set the coordinate axes pointer
    void SetCoordAxes(float *coordAxes, int* ArraySize);

    // Set the maximum number of coordinate axes to allocate
    void SetMaxAxesToAllocate(int MaxAxesToAllocate) { this->MaxAxesToAllocate = MaxAxesToAllocate; }
    
    // Get the coordinate axes pointer
    float *GetCoordAxesPtr_CPU() { return this->coordAxes->ptr; }

    // Set the number of coordinate axes
    void SetNumAxes(int numCoordAxes) { this->coordAxes->size[0] = numCoordAxes; }

    // Get the number of coordinate axes
    int GetNumAxes() { return this->coordAxes->GetSize(0) / 9; } // 9 elements per coordinate axe

    // Get the number of maximum axes allocated
    int GetMaxAxesToAllocate() { return this->MaxAxesToAllocate; }

    // Set the maskRadius parameter
    void SetMaskRadius(float maskRadius);

    // Get the maskRadius parameter
    float GetMaskRadius() { return this->maskRadius; }

    // Set the CAS images pointer
    void SetCASImages(float *CASimgs, int * ArraySize);
    
    // Set the interpolation factor parameter
    void SetInterpFactor(float interpFactor);

    // Get the pointer to the CAS images array on the CPU
    float *GetCASImgsPtr_CPU() { return this->CASimgs->ptr; }

    // Get the pointer to the images array on the CPU
    float *GetImgsPtr_CPU() { return this->imgs->ptr; }
};

