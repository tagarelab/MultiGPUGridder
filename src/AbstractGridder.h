#pragma once

#include <stdio.h>
#include <iostream>
#include "MemoryStruct.h"

class AbstractGridder
{

public:
    // Constructor
    AbstractGridder(int VolumeSize, int numCoordAxes, float interpFactor);

    // Deconstructor
    ~AbstractGridder();

    // Run the forward projection and return the projection images
    void ForwardProject();

    // Run the back projection and return the volume
    float *BackProject();

    // Reset the volume to all zeros
    void ResetVolume() { this->Volume->Reset(); };

    // Setter functions
    void SetVolume(float *Volume, int *VolumeSize);
    void SetKerBesselVector(float *ker_bessel_Vector, int *ArraySize);
    void SetCASVolume(float *CASVolume, int *ArraySize);
    void SetImages(float *imgs, int *ArraySize);
    void SetCoordAxes(float *coordAxes, int *ArraySize);
    void SetMaxAxesToAllocate(int MaxAxesToAllocate) { this->MaxAxesToAllocate = MaxAxesToAllocate; }
    void SetNumAxes(int numCoordAxes) { this->numCoordAxes = numCoordAxes; }
    void SetCASImages(float *CASimgs, int *ArraySize);
    void SetInterpFactor(float interpFactor) { this->interpFactor = interpFactor; };
    void SetMaskRadius(float maskRadius) { this->maskRadius = maskRadius; };

    // Getter functions
    int GetNumAxes() { return this->numCoordAxes; }
    int GetMaxAxesToAllocate() { return this->MaxAxesToAllocate; }
    int *GetVolumeSize() { return this->Volume->GetSize(); };
    int *GetCASVolumeSize() { return this->CASVolume->GetSize(); }
    int *GetCASImagesSize() { return this->CASimgs->GetSize(); }
    int *GetImgSize() { return this->imgs->GetSize(); }
    float *GetVolume() { return this->Volume->GetPointer(); };
    float *GetCASVolume() { return this->CASVolume->GetPointer(); }
    float *GetCoordAxesPtr_CPU() { return this->coordAxes->GetPointer(); }
    float GetMaskRadius() { return this->maskRadius; }
    float *GetCASImgsPtr_CPU() { return this->CASimgs->GetPointer(); }
    float *GetImgsPtr_CPU() { return this->imgs->GetPointer(); }

protected:
    // Create one instance of the following arrays to shared between objects of type AbstractGridder (and child objects)
    // All of these are on the CPU
    static MemoryStruct<float> *imgs;    // Output projection images
    static MemoryStruct<float> *CASimgs; // CAS Projection images
    static MemoryStruct<float> *Volume;
    static MemoryStruct<float> *CASVolume;
    static MemoryStruct<float> *coordAxes;
    static MemoryStruct<float> *ker_bessel_Vector;

    // Flag to test that all arrays were allocated successfully
    bool ErrorFlag;

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
};
