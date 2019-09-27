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
    void ResetVolume() { this->h_Volume->Reset(); };

    // Setter functions
    void SetVolume(float *h_Volume, int *VolumeSize);
    void SetKerBesselVector(float *h_KB_Table, int *ArraySize);
    void SetCASVolume(float *h_CASVolume, int *ArraySize);
    void SetImages(float *h_Imgs, int *ArraySize);
    void SetCoordAxes(float *h_CoordAxes, int *ArraySize);
    void SetMaxAxesToAllocate(int MaxAxesToAllocate) { this->MaxAxesToAllocate = MaxAxesToAllocate; }
    void SetNumAxes(int numCoordAxes) { this->numCoordAxes = numCoordAxes; }
    void SetCASImages(float *h_CASImgs, int *ArraySize);
    void SetPlaneDensity(float *h_PlaneDensity, int *ArraySize);
    void SetKBPreCompArray(float *h_KBPreComp, int *ArraySize);
    void SetInterpFactor(float interpFactor) { this->interpFactor = interpFactor; };
    void SetMaskRadius(float maskRadius) { this->maskRadius = maskRadius; };

    // Getter functions
    int GetNumAxes() { return this->numCoordAxes; }
    int GetMaxAxesToAllocate() { return this->MaxAxesToAllocate; }
    int *GetVolumeSize() { return this->h_Volume->GetSize(); };
    int *GetCASVolumeSize() { return this->h_CASVolume->GetSize(); }
    int *GetCASImagesSize() { return this->h_CASImgs->GetSize(); }
    int *GetImgSize() { return this->h_Imgs->GetSize(); }
    float *GetVolume() { return this->h_Volume->GetPointer(); };
    float *GetCASVolume() { return this->h_CASVolume->GetPointer(); }
    float *GetCoordAxesPtr_CPU() { return this->h_CoordAxes->GetPointer(); }
    float GetMaskRadius() { return this->maskRadius; }
    float *GetCASImgsPtr_CPU() { return this->h_CASImgs->GetPointer(); }
    float *GetImgsPtr_CPU() { return this->h_Imgs->GetPointer(); }

protected:
    // Create one instance of the following arrays to shared between objects of type AbstractGridder (and child objects)
    // All of these are on the CPU
    static MemoryStruct<float> *h_Imgs;
    static MemoryStruct<float> *h_Volume;
    static MemoryStruct<float> *h_CoordAxes;
    static MemoryStruct<float> *h_KB_Table;
    static MemoryStruct<float> *h_KBPreComp;    // Kaiser Bessel precompensation array (currently set using Matlab getPreComp())
    static MemoryStruct<float> *h_CASVolume;    // Optional inputs
    static MemoryStruct<float> *h_CASImgs;      // Optional inputs
    static MemoryStruct<float> *h_PlaneDensity; // Optional inputs

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

private:
    // Flags to see if the host arrays have been initialized already
    bool ImgsInitialized;
    bool VolumeInitialized;
    bool CASImgsInitialized;
    bool KB_TableInitialized;
    bool KBPreCompInitialized;
    bool CASVolumeInitialized;
    bool CoordAxesInitialized;
    bool PlaneDensityInitialized;
};
