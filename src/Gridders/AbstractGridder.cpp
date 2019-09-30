#include "AbstractGridder.h"

// Initialize static members
MemoryStruct<float> *AbstractGridder::h_Volume;
MemoryStruct<float> *AbstractGridder::h_CASVolume;
MemoryStruct<float> *AbstractGridder::h_Imgs;
MemoryStruct<float> *AbstractGridder::h_CASImgs;
MemoryStruct<float> *AbstractGridder::h_CoordAxes;
MemoryStruct<float> *AbstractGridder::h_KB_Table;
MemoryStruct<float> *AbstractGridder::h_PlaneDensity;
MemoryStruct<float> *AbstractGridder::h_KBPreComp;

AbstractGridder::AbstractGridder(int VolumeSize, int numCoordAxes, float interpFactor)
{
    // Constructor for the abstract gridder class
    // Initialize parameters to default values
    this->numCoordAxes = numCoordAxes;
    this->interpFactor = interpFactor;
    this->kerSize = 501;
    this->kerHWidth = 2;
    this->extraPadding = 3;
    this->ErrorFlag = false;
    this->maskRadius = (VolumeSize * this->interpFactor) / 2 - 1;
    this->h_CASImgs = NULL;      // Optional input
    this->h_CASVolume = NULL;    // Optional input
    this->h_PlaneDensity = NULL; // Optional input
    this->MaxAxesToAllocate = 1000;
    this->SetNumAxes(numCoordAxes);

    // Create empty objects so we can know if they've been initialized or not
    int *ArraySize = new int[1];
    ArraySize[0] = 1;

    // Set the initialization flags to false
    this->ImgsInitialized = false;
    this->VolumeInitialized = false;
    this->CASImgsInitialized = false;
    this->KB_TableInitialized = false;
    this->KBPreCompInitialized = false;
    this->CASVolumeInitialized = false;
    this->CoordAxesInitialized = false;
    this->PlaneDensityInitialized = false;
}

AbstractGridder::~AbstractGridder()
{
    // Deconstructor for the abstract gridder class

    // Free all of the allocated memory
    this->FreeMemory();
}

void AbstractGridder::SetKerBesselVector(float *ker_bessel_Vector, int *ArraySize)
{
    // Set the keiser bessel vector
    if (this->KB_TableInitialized == false)
    {
        this->h_KB_Table = new MemoryStruct<float>(1, ArraySize);
        this->h_KB_Table->CopyPointer(ker_bessel_Vector);
        this->h_KB_Table->PinArray();

        this->KB_TableInitialized = true;
    }
    else
    {
        // Just copy the pointer
        this->h_KB_Table->CopyPointer(ker_bessel_Vector);
    }
}

void AbstractGridder::SetVolume(float *Volume, int *ArraySize)
{
    // First save the given pointer
    if (this->VolumeInitialized == false)
    {
        this->h_Volume = new MemoryStruct<float>(3, ArraySize);
        this->h_Volume->CopyPointer(Volume);
        this->h_Volume->PinArray();

        this->VolumeInitialized = true;
    }
    else
    {
        // Just copy the pointer
        this->h_Volume->CopyPointer(Volume);
    }
}

void AbstractGridder::SetCASVolume(float *CASVolume, int *ArraySize)
{
    // Set the CAS volume (an optional input to the gridder)
    if (this->CASVolumeInitialized == false)
    {
        this->h_CASVolume = new MemoryStruct<float>(3, ArraySize);
        this->h_CASVolume->CopyPointer(CASVolume);
        this->h_CASVolume->PinArray();

        this->CASVolumeInitialized = true;
    }
    else
    {
        // Just copy the pointer
        this->h_CASVolume->CopyPointer(CASVolume);
    }
}

void AbstractGridder::SetPlaneDensity(float *PlaneDensity, int *ArraySize)
{
    // Set the plane density array (optional)
    if (this->PlaneDensityInitialized == false)
    {
        this->h_PlaneDensity = new MemoryStruct<float>(3, ArraySize);
        this->h_PlaneDensity->CopyPointer(PlaneDensity);
        this->h_PlaneDensity->PinArray();

        this->PlaneDensityInitialized = true;
    }
    else
    {
        // Just copy the pointer
        this->h_PlaneDensity->CopyPointer(PlaneDensity);
    }
}

void AbstractGridder::SetKBPreCompArray(float *KBPreCompArray, int *ArraySize)
{
    // Set the Kaiser Bessel precompentation array (currently set using Matlab's getPreComp() function)
    if (this->KBPreCompInitialized == false)
    {
        this->h_KBPreComp = new MemoryStruct<float>(3, ArraySize);
        this->h_KBPreComp->CopyPointer(KBPreCompArray);
        this->h_KBPreComp->PinArray();

        this->KBPreCompInitialized = true;
    }
    else
    {
        // Just copy the pointer
        this->h_KBPreComp->CopyPointer(KBPreCompArray);
    }
}

void AbstractGridder::SetImages(float *imgs, int *ArraySize)
{
    // Set the images array
    if (this->ImgsInitialized == false)
    {
        this->h_Imgs = new MemoryStruct<float>(3, ArraySize);
        this->h_Imgs->CopyPointer(imgs);
        this->h_Imgs->PinArray();

        this->ImgsInitialized = true;
    }
    else
    {
        // Just copy the pointer
        this->h_Imgs->CopyPointer(imgs);
    }
}

void AbstractGridder::SetCASImages(float *CASimgs, int *ArraySize)
{
    // Set the CAS images array
    if (this->CASImgsInitialized == false)
    {
        this->h_CASImgs = new MemoryStruct<float>(3, ArraySize);
        this->h_CASImgs->CopyPointer(CASimgs);
        this->h_CASImgs->PinArray();

        this->CASImgsInitialized = true;
    }
    else
    {
        // Just copy the pointer
        this->h_CASImgs->CopyPointer(CASimgs);
    }
}

void AbstractGridder::SetCoordAxes(float *coordAxes, int *ArraySize)
{
    // Set the coordinate axes pointer
    if (this->CoordAxesInitialized == false)
    {
        this->h_CoordAxes = new MemoryStruct<float>(1, ArraySize);
        this->h_CoordAxes->CopyPointer(coordAxes);
        this->h_CoordAxes->PinArray();

        this->CoordAxesInitialized = true;
    }
    else
    {
        // Just copy the pointer
        this->h_CoordAxes->CopyPointer(coordAxes);
    }

    // Set the number of coordinate axes by dividing by the number of elements per axe (i.e. 9)
    this->SetNumAxes(ArraySize[0] / 9);
}

void AbstractGridder::ResetVolume()
{
    this->h_Volume->Reset();
};
