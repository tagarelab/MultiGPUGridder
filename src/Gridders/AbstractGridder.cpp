#include "AbstractGridder.h"

AbstractGridder::AbstractGridder(int VolumeSize, int numCoordAxes, float interpFactor, int extraPadding)
{
    // Constructor for the abstract gridder class
    // Initialize parameters to default values
    this->numCoordAxes = numCoordAxes;
    this->interpFactor = interpFactor;
    this->kerSize = 501;
    this->kerHWidth = 2;
    this->extraPadding = extraPadding;
    this->ErrorFlag = false;
    this->maskRadius = (VolumeSize * this->interpFactor) / 2 - 1;
    this->h_CASImgs = NULL;      // Optional input
    this->h_CASVolume = NULL;    // Optional input
    this->h_PlaneDensity = NULL; // Optional input
    this->h_CTFs = NULL;
    this->MaxAxesToAllocate = 1000;
    this->SetNumAxes(numCoordAxes);

    // Create empty objects so we can know if they've been initialized or not
    int *ArraySize = new int[1];
    ArraySize[0] = 1;

    // Set the initialization flags to false
    this->ImgsInitialized = false;
    this->CTFsInitialized = false;
    this->VolumeInitialized = false;
    this->CASImgsInitialized = false;
    this->KB_TableInitialized = false;
    this->KBPreCompInitialized = false;
    this->CASVolumeInitialized = false;
    this->CoordAxesInitialized = false;
    this->PlaneDensityInitialized = false;
    this->ApplyCTFs = true;
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
        this->h_KB_Table = new HostMemory<float>(1, ArraySize);
        this->h_KB_Table->CopyPointer(ker_bessel_Vector);
        this->h_KB_Table->PinArray();

        this->KB_TableInitialized = true;
    }
    else
    {
        // Just copy the pointer
        this->h_KB_Table->CopyPointer(ker_bessel_Vector);

        // Check to see if we need to pin the array again (if the pointer is different)
        if (ker_bessel_Vector != this->h_KB_Table->GetPointer())
        {
            this->h_KB_Table->PinArray();
        }
    }
}

void AbstractGridder::SetVolume(float *Volume, int *ArraySize)
{
    // First save the given pointer
    if (this->VolumeInitialized == false)
    {
        this->h_Volume = new HostMemory<float>(3, ArraySize);
        this->h_Volume->CopyPointer(Volume);
        this->h_Volume->PinArray();

        this->VolumeInitialized = true;
    }
    else
    {
        // Just copy the pointer
        this->h_Volume->CopyPointer(Volume);

        // Check to see if we need to pin the array again (if the pointer is different)
        if (Volume != this->h_Volume->GetPointer())
        {
            this->h_Volume->PinArray();
        }
    }
}

void AbstractGridder::SetCASVolume(float *CASVolume, int *ArraySize)
{
    // Set the CAS volume (an optional input to the gridder)
    if (this->CASVolumeInitialized == false)
    {
        this->h_CASVolume = new HostMemory<float>(3, ArraySize);
        this->h_CASVolume->CopyPointer(CASVolume);
        this->h_CASVolume->PinArray();

        this->CASVolumeInitialized = true;
    }
    else
    {
        // Just copy the pointer
        this->h_CASVolume->CopyPointer(CASVolume);

        // Check to see if we need to pin the array again (if the pointer is different)
        if (CASVolume != this->h_CASVolume->GetPointer())
        {
            this->h_CASVolume->PinArray();
        }
    }
}

void AbstractGridder::SetPlaneDensity(float *PlaneDensity, int *ArraySize)
{
    // Set the plane density array (optional)
    if (this->PlaneDensityInitialized == false)
    {
        this->h_PlaneDensity = new HostMemory<float>(3, ArraySize);
        this->h_PlaneDensity->CopyPointer(PlaneDensity);
        this->h_PlaneDensity->PinArray();

        this->PlaneDensityInitialized = true;
    }
    else
    {
        // Just copy the pointer
        this->h_PlaneDensity->CopyPointer(PlaneDensity);

        // Check to see if we need to pin the array again (if the pointer is different)
        if (PlaneDensity != this->h_PlaneDensity->GetPointer())
        {
            this->h_PlaneDensity->PinArray();
        }
    }
}

void AbstractGridder::SetKBPreCompArray(float *KBPreCompArray, int *ArraySize)
{
    // Set the Kaiser Bessel precompentation array (currently set using Matlab's getPreComp() function)
    if (this->KBPreCompInitialized == false)
    {
        this->h_KBPreComp = new HostMemory<float>(3, ArraySize);
        this->h_KBPreComp->CopyPointer(KBPreCompArray);
        this->h_KBPreComp->PinArray();

        this->KBPreCompInitialized = true;
    }
    else
    {
        // Just copy the pointer
        this->h_KBPreComp->CopyPointer(KBPreCompArray);

        // Check to see if we need to pin the array again (if the pointer is different)
        if (KBPreCompArray != this->h_KBPreComp->GetPointer())
        {
            this->h_KBPreComp->PinArray();
        }
    }
}

void AbstractGridder::SetImages(float *imgs, int *ArraySize)
{
    // Set the images array
    if (this->ImgsInitialized == false)
    {
        this->h_Imgs = new HostMemory<float>(3, ArraySize);
        this->h_Imgs->CopyPointer(imgs);
        this->h_Imgs->PinArray();

        this->ImgsInitialized = true;
    }
    else
    {
        // Just copy the pointer
        this->h_Imgs->CopyPointer(imgs);

        // Check to see if we need to pin the array again (if the pointer is different)
        if (imgs != this->h_Imgs->GetPointer())
        {
            this->h_Imgs->PinArray();
        }
    }
}

void AbstractGridder::SetCASImages(float *CASimgs, int *ArraySize)
{
    // Set the CAS images array
    if (this->CASImgsInitialized == false)
    {
        this->h_CASImgs = new HostMemory<float>(3, ArraySize);
        this->h_CASImgs->CopyPointer(CASimgs);
        this->h_CASImgs->PinArray();

        this->CASImgsInitialized = true;
    }
    else
    {
        // Just copy the pointer
        this->h_CASImgs->CopyPointer(CASimgs);

        // Check to see if we need to pin the array again (if the pointer is different)
        if (CASimgs != this->h_CASImgs->GetPointer())
        {
            this->h_CASImgs->PinArray();
        }
    }
}

void AbstractGridder::SetCTFs(float *ctfs, int *ArraySize)
{

   
    std::cout << "h_CTFs: " << "size" << ArraySize[0] << " " << ArraySize[1] << " " << ArraySize[2] << '\n';
    // Set the images array
    if (this->CTFsInitialized == false)
    {
        this->h_CTFs = new HostMemory<float>(3, ArraySize);
        this->h_CTFs->CopyPointer(ctfs);
        this->h_CTFs->PinArray();

        this->CTFsInitialized = true;
    }
    else
    {
        // Just copy the pointer
        this->h_CTFs->CopyPointer(ctfs);

        // Check to see if we need to pin the array again (if the pointer is different)
        if (ctfs != this->h_CTFs->GetPointer())
        {
            this->h_CTFs->PinArray();
        }
    }
}

void AbstractGridder::SetCoordAxes(float *coordAxes, int *ArraySize)
{
    // Set the coordinate axes pointer
    if (this->CoordAxesInitialized == false)
    {
        this->h_CoordAxes = new HostMemory<float>(2, ArraySize);
        this->h_CoordAxes->CopyPointer(coordAxes);
        this->h_CoordAxes->PinArray();

        this->CoordAxesInitialized = true;
    }
    else
    {
        // Just copy the pointer
        this->h_CoordAxes->CopyPointer(coordAxes);

        // Check to see if we need to pin the array again (if the pointer is different)
        if (coordAxes != this->h_CoordAxes->GetPointer())
        {
            this->h_CoordAxes->PinArray();
        }
    }

    // Set the number of coordinate axes by dividing by the number of elements per axe (i.e. 9)
    this->SetNumAxes(ArraySize[0] / 9);
}

void AbstractGridder::ResetVolume()
{
    this->h_Volume->Reset();
};
