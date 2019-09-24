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

    this->h_Imgs = new MemoryStruct<float>(1, ArraySize);
    std::cout << "this->h_Imgs->IsInitialized(): " << this->h_Imgs->IsInitialized() << '\n';
    this->h_Volume = new MemoryStruct<float>(1, ArraySize);
    this->h_CoordAxes = new MemoryStruct<float>(1, ArraySize);
    this->h_KB_Table = new MemoryStruct<float>(1, ArraySize);
    this->h_KBPreComp = new MemoryStruct<float>(1, ArraySize);
    this->h_CASVolume = new MemoryStruct<float>(1, ArraySize);    // Optional inputs
    this->h_CASImgs = new MemoryStruct<float>(1, ArraySize);      // Optional inputs
    this->h_PlaneDensity = new MemoryStruct<float>(1, ArraySize); // Optional inputs
}

AbstractGridder::~AbstractGridder()
{
    // Deconstructor for the abstract gridder class

    // Free all of the allocated memory
    FreeMemory();
}

void AbstractGridder::SetKerBesselVector(float *ker_bessel_Vector, int *ArraySize)
{
    // Set the keiser bessel vector
    std::cout << "this->h_KB_Table->IsInitialized(): " << this->h_KB_Table->IsInitialized() << '\n';
    if (this->h_KB_Table->IsInitialized() == false)
    {
        this->h_KB_Table = new MemoryStruct<float>(1, ArraySize);
        this->h_KB_Table->CopyPointer(ker_bessel_Vector);
        this->h_KB_Table->PinArray();
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
    std::cout << "this->h_Volume->IsInitialized(): " << this->h_Volume->IsInitialized() << '\n';
    if (this->h_Volume->IsInitialized() == false)
    {
        this->h_Volume = new MemoryStruct<float>(3, ArraySize);
        this->h_Volume->CopyPointer(Volume);
        this->h_Volume->PinArray();
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
    if (this->h_CASVolume->IsInitialized() == false)
    {
        this->h_CASVolume = new MemoryStruct<float>(3, ArraySize);
        this->h_CASVolume->CopyPointer(CASVolume);
        this->h_CASVolume->PinArray();
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
    if (this->h_PlaneDensity->IsInitialized() == false)
    {
        this->h_PlaneDensity = new MemoryStruct<float>(3, ArraySize);
        this->h_PlaneDensity->CopyPointer(PlaneDensity);
        this->h_PlaneDensity->PinArray();
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
    if (this->h_KBPreComp->IsInitialized() == false)
    {
        this->h_KBPreComp = new MemoryStruct<float>(3, ArraySize);
        this->h_KBPreComp->CopyPointer(KBPreCompArray);
        this->h_KBPreComp->PinArray();
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
    if (this->h_Imgs->IsInitialized() == false)
    {
        this->h_Imgs = new MemoryStruct<float>(3, ArraySize);
        this->h_Imgs->CopyPointer(imgs);
        this->h_Imgs->PinArray();
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
    if (this->h_CASImgs->IsInitialized() == false)
    {
        this->h_CASImgs = new MemoryStruct<float>(3, ArraySize);
        this->h_CASImgs->CopyPointer(CASimgs);
        this->h_CASImgs->PinArray();
    }
    else
    {
        // Just copy the pointer
        this->h_CASImgs->CopyPointer(CASimgs);
    }
}

void AbstractGridder::SetCoordAxes(float *coordAxes, int *ArraySize)
{
     std::cout << "this->h_CoordAxes->IsInitialized(): " << this->h_CoordAxes->IsInitialized() << '\n';

    // Set the coordinate axes pointer
    if (this->h_CoordAxes->IsInitialized() == false)
    {
        std::cout << "new MemoryStruct<float>(1, ArraySize);" << '\n';
        this->h_CoordAxes = new MemoryStruct<float>(1, ArraySize);
        std::cout << "->CopyPointer(coordAxes);" << '\n';
        this->h_CoordAxes->CopyPointer(coordAxes);
        std::cout << "->PinArray();" << '\n';
        this->h_CoordAxes->PinArray();
    }
    else
    {
        // Just copy the pointer
        this->h_CoordAxes->CopyPointer(coordAxes);
    }

    std::cout << "this->h_CoordAxes->GetPointer(): " << this->h_CoordAxes->GetPointer() << '\n';

    // Set the number of coordinate axes by dividing by the number of elements per axe (i.e. 9)
    this->SetNumAxes(ArraySize[0] / 9);
}