#include "AbstractGridder.h"

// Initialize static members
MemoryStruct<float> *AbstractGridder::Volume;
MemoryStruct<float> *AbstractGridder::CASVolume;
MemoryStruct<float> *AbstractGridder::imgs;
MemoryStruct<float> *AbstractGridder::CASimgs;
MemoryStruct<float> *AbstractGridder::coordAxes;
MemoryStruct<float> *AbstractGridder::ker_bessel_Vector;

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
    this->CASimgs = nullptr; // CASimgs on the CPU is optional
    this->MaxAxesToAllocate = 1000;
    this->SetNumAxes(numCoordAxes);
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
    this->ker_bessel_Vector = new MemoryStruct<float>(1, ArraySize);
    this->ker_bessel_Vector->CopyPointer(ker_bessel_Vector);
    this->ker_bessel_Vector->PinArray();
}

void AbstractGridder::SetVolume(float *Volume, int *ArraySize)
{
    // First save the given pointer
    this->Volume = new MemoryStruct<float>(3, ArraySize);
    this->Volume->CopyPointer(Volume);

    // Next, pin the volume to host (i.e. CPU) memory in order to enable the async CUDA stream copying
    // This will let us copy the volume to all GPUs at the same time
    this->Volume->PinArray();
}

void AbstractGridder::SetCASVolume(float *CASVolume, int *ArraySize)
{
    // Set the CAS volume
    this->CASVolume = new MemoryStruct<float>(3, ArraySize);
    this->CASVolume->CopyPointer(CASVolume);
    this->CASVolume->PinArray();
}

void AbstractGridder::SetImages(float *imgs, int *ArraySize)
{
    // Set the images array
    this->imgs = new MemoryStruct<float>(3, ArraySize);
    this->imgs->CopyPointer(imgs);
    this->imgs->PinArray();
}

void AbstractGridder::SetCASImages(float *CASimgs, int *ArraySize)
{
    // Set the CAS images array
    this->CASimgs = new MemoryStruct<float>(3, ArraySize);
    this->CASimgs->CopyPointer(CASimgs);
    this->CASimgs->PinArray();
}

void AbstractGridder::SetCoordAxes(float *coordAxes, int *ArraySize)
{
    // Set the coordinate axes pointer
    this->coordAxes = new MemoryStruct<float>(1, ArraySize);
    this->coordAxes->CopyPointer(coordAxes);
    this->coordAxes->PinArray();

    // Set the number of coordinate axes by dividing by the number of elements per axe (i.e. 9)
    this->SetNumAxes(ArraySize[0] / 9);
}
