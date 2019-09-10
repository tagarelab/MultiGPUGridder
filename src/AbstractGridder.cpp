#include "AbstractGridder.h"

// Initialize static members
MemoryStruct * AbstractGridder::Volume;
MemoryStruct * AbstractGridder::CASVolume;
MemoryStruct * AbstractGridder::imgs;
MemoryStruct * AbstractGridder::CASimgs;
MemoryStruct * AbstractGridder::coordAxes;
MemoryStruct * AbstractGridder::ker_bessel_Vector;

AbstractGridder::AbstractGridder(int VolumeSize, int numCoordAxes, float interpFactor)
{
    // Constructor for the abstract gridder class

    // Set the interpolation factor parameter
    if (interpFactor > 0) // Check that the interpolation factor is greater than zero
    {
        this->interpFactor = interpFactor;
    }
    else
    {
        std::cerr << "Interpolation factor must be a non-zero float value." << '\n';
    }

    // Initlize these variable here for now
    this->kerSize = 501;
    this->kerHWidth = 2;
    this->extraPadding = 3;
    this->ErrorFlag = false;
    this->maskRadius = (VolumeSize * this->interpFactor) / 2 - 1;
    this->CASimgs = nullptr;
    this->MaxAxesToAllocate = 1000;
    this->numCoordAxes = numCoordAxes;

    return;

}

AbstractGridder::~AbstractGridder()
{
    // Deconstructor for the abstract gridder class

    // Free all of the allocated memory
    FreeMemory();
}

void AbstractGridder::FreeMemory()
{
    // Free all of the allocated CPU memory
}

void AbstractGridder::SetInterpFactor(float interpFactor)
{
    // Set the interpolation factor
    if (interpFactor >= 0)
    {
        this->interpFactor = interpFactor;
    }
    else
    {
        std::cerr << "Interpolation factor must be greater than zero." << '\n';
    }
}

void AbstractGridder::SetKerBesselVector(float *ker_bessel_Vector, int *ArraySize)
{
    // Set the keiser bessel vector
    this->ker_bessel_Vector = new MemoryStruct(1, ArraySize);
    this->ker_bessel_Vector->CopyPointer(ker_bessel_Vector);
    this->ker_bessel_Vector->PinArray();

}

float *AbstractGridder::GetImages()
{
    // Return the projection images as a float array
    return this->imgs->GetPointer();
}

void AbstractGridder::SetVolume(float *Volume, int *ArraySize)
{
    // First save the given pointer
    this->Volume = new MemoryStruct(3, ArraySize);
    this->Volume->CopyPointer(Volume);

    // Next, pin the volume to host (i.e. CPU) memory in order to enable the async CUDA stream copying
    // This will let us copy the volume to all GPUs at the same time
    this->Volume->PinArray();
}

void AbstractGridder::ResetVolume()
{
    // Reset the volume (of type MemoryStruct)
    this->Volume->Reset();
}

int *AbstractGridder::GetVolumeSize()
{
    return this->Volume->GetSize();
}

void AbstractGridder::SetCASVolume(float *CASVolume, int *ArraySize)
{
    // Set the CAS volume
    this->CASVolume = new MemoryStruct(3, ArraySize);
    this->CASVolume->CopyPointer(CASVolume);
    this->CASVolume->PinArray();
}

void AbstractGridder::SetMaskRadius(float maskRadius)
{
    this->maskRadius = maskRadius;
}

float *AbstractGridder::GetVolume()
{
    return this->Volume->GetPointer();
}

void AbstractGridder::SetImages(float *imgs, int *ArraySize)
{
    // Set the images array
    this->imgs = new MemoryStruct(3, ArraySize);
    this->imgs->CopyPointer(imgs);
    this->imgs->PinArray();
}

void AbstractGridder::SetCASImages(float *CASimgs, int *ArraySize)
{
    // Set the CAS images array
    this->CASimgs = new MemoryStruct(3, ArraySize);
    this->CASimgs->CopyPointer(CASimgs);
    this->CASimgs->PinArray();
}

void AbstractGridder::SetCoordAxes(float *coordAxes, int *ArraySize)
{   
    // Set the coordinate axes pointer
    this->coordAxes = new MemoryStruct(1, ArraySize);
    this->coordAxes->CopyPointer(coordAxes);
    this->coordAxes->PinArray();

    // Set the number of coordinate axes by dividing by the number of elements per axe (i.e. 9)
    this->SetNumAxes(ArraySize[0] / 9);

}
