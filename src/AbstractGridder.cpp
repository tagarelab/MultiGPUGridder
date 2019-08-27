#include "AbstractGridder.h"

AbstractGridder::AbstractGridder(int VolumeSize, int numCoordAxes, float interpFactor)
{
    // Constructor for the abstract gridder class

    // Initlize these variable here for now
    this->kerSize = 501;
    this->kerHWidth = 2;
    this->extraPadding = 3;
    this->ErrorFlag = false;
    this->interpFactor = 2;
    this->MaxAxesAllocated = 1000;
    this->maskRadius = (VolumeSize * this->interpFactor) / 2 - 1;

    // Set the volume size
    if (VolumeSize > 0 && VolumeSize % 2 == 0) // Check that the size is greater than zero and an even number
    {
        SetVolumeSize(VolumeSize);
    }
    else
    {
        std::cerr << "Volume size must be a non-zero even integer." << '\n';
    }

    // Set the coordinate axes size
    if (numCoordAxes > 0) // Check that the number of coordinate axes is greater than zero
    {
        this->numCoordAxes = numCoordAxes;
    }
    else
    {
        std::cerr << "Number of coordinate axes must be a non-zero integer." << '\n';
    }

    // Set the interpolation factor parameter
    if (interpFactor > 0) // Check that the interpolation factor is greater than zero
    {
        this->interpFactor = interpFactor;
    }
    else
    {
        std::cerr << "Interpolation factor must be a non-zero float value." << '\n';
    }
}

// AbstractGridder::~AbstractGridder()
// {
//     // Deconstructor for the abstract gridder class

//     // Free all of the allocated memory
//     FreeMemory();
// }

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

void AbstractGridder::SetKerBesselVector(float *ker_bessel_Vector, int kerSize)
{
    // Set the keiser bessel vector
    this->kerSize = kerSize;

    for (int i = 0; i < kerSize; i++)
    {
        this->ker_bessel_Vector[i] = ker_bessel_Vector[i];
    }
}

void AbstractGridder::SetImgSize(int *imgSize)
{
    // Set the projection image size
    this->imgSize = new int[3];
    this->imgSize[0] = imgSize[0];
    this->imgSize[1] = imgSize[1];
    this->imgSize[2] = imgSize[2];
}

float *AbstractGridder::GetImages()
{
    // Return the projection images as a float array
    return this->imgs;
}

void AbstractGridder::SetVolumeSize(int VolumeSize)
{
    std::cout << "VolumeSize: " << VolumeSize << '\n';
    this->VolumeSize = new int[3];
    this->VolumeSize[0] = VolumeSize;
    this->VolumeSize[1] = VolumeSize;
    this->VolumeSize[2] = VolumeSize;
};

int *AbstractGridder::GetVolumeSize()
{
    return this->VolumeSize;
}

void AbstractGridder::SetVolume(float *Volume)
{
    this->Volume = Volume;
}

void AbstractGridder::SetCASVolume(float *CASVolume)
{
    this->CASVolume = CASVolume;
}

void AbstractGridder::SetMaskRadius(float maskRadius)
{
    this->maskRadius = maskRadius;
}

float *AbstractGridder::GetVolume()
{
    return this->Volume;
}

void AbstractGridder::SetImageSize(int *imgSize)
{
    // Set the output image size parameter
    this->imgSize = new int[3];
    this->imgSize[0] = imgSize[0];
    this->imgSize[1] = imgSize[1];
    this->imgSize[2] = imgSize[2];
};

void AbstractGridder::SetCASImageSize(int *imgSize)
{
    // Set the CAS image size parameter
    this->CASimgSize = new int[3];
    this->CASimgSize[0] = imgSize[0];
    this->CASimgSize[1] = imgSize[1];
    this->CASimgSize[2] = imgSize[2];
};

void AbstractGridder::SetImages(float *imgs)
{
    this->imgs = imgs;
}
