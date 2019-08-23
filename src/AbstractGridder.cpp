#include "AbstractGridder.h"

AbstractGridder::AbstractGridder(int VolumeSize, int numCoordAxes, float interpFactor)
{
    // Constructor for the abstract gridder class
    // Set the volume size
    if (VolumeSize > 0 && VolumeSize % 2 == 0) // Check that the size is greater than zero and an even number
    {
        this->VolumeSize = VolumeSize;
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

AbstractGridder::~AbstractGridder()
{
    // Deconstructor for the abstract gridder class

    // Free all of the allocated memory
    FreeMemory();
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