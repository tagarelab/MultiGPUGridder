#include "AbstractGridder.h"

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

    return;

    // Kaiser bessel window function array of predefined values
    // Provide default vector values (useful if Matlab is not available)
    // float ker_bessel_Vector[501] = {0, 4.7666021e-05, 9.9947727e-05, 0.00015710578, 0.00021940906, 0.00028713472, 0.00036056826, 0.00044000358, 0.000525743, 0.00061809726, 0.00071738561, 0.00082393584, 0.00093808415, 0.0010601754, 0.0011905626, 0.0013296079, 0.0014776813, 0.0016351618, 0.0018024362, 0.0019799001, 0.0021679574, 0.0023670201, 0.0025775086, 0.0027998511, 0.0030344841, 0.0032818518, 0.0035424063, 0.0038166076, 0.004104923, 0.0044078273, 0.0047258027, 0.0050593382, 0.0054089306, 0.0057750829, 0.0061583039, 0.0065591107, 0.0069780252, 0.0074155759, 0.0078722965, 0.0083487276, 0.0088454131, 0.0093629034, 0.0099017536, 0.010462523, 0.011045775, 0.011652078, 0.012282003, 0.012936124, 0.01361502, 0.014319269, 0.015049456, 0.015806165, 0.016589981, 0.017401494, 0.018241292, 0.019109964, 0.020008098, 0.020936286, 0.021895116, 0.022885174, 0.023907047, 0.024961319, 0.026048571, 0.027169384, 0.028324334, 0.029513991, 0.030738926, 0.0319997, 0.033296872, 0.034630999, 0.036002625, 0.03741229, 0.038860526, 0.040347867, 0.041874826, 0.04344191, 0.045049626, 0.046698466, 0.04838891, 0.050121427, 0.051896479, 0.053714518, 0.055575978, 0.057481285, 0.059430853, 0.061425079, 0.063464351, 0.065549031, 0.06767948, 0.06985604, 0.072079033, 0.074348763, 0.076665528, 0.079029597, 0.081441231, 0.08390066, 0.086408108, 0.088963777, 0.091567844, 0.094220467, 0.096921794, 0.099671938, 0.10247099, 0.10531905, 0.10821614, 0.11116232, 0.11415757, 0.11720191, 0.12029527, 0.12343761, 0.12662882, 0.12986881, 0.13315745, 0.13649455, 0.13987994, 0.14331342, 0.14679474, 0.15032363, 0.1538998, 0.15752295, 0.16119272, 0.16490874, 0.16867061, 0.17247792, 0.17633021, 0.18022698, 0.18416776, 0.18815197, 0.19217908, 0.19624849, 0.20035958, 0.2045117, 0.20870419, 0.21293631, 0.21720737, 0.22151661, 0.22586322, 0.23024639, 0.23466532, 0.2391191, 0.24360684, 0.24812764, 0.25268054, 0.25726455, 0.26187873, 0.26652199, 0.2711933, 0.2758916, 0.28061575, 0.28536466, 0.2901372, 0.29493213, 0.2997483, 0.30458447, 0.30943942, 0.31431186, 0.31920055, 0.32410413, 0.32902128, 0.33395067, 0.33889091, 0.34384063, 0.34879845, 0.35376289, 0.35873255, 0.36370599, 0.3686817, 0.37365818, 0.37863398, 0.38360757, 0.38857737, 0.3935419, 0.39849958, 0.40344885, 0.40838817, 0.41331589, 0.41823044, 0.42313024, 0.42801368, 0.43287915, 0.43772501, 0.44254965, 0.44735143, 0.45212874, 0.45687994, 0.4616034, 0.46629748, 0.47096053, 0.47559094, 0.48018709, 0.48474735, 0.48927006, 0.49375367, 0.49819651, 0.50259697, 0.50695354, 0.5112645, 0.51552838, 0.51974356, 0.52390844, 0.52802157, 0.53208143, 0.53608638, 0.54003495, 0.5439257, 0.54775715, 0.55152786, 0.55523634, 0.55888116, 0.56246096, 0.56597435, 0.56942004, 0.57279658, 0.57610273, 0.57933718, 0.58249867, 0.58558595, 0.58859777, 0.59153306, 0.59439057, 0.59716916, 0.59986782, 0.60248536, 0.60502082, 0.60747313, 0.60984135, 0.6121245, 0.61432171, 0.61643207, 0.61845469, 0.62038887, 0.62223369, 0.62398845, 0.62565249, 0.6272251, 0.62870562, 0.63009351, 0.63138813, 0.63258898, 0.6336956, 0.63470745, 0.63562423, 0.63644552, 0.63717103, 0.63780034, 0.63833326, 0.63876963, 0.63910919, 0.63935184, 0.63949746, 0.63954604, 0.63949746, 0.63935184, 0.63910919, 0.63876963, 0.63833326, 0.63780034, 0.63717103, 0.63644552, 0.63562423, 0.63470745, 0.6336956, 0.63258898, 0.63138813, 0.63009351, 0.62870562, 0.6272251, 0.62565249, 0.62398845, 0.62223369, 0.62038887, 0.61845469, 0.61643207, 0.61432171, 0.6121245, 0.60984135, 0.60747313, 0.60502082, 0.60248536, 0.59986782, 0.59716916, 0.59439057, 0.59153306, 0.58859777, 0.58558595, 0.58249867, 0.57933718, 0.57610273, 0.57279658, 0.56942004, 0.56597435, 0.56246096, 0.55888116, 0.55523634, 0.55152786, 0.54775715, 0.5439257, 0.54003495, 0.53608638, 0.53208143, 0.52802157, 0.52390844, 0.51974356, 0.51552838, 0.5112645, 0.50695354, 0.50259697, 0.49819651, 0.49375367, 0.48927006, 0.48474735, 0.48018709, 0.47559094, 0.47096053, 0.46629748, 0.4616034, 0.45687994, 0.45212874, 0.44735143, 0.44254965, 0.43772501, 0.43287915, 0.42801368, 0.42313024, 0.41823044, 0.41331589, 0.40838817, 0.40344885, 0.39849958, 0.3935419, 0.38857737, 0.38360757, 0.37863398, 0.37365818, 0.3686817, 0.36370599, 0.35873255, 0.35376289, 0.34879845, 0.34384063, 0.33889091, 0.33395067, 0.32902128, 0.32410413, 0.31920055, 0.31431186, 0.30943942, 0.30458447, 0.2997483, 0.29493213, 0.2901372, 0.28536466, 0.28061575, 0.2758916, 0.2711933, 0.26652199, 0.26187873, 0.25726455, 0.25268054, 0.24812764, 0.24360684, 0.2391191, 0.23466532, 0.23024639, 0.22586322, 0.22151661, 0.21720737, 0.21293631, 0.20870419, 0.2045117, 0.20035958, 0.19624849, 0.19217908, 0.18815197, 0.18416776, 0.18022698, 0.17633021, 0.17247792, 0.16867061, 0.16490874, 0.16119272, 0.15752295, 0.1538998, 0.15032363, 0.14679474, 0.14331342, 0.13987994, 0.13649455, 0.13315745, 0.12986881, 0.12662882, 0.12343761, 0.12029527, 0.11720191, 0.11415757, 0.11116232, 0.10821614, 0.10531905, 0.10247099, 0.099671938, 0.096921794, 0.094220467, 0.091567844, 0.088963777, 0.086408108, 0.08390066, 0.081441231, 0.079029597, 0.076665528, 0.074348763, 0.072079033, 0.06985604, 0.06767948, 0.065549031, 0.063464351, 0.061425079, 0.059430853, 0.057481285, 0.055575978, 0.053714518, 0.051896479, 0.050121427, 0.04838891, 0.046698466, 0.045049626, 0.04344191, 0.041874826, 0.040347867, 0.038860526, 0.03741229, 0.036002625, 0.034630999, 0.033296872, 0.0319997, 0.030738926, 0.029513991, 0.028324334, 0.027169384, 0.026048571, 0.024961319, 0.023907047, 0.022885174, 0.021895116, 0.020936286, 0.020008098, 0.019109964, 0.018241292, 0.017401494, 0.016589981, 0.015806165, 0.015049456, 0.014319269, 0.01361502, 0.012936124, 0.012282003, 0.011652078, 0.011045775, 0.010462523, 0.0099017536, 0.0093629034, 0.0088454131, 0.0083487276, 0.0078722965, 0.0074155759, 0.0069780252, 0.0065591107, 0.0061583039, 0.0057750829, 0.0054089306, 0.0050593382, 0.0047258027, 0.0044078273, 0.004104923, 0.0038166076, 0.0035424063, 0.0032818518, 0.0030344841, 0.0027998511, 0.0025775086, 0.0023670201, 0.0021679574, 0.0019799001, 0.0018024362, 0.0016351618, 0.0014776813, 0.0013296079, 0.0011905626, 0.0010601754, 0.00093808415, 0.00082393584, 0.00071738561, 0.00061809726, 0.000525743, 0.00044000358, 0.00036056826, 0.00028713472, 0.00021940906, 0.00015710578, 9.9947727e-05, 4.7666021e-05, 0};
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

void AbstractGridder::SetKerBesselVector(float *ker_bessel_Vector, int *ArraySize)
{
    // Set the keiser bessel vector
    this->ker_bessel_Vector = new MemoryStruct(1, ArraySize);
    this->ker_bessel_Vector->ptr = ker_bessel_Vector;

    // for (int i = 0; i < kerSize; i++)
    // {
    //     this->ker_bessel_Vector->ptr[i] = ker_bessel_Vector[i];
    // }

    // std::cout << "this->ker_bessel_Vector->size[0]: " << this->ker_bessel_Vector->size[0] << '\n';
    // std::cout << "this->ker_bessel_Vector->length(): " << this->ker_bessel_Vector->length() << '\n';
}

float *AbstractGridder::GetImages()
{
    // Return the projection images as a float array
    return this->imgs->ptr;
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
    return this->Volume->size;
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
    return this->Volume->ptr;
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
    this->CASimgs->ptr = CASimgs;
}

void AbstractGridder::SetCoordAxes(float *coordAxes, int *ArraySize)
{   
    std::cout << "AbstractGridder::SetCoordAxes()" << '\n';
    
    // Set the coordinate axes pointer
    this->coordAxes = new MemoryStruct(1, ArraySize);
    this->coordAxes->CopyPointer(coordAxes);
    this->coordAxes->PinArray();
}
