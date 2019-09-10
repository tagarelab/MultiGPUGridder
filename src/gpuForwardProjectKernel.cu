#include "gpuForwardProject.h"
#include <math.h> /* round, floor, ceil, trunc */

__global__ void gpuForwardProjectKernel(const float *vol, int volSize, float *img, int imgSize, float *axes, int nAxes, float maskRadius,
                                        float *ker, int kerSize, float kerHWidth)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int volCenter = volSize / 2;
    int imgCenter = imgSize / 2;
    float f_vol_i, f_vol_j, f_vol_k;
    int img_i;
    float *img_ptr;
    int int_vol_i, int_vol_j, int_vol_k;
    int i1, j1, k1; //,kerIndex;
    float r = sqrtf((float)(i - imgCenter) * (i - imgCenter) + (j - imgCenter) * (j - imgCenter));
    float *nx, *ny;
    int convW = roundf(kerHWidth);
    float ri, rj, rk, w;
    //float sigma=0.33*convW;
    float wi, wj, wk;
    float kerCenter = ((float)kerSize - 1) / 2;
    float kerScale = kerCenter / kerHWidth;
    int kerIndex;

    __shared__ float locKer[1000];
    // __shared__ float locKer[501];

    if (threadIdx.x == 0)
    {
        /* Copy over the kernel */
        for (kerIndex = 0; kerIndex < kerSize; kerIndex++)
        {
            locKer[kerIndex] = *(ker + kerIndex);
        }
    }
    __syncthreads();

    // Are we inside the image bounds?
    if (i < 0 || i > volSize || j < 0 || j > volSize)
    {
        return;
    }

    for (img_i = 0; img_i < nAxes; img_i++)
    {
        img_ptr = img + img_i * imgSize * imgSize;

        if (r <= maskRadius)
        {
            nx = axes + 9 * img_i;
            ny = nx + 3;

            f_vol_i = (*(nx)) * ((float)(i - imgCenter)) + (*(ny)) * ((float)(j - imgCenter)) + (float)volCenter;
            f_vol_j = (*(nx + 1)) * ((float)(i - imgCenter)) + (*(ny + 1)) * ((float)(j - imgCenter)) + (float)volCenter;
            f_vol_k = (*(nx + 2)) * ((float)(i - imgCenter)) + (*(ny + 2)) * ((float)(j - imgCenter)) + (float)volCenter;

            int_vol_i = roundf(f_vol_i);
            int_vol_j = roundf(f_vol_j);
            int_vol_k = roundf(f_vol_k);

            *(img_ptr + j * imgSize + i) = 0;

            for (i1 = int_vol_i - convW; i1 <= int_vol_i + convW; i1++)
            {
                ri = (float)i1 - f_vol_i;
                ri = min(max(ri, (float)-convW), (float)convW);
                kerIndex = roundf(ri * kerScale + kerCenter);
                kerIndex = min(max(kerIndex, 0), kerSize - 1);
                wi = *(locKer + kerIndex);

                for (j1 = int_vol_j - convW; j1 <= int_vol_j + convW; j1++)
                {

                    rj = (float)j1 - f_vol_j;
                    rj = min(max(rj, (float)-convW), (float)convW);
                    kerIndex = roundf(rj * kerScale + kerCenter);
                    kerIndex = min(max(kerIndex, 0), kerSize - 1);
                    wj = *(locKer + kerIndex);

                    for (k1 = int_vol_k - convW; k1 <= int_vol_k + convW; k1++)
                    {
                        rk = (float)k1 - f_vol_k;
                        rk = min(max(rk, (float)-convW), (float)convW);
                        kerIndex = roundf(rk * kerScale + kerCenter);
                        kerIndex = min(max(kerIndex, 0), kerSize - 1);
                        wk = *(locKer + kerIndex);
                        w = wi * wj * wk;

                        *(img_ptr + j * imgSize + i) = *(img_ptr + j * imgSize + i) +
                                                       w * (*(vol + k1 * volSize * volSize + j1 * volSize + i1));

                    } //End k1
                }     //End j1
            }         //End i1
        }             //End if r
    }                 //End img_i
}

gpuForwardProject::Offsets gpuForwardProject::PlanOffsetValues()
{
    // Loop through all of the coordinate axes and calculate the corresponding pointer offset values
    // which are needed for running the CUDA kernels

    // Log2("PlanOffsetValues()", 0);

    // For compactness define the CASImgSize, CASVolSize, and ImgSize here
    int CASImgSize = this->d_CASImgs->GetSize(0);
    int CASVolSize = this->d_CASVolume->GetSize(0);
    int ImgSize = this->d_Imgs->GetSize(0);

    // Log2("coordAxesOffset", coordAxesOffset);
    // Log2("gridSize", this->gridSize);
    // Log2("blockSize", this->blockSize);
    // Log2("nAxes", this->nAxes);
    // Log2("MaxAxesAllocated", this->MaxAxesAllocated);
    // Log2("nStreams", this->nStreams);
    // Log2("GPU_Device", this->GPU_Device);
    // Log2("maskRadius", this->maskRadius);
    // Log2("ImgSize", ImgSize);
    // Log2("CASVolSize", CASVolSize);
    // Log2("CASImgSize", CASImgSize);

    // Create an instance of the Offsets struct
    Offsets Offsets_obj;
    Offsets_obj.num_offsets = 0;

    // Cumulative number of axes which have already been assigned to a CUDA stream
    int processed_nAxes = 0;

    // While we have coordinate axes to process, loop through the GPUs and the streams
    int MaxBatches = 10000; // Maximum iterations in case we get stuck in the while loop for some reason
    int batch = 0;

    // Initilize a variable which will remember how many axes have been assigned to the GPU
    // during the current batch. This is needed for calculating the offset for the stream when
    // the number of streams is greater than the number of GPUs. This resets to zeros after each
    // batch since the same GPU memory is used for the next batch.
    int numAxesGPU_Batch = 0;

    Log2("this->MaxAxesAllocated: ", this->MaxAxesAllocated);
    Log2("this->nAxes: ", this->nAxes);
    Log2("this->nStreams: ", this->nStreams);

    // Estimate how many coordinate axes to assign to each stream
    int EstimatedNumAxesPerStream;
    if (this->nAxes <= this->MaxAxesAllocated)
    {
        // The number of coordinate axes is less than or equal to the total number of axes to process
        EstimatedNumAxesPerStream = floor((double)this->nAxes / (double)this->nStreams);
    }
    else
    {
        // Several batches will be needed so evenly split the MaxAxesAllocated by the number of streams
        EstimatedNumAxesPerStream = floor((double)this->MaxAxesAllocated / (double)this->nStreams);
    }

    while (processed_nAxes < this->nAxes && batch < MaxBatches)
    {
        // Log2("processed_nAxes", processed_nAxes);

        for (int i = 0; i < this->nStreams; i++) // Loop through the streams
        {
            // Log2("i: ", i);
            // Log2("EstimatedNumAxesPerStream: ", EstimatedNumAxesPerStream);
            // Log2("this->nAxes: ", this->nAxes);

            // If we're about to process more than the number of coordinate axes, process the remaining faction of numAxesPerStream
            if (processed_nAxes + EstimatedNumAxesPerStream >= this->nAxes)
            {
                // Process the remaining fraction of EstimatedNumAxesPerStream
                // Log2("EstimatedNumAxesPerStream: ", i);
                Offsets_obj.numAxesPerStream.push_back(min(EstimatedNumAxesPerStream, nAxes - processed_nAxes));
                // Log2(" after EstimatedNumAxesPerStream: ", i);
            }
            else
            {
                // Save the estimated number of axes to the numAxesPerStream
                Offsets_obj.numAxesPerStream.push_back(EstimatedNumAxesPerStream);
            }

            //  Log2("this->nAxes: ", this->nAxes);

            // Calculate the offsets (in bytes) to determine which part of the array to copy for this stream
            // When using multiple GPUs coordAxesOffset will be the number already assigned to other GPUs
            Offsets_obj.CoordAxes_CPU_Offset.push_back((processed_nAxes + coordAxesOffset) * 9); // Each axes has 9 elements (X, Y, Z)
            Offsets_obj.coord_Axes_CPU_streamBytes.push_back(Offsets_obj.numAxesPerStream.back() * 9 * sizeof(float));

            // Log2("CoordAxes_CPU_Offset: ", i);

            // Use the number of axes already assigned to this GPU since starting the current batch to calculate the currect offset
            Offsets_obj.gpuCASImgs_Offset.push_back(numAxesGPU_Batch * CASImgSize * CASImgSize);
            Offsets_obj.gpuImgs_Offset.push_back(numAxesGPU_Batch * ImgSize * ImgSize);
            Offsets_obj.gpuCoordAxes_Stream_Offset.push_back(numAxesGPU_Batch * 9);

            // Log2("gpuCoordAxes_Stream_Offset: ", i);

            // Optionally: Copy the resulting CAS images back to the host pinned memory (CPU)
            if (this->CASImgs_CPU_Pinned != NULL)
            {
                // Have to use unsigned long long since the array may be longer than the max value int32 can represent
                // imgSize is the size of the zero padded projection images
                unsigned long long *CASImgs_Offset = new unsigned long long[3];
                CASImgs_Offset[0] = (unsigned long long)(CASImgSize);
                CASImgs_Offset[1] = (unsigned long long)(CASImgSize);
                CASImgs_Offset[2] = (unsigned long long)(processed_nAxes + coordAxesOffset);

                Offsets_obj.CASImgs_CPU_Offset.push_back(CASImgs_Offset[0] * CASImgs_Offset[1] * CASImgs_Offset[2]);

                // How many bytes are the output images?
                Offsets_obj.gpuCASImgs_streamBytes.push_back(CASImgSize * CASImgSize * Offsets_obj.numAxesPerStream.back() * sizeof(float));
            }

            // Have to use unsigned long long since the array may be longer than the max value int32 can represent
            // imgSize is the size of the zero padded projection images
            unsigned long long *Imgs_Offset = new unsigned long long[3];
            Imgs_Offset[0] = (unsigned long long)(ImgSize);
            Imgs_Offset[1] = (unsigned long long)(ImgSize);
            Imgs_Offset[2] = (unsigned long long)(processed_nAxes + coordAxesOffset);

            Offsets_obj.Imgs_CPU_Offset.push_back(Imgs_Offset[0] * Imgs_Offset[1] * Imgs_Offset[2]);

            // Log2("Imgs_CPU_Offset: ", i);

            // How many bytes are the output images?
            Offsets_obj.gpuImgs_streamBytes.push_back(ImgSize * ImgSize * Offsets_obj.numAxesPerStream.back() * sizeof(float));

            // Update the overall number of coordinate axes which have already been assigned to a CUDA stream
            processed_nAxes = processed_nAxes + Offsets_obj.numAxesPerStream.back();

            // Update the number of axes which have been assigned to this GPU during the current batch
            numAxesGPU_Batch = numAxesGPU_Batch + Offsets_obj.numAxesPerStream.back();

            // Log2("numAxesGPU_Batch: ", i);

            // Add one to the number of offset values
            Offsets_obj.num_offsets++;

            // Remember which stream this is
            Offsets_obj.stream_ID.push_back(i);

            // Log2("stream_ID: ", i);
        }

        // Increment the batch number
        batch++;

        // Reset the number of axes processed during the current batch variable
        numAxesGPU_Batch = 0;
    }

    return Offsets_obj;
}

void gpuForwardProject::Execute()
{
    // std::cout << "Running gpuForwardProject()..." << '\n';

    // For compactness define the CASImgSize, CASVolSize, and ImgSize here
    int CASImgSize = this->d_CASImgs->GetSize(0);
    int CASVolSize = this->d_CASVolume->GetSize(0);
    int ImgSize = this->d_Imgs->GetSize(0);

    // Plan the pointer offset values
    gpuForwardProject::Offsets Offsets_obj = PlanOffsetValues();

    // Define CUDA kernel dimensions
    dim3 dimGrid(this->gridSize, this->gridSize, 1);
    dim3 dimBlock(this->blockSize, this->blockSize, 1);

    for (int i = 0; i < Offsets_obj.num_offsets; i++)
    {
        if (Offsets_obj.numAxesPerStream[i] < 1)
        {
            continue;
        }

        std::cout << "GPU: " << this->GPU_Device << " stream " << Offsets_obj.stream_ID[i]
                  << " processing " << Offsets_obj.numAxesPerStream[i] << " axes " << '\n';

        // Copy the section of gpuCoordAxes which this stream will process on the current GPU
        cudaMemcpyAsync(
            this->d_CoordAxes->GetPointer(Offsets_obj.gpuCoordAxes_Stream_Offset[i]),
            &coordAxes_CPU_Pinned[Offsets_obj.CoordAxes_CPU_Offset[i]],
            Offsets_obj.coord_Axes_CPU_streamBytes[i],
            cudaMemcpyHostToDevice, streams[Offsets_obj.stream_ID[i]]);

        // Run the forward projection kernel
        gpuForwardProjectKernel<<<dimGrid, dimBlock, 0, streams[Offsets_obj.stream_ID[i]]>>>(
            this->d_CASVolume->GetPointer(),
            CASVolSize,
            this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
            CASImgSize,
            this->d_CoordAxes->GetPointer(Offsets_obj.gpuCoordAxes_Stream_Offset[i]),
            Offsets_obj.numAxesPerStream[i],
            this->maskRadius,
            this->d_KB_Table->GetPointer(),
            this->d_KB_Table->GetSize(0),
            this->kerHWidth);

        // Optionally: Copy the resulting CAS images back to the host pinned memory (CPU)
        if (this->CASImgs_CPU_Pinned != NULL)
        {
            cudaMemcpyAsync(
                &CASImgs_CPU_Pinned[Offsets_obj.CASImgs_CPU_Offset[i]],
                this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
                Offsets_obj.gpuCASImgs_streamBytes[i],
                cudaMemcpyDeviceToHost,
                streams[Offsets_obj.stream_ID[i]]);
        }

        // Convert the CAS projection images back to images using an inverse FFT and cropping out the zero padding
        this->gpuFFT_obj->CASImgsToImgs(
            streams[Offsets_obj.stream_ID[i]],
            CASImgSize,
            ImgSize,
            this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
            this->d_Imgs->GetPointer(Offsets_obj.gpuImgs_Offset[i]),
            this->d_CASImgsComplex->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
            Offsets_obj.numAxesPerStream[i]);

        // Lastly, copy the resulting cropped projection images back to the host pinned memory (CPU)
        cudaMemcpyAsync(
            &Imgs_CPU_Pinned[Offsets_obj.Imgs_CPU_Offset[i]],
            this->d_Imgs->GetPointer(Offsets_obj.gpuImgs_Offset[i]),
            Offsets_obj.gpuImgs_streamBytes[i],
            cudaMemcpyDeviceToHost,
            streams[Offsets_obj.stream_ID[i]]);
    }

    return;
}
