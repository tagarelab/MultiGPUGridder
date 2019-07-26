#include "gpuForwardProject.h"

__global__ void gpuForwardProjectKernel(const float* vol, int volSize, float* img,int imgSize, float *axes, int nAxes,float maskRadius,
                                    float* ker, int kerSize, float kerHWidth)
{
int i=blockIdx.x*blockDim.x+threadIdx.x;
int j=blockIdx.y*blockDim.y+threadIdx.y;
int volCenter= volSize/2;
int imgCenter=imgSize/2;
float f_vol_i,f_vol_j,f_vol_k;
int img_i;
float *img_ptr;
int int_vol_i,int_vol_j,int_vol_k;
int i1,j1,k1;//,kerIndex;
float r=sqrtf( (float) (i-imgCenter)*(i-imgCenter)+(j-imgCenter)*(j-imgCenter));
float *nx,*ny;
int convW=roundf(kerHWidth);
float ri,rj,rk,w;
//float sigma=0.33*convW;
float wi,wj,wk;
float kerCenter=((float)kerSize-1)/2;
float kerScale=kerCenter/kerHWidth;
int kerIndex;

__shared__ float locKer[1000];

    if (threadIdx.x==0)
    {
        /* Copy over the kernel */
        for (kerIndex=0;kerIndex<kerSize;kerIndex++) 
                locKer[kerIndex]=*(ker+kerIndex);
    }
    __syncthreads();



        for(img_i=0;img_i<nAxes;img_i++)
            {
            img_ptr=img+img_i*imgSize*imgSize;

            if (r<=maskRadius)
                {
                    nx=axes+9*img_i;
                    ny=nx+3;
            
                    f_vol_i= (*(nx))*((float)(i-imgCenter))+(*(ny))*((float)(j-imgCenter))+(float)volCenter;
                    f_vol_j= (*(nx+1))*((float)(i-imgCenter))+(*(ny+1))*((float)(j-imgCenter))+(float)volCenter;
                    f_vol_k= (*(nx+2))*((float)(i-imgCenter))+(*(ny+2))*((float)(j-imgCenter))+(float)volCenter;


                    int_vol_i= roundf(f_vol_i);
                    int_vol_j= roundf(f_vol_j);
                    int_vol_k= roundf(f_vol_k);

                    *(img_ptr+j*imgSize+i)=0;
                    
                    for (i1=int_vol_i-convW;i1<=int_vol_i+convW;i1++)
                        {
                            ri= (float)i1-f_vol_i;
                            ri=min(max(ri,(float)-convW),(float)convW);
                            kerIndex=roundf( ri*kerScale+kerCenter);
                            kerIndex=min(max(kerIndex,0),kerSize-1);
                          //  wi=*(ker+kerIndex);
                            wi=*(locKer+kerIndex);
                        
                            for (j1=int_vol_j-convW;j1<=int_vol_j+convW;j1++)
                            {
                  
                                rj= (float)j1-f_vol_j;
                                rj=min(max(rj,(float)-convW),(float)convW);
                                kerIndex=roundf( rj*kerScale+kerCenter);
                                kerIndex=min(max(kerIndex,0),kerSize-1);
                              //  wj=*(ker+kerIndex);
                                wj=*(locKer+kerIndex);

                                for (k1=int_vol_k-convW;k1<=int_vol_k+convW;k1++)
                                {
                                    rk= (float)k1-f_vol_k;
                                    rk=min(max(rk,(float)-convW),(float)convW);
                                    kerIndex=roundf( rk*kerScale+kerCenter);
                                    kerIndex=min(max(kerIndex,0),kerSize-1);
                                 //   wk=*(ker+kerIndex);
                                    wk=*(locKer+kerIndex);
                                    w=wi*wj*wk;

                                     //w=expf(-(ri*ri+rj*rj+rk*rk)/(2*sigma*sigma));
                                     *(img_ptr+j*imgSize+i)=*(img_ptr+j*imgSize+i)+//w;
                                            w*( *(vol+k1*volSize*volSize+j1*volSize+i1));
                                } //End k1
                            }//End j1   
                        }//End i1
                }//End if r
            }//End img_i

}


void gpuForwardProject() 
{

    std::cout << "gpuForwardProject()" << '\n';

}

//void Forward_Project(
//) 
//{

        //const float* vol, float* img, float *axes, float* ker, // GPU arrays
    //int volSize, int imgSize, int nAxes, float maskRadius, int kerSize, float kerHWidth // Parameters

//std::cout << " Forward_Project() " << '\n';

// GPU arrays
// const float* vol
// float* img
// float *axes
// float* ker

// Parameters
// int volSize
// int imgSize
// int nAxes
// float maskRadius
// int kerSize
// float kerHWidth

// Run the forward projection kernel
//gpuForwardProjectKernel<<<1, 1>>>(vol, volSize, img,imgSize, axes, nAxes, maskRadius,ker, kerSize, kerHWidth);



// gpuForwardProjectKernel(const float* vol, int volSize, float* img,int imgSize, float *axes, int nAxes,float maskRadius,
//                                     float* ker, int kerSize, float kerHWidth)







//}