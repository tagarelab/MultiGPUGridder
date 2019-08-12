import ctypes
import numpy as np
from matplotlib import pyplot as plt # For plotting resulting images

# lib = ctypes.cdll.LoadLibrary("../bin/Release/MultiGPUGridder.dll") 
lib = ctypes.cdll.LoadLibrary("../bin/libMultiGPUGridder.so") 

class MultiGPUGridder(object):
    def __init__(self):
        lib.Gridder_new.argtypes = []
        lib.Gridder_new.restype = ctypes.c_void_p

        lib.SetNumberGPUs.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetNumberGPUs.restype = ctypes.c_void_p
        
        lib.SetNumberStreams.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetNumberStreams.restype = ctypes.c_void_p

        lib.SetNumberBatches.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetNumberBatches.restype = ctypes.c_void_p

        lib.SetVolume.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)]
        lib.SetVolume.restype = ctypes.c_void_p

        lib.GetVolume.argtypes = [ctypes.c_void_p]
        lib.GetVolume.restype = ctypes.POINTER(ctypes.c_float)

        lib.ResetVolume.argtypes = [ctypes.c_void_p]
        lib.ResetVolume.restype = ctypes.c_void_p
        
        lib.SetImages.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        lib.SetImages.restype = ctypes.c_void_p

        lib.GetImages.argtypes = [ctypes.c_void_p]
        lib.GetImages.restype = ctypes.POINTER(ctypes.c_float)

        lib.SetAxes.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)]
        lib.SetAxes.restype = ctypes.c_void_p

        lib.SetImgSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        lib.SetImgSize.restype = ctypes.c_void_p

        lib.SetMaskRadius.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        lib.SetMaskRadius.restype = ctypes.c_void_p

        lib.Forward_Project.argtypes = [ctypes.c_void_p]
        lib.Forward_Project.restype = ctypes.c_void_p

        lib.Back_Project.argtypes = [ctypes.c_void_p]
        lib.Back_Project.restype = ctypes.c_void_p

        self.obj = lib.Gridder_new()

    def SetNumberGPUs(self, numGPUs):
        lib.SetNumberGPUs(self.obj, numGPUs)

    def SetNumberStreams(self, nStreams):
        lib.SetNumberStreams(self.obj, nStreams) 

    def SetNumberBatches(self, nBatches):
        lib.SetNumberBatches(self.obj, nBatches) 

    def SetVolume(self, gpuVol, gpuVolSize):
        lib.SetVolume(self.obj, gpuVol, gpuVolSize) 

    def GetVolume(self):
        return lib.GetVolume(self.obj) 

    def ResetVolume(self):
        lib.ResetVolume(self.obj)

    def SetImages(self, newCASImgs):
        lib.SetImages(self.obj, newCASImgs) 

    def GetImages(self):
        return lib.GetImages(self.obj)

    def SetAxes(self, coordAxes, axesSize):
        lib.SetAxes(self.obj, coordAxes, axesSize) 

    def SetImgSize(self, imgSize):
        lib.SetImgSize(self.obj, imgSize) 

    def SetMaskRadius(self, maskRadius):
        return lib.SetMaskRadius(self.obj, maskRadius) 

    def Forward_Project(self):
        return lib.Forward_Project(self.obj) 
    
    def Back_Project(self):
        return lib.Back_Project(self.obj) 



# Create the initial parameters and test data
print("Creating input volume...")

py_Vol = np.zeros((128,128,128)) #[1, 2, 3, 4]

for i in range(0,128):
    for j in range(0,128):
        for k in range(0,128):
            # Distance from the center of the volume (i.e. fuzzy sphere)
            py_Vol[i][j][k] = np.sqrt((i-128/2)*(i-128/2) + (j-128/2)*(j-128/2) +(k-128/2)*(k-128/2))


# Take the FFT of the volume
print("Taking fourier transform...")
py_Vol = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(py_Vol)))

# Combine the real and imaginary components (i.e. CAS images)
py_Vol = np.real(py_Vol) + np.imag(py_Vol)

float_Vol = (ctypes.c_float * len(py_Vol.flatten()))(*py_Vol.flatten())

py_VolSize = [128, 128, 128]
int_VolSize = (ctypes.c_int * len(py_VolSize))(*py_VolSize)

py_CoordAxes = [1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1]
float_CoordAxes = (ctypes.c_float * len(py_CoordAxes))(*py_CoordAxes)

py_AxesSize = [90,1,1]
int_AxesSize = (ctypes.c_int * len(py_AxesSize))(*py_AxesSize)

py_ImgSize = [128,128,128]
int_ImgSize = (ctypes.c_int * len(py_ImgSize))(*py_ImgSize)


# Create an instance of the multi GPU gridder object and run the forward projection
gridder=MultiGPUGridder()
gridder.SetNumberGPUs(1)
gridder.SetNumberStreams(4)
gridder.SetNumberBatches(1)
gridder.SetAxes(float_CoordAxes, int_AxesSize)
gridder.SetVolume(float_Vol, int_VolSize)
gridder.SetImgSize(int_ImgSize)
gridder.SetMaskRadius(ctypes.c_float(128/2 - 1)) 
gridder.Forward_Project()
outputImgs = gridder.GetImages()





# Convert the CASImgs output to a numpy array
outputImgs_numpy_arr = np.ctypeslib.as_array((ctypes.c_float * 128 * 128 * 10).from_address(ctypes.addressof(outputImgs.contents)))

# Take the inverse FFT of the first projection (i.. CASImgs)
print("Taking fourier transform...")
example_CAS_Img = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(outputImgs_numpy_arr[:][:][0]))))

# Plot the forward projections
nrows = 2
ncols = 5

for i in range(0,10):
    subPlot = plt.subplot(nrows, ncols, i+1)
    subPlot.title.set_text('Projection ' + str(i+1))

    example_CAS_Img = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(outputImgs_numpy_arr[:][:][i]))))

    plt.imshow(example_CAS_Img, interpolation='nearest', cmap='gray') #, vmin=0, vmax = 3)




plt.show()