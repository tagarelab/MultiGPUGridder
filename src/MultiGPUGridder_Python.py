import ctypes
import numpy as np
from matplotlib import pyplot as plt # For plotting resulting images

#lib = ctypes.cdll.LoadLibrary("../bin/Release/MultiGPUGridder.dll") 
lib = ctypes.cdll.LoadLibrary("C:/GitRepositories/MultiGPUGridder/bin/Release/MultiGPUGridder.dll") 

# lib = ctypes.cdll.LoadLibrary("/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj/bin/libMultiGPUGridder.so")

#cls & python C:\GitRepositories\MultiGPUGridder\src\src\MultiGPUGridder_Python.py

#lib = ctypes.cdll.LoadLibrary("../bin/libMultiGPUGridder.so") 

class MultiGPUGridder(object):
    def __init__(self):
        lib.Gridder_new.argtypes = []
        lib.Gridder_new.restype = ctypes.c_void_p

        lib.SetNumberGPUs.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetNumberGPUs.restype = ctypes.c_void_p
        
        lib.SetNumberStreams.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetNumberStreams.restype = ctypes.c_void_p

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

volSize = 128

py_Vol = np.zeros((volSize,volSize,volSize)) #[1, 2, 3, 4]

for i in range(0,volSize):
    for j in range(0,volSize):
        for k in range(0,volSize):

            if (i > volSize / 4 and i < volSize *3/4 and j > volSize / 4 and j < volSize *3/4 and k > volSize / 4 and k < volSize * 3/4):
                # Distance from the center of the volume (i.e. fuzzy sphere)
                py_Vol[i][j][k] = i*j*k #np.sqrt((i-volSize/2)*(i-volSize/2) + (j-volSize/2)*(j-volSize/2) +(k-volSize/2)*(k-volSize/2))


# Take the FFT of the volume
print("Taking fourier transform...")
py_Vol = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(py_Vol)))

# Combine the real and imaginary components (i.e. CAS images)
py_Vol = np.real(py_Vol) + np.imag(py_Vol)

float_Vol = (ctypes.c_float * len(py_Vol.flatten()))(*py_Vol.flatten())

py_VolSize = [volSize, volSize, volSize]
int_VolSize = (ctypes.c_int * len(py_VolSize))(*py_VolSize)

nAxes = 100

py_CoordAxes = []

for i in range(0, 50):
    py_CoordAxes = py_CoordAxes + [1,0,0,0,1,0,0,0,1] + [1,0,0,1,0,0,0,0,1]

float_CoordAxes = (ctypes.c_float * len(py_CoordAxes))(*py_CoordAxes)

py_AxesSize = [nAxes * 9, 1, 1]
int_AxesSize = (ctypes.c_int * len(py_AxesSize))(*py_AxesSize)

py_ImgSize = [volSize,volSize,volSize]
int_ImgSize = (ctypes.c_int * len(py_ImgSize))(*py_ImgSize)


# Create an instance of the multi GPU gridder object and run the forward projection
print("MultiGPUGridder()...")
gridder=MultiGPUGridder()
gridder.SetNumberGPUs(1)
gridder.SetNumberStreams(20)
gridder.SetAxes(float_CoordAxes, int_AxesSize)
gridder.SetVolume(float_Vol, int_VolSize)
gridder.SetImgSize(int_ImgSize)
gridder.SetMaskRadius(ctypes.c_float(volSize/2 - 2))  # volSize/2 - 1
# gridder.SetMaskRadius(ctypes.c_float(2))  # volSize/2 - 1

print("gridder.Forward_Project()...")
gridder.Forward_Project()
outputImgs = gridder.GetImages()

# Convert the CASImgs output to a numpy array
outputImgs_numpy_arr = np.ctypeslib.as_array((ctypes.c_float * volSize * volSize * nAxes).from_address(ctypes.addressof(outputImgs.contents)))

# Plot the forward projections
nrows = 10
ncols = 10

for i in range(0,100):
    subPlot = plt.subplot(nrows, ncols, i+1)
    subPlot.title.set_text('Projection ' + str(i+1))

    example_CAS_Img = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(outputImgs_numpy_arr[:][:][i]))))
    # example_CAS_Img = np.real(outputImgs_numpy_arr[:][:][i])


    plt.imshow(example_CAS_Img, interpolation='nearest', cmap='gray') #, vmin=0, vmax = 3)



plt.show()

gridder.ResetVolume()
gridder.SetImages(outputImgs)

gridder.Back_Project()

outputVol = gridder.GetVolume()


# Convert the CASImgs output to a numpy array
outputVol_numpy_arr = np.ctypeslib.as_array((ctypes.c_float * volSize * volSize * volSize).from_address(ctypes.addressof(outputVol.contents)))

outputVol_numpy_arr = np.real(np.fft.fftshift(np.fft.fftn(np.fft.fftshift(outputVol_numpy_arr))))

# Plot the back projections
nrows = 10
ncols = 10

for i in range(0,100):
    subPlot = plt.subplot(nrows, ncols, i+1)
    subPlot.title.set_text('Back Projection ' + str(i+1))

    # example_CAS_Img = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(outputVol_numpy_arr[:][:][i]))))
    example_CAS_Img = np.real(outputVol_numpy_arr[:][:][i])


    plt.imshow(example_CAS_Img,  cmap='gray') #, vmin=0, vmax = 3)

    
plt.show()

print("Done!")