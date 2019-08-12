import ctypes
import numpy as np
from matplotlib import pyplot as plt # For plotting resulting images

# lib = ctypes.cdll.LoadLibrary("C:/GitRepositories/MultiGPUGridder/bin/Release/MultiGPUGridder.dll") 
lib = ctypes.cdll.LoadLibrary("/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj/bin/libMultiGPUGridder.so") 

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
py_Vol = np.ones((128,128,128)) #[1, 2, 3, 4]
float_Vol = (ctypes.c_float * len(py_Vol.flatten()))(*py_Vol.flatten())

py_VolSize = [128, 128, 128]
int_VolSize = (ctypes.c_int * len(py_VolSize))(*py_VolSize)


py_CoordAxes = [1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1]
float_CoordAxes = (ctypes.c_float * len(py_CoordAxes))(*py_CoordAxes)

py_AxesSize = [90,1,1]
int_AxesSize = (ctypes.c_int * len(py_AxesSize))(*py_AxesSize)


py_ImgSize = [128,128,128]
int_ImgSize = (ctypes.c_int * len(py_ImgSize))(*py_ImgSize)


# Create an instance of the multi GPU gridder object
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

print("outputImgs")
print(outputImgs)

# for i in range(0,10 ): #len(py_Vol.flatten())
#    print(outputImgs[i])


outputImgs_numpy_arr = np.ctypeslib.as_array((ctypes.c_float * 128 * 128 * 10).from_address(ctypes.addressof(outputImgs.contents)))
print("outputImgs_numpy_arr")
print(outputImgs_numpy_arr)

print("Max: ")
print(np.amax(outputImgs_numpy_arr))

# Plot the forward projections
plt.imshow(outputImgs_numpy_arr[:][:][1], interpolation='nearest')
plt.show()