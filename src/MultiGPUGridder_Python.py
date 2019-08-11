import ctypes
import numpy as np

# c_float_p = ctypes.POINTER(ctypes.c_float)

lib = ctypes.cdll.LoadLibrary("C:/GitRepositories/MultiGPUGridder/bin/Release/MultiGPUGridder.dll") 

class MultiGPUGridder(object):
    def __init__(self):
        lib.Gridder_new.argtypes = []
        lib.Gridder_new.restype = ctypes.c_void_p

        lib.SetNumberGPUs.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetNumberGPUs.restype = ctypes.c_void_p
        
        lib.SetNumberStreams.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SetNumberStreams.restype = ctypes.c_void_p

        lib.Projection_Initilize.argtypes = [ctypes.c_void_p]
        lib.Projection_Initilize.restype = ctypes.c_void_p

        #lib.Foo_foobar.argtypes = [ctypes.c_void_p, ctypes.c_int]
        #lib.Foo_foobar.restype = ctypes.c_int        

       # lib.Foo_foosquare.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
       # lib.Foo_foosquare.restype = ctypes.POINTER(ctypes.c_float)

        self.obj = lib.Gridder_new()

    def SetNumberGPUs(self, numGPUs):
        lib.SetNumberGPUs(self.obj, numGPUs)

    def SetNumberStreams(self, nStreams):
        lib.SetNumberStreams(self.obj, nStreams) 

    def Projection_Initilize(self):
        lib.Projection_Initilize(self.obj) 

        #  def foobar(self, val):
        #     return lib.Foo_foobar(self.obj, val)

        #   def foosquare(self, val, size):
        #      return lib.Foo_foosquare(self.obj, val, size)

# from foo import Foo
# We'll create a Foo object with a value of 5...
gridder=MultiGPUGridder()

gridder.SetNumberGPUs(1)
gridder.SetNumberStreams(4)
#gridder.Projection_Initilize()





# Calling f.bar() will print a message including the value...
#f.bar()
# Now we'll use foobar to add a value to that stored in our Foo object, f
#print (f.foobar(7))
# Now we'll do the same thing - but this time demonstrate that it's a normal
# Python integer...
#x = f.foobar(2)
#print (type(x))

#pyarr = [1, 2, 3, 4]
#arr = (ctypes.c_float * len(pyarr))(*pyarr)

#print ("type(arr)")
#print (type(arr))
#print(arr)

#output =  f.foosquare(arr, len(pyarr))

#print("")
#print ("output")
#print (output)
#print (type(output))
#print(output.contents)

#for i in range(0,len(pyarr)):
#    print(output[i])
