x.gpuVol = gather(obj.gpuVol);
x.CASBox_size =  obj.CASBox.size
x.gpuCASImgs = gather(obj.gpuCASImgs)
x.imgSize = obj.imgSize;
x.gpuCoordAxes = gather(obj.gpuCoordAxes)
x.nAxes = nAxes;
x.rMax = single(obj.rMax);
x.gpuKerTbl = gather(obj.gpuKerTbl);
x.kerTblSize = int32(obj.kerTblSize)
x.kerHWidth = single(obj.kerHWidth)
x.interpBox = obj.interpBox;
x.fftinfo = obj.fftinfo;

save('Forward_Project_Input.mat','x')