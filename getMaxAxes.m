function nAxesMax=getMaxAxes(mem,volSize,imgSize,kerTblSize)
memFrac=0.8;
sBytes=4;
%Use only a fraction of memory
mem=mem*memFrac;
%Allocate memory for volume and kernel table
mem=mem-(volSize^3+kerTblSize)*sBytes;
%CAS images + (2) Complex FFT images  +CoordAxes
nAxesMax=max(floor(mem/((3*imgSize^2+9)*sBytes)),0);