function w=getKernelFiltTable(hWidth,tableSize)

alpha=kernelAlpha(hWidth);
delta=2*hWidth/(tableSize-1);
x=[-hWidth:delta:hWidth];
%x=x(find((x>=-nw/2)&(x<=nw/2)));
w=gridLibKaiser(hWidth,alpha,x);
w=w./sum(w*delta);
wCenter=(numel(x)+1)/2;
w=w';
%Set boundaries to zero
w=w-w(1);
w(end)=0;
%Normalize for 3d kernel
x=[-hWidth:hWidth];
center=(tableSize+1)/2;
scale=(center-1)/hWidth;
index=x*scale+center;
ker=w(index);
ker=kron(kron(ker,ker'),ker);
denom=sum(ker(:));
w=w/(denom^(1/3));


