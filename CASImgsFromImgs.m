function CASimgs=CASImgsFromImgs(imgs,interpBox,fftinfo)

nImgs=size(imgs,3);
nAxes=int32(nImgs);
interpImgs=zeros([interpBox.size interpBox.size nImgs],'single');
interpImgs(interpBox.origB:interpBox.origE,...
        interpBox.origB:interpBox.origE,:)=imgs;
fftw('swisdom',fftinfo);
interpImgs=fftshift2(fft2(fftshift2(interpImgs)));
CASimgs=real(interpImgs)+imag(interpImgs);