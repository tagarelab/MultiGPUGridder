function CASimgs=CASImgsFromImgsCTFs(imgs,ctfs,interpBox,fftinfo)

nImgs=size(imgs,3);
nAxes=int32(nImgs);
interpImgs=zeros([interpBox.size interpBox.size nImgs],'single');
interpImgs(interpBox.origB:interpBox.origE,...
        interpBox.origB:interpBox.origE,:)=imgs;
    
    interpCTFs=zeros([interpBox.size interpBox.size nImgs],'single');
interpCTFs(interpBox.origB:interpBox.origE,...
        interpBox.origB:interpBox.origE,:)=fftshift2(ctfs);

% fftw('swisdom',fftinfo);
interpImgs=fftshift2(fft2(fftshift2(interpImgs))).* interpCTFs;
CASimgs=real(interpImgs)+imag(interpImgs);