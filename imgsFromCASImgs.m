function imgs=imgsFromCASImgs(r,interpBox,fftinfo)
% Convert the "cosine and sine" real array r back to complex c.  We assume
% the origin is at the center of the arrays (i.e. fftshift has been
% applied.)  The -n/2 point is set to zero, which is at c(1,1).
% This routine handles a stack of 2D arraysS

[s1, s2, s3]=size(r);
o1=min(2,s1);
o2=min(2,s2);
imgs=complex(zeros(s1,s2,s3,'single'));
imgs(o1:s1,o2:s2,:)=0.5*(r(o1:s1,o2:s2,:)+r(s1:-1:o1,s2:-1:o2,:)...
    +1i*(r(o1:s1,o2:s2,:)-r(s1:-1:o1,s2:-1:o2,:)));

fftw('swisdom',fftinfo);
imgs=real(fftshift2(ifft2(fftshift2(imgs))));  %Inverse Fft

%Extract out the original images
interpB=interpBox.origB;
interpE=interpBox.origE;
imgs=imgs(interpBox.origB:interpBox.origE,...
            interpBox.origB:interpBox.origE,:);
